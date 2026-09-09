"""
Policy Knowledge RAG Agent.

Policy RAG flow:

    START
      |
      v
    retrieve
      |
      v
    rerank
      |
      v
    grade_evidence
      |
      +----------------------+
      |                      |
     good                   weak
      |                      |
      v                      v
 generate_answer        rewrite_query
      |                      |
      v                      |
 evaluate_answer <-----------+
      |
      +----------------------+
      |                      |
     pass                   retry
      |                      |
      v                      v
 build_evidence        self_correct
      |                      |
      v                      |
     END <-------------------+

The retry loop is bounded by MAX_RETRIES.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .policy_rag_retriever import PolicyVectorSearchRetriever
from .reranker import CrossEncoderReranker
from backend.cache.cache_factory import build_configured_retrieval_cache


MAX_RETRIES = 2
MIN_EVIDENCE_SCORE = 0.80
MIN_GROUNDEDNESS_SCORE = 0.80
MIN_COMPLETENESS_SCORE = 0.80
MIN_CITATION_SCORE = 0.80

RETRIEVAL_CACHE_TTL_SECONDS = 300
RETRIEVAL_CACHE_NAMESPACE = "policy_retrieval"
RETRIEVAL_CACHE_VERSION = "policy_vector_v1"


class PolicyKnowledgeState(TypedDict, total=False):
    question: str
    current_query: str
    authorization_context: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    evidence_grade: Literal["good", "weak"]
    evidence_score: float
    evidence_reason: str
    answer: str
    source_used: str
    groundedness_score: float
    completeness_score: float
    citation_score: float
    answer_grade: Literal["pass", "retry"]
    answer_feedback: str
    retry_count: int
    evidence: List[Dict[str, Any]]


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class SelfRAGAnswerGrade(BaseModel):
    groundedness_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["pass", "retry"]
    feedback: str


def _documents_to_context(documents: List[Document]) -> str:
    if not documents:
        return "No approved Policy evidence was retrieved."

    context_parts: List[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        context_parts.append(
            f"""
SOURCE {index}
Retrieval Type: {metadata.get("retrieval_type", "vector")}
Retrieval Source: {metadata.get("retrieval_source", "databricks_vector_search")}
Chunk ID: {metadata.get("chunk_id", "unknown")}
Document ID: {metadata.get("document_id", "unknown")}
File: {metadata.get("file_name", "unknown")}
Page: {metadata.get("page_number", "unknown")}
Section: {metadata.get("section", "unknown")}
Document Type: {metadata.get("document_type", "policy_document")}

{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


def build_policy_rag_agent(
    llm: Any,
    retriever: PolicyVectorSearchRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
):
    """Build the complete Policy RAG LangGraph."""

    retriever = retriever or PolicyVectorSearchRetriever()
    reranker = reranker or CrossEncoderReranker(top_n=10)

    retrieval_cache = build_configured_retrieval_cache(
    namespace=RETRIEVAL_CACHE_NAMESPACE,
)
    evidence_grader_llm = llm.with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )
    answer_grader_llm = llm.with_structured_output(
        SelfRAGAnswerGrade,
        method="json_mode",
    )

    def retrieve_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        query = state.get("current_query", state["question"])
        authorization_context = state.get(
            "authorization_context",
            "anonymous",
        )

        cached_documents = retrieval_cache.get(
            domain="policy",
            query=query,
            authorization_context=authorization_context,
            retrieval_version=RETRIEVAL_CACHE_VERSION,
        )

        if cached_documents is not None:
            return {
                "retrieved_docs": cached_documents,
            }

        documents = retriever.invoke(query)

        if documents:
            retrieval_cache.set(
                domain="policy",
                query=query,
                authorization_context=authorization_context,
                retrieval_version=RETRIEVAL_CACHE_VERSION,
                documents=documents,
            )

        return {
            "retrieved_docs": documents,
        }

    def rerank_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        query = state.get("current_query", state["question"])
        documents = state.get("retrieved_docs", [])

        ranked = reranker.rerank(
            query=query,
            documents=documents,
        )

        return {
            "reranked_docs": [
                document
                for document, _score in ranked
            ]
        }

    def grade_evidence_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        question = state.get("current_query", state["question"])
        documents = state.get("reranked_docs", [])

        if not documents:
            return {
                "evidence_grade": "weak",
                "evidence_score": 0.0,
                "evidence_reason": "No approved Policy evidence was retrieved.",
            }

        context = _documents_to_context(documents)
        prompt = f"""
You are an enterprise Policy RAG evidence evaluator.

Evaluate ONLY the supplied private Policy evidence.
Do not use outside knowledge.

Question:
{question}

Retrieved Policy Evidence:
{context}

Grade the evidence:

good:
- directly relevant
- sufficiently complete
- able to support a grounded answer

weak:
- missing
- irrelevant
- incomplete
- does not support the requested answer

Return ONLY valid JSON with:
- grade: "good" or "weak"
- score: number from 0 to 1
- reason: concise explanation
"""

        result = evidence_grader_llm.invoke(prompt)
        return {
            "evidence_grade": result.grade,
            "evidence_score": result.score,
            "evidence_reason": result.reason,
        }

    def rewrite_query_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        question = state["question"]
        current_query = state.get("current_query", question)
        retry_count = state.get("retry_count", 0) + 1

        prompt = f"""
You are a corrective enterprise Policy RAG retrieval-query optimizer.

Original user question:
{question}

Previous retrieval query:
{current_query}

Create a better Policy retrieval query.

Requirements:
1. Preserve the original intent.
2. Add useful policy/procedure/control terminology.
3. Make the query more precise.
4. Focus on concepts likely to exist in the approved Policy knowledge base.
5. Remove conversational wording.
6. Do not answer the question.
7. Do not invent facts.
8. Return ONLY the improved retrieval query.

Improved retrieval query:
"""

        result = llm.invoke(prompt)
        rewritten_query = getattr(result, "content", str(result)).strip()

        return {
            "current_query": rewritten_query or current_query,
            "retry_count": retry_count,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "answer_feedback": "",
        }

    def generate_answer_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        question = state["question"]
        documents = state.get("reranked_docs", [])
        context = _documents_to_context(documents)

        prompt = f"""
You are an enterprise Policy assistant.

Answer the user's question using ONLY the approved private Policy evidence.

Question:
{question}

Policy Evidence:
{context}

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. Every factual claim must be supported by the supplied evidence.
4. If the evidence is insufficient, say so.
5. Keep the answer concise but useful.
6. Cite the source file and page when available.
7. Do not cite sources that are not present in the evidence.

Answer:
"""

        result = llm.invoke(prompt)
        answer = getattr(result, "content", str(result)).strip()

        return {
            "answer": answer,
            "source_used": "policy_documents",
        }

    def evaluate_answer_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        question = state["question"]
        answer = state.get("answer", "")
        documents = state.get("reranked_docs", [])
        context = _documents_to_context(documents)

        prompt = f"""
You are a Self-RAG Policy answer evaluator.

Evaluate the generated answer against the original user question and the
supplied Policy evidence only.

Question:
{question}

Generated Answer:
{answer}

Policy Evidence:
{context}

Evaluate:
- groundedness_score: every factual claim supported by evidence
- completeness_score: adequately addresses the question
- citation_score: source/page references are supported by evidence

PASS only when all three scores are >= 0.80.
Otherwise return RETRY.

Do not use outside knowledge.

Return ONLY valid JSON with:
- groundedness_score: number from 0 to 1
- completeness_score: number from 0 to 1
- citation_score: number from 0 to 1
- decision: "pass" or "retry"
- feedback: concise explanation
"""

        result = answer_grader_llm.invoke(prompt)

        scores_pass = (
            result.groundedness_score >= MIN_GROUNDEDNESS_SCORE
            and result.completeness_score >= MIN_COMPLETENESS_SCORE
            and result.citation_score >= MIN_CITATION_SCORE
        )

        decision = (
            "pass"
            if result.decision == "pass" and scores_pass
            else "retry"
        )

        return {
            "groundedness_score": result.groundedness_score,
            "completeness_score": result.completeness_score,
            "citation_score": result.citation_score,
            "answer_grade": decision,
            "answer_feedback": result.feedback,
        }

    def self_correct_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        question = state["question"]
        current_query = state.get("current_query", question)
        feedback = state.get("answer_feedback", "")
        retry_count = state.get("retry_count", 0) + 1

        prompt = f"""
You are a Self-RAG corrective Policy retrieval agent.

Original question:
{question}

Previous retrieval query:
{current_query}

Self-RAG evaluator feedback:
{feedback}

Create a better retrieval query that specifically addresses the missing or
unsupported evidence.

Requirements:
1. Preserve the original intent.
2. Target missing evidence.
3. Remove unsupported assumptions.
4. Add precise policy terminology.
5. Improve retrieval specificity.
6. Do not answer the question.
7. Return ONLY the improved retrieval query.

Improved retrieval query:
"""

        result = llm.invoke(prompt)
        improved_query = getattr(result, "content", str(result)).strip()

        return {
            "current_query": improved_query or current_query,
            "retry_count": retry_count,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "answer_feedback": "",
        }

    def insufficient_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        return {
            "answer": (
                "I could not find sufficient approved Policy evidence "
                "to answer this question."
            ),
            "source_used": "policy_documents",
            "evidence": [],
        }

    def build_evidence_node(
        state: PolicyKnowledgeState,
    ) -> PolicyKnowledgeState:
        documents = state.get("reranked_docs", [])
        evidence: List[Dict[str, Any]] = []

        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)

            evidence.append(
                {
                    "evidence_id": f"policy-evidence-{index}",
                    "source": metadata.get(
                        "retrieval_source",
                        "databricks_vector_search",
                    ),
                    "tool": "policy_rag",
                    "retrieval_type": metadata.get(
                        "retrieval_type",
                        "vector",
                    ),
                    "data": {
                        "chunk_id": metadata.get("chunk_id"),
                        "document_id": metadata.get("document_id"),
                        "file_name": metadata.get("file_name"),
                        "file_path": metadata.get("file_path"),
                        "page_number": metadata.get("page_number"),
                        "section": metadata.get("section"),
                        "document_type": metadata.get("document_type"),
                        "chunk_text": document.page_content,
                    },
                    "metadata": metadata,
                }
            )

        return {"evidence": evidence}

    def route_after_evidence(state: PolicyKnowledgeState) -> str:
        if (
            state.get("evidence_grade") == "good"
            and state.get("evidence_score", 0.0) >= MIN_EVIDENCE_SCORE
        ):
            return "generate_answer"

        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite_query"

        return "insufficient"

    def route_after_answer(state: PolicyKnowledgeState) -> str:
        if state.get("answer_grade") == "pass":
            return "build_evidence"

        if state.get("retry_count", 0) < MAX_RETRIES:
            return "self_correct"

        return "insufficient"

    graph = StateGraph(PolicyKnowledgeState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_evidence", grade_evidence_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("self_correct", self_correct_node)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_node("insufficient", insufficient_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_evidence")

    graph.add_conditional_edges(
        "grade_evidence",
        route_after_evidence,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", "evaluate_answer")

    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_answer,
        {
            "build_evidence": "build_evidence",
            "self_correct": "self_correct",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("self_correct", "retrieve")
    graph.add_edge("build_evidence", END)
    graph.add_edge("insufficient", END)

    return graph.compile()