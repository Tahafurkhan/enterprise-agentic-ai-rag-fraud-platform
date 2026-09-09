"""
Company Knowledge RAG Agent.

Enterprise Company Knowledge RAG flow:

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

This module is deliberately limited to Company Knowledge.
It does not use MCP and does not perform external web search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .databricks_retriever import DatabricksVectorSearchRetriever
from .reranker import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 2

MIN_EVIDENCE_SCORE = 0.80
MIN_GROUNDEDNESS_SCORE = 0.80
MIN_COMPLETENESS_SCORE = 0.80
MIN_CITATION_SCORE = 0.80


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CompanyKnowledgeState(TypedDict, total=False):
    # Original question
    question: str

    # Current retrieval query.
    # This changes during Corrective RAG.
    current_query: str

    # Retrieval
    retrieved_docs: List[Document]
    reranked_docs: List[Document]

    # KB evidence grading
    evidence_grade: Literal["good", "weak"]
    evidence_score: float
    evidence_reason: str

    # Answer
    answer: str
    source_used: str

    # Self-RAG evaluation
    groundedness_score: float
    completeness_score: float
    citation_score: float
    answer_grade: Literal["pass", "retry"]
    answer_feedback: str

    # Corrective RAG
    retry_count: int

    # Final evidence
    evidence: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _documents_to_context(
    documents: List[Document],
) -> str:
    """
    Convert retrieved documents into controlled LLM context.

    Supports both:
        - Databricks Vector Search evidence
        - Neo4j Graph RAG evidence
    """

    if not documents:
        return (
            "No approved Company Knowledge evidence "
            "was retrieved."
        )

    context_parts: List[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        retrieval_type = metadata.get(
            "retrieval_type",
            "unknown",
        )

        retrieval_source = metadata.get(
            "retrieval_source",
            "company_knowledge",
        )

        # --------------------------------------------------------
        # Vector Search document
        # --------------------------------------------------------

        if retrieval_type == "vector":

            file_name = metadata.get(
                "file_name",
                "unknown",
            )

            page_number = metadata.get(
                "page_number",
                "unknown",
            )

            section = metadata.get(
                "section",
                "unknown",
            )

            chunk_id = metadata.get(
                "chunk_id",
                "unknown",
            )

            context_parts.append(
                f"""
SOURCE {index}
Retrieval Type: Vector
Retrieval Source: {retrieval_source}
Chunk ID: {chunk_id}
File: {file_name}
Page: {page_number}
Section: {section}

{document.page_content}
""".strip()
            )

        # --------------------------------------------------------
        # Graph RAG document
        # --------------------------------------------------------

        elif retrieval_type == "graph":

            labels = metadata.get(
                "graph_labels",
                [],
            )

            name = metadata.get(
                "name",
                "unknown",
            )

            context_parts.append(
                f"""
SOURCE {index}
Retrieval Type: Graph
Retrieval Source: {retrieval_source}
Graph Labels: {labels}
Entity: {name}

{document.page_content}
""".strip()
            )

        # --------------------------------------------------------
        # Unknown / future retrieval source
        # --------------------------------------------------------

        else:

            context_parts.append(
                f"""
SOURCE {index}
Retrieval Type: {retrieval_type}
Retrieval Source: {retrieval_source}

{document.page_content}
""".strip()
            )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_company_knowledge_agent(
    llm: Any,
    retriever: DatabricksVectorSearchRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
):
    """
    Build the complete Company Knowledge RAG LangGraph.

    Components:

    1. Vector retrieval
    2. Cross-encoder reranking
    3. Evidence grading
    4. Corrective query rewriting
    5. Answer generation
    6. Self-RAG answer evaluation
    7. Self-correction
    8. Bounded retry
    9. Evidence construction
    """

    retriever = (
        retriever
        or DatabricksVectorSearchRetriever()
    )

    reranker = (
        reranker
        or CrossEncoderReranker(top_n=5)
    )

    # -----------------------------------------------------------------------
    # Structured LLM evaluators
    #
    # Groq JSON mode requires the messages to explicitly contain the word
    # "JSON". The prompts below therefore explicitly instruct the model to
    # return JSON.
    # -----------------------------------------------------------------------

    evidence_grader_llm = llm.with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    answer_grader_llm = llm.with_structured_output(
        SelfRAGAnswerGrade,
        method="json_mode",
    )

    # -----------------------------------------------------------------------
    # Retrieve
    # -----------------------------------------------------------------------

    def retrieve_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        query = state.get(
            "current_query",
            state["question"],
        )

        documents = retriever.invoke(query)

        return {
            "retrieved_docs": documents,
        }

    # -----------------------------------------------------------------------
    # Rerank
    # -----------------------------------------------------------------------

    def rerank_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        query = state.get(
            "current_query",
            state["question"],
        )

        documents = state.get(
            "retrieved_docs",
            [],
        )

        ranked = reranker.rerank(
            query=query,
            documents=documents,
        )

        final_documents = [
            document
            for document, _score in ranked
        ]

        return {
            "reranked_docs": final_documents,
        }

    # -----------------------------------------------------------------------
    # KB Evidence Grader
    # -----------------------------------------------------------------------

    def grade_evidence_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        question = state.get(
            "current_query",
            state["question"],
        )

        documents = state.get(
            "reranked_docs",
            [],
        )

        if not documents:
            return {
                "evidence_grade": "weak",
                "evidence_score": 0.0,
                "evidence_reason": (
                    "No approved Company Knowledge "
                    "evidence was retrieved."
                ),
            }

        context = _documents_to_context(
            documents
        )

        prompt = f"""
You are an enterprise Company Knowledge
retrieval evidence evaluator.

Evaluate ONLY the supplied private evidence.

Question:
{question}

Retrieved Company Knowledge Evidence:
{context}

Determine whether the evidence is sufficient
to answer the question.

Grade:

good:
- evidence is directly relevant
- evidence is sufficiently complete
- evidence can support a grounded answer

weak:
- evidence is missing
- evidence is irrelevant
- evidence is too incomplete
- evidence does not support the requested answer

Do not use outside knowledge.

Return the evaluation as valid JSON.

The JSON object must contain:
- "grade": either "good" or "weak"
- "score": a number from 0 to 1
- "reason": a concise explanation

Return ONLY valid JSON.
"""

        result = evidence_grader_llm.invoke(
            prompt
        )

        return {
            "evidence_grade": result.grade,
            "evidence_score": result.score,
            "evidence_reason": result.reason,
        }

    # -----------------------------------------------------------------------
    # Corrective RAG: Query Rewrite
    # -----------------------------------------------------------------------

    def rewrite_query_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        question = state["question"]

        current_query = state.get(
            "current_query",
            question,
        )

        retry_count = (
            state.get("retry_count", 0) + 1
        )

        prompt = f"""
You are a corrective enterprise RAG
retrieval-query optimizer.

The previous Company Knowledge retrieval
did not provide sufficiently strong evidence.

Original user question:
{question}

Previous retrieval query:
{current_query}

Create a better retrieval query.

Requirements:

1. Preserve the original user intent.
2. Add useful technical terminology.
3. Make the query more precise.
4. Focus on concepts likely to exist in the
   enterprise knowledge base.
5. Remove conversational wording.
6. Do not answer the question.
7. Do not invent facts.
8. Return ONLY the improved retrieval query.

Improved retrieval query:
"""

        result = llm.invoke(prompt)

        rewritten_query = getattr(
            result,
            "content",
            str(result),
        ).strip()

        if not rewritten_query:
            rewritten_query = current_query

        return {
            "current_query": rewritten_query,
            "retry_count": retry_count,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "answer_feedback": "",
        }

    # -----------------------------------------------------------------------
    # Generate Answer
    # -----------------------------------------------------------------------

    def generate_answer_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        question = state["question"]

        documents = state.get(
            "reranked_docs",
            [],
        )

        context = _documents_to_context(
            documents
        )

        prompt = f"""
You are an enterprise Company Knowledge assistant.

Answer the user's question using ONLY the
approved private Company Knowledge evidence.

Question:
{question}

Evidence:
{context}

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. Every factual claim must be supported by
   the supplied evidence.
4. If the evidence is insufficient, say so.
5. Keep the answer concise but useful.
6. Cite the source file and page when available.
7. Do not cite sources that are not present
   in the evidence.

Answer:
"""

        result = llm.invoke(prompt)

        answer = getattr(
            result,
            "content",
            str(result),
        ).strip()

        return {
            "answer": answer,
            "source_used": "company_knowledge",
        }

    # -----------------------------------------------------------------------
    # Self-RAG Answer Evaluation
    # -----------------------------------------------------------------------

    def evaluate_answer_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        question = state["question"]

        answer = state.get(
            "answer",
            "",
        )

        documents = state.get(
            "reranked_docs",
            [],
        )

        context = _documents_to_context(
            documents
        )

        prompt = f"""
You are a Self-RAG answer evaluator.

Evaluate the generated answer against:

1. The original user question.
2. The retrieved Company Knowledge evidence.

Question:
{question}

Generated Answer:
{answer}

Company Knowledge Evidence:
{context}

Evaluate:

groundedness_score:
How strongly is every factual claim supported
by the supplied evidence?

completeness_score:
Does the answer adequately address the
user's question?

citation_score:
Are source file/page references supported
by the supplied evidence?

Decision:

PASS only when:
- groundedness >= 0.80
- completeness >= 0.80
- citation quality >= 0.80

Otherwise:
RETRY

If RETRY, explain specifically what is missing,
unsupported, or incorrect.

Do not use outside knowledge.

Return the evaluation as valid JSON.

The JSON object must contain:
- "groundedness_score": number from 0 to 1
- "completeness_score": number from 0 to 1
- "citation_score": number from 0 to 1
- "decision": either "pass" or "retry"
- "feedback": a concise explanation

Return ONLY valid JSON.
"""

        result = answer_grader_llm.invoke(
            prompt
        )

        # Defensive normalization:
        # the LLM's structured decision remains
        # authoritative, but scores also enforce
        # the enterprise threshold.

        scores_pass = (
            result.groundedness_score
            >= MIN_GROUNDEDNESS_SCORE
            and result.completeness_score
            >= MIN_COMPLETENESS_SCORE
            and result.citation_score
            >= MIN_CITATION_SCORE
        )

        decision = (
            "pass"
            if result.decision == "pass"
            and scores_pass
            else "retry"
        )

        return {
            "groundedness_score": (
                result.groundedness_score
            ),
            "completeness_score": (
                result.completeness_score
            ),
            "citation_score": (
                result.citation_score
            ),
            "answer_grade": decision,
            "answer_feedback": result.feedback,
        }

    # -----------------------------------------------------------------------
    # Self-RAG Corrective Loop
    # -----------------------------------------------------------------------

    def self_correct_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        question = state["question"]

        current_query = state.get(
            "current_query",
            question,
        )

        feedback = state.get(
            "answer_feedback",
            "",
        )

        retry_count = (
            state.get("retry_count", 0) + 1
        )

        prompt = f"""
You are a Self-RAG corrective retrieval agent.

The generated answer failed evaluation.

Original question:
{question}

Previous retrieval query:
{current_query}

Self-RAG evaluator feedback:
{feedback}

Create a better retrieval query that specifically
addresses the evaluator's feedback.

Requirements:

1. Preserve the original intent.
2. Target missing evidence.
3. Remove unsupported assumptions.
4. Add precise enterprise terminology.
5. Improve retrieval specificity.
6. Do not answer the question.
7. Return ONLY the improved retrieval query.

Improved retrieval query:
"""

        result = llm.invoke(prompt)

        improved_query = getattr(
            result,
            "content",
            str(result),
        ).strip()

        if not improved_query:
            improved_query = current_query

        return {
            "current_query": improved_query,
            "retry_count": retry_count,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "answer_feedback": "",
        }

    # -----------------------------------------------------------------------
    # Insufficient Evidence
    # -----------------------------------------------------------------------

    def insufficient_node(
        state: CompanyKnowledgeState,
    ) -> CompanyKnowledgeState:

        return {
            "answer": (
                "I could not find sufficient approved "
                "Company Knowledge evidence to answer "
                "this question."
            ),
            "source_used": "company_knowledge",
            "evidence": [],
        }

    # -----------------------------------------------------------------------
    # Evidence Builder
    # -----------------------------------------------------------------------

    def build_evidence_node(
    state: CompanyKnowledgeState,
) -> CompanyKnowledgeState:

        documents = state.get(
            "reranked_docs",
            [],
        )

        evidence: List[
            Dict[str, Any]
        ] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = dict(
                document.metadata
            )

            retrieval_type = metadata.get(
                "retrieval_type",
                "unknown",
            )

            retrieval_source = metadata.get(
                "retrieval_source",
                "company_knowledge",
            )

            evidence.append(
                {
                    "evidence_id": (
                        f"company-evidence-{index}"
                    ),
                    "source": (
                        retrieval_source
                    ),
                    "tool": (
                        "company_knowledge_rag"
                    ),
                    "retrieval_type": (
                        retrieval_type
                    ),
                    "data": {
                        "chunk_id": metadata.get(
                            "chunk_id"
                        ),
                        "document_id": metadata.get(
                            "document_id"
                        ),
                        "file_name": metadata.get(
                            "file_name"
                        ),
                        "page_number": metadata.get(
                            "page_number"
                        ),
                        "section": metadata.get(
                            "section"
                        ),
                        "graph_labels": metadata.get(
                            "graph_labels"
                        ),
                        "entity_name": metadata.get(
                            "name"
                        ),
                        "chunk_text": (
                            document.page_content
                        ),
                    },
                    "metadata": metadata,
                }
            )

        return {
            "evidence": evidence,
        }

    # -----------------------------------------------------------------------
    # Routing functions
    # -----------------------------------------------------------------------

    def route_after_evidence(
        state: CompanyKnowledgeState,
    ) -> str:

        if (
            state.get("evidence_grade")
            == "good"
            and state.get("evidence_score", 0.0)
            >= MIN_EVIDENCE_SCORE
        ):
            return "generate_answer"

        if (
            state.get("retry_count", 0)
            < MAX_RETRIES
        ):
            return "rewrite_query"

        return "insufficient"

    def route_after_answer(
        state: CompanyKnowledgeState,
    ) -> str:

        if (
            state.get("answer_grade")
            == "pass"
        ):
            return "build_evidence"

        if (
            state.get("retry_count", 0)
            < MAX_RETRIES
        ):
            return "self_correct"

        return "insufficient"

    # -----------------------------------------------------------------------
    # Build LangGraph
    # -----------------------------------------------------------------------

    graph = StateGraph(
        CompanyKnowledgeState
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "rerank",
        rerank_node,
    )

    graph.add_node(
        "grade_evidence",
        grade_evidence_node,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    graph.add_node(
        "generate_answer",
        generate_answer_node,
    )

    graph.add_node(
        "evaluate_answer",
        evaluate_answer_node,
    )

    graph.add_node(
        "self_correct",
        self_correct_node,
    )

    graph.add_node(
        "build_evidence",
        build_evidence_node,
    )

    graph.add_node(
        "insufficient",
        insufficient_node,
    )

    # -----------------------------------------------------------------------
    # Initial retrieval
    # -----------------------------------------------------------------------

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "rerank",
    )

    graph.add_edge(
        "rerank",
        "grade_evidence",
    )

    # -----------------------------------------------------------------------
    # Corrective RAG routing
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(
        "grade_evidence",
        route_after_evidence,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # Corrective retrieval loop
    graph.add_edge(
        "rewrite_query",
        "retrieve",
    )

    # -----------------------------------------------------------------------
    # Answer generation
    # -----------------------------------------------------------------------

    graph.add_edge(
        "generate_answer",
        "evaluate_answer",
    )

    # -----------------------------------------------------------------------
    # Self-RAG routing
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_answer,
        {
            "build_evidence": "build_evidence",
            "self_correct": "self_correct",
            "insufficient": "insufficient",
        },
    )

    # Self-correction returns to retrieval
    graph.add_edge(
        "self_correct",
        "retrieve",
    )

    # -----------------------------------------------------------------------
    # Successful answer
    # -----------------------------------------------------------------------

    graph.add_edge(
        "build_evidence",
        END,
    )

    # -----------------------------------------------------------------------
    # Failed bounded retries
    # -----------------------------------------------------------------------

    graph.add_edge(
        "insufficient",
        END,
    )

    return graph.compile()