from __future__ import annotations

from typing import Any

from ..agents.graph_state import AgentState
from ..evidence.evidence_normalizer import normalize_evidence
from .policy_rag_agent import (
    PolicyKnowledgeState,
    build_policy_rag_agent,
)
from .policy_rag_retriever import PolicyVectorSearchRetriever


def build_policy_rag_node(
    llm: Any,
    retriever: Any = None,
    reranker: Any = None,
):
    """
    Build the Policy RAG node and adapt it to enterprise AgentState.
    
    Creates a Policy RAG agent that retrieves and evaluates policy
    documents to answer user queries.
    """

    if retriever is None:
        retriever = PolicyVectorSearchRetriever()

    rag_agent = build_policy_rag_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    async def policy_rag_node(state: AgentState) -> AgentState:
        """
        Process policy RAG node.
        
        Converts enterprise AgentState to PolicyKnowledgeState,
        runs the RAG agent, normalizes evidence, and returns
        updated AgentState.
        """
        query = state["query"]

        rag_state: PolicyKnowledgeState = {
            "question": query,
            "current_query": query,
            "authorization_context": state.get("user_id", "anonymous"),
            "retry_count": 0,
            "retrieved_docs": [],
            "reranked_docs": [],
            "evidence": [],
        }

        result = await rag_agent.ainvoke(rag_state)

        # Shared Evidence Layer
        normalized_evidence = normalize_evidence(
            result.get("evidence", [])
        )

        return {
            **state,
            "current_query": result.get(
                "current_query",
                query,
            ),
            "kb_docs": result.get(
                "reranked_docs",
                [],
            ),
            "answer": result.get("answer"),
            "response": result.get("answer"),
            "source_used": result.get(
                "source_used",
                "policy_documents",
            ),
            "evidence": normalized_evidence,
            "retry_count": result.get(
                "retry_count",
                0,
            ),
            "groundedness_score": result.get(
                "groundedness_score",
                0.0,
            ),
            "completeness_score": result.get(
                "completeness_score",
                0.0,
            ),
            "citation_score": result.get(
                "citation_score",
                0.0,
            ),
            "answer_grade": result.get(
                "answer_grade"
            ),
            "answer_feedback": result.get(
                "answer_feedback",
                "",
            ),
            "current_stage": "policy_rag",
        }

    return policy_rag_node