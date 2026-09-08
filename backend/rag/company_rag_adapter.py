"""
Adapter between enterprise AgentState and Company Knowledge RAG.

The Company Knowledge RAG graph owns its specialized retrieval state.
The enterprise graph owns the global AgentState.
"""

from __future__ import annotations

from typing import Any

from ..agents.graph_state import AgentState
from .company_rag_agent import (
    CompanyKnowledgeState,
    build_company_knowledge_agent,
)


def build_company_knowledge_node(
    llm: Any,
    retriever: Any = None,
    reranker: Any = None,
):
    """
    Return a LangGraph node compatible with AgentState.
    """

    rag_agent = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    async def company_knowledge_node(
        state: AgentState,
    ) -> AgentState:

        query = state["query"]

        rag_state: CompanyKnowledgeState = {
            "question": query,
            "current_query": query,
            "retry_count": 0,
            "retrieved_docs": [],
            "reranked_docs": [],
            "evidence": [],
        }

        result = await rag_agent.ainvoke(
            rag_state
        )

        return {
            "current_query": result.get(
                "current_query",
                query,
            ),
            "kb_docs": result.get(
                "reranked_docs",
                [],
            ),
            "answer": result.get(
                "answer",
            ),
            "response": result.get(
                "answer",
            ),
            "source_used": result.get(
                "source_used",
                "company_knowledge",
            ),
            "evidence": result.get(
                "evidence",
                [],
            ),
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
                "answer_grade",
            ),
            "answer_feedback": result.get(
                "answer_feedback",
                "",
            ),
        }

    return company_knowledge_node