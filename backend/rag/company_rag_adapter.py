from __future__ import annotations

from typing import Any

from ..agents.graph_state import AgentState
from .company_rag_agent import (
    CompanyKnowledgeState,
    build_company_knowledge_agent,
)
from .databricks_retriever import DatabricksVectorSearchRetriever
from .graph_rag_retriever import Neo4jGraphRAGRetriever
from .hybrid_rag_retriever import HybridRAGRetriever


def build_company_knowledge_node(
    llm: Any,
    retriever: Any = None,
    reranker: Any = None,
):
    """
    Build the Company Knowledge RAG node.

    By default, Company Knowledge uses Hybrid RAG:

        Databricks Vector Search
                  +
             Neo4j Graph RAG
                  ↓
           Hybrid RAG Retriever
                  ↓
             Company RAG
                  ↓
              Reranking
                  ↓
          Corrective RAG / Self-RAG

    A custom retriever can still be injected for tests or
    specialized deployments.
    """

    # ------------------------------------------------------------
    # Default production retrieval path
    # ------------------------------------------------------------

    if retriever is None:
        vector_retriever = DatabricksVectorSearchRetriever()
        graph_retriever = Neo4jGraphRAGRetriever()

        retriever = HybridRAGRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
        )

    # ------------------------------------------------------------
    # Build Company Knowledge RAG
    # ------------------------------------------------------------

    rag_agent = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    # ------------------------------------------------------------
    # Adapter: AgentState → CompanyKnowledgeState
    # ------------------------------------------------------------

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
            **state,
            "current_query": result.get(
                "current_query",
                query,
            ),
            "kb_docs": result.get(
                "reranked_docs",
                [],
            ),
            "answer": result.get(
                "answer"
            ),
            "response": result.get(
                "answer"
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
                "answer_grade"
            ),
            "answer_feedback": result.get(
                "answer_feedback",
                "",
            ),
            "current_stage": "company_knowledge",
        }

    return company_knowledge_node