from __future__ import annotations

from typing import Any

from ..agents.graph_state import AgentState
from ..evidence.evidence_normalizer import normalize_evidence
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
             Retrieval Cache
                  ↓
             Company RAG
                  ↓
              Reranking
                  ↓
          Corrective RAG / Self-RAG
                  ↓
          Evidence Normalizer

    A custom retriever can still be injected for tests or
    specialized deployments.
    """

    # Default production retrieval path
    if retriever is None:
        vector_retriever = DatabricksVectorSearchRetriever()
        graph_retriever = Neo4jGraphRAGRetriever()

        retriever = HybridRAGRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=graph_retriever,
        )

    # Build Company Knowledge RAG
    rag_agent = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    # Adapter: AgentState → CompanyKnowledgeState
    async def company_knowledge_node(
        state: AgentState,
    ) -> AgentState:
        """
        Process company knowledge RAG node.

        Authorization context for retrieval caching:
        - user_id is already established by the authentication layer
          before Company Knowledge RAG is reached.
        - This value is carried into CompanyKnowledgeState so the
          retrieval cache cannot accidentally share cached retrieval
          results between different users.
        """
        query = state["query"]

        authorization_context = state.get(
            "user_id",
            "anonymous",
        )

        rag_state: CompanyKnowledgeState = {
            "question": query,
            "current_query": query,
            "retry_count": 0,
            "retrieved_docs": [],
            "reranked_docs": [],
            "evidence": [],
            "authorization_context": authorization_context,
        }

        result = await rag_agent.ainvoke(rag_state)

        # Shared Evidence Layer
        normalized_evidence = normalize_evidence(
            result.get(
                "evidence",
                [],
            )
        )

        # Return updated AgentState
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
                "company_knowledge",
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
            "answer_grade": result.get("answer_grade"),
            "answer_feedback": result.get(
                "answer_feedback",
                "",
            ),
            "current_stage": "company_knowledge",
        }

    return company_knowledge_node