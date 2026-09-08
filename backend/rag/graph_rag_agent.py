"""
LangGraph Graph RAG Agent.

Architecture:

    START
      |
      v
    retrieve
      |
      v
    build_evidence
      |
      v
     END

This is intentionally a retrieval-focused subgraph.

Corrective RAG and Self-RAG remain owned by the existing
Company Knowledge RAG pipeline until Hybrid RAG is introduced.
"""

from __future__ import annotations

from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


class GraphRAGState(TypedDict, total=False):

    question: str
    current_query: str

    retrieved_docs: List[Document]

    evidence: List[dict[str, Any]]

    source_used: str

    error: str


# ----------------------------------------------------------------------
# Retrieve
# ----------------------------------------------------------------------

def retrieve_node(
    state: GraphRAGState,
    retriever: Any,
) -> GraphRAGState:

    query = state.get(
        "current_query",
        state.get("question", ""),
    )

    documents = retriever.invoke(
        query
    )

    return {
        **state,
        "retrieved_docs": documents,
        "source_used": "company_knowledge_graph",
    }


# ----------------------------------------------------------------------
# Evidence
# ----------------------------------------------------------------------

def build_evidence_node(
    state: GraphRAGState,
) -> GraphRAGState:

    documents = state.get(
        "retrieved_docs",
        [],
    )

    evidence = []

    for index, document in enumerate(
        documents
    ):

        evidence.append(
            {
                "evidence_id": (
                    f"graph-evidence-{index + 1}"
                ),
                "source": (
                    document.metadata.get(
                        "source",
                        "company_knowledge_graph",
                    )
                ),
                "retrieval_type": "graph",
                "content": document.page_content,
                "metadata": dict(
                    document.metadata
                ),
            }
        )

    return {
        **state,
        "evidence": evidence,
    }


# ----------------------------------------------------------------------
# Graph Builder
# ----------------------------------------------------------------------

def build_graph_rag_agent(
    retriever: Any,
):
    """
    Build the Graph RAG LangGraph subgraph.
    """

    builder = StateGraph(
        GraphRAGState
    )

    builder.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            retriever,
        ),
    )

    builder.add_node(
        "build_evidence",
        build_evidence_node,
    )

    builder.add_edge(
        START,
        "retrieve",
    )

    builder.add_edge(
        "retrieve",
        "build_evidence",
    )

    builder.add_edge(
        "build_evidence",
        END,
    )

    return builder.compile()