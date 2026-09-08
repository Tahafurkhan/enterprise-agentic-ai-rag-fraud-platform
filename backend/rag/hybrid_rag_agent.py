"""
LangGraph Hybrid RAG Agent.

Combines:
- Databricks Vector Search
- Neo4j Graph RAG

The Hybrid RAG Agent is responsible for:
- Executing hybrid retrieval
- Building normalized evidence
- Preserving retrieval source information
- Returning retrieval metadata

It does not:
- Generate answers
- Call MCP
- Perform external web search
- Modify the existing Company Knowledge RAG graph

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
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


class HybridRAGState(TypedDict, total=False):
    """
    State used by the Hybrid RAG subgraph.
    """

    question: str
    current_query: str

    retrieved_docs: List[Document]
    evidence: List[Dict[str, Any]]

    vector_documents: int
    graph_documents: int
    total_documents: int

    source_used: str
    retrieval_type: str

    error: str


def retrieve_node(
    state: HybridRAGState,
    retriever: Any,
) -> HybridRAGState:
    """
    Execute hybrid retrieval.
    """

    query = state.get(
        "current_query",
        state.get("question", ""),
    )

    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    documents = retriever.invoke(query)

    if documents is None:
        documents = []

    if not isinstance(documents, list):
        raise TypeError(
            "Hybrid retriever must return a list of Documents."
        )

    vector_count = sum(
        1
        for document in documents
        if isinstance(document, Document)
        and document.metadata.get("retrieval_type") == "vector"
    )

    graph_count = sum(
        1
        for document in documents
        if isinstance(document, Document)
        and document.metadata.get("retrieval_type") == "graph"
    )

    valid_documents = [
        document
        for document in documents
        if isinstance(document, Document)
    ]

    return {
        **state,
        "current_query": query.strip(),
        "retrieved_docs": valid_documents,
        "vector_documents": vector_count,
        "graph_documents": graph_count,
        "total_documents": len(valid_documents),
        "source_used": "hybrid",
        "retrieval_type": "hybrid",
    }


def build_evidence_node(
    state: HybridRAGState,
) -> HybridRAGState:
    """
    Convert retrieved Documents into normalized evidence objects.
    """

    documents = state.get(
        "retrieved_docs",
        [],
    )

    evidence: List[Dict[str, Any]] = []

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
            "unknown",
        )

        evidence.append(
            {
                "evidence_id": f"hybrid-evidence-{index}",
                "source": retrieval_source,
                "retrieval_type": retrieval_type,
                "content": document.page_content,
                "metadata": metadata,
            }
        )

    return {
        **state,
        "evidence": evidence,
    }


def build_hybrid_rag_agent(
    retriever: Any,
):
    """
    Build the LangGraph Hybrid RAG agent.
    """

    if retriever is None:
        raise ValueError(
            "Hybrid retriever is required."
        )

    builder = StateGraph(
        HybridRAGState
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