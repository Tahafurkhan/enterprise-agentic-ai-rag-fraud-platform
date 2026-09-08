from unittest.mock import MagicMock

from langchain_core.documents import Document

from backend.rag.graph_rag_agent import (
    build_graph_rag_agent,
)


def test_graph_rag_agent():

    retriever = MagicMock()

    retriever.invoke.return_value = [
        Document(
            page_content=(
                "Quality management procedures "
                "are documented in the quality manual."
            ),
            metadata={
                "source": (
                    "company_knowledge_graph"
                ),
                "retrieval_type": "graph",
                "graph_labels": [
                    "Procedure"
                ],
            },
        )
    ]

    graph = build_graph_rag_agent(
        retriever=retriever,
    )

    result = graph.invoke(
        {
            "question": (
                "What are the quality management procedures?"
            ),
            "current_query": (
                "What are the quality management procedures?"
            ),
            "retrieved_docs": [],
            "evidence": [],
        }
    )

    assert len(
        result["retrieved_docs"]
    ) == 1

    assert len(
        result["evidence"]
    ) == 1

    assert (
        result["source_used"]
        == "company_knowledge_graph"
    )

    assert (
        result["evidence"][0][
            "retrieval_type"
        ]
        == "graph"
    )