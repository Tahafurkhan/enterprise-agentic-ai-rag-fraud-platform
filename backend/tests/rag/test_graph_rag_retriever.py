from unittest.mock import MagicMock

from backend.rag.graph_rag_retriever import (
    Neo4jGraphRAGRetriever,
)


def build_mock_driver():

    driver = MagicMock()

    session = MagicMock()

    driver.session.return_value.__enter__.return_value = (
        session
    )

    return driver, session


def test_graph_rag_retriever_returns_documents():

    driver, session = build_mock_driver()

    session.run.return_value = [
        {
            "labels": ["Procedure"],
            "properties": {
                "name": "Quality Management Procedure",
                "description": (
                    "Procedure for quality management."
                ),
            },
        }
    ]

    retriever = Neo4jGraphRAGRetriever(
        driver=driver,
        top_k=5,
    )

    documents = retriever.retrieve(
        "quality management"
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        document.metadata["retrieval_type"]
        == "graph"
    )

    assert (
        document.metadata["source"]
        == "company_knowledge_graph"
    )

    assert (
        "Quality Management Procedure"
        in document.page_content
    )


def test_graph_rag_retriever_rejects_empty_query():

    driver = MagicMock()

    retriever = Neo4jGraphRAGRetriever(
        driver=driver,
    )

    try:

        retriever.retrieve("")

        assert False

    except ValueError as exc:

        assert (
            "Query must not be empty"
            in str(exc)
        )


def test_graph_rag_retriever_invoke():

    driver, session = build_mock_driver()

    session.run.return_value = []

    retriever = Neo4jGraphRAGRetriever(
        driver=driver,
    )

    documents = retriever.invoke(
        "quality"
    )

    assert documents == []