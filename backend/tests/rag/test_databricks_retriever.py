from unittest.mock import Mock

from backend.rag.databricks_retriever import (
    DatabricksVectorSearchRetriever,
)


def test_embedding_dimension_validation():
    retriever = DatabricksVectorSearchRetriever.__new__(
        DatabricksVectorSearchRetriever
    )

    response = {
        "predictions": [[0.1, 0.2]]
    }

    try:
        retriever._extract_embedding(response)
        assert False, "Expected dimension validation failure"
    except ValueError as exc:
        assert "1024" in str(exc)


def test_embedding_response_is_extracted():
    retriever = DatabricksVectorSearchRetriever.__new__(
        DatabricksVectorSearchRetriever
    )

    vector = [0.1] * 1024

    result = retriever._extract_embedding(
        {"predictions": [vector]}
    )

    assert len(result) == 1024
    assert result[0] == 0.1