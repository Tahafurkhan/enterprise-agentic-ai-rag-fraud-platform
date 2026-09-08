
"""
Databricks Vector Search retriever for Company Knowledge RAG.

The existing Vector Search index uses self-managed embeddings generated
with BAAI/bge-large-en-v1.5. Therefore every query must be embedded with
the same model before querying the index.
"""

from __future__ import annotations

from typing import Any, Dict, List

from databricks.vector_search.client import VectorSearchClient
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class DatabricksVectorSearchRetriever:
    """
    Retrieves Company Knowledge documents from Databricks Vector Search.
    """

    INDEX_NAME = "enterprise_rag.gold.rag_chunks_vs_index"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024

    DEFAULT_COLUMNS = [
        "chunk_id",
        "document_id",
        "file_name",
        "file_path",
        "page_number",
        "section",
        "source_element_id",
        "chunk_index_in_element",
        "chunk_text",
        "chunk_tokens",
        "content_type",
        "quality_score",
        "quality_passed",
        "coherence",
        "semantic_validity",
        "structure_integrity",
        "quality_reason",
        "element_type",
        "file_type",
        "processing_timestamp",
    ]

    def __init__(
        self,
        index_name: str | None = None,
        embedding_model: str | None = None,
        num_results: int = 10,
        client: VectorSearchClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.index_name = index_name or self.INDEX_NAME
        self.num_results = num_results

        self.client = client or VectorSearchClient()

        self.model = model or SentenceTransformer(
            embedding_model or self.EMBEDDING_MODEL
        )

    @classmethod
    def _extract_embedding(cls, response: Dict[str, Any]) -> List[float]:
        """
        Extract and validate a single embedding vector.

        This helper supports embedding-service style responses such as:

            {
                "predictions": [
                    [0.1, 0.2, ...]
                ]
            }

        The vector must contain exactly 1024 dimensions.
        """

        if not isinstance(response, dict):
            raise ValueError(
                "Embedding response must be a dictionary."
            )

        predictions = response.get("predictions")

        if not isinstance(predictions, list) or not predictions:
            raise ValueError(
                "Embedding response does not contain predictions."
            )

        embedding = predictions[0]

        if not isinstance(embedding, (list, tuple)):
            raise ValueError(
                "Embedding prediction must be a list or tuple."
            )

        vector = [float(value) for value in embedding]

        if len(vector) != cls.EMBEDDING_DIMENSION:
            raise ValueError(
                "Invalid embedding dimension. "
                f"Expected {cls.EMBEDDING_DIMENSION}, "
                f"received {len(vector)}."
            )

        return vector

    def _embed_query(self, query: str) -> List[float]:
        """
        Generate a query embedding using the same BGE model used
        to create the stored Vector Search embeddings.

        Normalization is intentionally not enabled here because the
        existing validated notebook query uses:

            model.encode([query])[0].tolist()
        """

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        embedding = self.model.encode([query])[0]

        vector = embedding.tolist()

        if len(vector) != self.EMBEDDING_DIMENSION:
            raise ValueError(
                "Invalid embedding dimension. "
                f"Expected {self.EMBEDDING_DIMENSION}, "
                f"received {len(vector)}."
            )

        return vector

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve candidate documents using semantic vector search.
        """

        query_vector = self._embed_query(query)

        index = self.client.get_index(
            index_name=self.index_name
        )

        results = index.similarity_search(
            query_vector=query_vector,
            columns=self.DEFAULT_COLUMNS,
            num_results=self.num_results,
        )

        return self._to_documents(results)

    @staticmethod
    def _to_documents(results: Dict[str, Any]) -> List[Document]:
        """
        Convert Databricks Vector Search data_array results into
        LangChain Documents.

        The current Databricks SDK returns rows as positional lists,
        while column names are provided by result.manifest.columns.
        """

        documents: List[Document] = []

        if not isinstance(results, dict):
            return documents

        result = results.get("result", {})

        if not isinstance(result, dict):
            return documents

        rows = result.get("data_array", [])
        manifest = results.get("manifest", {})

        columns = manifest.get("columns", [])

        column_names = [
            column.get("name")
            for column in columns
            if isinstance(column, dict) and column.get("name")
        ]

        if not column_names:
            return documents

        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue

            row_data = dict(
                zip(column_names, row)
            )

            chunk_text = row_data.get("chunk_text")

            if not chunk_text:
                continue

            metadata = {
                key: value
                for key, value in row_data.items()
                if key != "chunk_text"
            }

            documents.append(
                Document(
                    page_content=str(chunk_text),
                    metadata=metadata,
                )
            )

        return documents

    def invoke(self, query: str) -> List[Document]:
        """
        LangChain-style retriever interface.
        """

        return self.retrieve(query)

