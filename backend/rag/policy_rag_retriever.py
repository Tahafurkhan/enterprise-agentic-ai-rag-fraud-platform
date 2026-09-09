"""
Policy-scoped Databricks Vector Search retriever.

The shared Vector Search index contains both Company and Policy chunks.
This retriever deliberately queries the shared index with a larger candidate
set and then retains only documents belonging to the configured Policy
source folder.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from databricks.vector_search.client import VectorSearchClient
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class PolicyVectorSearchRetriever:
    """Retrieve only approved Policy documents from the shared index."""

    INDEX_NAME = "enterprise_rag.gold.rag_chunks_vs_index"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024
    DEFAULT_CANDIDATE_RESULTS = 100

    # This is the actual Policy ingestion scope already used in Databricks.
    DEFAULT_POLICY_SOURCE_PATH = (
        "/Volumes/enterprise_rag/source/rag_docs/incoming/"
        "Fraud_policy_documents/"
    )

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
        num_results: int | None = None,
        policy_source_path: str | None = None,
        client: VectorSearchClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.index_name = index_name or self.INDEX_NAME
        self.num_results = max(
            int(
                num_results
                or os.getenv(
                    "POLICY_RAG_CANDIDATE_RESULTS",
                    self.DEFAULT_CANDIDATE_RESULTS,
                )
            ),
            1,
        )
        self.policy_source_path = self._normalize_path(
            policy_source_path
            or os.getenv(
                "POLICY_RAG_SOURCE_PATH",
                self.DEFAULT_POLICY_SOURCE_PATH,
            )
        )

        self.client = client or VectorSearchClient()
        self.model = model or SentenceTransformer(
            embedding_model or self.EMBEDDING_MODEL
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").lower()

    def _is_policy_document(self, file_path: Any) -> bool:
        if not file_path:
            return False

        normalized_file_path = self._normalize_path(str(file_path))
        return normalized_file_path.startswith(
            self.policy_source_path
        )

    def _embed_query(self, query: str) -> List[float]:
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
        """Retrieve shared-index candidates and keep only Policy chunks."""

        query_vector = self._embed_query(query)
        index = self.client.get_index(index_name=self.index_name)

        results = index.similarity_search(
            query_vector=query_vector,
            columns=self.DEFAULT_COLUMNS,
            num_results=self.num_results,
        )

        documents = self._to_documents(results)

        policy_documents = [
            document
            for document in documents
            if self._is_policy_document(
                document.metadata.get("file_path")
            )
        ]

        return policy_documents

    @staticmethod
    def _to_documents(results: Dict[str, Any]) -> List[Document]:
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

            row_data = dict(zip(column_names, row))
            chunk_text = row_data.get("chunk_text")

            if not chunk_text:
                continue

            metadata = {
                key: value
                for key, value in row_data.items()
                if key != "chunk_text"
            }

            metadata["knowledge_domain"] = "policy"
            metadata["namespace"] = "policy_knowledge"
            metadata["source"] = "policy_documents"
            metadata["retrieval_type"] = "vector"
            metadata["retrieval_source"] = "databricks_vector_search"

            documents.append(
                Document(
                    page_content=str(chunk_text),
                    metadata=metadata,
                )
            )

        return documents

    def invoke(self, query: str) -> List[Document]:
        return self.retrieve(query)
