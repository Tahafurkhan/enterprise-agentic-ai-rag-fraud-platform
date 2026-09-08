"""
Hybrid RAG Retriever.

Combines:
- Databricks Vector Search
- Neo4j Graph RAG

The retriever performs retrieval only.

It does not:
- generate answers
- call MCP
- perform external web search
- modify the existing Company Knowledge RAG graph

Architecture:

                    User Query
                        |
              +---------+---------+
              |                   |
              v                   v
      Databricks Vector       Neo4j Graph
          Search                 RAG
              |                   |
              +---------+---------+
                        |
                        v
                 Merge Evidence
                        |
                        v
                  Deduplicate
                        |
                        v
                 Hybrid Documents
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


class HybridRAGRetriever:
    """
    Combines multiple retrieval systems into one retrieval interface.

    The retrievers are injected so the class remains independent from
    concrete Databricks or Neo4j implementations.

    Expected retriever interface:

        retriever.invoke(query) -> List[Document]

    or:

        retriever.retrieve(query) -> List[Document]
    """

    def __init__(
        self,
        vector_retriever: Any,
        graph_retriever: Any,
        vector_top_k: int = 5,
        graph_top_k: int = 5,
        final_top_k: int = 10,
    ) -> None:
        if vector_retriever is None:
            raise ValueError("vector_retriever is required.")

        if graph_retriever is None:
            raise ValueError("graph_retriever is required.")

        if vector_top_k < 1:
            raise ValueError("vector_top_k must be at least 1.")

        if graph_top_k < 1:
            raise ValueError("graph_top_k must be at least 1.")

        if final_top_k < 1:
            raise ValueError("final_top_k must be at least 1.")

        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents from both retrieval systems and merge them.
        """

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        normalized_query = query.strip()

        vector_documents = self._retrieve(
            self.vector_retriever,
            normalized_query,
        )

        graph_documents = self._retrieve(
            self.graph_retriever,
            normalized_query,
        )

        vector_documents = vector_documents[: self.vector_top_k]
        graph_documents = graph_documents[: self.graph_top_k]

        merged_documents = self._merge_documents(
            vector_documents,
            graph_documents,
        )

        deduplicated_documents = self._deduplicate_documents(
            merged_documents
        )

        return deduplicated_documents[: self.final_top_k]

    def invoke(self, query: str) -> List[Document]:
        """
        LangChain-compatible invocation method.
        """

        return self.retrieve(query)

    @staticmethod
    def _retrieve(
        retriever: Any,
        query: str,
    ) -> List[Document]:
        """
        Call either invoke() or retrieve() on a retriever.
        """

        if hasattr(retriever, "invoke"):
            result = retriever.invoke(query)
        elif hasattr(retriever, "retrieve"):
            result = retriever.retrieve(query)
        else:
            raise TypeError(
                "Retriever must provide invoke() or retrieve()."
            )

        if result is None:
            return []

        if not isinstance(result, list):
            raise TypeError(
                "Retriever must return a list of Documents."
            )

        return [
            document
            for document in result
            if isinstance(document, Document)
        ]

    @staticmethod
    def _merge_documents(
        vector_documents: Sequence[Document],
        graph_documents: Sequence[Document],
    ) -> List[Document]:
        """
        Merge vector and graph retrieval results.

        Source metadata is added so downstream components know
        where each piece of evidence originated.
        """

        merged: List[Document] = []

        for rank, document in enumerate(vector_documents, start=1):
            metadata = dict(document.metadata)

            metadata.setdefault(
                "retrieval_type",
                "vector",
            )

            metadata.setdefault(
                "retrieval_source",
                "databricks_vector_search",
            )

            metadata["retrieval_rank"] = rank

            merged.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        for rank, document in enumerate(graph_documents, start=1):
            metadata = dict(document.metadata)

            metadata.setdefault(
                "retrieval_type",
                "graph",
            )

            metadata.setdefault(
                "retrieval_source",
                "neo4j_graph_rag",
            )

            metadata["retrieval_rank"] = rank

            merged.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return merged

    @staticmethod
    def _deduplicate_documents(
        documents: Sequence[Document],
    ) -> List[Document]:
        """
        Remove duplicate evidence.

        Deduplication first uses a stable document/chunk identifier
        when available. Otherwise the normalized page content is used.
        """

        unique_documents: List[Document] = []
        seen_keys: set[Tuple[str, str]] = set()

        for document in documents:
            metadata = document.metadata

            chunk_id = metadata.get("chunk_id")
            document_id = metadata.get("document_id")

            if chunk_id:
                key = (
                    "chunk_id",
                    str(chunk_id),
                )
            elif document_id:
                key = (
                    "document_id",
                    str(document_id),
                )
            else:
                normalized_content = " ".join(
                    document.page_content.lower().split()
                )

                key = (
                    "content",
                    normalized_content,
                )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            unique_documents.append(document)

        return unique_documents

    def retrieve_with_metadata(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Retrieve documents and return retrieval statistics.

        This method is useful for debugging, evaluation, and
        observability without changing the normal invoke() contract.
        """

        documents = self.retrieve(query)

        vector_count = sum(
            1
            for document in documents
            if document.metadata.get("retrieval_type") == "vector"
        )

        graph_count = sum(
            1
            for document in documents
            if document.metadata.get("retrieval_type") == "graph"
        )

        return {
            "query": query.strip(),
            "documents": documents,
            "document_count": len(documents),
            "vector_documents": vector_count,
            "graph_documents": graph_count,
        }