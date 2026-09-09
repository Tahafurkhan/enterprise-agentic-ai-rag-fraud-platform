from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document


class HybridRAGRetriever:
    """
    Hybrid retriever combining:

        Databricks Vector Search
                  +
             Neo4j Graph RAG
                  ↓
            Hybrid Results
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
            raise ValueError("vector_retriever is required")

        if graph_retriever is None:
            raise ValueError("graph_retriever is required")

        if vector_top_k <= 0:
            raise ValueError("vector_top_k must be greater than 0")

        if graph_top_k <= 0:
            raise ValueError("graph_top_k must be greater than 0")

        if final_top_k <= 0:
            raise ValueError("final_top_k must be greater than 0")

        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents from both vector and graph sources.
        """

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        vector_documents = self._retrieve(
            retriever=self.vector_retriever,
            query=query,
        )

        graph_documents = self._retrieve(
            retriever=self.graph_retriever,
            query=query,
        )

        vector_documents = vector_documents[: self.vector_top_k]
        graph_documents = graph_documents[: self.graph_top_k]

        merged_documents = self._merge_documents(
            vector_documents=vector_documents,
            graph_documents=graph_documents,
        )

        deduplicated_documents = self._deduplicate_documents(
            merged_documents
        )

        return deduplicated_documents[: self.final_top_k]

    def _retrieve(
        self,
        retriever: Any,
        query: str,
    ) -> List[Document]:
        """
        Support retrievers exposing either invoke() or retrieve().
        """

        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(query)

        elif hasattr(retriever, "retrieve"):
            documents = retriever.retrieve(query)

        else:
            raise TypeError(
                "Retriever must implement invoke() or retrieve()"
            )

        if documents is None:
            return []

        if not isinstance(documents, list):
            documents = list(documents)

        for document in documents:
            if not isinstance(document, Document):
                raise TypeError(
                    "Retriever must return LangChain Document objects"
                )

        return documents

    def _merge_documents(
        self,
        vector_documents: List[Document],
        graph_documents: List[Document],
    ) -> List[Document]:
        """
        Merge vector and graph documents while adding normalized
        retrieval metadata.
        """

        merged: List[Document] = []

        for rank, document in enumerate(
            vector_documents,
            start=1,
        ):
            metadata = dict(document.metadata or {})

            retrieval_type = metadata.get("retrieval_type")
            if not retrieval_type:
                metadata["retrieval_type"] = "vector"

            retrieval_source = metadata.get("retrieval_source")
            if not retrieval_source:
                metadata["retrieval_source"] = (
                    "databricks_vector_search"
                )

            metadata["retrieval_rank"] = rank

            merged.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        for rank, document in enumerate(
            graph_documents,
            start=1,
        ):
            metadata = dict(document.metadata or {})

            retrieval_type = metadata.get("retrieval_type")
            if not retrieval_type:
                metadata["retrieval_type"] = "graph"

            retrieval_source = metadata.get("retrieval_source")
            if not retrieval_source:
                metadata["retrieval_source"] = (
                    "neo4j_graph_rag"
                )

            metadata["retrieval_rank"] = rank

            merged.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return merged

    def _deduplicate_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Deduplicate documents using:

        1. chunk_id
        2. document_id
        3. normalized page content
        """

        seen = set()
        unique_documents: List[Document] = []

        for document in documents:
            metadata = document.metadata or {}

            chunk_id = metadata.get("chunk_id")
            document_id = metadata.get("document_id")

            if chunk_id:
                dedup_key = (
                    "chunk_id",
                    str(chunk_id),
                )

            elif document_id:
                dedup_key = (
                    "document_id",
                    str(document_id),
                )

            else:
                normalized_content = " ".join(
                    document.page_content.split()
                ).lower()

                dedup_key = (
                    "content",
                    normalized_content,
                )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            unique_documents.append(document)

        return unique_documents

    def invoke(self, query: str) -> List[Document]:
        """
        LangChain-compatible invocation.
        """

        return self.retrieve(query)

    def retrieve_with_metadata(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Retrieve documents together with source statistics.
        """

        documents = self.retrieve(query)

        vector_count = sum(
            1
            for document in documents
            if document.metadata.get("retrieval_type")
            == "vector"
        )

        graph_count = sum(
            1
            for document in documents
            if document.metadata.get("retrieval_type")
            == "graph"
        )

        return {
            "query": query,
            "documents": documents,

            # Preserve the existing test/API contract.
            "document_count": len(documents),

            # Additional descriptive statistics.
            "total_documents": len(documents),
            "vector_documents": vector_count,
            "graph_documents": graph_count,
        }