"""
Cross-encoder reranker for Company Knowledge RAG.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            (document, float(score))
            for document, score in ranked[:self.top_n]
        ]