
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from backend.rag.policy_rag_adapter import build_policy_rag_node


class FakeEvidenceGrade:
    def __init__(
        self,
        grade: str = "good",
        score: float = 0.95,
        reason: str = (
            "The retrieved Policy evidence is directly "
            "relevant and sufficient."
        ),
    ) -> None:
        self.grade = grade
        self.score = score
        self.reason = reason



class FakeAnswerGrade:
    def __init__(
        self,
        groundedness_score: float = 0.95,
        completeness_score: float = 0.95,
        citation_score: float = 0.95,
        decision: str = "pass",
        feedback: str = (
            "The answer is grounded in the retrieved "
            "Policy evidence."
        ),
    ) -> None:
        self.groundedness_score = groundedness_score
        self.completeness_score = completeness_score
        self.citation_score = citation_score
        self.decision = decision
        self.feedback = feedback




class FakePolicyRetriever:
    def invoke(self, query: str) -> list[Document]:
        return [
            Document(
                page_content=(
                    "Fraud investigations must follow approved "
                    "investigation procedures."
                ),
                metadata={
                    "chunk_id": "policy-chunk-001",
                    "document_id": "policy-doc-001",
                    "file_name": "Fraud Investigation Policy.pdf",
                    "file_path": (
                        "/Volumes/enterprise_rag/source/rag_docs/"
                        "incoming/Fraud_policy_documents/"
                        "Fraud Investigation Policy.pdf"
                    ),
                    "page_number": 5,
                    "section": "Investigation Procedure",
                    "document_type": "policy",
                    "retrieval_type": "vector",
                    "retrieval_source": "databricks_vector_search",
                },
            )
        ]

    async def ainvoke(self, query: str) -> list[Document]:
        return self.invoke(query)


class FakePolicyReranker:
    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[tuple[Document, float]]:
        return [
            (document, 0.95)
            for document in documents
        ]

    def invoke(
        self,
        query: str,
        documents: list[Document],
    ) -> list[tuple[Document, float]]:
        return self.rerank(query, documents)


class FakeStructuredLLM:
    def __init__(
        self,
        response: Any,
    ) -> None:
        self.response = response

    def invoke(
        self,
        prompt: Any,
    ) -> Any:
        return self.response

    async def ainvoke(
        self,
        prompt: Any,
    ) -> Any:
        return self.response


class FakePolicyLLM:
    def with_structured_output(
        self,
        schema: Any,
        method: str | None = None,
        **kwargs: Any,
    ) -> FakeStructuredLLM:
        schema_name = getattr(
            schema,
            "__name__",
            "",
        )

        if schema_name == "EvidenceGrade":
            return FakeStructuredLLM(
                FakeEvidenceGrade()
            )

        return FakeStructuredLLM(
            FakeAnswerGrade()
        )

    async def ainvoke(
        self,
        prompt: Any,
    ) -> str:
        return (
            "Fraud investigations must follow approved "
            "investigation procedures."
        )

    def invoke(
        self,
        prompt: Any,
    ) -> str:
        return (
            "Fraud investigations must follow approved "
            "investigation procedures."
        )


@pytest.mark.asyncio
async def test_policy_rag_adapter_normalizes_evidence():
    node = build_policy_rag_node(
        llm=FakePolicyLLM(),
        retriever=FakePolicyRetriever(),
        reranker=FakePolicyReranker(),
    )

    state = {
        "query": (
            "What procedure must fraud investigations follow?"
        ),
        "current_stage": "policy_rag",
    }

    result = await node(state)

    assert "evidence" in result
    assert len(result["evidence"]) > 0

    evidence = result["evidence"][0]

    # Shared Evidence Layer fields.
    assert evidence["evidence_id"] == "policy-evidence-1"
    assert evidence["source_domain"] == "policy"
    assert evidence["source"] == "databricks_vector_search"
    assert evidence["retrieval_type"] == "vector"
    assert (
        evidence["retrieval_source"]
        == "databricks_vector_search"
    )
    assert evidence["tool"] == "policy_rag"

    # Existing evidence data must remain intact.
    assert (
        evidence["data"]["chunk_id"]
        == "policy-chunk-001"
    )
    assert (
        evidence["data"]["document_id"]
        == "policy-doc-001"
    )
    assert (
        evidence["data"]["file_name"]
        == "Fraud Investigation Policy.pdf"
    )
    assert evidence["data"]["page_number"] == 5
    assert (
        evidence["data"]["section"]
        == "Investigation Procedure"
    )
    assert (
        evidence["data"]["document_type"]
        == "policy"
    )

    # Existing AgentState fields must remain intact.
    assert result["current_stage"] == "policy_rag"
    assert (
        result["source_used"]
        == "policy_documents"
    )


@pytest.mark.asyncio
async def test_policy_rag_adapter_preserves_policy_answer():
    node = build_policy_rag_node(
        llm=FakePolicyLLM(),
        retriever=FakePolicyRetriever(),
        reranker=FakePolicyReranker(),
    )

    state = {
        "query": (
            "What procedure must fraud investigations follow?"
        ),
    }

    result = await node(state)

    assert result["answer"] is not None
    assert result["response"] == result["answer"]
    assert result["current_stage"] == "policy_rag"


@pytest.mark.asyncio
async def test_policy_rag_adapter_preserves_input_state():
    node = build_policy_rag_node(
        llm=FakePolicyLLM(),
        retriever=FakePolicyRetriever(),
        reranker=FakePolicyReranker(),
    )

    state = {
        "user_id": "user-001",
        "query": (
            "What procedure must fraud investigations follow?"
        ),
        "authenticated": True,
        "authorization_allowed": True,
    }

    result = await node(state)

    assert result["user_id"] == "user-001"
    assert result["query"] == state["query"]
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

