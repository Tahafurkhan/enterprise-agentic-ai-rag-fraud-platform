
import os
import uuid

import pytest
from langchain_core.documents import Document

from backend.cache.redis_factory import build_redis_retrieval_cache
from backend.rag.policy_rag_agent import (
    RETRIEVAL_CACHE_VERSION,
    build_policy_rag_agent,
)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.documents


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append(query)

        return [
            (document, 1.0)
            for document in documents
        ]


class FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class FakeEvidenceResult:
    def __init__(
        self,
        grade="good",
        score=0.95,
        reason="Relevant Policy evidence.",
    ):
        self.grade = grade
        self.score = score
        self.reason = reason


class FakeAnswerResult:
    def __init__(
        self,
        groundedness_score=0.95,
        completeness_score=0.95,
        citation_score=0.95,
        decision="pass",
        feedback="Answer is grounded.",
    ):
        self.groundedness_score = groundedness_score
        self.completeness_score = completeness_score
        self.citation_score = citation_score
        self.decision = decision
        self.feedback = feedback


class FakeStructuredLLM:
    def __init__(
        self,
        evidence_result=None,
        answer_result=None,
    ):
        self.evidence_result = evidence_result
        self.answer_result = answer_result

    def invoke(self, prompt):
        if self.evidence_result is not None:
            return self.evidence_result

        return self.answer_result


class FakeLLM:
    def __init__(self):
        self.evidence_result = FakeEvidenceResult()
        self.answer_result = FakeAnswerResult()

    def with_structured_output(
        self,
        schema,
        method=None,
    ):
        name = getattr(
            schema,
            "__name__",
            "",
        )

        if name == "EvidenceGrade":
            return FakeStructuredLLM(
                evidence_result=self.evidence_result
            )

        return FakeStructuredLLM(
            answer_result=self.answer_result
        )

    def invoke(self, prompt):
        return FakeLLMResponse(
            "Approved Policy answer."
        )


def test_policy_rag_uses_real_redis_cache():
    if os.getenv("RETRIEVAL_CACHE_BACKEND") != "redis":
        pytest.skip(
            "Redis E2E test requires RETRIEVAL_CACHE_BACKEND=redis"
        )

    namespace = (
        f"policy_redis_e2e_{uuid.uuid4().hex}"
    )

    query = (
        "What is the policy for corrective action?"
    )

    authorization_context = "policy-redis-e2e-user"

    documents = [
        Document(
            page_content=(
                "The Policy requires documented "
                "corrective action."
            ),
            metadata={
                "document_id": "policy-redis-e2e-doc",
                "file_name": "Policy Manual.pdf",
                "page_number": 35,
                "section": "Corrective Action",
            },
        )
    ]

    cache = build_redis_retrieval_cache(
        ttl_seconds=300,
        namespace=namespace,
    )

    retriever = FakeRetriever(documents)
    reranker = FakeReranker()
    llm = FakeLLM()

    agent = build_policy_rag_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        retrieval_cache=cache,
    )

    state = {
        "question": query,
        "current_query": query,
        "authorization_context": authorization_context,
        "retry_count": 0,
    }

    try:
        # First invocation:
        # Redis cache miss -> real retriever is called.
        first_result = agent.invoke(state)

        assert retriever.calls == [query]

        assert (
            first_result["retrieved_docs"]
            == documents
        )

        # Verify that the retrieved Policy documents
        # were actually written to Redis.
        cached_documents = cache.get(
            domain="policy",
            query=query,
            authorization_context=authorization_context,
            retrieval_version=RETRIEVAL_CACHE_VERSION,
        )

        assert cached_documents == documents

        # Second invocation:
        # Redis cache hit -> retriever must NOT be called again.
        second_result = agent.invoke(state)

        assert (
            second_result["retrieved_docs"]
            == documents
        )

        assert retriever.calls == [query]

        # The isolated namespace should contain exactly
        # one cached retrieval.
        assert cache.size() == 1

    finally:
        # Remove only this test's namespace.
        cache.clear()

