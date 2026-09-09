
import os
import uuid

import pytest
from langchain_core.documents import Document

from backend.cache.redis_factory import build_redis_retrieval_cache
from backend.rag.company_rag_agent import build_company_knowledge_agent


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
        reason="Relevant evidence.",
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
        answer_results=None,
    ):
        self.evidence_result = evidence_result
        self.answer_result = answer_result
        self.answer_results = answer_results
        self.answer_index = 0

    def invoke(self, prompt):
        if self.evidence_result is not None:
            return self.evidence_result

        if self.answer_results is not None:
            result = self.answer_results[
                min(
                    self.answer_index,
                    len(self.answer_results) - 1,
                )
            ]
            self.answer_index += 1
            return result

        return self.answer_result


class FakeLLM:
    def __init__(
        self,
        answer="Approved answer.",
        evidence_result=None,
        answer_result=None,
        rewritten_query="improved enterprise retrieval query",
    ):
        self.answer = answer
        self.evidence_result = (
            evidence_result
            or FakeEvidenceResult()
        )
        self.answer_result = (
            answer_result
            or FakeAnswerResult()
        )
        self.rewritten_query = rewritten_query

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
        if (
            "improved retrieval query"
            in prompt.lower()
        ):
            return FakeLLMResponse(
                self.rewritten_query
            )

        if (
            "better retrieval query"
            in prompt.lower()
        ):
            return FakeLLMResponse(
                self.rewritten_query
            )

        return FakeLLMResponse(
            self.answer
        )


def test_company_rag_uses_real_redis_cache():
    if os.getenv("RETRIEVAL_CACHE_BACKEND") != "redis":
        pytest.skip(
            "Redis E2E test requires RETRIEVAL_CACHE_BACKEND=redis"
        )

    namespace = (
        f"company_redis_e2e_{uuid.uuid4().hex}"
    )

    query = "redis company retrieval e2e test"
    authorization_context = "redis-e2e-user"
    retrieval_version = "company_hybrid_v1"

    documents = [
        Document(
            page_content=(
                "Redis E2E company knowledge document."
            ),
            metadata={
                "document_id": "redis-e2e-doc",
                "source": "redis-e2e",
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

    agent = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        retrieval_cache=cache,
    )

    try:
        first_result = agent.invoke(
            {
                "question": query,
                "current_query": query,
                "authorization_context": (
                    authorization_context
                ),
                "retry_count": 0,
            }
        )

        assert retriever.calls == [query]

        assert (
            first_result["retrieved_docs"]
            == documents
        )

        cached_documents = cache.get(
            domain="company",
            query=query,
            authorization_context=(
                authorization_context
            ),
            retrieval_version=retrieval_version,
        )

        assert cached_documents == documents

        second_result = agent.invoke(
            {
                "question": query,
                "current_query": query,
                "authorization_context": (
                    authorization_context
                ),
                "retry_count": 0,
            }
        )

        assert (
            second_result["retrieved_docs"]
            == documents
        )

        assert retriever.calls == [query]

        assert cache.size() == 1

    finally:
        cache.clear()

