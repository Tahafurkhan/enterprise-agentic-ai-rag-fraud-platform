from __future__ import annotations

from langchain_core.documents import Document

from backend.cache.retrieval_cache import build_retrieval_cache
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
    def rerank(self, query, documents):
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


def _build_agent(
    documents,
    authorization_context="policy-test-user",
):
    retriever = FakeRetriever(documents)
    reranker = FakeReranker()
    llm = FakeLLM()

    cache = build_retrieval_cache(
        ttl_seconds=300,
        namespace="policy_cache_test",
    )

    agent = build_policy_rag_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        retrieval_cache=cache,
    )

    initial_state = {
        "question": "What is the policy for corrective action?",
        "current_query": (
            "What is the policy for corrective action?"
        ),
        "authorization_context": authorization_context,
        "retry_count": 0,
    }

    return (
        agent,
        retriever,
        cache,
        initial_state,
    )


def test_policy_retrieval_first_call_populates_cache():
    documents = [
        Document(
            page_content=(
                "The quality management policy requires "
                "documented corrective action."
            ),
            metadata={
                "document_id": "policy-doc-001",
                "file_name": "Quality Policy.pdf",
                "page_number": 10,
            },
        )
    ]

    (
        agent,
        retriever,
        cache,
        state,
    ) = _build_agent(documents)

    result = agent.invoke(state)

    assert retriever.calls == [
        "What is the policy for corrective action?"
    ]

    assert result["retrieved_docs"] == documents

    cached_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="policy-test-user",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
    )

    assert cached_documents == documents
    assert cache.size() == 1


def test_policy_retrieval_second_call_uses_cache():
    documents = [
        Document(
            page_content=(
                "Corrective actions must be documented "
                "and tracked."
            ),
            metadata={
                "document_id": "policy-doc-002",
                "file_name": "Quality Policy.pdf",
                "page_number": 11,
            },
        )
    ]

    (
        agent,
        retriever,
        cache,
        state,
    ) = _build_agent(documents)

    first_result = agent.invoke(state)

    second_result = agent.invoke(state)

    assert first_result["retrieved_docs"] == documents
    assert second_result["retrieved_docs"] == documents

    assert retriever.calls == [
        "What is the policy for corrective action?"
    ]

    assert cache.size() == 1


def test_policy_cache_isolated_by_authorization_context():
    documents = [
        Document(
            page_content="Authorized Policy evidence.",
            metadata={
                "document_id": "policy-doc-003",
            },
        )
    ]

    (
        agent,
        retriever,
        cache,
        state,
    ) = _build_agent(
        documents,
        authorization_context="user-a",
    )

    agent.invoke(state)

    user_b_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="user-b",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
    )

    assert user_b_documents is None

    user_a_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="user-a",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
    )

    assert user_a_documents == documents
    assert retriever.calls == [
        "What is the policy for corrective action?"
    ]


def test_policy_cache_isolated_by_retrieval_version():
    documents = [
        Document(
            page_content="Versioned Policy evidence.",
            metadata={
                "document_id": "policy-doc-004",
            },
        )
    ]

    (
        agent,
        retriever,
        cache,
        state,
    ) = _build_agent(documents)

    agent.invoke(state)

    current_version_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="policy-test-user",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
    )

    old_version_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="policy-test-user",
        retrieval_version="policy_vector_old_v1",
    )

    assert current_version_documents == documents
    assert old_version_documents is None

    assert retriever.calls == [
        "What is the policy for corrective action?"
    ]


def test_policy_empty_retrieval_result_is_not_cached():
    cache = build_retrieval_cache(
        ttl_seconds=300,
        namespace="policy_empty_cache_test",
    )

    documents = []

    cache.set(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="policy-test-user",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
        documents=documents,
    )

    cached_documents = cache.get(
        domain="policy",
        query="What is the policy for corrective action?",
        authorization_context="policy-test-user",
        retrieval_version=RETRIEVAL_CACHE_VERSION,
    )

    assert cached_documents is None
    assert cache.size() == 0
