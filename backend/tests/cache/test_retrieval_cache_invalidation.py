
from __future__ import annotations

from langchain_core.documents import Document

from backend.cache.retrieval_cache import build_retrieval_cache


def _build_cache(namespace: str = "invalidation_test"):
    return build_retrieval_cache(
        ttl_seconds=300,
        namespace=namespace,
    )


def _build_documents():
    return [
        Document(
            page_content="Approved enterprise retrieval evidence.",
            metadata={
                "document_id": "doc-001",
                "chunk_id": "chunk-001",
                "file_name": "Policy Manual.pdf",
            },
        )
    ]


def test_same_cache_key_returns_stored_documents():
    cache = _build_cache()

    documents = _build_documents()

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    cached_documents = cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    )

    assert cached_documents == documents
    assert cache.size() == 1


def test_different_query_creates_separate_cache_entry():
    cache = _build_cache(
        namespace="different_query_test",
    )

    documents = _build_documents()

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    cache.set(
        domain="policy",
        query="document retention policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    ) == documents

    assert cache.get(
        domain="policy",
        query="document retention policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    ) == documents

    assert cache.size() == 2


def test_different_authorization_context_creates_separate_cache_entry():
    cache = _build_cache(
        namespace="authorization_isolation_test",
    )

    documents = _build_documents()

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-b",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    ) == documents

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-b",
        retrieval_version="policy_vector_v1",
    ) == documents

    assert cache.size() == 2


def test_different_retrieval_version_creates_separate_cache_entry():
    cache = _build_cache(
        namespace="version_isolation_test",
    )

    documents = _build_documents()

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v2",
        documents=documents,
    )

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    ) == documents

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v2",
    ) == documents

    assert cache.size() == 2


def test_clear_removes_all_entries_in_namespace():
    cache = _build_cache(
        namespace="clear_namespace_test",
    )

    documents = _build_documents()

    cache.set(
        domain="company",
        query="quality management",
        authorization_context="user-a",
        retrieval_version="company_hybrid_v1",
        documents=documents,
    )

    cache.set(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
        documents=documents,
    )

    assert cache.size() == 2

    cache.clear()

    assert cache.size() == 0

    assert cache.get(
        domain="company",
        query="quality management",
        authorization_context="user-a",
        retrieval_version="company_hybrid_v1",
    ) is None

    assert cache.get(
        domain="policy",
        query="corrective action policy",
        authorization_context="user-a",
        retrieval_version="policy_vector_v1",
    ) is None


def test_clear_on_empty_cache_is_safe():
    cache = _build_cache(
        namespace="empty_clear_test",
    )

    assert cache.size() == 0

    cache.clear()

    assert cache.size() == 0

