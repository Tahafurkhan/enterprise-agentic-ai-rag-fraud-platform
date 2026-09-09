
from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from redis.exceptions import RedisError

from backend.cache.redis_retrieval_cache import RedisRetrievalCache


class FakeRedis:
    """Small in-memory Redis test double."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.get_calls: list[str] = []
        self.setex_calls: list[tuple[str, int, str]] = []
        self.delete_calls: list[str] = []
        self.scan_calls: list[str] = []

        self.fail_get = False
        self.fail_setex = False
        self.fail_delete = False
        self.fail_scan = False

    def get(self, key: str) -> str | None:
        self.get_calls.append(key)

        if self.fail_get:
            raise RedisError("Redis GET failure")

        return self.store.get(key)

    def setex(
        self,
        key: str,
        ttl: int,
        value: str,
    ) -> bool:
        self.setex_calls.append((key, ttl, value))

        if self.fail_setex:
            raise RedisError("Redis SET failure")

        self.store[key] = value
        self.ttls[key] = ttl

        return True

    def delete(self, key: str) -> int:
        self.delete_calls.append(key)

        if self.fail_delete:
            raise RedisError("Redis DELETE failure")

        existed = key in self.store

        self.store.pop(key, None)
        self.ttls.pop(key, None)

        return 1 if existed else 0

    def scan_iter(self, match: str):
        self.scan_calls.append(match)

        if self.fail_scan:
            raise RedisError("Redis SCAN failure")

        prefix = match.replace("*", "")

        for key in list(self.store):
            if key.startswith(prefix):
                yield key


def _documents() -> list[Document]:
    return [
        Document(
            page_content="Approved policy evidence.",
            metadata={
                "document_id": "DOC-001",
                "file_name": "policy.pdf",
                "source": "policy_documents",
            },
        ),
        Document(
            page_content="Second approved evidence item.",
            metadata={
                "document_id": "DOC-002",
                "file_name": "policy_2.pdf",
                "source": "policy_documents",
            },
        ),
    ]


def _build_cache(
    redis_client: FakeRedis | None = None,
) -> tuple[RedisRetrievalCache, FakeRedis]:
    redis_client = redis_client or FakeRedis()

    cache = RedisRetrievalCache(
        redis_client=redis_client,
        ttl_seconds=300,
        namespace="test_retrieval",
    )

    return cache, redis_client


def test_redis_cache_miss_returns_none() -> None:
    cache, redis_client = _build_cache()

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None
    assert len(redis_client.get_calls) == 1


def test_redis_cache_hit_returns_documents() -> None:
    cache, _ = _build_cache()

    documents = _documents()

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=documents,
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is not None
    assert len(result) == 2

    assert result[0].page_content == "Approved policy evidence."
    assert result[0].metadata["document_id"] == "DOC-001"

    assert result[1].page_content == "Second approved evidence item."
    assert result[1].metadata["document_id"] == "DOC-002"


def test_redis_cache_serializes_documents_as_json() -> None:
    cache, redis_client = _build_cache()

    documents = _documents()

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=documents,
    )

    assert len(redis_client.setex_calls) == 1

    _, _, payload = redis_client.setex_calls[0]

    decoded = json.loads(payload)

    assert isinstance(decoded, list)
    assert len(decoded) == 2

    assert decoded[0]["page_content"] == "Approved policy evidence."
    assert decoded[0]["metadata"]["document_id"] == "DOC-001"


def test_redis_cache_uses_configured_ttl() -> None:
    redis_client = FakeRedis()

    cache = RedisRetrievalCache(
        redis_client=redis_client,
        ttl_seconds=600,
        namespace="test_retrieval",
    )

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    assert len(redis_client.setex_calls) == 1

    _, ttl, _ = redis_client.setex_calls[0]

    assert ttl == 600


def test_normalized_queries_share_cache_entry() -> None:
    cache, redis_client = _build_cache()

    cache.set(
        domain="policy",
        query="  Approval   Policy  ",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is not None
    assert len(redis_client.get_calls) == 1


def test_authorization_contexts_are_isolated() -> None:
    cache, _ = _build_cache()

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-b",
        retrieval_version="policy_v1",
    )

    assert result is None


def test_domains_are_isolated() -> None:
    cache, _ = _build_cache()

    cache.set(
        domain="company",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="company_v1",
        documents=_documents(),
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="company_v1",
    )

    assert result is None


def test_retrieval_versions_are_isolated() -> None:
    cache, _ = _build_cache()

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v2",
    )

    assert result is None


def test_empty_documents_are_not_cached() -> None:
    cache, redis_client = _build_cache()

    cache.set(
        domain="policy",
        query="empty query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=[],
    )

    result = cache.get(
        domain="policy",
        query="empty query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None
    assert len(redis_client.setex_calls) == 0


def test_none_documents_are_not_cached() -> None:
    cache, redis_client = _build_cache()

    cache.set(
        domain="policy",
        query="empty query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=None,
    )

    result = cache.get(
        domain="policy",
        query="empty query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None
    assert len(redis_client.setex_calls) == 0


def test_invalidate_removes_entry() -> None:
    cache, _ = _build_cache()

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    assert cache.invalidate(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    ) is True

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None


def test_invalidate_missing_entry_returns_false() -> None:
    cache, _ = _build_cache()

    result = cache.invalidate(
        domain="policy",
        query="missing query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is False


def test_clear_uses_scan_and_removes_namespace_entries() -> None:
    cache, redis_client = _build_cache()

    cache.set(
        domain="company",
        query="company query",
        authorization_context="user-a",
        retrieval_version="company_v1",
        documents=_documents(),
    )

    cache.set(
        domain="policy",
        query="policy query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    cache.clear()

    assert len(redis_client.scan_calls) == 1
    assert redis_client.store == {}


def test_size_reports_namespace_entries() -> None:
    cache, _ = _build_cache()

    cache.set(
        domain="company",
        query="company query",
        authorization_context="user-a",
        retrieval_version="company_v1",
        documents=_documents(),
    )

    cache.set(
        domain="policy",
        query="policy query",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    assert cache.size() == 2


def test_redis_get_failure_fails_open() -> None:
    redis_client = FakeRedis()
    redis_client.fail_get = True

    cache, _ = _build_cache(redis_client)

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None


def test_redis_set_failure_fails_open() -> None:
    redis_client = FakeRedis()
    redis_client.fail_setex = True

    cache, _ = _build_cache(redis_client)

    cache.set(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
        documents=_documents(),
    )

    assert redis_client.store == {}


def test_redis_delete_failure_fails_open() -> None:
    redis_client = FakeRedis()
    redis_client.fail_delete = True

    cache, _ = _build_cache(redis_client)

    result = cache.invalidate(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is False


def test_redis_scan_failure_fails_open() -> None:
    redis_client = FakeRedis()
    redis_client.fail_scan = True

    cache, _ = _build_cache(redis_client)

    cache.clear()

    assert redis_client.store == {}


def test_malformed_cached_payload_is_treated_as_cache_miss() -> None:
    cache, redis_client = _build_cache()

    key = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    redis_client.store[key] = '{"invalid": "payload"}'

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None


def test_non_object_document_payload_is_treated_as_cache_miss() -> None:
    cache, redis_client = _build_cache()

    key = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    redis_client.store[key] = json.dumps(
        [
            "invalid document",
        ]
    )

    result = cache.get(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert result is None


def test_cache_key_is_deterministic() -> None:
    cache, _ = _build_cache()

    key_one = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    key_two = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    assert key_one == key_two


def test_different_authorization_contexts_generate_different_keys() -> None:
    cache, _ = _build_cache()

    key_one = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    key_two = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-b",
        retrieval_version="policy_v1",
    )

    assert key_one != key_two


def test_different_domains_generate_different_keys() -> None:
    cache, _ = _build_cache()

    key_one = cache._build_key(
        domain="company",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="v1",
    )

    key_two = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="v1",
    )

    assert key_one != key_two


def test_different_retrieval_versions_generate_different_keys() -> None:
    cache, _ = _build_cache()

    key_one = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v1",
    )

    key_two = cache._build_key(
        domain="policy",
        query="approval policy",
        authorization_context="user-a",
        retrieval_version="policy_v2",
    )

    assert key_one != key_two

