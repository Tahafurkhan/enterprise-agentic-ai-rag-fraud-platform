
from __future__ import annotations

import pytest

from backend.cache.cache_factory import (
    build_configured_retrieval_cache,
)
from backend.cache.redis_retrieval_cache import RedisRetrievalCache
from backend.cache.retrieval_cache import RetrievalCache


def test_factory_defaults_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "RETRIEVAL_CACHE_BACKEND",
        raising=False,
    )

    cache = build_configured_retrieval_cache(
        namespace="test_memory",
    )

    assert isinstance(cache, RetrievalCache)


def test_factory_selects_memory_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    cache = build_configured_retrieval_cache(
        namespace="test_memory",
    )

    assert isinstance(cache, RetrievalCache)


def test_factory_selects_redis_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "redis",
    )

    monkeypatch.setenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    cache = build_configured_retrieval_cache(
        namespace="test_redis",
    )

    assert isinstance(cache, RedisRetrievalCache)


def test_factory_rejects_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "invalid",
    )

    with pytest.raises(ValueError, match="memory.*redis"):
        build_configured_retrieval_cache(
            namespace="test_invalid",
        )


def test_factory_rejects_invalid_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="RETRIEVAL_CACHE_TTL_SECONDS",
    ):
        build_configured_retrieval_cache(
            namespace="test_invalid_ttl",
        )


def test_factory_rejects_non_positive_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "0",
    )

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        build_configured_retrieval_cache(
            namespace="test_zero_ttl",
        )

