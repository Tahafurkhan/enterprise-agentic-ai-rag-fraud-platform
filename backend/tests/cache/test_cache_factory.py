
from __future__ import annotations

from unittest.mock import Mock

import pytest

import backend.cache.cache_factory as cache_factory
from backend.cache.redis_retrieval_cache import RedisRetrievalCache
from backend.cache.retrieval_cache import RetrievalCache


def test_default_backend_is_memory(monkeypatch):
    monkeypatch.delenv(
        "RETRIEVAL_CACHE_BACKEND",
        raising=False,
    )

    cache = cache_factory.build_configured_retrieval_cache(
        namespace="test_default",
    )

    assert isinstance(cache, RetrievalCache)


def test_memory_backend_returns_memory_cache(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    cache = cache_factory.build_configured_retrieval_cache(
        namespace="test_memory",
    )

    assert isinstance(cache, RetrievalCache)


def test_redis_backend_uses_redis_factory(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "redis",
    )

    fake_cache = Mock(spec=RedisRetrievalCache)

    build_redis_mock = Mock(
        return_value=fake_cache,
    )

    monkeypatch.setattr(
        cache_factory,
        "build_redis_retrieval_cache",
        build_redis_mock,
    )

    cache = cache_factory.build_configured_retrieval_cache(
        namespace="test_redis",
    )

    assert cache is fake_cache

    build_redis_mock.assert_called_once_with(
        ttl_seconds=300,
        namespace="test_redis",
    )


def test_custom_ttl_is_passed_to_redis_backend(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "redis",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "600",
    )

    fake_cache = Mock(spec=RedisRetrievalCache)

    build_redis_mock = Mock(
        return_value=fake_cache,
    )

    monkeypatch.setattr(
        cache_factory,
        "build_redis_retrieval_cache",
        build_redis_mock,
    )

    cache = cache_factory.build_configured_retrieval_cache(
        namespace="custom_ttl",
    )

    assert cache is fake_cache

    build_redis_mock.assert_called_once_with(
        ttl_seconds=600,
        namespace="custom_ttl",
    )


def test_invalid_backend_raises_value_error(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "invalid_backend",
    )

    with pytest.raises(ValueError):
        cache_factory.build_configured_retrieval_cache(
            namespace="invalid_backend",
        )


def test_invalid_ttl_zero_raises_value_error(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "0",
    )

    with pytest.raises(ValueError):
        cache_factory.build_configured_retrieval_cache(
            namespace="invalid_ttl",
        )


def test_invalid_ttl_negative_raises_value_error(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "-10",
    )

    with pytest.raises(ValueError):
        cache_factory.build_configured_retrieval_cache(
            namespace="negative_ttl",
        )


def test_invalid_ttl_non_integer_raises_value_error(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "not-an-integer",
    )

    with pytest.raises(ValueError):
        cache_factory.build_configured_retrieval_cache(
            namespace="invalid_ttl_text",
        )


def test_memory_backend_respects_custom_ttl(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    )

    monkeypatch.setenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "600",
    )

    cache = cache_factory.build_configured_retrieval_cache(
        namespace="custom_memory_ttl",
    )

    assert isinstance(cache, RetrievalCache)
    assert cache.ttl_seconds == 600


def test_redis_backend_respects_custom_namespace(monkeypatch):
    monkeypatch.setenv(
        "RETRIEVAL_CACHE_BACKEND",
        "redis",
    )

    fake_cache = Mock(spec=RedisRetrievalCache)

    build_redis_mock = Mock(
        return_value=fake_cache,
    )

    monkeypatch.setattr(
        cache_factory,
        "build_redis_retrieval_cache",
        build_redis_mock,
    )

    cache_factory.build_configured_retrieval_cache(
        namespace="policy_retrieval",
    )

    build_redis_mock.assert_called_once_with(
        ttl_seconds=300,
        namespace="policy_retrieval",
    )

