from __future__ import annotations

import os
from typing import Any

from .redis_factory import build_redis_retrieval_cache
from .retrieval_cache import build_retrieval_cache


def _get_cache_backend() -> str:
    """
    Return the configured cache backend.

    Supported values:
        memory
        redis
    """

    backend = os.getenv(
        "RETRIEVAL_CACHE_BACKEND",
        "memory",
    ).strip().lower()

    if backend not in {"memory", "redis"}:
        raise ValueError(
            "RETRIEVAL_CACHE_BACKEND must be either "
            "'memory' or 'redis'."
        )

    return backend


def _get_cache_ttl() -> int:
    """Return the configured cache TTL in seconds."""

    raw_ttl = os.getenv(
        "RETRIEVAL_CACHE_TTL_SECONDS",
        "300",
    )

    try:
        ttl_seconds = int(raw_ttl)
    except ValueError as exc:
        raise ValueError(
            "RETRIEVAL_CACHE_TTL_SECONDS must be an integer."
        ) from exc

    if ttl_seconds <= 0:
        raise ValueError(
            "RETRIEVAL_CACHE_TTL_SECONDS must be greater than 0."
        )

    return ttl_seconds


def build_configured_retrieval_cache(
    namespace: str = "retrieval",
) -> Any:
    """
    Build the retrieval cache according to environment configuration.

    Local/test default:
        RetrievalCache

    Production:
        RedisRetrievalCache

    The returned objects expose the same cache contract.
    """

    backend = _get_cache_backend()
    ttl_seconds = _get_cache_ttl()

    if backend == "redis":
        return build_redis_retrieval_cache(
            ttl_seconds=ttl_seconds,
            namespace=namespace,
        )

    return build_retrieval_cache(
        ttl_seconds=ttl_seconds,
        namespace=namespace,
    )

