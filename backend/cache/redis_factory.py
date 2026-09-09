
from __future__ import annotations

import os

from redis import Redis

from .redis_retrieval_cache import RedisRetrievalCache


def build_redis_retrieval_cache(
    ttl_seconds: int = 300,
    namespace: str = "retrieval",
) -> RedisRetrievalCache:
    """
    Build a Redis-backed retrieval cache from environment configuration.
    """

    redis_url = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    socket_timeout = float(
        os.getenv(
            "REDIS_SOCKET_TIMEOUT_SECONDS",
            "1.5",
        )
    )

    socket_connect_timeout = float(
        os.getenv(
            "REDIS_CONNECT_TIMEOUT_SECONDS",
            "1.5",
        )
    )

    max_connections = int(
        os.getenv(
            "REDIS_MAX_CONNECTIONS",
            "20",
        )
    )

    redis_client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
        max_connections=max_connections,
        health_check_interval=30,
    )

    return RedisRetrievalCache(
        redis_client=redis_client,
        ttl_seconds=ttl_seconds,
        namespace=namespace,
    )

