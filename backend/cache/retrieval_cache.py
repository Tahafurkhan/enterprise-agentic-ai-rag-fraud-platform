
"""
Shared retrieval cache for enterprise RAG agents.

The cache is designed to be shared by:
    - Company RAG
    - Policy RAG

Cache isolation is based on:
    - domain
    - query
    - authorization context
    - retrieval version

The current implementation uses an in-memory cache suitable for
local development and testing.

A Redis-backed implementation can later replace this storage layer
without changing the Company RAG or Policy RAG cache contract.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import List

from langchain_core.documents import Document


@dataclass
class _CacheEntry:
    documents: List[Document]
    expires_at: float


class RetrievalCache:
    """Thread-safe in-memory retrieval cache."""

    def __init__(
        self,
        ttl_seconds: int = 300,
        namespace: str = "retrieval",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0.")

        if not namespace:
            raise ValueError("namespace must not be empty.")

        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

        self._cache: dict[str, _CacheEntry] = {}
        self._lock = Lock()

    def _build_key(
        self,
        domain: str,
        query: str,
        authorization_context: str,
        retrieval_version: str,
    ) -> str:
        """Build a deterministic cache key."""

        raw_key = "|".join(
            [
                self.namespace,
                domain,
                query.strip(),
                authorization_context,
                retrieval_version,
            ]
        )

        return hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
    ) -> List[Document] | None:
        """
        Retrieve documents from cache.

        Returns:
            Cached documents when a valid entry exists.
            None when the cache misses or the entry has expired.
        """

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            if time.monotonic() >= entry.expires_at:
                del self._cache[key]
                return None

            return list(entry.documents)

    def set(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
        documents: List[Document] | None = None,
    ) -> None:
        """Store retrieved documents in the cache."""

        if not documents:
            return

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        entry = _CacheEntry(
            documents=list(documents),
            expires_at=time.monotonic() + self.ttl_seconds,
        )

        with self._lock:
            self._cache[key] = entry

    def delete(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
    ) -> None:
        """Delete one cache entry."""

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""

        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return the number of currently stored cache entries."""

        with self._lock:
            return len(self._cache)


def build_retrieval_cache(
    ttl_seconds: int = 300,
    namespace: str = "retrieval",
) -> RetrievalCache:
    """Build a shared retrieval cache instance."""

    return RetrievalCache(
        ttl_seconds=ttl_seconds,
        namespace=namespace,
    )

