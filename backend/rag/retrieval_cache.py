"""
Retrieval cache for the enterprise RAG platform.

The cache stores retrieval/evidence results, not final generated answers.

Security principles:
- Cache keys are domain-aware.
- Cache keys include authorization context.
- Queries are normalized before key generation.
- Empty results are not cached.
- Failed retrievals should not be cached.
- Entries expire through TTL.
- Cache contents can be explicitly invalidated.

This first implementation is intentionally in-memory and process-local.
It can later be replaced or backed by Redis without changing the
retrieval/RAG interfaces.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional


# ============================================================
# CACHE ENTRY
# ============================================================


@dataclass
class CacheEntry:
    """
    One cached retrieval result.
    """

    value: Any
    created_at: float
    expires_at: float


# ============================================================
# RETRIEVAL CACHE
# ============================================================


class RetrievalCache:
    """
    Process-local retrieval cache.

    Cache keys are derived from:

        domain
        authorization context
        normalized query
        retrieval configuration/version

    This prevents retrieval results from one security/domain context
    being accidentally reused in another context.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        namespace: str = "retrieval",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        if not namespace.strip():
            raise ValueError(
                "namespace must not be empty."
            )

        self.ttl_seconds = ttl_seconds
        self.namespace = namespace.strip()

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    # ========================================================
    # QUERY NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_query(
        query: str,
    ) -> str:
        """
        Normalize a retrieval query for cache-key generation.

        The original query is not modified.

        Example:

            "  What is   corrective action?  "

        becomes:

            "what is corrective action?"
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        return " ".join(
            query.strip().lower().split()
        )

    # ========================================================
    # AUTHORIZATION NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_authorization_context(
        authorization_context: Optional[Hashable],
    ) -> str:
        """
        Convert authorization context into a deterministic string.

        The caller should provide only the authorization information
        necessary to distinguish access boundaries.

        Examples:

            "fraud_investigator"
            "fraud_manager"

        or:

            ("fraud_investigator", "fraud")
        """

        if authorization_context is None:
            return "anonymous"

        if isinstance(
            authorization_context,
            (list, tuple, set, frozenset),
        ):
            values = sorted(
                str(value)
                for value in authorization_context
            )

            return "|".join(values)

        return str(
            authorization_context
        ).strip().lower()

    # ========================================================
    # CACHE KEY
    # ========================================================

    def build_key(
        self,
        *,
        domain: str,
        query: str,
        authorization_context: Optional[Hashable] = None,
        retrieval_version: str = "v1",
    ) -> str:
        """
        Build a deterministic cache key.

        The key includes:

            namespace
            domain
            authorization context
            normalized query
            retrieval version

        The resulting key is hashed so that arbitrary query/context
        values do not become directly visible in the dictionary key.
        """

        if not isinstance(domain, str):
            raise TypeError(
                "domain must be a string."
            )

        if not domain.strip():
            raise ValueError(
                "domain must not be empty."
            )

        if not isinstance(
            retrieval_version,
            str,
        ):
            raise TypeError(
                "retrieval_version must be a string."
            )

        if not retrieval_version.strip():
            raise ValueError(
                "retrieval_version must not be empty."
            )

        normalized_query = self.normalize_query(
            query
        )

        if not normalized_query:
            raise ValueError(
                "query must not be empty."
            )

        normalized_auth = (
            self.normalize_authorization_context(
                authorization_context
            )
        )

        raw_key = "|".join(
            [
                self.namespace,
                domain.strip().lower(),
                normalized_auth,
                retrieval_version.strip(),
                normalized_query,
            ]
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return f"{self.namespace}:{digest}"

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        *,
        domain: str,
        query: str,
        authorization_context: Optional[Hashable] = None,
        retrieval_version: str = "v1",
    ) -> Any:
        """
        Retrieve a cached value.

        Returns:
            Cached value when present and valid.
            None when there is a cache miss or expired entry.
        """

        key = self.build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        now = time.monotonic()

        with self._lock:

            entry = self._cache.get(key)

            if entry is None:
                return None

            if now >= entry.expires_at:
                del self._cache[key]
                return None

            return entry.value

    # ========================================================
    # SET
    # ========================================================

    def set(
        self,
        *,
        domain: str,
        query: str,
        value: Any,
        authorization_context: Optional[Hashable] = None,
        retrieval_version: str = "v1",
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store a successful retrieval result.

        Empty results are deliberately not cached.

        Returns:
            True when the value was cached.
            False when the value was rejected as empty.
        """

        if value is None:
            return False

        if isinstance(value, (list, tuple, set, frozenset)):
            if len(value) == 0:
                return False

        effective_ttl = (
            self.ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )

        if effective_ttl <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        key = self.build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        now = time.monotonic()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=(
                now + effective_ttl
            ),
        )

        with self._lock:
            self._cache[key] = entry

        return True

    # ========================================================
    # INVALIDATE
    # ========================================================

    def invalidate(
        self,
        *,
        domain: str,
        query: str,
        authorization_context: Optional[Hashable] = None,
        retrieval_version: str = "v1",
    ) -> bool:
        """
        Remove one cache entry.

        Returns:
            True if an entry existed and was removed.
            False otherwise.
        """

        key = self.build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        with self._lock:

            if key not in self._cache:
                return False

            del self._cache[key]

            return True

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """
        Clear the entire cache.
        """

        with self._lock:
            self._cache.clear()

    # ========================================================
    # EXPIRE
    # ========================================================

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of removed entries.
        """

        now = time.monotonic()

        removed = 0

        with self._lock:

            expired_keys = [
                key
                for key, entry in self._cache.items()
                if now >= entry.expires_at
            ]

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    # ========================================================
    # SIZE
    # ========================================================

    def size(self) -> int:
        """
        Return the number of currently stored entries.

        Expired entries are cleaned before calculating the size.
        """

        self.cleanup_expired()

        with self._lock:
            return len(self._cache)


# ============================================================
# FACTORY
# ============================================================


def build_retrieval_cache(
    ttl_seconds: int = 300,
    namespace: str = "retrieval",
) -> RetrievalCache:
    """
    Build the process-local retrieval cache.
    """

    return RetrievalCache(
        ttl_seconds=ttl_seconds,
        namespace=namespace,
    )

