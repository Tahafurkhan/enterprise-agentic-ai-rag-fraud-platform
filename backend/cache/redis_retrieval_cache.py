
from __future__ import annotations

import hashlib
import json
from typing import Any, List

from langchain_core.documents import Document
from redis import Redis
from redis.exceptions import RedisError


class RedisRetrievalCache:
    """
    Production Redis-backed retrieval cache.

    The cache is intentionally fail-open:
    Redis failures do not break the RAG request. A cache failure
    results in a cache miss so the caller can retrieve fresh evidence.

    Cache isolation is based on:
        - namespace
        - domain
        - normalized query
        - authorization context
        - retrieval version
    """

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = 300,
        namespace: str = "retrieval",
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0.")

        if not namespace:
            raise ValueError("namespace must not be empty.")

        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query text for deterministic cache keys."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        normalized = " ".join(query.strip().split())

        if not normalized:
            raise ValueError("query must not be empty.")

        return normalized.casefold()

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> None:
        """Validate required string inputs."""

        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")

        if not value.strip():
            raise ValueError(f"{field_name} must not be empty.")

    def _build_key(
        self,
        domain: str,
        query: str,
        authorization_context: str,
        retrieval_version: str,
    ) -> str:
        """Build a deterministic Redis cache key."""

        self._validate_text(domain, "domain")
        self._validate_text(
            authorization_context,
            "authorization_context",
        )
        self._validate_text(
            retrieval_version,
            "retrieval_version",
        )

        normalized_query = self._normalize_query(query)

        raw_key = "|".join(
            [
                self.namespace,
                domain.strip(),
                normalized_query,
                authorization_context.strip(),
                retrieval_version.strip(),
            ]
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return f"{self.namespace}:retrieval:{digest}"

    @staticmethod
    def _serialize_documents(
        documents: List[Document],
    ) -> str:
        """Serialize LangChain documents into JSON."""

        payload = [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            for document in documents
        ]

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize_documents(
        payload: str,
    ) -> List[Document]:
        """Deserialize JSON into LangChain documents."""

        data: Any = json.loads(payload)

        if not isinstance(data, list):
            raise ValueError("Cached retrieval payload must be a list.")

        documents: List[Document] = []

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Cached retrieval document must be an object."
                )

            page_content = item.get("page_content", "")
            metadata = item.get("metadata", {})

            if not isinstance(page_content, str):
                raise ValueError(
                    "Cached document page_content must be a string."
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Cached document metadata must be an object."
                )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        return documents

    def get(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
    ) -> List[Document] | None:
        """
        Retrieve documents from Redis.

        Redis failures are treated as cache misses so that
        retrieval can continue from the source system.
        """

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        try:
            payload = self.redis_client.get(key)

            if payload is None:
                return None

            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")

            if not isinstance(payload, str):
                raise ValueError(
                    "Redis cache payload must be a string."
                )

            return self._deserialize_documents(payload)

        except (RedisError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def set(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
        documents: List[Document] | None = None,
    ) -> None:
        """Store documents in Redis with TTL."""

        if not documents:
            return

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        payload = self._serialize_documents(documents)

        try:
            self.redis_client.setex(
                key,
                self.ttl_seconds,
                payload,
            )
        except RedisError:
            # Fail open. Retrieval should continue even when
            # Redis is temporarily unavailable.
            return

    def invalidate(
        self,
        domain: str,
        query: str,
        authorization_context: str = "anonymous",
        retrieval_version: str = "v1",
    ) -> bool:
        """Invalidate one cache entry."""

        key = self._build_key(
            domain=domain,
            query=query,
            authorization_context=authorization_context,
            retrieval_version=retrieval_version,
        )

        try:
            return bool(self.redis_client.delete(key))
        except RedisError:
            return False

    def clear(self) -> None:
        """
        Clear cache entries belonging to this namespace.

        Uses SCAN rather than KEYS to avoid blocking Redis.
        """

        pattern = f"{self.namespace}:retrieval:*"

        try:
            keys = self.redis_client.scan_iter(match=pattern)

            for key in keys:
                self.redis_client.delete(key)

        except RedisError:
            return

    def size(self) -> int:
        """Return the number of entries in this cache namespace."""

        pattern = f"{self.namespace}:retrieval:*"

        try:
            return sum(
                1
                for _ in self.redis_client.scan_iter(
                    match=pattern
                )
            )
        except RedisError:
            return 0
