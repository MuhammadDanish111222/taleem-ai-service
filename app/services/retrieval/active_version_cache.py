"""Small Redis cache for active corpus-version configuration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import Any, Protocol

from redis import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_CACHE_FIELDS = (
    "id",
    "embedding_model",
    "embedding_revision",
    "embedding_dim",
    "normalize_embeddings",
    "query_instruction",
    "embedding_config_fingerprint",
)


class RedisClient(Protocol):
    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: str, *, ex: int) -> Any: ...

    def delete(self, *names: str) -> Any: ...


class ActiveCorpusVersionCache:
    """Caches only the non-sensitive configuration needed by retrieval."""

    def __init__(self, redis_client: RedisClient, *, ttl_seconds: int = 300) -> None:
        if ttl_seconds < 1:
            raise ValueError("ACTIVE_CORPUS_CACHE_TTL_MUST_BE_POSITIVE")
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(board_id: str, class_id: str, subject_id: str) -> str:
        scope = "\x1f".join((board_id, class_id, subject_id)).encode("utf-8")
        return f"rag:active-corpus:v1:{hashlib.sha256(scope).hexdigest()}"

    async def get(
        self, board_id: str, class_id: str, subject_id: str
    ) -> dict[str, Any] | None:
        try:
            raw = await asyncio.to_thread(
                self._redis.get, self._key(board_id, class_id, subject_id)
            )
            if raw is None:
                return None
            parsed = json.loads(raw)
            if (
                not isinstance(parsed, dict)
                or any(field not in parsed for field in _CACHE_FIELDS)
                or not isinstance(parsed["id"], str)
            ):
                return None
            return parsed
        except (RedisError, OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Active corpus cache read failed; using PostgreSQL")
            return None

    async def set(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        version: dict[str, Any],
    ) -> None:
        try:
            payload = {field: version[field] for field in _CACHE_FIELDS}
            payload["id"] = str(payload["id"])
            await asyncio.to_thread(
                self._redis.set,
                self._key(board_id, class_id, subject_id),
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                ex=self._ttl_seconds,
            )
        except (RedisError, OSError, ValueError, TypeError, KeyError):
            logger.warning("Active corpus cache write failed; continuing without cache")

    async def invalidate(self, board_id: str, class_id: str, subject_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._redis.delete, self._key(board_id, class_id, subject_id)
            )
        except (RedisError, OSError):
            logger.warning("Active corpus cache invalidation failed")


class DisabledActiveCorpusVersionCache(ActiveCorpusVersionCache):
    """No-op cache for isolated database test runs."""

    def __init__(self) -> None:
        pass

    async def get(
        self, board_id: str, class_id: str, subject_id: str
    ) -> dict[str, Any] | None:
        return None

    async def set(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        version: dict[str, Any],
    ) -> None:
        return None

    async def invalidate(self, board_id: str, class_id: str, subject_id: str) -> None:
        return None


@lru_cache(maxsize=1)
def get_active_corpus_version_cache() -> ActiveCorpusVersionCache:
    # Disposable test databases frequently reuse the same human-readable
    # scope with different UUIDs across runs; they must never share live Redis.
    if os.getenv("TEST_DATABASE_URL"):
        return DisabledActiveCorpusVersionCache()
    settings = get_settings()
    redis_client = Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
    )
    return ActiveCorpusVersionCache(
        redis_client,
        ttl_seconds=settings.ACTIVE_CORPUS_CACHE_TTL_SECONDS,
    )
