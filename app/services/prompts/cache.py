"""Bounded shared prompt-cache contracts and Redis implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Protocol

from redis.exceptions import RedisError

from app.services.prompts.models import (
    AnswerMode,
    PromptKey,
    PromptRecord,
    PromptScope,
    PromptStatus,
)


class RedisPromptClient(Protocol):
    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: str, *, ex: int) -> Any: ...

    def incr(self, name: str) -> Any: ...


class PromptCache(Protocol):
    async def get(
        self, prompt_key: PromptKey, answer_mode: AnswerMode, scope: PromptScope
    ) -> PromptRecord | None: ...

    async def set(self, scope: PromptScope, record: PromptRecord) -> None: ...

    async def invalidate_family(
        self, prompt_key: PromptKey, answer_mode: AnswerMode
    ) -> None: ...


class NullPromptCache:
    async def get(
        self, prompt_key: PromptKey, answer_mode: AnswerMode, scope: PromptScope
    ) -> PromptRecord | None:
        return None

    async def set(self, scope: PromptScope, record: PromptRecord) -> None:
        return None

    async def invalidate_family(
        self, prompt_key: PromptKey, answer_mode: AnswerMode
    ) -> None:
        return None


class RedisPromptCache:
    """Shared cache using O(1) family-generation invalidation.

    Old entries expire through the bounded TTL. Activation and rollback only
    increment a fixed generation key; neither operation performs a key scan.
    """

    def __init__(self, redis_client: RedisPromptClient, *, ttl_seconds: int = 300):
        if not 1 <= ttl_seconds <= 900:
            raise ValueError("PROMPT_CACHE_TTL_OUT_OF_RANGE")
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    async def get(
        self, prompt_key: PromptKey, answer_mode: AnswerMode, scope: PromptScope
    ) -> PromptRecord | None:
        try:
            generation = await self._generation(prompt_key, answer_mode)
            raw = await asyncio.to_thread(
                self._redis.get,
                self._record_key(prompt_key, answer_mode, scope, generation),
            )
            if raw is None:
                return None
            return _deserialize_record(raw)
        except (
            RedisError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None

    async def set(self, scope: PromptScope, record: PromptRecord) -> None:
        try:
            generation = await self._generation(record.prompt_key, record.answer_mode)
            await asyncio.to_thread(
                self._redis.set,
                self._record_key(
                    record.prompt_key, record.answer_mode, scope, generation
                ),
                _serialize_record(record),
                ex=self._ttl_seconds,
            )
        except (RedisError, OSError, TypeError, ValueError):
            return None

    async def invalidate_family(
        self, prompt_key: PromptKey, answer_mode: AnswerMode
    ) -> None:
        try:
            await asyncio.to_thread(
                self._redis.incr, self._generation_key(prompt_key, answer_mode)
            )
        except (RedisError, OSError, TypeError, ValueError):
            return None

    async def _generation(self, prompt_key: PromptKey, answer_mode: AnswerMode) -> int:
        raw = await asyncio.to_thread(
            self._redis.get, self._generation_key(prompt_key, answer_mode)
        )
        try:
            return max(0, int(raw)) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _generation_key(prompt_key: PromptKey, answer_mode: AnswerMode) -> str:
        return f"prompts:generation:v1:{prompt_key.value}:{answer_mode.value}"

    @staticmethod
    def _record_key(
        prompt_key: PromptKey,
        answer_mode: AnswerMode,
        scope: PromptScope,
        generation: int,
    ) -> str:
        identity = "\x1f".join(
            (
                prompt_key.value,
                answer_mode.value,
                scope.board_id or "",
                scope.class_id or "",
                scope.subject_id or "",
            )
        ).encode()
        digest = hashlib.sha256(identity).hexdigest()
        return f"prompts:active:v1:{generation}:{digest}"


def _serialize_record(record: PromptRecord) -> str:
    payload = {
        "id": record.id,
        "prompt_key": record.prompt_key.value,
        "answer_mode": record.answer_mode.value,
        "scope": {
            "board_id": record.scope.board_id,
            "class_id": record.scope.class_id,
            "subject_id": record.scope.subject_id,
        },
        "version": record.version,
        "content": record.content,
        "status": record.status.value,
        "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
        "activated_by": record.activated_by,
        "activated_at": (
            record.activated_at.isoformat() if record.activated_at else None
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _deserialize_record(raw: str | bytes) -> PromptRecord:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("scope"), dict):
        raise ValueError("PROMPT_CACHE_PAYLOAD_INVALID")
    scope_payload = payload["scope"]
    return PromptRecord(
        id=payload["id"],
        prompt_key=PromptKey(payload["prompt_key"]),
        answer_mode=AnswerMode(payload["answer_mode"]),
        scope=PromptScope(
            board_id=scope_payload.get("board_id"),
            class_id=scope_payload.get("class_id"),
            subject_id=scope_payload.get("subject_id"),
        ),
        version=payload["version"],
        content=payload["content"],
        status=PromptStatus(payload["status"]),
        created_by=payload["created_by"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        activated_by=payload.get("activated_by"),
        activated_at=(
            datetime.fromisoformat(payload["activated_at"])
            if payload.get("activated_at")
            else None
        ),
    )
