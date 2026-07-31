"""Cross-process prompt cache keyed by a PostgreSQL generation."""

from __future__ import annotations

import asyncio
import json

import asyncpg
import redis

from app.services.prompts.cache import _deserialize_record, _serialize_record
from app.services.prompts.models import (
    AnswerMode,
    PromptKey,
    PromptRecord,
    PromptScope,
)


class SharedPromptCache:
    def __init__(
        self,
        conn: asyncpg.Connection,
        redis_client: redis.Redis,
        *,
        ttl_seconds: int = 300,
    ):
        if not 1 <= ttl_seconds <= 900:
            raise ValueError("PROMPT_CACHE_TTL_OUT_OF_RANGE")
        self._conn = conn
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _family(prompt_key: PromptKey, answer_mode: AnswerMode) -> str:
        return f"{prompt_key.value}:{answer_mode.value}"

    async def _generation(self, prompt_key: PromptKey, answer_mode: AnswerMode) -> int:
        generation_key = self._generation_key(prompt_key, answer_mode)
        cached_generation: int | None = None
        try:
            cached = await asyncio.to_thread(self._redis.get, generation_key)
            if cached is not None:
                cached_generation = max(1, int(cached))
        except Exception:
            pass
        # PostgreSQL remains the transactionally committed invalidation
        # authority. Checking its tiny generation row guarantees activation
        # affects the next request even if the publishing process lost Redis
        # connectivity immediately after commit.
        value = await self._conn.fetchval(
            """SELECT generation FROM cache_generations
               WHERE namespace='prompt' AND cache_key=$1""",
            self._family(prompt_key, answer_mode),
        )
        generation = int(value or 1)
        if cached_generation != generation:
            try:
                await asyncio.to_thread(
                    self._redis.set,
                    generation_key,
                    generation,
                    ex=self._ttl,
                )
            except Exception:
                pass
        return generation

    async def get(
        self, prompt_key: PromptKey, answer_mode: AnswerMode, scope: PromptScope
    ) -> PromptRecord | None:
        generation = await self._generation(prompt_key, answer_mode)
        key = self._key(prompt_key, answer_mode, scope, generation)
        try:
            raw = await asyncio.to_thread(self._redis.get, key)
            return _deserialize_record(raw) if raw is not None else None
        except Exception:
            return None

    async def set(self, scope: PromptScope, record: PromptRecord) -> None:
        generation = await self._generation(record.prompt_key, record.answer_mode)
        try:
            await asyncio.to_thread(
                self._redis.set,
                self._key(record.prompt_key, record.answer_mode, scope, generation),
                _serialize_record(record),
                ex=self._ttl,
            )
        except Exception:
            return None

    async def invalidate_family(
        self, prompt_key: PromptKey, answer_mode: AnswerMode
    ) -> None:
        # The repository already bumped the PostgreSQL generation in the same
        # transaction as activation. Publish that committed value to shared
        # Redis so every service instance observes it on its next request.
        value = await self._conn.fetchval(
            """SELECT generation FROM cache_generations
               WHERE namespace='prompt' AND cache_key=$1""",
            self._family(prompt_key, answer_mode),
        )
        try:
            await asyncio.to_thread(
                self._redis.set,
                self._generation_key(prompt_key, answer_mode),
                int(value or 1),
                ex=self._ttl,
            )
        except Exception:
            return None

    @staticmethod
    def _key(
        prompt_key: PromptKey,
        answer_mode: AnswerMode,
        scope: PromptScope,
        generation: int,
    ) -> str:
        payload = json.dumps(
            [
                prompt_key.value,
                answer_mode.value,
                scope.board_id,
                scope.class_id,
                scope.subject_id,
            ],
            separators=(",", ":"),
        )
        import hashlib

        return f"prompts:active:v2:{generation}:{hashlib.sha256(payload.encode()).hexdigest()}"

    @staticmethod
    def _generation_key(prompt_key: PromptKey, answer_mode: AnswerMode) -> str:
        return f"prompts:generation:v2:{prompt_key.value}:{answer_mode.value}"
