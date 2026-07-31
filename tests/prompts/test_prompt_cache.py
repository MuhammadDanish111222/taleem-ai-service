from datetime import UTC, datetime

import pytest

from app.services.prompts.cache import RedisPromptCache
from app.services.prompts.models import (
    AnswerMode,
    PromptKey,
    PromptRecord,
    PromptScope,
    PromptStatus,
)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.incremented = []

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value, *, ex):
        self.values[name] = value
        self.values[f"{name}:ttl"] = ex

    def incr(self, name):
        self.incremented.append(name)
        value = int(self.values.get(name, 0)) + 1
        self.values[name] = value
        return value


@pytest.mark.asyncio
async def test_generation_bump_invalidates_without_key_scan():
    redis = FakeRedis()
    cache = RedisPromptCache(redis, ttl_seconds=120)
    scope = PromptScope(board_id="fbise", class_id="10", subject_id="biology")
    record = PromptRecord(
        id="prompt-1",
        prompt_key=PromptKey.ASK_GROUNDED,
        answer_mode=AnswerMode.SHORT,
        scope=PromptScope(subject_id="biology"),
        version=1,
        content="Explain clearly.",
        status=PromptStatus.ACTIVE,
        created_by="admin-1",
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        activated_by="admin-1",
        activated_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    await cache.set(scope, record)
    assert await cache.get(record.prompt_key, record.answer_mode, scope) == record

    await cache.invalidate_family(record.prompt_key, record.answer_mode)

    assert redis.incremented == ["prompts:generation:v1:ask_grounded:short"]
    assert await cache.get(record.prompt_key, record.answer_mode, scope) is None
