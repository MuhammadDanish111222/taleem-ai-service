import os

import asyncpg
import pytest
import redis

from app.repositories.prompt_cache import SharedPromptCache
from app.repositories.prompt_repository import PostgresPromptRepository
from app.services.prompts.models import AnswerMode, PromptKey, PromptScope
from app.services.prompts.service import PromptService

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(DB_URL)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


@pytest.mark.asyncio
async def test_postgres_prompt_hierarchy_and_shared_generation_invalidation(conn):
    cache = SharedPromptCache(
        conn,
        redis.Redis.from_url(REDIS_URL, decode_responses=True),
        ttl_seconds=60,
    )
    service = PromptService(PostgresPromptRepository(conn), cache=cache)
    scopes = [
        PromptScope(),
        PromptScope(subject_id="physics"),
        PromptScope(class_id="class-9", subject_id="physics"),
        PromptScope(board_id="punjab", class_id="class-9", subject_id="physics"),
    ]
    for index, scope in enumerate(scopes, start=1):
        draft = await service.create_draft(
            prompt_key=PromptKey.ASK_GROUNDED,
            answer_mode=AnswerMode.SHORT,
            scope=scope,
            content=f"Teaching prompt {index}",
            actor_id="admin",
        )
        await service.activate(prompt_id=draft.id, actor_id="admin")

    resolved = await service.resolve_active(
        prompt_key=PromptKey.ASK_GROUNDED,
        answer_mode=AnswerMode.SHORT,
        scope=PromptScope(board_id="punjab", class_id="class-9", subject_id="physics"),
    )
    assert resolved.record.content == "Teaching prompt 4"
    generation = await conn.fetchval(
        """SELECT generation FROM cache_generations
           WHERE namespace='prompt' AND cache_key='ask_grounded:short'"""
    )
    assert generation == 5
    assert (
        await conn.fetchval(
            """SELECT COUNT(*) FROM admin_audit_logs
               WHERE action='prompt.activated'"""
        )
        == 4
    )
