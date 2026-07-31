from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
import redis

from app.core.config import get_settings
from app.services.usage.models import AccountTier
from app.services.usage.service import UsageLimitExceeded, UsageService

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(autouse=True)
def usage_secrets(monkeypatch):
    monkeypatch.setenv("USAGE_UID_HMAC_SECRET", "module4-test-usage-secret")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _reserve(uid: str, request_id: str, *, redis_client=None):
    conn = await asyncpg.connect(DB_URL)
    try:
        async with conn.transaction():
            return await UsageService(redis_client=redis_client).reserve(
                conn,
                request_id=request_id,
                uid=uid,
                tier=AccountTier.ANONYMOUS,
            )
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_concurrent_near_limit_allows_exactly_one_remaining():
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        await asyncio.to_thread(client.ping)
        probe = await asyncpg.connect(DB_URL)
        await probe.close()
    except (ConnectionRefusedError, OSError, redis.RedisError):
        pytest.skip("Disposable PostgreSQL/Redis is unavailable")

    uid = f"concurrent-{uuid.uuid4()}"
    for _ in range(4):
        await _reserve(uid, str(uuid.uuid4()), redis_client=client)

    results = await asyncio.gather(
        *(_reserve(uid, str(uuid.uuid4()), redis_client=client) for _ in range(10)),
        return_exceptions=True,
    )
    allowed = [item for item in results if not isinstance(item, BaseException)]
    blocked = [item for item in results if isinstance(item, UsageLimitExceeded)]
    assert len(allowed) == 1
    assert len(blocked) == 9
    assert allowed[0].used == 5
    assert all(item.used == 5 and item.limit == 5 for item in blocked)
    assert all(item.student_visible for item in blocked)
    assert all(item.resets_at.tzinfo is not None for item in blocked)


@pytest.mark.asyncio
async def test_idempotent_retry_does_not_increment_twice():
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        await asyncio.to_thread(client.ping)
    except redis.RedisError:
        pytest.skip("Disposable Redis is unavailable")
    uid = f"idempotent-{uuid.uuid4()}"
    request_id = str(uuid.uuid4())
    first, second = await asyncio.gather(
        _reserve(uid, request_id, redis_client=client),
        _reserve(uid, request_id, redis_client=client),
    )
    assert first.used == second.used == 1
    assert first.duplicate is not second.duplicate


@pytest.mark.asyncio
async def test_same_client_request_id_is_isolated_between_identities():
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        await asyncio.to_thread(client.ping)
    except redis.RedisError:
        pytest.skip("Disposable Redis is unavailable")
    request_id = str(uuid.uuid4())
    first, second = await asyncio.gather(
        _reserve(f"identity-a-{uuid.uuid4()}", request_id, redis_client=client),
        _reserve(f"identity-b-{uuid.uuid4()}", request_id, redis_client=client),
    )
    assert first.used == second.used == 1
    assert not first.duplicate
    assert not second.duplicate


class FailingRedis:
    def eval(self, *_args, **_kwargs):
        raise redis.ConnectionError("unavailable")


@pytest.mark.asyncio
async def test_redis_failure_uses_guarded_postgres_fallback():
    uid = f"fallback-{uuid.uuid4()}"
    results = await asyncio.gather(
        *(
            _reserve(uid, str(uuid.uuid4()), redis_client=FailingRedis())
            for _ in range(8)
        ),
        return_exceptions=True,
    )
    allowed = [item for item in results if not isinstance(item, BaseException)]
    blocked = [item for item in results if isinstance(item, UsageLimitExceeded)]
    assert len(allowed) == 5
    assert len(blocked) == 3
    assert {item.backend for item in allowed} == {"postgresql"}
