import os
import uuid
from datetime import datetime, timezone

import pytest

from app.core.internal_auth import _record_jti_postgres
from app.db.pool import close_db_pool, init_db_pool

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)


@pytest.mark.asyncio
async def test_postgres_jti_claim_is_atomic_and_replay_safe():
    try:
        await init_db_pool(DB_URL)
    except (ConnectionRefusedError, OSError):
        pytest.skip("Disposable PostgreSQL is unavailable")
    digest = uuid.uuid4().hex + uuid.uuid4().hex
    expires_at = datetime.now(timezone.utc).timestamp() + 60
    try:
        first = await _record_jti_postgres(
            jti_hash=digest,
            expires_at=expires_at,
            fallback_event=True,
        )
        second = await _record_jti_postgres(
            jti_hash=digest,
            expires_at=expires_at,
            fallback_event=True,
        )
        assert first is True
        assert second is False
    finally:
        await close_db_pool()
