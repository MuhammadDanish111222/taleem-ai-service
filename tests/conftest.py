import asyncio
import os

import asyncpg
import pytest

from app.db.migrator import run_migrations

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/taleem_dev"
)


async def _ensure_migrated():
    conn = await asyncpg.connect(DB_URL)
    try:
        await run_migrations(conn)
    finally:
        await conn.close()


@pytest.fixture(autouse=True, scope="session")
def ensure_db_migrated():
    """Applies database migrations once per session when PostgreSQL is reachable.

    Tests that do not need a database (pure unit tests) should still run when
    the database is unavailable — the fixture silently skips migration in that
    case.  Tests that actually open a connection will fail naturally on their
    own if PostgreSQL is not available.
    """
    try:
        asyncio.run(_ensure_migrated())
    except (ConnectionRefusedError, OSError):
        # PostgreSQL is not running locally — skip migration.
        # Pure unit tests will still execute; integration tests that need
        # a real connection will fail at their own connection step.
        pass
