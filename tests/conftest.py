"""Pytest global configuration and database fixtures."""

import asyncio
import pytest
import asyncpg
from app.db.migrator import run_migrations

DB_URL = "postgresql://postgres:postgres@localhost:5432/taleem_dev"

async def _reset_and_migrate():
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
        await run_migrations(conn)
    finally:
        await conn.close()

@pytest.fixture(autouse=True, scope="package")
def ensure_db_migrated():
    """Resets public schema and applies all migrations once per test execution package."""
    asyncio.run(_reset_and_migrate())
