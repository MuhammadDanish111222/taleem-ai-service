"""Pytest global configuration and database fixtures."""

import pytest
import asyncpg
from app.db.migrator import run_migrations

DB_URL = "postgresql://postgres:postgres@localhost:5432/taleem_dev"

@pytest.fixture(autouse=True, scope="module")
async def ensure_db_migrated():
    """Resets public schema and applies all migrations once per test module execution."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
        await run_migrations(conn)
    finally:
        await conn.close()
