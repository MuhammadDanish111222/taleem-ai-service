import os
import asyncio
import pytest
import asyncpg
from app.db.migrator import run_migrations

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/taleem_dev")

async def _ensure_migrated():
    conn = await asyncpg.connect(DB_URL)
    try:
        await run_migrations(conn)
    finally:
        await conn.close()

@pytest.fixture(autouse=True, scope="session")
def ensure_db_migrated():
    """Applies database migrations if needed once per test execution session."""
    asyncio.run(_ensure_migrated())
