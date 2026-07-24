"""Database Migration Runner.

Loads and executes SQL migrations in sorted sequence and tracks applied files
in the schema_migrations table.
"""

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

INIT_MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def run_migrations(connection: asyncpg.Connection) -> list[str]:
    """Applies all pending SQL migration files from the migrations directory."""
    await connection.execute(INIT_MIGRATION_TABLE_SQL)

    rows = await connection.fetch("SELECT version FROM schema_migrations;")
    applied = {row["version"] for row in rows}

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    newly_applied = []

    for filepath in migration_files:
        version = filepath.name
        if version in applied:
            continue

        logger.info(f"Applying database migration: {version}")
        sql_content = filepath.read_text(encoding="utf-8")

        async with connection.transaction():
            await connection.execute(sql_content)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1);", version
            )
        newly_applied.append(version)

    return newly_applied
