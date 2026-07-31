"""Fresh/upgrade compatibility for the forward-only Module 4 migration."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)


def _database_url(name: str) -> str:
    parsed = urlsplit(TEST_DATABASE_URL)
    return urlunsplit(parsed._replace(path=f"/{name}"))


@pytest.mark.asyncio
async def test_upgrade_preserves_legacy_answer_and_bank_rows():
    database_name = f"taleem_m4_upgrade_{uuid.uuid4().hex[:12]}"
    try:
        admin = await asyncpg.connect(_database_url("postgres"))
    except (ConnectionRefusedError, OSError):
        pytest.skip("Disposable PostgreSQL is unavailable")
    try:
        await admin.execute(f'CREATE DATABASE "{database_name}"')
    finally:
        await admin.close()

    connection = await asyncpg.connect(_database_url(database_name))
    try:
        for migration in sorted(MIGRATIONS.glob("*.sql")):
            if migration.name >= "0009_module4_ask_foundation.sql":
                break
            await connection.execute(migration.read_text(encoding="utf-8"))

        legacy_request_id = await connection.fetchval(
            """INSERT INTO ai_requests(
                 board_id,class_id,subject_id,language,answer_mode,raw_question,
                 normalized_question,question_hash,prompt_version,status
               ) VALUES(
                 'punjab','class-9','physics','en','concise','What is force?',
                 'what is force',$1,'v1','completed'
               ) RETURNING id""",
            "a" * 64,
        )
        await connection.execute(
            """INSERT INTO ai_answers(request_id,answer_text)
               VALUES($1,'Force is a push or pull.')""",
            legacy_request_id,
        )
        legacy_bank_id = await connection.fetchval(
            """INSERT INTO approved_question_bank(
                 board_id,class_id,subject_id,normalized_question,question_hash,
                 answer_text,status,source
               ) VALUES(
                 'punjab','class-9','physics','what is force',$1,
                 'Force is a push or pull.','approved','legacy_admin'
               ) RETURNING id""",
            "a" * 64,
        )

        await connection.execute(
            (MIGRATIONS / "0009_module4_ask_foundation.sql").read_text(encoding="utf-8")
        )

        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM ai_requests WHERE id=$1", legacy_request_id
            )
            == 1
        )
        assert (
            await connection.fetchval(
                "SELECT COUNT(*) FROM ai_answers WHERE request_id=$1",
                legacy_request_id,
            )
            == 1
        )
        revision = await connection.fetchrow(
            """SELECT q.id,r.review_status,r.source,r.answer_blocks
               FROM question_bank_questions q
               JOIN question_bank_revisions r ON r.question_id=q.id
               WHERE q.id=$1""",
            legacy_bank_id,
        )
        assert revision["id"] == legacy_bank_id
        assert revision["review_status"] == "approved"
        assert revision["source"] == "legacy_admin"
        blocks = revision["answer_blocks"]
        if isinstance(blocks, str):
            blocks = json.loads(blocks)
        assert blocks[0]["text"] == "Force is a push or pull."
        assert not await connection.fetchval(
            """SELECT EXISTS(
                 SELECT 1 FROM pg_class WHERE relname='approved_question_bank'
               )"""
        )
    finally:
        await connection.close()
        admin = await asyncpg.connect(_database_url("postgres"))
        try:
            await admin.execute(
                """SELECT pg_terminate_backend(pid) FROM pg_stat_activity
                   WHERE datname=$1""",
                database_name,
            )
            await admin.execute(f'DROP DATABASE "{database_name}"')
        finally:
            await admin.close()
