"""JSONL audit tests against the real disposable PostgreSQL schema."""

import json
import uuid
from contextlib import asynccontextmanager

import asyncpg
import pytest

from app.api.v1 import internal
from app.core.internal_auth import AuthContext


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(
        "postgresql://postgres:postgres@localhost:5432/taleem_dev"
    )
    tx = connection.transaction()
    await tx.start()
    yield connection
    await tx.rollback()
    await connection.close()


@pytest.mark.asyncio
async def test_accepted_and_rejected_jsonl_actions_are_auditable_without_source_content(
    conn, monkeypatch
):
    @asynccontextmanager
    async def fake_connection():
        yield conn

    scope = {
        "board_id": "fbise",
        "class_id": "class_9",
        "subject_id": "physics",
        "chapter_id": "ch_1",
    }
    chunks = [{**scope, "chunk_order": 0}]
    monkeypatch.setattr(internal, "get_db_connection", fake_connection)
    monkeypatch.setattr(internal, "_get_firestore_db", lambda: object())
    monkeypatch.setattr(
        internal,
        "validate_and_parse_jsonl",
        lambda *_args, **_kwargs: _validated(chunks, []),
    )

    auth = AuthContext(
        uid="admin-audit", is_admin=True, feature="jsonl_ingest", request_id="req-audit"
    )
    accepted = await internal.submit_jsonl_ingest(
        internal.JsonlIngestRequest(
            jsonl_content="PRIVATE_JSONL_CONTENT", idempotency_key=f"key-{uuid.uuid4()}"
        ),
        auth,
    )
    assert accepted["status"] == "queued"

    accepted_audit = await conn.fetchrow(
        "SELECT actor_id, action, target_type, target_id, after_value FROM admin_audit_logs WHERE target_id = $1;",
        accepted["job_id"],
    )
    assert accepted_audit["actor_id"] == "admin-audit"
    assert accepted_audit["action"] == "jsonl_ingest_submission"
    assert accepted_audit["after_value"]["outcome"] == "accepted"
    assert accepted_audit["after_value"]["request_id"] == "req-audit"
    assert accepted_audit["after_value"]["board_id"] == "fbise"
    assert "PRIVATE_JSONL_CONTENT" not in json.dumps(dict(accepted_audit))

    errors = [
        {
            "row": 2,
            "field": "scope",
            "reason": "scope_mismatch",
            "code": "JSONL_SCOPE_MISMATCH",
        }
    ]
    monkeypatch.setattr(
        internal,
        "validate_and_parse_jsonl",
        lambda *_args, **_kwargs: _validated([], errors),
    )

    with pytest.raises(Exception) as exc_info:
        await internal.submit_jsonl_ingest(
            internal.JsonlIngestRequest(
                jsonl_content=json.dumps(
                    {**scope, "chunk_text": "PRIVATE_JSONL_CONTENT_REJECTED"}
                )
            ),
            auth,
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "JSONL_SCOPE_MISMATCH"

    rejected_audit = await conn.fetchrow(
        "SELECT after_value FROM admin_audit_logs WHERE after_value->>'outcome' = 'rejected' ORDER BY created_at DESC LIMIT 1;"
    )
    assert rejected_audit["after_value"]["error_code"] == "JSONL_SCOPE_MISMATCH"
    assert rejected_audit["after_value"]["chapter_id"] == "ch_1"
    assert "PRIVATE_JSONL_CONTENT_REJECTED" not in json.dumps(dict(rejected_audit))


async def _validated(chunks, errors):
    return chunks, errors
