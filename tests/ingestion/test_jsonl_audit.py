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


@pytest.mark.asyncio
async def test_audit_repository_sanitization_without_database():
    """Unit test: verifies AuditRepository sanitization payload structure with mocked connection."""
    from unittest.mock import AsyncMock, MagicMock
    from app.repositories.audit_repository import AuditRepository

    mock_conn = MagicMock()
    mock_conn.fetchrow = AsyncMock(
        side_effect=lambda query, actor_id, action, target_type, target_id, before, after: {
            "actor_id": actor_id,
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "before_value": before,
            "after_value": json.loads(after),
        }
    )

    repo = AuditRepository(mock_conn)
    sensitive_source = "SECRET_RAW_JSONL_WITH_CHUNK_TEXT_AND_EXPECTED_QUESTIONS"
    import hashlib
    source_hash = hashlib.sha256(sensitive_source.encode()).hexdigest()

    result = await repo.create_jsonl_ingestion_audit(
        actor_id="admin-user-777",
        request_id="req-audit-999",
        scope={
            "board_id": "fbise",
            "class_id": "class_9",
            "subject_id": "physics",
            "chapter_id": "ch_1",
        },
        outcome="rejected",
        error_code="JSONL_SCOPE_MISMATCH",
        job_id=None,
        source_hash=source_hash,
        idempotency_key_hash="hash-123",
    )

    assert result["actor_id"] == "admin-user-777"
    assert result["action"] == "jsonl_ingest_submission"
    assert result["target_type"] == "jsonl_ingest"
    assert result["target_id"] == source_hash

    after = result["after_value"]
    assert after["outcome"] == "rejected"
    assert after["error_code"] == "JSONL_SCOPE_MISMATCH"
    assert after["board_id"] == "fbise"
    assert after["class_id"] == "class_9"
    assert after["subject_id"] == "physics"
    assert after["chapter_id"] == "ch_1"
    assert after["source_hash"] == source_hash

    # Assert zero sensitive text in the serialized audit payload
    serialized = json.dumps(result)
    assert sensitive_source not in serialized
    assert "chunk_text" not in serialized
    assert "expected_questions" not in serialized
    assert "secret" not in serialized.lower()

