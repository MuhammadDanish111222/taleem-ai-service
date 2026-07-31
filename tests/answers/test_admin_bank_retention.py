from __future__ import annotations

import os
import uuid
from datetime import date

import asyncpg
import pytest

from app.repositories.ask_repository import AskRepository
from app.repositories.question_bank_repository import QuestionBankRepository
from app.schemas.ask import AnswerMode, AnswerStyle
from app.schemas.ask_admin import AskAdminRequest
from app.services.answers.normalization import normalize_question, question_hash
from app.services.answers.retention import CandidateRetentionService

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)


def test_admin_list_contract_preserves_optional_chapter_filter():
    request = AskAdminRequest(
        operation="bank_list",
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="chapter-1",
    )
    assert request.chapter_id == "chapter-1"


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(DB_URL)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


async def _approved(conn, question: str) -> str:
    normalized = normalize_question(question)
    return await QuestionBankRepository(conn).create_approved_revision(
        actor_id="admin",
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="motion",
        answer_mode=AnswerMode.SHORT,
        answer_style=AnswerStyle.EXAM_STYLE,
        difficulty="easy",
        marks=2,
        question_text=question,
        normalized_question=normalized,
        question_hash=question_hash(normalized),
        blocks=[{"type": "paragraph", "text": "Approved answer"}],
        source="admin_authored",
    )


@pytest.mark.asyncio
async def test_bank_list_history_variation_embedding_and_archive(conn):
    bank = QuestionBankRepository(conn)
    revision_id = await _approved(conn, "Define momentum.")
    variation = "What is momentum?"
    normalized = normalize_question(variation)
    variation_id = await bank.add_variation(
        revision_id=revision_id,
        variation_text=variation,
        normalized_variation=normalized,
        variation_hash=question_hash(normalized),
        actor_id="admin",
    )

    listed = await bank.list_approved(
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="motion",
        answer_mode="short",
    )
    assert [item["revision_id"] for item in listed] == [revision_id]
    assert listed[0]["embedding_status"] == "pending"

    inactive = await bank.set_variation_active(variation_id=variation_id, active=False)
    assert inactive["active"] is False
    await bank.set_variation_active(variation_id=variation_id, active=True)
    await bank.reset_embedding(
        revision_id=revision_id,
        variation_id=variation_id,
    )
    history = await bank.revision_history(revision_id=revision_id)
    assert history is not None
    assert history["revisions"][0]["review_status"] == "approved"
    assert history["variations"][0]["embedding_status"] == "pending"

    await bank.archive_revision(revision_id=revision_id)
    assert await bank.list_approved(subject_id="physics") == []
    assert (
        await bank.find_exact(
            board_id="punjab",
            class_id="class-9",
            subject_id="physics",
            chapter_id="motion",
            answer_mode=AnswerMode.SHORT,
            normalized_question=normalize_question("Define momentum."),
        )
        is None
    )


@pytest.mark.asyncio
async def test_retention_preview_and_cleanup_are_bounded_and_preserve_attempts(conn):
    asks = AskRepository(conn)
    request_id = str(uuid.uuid4())
    pending = await asks.create_pending(
        client_request_id=request_id,
        uid_hash="a" * 64,
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="motion",
        answer_mode="short",
        answer_style="exam_style",
        raw_question="Temporary candidate",
        normalized_question="temporary candidate",
        question_hash="b" * 64,
        usage_business_date=date.today(),
    )
    await asks.complete(
        ai_request_id=str(pending["id"]),
        answer_source="general_knowledge",
        blocks=[{"type": "paragraph", "text": "Temporary answer"}],
        citations=[],
        visual_ids=[],
        prompt_version="fixture",
        corpus_version_id=None,
        provider="fake",
        model="fake-model",
    )
    await conn.execute(
        """UPDATE ai_requests SET retention_expires_at=NOW()-INTERVAL '1 day'
           WHERE id=$1::uuid""",
        pending["id"],
    )
    await conn.execute(
        """UPDATE ai_answers SET review_status='rejected',
                   retention_expires_at=NOW()-INTERVAL '1 day'
           WHERE request_id=$1::uuid""",
        pending["id"],
    )
    attempt_id = await conn.fetchval(
        """INSERT INTO provider_attempts(
             ai_request_id,provider,model,status
           ) VALUES($1::uuid,'fake','fake-model','success') RETURNING id""",
        pending["id"],
    )
    orphan = await asks.create_pending(
        client_request_id=str(uuid.uuid4()),
        uid_hash="c" * 64,
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="motion",
        answer_mode="short",
        answer_style="exam_style",
        raw_question="Expired failed request",
        normalized_question="expired failed request",
        question_hash="d" * 64,
        usage_business_date=date.today(),
    )
    await conn.execute(
        """UPDATE ai_requests SET status='failed',
                   retention_expires_at=NOW()-INTERVAL '1 day'
           WHERE id=$1::uuid""",
        orphan["id"],
    )

    service = CandidateRetentionService(conn)
    preview = await service.preview()
    assert preview.eligible_answers == 1
    assert preview.eligible_requests_without_answer == 1
    deleted = await service.cleanup(actor_id="admin", limit=1)
    assert deleted.eligible_answers == 1
    assert (
        await conn.fetchval(
            "SELECT COUNT(*) FROM ai_requests WHERE id=$1", orphan["id"]
        )
        == 1
    )
    assert (
        await conn.fetchval(
            "SELECT COUNT(*) FROM ai_requests WHERE id=$1", pending["id"]
        )
        == 0
    )
    attempt = await conn.fetchrow(
        "SELECT ai_request_id FROM provider_attempts WHERE id=$1", attempt_id
    )
    assert attempt is not None
    assert attempt["ai_request_id"] is None
    audit = await conn.fetchrow(
        """SELECT after_value FROM admin_audit_logs
           WHERE action='candidate.retention_cleanup'
           ORDER BY created_at DESC LIMIT 1"""
    )
    assert "Temporary" not in str(audit["after_value"])
    second = await service.cleanup(actor_id="admin", limit=1)
    assert second.eligible_requests_without_answer == 1
    third = await service.cleanup(actor_id="admin", limit=1)
    assert third.total == 0
