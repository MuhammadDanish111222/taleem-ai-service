"""Run 2 safety regressions that do not require a live PostgreSQL server."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.api.v1.internal import _multiple_ask_status_response
from app.services.multiple_ask_extraction_service import (
    MultipleAskExtractionError,
    MultipleAskExtractionService,
)
from app.services.usage.service import UsageService


def test_corrected_mcq_requires_exactly_four_ordered_options() -> None:
    validate = MultipleAskExtractionService._validate_correction_options
    assert validate(
        "mcq",
        [
            {"label": "A", "text": "one"},
            {"label": "B", "text": "two"},
            {"label": "C", "text": "three"},
            {"label": "D", "text": "four"},
        ],
    ) == [
        {"label": "A", "text": "one"},
        {"label": "B", "text": "two"},
        {"label": "C", "text": "three"},
        {"label": "D", "text": "four"},
    ]
    with pytest.raises(MultipleAskExtractionError, match="OPTIONS_INVALID"):
        validate("mcq", [{"label": "A", "text": "one"}])
    with pytest.raises(MultipleAskExtractionError, match="OPTIONS_INVALID"):
        validate("short", [{"label": "A", "text": "one"}])


@pytest.mark.asyncio
async def test_post_charge_extraction_failure_refunds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Connection:
        def transaction(self):
            return Transaction()

    class Repository:
        def __init__(self):
            self.context = {
                "id": "job-1",
                "client_request_id": "request-1",
                "uid_hash": "hash-1",
                "workflow_status": "extracting",
                "quota_status": "committed",
            }
            self.terminal_calls: list[dict[str, object]] = []

        async def lock_extraction_context(self, _session_id):
            return self.context

        async def mark_terminal(self, _job_id, workflow_status, _expiry, **kwargs):
            self.context["workflow_status"] = workflow_status
            self.context["quota_status"] = "refunded"
            self.terminal_calls.append(kwargs)

    calls: list[tuple[str, str]] = []

    async def refund_once(_self, _conn, request_id, uid_hash):
        calls.append((request_id, uid_hash))
        return True

    monkeypatch.setattr(UsageService, "refund_committed", refund_once)
    service = MultipleAskExtractionService(Connection(), storage=object(), ocr=object())
    repository = Repository()
    service._repo = repository

    await service.fail_and_refund(
        session_id="session-1",
        workflow_status="failed",
        error_code="MULTIPLE_ASK_EXTRACTION_FAILED",
    )
    await service.fail_and_refund(
        session_id="session-1",
        workflow_status="failed",
        error_code="MULTIPLE_ASK_EXTRACTION_FAILED",
    )

    assert calls == [("request-1", "hash-1")]
    assert repository.terminal_calls == [
        {"error_code": "MULTIPLE_ASK_EXTRACTION_FAILED", "quota_refunded": True}
    ]


def test_polling_response_has_retention_terminal_state_and_safe_item_context() -> None:
    now = datetime(2026, 8, 6, tzinfo=UTC)
    response = _multiple_ask_status_response(
        {
            "id": "job-1",
            "workflow_status": "too_many_questions",
            "input_kind": "pdf",
            "board_id": "punjab",
            "class_id": "class-9",
            "subject_id": "chemistry",
            "chapter_id": None,
            "created_at": now,
            "updated_at": now,
            "retention_expires_at": now,
            "terminal_error_code": "MULTIPLE_ASK_TOO_MANY_QUESTIONS",
            "queue_status": "failed",
            "queue_stage": "extracting",
            "queue_progress": 100,
            "items": [
                {
                    "id": "item-1",
                    "item_index": 0,
                    "display_label": "2(ii)",
                    "section_context": "Write short answers",
                    "item_status": "needs_correction",
                    "normalized_question": None,
                    "answer_mode": "not_clear",
                    "mcq_options": [],
                    "unclear_reason": "QUESTION_TEXT_UNCLEAR",
                    "source_locator": {"page_number": 1},
                    "extraction_version": 1,
                    "correction_version": 0,
                    "corrected_at": None,
                }
            ],
        }
    )
    assert response["retention_expires_at"] == now.isoformat()
    assert response["terminal_error_code"] == "MULTIPLE_ASK_TOO_MANY_QUESTIONS"
    assert response["summary"] == {
        "total": 1,
        "short": 0,
        "long": 0,
        "mcq": 0,
        "not_clear": 1,
    }
    assert response["items"][0]["display_label"] == "2(ii)"
    assert "source_text" not in str(response)
