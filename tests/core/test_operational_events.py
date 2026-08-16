from unittest.mock import AsyncMock

import pytest

from app.services.operational_events import record_operational_event
from app.services.operations_dashboard import safe_failure


@pytest.mark.asyncio
async def test_operational_event_is_content_free_and_fail_soft():
    conn = AsyncMock()
    await record_operational_event(
        conn,
        feature="single_question",
        event_type="retrieval_outcome",
        outcome="empty",
        error_code=None,
        request_id="123e4567-e89b-42d3-a456-426614174000",
    )
    sql, *values = conn.execute.await_args.args
    assert "question" not in sql.lower() and "metadata" not in sql.lower()
    assert values == [
        "single_question",
        "retrieval_outcome",
        "empty",
        None,
        "123e4567-e89b-42d3-a456-426614174000",
        None,
    ]
    conn.execute.side_effect = RuntimeError("database detail must not escape")
    await record_operational_event(
        conn, feature="single_question", event_type="quota_block", outcome="denied"
    )


def test_unknown_operational_error_code_is_generic():
    safe = safe_failure(
        {
            "source": "provider_attempts",
            "error_code": "raw provider response",
            "created_at": type("T", (), {"isoformat": lambda _: "now"})(),
        }
    )
    assert safe["message"] == "Operation failed."
    assert "raw provider response" not in safe["message"]
    assert safe["error_code"] == "OPERATION_FAILED"
