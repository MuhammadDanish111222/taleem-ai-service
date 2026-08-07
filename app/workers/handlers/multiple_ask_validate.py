"""Railway-owned validation stage which durably schedules post-quota extraction."""

from __future__ import annotations

from typing import Any

import asyncpg

from app.services.multiple_ask import MultipleAskService
from app.services.multiple_ask_extraction_service import MultipleAskExtractionService


async def handle_multiple_ask_validate(
    job: dict[str, Any], conn: asyncpg.Connection
) -> None:
    payload = job.get("payload") or {}
    session_id = payload.get("multiple_ask_session_id")
    if not isinstance(session_id, str):
        raise ValueError("MULTIPLE_ASK_JOB_PAYLOAD_INVALID")
    # Validation itself reads bounded private bytes only and must finish before
    # any OCR is considered. The next durable job owns OCR/extraction.
    workflow_status = await MultipleAskService(conn).validate_and_charge(session_id)
    if workflow_status == "validated":
        await MultipleAskExtractionService(conn).start_initial_extraction(session_id)
