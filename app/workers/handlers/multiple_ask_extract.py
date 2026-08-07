"""Railway-owned OCR/extraction stage; deliberately no answer generation."""

from __future__ import annotations

from typing import Any

import asyncpg

from app.services.multiple_ask_extraction_service import MultipleAskExtractionService


async def handle_multiple_ask_extract(
    job: dict[str, Any], conn: asyncpg.Connection
) -> None:
    payload = job.get("payload") or {}
    session_id, epoch = payload.get("multiple_ask_session_id"), payload.get("epoch")
    if not isinstance(session_id, str) or not isinstance(epoch, int):
        raise ValueError("MULTIPLE_ASK_JOB_PAYLOAD_INVALID")
    await MultipleAskExtractionService(conn).extract(
        session_id=session_id, epoch=epoch, resume=payload.get("resume") is True
    )
