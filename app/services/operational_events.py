"""Fail-soft, content-free operational event recording."""

from __future__ import annotations

import logging
from typing import Literal

import asyncpg

logger = logging.getLogger(__name__)

EventType = Literal["retrieval_outcome", "quota_block", "test_generation_failure"]


async def record_operational_event(
    conn: asyncpg.Connection,
    *,
    feature: str,
    event_type: EventType,
    outcome: str,
    error_code: str | None = None,
    request_id: str | None = None,
    job_id: str | None = None,
) -> None:
    """Best-effort telemetry; never let a dashboard write change request behaviour."""
    try:
        await conn.execute(
            """INSERT INTO operational_events(feature,event_type,outcome,error_code,request_id,job_id)
               VALUES($1,$2,$3,$4,$5::uuid,$6::uuid)""",
            feature,
            event_type,
            outcome,
            error_code,
            request_id,
            job_id,
        )
    except Exception:
        logger.warning(
            "operational_event_recording_failed", extra={"event_type": event_type}
        )
