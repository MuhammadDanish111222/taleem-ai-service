"""Bounded, auditable retention for unapproved generated-answer candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import asyncpg

from app.repositories.audit_repository import AuditRepository


@dataclass(frozen=True, slots=True)
class CandidateRetentionCounts:
    eligible_answers: int
    eligible_requests_without_answer: int

    @property
    def total(self) -> int:
        return self.eligible_answers + self.eligible_requests_without_answer


class CandidateRetentionService:
    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    async def preview(self) -> CandidateRetentionCounts:
        row = await self._conn.fetchrow(
            """SELECT
                 (
                   SELECT COUNT(*)
                   FROM ai_answers a
                   WHERE a.review_status IN ('pending','rejected')
                     AND a.approved_revision_id IS NULL
                     AND a.retention_expires_at IS NOT NULL
                     AND a.retention_expires_at <= NOW()
                 ) AS eligible_answers,
                 (
                   SELECT COUNT(*)
                   FROM ai_requests r
                   WHERE r.status IN ('pending','failed')
                     AND r.retention_expires_at IS NOT NULL
                     AND r.retention_expires_at <= NOW()
                     AND NOT EXISTS(
                       SELECT 1 FROM ai_answers a WHERE a.request_id=r.id
                     )
                 ) AS eligible_requests_without_answer"""
        )
        return CandidateRetentionCounts(
            eligible_answers=int(row["eligible_answers"]),
            eligible_requests_without_answer=int(
                row["eligible_requests_without_answer"]
            ),
        )

    async def cleanup(
        self, *, actor_id: str, limit: int = 100
    ) -> CandidateRetentionCounts:
        if not actor_id.strip():
            raise ValueError("RETENTION_ACTOR_REQUIRED")
        if not 1 <= limit <= 500:
            raise ValueError("RETENTION_LIMIT_OUT_OF_RANGE")

        async with self._conn.transaction():
            deleted_answers = await self._conn.fetch(
                """WITH eligible AS (
                     SELECT a.id
                     FROM ai_answers a
                     WHERE a.review_status IN ('pending','rejected')
                       AND a.approved_revision_id IS NULL
                       AND a.retention_expires_at IS NOT NULL
                       AND a.retention_expires_at <= NOW()
                     ORDER BY a.retention_expires_at,a.id
                     FOR UPDATE SKIP LOCKED
                     LIMIT $1
                   )
                   DELETE FROM ai_answers a USING eligible e
                   WHERE a.id=e.id
                   RETURNING a.request_id""",
                limit,
            )
            remaining = limit - len(deleted_answers)
            deleted_request_ids: set[str] = set()
            if deleted_answers:
                rows = await self._conn.fetch(
                    """DELETE FROM ai_requests r
                       WHERE r.id=ANY($1::uuid[])
                         AND r.retention_expires_at IS NOT NULL
                         AND r.retention_expires_at <= NOW()
                         AND NOT EXISTS(
                           SELECT 1 FROM ai_answers a WHERE a.request_id=r.id
                         )
                       RETURNING r.id::text""",
                    [str(row["request_id"]) for row in deleted_answers],
                )
                deleted_request_ids.update(row["id"] for row in rows)
            orphan_count = 0
            if remaining > 0:
                rows = await self._conn.fetch(
                    """WITH eligible AS (
                         SELECT r.id
                         FROM ai_requests r
                         WHERE r.status IN ('pending','failed')
                           AND r.retention_expires_at IS NOT NULL
                           AND r.retention_expires_at <= NOW()
                           AND NOT EXISTS(
                             SELECT 1 FROM ai_answers a WHERE a.request_id=r.id
                           )
                         ORDER BY r.retention_expires_at,r.id
                         FOR UPDATE SKIP LOCKED
                         LIMIT $1
                       )
                       DELETE FROM ai_requests r USING eligible e
                       WHERE r.id=e.id
                       RETURNING r.id::text""",
                    remaining,
                )
                newly_deleted = {
                    row["id"] for row in rows if row["id"] not in deleted_request_ids
                }
                orphan_count = len(newly_deleted)
                deleted_request_ids.update(newly_deleted)

            counts = CandidateRetentionCounts(
                eligible_answers=len(deleted_answers),
                eligible_requests_without_answer=orphan_count,
            )
            await AuditRepository(self._conn).create_audit_log(
                actor_id=actor_id,
                action="candidate.retention_cleanup",
                target_type="candidate_retention",
                target_id="bounded_cleanup",
                after_value={
                    **asdict(counts),
                    "deleted_requests_total": len(deleted_request_ids),
                    "batch_limit": limit,
                },
            )
        return counts
