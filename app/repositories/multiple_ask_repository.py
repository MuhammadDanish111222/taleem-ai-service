"""Durable, private Multiple Ask session, validation, and cleanup persistence."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg


class MultipleAskStateError(ValueError):
    pass


class MultipleAskRepository:
    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    async def create_or_get_session(
        self,
        *,
        session_id: str,
        client_request_id: str,
        uid_hash: str,
        account_tier: str,
        input_kind: str,
        expected_content_type: str | None,
        expected_size_bytes: int | None,
        storage_bucket: str | None,
        storage_object_key: str | None,
        upload_capability_expires_at: datetime | None,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        text: str | None = None,
    ) -> dict[str, Any]:
        row = await self._conn.fetchrow(
            """INSERT INTO multiple_ask_upload_sessions(
                   id,client_request_id,uid_hash,account_tier,input_kind,
                   expected_content_type,expected_size_bytes,storage_bucket,
                   storage_object_key,upload_url_expires_at,upload_capability_expires_at,
                   board_id,class_id,subject_id,chapter_id
                 ) VALUES($1::uuid,$2::uuid,$3,$4,$5,$6,$7,$8,$9,$10,$10,$11,$12,$13,$14)
                 ON CONFLICT(uid_hash,client_request_id) DO NOTHING RETURNING *""",
            session_id,
            client_request_id,
            uid_hash,
            account_tier,
            input_kind,
            expected_content_type,
            expected_size_bytes,
            storage_bucket,
            storage_object_key,
            upload_capability_expires_at,
            board_id,
            class_id,
            subject_id,
            chapter_id,
        )
        if row is not None:
            if text is not None:
                await self._conn.execute(
                    "INSERT INTO multiple_ask_text_inputs(session_id,input_text) VALUES($1::uuid,$2)",
                    session_id,
                    text,
                )
            return dict(row)
        existing = await self._conn.fetchrow(
            """SELECT * FROM multiple_ask_upload_sessions
               WHERE uid_hash=$1 AND client_request_id=$2::uuid""",
            uid_hash,
            client_request_id,
        )
        if existing is None:
            raise RuntimeError("MULTIPLE_ASK_SESSION_CREATE_FAILED")
        result = dict(existing)
        immutable = (
            result["input_kind"],
            result["expected_content_type"],
            result["expected_size_bytes"],
            result["board_id"],
            result["class_id"],
            result["subject_id"],
            result["chapter_id"],
        )
        requested = (
            input_kind,
            expected_content_type,
            expected_size_bytes,
            board_id,
            class_id,
            subject_id,
            chapter_id,
        )
        if immutable != requested:
            raise MultipleAskStateError("MULTIPLE_ASK_IDEMPOTENCY_CONFLICT")
        if text is not None:
            stored = await self._conn.fetchval(
                "SELECT input_text FROM multiple_ask_text_inputs WHERE session_id=$1::uuid",
                result["id"],
            )
            if stored != text:
                raise MultipleAskStateError("MULTIPLE_ASK_IDEMPOTENCY_CONFLICT")
        return result

    async def lock_session(
        self, *, session_id: str, uid_hash: str, client_request_id: str
    ) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            """SELECT * FROM multiple_ask_upload_sessions
               WHERE id=$1::uuid AND uid_hash=$2 AND client_request_id=$3::uuid FOR UPDATE""",
            session_id,
            uid_hash,
            client_request_id,
        )
        return dict(row) if row else None

    async def job_for_session(self, session_id: str) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            "SELECT * FROM multiple_ask_jobs WHERE upload_session_id=$1::uuid",
            session_id,
        )
        return dict(row) if row else None

    async def finalize_with_validation_job(
        self,
        *,
        session: dict[str, Any],
        queue_job_id: str,
        raw_source_expires_at: datetime,
    ) -> dict[str, Any]:
        await self._conn.execute(
            """UPDATE multiple_ask_upload_sessions
               SET status='finalized', finalized_at=NOW(), raw_source_expires_at=$2,
                   updated_at=NOW() WHERE id=$1::uuid""",
            session["id"],
            raw_source_expires_at,
        )
        row = await self._conn.fetchrow(
            """INSERT INTO multiple_ask_jobs(
                   upload_session_id,queue_job_id,uid_hash,client_request_id,account_tier,input_kind,
                   board_id,class_id,subject_id,chapter_id,workflow_status
                 ) VALUES($1::uuid,$2::uuid,$3,$4::uuid,$5,$6,$7,$8,$9,$10,'queued') RETURNING *""",
            session["id"],
            queue_job_id,
            session["uid_hash"],
            session["client_request_id"],
            session["account_tier"],
            session["input_kind"],
            session["board_id"],
            session["class_id"],
            session["subject_id"],
            session["chapter_id"],
        )
        return dict(row)

    async def lock_validation_context(self, session_id: str) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            """SELECT j.*, s.storage_bucket AS source_bucket, s.storage_object_key AS source_key,
                      s.expected_content_type, s.expected_size_bytes, s.raw_source_expires_at,
                      s.raw_source_purged_at, s.status AS session_status,
                      t.input_text
               FROM multiple_ask_jobs j
               JOIN multiple_ask_upload_sessions s ON s.id=j.upload_session_id
               LEFT JOIN multiple_ask_text_inputs t ON t.session_id=s.id
               WHERE s.id=$1::uuid FOR UPDATE OF j, s""",
            session_id,
        )
        return dict(row) if row else None

    async def mark_validating(self, job_id: str) -> bool:
        return bool(
            await self._conn.fetchval(
                """UPDATE multiple_ask_jobs SET workflow_status='validating', updated_at=NOW()
               WHERE id=$1::uuid AND workflow_status='queued' RETURNING TRUE""",
                job_id,
            )
        )

    async def mark_validated_and_charged(self, job_id: str) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_jobs SET workflow_status='validated', quota_status='committed',
                   updated_at=NOW() WHERE id=$1::uuid""",
            job_id,
        )

    async def lock_extraction_context(self, session_id: str) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            """SELECT j.*, s.storage_bucket AS source_bucket, s.storage_object_key AS source_key,
                      s.expected_content_type, t.input_text,
                      n.normalized_text, n.source_kind AS normalized_source_kind,
                      n.source_locators
               FROM multiple_ask_jobs j
               JOIN multiple_ask_upload_sessions s ON s.id=j.upload_session_id
               LEFT JOIN multiple_ask_text_inputs t ON t.session_id=s.id
               LEFT JOIN multiple_ask_normalized_sources n ON n.session_id=s.id
               WHERE s.id=$1::uuid FOR UPDATE OF j, s""",
            session_id,
        )
        return dict(row) if row else None

    async def save_normalized_source_once(
        self,
        *,
        session_id: str,
        normalized_text: str,
        source_locators: list[dict[str, Any]],
        source_kind: str,
        ocr_provider: str | None,
    ) -> dict[str, Any]:
        row = await self._conn.fetchrow(
            """INSERT INTO multiple_ask_normalized_sources(
                   session_id,normalized_text,source_locators,source_kind,ocr_provider
                 ) VALUES($1::uuid,$2,$3::jsonb,$4,$5)
                 ON CONFLICT(session_id) DO NOTHING RETURNING *""",
            session_id,
            normalized_text,
            json.dumps(source_locators),
            source_kind,
            ocr_provider,
        )
        if row is None:
            row = await self._conn.fetchrow(
                "SELECT * FROM multiple_ask_normalized_sources WHERE session_id=$1::uuid",
                session_id,
            )
        if row is None:
            raise RuntimeError("MULTIPLE_ASK_SOURCE_CACHE_FAILED")
        return dict(row)

    async def start_extraction(
        self, *, job_id: str, queue_job_id: str, epoch: int
    ) -> bool:
        return bool(
            await self._conn.fetchval(
                """UPDATE multiple_ask_jobs
                   SET workflow_status='extracting', queue_job_id=$2::uuid,
                       extraction_epoch=$3, updated_at=NOW()
                   WHERE id=$1::uuid AND workflow_status IN ('validated','needs_correction')
                     AND extraction_epoch=$3-1 RETURNING TRUE""",
                job_id,
                queue_job_id,
                epoch,
            )
        )

    async def insert_extracted_items(
        self, *, job_id: str, items: list[dict[str, Any]]
    ) -> None:
        for item in items:
            await self._conn.execute(
                """INSERT INTO multiple_ask_job_items(
                       multiple_ask_job_id,item_index,display_label,section_context,source_text,source_locator,extracted_text,
                       normalized_question,question_hash,answer_mode,item_status,mcq_options,
                       unclear_reason,retention_expires_at
                   ) VALUES($1::uuid,$2,$3,$4,$5,$6::jsonb,$5,$7,$8,$9,$10,$11::jsonb,$12,NULL)""",
                job_id,
                item["item_index"],
                item["display_label"],
                item["section_context"],
                item["question_text"],
                json.dumps(item["source_locator"]),
                item["normalized_question"],
                item["question_hash"],
                item["answer_mode"],
                item["item_status"],
                json.dumps(item["mcq_options"]),
                item["unclear_reason"],
            )

    async def update_extracted_correction_item(
        self, *, item_id: str, item: dict[str, Any]
    ) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_job_items SET source_text=$2,extracted_text=$2,
                   normalized_question=$3,question_hash=$4,answer_mode=$5,item_status=$6,
                   mcq_options=$7::jsonb,unclear_reason=$8,extraction_version=extraction_version+1,
                   updated_at=NOW() WHERE id=$1::uuid AND item_status='pending_extraction'""",
            item_id,
            item["question_text"],
            item["normalized_question"],
            item["question_hash"],
            item["answer_mode"],
            item["item_status"],
            json.dumps(item["mcq_options"]),
            item["unclear_reason"],
        )

    async def finish_extraction(self, *, job_id: str, needs_correction: bool) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_jobs SET workflow_status=$2,updated_at=NOW()
               WHERE id=$1::uuid AND workflow_status='extracting'""",
            job_id,
            "needs_correction" if needs_correction else "ready_to_answer",
        )

    async def start_answering(
        self, *, job_id: str, queue_job_id: str, epoch: int
    ) -> bool:
        return bool(
            await self._conn.fetchval(
                """UPDATE multiple_ask_jobs
                   SET workflow_status='answering',queue_job_id=$2::uuid,
                       answer_epoch=$3,updated_at=NOW()
                   WHERE id=$1::uuid AND workflow_status='ready_to_answer'
                     AND answer_epoch=$3-1
                   RETURNING TRUE""",
                job_id,
                queue_job_id,
                epoch,
            )
        )

    async def lock_answer_context(self, session_id: str) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            """SELECT j.* FROM multiple_ask_jobs j
               JOIN multiple_ask_upload_sessions s ON s.id=j.upload_session_id
               WHERE s.id=$1::uuid FOR UPDATE OF j""",
            session_id,
        )
        return dict(row) if row else None

    async def link_item_request(self, *, item_id: str, ai_request_id: str) -> bool:
        return bool(
            await self._conn.fetchval(
                """UPDATE multiple_ask_job_items
                   SET ai_request_id=COALESCE(ai_request_id,$2::uuid),updated_at=NOW()
                   WHERE id=$1::uuid
                     AND (ai_request_id IS NULL OR ai_request_id=$2::uuid)
                   RETURNING TRUE""",
                item_id,
                ai_request_id,
            )
        )

    async def mark_item_answering(self, item_id: str) -> bool:
        return bool(
            await self._conn.fetchval(
                """UPDATE multiple_ask_job_items SET item_status='answering',updated_at=NOW()
                   WHERE id=$1::uuid AND item_status IN ('ready_to_answer','answering')
                   RETURNING TRUE""",
                item_id,
            )
        )

    async def complete_answer_item(
        self,
        *,
        item_id: str,
        ai_answer_id: str,
        answer_source: str,
        approved_revision_id: str | None,
    ) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_job_items
               SET item_status='answered',ai_answer_id=$2::uuid,answer_source=$3,
                   approved_revision_id=$4::uuid,terminal_error_code=NULL,updated_at=NOW()
               WHERE id=$1::uuid AND item_status IN ('ready_to_answer','answering')""",
            item_id,
            ai_answer_id,
            answer_source,
            approved_revision_id,
        )

    async def fail_answer_item(self, *, item_id: str, error_code: str) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_job_items
               SET item_status='failed',terminal_error_code=$2,updated_at=NOW()
               WHERE id=$1::uuid AND item_status IN ('ready_to_answer','answering')""",
            item_id,
            error_code,
        )

    async def finish_answers(self, job_id: str) -> str:
        counts = await self._conn.fetchrow(
            """SELECT count(*) FILTER (WHERE item_status='answered') AS answered,
                      count(*) FILTER (WHERE item_status='failed') AS failed,
                      count(*) FILTER (WHERE item_status IN ('ready_to_answer','answering')) AS pending
               FROM multiple_ask_job_items WHERE multiple_ask_job_id=$1::uuid""",
            job_id,
        )
        if counts is None or counts["pending"]:
            return "answering"
        workflow = "completed" if not counts["failed"] else "partially_completed"
        if not counts["answered"]:
            workflow = "failed"
        await self._conn.execute(
            """UPDATE multiple_ask_jobs SET workflow_status=$2,terminal_at=NOW(),
                   retention_expires_at=NOW()+INTERVAL '7 days',updated_at=NOW()
               WHERE id=$1::uuid AND workflow_status='answering'""",
            job_id,
            workflow,
        )
        return workflow

    async def get_owned_job_status(
        self, *, job_id: str, uid_hash: str
    ) -> dict[str, Any] | None:
        job = await self._conn.fetchrow(
            """SELECT j.id,j.workflow_status,j.input_kind,j.board_id,j.class_id,j.subject_id,
                      j.chapter_id,j.created_at,j.updated_at,j.retention_expires_at,j.terminal_error_code,
                      q.status AS queue_status,
                      q.stage AS queue_stage,q.progress AS queue_progress
               FROM multiple_ask_jobs j LEFT JOIN job_queue q ON q.id=j.queue_job_id
               WHERE j.id=$1::uuid AND j.uid_hash=$2""",
            job_id,
            uid_hash,
        )
        if job is None:
            return None
        items = await self._conn.fetch(
            """SELECT i.id,i.item_index,i.display_label,i.section_context,i.item_status,i.normalized_question,i.answer_mode,i.mcq_options,
                      i.unclear_reason,i.source_locator,i.extraction_version,i.correction_version,i.corrected_at,
                      i.answer_source,i.terminal_error_code,i.approved_revision_id,
                      a.answer_blocks,a.citation_sources,a.visual_ids,a.answer_source AS persisted_answer_source
               FROM multiple_ask_job_items i
               LEFT JOIN ai_answers a ON a.id=i.ai_answer_id
               WHERE i.multiple_ask_job_id=$1::uuid
               ORDER BY i.item_index""",
            job_id,
        )
        result = dict(job)
        result["items"] = [dict(item) for item in items]
        return result

    async def lock_owned_job(
        self, *, job_id: str, uid_hash: str
    ) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            "SELECT * FROM multiple_ask_jobs WHERE id=$1::uuid AND uid_hash=$2 FOR UPDATE",
            job_id,
            uid_hash,
        )
        return dict(row) if row else None

    async def lock_job_items(self, *, job_id: str) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(
            """SELECT * FROM multiple_ask_job_items WHERE multiple_ask_job_id=$1::uuid
               ORDER BY item_index FOR UPDATE""",
            job_id,
        )
        return [dict(row) for row in rows]

    async def apply_correction(
        self,
        *,
        job_id: str,
        item_id: str,
        request_id: str,
        question_text: str,
        normalized_question: str,
        question_hash: str,
        answer_mode: str,
        mcq_options: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        row = await self._conn.fetchrow(
            """UPDATE multiple_ask_job_items SET correction_text=$4,correction_request_id=$3::uuid,
                   correction_answer_mode=$5,correction_mcq_options=$6::jsonb,
                   correction_version=correction_version+1,corrected_at=NOW(),
                   source_text=$4,extracted_text=$4,normalized_question=$7,question_hash=$8,
                   answer_mode=$5,item_status='ready_to_answer',mcq_options=$6::jsonb,
                   unclear_reason=NULL,updated_at=NOW()
               WHERE id=$2::uuid AND multiple_ask_job_id=$1::uuid
                 AND item_status='needs_correction' AND answer_mode='not_clear'
               RETURNING *""",
            job_id,
            item_id,
            request_id,
            question_text,
            answer_mode,
            json.dumps(mcq_options),
            normalized_question,
            question_hash,
        )
        return dict(row) if row else None

    async def finish_corrections_if_resolved(self, *, job_id: str) -> None:
        unresolved = await self._conn.fetchval(
            """SELECT EXISTS(
                   SELECT 1 FROM multiple_ask_job_items
                   WHERE multiple_ask_job_id=$1::uuid AND item_status='needs_correction'
               )""",
            job_id,
        )
        if not unresolved:
            await self._conn.execute(
                """UPDATE multiple_ask_jobs SET workflow_status='ready_to_answer',updated_at=NOW()
                   WHERE id=$1::uuid AND workflow_status='needs_correction'""",
                job_id,
            )

    async def mark_terminal(
        self,
        job_id: str,
        workflow_status: str,
        retention_expires_at: datetime,
        *,
        error_code: str | None = None,
        quota_refunded: bool = False,
    ) -> None:
        await self._conn.execute(
            """UPDATE multiple_ask_jobs SET workflow_status=$2, terminal_at=NOW(),
                   retention_expires_at=$3, terminal_error_code=$4,
                   quota_status=CASE WHEN $5 THEN 'refunded' ELSE quota_status END,
                   quota_refunded_at=CASE WHEN $5 THEN NOW() ELSE quota_refunded_at END,
                   updated_at=NOW() WHERE id=$1::uuid""",
            job_id,
            workflow_status,
            retention_expires_at,
            error_code,
            quota_refunded,
        )

    async def claim_expired_unfinalized(self, *, limit: int) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(
            """WITH due AS (
                   SELECT id FROM multiple_ask_upload_sessions
                   WHERE status IN ('created','uploaded')
                     AND upload_capability_expires_at <= NOW()
                     AND (cleanup_claimed_at IS NULL OR cleanup_claimed_at < NOW()-INTERVAL '5 minutes')
                   ORDER BY upload_capability_expires_at,id FOR UPDATE SKIP LOCKED LIMIT $1
                 ) UPDATE multiple_ask_upload_sessions s SET cleanup_claimed_at=NOW(),updated_at=NOW()
                 FROM due WHERE s.id=due.id RETURNING s.*""",
            limit,
        )
        return [dict(row) for row in rows]

    async def claim_expired_raw_sources(self, *, limit: int) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(
            """WITH due AS (
                   SELECT id FROM multiple_ask_upload_sessions
                   WHERE status='finalized' AND raw_source_expires_at <= NOW()
                     AND raw_source_purged_at IS NULL
                     AND (cleanup_claimed_at IS NULL OR cleanup_claimed_at < NOW()-INTERVAL '5 minutes')
                   ORDER BY raw_source_expires_at,id FOR UPDATE SKIP LOCKED LIMIT $1
                 ) UPDATE multiple_ask_upload_sessions s SET cleanup_claimed_at=NOW(),updated_at=NOW()
                 FROM due WHERE s.id=due.id RETURNING s.*""",
            limit,
        )
        return [dict(row) for row in rows]

    async def purge_raw_source(self, session_id: str) -> None:
        await self._conn.execute(
            "DELETE FROM multiple_ask_normalized_sources WHERE session_id=$1::uuid",
            session_id,
        )
        await self._conn.execute(
            "DELETE FROM multiple_ask_text_inputs WHERE session_id=$1::uuid", session_id
        )
        await self._conn.execute(
            """UPDATE multiple_ask_upload_sessions SET status='raw_source_purged',
                   storage_bucket=NULL,storage_object_key=NULL,raw_source_purged_at=NOW(),
                   cleanup_claimed_at=NULL,updated_at=NOW() WHERE id=$1::uuid""",
            session_id,
        )

    async def release_cleanup_claim(self, session_id: str) -> None:
        await self._conn.execute(
            "UPDATE multiple_ask_upload_sessions SET cleanup_claimed_at=NULL,updated_at=NOW() WHERE id=$1::uuid",
            session_id,
        )

    async def delete_unfinalized_session(self, session_id: str) -> None:
        await self._conn.execute(
            "DELETE FROM multiple_ask_upload_sessions WHERE id=$1::uuid", session_id
        )

    async def claim_expired_jobs(self, *, limit: int) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(
            """WITH due AS (
                   SELECT id FROM multiple_ask_jobs
                   WHERE retention_expires_at <= NOW()
                     AND (cleanup_claimed_at IS NULL OR cleanup_claimed_at < NOW()-INTERVAL '5 minutes')
                   ORDER BY retention_expires_at,id FOR UPDATE SKIP LOCKED LIMIT $1
                 ) UPDATE multiple_ask_jobs j SET cleanup_claimed_at=NOW(),updated_at=NOW()
                 FROM due WHERE j.id=due.id
                 RETURNING j.*, (SELECT storage_bucket FROM multiple_ask_upload_sessions s WHERE s.id=j.upload_session_id) AS source_bucket,
                 (SELECT storage_object_key FROM multiple_ask_upload_sessions s WHERE s.id=j.upload_session_id) AS source_key""",
            limit,
        )
        return [dict(row) for row in rows]

    async def delete_expired_job_and_session(
        self, *, job_id: str, session_id: str
    ) -> None:
        await self._conn.execute(
            "DELETE FROM multiple_ask_jobs WHERE id=$1::uuid", job_id
        )
        await self._conn.execute(
            "DELETE FROM multiple_ask_upload_sessions WHERE id=$1::uuid", session_id
        )

    async def release_job_cleanup_claim(self, job_id: str) -> None:
        await self._conn.execute(
            "UPDATE multiple_ask_jobs SET cleanup_claimed_at=NULL,updated_at=NOW() WHERE id=$1::uuid",
            job_id,
        )

    async def audit_cleanup(
        self,
        *,
        run_id: str,
        session_id: str | None,
        subject_kind: str,
        action: str,
        error_code: str | None = None,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO multiple_ask_cleanup_audit(run_id,session_id,subject_kind,action,error_code)
               VALUES($1::uuid,$2::uuid,$3,$4,$5)
               ON CONFLICT(run_id,session_id,action) DO NOTHING""",
            run_id,
            session_id,
            subject_kind,
            action,
            error_code,
        )
