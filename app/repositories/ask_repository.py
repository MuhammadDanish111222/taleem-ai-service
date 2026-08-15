"""Idempotent Ask request/candidate persistence on the existing AI tables."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import asyncpg


class AskRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def by_client_request_id(
        self, request_id: str, uid_hash: str
    ) -> dict[str, Any] | None:
        row = await self.conn.fetchrow(
            """SELECT r.*, a.id AS answer_id, a.answer_blocks, a.answer_source,
                      a.answer_mode AS stored_answer_mode, a.answer_style AS stored_answer_style,
                      a.citation_sources, a.visual_ids, a.review_status,
                      a.approved_revision_id
               FROM ai_requests r
               LEFT JOIN ai_answers a ON a.request_id=r.id
               WHERE r.client_request_id=$1::uuid AND r.uid_hash=$2""",
            request_id,
            uid_hash,
        )
        return dict(row) if row else None

    async def answer_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        row = await self.conn.fetchrow(
            """SELECT * FROM ai_answers WHERE request_id = $1::uuid""",
            request_id,
        )
        return dict(row) if row else None

    async def create_pending(
        self,
        *,
        client_request_id: str,
        uid_hash: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: str,
        answer_style: str,
        raw_question: str,
        normalized_question: str,
        question_hash: str,
        usage_business_date: date,
    ) -> dict[str, Any]:
        row = await self.conn.fetchrow(
            """INSERT INTO ai_requests(
                 client_request_id,uid_hash,board_id,class_id,subject_id,chapter_id,
                 language,answer_mode,answer_style,raw_question,normalized_question,
                 question_hash,prompt_version,status,source_feature,
                 normalization_version,usage_business_date,retention_expires_at
               ) VALUES(
                 $1::uuid,$2,$3,$4,$5,$6,'en',$7,$8,$9,$10,$11,'unresolved',
                 'pending','single_question',1,$12,NOW()+INTERVAL '90 days'
               )
               ON CONFLICT DO NOTHING
               RETURNING *""",
            client_request_id,
            uid_hash,
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode,
            answer_style,
            raw_question,
            normalized_question,
            question_hash,
            usage_business_date,
        )
        newly_created = row is not None
        if row is None:
            row = await self.conn.fetchrow(
                """SELECT * FROM ai_requests
                   WHERE client_request_id=$1::uuid AND uid_hash=$2""",
                client_request_id,
                uid_hash,
            )
        result = dict(row)
        result["_newly_created"] = newly_created
        return result

    async def create_pending_multiple_ask(
        self,
        *,
        client_request_id: str,
        uid_hash: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: str,
        raw_question: str,
        normalized_question: str,
        question_hash: str,
    ) -> dict[str, Any]:
        """Create the one reviewable candidate request for a durable paper item.

        This intentionally has no usage reservation: Run 1 already committed the
        one batch quota before extraction.  The deterministic client request id
        supplied by the caller makes restarts return this exact row rather than
        creating another candidate.
        """
        row = await self.conn.fetchrow(
            """INSERT INTO ai_requests(
                 client_request_id,uid_hash,board_id,class_id,subject_id,chapter_id,
                 language,answer_mode,answer_style,raw_question,normalized_question,
                 question_hash,prompt_version,status,source_feature,
                 normalization_version,retention_expires_at
               ) VALUES(
                 $1::uuid,$2,$3,$4,$5,$6,'en',$7,'exam_style',$8,$9,$10,
                 'unresolved','pending','multiple_ask',1,NOW()+INTERVAL '7 days'
               ) ON CONFLICT DO NOTHING RETURNING *""",
            client_request_id,
            uid_hash,
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode,
            raw_question,
            normalized_question,
            question_hash,
        )
        newly_created = row is not None
        if row is None:
            row = await self.conn.fetchrow(
                """SELECT * FROM ai_requests
                   WHERE client_request_id=$1::uuid AND uid_hash=$2""",
                client_request_id,
                uid_hash,
            )
        if row is None:
            raise RuntimeError("MULTIPLE_ASK_CANDIDATE_CREATE_FAILED")
        result = dict(row)
        result["_newly_created"] = newly_created
        return result

    async def complete(
        self,
        *,
        ai_request_id: str,
        answer_source: str,
        blocks: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        visual_ids: list[str],
        prompt_version: str,
        corpus_version_id: str | None,
        provider: str | None,
        model: str | None,
        tokens_used: int = 0,
        latency_ms: int = 0,
        approved_revision_id: str | None = None,
    ) -> dict[str, Any]:
        answer_text = "\n\n".join(
            item.get("text", "") for item in blocks if item.get("type") == "paragraph"
        )
        row = await self.conn.fetchrow(
            """INSERT INTO ai_answers(
                 request_id,answer_text,answer_blocks,answer_source,answer_mode,
                 answer_style,citation_sources,citation_ids,visual_ids,provider,model,
                 review_status,tokens_used,latency_ms,retention_expires_at,
                 approved_revision_id
               )
               SELECT $1::uuid,$2,$3::jsonb,$4,r.answer_mode,r.answer_style,
                      $5::jsonb,$6::jsonb,$7::jsonb,$8,$9,
                      CASE WHEN $4='approved_bank' THEN 'approved' ELSE 'pending' END,
                      $10,$11,
                      CASE WHEN $4='approved_bank' THEN NULL ELSE NOW()+INTERVAL '90 days' END,
                      $12::uuid
               FROM ai_requests r WHERE r.id=$1::uuid
               ON CONFLICT(request_id) DO UPDATE SET request_id=EXCLUDED.request_id
               RETURNING *""",
            ai_request_id,
            answer_text,
            json.dumps(blocks),
            answer_source,
            json.dumps(citations),
            json.dumps([item["citation_id"] for item in citations]),
            json.dumps(visual_ids),
            provider,
            model,
            tokens_used,
            latency_ms,
            approved_revision_id,
        )
        await self.conn.execute(
            """UPDATE ai_requests
               SET status='completed',answer_source=$2,prompt_version=$3,
                   corpus_version_id=$4::uuid,updated_at=NOW()
               WHERE id=$1::uuid""",
            ai_request_id,
            answer_source,
            prompt_version,
            corpus_version_id,
        )
        return dict(row)

    async def no_answer(
        self, ai_request_id: str, *, error_code: str, prompt_version: str | None
    ) -> None:
        await self.conn.execute(
            """UPDATE ai_requests SET status='no_answer',terminal_error_code=$2,
                      prompt_version=COALESCE($3,prompt_version),updated_at=NOW()
               WHERE id=$1::uuid""",
            ai_request_id,
            error_code,
            prompt_version,
        )

    async def fail(self, ai_request_id: str, *, error_code: str) -> None:
        await self.conn.execute(
            """UPDATE ai_requests SET status='failed',terminal_error_code=$2,
                      updated_at=NOW()
               WHERE id=$1::uuid""",
            ai_request_id,
            error_code,
        )

    async def list_pending(
        self,
        *,
        board_id: str | None = None,
        class_id: str | None = None,
        subject_id: str | None = None,
        chapter_id: str | None = None,
        answer_mode: str | None = None,
        answer_source: str | None = None,
        source_feature: str | None = None,
        provider: str | None = None,
        age_days: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = await self.conn.fetch(
            """SELECT a.id,a.request_id,r.client_request_id,r.board_id,r.class_id,
                      r.subject_id,r.chapter_id,r.answer_mode,r.answer_style,
                      r.answer_source,r.source_feature,r.created_at,a.provider,a.model,
                      r.prompt_version,r.corpus_version_id
               FROM ai_answers a JOIN ai_requests r ON r.id=a.request_id
               WHERE a.review_status='pending'
                 AND ($1::text IS NULL OR r.board_id=$1)
                 AND ($2::text IS NULL OR r.class_id=$2)
                 AND ($3::text IS NULL OR r.subject_id=$3)
                 AND ($4::text IS NULL OR r.chapter_id=$4)
                 AND ($5::text IS NULL OR r.answer_mode=$5)
                 AND ($6::text IS NULL OR r.answer_source=$6)
                 AND ($7::text IS NULL OR r.source_feature=$7)
                 AND ($8::text IS NULL OR a.provider=$8)
                 AND ($9::integer IS NULL OR
                      r.created_at <= NOW()-($9 * INTERVAL '1 day'))
               ORDER BY r.created_at,a.id LIMIT $10""",
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode,
            answer_source,
            source_feature,
            provider,
            age_days,
            limit,
        )
        return [dict(row) for row in rows]

    async def inspect_candidate(self, answer_id: str) -> dict[str, Any] | None:
        row = await self.conn.fetchrow(
            """SELECT a.id,a.request_id,a.answer_blocks,a.answer_source,
                      a.answer_mode,a.answer_style,a.citation_sources,a.visual_ids,
                      a.provider,a.model,a.review_status,a.approved_revision_id,
                      r.client_request_id,r.board_id,r.class_id,r.subject_id,
                      r.chapter_id,r.raw_question,r.normalized_question,
                      r.question_hash,r.source_feature,r.prompt_version,
                      r.corpus_version_id,r.created_at
               FROM ai_answers a JOIN ai_requests r ON r.id=a.request_id
               WHERE a.id=$1::uuid""",
            answer_id,
        )
        return dict(row) if row else None

    async def visual_stream_reference(
        self, *, client_request_id: str, uid_hash: str, visual_id: str
    ) -> dict[str, str] | None:
        """Resolve one reviewed visual selected by this identity's completed Ask."""

        rows = await self.conn.fetch(
            """WITH selected_answer AS (
                 SELECT r.corpus_version_id,a.approved_revision_id
                 FROM ai_requests r
                 JOIN ai_answers a ON a.request_id=r.id
                 WHERE r.client_request_id=$1::uuid AND r.uid_hash=$2
                   AND r.status='completed'
                   AND a.visual_ids @> jsonb_build_array($3::text)
               ),
               eligible AS (
                 SELECT v.storage_provider,v.storage_key
                 FROM selected_answer s
                 JOIN rag_chunks c ON c.corpus_version_id=s.corpus_version_id
                 JOIN rag_visuals v ON v.chunk_id=c.id AND v.visual_id=$3
                 WHERE s.approved_revision_id IS NULL
                   AND v.review_status='approved'
                 UNION ALL
                 SELECT v.storage_provider,v.storage_key
                 FROM selected_answer s
                 JOIN question_bank_revision_visuals l
                   ON l.revision_id=s.approved_revision_id
                 JOIN rag_visuals v ON v.id=l.visual_id AND v.visual_id=$3
                 WHERE v.review_status='approved'
               )
               SELECT storage_provider,storage_key FROM eligible LIMIT 2""",
            client_request_id,
            uid_hash,
            visual_id,
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        if (
            row["storage_provider"] != "google_drive"
            or not row["storage_key"]
            or not row["storage_key"].strip()
        ):
            return None
        return {
            "storage_provider": row["storage_provider"],
            "storage_key": row["storage_key"],
        }

    async def multiple_ask_visual_stream_reference(
        self, *, job_id: str, uid_hash: str, visual_id: str
    ) -> dict[str, str] | None:
        """Resolve a reviewed visual only when it belongs to this student's job."""

        rows = await self.conn.fetch(
            """WITH selected_answer AS (
                 SELECT r.corpus_version_id,a.approved_revision_id
                 FROM multiple_ask_jobs j
                 JOIN multiple_ask_job_items i ON i.multiple_ask_job_id=j.id
                 JOIN ai_answers a ON a.id=i.ai_answer_id
                 JOIN ai_requests r ON r.id=a.request_id
                 WHERE j.id=$1::uuid AND j.uid_hash=$2
                   AND i.item_status='answered' AND r.status='completed'
                   AND a.visual_ids @> jsonb_build_array($3::text)
               ),
               eligible AS (
                 SELECT v.storage_provider,v.storage_key
                 FROM selected_answer s
                 JOIN rag_chunks c ON c.corpus_version_id=s.corpus_version_id
                 JOIN rag_visuals v ON v.chunk_id=c.id AND v.visual_id=$3
                 WHERE s.approved_revision_id IS NULL
                   AND v.review_status='approved'
                   AND v.display_policy IN ('always','llm_decide')
                 UNION ALL
                 SELECT v.storage_provider,v.storage_key
                 FROM selected_answer s
                 JOIN question_bank_revision_visuals l
                   ON l.revision_id=s.approved_revision_id
                 JOIN rag_visuals v ON v.id=l.visual_id AND v.visual_id=$3
                 WHERE v.review_status='approved'
                   AND v.display_policy IN ('always','llm_decide')
               )
               SELECT storage_provider,storage_key FROM eligible LIMIT 2""",
            job_id,
            uid_hash,
            visual_id,
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        if row["storage_provider"] != "google_drive" or not row["storage_key"]:
            return None
        return {
            "storage_provider": row["storage_provider"],
            "storage_key": row["storage_key"],
        }

    async def visual_metadata_for_completed_request(
        self, *, ai_request_id: str, visual_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not visual_ids:
            return []
        rows = await self.conn.fetch(
            """WITH selected_answer AS (
                 SELECT r.corpus_version_id,a.approved_revision_id
                 FROM ai_requests r
                 JOIN ai_answers a ON a.request_id=r.id
                 WHERE r.id=$1::uuid AND r.status='completed'
               ),
               eligible AS (
                 SELECT v.id,v.visual_id,v.title,v.description,v.display_policy
                 FROM selected_answer s
                 JOIN rag_chunks c ON c.corpus_version_id=s.corpus_version_id
                 JOIN rag_visuals v ON v.chunk_id=c.id
                 WHERE s.approved_revision_id IS NULL
                   AND v.visual_id=ANY($2::text[])
                   AND v.review_status='approved'
                 UNION ALL
                 SELECT v.id,v.visual_id,v.title,v.description,v.display_policy
                 FROM selected_answer s
                 JOIN question_bank_revision_visuals l
                   ON l.revision_id=s.approved_revision_id
                 JOIN rag_visuals v ON v.id=l.visual_id
                 WHERE v.visual_id=ANY($2::text[])
                   AND v.review_status='approved'
               )
               SELECT visual_id,title,description,display_policy
               FROM eligible ORDER BY visual_id,id""",
            ai_request_id,
            visual_ids,
        )
        grouped: dict[str, list[asyncpg.Record]] = {}
        for row in rows:
            grouped.setdefault(row["visual_id"], []).append(row)
        if any(len(grouped.get(visual_id, ())) != 1 for visual_id in visual_ids):
            return []
        return [
            {
                "visual_id": visual_id,
                "title": grouped[visual_id][0]["title"],
                "description": grouped[visual_id][0]["description"],
                "display_policy": grouped[visual_id][0]["display_policy"],
                "display_order": order,
            }
            for order, visual_id in enumerate(visual_ids)
        ]

    async def approve_candidate(
        self, *, answer_id: str, revision_id: str, actor_id: str
    ) -> None:
        result = await self.conn.execute(
            """UPDATE ai_answers
               SET review_status='approved',approved_revision_id=$2::uuid,
                   reviewed_by=$3,reviewed_at=NOW(),retention_expires_at=NULL
               WHERE id=$1::uuid AND review_status='pending'""",
            answer_id,
            revision_id,
            actor_id,
        )
        if result != "UPDATE 1":
            raise ValueError("CANDIDATE_NOT_PENDING")

    async def reject_candidate(
        self, *, answer_id: str, reason: str, actor_id: str
    ) -> None:
        result = await self.conn.execute(
            """UPDATE ai_answers
               SET review_status='rejected',rejection_reason=$2,reviewed_by=$3,
                   reviewed_at=NOW()
               WHERE id=$1::uuid AND review_status='pending'""",
            answer_id,
            reason,
            actor_id,
        )
        if result != "UPDATE 1":
            raise ValueError("CANDIDATE_NOT_PENDING")
