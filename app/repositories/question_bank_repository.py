"""Typed SQL boundary for the one trusted approved Question-Answer Bank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import asyncpg

from app.schemas.ask import AnswerMode, AnswerStyle


@dataclass(frozen=True)
class ApprovedBankAnswer:
    revision_id: str
    board_id: str
    class_id: str
    subject_id: str
    chapter_id: str | None
    answer_mode: str
    answer_style: str
    blocks: tuple[dict[str, Any], ...]
    citations: tuple[dict[str, Any], ...]
    visuals: tuple[dict[str, Any], ...]


class QuestionBankRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def _hydrate(self, row: asyncpg.Record) -> ApprovedBankAnswer:
        revision_id = str(row["id"])
        citation_rows = await self.conn.fetch(
            """SELECT c.id::text AS citation_id, c.chapter_id, c.topic_no,
                      c.topic_title, c.page_start, c.page_end
               FROM question_bank_revision_citations l
               JOIN rag_chunks c ON c.id=l.chunk_id
               WHERE l.revision_id=$1::uuid
               ORDER BY l.display_order, c.id""",
            revision_id,
        )
        visual_rows = await self.conn.fetch(
            """SELECT v.visual_id, v.title, v.description, v.display_policy,
                      l.display_order
               FROM question_bank_revision_visuals l
               JOIN rag_visuals v ON v.id=l.visual_id
               WHERE l.revision_id=$1::uuid
                 AND v.review_status='approved'
                 AND v.display_policy IN ('always','llm_decide')
               ORDER BY l.display_order, v.visual_id""",
            revision_id,
        )
        blocks = row["answer_blocks"]
        if isinstance(blocks, str):
            blocks = json.loads(blocks)
        return ApprovedBankAnswer(
            revision_id=revision_id,
            board_id=row["board_id"],
            class_id=row["class_id"],
            subject_id=row["subject_id"],
            chapter_id=row["chapter_id"],
            answer_mode=row["answer_mode"],
            answer_style=row["answer_style"],
            blocks=tuple(blocks),
            citations=tuple(dict(item) for item in citation_rows),
            visuals=tuple(dict(item) for item in visual_rows),
        )

    @staticmethod
    def _unambiguous(rows: list[asyncpg.Record], chapter_id: str | None):
        if not rows:
            return None
        if chapter_id is not None:
            return rows[0]
        chapters = {row["chapter_id"] for row in rows}
        return rows[0] if len(chapters) == 1 else None

    async def find_exact(
        self,
        *,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: AnswerMode,
        normalized_question: str,
    ) -> ApprovedBankAnswer | None:
        rows = await self.conn.fetch(
            """SELECT *
               FROM question_bank_revisions
               WHERE board_id=$1 AND class_id=$2 AND subject_id=$3
                 AND ($4::text IS NULL OR chapter_id=$4)
                 AND answer_mode=$5 AND normalized_question=$6
                 AND review_status='approved' AND superseded_at IS NULL
               ORDER BY chapter_id NULLS LAST, approved_at DESC, id
               LIMIT 3""",
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode.value,
            normalized_question,
        )
        selected = self._unambiguous(list(rows), chapter_id)
        return await self._hydrate(selected) if selected is not None else None

    async def find_exact_variation(
        self,
        *,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: AnswerMode,
        normalized_question: str,
    ) -> ApprovedBankAnswer | None:
        rows = await self.conn.fetch(
            """SELECT r.*
               FROM question_bank_variations v
               JOIN question_bank_revisions r ON r.id=v.revision_id
               WHERE r.board_id=$1 AND r.class_id=$2 AND r.subject_id=$3
                 AND ($4::text IS NULL OR r.chapter_id=$4)
                 AND r.answer_mode=$5 AND v.normalized_variation=$6
                 AND v.active
                 AND r.review_status='approved' AND r.superseded_at IS NULL
               ORDER BY r.chapter_id NULLS LAST, r.approved_at DESC, r.id
               LIMIT 3""",
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode.value,
            normalized_question,
        )
        selected = self._unambiguous(list(rows), chapter_id)
        return await self._hydrate(selected) if selected is not None else None

    async def find_semantic(
        self,
        *,
        query_embedding: list[float],
        evaluated_threshold: float | None,
        enabled: bool,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: AnswerMode,
    ) -> ApprovedBankAnswer | None:
        """No configured evaluated threshold means semantic reuse is disabled."""
        if not enabled or evaluated_threshold is None:
            return None
        rows = await self.conn.fetch(
            """WITH candidates AS (
                 SELECT r.*, r.embedding <=> $6::text::vector AS distance
                 FROM question_bank_revisions r
                 WHERE r.board_id=$1 AND r.class_id=$2 AND r.subject_id=$3
                   AND ($4::text IS NULL OR r.chapter_id=$4)
                   AND r.answer_mode=$5
                   AND r.review_status='approved' AND r.superseded_at IS NULL
                   AND r.embedding_status='embedded' AND r.embedding IS NOT NULL
                 UNION ALL
                 SELECT r.*, v.embedding <=> $6::text::vector AS distance
                 FROM question_bank_variations v
                 JOIN question_bank_revisions r ON r.id=v.revision_id
                 WHERE r.board_id=$1 AND r.class_id=$2 AND r.subject_id=$3
                   AND ($4::text IS NULL OR r.chapter_id=$4)
                   AND r.answer_mode=$5
                   AND r.review_status='approved' AND r.superseded_at IS NULL
                   AND v.active AND v.embedding_status='embedded'
                   AND v.embedding IS NOT NULL
               )
               SELECT * FROM candidates
               WHERE distance <= $7
               ORDER BY distance, id
               LIMIT 3""",
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode.value,
            json.dumps(query_embedding),
            evaluated_threshold,
        )
        selected = self._unambiguous(list(rows), chapter_id)
        return await self._hydrate(selected) if selected is not None else None

    async def create_approved_revision(
        self,
        *,
        actor_id: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: AnswerMode,
        answer_style: AnswerStyle,
        difficulty: str,
        marks: float,
        question_text: str,
        normalized_question: str,
        question_hash: str,
        blocks: list[dict[str, Any]],
        source: str,
        question_id: str | None = None,
        citation_chunk_ids: Iterable[str] = (),
        visual_row_ids: Iterable[str] = (),
        mcq_options: Iterable[dict[str, Any]] = (),
    ) -> str:
        if question_id is None:
            question_id = str(
                await self.conn.fetchval(
                    """INSERT INTO question_bank_questions(source,created_by)
                       VALUES($1,$2) RETURNING id""",
                    source,
                    actor_id,
                )
            )
        else:
            await self.conn.execute(
                """UPDATE question_bank_revisions SET superseded_at=NOW()
                   WHERE question_id=$1::uuid AND review_status='approved'
                     AND superseded_at IS NULL""",
                question_id,
            )
        version = await self.conn.fetchval(
            """SELECT COALESCE(MAX(version_no),0)+1
               FROM question_bank_revisions WHERE question_id=$1::uuid""",
            question_id,
        )
        revision_id = str(
            await self.conn.fetchval(
                """INSERT INTO question_bank_revisions(
                     question_id,version_no,board_id,class_id,subject_id,chapter_id,
                     answer_mode,answer_style,difficulty,marks,question_text,
                     normalized_question,question_hash,answer_blocks,review_status,
                     source,approved_by,approved_at,created_by
                   ) VALUES(
                     $1::uuid,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb,
                     'approved',$15,$16,NOW(),$16
                   ) RETURNING id""",
                question_id,
                version,
                board_id,
                class_id,
                subject_id,
                chapter_id,
                answer_mode.value,
                answer_style.value,
                difficulty,
                marks,
                question_text,
                normalized_question,
                question_hash,
                json.dumps(blocks),
                source,
                actor_id,
            )
        )
        for order, chunk_id in enumerate(citation_chunk_ids):
            await self.conn.execute(
                """INSERT INTO question_bank_revision_citations(
                     revision_id,chunk_id,display_order
                   ) VALUES($1::uuid,$2::uuid,$3)""",
                revision_id,
                chunk_id,
                order,
            )
        for order, visual_id in enumerate(visual_row_ids):
            await self.conn.execute(
                """INSERT INTO question_bank_revision_visuals(
                     revision_id,visual_id,display_order
                   ) VALUES($1::uuid,$2::uuid,$3)""",
                revision_id,
                visual_id,
                order,
            )
        for order, option in enumerate(mcq_options):
            await self.conn.execute(
                """INSERT INTO question_bank_mcq_options(
                     revision_id,option_key,option_text,display_order,is_correct
                   ) VALUES($1::uuid,$2,$3,$4,$5)""",
                revision_id,
                option["key"],
                option["text"],
                order,
                bool(option.get("is_correct")),
            )
        return revision_id

    async def add_variation(
        self,
        *,
        revision_id: str,
        variation_text: str,
        normalized_variation: str,
        variation_hash: str,
        actor_id: str,
    ) -> str:
        revision_is_approved = await self.conn.fetchval(
            """SELECT EXISTS(
                 SELECT 1 FROM question_bank_revisions
                 WHERE id=$1::uuid AND review_status='approved'
                   AND superseded_at IS NULL
               )""",
            revision_id,
        )
        if not revision_is_approved:
            raise ValueError("APPROVED_REVISION_NOT_ACTIVE")
        return str(
            await self.conn.fetchval(
                """INSERT INTO question_bank_variations(
                     revision_id,variation_text,normalized_variation,
                     variation_hash,created_by
                   ) VALUES($1::uuid,$2,$3,$4,$5)
                   ON CONFLICT(revision_id,variation_hash) DO UPDATE
                   SET active=TRUE
                   RETURNING id""",
                revision_id,
                variation_text,
                normalized_variation,
                variation_hash,
                actor_id,
            )
        )

    async def get_revision(self, revision_id: str) -> ApprovedBankAnswer | None:
        row = await self.conn.fetchrow(
            """SELECT * FROM question_bank_revisions
               WHERE id=$1::uuid AND review_status='approved'""",
            revision_id,
        )
        return await self._hydrate(row) if row else None

    async def list_approved(
        self,
        *,
        board_id: str | None = None,
        class_id: str | None = None,
        subject_id: str | None = None,
        chapter_id: str | None = None,
        answer_mode: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = await self.conn.fetch(
            """SELECT r.id::text AS revision_id,r.question_id::text,
                      r.version_no,r.question_text,r.board_id,r.class_id,
                      r.subject_id,r.chapter_id,r.answer_mode,r.answer_style,
                      r.difficulty,r.marks,r.source,r.approved_by,r.approved_at,
                      r.embedding_status,r.superseded_at,
                      (SELECT COUNT(*) FROM question_bank_variations v
                       WHERE v.revision_id=r.id AND v.active) AS variation_count
               FROM question_bank_revisions r
               WHERE r.review_status='approved' AND r.superseded_at IS NULL
                 AND ($1::text IS NULL OR r.board_id=$1)
                 AND ($2::text IS NULL OR r.class_id=$2)
                 AND ($3::text IS NULL OR r.subject_id=$3)
                 AND ($4::text IS NULL OR r.chapter_id=$4)
                 AND ($5::text IS NULL OR r.answer_mode=$5)
                 AND ($6::text IS NULL OR r.source=$6)
               ORDER BY r.approved_at DESC,r.id
               LIMIT $7""",
            board_id,
            class_id,
            subject_id,
            chapter_id,
            answer_mode,
            source,
            limit,
        )
        return [dict(row) for row in rows]

    async def revision_history(
        self, *, revision_id: str | None = None, question_id: str | None = None
    ) -> dict[str, Any] | None:
        if question_id is None:
            question_id = await self.conn.fetchval(
                """SELECT question_id::text FROM question_bank_revisions
                   WHERE id=$1::uuid""",
                revision_id,
            )
        if question_id is None:
            return None
        revisions = await self.conn.fetch(
            """SELECT id::text AS revision_id,question_id::text,version_no,
                      board_id,class_id,subject_id,chapter_id,answer_mode,
                      answer_style,difficulty,marks,question_text,answer_blocks,
                      review_status,source,approved_by,approved_at,rejected_by,
                      rejected_at,rejection_reason,superseded_at,embedding_status,
                      embedding_model,embedding_revision,
                      embedding_config_fingerprint,created_by,created_at
               FROM question_bank_revisions
               WHERE question_id=$1::uuid
               ORDER BY version_no DESC,id""",
            question_id,
        )
        if not revisions:
            return None
        revision_ids = [row["revision_id"] for row in revisions]
        variations = await self.conn.fetch(
            """SELECT id::text AS variation_id,revision_id::text,variation_text,
                      active,embedding_status,embedding_model,embedding_revision,
                      embedding_config_fingerprint,created_by,created_at
               FROM question_bank_variations
               WHERE revision_id=ANY($1::uuid[])
               ORDER BY created_at,id""",
            revision_ids,
        )
        options = await self.conn.fetch(
            """SELECT revision_id::text,option_key,option_text,display_order,
                      is_correct
               FROM question_bank_mcq_options
               WHERE revision_id=ANY($1::uuid[])
               ORDER BY revision_id,display_order""",
            revision_ids,
        )
        return {
            "question_id": question_id,
            "revisions": [dict(row) for row in revisions],
            "variations": [dict(row) for row in variations],
            "mcq_options": [dict(row) for row in options],
        }

    async def archive_revision(self, *, revision_id: str) -> dict[str, Any]:
        row = await self.conn.fetchrow(
            """UPDATE question_bank_revisions
               SET review_status='archived',superseded_at=COALESCE(superseded_at,NOW())
               WHERE id=$1::uuid AND review_status='approved'
                 AND superseded_at IS NULL
               RETURNING question_id::text,version_no,review_status,superseded_at""",
            revision_id,
        )
        if row is None:
            raise ValueError("APPROVED_REVISION_NOT_ACTIVE")
        return dict(row)

    async def set_variation_active(
        self, *, variation_id: str, active: bool
    ) -> dict[str, Any]:
        row = await self.conn.fetchrow(
            """UPDATE question_bank_variations v SET active=$2
               FROM question_bank_revisions r
               WHERE v.id=$1::uuid AND r.id=v.revision_id
                 AND r.review_status='approved' AND r.superseded_at IS NULL
               RETURNING v.id::text AS variation_id,
                         v.revision_id::text,v.active,v.embedding_status""",
            variation_id,
            active,
        )
        if row is None:
            raise ValueError("APPROVED_VARIATION_NOT_FOUND")
        return dict(row)

    async def reset_embedding(
        self, *, revision_id: str, variation_id: str | None = None
    ) -> None:
        if variation_id is None:
            result = await self.conn.execute(
                """UPDATE question_bank_revisions
                   SET embedding=NULL,embedding_model=NULL,embedding_revision=NULL,
                       embedding_config_fingerprint=NULL,embedding_status='pending'
                   WHERE id=$1::uuid AND review_status='approved'
                     AND superseded_at IS NULL""",
                revision_id,
            )
        else:
            result = await self.conn.execute(
                """UPDATE question_bank_variations v
                   SET embedding=NULL,embedding_model=NULL,embedding_revision=NULL,
                       embedding_config_fingerprint=NULL,embedding_status='pending'
                   FROM question_bank_revisions r
                   WHERE v.id=$2::uuid AND v.revision_id=$1::uuid
                     AND r.id=v.revision_id AND r.review_status='approved'
                     AND r.superseded_at IS NULL AND v.active""",
                revision_id,
                variation_id,
            )
        if result != "UPDATE 1":
            raise ValueError("EMBEDDING_TARGET_NOT_ACTIVE")

    async def set_visual_links(
        self, *, revision_id: str, visual_row_ids: list[str]
    ) -> None:
        exists = await self.conn.fetchval(
            """SELECT EXISTS(
                 SELECT 1 FROM question_bank_revisions
                 WHERE id=$1::uuid AND review_status='approved'
               )""",
            revision_id,
        )
        if not exists:
            raise ValueError("APPROVED_REVISION_NOT_FOUND")
        if visual_row_ids:
            eligible = await self.conn.fetchval(
                """SELECT COUNT(*) FROM rag_visuals
                   WHERE id=ANY($1::uuid[]) AND review_status='approved'
                     AND display_policy IN ('always','llm_decide')""",
                visual_row_ids,
            )
            if eligible != len(set(visual_row_ids)):
                raise ValueError("VISUAL_LINK_NOT_REVIEWED")
        await self.conn.execute(
            "DELETE FROM question_bank_revision_visuals WHERE revision_id=$1::uuid",
            revision_id,
        )
        for order, visual_id in enumerate(visual_row_ids):
            await self.conn.execute(
                """INSERT INTO question_bank_revision_visuals(
                     revision_id,visual_id,display_order
                   ) VALUES($1::uuid,$2::uuid,$3)""",
                revision_id,
                visual_id,
                order,
            )
