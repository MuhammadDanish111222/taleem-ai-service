"""RAG Schema Repository using Asyncpg for vector and lexical queries."""

import hashlib
import json
from typing import Any, Dict, List, Optional

import asyncpg

from app.providers.embeddings.voyage import (
    EMBEDDING_DIMENSIONS,
    MODEL_NAME,
    MODEL_REVISION,
)
from app.services.ingestion.normalization import normalize_expected_question


class RagRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    def _validate_vector(self, vector: List[float]) -> None:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match expected {EMBEDDING_DIMENSIONS}."
            )

    async def get_or_create_corpus(
        self, board_id: str, class_id: str, subject_id: str
    ) -> Dict[str, Any]:
        """Fetches or creates a RAG corpus record for a scope."""
        query = """
        INSERT INTO rag_corpora (board_id, class_id, subject_id)
        VALUES ($1, $2, $3)
        ON CONFLICT (board_id, class_id, subject_id) DO UPDATE SET created_at = rag_corpora.created_at
        RETURNING *;
        """
        row = await self.conn.fetchrow(query, board_id, class_id, subject_id)
        return dict(row)

    async def get_or_create_building_corpus_version(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        embedding_model: str = MODEL_NAME,
        embedding_revision: str = MODEL_REVISION,
        embedding_dim: int = EMBEDDING_DIMENSIONS,
        embedding_config_fingerprint: str = "",
        normalize_embeddings: bool = True,
        query_instruction: Optional[str] = None,
        *,
        require_existing_draft_after_activation: bool = False,
    ) -> Dict[str, Any]:
        """Fetches or creates a single 'building' corpus version for a board/class/subject scope.

        Locks parent rag_corpora row via ON CONFLICT DO UPDATE + FOR UPDATE to prevent
        concurrent check-then-act version creation races.
        """
        # 1. Atomic upsert to acquire/guarantee parent corpora row
        upsert_query = """
        INSERT INTO rag_corpora (board_id, class_id, subject_id)
        VALUES ($1, $2, $3)
        ON CONFLICT (board_id, class_id, subject_id) DO UPDATE SET created_at = rag_corpora.created_at
        RETURNING id;
        """
        row = await self.conn.fetchrow(upsert_query, board_id, class_id, subject_id)
        corpus_id = row["id"]

        # 2. Lock parent corpora row for UPDATE to serialize building version checks/creations
        await self.conn.execute(
            "SELECT id FROM rag_corpora WHERE id = $1::uuid FOR UPDATE;", corpus_id
        )

        # 3. Check for existing building version
        existing = await self.conn.fetchrow(
            """
            SELECT * FROM rag_corpus_versions
            WHERE corpus_id = $1::uuid AND status = 'building'
            ORDER BY version_no DESC
            LIMIT 1;
            """,
            corpus_id,
        )
        if existing:
            existing_dict = dict(existing)
            requested = (
                embedding_model,
                embedding_revision,
                embedding_dim,
                embedding_config_fingerprint,
                normalize_embeddings,
                query_instruction,
            )
            stored = (
                existing_dict["embedding_model"],
                existing_dict["embedding_revision"],
                existing_dict["embedding_dim"],
                existing_dict.get("embedding_config_fingerprint", ""),
                existing_dict["normalize_embeddings"],
                existing_dict["query_instruction"],
            )
            if requested != stored:
                raise ValueError(
                    "EMBEDDING_CONFIGURATION_CHANGE_REQUIRES_NEW_CORPUS_VERSION"
                )
            return existing_dict

        # 4. A subject may be imported chapter-by-chapter before its first
        # activation. Reopen the latest QA-ready snapshot when its embedding
        # configuration is unchanged so the next chapter extends that same
        # subject version instead of creating a chapter-only replacement.
        qa_ready = await self.conn.fetchrow(
            """
            SELECT * FROM rag_corpus_versions
            WHERE corpus_id = $1::uuid AND status = 'qa_ready'
            ORDER BY version_no DESC
            LIMIT 1
            FOR UPDATE;
            """,
            corpus_id,
        )
        if qa_ready:
            qa_ready_dict = dict(qa_ready)
            requested = (
                embedding_model,
                embedding_revision,
                embedding_dim,
                embedding_config_fingerprint,
                normalize_embeddings,
                query_instruction,
            )
            stored = (
                qa_ready_dict["embedding_model"],
                qa_ready_dict["embedding_revision"],
                qa_ready_dict["embedding_dim"],
                qa_ready_dict.get("embedding_config_fingerprint", ""),
                qa_ready_dict["normalize_embeddings"],
                qa_ready_dict["query_instruction"],
            )
            if requested == stored:
                reopened = await self.conn.fetchrow(
                    """
                    UPDATE rag_corpus_versions
                    SET status = 'building'
                    WHERE id = $1::uuid
                    RETURNING *;
                    """,
                    qa_ready_dict["id"],
                )
                await self.conn.execute(
                    """
                    UPDATE rag_corpus_qa_approvals
                    SET invalidated_at = NOW(), invalidated_reason = 'chapter_imported'
                    WHERE corpus_version_id = $1::uuid
                      AND invalidated_at IS NULL;
                    """,
                    qa_ready_dict["id"],
                )
                return dict(reopened)

        if require_existing_draft_after_activation and await self.conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM rag_corpus_versions
                WHERE corpus_id = $1::uuid AND status = 'active'
            );
            """,
            corpus_id,
        ):
            raise ValueError("ACTIVE_CORPUS_REQUIRES_EDITABLE_DRAFT")

        # 5. Create new building version (version_no = max + 1)
        max_v_row = await self.conn.fetchrow(
            "SELECT MAX(version_no) as max_v FROM rag_corpus_versions WHERE corpus_id = $1::uuid;",
            corpus_id,
        )
        max_v = (
            max_v_row["max_v"] if max_v_row and max_v_row["max_v"] is not None else 0
        )
        new_version_no = max_v + 1

        insert_version_query = """
        INSERT INTO rag_corpus_versions (
            corpus_id, version_no, embedding_model, embedding_revision, embedding_dim,
            normalize_embeddings, query_instruction, embedding_config_fingerprint, status
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, 'building')
        RETURNING *;
        """
        new_version = await self.conn.fetchrow(
            insert_version_query,
            corpus_id,
            new_version_no,
            embedding_model,
            embedding_revision,
            embedding_dim,
            normalize_embeddings,
            query_instruction,
            embedding_config_fingerprint,
        )
        return dict(new_version)

    async def create_corpus_version(
        self,
        corpus_id: str,
        version_no: int,
        embedding_model: str,
        embedding_revision: str,
        embedding_dim: int = 512,
        chunking_config: Optional[Dict[str, Any]] = None,
        embedding_config_fingerprint: str = "",
        normalize_embeddings: bool = True,
        query_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a new corpus version in building status."""
        query = """
        INSERT INTO rag_corpus_versions (
            corpus_id, version_no, embedding_model, embedding_revision, embedding_dim,
            normalize_embeddings, query_instruction, embedding_config_fingerprint,
            chunking_config, status
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, 'building')
        RETURNING *;
        """
        config_json = json.dumps(chunking_config or {})
        row = await self.conn.fetchrow(
            query,
            corpus_id,
            version_no,
            embedding_model,
            embedding_revision,
            embedding_dim,
            normalize_embeddings,
            query_instruction,
            embedding_config_fingerprint,
            config_json,
        )
        return dict(row)

    async def activate_corpus_version(
        self, corpus_version_id: str, activated_by: str
    ) -> bool:
        """Activates a corpus version, automatically superseding any existing active version."""
        row = await self.conn.fetchrow(
            "SELECT corpus_id, status FROM rag_corpus_versions WHERE id = $1::uuid;",
            corpus_version_id,
        )
        if not row or row["status"] not in {"qa_ready", "superseded"}:
            return False
        corpus_id = row["corpus_id"]

        await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET status = 'superseded'
            WHERE corpus_id = $1 AND status = 'active';
            """,
            corpus_id,
        )

        result = await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET status = 'active', activated_at = NOW(), activated_by = $2
            WHERE id = $1::uuid;
            """,
            corpus_version_id,
            activated_by,
        )
        return result.endswith("1")

    async def get_active_corpus_version(
        self, board_id: str, class_id: str, subject_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the active corpus version for a given board/class/subject scope."""
        query = """
        SELECT cv.*
        FROM rag_corpus_versions cv
        JOIN rag_corpora c ON cv.corpus_id = c.id
        WHERE c.board_id = $1 AND c.class_id = $2 AND c.subject_id = $3
          AND cv.status = 'active';
        """
        row = await self.conn.fetchrow(query, board_id, class_id, subject_id)
        return dict(row) if row else None

    async def create_document_version(
        self,
        corpus_version_id: str,
        resource_id: str,
        resource_version_id: str,
        pipeline_version: str,
        doc_title: str,
        total_chunks: int = 0,
    ) -> Dict[str, Any]:
        """Links a Module 2 resource version or JSONL ingestion doc version to a RAG corpus version."""
        query = """
        INSERT INTO rag_document_versions (
            corpus_version_id, resource_id, resource_version_id, pipeline_version, doc_title, total_chunks
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6)
        ON CONFLICT (resource_id, resource_version_id, pipeline_version, corpus_version_id)
        DO UPDATE SET doc_title = EXCLUDED.doc_title, total_chunks = EXCLUDED.total_chunks
        RETURNING *;
        """
        row = await self.conn.fetchrow(
            query,
            corpus_version_id,
            resource_id,
            resource_version_id,
            pipeline_version,
            doc_title,
            total_chunks,
        )
        return dict(row)

    async def replace_chapter_chunks(
        self,
        corpus_version_id: str,
        document_version_id: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Atomically replaces chunks for a document version within a building corpus version.

        1. Locks corpus version row FOR UPDATE and verifies status == 'building'.
        2. Calculates chunk count delta (new_count - old_count).
        3. Deletes existing chunks for document_version_id.
        4. Inserts new rag_chunks and chunk_expected_questions.
        5. Updates expected_chunk_count by delta and reconciles embedded_chunk_count.
        """
        # 1. Lock corpus version FOR UPDATE and verify status
        status_row = await self.conn.fetchrow(
            "SELECT status FROM rag_corpus_versions WHERE id = $1::uuid FOR UPDATE;",
            corpus_version_id,
        )
        if not status_row:
            raise RuntimeError(f"Corpus version '{corpus_version_id}' not found.")
        status = status_row["status"]
        if status != "building":
            raise RuntimeError(
                f"Corpus version '{corpus_version_id}' status is '{status}', expected 'building'."
            )

        new_chunk_count = len(chunks)

        # 2. Delete existing chunks for document_version_id (CASCADE deletes questions)
        await self.conn.execute(
            "DELETE FROM rag_chunks WHERE document_version_id = $1::uuid;",
            document_version_id,
        )

        inserted_chunks: List[Dict[str, Any]] = []

        # 3. Insert new chunks and expected questions
        for chunk in chunks:
            chunk_query = """
            INSERT INTO rag_chunks (
                document_version_id, corpus_version_id, chunk_index, content,
                chapter_id, topic_no, topic_title, page_start, page_end,
                content_type, metadata, content_hash, language, token_count
            )
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13, $14)
            RETURNING *;
            """
            metadata_json = json.dumps(chunk.get("metadata") or {})
            c_row = await self.conn.fetchrow(
                chunk_query,
                document_version_id,
                corpus_version_id,
                chunk["chunk_order"],
                chunk["chunk_text"],
                chunk["chapter_id"],
                chunk["topic_no"],
                chunk["topic_title"],
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk["content_type"],
                metadata_json,
                chunk["content_hash"],
                chunk.get("language", "en"),
                chunk.get("token_count", 0),
            )
            chunk_dict = dict(c_row)
            chunk_id = chunk_dict["id"]

            # Insert expected questions with NULL embedding
            expected_questions = chunk.get("expected_questions") or []
            for q_text in expected_questions:
                if q_text and isinstance(q_text, str) and q_text.strip():
                    await self.conn.execute(
                        """
                        INSERT INTO chunk_expected_questions (
                            chunk_id, question_text, question_normalized, question_hash, embedding
                        )
                        VALUES ($1::uuid, $2, $3, $4, NULL);
                        """,
                        chunk_id,
                        q_text.strip(),
                        normalize_expected_question(q_text),
                        hashlib.sha256(
                            normalize_expected_question(q_text).encode("utf-8")
                        ).hexdigest(),
                    )

            # JSONL visual keys are persisted only in the service database.  New
            # imports are intentionally non-displayable until reviewed locally.
            for visual in chunk.get("visuals") or []:
                title = visual["title"].strip()
                description = visual["description"].strip()
                visual_text = " ".join(f"{title} {description}".split()).lower()
                await self.conn.execute(
                    """
                    INSERT INTO rag_visuals (
                        chunk_id, visual_id, visual_type, storage_path, title, description,
                        storage_provider, storage_key, display_policy, review_status,
                        visual_text_hash
                    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, 'google_drive', $4,
                              'llm_decide', 'pending', $7);
                    """,
                    chunk_id,
                    visual["visual_id"],
                    visual["visual_type"],
                    visual["storage_key"],
                    title,
                    description,
                    hashlib.sha256(visual_text.encode("utf-8")).hexdigest(),
                )

            inserted_chunks.append(chunk_dict)

        # 4. Update total_chunks on document_version
        await self.conn.execute(
            "UPDATE rag_document_versions SET total_chunks = $1 WHERE id = $2::uuid;",
            new_chunk_count,
            document_version_id,
        )

        # 6. Reconcile both populations and record the building-version input state.
        # A source change is permitted only while building; each row still carries its
        # own input hash so unchanged vectors remain resumable.
        await self.refresh_embedding_counts(corpus_version_id)
        await self.refresh_embedding_input_fingerprint(corpus_version_id)

        return inserted_chunks

    async def refresh_embedding_input_fingerprint(
        self, corpus_version_id: str, allow_active: bool = False
    ) -> str:
        """Stores a deterministic fingerprint of the current building or active inputs."""
        rows = await self.conn.fetch(
            """
            SELECT c.id::text AS chunk_id, c.content_hash, c.topic_no, c.topic_title,
                   c.content,
                   COALESCE(array_agg(DISTINCT q.question_hash ORDER BY q.question_hash)
                       FILTER (WHERE q.id IS NOT NULL), '{}') AS question_hashes
                   ,COALESCE(array_agg(DISTINCT (v.visual_id || ':' || v.visual_text_hash)
                       ORDER BY (v.visual_id || ':' || v.visual_text_hash))
                       FILTER (WHERE v.id IS NOT NULL AND v.review_status = 'approved'), '{}') AS approved_visual_hashes
            FROM rag_chunks c
            LEFT JOIN chunk_expected_questions q ON q.chunk_id = c.id
            LEFT JOIN rag_visuals v ON v.chunk_id = c.id
            WHERE c.corpus_version_id = $1::uuid
            GROUP BY c.id
            ORDER BY c.id;
            """,
            corpus_version_id,
        )
        payload = [dict(row) for row in rows]
        fingerprint = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        condition = "status IN ('building', 'active')" if allow_active else "status = 'building'"
        await self.conn.execute(
            f"""
            UPDATE rag_corpus_versions
            SET embedding_input_fingerprint = $2
            WHERE id = $1::uuid AND {condition};
            """,
            corpus_version_id,
            fingerprint,
        )
        return fingerprint

    async def refresh_embedding_counts(self, corpus_version_id: str) -> Dict[str, int]:
        """Reconciles stored counters against valid vectors for this exact configuration."""
        row = await self.conn.fetchrow(
            """
            WITH version AS (
                SELECT * FROM rag_corpus_versions WHERE id = $1::uuid
            ), counts AS (
                SELECT
                    (SELECT COUNT(*) FROM rag_chunks c WHERE c.corpus_version_id = version.id) AS expected_chunks,
                    (SELECT COUNT(*) FROM rag_chunks c WHERE c.corpus_version_id = version.id
                        AND c.embedding IS NOT NULL AND vector_dims(c.embedding) = version.embedding_dim
                        AND c.embedding_status = 'embedded' AND c.embedding_model = version.embedding_model
                        AND c.embedding_revision = version.embedding_revision
                        AND c.embedding_config_fingerprint = version.embedding_config_fingerprint) AS embedded_chunks,
                    (SELECT COUNT(*) FROM chunk_expected_questions q JOIN rag_chunks c ON c.id = q.chunk_id
                        WHERE c.corpus_version_id = version.id) AS expected_questions,
                    (SELECT COUNT(*) FROM chunk_expected_questions q JOIN rag_chunks c ON c.id = q.chunk_id
                        WHERE c.corpus_version_id = version.id AND q.embedding IS NOT NULL
                        AND vector_dims(q.embedding) = version.embedding_dim AND q.embedding_status = 'embedded'
                        AND q.embedding_model = version.embedding_model
                        AND q.embedding_revision = version.embedding_revision
                        AND q.embedding_config_fingerprint = version.embedding_config_fingerprint) AS embedded_questions
                FROM version
            )
            UPDATE rag_corpus_versions cv
            SET expected_chunk_count = counts.expected_chunks,
                embedded_chunk_count = counts.embedded_chunks,
                expected_question_count = counts.expected_questions,
                embedded_question_count = counts.embedded_questions
            FROM counts
            WHERE cv.id = $1::uuid
            RETURNING cv.expected_chunk_count, cv.embedded_chunk_count,
                      cv.expected_question_count, cv.embedded_question_count;
            """,
            corpus_version_id,
        )
        return dict(row)

    async def fetch_chunks_for_embedding(
        self, corpus_version_id: str, configuration_fingerprint: str
    ) -> List[Dict[str, Any]]:
        """Fetches building-version chunk work without exposing unrelated metadata."""
        rows = await self.conn.fetch(
            """
            SELECT c.id, c.topic_no, c.topic_title, c.content, c.language, c.embedding,
                   c.embedding_status, c.embedding_input_hash,
                   c.embedding_config_fingerprint, c.embedding_model, c.embedding_revision,
                   COALESCE(jsonb_agg(jsonb_build_object('visual_id', v.visual_id, 'title', v.title, 'description', v.description)
                       ORDER BY v.visual_id) FILTER (WHERE v.id IS NOT NULL AND v.review_status = 'approved'), '[]'::jsonb) AS approved_visuals
            FROM rag_chunks c
            JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
            LEFT JOIN rag_visuals v ON v.chunk_id = c.id
            WHERE c.corpus_version_id = $1::uuid
              AND cv.status = 'building'
              AND cv.embedding_config_fingerprint = $2
            GROUP BY c.id
            ORDER BY c.created_at, c.id;
            """,
            corpus_version_id,
            configuration_fingerprint,
        )
        return [dict(row) for row in rows]

    async def fetch_questions_for_embedding(
        self, corpus_version_id: str, configuration_fingerprint: str
    ) -> List[Dict[str, Any]]:
        rows = await self.conn.fetch(
            """
            SELECT q.id, q.question_text, q.question_hash, c.language, q.embedding,
                   q.embedding_status, q.embedding_input_hash,
                   q.embedding_config_fingerprint, q.embedding_model, q.embedding_revision
            FROM chunk_expected_questions q
            JOIN rag_chunks c ON c.id = q.chunk_id
            JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
            WHERE c.corpus_version_id = $1::uuid
              AND cv.status = 'building'
              AND cv.embedding_config_fingerprint = $2
            ORDER BY q.created_at, q.id;
            """,
            corpus_version_id,
            configuration_fingerprint,
        )
        return [dict(row) for row in rows]

    async def get_corpus_version(
        self, corpus_version_id: str
    ) -> Optional[Dict[str, Any]]:
        row = await self.conn.fetchrow(
            "SELECT * FROM rag_corpus_versions WHERE id = $1::uuid;", corpus_version_id
        )
        return dict(row) if row else None

    async def write_chunk_embedding(
        self,
        chunk_id: str,
        corpus_version_id: str,
        vector: List[float],
        input_hash: str,
        embedding_model: str,
        embedding_revision: str,
        configuration_fingerprint: str,
    ) -> bool:
        self._validate_vector(vector)
        result = await self.conn.execute(
            """
            UPDATE rag_chunks c
            SET embedding = $3::text::halfvec, embedding_model = $5, embedding_revision = $6,
                embedding_config_fingerprint = $7, embedding_input_hash = $4,
                embedding_status = 'embedded', embedding_started_at = COALESCE(embedding_started_at, NOW()),
                embedding_completed_at = NOW(), embedding_error_code = NULL
            FROM rag_corpus_versions cv
            WHERE c.id = $1::uuid AND c.corpus_version_id = $2::uuid AND cv.id = c.corpus_version_id
              AND cv.status = 'building' AND cv.embedding_model = $5
              AND cv.embedding_revision = $6 AND cv.embedding_dim = 512
              AND cv.embedding_config_fingerprint = $7;
            """,
            chunk_id,
            corpus_version_id,
            str(vector),
            input_hash,
            embedding_model,
            embedding_revision,
            configuration_fingerprint,
        )
        return result.endswith("1")

    async def write_question_embedding(
        self,
        question_id: str,
        corpus_version_id: str,
        vector: List[float],
        input_hash: str,
        embedding_model: str,
        embedding_revision: str,
        configuration_fingerprint: str,
    ) -> bool:
        self._validate_vector(vector)
        result = await self.conn.execute(
            """
            UPDATE chunk_expected_questions q
            SET embedding = $3::text::halfvec, embedding_model = $5, embedding_revision = $6,
                embedding_config_fingerprint = $7, embedding_input_hash = $4,
                embedding_status = 'embedded',
                embedding_started_at = COALESCE(q.embedding_started_at, NOW()),
                embedding_completed_at = NOW(), embedding_error_code = NULL
            FROM rag_chunks c, rag_corpus_versions cv
            WHERE q.id = $1::uuid AND c.id = q.chunk_id AND c.corpus_version_id = $2::uuid
              AND cv.id = c.corpus_version_id AND cv.status = 'building'
              AND cv.embedding_model = $5 AND cv.embedding_revision = $6
              AND cv.embedding_dim = 512 AND cv.embedding_config_fingerprint = $7;
            """,
            question_id,
            corpus_version_id,
            str(vector),
            input_hash,
            embedding_model,
            embedding_revision,
            configuration_fingerprint,
        )
        return result.endswith("1")

    async def mark_embedding_failed(
        self, table: str, row_id: str, error_code: str
    ) -> bool:
        if table not in {"rag_chunks", "chunk_expected_questions"}:
            raise ValueError("Unsupported embedding table.")
        result = await self.conn.execute(
            f"""UPDATE {table} SET embedding_status = 'failed', embedding_error_code = $2,
                embedding_completed_at = NOW() WHERE id = $1::uuid;""",
            row_id,
            error_code[:80],
        )
        return result.endswith("1")

    async def completeness_report(
        self, corpus_version_id: str, *, require_building: bool = True
    ) -> Dict[str, Any]:
        """Returns sanitized readiness evidence; it never exposes content or vectors.

        ``qa_ready`` snapshots need the same provenance check again immediately
        before activation.  In that case callers set ``require_building=False``;
        this keeps the calculation shared with the Phase 3D completeness gate
        instead of trusting persisted counters alone.
        """
        version = await self.conn.fetchrow(
            "SELECT * FROM rag_corpus_versions WHERE id = $1::uuid;", corpus_version_id
        )
        if not version:
            return {"ready": False, "reasons": ["CORPUS_VERSION_NOT_FOUND"]}
        cv = dict(version)
        actual = await self.conn.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM rag_chunks WHERE corpus_version_id = $1::uuid) AS chunks,
                (SELECT COUNT(*) FROM rag_chunks WHERE corpus_version_id = $1::uuid
                    AND (embedding IS NULL OR vector_dims(embedding) <> $2 OR embedding_status <> 'embedded'
                         OR embedding_model <> $3 OR embedding_revision <> $4
                         OR embedding_config_fingerprint <> $5)) AS invalid_chunks,
                (SELECT COUNT(*) FROM chunk_expected_questions q JOIN rag_chunks c ON c.id = q.chunk_id
                    WHERE c.corpus_version_id = $1::uuid) AS questions,
                (SELECT COUNT(*) FROM chunk_expected_questions q JOIN rag_chunks c ON c.id = q.chunk_id
                    WHERE c.corpus_version_id = $1::uuid
                      AND (q.embedding IS NULL OR vector_dims(q.embedding) <> $2
                           OR q.embedding_status <> 'embedded' OR q.embedding_model <> $3
                           OR q.embedding_revision <> $4 OR q.embedding_config_fingerprint <> $5)) AS invalid_questions;
            """,
            corpus_version_id,
            cv["embedding_dim"],
            cv["embedding_model"],
            cv["embedding_revision"],
            cv["embedding_config_fingerprint"],
        )
        blocking_jobs = await self.conn.fetchval(
            """
            SELECT COUNT(*) FROM job_queue
            WHERE job_type IN ('embed_chunks', 'embed_questions')
              AND payload->>'corpus_version_id' = $1
              AND payload->>'embedding_config_fingerprint' = $2
              AND payload->>'embedding_input_fingerprint' = $3
              AND status IN ('queued', 'leased', 'running', 'retry_wait', 'failed');
            """,
            corpus_version_id,
            cv["embedding_config_fingerprint"],
            cv["embedding_input_fingerprint"],
        )
        reasons = []
        if require_building and cv["status"] != "building":
            reasons.append("CORPUS_NOT_BUILDING")
        if cv["embedding_dim"] != 512 or not cv["embedding_config_fingerprint"]:
            reasons.append("INVALID_EMBEDDING_CONFIGURATION")
        if cv["expected_chunk_count"] != actual["chunks"]:
            reasons.append("CHUNK_COUNT_MISMATCH")
        if cv["embedded_chunk_count"] != actual["chunks"] or actual["invalid_chunks"]:
            reasons.append("CHUNK_EMBEDDINGS_INCOMPLETE")
        if cv["expected_question_count"] != actual["questions"]:
            reasons.append("QUESTION_COUNT_MISMATCH")
        if (
            cv["embedded_question_count"] != actual["questions"]
            or actual["invalid_questions"]
        ):
            reasons.append("QUESTION_EMBEDDINGS_INCOMPLETE")
        if blocking_jobs:
            reasons.append("EMBEDDING_JOBS_NOT_SETTLED")
        return {"ready": not reasons, "reasons": reasons, "counts": dict(actual)}

    async def mark_qa_ready(self, corpus_version_id: str) -> Dict[str, Any]:
        report = await self.completeness_report(corpus_version_id)
        if not report["ready"]:
            return report
        result = await self.conn.execute(
            """UPDATE rag_corpus_versions SET status = 'qa_ready'
               WHERE id = $1::uuid AND status = 'building';""",
            corpus_version_id,
        )
        if not result.endswith("1"):
            return {"ready": False, "reasons": ["QA_READY_TRANSITION_REJECTED"]}
        return {"ready": True, "reasons": []}

    @staticmethod
    def _validate_vector(vector: List[float]) -> None:
        if len(vector) != 512:
            raise ValueError("Embedding vector dimension must be exactly 512.")

    async def insert_chunk(
        self,
        document_version_id: str,
        corpus_version_id: str,
        chunk_index: int,
        content: str,
        chapter_id: Optional[str] = None,
        topic_no: Optional[str] = None,
        topic_title: Optional[str] = None,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Inserts a single RAG chunk with chapter, topic, page range, and embedding."""
        query = """
        INSERT INTO rag_chunks (
            document_version_id, corpus_version_id, chunk_index, content,
            chapter_id, topic_no, topic_title, page_start, page_end, embedding
        )
        VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, document_version_id, corpus_version_id, chunk_index, content, chapter_id, topic_no, topic_title, page_start, page_end, created_at;
        """
        vec_str = str(embedding) if embedding is not None else None
        row = await self.conn.fetchrow(
            query,
            document_version_id,
            corpus_version_id,
            chunk_index,
            content,
            chapter_id,
            topic_no,
            topic_title,
            page_start,
            page_end,
            vec_str,
        )
        return dict(row)

    async def search_chunks_vector(
        self, corpus_version_id: str, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Executes an exact vector similarity search using cosine distance (<=>)."""
        self._validate_vector(query_embedding)
        query = """
        SELECT id, document_version_id, corpus_version_id, chunk_index, content,
               chapter_id, topic_no, topic_title, page_start, page_end,
               (embedding <=> $2::text::halfvec) AS distance
        FROM rag_chunks
        WHERE corpus_version_id = $1::uuid AND embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT $3;
        """
        vec_str = str(query_embedding)
        rows = await self.conn.fetch(query, corpus_version_id, vec_str, top_k)
        return [dict(r) for r in rows]

    async def search_chunks_lexical(
        self, corpus_version_id: str, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Executes a language-aware full-text search using 'simple' tsvector configuration."""
        query = """
        SELECT id, document_version_id, corpus_version_id, chunk_index, content,
               chapter_id, topic_no, topic_title, page_start, page_end,
               ts_rank(content_tsvector, plainto_tsquery('simple', $2)) AS rank
        FROM rag_chunks
        WHERE corpus_version_id = $1::uuid
          AND content_tsvector @@ plainto_tsquery('simple', $2)
        ORDER BY rank DESC
        LIMIT $3;
        """
        rows = await self.conn.fetch(query, corpus_version_id, query_text, top_k)
        return [dict(r) for r in rows]

    async def active_chapter_exists(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        corpus_version_id: str,
        chapter_id: str,
    ) -> bool:
        """Checks a chapter against the exact active corpus scope in SQL."""
        return bool(
            await self.conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM rag_chunks c
                    JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
                    JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
                    WHERE corpus.board_id = $1
                      AND corpus.class_id = $2
                      AND corpus.subject_id = $3
                      AND cv.id = $4::uuid
                      AND cv.status = 'active'
                      AND c.corpus_version_id = cv.id
                      AND c.chapter_id = $5
                );
                """,
                board_id,
                class_id,
                subject_id,
                corpus_version_id,
                chapter_id,
            )
        )

    async def search_active_chunks_cosine(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        corpus_version_id: str,
        query_embedding: List[float],
        chapter_id: Optional[str],
        top_k: int,
        allow_named_draft: bool = False,
    ) -> List[Dict[str, Any]]:
        """Exact scoped cosine retrieval of verified chunk vectors.

        The board/class/subject and active corpus-version predicates deliberately
        appear in this SQL rather than relying on caller-side filtering.
        """
        self._validate_vector(query_embedding)
        rows = await self.conn.fetch(
            """
            SELECT c.id::text AS citation_id, c.content, c.chapter_id, c.topic_no,
                   c.topic_title, c.page_start, c.page_end
            FROM rag_chunks c
            JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
            JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
            WHERE corpus.board_id = $1
              AND corpus.class_id = $2
              AND corpus.subject_id = $3
              AND cv.id = $4::uuid
              AND (cv.status = 'active' OR ($8::boolean AND cv.status IN ('building', 'qa_ready')))
              AND c.corpus_version_id = cv.id
              AND ($6::text IS NULL OR c.chapter_id = $6)
              AND c.embedding IS NOT NULL
              AND vector_dims(c.embedding) = cv.embedding_dim
              AND c.embedding_status = 'embedded'
              AND c.embedding_model = cv.embedding_model
              AND c.embedding_revision = cv.embedding_revision
              AND c.embedding_config_fingerprint = cv.embedding_config_fingerprint
            ORDER BY c.embedding <=> $5::text::halfvec ASC, c.id ASC
            LIMIT $7;
            """,
            board_id,
            class_id,
            subject_id,
            corpus_version_id,
            str(query_embedding),
            chapter_id,
            top_k,
            allow_named_draft,
        )
        return [dict(row) for row in rows]

    async def search_active_expected_questions_cosine(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        corpus_version_id: str,
        query_embedding: List[float],
        chapter_id: Optional[str],
        top_k: int,
        allow_named_draft: bool = False,
    ) -> List[Dict[str, Any]]:
        """Searches expected-question vectors and ranks deduplicated parent chunks.

        Each parent retains its best individual expected-question match, then the
        distinct parents receive contiguous channel ranks. Expected-question IDs,
        text, and raw distances never leave this query.
        """
        self._validate_vector(query_embedding)
        rows = await self.conn.fetch(
            """
            WITH parent_best_matches AS (
                SELECT DISTINCT ON (c.id)
                       c.id::text AS citation_id, c.content, c.chapter_id, c.topic_no,
                       c.topic_title, c.page_start, c.page_end,
                       q.embedding <=> $5::text::halfvec AS best_question_distance
                FROM chunk_expected_questions q
                JOIN rag_chunks c ON c.id = q.chunk_id
                JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
                JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
                WHERE corpus.board_id = $1
                  AND corpus.class_id = $2
                  AND corpus.subject_id = $3
                  AND cv.id = $4::uuid
                  AND (cv.status = 'active' OR ($8::boolean AND cv.status IN ('building', 'qa_ready')))
                  AND c.corpus_version_id = cv.id
                  AND ($6::text IS NULL OR c.chapter_id = $6)
                  AND q.embedding IS NOT NULL
                  AND vector_dims(q.embedding) = cv.embedding_dim
                  AND q.embedding_status = 'embedded'
                  AND q.embedding_model = cv.embedding_model
                  AND q.embedding_revision = cv.embedding_revision
                  AND q.embedding_config_fingerprint = cv.embedding_config_fingerprint
                ORDER BY c.id, q.embedding <=> $5::text::halfvec ASC, q.id ASC
            )
            SELECT citation_id, content, chapter_id, topic_no, topic_title, page_start,
                   page_end,
                   row_number() OVER (
                       ORDER BY best_question_distance ASC, citation_id ASC
                   ) AS expected_question_rank
            FROM parent_best_matches
            ORDER BY best_question_distance ASC, citation_id ASC
            LIMIT $7;
            """,
            board_id,
            class_id,
            subject_id,
            corpus_version_id,
            str(query_embedding),
            chapter_id,
            top_k,
            allow_named_draft,
        )
        return [dict(row) for row in rows]

    async def search_active_chunks_lexical(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        corpus_version_id: str,
        query_text: str,
        chapter_id: Optional[str],
        top_k: int,
        allow_named_draft: bool = False,
    ) -> List[Dict[str, Any]]:
        """Exact scoped PostgreSQL ``simple`` full-text chunk retrieval."""
        rows = await self.conn.fetch(
            """
            SELECT c.id::text AS citation_id, c.content, c.chapter_id, c.topic_no,
                   c.topic_title, c.page_start, c.page_end
            FROM rag_chunks c
            JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
            JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
            WHERE corpus.board_id = $1
              AND corpus.class_id = $2
              AND corpus.subject_id = $3
              AND cv.id = $4::uuid
              AND (cv.status = 'active' OR ($8::boolean AND cv.status IN ('building', 'qa_ready')))
              AND c.corpus_version_id = cv.id
              AND ($6::text IS NULL OR c.chapter_id = $6)
              AND c.content_tsvector @@ plainto_tsquery('simple', $5)
            ORDER BY ts_rank(c.content_tsvector, plainto_tsquery('simple', $5)) DESC,
                     c.id ASC
            LIMIT $7;
            """,
            board_id,
            class_id,
            subject_id,
            corpus_version_id,
            query_text,
            chapter_id,
            top_k,
            allow_named_draft,
        )
        return [dict(row) for row in rows]

    async def get_eligible_retrieval_visuals(
        self, citation_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return only safe reviewed metadata; never storage identifiers."""
        if not citation_ids:
            return {}
        rows = await self.conn.fetch(
            """
            SELECT v.chunk_id::text AS citation_id, v.visual_id, v.title,
                   v.description, v.display_policy
            FROM rag_visuals v
            WHERE v.chunk_id = ANY($1::uuid[])
              AND v.review_status = 'approved'
              AND v.display_policy IN ('always', 'llm_decide')
            ORDER BY v.chunk_id, v.visual_id
            """,
            citation_ids,
        )
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            citation_id = item.pop("citation_id")
            result.setdefault(citation_id, []).append(item)
        return result

    async def get_active_topic_chunks(
        self,
        *,
        board_id: str,
        class_id: str,
        subject_id: str,
        corpus_version_id: str,
        anchor_citation_ids: List[str],
        max_topics: int = 3,
        max_chunks: int = 12,
    ) -> List[Dict[str, Any]]:
        """Expand ranked anchors to complete, ordered textbook topics.

        Topic number is the primary identity because ingestion may give split
        chunks titles such as ``Part 1``/``Part 2``.  Title is used only when a
        source has no topic number.  Every query repeats the active corpus and
        board/class/subject predicates so expansion cannot cross a scope.
        """
        if not anchor_citation_ids or max_topics < 1 or max_chunks < 1:
            return []
        anchors = await self.conn.fetch(
            """
            SELECT c.id::text AS citation_id, c.chapter_id, c.topic_no,
                   c.topic_title
            FROM unnest($5::uuid[]) WITH ORDINALITY AS requested(id, position)
            JOIN rag_chunks c ON c.id = requested.id
            JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
            JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
            WHERE corpus.board_id = $1
              AND corpus.class_id = $2
              AND corpus.subject_id = $3
              AND cv.id = $4::uuid
              AND cv.status = 'active'
              AND c.corpus_version_id = cv.id
            ORDER BY requested.position
            """,
            board_id,
            class_id,
            subject_id,
            corpus_version_id,
            anchor_citation_ids,
        )
        topics: list[tuple[str | None, str | None, str | None]] = []
        seen_topics: set[tuple[str | None, str, str]] = set()
        for anchor in anchors:
            topic_no = anchor["topic_no"]
            topic_title = anchor["topic_title"]
            topic_key = (
                anchor["chapter_id"],
                "number" if topic_no else "title",
                topic_no or topic_title or anchor["citation_id"],
            )
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            topics.append((anchor["chapter_id"], topic_no, topic_title))
            if len(topics) >= max_topics:
                break

        expanded: list[Dict[str, Any]] = []
        seen_chunks: set[str] = set()
        for chapter_id, topic_no, topic_title in topics:
            remaining = max_chunks - len(expanded)
            if remaining <= 0:
                break
            rows = await self.conn.fetch(
                """
                SELECT c.id::text AS citation_id, c.content, c.chapter_id,
                       c.topic_no, c.topic_title, c.page_start, c.page_end
                FROM rag_chunks c
                JOIN rag_corpus_versions cv ON cv.id = c.corpus_version_id
                JOIN rag_corpora corpus ON corpus.id = cv.corpus_id
                WHERE corpus.board_id = $1
                  AND corpus.class_id = $2
                  AND corpus.subject_id = $3
                  AND cv.id = $4::uuid
                  AND cv.status = 'active'
                  AND c.corpus_version_id = cv.id
                  AND c.chapter_id IS NOT DISTINCT FROM $5::text
                  AND (
                    ($6::text IS NOT NULL AND c.topic_no = $6)
                    OR ($6::text IS NULL AND c.topic_no IS NULL
                        AND c.topic_title IS NOT DISTINCT FROM $7::text)
                  )
                ORDER BY c.chunk_index, c.id
                LIMIT $8
                """,
                board_id,
                class_id,
                subject_id,
                corpus_version_id,
                chapter_id,
                topic_no,
                topic_title,
                remaining,
            )
            for row in rows:
                item = dict(row)
                if item["citation_id"] not in seen_chunks:
                    seen_chunks.add(item["citation_id"])
                    expanded.append(item)
        return expanded

    async def promote_chapter_from_temp_to_active(
        self,
        *,
        temp_version_id: str,
        active_version_id: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str,
        is_replace: bool = False,
    ) -> Dict[str, Any]:
        """Atomically moves chapter documents/chunks from temp building corpus into active corpus.

        1. Hard-verifies matching embedding configuration between temp and active versions.
        2. Cleans up citations, visuals, and LLM-generated Q&A for chapter.
        3. If replacing: deletes old chapter chunks & document versions from active version.
        4. Updates corpus_version_id = active_version_id on temp document version and chunks.
        5. Reconciles active version counts and active fingerprint.
        6. Invalidates active QA approvals.
        7. Deletes empty temp building version.
        """
        temp_ver = await self.conn.fetchrow(
            "SELECT * FROM rag_corpus_versions WHERE id = $1::uuid FOR UPDATE;", temp_version_id
        )
        active_ver = await self.conn.fetchrow(
            "SELECT * FROM rag_corpus_versions WHERE id = $1::uuid FOR UPDATE;", active_version_id
        )
        if not temp_ver or not active_ver:
            raise ValueError("CORPUS_VERSION_NOT_FOUND")

        temp_cfg = (
            temp_ver["embedding_model"],
            temp_ver["embedding_revision"],
            temp_ver["embedding_dim"],
            temp_ver["embedding_config_fingerprint"],
            temp_ver["normalize_embeddings"],
        )
        active_cfg = (
            active_ver["embedding_model"],
            active_ver["embedding_revision"],
            active_ver["embedding_dim"],
            active_ver["embedding_config_fingerprint"],
            active_ver["normalize_embeddings"],
        )
        if temp_cfg != active_cfg:
            raise ValueError("EMBEDDING_CONFIGURATION_MISMATCH_CANNOT_PROMOTE")

        from app.repositories.question_bank_repository import QuestionBankRepository

        bank_repo = QuestionBankRepository(self.conn)

        old_chunks = await self.conn.fetch(
            "SELECT id::text FROM rag_chunks WHERE corpus_version_id = $1::uuid AND chapter_id = $2;",
            active_version_id,
            chapter_id,
        )
        old_chunk_ids = [r["id"] for r in old_chunks]

        old_visual_ids: list[str] = []
        if old_chunk_ids:
            old_visuals = await self.conn.fetch(
                "SELECT id::text FROM rag_visuals WHERE chunk_id = ANY($1::uuid[]);",
                old_chunk_ids,
            )
            old_visual_ids = [r["id"] for r in old_visuals]

        qa_summary = await bank_repo.cleanup_chapter_qa(
            board_id=board_id,
            class_id=class_id,
            subject_id=subject_id,
            chapter_id=chapter_id,
            old_chunk_ids=old_chunk_ids,
            old_visual_ids=old_visual_ids,
        )

        if old_chunk_ids:
            await self.conn.execute(
                "DELETE FROM rag_chunks WHERE corpus_version_id = $1::uuid AND chapter_id = $2;",
                active_version_id,
                chapter_id,
            )
            await self.conn.execute(
                """DELETE FROM rag_document_versions
                   WHERE corpus_version_id = $1::uuid AND resource_id = $2;""",
                active_version_id,
                f"jsonl:chapter:{chapter_id}",
            )

        await self.conn.execute(
            """UPDATE rag_document_versions
               SET corpus_version_id = $2::uuid
               WHERE corpus_version_id = $1::uuid;""",
            temp_version_id,
            active_version_id,
        )
        await self.conn.execute(
            """UPDATE rag_chunks
               SET corpus_version_id = $2::uuid
               WHERE corpus_version_id = $1::uuid;""",
            temp_version_id,
            active_version_id,
        )

        await self.refresh_embedding_counts(active_version_id)
        await self.refresh_embedding_input_fingerprint(active_version_id, allow_active=True)

        await self.conn.execute(
            """UPDATE rag_corpus_qa_approvals
               SET invalidated_at = NOW(), invalidated_reason = 'chapter_promoted'
               WHERE corpus_version_id = $1::uuid AND invalidated_at IS NULL;""",
            active_version_id,
        )

        await self.conn.execute(
            "DELETE FROM rag_corpus_versions WHERE id = $1::uuid;", temp_version_id
        )

        return {
            "status": "promoted",
            "active_version_id": active_version_id,
            "chapter_id": chapter_id,
            "qa_summary": qa_summary,
        }

    async def delete_chapter_from_active(
        self,
        *,
        active_version_id: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str,
    ) -> Dict[str, Any]:
        """Atomically deletes a chapter's chunks, expected questions, document versions, and LLM Q&A from active corpus."""
        active_ver = await self.conn.fetchrow(
            "SELECT * FROM rag_corpus_versions WHERE id = $1::uuid FOR UPDATE;", active_version_id
        )
        if not active_ver or active_ver["status"] != "active":
            raise ValueError("ACTIVE_CORPUS_VERSION_NOT_FOUND")

        from app.repositories.question_bank_repository import QuestionBankRepository

        bank_repo = QuestionBankRepository(self.conn)

        old_chunks = await self.conn.fetch(
            "SELECT id::text FROM rag_chunks WHERE corpus_version_id = $1::uuid AND chapter_id = $2;",
            active_version_id,
            chapter_id,
        )
        old_chunk_ids = [r["id"] for r in old_chunks]

        old_visual_ids: list[str] = []
        if old_chunk_ids:
            old_visuals = await self.conn.fetch(
                "SELECT id::text FROM rag_visuals WHERE chunk_id = ANY($1::uuid[]);",
                old_chunk_ids,
            )
            old_visual_ids = [r["id"] for r in old_visuals]

        qa_summary = await bank_repo.cleanup_chapter_qa(
            board_id=board_id,
            class_id=class_id,
            subject_id=subject_id,
            chapter_id=chapter_id,
            old_chunk_ids=old_chunk_ids,
            old_visual_ids=old_visual_ids,
        )

        if old_chunk_ids:
            await self.conn.execute(
                "DELETE FROM rag_chunks WHERE corpus_version_id = $1::uuid AND chapter_id = $2;",
                active_version_id,
                chapter_id,
            )
            await self.conn.execute(
                """DELETE FROM rag_document_versions
                   WHERE corpus_version_id = $1::uuid AND resource_id = $2;""",
                active_version_id,
                f"jsonl:chapter:{chapter_id}",
            )

        await self.refresh_embedding_counts(active_version_id)
        await self.refresh_embedding_input_fingerprint(active_version_id, allow_active=True)

        await self.conn.execute(
            """UPDATE rag_corpus_qa_approvals
               SET invalidated_at = NOW(), invalidated_reason = 'chapter_deleted'
               WHERE corpus_version_id = $1::uuid AND invalidated_at IS NULL;""",
            active_version_id,
        )

        return {
            "status": "deleted",
            "active_version_id": active_version_id,
            "chapter_id": chapter_id,
            "deleted_chunks": len(old_chunk_ids),
            "qa_summary": qa_summary,
        }

