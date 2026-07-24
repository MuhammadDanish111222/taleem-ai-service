"""RAG Schema Repository using Asyncpg for vector and lexical queries."""

import json
from typing import Any, Dict, List, Optional

import asyncpg

from app.services.ingestion.normalization import normalize_expected_question


class RagRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

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
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_revision: str = "main",
        embedding_dim: int = 768,
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
            return dict(existing)

        # 4. Create new building version (version_no = max + 1)
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
            corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, status
        )
        VALUES ($1::uuid, $2, $3, $4, $5, 'building')
        RETURNING *;
        """
        new_version = await self.conn.fetchrow(
            insert_version_query,
            corpus_id,
            new_version_no,
            embedding_model,
            embedding_revision,
            embedding_dim,
        )
        return dict(new_version)

    async def create_corpus_version(
        self,
        corpus_id: str,
        version_no: int,
        embedding_model: str,
        embedding_revision: str,
        embedding_dim: int = 768,
        chunking_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates a new corpus version in building status."""
        query = """
        INSERT INTO rag_corpus_versions (
            corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, chunking_config, status
        )
        VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, 'building')
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
            config_json,
        )
        return dict(row)

    async def activate_corpus_version(
        self, corpus_version_id: str, activated_by: str
    ) -> bool:
        """Activates a corpus version, automatically superseding any existing active version."""
        row = await self.conn.fetchrow(
            "SELECT corpus_id FROM rag_corpus_versions WHERE id = $1::uuid;",
            corpus_version_id,
        )
        if not row:
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

        # 2. Compute old chunk count for this document version
        old_count_row = await self.conn.fetchrow(
            "SELECT COUNT(*) as count FROM rag_chunks WHERE document_version_id = $1::uuid;",
            document_version_id,
        )
        old_chunk_count = old_count_row["count"] if old_count_row else 0
        new_chunk_count = len(chunks)
        delta = new_chunk_count - old_chunk_count

        # 3. Delete existing chunks for document_version_id (CASCADE deletes chunk_expected_questions)
        await self.conn.execute(
            "DELETE FROM rag_chunks WHERE document_version_id = $1::uuid;",
            document_version_id,
        )

        inserted_chunks: List[Dict[str, Any]] = []

        # 4. Insert new chunks and expected questions
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
                        INSERT INTO chunk_expected_questions (chunk_id, question_text, question_normalized, embedding)
                        VALUES ($1::uuid, $2, $3, NULL);
                        """,
                        chunk_id,
                        q_text.strip(),
                        normalize_expected_question(q_text),
                    )

            inserted_chunks.append(chunk_dict)

        # 5. Update total_chunks on document_version
        await self.conn.execute(
            "UPDATE rag_document_versions SET total_chunks = $1 WHERE id = $2::uuid;",
            new_chunk_count,
            document_version_id,
        )

        # 6. Update expected_chunk_count by delta
        await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET expected_chunk_count = GREATEST(0, expected_chunk_count + $1)
            WHERE id = $2::uuid;
            """,
            delta,
            corpus_version_id,
        )

        # 7. Reconcile embedded_chunk_count from actual non-null embeddings remaining
        await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET embedded_chunk_count = (
                SELECT COUNT(*) FROM rag_chunks WHERE corpus_version_id = $1::uuid AND embedding IS NOT NULL
            )
            WHERE id = $1::uuid;
            """,
            corpus_version_id,
        )

        return inserted_chunks

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
        """Executes an exact vector similarity search using L2 distance (<->)."""
        query = """
        SELECT id, document_version_id, corpus_version_id, chunk_index, content,
               chapter_id, topic_no, topic_title, page_start, page_end,
               (embedding <-> $2::vector) AS distance
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
