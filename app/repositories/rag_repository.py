"""RAG Schema Repository using Asyncpg for vector and lexical queries."""

from typing import Optional, Dict, Any, List
import json
import asyncpg

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

    async def create_corpus_version(
        self,
        corpus_id: str,
        version_no: int,
        embedding_model: str,
        embedding_revision: str,
        embedding_dim: int = 768,
        chunking_config: Optional[Dict[str, Any]] = None
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
            query, corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, config_json
        )
        return dict(row)

    async def activate_corpus_version(self, corpus_version_id: str, activated_by: str) -> bool:
        """Activates a corpus version, automatically superseding any existing active version."""
        # 1. Fetch corpus_id for this version
        row = await self.conn.fetchrow(
            "SELECT corpus_id FROM rag_corpus_versions WHERE id = $1::uuid;", corpus_version_id
        )
        if not row:
            return False
        corpus_id = row["corpus_id"]

        # 2. Mark current active versions as superseded
        await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET status = 'superseded'
            WHERE corpus_id = $1 AND status = 'active';
            """,
            corpus_id
        )

        # 3. Mark target version active (enforced by partial unique index)
        result = await self.conn.execute(
            """
            UPDATE rag_corpus_versions
            SET status = 'active', activated_at = NOW(), activated_by = $2
            WHERE id = $1::uuid;
            """,
            corpus_version_id, activated_by
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
        total_chunks: int = 0
    ) -> Dict[str, Any]:
        """Links a Module 2 resource version to a RAG corpus version."""
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
            query, corpus_version_id, resource_id, resource_version_id, pipeline_version, doc_title, total_chunks
        )
        return dict(row)

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
        embedding: Optional[List[float]] = None
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
            query, document_version_id, corpus_version_id, chunk_index, content,
            chapter_id, topic_no, topic_title, page_start, page_end, vec_str
        )
        return dict(row)

    async def search_chunks_vector(
        self,
        corpus_version_id: str,
        query_embedding: List[float],
        top_k: int = 5
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
        self,
        corpus_version_id: str,
        query_text: str,
        top_k: int = 5
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
