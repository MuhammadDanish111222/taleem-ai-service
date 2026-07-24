"""AI Requests and Answers Repository using Asyncpg."""

import json
from typing import Any, Dict, List, Optional

import asyncpg


class AIRequestRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def create_request(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        language: str,
        answer_mode: str,
        raw_question: str,
        normalized_question: str,
        question_hash: str,
        corpus_version_id: Optional[str] = None,
        prompt_version: str = "v1",
    ) -> Dict[str, Any]:
        """Creates an AI request record."""
        query = """
        INSERT INTO ai_requests (
            board_id, class_id, subject_id, language, answer_mode,
            raw_question, normalized_question, question_hash,
            corpus_version_id, prompt_version, status
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::uuid, $10, 'pending')
        RETURNING *;
        """
        row = await self.conn.fetchrow(
            query,
            board_id,
            class_id,
            subject_id,
            language,
            answer_mode,
            raw_question,
            normalized_question,
            question_hash,
            corpus_version_id,
            prompt_version,
        )
        return dict(row)

    async def create_answer(
        self,
        request_id: str,
        answer_text: str,
        citation_sources: Optional[List[Dict[str, Any]]] = None,
        chunk_text_score: Optional[float] = None,
        expected_question_score: Optional[float] = None,
        tokens_used: int = 0,
        latency_ms: int = 0,
    ) -> Dict[str, Any]:
        """Records an AI answer and updates request status to completed."""
        citations_json = json.dumps(citation_sources or [])
        query_answer = """
        INSERT INTO ai_answers (
            request_id, answer_text, citation_sources,
            chunk_text_score, expected_question_score, tokens_used, latency_ms
        )
        VALUES ($1::uuid, $2, $3::jsonb, $4, $5, $6, $7)
        RETURNING *;
        """
        answer_row = await self.conn.fetchrow(
            query_answer,
            request_id,
            answer_text,
            citations_json,
            chunk_text_score,
            expected_question_score,
            tokens_used,
            latency_ms,
        )

        await self.conn.execute(
            "UPDATE ai_requests SET status = 'completed' WHERE id = $1::uuid;",
            request_id,
        )
        return dict(answer_row)

    async def find_cached_answer(
        self,
        board_id: str,
        class_id: str,
        subject_id: str,
        answer_mode: str,
        language: str,
        question_hash: str,
        corpus_version_id: Optional[str] = None,
        prompt_version: str = "v1",
    ) -> Optional[Dict[str, Any]]:
        """Exact-answer cache lookup using composite key index."""
        query = """
        SELECT req.id AS request_id, req.board_id, req.class_id, req.subject_id,
               req.normalized_question, req.question_hash, req.answer_mode, req.language,
               ans.id AS answer_id, ans.answer_text, ans.citation_sources,
               ans.chunk_text_score, ans.expected_question_score, ans.created_at
        FROM ai_requests req
        JOIN ai_answers ans ON ans.request_id = req.id
        WHERE req.board_id = $1
          AND req.class_id = $2
          AND req.subject_id = $3
          AND req.answer_mode = $4
          AND req.language = $5
          AND req.question_hash = $6
          AND (req.corpus_version_id IS NOT DISTINCT FROM $7::uuid)
          AND req.prompt_version = $8
          AND req.status IN ('completed', 'cached')
        ORDER BY ans.created_at DESC
        LIMIT 1;
        """
        row = await self.conn.fetchrow(
            query,
            board_id,
            class_id,
            subject_id,
            answer_mode,
            language,
            question_hash,
            corpus_version_id,
            prompt_version,
        )
        return dict(row) if row else None
