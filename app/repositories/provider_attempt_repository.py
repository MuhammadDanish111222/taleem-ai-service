"""Provider Attempt Repository using Asyncpg."""

from typing import Optional, Dict, Any, List
import asyncpg

class ProviderAttemptRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def record_attempt(
        self,
        provider: str,
        model: str,
        status: str,
        attempt_no: int = 1,
        ai_request_id: Optional[str] = None,
        job_id: Optional[str] = None,
        provider_request_id: Optional[str] = None,
        system_fingerprint: Optional[str] = None,
        finish_reason: Optional[str] = None,
        prompt_tokens: int = 0,
        cache_tokens: int = 0,
        reasoning_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
        error_code: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records an external provider API attempt."""
        query = """
        INSERT INTO provider_attempts (
            ai_request_id, job_id, provider, model, attempt_no,
            provider_request_id, system_fingerprint, finish_reason,
            prompt_tokens, cache_tokens, reasoning_tokens, completion_tokens,
            latency_ms, status, error_code, trace_id
        )
        VALUES (
            $1::uuid, $2::uuid, $3, $4, $5,
            $6, $7, $8,
            $9, $10, $11, $12,
            $13, $14, $15, $16
        )
        RETURNING *;
        """
        row = await self.conn.fetchrow(
            query,
            ai_request_id, job_id, provider, model, attempt_no,
            provider_request_id, system_fingerprint, finish_reason,
            prompt_tokens, cache_tokens, reasoning_tokens, completion_tokens,
            latency_ms, status, error_code, trace_id
        )
        return dict(row)

    async def get_attempts_for_request(self, ai_request_id: str) -> List[Dict[str, Any]]:
        """Queries provider attempts for a specific AI request."""
        query = "SELECT * FROM provider_attempts WHERE ai_request_id = $1::uuid ORDER BY attempt_no ASC;"
        rows = await self.conn.fetch(query, ai_request_id)
        return [dict(r) for r in rows]

    async def get_attempts_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Queries provider attempts for a specific background job."""
        query = "SELECT * FROM provider_attempts WHERE job_id = $1::uuid ORDER BY attempt_no ASC;"
        rows = await self.conn.fetch(query, job_id)
        return [dict(r) for r in rows]
