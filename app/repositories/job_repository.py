"""Durable Job Queue Repository using Asyncpg and Explicit SQL."""

from typing import Optional, Dict, Any, List
import json
import asyncpg

class JobRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def create_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Queues a new background job with idempotency support."""
        query = """
        INSERT INTO job_queue (job_type, idempotency_key, payload, status, stage, progress, max_attempts)
        VALUES ($1, $2, $3::jsonb, 'queued', 'pending', 0, $4)
        ON CONFLICT (idempotency_key) DO UPDATE SET updated_at = NOW()
        RETURNING *;
        """
        row = await self.conn.fetchrow(
            query, job_type, idempotency_key, json.dumps(payload), max_attempts
        )
        return dict(row)

    async def lease_job(
        self,
        worker_id: str,
        supported_types: List[str],
        lease_duration_seconds: int = 300
    ) -> Optional[Dict[str, Any]]:
        """Atomically leases a queued or retryable job using FOR UPDATE SKIP LOCKED."""
        query = """
        WITH candidate AS (
            SELECT id FROM job_queue
            WHERE (status = 'queued' OR (status = 'retry_wait' AND next_retry_at <= NOW()))
              AND job_type = ANY($1::text[])
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE job_queue j
        SET status = 'leased',
            locked_by = $2,
            locked_at = NOW(),
            heartbeat_at = NOW(),
            attempt_count = j.attempt_count + 1,
            stage = 'leased',
            updated_at = NOW()
        FROM candidate
        WHERE j.id = candidate.id
        RETURNING j.*;
        """
        row = await self.conn.fetchrow(query, supported_types, worker_id)
        return dict(row) if row else None

    async def update_heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Extends worker lease heartbeat."""
        query = """
        UPDATE job_queue
        SET heartbeat_at = NOW(), updated_at = NOW()
        WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
        """
        result = await self.conn.execute(query, job_id, worker_id)
        return result.endswith("1")

    async def update_progress(self, job_id: str, stage: str, progress: float) -> bool:
        """Updates job progress and stage."""
        query = """
        UPDATE job_queue
        SET stage = $2, progress = $3, status = 'running', updated_at = NOW()
        WHERE id = $1::uuid;
        """
        result = await self.conn.execute(query, job_id, stage, progress)
        return result.endswith("1")

    async def complete_job(self, job_id: str, final_progress: float = 100.0) -> bool:
        """Marks a job as succeeded."""
        query = """
        UPDATE job_queue
        SET status = 'succeeded', progress = $2, stage = 'completed', completed_at = NOW(), updated_at = NOW()
        WHERE id = $1::uuid;
        """
        result = await self.conn.execute(query, job_id, final_progress)
        return result.endswith("1")

    async def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
        retry_delay_seconds: Optional[int] = None
    ) -> bool:
        """Fails a job or schedules it for retry depending on remaining attempts."""
        job = await self.get_job(job_id)
        if not job:
            return False
            
        if retry_delay_seconds and job["attempt_count"] < job["max_attempts"]:
            query = """
            UPDATE job_queue
            SET status = 'retry_wait',
                error_code = $2,
                error_message = $3,
                next_retry_at = NOW() + ($4 || ' seconds')::interval,
                locked_by = NULL,
                locked_at = NULL,
                updated_at = NOW()
            WHERE id = $1::uuid;
            """
            result = await self.conn.execute(query, job_id, error_code, error_message, str(retry_delay_seconds))
        else:
            query = """
            UPDATE job_queue
            SET status = 'failed',
                error_code = $2,
                error_message = $3,
                completed_at = NOW(),
                updated_at = NOW()
            WHERE id = $1::uuid;
            """
            result = await self.conn.execute(query, job_id, error_code, error_message)
        return result.endswith("1")

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a job by ID."""
        row = await self.conn.fetchrow("SELECT * FROM job_queue WHERE id = $1::uuid;", job_id)
        return dict(row) if row else None
