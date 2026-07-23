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
        """Extends worker lease heartbeat. Enforces worker_id lock ownership."""
        query = """
        UPDATE job_queue
        SET heartbeat_at = NOW(), updated_at = NOW()
        WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
        """
        result = await self.conn.execute(query, job_id, worker_id)
        return self._affected_rows(result) > 0

    async def update_progress(
        self,
        job_id: str,
        stage_or_worker: str,
        progress_or_stage: Any,
        progress: Optional[float] = None
    ) -> bool:
        """Updates job progress and stage. Enforces worker_id lock ownership when provided."""
        if progress is not None:
            worker_id = stage_or_worker
            stage = str(progress_or_stage)
            prog_val = float(progress)
            query = """
            UPDATE job_queue
            SET stage = $3, progress = $4, status = 'running', heartbeat_at = NOW(), updated_at = NOW()
            WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
            """
            result = await self.conn.execute(query, job_id, worker_id, stage, prog_val)
        else:
            stage = stage_or_worker
            prog_val = float(progress_or_stage)
            query = """
            UPDATE job_queue
            SET stage = $2, progress = $3, status = 'running', updated_at = NOW()
            WHERE id = $1::uuid;
            """
            result = await self.conn.execute(query, job_id, stage, prog_val)
        return self._affected_rows(result) > 0

    async def complete_job(
        self,
        job_id: str,
        worker_id_or_progress: Any = 100.0,
        final_progress: float = 100.0
    ) -> bool:
        """Marks a job as succeeded. Enforces worker_id lock ownership when provided as str."""
        if isinstance(worker_id_or_progress, str):
            worker_id = worker_id_or_progress
            query = """
            UPDATE job_queue
            SET status = 'succeeded', progress = $3, stage = 'completed', completed_at = NOW(), updated_at = NOW()
            WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
            """
            result = await self.conn.execute(query, job_id, worker_id, final_progress)
        else:
            prog_val = float(worker_id_or_progress)
            query = """
            UPDATE job_queue
            SET status = 'succeeded', progress = $2, stage = 'completed', completed_at = NOW(), updated_at = NOW()
            WHERE id = $1::uuid;
            """
            result = await self.conn.execute(query, job_id, prog_val)
        return self._affected_rows(result) > 0

    async def fail_job(
        self,
        job_id: str,
        error_code_or_worker: str,
        error_message_or_code: str,
        retry_delay_or_message: Any = None,
        retry_delay_seconds: Optional[int] = None,
        worker_id: Optional[str] = None
    ) -> bool:
        """Fails a job or schedules it for retry depending on remaining attempts. Enforces lock ownership when worker_id passed."""
        job = await self.get_job(job_id)
        if not job:
            return False

        if worker_id is not None:
            w_id = worker_id
            err_code = error_code_or_worker
            err_msg = error_message_or_code
            delay = retry_delay_or_message if isinstance(retry_delay_or_message, (int, float)) else retry_delay_seconds
        elif job.get("locked_by") is not None:
            w_id = error_code_or_worker
            err_code = error_message_or_code
            err_msg = str(retry_delay_or_message) if retry_delay_or_message is not None else ""
            delay = retry_delay_seconds
        else:
            w_id = None
            err_code = error_code_or_worker
            err_msg = error_message_or_code
            delay = retry_delay_or_message if isinstance(retry_delay_or_message, (int, float)) else retry_delay_seconds

        if w_id is not None and job.get("locked_by") != w_id:
            return False

        if delay and job["attempt_count"] < job["max_attempts"]:
            if w_id:
                query = """
                UPDATE job_queue
                SET status = 'retry_wait', error_code = $3, error_message = $4,
                    next_retry_at = NOW() + ($5 || ' seconds')::interval,
                    locked_by = NULL, locked_at = NULL, updated_at = NOW()
                WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
                """
                result = await self.conn.execute(query, job_id, w_id, err_code, err_msg, str(delay))
            else:
                query = """
                UPDATE job_queue
                SET status = 'retry_wait', error_code = $2, error_message = $3,
                    next_retry_at = NOW() + ($4 || ' seconds')::interval,
                    locked_by = NULL, locked_at = NULL, updated_at = NOW()
                WHERE id = $1::uuid;
                """
                result = await self.conn.execute(query, job_id, err_code, err_msg, str(delay))
        else:
            if w_id:
                query = """
                UPDATE job_queue
                SET status = 'failed', error_code = $3, error_message = $4, completed_at = NOW(), updated_at = NOW()
                WHERE id = $1::uuid AND locked_by = $2 AND status IN ('leased', 'running');
                """
                result = await self.conn.execute(query, job_id, w_id, err_code, err_msg)
            else:
                query = """
                UPDATE job_queue
                SET status = 'failed', error_code = $2, error_message = $3, completed_at = NOW(), updated_at = NOW()
                WHERE id = $1::uuid;
                """
                result = await self.conn.execute(query, job_id, err_code, err_msg)
        return self._affected_rows(result) > 0

    async def recover_stale_jobs(self, stale_threshold_seconds: int = 60) -> int:
        """Recovers jobs whose heartbeat is stale (older than stale_threshold_seconds).
        Resets retryable jobs to 'retry_wait' (next_retry_at = NOW() + 5s) and exhausted jobs to 'failed'.
        """
        query_retry = """
        UPDATE job_queue
        SET status = 'retry_wait',
            locked_by = NULL,
            locked_at = NULL,
            next_retry_at = NOW() + INTERVAL '5 seconds',
            updated_at = NOW()
        WHERE status IN ('leased', 'running')
          AND heartbeat_at < NOW() - ($1 || ' seconds')::interval
          AND attempt_count < max_attempts;
        """
        res1 = await self.conn.execute(query_retry, str(stale_threshold_seconds))
        count1 = self._affected_rows(res1)

        query_fail = """
        UPDATE job_queue
        SET status = 'failed',
            error_code = 'STALE_LEASE_EXHAUSTED',
            error_message = 'Job heartbeat expired and maximum attempts were reached',
            locked_by = NULL,
            locked_at = NULL,
            completed_at = NOW(),
            updated_at = NOW()
        WHERE status IN ('leased', 'running')
          AND heartbeat_at < NOW() - ($1 || ' seconds')::interval
          AND attempt_count >= max_attempts;
        """
        res2 = await self.conn.execute(query_fail, str(stale_threshold_seconds))
        count2 = self._affected_rows(res2)

        return count1 + count2

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a job by ID."""
        row = await self.conn.fetchrow("SELECT * FROM job_queue WHERE id = $1::uuid;", job_id)
        return dict(row) if row else None

    @staticmethod
    def _affected_rows(result: str) -> int:
        try:
            return int(result.split()[-1])
        except Exception:
            return 0
