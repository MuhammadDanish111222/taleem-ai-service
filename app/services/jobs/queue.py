"""Durable Job Queue Service wrapping JobRepository."""

from typing import Optional, Dict, Any, List
import asyncpg
from app.repositories.job_repository import JobRepository

class JobQueueService:
    def __init__(self, conn: asyncpg.Connection):
        self.repo = JobRepository(conn)

    async def enqueue_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        return await self.repo.create_job(job_type, payload, idempotency_key, max_attempts)

    async def lease_next_job(
        self,
        worker_id: str,
        supported_types: List[str],
        lease_duration_seconds: int = 300
    ) -> Optional[Dict[str, Any]]:
        return await self.repo.lease_job(worker_id, supported_types, lease_duration_seconds)

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        return await self.repo.update_heartbeat(job_id, worker_id)

    async def update_progress(self, job_id: str, worker_id: str, stage: str, progress: float) -> bool:
        return await self.repo.update_progress(job_id, worker_id, stage, progress)

    async def complete_job(self, job_id: str, worker_id: str, final_progress: float = 100.0) -> bool:
        return await self.repo.complete_job(job_id, worker_id, final_progress)

    async def fail_job(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_delay_seconds: Optional[int] = None
    ) -> bool:
        return await self.repo.fail_job(job_id, worker_id, error_code, error_message, retry_delay_seconds)

    async def recover_stale_jobs(self, stale_threshold_seconds: int = 60) -> int:
        return await self.repo.recover_stale_jobs(stale_threshold_seconds)

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return await self.repo.get_job(job_id)
