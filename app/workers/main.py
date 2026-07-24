"""Standalone Worker Runtime Process for Taleem AI Service.

Execution command:
    python -m app.workers.main
    (or: uv run python -m app.workers.main)
"""

import asyncio
import logging
import signal
import sys
import uuid
from typing import Dict, Callable, Any, Optional
import asyncpg

from app.core.config import get_settings
from app.services.jobs.queue import JobQueueService

from app.workers.handlers.jsonl_ingest import handle_jsonl_ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("worker_main")

# Handler registry for job types
HANDLERS: Dict[str, Callable[[Dict[str, Any], asyncpg.Connection], Any]] = {}

def register_handler(job_type: str, handler_func: Callable[[Dict[str, Any], asyncpg.Connection], Any]):
    """Registers a handler callable for a specific job_type."""
    HANDLERS[job_type] = handler_func

class Worker:
    def __init__(
        self,
        worker_id: Optional[str] = None,
        supported_types: Optional[list] = None,
        poll_interval: float = 2.0,
        heartbeat_interval: float = 5.0,
        stale_check_interval: float = 15.0,
        stale_threshold_seconds: int = 60
    ):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.supported_types = supported_types or ["test_job", "ingestion_job", "jsonl_ingest"]
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.stale_check_interval = stale_check_interval
        self.stale_threshold_seconds = stale_threshold_seconds
        self.running = False
        self.active_task: Optional[asyncio.Task] = None

    async def run(self, db_url: Optional[str] = None):
        settings = get_settings()
        url = db_url or settings.DATABASE_URL
        
        logger.info(f"Worker '{self.worker_id}' starting with supported job types: {self.supported_types}")
        
        pool = await asyncpg.create_pool(url, min_size=2, max_size=5)
        self.running = True
        
        # Setup signal handling for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                # Windows signal handler fallback
                pass

        stale_recovery_task = asyncio.create_task(self._stale_recovery_loop(pool))

        try:
            while self.running:
                async with pool.acquire() as conn:
                    service = JobQueueService(conn)
                    job = await service.lease_next_job(self.worker_id, self.supported_types)

                    if job:
                        logger.info(f"Worker '{self.worker_id}' leased job '{job['id']}' of type '{job['job_type']}'")
                        self.active_task = asyncio.create_task(self._process_job(pool, job))
                        await self.active_task
                        self.active_task = None
                    else:
                        await asyncio.sleep(self.poll_interval)
        finally:
            stale_recovery_task.cancel()
            await pool.close()
            logger.info(f"Worker '{self.worker_id}' stopped cleanly.")

    async def shutdown(self):
        logger.info(f"Worker '{self.worker_id}' received shutdown signal. Stopping job leasing...")
        self.running = False

    async def _stale_recovery_loop(self, pool: asyncpg.Pool):
        while self.running:
            try:
                await asyncio.sleep(self.stale_check_interval)
                async with pool.acquire() as conn:
                    service = JobQueueService(conn)
                    recovered = await service.recover_stale_jobs(self.stale_threshold_seconds)
                    if recovered > 0:
                        logger.warning(f"Recovered {recovered} stale job(s) in worker background loop.")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stale recovery loop: {e}")

    async def _process_job(self, pool: asyncpg.Pool, job: Dict[str, Any]):
        job_id = str(job["id"])
        job_type = job["job_type"]

        # Check for unsupported job type (Point 9 requirement)
        if job_type not in HANDLERS:
            logger.error(f"Unsupported job type '{job_type}' for job '{job_id}'. Marking terminally failed.")
            async with pool.acquire() as conn:
                service = JobQueueService(conn)
                await service.fail_job(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    error_code="UNSUPPORTED_JOB_TYPE",
                    error_message=f"No handler registered for job type '{job_type}'"
                )
            return

        handler = HANDLERS[job_type]
        heartbeat_running = True

        async def heartbeat_loop():
            while heartbeat_running and self.running:
                await asyncio.sleep(self.heartbeat_interval)
                try:
                    async with pool.acquire() as conn:
                        service = JobQueueService(conn)
                        hb_ok = await service.heartbeat(job_id, self.worker_id)
                        if not hb_ok:
                            logger.warning(f"Heartbeat failed for job '{job_id}'; worker may have lost lease ownership.")
                except Exception as hb_err:
                    logger.error(f"Error sending heartbeat for job '{job_id}': {hb_err}")

        hb_task = asyncio.create_task(heartbeat_loop())

        try:
            async with pool.acquire() as conn:
                service = JobQueueService(conn)
                await service.update_progress(job_id, self.worker_id, "processing", 10.0)

            # Execute actual handler logic
            async with pool.acquire() as conn:
                await handler(job, conn)

            async with pool.acquire() as conn:
                service = JobQueueService(conn)
                await service.complete_job(job_id, self.worker_id, 100.0)
            logger.info(f"Job '{job_id}' completed successfully.")
        except Exception as err:
            logger.error(f"Error processing job '{job_id}': {err}")
            async with pool.acquire() as conn:
                service = JobQueueService(conn)
                await service.fail_job(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    error_code="HANDLER_ERROR",
                    error_message=str(err),
                    retry_delay_seconds=5
                )
        finally:
            heartbeat_running = False
            hb_task.cancel()

# Sample default handler for testing
async def dummy_test_handler(job: Dict[str, Any], conn: asyncpg.Connection):
    logger.info(f"Executing dummy test handler for job {job.get('id')}")
    await asyncio.sleep(0.1)

register_handler("test_job", dummy_test_handler)
register_handler("jsonl_ingest", handle_jsonl_ingest)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        print("WORKER_SMOKE_TEST_SUCCESS", flush=True)
        sys.exit(0)
    worker = Worker()
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker process terminated.")

