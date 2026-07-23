"""Tests for Worker Runtime: Concurrency, Lock Ownership, Stale Recovery, Crash Idempotency, and Unsupported Job Handling."""

import pytest
import asyncio
import asyncpg
from app.repositories.job_repository import JobRepository
from app.services.jobs.queue import JobQueueService
from app.workers.main import Worker, register_handler, HANDLERS

DB_URL = "postgresql://postgres:postgres@localhost:5432/taleem_dev"

@pytest.fixture
async def db_conn():
    connection = await asyncpg.connect(DB_URL)
    await connection.execute("SET search_path = public, pg_catalog;")
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()

@pytest.mark.asyncio
async def test_concurrent_lease_with_independent_connections():
    """Point 8: Real concurrency test using two independent PostgreSQL connections and SKIP LOCKED."""
    conn1 = await asyncpg.connect(DB_URL)
    conn2 = await asyncpg.connect(DB_URL)
    try:
        repo1 = JobRepository(conn1)
        repo2 = JobRepository(conn2)

        # Enqueue a single job
        job = await repo1.create_job("concurrent_job", {"data": 123}, idempotency_key="idemp_concurrent_1")
        assert job["status"] == "queued"

        # Concurrently lease from both connections
        res1, res2 = await asyncio.gather(
            repo1.lease_job("worker_1", ["concurrent_job"]),
            repo2.lease_job("worker_2", ["concurrent_job"])
        )

        leased_results = [r for r in (res1, res2) if r is not None]
        assert len(leased_results) == 1, "Exactly one worker must acquire lease under FOR UPDATE SKIP LOCKED"
        
        winner_worker = leased_results[0]["locked_by"]
        assert winner_worker in ("worker_1", "worker_2")
    finally:
        # Cleanup test job
        await conn1.execute("DELETE FROM job_queue WHERE idempotency_key = 'idemp_concurrent_1';")
        await conn1.close()
        await conn2.close()

@pytest.mark.asyncio
async def test_lock_ownership_enforcement(db_conn):
    """Point 4: Verify worker cannot mutate jobs locked by another worker."""
    repo = JobRepository(db_conn)
    job = await repo.create_job("ownership_test_job", {"data": 1})
    leased = await repo.lease_job("worker_owner", ["ownership_test_job"])
    job_id = str(leased["id"])

    # Attempt mutations from worker_intruder
    assert await repo.update_heartbeat(job_id, "worker_intruder") is False
    assert await repo.update_progress(job_id, "worker_intruder", "stage", 50.0) is False
    assert await repo.complete_job(job_id, "worker_intruder") is False
    assert await repo.fail_job(job_id, "worker_intruder", "ERR", "msg") is False

    # Verify original owner succeeds
    assert await repo.update_heartbeat(job_id, "worker_owner") is True
    assert await repo.complete_job(job_id, "worker_owner") is True

@pytest.mark.asyncio
async def test_worker_crash_stale_recovery_and_idempotency(db_conn):
    """Points 7 & 8: Simulates Worker A crash mid-task, stale recovery, Worker B pickup, and single idempotent output."""
    service = JobQueueService(db_conn)
    idemp_key = "idemp_crash_test_999"

    # Enqueue job
    job = await service.enqueue_job("crash_test_job", {"target_id": "target_res_123"}, idempotency_key=idemp_key)
    job_id = str(job["id"])

    # 1. Worker A leases and produces side-effect, then crashes
    worker_a_lease = await service.lease_next_job("worker_A", ["crash_test_job"])
    assert worker_a_lease is not None
    assert worker_a_lease["attempt_count"] == 1

    # Worker A produces side effect in admin_audit_logs (using idempotency check)
    audit_exists = await db_conn.fetchval(
        "SELECT COUNT(*) FROM admin_audit_logs WHERE target_id = $1;", idemp_key
    )
    if audit_exists == 0:
        await db_conn.execute(
            """
            INSERT INTO admin_audit_logs (actor_id, action, target_type, target_id)
            VALUES ($1, $2, $3, $4);
            """,
            "worker_A", "process_crash_test", "resource", idemp_key
        )

    # Worker A crashes (heartbeat not updated, status left in leased, heartbeat_at set in past)
    await db_conn.execute(
        "UPDATE job_queue SET heartbeat_at = NOW() - INTERVAL '120 seconds' WHERE id = $1::uuid;",
        job_id
    )

    # 2. Stale lease recovery triggered
    recovered_count = await service.recover_stale_jobs(stale_threshold_seconds=60)
    assert recovered_count == 1

    recovered_job = await service.get_job(job_id)
    assert recovered_job["status"] == "retry_wait"
    assert recovered_job["locked_by"] is None

    # Manually reset next_retry_at to NOW() for immediate test pickup
    await db_conn.execute("UPDATE job_queue SET next_retry_at = NOW() WHERE id = $1::uuid;", job_id)

    # 3. Worker B leases the recovered job
    worker_b_lease = await service.lease_next_job("worker_B", ["crash_test_job"])
    assert worker_b_lease is not None
    assert worker_b_lease["attempt_count"] == 2

    # Worker B executes idempotent handler logic (check before write / ON CONFLICT)
    audit_exists_b = await db_conn.fetchval(
        "SELECT COUNT(*) FROM admin_audit_logs WHERE target_id = $1;", idemp_key
    )
    if audit_exists_b == 0:
        await db_conn.execute(
            """
            INSERT INTO admin_audit_logs (actor_id, action, target_type, target_id)
            VALUES ($1, $2, $3, $4);
            """,
            "worker_B", "process_crash_test", "resource", idemp_key
        )

    await service.complete_job(job_id, "worker_B")

    # 4. Assert final status and idempotent side-effect proof
    final_job = await service.get_job(job_id)
    assert final_job["status"] == "succeeded"
    assert final_job["attempt_count"] == 2

    # Assert exactly ONE logical output record exists
    output_count = await db_conn.fetchval(
        "SELECT COUNT(*) FROM admin_audit_logs WHERE target_id = $1;", idemp_key
    )
    assert output_count == 1, "Side-effect must be strictly idempotent with exactly 1 output record"

@pytest.mark.asyncio
async def test_unsupported_job_type_handling(db_conn):
    """Point 10: Verify worker immediately fails unsupported job types without infinite retries."""
    worker = Worker(worker_id="test_worker_unsupported", supported_types=["unsupported_type_xyz"])
    service = JobQueueService(db_conn)

    job = await service.enqueue_job("unsupported_type_xyz", {"payload": 123})
    job_id = str(job["id"])

    # Simulate worker processing
    leased = await service.lease_next_job(worker.worker_id, ["unsupported_type_xyz"])
    assert leased is not None

    # Process job with worker runtime
    pool_mock = type('PoolMock', (), {
        'acquire': lambda self_inner: type('AcquireCtx', (), {
            '__aenter__': lambda ctx_inner: asyncio.sleep(0, result=db_conn),
            '__aexit__': lambda ctx_inner, *args: asyncio.sleep(0)
        })()
    })()

    await worker._process_job(pool_mock, leased)

    # Assert job is terminally failed with UNSUPPORTED_JOB_TYPE
    failed_job = await service.get_job(job_id)
    assert failed_job["status"] == "failed"
    assert failed_job["error_code"] == "UNSUPPORTED_JOB_TYPE"
    assert failed_job["attempt_count"] == 1
