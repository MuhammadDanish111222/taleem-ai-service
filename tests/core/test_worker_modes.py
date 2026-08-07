import pytest

from app.core.config import get_settings
from app.core.worker_modes import (
    WorkerMode,
    WorkerModeConfigurationError,
    owned_job_types,
    resolve_worker_mode,
)
from app.workers.main import Worker


def test_worker_mode_must_be_explicit():
    with pytest.raises(WorkerModeConfigurationError):
        resolve_worker_mode("")


@pytest.mark.asyncio
async def test_unconfigured_worker_fails_before_connecting_to_database(monkeypatch):
    monkeypatch.setattr(get_settings(), "WORKER_MODE", "")
    with pytest.raises(WorkerModeConfigurationError):
        await Worker().run("postgresql://not-used")


@pytest.mark.asyncio
async def test_public_and_local_workers_reject_each_others_job_types():
    assert owned_job_types(WorkerMode.RAILWAY_PUBLIC) == frozenset(
        {"multiple_ask_validate", "multiple_ask_extract", "multiple_ask_answer"}
    )
    assert "multiple_ask_validate" not in owned_job_types(WorkerMode.LOCAL_ADMIN)
    with pytest.raises(ValueError, match="outside its ownership"):
        await Worker(
            worker_mode="railway_public", supported_types=["embed_chunks"]
        ).run("postgresql://not-used")
    with pytest.raises(ValueError, match="outside its ownership"):
        await Worker(
            worker_mode="local_admin", supported_types=["query_embedding"]
        ).run("postgresql://not-used")


@pytest.mark.asyncio
async def test_railway_worker_runs_multiple_ask_retention_cleanup(monkeypatch):
    calls = []

    class Acquire:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    class Pool:
        def acquire(self):
            return Acquire()

    class Cleanup:
        def __init__(self, _conn):
            pass

        async def cleanup_once(self, *, run_id):
            calls.append(run_id)
            worker.running = False
            return {"unfinalized": 0, "raw_sources": 0, "jobs": 0, "failed": 0}

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr("app.workers.main.MultipleAskRetentionService", Cleanup)
    monkeypatch.setattr("app.workers.main.asyncio.sleep", no_wait)
    worker = Worker(worker_mode="railway_public")
    worker.running = True
    await worker._multiple_ask_cleanup_loop(Pool())
    assert len(calls) == 1
