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
    assert owned_job_types(WorkerMode.RAILWAY_PUBLIC) == frozenset()
    with pytest.raises(ValueError, match="outside its ownership"):
        await Worker(
            worker_mode="railway_public", supported_types=["embed_chunks"]
        ).run("postgresql://not-used")
    with pytest.raises(ValueError, match="outside its ownership"):
        await Worker(
            worker_mode="local_admin", supported_types=["query_embedding"]
        ).run("postgresql://not-used")
