"""Explicit worker ownership rules for durable jobs.

Bulk corpus embedding is intentionally assigned only to a local-admin worker.
The public Railway process has no bulk job types and fails closed when its mode
is absent or invalid.
"""

from enum import StrEnum


class WorkerMode(StrEnum):
    LOCAL_ADMIN = "local_admin"
    RAILWAY_PUBLIC = "railway_public"


LOCAL_ADMIN_JOB_TYPES = frozenset(
    {
        "test_job",
        "ingestion_job",
        "jsonl_ingest",
        "embed_chunks",
        "embed_questions",
        "corpus_completeness",
        "question_bank_embeddings",
    }
)
# Phase 3D intentionally gives Railway no durable job types.  The bounded
# on-demand query-embedding path belongs to Phase 3E and must register its own
# tested handler before it is added here.
RAILWAY_PUBLIC_JOB_TYPES = frozenset()


class WorkerModeConfigurationError(RuntimeError):
    """Raised when a worker deployment has no explicit ownership configuration."""


def resolve_worker_mode(value: str) -> WorkerMode:
    try:
        return WorkerMode(value)
    except ValueError as exc:
        raise WorkerModeConfigurationError(
            "WORKER_MODE must be explicitly set to 'local_admin' or 'railway_public'."
        ) from exc


def owned_job_types(mode: WorkerMode) -> frozenset[str]:
    if mode is WorkerMode.LOCAL_ADMIN:
        return LOCAL_ADMIN_JOB_TYPES
    if mode is WorkerMode.RAILWAY_PUBLIC:
        return RAILWAY_PUBLIC_JOB_TYPES
    raise WorkerModeConfigurationError(f"Unsupported worker mode: {mode!s}")
