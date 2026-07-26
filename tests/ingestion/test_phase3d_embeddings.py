"""Focused Phase 3D tests; the fake provider never loads or downloads BGE."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import asyncpg
import pytest

from app.core.worker_modes import WorkerMode, owned_job_types
from app.providers.embeddings.bge import (
    BGEEmbeddingConfiguration,
    embedding_input_hash,
    format_chunk_embedding_input,
)
from app.repositories.job_repository import JobRepository
from app.repositories.rag_repository import RagRepository
from app.services.ingestion.embed_chunks import embed_chunks
from app.services.ingestion.embed_questions import embed_questions


@dataclass
class FakeEmbeddingProvider:
    configuration: BGEEmbeddingConfiguration = BGEEmbeddingConfiguration()

    def __post_init__(self):
        self.calls: list[list[str]] = []

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            value = (
                int(hashlib.sha256(text.encode()).hexdigest()[:4], 16) % 1000
            ) / 1000
            vectors.append([value] * 768)
        return vectors


@pytest.fixture
async def conn():
    try:
        connection = await asyncpg.connect(
            os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/taleem_dev",
            )
        )
    except (ConnectionRefusedError, OSError):
        pytest.skip("Supported disposable PostgreSQL database is unavailable.")
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


async def _building_corpus(conn, scope: str = "phase3d"):
    provider = FakeEmbeddingProvider()
    repo = RagRepository(conn)
    version = await repo.get_or_create_building_corpus_version(
        f"board-{scope}",
        "class-9",
        "physics",
        provider.configuration.model,
        provider.configuration.revision,
        768,
        provider.configuration_fingerprint,
        provider.configuration.normalize,
        provider.configuration.query_instruction,
    )
    document = await repo.create_document_version(
        str(version["id"]),
        f"resource-{scope}",
        "v1",
        "admin_jsonl_v1",
        "Phase 3D",
    )
    chunks = [
        {
            "chunk_order": 0,
            "chunk_text": "Mass measures resistance to acceleration.",
            "chapter_id": "chapter-1",
            "topic_no": "1.1",
            "topic_title": "Mass",
            "content_type": "explanation",
            "content_hash": "a" * 64,
            "expected_questions": ["What is mass?", "How is mass measured?"],
        }
    ]
    await repo.replace_chapter_chunks(str(version["id"]), str(document["id"]), chunks)
    return repo, provider, str(version["id"])


def test_chunk_embedding_input_only_uses_allowed_retrieval_text():
    text = format_chunk_embedding_input(
        topic_no="2.4",
        topic_title="Newton's laws",
        chunk_text="Force equals mass times acceleration.",
        approved_visual={
            "is_linked": True,
            "is_approved": True,
            "title": "Force diagram",
            "description": "A cart with an applied force arrow.",
            "id": "9b6d5cb3-1234-5678-9abc-1f0b0c0d0e0f",
            "storage_path": "private/drive-id/secret-key.png",
        },
    )
    assert "Topic 2.4" in text
    assert "Newton's laws" in text
    assert "cart with an applied force" in text
    assert "9b6d5cb3" not in text
    assert "drive-id" not in text
    assert "secret-key" not in text


def test_chunk_embedding_visuals_are_ordered_by_logical_id_without_embedding_ids():
    text = format_chunk_embedding_input(
        topic_no="2.4",
        topic_title="Forces",
        chunk_text="A force changes motion.",
        approved_visuals=[
            {
                "visual_id": "z-force",
                "title": "Zebra force",
                "description": "Second visual",
            },
            {
                "visual_id": "a-force",
                "title": "Arrow force",
                "description": "First visual",
            },
        ],
    )
    assert text.index("Arrow force") < text.index("Zebra force")
    assert "a-force" not in text
    assert "z-force" not in text


@pytest.mark.asyncio
async def test_each_expected_question_receives_its_own_vector(conn):
    repo, provider, corpus_version_id = await _building_corpus(conn, "questions")
    result = await embed_questions(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    assert result == {"embedded": 2, "expected": 2}
    assert len(provider.calls) == 1
    assert sorted(provider.calls[0]) == [
        "How is mass measured?",
        "What is mass?",
    ]
    rows = await conn.fetch(
        "SELECT question_text, embedding_input_hash, embedding_status FROM chunk_expected_questions ORDER BY question_text;"
    )
    assert len(rows) == 2
    assert all(row["embedding_status"] == "embedded" for row in rows)
    assert {row["embedding_input_hash"] for row in rows} == {
        embedding_input_hash("What is mass?"),
        embedding_input_hash("How is mass measured?"),
    }


@pytest.mark.asyncio
async def test_restart_skips_matching_chunk_vectors_without_duplicate_rows(conn):
    repo, provider, corpus_version_id = await _building_corpus(conn, "restart")
    first = await embed_chunks(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    second = await embed_chunks(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    assert first["embedded"] == 1
    assert second["embedded"] == 0
    assert len(provider.calls) == 1
    assert await conn.fetchval("SELECT COUNT(*) FROM rag_chunks;") == 1


@pytest.mark.asyncio
async def test_changed_configuration_requires_a_new_building_version(conn):
    repo, provider, corpus_version_id = await _building_corpus(conn, "config")
    await embed_chunks(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    await embed_questions(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    await repo.refresh_embedding_counts(corpus_version_id)
    assert (await repo.mark_qa_ready(corpus_version_id))["ready"] is True

    changed = BGEEmbeddingConfiguration(query_instruction="Changed instruction: ")
    next_version = await repo.get_or_create_building_corpus_version(
        "board-config",
        "class-9",
        "physics",
        changed.model,
        changed.revision,
        768,
        changed.fingerprint(),
        changed.normalize,
        changed.query_instruction,
    )
    assert str(next_version["id"]) != corpus_version_id
    assert next_version["status"] == "building"
    assert next_version["embedding_config_fingerprint"] == changed.fingerprint()


@pytest.mark.asyncio
async def test_worker_modes_cannot_claim_each_others_jobs(conn):
    jobs = JobRepository(conn)
    local_job = await jobs.create_job("embed_chunks", {"corpus_version_id": "scope"})
    public_lease = await jobs.lease_job(
        "railway", sorted(owned_job_types(WorkerMode.RAILWAY_PUBLIC))
    )
    assert public_lease is None
    assert (await jobs.get_job(str(local_job["id"])))["status"] == "queued"
    local_lease = await jobs.lease_job(
        "local", sorted(owned_job_types(WorkerMode.LOCAL_ADMIN))
    )
    assert local_lease["job_type"] == "embed_chunks"


@pytest.mark.asyncio
async def test_embedding_stage_dependencies_are_completed_and_enqueued_atomically(conn):
    jobs = JobRepository(conn)
    payload = {
        "corpus_version_id": "scope",
        "embedding_config_fingerprint": "config",
        "embedding_input_fingerprint": "input",
    }
    chunk = await jobs.create_job(
        "embed_chunks", payload, idempotency_key="phase3d-test-chunks"
    )
    assert (
        await conn.fetchval(
            "SELECT COUNT(*) FROM job_queue WHERE job_type = 'embed_questions'"
        )
        == 0
    )
    leased_chunk = await jobs.lease_job("local", ["embed_chunks"])
    assert str(leased_chunk["id"]) == str(chunk["id"])

    assert await jobs.complete_job_and_enqueue(
        str(chunk["id"]),
        "local",
        {
            "job_type": "embed_questions",
            "payload": payload,
            "idempotency_key": "phase3d-test-questions",
        },
    )
    assert (await jobs.get_job(str(chunk["id"])))["status"] == "succeeded"
    question = await jobs.lease_job("local", ["embed_questions"])
    assert question is not None
    assert (
        await conn.fetchval(
            "SELECT COUNT(*) FROM job_queue WHERE job_type = 'corpus_completeness'"
        )
        == 0
    )

    assert await jobs.complete_job_and_enqueue(
        str(question["id"]),
        "local",
        {
            "job_type": "corpus_completeness",
            "payload": payload,
            "idempotency_key": "phase3d-test-completeness",
        },
    )
    readiness = await jobs.lease_job("local", ["corpus_completeness"])
    assert readiness is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("rag_chunks", "embedding", None),
        ("rag_chunks", "embedding_status", "failed"),
        ("rag_chunks", "embedding_status", "pending"),
        ("rag_chunks", "embedding_config_fingerprint", "stale"),
        ("chunk_expected_questions", "embedding", None),
        ("chunk_expected_questions", "embedding_status", "failed"),
        ("chunk_expected_questions", "embedding_status", "pending"),
        ("chunk_expected_questions", "embedding_config_fingerprint", "stale"),
    ],
)
async def test_qa_ready_is_blocked_for_invalid_chunk_or_question_vectors(
    conn, table, column, value
):
    repo, provider, corpus_version_id = await _building_corpus(
        conn, f"blocked-{table}-{column}-{value}"
    )
    await embed_chunks(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    await embed_questions(
        corpus_version_id, provider.configuration_fingerprint, conn, provider
    )
    await repo.refresh_embedding_counts(corpus_version_id)
    await conn.execute(f"UPDATE {table} SET {column} = $1", value)
    report = await repo.mark_qa_ready(corpus_version_id)
    assert report["ready"] is False
    assert report["reasons"]


def test_wrong_dimension_vectors_are_rejected_before_database_write():
    with pytest.raises(ValueError, match="exactly 768"):
        RagRepository._validate_vector([0.0] * 767)
