"""Integration tests for Admin JSONL Chunk Ingestion Job Handler & Repository.

Environment: Verified against real PostgreSQL test database using asyncpg connection pool.
"""

import os
import uuid
import asyncio
import pytest
import asyncpg

from unittest.mock import patch
from app.workers.handlers.jsonl_ingest import handle_jsonl_ingest
from app.repositories.rag_repository import RagRepository

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/taleem_dev")


@pytest.fixture(autouse=True)
def mock_firestore_hierarchy_check():
    """Mocks Firestore hierarchy check to return True for test UUIDs during DB integration tests."""
    with patch("app.services.ingestion.jsonl_chunks.check_firestore_hierarchy", return_value=True):
        yield


@pytest.fixture
async def conn():
    """Acquires a real asyncpg connection for testing within an isolated transaction block."""
    connection = await asyncpg.connect(DB_URL)
    tx = connection.transaction()
    await tx.start()
    yield connection
    await tx.rollback()
    await connection.close()


@pytest.mark.asyncio
async def test_valid_jsonl_ingestion_job_execution(conn):
    """Executes jsonl_ingest job, verifying rag_chunks and chunk_expected_questions storage."""
    board_id = f"b_{uuid.uuid4().hex[:6]}"
    class_id = f"c_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_{uuid.uuid4().hex[:6]}"
    chapter_id = f"ch_{uuid.uuid4().hex[:6]}"

    raw_jsonl = (
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{chapter_id}",'
        f'"topic_no":"1.1","topic_title":"Introduction","chunk_order":0,"content_type":"explanation",'
        f'"chunk_text":"Physics studies nature.","expected_questions":["What does physics study?"],"page_range":[1,2]}}\n'
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{chapter_id}",'
        f'"topic_no":"1.2","topic_title":"Units","chunk_order":1,"content_type":"definition",'
        f'"chunk_text":"Length is measured in meters.","expected_questions":["Define meter."],"page_range":[3,4]}}'
    )

    job = {
        "id": str(uuid.uuid4()),
        "job_type": "jsonl_ingest",
        "payload": {"jsonl_content": raw_jsonl, "resource_version_id": "v1"},
    }

    result = await handle_jsonl_ingest(job, conn)

    assert result["status"] == "succeeded"
    assert result["chunks_ingested"] == 2

    corpus_version_id = result["corpus_version_id"]
    document_version_id = result["document_version_id"]

    # 1. Assert corpus version status remains 'building'
    cv_row = await conn.fetchrow(
        "SELECT status, expected_chunk_count, embedded_chunk_count, embedding_model FROM rag_corpus_versions WHERE id = $1::uuid;",
        corpus_version_id,
    )
    assert cv_row["status"] == "building"
    assert cv_row["expected_chunk_count"] == 2
    assert cv_row["embedded_chunk_count"] == 0
    assert cv_row["embedding_model"] == "BAAI/bge-base-en-v1.5"

    # 2. Assert rag_chunks rows stored with exact field mapping
    chunks = await conn.fetch(
        "SELECT * FROM rag_chunks WHERE document_version_id = $1::uuid ORDER BY chunk_index ASC;",
        document_version_id,
    )
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["content"] == "Physics studies nature."
    assert chunks[0]["chapter_id"] == chapter_id
    assert chunks[0]["content_type"] == "explanation"
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2

    # 3. Assert chunk_expected_questions stored with NULL embedding
    eq0 = await conn.fetch(
        "SELECT * FROM chunk_expected_questions WHERE chunk_id = $1::uuid;",
        chunks[0]["id"],
    )
    assert len(eq0) == 1
    assert eq0[0]["question_text"] == "What does physics study?"
    assert eq0[0]["embedding"] is None


@pytest.mark.asyncio
async def test_chapter_reupload_atomic_replace(conn):
    """Re-uploading a chapter replaces prior chunks and reconciles expected_chunk_count."""
    board_id = f"b_{uuid.uuid4().hex[:6]}"
    class_id = f"c_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_{uuid.uuid4().hex[:6]}"
    chapter_id = f"ch_{uuid.uuid4().hex[:6]}"

    # Batch 1: 3 chunks
    raw_jsonl_1 = "\n".join([
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{chapter_id}",'
        f'"topic_no":"1.{i}","topic_title":"Topic {i}","chunk_order":{i},"content_type":"explanation",'
        f'"chunk_text":"Content {i}","expected_questions":["Q {i}"],"page_range":[{i+1},{i+2}]}}'
        for i in range(3)
    ])

    job1 = {"id": str(uuid.uuid4()), "job_type": "jsonl_ingest", "payload": {"jsonl_content": raw_jsonl_1}}
    res1 = await handle_jsonl_ingest(job1, conn)
    assert res1["chunks_ingested"] == 3

    corpus_version_id = res1["corpus_version_id"]
    cv1 = await conn.fetchrow("SELECT expected_chunk_count FROM rag_corpus_versions WHERE id = $1::uuid;", corpus_version_id)
    assert cv1["expected_chunk_count"] == 3

    # Batch 2: Re-upload with 1 chunk
    raw_jsonl_2 = (
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{chapter_id}",'
        f'"topic_no":"1.0","topic_title":"Updated Topic","chunk_order":0,"content_type":"summary",'
        f'"chunk_text":"Updated single content","expected_questions":["New Q"],"page_range":[1,5]}}'
    )

    job2 = {"id": str(uuid.uuid4()), "job_type": "jsonl_ingest", "payload": {"jsonl_content": raw_jsonl_2}}
    res2 = await handle_jsonl_ingest(job2, conn)
    assert res2["chunks_ingested"] == 1

    # Assert expected_chunk_count updated by delta (3 - 2 = 1)
    cv2 = await conn.fetchrow("SELECT expected_chunk_count FROM rag_corpus_versions WHERE id = $1::uuid;", corpus_version_id)
    assert cv2["expected_chunk_count"] == 1

    # Assert old chunks deleted and replaced
    chunks = await conn.fetch("SELECT * FROM rag_chunks WHERE corpus_version_id = $1::uuid;", corpus_version_id)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Updated single content"


@pytest.mark.asyncio
async def test_multi_chapter_corpus_accumulation(conn):
    """Uploading Chapter 1 and Chapter 2 for same subject scope accumulates into single building corpus version."""
    board_id = f"b_{uuid.uuid4().hex[:6]}"
    class_id = f"c_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_{uuid.uuid4().hex[:6]}"

    ch1_id = f"ch_1_{uuid.uuid4().hex[:4]}"
    ch2_id = f"ch_2_{uuid.uuid4().hex[:4]}"

    jsonl_ch1 = (
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{ch1_id}",'
        f'"topic_no":"1.1","topic_title":"Ch1","chunk_order":0,"content_type":"explanation","chunk_text":"Ch1 text"}}'
    )
    jsonl_ch2 = (
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{ch2_id}",'
        f'"topic_no":"2.1","topic_title":"Ch2","chunk_order":0,"content_type":"explanation","chunk_text":"Ch2 text"}}'
    )

    job1 = {"id": str(uuid.uuid4()), "job_type": "jsonl_ingest", "payload": {"jsonl_content": jsonl_ch1}}
    res1 = await handle_jsonl_ingest(job1, conn)

    job2 = {"id": str(uuid.uuid4()), "job_type": "jsonl_ingest", "payload": {"jsonl_content": jsonl_ch2}}
    res2 = await handle_jsonl_ingest(job2, conn)

    # Assert both chapters share the exact same building corpus version
    assert res1["corpus_version_id"] == res2["corpus_version_id"]

    cv = await conn.fetchrow("SELECT expected_chunk_count FROM rag_corpus_versions WHERE id = $1::uuid;", res1["corpus_version_id"])
    assert cv["expected_chunk_count"] == 2


@pytest.mark.asyncio
async def test_non_building_status_lock_rejection(conn):
    """Worker aborts transaction if corpus version is not in 'building' status."""
    repo = RagRepository(conn)
    board_id = f"b_{uuid.uuid4().hex[:6]}"
    class_id = f"c_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_{uuid.uuid4().hex[:6]}"

    cv = await repo.get_or_create_building_corpus_version(board_id, class_id, subject_id)
    cv_id = str(cv["id"])

    # Simulate activation: mark status active
    await conn.execute("UPDATE rag_corpus_versions SET status = 'active' WHERE id = $1::uuid;", cv_id)

    chunks = [{
        "board_id": board_id, "class_id": class_id, "subject_id": subject_id, "chapter_id": "ch_1",
        "topic_no": "1", "topic_title": "T", "chunk_order": 0, "content_type": "explanation",
        "chunk_text": "text", "expected_questions": [], "content_hash": "hash", "token_count": 1
    }]

    with pytest.raises(RuntimeError, match="status is 'active', expected 'building'"):
        await repo.replace_chapter_chunks(cv_id, str(uuid.uuid4()), chunks)


@pytest.mark.asyncio
async def test_first_ingestion_concurrent_corpus_creation_race():
    """Simulates concurrent jobs creating the first corpus version ever for a brand-new scope.

    Uses two real asyncpg connections (conn1, conn2). conn1 holds parent rag_corpora FOR UPDATE
    lock while conn2 attempts concurrent get_or_create_building_corpus_version, verifying
    ON CONFLICT DO UPDATE + FOR UPDATE blocking behavior and single version creation.
    """
    conn1 = await asyncpg.connect(DB_URL)
    conn2 = await asyncpg.connect(DB_URL)

    board_id = f"b_race_{uuid.uuid4().hex[:6]}"
    class_id = f"c_race_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_race_{uuid.uuid4().hex[:6]}"

    repo1 = RagRepository(conn1)
    repo2 = RagRepository(conn2)

    tx1 = conn1.transaction()
    await tx1.start()

    # conn1 acquires corpora row and building version
    cv1 = await repo1.get_or_create_building_corpus_version(board_id, class_id, subject_id)

    # conn2 attempts get_or_create_building_corpus_version in background task
    async def conn2_task():
        tx2 = conn2.transaction()
        await tx2.start()
        try:
            res = await repo2.get_or_create_building_corpus_version(board_id, class_id, subject_id)
            await tx2.commit()
            return res
        except Exception as e:
            await tx2.rollback()
            raise e

    task2 = asyncio.create_task(conn2_task())
    await asyncio.sleep(0.1)

    # Assert task2 is blocked waiting for conn1 lock release
    assert not task2.done()

    # conn1 commits transaction
    await tx1.commit()

    # task2 unblocks and completes
    cv2 = await task2

    # Assert conn2 reused conn1's building version (same version ID)
    assert cv1["id"] == cv2["id"]

    # Verify database total version count for this corpus is 1
    total_versions = await conn1.fetchval(
        "SELECT COUNT(*) FROM rag_corpus_versions WHERE corpus_id = $1::uuid;", cv1["corpus_id"]
    )
    assert total_versions == 1

    await conn1.close()
    await conn2.close()


@pytest.mark.asyncio
async def test_jsonl_ingest_idempotency_key_replay(conn):
    """Enqueuing and processing a job with an identical idempotency key returns existing job without creating duplicate rows."""
    from app.services.jobs.queue import JobQueueService

    board_id = f"b_idemp_{uuid.uuid4().hex[:6]}"
    class_id = f"c_idemp_{uuid.uuid4().hex[:6]}"
    subject_id = f"s_idemp_{uuid.uuid4().hex[:6]}"
    chapter_id = f"ch_idemp_{uuid.uuid4().hex[:6]}"
    idempotency_key = f"key_{uuid.uuid4().hex}"

    raw_jsonl = (
        f'{{"board_id":"{board_id}","class_id":"{class_id}","subject_id":"{subject_id}","chapter_id":"{chapter_id}",'
        f'"topic_no":"1.1","topic_title":"Title","chunk_order":0,"content_type":"explanation",'
        f'"chunk_text":"Idempotency test content.","expected_questions":["Idempotent question?"],"page_range":[1,2]}}'
    )
    payload = {"jsonl_content": raw_jsonl, "resource_version_id": "v1"}

    service = JobQueueService(conn)
    # First enqueue, lease, execution, and completion
    job1 = await service.enqueue_job("jsonl_ingest", payload, idempotency_key=idempotency_key)
    leased_job = await service.lease_next_job("worker-1", ["jsonl_ingest"])
    res1 = await handle_jsonl_ingest(leased_job, conn)
    await service.complete_job(str(job1["id"]), "worker-1")

    # Record initial counts
    c_ver_count_1 = await conn.fetchval("SELECT COUNT(*) FROM rag_corpus_versions;")
    d_ver_count_1 = await conn.fetchval("SELECT COUNT(*) FROM rag_document_versions;")
    chunk_count_1 = await conn.fetchval("SELECT COUNT(*) FROM rag_chunks;")
    q_count_1 = await conn.fetchval("SELECT COUNT(*) FROM chunk_expected_questions;")

    # Second enqueue attempt with EXACT SAME idempotency key
    job2 = await service.enqueue_job("jsonl_ingest", payload, idempotency_key=idempotency_key)

    # Assert job2 returned the original job record without creating a new job row
    assert str(job2["id"]) == str(job1["id"])
    assert job2["status"] == "succeeded"

    # Record post-replay counts
    c_ver_count_2 = await conn.fetchval("SELECT COUNT(*) FROM rag_corpus_versions;")
    d_ver_count_2 = await conn.fetchval("SELECT COUNT(*) FROM rag_document_versions;")
    chunk_count_2 = await conn.fetchval("SELECT COUNT(*) FROM rag_chunks;")
    q_count_2 = await conn.fetchval("SELECT COUNT(*) FROM chunk_expected_questions;")

    # Assert ZERO new rows created across all tables
    assert c_ver_count_2 == c_ver_count_1
    assert d_ver_count_2 == d_ver_count_1
    assert chunk_count_2 == chunk_count_1
    assert q_count_2 == q_count_1

