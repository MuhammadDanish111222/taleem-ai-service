"""Integration tests for JobRepository, RagRepository, AIRequestRepository, and AuditRepository."""

import pytest
import asyncpg
from app.repositories.job_repository import JobRepository
from app.repositories.rag_repository import RagRepository
from app.repositories.ai_request_repository import AIRequestRepository
from app.repositories.audit_repository import AuditRepository

DB_URL = "postgresql://postgres:postgres@localhost:5432/taleem_dev"

@pytest.fixture
async def db_conn():
    connection = await asyncpg.connect(DB_URL)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()

@pytest.mark.asyncio
async def test_job_repository_lifecycle(db_conn):
    """Tests job creation, FOR UPDATE SKIP LOCKED leasing, progress, heartbeat, and completion."""
    repo = JobRepository(db_conn)
    
    # 1. Create job
    job = await repo.create_job("ingest_pipeline", {"resource_id": "res_001"}, idempotency_key="idemp_001")
    assert job["job_type"] == "ingest_pipeline"
    assert job["status"] == "queued"
    assert job["progress"] == 0
    
    # 2. Lease job atomically
    leased = await repo.lease_job("worker_a", ["ingest_pipeline"])
    assert leased is not None
    assert leased["id"] == job["id"]
    assert leased["status"] == "leased"
    assert leased["locked_by"] == "worker_a"
    
    # 3. Update heartbeat
    hb_ok = await repo.update_heartbeat(str(job["id"]), "worker_a")
    assert hb_ok is True
    
    # 4. Update progress
    prog_ok = await repo.update_progress(str(job["id"]), "chunking", 45.0)
    assert prog_ok is True
    
    updated_job = await repo.get_job(str(job["id"]))
    assert updated_job["status"] == "running"
    assert float(updated_job["progress"]) == 45.0
    
    # 5. Complete job
    comp_ok = await repo.complete_job(str(job["id"]))
    assert comp_ok is True
    
    completed_job = await repo.get_job(str(job["id"]))
    assert completed_job["status"] == "succeeded"
    assert float(completed_job["progress"]) == 100.0

@pytest.mark.asyncio
async def test_rag_repository_crud_vector_and_lexical(db_conn):
    """Tests corpus, corpus version activation, chunk indexing, vector search, and lexical search."""
    repo = RagRepository(db_conn)
    
    # 1. Get or create corpus
    corpus = await repo.get_or_create_corpus("fbise", "class_9", "physics")
    assert corpus["board_id"] == "fbise"
    
    # 2. Create corpus version
    cv = await repo.create_corpus_version(str(corpus["id"]), 1, "text-embedding-3-small", "rev1", 768)
    assert cv["status"] == "building"
    
    # 3. Activate corpus version
    act_ok = await repo.activate_corpus_version(str(cv["id"]), "admin_01")
    assert act_ok is True
    
    active_cv = await repo.get_active_corpus_version("fbise", "class_9", "physics")
    assert active_cv is not None
    assert str(active_cv["id"]) == str(cv["id"])
    
    # 4. Create document version
    doc_ver = await repo.create_document_version(
        str(cv["id"]), "res_physics_ch1", "v1.0", "p1.0", "Physics Chapter 1"
    )
    assert doc_ver["doc_title"] == "Physics Chapter 1"
    
    # 5. Insert chunk with metadata & 768-dim vector embedding
    sample_vector = [0.1] * 768
    chunk = await repo.insert_chunk(
        str(doc_ver["id"]),
        str(cv["id"]),
        0,
        "Physical quantities and measurement methods in physics.",
        chapter_id="ch1-introduction-to-physics",
        topic_no="1.1",
        topic_title="Physical Quantities",
        page_start=1,
        page_end=4,
        embedding=sample_vector
    )
    assert chunk["chunk_index"] == 0
    assert chunk["chapter_id"] == "ch1-introduction-to-physics"
    
    # 6. Vector search test
    vector_results = await repo.search_chunks_vector(str(cv["id"]), sample_vector, top_k=5)
    assert len(vector_results) == 1
    assert vector_results[0]["chapter_id"] == "ch1-introduction-to-physics"
    assert "distance" in vector_results[0]
    
    # 7. Lexical simple tsvector search test
    lexical_results = await repo.search_chunks_lexical(str(cv["id"]), "measurement physics", top_k=5)
    assert len(lexical_results) == 1
    assert lexical_results[0]["topic_title"] == "Physical Quantities"

@pytest.mark.asyncio
async def test_ai_request_repository_and_cache(db_conn):
    """Tests request creation, answer storing, and composite exact-answer cache lookup."""
    repo = AIRequestRepository(db_conn)
    
    # 1. Create request
    req = await repo.create_request(
        board_id="fbise",
        class_id="class_9",
        subject_id="physics",
        language="en",
        answer_mode="concise",
        raw_question="What is acceleration?",
        normalized_question="what is acceleration",
        question_hash="hash_accel_123",
        prompt_version="v1"
    )
    assert req["status"] == "pending"
    
    # 2. Record answer
    ans = await repo.create_answer(
        request_id=str(req["id"]),
        answer_text="Acceleration is the rate of change of velocity.",
        citation_sources=[{"chunk_id": "c1", "page": 12}],
        chunk_text_score=0.95,
        expected_question_score=0.92,
        tokens_used=120,
        latency_ms=350
    )
    assert ans["answer_text"] == "Acceleration is the rate of change of velocity."
    assert float(ans["chunk_text_score"]) == 0.95
    
    # 3. Exact-answer cache lookup test using composite key
    cached = await repo.find_cached_answer(
        board_id="fbise",
        class_id="class_9",
        subject_id="physics",
        answer_mode="concise",
        language="en",
        question_hash="hash_accel_123",
        prompt_version="v1"
    )
    assert cached is not None
    assert cached["answer_text"] == "Acceleration is the rate of change of velocity."
    assert str(cached["request_id"]) == str(req["id"])

@pytest.mark.asyncio
async def test_audit_repository(db_conn):
    """Tests writing and querying admin audit logs."""
    repo = AuditRepository(db_conn)
    
    log = await repo.create_audit_log(
        actor_id="admin_user_99",
        action="activate_corpus_version",
        target_type="rag_corpus_versions",
        target_id="cv_12345",
        before_value={"status": "building"},
        after_value={"status": "active"}
    )
    assert log["actor_id"] == "admin_user_99"
    assert log["action"] == "activate_corpus_version"
    
    logs = await repo.get_audit_logs(actor_id="admin_user_99")
    assert len(logs) >= 1
    assert logs[0]["target_id"] == "cv_12345"

@pytest.mark.asyncio
async def test_provider_attempt_repository(db_conn):
    """Tests recording and querying provider attempts."""
    from app.repositories.provider_attempt_repository import ProviderAttemptRepository
    repo = ProviderAttemptRepository(db_conn)
    
    attempt = await repo.record_attempt(
        provider="deepseek",
        model="deepseek-chat",
        status="success",
        attempt_no=1,
        provider_request_id="req_ds_001",
        system_fingerprint="fp_123",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=80,
        latency_ms=420,
        trace_id="tr_abc_789"
    )
    assert attempt["provider"] == "deepseek"
    assert attempt["status"] == "success"
    assert attempt["trace_id"] == "tr_abc_789"

