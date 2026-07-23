"""Integration tests for Database Migrations, Constraints, FK Cascade, and RLS Grants."""

import pytest
import asyncpg
from app.db.migrator import run_migrations

DB_URL = "postgresql://postgres:postgres@localhost:5432/taleem_dev"

@pytest.fixture
async def conn():
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
async def test_migrations_execution_and_idempotency():
    """Verifies that all SQL migrations are recorded and subsequent runs are idempotent."""
    connection = await asyncpg.connect(DB_URL)
    try:
        # Subsequent run should apply 0 new migrations
        applied_second = await run_migrations(connection)
        assert applied_second == []
        
        # Verify schema_migrations table contains all 3 migrations
        rows = await connection.fetch("SELECT version FROM schema_migrations ORDER BY version;")
        versions = [r["version"] for r in rows]
        assert "0001_platform_core.sql" in versions
        assert "0002_rag_schema.sql" in versions
        assert "0003_security_grants.sql" in versions
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_job_queue_check_constraints(conn):
    """Verifies CHECK constraints on job_queue status and progress."""
    # Invalid status
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO job_queue (job_type, payload, status)
                VALUES ('test_job', '{}'::jsonb, 'invalid_status');
                """
            )

    # Invalid progress (> 100)
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO job_queue (job_type, payload, status, progress)
                VALUES ('test_job', '{}'::jsonb, 'queued', 150.0);
                """
            )


@pytest.mark.asyncio
async def test_single_active_corpus_version_constraint(conn):
    """Verifies schema-level enforcement of max 1 active corpus version per corpus."""
    # 1. Create corpus
    corpus = await conn.fetchrow(
        "INSERT INTO rag_corpora (board_id, class_id, subject_id) VALUES ('fbise', 'class_9', 'physics') RETURNING id;"
    )
    corpus_id = corpus["id"]

    # 2. Insert first active version
    await conn.execute(
        """
        INSERT INTO rag_corpus_versions (corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, status)
        VALUES ($1, 1, 'text-embedding-3-small', 'v1', 768, 'active');
        """,
        corpus_id
    )

    # 3. Attempting to insert a second active version for same corpus must fail with UniqueViolationError
    with pytest.raises(asyncpg.UniqueViolationError):
        await conn.execute(
            """
            INSERT INTO rag_corpus_versions (corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, status)
            VALUES ($1, 2, 'text-embedding-3-small', 'v1', 768, 'active');
            """,
            corpus_id
        )

@pytest.mark.asyncio
async def test_foreign_key_on_delete_cascade_and_set_null(conn):
    """Verifies ON DELETE CASCADE and ON DELETE SET NULL behaviors."""
    # Setup corpus & active version
    corpus = await conn.fetchrow(
        "INSERT INTO rag_corpora (board_id, class_id, subject_id) VALUES ('bise_lahore', 'class_10', 'chemistry') RETURNING id;"
    )
    cv = await conn.fetchrow(
        """
        INSERT INTO rag_corpus_versions (corpus_id, version_no, embedding_model, embedding_revision, embedding_dim, status)
        VALUES ($1, 1, 'text-embedding-3-small', 'v1', 768, 'active') RETURNING id;
        """,
        corpus["id"]
    )
    
    # Create request with corpus_version_id
    req = await conn.fetchrow(
        """
        INSERT INTO ai_requests (board_id, class_id, subject_id, language, answer_mode, raw_question, normalized_question, question_hash, corpus_version_id, status)
        VALUES ('bise_lahore', 'class_10', 'chemistry', 'en', 'concise', 'What is water?', 'what is water', 'hash123', $1, 'completed')
        RETURNING id;
        """,
        cv["id"]
    )
    
    # Create answer
    await conn.execute(
        "INSERT INTO ai_answers (request_id, answer_text) VALUES ($1, 'H2O');",
        req["id"]
    )
    
    # Delete corpus version
    await conn.execute("DELETE FROM rag_corpus_versions WHERE id = $1;", cv["id"])
    
    # Verify ai_requests.corpus_version_id was set to NULL (ON DELETE SET NULL)
    req_updated = await conn.fetchrow("SELECT corpus_version_id FROM ai_requests WHERE id = $1;", req["id"])
    assert req_updated["corpus_version_id"] is None
    
    # Delete request
    await conn.execute("DELETE FROM ai_requests WHERE id = $1;", req["id"])
    
    # Verify answer was cascade deleted (ON DELETE CASCADE)
    ans_updated = await conn.fetchrow("SELECT id FROM ai_answers WHERE request_id = $1;", req["id"])
    assert ans_updated is None

ALL_TABLES = [
    "job_queue",
    "system_settings",
    "admin_audit_logs",
    "ai_requests",
    "ai_answers",
    "rag_corpora",
    "rag_corpus_versions",
    "rag_document_versions",
    "rag_chunks",
    "rag_visuals",
    "chunk_expected_questions",
    "approved_question_bank",
    "solved_papers",
    "provider_attempts",
]

@pytest.mark.asyncio
async def test_rls_deny_by_default_grants():
    """Verifies that anon and authenticated roles receive 42501 permission denied on all 14 application tables."""
    # 1. Test with anon role across all 14 tables
    anon_conn = await asyncpg.connect(DB_URL)
    try:
        await anon_conn.execute("SET ROLE anon; SET search_path = public;")
        for table in ALL_TABLES:
            with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
                await anon_conn.fetch(f"SELECT * FROM public.{table};")
            assert exc_info.value.sqlstate == "42501", f"Expected 42501 for anon role on table {table}, got {exc_info.value.sqlstate}"
    finally:
        await anon_conn.close()

    # 2. Test with authenticated role across all 14 tables
    auth_conn = await asyncpg.connect(DB_URL)
    try:
        await auth_conn.execute("SET ROLE authenticated; SET search_path = public;")
        for table in ALL_TABLES:
            with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
                await auth_conn.fetch(f"SELECT * FROM public.{table};")
            assert exc_info.value.sqlstate == "42501", f"Expected 42501 for authenticated role on table {table}, got {exc_info.value.sqlstate}"
    finally:
        await auth_conn.close()

@pytest.mark.asyncio
async def test_rls_deny_write_access():
    """Verifies that anon INSERT and authenticated UPDATE/DELETE attempts receive 42501 permission denied."""
    # 1. Test INSERT attempt as anon role
    anon_conn = await asyncpg.connect(DB_URL)
    try:
        await anon_conn.execute("SET ROLE anon; SET search_path = public;")
        with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
            await anon_conn.execute("INSERT INTO public.job_queue (job_type, payload) VALUES ('test_job', '{}');")
        assert exc_info.value.sqlstate == "42501"
    finally:
        await anon_conn.close()

    # 2. Test UPDATE and DELETE attempts as authenticated role
    auth_conn = await asyncpg.connect(DB_URL)
    try:
        await auth_conn.execute("SET ROLE authenticated; SET search_path = public;")
        with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
            await auth_conn.execute("UPDATE public.ai_requests SET status = 'completed';")
        assert exc_info.value.sqlstate == "42501"

        with pytest.raises(asyncpg.InsufficientPrivilegeError) as exc_info:
            await auth_conn.execute("DELETE FROM public.rag_corpora;")
        assert exc_info.value.sqlstate == "42501"
    finally:
        await auth_conn.close()




@pytest.mark.asyncio
async def test_provider_attempts_check_constraints(conn):
    """Verifies CHECK constraints on provider_attempts table."""
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO provider_attempts (provider, model, status, prompt_tokens)
                VALUES ('openai', 'gpt-4o', 'success', -10);
                """
            )
    with pytest.raises(asyncpg.CheckViolationError):
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO provider_attempts (provider, model, status)
                VALUES ('openai', 'gpt-4o', 'invalid_status');
                """
            )

