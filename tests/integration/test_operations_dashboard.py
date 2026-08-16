"""PostgreSQL aggregate, audit-ordering, and redaction coverage for Run 2."""

import hashlib
import os
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
import pytest

from app.api.v1 import operations_dashboard as dashboard_api
from app.core.internal_auth import AuthContext
from app.services.operations_dashboard import OperationsDashboardService

DB_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/taleem_dev"
)


async def _clean(conn: asyncpg.Connection) -> None:
    await conn.execute(
        "TRUNCATE operational_events,provider_attempts,ai_answers,ai_requests,job_queue,"
        "rag_chunks,rag_document_versions,rag_corpus_versions,rag_corpora,prompt_versions,"
        "admin_audit_logs CASCADE"
    )


async def _request(
    conn: asyncpg.Connection, *, question_hash: str, answer_source: str
) -> UUID:
    return await conn.fetchval(
        "INSERT INTO ai_requests(board_id,class_id,subject_id,language,answer_mode,raw_question,"
        "normalized_question,question_hash,prompt_version,status,source_feature,answer_source) "
        "VALUES('board','class','subject','en','short','TEST_PRIVATE_QUESTION',"
        "'normalized',$1,'v','completed','single_question',$2) RETURNING id",
        question_hash,
        answer_source,
    )


@pytest.mark.asyncio
async def test_dashboard_uses_exact_persisted_aggregates_and_redacts_content():
    conn = await asyncpg.connect(DB_URL)
    try:
        await _clean(conn)
        corpus = await conn.fetchval(
            "INSERT INTO rag_corpora(board_id,class_id,subject_id) "
            "VALUES('board','class','subject') RETURNING id"
        )
        version = await conn.fetchval(
            "INSERT INTO rag_corpus_versions(corpus_id,version_no,embedding_model,"
            "embedding_revision,embedding_dim,status) VALUES($1,1,'model','rev',512,'building') "
            "RETURNING id",
            corpus,
        )
        document = await conn.fetchval(
            "INSERT INTO rag_document_versions(corpus_version_id,resource_id,resource_version_id,"
            "pipeline_version,doc_title) VALUES($1,'resource','version','pipeline','safe') RETURNING id",
            version,
        )
        chunk = await conn.fetchval(
            "INSERT INTO rag_chunks(document_version_id,corpus_version_id,chunk_index,content) "
            "VALUES($1,$2,0,'TEST_PRIVATE_CHUNK') RETURNING id",
            document,
            version,
        )
        normalized = "safe expected question"
        await conn.execute(
            "INSERT INTO chunk_expected_questions(chunk_id,question_text,question_normalized,"
            "question_hash,embedding) VALUES($1,$2,$2,$3,array_fill(0::real,ARRAY[512])::halfvec)",
            chunk,
            normalized,
            hashlib.sha256(normalized.encode()).hexdigest(),
        )
        await conn.execute(
            "INSERT INTO prompt_versions(prompt_key,answer_mode,version,content,status,created_by,"
            "activated_by,activated_at) VALUES('ask_grounded','short',1,'TEST_PRIVATE_PROMPT',"
            "'active','test','test',NOW())"
        )
        for job_type in (
            "jsonl_ingest",
            "embed_chunks",
            "multiple_ask_extract",
            "multiple_ask_answer",
        ):
            await conn.execute(
                "INSERT INTO job_queue(job_type,payload,status,stage,error_code,error_message) "
                "VALUES($1,'{}','failed','failed','INGESTION_FAILED','TEST_PRIVATE_STACK')",
                job_type,
            )

        approved = await _request(
            conn, question_hash="a" * 64, answer_source="approved_bank"
        )
        general = await _request(
            conn, question_hash="b" * 64, answer_source="general_knowledge"
        )
        await conn.execute(
            "INSERT INTO ai_answers(request_id,answer_text,answer_source,answer_mode,review_status,"
            "retention_expires_at) VALUES($1,'TEST_PRIVATE_ANSWER','general_knowledge','short',"
            "'pending',NOW()+interval '1 day')",
            general,
        )
        provider_job = await conn.fetchval(
            "INSERT INTO job_queue(job_type,payload,status,stage) "
            "VALUES('jsonl_ingest','{}','failed','failed') RETURNING id"
        )
        await conn.execute(
            "INSERT INTO provider_attempts(job_id,provider,model,status,error_code,trace_id) "
            "VALUES($1,'provider','model','retryable_error','PROVIDER_TIMEOUT','safe-trace')",
            provider_job,
        )
        await conn.execute(
            "INSERT INTO provider_attempts(job_id,provider,model,status,error_code) "
            "VALUES($1,'provider','model','non_retryable_error','TEST_RAW_PROVIDER_ERROR')",
            provider_job,
        )
        # The approved-bank request intentionally has no retrieval outcome.
        for outcome in ("empty", "evidence_found"):
            await conn.execute(
                "INSERT INTO operational_events(feature,event_type,outcome,request_id) "
                "VALUES('single_question','retrieval_outcome',$1,$2)",
                outcome,
                general,
            )
        await conn.execute(
            "INSERT INTO operational_events(feature,event_type,outcome,error_code,request_id) "
            "VALUES('single_question','quota_block','denied','USAGE_LIMIT_REACHED',$1)",
            general,
        )
        await conn.execute(
            "INSERT INTO operational_events(feature,event_type,outcome,error_code,request_id) "
            "VALUES('test_generation','test_generation_failure','failed','TEST_GENERATION_FAILED',$1)",
            general,
        )

        data = await OperationsDashboardService(conn).dashboard("24h")
        assert data["retrieval"] == {
            "event_type": "retrieval_outcome",
            "numerator": 1,
            "denominator": 2,
            "count": 2,
            "rate": 0.5,
        }
        assert data["answers"] == {
            "pending_candidates": 1,
            "rejected_candidates": 0,
            "promoted_candidates": 0,
            "retention_eligible_candidates": 1,
            "approved_bank_hits": 1,
            "approved_bank_denominator": 2,
            "general_fallbacks": 1,
            "approved_bank_rate": 0.5,
        }
        assert data["rag"] == {
            "corpus_versions": 1,
            "chunks": 1,
            "expected_question_embeddings": 1,
            "expected_questions_pending": 1,
            "expected_questions_embedded": 0,
            "expected_questions_failed": 0,
            "prompt_versions": 1,
        }
        assert {row["job_type"] for row in data["jobs"]} >= {
            "jsonl_ingest",
            "embed_chunks",
            "multiple_ask_extract",
            "multiple_ask_answer",
        }
        assert data["quota"] == {"blocks": 1}
        assert data["test_generation"] == {"failures": 1}
        assert {row["error_code"] for row in data["providers"]} == {
            "PROVIDER_TIMEOUT",
            "OPERATION_FAILED",
        }
        response = str(data)
        for forbidden in (
            "TEST_PRIVATE_QUESTION",
            "TEST_PRIVATE_ANSWER",
            "TEST_PRIVATE_PROMPT",
            "TEST_PRIVATE_CHUNK",
            "TEST_PRIVATE_STACK",
            "TEST_RAW_PROVIDER_ERROR",
            "raw_question",
            "answer_text",
            "storage_object_key",
            "database_url",
            "token",
        ):
            assert forbidden not in response
        assert str(approved) not in response
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_empty_periods_are_zero_or_empty():
    conn = await asyncpg.connect(DB_URL)
    try:
        await _clean(conn)
        data = await OperationsDashboardService(conn).dashboard("24h")
        assert data["jobs"] == []
        assert data["providers"] == []
        assert data["recent_failures"] == []
        assert data["retrieval"] == {"numerator": 0, "denominator": 0, "rate": 0}
        assert data["answers"]["approved_bank_denominator"] == 0
        assert data["answers"]["approved_bank_rate"] == 0
        assert data["quota"] == {"blocks": 0}
        assert data["test_generation"] == {"failures": 0}
        assert all(value == 0 for value in data["rag"].values())
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_audit_search_is_deterministic_and_content_free(monkeypatch):
    conn = await asyncpg.connect(DB_URL)
    try:
        await _clean(conn)
        first = UUID("00000000-0000-0000-0000-000000000001")
        second = UUID("00000000-0000-0000-0000-000000000002")
        for audit_id in (first, second):
            await conn.execute(
                "INSERT INTO admin_audit_logs(id,actor_id,action,target_type,target_id,before_value,"
                "after_value,created_at) VALUES($1,'TEST_PRIVATE_UID','candidate.rejected','ai_answer',"
                "'storage-private-key', '{\"answer\":\"TEST_PRIVATE_ANSWER\"}',"
                '\'{"error_code":"PROVIDER_TIMEOUT","prompt":"TEST_PRIVATE_PROMPT"}\',NOW())',
                audit_id,
            )

        @asynccontextmanager
        async def connection():
            yield conn

        monkeypatch.setattr(dashboard_api, "get_db_connection", connection)
        auth = AuthContext(
            uid="admin",
            is_admin=True,
            feature="local_operations_dashboard",
            request_id="request",
        )
        first_page = await dashboard_api.audit_search(
            window="24h",
            limit=1,
            cursor=None,
            action="candidate.rejected",
            target_type=None,
            target_id=None,
            error_code="PROVIDER_TIMEOUT",
            auth=auth,
        )
        second_page = await dashboard_api.audit_search(
            window="24h",
            limit=1,
            cursor=UUID(first_page["next_cursor"]),
            action=None,
            target_type=None,
            target_id=None,
            error_code=None,
            auth=auth,
        )
        assert [row["id"] for row in first_page["items"] + second_page["items"]] == [
            str(second),
            str(first),
        ]
        assert first_page["items"][0]["target_id"] == "redacted"
        response = str(first_page) + str(second_page)
        for forbidden in (
            "TEST_PRIVATE_UID",
            "TEST_PRIVATE_ANSWER",
            "TEST_PRIVATE_PROMPT",
            "storage-private-key",
        ):
            assert forbidden not in response
    finally:
        await conn.close()
