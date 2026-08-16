"""Read-only, bounded, content-free operations dashboard."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

import asyncpg

Window = Literal["24h", "7d", "30d"]
WINDOWS: dict[Window, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
SAFE_MESSAGES = {
    "QUOTA_EXCEEDED": "Usage limit reached.",
    "FEATURE_NOT_ENABLED": "Feature is not available.",
    "PROVIDER_TIMEOUT": "Provider request timed out.",
    "PROVIDER_UNAVAILABLE": "Provider is unavailable.",
    "INGESTION_FAILED": "Ingestion failed.",
    "TEST_GENERATION_FAILED": "Test generation failed.",
    "USAGE_LIMIT_REACHED": "Usage limit reached.",
}


def safe_error_code(value: object) -> str:
    """Never reflect a provider/database exception string to the local UI."""
    code = value if isinstance(value, str) else ""
    return code if code in SAFE_MESSAGES else "OPERATION_FAILED"


def safe_failure(row: dict[str, Any]) -> dict[str, Any]:
    code = safe_error_code(row.get("error_code"))
    return {
        "source": row["source"],
        "feature": row.get("feature", row["source"]),
        "error_code": code,
        "message": SAFE_MESSAGES.get(code, "Operation failed."),
        "request_id": str(row["request_id"]) if row.get("request_id") else None,
        "job_id": str(row["job_id"]) if row.get("job_id") else None,
        "trace_id": row.get("trace_id"),
        "status": row.get("status", "failed"),
        "retryable": row.get("retryable"),
        "timestamp": row["created_at"].isoformat(),
    }


class OperationsDashboardService:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def dashboard(self, window: Window = "24h") -> dict[str, Any]:
        if window not in WINDOWS:
            raise ValueError("OPERATIONS_WINDOW_INVALID")
        seconds = int(WINDOWS[window].total_seconds())
        jobs, rag, answers, providers, events, failures = await self._queries(seconds)
        job_groups = [
            {"job_type": r["job_type"], "status": r["status"], "count": r["count"]}
            for r in jobs
        ]
        retrieval = next(
            (dict(r) for r in events if r["event_type"] == "retrieval_outcome"),
            {"numerator": 0, "denominator": 0},
        )
        quota = next(
            (dict(r) for r in events if r["event_type"] == "quota_block"), {"count": 0}
        )
        test = next(
            (dict(r) for r in events if r["event_type"] == "test_generation_failure"),
            {"count": 0},
        )
        answer_metrics = dict(answers)
        answer_metrics["approved_bank_rate"] = (
            answer_metrics["approved_bank_hits"]
            / answer_metrics["approved_bank_denominator"]
            if answer_metrics["approved_bank_denominator"]
            else 0
        )
        return {
            "window": window,
            "retained_data": {"multiple_ask": True},
            "summary": {"job_count": sum(r["count"] for r in jobs)},
            "jobs": job_groups,
            "rag": dict(rag),
            "answers": answer_metrics,
            "providers": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "status": r["status"],
                    "error_code": safe_error_code(r["error_code"]),
                    "count": r["count"],
                }
                for r in providers
            ],
            "quota": {"blocks": quota["count"]},
            "test_generation": {"failures": test["count"]},
            "recent_failures": [safe_failure(dict(r)) for r in failures],
            "retrieval": {
                **retrieval,
                "rate": retrieval["numerator"] / retrieval["denominator"]
                if retrieval["denominator"]
                else 0,
            },
        }

    async def _queries(self, seconds: int):
        # ``asyncpg.Connection`` permits one operation at a time.  These are a
        # fixed number of aggregate queries (not a per-row query loop).
        jobs = await self.conn.fetch(
            "SELECT job_type,status,count(*)::int count FROM job_queue "
            "WHERE created_at >= NOW() - $1::int * interval '1 second' "
            "GROUP BY job_type,status ORDER BY job_type,status",
            seconds,
        )
        rag = await self.conn.fetchrow(
            "SELECT (SELECT count(*)::int FROM rag_corpus_versions) corpus_versions, "
            "(SELECT count(*)::int FROM rag_chunks) chunks, "
            "(SELECT count(*)::int FROM chunk_expected_questions) expected_question_embeddings, "
            "(SELECT count(*)::int FROM chunk_expected_questions WHERE embedding_status='pending') expected_questions_pending, "
            "(SELECT count(*)::int FROM chunk_expected_questions WHERE embedding_status='embedded') expected_questions_embedded, "
            "(SELECT count(*)::int FROM chunk_expected_questions WHERE embedding_status='failed') expected_questions_failed, "
            "(SELECT count(*)::int FROM prompt_versions) prompt_versions"
        )
        answers = await self.conn.fetchrow(
            "SELECT "
            "(SELECT count(*)::int FROM ai_answers WHERE created_at >= NOW() - $1::int * interval '1 second' AND review_status='pending') pending_candidates, "
            "(SELECT count(*)::int FROM ai_answers WHERE created_at >= NOW() - $1::int * interval '1 second' AND review_status='rejected') rejected_candidates, "
            "(SELECT count(*)::int FROM ai_answers WHERE created_at >= NOW() - $1::int * interval '1 second' AND review_status='approved' AND approved_revision_id IS NOT NULL) promoted_candidates, "
            "(SELECT count(*)::int FROM ai_answers WHERE created_at >= NOW() - $1::int * interval '1 second' AND retention_expires_at IS NOT NULL AND review_status IN ('pending','rejected')) retention_eligible_candidates, "
            "(SELECT count(*)::int FROM ai_requests WHERE created_at >= NOW() - $1::int * interval '1 second' AND source_feature='single_question' AND status IN ('completed','no_answer') AND answer_source='approved_bank') approved_bank_hits, "
            "(SELECT count(*)::int FROM ai_requests WHERE created_at >= NOW() - $1::int * interval '1 second' AND source_feature='single_question' AND status IN ('completed','no_answer')) approved_bank_denominator, "
            "(SELECT count(*)::int FROM ai_requests WHERE created_at >= NOW() - $1::int * interval '1 second' AND source_feature='single_question' AND status='completed' AND answer_source='general_knowledge') general_fallbacks",
            seconds,
        )
        providers = await self.conn.fetch(
            "SELECT provider,model,status,error_code,count(*)::int count FROM provider_attempts "
            "WHERE created_at >= NOW() - $1::int * interval '1 second' AND status <> 'success' "
            "GROUP BY provider,model,status,error_code ORDER BY count DESC,provider LIMIT 100",
            seconds,
        )
        events = await self.conn.fetch(
            "SELECT event_type, count(*) FILTER (WHERE event_type='retrieval_outcome' AND outcome='empty')::int numerator, "
            "count(*) FILTER (WHERE event_type='retrieval_outcome')::int denominator, count(*)::int count "
            "FROM operational_events WHERE created_at >= NOW() - $1::int * interval '1 second' GROUP BY event_type",
            seconds,
        )
        failures = await self.conn.fetch(
            "SELECT 'job_queue' source,'jobs' feature,error_code,id job_id,NULL::uuid request_id,NULL::text trace_id,status,NULL::boolean retryable,created_at "
            "FROM job_queue WHERE created_at >= NOW() - $1::int * interval '1 second' AND status='failed' "
            "UNION ALL SELECT 'provider_attempts','providers',error_code,job_id,ai_request_id,trace_id,status,(status='retryable_error'),created_at "
            "FROM provider_attempts WHERE created_at >= NOW() - $1::int * interval '1 second' AND status <> 'success' "
            "UNION ALL SELECT 'operational_events',feature,error_code,job_id,request_id,NULL::text,outcome,NULL::boolean,created_at "
            "FROM operational_events WHERE created_at >= NOW() - $1::int * interval '1 second' AND event_type='test_generation_failure' "
            "ORDER BY created_at DESC LIMIT 100",
            seconds,
        )
        return jobs, rag, answers, providers, events, failures
