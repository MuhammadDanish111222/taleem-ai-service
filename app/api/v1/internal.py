import hashlib
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.firebase_admin import get_firebase_app
from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.repositories.audit_repository import AuditRepository
from app.services.ingestion.jsonl_chunks import (
    extract_safe_scope,
    get_validation_error_code,
    validate_and_parse_jsonl,
)
from app.services.jobs.queue import JobQueueService

router = APIRouter()


class JsonlIngestRequest(BaseModel):
    jsonl_content: str = Field(
        ...,
        description="Raw UTF-8 JSONL string content containing chapter chunk records",
    )
    idempotency_key: Optional[str] = Field(
        None, description="Optional unique idempotency key for job deduplication"
    )
    resource_version_id: Optional[str] = Field(
        "v1", description="Resource version string, defaults to 'v1'"
    )


def _safe_hash(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scope_from_valid_chunks(chunks: list[Dict[str, Any]]) -> Dict[str, str]:
    if not chunks:
        return {}
    first = chunks[0]
    return {
        key: first[key] for key in ("board_id", "class_id", "subject_id", "chapter_id")
    }


def _get_firestore_db():
    app = get_firebase_app()
    from firebase_admin import firestore

    return firestore.client(app=app)


@router.get("/internal/verify")
async def verify_internal_access(
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Protected internal endpoint used for verifying cross-repository identity propagation."""
    return {
        "status": "authenticated",
        "uid": auth_context.uid,
        "is_admin": auth_context.is_admin,
        "feature": auth_context.feature,
        "request_id": auth_context.request_id,
    }


@router.post("/internal/ingest/jsonl", status_code=status.HTTP_202_ACCEPTED)
async def submit_jsonl_ingest(
    request: JsonlIngestRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Protected internal endpoint for submitting admin pre-chunked JSONL files for ingestion."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN_NOT_ADMIN",
                "message": "Admin privileges required for JSONL ingestion",
            },
        )

    if not request.jsonl_content or not request.jsonl_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_PAYLOAD",
                "message": "jsonl_content cannot be empty",
            },
        )

    source_hash = _safe_hash(request.jsonl_content)
    idempotency_key_hash = _safe_hash(request.idempotency_key)
    try:
        firestore_db = _get_firestore_db()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FIRESTORE_UNAVAILABLE",
                "message": "Catalogue validation is unavailable",
            },
        )

    valid_chunks, errors = await validate_and_parse_jsonl(
        request.jsonl_content, firestore_db
    )
    scope = _scope_from_valid_chunks(valid_chunks) or extract_safe_scope(
        request.jsonl_content
    )

    async with get_db_connection() as conn:
        audit = AuditRepository(conn)
        if errors:
            error_code = get_validation_error_code(errors)
            async with conn.transaction():
                await audit.create_jsonl_ingestion_audit(
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                    scope=scope,
                    outcome="rejected",
                    error_code=error_code,
                    job_id=None,
                    source_hash=source_hash or "",
                    idempotency_key_hash=idempotency_key_hash,
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": error_code,
                    "message": "JSONL validation failed",
                    "errors": errors,
                },
            )

        service = JobQueueService(conn)
        job_payload = {
            "jsonl_content": request.jsonl_content,
            "resource_version_id": request.resource_version_id or "v1",
            "submitted_by": auth_context.uid,
            "request_id": auth_context.request_id,
            "source_hash": source_hash,
            "idempotency_key_hash": idempotency_key_hash,
            "scope": scope,
        }
        async with conn.transaction():
            job = await service.enqueue_job(
                job_type="jsonl_ingest",
                payload=job_payload,
                idempotency_key=request.idempotency_key,
            )
            await audit.create_jsonl_ingestion_audit(
                actor_id=auth_context.uid,
                request_id=auth_context.request_id,
                scope=scope,
                outcome="accepted",
                error_code=None,
                job_id=str(job["id"]),
                source_hash=source_hash or "",
                idempotency_key_hash=idempotency_key_hash,
            )

    return {
        "status": "queued",
        "job_id": str(job["id"]),
        "job_type": job["job_type"],
        "idempotency_key": job.get("idempotency_key"),
        "stage": job.get("stage"),
    }
