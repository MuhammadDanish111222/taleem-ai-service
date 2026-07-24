from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import asyncpg

from app.core.internal_auth import verify_internal_jwt, AuthContext
from app.db.pool import get_db_connection
from app.services.jobs.queue import JobQueueService

router = APIRouter()

class JsonlIngestRequest(BaseModel):
    jsonl_content: str = Field(..., description="Raw UTF-8 JSONL string content containing chapter chunk records")
    idempotency_key: Optional[str] = Field(None, description="Optional unique idempotency key for job deduplication")
    resource_version_id: Optional[str] = Field("v1", description="Resource version string, defaults to 'v1'")

@router.get("/internal/verify")
async def verify_internal_access(auth_context: AuthContext = Depends(verify_internal_jwt)):
    """Protected internal endpoint used for verifying cross-repository identity propagation."""
    return {
        "status": "authenticated",
        "uid": auth_context.uid,
        "is_admin": auth_context.is_admin,
        "feature": auth_context.feature,
        "request_id": auth_context.request_id
    }

@router.post("/internal/ingest/jsonl", status_code=status.HTTP_202_ACCEPTED)
async def submit_jsonl_ingest(
    request: JsonlIngestRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt)
):
    """Protected internal endpoint for submitting admin pre-chunked JSONL files for ingestion."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN_NOT_ADMIN", "message": "Admin privileges required for JSONL ingestion"}
        )

    if not request.jsonl_content or not request.jsonl_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_PAYLOAD", "message": "jsonl_content cannot be empty"}
        )

    async with get_db_connection() as conn:
        service = JobQueueService(conn)
        job_payload = {
            "jsonl_content": request.jsonl_content,
            "resource_version_id": request.resource_version_id or "v1",
            "submitted_by": auth_context.uid,
        }
        job = await service.enqueue_job(
            job_type="jsonl_ingest",
            payload=job_payload,
            idempotency_key=request.idempotency_key
        )

    return {
        "status": "queued",
        "job_id": str(job["id"]),
        "job_type": job["job_type"],
        "idempotency_key": job.get("idempotency_key"),
        "stage": job.get("stage"),
    }

