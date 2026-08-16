import hashlib
import json
from typing import Any, Dict, Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.firebase_admin import get_firebase_app
from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.repositories.ask_repository import AskRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.multiple_ask_repository import MultipleAskRepository
from app.services.ingestion.jsonl_chunks import (
    extract_safe_scope,
    get_validation_error_code,
    validate_and_parse_jsonl,
)
from app.services.jobs.queue import JobQueueService
from app.services.local_admin import LocalAdminError, LocalAdminService
from app.services.multiple_ask import MultipleAskError, MultipleAskService
from app.services.multiple_ask_answers import MultipleAskAnswerService
from app.services.multiple_ask_extraction_service import MultipleAskExtractionService
from app.services.retrieval.evidence import RetrievalScope
from app.services.retrieval.service import RetrievalScopeError, RetrievalService
from app.services.runtime_settings import RuntimeSettingsService, Scope
from app.services.usage.models import AccountTier
from app.services.usage.service import UsageLimitExceeded, UsageService

router = APIRouter()
_MULTIPLE_ASK_SCOPE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"


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


class RagAdminRequest(BaseModel):
    operation: str
    board_id: str
    class_id: str
    subject_id: str
    corpus_version_id: Optional[str] = None
    question_id: Optional[str] = None
    chunk_id: Optional[str] = None
    question_text: Optional[str] = None
    visual_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    review_status: Optional[str] = None
    display_policy: Optional[str] = None
    question: Optional[str] = None
    chapter_id: Optional[str] = None


class PairedImportAuditRequest(BaseModel):
    operation: str
    import_hash: str
    board_id: str
    class_id: str
    subject_id: str
    chunk_count: int = 0
    referenced_visual_count: int = 0
    unused_visual_count: int = 0
    asset_hashes: list[str] = Field(default_factory=list)
    job_id: Optional[str] = None
    error_code: Optional[str] = None


class PairedImportStatusRequest(BaseModel):
    import_hash: str


class MultipleAskFileSessionRequest(BaseModel):
    request_id: UUID
    input_kind: Literal["image", "pdf"]
    content_type: str = Field(..., min_length=1, max_length=100)
    size_bytes: int = Field(..., gt=0)
    board_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    class_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    subject_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    chapter_id: Optional[str] = Field(
        None, min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )


class MultipleAskFinalizeRequest(BaseModel):
    request_id: UUID
    session_id: UUID


class MultipleAskTextRequest(BaseModel):
    request_id: UUID
    text: str = Field(..., min_length=1, max_length=30000)
    board_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    class_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    subject_id: str = Field(
        ..., min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )
    chapter_id: Optional[str] = Field(
        None, min_length=1, max_length=120, pattern=_MULTIPLE_ASK_SCOPE_PATTERN
    )


class MultipleAskCorrectionOption(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str = Field(..., min_length=1, max_length=5000)


class MultipleAskCorrectionRequest(BaseModel):
    request_id: UUID
    question_text: str = Field(..., min_length=1, max_length=30000)
    answer_mode: Literal["short", "long", "mcq"]
    mcq_options: list[MultipleAskCorrectionOption] = Field(default_factory=list)


class MultipleAskResumeRequest(BaseModel):
    request_id: UUID


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
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


async def _multiple_ask_tier(auth: AuthContext) -> AccountTier:
    # The environment switch is no longer the public source of truth. The
    # default database lifecycle state remains disabled, so incomplete flows
    # stay fail-closed until an audited local-admin change enables them.
    async with get_db_connection() as conn:
        if (
            await RuntimeSettingsService(conn).get(
                "feature.multiple_ask", Scope(kind="global")
            )
            != "enabled"
        ):
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND"})
    if auth.feature != "multiple_ask":
        raise HTTPException(
            status_code=403,
            detail={"code": "AUTH_FEATURE_FORBIDDEN", "message": "Feature denied"},
        )
    try:
        return AccountTier(auth.account_tier)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Invalid account tier"},
        ) from None


def _multiple_ask_error(exc: MultipleAskError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": "Multiple Ask request rejected"},
    )


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


@router.post("/internal/multiple-ask/upload-sessions")
async def create_multiple_ask_upload_session(
    request: MultipleAskFileSessionRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Mint a browser-safe capability for one private temporary file object."""
    tier = await _multiple_ask_tier(auth_context)
    if str(request.request_id) != auth_context.request_id:
        raise HTTPException(status_code=409, detail={"code": "REQUEST_ID_MISMATCH"})
    try:
        async with get_db_connection() as conn:
            return await MultipleAskService(conn).create_file_session(
                client_request_id=str(request.request_id),
                uid=auth_context.uid,
                tier=tier,
                input_kind=request.input_kind,
                content_type=request.content_type,
                size_bytes=request.size_bytes,
                board_id=request.board_id,
                class_id=request.class_id,
                subject_id=request.subject_id,
                chapter_id=request.chapter_id,
            )
    except MultipleAskError as exc:
        raise _multiple_ask_error(exc) from None


@router.post("/internal/multiple-ask/upload-sessions/finalize")
async def finalize_multiple_ask_upload_session(
    request: MultipleAskFinalizeRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Finalize the source and atomically create its durable validation job."""
    tier = await _multiple_ask_tier(auth_context)
    if str(request.request_id) != auth_context.request_id:
        raise HTTPException(status_code=409, detail={"code": "REQUEST_ID_MISMATCH"})
    try:
        async with get_db_connection() as conn:
            return await MultipleAskService(conn).finalize_file(
                session_id=str(request.session_id),
                client_request_id=str(request.request_id),
                uid=auth_context.uid,
                tier=tier,
            )
    except UsageLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "USAGE_LIMIT_REACHED",
                "message": "Daily batch limit reached",
            },
        ) from exc
    except MultipleAskError as exc:
        raise _multiple_ask_error(exc) from None


@router.post("/internal/multiple-ask/text")
async def submit_multiple_ask_text(
    request: MultipleAskTextRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Small pasted text uses the same durable parent/quota model, not Storage."""
    tier = await _multiple_ask_tier(auth_context)
    if str(request.request_id) != auth_context.request_id:
        raise HTTPException(status_code=409, detail={"code": "REQUEST_ID_MISMATCH"})
    try:
        async with get_db_connection() as conn:
            return await MultipleAskService(conn).submit_text(
                client_request_id=str(request.request_id),
                uid=auth_context.uid,
                tier=tier,
                text=request.text,
                board_id=request.board_id,
                class_id=request.class_id,
                subject_id=request.subject_id,
                chapter_id=request.chapter_id,
            )
    except UsageLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "USAGE_LIMIT_REACHED",
                "message": "Daily batch limit reached",
            },
        ) from exc
    except MultipleAskError as exc:
        raise _multiple_ask_error(exc) from None


def _multiple_ask_status_response(record: dict[str, Any]) -> dict[str, Any]:
    """Return only polling-safe metadata, never source text, keys, or bytes."""

    def timestamp(value: Any) -> str | None:
        return value.isoformat() if hasattr(value, "isoformat") else None

    def json_value(value: Any, fallback: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return fallback
        return value if value is not None else fallback

    def topic_names(item: dict[str, Any]) -> list[str]:
        source = item.get("answer_source") or item.get("persisted_answer_source")
        if source not in {"approved_bank", "syllabus_grounded"}:
            return []
        limit = 1 if item.get("answer_mode") == "short" else 2
        names: list[str] = []
        for citation in json_value(item.get("citation_sources"), []):
            if not isinstance(citation, dict):
                continue
            name = citation.get("topic_title") or citation.get("topicTitle")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    return {
        "job_id": str(record["id"]),
        "workflow_status": record["workflow_status"],
        "input_kind": record["input_kind"],
        "scope": {
            "board_id": record["board_id"],
            "class_id": record["class_id"],
            "subject_id": record["subject_id"],
            "chapter_id": record["chapter_id"],
        },
        "created_at": timestamp(record["created_at"]),
        "updated_at": timestamp(record["updated_at"]),
        "retention_expires_at": timestamp(record.get("retention_expires_at")),
        "terminal_error_code": record.get("terminal_error_code"),
        "queue": {
            "status": record.get("queue_status"),
            "stage": record.get("queue_stage"),
            "progress": record.get("queue_progress"),
        },
        "items": [
            {
                "item_id": str(item["id"]),
                "item_index": item["item_index"],
                "display_label": item.get("display_label"),
                "section_context": item.get("section_context"),
                "question_text": item.get("source_text"),
                "item_status": item["item_status"],
                "normalized_question": item["normalized_question"],
                "answer_mode": item["answer_mode"],
                "mcq_options": json_value(item["mcq_options"], []),
                "unclear_reason": item["unclear_reason"],
                "terminal_error_code": item.get("terminal_error_code"),
                "source_locator": json_value(item["source_locator"], None),
                "extraction_version": item["extraction_version"],
                "correction_version": item["correction_version"],
                "corrected_at": timestamp(item["corrected_at"]),
                "result": (
                    {
                        "answer_source": item.get("answer_source")
                        or item.get("persisted_answer_source"),
                        "blocks": json_value(item.get("answer_blocks"), []),
                        "citations": (
                            json_value(item.get("citation_sources"), [])
                            if (
                                item.get("answer_source")
                                or item.get("persisted_answer_source")
                            )
                            in {"approved_bank", "syllabus_grounded"}
                            else []
                        ),
                        "visual_ids": (
                            json_value(item.get("visual_ids"), [])
                            if (
                                item.get("answer_source")
                                or item.get("persisted_answer_source")
                            )
                            in {"approved_bank", "syllabus_grounded"}
                            else []
                        ),
                        "approved_revision_id": (
                            str(item["approved_revision_id"])
                            if item.get("approved_revision_id")
                            else None
                        ),
                        "mcq_result": json_value(item.get("mcq_result"), None),
                        "topic_names": topic_names(item),
                        "visuals": json_value(item.get("visuals"), []),
                    }
                    if item["item_status"] == "answered"
                    else None
                ),
            }
            for item in record["items"]
        ],
        "summary": {
            "total": len(record["items"]),
            "short": sum(item["answer_mode"] == "short" for item in record["items"]),
            "long": sum(item["answer_mode"] == "long" for item in record["items"]),
            "mcq": sum(item["answer_mode"] == "mcq" for item in record["items"]),
            "not_clear": sum(
                item["answer_mode"] == "not_clear" for item in record["items"]
            ),
        },
    }


@router.get("/internal/multiple-ask/jobs/{job_id}")
async def get_multiple_ask_job_status(
    job_id: UUID, auth_context: AuthContext = Depends(verify_internal_jwt)
):
    await _multiple_ask_tier(auth_context)
    async with get_db_connection() as conn:
        record = await MultipleAskRepository(conn).get_owned_job_status(
            job_id=str(job_id), uid_hash=UsageService().uid_hash(auth_context.uid)
        )
    if record is None:
        raise HTTPException(
            status_code=404, detail={"code": "MULTIPLE_ASK_JOB_NOT_FOUND"}
        )
    return _multiple_ask_status_response(record)


@router.get("/internal/multiple-ask/jobs/{job_id}/visual/{visual_id}")
async def multiple_ask_visual_reference(
    job_id: UUID,
    visual_id: str,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    await _multiple_ask_tier(auth_context)
    if not visual_id or len(visual_id) > 160:
        raise HTTPException(status_code=404, detail={"code": "VISUAL_NOT_FOUND"})
    try:
        async with get_db_connection() as conn:
            reference = await AskRepository(conn).multiple_ask_visual_stream_reference(
                job_id=str(job_id),
                uid_hash=UsageService().uid_hash(auth_context.uid),
                visual_id=visual_id,
            )
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail={"code": "VISUAL_CONFIGURATION_ERROR"},
        ) from None
    if reference is None:
        raise HTTPException(status_code=404, detail={"code": "VISUAL_NOT_FOUND"})
    return reference


@router.post("/internal/multiple-ask/jobs/{job_id}/items/{item_id}/correction")
async def correct_multiple_ask_item(
    job_id: UUID,
    item_id: UUID,
    request: MultipleAskCorrectionRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    await _multiple_ask_tier(auth_context)
    if str(request.request_id) != auth_context.request_id:
        raise HTTPException(status_code=409, detail={"code": "REQUEST_ID_MISMATCH"})
    try:
        async with get_db_connection() as conn:
            service = MultipleAskExtractionService(conn)
            record = await service.apply_correction(
                job_id=str(job_id),
                item_id=str(item_id),
                uid=auth_context.uid,
                request_id=str(request.request_id),
                question_text=request.question_text,
                answer_mode=request.answer_mode,
                mcq_options=[option.model_dump() for option in request.mcq_options],
            )
        if record is None:
            raise HTTPException(
                status_code=404, detail={"code": "MULTIPLE_ASK_JOB_NOT_FOUND"}
            )
        return _multiple_ask_status_response(record)
    except MultipleAskError as exc:
        raise _multiple_ask_error(exc) from None


@router.post("/internal/multiple-ask/jobs/{job_id}/resume")
async def resume_multiple_ask_job(
    job_id: UUID,
    request: MultipleAskResumeRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    await _multiple_ask_tier(auth_context)
    if str(request.request_id) != auth_context.request_id:
        raise HTTPException(status_code=409, detail={"code": "REQUEST_ID_MISMATCH"})
    try:
        async with get_db_connection() as conn:
            return await MultipleAskAnswerService(conn).start_for_job(
                job_id=str(job_id), uid=auth_context.uid
            )
    except MultipleAskError as exc:
        raise _multiple_ask_error(exc) from None


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


@router.post("/internal/paired-import/audit")
async def paired_import_audit(
    request: PairedImportAuditRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Records non-sensitive paired-import state for retry and cleanup auditability."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN_NOT_ADMIN",
                "message": "Admin privileges required",
            },
        )
    if (
        request.operation not in {"started", "assets_uploaded", "queued", "failed"}
        or len(request.import_hash) != 64
        or any(character not in "0123456789abcdef" for character in request.import_hash)
        or not all(
            value.strip()
            for value in (request.board_id, request.class_id, request.subject_id)
        )
        or any(
            value < 0
            for value in (
                request.chunk_count,
                request.referenced_visual_count,
                request.unused_visual_count,
            )
        )
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in request.asset_hashes
        )
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "PAIRED_IMPORT_AUDIT_INVALID",
                "message": "Invalid paired import audit request",
            },
        )
    state = {
        "started": "validating",
        "assets_uploaded": "assets_uploaded",
        "queued": "queued",
        "failed": "failed",
    }[request.operation]
    async with get_db_connection() as conn:
        await conn.execute(
            """INSERT INTO rag_paired_imports
              (import_hash,actor_id,board_id,class_id,subject_id,status,chunk_count,referenced_visual_count,unused_visual_count,asset_hashes,job_id,error_code)
              VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::uuid,$12)
              ON CONFLICT (import_hash) DO UPDATE SET
                status=EXCLUDED.status, chunk_count=EXCLUDED.chunk_count,
                referenced_visual_count=EXCLUDED.referenced_visual_count,
                unused_visual_count=EXCLUDED.unused_visual_count,
                asset_hashes=CASE WHEN jsonb_array_length(EXCLUDED.asset_hashes)>0 THEN EXCLUDED.asset_hashes ELSE rag_paired_imports.asset_hashes END,
                job_id=COALESCE(EXCLUDED.job_id,rag_paired_imports.job_id),
                error_code=EXCLUDED.error_code, updated_at=NOW();""",
            request.import_hash,
            auth_context.uid,
            request.board_id,
            request.class_id,
            request.subject_id,
            state,
            request.chunk_count,
            request.referenced_visual_count,
            request.unused_visual_count,
            json.dumps(request.asset_hashes),
            request.job_id,
            request.error_code,
        )
    return {"status": state}


@router.post("/internal/paired-import/status")
async def paired_import_status(
    request: PairedImportStatusRequest,
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Returns only safe retry/deduplication state to the trusted local BFF."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN_NOT_ADMIN",
                "message": "Admin privileges required",
            },
        )
    if not _valid_sha256(request.import_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "PAIRED_IMPORT_HASH_INVALID",
                "message": "Invalid paired import hash",
            },
        )
    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            """SELECT p.status AS import_status, p.job_id,
                      j.status AS job_status, j.stage AS job_stage, j.progress
               FROM rag_paired_imports p
               LEFT JOIN job_queue j ON j.id = p.job_id
               WHERE p.import_hash = $1""",
            request.import_hash,
        )
    if not row:
        return {"found": False}
    return {
        "found": True,
        "import_status": row["import_status"],
        "job_id": str(row["job_id"]) if row["job_id"] else None,
        "job_status": row["job_status"],
        "job_stage": row["job_stage"],
        "progress": float(row["progress"]) if row["progress"] is not None else None,
    }


@router.post("/internal/paired-import/referenced-assets")
async def paired_import_referenced_assets(
    auth_context: AuthContext = Depends(verify_internal_jwt),
):
    """Trusted-BFF-only reference set used for conservative Drive cleanup."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN_NOT_ADMIN",
                "message": "Admin privileges required",
            },
        )
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """SELECT DISTINCT storage_key
               FROM rag_visuals
               WHERE storage_provider = 'google_drive'
                 AND storage_key IS NOT NULL
                 AND btrim(storage_key) <> ''"""
        )
    return {"storage_keys": [row["storage_key"] for row in rows]}


@router.post("/internal/admin/rag")
async def local_admin_rag(
    request: RagAdminRequest, auth_context: AuthContext = Depends(verify_internal_jwt)
):
    """Local-admin control plane. Responses use safe DTOs and never include vectors or keys."""
    if not auth_context.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FORBIDDEN_NOT_ADMIN",
                "message": "Admin privileges required",
            },
        )
    scope = {
        "board_id": request.board_id,
        "class_id": request.class_id,
        "subject_id": request.subject_id,
    }
    try:
        async with get_db_connection() as conn:
            service = LocalAdminService(conn)
            if request.operation == "overview":
                return await service.overview(scope)
            if request.operation == "get_chapter_visuals":
                return await service.get_chapter_visuals(
                    scope=scope, chapter_id=request.chapter_id or ""
                )
            if request.operation == "delete_chapter":
                return await service.delete_chapter(
                    scope=scope,
                    chapter_id=request.chapter_id or "",
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                )
            if request.operation == "visual_stream_ref":
                return await service.visual_stream_reference(
                    version_id=request.corpus_version_id or "",
                    scope=scope,
                    visual_id=request.visual_id or "",
                )
            if request.operation == "inspect_version":
                return await service.inspect_version(
                    version_id=request.corpus_version_id or "", scope=scope
                )
            if request.operation == "qa_search":
                result = await RetrievalService(conn).retrieve_named_version(
                    request.question or "",
                    RetrievalScope(**scope, chapter_id=request.chapter_id),
                    request.corpus_version_id or "",
                )
                return {
                    "strength": result.strength,
                    "reason": result.reason,
                    "results": [
                        {
                            "citation": {
                                "citation_id": item.citation.citation_id,
                                "content": item.citation.content,
                                "chapter_id": item.citation.chapter_id,
                                "topic_no": item.citation.topic_no,
                                "topic_title": item.citation.topic_title,
                                "page_start": item.citation.page_start,
                                "page_end": item.citation.page_end,
                                "visuals": [],
                            },
                            "fused_rank": item.fused_rank,
                            "contributions": [
                                {
                                    "channel": contribution.channel,
                                    "rank": contribution.rank,
                                }
                                for contribution in item.contributions
                            ],
                        }
                        for item in result.results
                    ],
                }
            if request.operation == "create_draft":
                return await service.create_draft(
                    active_version_id=request.corpus_version_id or "",
                    scope=scope,
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                )
            if request.operation == "approve_qa":
                await service.approve_qa(
                    version_id=request.corpus_version_id or "",
                    scope=scope,
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                )
                return {"status": "approved"}
            if request.operation in {"activate", "rollback"}:
                await service.activate(
                    version_id=request.corpus_version_id or "",
                    scope=scope,
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                    rollback=request.operation == "rollback",
                )
                return {"status": "active"}
            if request.operation in {
                "add_question",
                "edit_question",
                "delete_question",
            }:
                return await service.edit_question(
                    version_id=request.corpus_version_id or "",
                    scope=scope,
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                    question_id=request.question_id,
                    question_text=request.question_text,
                    chunk_id=request.chunk_id,
                    delete=request.operation == "delete_question",
                )
            if request.operation == "edit_visual":
                return await service.edit_visual(
                    version_id=request.corpus_version_id or "",
                    scope=scope,
                    actor_id=auth_context.uid,
                    request_id=auth_context.request_id,
                    visual_id=request.visual_id or "",
                    title=request.title,
                    description=request.description,
                    review_status=request.review_status,
                    display_policy=request.display_policy,
                )
    except (LocalAdminError, RetrievalScopeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": str(exc), "message": "Local admin operation rejected"},
        )
    raise HTTPException(
        status_code=400,
        detail={
            "code": "ADMIN_OPERATION_INVALID",
            "message": "Unsupported local admin operation",
        },
    )
