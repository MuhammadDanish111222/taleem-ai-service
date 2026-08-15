"""Signed local-admin Ask/prompt/candidate/bank operations for the Run 2 UI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import redis
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.providers.llm.deepseek import DeepSeekConfig, DeepSeekProvider
from app.repositories.ask_repository import AskRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.prompt_cache import SharedPromptCache
from app.repositories.prompt_repository import PostgresPromptRepository
from app.repositories.provider_attempt_repository import ProviderAttemptRepository
from app.repositories.question_bank_repository import QuestionBankRepository
from app.schemas.ask_admin import ApprovedQuestionInput, AskAdminRequest
from app.services.answers.normalization import normalize_question, question_hash
from app.services.answers.retention import CandidateRetentionService
from app.services.jobs.queue import JobQueueService
from app.services.prompts.models import AnswerMode, PromptKey, PromptScope
from app.services.prompts.service import PromptService

router = APIRouter()


def _required(value: Any, code: str):
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(code)
    return value


async def _visual_row_ids(
    conn,
    logical_ids: list[str],
    *,
    board_id: str,
    class_id: str,
    subject_id: str,
    chapter_id: str | None,
) -> list[str]:
    if not logical_ids:
        return []
    rows = await conn.fetch(
        """SELECT v.id::text,v.visual_id FROM rag_visuals v
           JOIN rag_chunks c ON c.id=v.chunk_id
           JOIN rag_corpus_versions cv ON cv.id=c.corpus_version_id
           JOIN rag_corpora co ON co.id=cv.corpus_id
           WHERE v.visual_id=ANY($1::text[]) AND v.review_status='approved'
             AND v.display_policy IN ('always','llm_decide')
             AND co.board_id=$2 AND co.class_id=$3 AND co.subject_id=$4
             AND ($5::text IS NULL OR c.chapter_id=$5)""",
        logical_ids,
        board_id,
        class_id,
        subject_id,
        chapter_id,
    )
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["visual_id"], []).append(row["id"])
    if (
        len(logical_ids) != len(set(logical_ids))
        or set(logical_ids) != set(grouped)
        or any(len(grouped[item]) != 1 for item in logical_ids)
    ):
        raise ValueError("VISUAL_LINK_NOT_REVIEWED")
    return [grouped[item][0] for item in logical_ids]


async def _validated_citation_ids(conn, value: ApprovedQuestionInput) -> list[str]:
    citation_ids = [str(item) for item in value.citation_ids]
    if not citation_ids:
        return []
    count = await conn.fetchval(
        """SELECT COUNT(DISTINCT c.id)
           FROM rag_chunks c
           JOIN rag_corpus_versions cv ON cv.id=c.corpus_version_id
           JOIN rag_corpora co ON co.id=cv.corpus_id
           WHERE c.id=ANY($1::uuid[])
             AND co.board_id=$2 AND co.class_id=$3 AND co.subject_id=$4
             AND ($5::text IS NULL OR c.chapter_id=$5)""",
        citation_ids,
        value.board_id,
        value.class_id,
        value.subject_id,
        value.chapter_id,
    )
    if count != len(citation_ids):
        raise ValueError("CITATION_LINK_OUTSIDE_SCOPE")
    return citation_ids


async def _create_approved(
    conn,
    bank: QuestionBankRepository,
    value: ApprovedQuestionInput,
    *,
    actor_id: str,
    source: str,
) -> str:
    normalized = normalize_question(value.question)
    visual_rows = await _visual_row_ids(
        conn,
        value.visual_ids,
        board_id=value.board_id,
        class_id=value.class_id,
        subject_id=value.subject_id,
        chapter_id=value.chapter_id,
    )
    citation_ids = await _validated_citation_ids(conn, value)
    return await bank.create_approved_revision(
        actor_id=actor_id,
        board_id=value.board_id,
        class_id=value.class_id,
        subject_id=value.subject_id,
        chapter_id=value.chapter_id,
        answer_mode=value.answer_mode,
        answer_style=value.answer_style,
        difficulty=value.difficulty,
        marks=value.marks,
        question_text=value.question,
        normalized_question=normalized,
        question_hash=question_hash(normalized),
        blocks=[item.model_dump() for item in value.blocks],
        source=source,
        citation_chunk_ids=citation_ids,
        visual_row_ids=visual_rows,
        mcq_options=[item.model_dump() for item in value.mcq_options],
    )


async def _enqueue_embeddings(
    conn, *, revision_id: str, variation_id: str | None = None
) -> None:
    suffix = variation_id or "revision"
    await JobQueueService(conn).enqueue_job(
        job_type="question_bank_embeddings",
        payload={
            "revision_id": revision_id,
            "variation_id": variation_id,
        },
        idempotency_key=f"question-bank-embedding:{revision_id}:{suffix}",
    )


@router.post("/internal/admin/ask")
async def ask_admin(
    request: AskAdminRequest,
    auth: AuthContext = Depends(verify_internal_jwt),
):
    if not auth.is_admin or auth.feature != "local_ask_admin":
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN_NOT_ADMIN", "message": "Admin required"},
        )
    try:
        async with get_db_connection() as conn:
            prompts = PostgresPromptRepository(conn)
            settings = get_settings()
            prompt_service = PromptService(
                prompts,
                cache=SharedPromptCache(
                    conn,
                    redis.Redis.from_url(settings.REDIS_URL, decode_responses=True),
                    ttl_seconds=settings.PROMPT_CACHE_TTL_SECONDS,
                ),
                provider=DeepSeekProvider(
                    DeepSeekConfig(
                        api_key=settings.DEEPSEEK_API_KEY,
                        model=settings.DEEPSEEK_MODEL,
                        base_url=settings.DEEPSEEK_API_URL.removesuffix(
                            "/chat/completions"
                        ),
                        timeout_seconds=settings.DEEPSEEK_TIMEOUT_SECONDS,
                        max_retries=settings.DEEPSEEK_MAX_RETRIES,
                        max_output_tokens=settings.DEEPSEEK_MAX_OUTPUT_TOKENS,
                        max_input_characters=(settings.DEEPSEEK_MAX_INPUT_CHARACTERS),
                    ),
                    attempt_recorder=ProviderAttemptRepository(conn),
                ),
            )
            scope = PromptScope(
                board_id=request.board_id,
                class_id=request.class_id,
                subject_id=request.subject_id,
            )
            if request.operation == "prompt_history":
                records = await prompt_service.list_history(
                    prompt_key=PromptKey(
                        _required(request.prompt_key, "PROMPT_KEY_REQUIRED")
                    ),
                    answer_mode=AnswerMode(
                        _required(request.answer_mode, "ANSWER_MODE_REQUIRED")
                    ),
                    scope=scope,
                    limit=request.limit,
                )
                return {"items": [asdict(record) for record in records]}
            if request.operation == "prompt_create_draft":
                record = await prompt_service.create_draft(
                    prompt_key=PromptKey(
                        _required(request.prompt_key, "PROMPT_KEY_REQUIRED")
                    ),
                    answer_mode=AnswerMode(
                        _required(request.answer_mode, "ANSWER_MODE_REQUIRED")
                    ),
                    scope=scope,
                    content=_required(request.content, "PROMPT_CONTENT_REQUIRED"),
                    actor_id=auth.uid,
                )
                return {"prompt_id": record.id, "version": record.version}
            if request.operation == "prompt_update_draft":
                record = await prompt_service.update_draft(
                    prompt_id=str(_required(request.prompt_id, "PROMPT_ID_REQUIRED")),
                    content=_required(request.content, "PROMPT_CONTENT_REQUIRED"),
                    actor_id=auth.uid,
                )
                return {
                    "prompt_id": record.id,
                    "version": record.version,
                    "status": record.status,
                }
            if request.operation == "prompt_test_draft":
                prompt_id = str(_required(request.prompt_id, "PROMPT_ID_REQUIRED"))
                result = await prompt_service.test_draft(
                    prompt_id=prompt_id,
                    question=_required(request.question, "QUESTION_REQUIRED"),
                    actor_id=auth.uid,
                )
                await AuditRepository(conn).create_audit_log(
                    actor_id=auth.uid,
                    action="prompt.draft_tested",
                    target_type="prompt",
                    target_id=prompt_id,
                    after_value={
                        "provider": result.provider,
                        "model": result.model,
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "latency_ms": result.latency_ms,
                    },
                )
                return {
                    "document": result.document,
                    "provider": result.provider,
                    "model": result.model,
                    "usage": asdict(result.usage),
                    "latency_ms": result.latency_ms,
                }
            if request.operation in {"prompt_activate", "prompt_rollback"}:
                prompt_id = str(_required(request.prompt_id, "PROMPT_ID_REQUIRED"))
                result = (
                    await prompt_service.activate(
                        prompt_id=prompt_id, actor_id=auth.uid
                    )
                    if request.operation == "prompt_activate"
                    else await prompt_service.rollback(
                        target_prompt_id=prompt_id, actor_id=auth.uid
                    )
                )
                return {"active_prompt_id": result.active.id}

            if request.operation == "source_policy_get":
                row = await conn.fetchrow(
                    """SELECT class_id,subject_id,semantic_reuse_enabled,
                              semantic_distance_threshold
                       FROM ask_source_policies
                       WHERE (class_id=$1 AND subject_id=$2)
                          OR (class_id IS NULL AND subject_id=$2)
                          OR (class_id IS NULL AND subject_id IS NULL)
                       ORDER BY CASE
                         WHEN class_id=$1 AND subject_id=$2 THEN 1
                         WHEN class_id IS NULL AND subject_id=$2 THEN 2
                         ELSE 3 END
                       LIMIT 1""",
                    request.class_id,
                    request.subject_id,
                )
                distance = (
                    float(row["semantic_distance_threshold"])
                    if row and row["semantic_distance_threshold"] is not None
                    else 0.18
                )
                return {
                    "scope": {
                        "class_id": row["class_id"] if row else None,
                        "subject_id": row["subject_id"] if row else None,
                    },
                    "semantic_reuse_enabled": bool(
                        row and row["semantic_reuse_enabled"]
                    ),
                    "semantic_similarity_threshold": round(1.0 - distance, 4),
                }
            if request.operation == "source_policy_set_semantic_threshold":
                similarity = float(request.semantic_similarity_threshold)
                row = await conn.fetchrow(
                    """INSERT INTO ask_source_policies(
                         class_id,subject_id,allow_general,semantic_reuse_enabled,
                         semantic_distance_threshold,updated_by
                       ) VALUES($1,$2,FALSE,TRUE,$3,$4)
                       ON CONFLICT (COALESCE(class_id, ''),COALESCE(subject_id, ''))
                       DO UPDATE SET semantic_reuse_enabled=TRUE,
                                     semantic_distance_threshold=EXCLUDED.semantic_distance_threshold,
                                     updated_by=EXCLUDED.updated_by,updated_at=NOW()
                       RETURNING class_id,subject_id,semantic_reuse_enabled,
                                 semantic_distance_threshold""",
                    request.class_id,
                    request.subject_id,
                    1.0 - similarity,
                    auth.uid,
                )
                await AuditRepository(conn).create_audit_log(
                    actor_id=auth.uid,
                    action="ask.semantic_threshold_changed",
                    target_type="ask_source_policy",
                    target_id=f"{request.class_id or 'subject-global'}:{request.subject_id}",
                    after_value={"semantic_similarity_threshold": similarity},
                )
                return {
                    "scope": {
                        "class_id": row["class_id"],
                        "subject_id": row["subject_id"],
                    },
                    "semantic_reuse_enabled": bool(row["semantic_reuse_enabled"]),
                    "semantic_similarity_threshold": round(
                        1.0 - float(row["semantic_distance_threshold"]), 4
                    ),
                }

            asks = AskRepository(conn)
            bank = QuestionBankRepository(conn)
            if request.operation == "candidate_list":
                return {
                    "items": await asks.list_pending(
                        board_id=request.board_id,
                        class_id=request.class_id,
                        subject_id=request.subject_id,
                        chapter_id=request.chapter_id,
                        answer_mode=(
                            request.answer_mode.value if request.answer_mode else None
                        ),
                        answer_source=request.answer_source,
                        source_feature=request.source_feature,
                        provider=request.provider,
                        age_days=request.age_days,
                        limit=request.limit,
                    )
                }
            if request.operation == "candidate_inspect":
                item = await asks.inspect_candidate(
                    str(_required(request.candidate_id, "CANDIDATE_ID_REQUIRED"))
                )
                if item is None:
                    raise LookupError("CANDIDATE_NOT_FOUND")
                return {"item": item}
            if request.operation == "candidate_reject":
                candidate_id = str(
                    _required(request.candidate_id, "CANDIDATE_ID_REQUIRED")
                )
                reason = _required(
                    request.rejection_reason, "REJECTION_REASON_REQUIRED"
                )
                async with conn.transaction():
                    before = await asks.inspect_candidate(candidate_id)
                    if before is None or before["review_status"] != "pending":
                        raise ValueError("CANDIDATE_NOT_PENDING")
                    await asks.reject_candidate(
                        answer_id=candidate_id, reason=reason, actor_id=auth.uid
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="candidate.rejected",
                        target_type="ai_answer",
                        target_id=candidate_id,
                        before_value={"review_status": before["review_status"]},
                        after_value={
                            "review_status": "rejected",
                            "reason": reason,
                        },
                    )
                return {"status": "rejected"}
            if request.operation == "candidate_retention_preview":
                counts = await CandidateRetentionService(conn).preview()
                return {
                    "dry_run": True,
                    "eligible_answers": counts.eligible_answers,
                    "eligible_requests_without_answer": (
                        counts.eligible_requests_without_answer
                    ),
                    "eligible_total": counts.total,
                }
            if request.operation == "candidate_retention_cleanup":
                _required(request.reason, "RETENTION_AUTHORIZATION_REASON_REQUIRED")
                counts = await CandidateRetentionService(conn).cleanup(
                    actor_id=auth.uid,
                    limit=request.limit,
                )
                return {
                    "dry_run": False,
                    "deleted_answers": counts.eligible_answers,
                    "deleted_requests_without_answer": (
                        counts.eligible_requests_without_answer
                    ),
                    "deleted_total": counts.total,
                }
            if request.operation == "candidate_approve":
                candidate_id = str(
                    _required(request.candidate_id, "CANDIDATE_ID_REQUIRED")
                )
                value = _required(
                    request.approved_question, "APPROVED_QUESTION_REQUIRED"
                )
                async with conn.transaction():
                    candidate = await asks.inspect_candidate(candidate_id)
                    if candidate is None or candidate["review_status"] != "pending":
                        raise ValueError("CANDIDATE_NOT_PENDING")
                    revision_id = await _create_approved(
                        conn,
                        bank,
                        value,
                        actor_id=auth.uid,
                        source="generated_candidate",
                    )
                    await _enqueue_embeddings(conn, revision_id=revision_id)
                    await asks.approve_candidate(
                        answer_id=candidate_id,
                        revision_id=revision_id,
                        actor_id=auth.uid,
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="candidate.approved",
                        target_type="ai_answer",
                        target_id=candidate_id,
                        before_value={"review_status": "pending"},
                        after_value={
                            "review_status": "approved",
                            "approved_revision_id": revision_id,
                        },
                    )
                return {"status": "approved", "revision_id": revision_id}
            if request.operation == "bank_list":
                return {
                    "items": await bank.list_approved(
                        board_id=request.board_id,
                        class_id=request.class_id,
                        subject_id=request.subject_id,
                        chapter_id=request.chapter_id,
                        answer_mode=(
                            request.answer_mode.value if request.answer_mode else None
                        ),
                        source=request.bank_source,
                        limit=request.limit,
                    )
                }
            if request.operation == "bank_create":
                value = _required(
                    request.approved_question, "APPROVED_QUESTION_REQUIRED"
                )
                async with conn.transaction():
                    revision_id = await _create_approved(
                        conn, bank, value, actor_id=auth.uid, source="admin_authored"
                    )
                    await _enqueue_embeddings(conn, revision_id=revision_id)
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.admin_approved_created",
                        target_type="question_revision",
                        target_id=revision_id,
                        after_value={
                            "review_status": "approved",
                            "board_id": value.board_id,
                            "class_id": value.class_id,
                            "subject_id": value.subject_id,
                            "answer_mode": value.answer_mode.value,
                            "difficulty": value.difficulty,
                            "marks": value.marks,
                        },
                    )
                return {"status": "approved", "revision_id": revision_id}
            if request.operation == "bank_import":
                import_key = _required(request.import_key, "IMPORT_KEY_REQUIRED")
                if not request.import_questions:
                    raise ValueError("IMPORT_QUESTIONS_REQUIRED")
                import_questions = [
                    item.as_approved_question(
                        board_id=_required(request.board_id, "IMPORT_BOARD_REQUIRED"),
                        class_id=_required(request.class_id, "IMPORT_CLASS_REQUIRED"),
                        subject_id=_required(
                            request.subject_id, "IMPORT_SUBJECT_REQUIRED"
                        ),
                        chapter_id=_required(
                            request.chapter_id, "IMPORT_CHAPTER_REQUIRED"
                        ),
                    )
                    for item in request.import_questions
                ]
                payload = json.dumps(
                    [item.model_dump(mode="json") for item in import_questions],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                payload_hash = hashlib.sha256(payload.encode()).hexdigest()
                async with conn.transaction():
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                        import_key,
                    )
                    existing = await conn.fetchrow(
                        """SELECT payload_hash,revision_ids FROM question_bank_imports
                           WHERE import_key=$1 FOR UPDATE""",
                        import_key,
                    )
                    if existing:
                        if existing["payload_hash"] != payload_hash:
                            raise ValueError("IMPORT_KEY_CONFLICT")
                        return {
                            "status": "already_imported",
                            "revision_ids": [
                                str(item) for item in existing["revision_ids"]
                            ],
                        }
                    # Confirm every external visual reference before creating
                    # any bank rows.  The enclosing transaction also protects
                    # against a later database failure.
                    for index, item in enumerate(import_questions, start=1):
                        try:
                            await _visual_row_ids(
                                conn,
                                item.visual_ids,
                                board_id=item.board_id,
                                class_id=item.class_id,
                                subject_id=item.subject_id,
                                chapter_id=item.chapter_id,
                            )
                        except ValueError as exc:
                            # This code is deliberately safe for the local BFF
                            # to surface; it contains no source metadata.
                            raise ValueError(f"IMPORT_QUESTION_{index}_{exc}") from None
                    revision_ids = []
                    for item in import_questions:
                        revision_id = await _create_approved(
                            conn,
                            bank,
                            item,
                            actor_id=auth.uid,
                            source="admin_import",
                        )
                        await _enqueue_embeddings(conn, revision_id=revision_id)
                        revision_ids.append(revision_id)
                    await conn.execute(
                        """INSERT INTO question_bank_imports(
                             import_key,payload_hash,revision_ids,actor_id
                           ) VALUES($1,$2,$3::uuid[],$4)""",
                        import_key,
                        payload_hash,
                        revision_ids,
                        auth.uid,
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.import_approved",
                        target_type="question_import",
                        target_id=import_key,
                        after_value={
                            "payload_hash": payload_hash,
                            "approved_count": len(revision_ids),
                        },
                    )
                return {"status": "approved", "revision_ids": revision_ids}
            if request.operation == "bank_view":
                revision_id = str(
                    _required(request.revision_id, "REVISION_ID_REQUIRED")
                )
                revision = await bank.get_revision(revision_id)
                if revision is None:
                    raise LookupError("REVISION_NOT_FOUND")
                return {
                    "revision": asdict(revision),
                    "history": await bank.revision_history(revision_id=revision_id),
                }
            if request.operation == "bank_history":
                history = await bank.revision_history(
                    revision_id=(
                        str(request.revision_id) if request.revision_id else None
                    ),
                    question_id=(
                        str(request.question_id) if request.question_id else None
                    ),
                )
                if history is None:
                    raise LookupError("QUESTION_HISTORY_NOT_FOUND")
                return history
            if request.operation == "bank_archive":
                revision_id = str(
                    _required(request.revision_id, "REVISION_ID_REQUIRED")
                )
                reason = _required(request.reason, "ARCHIVE_REASON_REQUIRED")
                async with conn.transaction():
                    before = await bank.revision_history(revision_id=revision_id)
                    if before is None:
                        raise LookupError("REVISION_NOT_FOUND")
                    archived = await bank.archive_revision(revision_id=revision_id)
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.revision_archived",
                        target_type="question_revision",
                        target_id=revision_id,
                        before_value={
                            "review_status": "approved",
                            "version_no": archived["version_no"],
                        },
                        after_value={
                            "review_status": "archived",
                            "reason": reason,
                        },
                    )
                return {"status": "archived"}
            if request.operation == "bank_add_variation":
                revision_id = str(
                    _required(request.revision_id, "REVISION_ID_REQUIRED")
                )
                variation = _required(request.variation, "VARIATION_REQUIRED")
                normalized = normalize_question(variation)
                async with conn.transaction():
                    variation_id = await bank.add_variation(
                        revision_id=revision_id,
                        variation_text=variation,
                        normalized_variation=normalized,
                        variation_hash=question_hash(normalized),
                        actor_id=auth.uid,
                    )
                    await _enqueue_embeddings(
                        conn,
                        revision_id=revision_id,
                        variation_id=variation_id,
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.variation_added",
                        target_type="question_revision",
                        target_id=revision_id,
                        after_value={"variation_id": variation_id},
                    )
                return {"variation_id": variation_id, "embedding_status": "pending"}
            if request.operation == "bank_set_variation_active":
                variation_id = str(
                    _required(request.variation_id, "VARIATION_ID_REQUIRED")
                )
                active = _required(request.active, "VARIATION_ACTIVE_REQUIRED")
                async with conn.transaction():
                    updated = await bank.set_variation_active(
                        variation_id=variation_id, active=bool(active)
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.variation_state_changed",
                        target_type="question_variation",
                        target_id=variation_id,
                        after_value={"active": bool(active)},
                    )
                return updated
            if request.operation == "bank_requeue_embedding":
                revision_id = str(
                    _required(request.revision_id, "REVISION_ID_REQUIRED")
                )
                variation_id = (
                    str(request.variation_id) if request.variation_id else None
                )
                async with conn.transaction():
                    await bank.reset_embedding(
                        revision_id=revision_id,
                        variation_id=variation_id,
                    )
                    await _enqueue_embeddings(
                        conn,
                        revision_id=revision_id,
                        variation_id=variation_id,
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.embedding_requeued",
                        target_type=(
                            "question_variation"
                            if variation_id
                            else "question_revision"
                        ),
                        target_id=variation_id or revision_id,
                        after_value={"embedding_status": "pending"},
                    )
                return {"embedding_status": "pending"}
            if request.operation == "bank_set_visuals":
                revision_id = str(
                    _required(request.revision_id, "REVISION_ID_REQUIRED")
                )
                revision = await bank.get_revision(revision_id)
                if revision is None:
                    raise LookupError("REVISION_NOT_FOUND")
                referenced_visuals = {
                    block["visual_id"]
                    for block in revision.blocks
                    if block.get("type") == "visual_ref"
                }
                if referenced_visuals != set(request.visual_ids):
                    raise ValueError("VISUAL_BLOCK_LINK_MISMATCH")
                visual_rows = await _visual_row_ids(
                    conn,
                    request.visual_ids,
                    board_id=revision.board_id,
                    class_id=revision.class_id,
                    subject_id=revision.subject_id,
                    chapter_id=revision.chapter_id,
                )
                async with conn.transaction():
                    await bank.set_visual_links(
                        revision_id=revision_id, visual_row_ids=visual_rows
                    )
                    await AuditRepository(conn).create_audit_log(
                        actor_id=auth.uid,
                        action="bank.visual_links_changed",
                        target_type="question_revision",
                        target_id=revision_id,
                        after_value={"visual_ids": request.visual_ids},
                    )
                return {"status": "updated"}
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": str(exc), "message": "Admin object not found"},
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": str(exc), "message": "Admin operation rejected"},
        ) from None
    raise HTTPException(
        status_code=400,
        detail={"code": "ADMIN_OPERATION_INVALID", "message": "Invalid operation"},
    )
