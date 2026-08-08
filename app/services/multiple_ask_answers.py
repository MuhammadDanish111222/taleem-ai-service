"""Railway-owned, restart-safe answers for extracted Multiple Ask items."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import TypeAdapter, ValidationError

from app.core.config import get_settings
from app.providers.embeddings.voyage import (
    VoyageEmbeddingConfiguration,
    VoyageEmbeddingProvider,
)
from app.providers.llm.deepseek import DeepSeekProviderError
from app.repositories.ask_repository import AskRepository
from app.repositories.multiple_ask_repository import MultipleAskRepository
from app.repositories.rag_repository import RagRepository
from app.schemas.ask import (
    AnswerBlock,
    AnswerMode,
    AnswerSource,
    CitationDto,
    VisualDto,
)
from app.services.answers.context import assemble_context
from app.services.answers.generate import (
    _MAX_COMPLETE_TOPIC_CHUNKS,
    AskService,
    AskServiceError,
    _normalize_provider_block_aliases,
)
from app.services.answers.validation import (
    AnswerValidationError,
    validate_generated_answer,
)
from app.services.jobs.queue import JobQueueService
from app.services.multiple_ask import MultipleAskError
from app.services.prompts.models import AnswerMode as PromptAnswerMode
from app.services.prompts.models import PromptKey, PromptScope
from app.services.retrieval.evidence import RetrievalScope
from app.services.usage.service import UsageService

logger = logging.getLogger(__name__)

_BLOCKS = TypeAdapter(list[AnswerBlock])


class MultipleAskAnswerError(MultipleAskError):
    pass


class MultipleAskAnswerService:
    """Answer ready items in small durable batches without duplicate LLM calls."""

    def __init__(self, conn: Any, *, ask_service: AskService | None = None):
        self._conn = conn
        self._repo = MultipleAskRepository(conn)
        self._ask = ask_service or AskService(conn)
        self._asks = AskRepository(conn)

    async def start_for_job(
        self, *, job_id: str, uid: str | None = None
    ) -> dict[str, Any]:
        """Idempotently enqueue the one Railway answer job after all corrections."""
        uid_hash = UsageService().uid_hash(uid) if uid is not None else None
        async with self._conn.transaction():
            if uid_hash is None:
                job = await self._conn.fetchrow(
                    "SELECT * FROM multiple_ask_jobs WHERE id=$1::uuid FOR UPDATE",
                    job_id,
                )
                job = dict(job) if job else None
            else:
                job = await self._repo.lock_owned_job(job_id=job_id, uid_hash=uid_hash)
            if job is None:
                raise MultipleAskAnswerError(
                    "MULTIPLE_ASK_JOB_NOT_FOUND", status_code=404
                )
            items = await self._repo.lock_job_items(job_id=job_id)
            if any(item["item_status"] == "needs_correction" for item in items):
                raise MultipleAskAnswerError(
                    "MULTIPLE_ASK_ANSWERS_BLOCKED_BY_CORRECTION", status_code=409
                )
            if job["workflow_status"] == "answering":
                return {
                    "job_id": job_id,
                    "workflow_status": "answering",
                    "queue_status": "queued",
                }
            if job["workflow_status"] in {
                "completed",
                "partially_completed",
                "failed",
                "cancelled",
            }:
                return {
                    "job_id": job_id,
                    "workflow_status": job["workflow_status"],
                    "queue_status": "succeeded",
                }
            if job["workflow_status"] != "ready_to_answer":
                raise MultipleAskAnswerError(
                    "MULTIPLE_ASK_JOB_NOT_READY_TO_ANSWER", status_code=409
                )
            epoch = int(job["answer_epoch"]) + 1
            queue = await JobQueueService(self._conn).enqueue_job(
                job_type="multiple_ask_answer",
                payload={
                    "multiple_ask_session_id": str(job["upload_session_id"]),
                    "epoch": epoch,
                },
                idempotency_key=f"multiple-ask-answer:{job_id}:{epoch}",
            )
            if not await self._repo.start_answering(
                job_id=job_id, queue_job_id=str(queue["id"]), epoch=epoch
            ):
                raise MultipleAskAnswerError(
                    "MULTIPLE_ASK_ANSWER_STATE_CONFLICT", status_code=409
                )
            return {
                "job_id": job_id,
                "workflow_status": "answering",
                "queue_status": queue["status"],
            }

    async def answer(self, *, session_id: str, epoch: int) -> str | dict[str, Any]:
        """Runs a small batch of items; returns _next_job to chain until all items reach a terminal state."""
        batch_size = get_settings().MULTIPLE_ASK_ANSWER_BATCH_SIZE
        async with self._conn.transaction():
            job = await self._repo.lock_answer_context(session_id)
            if job is None:
                raise MultipleAskAnswerError("MULTIPLE_ASK_PARENT_NOT_FOUND")
            if job["workflow_status"] != "answering" or job["answer_epoch"] != epoch:
                return str(job["workflow_status"])
            items = await self._repo.lock_job_items(job_id=str(job["id"]))
            if any(item["item_status"] == "needs_correction" for item in items):
                raise MultipleAskAnswerError(
                    "MULTIPLE_ASK_ANSWERS_BLOCKED_BY_CORRECTION"
                )

        remaining = [
            item
            for item in items
            if item["item_status"] in {"ready_to_answer", "answering"}
        ]
        if not remaining:
            async with self._conn.transaction():
                return await self._repo.finish_answers(str(job["id"]))

        current_batch = remaining[:batch_size]

        # Step 1: Check existing completions and exact/variation matches in approved bank
        missed_items: list[tuple[dict[str, Any], Any]] = []
        for item in current_batch:
            try:
                request = await self._candidate_request(job, item)
                completed = await self._existing_completion(item, request)
                if completed:
                    continue
                async with self._conn.transaction():
                    await self._repo.mark_item_answering(str(item["id"]))
                mode = AnswerMode(item["answer_mode"])
                approved = await self._ask._bank.find_exact(
                    board_id=job["board_id"],
                    class_id=job["class_id"],
                    subject_id=job["subject_id"],
                    chapter_id=job["chapter_id"],
                    answer_mode=mode,
                    normalized_question=item["normalized_question"],
                )
                if approved is None:
                    approved = await self._ask._bank.find_exact_variation(
                        board_id=job["board_id"],
                        class_id=job["class_id"],
                        subject_id=job["subject_id"],
                        chapter_id=job["chapter_id"],
                        answer_mode=mode,
                        normalized_question=item["normalized_question"],
                    )
                if approved is not None:
                    answer = await self._asks.complete(
                        ai_request_id=str(request["id"]),
                        answer_source=AnswerSource.APPROVED_BANK.value,
                        blocks=list(approved.blocks),
                        citations=list(approved.citations),
                        visual_ids=[visual["visual_id"] for visual in approved.visuals],
                        prompt_version="approved-bank",
                        corpus_version_id=None,
                        provider=None,
                        model=None,
                        approved_revision_id=approved.revision_id,
                    )
                    async with self._conn.transaction():
                        await self._repo.complete_answer_item(
                            item_id=str(item["id"]),
                            ai_answer_id=str(answer["id"]),
                            answer_source=AnswerSource.APPROVED_BANK.value,
                            approved_revision_id=approved.revision_id,
                        )
                    continue

                missed_items.append((item, request))
            except Exception as exc:
                code = getattr(exc, "code", str(exc))
                if hasattr(code, "value"):
                    code = code.value
                logger.warning(
                    "Error during pre-check for Multiple Ask item_id=%s: %s",
                    item.get("id"),
                    code,
                )
                await self._fail_item(item, str(code))

        # Step 2: For missed questions in the batch, fetch Voyage query embeddings in ONE call
        query_vectors: list[list[float]] = []
        if missed_items:
            missed_texts = [" ".join(item["source_text"].split()) for item, _ in missed_items]
            scope = RetrievalScope(
                job["board_id"], job["class_id"], job["subject_id"], job["chapter_id"]
            )
            active_version = await RagRepository(self._conn).get_active_corpus_version(
                job["board_id"], job["class_id"], job["subject_id"]
            )
            if active_version is not None:
                config = VoyageEmbeddingConfiguration(
                    model=active_version["embedding_model"],
                    revision=active_version["embedding_revision"],
                    dimensions=active_version["embedding_dim"],
                )
                provider = VoyageEmbeddingProvider(config, input_type="query")
                query_vectors = await provider.embed_queries(missed_texts)
            else:
                query_vectors = [[] for _ in missed_items]

        # Step 3: Process retrieval + DeepSeek for each missed item
        for (item, request), vector in zip(missed_items, query_vectors, strict=False):
            try:
                mode = AnswerMode(item["answer_mode"])
                policy = await self._ask._source_policy(job["class_id"], job["subject_id"])
                approved = None
                if (
                    vector
                    and policy["semantic_reuse_enabled"]
                    and policy["semantic_distance_threshold"] is not None
                ):
                    approved = await self._ask._bank.find_semantic(
                        query_embedding=vector,
                        evaluated_threshold=float(policy["semantic_distance_threshold"]),
                        enabled=True,
                        board_id=job["board_id"],
                        class_id=job["class_id"],
                        subject_id=job["subject_id"],
                        chapter_id=job["chapter_id"],
                        answer_mode=mode,
                    )
                if approved is not None:
                    answer = await self._asks.complete(
                        ai_request_id=str(request["id"]),
                        answer_source=AnswerSource.APPROVED_BANK.value,
                        blocks=list(approved.blocks),
                        citations=list(approved.citations),
                        visual_ids=[visual["visual_id"] for visual in approved.visuals],
                        prompt_version="approved-bank",
                        corpus_version_id=None,
                        provider=None,
                        model=None,
                        approved_revision_id=approved.revision_id,
                    )
                    async with self._conn.transaction():
                        await self._repo.complete_answer_item(
                            item_id=str(item["id"]),
                            ai_answer_id=str(answer["id"]),
                            answer_source=AnswerSource.APPROVED_BANK.value,
                            approved_revision_id=approved.revision_id,
                        )
                    continue

                # Unified retrieval with pre-computed query vector
                scope = RetrievalScope(
                    job["board_id"], job["class_id"], job["subject_id"], job["chapter_id"]
                )
                evidence = await self._ask._retrieval.retrieve(
                    item["source_text"], scope, query_vector=vector if vector else None
                )
                active = await RagRepository(self._conn).get_active_corpus_version(
                    job["board_id"], job["class_id"], job["subject_id"]
                )

                if mode == AnswerMode.MCQ:
                    await self._answer_single_mcq(job, item, request, evidence, active, policy)
                else:
                    await self._answer_single_short_or_long(
                        job, item, request, mode, evidence, active, policy
                    )
            except (
                ValidationError,
                AnswerValidationError,
                MultipleAskAnswerError,
                AskServiceError,
                DeepSeekProviderError,
            ) as exc:
                code = getattr(exc, "code", str(exc))
                if hasattr(code, "value"):
                    code = code.value
                logger.warning(
                    "Handled error for Multiple Ask item_id=%s job_id=%s: %s",
                    item.get("id"),
                    job.get("id"),
                    code,
                )
                await self._fail_item(item, str(code))
            except Exception:
                # Transient network / 5xx / timeout errors bubble to queue runtime for retry backoff
                logger.exception(
                    "Transient error answering Multiple Ask item_id=%s job_id=%s",
                    item.get("id"),
                    job.get("id"),
                )
                raise

        # Step 4: If remaining items exist after this batch, chain next job via _next_job
        if len(remaining) > len(current_batch):
            return {
                "workflow_status": "answering",
                "_next_job": {
                    "job_type": "multiple_ask_answer",
                    "payload": {
                        "multiple_ask_session_id": session_id,
                        "epoch": epoch,
                    },
                    "idempotency_key": (
                        f"multiple-ask-answer:{job['id']}:{epoch}:{len(remaining) - len(current_batch)}"
                    ),
                },
            }

        async with self._conn.transaction():
            return await self._repo.finish_answers(str(job["id"]))

    async def mark_queue_failure(self, session_id: str) -> None:
        """Make exhausted queue retries visible as terminal item/job state."""
        async with self._conn.transaction():
            job = await self._repo.lock_answer_context(session_id)
            if job is None or job["workflow_status"] != "answering":
                return
            for item in await self._repo.lock_job_items(job_id=str(job["id"])):
                if item["item_status"] in {"ready_to_answer", "answering"}:
                    await self._repo.fail_answer_item(
                        item_id=str(item["id"]), error_code="MULTIPLE_ASK_ANSWER_FAILED"
                    )
            await self._repo.finish_answers(str(job["id"]))

    async def _answer_single_short_or_long(
        self,
        job: dict[str, Any],
        item: dict[str, Any],
        request: dict[str, Any],
        mode: AnswerMode,
        evidence: Any,
        active: Any,
        policy: dict[str, Any],
    ) -> None:
        source = AnswerSource.SYLLABUS_GROUNDED
        citations: dict[str, CitationDto] = {}
        visuals: dict[str, VisualDto] = {}
        if evidence.results:
            results = evidence.results[:1]
            if active:
                request_context = SimpleNamespace(
                    board_id=job["board_id"],
                    class_id=job["class_id"],
                    subject_id=job["subject_id"],
                    answer_mode=mode,
                )
                results = await self._ask._expand_answer_topics(
                    request=request_context,
                    corpus_version_id=str(active["id"]),
                    ranked_results=evidence.results,
                )
            context = assemble_context(
                results,
                max_chunks=_MAX_COMPLETE_TOPIC_CHUNKS,
                max_characters=24000 if mode is AnswerMode.LONG else 12000,
            )
            citations = {
                value.citation.citation_id: CitationDto(
                    citation_id=value.citation.citation_id,
                    chapter_id=value.citation.chapter_id,
                    topic_no=value.citation.topic_no,
                    topic_title=value.citation.topic_title,
                    page_start=value.citation.page_start,
                    page_end=value.citation.page_end,
                )
                for value in results
            }
            visual_order = 0
            for result in results:
                for visual in result.citation.visuals:
                    if visual.visual_id in visuals:
                        raise MultipleAskAnswerError("RETRIEVED_VISUAL_ID_AMBIGUOUS")
                    visuals[visual.visual_id] = VisualDto(
                        visual_id=visual.visual_id,
                        title=visual.title,
                        description=visual.description,
                        display_policy=visual.display_policy,
                        display_order=visual_order,
                    )
                    visual_order += 1
            prompt_key = PromptKey.ASK_GROUNDED
            payload: dict[str, Any] = {
                "question": item["source_text"],
                "answer_mode": mode.value,
                "answer_style": "exam_style",
                "allow_general": bool(policy["allow_general"]),
                "evidence": [asdict(value) for value in context],
                "allowed_visuals": [
                    visual.model_dump() for visual in visuals.values()
                ],
            }
        elif policy["allow_general"]:
            source = AnswerSource.GENERAL_KNOWLEDGE
            citations, visuals, prompt_key = {}, {}, PromptKey.ASK_GENERAL
            payload = {
                "question": item["source_text"],
                "answer_mode": mode.value,
                "answer_style": "exam_style",
            }
        else:
            raise MultipleAskAnswerError("GENERAL_AI_DISABLED", status_code=409)

        resolved = await self._ask._prompts.resolve_active(
            prompt_key=prompt_key,
            answer_mode=PromptAnswerMode(mode.value),
            scope=PromptScope(
                board_id=job["board_id"],
                class_id=job["class_id"],
                subject_id=job["subject_id"],
            ),
        )
        if self._ask._provider is None:
            raise MultipleAskAnswerError("PROVIDER_UNAVAILABLE")
        generated = await self._ask._provider.generate(
            system_prompt=resolved.system_prompt,
            user_prompt=json.dumps(payload, separators=(",", ":")),
            ai_request_id=str(request["id"]),
            trace_id=str(item["id"]),
        )
        blocks = _BLOCKS.validate_python(
            _normalize_provider_block_aliases(generated.document.get("blocks"))
        )
        cited_ids = generated.document.get("cited_chunk_ids", [])
        if not isinstance(cited_ids, list) or not all(
            isinstance(value, str) for value in cited_ids
        ):
            raise AnswerValidationError("ANSWER_CITATIONS_INVALID")
        if source is AnswerSource.SYLLABUS_GROUNDED and not cited_ids:
            if not policy["allow_general"]:
                raise MultipleAskAnswerError("GENERAL_AI_DISABLED", status_code=409)
            source, citations, visuals = AnswerSource.GENERAL_KNOWLEDGE, {}, {}
        validated = validate_generated_answer(
            source=source,
            blocks=blocks,
            citation_ids=cited_ids,
            allowed_citations=citations,
            allowed_visuals=visuals,
            include_all_allowed_visuals=(
                source is AnswerSource.SYLLABUS_GROUNDED and mode is AnswerMode.LONG
            ),
        )
        answer = await self._asks.complete(
            ai_request_id=str(request["id"]),
            answer_source=source.value,
            blocks=[block.model_dump() for block in validated.blocks],
            citations=[citation.model_dump() for citation in validated.citations],
            visual_ids=[visual.visual_id for visual in validated.visuals],
            prompt_version=f"{resolved.record.id}:{resolved.record.version}",
            corpus_version_id=str(active["id"]) if active else None,
            provider=generated.provider,
            model=generated.model,
            tokens_used=generated.usage.prompt_tokens
            + generated.usage.completion_tokens,
            latency_ms=generated.latency_ms,
        )
        async with self._conn.transaction():
            await self._repo.complete_answer_item(
                item_id=str(item["id"]),
                ai_answer_id=str(answer["id"]),
                answer_source=source.value,
                approved_revision_id=None,
            )

    async def _answer_single_mcq(
        self,
        job: dict[str, Any],
        item: dict[str, Any],
        request: dict[str, Any],
        evidence: Any,
        active: Any,
        policy: dict[str, Any],
    ) -> None:
        source = AnswerSource.SYLLABUS_GROUNDED
        citations: dict[str, CitationDto] = {}
        if evidence.results:
            results = evidence.results[:1]
            if active:
                request_context = SimpleNamespace(
                    board_id=job["board_id"],
                    class_id=job["class_id"],
                    subject_id=job["subject_id"],
                    answer_mode=AnswerMode.MCQ,
                )
                results = await self._ask._expand_answer_topics(
                    request=request_context,
                    corpus_version_id=str(active["id"]),
                    ranked_results=evidence.results,
                )
            context = assemble_context(
                results,
                max_chunks=_MAX_COMPLETE_TOPIC_CHUNKS,
                max_characters=12000,
            )
            citations = {
                value.citation.citation_id: CitationDto(
                    citation_id=value.citation.citation_id,
                    chapter_id=value.citation.chapter_id,
                    topic_no=value.citation.topic_no,
                    topic_title=value.citation.topic_title,
                    page_start=value.citation.page_start,
                    page_end=value.citation.page_end,
                )
                for value in results
            }
            prompt_key = PromptKey.ASK_GROUNDED
            payload: dict[str, Any] = {
                "question": item["source_text"],
                "answer_mode": "mcq",
                "mcq_options": item["mcq_options"],
                "answer_style": "exam_style",
                "allow_general": bool(policy["allow_general"]),
                "evidence": [asdict(value) for value in context],
            }
        elif policy["allow_general"]:
            source = AnswerSource.GENERAL_KNOWLEDGE
            prompt_key = PromptKey.ASK_GENERAL
            payload = {
                "question": item["source_text"],
                "answer_mode": "mcq",
                "mcq_options": item["mcq_options"],
                "answer_style": "exam_style",
            }
        else:
            raise MultipleAskAnswerError("GENERAL_AI_DISABLED", status_code=409)

        resolved = await self._ask._prompts.resolve_active(
            prompt_key=prompt_key,
            answer_mode=PromptAnswerMode.MCQ,
            scope=PromptScope(
                board_id=job["board_id"],
                class_id=job["class_id"],
                subject_id=job["subject_id"],
            ),
        )
        if self._ask._provider is None:
            raise MultipleAskAnswerError("PROVIDER_UNAVAILABLE")
        generated = await self._ask._provider.generate(
            system_prompt=resolved.system_prompt,
            user_prompt=json.dumps(payload, separators=(",", ":")),
            ai_request_id=str(request["id"]),
            trace_id=str(item["id"]),
        )
        blocks = _BLOCKS.validate_python(
            _normalize_provider_block_aliases(generated.document.get("blocks"))
        )
        cited_ids = generated.document.get("cited_chunk_ids", [])
        if not isinstance(cited_ids, list) or not all(
            isinstance(value, str) for value in cited_ids
        ):
            cited_ids = []
        answer = await self._asks.complete(
            ai_request_id=str(request["id"]),
            answer_source=source.value,
            blocks=[block.model_dump() for block in blocks],
            citations=[citation.model_dump() for citation in citations.values()]
            if source == AnswerSource.SYLLABUS_GROUNDED
            else [],
            visual_ids=[],
            prompt_version=f"{resolved.record.id}:{resolved.record.version}",
            corpus_version_id=str(active["id"]) if active else None,
            provider=generated.provider,
            model=generated.model,
            tokens_used=generated.usage.prompt_tokens
            + generated.usage.completion_tokens,
            latency_ms=generated.latency_ms,
        )
        async with self._conn.transaction():
            await self._repo.complete_answer_item(
                item_id=str(item["id"]),
                ai_answer_id=str(answer["id"]),
                answer_source=source.value,
                approved_revision_id=None,
            )

    async def _candidate_request(
        self, job: dict[str, Any], item: dict[str, Any]
    ) -> dict[str, Any]:
        request = await self._asks.create_pending_multiple_ask(
            client_request_id=str(
                uuid5(NAMESPACE_URL, f"multiple-ask:{job['id']}:{item['id']}")
            ),
            uid_hash=job["uid_hash"],
            board_id=job["board_id"],
            class_id=job["class_id"],
            subject_id=job["subject_id"],
            chapter_id=job["chapter_id"],
            answer_mode=item["answer_mode"],
            raw_question=item["source_text"],
            normalized_question=item["normalized_question"],
            question_hash=item["question_hash"],
        )
        async with self._conn.transaction():
            if not await self._repo.link_item_request(
                item_id=str(item["id"]), ai_request_id=str(request["id"])
            ):
                raise MultipleAskAnswerError("MULTIPLE_ASK_ITEM_REQUEST_CONFLICT")
        return request

    async def _existing_completion(
        self, item: dict[str, Any], request: dict[str, Any]
    ) -> bool:
        """Re-links existing completion if worker restarted after persisting answer."""
        existing_answer = await self._asks.answer_by_request_id(str(request["id"]))
        if existing_answer is None:
            return False
        async with self._conn.transaction():
            await self._repo.complete_answer_item(
                item_id=str(item["id"]),
                ai_answer_id=str(existing_answer["id"]),
                answer_source=str(existing_answer["answer_source"]),
                approved_revision_id=None,
            )
        return True

    async def _fail_item(self, item: dict[str, Any], code: str) -> None:
        async with self._conn.transaction():
            await self._repo.fail_answer_item(item_id=str(item["id"]), error_code=code)
