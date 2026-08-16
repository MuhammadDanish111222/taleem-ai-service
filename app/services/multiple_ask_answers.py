"""Railway-owned, restart-safe answers for extracted Multiple Ask items."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from hashlib import sha256
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
from app.services.retrieval.evidence import EvidenceStrength, RetrievalScope
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

        # MCQs are deliberately completed before, and completely outside, the
        # written-answer path.  In particular, no bank, embeddings, retrieval,
        # corpus, citation, or visual work is allowed to precede this batch.
        mcq_items = [item for item in items if item["answer_mode"] == "mcq"]
        pending_mcqs = [
            item
            for item in mcq_items
            if item["item_status"] in {"ready_to_answer", "answering"}
        ]
        if pending_mcqs:
            try:
                await self._answer_mcq_batch(job, mcq_items, epoch)
            except DeepSeekProviderError as exc:
                if exc.retryable and exc.code.value != "bad_response":
                    raise
                await self._fail_mcq_batch(pending_mcqs, exc.code.value)
            except (MultipleAskAnswerError, ValidationError) as exc:
                await self._fail_mcq_batch(pending_mcqs, str(exc))

        # Refresh status after MCQ materialization, then keep the existing
        # durable small-batch behavior for written questions only.
        async with self._conn.transaction():
            items = await self._repo.lock_job_items(job_id=str(job["id"]))
        remaining = [
            item
            for item in items
            if item["answer_mode"] in {"short", "long"}
            and item["item_status"] in {"ready_to_answer", "answering"}
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
                approved = None
                if mode in {AnswerMode.SHORT, AnswerMode.LONG}:
                    approved, route = await self._ask.find_approved_without_embedding(
                        board_id=job["board_id"],
                        class_id=job["class_id"],
                        subject_id=job["subject_id"],
                        chapter_id=job["chapter_id"],
                        answer_mode=mode,
                        normalized_question=item["normalized_question"],
                    )
                if approved is not None:
                    logger.info(
                        "multiple_ask_answer_route=%s item_id=%s", route, item["id"]
                    )
                    answer = await self._asks.complete(
                        ai_request_id=str(request["id"]),
                        answer_source=AnswerSource.APPROVED_BANK.value,
                        blocks=list(approved.blocks),
                        citations=list(approved.citations),
                        visual_ids=[visual["visual_id"] for visual in approved.answer_visuals],
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
            missed_texts = [
                " ".join(item["source_text"].split()) for item, _ in missed_items
            ]
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
                policy = await self._ask._source_policy(
                    job["class_id"], job["subject_id"]
                )
                approved = None
                if (
                    mode in {AnswerMode.SHORT, AnswerMode.LONG}
                    and vector
                    and policy["semantic_reuse_enabled"]
                    and policy["semantic_distance_threshold"] is not None
                ):
                    approved = await self._ask._bank.find_semantic(
                        query_embedding=vector,
                        evaluated_threshold=float(
                            policy["semantic_distance_threshold"]
                        ),
                        enabled=True,
                        board_id=job["board_id"],
                        class_id=job["class_id"],
                        subject_id=job["subject_id"],
                        chapter_id=job["chapter_id"],
                        answer_mode=mode,
                    )
                if approved is not None:
                    logger.info(
                        "multiple_ask_answer_route=approved_semantic item_id=%s",
                        item["id"],
                    )
                    answer = await self._asks.complete(
                        ai_request_id=str(request["id"]),
                        answer_source=AnswerSource.APPROVED_BANK.value,
                        blocks=list(approved.blocks),
                        citations=list(approved.citations),
                        visual_ids=[visual["visual_id"] for visual in approved.answer_visuals],
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
                    job["board_id"],
                    job["class_id"],
                    job["subject_id"],
                    job["chapter_id"],
                )
                evidence = await self._ask._retrieval.retrieve(
                    item["source_text"], scope, query_vector=vector if vector else None
                )
                active = await RagRepository(self._conn).get_active_corpus_version(
                    job["board_id"], job["class_id"], job["subject_id"]
                )

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
                if isinstance(exc, DeepSeekProviderError) and exc.retryable:
                    logger.warning(
                        "Retryable DeepSeek error for Multiple Ask item_id=%s job_id=%s: %s (re-raising for worker retry backoff)",
                        item.get("id"),
                        job.get("id"),
                        exc,
                    )
                    raise
                code = getattr(exc, "code", str(exc))
                if hasattr(code, "value"):
                    code = code.value
                logger.warning(
                    "Handled non-retryable error for Multiple Ask item_id=%s job_id=%s: %s",
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
        evidence_strength = getattr(
            evidence,
            "strength",
            EvidenceStrength.STRONG if evidence.results else EvidenceStrength.NONE,
        )
        if evidence_strength is EvidenceStrength.STRONG:
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
                "evidence": [asdict(value) for value in context],
                "allowed_visuals": [visual.model_dump() for visual in visuals.values()],
            }
        else:
            source = AnswerSource.GENERAL_KNOWLEDGE
            citations, visuals, prompt_key = {}, {}, PromptKey.ASK_GENERAL
            payload = {
                "question": item["source_text"],
                "answer_mode": mode.value,
                "answer_style": "exam_style",
            }
        logger.info(
            "multiple_ask_answer_route=%s item_id=%s evidence=%s",
            "rag_grounded"
            if source is AnswerSource.SYLLABUS_GROUNDED
            else "general_fallback",
            item["id"],
            evidence_strength,
        )

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
        validated = validate_generated_answer(
            source=source,
            blocks=blocks,
            citation_ids=cited_ids,
            allowed_citations=citations,
            allowed_visuals=visuals,
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

    async def _answer_mcq_batch(
        self, job: dict[str, Any], items: list[dict[str, Any]], epoch: int
    ) -> None:
        """Generate every paper MCQ once, persist it, then materialize items."""
        item_ids = [str(item["id"]) for item in items]
        identity = str(
            uuid5(
                NAMESPACE_URL,
                f"multiple-ask-mcq:{job['id']}:{epoch}:{','.join(item_ids)}",
            )
        )
        batch_identity = sha256(identity.encode("utf-8")).hexdigest()
        batch = await self._repo.get_mcq_batch(
            job_id=str(job["id"]), answer_epoch=epoch
        )
        if batch is None:
            resolved = await self._ask._prompts.resolve_active(
                prompt_key=PromptKey.ASK_GENERAL,
                answer_mode=PromptAnswerMode.SHORT,
                scope=PromptScope(
                    board_id=job["board_id"],
                    class_id=job["class_id"],
                    subject_id=job["subject_id"],
                ),
            )
            payload = {
                "instructions": (
                    "Use general knowledge only. Answer every supplied MCQ. "
                    "Do not use textbook citations or visuals. Do not change item IDs "
                    "or add/remove questions. With options select exactly one supplied "
                    "label; without options return a direct answer. Give a very short "
                    "explanation. Return JSON only: {results:[{item_id,selected_option,"
                    "answer_text,explanation}]}."
                ),
                "items": [
                    {
                        "item_id": str(item["id"]),
                        "question": item["source_text"],
                        "options": item["mcq_options"],
                    }
                    for item in items
                ],
            }
            user_prompt = json.dumps(payload, separators=(",", ":"))
            if (
                len(resolved.system_prompt) + len(user_prompt)
                > get_settings().DEEPSEEK_MAX_INPUT_CHARACTERS
            ):
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_BATCH_TOO_LARGE")
            if self._ask._provider is None:
                raise MultipleAskAnswerError("PROVIDER_UNAVAILABLE")
            generated = await self._ask._provider.generate(
                system_prompt=resolved.system_prompt,
                user_prompt=user_prompt,
                ai_request_id=identity,
                trace_id=batch_identity,
            )
            results = self._validate_mcq_results(generated.document, items)
            async with self._conn.transaction():
                batch = await self._repo.save_mcq_batch(
                    job_id=str(job["id"]),
                    answer_epoch=epoch,
                    batch_identity=batch_identity,
                    item_ids=item_ids,
                    results=results,
                    prompt_version=f"{resolved.record.id}:{resolved.record.version}",
                    provider=generated.provider,
                    model=generated.model,
                    tokens_used=generated.usage.prompt_tokens
                    + generated.usage.completion_tokens,
                    latency_ms=generated.latency_ms,
                )

        results = batch["results"]
        if isinstance(results, str):
            results = json.loads(results)
        result_by_id = {result["item_id"]: result for result in results}
        for item in items:
            if item.get("item_status") == "answered":
                continue
            result = result_by_id[str(item["id"])]
            request = await self._candidate_request(job, item)
            if await self._existing_completion(item, request, mcq_result=result):
                continue
            answer = await self._asks.complete(
                ai_request_id=str(request["id"]),
                answer_source=AnswerSource.GENERAL_KNOWLEDGE.value,
                blocks=[
                    {
                        "type": "paragraph",
                        "text": (
                            f"Correct answer: {result['correct_answer_text']}\n\n"
                            f"{result['explanation']}"
                        ),
                    }
                ],
                citations=[],
                visual_ids=[],
                prompt_version=str(batch["prompt_version"]),
                corpus_version_id=None,
                provider=str(batch["provider"]),
                model=str(batch["model"]),
                # Batch usage is durably recorded once on multiple_ask_mcq_batches.
                tokens_used=0,
                latency_ms=0,
            )
            async with self._conn.transaction():
                await self._repo.complete_answer_item(
                    item_id=str(item["id"]),
                    ai_answer_id=str(answer["id"]),
                    answer_source=AnswerSource.GENERAL_KNOWLEDGE.value,
                    approved_revision_id=None,
                    mcq_result=result,
                )

    @staticmethod
    def _validate_mcq_results(
        document: Any, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(document, dict) or any(
            key in document
            for key in (
                "citations",
                "citation_sources",
                "citation_ids",
                "visuals",
                "visual_ids",
                "visual_id",
                "cited_chunk_ids",
            )
        ):
            raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_RESPONSE_INVALID")
        results = document.get("results")
        if not isinstance(results, list) or len(results) != len(items):
            raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_RESULTS_MISMATCH")
        item_by_id = {str(item["id"]): item for item in items}
        validated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in results:
            if not isinstance(result, dict) or any(
                key in result
                for key in (
                    "citations",
                    "citation_sources",
                    "citation_ids",
                    "visuals",
                    "visual_ids",
                    "visual_id",
                    "cited_chunk_ids",
                )
            ):
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_RESULT_INVALID")
            item_id = result.get("item_id")
            if not isinstance(item_id, str) or item_id not in item_by_id:
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_UNKNOWN_ITEM")
            if item_id in seen:
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_DUPLICATE_ITEM")
            seen.add(item_id)
            explanation = result.get("explanation")
            if (
                not isinstance(explanation, str)
                or not explanation.strip()
                or len(explanation.strip()) > 600
            ):
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_EXPLANATION_INVALID")
            item = item_by_id[item_id]
            options = item["mcq_options"]
            if "selected_option" not in result:
                raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_OPTION_INVALID")
            selected = result.get("selected_option")
            answer_text = result.get("answer_text")
            if options:
                labels = {option["label"]: option["text"] for option in options}
                if not isinstance(selected, str) or selected not in labels:
                    raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_OPTION_INVALID")
                if answer_text is not None and not isinstance(answer_text, str):
                    raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_RESULT_INVALID")
                correct_answer_text = labels[selected]
            else:
                if (
                    selected is not None
                    or not isinstance(answer_text, str)
                    or not answer_text.strip()
                ):
                    raise MultipleAskAnswerError(
                        "MULTIPLE_ASK_MCQ_DIRECT_ANSWER_INVALID"
                    )
                correct_answer_text = answer_text.strip()
            validated.append(
                {
                    "item_id": item_id,
                    "selected_option": selected,
                    "correct_answer_text": correct_answer_text,
                    "explanation": explanation.strip(),
                }
            )
        if seen != set(item_by_id):
            raise MultipleAskAnswerError("MULTIPLE_ASK_MCQ_MISSING_ITEM")
        return validated

    async def _fail_mcq_batch(self, items: list[dict[str, Any]], code: str) -> None:
        for item in items:
            await self._fail_item(item, code)

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
        self,
        item: dict[str, Any],
        request: dict[str, Any],
        *,
        mcq_result: dict[str, Any] | None = None,
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
                mcq_result=mcq_result,
            )
        return True

    async def _fail_item(self, item: dict[str, Any], code: str) -> None:
        async with self._conn.transaction():
            await self._repo.fail_answer_item(item_id=str(item["id"]), error_code=code)
