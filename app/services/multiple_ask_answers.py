"""Railway-owned, restart-safe answers for extracted Multiple Ask items.

The implementation deliberately writes generated output through Module 4's
``ai_requests``/``ai_answers`` candidate foundation.  It never treats paper
content as a cache: only immutable approved-bank revisions are reusable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import TypeAdapter, ValidationError

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
    """Answer ready items without consuming a second quota reservation."""

    def __init__(self, conn: Any, *, ask_service: AskService | None = None):
        self._conn = conn
        self._repo = MultipleAskRepository(conn)
        # AskService is composition, not a nested public Ask request: its
        # trusted bank/retrieval/prompt/validation dependencies are reused but
        # no single-question usage reservation is made here.
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

    async def answer(self, *, session_id: str, epoch: int) -> str:
        """Run remaining persisted items only; terminal rows are never regenerated."""
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
        for item in remaining:
            if item["answer_mode"] in {"short", "long"}:
                await self._answer_short_or_long(job, item)
        mcqs = [item for item in remaining if item["answer_mode"] == "mcq"]
        for group in self._mcq_groups(mcqs):
            await self._answer_mcq_group(job, group)
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

    async def _answer_short_or_long(
        self, job: dict[str, Any], item: dict[str, Any]
    ) -> None:
        try:
            request = await self._candidate_request(job, item)
            completed = await self._existing_completion(item, request)
            if completed:
                return
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
            # Semantic bank reuse is intentionally absent.  It remains disabled
            # even if a future source policy is configured differently.
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
                return
            policy = await self._ask._source_policy(job["class_id"], job["subject_id"])
            scope = RetrievalScope(
                job["board_id"], job["class_id"], job["subject_id"], job["chapter_id"]
            )
            evidence = await self._ask._retrieval.retrieve(item["source_text"], scope)
            active = await RagRepository(self._conn).get_active_corpus_version(
                job["board_id"], job["class_id"], job["subject_id"]
            )
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
                            raise MultipleAskAnswerError(
                                "RETRIEVED_VISUAL_ID_AMBIGUOUS"
                            )
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
            logger.exception(
                "Unexpected error for Multiple Ask item_id=%s job_id=%s",
                item.get("id"),
                job.get("id"),
            )
            await self._fail_item(item, "MULTIPLE_ASK_ANSWER_FAILED")

    async def _answer_mcq_group(
        self, job: dict[str, Any], group: list[dict[str, Any]]
    ) -> None:
        requests: dict[str, dict[str, Any]] = {}
        for item in group:
            try:
                request = await self._candidate_request(job, item)
                if not await self._existing_completion(item, request):
                    requests[str(item["id"])] = request
                    async with self._conn.transaction():
                        await self._repo.mark_item_answering(str(item["id"]))
            except Exception:
                await self._fail_item(item, "MULTIPLE_ASK_CANDIDATE_FAILED")
        pending = [item for item in group if str(item["id"]) in requests]
        if not pending:
            return
        try:
            resolved = await self._ask._prompts.resolve_active(
                prompt_key=PromptKey.ASK_GENERAL,
                answer_mode=PromptAnswerMode.MCQ,
                scope=PromptScope(
                    board_id=job["board_id"],
                    class_id=job["class_id"],
                    subject_id=job["subject_id"],
                ),
            )
            if self._ask._provider is None:
                raise MultipleAskAnswerError("PROVIDER_UNAVAILABLE")
            prompt = {
                "items": [
                    {
                        "item_id": str(item["id"]),
                        "question": item["source_text"],
                        "options": item["mcq_options"],
                    }
                    for item in pending
                ],
                "required_result": {
                    "item_id": "UUID",
                    "selected_option": "A|B|C|D",
                    "explanation": "concise text",
                },
                "rules": [
                    "Use general knowledge only",
                    "No citations",
                    "No visuals",
                    "Return JSON results only",
                ],
            }
            generated = await self._ask._provider.generate(
                system_prompt=resolved.system_prompt
                + "\nReturn one strict result for every supplied item; never add IDs.",
                user_prompt=json.dumps(prompt, separators=(",", ":")),
                ai_request_id=str(next(iter(requests.values()))["id"]),
                trace_id=str(job["id"]),
            )
            results = generated.document.get("results")
            if not isinstance(results, list):
                raise MultipleAskAnswerError("MCQ_PROVIDER_RESPONSE_INVALID")
            by_id: dict[str, list[dict[str, Any]]] = {}
            for result in results:
                if isinstance(result, dict) and isinstance(result.get("item_id"), str):
                    by_id.setdefault(result["item_id"], []).append(result)
            allowed = {str(item["id"]): item for item in pending}
            for item_id, item in allowed.items():
                values = by_id.get(item_id, [])
                if len(values) != 1:
                    await self._fail_item(item, "MCQ_RESULT_MISSING_OR_DUPLICATE")
                    continue
                result = values[0]
                option, explanation = (
                    result.get("selected_option"),
                    result.get("explanation"),
                )
                if (
                    option not in {"A", "B", "C", "D"}
                    or not isinstance(explanation, str)
                    or not explanation.strip()
                ):
                    await self._fail_item(item, "MCQ_RESULT_INVALID")
                    continue
                answer = await self._asks.complete(
                    ai_request_id=str(requests[item_id]["id"]),
                    answer_source=AnswerSource.GENERAL_KNOWLEDGE.value,
                    blocks=[
                        {
                            "type": "paragraph",
                            "text": f"Selected option: {option}\n\n{explanation.strip()}",
                        }
                    ],
                    citations=[],
                    visual_ids=[],
                    prompt_version=f"{resolved.record.id}:{resolved.record.version}",
                    corpus_version_id=None,
                    provider=generated.provider,
                    model=generated.model,
                    tokens_used=generated.usage.prompt_tokens
                    + generated.usage.completion_tokens,
                    latency_ms=generated.latency_ms,
                )
                async with self._conn.transaction():
                    await self._repo.complete_answer_item(
                        item_id=item_id,
                        ai_answer_id=str(answer["id"]),
                        answer_source=AnswerSource.GENERAL_KNOWLEDGE.value,
                        approved_revision_id=None,
                    )
        except Exception as exc:
            for item in pending:
                await self._fail_item(item, getattr(exc, "code", "MCQ_BATCH_FAILED"))

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
        if request["status"] != "completed":
            return False
        answer = await self._conn.fetchrow(
            "SELECT id,answer_source,approved_revision_id FROM ai_answers WHERE request_id=$1::uuid",
            request["id"],
        )
        if answer is None:
            return False
        async with self._conn.transaction():
            await self._repo.complete_answer_item(
                item_id=str(item["id"]),
                ai_answer_id=str(answer["id"]),
                answer_source=answer["answer_source"],
                approved_revision_id=str(answer["approved_revision_id"])
                if answer["approved_revision_id"]
                else None,
            )
        return True

    async def _fail_item(self, item: dict[str, Any], error_code: str) -> None:
        async with self._conn.transaction():
            await self._repo.fail_answer_item(
                item_id=str(item["id"]), error_code=error_code[:120]
            )

    @staticmethod
    def _mcq_groups(
        items: list[dict[str, Any]], limit: int = 24_000
    ) -> list[list[dict[str, Any]]]:
        """Deterministically bound provider input while retaining extraction order."""
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_size = 0
        for item in items:
            size = len(
                json.dumps(
                    {"q": item["source_text"], "o": item["mcq_options"]},
                    separators=(",", ":"),
                )
            )
            if current and current_size + size > limit:
                groups.append(current)
                current, current_size = [], 0
            current.append(item)
            current_size += size
        if current:
            groups.append(current)
        return groups
