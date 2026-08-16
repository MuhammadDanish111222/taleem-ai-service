"""Approved-bank-first single-question orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

import asyncpg
import redis
from pydantic import TypeAdapter, ValidationError

from app.core.config import get_settings
from app.providers.llm.deepseek import (
    DeepSeekConfig,
    DeepSeekProvider,
    DeepSeekProviderError,
)
from app.repositories.ask_repository import AskRepository
from app.repositories.prompt_cache import SharedPromptCache
from app.repositories.prompt_repository import PostgresPromptRepository
from app.repositories.provider_attempt_repository import ProviderAttemptRepository
from app.repositories.question_bank_repository import (
    ApprovedBankAnswer,
    QuestionBankRepository,
)
from app.repositories.rag_repository import RagRepository
from app.schemas.ask import (
    GENERAL_AI_LABEL,
    AnswerBlock,
    AnswerMode,
    AnswerSource,
    AskRequest,
    AskResponse,
    CitationDto,
    TerminalStatus,
    UsageDto,
    VisualDto,
)
from app.services.answers.context import assemble_context
from app.services.answers.normalization import normalize_question, question_hash
from app.services.answers.validation import (
    AnswerValidationError,
    validate_generated_answer,
)
from app.services.prompts.models import (
    AnswerMode as PromptAnswerMode,
)
from app.services.prompts.models import (
    PromptConfigurationError,
    PromptKey,
    PromptScope,
)
from app.services.prompts.service import PromptService
from app.services.retrieval.evidence import (
    Citation,
    EvidenceStrength,
    RetrievalScope,
    RetrievedEvidence,
    RetrievedVisual,
)
from app.services.retrieval.service import RetrievalService
from app.services.usage.models import AccountTier, UsageReservation
from app.services.usage.service import UsageLimitExceeded, UsageService

_BLOCKS = TypeAdapter(list[AnswerBlock])
_TOPIC_ANCHOR_WINDOW = 5
_MAX_COMPLETE_TOPIC_CHUNKS = 12
logger = logging.getLogger(__name__)


def _normalize_provider_block_aliases(value: object) -> object:
    """Normalize one observed DeepSeek alias without weakening strict validation."""
    if not isinstance(value, list):
        return value
    normalized: list[object] = []
    for item in value:
        if (
            isinstance(item, dict)
            and item.get("type") == "paragraph"
            and "text" not in item
            and type(item.get("content")) is str
            and set(item) == {"type", "content"}
        ):
            normalized.append({"type": "paragraph", "text": item["content"]})
        else:
            normalized.append(item)
    return normalized


def _topic_key(result: RetrievedEvidence) -> tuple[str | None, str, str]:
    """Return the stable ingestion identity for one retrieved textbook topic."""

    citation = result.citation
    if citation.topic_no:
        return (citation.chapter_id, "number", citation.topic_no)
    if citation.topic_title:
        return (citation.chapter_id, "title", citation.topic_title)
    return (citation.chapter_id, "chunk", citation.citation_id)


def _has_independent_strong_support(result: RetrievedEvidence) -> bool:
    """Use the approved multi-channel rule for a possible second topic."""

    return (
        len(
            {
                contribution.channel
                for contribution in result.contributions
                if contribution.rank <= 3
            }
        )
        >= 2
    )


def _select_topic_anchor_ids(
    ranked_results: tuple[RetrievedEvidence, ...], answer_mode: AnswerMode
) -> list[str]:
    """Select one topic for short answers and at most two for long answers.

    The first ranked topic is always selected. A second long-answer topic must
    be independently strong or appear more than once inside the top-five
    retrieval window; merely being a lower-ranked neighbour is not enough.
    """

    candidates = ranked_results[:_TOPIC_ANCHOR_WINDOW]
    if not candidates:
        return []
    selected = [candidates[0].citation.citation_id]
    if answer_mode is AnswerMode.SHORT:
        return selected

    first_topic = _topic_key(candidates[0])
    topic_counts: dict[tuple[str | None, str, str], int] = {}
    for candidate in candidates:
        key = _topic_key(candidate)
        topic_counts[key] = topic_counts.get(key, 0) + 1
    for candidate in candidates[1:]:
        key = _topic_key(candidate)
        if key == first_topic:
            continue
        if topic_counts[key] >= 2 or _has_independent_strong_support(candidate):
            selected.append(candidate.citation.citation_id)
            break
    return selected


class AskServiceError(RuntimeError):
    def __init__(self, code: str, *, status_code: int = 500):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


class AskService:
    def __init__(
        self,
        conn: asyncpg.Connection,
        *,
        retrieval: RetrievalService | None = None,
        provider: DeepSeekProvider | None = None,
        usage: UsageService | None = None,
        prompt_service: PromptService | None = None,
    ):
        self._conn = conn
        self._asks = AskRepository(conn)
        self._bank = QuestionBankRepository(conn)
        self._retrieval = retrieval or RetrievalService(conn)
        self._usage = usage or UsageService()
        if prompt_service is None:
            settings = get_settings()
            prompt_cache = SharedPromptCache(
                conn,
                redis.Redis.from_url(settings.REDIS_URL, decode_responses=True),
                ttl_seconds=settings.PROMPT_CACHE_TTL_SECONDS,
            )
            provider = provider or DeepSeekProvider(
                DeepSeekConfig(
                    api_key=settings.DEEPSEEK_API_KEY,
                    model=settings.DEEPSEEK_MODEL,
                    base_url=settings.DEEPSEEK_API_URL.removesuffix(
                        "/chat/completions"
                    ),
                    timeout_seconds=settings.DEEPSEEK_TIMEOUT_SECONDS,
                    max_retries=settings.DEEPSEEK_MAX_RETRIES,
                    max_output_tokens=settings.DEEPSEEK_MAX_OUTPUT_TOKENS,
                    max_input_characters=settings.DEEPSEEK_MAX_INPUT_CHARACTERS,
                ),
                attempt_recorder=ProviderAttemptRepository(conn),
            )
            prompt_service = PromptService(
                PostgresPromptRepository(conn),
                cache=prompt_cache,
                provider=provider,
            )
        self._provider = provider
        self._prompts = prompt_service

    async def ask(
        self, request: AskRequest, *, uid: str, tier: AccountTier
    ) -> AskResponse:
        request_id = str(request.request_id)
        safe_uid = self._usage.uid_hash(uid)
        existing = await self._asks.by_client_request_id(request_id, safe_uid)
        if existing is not None:
            if existing["status"] in {"completed", "no_answer"}:
                snapshot = await self._usage.snapshot(self._conn, uid=uid, tier=tier)
                return await self._response_from_existing(existing, snapshot)
            raise AskServiceError("REQUEST_ALREADY_IN_PROGRESS", status_code=409)

        normalized = normalize_question(request.question)
        if not normalized:
            raise AskServiceError("QUESTION_BLANK", status_code=422)
        reservation: UsageReservation | None = None
        ai_request: dict[str, Any] | None = None
        provider_started = False
        try:
            async with self._conn.transaction():
                reservation = await self._usage.reserve(
                    self._conn,
                    request_id=request_id,
                    uid=uid,
                    tier=tier,
                )
                created_request = await self._asks.create_pending(
                    client_request_id=request_id,
                    uid_hash=safe_uid,
                    board_id=request.board_id,
                    class_id=request.class_id,
                    subject_id=request.subject_id,
                    chapter_id=request.chapter_id,
                    answer_mode=request.answer_mode.value,
                    answer_style=request.answer_style.value,
                    raw_question=request.question,
                    normalized_question=normalized,
                    question_hash=question_hash(normalized),
                    usage_business_date=reservation.window.business_date,
                )
            if not created_request.pop("_newly_created", False):
                existing = await self._asks.by_client_request_id(request_id, safe_uid)
                if existing is not None and existing["status"] in {
                    "completed",
                    "no_answer",
                }:
                    return await self._response_from_existing(existing, reservation)
                # This operation does not own the already-pending row. It must
                # never fail/refund the primary operation in the exception path.
                ai_request = None
                raise AskServiceError("REQUEST_ALREADY_IN_PROGRESS", status_code=409)
            ai_request = created_request

            approved, route = await self.find_approved_without_embedding(
                board_id=request.board_id,
                class_id=request.class_id,
                subject_id=request.subject_id,
                chapter_id=request.chapter_id,
                answer_mode=AnswerMode(request.answer_mode),
                normalized_question=normalized,
            )
            source_policy = await self._source_policy(
                request.class_id, request.subject_id
            )
            retrieval_scope = RetrievalScope(
                request.board_id,
                request.class_id,
                request.subject_id,
                request.chapter_id,
            )
            query_vector: list[float] | None = None
            if approved is None:
                # Generate query vector once on miss for potential semantic reuse & hybrid retrieval
                query_vector = await self._retrieval.embed_live_query(
                    request.question, retrieval_scope
                )
                if (
                    query_vector is not None
                    and source_policy["semantic_reuse_enabled"]
                    and source_policy["semantic_distance_threshold"] is not None
                ):
                    approved = await self._bank.find_semantic(
                        query_embedding=query_vector,
                        evaluated_threshold=float(
                            source_policy["semantic_distance_threshold"]
                        ),
                        enabled=True,
                        board_id=request.board_id,
                        class_id=request.class_id,
                        subject_id=request.subject_id,
                        chapter_id=request.chapter_id,
                        answer_mode=AnswerMode(request.answer_mode),
                    )
                    if approved is not None:
                        route = "approved_semantic"

            if approved is not None:
                logger.info("ask_answer_route=%s request_id=%s", route, request_id)
                async with self._conn.transaction():
                    await self._persist_approved(ai_request, approved)
                    await self._usage.commit(self._conn, request_id, safe_uid)
                return self._approved_response(request, approved, reservation)

            evidence = await self._retrieval.retrieve(
                request.question,
                retrieval_scope,
                query_vector=query_vector,
            )
            active_version = await RagRepository(self._conn).get_active_corpus_version(
                request.board_id, request.class_id, request.subject_id
            )

            if evidence.strength is EvidenceStrength.STRONG:
                prompt_key = PromptKey.ASK_GROUNDED
                generation_results = evidence.results[:1]
                if active_version:
                    generation_results = await self._expand_answer_topics(
                        request=request,
                        corpus_version_id=str(active_version["id"]),
                        ranked_results=evidence.results,
                    )
                context = assemble_context(
                    generation_results,
                    max_chunks=_MAX_COMPLETE_TOPIC_CHUNKS,
                    max_characters=(
                        24000 if request.answer_mode == AnswerMode.LONG else 12000
                    ),
                )
                allowed_citations = {
                    item.citation.citation_id: CitationDto(
                        citation_id=item.citation.citation_id,
                        chapter_id=item.citation.chapter_id,
                        topic_no=item.citation.topic_no,
                        topic_title=item.citation.topic_title,
                        page_start=item.citation.page_start,
                        page_end=item.citation.page_end,
                    )
                    for item in generation_results
                }
                allowed_visuals: dict[str, VisualDto] = {}
                visual_order = 0
                for item in generation_results:
                    for visual in item.citation.visuals:
                        if visual.visual_id in allowed_visuals:
                            raise AskServiceError(
                                "RETRIEVED_VISUAL_ID_AMBIGUOUS",
                                status_code=502,
                            )
                        allowed_visuals[visual.visual_id] = VisualDto(
                            visual_id=visual.visual_id,
                            title=visual.title,
                            description=visual.description,
                            display_policy=visual.display_policy,
                            display_order=visual_order,
                        )
                        visual_order += 1
                user_prompt = json.dumps(
                    {
                        "question": request.question,
                        "answer_mode": request.answer_mode.value,
                        "answer_style": request.answer_style.value,
                        "evidence": [asdict(item) for item in context],
                        "allowed_visuals": [
                            visual.model_dump() for visual in allowed_visuals.values()
                        ],
                    },
                    separators=(",", ":"),
                )
                source = AnswerSource.SYLLABUS_GROUNDED
            else:
                prompt_key = PromptKey.ASK_GENERAL
                allowed_citations = {}
                allowed_visuals = {}
                user_prompt = json.dumps(
                    {
                        "question": request.question,
                        "answer_mode": request.answer_mode.value,
                        "answer_style": request.answer_style.value,
                    },
                    separators=(",", ":"),
                )
                source = AnswerSource.GENERAL_KNOWLEDGE

            logger.info(
                "ask_answer_route=%s request_id=%s evidence=%s",
                "rag_grounded"
                if source is AnswerSource.SYLLABUS_GROUNDED
                else "general_fallback",
                request_id,
                evidence.strength,
            )

            resolved = await self._prompts.resolve_active(
                prompt_key=prompt_key,
                answer_mode=PromptAnswerMode(request.answer_mode.value),
                scope=PromptScope(
                    board_id=request.board_id,
                    class_id=request.class_id,
                    subject_id=request.subject_id,
                ),
            )
            if self._provider is None:
                raise AskServiceError("PROVIDER_UNAVAILABLE")
            provider_started = True
            generation = await self._provider.generate(
                system_prompt=resolved.system_prompt,
                user_prompt=user_prompt,
                ai_request_id=str(ai_request["id"]),
                trace_id=request_id,
            )
            try:
                blocks = _BLOCKS.validate_python(
                    _normalize_provider_block_aliases(generation.document.get("blocks"))
                )
                cited_ids = generation.document.get("cited_chunk_ids", [])
                if not isinstance(cited_ids, list) or not all(
                    isinstance(item, str) for item in cited_ids
                ):
                    raise AnswerValidationError("ANSWER_CITATIONS_INVALID")
            except (ValidationError, AnswerValidationError, AttributeError) as exc:
                raise AskServiceError("PROVIDER_RESPONSE_INVALID") from exc

            prompt_version = f"{resolved.record.id}:{resolved.record.version}"
            if not blocks and not cited_ids:
                async with self._conn.transaction():
                    await self._asks.no_answer(
                        str(ai_request["id"]),
                        error_code="TEXTBOOK_EVIDENCE_INSUFFICIENT",
                        prompt_version=prompt_version,
                    )
                    await self._usage.commit(self._conn, request_id, safe_uid)
                return self._no_answer_response(
                    request,
                    reservation,
                    error_code="TEXTBOOK_EVIDENCE_INSUFFICIENT",
                )

            validated = validate_generated_answer(
                source=source,
                blocks=blocks,
                citation_ids=cited_ids,
                allowed_citations=allowed_citations,
                allowed_visuals=allowed_visuals,
            )
            async with self._conn.transaction():
                await self._asks.complete(
                    ai_request_id=str(ai_request["id"]),
                    answer_source=source.value,
                    blocks=[item.model_dump() for item in validated.blocks],
                    citations=[item.model_dump() for item in validated.citations],
                    visual_ids=[item.visual_id for item in validated.visuals],
                    prompt_version=prompt_version,
                    corpus_version_id=(
                        str(active_version["id"]) if active_version else None
                    ),
                    provider=generation.provider,
                    model=generation.model,
                    tokens_used=(
                        generation.usage.prompt_tokens
                        + generation.usage.completion_tokens
                    ),
                    latency_ms=generation.latency_ms,
                )
                await self._usage.commit(self._conn, request_id, safe_uid)
            return AskResponse(
                request_id=request.request_id,
                answer_source=source,
                answer_mode=request.answer_mode,
                answer_style=request.answer_style,
                blocks=list(validated.blocks),
                citations=list(validated.citations),
                visuals=list(validated.visuals),
                general_ai_label=(
                    GENERAL_AI_LABEL
                    if source == AnswerSource.GENERAL_KNOWLEDGE
                    else None
                ),
                prompt_version=prompt_version,
                corpus_version=(str(active_version["id"]) if active_version else None),
                approved_revision_id=None,
                usage=self._usage_dto(reservation),
                terminal_status=TerminalStatus.ANSWERED,
                error_code=None,
            )
        except UsageLimitExceeded:
            raise
        except BaseException as exc:
            if ai_request is not None:
                async with self._conn.transaction():
                    await self._asks.fail(
                        str(ai_request["id"]),
                        error_code=self._safe_error_code(exc),
                    )
                    if provider_started and isinstance(exc, asyncio.CancelledError):
                        await self._usage.commit(self._conn, request_id, safe_uid)
                    else:
                        await self._usage.refund(self._conn, request_id, safe_uid)
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, (AskServiceError, UsageLimitExceeded)):
                raise
            if isinstance(exc, DeepSeekProviderError):
                raise AskServiceError(exc.code.value, status_code=503) from None
            if isinstance(exc, AnswerValidationError):
                raise AskServiceError(str(exc), status_code=502) from None
            if isinstance(exc, PromptConfigurationError):
                raise AskServiceError(
                    "PROMPT_CONFIGURATION_MISSING", status_code=503
                ) from exc
            raise AskServiceError("ASK_INTERNAL_FAILURE", status_code=500) from None

    async def find_approved_without_embedding(
        self,
        *,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
        answer_mode: AnswerMode,
        normalized_question: str,
    ) -> tuple[ApprovedBankAnswer | None, str | None]:
        """Run the deterministic, no-provider approved-bank prefix."""

        if answer_mode not in {AnswerMode.SHORT, AnswerMode.LONG}:
            return None, None
        scope = {
            "board_id": board_id,
            "class_id": class_id,
            "subject_id": subject_id,
            "chapter_id": chapter_id,
            "answer_mode": answer_mode,
            "normalized_question": normalized_question,
        }
        approved = await self._bank.find_exact(**scope)
        if approved is not None:
            return approved, "approved_exact"
        approved = await self._bank.find_exact_variation(**scope)
        if approved is not None:
            return approved, "approved_variation"
        approved = await self._bank.find_lexical(**scope)
        if approved is not None:
            return approved, "approved_lexical"
        return None, None

    async def _expand_answer_topics(
        self,
        *,
        request: AskRequest,
        corpus_version_id: str,
        ranked_results: tuple[RetrievedEvidence, ...],
    ) -> tuple[RetrievedEvidence, ...]:
        """Expand the selected ranked anchors to complete scoped topics."""
        repo = RagRepository(self._conn)
        anchor_ids = _select_topic_anchor_ids(ranked_results, request.answer_mode)
        rows = await repo.get_active_topic_chunks(
            board_id=request.board_id,
            class_id=request.class_id,
            subject_id=request.subject_id,
            corpus_version_id=corpus_version_id,
            anchor_citation_ids=anchor_ids,
            max_topics=2 if request.answer_mode is AnswerMode.LONG else 1,
            max_chunks=_MAX_COMPLETE_TOPIC_CHUNKS,
        )
        if not rows:
            selected = set(anchor_ids)
            return tuple(
                item for item in ranked_results if item.citation.citation_id in selected
            )
        visual_map = await repo.get_eligible_retrieval_visuals(
            [row["citation_id"] for row in rows]
        )
        ranked_by_id = {item.citation.citation_id: item for item in ranked_results}
        expanded: list[RetrievedEvidence] = []
        for position, row in enumerate(rows, start=1):
            ranked = ranked_by_id.get(row["citation_id"])
            expanded.append(
                RetrievedEvidence(
                    citation=Citation(
                        citation_id=row["citation_id"],
                        content=row["content"],
                        chapter_id=row["chapter_id"],
                        topic_no=row["topic_no"],
                        topic_title=row["topic_title"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        visuals=tuple(
                            RetrievedVisual(
                                visual_id=visual["visual_id"],
                                title=visual["title"],
                                description=visual["description"],
                                display_policy=visual["display_policy"],
                            )
                            for visual in visual_map.get(row["citation_id"], ())
                        ),
                    ),
                    fused_rank=ranked.fused_rank if ranked else position,
                    contributions=ranked.contributions if ranked else (),
                )
            )
        return tuple(expanded)

    async def usage(self, *, uid: str, tier: AccountTier) -> UsageDto:
        return self._usage_dto(
            await self._usage.snapshot(self._conn, uid=uid, tier=tier)
        )

    async def _source_policy(self, class_id: str, subject_id: str) -> dict[str, Any]:
        row = await self._conn.fetchrow(
            """SELECT allow_general,semantic_reuse_enabled,
                      semantic_distance_threshold
               FROM ask_source_policies
               WHERE (class_id=$1 AND subject_id=$2)
                  OR (class_id IS NULL AND subject_id=$2)
                  OR (class_id IS NULL AND subject_id IS NULL)
               ORDER BY
                 CASE
                   WHEN class_id=$1 AND subject_id=$2 THEN 1
                   WHEN class_id IS NULL AND subject_id=$2 THEN 2
                   ELSE 3
                 END
               LIMIT 1""",
            class_id,
            subject_id,
        )
        return {
            "allow_general": bool(row and row["allow_general"]),
            "semantic_reuse_enabled": bool(row and row["semantic_reuse_enabled"]),
            "semantic_distance_threshold": (
                row["semantic_distance_threshold"] if row else None
            ),
        }

    async def _persist_approved(
        self, ai_request: dict[str, Any], approved: ApprovedBankAnswer
    ) -> None:
        await self._asks.complete(
            ai_request_id=str(ai_request["id"]),
            answer_source=AnswerSource.APPROVED_BANK.value,
            blocks=list(approved.blocks),
            citations=list(approved.citations),
            visual_ids=[item["visual_id"] for item in approved.answer_visuals],
            prompt_version="approved-bank",
            corpus_version_id=None,
            provider=None,
            model=None,
            approved_revision_id=approved.revision_id,
        )

    def _approved_response(
        self,
        request: AskRequest,
        approved: ApprovedBankAnswer,
        reservation: UsageReservation,
    ) -> AskResponse:
        return AskResponse(
            request_id=request.request_id,
            answer_source=AnswerSource.APPROVED_BANK,
            answer_mode=request.answer_mode,
            answer_style=request.answer_style,
            blocks=_BLOCKS.validate_python(list(approved.blocks)),
            citations=[CitationDto.model_validate(item) for item in approved.citations],
            visuals=[
                VisualDto.model_validate(item) for item in approved.answer_visuals
            ],
            general_ai_label=None,
            prompt_version=None,
            corpus_version=None,
            approved_revision_id=approved.revision_id,
            usage=self._usage_dto(reservation),
            terminal_status=TerminalStatus.ANSWERED,
            error_code=None,
        )

    def _no_answer_response(
        self,
        request: AskRequest,
        reservation: UsageReservation,
        *,
        error_code: str,
    ) -> AskResponse:
        return AskResponse(
            request_id=request.request_id,
            answer_source=None,
            answer_mode=request.answer_mode,
            answer_style=request.answer_style,
            usage=self._usage_dto(reservation),
            terminal_status=TerminalStatus.NO_ANSWER,
            error_code=error_code,
        )

    async def _response_from_existing(
        self, existing: dict[str, Any], reservation: UsageReservation
    ) -> AskResponse:
        mode = AnswerMode(existing["answer_mode"])
        if existing["status"] == "no_answer":
            return AskResponse(
                request_id=existing["client_request_id"],
                answer_source=None,
                answer_mode=mode,
                answer_style=existing["answer_style"],
                usage=self._usage_dto(reservation),
                terminal_status=TerminalStatus.NO_ANSWER,
                error_code=existing["terminal_error_code"],
            )
        if existing["approved_revision_id"]:
            approved = await self._bank.get_revision(
                str(existing["approved_revision_id"])
            )
            if approved is not None:
                return AskResponse(
                    request_id=existing["client_request_id"],
                    answer_source=AnswerSource.APPROVED_BANK,
                    answer_mode=mode,
                    answer_style=existing["answer_style"],
                    blocks=_BLOCKS.validate_python(list(approved.blocks)),
                    citations=[
                        CitationDto.model_validate(item) for item in approved.citations
                    ],
                    visuals=[
                        VisualDto.model_validate(item)
                        for item in approved.answer_visuals
                    ],
                    approved_revision_id=approved.revision_id,
                    usage=self._usage_dto(reservation),
                    terminal_status=TerminalStatus.ANSWERED,
                )
        blocks = existing["answer_blocks"] or []
        citations = existing["citation_sources"] or []
        if isinstance(blocks, str):
            blocks = json.loads(blocks)
        if isinstance(citations, str):
            citations = json.loads(citations)
        visual_ids = existing["visual_ids"] or []
        if isinstance(visual_ids, str):
            visual_ids = json.loads(visual_ids)
        visuals = (
            await self._asks.visual_metadata_for_completed_request(
                ai_request_id=str(existing["id"]),
                visual_ids=visual_ids,
            )
            if existing["answer_source"] == AnswerSource.SYLLABUS_GROUNDED
            else []
        )
        return AskResponse(
            request_id=existing["client_request_id"],
            answer_source=existing["answer_source"],
            answer_mode=mode,
            answer_style=existing["answer_style"],
            blocks=_BLOCKS.validate_python(blocks),
            citations=[CitationDto.model_validate(item) for item in citations],
            visuals=[VisualDto.model_validate(item) for item in visuals],
            general_ai_label=(
                GENERAL_AI_LABEL
                if existing["answer_source"] == AnswerSource.GENERAL_KNOWLEDGE
                else None
            ),
            prompt_version=existing["prompt_version"],
            corpus_version=(
                str(existing["corpus_version_id"])
                if existing["corpus_version_id"]
                else None
            ),
            approved_revision_id=existing["approved_revision_id"],
            usage=self._usage_dto(reservation),
            terminal_status=TerminalStatus.ANSWERED,
            error_code=None,
        )

    @staticmethod
    def _usage_dto(reservation: UsageReservation) -> UsageDto:
        visible_limit = reservation.limit if reservation.student_visible else None
        return UsageDto(
            used=reservation.used,
            limit=visible_limit,
            remaining=(
                max(0, reservation.limit - reservation.used)
                if visible_limit is not None
                else None
            ),
            resets_at=reservation.window.resets_at,
        )

    @staticmethod
    def _safe_error_code(exc: BaseException) -> str:
        if isinstance(exc, AskServiceError):
            return exc.code
        if isinstance(exc, DeepSeekProviderError):
            return exc.code.value
        if isinstance(exc, AnswerValidationError):
            return str(exc)
        if isinstance(exc, PromptConfigurationError):
            return "PROMPT_CONFIGURATION_MISSING"
        return "ASK_INTERNAL_FAILURE"
