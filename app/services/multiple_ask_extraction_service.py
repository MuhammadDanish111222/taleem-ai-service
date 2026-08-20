"""Railway-only OCR and deterministic extraction for temporary Multiple Ask input."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any

import fitz
from pypdf import PdfReader

from app.core.config import get_settings
from app.providers.ocr.base import OCRExtractedQuestion, OCRProvider, OCRProviderError
from app.providers.ocr.gemini import GeminiOCRProvider
from app.repositories.multiple_ask_repository import MultipleAskRepository
from app.services.answers.normalization import normalize_question, question_hash
from app.services.jobs.queue import JobQueueService
from app.services.multiple_ask import MultipleAskError
from app.services.multiple_ask_answers import MultipleAskAnswerService
from app.services.multiple_ask_extraction import (
    ExtractedQuestion,
    QuestionLimitExceeded,
    extract_ordered_questions,
)
from app.services.multiple_ask_storage import TemporaryUploadStorage
from app.services.runtime_settings import RuntimeSettingsService, Scope
from app.services.usage.service import UsageService


class MultipleAskExtractionError(MultipleAskError):
    pass


def normalize_source_text(text: str, *, max_characters: int) -> str:
    """Normalize layout only: preserve page boundaries and never correct content."""
    pages = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    normalized_pages = []
    for page in pages:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in page.split("\n")]
        normalized_pages.append("\n".join(lines).strip())
    normalized = "\f\n".join(normalized_pages).strip()
    if len(normalized) > max_characters:
        raise MultipleAskExtractionError("MULTIPLE_ASK_SOURCE_TEXT_INVALID")
    return normalized


def _meaningful_embedded_text(text: str) -> bool:
    return len(re.findall(r"[A-Za-z0-9]", text)) >= 8


def _structured_section_mode(context: str | None) -> str | None:
    value = re.sub(r"[^a-z0-9]+", " ", (context or "").lower()).strip()
    if re.search(
        r"\b(?:mcqs?|multiple choice questions?|objective questions?|choose (?:the )?correct answer)\b",
        value,
    ):
        return "mcq"
    if re.search(
        r"\b(?:short questions?|short answer questions?|answer the following short questions?)\b",
        value,
    ):
        return "short"
    if re.search(
        r"\b(?:long questions?|long answer questions?|detailed questions?)\b", value
    ):
        return "long"
    return None


def _valid_dynamic_options(options: list[dict[str, str]]) -> bool:
    return (
        len(options) >= 2
        and [option["label"] for option in options]
        == [chr(ord("A") + index) for index in range(len(options))]
        and all(option["text"] for option in options)
    )


class MultipleAskExtractionService:
    def __init__(
        self,
        conn: Any,
        *,
        storage: TemporaryUploadStorage | None = None,
        ocr: OCRProvider | None = None,
    ):
        self._conn = conn
        self._repo = MultipleAskRepository(conn)
        self._storage = storage or TemporaryUploadStorage()
        self._ocr = ocr

    async def _ocr_provider(self) -> OCRProvider:
        if self._ocr is not None:
            return self._ocr
        return GeminiOCRProvider(
            timeout_seconds=int(
                await RuntimeSettingsService(self._conn).get(
                    "multiple_ask.ocr_timeout_seconds", Scope(kind="global")
                )
            )
        )

    async def start_initial_extraction(self, session_id: str) -> dict[str, Any] | None:
        """Atomically make the post-validation extraction job durable and visible."""
        async with self._conn.transaction():
            context = await self._repo.lock_extraction_context(session_id)
            if context is None:
                raise MultipleAskExtractionError("MULTIPLE_ASK_PARENT_NOT_FOUND")
            if context["workflow_status"] != "validated":
                return None
            return await self._enqueue_extraction_locked(context, resume=False)

    async def resume_extraction(
        self, *, job_id: str, uid: str, request_id: str
    ) -> dict[str, Any]:
        uid_hash = UsageService().uid_hash(uid)
        async with self._conn.transaction():
            job = await self._repo.lock_owned_job(job_id=job_id, uid_hash=uid_hash)
            if job is None:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_JOB_NOT_FOUND", status_code=404
                )
            if job["workflow_status"] == "ready_to_answer":
                # Corrections now persist the student's selected type/options
                # immediately. This compatibility endpoint therefore never
                # re-runs extraction or alters a corrected item.
                return {
                    "job_id": str(job["id"]),
                    "workflow_status": "ready_to_answer",
                    "queue_status": "succeeded",
                }
            if job["workflow_status"] != "needs_correction":
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_JOB_NOT_RESUMABLE", status_code=409
                )
            items = await self._repo.lock_job_items(job_id=str(job["id"]))
            if any(item["item_status"] == "needs_correction" for item in items):
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_JOB_NOT_RESUMABLE", status_code=409
                )
            raise MultipleAskExtractionError(
                "MULTIPLE_ASK_JOB_NOT_RESUMABLE", status_code=409
            )

    async def apply_correction(
        self,
        *,
        job_id: str,
        item_id: str,
        uid: str,
        request_id: str,
        question_text: str,
        answer_mode: str,
        mcq_options: list[dict[str, str]],
    ) -> dict[str, Any]:
        if (
            not question_text.strip()
            or len(question_text) > get_settings().MULTIPLE_ASK_MAX_TEXT_CHARACTERS
            or "\x00" in question_text
        ):
            raise MultipleAskExtractionError("MULTIPLE_ASK_CORRECTION_INVALID")
        if answer_mode not in {"short", "long", "mcq"}:
            raise MultipleAskExtractionError("MULTIPLE_ASK_CORRECTION_INVALID")
        normalized_options = self._validate_correction_options(answer_mode, mcq_options)
        normalized = normalize_question(question_text)
        if not normalized:
            raise MultipleAskExtractionError("MULTIPLE_ASK_CORRECTION_INVALID")
        uid_hash = UsageService().uid_hash(uid)
        async with self._conn.transaction():
            job = await self._repo.lock_owned_job(job_id=job_id, uid_hash=uid_hash)
            if job is None:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_JOB_NOT_FOUND", status_code=404
                )
            items = await self._repo.lock_job_items(job_id=job_id)
            target = next((item for item in items if str(item["id"]) == item_id), None)
            if target is None:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_ITEM_NOT_FOUND", status_code=404
                )
            if (
                target.get("correction_request_id")
                and str(target["correction_request_id"]) == request_id
            ):
                if (
                    target.get("correction_text") != question_text.strip()
                    or target.get("correction_answer_mode") != answer_mode
                    or list(target.get("correction_mcq_options") or [])
                    != normalized_options
                ):
                    raise MultipleAskExtractionError(
                        "MULTIPLE_ASK_CORRECTION_IDEMPOTENCY_CONFLICT", status_code=409
                    )
                return await self._owned_status_locked(job)
            if job["workflow_status"] != "needs_correction" or (
                target["item_status"] != "needs_correction"
                or target["answer_mode"] != "not_clear"
            ):
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_ITEM_NOT_CORRECTABLE", status_code=409
                )
            updated = await self._repo.apply_correction(
                job_id=job_id,
                item_id=item_id,
                request_id=request_id,
                question_text=question_text.strip(),
                normalized_question=normalized,
                question_hash=question_hash(normalized),
                answer_mode=answer_mode,
                mcq_options=normalized_options,
            )
            if updated is None:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_ITEM_NOT_CORRECTABLE", status_code=409
                )
            await self._repo.finish_corrections_if_resolved(job_id=job_id)
            record = await self._owned_status_locked(job)
        # Final correction begins the durable answer flow automatically. This
        # uses the original batch reservation; no correction can charge again.
        if record["workflow_status"] == "ready_to_answer":
            await MultipleAskAnswerService(self._conn).start_for_job(
                job_id=job_id, uid=uid
            )
            refreshed = await self._repo.get_owned_job_status(
                job_id=job_id, uid_hash=UsageService().uid_hash(uid)
            )
            if refreshed is None:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_JOB_NOT_FOUND", status_code=404
                )
            return refreshed
        return record

    async def extract(self, *, session_id: str, epoch: int, resume: bool) -> str:
        """Run one durable, bounded extraction epoch. No answer provider is involved."""
        async with self._conn.transaction():
            context = await self._repo.lock_extraction_context(session_id)
            if context is None:
                raise MultipleAskExtractionError("MULTIPLE_ASK_PARENT_NOT_FOUND")
            if (
                context["workflow_status"] != "extracting"
                or context["extraction_epoch"] != epoch
            ):
                return str(context["workflow_status"])
        if resume:
            return await self._extract_corrections(context)
        try:
            source = await self._normalized_source(context)
            extracted = source.get("structured_questions") or extract_ordered_questions(
                source["normalized_text"],
                max_questions=int(
                    await RuntimeSettingsService(self._conn).get(
                        "multiple_ask.max_extracted_questions", Scope(kind="global")
                    )
                ),
            )
        except QuestionLimitExceeded:
            await self.fail_and_refund(
                session_id=session_id,
                workflow_status="too_many_questions",
                error_code="MULTIPLE_ASK_TOO_MANY_QUESTIONS",
            )
            return "too_many_questions"
        records = [
            self._record(ExtractedQuestion(**{**item.__dict__, "source_order": index}))
            for index, item in enumerate(extracted)
        ]
        if not records:
            records = [
                {
                    "item_index": 0,
                    "question_text": "",
                    "normalized_question": None,
                    "question_hash": None,
                    "answer_mode": "not_clear",
                    "item_status": "needs_correction",
                    "mcq_options": [],
                    "unclear_reason": "NO_QUESTIONS_DETECTED",
                    "source_locator": {"page_number": 1, "display_label": "1"},
                    "display_label": "1",
                    "section_context": None,
                }
            ]
        async with self._conn.transaction():
            current = await self._repo.lock_extraction_context(session_id)
            if (
                current is None
                or current["workflow_status"] != "extracting"
                or current["extraction_epoch"] != epoch
            ):
                return str(current["workflow_status"]) if current else "failed"
            await self._repo.insert_extracted_items(
                job_id=str(current["id"]), items=records
            )
            await self._repo.finish_extraction(
                job_id=str(current["id"]),
                needs_correction=any(
                    item["item_status"] == "needs_correction" for item in records
                ),
            )
        status = (
            "needs_correction"
            if any(item["item_status"] == "needs_correction" for item in records)
            else "ready_to_answer"
        )
        if status == "ready_to_answer":
            await MultipleAskAnswerService(self._conn).start_for_job(
                job_id=str(current["id"])
            )
            return "answering"
        return status

    async def _enqueue_extraction_locked(
        self, context: dict[str, Any], *, resume: bool
    ) -> dict[str, Any]:
        epoch = int(context["extraction_epoch"]) + 1
        queue = await JobQueueService(self._conn).enqueue_job(
            job_type="multiple_ask_extract",
            payload={
                "multiple_ask_session_id": str(context["upload_session_id"]),
                "epoch": epoch,
                "resume": resume,
            },
            idempotency_key=f"multiple-ask-extract:{context['id']}:{epoch}",
        )
        if not await self._repo.start_extraction(
            job_id=str(context["id"]), queue_job_id=str(queue["id"]), epoch=epoch
        ):
            raise MultipleAskExtractionError(
                "MULTIPLE_ASK_EXTRACTION_STATE_CONFLICT", status_code=409
            )
        return {
            "job_id": str(context["id"]),
            "workflow_status": "extracting",
            "queue_status": queue["status"],
        }

    async def _normalized_source(self, context: dict[str, Any]) -> dict[str, Any]:
        if context.get("normalized_text"):
            return {"normalized_text": context["normalized_text"]}
        settings = get_settings()
        if context["input_kind"] == "text":
            text = context.get("input_text")
            if not isinstance(text, str):
                raise MultipleAskExtractionError("MULTIPLE_ASK_SOURCE_UNAVAILABLE")
            normalized = normalize_source_text(
                text, max_characters=settings.MULTIPLE_ASK_MAX_TEXT_CHARACTERS
            )
            kind, locators, provider = (
                "pasted_text",
                [{"page_number": 1, "source_kind": "pasted_text"}],
                None,
            )
        else:
            raw = await self._storage.read_object_limited(
                bucket=context["source_bucket"],
                object_key=context["source_key"],
                max_bytes=settings.MULTIPLE_ASK_MAX_IMAGE_BYTES
                if context["input_kind"] == "image"
                else settings.MULTIPLE_ASK_MAX_PDF_BYTES,
            )
            if context["input_kind"] == "image":
                structured = await self._ocr_questions(raw, page_number=1)
                normalized = self._structured_source_text(structured)
                kind, locators, provider = (
                    "image_structured_extraction",
                    [{"page_number": 1, "source_kind": "ocr"}],
                    "gemini",
                )
            else:
                normalized, kind, locators, provider, structured = await self._pdf_text(
                    raw
                )
        async with self._conn.transaction():
            cached = await self._repo.save_normalized_source_once(
                session_id=str(context["upload_session_id"]),
                normalized_text=normalized,
                source_locators=locators,
                source_kind=kind,
                ocr_provider=provider,
            )
        if context["input_kind"] == "image":
            return {**cached, "structured_questions": structured}
        return {
            **cached,
            **({"structured_questions": structured} if structured else {}),
        }

    async def _pdf_text(
        self, raw: bytes
    ) -> tuple[str, str, list[dict[str, Any]], str | None, list[ExtractedQuestion]]:
        reader = PdfReader(BytesIO(raw), strict=True)
        document = fitz.open(stream=raw, filetype="pdf")
        pages: list[str] = []
        locators: list[dict[str, Any]] = []
        structured: list[ExtractedQuestion] = []
        used_ocr = False
        try:
            for index, page in enumerate(reader.pages):
                embedded = page.extract_text() or ""
                page_number = index + 1
                if _meaningful_embedded_text(embedded):
                    pages.append(embedded)
                    structured.extend(
                        self._with_page_number(
                            extract_ordered_questions(embedded), page_number
                        )
                    )
                    locators.append(
                        {"page_number": page_number, "source_kind": "embedded_text"}
                    )
                    continue
                used_ocr = True
                pdf_page = document.load_page(index)
                rect = pdf_page.rect
                desired_scale = 200 / 72
                page_area = max(float(rect.width) * float(rect.height), 1.0)
                pixel_cap = get_settings().MULTIPLE_ASK_MAX_RENDERED_PDF_PAGE_PIXELS
                safe_scale = min(desired_scale, (pixel_cap / page_area) ** 0.5)
                pixmap = pdf_page.get_pixmap(
                    matrix=fitz.Matrix(safe_scale, safe_scale), alpha=False
                )
                try:
                    if pixmap.width * pixmap.height > pixel_cap:
                        raise MultipleAskExtractionError(
                            "MULTIPLE_ASK_PDF_RENDER_LIMIT"
                        )
                    scanned = await self._ocr_questions(
                        pixmap.tobytes("png"), page_number=page_number
                    )
                    structured.extend(scanned)
                    pages.append(self._structured_source_text(scanned))
                finally:
                    del pixmap
                locators.append({"page_number": page_number, "source_kind": "ocr"})
        finally:
            document.close()
        max_questions = int(
            await RuntimeSettingsService(self._conn).get(
                "multiple_ask.max_extracted_questions", Scope(kind="global")
            )
        )
        if len(structured) > max_questions:
            raise QuestionLimitExceeded(len(structured), max_questions)
        normalized = normalize_source_text(
            "\f\n".join(pages),
            max_characters=get_settings().MULTIPLE_ASK_MAX_TEXT_CHARACTERS,
        )
        return (
            normalized,
            "pdf_ocr" if used_ocr else "pdf_embedded_text",
            locators,
            "gemini" if used_ocr else None,
            structured,
        )

    @staticmethod
    def _with_page_number(
        items: list[ExtractedQuestion], page_number: int
    ) -> list[ExtractedQuestion]:
        return [
            ExtractedQuestion(
                **{
                    **item.__dict__,
                    "source_locator": {
                        **item.source_locator,
                        "page_number": page_number,
                    },
                }
            )
            for item in items
        ]

    @staticmethod
    def _structured_source_text(items: list[ExtractedQuestion]) -> str:
        lines: list[str] = []
        current_section: str | None = None
        for item in items:
            if item.section_context and item.section_context != current_section:
                lines.append(item.section_context)
                current_section = item.section_context
            lines.append(f"{item.display_label}. {item.question_text}")
            lines.extend(
                f"{option['label']}. {option['text']}" for option in item.mcq_options
            )
        return "\n".join(lines)

    async def _ocr_questions(
        self, image_bytes: bytes, *, page_number: int
    ) -> list[ExtractedQuestion]:
        provider = await self._ocr_provider()
        structured_extract = getattr(provider, "extract_image_questions", None)
        if structured_extract is None:
            # Compatibility for bounded fakes and legacy providers; production
            # Gemini always takes the structured path.
            return self._with_page_number(
                extract_ordered_questions(await self._ocr_text(image_bytes, provider)),
                page_number,
            )
        try:
            raw_items: tuple[OCRExtractedQuestion, ...] = await structured_extract(
                image_bytes
            )
        except OCRProviderError as exc:
            raise MultipleAskExtractionError(exc.code, status_code=503) from exc
        output: list[ExtractedQuestion] = []
        for source_order, item in enumerate(raw_items):
            output.append(
                self._validated_structured_item(item, source_order, page_number)
            )
        return output

    @staticmethod
    def _validated_structured_item(
        item: OCRExtractedQuestion, source_order: int, page_number: int
    ) -> ExtractedQuestion:
        display_label = item.display_label.strip() if item.display_label else str(source_order + 1)
        options = [
            {"label": option["label"].upper().strip(), "text": option["text"].strip()}
            for option in item.mcq_options
        ]
        if item.answer_mode not in {
            "short",
            "long",
            "mcq",
            "not_clear",
        }:
            raise MultipleAskExtractionError("MULTIPLE_ASK_OCR_FAILED", status_code=503)
        if item.answer_mode == "not_clear" or not item.question_text:
            return ExtractedQuestion(
                source_order,
                display_label,
                item.section_context,
                item.question_text or "",
                "not_clear",
                (),
                item.unclear_reason or "QUESTION_TEXT_UNCLEAR",
                {"page_number": page_number, "display_label": display_label},
            )
        section_mode = _structured_section_mode(item.section_context)
        expected = section_mode or ("mcq" if options else item.answer_mode if item.answer_mode in {"short", "long"} else "short")
        if options and not _valid_dynamic_options(options):
            raise MultipleAskExtractionError("MULTIPLE_ASK_OCR_FAILED", status_code=503)
        if (
            item.answer_mode != expected
            or not item.question_text
            or (item.answer_mode != "mcq" and options)
        ):
            raise MultipleAskExtractionError("MULTIPLE_ASK_OCR_FAILED", status_code=503)
        return ExtractedQuestion(
            source_order,
            display_label,
            item.section_context,
            item.question_text,
            item.answer_mode,
            tuple(options),
            None,
            {"page_number": page_number, "display_label": display_label},
        )

    async def _ocr_text(
        self, image_bytes: bytes, provider: OCRProvider | None = None
    ) -> str:
        try:
            return await (provider or await self._ocr_provider()).extract_image_text(
                image_bytes
            )
        except OCRProviderError as exc:
            raise MultipleAskExtractionError(exc.code, status_code=503) from exc

    @staticmethod
    def _record(item: ExtractedQuestion) -> dict[str, Any]:
        normalized_question = normalize_question(item.question_text) or None
        return {
            "item_index": item.source_order,
            "display_label": item.display_label,
            "section_context": item.section_context,
            "question_text": item.question_text,
            "normalized_question": normalized_question,
            "question_hash": question_hash(normalized_question)
            if normalized_question
            else None,
            "answer_mode": item.answer_mode,
            "item_status": "needs_correction"
            if item.answer_mode == "not_clear"
            else "ready_to_answer",
            "mcq_options": list(item.mcq_options),
            "unclear_reason": item.unclear_reason,
            "source_locator": item.source_locator,
        }

    async def _extract_corrections(self, context: dict[str, Any]) -> str:
        async with self._conn.transaction():
            items = await self._repo.lock_job_items(job_id=str(context["id"]))
            if any(item["item_status"] == "pending_extraction" for item in items):
                raise MultipleAskExtractionError("MULTIPLE_ASK_LEGACY_RESUME_FORBIDDEN")
            remaining = await self._repo.lock_job_items(job_id=str(context["id"]))
            needs = any(item["item_status"] == "needs_correction" for item in remaining)
            await self._repo.finish_extraction(
                job_id=str(context["id"]), needs_correction=needs
            )
        return "needs_correction" if needs else "ready_to_answer"

    async def mark_queue_failure(
        self, session_id: str, *, queue_error_code: str | None = None
    ) -> None:
        """Reflect exhausted shared-queue retries in the separate business state.

        Keep the student-visible terminal reason deliberately small and stable.
        Queue errors otherwise contain implementation details that must not be
        exposed through the polling DTO.
        """
        safe_code = (
            queue_error_code
            if queue_error_code
            in {
                "MULTIPLE_ASK_OCR_TIMEOUT",
                "MULTIPLE_ASK_OCR_UNAVAILABLE",
                "MULTIPLE_ASK_OCR_FAILED",
            }
            else "MULTIPLE_ASK_EXTRACTION_FAILED"
        )
        await self.fail_and_refund(
            session_id=session_id,
            workflow_status="failed",
            error_code=safe_code,
        )

    async def fail_and_refund(
        self, *, session_id: str, workflow_status: str, error_code: str
    ) -> None:
        """Mark a post-charge extraction failure and refund the batch once."""
        async with self._conn.transaction():
            context = await self._repo.lock_extraction_context(session_id)
            if context is None or context["workflow_status"] in {
                "invalid",
                "limit_reached",
                "cancelled",
                "completed",
                "failed",
                "too_many_questions",
            }:
                return
            refunded = False
            if context.get("quota_status") == "committed":
                refunded = await UsageService().refund_committed(
                    self._conn, str(context["client_request_id"]), context["uid_hash"]
                )
            await self._repo.mark_terminal(
                str(context["id"]),
                workflow_status,
                datetime.now(UTC)
                + timedelta(days=get_settings().MULTIPLE_ASK_JOB_RETENTION_DAYS),
                error_code=error_code,
                quota_refunded=refunded,
            )

    @staticmethod
    def _validate_correction_options(
        answer_mode: str, options: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if answer_mode != "mcq":
            if options:
                raise MultipleAskExtractionError(
                    "MULTIPLE_ASK_CORRECTION_OPTIONS_INVALID"
                )
            return []
        normalized = [
            {
                "label": str(option.get("label", "")).upper(),
                "text": str(option.get("text", "")).strip(),
            }
            for option in options
        ]
        if normalized and not _valid_dynamic_options(normalized):
            raise MultipleAskExtractionError("MULTIPLE_ASK_CORRECTION_OPTIONS_INVALID")
        return normalized

    async def _owned_status_locked(self, job: dict[str, Any]) -> dict[str, Any]:
        record = await self._repo.get_owned_job_status(
            job_id=str(job["id"]), uid_hash=job["uid_hash"]
        )
        if record is None:
            raise MultipleAskExtractionError(
                "MULTIPLE_ASK_JOB_NOT_FOUND", status_code=404
            )
        return record
