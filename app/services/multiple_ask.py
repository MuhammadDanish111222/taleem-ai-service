"""Feature-gated Module 5 Run 1 orchestration. No OCR or answer generation."""

from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any
from uuid import uuid4

import asyncpg
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import get_settings
from app.repositories.multiple_ask_repository import (
    MultipleAskRepository,
    MultipleAskStateError,
)
from app.services.jobs.queue import JobQueueService
from app.services.multiple_ask_storage import (
    TemporaryStorageError,
    TemporaryStorageObjectNotFound,
    TemporaryStorageObjectTooLarge,
    TemporaryUploadStorage,
)
from app.services.usage.models import AccountTier
from app.services.usage.service import UsageLimitExceeded, UsageService

_SCOPE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_FILE_CONTENT_TYPES = {
    "image": {"image/jpeg", "image/png", "image/webp"},
    "pdf": {"application/pdf"},
}


class MultipleAskError(RuntimeError):
    def __init__(self, code: str, *, status_code: int = 400):
        self.code, self.status_code = code, status_code
        super().__init__(code)


class CanonicalValidationError(MultipleAskError):
    pass


def normalize_scope_id(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise MultipleAskError("MULTIPLE_ASK_SCOPE_INVALID")
        return None
    normalized = value.strip()
    if not _SCOPE_IDENTIFIER.fullmatch(normalized):
        raise MultipleAskError("MULTIPLE_ASK_SCOPE_INVALID")
    return normalized


class MultipleAskService:
    def __init__(
        self,
        conn: asyncpg.Connection,
        *,
        storage: TemporaryUploadStorage | None = None,
        usage: UsageService | None = None,
    ):
        self._conn, self._repo = conn, MultipleAskRepository(conn)
        self._storage, self._usage = (
            storage or TemporaryUploadStorage(),
            usage or UsageService(),
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _scope(
        board_id: str, class_id: str, subject_id: str, chapter_id: str | None
    ) -> tuple[str, str, str, str | None]:
        return (
            normalize_scope_id(board_id, required=True) or "",
            normalize_scope_id(class_id, required=True) or "",
            normalize_scope_id(subject_id, required=True) or "",
            normalize_scope_id(chapter_id, required=False),
        )

    def _max_bytes(self, input_kind: str) -> int:
        settings = get_settings()
        if input_kind == "image":
            return settings.MULTIPLE_ASK_MAX_IMAGE_BYTES
        if input_kind == "pdf":
            return settings.MULTIPLE_ASK_MAX_PDF_BYTES
        raise MultipleAskError("MULTIPLE_ASK_INPUT_INVALID")

    async def create_file_session(
        self,
        *,
        client_request_id: str,
        uid: str,
        tier: AccountTier,
        input_kind: str,
        content_type: str,
        size_bytes: int,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
    ) -> dict[str, Any]:
        if (
            input_kind not in _FILE_CONTENT_TYPES
            or content_type not in _FILE_CONTENT_TYPES[input_kind]
        ):
            raise MultipleAskError("MULTIPLE_ASK_INPUT_INVALID")
        if not 0 < size_bytes <= self._max_bytes(input_kind):
            raise MultipleAskError("MULTIPLE_ASK_INPUT_TOO_LARGE")
        scope = self._scope(board_id, class_id, subject_id, chapter_id)
        now, settings = self._now(), get_settings()
        capability_expires_at = now + timedelta(
            seconds=settings.MULTIPLE_ASK_SESSION_TTL_SECONDS
        )
        session_id = str(uuid4())
        try:
            async with self._conn.transaction():
                session = await self._repo.create_or_get_session(
                    session_id=session_id,
                    client_request_id=client_request_id,
                    uid_hash=self._usage.uid_hash(uid),
                    account_tier=tier.value,
                    input_kind=input_kind,
                    expected_content_type=content_type,
                    expected_size_bytes=size_bytes,
                    storage_bucket=settings.MULTIPLE_ASK_TEMPORARY_BUCKET,
                    storage_object_key=f"multiple-ask-temp/{session_id}/source",
                    upload_capability_expires_at=capability_expires_at,
                    board_id=scope[0],
                    class_id=scope[1],
                    subject_id=scope[2],
                    chapter_id=scope[3],
                )
        except MultipleAskStateError as exc:
            raise MultipleAskError(str(exc), status_code=409) from exc
        if (
            session["status"] not in {"created", "uploaded"}
            or session["upload_capability_expires_at"] <= now
        ):
            raise MultipleAskError("MULTIPLE_ASK_SESSION_UNAVAILABLE", status_code=409)
        try:
            capability = await self._storage.create_signed_upload_url(
                bucket=session["storage_bucket"],
                object_key=session["storage_object_key"],
                content_type=content_type,
            )
        except TemporaryStorageError as exc:
            raise MultipleAskError(str(exc), status_code=503) from exc
        return {
            "session_id": str(session["id"]),
            "upload_url": capability.upload_url,
            "upload_method": capability.method,
            "upload_headers": capability.required_headers,
            # The service accepts finalization only until this expiry. Supabase's
            # signed-upload token lifetime itself follows its supported API.
            "upload_capability_expires_at": capability_expires_at.isoformat(),
        }

    async def submit_text(
        self,
        *,
        client_request_id: str,
        uid: str,
        tier: AccountTier,
        text: str,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str | None,
    ) -> dict[str, Any]:
        # This is only early feedback. Railway validates the persisted text again.
        if (
            not text.strip()
            or len(text) > get_settings().MULTIPLE_ASK_MAX_TEXT_CHARACTERS
        ):
            raise MultipleAskError("MULTIPLE_ASK_INPUT_INVALID")
        scope = self._scope(board_id, class_id, subject_id, chapter_id)
        try:
            async with self._conn.transaction():
                session = await self._repo.create_or_get_session(
                    session_id=str(uuid4()),
                    client_request_id=client_request_id,
                    uid_hash=self._usage.uid_hash(uid),
                    account_tier=tier.value,
                    input_kind="text",
                    expected_content_type=None,
                    expected_size_bytes=None,
                    storage_bucket=None,
                    storage_object_key=None,
                    upload_capability_expires_at=None,
                    board_id=scope[0],
                    class_id=scope[1],
                    subject_id=scope[2],
                    chapter_id=scope[3],
                    text=text,
                )
                return await self._enqueue_validation_locked(session)
        except MultipleAskStateError as exc:
            raise MultipleAskError(str(exc), status_code=409) from exc

    async def finalize_file(
        self, *, session_id: str, client_request_id: str, uid: str, tier: AccountTier
    ) -> dict[str, Any]:
        async with self._conn.transaction():
            session = await self._repo.lock_session(
                session_id=session_id,
                uid_hash=self._usage.uid_hash(uid),
                client_request_id=client_request_id,
            )
            if session is None:
                raise MultipleAskError(
                    "MULTIPLE_ASK_SESSION_NOT_FOUND", status_code=404
                )
            if session["input_kind"] == "text":
                raise MultipleAskError("MULTIPLE_ASK_INPUT_INVALID")
            if session["status"] == "finalized":
                existing = await self._repo.job_for_session(session_id)
                if existing is None:
                    raise MultipleAskError(
                        "MULTIPLE_ASK_FINALIZE_INCOMPLETE", status_code=409
                    )
                return self._job_response(existing)
            if (
                session["status"] not in {"created", "uploaded"}
                or session["upload_capability_expires_at"] <= self._now()
            ):
                raise MultipleAskError(
                    "MULTIPLE_ASK_SESSION_UNAVAILABLE", status_code=409
                )
            # Metadata is only advisory. Canonical magic/PDF inspection happens in
            # the Railway-owned handler after this fast durable enqueue.
            return await self._enqueue_validation_locked(session)

    async def _enqueue_validation_locked(
        self, session: dict[str, Any]
    ) -> dict[str, Any]:
        existing = await self._repo.job_for_session(str(session["id"]))
        if existing is not None:
            return self._job_response(existing)
        queue_job = await JobQueueService(self._conn).enqueue_job(
            job_type="multiple_ask_validate",
            payload={
                "multiple_ask_session_id": str(session["id"]),
                "schema_version": 2,
            },
            idempotency_key=f"multiple-ask:{session['id']}",
        )
        job = await self._repo.finalize_with_validation_job(
            session=session,
            queue_job_id=str(queue_job["id"]),
            raw_source_expires_at=self._now()
            + timedelta(hours=get_settings().MULTIPLE_ASK_RAW_SOURCE_RETENTION_HOURS),
        )
        return self._job_response(job, queue_status=queue_job["status"])

    async def canonical_validate_context(self, context: dict[str, Any]) -> None:
        """Read bounded private source bytes only; never call OCR or a text LLM."""
        settings = get_settings()
        if context["input_kind"] == "text":
            text = context.get("input_text")
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > settings.MULTIPLE_ASK_MAX_TEXT_CHARACTERS
            ):
                raise CanonicalValidationError("MULTIPLE_ASK_TEXT_INVALID")
            if "\x00" in text:
                raise CanonicalValidationError("MULTIPLE_ASK_TEXT_INVALID")
            return
        bucket, key = context.get("source_bucket"), context.get("source_key")
        if not isinstance(bucket, str) or not isinstance(key, str):
            raise CanonicalValidationError("MULTIPLE_ASK_SOURCE_UNAVAILABLE")
        max_bytes = self._max_bytes(context["input_kind"])
        try:
            prefix = await self._storage.read_object_prefix(
                bucket=bucket, object_key=key, max_bytes=16
            )
            if context["input_kind"] == "image":
                # Reading at most the configured image maximum makes actual
                # object size authoritative rather than browser metadata.
                raw_image = await self._storage.read_object_limited(
                    bucket=bucket, object_key=key, max_bytes=max_bytes
                )
                expected = context["expected_content_type"]
                observed = self._image_type(raw_image)
                if observed is None or observed != expected:
                    raise CanonicalValidationError("MULTIPLE_ASK_IMAGE_MAGIC_INVALID")
                self._verify_image(
                    raw_image, expected, settings.MULTIPLE_ASK_MAX_IMAGE_PIXELS
                )
                return
            if not prefix.startswith(b"%PDF-"):
                raise CanonicalValidationError("MULTIPLE_ASK_PDF_INVALID")
            raw_pdf = await self._storage.read_object_limited(
                bucket=bucket, object_key=key, max_bytes=max_bytes
            )
        except TemporaryStorageObjectTooLarge as exc:
            raise CanonicalValidationError("MULTIPLE_ASK_SOURCE_TOO_LARGE") from exc
        except TemporaryStorageObjectNotFound as exc:
            raise CanonicalValidationError("MULTIPLE_ASK_SOURCE_UNAVAILABLE") from exc
        except TemporaryStorageError:
            # Network/storage availability is retryable, so do not terminally mark it invalid.
            raise
        try:
            reader = PdfReader(BytesIO(raw_pdf), strict=True)
            if not 1 <= len(reader.pages) <= settings.MULTIPLE_ASK_MAX_PDF_PAGES:
                raise CanonicalValidationError("MULTIPLE_ASK_PDF_PAGE_LIMIT")
            extracted_characters = 0
            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_characters += len(page_text)
                if (
                    extracted_characters
                    > settings.MULTIPLE_ASK_MAX_PDF_EXTRACTED_CHARACTERS
                ):
                    raise CanonicalValidationError("MULTIPLE_ASK_PDF_TEXT_LIMIT")
        except CanonicalValidationError:
            raise
        except (PdfReadError, ValueError, OSError, EOFError, RuntimeError) as exc:
            raise CanonicalValidationError("MULTIPLE_ASK_PDF_INVALID") from exc

    async def validate_and_charge(self, session_id: str) -> str:
        """Railway-owned canonical validation and exactly-once batch charging."""
        settings = get_settings()
        async with self._conn.transaction():
            context = await self._repo.lock_validation_context(session_id)
            if context is None:
                raise MultipleAskError("MULTIPLE_ASK_PARENT_NOT_FOUND")
            if context["workflow_status"] in {
                "validated",
                "invalid",
                "limit_reached",
                "cancelled",
            }:
                return context["workflow_status"]
            if context["workflow_status"] == "queued":
                await self._repo.mark_validating(str(context["id"]))
                context["workflow_status"] = "validating"

        try:
            await self.canonical_validate_context(context)
        except CanonicalValidationError:
            async with self._conn.transaction():
                current = await self._repo.lock_validation_context(session_id)
                if current is not None and current["workflow_status"] not in {
                    "validated",
                    "invalid",
                    "limit_reached",
                    "cancelled",
                }:
                    await self._repo.mark_terminal(
                        str(current["id"]),
                        "invalid",
                        self._now()
                        + timedelta(days=settings.MULTIPLE_ASK_JOB_RETENTION_DAYS),
                    )
            return "invalid"

        async with self._conn.transaction():
            current = await self._repo.lock_validation_context(session_id)
            if current is None:
                raise MultipleAskError("MULTIPLE_ASK_PARENT_NOT_FOUND")
            if current["workflow_status"] in {
                "validated",
                "invalid",
                "limit_reached",
                "cancelled",
            }:
                return current["workflow_status"]
            try:
                # A nested transaction rolls back a rejected reservation while
                # the outer transaction records the terminal limit state.
                async with self._conn.transaction():
                    await self._usage.reserve_for_uid_hash(
                        self._conn,
                        request_id=str(current["client_request_id"]),
                        uid_hash=current["uid_hash"],
                        tier=AccountTier(current["account_tier"]),
                        feature="multiple_question_batch",
                    )
                    await self._usage.commit(
                        self._conn,
                        str(current["client_request_id"]),
                        current["uid_hash"],
                    )
            except UsageLimitExceeded:
                await self._repo.mark_terminal(
                    str(current["id"]),
                    "limit_reached",
                    self._now()
                    + timedelta(days=settings.MULTIPLE_ASK_JOB_RETENTION_DAYS),
                )
                return "limit_reached"
            await self._repo.mark_validated_and_charged(str(current["id"]))
            return "validated"

    @staticmethod
    def _image_type(prefix: bytes) -> str | None:
        if prefix.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
            return "image/webp"
        return None

    @staticmethod
    def _verify_image(
        raw_image: bytes, expected_content_type: str, max_pixels: int
    ) -> None:
        """Confirm that a magic-byte match is a decodable, bounded image."""
        expected_format = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }[expected_content_type]
        try:
            # Pillow raises a decompression-bomb warning before allocating the
            # full image. Treat that warning as a validation failure.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw_image)) as probe:
                    if probe.format != expected_format:
                        raise CanonicalValidationError(
                            "MULTIPLE_ASK_IMAGE_FORMAT_INVALID"
                        )
                    width, height = probe.size
                    if width < 1 or height < 1 or width * height > max_pixels:
                        raise CanonicalValidationError(
                            "MULTIPLE_ASK_IMAGE_DIMENSIONS_INVALID"
                        )
                    probe.verify()
                # verify() intentionally invalidates the first handle; loading
                # a new one detects truncated data that passed only the header.
                with Image.open(BytesIO(raw_image)) as verified:
                    verified.load()
        except CanonicalValidationError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise CanonicalValidationError("MULTIPLE_ASK_IMAGE_INVALID") from exc

    @staticmethod
    def _job_response(
        job: dict[str, Any], *, queue_status: str = "queued"
    ) -> dict[str, Any]:
        return {
            "job_id": str(job["id"]),
            "workflow_status": job["workflow_status"],
            "queue_status": queue_status,
        }


class MultipleAskRetentionService:
    """Bounded, restart-safe retention without retaining source material in audit."""

    def __init__(
        self, conn: asyncpg.Connection, *, storage: TemporaryUploadStorage | None = None
    ):
        self._conn, self._repo = conn, MultipleAskRepository(conn)
        self._storage = storage or TemporaryUploadStorage()

    async def cleanup_once(
        self, *, run_id: str, limit: int | None = None
    ) -> dict[str, int]:
        batch = limit or get_settings().MULTIPLE_ASK_CLEANUP_BATCH_SIZE
        counts = {"unfinalized": 0, "raw_sources": 0, "jobs": 0, "failed": 0}
        async with self._conn.transaction():
            unfinalized = await self._repo.claim_expired_unfinalized(limit=batch)
        for session in unfinalized:
            if await self._delete_storage_if_present(session):
                async with self._conn.transaction():
                    await self._repo.audit_cleanup(
                        run_id=run_id,
                        session_id=str(session["id"]),
                        subject_kind="upload_capability",
                        action="storage_delete_requested",
                    )
                    await self._repo.delete_unfinalized_session(str(session["id"]))
                counts["unfinalized"] += 1
            else:
                await self._cleanup_failure(
                    run_id, str(session["id"]), "upload_capability"
                )
                counts["failed"] += 1
        async with self._conn.transaction():
            raw_sources = await self._repo.claim_expired_raw_sources(limit=batch)
        for session in raw_sources:
            if await self._delete_storage_if_present(session):
                async with self._conn.transaction():
                    await self._repo.purge_raw_source(str(session["id"]))
                    await self._repo.audit_cleanup(
                        run_id=run_id,
                        session_id=str(session["id"]),
                        subject_kind="raw_source",
                        action="metadata_purged",
                    )
                counts["raw_sources"] += 1
            else:
                await self._cleanup_failure(run_id, str(session["id"]), "raw_source")
                counts["failed"] += 1
        async with self._conn.transaction():
            jobs = await self._repo.claim_expired_jobs(limit=batch)
        for job in jobs:
            if await self._delete_storage_if_present(job):
                async with self._conn.transaction():
                    await self._repo.audit_cleanup(
                        run_id=run_id,
                        session_id=str(job["upload_session_id"]),
                        subject_kind="job",
                        action="metadata_purged",
                    )
                    await self._repo.delete_expired_job_and_session(
                        job_id=str(job["id"]), session_id=str(job["upload_session_id"])
                    )
                counts["jobs"] += 1
            else:
                async with self._conn.transaction():
                    await self._repo.release_job_cleanup_claim(str(job["id"]))
                    await self._repo.audit_cleanup(
                        run_id=run_id,
                        session_id=str(job["upload_session_id"]),
                        subject_kind="job",
                        action="failed",
                        error_code="STORAGE_DELETE_FAILED",
                    )
                counts["failed"] += 1
        return counts

    async def _delete_storage_if_present(self, record: dict[str, Any]) -> bool:
        bucket, key = (
            record.get("storage_bucket") or record.get("source_bucket"),
            record.get("storage_object_key") or record.get("source_key"),
        )
        if not bucket or not key:
            return True
        try:
            await self._storage.delete_object(bucket=bucket, object_key=key)
            return True
        except TemporaryStorageError:
            return False

    async def _cleanup_failure(
        self, run_id: str, session_id: str, subject_kind: str
    ) -> None:
        async with self._conn.transaction():
            await self._repo.release_cleanup_claim(session_id)
            await self._repo.audit_cleanup(
                run_id=run_id,
                session_id=session_id,
                subject_kind=subject_kind,
                action="failed",
                error_code="STORAGE_DELETE_FAILED",
            )
