"""Run 1 foundation guardrails without a live Storage account or provider."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter

from app.repositories.multiple_ask_repository import (
    MultipleAskRepository,
    MultipleAskStateError,
)
from app.services.jobs.queue import JobQueueService
from app.services.multiple_ask import (
    CanonicalValidationError,
    MultipleAskError,
    MultipleAskRetentionService,
    MultipleAskService,
    normalize_scope_id,
)
from app.services.multiple_ask_storage import (
    TemporaryStorageError,
    TemporaryUploadStorage,
)
from app.workers.handlers.multiple_ask_validate import handle_multiple_ask_validate


def test_module5_correction_migration_preserves_shared_boundaries():
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "0011_module5_run1_foundation_corrections.sql"
    ).read_text(encoding="utf-8")
    assert (
        "board_id" in migration
        and "class_id" in migration
        and "subject_id" in migration
    )
    assert "multiple_ask_jobs_workflow_check" in migration
    assert "'mixed'" not in migration
    assert "ALTER TABLE job_queue" not in migration
    assert "solved_papers" not in migration
    assert "REFERENCES ai_answers(id) ON DELETE SET NULL" in migration
    assert "ON DELETE SET NULL" in migration


def test_scope_identifiers_are_required_safe_and_normalized():
    assert normalize_scope_id(" class-9 ", required=True) == "class-9"
    assert normalize_scope_id(None, required=False) is None
    with pytest.raises(MultipleAskError, match="SCOPE_INVALID"):
        normalize_scope_id(" ", required=True)
    with pytest.raises(MultipleAskError, match="SCOPE_INVALID"):
        normalize_scope_id("class 9", required=True)


@pytest.mark.asyncio
async def test_idempotent_retry_cannot_change_persisted_scope():
    existing = {
        "input_kind": "pdf",
        "expected_content_type": "application/pdf",
        "expected_size_bytes": 123,
        "board_id": "punjab",
        "class_id": "class-9",
        "subject_id": "physics",
        "chapter_id": "motion",
    }

    class Conn:
        def __init__(self):
            self.calls = 0

        async def fetchrow(self, *_args):
            self.calls += 1
            return None if self.calls == 1 else existing

    repository = MultipleAskRepository(Conn())
    with pytest.raises(MultipleAskStateError, match="IDEMPOTENCY_CONFLICT"):
        await repository.create_or_get_session(
            session_id="123e4567-e89b-42d3-a456-426614174000",
            client_request_id="123e4567-e89b-42d3-a456-426614174001",
            uid_hash="a" * 64,
            account_tier="google",
            input_kind="pdf",
            expected_content_type="application/pdf",
            expected_size_bytes=123,
            storage_bucket="private",
            storage_object_key="opaque",
            upload_capability_expires_at=None,
            board_id="punjab",
            class_id="class-9",
            subject_id="chemistry",
            chapter_id="motion",
        )


@pytest.mark.asyncio
async def test_session_scope_is_copied_into_the_durable_validation_job(monkeypatch):
    class Repo:
        existing = None

        async def job_for_session(self, _session_id):
            return self.existing

        async def finalize_with_validation_job(
            self, *, session, queue_job_id, raw_source_expires_at
        ):
            assert (
                session["board_id"],
                session["class_id"],
                session["subject_id"],
                session["chapter_id"],
            ) == ("punjab", "class-9", "physics", "motion")
            assert queue_job_id == "123e4567-e89b-42d3-a456-426614174002"
            self.existing = {
                "id": "123e4567-e89b-42d3-a456-426614174003",
                "workflow_status": "queued",
            }
            return self.existing

    async def enqueue_once(_self, **_kwargs):
        enqueue_once.calls += 1
        return {"id": "123e4567-e89b-42d3-a456-426614174002", "status": "queued"}

    enqueue_once.calls = 0
    monkeypatch.setattr(JobQueueService, "enqueue_job", enqueue_once)
    service = MultipleAskService(object(), storage=object())
    service._repo = Repo()
    session = {
        "id": "123e4567-e89b-42d3-a456-426614174000",
        "board_id": "punjab",
        "class_id": "class-9",
        "subject_id": "physics",
        "chapter_id": "motion",
    }
    first = await service._enqueue_validation_locked(session)
    second = await service._enqueue_validation_locked(session)
    assert first["workflow_status"] == second["workflow_status"] == "queued"
    assert enqueue_once.calls == 1


def test_finalized_sources_are_not_selected_by_upload_capability_expiry_cleanup():
    repository = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "repositories"
        / "multiple_ask_repository.py"
    ).read_text(encoding="utf-8")
    capability_section = repository.split("async def claim_expired_unfinalized", 1)[
        1
    ].split("async def claim_expired_raw_sources", 1)[0]
    assert "status IN ('created','uploaded')" in capability_section
    assert "finalized" not in capability_section


@pytest.mark.asyncio
async def test_signed_upload_capability_is_same_origin_put_without_credentials(
    monkeypatch,
):
    storage = TemporaryUploadStorage(
        base_url="https://project.supabase.co", service_key="test"
    )
    monkeypatch.setattr(
        storage,
        "_request_json",
        lambda *_args: {"url": "/object/upload/sign/private/opaque?token=short"},
    )
    capability = await storage.create_signed_upload_url(
        bucket="private", object_key="opaque", content_type="application/pdf"
    )
    assert capability.upload_url.startswith("https://project.supabase.co/storage/v1/")
    assert capability.method == "PUT"
    assert capability.required_headers == {"Content-Type": "application/pdf"}
    assert "Authorization" not in capability.required_headers


@pytest.mark.asyncio
async def test_signed_upload_url_cannot_redirect_to_another_origin(monkeypatch):
    storage = TemporaryUploadStorage(
        base_url="https://project.supabase.co", service_key="test"
    )
    monkeypatch.setattr(
        storage, "_request_json", lambda *_args: {"url": "https://bad.example/upload"}
    )
    with pytest.raises(TemporaryStorageError, match="INVALID_RESPONSE"):
        await storage.create_signed_upload_url(
            bucket="temporary", object_key="opaque", content_type="image/png"
        )


@pytest.mark.asyncio
async def test_fake_image_mime_or_magic_is_rejected_without_any_llm_path():
    class Storage:
        async def read_object_prefix(self, **_kwargs):
            return b"not-an-image"

        async def read_object_limited(self, **_kwargs):
            return b"not-an-image"

    service = MultipleAskService(None, storage=Storage())
    with pytest.raises(CanonicalValidationError, match="IMAGE_MAGIC_INVALID"):
        await service.canonical_validate_context(
            {
                "input_kind": "image",
                "source_bucket": "private",
                "source_key": "opaque",
                "expected_content_type": "image/png",
            }
        )


@pytest.mark.asyncio
async def test_image_with_valid_magic_but_invalid_body_is_rejected_before_quota():
    class Storage:
        async def read_object_prefix(self, **_kwargs):
            return b"\x89PNG\r\n\x1a\nnot-a-real-png"

        async def read_object_limited(self, **_kwargs):
            return b"\x89PNG\r\n\x1a\nnot-a-real-png"

    service = MultipleAskService(None, storage=Storage())
    with pytest.raises(CanonicalValidationError, match="IMAGE_INVALID"):
        await service.canonical_validate_context(
            {
                "input_kind": "image",
                "source_bucket": "private",
                "source_key": "opaque",
                "expected_content_type": "image/png",
            }
        )


def test_image_decoder_enforces_pixel_limit_before_any_ocr():
    image = Image.new("RGB", (2, 2), color="white")
    raw = BytesIO()
    image.save(raw, format="PNG")
    with pytest.raises(CanonicalValidationError, match="IMAGE_DIMENSIONS_INVALID"):
        MultipleAskService._verify_image(raw.getvalue(), "image/png", max_pixels=1)


@pytest.mark.asyncio
async def test_invalid_or_excessive_pasted_text_is_rejected_before_quota():
    service = MultipleAskService(None, storage=object())
    with pytest.raises(CanonicalValidationError, match="TEXT_INVALID"):
        await service.canonical_validate_context(
            {"input_kind": "text", "input_text": "\x00bad"}
        )


@pytest.mark.asyncio
async def test_malformed_and_excessive_page_pdf_are_rejected_with_bounded_reads():
    class Storage:
        def __init__(self, payload):
            self.payload = payload

        async def read_object_prefix(self, **_kwargs):
            return self.payload[:16]

        async def read_object_limited(self, **_kwargs):
            return self.payload

    malformed = MultipleAskService(None, storage=Storage(b"%PDF-not-real"))
    with pytest.raises(CanonicalValidationError, match="PDF_INVALID"):
        await malformed.canonical_validate_context(
            {"input_kind": "pdf", "source_bucket": "p", "source_key": "k"}
        )
    stream = BytesIO()
    writer = PdfWriter()
    for _ in range(11):
        writer.add_blank_page(width=72, height=72)
    writer.write(stream)
    too_many = MultipleAskService(None, storage=Storage(stream.getvalue()))
    with pytest.raises(CanonicalValidationError, match="PDF_PAGE_LIMIT"):
        await too_many.canonical_validate_context(
            {"input_kind": "pdf", "source_bucket": "p", "source_key": "k"}
        )


@pytest.mark.asyncio
async def test_empty_pdf_and_excessive_embedded_text_are_rejected(monkeypatch):
    class Storage:
        async def read_object_prefix(self, **_kwargs):
            return b"%PDF-1.7"

        async def read_object_limited(self, **_kwargs):
            return b"%PDF-1.7"

    class EmptyReader:
        pages: list[object] = []

    monkeypatch.setattr(
        "app.services.multiple_ask.PdfReader", lambda *_args, **_kwargs: EmptyReader()
    )
    service = MultipleAskService(None, storage=Storage())
    with pytest.raises(CanonicalValidationError, match="PDF_PAGE_LIMIT"):
        await service.canonical_validate_context(
            {"input_kind": "pdf", "source_bucket": "p", "source_key": "k"}
        )

    class Page:
        def extract_text(self):
            return "x" * 30_001

    class TextHeavyReader:
        pages = [Page()]

    monkeypatch.setattr(
        "app.services.multiple_ask.PdfReader",
        lambda *_args, **_kwargs: TextHeavyReader(),
    )
    with pytest.raises(CanonicalValidationError, match="PDF_TEXT_LIMIT"):
        await service.canonical_validate_context(
            {"input_kind": "pdf", "source_bucket": "p", "source_key": "k"}
        )


@pytest.mark.asyncio
async def test_valid_canonical_validation_charges_once_after_validation_only():
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Conn:
        def transaction(self):
            return Transaction()

    class Repo:
        def __init__(self):
            self.context = {
                "id": "job",
                "client_request_id": "123e4567-e89b-42d3-a456-426614174000",
                "uid_hash": "a" * 64,
                "account_tier": "google",
                "workflow_status": "queued",
            }
            self.validated = 0

        async def lock_validation_context(self, _session_id):
            return self.context

        async def mark_validating(self, _job_id):
            self.context["workflow_status"] = "validating"
            return True

        async def mark_validated_and_charged(self, _job_id):
            self.context["workflow_status"] = "validated"
            self.validated += 1

        async def mark_terminal(self, *_args):
            raise AssertionError("valid input must not become terminal")

    class Usage:
        def __init__(self):
            self.reserves = self.commits = 0

        async def reserve_for_uid_hash(self, *_args, **_kwargs):
            self.reserves += 1

        async def commit(self, *_args):
            self.commits += 1

    service = MultipleAskService(Conn(), storage=object(), usage=Usage())
    repo = Repo()
    service._repo = repo

    async def canonical(_context):
        return None

    service.canonical_validate_context = canonical  # type: ignore[method-assign]
    assert await service.validate_and_charge("session") == "validated"
    assert await service.validate_and_charge("session") == "validated"
    assert service._usage.reserves == 1
    assert service._usage.commits == 1
    assert repo.validated == 1


@pytest.mark.asyncio
async def test_invalid_canonical_validation_never_reserves_quota():
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Conn:
        def transaction(self):
            return Transaction()

    class Repo:
        context = {"id": "job", "workflow_status": "queued"}

        async def lock_validation_context(self, _session_id):
            return self.context

        async def mark_validating(self, _job_id):
            self.context["workflow_status"] = "validating"
            return True

        async def mark_terminal(self, _job_id, status, _retention):
            self.context["workflow_status"] = status

    class Usage:
        async def reserve_for_uid_hash(self, *_args, **_kwargs):
            raise AssertionError("invalid content must not reserve quota")

    service = MultipleAskService(Conn(), storage=object(), usage=Usage())
    service._repo = Repo()

    async def invalid(_context):
        raise CanonicalValidationError("MULTIPLE_ASK_TEXT_INVALID")

    service.canonical_validate_context = invalid  # type: ignore[method-assign]
    assert await service.validate_and_charge("session") == "invalid"


@pytest.mark.asyncio
async def test_raw_source_cleanup_is_bounded_and_redacts_after_storage_delete():
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Conn:
        def transaction(self):
            return Transaction()

    class Repo:
        async def claim_expired_unfinalized(self, **_kwargs):
            return []

        async def claim_expired_raw_sources(self, **_kwargs):
            return [
                {
                    "id": "123e4567-e89b-42d3-a456-426614174000",
                    "storage_bucket": "private",
                    "storage_object_key": "opaque",
                }
            ]

        async def claim_expired_jobs(self, **_kwargs):
            return []

        async def purge_raw_source(self, session_id):
            self.purged = session_id

        async def audit_cleanup(self, **kwargs):
            self.audit = kwargs

    class Storage:
        async def delete_object(self, **kwargs):
            self.deleted = kwargs

    retention = MultipleAskRetentionService(Conn(), storage=Storage())
    repo = Repo()
    retention._repo = repo
    result = await retention.cleanup_once(
        run_id="123e4567-e89b-42d3-a456-426614174099", limit=1
    )
    assert result == {"unfinalized": 0, "raw_sources": 1, "jobs": 0, "failed": 0}
    assert repo.purged == "123e4567-e89b-42d3-a456-426614174000"
    assert repo.audit["subject_kind"] == "raw_source"
    assert "opaque" not in str(repo.audit)


@pytest.mark.asyncio
async def test_worker_validates_before_durably_scheduling_extraction(monkeypatch):
    calls = []

    async def fake_validate_and_charge(_self, session_id):
        calls.append(session_id)
        return "validated"

    monkeypatch.setattr(
        MultipleAskService, "validate_and_charge", fake_validate_and_charge
    )
    scheduled = []

    async def fake_start(_self, session_id):
        scheduled.append(session_id)

    monkeypatch.setattr(
        "app.workers.handlers.multiple_ask_validate.MultipleAskExtractionService.start_initial_extraction",
        fake_start,
    )
    await handle_multiple_ask_validate(
        {
            "payload": {
                "multiple_ask_session_id": "123e4567-e89b-42d3-a456-426614174000"
            }
        },
        object(),
    )
    assert calls == ["123e4567-e89b-42d3-a456-426614174000"]
    assert scheduled == ["123e4567-e89b-42d3-a456-426614174000"]


def test_multiple_ask_validation_has_no_deepseek_or_ocr_path():
    source = (
        (Path(__file__).resolve().parents[1] / "app" / "services" / "multiple_ask.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    handler = (
        (
            Path(__file__).resolve().parents[1]
            / "app"
            / "workers"
            / "handlers"
            / "multiple_ask_validate.py"
        )
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "from app.providers" not in source
    assert "from app.providers" not in handler
    assert "generate(" not in source and "generate(" not in handler
