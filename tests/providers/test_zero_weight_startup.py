"""Regression tests ensuring zero-weight startup without torch, transformers, onnxruntime, or pytesseract,
and validating Railway production startup configuration.
"""

import sys
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.core.worker_modes import owned_job_types, resolve_worker_mode
from app.providers.embeddings.voyage import VoyageEmbeddingProvider
from app.providers.ocr.base import OCRProviderError
from app.providers.ocr.gemini import GeminiOCRProvider
from app.workers.main import Worker


def test_procfile_contains_no_dev_and_branches():
    procfile_path = Path(__file__).parents[2] / "Procfile"
    assert procfile_path.exists(), "Procfile must exist in repo root"
    content = procfile_path.read_text(encoding="utf-8")

    assert "uv run --no-dev" in content, "Procfile must use 'uv run --no-dev'"
    assert "TALEEM_PROCESS_ROLE" in content, "Procfile must handle TALEEM_PROCESS_ROLE"
    assert "app.workers.main" in content, "Worker process branch must launch app.workers.main"
    assert "uvicorn app.main:app" in content, "API process branch must launch uvicorn app.main:app"


def test_zero_weight_imports_do_not_load_heavy_libraries():
    # Force import of main application and worker entry points
    import app.main  # noqa: F401
    import app.providers.embeddings.voyage  # noqa: F401
    import app.providers.llm.deepseek  # noqa: F401
    import app.providers.ocr.gemini  # noqa: F401
    import app.workers.main  # noqa: F401

    forbidden = {"torch", "transformers", "onnxruntime", "pytesseract", "huggingface_hub"}
    loaded = set(sys.modules.keys())
    intersection = forbidden.intersection(loaded)
    assert not intersection, f"Forbidden heavy modules were imported: {intersection}"


def test_railway_worker_mode_and_credentials(monkeypatch):
    monkeypatch.setattr(get_settings(), "WORKER_MODE", "railway_public")
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "query_live_key")
    monkeypatch.setattr(get_settings(), "VOYAGE_ADMIN_API_KEY", "")

    mode = resolve_worker_mode("railway_public")
    types = owned_job_types(mode)
    assert types == frozenset({"multiple_ask_validate", "multiple_ask_extract", "multiple_ask_answer"})

    # Ensure Local Admin ingestion jobs are excluded from Railway
    admin_jobs = {"embed_chunks", "embed_questions", "jsonl_ingest", "ingestion_job"}
    assert admin_jobs.isdisjoint(types)

    # Instantiate worker and verify supported types
    worker = Worker(worker_mode="railway_public")
    assert worker.worker_mode == "railway_public"

    # Verify query provider works with VOYAGE_API_KEY and does not need VOYAGE_ADMIN_API_KEY
    query_provider = VoyageEmbeddingProvider(input_type="query")
    key = query_provider._resolve_api_key("query")
    assert key == "query_live_key"

    # Verify document provider raises error when VOYAGE_ADMIN_API_KEY is missing
    doc_provider = VoyageEmbeddingProvider(input_type="document")
    with pytest.raises(RuntimeError, match="Voyage Admin API key is not configured"):
        doc_provider._resolve_api_key("document")


@pytest.mark.asyncio
async def test_no_local_embedding_or_ocr_fallback(monkeypatch):
    # Embedding fallback prevention
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "")
    query_provider = VoyageEmbeddingProvider(input_type="query")
    with pytest.raises(RuntimeError, match="Voyage API key is not configured"):
        await query_provider.embed_queries(["test query"])

    # OCR fallback prevention
    monkeypatch.setattr(get_settings(), "GEMINI_API_KEY", "")
    ocr_provider = GeminiOCRProvider(api_key="")
    with pytest.raises(OCRProviderError) as exc_info:
        await ocr_provider.extract_image_text(b"image_bytes")
    assert exc_info.value.code == "MULTIPLE_ASK_OCR_UNAVAILABLE"
    assert exc_info.value.retryable is False

