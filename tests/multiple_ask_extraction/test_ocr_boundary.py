"""Bounded local OCR routing tests; no real Tesseract binary is required."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.providers.ocr.base import OCRExtractedQuestion
from app.services.multiple_ask_extraction_service import (
    MultipleAskExtractionError,
    MultipleAskExtractionService,
    normalize_source_text,
)


class FakeOCR:
    def __init__(self):
        self.calls: list[bytes] = []

    async def extract_image_text(self, image_bytes: bytes) -> str:
        self.calls.append(image_bytes)
        return "1. Define velocity."


class FakeStructuredOCR(FakeOCR):
    async def extract_image_questions(self, image_bytes: bytes):
        self.calls.append(image_bytes)
        return (
            OCRExtractedQuestion(
                display_label="1",
                section_context="Multiple Choice Questions",
                question_text="Which form conducts?",
                answer_mode="mcq",
                mcq_options=(),
                unclear_reason=None,
            ),
        )


def test_source_normalization_preserves_page_boundary_without_spell_correction():
    assert (
        normalize_source_text(
            "  1) Defne  momentum. \r\n\f\r\n 2) momenturn unit?  ",
            max_characters=30_000,
        )
        == "1) Defne momentum.\f\n2) momenturn unit?"
    )


@pytest.mark.asyncio
async def test_scanned_pdf_page_is_rendered_and_ocrd_one_page_at_a_time():
    stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(stream)
    fake = FakeOCR()
    service = MultipleAskExtractionService(object(), storage=object(), ocr=fake)

    normalized, kind, locators, provider, structured = await service._pdf_text(
        stream.getvalue()
    )

    assert normalized == "1. Define velocity."
    assert kind == "pdf_ocr"
    assert locators == [{"page_number": 1, "source_kind": "ocr"}]
    assert provider == "gemini"
    assert structured[0].question_text == "Define velocity."
    assert len(fake.calls) == 1 and fake.calls[0].startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_scanned_page_uses_validated_gemini_structured_questions():
    fake = FakeStructuredOCR()
    service = MultipleAskExtractionService(object(), storage=object(), ocr=fake)
    items = await service._ocr_questions(b"\x89PNG\r\n\x1a\nbytes", page_number=2)
    assert [
        (
            item.display_label,
            item.answer_mode,
            item.mcq_options,
            item.source_locator["page_number"],
        )
        for item in items
    ] == [("1", "mcq", (), 2)]


@pytest.mark.asyncio
async def test_embedded_pdf_never_calls_gemini():
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "1. Define velocity.")
    raw = document.tobytes()
    document.close()
    fake = FakeStructuredOCR()
    service = MultipleAskExtractionService(object(), storage=object(), ocr=fake)
    _, kind, _, provider, _ = await service._pdf_text(raw)
    assert kind == "pdf_embedded_text"
    assert provider is None
    assert fake.calls == []


def test_invalid_structured_gemini_options_are_rejected_before_persistence():
    bad = OCRExtractedQuestion(
        display_label="1",
        section_context=None,
        question_text="Select.",
        answer_mode="mcq",
        mcq_options=({"label": "A", "text": "one"}, {"label": "C", "text": "three"}),
    )
    with pytest.raises(MultipleAskExtractionError, match="MULTIPLE_ASK_OCR_FAILED"):
        MultipleAskExtractionService._validated_structured_item(bad, 0, 1)
