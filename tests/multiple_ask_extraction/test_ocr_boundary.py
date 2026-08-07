"""Bounded local OCR routing tests; no real Tesseract binary is required."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.services.multiple_ask_extraction_service import (
    MultipleAskExtractionService,
    normalize_source_text,
)


class FakeOCR:
    def __init__(self):
        self.calls: list[bytes] = []

    async def extract_image_text(self, image_bytes: bytes) -> str:
        self.calls.append(image_bytes)
        return "1. Define velocity."


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

    normalized, kind, locators, provider = await service._pdf_text(stream.getvalue())

    assert normalized == "1. Define velocity."
    assert kind == "pdf_ocr"
    assert locators == [{"page_number": 1, "source_kind": "ocr"}]
    assert provider == "tesseract"
    assert len(fake.calls) == 1 and fake.calls[0].startswith(b"\x89PNG\r\n\x1a\n")
