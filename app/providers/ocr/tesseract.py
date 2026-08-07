"""Tesseract implementation for local Railway OCR only."""

from __future__ import annotations

import asyncio
from io import BytesIO

import pytesseract
from PIL import Image

from app.providers.ocr.base import OCRProviderError


class TesseractOCRProvider:
    """Runs the installed English Tesseract binary with bounded input only."""

    def __init__(self, *, timeout_seconds: int = 20):
        self._timeout_seconds = timeout_seconds

    async def extract_image_text(self, image_bytes: bytes) -> str:
        return await asyncio.to_thread(self._extract_sync, image_bytes)

    def _extract_sync(self, image_bytes: bytes) -> str:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return str(
                    pytesseract.image_to_string(
                        image,
                        lang="eng",
                        config="--oem 1 --psm 6",
                        timeout=self._timeout_seconds,
                    )
                )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRProviderError(
                "MULTIPLE_ASK_OCR_UNAVAILABLE", retryable=False
            ) from exc
        except RuntimeError as exc:
            # pytesseract reports a timed-out subprocess through RuntimeError.
            raise OCRProviderError("MULTIPLE_ASK_OCR_TIMEOUT", retryable=True) from exc
        except (OSError, ValueError) as exc:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True) from exc
