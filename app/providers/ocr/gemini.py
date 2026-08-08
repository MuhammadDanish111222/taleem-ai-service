"""Google Gemini Vision OCR provider with async HTTP transport."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from app.core.config import get_settings
from app.providers.ocr.base import OCRProviderError

DEFAULT_GEMINI_OCR_MODEL = "gemini-3.6-flash"


def _detect_mime_type(image_bytes: bytes, explicit_mime: str | None = None) -> str:
    if explicit_mime and explicit_mime.strip():
        return explicit_mime.strip()
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class GeminiOCRProvider:
    """Non-blocking Gemini Vision OCR provider extracting plain text from page images."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self._model = model or settings.GEMINI_OCR_MODEL or DEFAULT_GEMINI_OCR_MODEL
        self._timeout_seconds = timeout_seconds
        self._client = client

    def _resolve_api_key(self) -> str:
        key = self._api_key.strip()
        if not key:
            raise OCRProviderError("MULTIPLE_ASK_OCR_UNAVAILABLE", retryable=False)
        return key

    async def extract_image_text(
        self, image_bytes: bytes, mime_type: str | None = None
    ) -> str:
        """Sends an image to Gemini Vision generateContent and extracts English text."""
        if not image_bytes:
            return ""
        api_key = self._resolve_api_key()
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        resolved_mime = _detect_mime_type(image_bytes, mime_type)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={api_key}"
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Extract and transcribe all printed text from this image exactly as written. "
                                "Preserve math formulas, tables, and question structure accurately. "
                                "Return plain text only, without conversational commentary."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": resolved_mime,
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ]
        }

        try:
            if self._client is not None:
                response = await self._client.post(
                    url,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(
                        url,
                        json=payload,
                    )
        except httpx.TimeoutException:
            raise OCRProviderError("MULTIPLE_ASK_OCR_TIMEOUT", retryable=True)
        except Exception:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

        if response.status_code in (401, 403):
            raise OCRProviderError("MULTIPLE_ASK_OCR_UNAVAILABLE", retryable=False)
        if response.status_code == 429:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)
        if response.status_code >= 500:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)
        if response.status_code != 200:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

        try:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            extracted_parts = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and "text" in part
            ]
            return "\n".join(extracted_parts).strip()
        except Exception:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)
