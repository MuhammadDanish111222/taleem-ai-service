"""Google Gemini Vision OCR provider with async HTTP transport."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import Any

import httpx

from app.core.config import get_settings
from app.providers.ocr.base import OCRExtractedQuestion, OCRProviderError

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_OCR_MODEL = "gemini-3.6-flash"


def _detect_mime_type(image_bytes: bytes, explicit_mime: str | None = None) -> str:
    if explicit_mime and explicit_mime.strip():
        return explicit_mime.strip()
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if (
        image_bytes.startswith(b"RIFF")
        and len(image_bytes) >= 12
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return "image/png"


class GeminiOCRProvider:
    """Non-blocking Gemini Vision extraction provider for scanned paper pages."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 90.0,
        client: httpx.AsyncClient | None = None,
    ):
        settings = get_settings()
        self._api_key = (
            api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        )
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
            return self._candidate_text(await self._post(url, payload))
        except OCRProviderError:
            raise
        except Exception:
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

    async def extract_image_questions(
        self, image_bytes: bytes, mime_type: str | None = None
    ) -> tuple[OCRExtractedQuestion, ...]:
        """Extract only paper structure; answers and educational guesses are forbidden."""
        if not image_bytes:
            return ()
        api_key = self._resolve_api_key()
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        resolved_mime = _detect_mime_type(image_bytes, mime_type)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={api_key}"
        schema = {
            "type": "OBJECT",
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "required": [
                            "display_label",
                            "question_text",
                            "answer_mode",
                            "mcq_options",
                        ],
                        "properties": {
                            "display_label": {"type": "STRING"},
                            "section_context": {
                                "type": "STRING",
                                "nullable": True,
                            },
                            "question_text": {"type": "STRING"},
                            "answer_mode": {
                                "type": "STRING",
                                "enum": ["mcq", "short", "long", "not_clear"],
                            },
                            "mcq_options": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "required": ["label", "text"],
                                    "properties": {
                                        "label": {"type": "STRING"},
                                        "text": {"type": "STRING"},
                                    },
                                },
                            },
                            "unclear_reason": {
                                "type": "STRING",
                                "nullable": True,
                            },
                        },
                    },
                },
            },
        }
        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Extract this scanned examination-paper page into the supplied JSON schema. "
                                "Return questions only: never solve, answer, explain, use outside knowledge, "
                                "or infer missing text/options. Preserve printed order, labels, wording, section "
                                "headings, and option text. Classification priority is exact: a recognized Multiple "
                                "Choice/MCQ/Objective/Choose-correct section means mcq; a recognized Short Questions/"
                                "Short Answer section means short; a recognized Long Questions/Long Answer/Detailed "
                                "Questions section means long. A recognized section wins over verbs and options. "
                                "Without a recognized type section, valid contiguous A-starting options mean mcq and "
                                "no options mean short. MCQ-section questions may have zero options. Never invent "
                                "options; use not_clear with an unclear_reason when text or printed options are unreadable."
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
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        data = await self._post(url, payload)
        try:
            text = self._candidate_text(data)
            cleaned_text = text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(
                    r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE
                )
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()
            decoded = json.loads(cleaned_text)
            questions = decoded.get("questions") if isinstance(decoded, dict) else None
            if not isinstance(questions, list):
                raise ValueError(
                    "JSON response does not contain a valid 'questions' array"
                )
            output: list[OCRExtractedQuestion] = []
            for idx, question in enumerate(questions):
                if not isinstance(question, dict):
                    continue
                display_label = str(question.get("display_label") or (idx + 1)).strip()
                section_context = question.get("section_context")
                if section_context is not None:
                    section_context = str(section_context).strip() or None
                question_text = str(question.get("question_text") or "").strip()
                answer_mode = (
                    str(question.get("answer_mode") or "short").strip().lower()
                )
                if answer_mode not in {"mcq", "short", "long", "not_clear"}:
                    answer_mode = "short"

                raw_options = question.get("mcq_options")
                parsed_options: list[dict[str, str]] = []
                if isinstance(raw_options, list):
                    for opt_idx, opt in enumerate(raw_options):
                        if isinstance(opt, dict):
                            label = (
                                str(opt.get("label") or chr(ord("A") + opt_idx))
                                .strip()
                                .upper()
                            )
                            text_val = str(opt.get("text") or "").strip()
                            if text_val:
                                parsed_options.append(
                                    {"label": label, "text": text_val}
                                )

                unclear_reason = question.get("unclear_reason")
                if unclear_reason is not None:
                    unclear_reason = str(unclear_reason).strip() or None

                if not question_text and answer_mode != "not_clear":
                    answer_mode = "not_clear"
                    unclear_reason = unclear_reason or "QUESTION_TEXT_UNCLEAR"

                output.append(
                    OCRExtractedQuestion(
                        display_label=display_label,
                        section_context=section_context,
                        question_text=question_text,
                        answer_mode=answer_mode,
                        mcq_options=tuple(parsed_options),
                        unclear_reason=unclear_reason,
                    )
                )
            if not output:
                raise ValueError("No questions extracted from response")
            return tuple(output)
        except Exception as exc:
            logger.error("Gemini OCR structured extraction parse error: %s", exc)
            raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if self._client is not None:
                    response = await self._client.post(
                        url, json=payload, timeout=self._timeout_seconds
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self._timeout_seconds
                    ) as client:
                        response = await client.post(url, json=payload)
            except httpx.TimeoutException:
                logger.error(
                    "Gemini OCR request timed out (timeout: %ss)", self._timeout_seconds
                )
                raise OCRProviderError("MULTIPLE_ASK_OCR_TIMEOUT", retryable=True)
            except Exception as exc:
                logger.error("Gemini OCR transport exception: %s", exc)
                raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

            if response.status_code in (401, 403):
                logger.error(
                    "Gemini OCR auth error (%s): %s",
                    response.status_code,
                    response.text,
                )
                raise OCRProviderError("MULTIPLE_ASK_OCR_UNAVAILABLE", retryable=False)

            if response.status_code in (503, 429) and attempt < max_attempts - 1:
                logger.warning(
                    "Gemini OCR temporary API status %s on attempt %s/%s. Retrying in %ss...",
                    response.status_code,
                    attempt + 1,
                    max_attempts,
                    (attempt + 1) * 2,
                )
                await asyncio.sleep((attempt + 1) * 2)
                continue

            if response.status_code != 200:
                logger.error(
                    "Gemini OCR API error (%s): %s", response.status_code, response.text
                )
                raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

            try:
                return response.json()
            except Exception as exc:
                logger.error(
                    "Gemini OCR JSON decode error: %s (raw response: %s)",
                    exc,
                    response.text,
                )
                raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

        raise OCRProviderError("MULTIPLE_ASK_OCR_FAILED", retryable=True)

    @staticmethod
    def _candidate_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError
        return "\n".join(
            part.get("text", "")
            for part in candidates[0].get("content", {}).get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
