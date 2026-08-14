"""Small, provider-neutral boundary for bounded printed-English OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


class OCRProviderError(RuntimeError):
    """Sanitized OCR failure; source bytes and text must never enter the error."""

    def __init__(self, code: str, *, retryable: bool):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True)
class OCRPage:
    page_number: int | None
    text: str
    source_kind: str


@dataclass(frozen=True)
class OCRExtractedQuestion:
    """Untrusted structured content returned for one scanned page."""

    display_label: str
    section_context: str | None
    question_text: str
    answer_mode: Literal["short", "long", "mcq", "not_clear"]
    mcq_options: tuple[dict[str, str], ...]
    unclear_reason: str | None = None


class OCRProvider(Protocol):
    async def extract_image_text(self, image_bytes: bytes) -> str: ...

    async def extract_image_questions(
        self, image_bytes: bytes, mime_type: str | None = None
    ) -> tuple[OCRExtractedQuestion, ...]: ...
