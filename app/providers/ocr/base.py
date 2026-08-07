"""Small, provider-neutral boundary for bounded printed-English OCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


class OCRProvider(Protocol):
    async def extract_image_text(self, image_bytes: bytes) -> str: ...
