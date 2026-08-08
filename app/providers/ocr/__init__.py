"""OCR providers for student-uploaded sources."""

from app.providers.ocr.base import OCRPage, OCRProvider, OCRProviderError
from app.providers.ocr.gemini import GeminiOCRProvider

__all__ = ["OCRProvider", "OCRProviderError", "OCRPage", "GeminiOCRProvider"]
