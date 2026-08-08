"""Unit tests for Google Gemini Vision OCR provider with async transport and error handling."""

from __future__ import annotations

import httpx
import pytest

from app.providers.ocr.base import OCRProviderError
from app.providers.ocr.gemini import GeminiOCRProvider


@pytest.mark.asyncio
async def test_gemini_ocr_extracts_text_successfully():
    def mock_post(request: httpx.Request):
        url_str = str(request.url)
        assert "gemini-3.6-flash" in url_str
        assert "key=test_gemini_key" in url_str
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Q1: Explain Newton's Third Law.\nAction equals reaction."}
                            ]
                        }
                    }
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GeminiOCRProvider(
            api_key="test_gemini_key", model="gemini-3.6-flash", client=client
        )
        text = await provider.extract_image_text(b"\x89PNG\r\n\x1a\nfake_image_bytes")
        assert "Newton's Third Law" in text
        assert "Action equals reaction." in text


@pytest.mark.asyncio
async def test_gemini_ocr_handles_missing_key():
    provider = GeminiOCRProvider(api_key="")
    with pytest.raises(OCRProviderError) as exc:
        await provider.extract_image_text(b"some_bytes")
    assert exc.value.code == "MULTIPLE_ASK_OCR_UNAVAILABLE"
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_gemini_ocr_maps_401_403_to_non_retryable_unavailable():
    def mock_post(request: httpx.Request):
        return httpx.Response(401, text="Unauthorized", request=request)

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GeminiOCRProvider(api_key="bad_key", client=client)
        with pytest.raises(OCRProviderError) as exc:
            await provider.extract_image_text(b"some_bytes")
        assert exc.value.code == "MULTIPLE_ASK_OCR_UNAVAILABLE"
        assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_gemini_ocr_maps_timeout_to_retryable():
    def mock_post(request: httpx.Request):
        raise httpx.ReadTimeout("Request timed out", request=request)

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GeminiOCRProvider(api_key="test_key", client=client)
        with pytest.raises(OCRProviderError) as exc:
            await provider.extract_image_text(b"some_bytes")
        assert exc.value.code == "MULTIPLE_ASK_OCR_TIMEOUT"
        assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_gemini_ocr_maps_500_to_retryable_failed():
    def mock_post(request: httpx.Request):
        return httpx.Response(500, text="Internal Server Error", request=request)

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GeminiOCRProvider(api_key="test_key", client=client)
        with pytest.raises(OCRProviderError) as exc:
            await provider.extract_image_text(b"some_bytes")
        assert exc.value.code == "MULTIPLE_ASK_OCR_FAILED"
        assert exc.value.retryable is True
