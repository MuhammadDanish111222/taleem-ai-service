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
                                {
                                    "text": "Q1: Explain Newton's Third Law.\nAction equals reaction."
                                }
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


@pytest.mark.asyncio
async def test_gemini_ocr_detects_jpeg_mime_type():
    import json

    captured_mime = ""

    def mock_post(request: httpx.Request):
        nonlocal captured_mime
        body = json.loads(request.content)
        captured_mime = body["contents"][0]["parts"][1]["inline_data"]["mime_type"]
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "JPEG OCR text"}]}}]},
            request=request,
        )

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = GeminiOCRProvider(api_key="test_key", client=client)
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        text = await provider.extract_image_text(jpeg_bytes)
        assert text == "JPEG OCR text"
        assert captured_mime == "image/jpeg"


@pytest.mark.asyncio
async def test_gemini_structured_extraction_accepts_questions_only_contract():
    import json

    def mock_post(request: httpx.Request):
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "never solve" in body["contents"][0]["parts"][0]["text"]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "questions": [
                                                {
                                                    "display_label": "1",
                                                    "section_context": "Multiple Choice Questions",
                                                    "question_text": "Which form conducts?",
                                                    "answer_mode": "mcq",
                                                    "mcq_options": [
                                                        {
                                                            "label": "A",
                                                            "text": "Diamond",
                                                        },
                                                        {
                                                            "label": "B",
                                                            "text": "Graphite",
                                                        },
                                                    ],
                                                    "unclear_reason": None,
                                                }
                                            ]
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_post)) as client:
        questions = await GeminiOCRProvider(
            api_key="test_key", client=client
        ).extract_image_questions(b"\x89PNG\r\n\x1a\nimage")
    assert questions[0].answer_mode == "mcq"
    assert questions[0].mcq_options[1] == {"label": "B", "text": "Graphite"}


@pytest.mark.asyncio
async def test_gemini_structured_extraction_rejects_arbitrary_or_invalid_json():
    def mock_post(request: httpx.Request):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"answer":"B"}'}]}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_post)) as client:
        with pytest.raises(OCRProviderError, match="MULTIPLE_ASK_OCR_FAILED"):
            await GeminiOCRProvider(
                api_key="test_key", client=client
            ).extract_image_questions(b"image")
