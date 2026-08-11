"""Unit and integration tests for Voyage-4-lite embedding provider and credential isolation."""

from __future__ import annotations

import json

import httpx
import pytest

from app.providers.embeddings.voyage import (
    EMBEDDING_DIMENSIONS,
    MODEL_NAME,
    MODEL_REVISION,
    VoyageEmbeddingConfiguration,
    VoyageEmbeddingProvider,
)


@pytest.mark.asyncio
async def test_voyage_configuration_defaults_and_fingerprint():
    config = VoyageEmbeddingConfiguration()
    assert config.model == MODEL_NAME
    assert config.revision == MODEL_REVISION
    assert config.dimensions == EMBEDDING_DIMENSIONS
    assert config.output_dtype == "float"
    fp1 = config.fingerprint()
    fp2 = config.fingerprint()
    assert fp1 == fp2 and len(fp1) == 64


@pytest.mark.asyncio
async def test_voyage_provider_enforces_admin_key_for_documents(monkeypatch):
    monkeypatch.setenv("VOYAGE_ADMIN_API_KEY", "admin_secret_key")
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VOYAGE_ADMIN_API_KEY", "admin_secret_key")
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "")

    def mock_post(request: httpx.Request):
        assert request.headers.get("Authorization") == "Bearer admin_secret_key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload.get("input_type") == "document"
        assert payload.get("output_dimension") == 512
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1] * 512},
                    {"index": 1, "embedding": [0.2] * 512},
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = VoyageEmbeddingProvider(
            input_type="document", batch_size=64, client=client
        )
        vectors = await provider.embed_documents(["Text A", "Text B"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 512
        assert len(vectors[1]) == 512


@pytest.mark.asyncio
async def test_voyage_provider_enforces_query_key_for_queries(monkeypatch):
    monkeypatch.setenv("VOYAGE_ADMIN_API_KEY", "")
    monkeypatch.setenv("VOYAGE_API_KEY", "query_secret_key")
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VOYAGE_ADMIN_API_KEY", "")
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "query_secret_key")

    def mock_post(request: httpx.Request):
        assert request.headers.get("Authorization") == "Bearer query_secret_key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload.get("input_type") == "query"
        assert payload.get("output_dimension") == 512
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.5] * 512},
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = VoyageEmbeddingProvider(
            input_type="query", batch_size=64, client=client
        )
        vectors = await provider.embed_queries(["What is momentum?"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 512


@pytest.mark.asyncio
async def test_voyage_admin_never_falls_back_to_railway_key(monkeypatch):
    monkeypatch.setenv("VOYAGE_ADMIN_API_KEY", "")
    monkeypatch.setenv("VOYAGE_API_KEY", "query_key_only")
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VOYAGE_ADMIN_API_KEY", "")
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "query_key_only")

    provider = VoyageEmbeddingProvider(input_type="document")
    with pytest.raises(RuntimeError, match="Voyage Admin API key is not configured"):
        await provider.embed_documents(["Some textbook chunk"])


@pytest.mark.asyncio
async def test_voyage_railway_never_falls_back_to_admin_key(monkeypatch):
    monkeypatch.setenv("VOYAGE_ADMIN_API_KEY", "admin_key_only")
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VOYAGE_ADMIN_API_KEY", "admin_key_only")
    monkeypatch.setattr(get_settings(), "VOYAGE_API_KEY", "")

    provider = VoyageEmbeddingProvider(input_type="query")
    with pytest.raises(RuntimeError, match="Voyage API key is not configured"):
        await provider.embed_queries(["Student live question"])


@pytest.mark.asyncio
async def test_voyage_batches_inputs_correctly():
    call_counts = 0

    def mock_post(request: httpx.Request):
        nonlocal call_counts
        call_counts += 1
        payload = json.loads(request.content.decode("utf-8"))
        inputs = payload.get("input", [])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": idx, "embedding": [0.1] * 512}
                    for idx in range(len(inputs))
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(mock_post)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = VoyageEmbeddingProvider(
            api_key="test_key",
            input_type="document",
            batch_size=2,
            client=client,
        )
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        vectors = await provider.embed_documents(texts)
        assert len(vectors) == 5
        # 5 texts in batches of 2 = 3 HTTP calls
        assert call_counts == 3
