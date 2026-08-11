"""Voyage AI embedding provider with async HTTP transport and separate Admin/Railway credentials."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import httpx

from app.core.config import get_settings

MODEL_NAME = "voyage-4-lite"
MODEL_REVISION = "voyage-4-lite-512-v1"
EMBEDDING_DIMENSIONS = 512
OUTPUT_DTYPE = "float"
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
DOCUMENT_INPUT_FORMAT = "topic-heading-approved-visuals-v2"
QUERY_INPUT_FORMAT = "voyage-query-v1"
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class VoyageEmbeddingConfiguration:
    model: str = MODEL_NAME
    revision: str = MODEL_REVISION
    dimensions: int = EMBEDDING_DIMENSIONS
    output_dtype: str = OUTPUT_DTYPE
    normalize: bool = True
    document_input_format: str = DOCUMENT_INPUT_FORMAT
    query_input_format: str = QUERY_INPUT_FORMAT

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def format_chunk_embedding_input(
    *,
    topic_no: object,
    topic_title: object,
    chunk_text: object,
    approved_visuals: Sequence[dict[str, object]] = (),
    approved_visual: dict[str, object] | None = None,
) -> str:
    """Builds the deterministic chunk embedding text."""
    lines = []
    if number := _clean_text(topic_no):
        lines.append(f"Topic {number}")
    if title := _clean_text(topic_title):
        lines.append(f"Title: {title}")
    text = _clean_text(chunk_text)
    if not text:
        raise ValueError("Chunk embedding text must not be blank.")
    lines.append(f"Content: {text}")
    visuals = sorted(
        approved_visuals,
        key=lambda item: _clean_text(item.get("visual_id")),
    )
    if (
        approved_visual
        and approved_visual.get("is_linked")
        and approved_visual.get("is_approved")
    ):
        visuals.append(approved_visual)
    for visual in visuals:
        if visual_title := _clean_text(visual.get("title")):
            lines.append(f"Visual: {visual_title}")
        if visual_description := _clean_text(visual.get("description")):
            lines.append(f"Visual description: {visual_description}")
    return "\n".join(lines)


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VoyageEmbeddingProvider:
    """Non-blocking Voyage-4-lite provider for Admin corpus preparation and Railway query retrieval."""

    def __init__(
        self,
        configuration: VoyageEmbeddingConfiguration | None = None,
        *,
        api_key: str | None = None,
        input_type: str = "document",
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.configuration = configuration or VoyageEmbeddingConfiguration()
        if (
            self.configuration.model != MODEL_NAME
            or self.configuration.dimensions != EMBEDDING_DIMENSIONS
            or self.configuration.output_dtype != OUTPUT_DTYPE
        ):
            raise ValueError(
                f"Voyage embedding provider requires {MODEL_NAME} with {EMBEDDING_DIMENSIONS} dimensions."
            )
        self.input_type = input_type
        self.batch_size = max(1, batch_size)
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key
        self._client = client

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def _resolve_api_key(self, input_type: str) -> str:
        if self._api_key:
            return self._api_key
        settings = get_settings()
        if input_type == "document":
            key = (
                settings.VOYAGE_ADMIN_API_KEY
                or os.getenv("VOYAGE_ADMIN_API_KEY", "")
                or settings.VOYAGE_API_KEY
                or os.getenv("VOYAGE_API_KEY", "")
            ).strip()
            if not key:
                raise RuntimeError(
                    "Voyage Admin API key is not configured for document embedding."
                )
            return key
        if input_type == "query":
            key = (
                settings.VOYAGE_API_KEY
                or os.getenv("VOYAGE_API_KEY", "")
                or settings.VOYAGE_ADMIN_API_KEY
                or os.getenv("VOYAGE_ADMIN_API_KEY", "")
            ).strip()
            if not key:
                raise RuntimeError(
                    "Voyage API key is not configured for query embedding."
                )
            return key
        raise ValueError(f"Unsupported Voyage input_type '{input_type}'.")

    async def _call_voyage_api_async(
        self, texts: Sequence[str], input_type: str
    ) -> list[list[float]]:
        if not texts:
            return []
        api_key = self._resolve_api_key(input_type)

        all_embeddings: list[list[float]] = []
        # Batch multiple inputs into slices of self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[i : i + self.batch_size])
            payload: dict[str, Any] = {
                "input": batch_texts,
                "model": self.configuration.model,
                "input_type": input_type,
                "output_dimension": self.configuration.dimensions,
                "output_dtype": self.configuration.output_dtype,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if self._client is not None:
                response = await self._client.post(
                    VOYAGE_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        VOYAGE_API_URL,
                        json=payload,
                        headers=headers,
                    )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Voyage API request failed with status {response.status_code}: {response.text}"
                )
            data = response.json()
            items = data.get("data", [])
            sorted_items = sorted(items, key=lambda x: x.get("index", 0))
            for item in sorted_items:
                vec = item.get("embedding", [])
                if len(vec) != self.configuration.dimensions:
                    raise ValueError(
                        f"Voyage returned vector with dimension {len(vec)}, expected {self.configuration.dimensions}"
                    )
                all_embeddings.append([float(v) for v in vec])

        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Voyage returned {len(all_embeddings)} vectors for {len(texts)} texts."
            )
        return all_embeddings

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds document texts using input_type='document' and VOYAGE_ADMIN_API_KEY."""
        clean_texts = [_clean_text(t) for t in texts]
        return await self._call_voyage_api_async(clean_texts, input_type="document")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds query texts using input_type='query' and VOYAGE_API_KEY."""
        clean_texts = [_clean_text(t) for t in texts]
        return await self._call_voyage_api_async(clean_texts, input_type="query")
