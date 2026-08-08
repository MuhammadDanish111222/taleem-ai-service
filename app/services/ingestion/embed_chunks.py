"""Local-worker chunk embedding service with retry-safe per-row writes."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Protocol

import asyncpg

from app.providers.embeddings.voyage import (
    VoyageEmbeddingProvider,
    embedding_input_hash,
    format_chunk_embedding_input,
)
from app.repositories.rag_repository import RagRepository


class DocumentEmbeddingProvider(Protocol):
    configuration_fingerprint: str
    configuration: Any

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def _requires_embedding(row: Dict[str, Any], input_hash: str, provider: Any) -> bool:
    return not (
        row["embedding"] is not None
        and row["embedding_status"] == "embedded"
        and row["embedding_input_hash"] == input_hash
        and row["embedding_config_fingerprint"] == provider.configuration_fingerprint
        and row["embedding_model"] == provider.configuration.model
        and row["embedding_revision"] == provider.configuration.revision
    )


async def embed_chunks(
    corpus_version_id: str,
    configuration_fingerprint: str,
    conn: asyncpg.Connection,
    provider: DocumentEmbeddingProvider | None = None,
    batch_size: int = 64,
) -> Dict[str, int]:
    """Embeds only missing/stale chunk rows; existing valid vectors are untouched."""
    provider = provider or VoyageEmbeddingProvider(input_type="document", batch_size=batch_size)
    if provider.configuration_fingerprint != configuration_fingerprint:
        raise ValueError("EMBEDDING_CONFIGURATION_MISMATCH_REQUIRES_NEW_CORPUS_VERSION")
    repo = RagRepository(conn)
    version = await repo.get_corpus_version(corpus_version_id)
    if not version or version["status"] != "building":
        raise ValueError("CORPUS_VERSION_NOT_BUILDING")
    if version["embedding_config_fingerprint"] != configuration_fingerprint:
        raise ValueError("EMBEDDING_CONFIGURATION_MISMATCH_REQUIRES_NEW_CORPUS_VERSION")

    candidates = []
    for row in await repo.fetch_chunks_for_embedding(
        corpus_version_id, configuration_fingerprint
    ):
        if row["language"] != "en":
            await repo.mark_embedding_failed(
                "rag_chunks", str(row["id"]), "ENGLISH_ONLY_EMBEDDING_UNSUPPORTED"
            )
            raise ValueError("ENGLISH_ONLY_EMBEDDING_UNSUPPORTED")
        approved_visuals = row.get("approved_visuals") or []
        if isinstance(approved_visuals, str):
            approved_visuals = json.loads(approved_visuals)
        text = format_chunk_embedding_input(
            topic_no=row["topic_no"],
            topic_title=row["topic_title"],
            chunk_text=row["content"],
            approved_visuals=approved_visuals,
        )
        input_hash = embedding_input_hash(text)
        if _requires_embedding(row, input_hash, provider):
            candidates.append((row, text, input_hash))

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        try:
            if asyncio.iscoroutinefunction(provider.embed_documents):
                vectors = await provider.embed_documents([item[1] for item in batch])
            else:
                vectors = await asyncio.to_thread(
                    provider.embed_documents, [item[1] for item in batch]
                )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned an unexpected chunk vector count."
                )
            for (row, _, input_hash), vector in zip(batch, vectors, strict=True):
                await repo.write_chunk_embedding(
                    str(row["id"]),
                    corpus_version_id,
                    vector,
                    input_hash,
                    provider.configuration.model,
                    provider.configuration.revision,
                    provider.configuration_fingerprint,
                )
        except Exception:
            for row, _, _ in batch:
                await repo.mark_embedding_failed(
                    "rag_chunks", str(row["id"]), "CHUNK_EMBEDDING_FAILED"
                )
            raise
    counts = await repo.refresh_embedding_counts(corpus_version_id)
    return {"embedded": len(candidates), "expected": counts["expected_chunk_count"]}


async def handle_embed_chunks(
    job: Dict[str, Any], conn: asyncpg.Connection
) -> Dict[str, Any]:
    payload = job.get("payload") or {}
    corpus_version_id = str(payload.get("corpus_version_id") or "")
    configuration_fingerprint = str(payload.get("embedding_config_fingerprint") or "")
    input_fingerprint = str(payload.get("embedding_input_fingerprint") or "")
    repo = RagRepository(conn)
    version = await repo.get_corpus_version(corpus_version_id)
    if not version or (
        version["embedding_config_fingerprint"] != configuration_fingerprint
        or version["embedding_input_fingerprint"] != input_fingerprint
    ):
        return {"status": "stale_generation"}

    result = await embed_chunks(corpus_version_id, configuration_fingerprint, conn)
    current = await repo.get_corpus_version(corpus_version_id)
    if not current or current["embedding_input_fingerprint"] != input_fingerprint:
        return {"status": "stale_generation"}
    result["_next_job"] = {
        "job_type": "embed_questions",
        "payload": payload,
        "idempotency_key": (
            f"phase3d:embed_questions:{corpus_version_id}:{input_fingerprint}"
        ),
    }
    return result
