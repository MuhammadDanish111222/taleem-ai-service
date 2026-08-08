"""Local-admin-only embedding job for approved questions and variations."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import asyncpg

from app.providers.embeddings.voyage import VoyageEmbeddingProvider


@lru_cache(maxsize=1)
def _provider() -> VoyageEmbeddingProvider:
    return VoyageEmbeddingProvider(input_type="document")


async def handle_question_bank_embeddings(
    job: dict[str, Any], conn: asyncpg.Connection
) -> dict[str, Any]:
    payload = job.get("payload") or {}
    revision_id = payload.get("revision_id")
    variation_id = payload.get("variation_id")
    rows = await conn.fetch(
        """SELECT 'revision' AS kind,id::text,text_value FROM (
             SELECT id,question_text AS text_value
             FROM question_bank_revisions
             WHERE review_status='approved' AND superseded_at IS NULL
               AND embedding_status='pending'
               AND ($1::uuid IS NULL OR id=$1::uuid)
           ) r
           UNION ALL
           SELECT 'variation' AS kind,id::text,text_value FROM (
             SELECT v.id,v.variation_text AS text_value
             FROM question_bank_variations v
             JOIN question_bank_revisions r ON r.id=v.revision_id
             WHERE v.active AND v.embedding_status='pending'
               AND r.review_status='approved' AND r.superseded_at IS NULL
               AND ($2::uuid IS NULL OR v.id=$2::uuid)
               AND ($1::uuid IS NULL OR r.id=$1::uuid)
           ) v
           ORDER BY kind,id
           LIMIT 256""",
        revision_id,
        variation_id,
    )
    if not rows:
        return {"embedded": 0}
    provider = _provider()
    if asyncio.iscoroutinefunction(provider.embed_documents):
        vectors = await provider.embed_documents([row["text_value"] for row in rows])
    else:
        vectors = await asyncio.to_thread(
            provider.embed_documents, [row["text_value"] for row in rows]
        )
    async with conn.transaction():
        for row, vector in zip(rows, vectors, strict=True):
            table = (
                "question_bank_revisions"
                if row["kind"] == "revision"
                else "question_bank_variations"
            )
            await conn.execute(
                f"""UPDATE {table}
                    SET embedding=$2::text::halfvec,embedding_model=$3,
                        embedding_revision=$4,embedding_config_fingerprint=$5,
                        embedding_status='embedded'
                    WHERE id=$1::uuid""",
                row["id"],
                str(vector),
                provider.configuration.model,
                provider.configuration.revision,
                provider.configuration_fingerprint,
            )
    return {"embedded": len(rows)}
