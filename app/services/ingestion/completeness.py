"""Corpus completeness gate used by the local-admin worker."""

from __future__ import annotations

from typing import Any, Dict

import asyncpg

from app.repositories.rag_repository import RagRepository


class CorpusIncompleteError(RuntimeError):
    code = "CORPUS_INCOMPLETE"


async def confirm_corpus_completeness(
    corpus_version_id: str, conn: asyncpg.Connection
) -> Dict[str, Any]:
    repo = RagRepository(conn)
    await repo.refresh_embedding_counts(corpus_version_id)
    result = await repo.mark_qa_ready(corpus_version_id)
    if not result["ready"]:
        raise CorpusIncompleteError(
            ",".join(result["reasons"]) or "QA_READY_TRANSITION_REJECTED"
        )
    return result


async def handle_corpus_completeness(
    job: Dict[str, Any], conn: asyncpg.Connection
) -> Dict[str, Any]:
    payload = job.get("payload") or {}
    corpus_version_id = str(payload.get("corpus_version_id") or "")
    repo = RagRepository(conn)
    version = await repo.get_corpus_version(corpus_version_id)
    if not version or (
        version["embedding_config_fingerprint"]
        != str(payload.get("embedding_config_fingerprint") or "")
        or version["embedding_input_fingerprint"]
        != str(payload.get("embedding_input_fingerprint") or "")
    ):
        return {"status": "stale_generation"}
    return await confirm_corpus_completeness(corpus_version_id, conn)
