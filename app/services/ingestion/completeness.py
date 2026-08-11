"""Corpus completeness gate used by the local-admin worker."""

from __future__ import annotations

from typing import Any, Dict

import asyncpg

from app.repositories.rag_repository import RagRepository


class CorpusIncompleteError(RuntimeError):
    code = "CORPUS_INCOMPLETE"


async def confirm_corpus_completeness(
    corpus_version_id: str,
    conn: asyncpg.Connection,
    auto_promote: bool = False,
    target_chapter_id: str | None = None,
) -> Dict[str, Any]:
    repo = RagRepository(conn)
    await repo.refresh_embedding_counts(corpus_version_id)
    result = await repo.mark_qa_ready(corpus_version_id)
    if not result["ready"]:
        raise CorpusIncompleteError(
            ",".join(result["reasons"]) or "QA_READY_TRANSITION_REJECTED"
        )

    if auto_promote and target_chapter_id:
        c_info = await conn.fetchrow(
            """SELECT c.board_id, c.class_id, c.subject_id, cv.status
               FROM rag_corpus_versions cv
               JOIN rag_corpora c ON c.id = cv.corpus_id
               WHERE cv.id = $1::uuid;""",
            corpus_version_id,
        )
        if c_info:
            active_ver = await repo.get_active_corpus_version(
                c_info["board_id"], c_info["class_id"], c_info["subject_id"]
            )
            if active_ver and str(active_ver["id"]) != corpus_version_id:
                promotion = await repo.promote_chapter_from_temp_to_active(
                    temp_version_id=corpus_version_id,
                    active_version_id=str(active_ver["id"]),
                    board_id=c_info["board_id"],
                    class_id=c_info["class_id"],
                    subject_id=c_info["subject_id"],
                    chapter_id=target_chapter_id,
                )
                from app.services.retrieval.active_version_cache import (
                    get_active_corpus_version_cache,
                )

                await get_active_corpus_version_cache().invalidate(
                    c_info["board_id"], c_info["class_id"], c_info["subject_id"]
                )
                return {**result, "promoted": True, "promotion": promotion}

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
    auto_promote = bool(payload.get("auto_promote"))
    chapter_id = str(payload.get("chapter_id")) if payload.get("chapter_id") else None
    return await confirm_corpus_completeness(
        corpus_version_id, conn, auto_promote=auto_promote, target_chapter_id=chapter_id
    )
