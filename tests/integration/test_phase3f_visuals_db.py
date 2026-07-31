"""Non-destructive Phase 3F database regression tests.

Each test wraps its own inserts in a transaction and rolls them back; migrations
are applied separately by the session fixture.
"""

import os
from uuid import uuid4

import asyncpg
import pytest

from app.providers.embeddings.bge import BGEEmbeddingConfiguration
from app.repositories.rag_repository import RagRepository
from app.services.local_admin import LocalAdminError, LocalAdminService


@pytest.mark.asyncio
async def test_visual_insert_gets_updated_at_and_active_clone_copies_visual():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    transaction = conn.transaction()
    await transaction.start()
    try:
        suffix = uuid4().hex
        scope = {
            "board_id": f"p3f-board-{suffix}",
            "class_id": "class",
            "subject_id": "subject",
        }
        configuration = BGEEmbeddingConfiguration()
        repo = RagRepository(conn)
        corpus = await repo.get_or_create_corpus(**scope)
        version = await repo.create_corpus_version(
            str(corpus["id"]),
            1,
            configuration.model,
            configuration.revision,
            configuration.dimensions,
            embedding_config_fingerprint=configuration.fingerprint(),
            normalize_embeddings=configuration.normalize,
            query_instruction=configuration.query_instruction,
        )
        document = await repo.create_document_version(
            str(version["id"]), f"resource-{suffix}", "v1", "test", "Visual test"
        )
        inserted = await repo.replace_chapter_chunks(
            str(version["id"]),
            str(document["id"]),
            [
                {
                    "chapter_id": "chapter",
                    "topic_no": "1",
                    "topic_title": "Forces",
                    "chunk_order": 0,
                    "chunk_text": "A force changes motion.",
                    "content_type": "explanation",
                    "content_hash": "hash",
                    "language": "en",
                    "token_count": 1,
                    "metadata": {},
                    "expected_questions": [],
                    "visuals": [
                        {
                            "visual_id": "force-diagram",
                            "visual_type": "diagram",
                            "title": "Force arrows",
                            "description": "A block and arrows",
                            "storage_key": "test-drive-key",
                        }
                    ],
                }
            ],
        )
        visual = await conn.fetchrow(
            "SELECT updated_at, review_status, display_policy FROM rag_visuals WHERE chunk_id=$1",
            inserted[0]["id"],
        )
        assert visual["updated_at"] is not None
        assert visual["review_status"] == "pending"
        assert visual["display_policy"] == "llm_decide"
        await repo.write_chunk_embedding(
            str(inserted[0]["id"]),
            str(version["id"]),
            [0.0] * 768,
            "input",
            configuration.model,
            configuration.revision,
            configuration.fingerprint(),
        )
        await repo.refresh_embedding_counts(str(version["id"]))
        await conn.execute(
            "UPDATE rag_corpus_versions SET status='qa_ready' WHERE id=$1",
            version["id"],
        )
        await conn.execute(
            "UPDATE rag_corpus_versions SET status='active' WHERE id=$1", version["id"]
        )
        draft = await LocalAdminService(conn).create_draft(
            active_version_id=str(version["id"]),
            scope=scope,
            actor_id="test-admin",
            request_id=f"test-{suffix}",
        )
        clone = await conn.fetchrow(
            """SELECT v.updated_at, v.storage_key FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id
            WHERE c.corpus_version_id=$1::uuid""",
            draft["id"],
        )
        assert clone["updated_at"] is not None
        assert clone["storage_key"] == "test-drive-key"
        service = LocalAdminService(conn)
        with pytest.raises(LocalAdminError, match="ACTIVE_VERSION_IMMUTABLE"):
            await service.edit_question(
                version_id=str(version["id"]),
                scope=scope,
                actor_id="test-admin",
                request_id="active-edit",
                question_id=None,
                question_text="No direct edits",
                chunk_id=str(inserted[0]["id"]),
            )
        draft_visual = await conn.fetchrow(
            """SELECT v.id, c.id AS chunk_id FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id
            WHERE c.corpus_version_id=$1::uuid""",
            draft["id"],
        )
        await conn.execute(
            "INSERT INTO rag_corpus_qa_approvals (corpus_version_id,reviewer_id,request_id,summary) VALUES ($1,'admin','test','{}')",
            draft["id"],
        )
        await service.edit_visual(
            version_id=draft["id"],
            scope=scope,
            actor_id="test-admin",
            request_id="policy",
            visual_id=str(draft_visual["id"]),
            title=None,
            description=None,
            review_status=None,
            display_policy="always",
        )
        chunk = await conn.fetchrow(
            "SELECT embedding_status, embedding IS NOT NULL AS has_embedding FROM rag_chunks WHERE id=$1",
            draft_visual["chunk_id"],
        )
        assert chunk["embedding_status"] == "embedded" and chunk["has_embedding"]
        assert await conn.fetchval(
            "SELECT NOT EXISTS(SELECT 1 FROM rag_corpus_qa_approvals WHERE corpus_version_id=$1 AND invalidated_at IS NULL)",
            draft["id"],
        )
    finally:
        await transaction.rollback()
        await conn.close()
