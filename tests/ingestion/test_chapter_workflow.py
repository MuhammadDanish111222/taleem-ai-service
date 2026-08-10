"""Tests for simplified Local RAG chapter workflow, atomic promotions, storage cleanup, and Q&A retention rules."""

import pytest

from app.repositories.question_bank_repository import QuestionBankRepository
from app.repositories.rag_repository import RagRepository


class MockTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_hard_verification_embedding_config_mismatch():
    """Verify promotion aborts if embedding configurations between temp and active version mismatch."""
    class MockConn:
        def transaction(self):
            return MockTx()

        async def fetchrow(self, query, *args):
            if "FOR UPDATE" in query:
                if args[0] == "temp-1":
                    return {
                        "id": "temp-1",
                        "status": "building",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_temp",
                        "normalize_embeddings": True,
                    }
                if args[0] == "active-1":
                    return {
                        "id": "active-1",
                        "status": "active",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_active_different",
                        "normalize_embeddings": True,
                    }
            return None

        async def fetch(self, query, *args):
            if "DISTINCT chapter_id" in query:
                return [{"chapter_id": "ch01"}]
            if "rag_document_versions" in query:
                return [{"resource_id": "jsonl:chapter:ch01"}]
            return []

    repo = RagRepository(MockConn())  # type: ignore
    with pytest.raises(ValueError, match="EMBEDDING_CONFIGURATION_MISMATCH_CANNOT_PROMOTE"):
        await repo.promote_chapter_from_temp_to_active(
            temp_version_id="temp-1",
            active_version_id="active-1",
            board_id="fbise",
            class_id="class-9",
            subject_id="chemistry",
            chapter_id="ch01",
        )


@pytest.mark.asyncio
async def test_multi_chapter_temp_corpus_rejected():
    """Verify promotion aborts if temp building corpus contains multiple chapters."""
    class MockConn:
        def transaction(self):
            return MockTx()

        async def fetchrow(self, query, *args):
            if "FOR UPDATE" in query:
                if args[0] == "temp-multi":
                    return {
                        "id": "temp-multi",
                        "status": "building",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_same",
                        "normalize_embeddings": True,
                    }
                if args[0] == "active-1":
                    return {
                        "id": "active-1",
                        "status": "active",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_same",
                        "normalize_embeddings": True,
                    }
            return None

        async def fetch(self, query, *args):
            if "DISTINCT chapter_id" in query:
                return [{"chapter_id": "ch01"}, {"chapter_id": "ch02"}]
            return []

    repo = RagRepository(MockConn())  # type: ignore
    with pytest.raises(ValueError, match="TEMP_CORPUS_MUST_CONTAIN_EXACTLY_ONE_CHAPTER"):
        await repo.promote_chapter_from_temp_to_active(
            temp_version_id="temp-multi",
            active_version_id="active-1",
            board_id="fbise",
            class_id="class-9",
            subject_id="chemistry",
            chapter_id="ch01",
        )


@pytest.mark.asyncio
async def test_qa_cleanup_distinction():
    """Verify cleanup_chapter_qa removes citations for manual Q&A while deleting LLM Q&A."""
    queries = []

    class MockConn:
        async def execute(self, query, *args):
            queries.append((query, args))
            return "DELETE 1"

        async def fetch(self, query, *args):
            queries.append((query, args))
            if "question_bank_revisions" in query:
                return [{"revision_id": "rev-llm-1", "question_id": "q-llm-1"}]
            return []

        async def fetchval(self, query, *args):
            queries.append((query, args))
            if "COUNT(*)" in query and "question_bank_revisions" in query:
                return 0
            return 0

    repo = QuestionBankRepository(MockConn())  # type: ignore
    result = await repo.cleanup_chapter_qa(
        board_id="fbise",
        class_id="class-9",
        subject_id="chemistry",
        chapter_id="ch01",
        old_chunk_ids=["chunk-old-1"],
        old_visual_ids=["vis-old-1"],
    )

    assert result["deleted_llm_revisions"] == 1
    assert result["deleted_llm_questions"] == 1

    deleted_chunk_citations = any(
        "DELETE FROM question_bank_revision_citations WHERE chunk_id = ANY" in q[0]
        for q in queries
    )
    assert deleted_chunk_citations, "Must remove citation links for manual Q&A referencing deleted chunks"

    deleted_visual_links = any(
        "DELETE FROM question_bank_revision_visuals WHERE visual_id = ANY" in q[0]
        for q in queries
    )
    assert deleted_visual_links, "Must remove visual links referencing deleted visuals"
