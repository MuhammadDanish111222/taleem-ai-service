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


@pytest.mark.asyncio
async def test_fresh_temp_corpus_for_active_subject():
    """When an active corpus exists, create_building_corpus_version always creates a fresh version
    rather than reusing any existing building/qa_ready draft."""
    queries = []

    class MockConn:
        async def execute(self, query, *args):
            queries.append(("execute", query, args))

        async def fetchrow(self, query, *args):
            queries.append(("fetchrow", query, args))
            if "FOR UPDATE" in query and "rag_corpora" in query:
                return None  # Lock acquired
            if "MAX(version_no)" in query:
                return {"max_v": 3}  # Already has 3 versions
            if "INSERT INTO rag_corpus_versions" in query:
                return {
                    "id": "fresh-building-version",
                    "corpus_id": "corpus-1",
                    "version_no": 4,
                    "status": "building",
                    "embedding_model": "voyage-3-lite",
                    "embedding_revision": "v1",
                    "embedding_dim": 512,
                    "embedding_config_fingerprint": "fp1",
                    "normalize_embeddings": True,
                    "query_instruction": None,
                }
            return None

    repo = RagRepository(MockConn())  # type: ignore
    result = await repo.create_building_corpus_version(
        corpus_id="corpus-1",
        embedding_model="voyage-3-lite",
        embedding_revision="v1",
        embedding_dim=512,
        embedding_config_fingerprint="fp1",
        normalize_embeddings=True,
    )

    assert result["id"] == "fresh-building-version"
    assert result["version_no"] == 4
    assert result["status"] == "building"

    # Verify it did NOT attempt to look up existing building/qa_ready versions
    select_building_or_qa = [
        q for q in queries
        if "status = 'building'" in q[1] or "status = 'qa_ready'" in q[1]
    ]
    assert len(select_building_or_qa) == 0, (
        "create_building_corpus_version must NOT look for existing building/qa_ready versions"
    )


@pytest.mark.asyncio
async def test_successful_promotion_deletes_old_and_moves_new_chunks():
    """Verify that successful promotion deletes old chapter chunks from active version
    and moves new chunks from temp to active version."""
    queries = []

    class MockConn:
        def transaction(self):
            return MockTx()

        async def fetchrow(self, query, *args):
            queries.append(("fetchrow", query, args))
            if "FOR UPDATE" in query:
                if args[0] == "temp-v":
                    return {
                        "id": "temp-v",
                        "status": "building",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_same",
                        "normalize_embeddings": True,
                    }
                if args[0] == "active-v":
                    return {
                        "id": "active-v",
                        "status": "active",
                        "embedding_model": "voyage-3-lite",
                        "embedding_revision": "v1",
                        "embedding_dim": 512,
                        "embedding_config_fingerprint": "fp_same",
                        "normalize_embeddings": True,
                    }
            if "expected_chunk_count" in query or "embedded_chunk_count" in query:
                return {
                    "expected_chunk_count": 5,
                    "embedded_chunk_count": 5,
                    "expected_question_count": 10,
                    "embedded_question_count": 10,
                }
            return None

        async def fetch(self, query, *args):
            queries.append(("fetch", query, args))
            if "DISTINCT chapter_id" in query:
                return [{"chapter_id": "ch01"}]
            if "rag_document_versions" in query and "temp-v" in str(args):
                return [{"resource_id": "jsonl:chapter:ch01"}]
            if "rag_chunks" in query and "id::text" in query and "active-v" in str(args):
                return [{"id": "old-chunk-1"}, {"id": "old-chunk-2"}]
            if "rag_visuals" in query and "id::text" in query:
                return [{"id": "old-vis-1"}]
            return []

        async def fetchval(self, query, *args):
            queries.append(("fetchval", query, args))
            if "COUNT(*)" in query:
                return 0
            return 0

        async def execute(self, query, *args):
            queries.append(("execute", query, args))
            return "UPDATE 1"

    repo = RagRepository(MockConn())  # type: ignore
    await repo.promote_chapter_from_temp_to_active(
        temp_version_id="temp-v",
        active_version_id="active-v",
        board_id="fbise",
        class_id="class-9",
        subject_id="chemistry",
        chapter_id="ch01",
    )

    # Verify old chapter chunks were deleted
    delete_chunk_queries = [
        q for q in queries
        if q[0] == "execute"
        and "DELETE" in q[1]
        and "rag_chunks" in q[1]
        and "active-v" in str(q[2])
    ]
    assert len(delete_chunk_queries) > 0, "Must delete old chapter chunks from active version"

    # Verify new chunks were moved from temp to active
    update_chunk_queries = [
        q for q in queries
        if q[0] == "execute"
        and "UPDATE" in q[1]
        and "rag_chunks" in q[1]
        and "corpus_version_id" in q[1]
    ]
    assert len(update_chunk_queries) > 0, "Must move temp chunks to active version"

    # Verify temp version was deleted
    delete_temp_queries = [
        q for q in queries
        if q[0] == "execute"
        and "DELETE" in q[1]
        and "rag_corpus_versions" in q[1]
    ]
    assert len(delete_temp_queries) > 0, "Must delete temp building corpus version after promotion"


@pytest.mark.asyncio
async def test_visual_review_status_from_chunk_data():
    """Verify replace_chapter_chunks creates visuals with review_status and display_policy
    from the visual dict (trusted admin pipeline) instead of hardcoded 'pending'."""
    visual_inserts = []

    class MockConn:
        async def fetchrow(self, query, *args):
            if "FOR UPDATE" in query:
                return {"status": "building"}
            if "INSERT INTO rag_chunks" in query:
                return {
                    "id": "chunk-1",
                    "document_version_id": "doc-1",
                    "corpus_version_id": "cv-1",
                    "chunk_index": 0,
                    "content": "test",
                    "chapter_id": "ch01",
                    "topic_no": "1.1",
                    "topic_title": "Test",
                    "content_type": "explanation",
                    "content_hash": "abc",
                }
            if "expected_chunk_count" in query or "embedded_chunk_count" in query:
                return {
                    "expected_chunk_count": 1,
                    "embedded_chunk_count": 0,
                    "expected_question_count": 0,
                    "embedded_question_count": 0,
                }
            return None

        async def fetch(self, query, *args):
            return []

        async def execute(self, query, *args):
            if "INSERT INTO rag_visuals" in query:
                visual_inserts.append(args)
            return "INSERT 1"

    repo = RagRepository(MockConn())  # type: ignore
    await repo.replace_chapter_chunks(
        corpus_version_id="cv-1",
        document_version_id="doc-1",
        chunks=[{
            "chunk_order": 0,
            "chunk_text": "Cells are basic units.",
            "chapter_id": "ch01",
            "topic_no": "1.1",
            "topic_title": "Cells",
            "content_type": "explanation",
            "content_hash": "abc123",
            "metadata": {},
            "expected_questions": [],
            "visuals": [
                {
                    "visual_id": "v1",
                    "visual_type": "diagram",
                    "storage_key": "drive-key-v1",
                    "title": "Cell Diagram",
                    "description": "Eukaryotic cell",
                    "review_status": "approved",
                    "display_policy": "always_show",
                },
                {
                    "visual_id": "v2",
                    "visual_type": "figure",
                    "storage_key": "drive-key-v2",
                    "title": "Mitosis",
                    "description": "Cell division",
                    # No review_status/display_policy — should default to 'approved'/'llm_decide'
                },
            ],
        }],
    )

    assert len(visual_inserts) == 2, "Expected 2 visual inserts"

    # Visual 1: explicitly approved with always_show
    v1_args = visual_inserts[0]
    # Args order: chunk_id, visual_id, visual_type, storage_key, title, description, display_policy, review_status, visual_text_hash
    assert v1_args[1] == "v1"
    assert v1_args[6] == "always_show", "display_policy should be 'always_show' from visual dict"
    assert v1_args[7] == "approved", "review_status should be 'approved' from visual dict"

    # Visual 2: defaults to approved/llm_decide
    v2_args = visual_inserts[1]
    assert v2_args[1] == "v2"
    assert v2_args[6] == "llm_decide", "display_policy should default to 'llm_decide'"
    assert v2_args[7] == "approved", "review_status should default to 'approved'"
