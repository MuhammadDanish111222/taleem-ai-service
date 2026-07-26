"""Rollback-safe Phase 3F activation, editing, and draft-QA integration tests.

All tests except the lock-contention case run inside one transaction that is
rolled back.  PostgreSQL row-lock contention needs independent committed
connections; that test deletes only its generated corpus and audit targets in
its ``finally`` block.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from uuid import uuid4

import asyncpg
import pytest

from app.providers.embeddings.bge import BGEEmbeddingConfiguration
from app.repositories.rag_repository import RagRepository
from app.services.local_admin import LocalAdminError, LocalAdminService
from app.services.retrieval.evidence import RetrievalScope
from app.services.retrieval.service import RetrievalScopeError, RetrievalService


def _vector(index: int = 0) -> list[float]:
    result = [0.0] * 768
    result[index] = 1.0
    return result


@dataclass
class _QueryProvider:
    configuration: BGEEmbeddingConfiguration

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [_vector()]


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


async def _new_version(
    conn: asyncpg.Connection,
    *,
    scope: dict[str, str],
    version_no: int,
    label: str,
    chunks: int = 2,
    ready: bool = True,
) -> dict[str, object]:
    """Create a complete, provenance-valid snapshot without running a model."""
    repo = RagRepository(conn)
    configuration = BGEEmbeddingConfiguration()
    corpus = await repo.get_or_create_corpus(**scope)
    version = await repo.create_corpus_version(
        str(corpus["id"]),
        version_no,
        configuration.model,
        configuration.revision,
        configuration.dimensions,
        embedding_config_fingerprint=configuration.fingerprint(),
        normalize_embeddings=configuration.normalize,
        query_instruction=configuration.query_instruction,
    )
    document = await repo.create_document_version(
        str(version["id"]), f"resource-{label}", "v1", "phase3f-test", label
    )
    input_chunks = [
        {
            "chapter_id": "chapter-1",
            "topic_no": str(index + 1),
            "topic_title": f"Topic {index + 1}",
            "chunk_order": index,
            "chunk_text": f"{label} retrieval evidence chunk {index}",
            "content_type": "explanation",
            "content_hash": f"{index + version_no + 1:064x}",
            "language": "en",
            "token_count": 6,
            "metadata": {},
            "expected_questions": [f"{label} question {index}"],
            "visuals": [
                {
                    "visual_id": f"{label}-visual-{index}",
                    "visual_type": "diagram",
                    "title": f"{label} diagram {index}",
                    "description": f"{label} description {index}",
                    "storage_key": f"server-only-{label}-{index}",
                }
            ],
        }
        for index in range(chunks)
    ]
    inserted = await repo.replace_chapter_chunks(
        str(version["id"]), str(document["id"]), input_chunks
    )
    # Visual text participates only after human approval.  Approve before
    # calculating the embedding input and writing the test provenance.
    await conn.execute(
        "UPDATE rag_visuals SET review_status='approved' WHERE chunk_id IN (SELECT id FROM rag_chunks WHERE corpus_version_id=$1::uuid)",
        str(version["id"]),
    )
    await repo.refresh_embedding_input_fingerprint(str(version["id"]))
    for index, chunk in enumerate(inserted):
        assert await repo.write_chunk_embedding(
            str(chunk["id"]),
            str(version["id"]),
            _vector(index),
            f"chunk-{label}-{index}",
            configuration.model,
            configuration.revision,
            configuration.fingerprint(),
        )
    questions = await conn.fetch(
        """SELECT q.id FROM chunk_expected_questions q JOIN rag_chunks c ON c.id=q.chunk_id
           WHERE c.corpus_version_id=$1::uuid ORDER BY c.chunk_index, q.id""",
        str(version["id"]),
    )
    for index, question in enumerate(questions):
        assert await repo.write_question_embedding(
            str(question["id"]),
            str(version["id"]),
            _vector(index),
            f"question-{label}-{index}",
            configuration.model,
            configuration.revision,
            configuration.fingerprint(),
        )
    await repo.refresh_embedding_counts(str(version["id"]))
    if ready:
        assert (await repo.mark_qa_ready(str(version["id"])))["ready"]
    return {
        "corpus_id": str(corpus["id"]),
        "version_id": str(version["id"]),
        "configuration": configuration,
    }


async def _approve(
    conn: asyncpg.Connection,
    version_id: str,
    scope: dict[str, str],
    request_id: str = "qa",
) -> None:
    await LocalAdminService(conn).approve_qa(
        version_id=version_id,
        scope=scope,
        actor_id="local-admin",
        request_id=request_id,
    )


async def _activate_initial(
    conn: asyncpg.Connection, version_id: str, scope: dict[str, str]
) -> None:
    await _approve(conn, version_id, scope, "initial-qa")
    await LocalAdminService(conn).activate(
        version_id=version_id,
        scope=scope,
        actor_id="local-admin",
        request_id="initial-activation",
    )


def _scope() -> dict[str, str]:
    suffix = uuid4().hex
    return {"board_id": f"p3f-{suffix}", "class_id": "class-9", "subject_id": "physics"}


@pytest.mark.asyncio
async def test_activation_rechecks_completeness_provenance_and_qa_approval(conn):
    scope = _scope()
    incomplete = await _new_version(
        conn, scope=scope, version_no=1, label="incomplete", ready=False
    )
    service = LocalAdminService(conn)
    with pytest.raises(LocalAdminError, match="ACTIVATION_TARGET_NOT_QA_READY"):
        await service.activate(
            version_id=incomplete["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="incomplete",
        )

    failed = await _new_version(conn, scope=scope, version_no=2, label="failed")
    await _approve(conn, failed["version_id"], scope, "failed-qa")
    failed_chunk = await conn.fetchval(
        "SELECT id FROM rag_chunks WHERE corpus_version_id=$1::uuid LIMIT 1",
        failed["version_id"],
    )
    assert await RagRepository(conn).mark_embedding_failed(
        "rag_chunks", str(failed_chunk), "TEST_FAILURE"
    )
    with pytest.raises(LocalAdminError, match="ACTIVATION_CORPUS_INCOMPLETE"):
        await service.activate(
            version_id=failed["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="failed",
        )

    invalid_provenance = await _new_version(
        conn, scope=scope, version_no=3, label="invalid-provenance"
    )
    await _approve(conn, invalid_provenance["version_id"], scope, "provenance-qa")
    await conn.execute(
        "UPDATE rag_chunks SET embedding_revision='wrong-revision' WHERE corpus_version_id=$1::uuid",
        invalid_provenance["version_id"],
    )
    with pytest.raises(LocalAdminError, match="ACTIVATION_CORPUS_INCOMPLETE"):
        await service.activate(
            version_id=invalid_provenance["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="invalid-provenance",
        )

    stale = await _new_version(conn, scope=scope, version_no=4, label="stale")
    await _approve(conn, stale["version_id"], scope, "stale-qa")
    await conn.execute(
        "UPDATE rag_corpus_qa_approvals SET invalidated_at=NOW() WHERE corpus_version_id=$1::uuid",
        stale["version_id"],
    )
    with pytest.raises(LocalAdminError, match="ACTIVATION_QA_APPROVAL_REQUIRED"):
        await service.activate(
            version_id=stale["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="stale",
        )


@pytest.mark.asyncio
async def test_question_edits_are_draft_only_and_targeted(conn):
    scope = _scope()
    active = await _new_version(
        conn, scope=scope, version_no=1, label="question-active"
    )
    await _activate_initial(conn, active["version_id"], scope)
    service = LocalAdminService(conn)
    active_question = await conn.fetchval(
        """SELECT q.id FROM chunk_expected_questions q JOIN rag_chunks c ON c.id=q.chunk_id
           WHERE c.corpus_version_id=$1::uuid LIMIT 1""",
        active["version_id"],
    )
    with pytest.raises(LocalAdminError, match="ACTIVE_VERSION_IMMUTABLE"):
        await service.edit_question(
            version_id=active["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="active-edit",
            question_id=str(active_question),
            question_text="must fail",
            chunk_id=None,
        )

    draft = await _new_version(conn, scope=scope, version_no=2, label="question-draft")
    await _approve(conn, draft["version_id"], scope, "draft-qa")
    rows = await conn.fetch(
        """SELECT q.id, q.chunk_id, q.embedding_input_hash FROM chunk_expected_questions q
           JOIN rag_chunks c ON c.id=q.chunk_id WHERE c.corpus_version_id=$1::uuid ORDER BY q.id""",
        draft["version_id"],
    )
    changed, untouched = rows
    await service.edit_question(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="edit",
        question_id=str(changed["id"]),
        question_text="changed expected question",
        chunk_id=None,
    )
    version_status = await conn.fetchval(
        "SELECT status FROM rag_corpus_versions WHERE id=$1::uuid", draft["version_id"]
    )
    assert version_status == "building"
    changed_state = await conn.fetchrow(
        "SELECT embedding_status, embedding FROM chunk_expected_questions WHERE id=$1",
        changed["id"],
    )
    untouched_state = await conn.fetchrow(
        "SELECT embedding_status, embedding_input_hash FROM chunk_expected_questions WHERE id=$1",
        untouched["id"],
    )
    assert (
        changed_state["embedding_status"] == "pending"
        and changed_state["embedding"] is None
    )
    assert (
        untouched_state["embedding_status"] == "embedded"
        and untouched_state["embedding_input_hash"] == untouched["embedding_input_hash"]
    )
    assert await conn.fetchval(
        "SELECT NOT EXISTS(SELECT 1 FROM rag_corpus_qa_approvals WHERE corpus_version_id=$1::uuid AND invalidated_at IS NULL)",
        draft["version_id"],
    )
    job_count = await conn.fetchval(
        "SELECT count(*) FROM job_queue WHERE job_type='embed_questions' AND payload->>'corpus_version_id'=$1",
        draft["version_id"],
    )
    assert job_count == 1

    config = draft["configuration"]
    assert await RagRepository(conn).write_question_embedding(
        str(changed["id"]),
        draft["version_id"],
        _vector(),
        "changed-question",
        config.model,
        config.revision,
        config.fingerprint(),
    )
    added = await service.edit_question(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="add",
        question_id=None,
        question_text="new expected question",
        chunk_id=str(changed["chunk_id"]),
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM chunk_expected_questions WHERE id=$1::uuid",
            added["id"],
        )
        == "pending"
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM chunk_expected_questions WHERE id=$1",
            untouched["id"],
        )
        == "embedded"
    )
    assert await RagRepository(conn).write_question_embedding(
        added["id"],
        draft["version_id"],
        _vector(),
        "new-question",
        config.model,
        config.revision,
        config.fingerprint(),
    )
    await service.edit_question(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="delete",
        question_id=added["id"],
        question_text=None,
        chunk_id=None,
        delete=True,
    )
    assert not await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM chunk_expected_questions WHERE id=$1::uuid)",
        added["id"],
    )
    counts = await RagRepository(conn).refresh_embedding_counts(draft["version_id"])
    assert counts["expected_question_count"] == counts["embedded_question_count"] == 2


@pytest.mark.asyncio
async def test_draft_clone_reuses_only_valid_embedding_provenance(conn):
    scope = _scope()
    active = await _new_version(
        conn, scope=scope, version_no=1, label="clone-provenance"
    )
    await _activate_initial(conn, active["version_id"], scope)
    invalid_chunk = await conn.fetchval(
        "SELECT id FROM rag_chunks WHERE corpus_version_id=$1::uuid ORDER BY chunk_index LIMIT 1",
        active["version_id"],
    )
    invalid_question = await conn.fetchval(
        """SELECT q.id FROM chunk_expected_questions q JOIN rag_chunks c ON c.id=q.chunk_id
           WHERE c.corpus_version_id=$1::uuid ORDER BY c.chunk_index LIMIT 1""",
        active["version_id"],
    )
    repo = RagRepository(conn)
    assert await repo.mark_embedding_failed(
        "rag_chunks", str(invalid_chunk), "TEST_STALE"
    )
    assert await repo.mark_embedding_failed(
        "chunk_expected_questions", str(invalid_question), "TEST_STALE"
    )
    draft = await LocalAdminService(conn).create_draft(
        active_version_id=active["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="clone-stale",
    )
    cloned_chunks = await conn.fetch(
        "SELECT embedding_status, embedding IS NOT NULL AS has_embedding FROM rag_chunks WHERE corpus_version_id=$1::uuid ORDER BY chunk_index",
        draft["id"],
    )
    cloned_questions = await conn.fetch(
        """SELECT q.embedding_status, q.embedding IS NOT NULL AS has_embedding
           FROM chunk_expected_questions q JOIN rag_chunks c ON c.id=q.chunk_id
           WHERE c.corpus_version_id=$1::uuid ORDER BY c.chunk_index""",
        draft["id"],
    )
    assert [row["embedding_status"] for row in cloned_chunks] == ["pending", "embedded"]
    assert [row["has_embedding"] for row in cloned_chunks] == [False, True]
    assert [row["embedding_status"] for row in cloned_questions] == [
        "pending",
        "embedded",
    ]
    assert [row["has_embedding"] for row in cloned_questions] == [False, True]


@pytest.mark.asyncio
async def test_visual_edits_only_invalidate_parent_chunk_and_active_is_immutable(conn):
    scope = _scope()
    active = await _new_version(conn, scope=scope, version_no=1, label="visual-active")
    await _activate_initial(conn, active["version_id"], scope)
    service = LocalAdminService(conn)
    active_visual = await conn.fetchval(
        "SELECT v.id FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id WHERE c.corpus_version_id=$1::uuid LIMIT 1",
        active["version_id"],
    )
    with pytest.raises(LocalAdminError, match="ACTIVE_VERSION_IMMUTABLE"):
        await service.edit_visual(
            version_id=active["version_id"],
            scope=scope,
            actor_id="admin",
            request_id="active-visual",
            visual_id=str(active_visual),
            title="new",
            description=None,
            review_status=None,
            display_policy=None,
        )

    draft = await _new_version(conn, scope=scope, version_no=2, label="visual-draft")
    await _approve(conn, draft["version_id"], scope, "visual-qa")
    rows = await conn.fetch(
        """SELECT v.id AS visual_id, v.chunk_id FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id
           WHERE c.corpus_version_id=$1::uuid ORDER BY v.visual_id""",
        draft["version_id"],
    )
    changed, unrelated = rows
    await service.edit_visual(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="title",
        visual_id=str(changed["visual_id"]),
        title="changed approved title",
        description=None,
        review_status=None,
        display_policy=None,
    )
    changed_state = await conn.fetchrow(
        "SELECT embedding_status, embedding FROM rag_chunks WHERE id=$1",
        changed["chunk_id"],
    )
    unrelated_state = await conn.fetchrow(
        "SELECT embedding_status, embedding FROM rag_chunks WHERE id=$1",
        unrelated["chunk_id"],
    )
    assert (
        changed_state["embedding_status"] == "pending"
        and changed_state["embedding"] is None
    )
    assert (
        unrelated_state["embedding_status"] == "embedded"
        and unrelated_state["embedding"] is not None
    )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM job_queue WHERE job_type='embed_chunks' AND payload->>'corpus_version_id'=$1",
            draft["version_id"],
        )
        == 1
    )
    config = draft["configuration"]
    assert await RagRepository(conn).write_chunk_embedding(
        str(changed["chunk_id"]),
        draft["version_id"],
        _vector(),
        "changed-visual",
        config.model,
        config.revision,
        config.fingerprint(),
    )
    await service.edit_visual(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="review",
        visual_id=str(changed["visual_id"]),
        title=None,
        description=None,
        review_status="rejected",
        display_policy=None,
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM rag_chunks WHERE id=$1", changed["chunk_id"]
        )
        == "pending"
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM rag_chunks WHERE id=$1", unrelated["chunk_id"]
        )
        == "embedded"
    )
    assert await RagRepository(conn).write_chunk_embedding(
        str(changed["chunk_id"]),
        draft["version_id"],
        _vector(),
        "review-visual",
        config.model,
        config.revision,
        config.fingerprint(),
    )
    await conn.execute(
        "INSERT INTO rag_corpus_qa_approvals (corpus_version_id,reviewer_id,request_id,summary) VALUES ($1::uuid,'admin','policy','{}')",
        draft["version_id"],
    )
    await service.edit_visual(
        version_id=draft["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="policy",
        visual_id=str(changed["visual_id"]),
        title=None,
        description=None,
        review_status=None,
        display_policy="always",
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM rag_chunks WHERE id=$1", changed["chunk_id"]
        )
        == "embedded"
    )
    assert (
        await conn.fetchval(
            "SELECT embedding_status FROM rag_chunks WHERE id=$1", unrelated["chunk_id"]
        )
        == "embedded"
    )
    assert await conn.fetchval(
        "SELECT NOT EXISTS(SELECT 1 FROM rag_corpus_qa_approvals WHERE corpus_version_id=$1::uuid AND invalidated_at IS NULL)",
        draft["version_id"],
    )
    inspected = await service.inspect_version(
        version_id=draft["version_id"], scope=scope
    )
    assert "server-only-visual-draft" not in str(inspected)


@pytest.mark.asyncio
async def test_named_draft_qa_is_isolated_and_rollback_changes_active_retrieval(conn):
    scope = _scope()
    first = await _new_version(conn, scope=scope, version_no=1, label="first-active")
    await _activate_initial(conn, first["version_id"], scope)
    second = await _new_version(conn, scope=scope, version_no=2, label="named-draft")
    retrieval = RetrievalService(
        conn, lambda configuration: _QueryProvider(configuration)
    )
    active_before = await retrieval.retrieve(
        "retrieval evidence", RetrievalScope(**scope)
    )
    named = await retrieval.retrieve_named_version(
        "retrieval evidence", RetrievalScope(**scope), second["version_id"]
    )
    assert any(
        "first-active" in item.citation.content for item in active_before.results
    )
    assert any("named-draft" in item.citation.content for item in named.results)
    assert (
        str((await RagRepository(conn).get_active_corpus_version(**scope))["id"])
        == first["version_id"]
    )
    with pytest.raises(RetrievalScopeError, match="OUTSIDE_SCOPE"):
        await retrieval.retrieve_named_version(
            "retrieval",
            RetrievalScope(scope["board_id"], scope["class_id"], "other-subject"),
            second["version_id"],
        )

    await _approve(conn, second["version_id"], scope, "second-qa")
    service = LocalAdminService(conn)
    await service.activate(
        version_id=second["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="activate-second",
    )
    active_second = await retrieval.retrieve(
        "retrieval evidence", RetrievalScope(**scope)
    )
    assert any("named-draft" in item.citation.content for item in active_second.results)
    await service.activate(
        version_id=first["version_id"],
        scope=scope,
        actor_id="admin",
        request_id="rollback-first",
        rollback=True,
    )
    active_rolled_back = await retrieval.retrieve(
        "retrieval evidence", RetrievalScope(**scope)
    )
    assert any(
        "first-active" in item.citation.content for item in active_rolled_back.results
    )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM rag_corpus_versions WHERE corpus_id=$1::uuid AND status='active'",
            first["corpus_id"],
        )
        == 1
    )
    assert await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM admin_audit_logs WHERE target_id=$1 AND action='corpus_rollback')",
        first["version_id"],
    )
    wrong_scope = {**scope, "subject_id": "other-subject"}
    with pytest.raises(LocalAdminError, match="CORPUS_SCOPE_NOT_FOUND"):
        await service.activate(
            version_id=first["version_id"],
            scope=wrong_scope,
            actor_id="admin",
            request_id="cross-scope",
        )


@pytest.mark.asyncio
async def test_concurrent_activations_keep_one_active_and_auditable():
    """Uses committed fixture rows only because two DB sessions must contend."""
    setup = await asyncpg.connect(os.environ["DATABASE_URL"])
    scope = _scope()
    corpus_id = ""
    version_ids: list[str] = []
    try:
        initial = await _new_version(
            setup, scope=scope, version_no=1, label="concurrent-initial"
        )
        corpus_id = initial["corpus_id"]
        await _activate_initial(setup, initial["version_id"], scope)
        first = await _new_version(
            setup, scope=scope, version_no=2, label="concurrent-first"
        )
        second = await _new_version(
            setup, scope=scope, version_no=3, label="concurrent-second"
        )
        version_ids = [initial["version_id"], first["version_id"], second["version_id"]]
        await _approve(setup, first["version_id"], scope, "concurrent-first-qa")
        await _approve(setup, second["version_id"], scope, "concurrent-second-qa")

        async def activate(version_id: str, request_id: str) -> None:
            connection = await asyncpg.connect(os.environ["DATABASE_URL"])
            try:
                await LocalAdminService(connection).activate(
                    version_id=version_id,
                    scope=scope,
                    actor_id="admin",
                    request_id=request_id,
                )
            finally:
                await connection.close()

        await asyncio.gather(
            activate(first["version_id"], "concurrent-a"),
            activate(second["version_id"], "concurrent-b"),
        )
        active_count = await setup.fetchval(
            "SELECT count(*) FROM rag_corpus_versions WHERE corpus_id=$1::uuid AND status='active'",
            corpus_id,
        )
        assert active_count == 1
        assert (
            await setup.fetchval(
                "SELECT count(*) FROM admin_audit_logs WHERE action='corpus_activated' AND target_id = ANY($1::text[])",
                version_ids,
            )
            == 3
        )
        statuses = await setup.fetch(
            "SELECT status FROM rag_corpus_versions WHERE corpus_id=$1::uuid", corpus_id
        )
        assert {row["status"] for row in statuses} <= {"active", "superseded"}
    finally:
        if version_ids:
            await setup.execute(
                "DELETE FROM admin_audit_logs WHERE target_id = ANY($1::text[])",
                version_ids,
            )
        if corpus_id:
            await setup.execute("DELETE FROM rag_corpora WHERE id=$1::uuid", corpus_id)
        await setup.close()
