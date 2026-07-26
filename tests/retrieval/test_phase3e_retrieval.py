"""Phase 3E retrieval tests using a disposable PostgreSQL + pgvector database."""

from __future__ import annotations

import os
from dataclasses import dataclass

import asyncpg
import pytest

from app.core.worker_modes import WorkerMode, owned_job_types
from app.providers.embeddings.bge import BGEEmbeddingConfiguration
from app.repositories.rag_repository import RagRepository
from app.services.retrieval.evidence import (
    Citation,
    EvidenceStrength,
    RetrievalChannel,
    RetrievalScope,
    classify_evidence,
)
from app.services.retrieval.fusion import RankedChannelHit, fuse_ranked_hits
from app.services.retrieval.service import (
    RetrievalConfigurationError,
    RetrievalService,
)


@dataclass
class FakeQueryEmbeddingProvider:
    configuration: BGEEmbeddingConfiguration
    vector: list[float]

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [self.vector]


@pytest.fixture
async def conn():
    try:
        connection = await asyncpg.connect(
            os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/taleem_dev",
            )
        )
    except (ConnectionRefusedError, OSError):
        pytest.skip("Supported disposable PostgreSQL database is unavailable.")
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


def _vector(index: int = 0) -> list[float]:
    value = [0.0] * 768
    value[index] = 1.0
    return value


async def _activate_corpus(
    conn,
    *,
    board: str,
    class_id: str = "class-9",
    subject: str = "physics",
    suffix: str,
    chunks,
):
    repo = RagRepository(conn)
    configuration = BGEEmbeddingConfiguration()
    version = await repo.get_or_create_building_corpus_version(
        board,
        class_id,
        subject,
        configuration.model,
        configuration.revision,
        configuration.dimensions,
        configuration.fingerprint(),
        configuration.normalize,
        configuration.query_instruction,
    )
    document = await repo.create_document_version(
        str(version["id"]), f"resource-{suffix}", "v1", "admin_jsonl_v1", suffix
    )
    await repo.replace_chapter_chunks(str(version["id"]), str(document["id"]), chunks)
    chunk_rows = await conn.fetch(
        "SELECT id FROM rag_chunks WHERE corpus_version_id = $1::uuid ORDER BY chunk_index;",
        str(version["id"]),
    )
    for chunk, row in zip(chunks, chunk_rows, strict=True):
        await repo.write_chunk_embedding(
            str(row["id"]),
            str(version["id"]),
            _vector(chunk.get("vector_index", 0)),
            f"chunk-input-{suffix}-{row['id']}",
            configuration.model,
            configuration.revision,
            configuration.fingerprint(),
        )
    question_rows = await conn.fetch(
        """
        SELECT q.id, c.chunk_index
        FROM chunk_expected_questions q
        JOIN rag_chunks c ON c.id = q.chunk_id
        WHERE c.corpus_version_id = $1::uuid
        ORDER BY c.chunk_index, q.created_at, q.id;
        """,
        str(version["id"]),
    )
    for question in question_rows:
        await repo.write_question_embedding(
            str(question["id"]),
            str(version["id"]),
            _vector(int(question["chunk_index"])),
            f"question-input-{suffix}-{question['id']}",
            configuration.model,
            configuration.revision,
            configuration.fingerprint(),
        )
    await repo.refresh_embedding_counts(str(version["id"]))
    assert (await repo.mark_qa_ready(str(version["id"])))["ready"]
    assert await repo.activate_corpus_version(str(version["id"]), "test-admin")
    return str(version["id"])


def _chunk(
    index: int, chapter: str, text: str, questions: list[str], vector_index: int = 0
):
    return {
        "chunk_order": index,
        "chunk_text": text,
        "chapter_id": chapter,
        "topic_no": str(index + 1),
        "topic_title": f"Topic {index + 1}",
        "content_type": "explanation",
        "content_hash": f"{index + 1:064x}",
        "expected_questions": questions,
        "vector_index": vector_index,
    }


@pytest.mark.asyncio
async def test_three_channels_are_scoped_to_active_version_and_optional_chapter(conn):
    await _activate_corpus(
        conn,
        board="board-a",
        suffix="superseded",
        chunks=[
            _chunk(
                0,
                "chapter-one",
                "shared orbit evidence superseded",
                ["shared orbit old"],
            )
        ],
    )
    await _activate_corpus(
        conn,
        board="board-a",
        suffix="active",
        chunks=[
            _chunk(
                0,
                "chapter-one",
                "shared orbit evidence active chapter one",
                ["shared orbit one", "shared orbit again"],
            ),
            _chunk(
                1,
                "chapter-two",
                "shared orbit evidence active chapter two",
                ["shared orbit two"],
            ),
        ],
    )
    await _activate_corpus(
        conn,
        board="board-b",
        suffix="other-board",
        chunks=[
            _chunk(
                0,
                "chapter-one",
                "shared orbit evidence hidden board",
                ["shared orbit hidden"],
            )
        ],
    )
    service = RetrievalService(
        conn,
        lambda config: FakeQueryEmbeddingProvider(config, _vector()),
        dense_top_k=10,
        expected_question_top_k=10,
        lexical_top_k=10,
    )
    result = await service.retrieve(
        "shared orbit", RetrievalScope("board-a", "class-9", "physics")
    )
    assert {item.citation.chapter_id for item in result.results} == {
        "chapter-one",
        "chapter-two",
    }
    assert all("hidden" not in item.citation.content for item in result.results)
    assert all("superseded" not in item.citation.content for item in result.results)
    assert (
        await RagRepository(conn).get_active_corpus_version(
            "board-a", "class-9", "physics"
        )
    )["status"] == "active"

    chapter_result = await service.retrieve(
        "shared orbit", RetrievalScope("board-a", "class-9", "physics", "chapter-two")
    )
    assert chapter_result.results
    assert {item.citation.chapter_id for item in chapter_result.results} == {
        "chapter-two"
    }


@pytest.mark.asyncio
async def test_expected_question_hits_resolve_once_to_parent_chunk_with_best_rank(conn):
    await _activate_corpus(
        conn,
        board="board-expected",
        suffix="expected",
        chunks=[
            _chunk(0, "chapter-one", "first parent text", ["question a", "question b"]),
            _chunk(1, "chapter-one", "second parent text", ["question c"]),
        ],
    )
    service = RetrievalService(
        conn,
        lambda config: FakeQueryEmbeddingProvider(config, _vector()),
        dense_top_k=1,
        expected_question_top_k=10,
        lexical_top_k=1,
    )
    result = await service.retrieve(
        "no lexical match", RetrievalScope("board-expected", "class-9", "physics")
    )
    expected_only = [
        item
        for item in result.results
        if any(
            part.channel is RetrievalChannel.EXPECTED_QUESTION
            for part in item.contributions
        )
    ]
    assert len(expected_only) == 2
    expected_ranks = [
        next(
            part.rank
            for part in item.contributions
            if part.channel is RetrievalChannel.EXPECTED_QUESTION
        )
        for item in expected_only
    ]
    assert sorted(expected_ranks) == [1, 2]
    assert all(
        sum(
            contribution.channel is RetrievalChannel.EXPECTED_QUESTION
            for contribution in item.contributions
        )
        == 1
        for item in expected_only
    )
    assert all(
        not hasattr(item.citation, "expected_question_id") for item in expected_only
    )
    assert all("question_text" not in item.citation.__dict__ for item in expected_only)


@pytest.mark.asyncio
async def test_empty_evidence_and_mismatched_active_configuration_fail_safely(conn):
    service = RetrievalService(
        conn, lambda config: FakeQueryEmbeddingProvider(config, _vector())
    )
    empty = await service.retrieve(
        "nothing", RetrievalScope("missing", "class-9", "physics")
    )
    assert empty.strength is EvidenceStrength.NONE
    assert empty.results == ()

    version_id = await _activate_corpus(
        conn,
        board="board-bad-config",
        suffix="bad-config",
        chunks=[_chunk(0, "chapter-one", "active text", ["active question"])],
    )
    await conn.execute(
        "UPDATE rag_corpus_versions SET embedding_config_fingerprint = 'mismatch' WHERE id = $1::uuid;",
        version_id,
    )
    with pytest.raises(
        RetrievalConfigurationError, match="ACTIVE_CORPUS_CONFIGURATION_MISMATCH"
    ):
        await service.retrieve(
            "active", RetrievalScope("board-bad-config", "class-9", "physics")
        )


def test_rrf_is_deterministic_and_never_exposes_confidence():
    citation_a = Citation("citation-a", "text a", None, None, None, None, None)
    citation_b = Citation("citation-b", "text b", None, None, None, None, None)
    input_hits = [
        RankedChannelHit(citation_a, RetrievalChannel.DENSE, 1),
        RankedChannelHit(citation_b, RetrievalChannel.DENSE, 2),
        RankedChannelHit(citation_b, RetrievalChannel.LEXICAL, 1),
    ]
    first = fuse_ranked_hits(input_hits)
    second = fuse_ranked_hits(reversed(input_hits))
    assert first == second
    assert first[0].citation.citation_id == "citation-b"
    assert {part.channel for part in first[0].contributions} == {
        RetrievalChannel.DENSE,
        RetrievalChannel.LEXICAL,
    }
    assert not hasattr(first[0], "confidence")
    assert not hasattr(first[0], "probability")
    assert not hasattr(first[0], "fusion_weight")
    assert all(
        not hasattr(contribution, "rrf_contribution")
        for contribution in first[0].contributions
    )


def test_approved_evidence_policy_uses_only_top_parent_channel_ranks():
    citation = Citation("citation", "text", None, None, None, None, None)
    assert classify_evidence(()).strength is EvidenceStrength.NONE
    single_channel = classify_evidence(
        fuse_ranked_hits([RankedChannelHit(citation, RetrievalChannel.DENSE, 1)])
    )
    assert single_channel.strength is EvidenceStrength.WEAK

    strong = classify_evidence(
        fuse_ranked_hits(
            [
                RankedChannelHit(citation, RetrievalChannel.DENSE, 1),
                RankedChannelHit(citation, RetrievalChannel.LEXICAL, 3),
            ]
        )
    )
    assert strong.strength is EvidenceStrength.STRONG

    outside_top_three = classify_evidence(
        fuse_ranked_hits(
            [
                RankedChannelHit(citation, RetrievalChannel.DENSE, 1),
                RankedChannelHit(citation, RetrievalChannel.LEXICAL, 4),
            ]
        )
    )
    assert outside_top_three.strength is EvidenceStrength.WEAK

    duplicate_expected_questions = classify_evidence(
        fuse_ranked_hits(
            [
                RankedChannelHit(citation, RetrievalChannel.EXPECTED_QUESTION, 1),
                RankedChannelHit(citation, RetrievalChannel.EXPECTED_QUESTION, 2),
            ]
        )
    )
    assert duplicate_expected_questions.strength is EvidenceStrength.WEAK
    assert len(duplicate_expected_questions.results[0].contributions) == 1

    top_weak = Citation("top-weak", "top", None, None, None, None, None)
    lower_strong = Citation("lower-strong", "lower", None, None, None, None, None)
    top_only_policy = classify_evidence(
        fuse_ranked_hits(
            [
                RankedChannelHit(top_weak, RetrievalChannel.DENSE, 1),
                RankedChannelHit(top_weak, RetrievalChannel.LEXICAL, 4),
                RankedChannelHit(lower_strong, RetrievalChannel.DENSE, 3),
                RankedChannelHit(lower_strong, RetrievalChannel.EXPECTED_QUESTION, 3),
            ]
        )
    )
    assert top_only_policy.results[0].citation.citation_id == "top-weak"
    assert top_only_policy.strength is EvidenceStrength.WEAK


def test_railway_public_still_owns_no_durable_jobs():
    assert owned_job_types(WorkerMode.RAILWAY_PUBLIC) == frozenset()
