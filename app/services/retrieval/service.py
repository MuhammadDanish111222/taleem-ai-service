"""On-demand, internal-only orchestration for Phase 3E retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

import asyncpg

from app.providers.embeddings.bge import BGEEmbeddingConfiguration, BGEEmbeddingProvider
from app.repositories.rag_repository import RagRepository
from app.services.retrieval.evidence import (
    Citation,
    EvidenceResult,
    RetrievalChannel,
    RetrievalScope,
    classify_evidence,
)
from app.services.retrieval.fusion import RankedChannelHit, fuse_ranked_hits


class QueryEmbeddingProvider(Protocol):
    configuration: BGEEmbeddingConfiguration
    configuration_fingerprint: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class RetrievalConfigurationError(RuntimeError):
    """The active corpus configuration cannot safely be used for a query."""


class RetrievalScopeError(ValueError):
    """The optional chapter is outside the supplied active corpus scope."""


class RetrievalService:
    """Retrieves only from the exact active corpus selected by a supplied scope."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        provider_factory: Callable[[BGEEmbeddingConfiguration], QueryEmbeddingProvider]
        | None = None,
        *,
        dense_top_k: int = 10,
        expected_question_top_k: int = 10,
        lexical_top_k: int = 10,
    ):
        if min(dense_top_k, expected_question_top_k, lexical_top_k) < 1:
            raise ValueError("RETRIEVAL_TOP_K_MUST_BE_POSITIVE")
        self._repo = RagRepository(conn)
        self._provider_factory = provider_factory or BGEEmbeddingProvider
        self._dense_top_k = dense_top_k
        self._expected_question_top_k = expected_question_top_k
        self._lexical_top_k = lexical_top_k

    async def retrieve(self, question: str, scope: RetrievalScope) -> EvidenceResult:
        """Run all three channels without logging question text or provider output."""
        active_version = await self._repo.get_active_corpus_version(
            scope.board_id, scope.class_id, scope.subject_id
        )
        if active_version is None:
            return classify_evidence(())
        return await self._retrieve_version(question, scope, active_version, allow_named_draft=False)

    async def retrieve_named_version(self, question: str, scope: RetrievalScope, corpus_version_id: str) -> EvidenceResult:
        """Local QA only: named building/qa-ready snapshot; never changes active resolution."""
        version = await self._repo.get_corpus_version(corpus_version_id)
        if not version or version["status"] not in {"building", "qa_ready"}:
            raise RetrievalScopeError("QA_CORPUS_VERSION_NOT_ELIGIBLE")
        scoped = await self._repo.conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM rag_corpus_versions cv JOIN rag_corpora c ON c.id=cv.corpus_id
               WHERE cv.id=$1::uuid AND c.board_id=$2 AND c.class_id=$3 AND c.subject_id=$4)""",
            corpus_version_id, scope.board_id, scope.class_id, scope.subject_id,
        )
        if not scoped:
            raise RetrievalScopeError("QA_CORPUS_VERSION_OUTSIDE_SCOPE")
        return await self._retrieve_version(question, scope, version, allow_named_draft=True)

    async def _retrieve_version(self, question: str, scope: RetrievalScope, active_version: dict[str, Any], *, allow_named_draft: bool) -> EvidenceResult:
        normalized_question = " ".join(question.split())
        if not normalized_question:
            raise ValueError("RETRIEVAL_QUESTION_BLANK")
        corpus_version_id = str(active_version["id"])
        chapter_exists = await self._repo.active_chapter_exists(
            scope.board_id, scope.class_id, scope.subject_id, corpus_version_id, scope.chapter_id
        ) if scope.chapter_id is not None and not allow_named_draft else (
            await self._repo.conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_chunks WHERE corpus_version_id=$1::uuid AND chapter_id=$2)", corpus_version_id, scope.chapter_id)
            if scope.chapter_id is not None else True
        )
        if not chapter_exists:
            raise RetrievalScopeError("CHAPTER_NOT_IN_ACTIVE_CORPUS")

        configuration = self._configuration_from_active_version(active_version)
        provider = self._provider_factory(configuration)
        if provider.configuration_fingerprint != configuration.fingerprint():
            raise RetrievalConfigurationError("QUERY_PROVIDER_CONFIGURATION_MISMATCH")

        # Model inference is bounded and on-demand, but still kept off FastAPI's
        # event loop. It is intentionally not represented by a durable worker job.
        vectors = await asyncio.to_thread(provider.embed_queries, [normalized_question])
        if len(vectors) != 1 or len(vectors[0]) != configuration.dimensions:
            raise RetrievalConfigurationError("QUERY_EMBEDDING_DIMENSION_MISMATCH")
        query_vector = vectors[0]

        # A repository is bound to one asyncpg connection, which cannot execute
        # concurrent commands. Keep the channel SQL serial here; callers that need
        # connection-level concurrency can create separate service instances.
        dense_rows = await self._repo.search_active_chunks_cosine(
            scope.board_id, scope.class_id, scope.subject_id, corpus_version_id,
            query_vector, scope.chapter_id, self._dense_top_k, allow_named_draft,
        )
        expected_rows = await self._repo.search_active_expected_questions_cosine(
            scope.board_id, scope.class_id, scope.subject_id, corpus_version_id,
            query_vector, scope.chapter_id, self._expected_question_top_k, allow_named_draft,
        )
        lexical_rows = await self._repo.search_active_chunks_lexical(
            scope.board_id, scope.class_id, scope.subject_id, corpus_version_id,
            normalized_question, scope.chapter_id, self._lexical_top_k, allow_named_draft,
        )
        hits = [
            *self._channel_hits(dense_rows, RetrievalChannel.DENSE),
            *self._channel_hits(expected_rows, RetrievalChannel.EXPECTED_QUESTION),
            *self._channel_hits(lexical_rows, RetrievalChannel.LEXICAL),
        ]
        return classify_evidence(fuse_ranked_hits(hits))

    @staticmethod
    def _configuration_from_active_version(version: dict[str, Any]) -> BGEEmbeddingConfiguration:
        try:
            configuration = BGEEmbeddingConfiguration(
                model=version["embedding_model"],
                revision=version["embedding_revision"],
                dimensions=version["embedding_dim"],
                normalize=version["normalize_embeddings"],
                query_instruction=version["query_instruction"],
            )
        except (KeyError, TypeError) as exc:
            raise RetrievalConfigurationError("ACTIVE_CORPUS_CONFIGURATION_INVALID") from exc
        if version.get("embedding_config_fingerprint") != configuration.fingerprint():
            raise RetrievalConfigurationError("ACTIVE_CORPUS_CONFIGURATION_MISMATCH")
        return configuration

    @staticmethod
    def _channel_hits(rows: list[dict[str, Any]], channel: RetrievalChannel) -> list[RankedChannelHit]:
        hits = []
        for position, row in enumerate(rows, start=1):
            rank = int(row.get("expected_question_rank", position))
            hits.append(
                RankedChannelHit(
                    citation=Citation(
                        citation_id=row["citation_id"],
                        content=row["content"],
                        chapter_id=row["chapter_id"],
                        topic_no=row["topic_no"],
                        topic_title=row["topic_title"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                    ),
                    channel=channel,
                    rank=rank,
                )
            )
        return hits
