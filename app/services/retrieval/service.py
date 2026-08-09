"""On-demand, internal-only orchestration for Phase 3E retrieval with Voyage halfvec(512)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol

import asyncpg

from app.providers.embeddings.voyage import (
    VoyageEmbeddingConfiguration,
    VoyageEmbeddingProvider,
)
from app.repositories.rag_repository import RagRepository
from app.services.retrieval.active_version_cache import (
    ActiveCorpusVersionCache,
    get_active_corpus_version_cache,
)
from app.services.retrieval.evidence import (
    Citation,
    EvidenceResult,
    RetrievalChannel,
    RetrievalScope,
    RetrievedVisual,
    classify_evidence,
)
from app.services.retrieval.fusion import RankedChannelHit, fuse_ranked_hits


class QueryEmbeddingProvider(Protocol):
    configuration: VoyageEmbeddingConfiguration
    configuration_fingerprint: str

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class RetrievalConfigurationError(RuntimeError):
    """The active corpus configuration cannot safely be used for a query."""


class RetrievalScopeError(ValueError):
    """The optional chapter is outside the supplied active corpus scope."""


@lru_cache(maxsize=4)
def _cached_voyage_provider(
    configuration: VoyageEmbeddingConfiguration,
) -> VoyageEmbeddingProvider:
    """Reuse cached provider for query embeddings with VOYAGE_API_KEY."""
    return VoyageEmbeddingProvider(configuration, input_type="query")


class RetrievalService:
    """Retrieves only from the exact active corpus selected by a supplied scope."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        provider_factory: Callable[
            [VoyageEmbeddingConfiguration], QueryEmbeddingProvider
        ]
        | None = None,
        *,
        dense_top_k: int = 10,
        expected_question_top_k: int = 10,
        lexical_top_k: int = 10,
        active_version_cache: ActiveCorpusVersionCache | None = None,
    ):
        if min(dense_top_k, expected_question_top_k, lexical_top_k) < 1:
            raise ValueError("RETRIEVAL_TOP_K_MUST_BE_POSITIVE")
        self._repo = RagRepository(conn)
        self._provider_factory = provider_factory or _cached_voyage_provider
        self._dense_top_k = dense_top_k
        self._expected_question_top_k = expected_question_top_k
        self._lexical_top_k = lexical_top_k
        self._active_version_cache = (
            active_version_cache or get_active_corpus_version_cache()
        )

    def _configuration_from_active_version(
        self, active_version: dict[str, Any]
    ) -> VoyageEmbeddingConfiguration:
        configuration = VoyageEmbeddingConfiguration(
            model=active_version["embedding_model"],
            revision=active_version["embedding_revision"],
            dimensions=active_version["embedding_dim"],
            normalize=bool(active_version["normalize_embeddings"]),
        )
        stored_fingerprint = str(
            active_version.get("embedding_config_fingerprint") or ""
        )
        if stored_fingerprint != configuration.fingerprint():
            raise RetrievalConfigurationError("ACTIVE_CORPUS_CONFIGURATION_MISMATCH")
        return configuration

    async def embed_live_query(
        self, question: str, scope: RetrievalScope
    ) -> list[float] | None:
        """Generates a single 512-dim live query vector on Railway using VOYAGE_API_KEY."""
        normalized = " ".join(question.split())
        if not normalized:
            raise ValueError("RETRIEVAL_QUESTION_BLANK")
        active_version = await self._active_version_cache.get(
            scope.board_id, scope.class_id, scope.subject_id
        )
        if active_version is None:
            active_version = await self._repo.get_active_corpus_version(
                scope.board_id, scope.class_id, scope.subject_id
            )
        if active_version is None:
            return None
        configuration = self._configuration_from_active_version(active_version)
        provider = self._provider_factory(configuration)
        if provider.configuration_fingerprint != configuration.fingerprint():
            raise RetrievalConfigurationError("QUERY_PROVIDER_CONFIGURATION_MISMATCH")
        if asyncio.iscoroutinefunction(provider.embed_queries):
            vectors = await provider.embed_queries([normalized])
        else:
            vectors = await asyncio.to_thread(provider.embed_queries, [normalized])
        if len(vectors) != 1 or len(vectors[0]) != configuration.dimensions:
            raise RetrievalConfigurationError("QUERY_EMBEDDING_DIMENSION_MISMATCH")
        return vectors[0]

    async def retrieve(
        self,
        question: str,
        scope: RetrievalScope,
        query_vector: list[float] | None = None,
    ) -> EvidenceResult:
        """Run all three channels without logging question text or provider output."""
        active_version = await self._active_version_cache.get(
            scope.board_id, scope.class_id, scope.subject_id
        )
        if active_version is None:
            active_version = await self._repo.get_active_corpus_version(
                scope.board_id, scope.class_id, scope.subject_id
            )
            if active_version is not None:
                await self._active_version_cache.set(
                    scope.board_id,
                    scope.class_id,
                    scope.subject_id,
                    active_version,
                )
        if active_version is None:
            return classify_evidence(())
        return await self._retrieve_version(
            question,
            scope,
            active_version,
            query_vector=query_vector,
            allow_named_draft=False,
        )

    async def retrieve_named_version(
        self, question: str, scope: RetrievalScope, corpus_version_id: str
    ) -> EvidenceResult:
        """Local QA only: named building/qa-ready snapshot; never changes active resolution."""
        version = await self._repo.get_corpus_version(corpus_version_id)
        if not version or version["status"] not in {"building", "qa_ready"}:
            raise RetrievalScopeError("QA_CORPUS_VERSION_NOT_ELIGIBLE")
        scoped = await self._repo.conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM rag_corpus_versions cv JOIN rag_corpora c ON c.id=cv.corpus_id
               WHERE cv.id=$1::uuid AND c.board_id=$2 AND c.class_id=$3 AND c.subject_id=$4)""",
            corpus_version_id,
            scope.board_id,
            scope.class_id,
            scope.subject_id,
        )
        if not scoped:
            raise RetrievalScopeError("QA_CORPUS_VERSION_OUTSIDE_SCOPE")
        return await self._retrieve_version(
            question, scope, version, query_vector=None, allow_named_draft=True
        )

    async def _retrieve_version(
        self,
        question: str,
        scope: RetrievalScope,
        active_version: dict[str, Any],
        *,
        query_vector: list[float] | None = None,
        allow_named_draft: bool,
    ) -> EvidenceResult:
        normalized_question = " ".join(question.split())
        if not normalized_question:
            raise ValueError("RETRIEVAL_QUESTION_BLANK")
        corpus_version_id = str(active_version["id"])
        chapter_exists = (
            await self._repo.active_chapter_exists(
                scope.board_id,
                scope.class_id,
                scope.subject_id,
                corpus_version_id,
                scope.chapter_id,
            )
            if scope.chapter_id is not None and not allow_named_draft
            else (
                await self._repo.conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM rag_chunks WHERE corpus_version_id=$1::uuid AND chapter_id=$2)",
                    corpus_version_id,
                    scope.chapter_id,
                )
                if scope.chapter_id is not None
                else True
            )
        )
        if not chapter_exists:
            raise RetrievalScopeError("CHAPTER_NOT_IN_ACTIVE_CORPUS")

        configuration = self._configuration_from_active_version(active_version)
        if query_vector is None:
            provider = self._provider_factory(configuration)
            if provider.configuration_fingerprint != configuration.fingerprint():
                raise RetrievalConfigurationError(
                    "QUERY_PROVIDER_CONFIGURATION_MISMATCH"
                )
            if asyncio.iscoroutinefunction(provider.embed_queries):
                vectors = await provider.embed_queries([normalized_question])
            else:
                vectors = await asyncio.to_thread(
                    provider.embed_queries, [normalized_question]
                )
            if len(vectors) != 1 or len(vectors[0]) != configuration.dimensions:
                raise RetrievalConfigurationError("QUERY_EMBEDDING_DIMENSION_MISMATCH")
            query_vector = vectors[0]
        elif len(query_vector) != configuration.dimensions:
            raise RetrievalConfigurationError("QUERY_EMBEDDING_DIMENSION_MISMATCH")

        dense_rows = await self._repo.search_active_chunks_cosine(
            scope.board_id,
            scope.class_id,
            scope.subject_id,
            corpus_version_id,
            query_vector,
            scope.chapter_id,
            self._dense_top_k,
            allow_named_draft,
        )
        expected_rows = await self._repo.search_active_expected_questions_cosine(
            scope.board_id,
            scope.class_id,
            scope.subject_id,
            corpus_version_id,
            query_vector,
            scope.chapter_id,
            self._expected_question_top_k,
            allow_named_draft,
        )
        lexical_rows = await self._repo.search_active_chunks_lexical(
            scope.board_id,
            scope.class_id,
            scope.subject_id,
            corpus_version_id,
            normalized_question,
            scope.chapter_id,
            self._lexical_top_k,
            allow_named_draft,
        )

        row_lookup: dict[str, dict[str, Any]] = {}
        for row in (*dense_rows, *expected_rows, *lexical_rows):
            row_lookup.setdefault(row["citation_id"], row)

        visual_map = await self._repo.get_eligible_retrieval_visuals(
            list(row_lookup.keys())
        )
        citations: dict[str, Citation] = {}
        for citation_id, row in row_lookup.items():
            citations[citation_id] = Citation(
                citation_id=citation_id,
                content=row["content"],
                chapter_id=row["chapter_id"],
                topic_no=row["topic_no"],
                topic_title=row["topic_title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                visuals=tuple(
                    RetrievedVisual(
                        visual_id=visual["visual_id"],
                        title=visual["title"],
                        description=visual["description"],
                        display_policy=visual["display_policy"],
                    )
                    for visual in visual_map.get(citation_id, [])
                ),
            )

        dense_hits = [
            RankedChannelHit(
                citation=citations[row["citation_id"]],
                channel=RetrievalChannel.DENSE,
                rank=idx + 1,
            )
            for idx, row in enumerate(dense_rows)
        ]
        expected_hits = [
            RankedChannelHit(
                citation=citations[row["citation_id"]],
                channel=RetrievalChannel.EXPECTED_QUESTION,
                rank=int(row["expected_question_rank"]),
            )
            for row in expected_rows
        ]
        lexical_hits = [
            RankedChannelHit(
                citation=citations[row["citation_id"]],
                channel=RetrievalChannel.LEXICAL,
                rank=idx + 1,
            )
            for idx, row in enumerate(lexical_rows)
        ]

        fused = fuse_ranked_hits((*dense_hits, *expected_hits, *lexical_hits))
        return classify_evidence(fused)
