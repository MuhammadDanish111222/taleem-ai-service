from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import asyncpg
import pytest

from app.providers.embeddings.bge import BGEEmbeddingConfiguration
from app.providers.llm.deepseek import (
    DeepSeekProviderError,
    ProviderErrorCode,
    StructuredGeneration,
    TokenUsage,
)
from app.repositories.ask_repository import AskRepository
from app.repositories.question_bank_repository import QuestionBankRepository
from app.repositories.rag_repository import RagRepository
from app.schemas.ask import (
    AnswerMode,
    AnswerSource,
    AnswerStyle,
    AskRequest,
    TerminalStatus,
)
from app.services.answers.generate import AskService, AskServiceError
from app.services.answers.normalization import normalize_question, question_hash
from app.services.prompts.models import (
    PromptKey,
    PromptRecord,
    PromptStatus,
    ResolvedPrompt,
)
from app.services.retrieval.evidence import (
    ChannelContribution,
    Citation,
    EvidenceResult,
    EvidenceStrength,
    RetrievalChannel,
    RetrievedEvidence,
)
from app.services.usage.models import (
    AccountTier,
    BusinessWindow,
    UsageReservation,
)

DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/taleem_dev",
)


@pytest.fixture
async def conn():
    connection = await asyncpg.connect(DB_URL)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


class FakeUsage:
    def __init__(self):
        self.reserved = 0
        self.committed = 0
        self.refunded = 0

    @staticmethod
    def uid_hash(_uid):
        return "b" * 64

    def _value(self, request_id=""):
        return UsageReservation(
            request_id=request_id,
            used=1,
            limit=5,
            student_visible=True,
            window=BusinessWindow(
                business_date=datetime(2026, 7, 30).date(),
                resets_at=datetime(2026, 7, 30, 19, tzinfo=timezone.utc),
                ttl_seconds=100,
            ),
            backend="redis",
        )

    async def reserve(self, _conn, *, request_id, **_kwargs):
        self.reserved += 1
        return self._value(request_id)

    async def snapshot(self, _conn, **_kwargs):
        return self._value()

    async def commit(self, _conn, _request_id, _uid_hash):
        self.committed += 1

    async def refund(self, _conn, _request_id, _uid_hash):
        self.refunded += 1


class FakeRetrieval:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.semantic_calls = 0

    async def retrieve(self, *_args, **_kwargs):
        self.calls += 1
        return self.result

    async def embed_query_for_approved_reuse(self, *_args, **_kwargs):
        self.semantic_calls += 1
        return [1.0] + [0.0] * 767


class FakePrompts:
    def __init__(self):
        self.keys = []

    async def resolve_active(self, *, prompt_key, answer_mode, scope):
        self.keys.append(prompt_key)
        return ResolvedPrompt(
            record=PromptRecord(
                id=str(uuid.uuid4()),
                prompt_key=prompt_key,
                answer_mode=answer_mode,
                scope=scope,
                version=3,
                content="Teach clearly.",
                status=PromptStatus.ACTIVE,
                created_by="admin",
                created_at=datetime.now(timezone.utc),
                activated_by="admin",
                activated_at=datetime.now(timezone.utc),
            ),
            system_prompt="safe prompt",
        )


class FakeProvider:
    def __init__(self, document=None, error=None):
        self.document = document or {
            "blocks": [{"type": "paragraph", "text": "Generated answer"}],
            "cited_chunk_ids": ["chunk-1"],
        }
        self.error = error
        self.calls = 0

    async def generate(self, **_kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return StructuredGeneration(
            document=self.document,
            provider="fake",
            model="fake-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency_ms=4,
            provider_request_id=None,
            finish_reason="stop",
        )


def request(question="What is force?", chapter_id="chapter-1"):
    return AskRequest(
        request_id=uuid.uuid4(),
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id=chapter_id,
        question=question,
        answer_mode="short",
        answer_style="exam_style",
    )


async def create_approved(conn, question, chapter_id):
    normalized = normalize_question(question)
    return await QuestionBankRepository(conn).create_approved_revision(
        actor_id="admin",
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id=chapter_id,
        answer_mode=AnswerMode.SHORT,
        answer_style=AnswerStyle.EXAM_STYLE,
        difficulty="easy",
        marks=2,
        question_text=question,
        normalized_question=normalized,
        question_hash=question_hash(normalized),
        blocks=[{"type": "paragraph", "text": "Approved answer"}],
        source="admin_authored",
    )


async def create_reviewed_visual(conn):
    suffix = uuid.uuid4().hex
    configuration = BGEEmbeddingConfiguration()
    rag = RagRepository(conn)
    corpus = await rag.get_or_create_corpus(
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
    )
    version = await rag.create_corpus_version(
        str(corpus["id"]),
        1000 + int(suffix[:4], 16),
        configuration.model,
        configuration.revision,
        configuration.dimensions,
        embedding_config_fingerprint=configuration.fingerprint(),
        normalize_embeddings=configuration.normalize,
        query_instruction=configuration.query_instruction,
    )
    document = await rag.create_document_version(
        str(version["id"]),
        f"approved-visual-{suffix}",
        "v1",
        "test",
        "Approved visual test",
    )
    chunks = await rag.replace_chapter_chunks(
        str(version["id"]),
        str(document["id"]),
        [
            {
                "chapter_id": "chapter-1",
                "topic_no": "1",
                "topic_title": "Force",
                "chunk_order": 0,
                "chunk_text": "Force is a push or pull.",
                "content_type": "explanation",
                "content_hash": suffix,
                "language": "en",
                "token_count": 6,
                "metadata": {},
                "expected_questions": [],
                "visuals": [
                    {
                        "visual_id": f"force-visual-{suffix}",
                        "visual_type": "diagram",
                        "title": "Force diagram",
                        "description": "Reviewed force arrows",
                        "storage_key": f"drive-{suffix}",
                    }
                ],
            }
        ],
    )
    visual = await conn.fetchrow(
        """UPDATE rag_visuals SET review_status='approved'
           WHERE chunk_id=$1::uuid RETURNING id,visual_id""",
        chunks[0]["id"],
    )
    return visual


@pytest.mark.asyncio
async def test_exact_approved_match_bypasses_retrieval_and_provider(conn):
    revision_id = await create_approved(conn, "What is force?", "chapter-1")
    retrieval = FakeRetrieval(
        EvidenceResult(EvidenceStrength.NONE, (), "NO_SCOPED_EVIDENCE")
    )
    provider = FakeProvider()
    usage = FakeUsage()
    service = AskService(
        conn,
        retrieval=retrieval,
        provider=provider,
        usage=usage,
        prompt_service=FakePrompts(),
    )
    ask_request = request()
    result = await service.ask(ask_request, uid="student", tier=AccountTier.ANONYMOUS)
    assert result.answer_source == AnswerSource.APPROVED_BANK
    assert str(result.approved_revision_id) == revision_id
    assert retrieval.calls == provider.calls == 0
    assert usage.committed == 1
    duplicate = await service.ask(
        ask_request, uid="student", tier=AccountTier.ANONYMOUS
    )
    assert duplicate.blocks == result.blocks
    assert usage.reserved == 1
    assert retrieval.calls == provider.calls == 0


@pytest.mark.asyncio
async def test_approved_visual_is_rehydrated_on_idempotent_replay(conn):
    visual = await create_reviewed_visual(conn)
    question = "Illustrate force."
    normalized = normalize_question(question)
    revision_id = await QuestionBankRepository(conn).create_approved_revision(
        actor_id="admin",
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="chapter-1",
        answer_mode=AnswerMode.SHORT,
        answer_style=AnswerStyle.EXAM_STYLE,
        difficulty="easy",
        marks=2,
        question_text=question,
        normalized_question=normalized,
        question_hash=question_hash(normalized),
        blocks=[
            {"type": "paragraph", "text": "Approved answer"},
            {"type": "visual_ref", "visual_id": visual["visual_id"]},
        ],
        source="admin_authored",
        visual_row_ids=[str(visual["id"])],
    )
    service = AskService(
        conn,
        retrieval=FakeRetrieval(
            EvidenceResult(EvidenceStrength.NONE, (), "NO_SCOPED_EVIDENCE")
        ),
        provider=FakeProvider(),
        usage=FakeUsage(),
        prompt_service=FakePrompts(),
    )
    ask_request = request(question)

    first = await service.ask(ask_request, uid="student", tier=AccountTier.ANONYMOUS)
    replay = await service.ask(ask_request, uid="student", tier=AccountTier.ANONYMOUS)

    assert str(first.approved_revision_id) == revision_id
    assert [item.visual_id for item in first.visuals] == [visual["visual_id"]]
    assert replay.visuals == first.visuals


@pytest.mark.asyncio
async def test_exact_variation_bypasses_retrieval_and_provider(conn):
    revision_id = await create_approved(conn, "Define force.", "chapter-1")
    normalized = normalize_question("What is force?")
    await QuestionBankRepository(conn).add_variation(
        revision_id=revision_id,
        variation_text="What is force?",
        normalized_variation=normalized,
        variation_hash=question_hash(normalized),
        actor_id="admin",
    )
    retrieval = FakeRetrieval(
        EvidenceResult(EvidenceStrength.NONE, (), "NO_SCOPED_EVIDENCE")
    )
    provider = FakeProvider()
    result = await AskService(
        conn,
        retrieval=retrieval,
        provider=provider,
        usage=FakeUsage(),
        prompt_service=FakePrompts(),
    ).ask(request(), uid="student", tier=AccountTier.ANONYMOUS)
    assert result.answer_source == AnswerSource.APPROVED_BANK
    assert retrieval.calls == provider.calls == 0


@pytest.mark.asyncio
async def test_semantic_reuse_is_disabled_without_evaluated_policy(conn):
    retrieval = FakeRetrieval(
        EvidenceResult(EvidenceStrength.NONE, (), "NO_SCOPED_EVIDENCE")
    )
    result = await AskService(
        conn,
        retrieval=retrieval,
        provider=FakeProvider(),
        usage=FakeUsage(),
        prompt_service=FakePrompts(),
    ).ask(
        request(question="A novel paraphrase"),
        uid="student",
        tier=AccountTier.ANONYMOUS,
    )
    assert retrieval.semantic_calls == 0
    assert retrieval.calls == 1
    assert result.error_code == "NO_ACTIVE_CORPUS"


@pytest.mark.asyncio
async def test_semantic_repository_never_reuses_unapproved_rows(conn):
    revision_id = await create_approved(conn, "Define momentum.", "chapter-1")
    vector = [1.0] + [0.0] * 767
    await conn.execute(
        """UPDATE question_bank_revisions
           SET review_status='pending',approved_by=NULL,approved_at=NULL,
               embedding=$2::text::vector,embedding_status='embedded'
           WHERE id=$1::uuid""",
        revision_id,
        str(vector),
    )
    result = await QuestionBankRepository(conn).find_semantic(
        query_embedding=vector,
        evaluated_threshold=0.25,
        enabled=True,
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="chapter-1",
        answer_mode=AnswerMode.SHORT,
    )
    assert result is None


@pytest.mark.asyncio
async def test_ambiguous_chapterless_match_continues_to_retrieval(conn):
    await create_approved(conn, "What is force?", "chapter-1")
    await create_approved(conn, "What is force?", "chapter-2")
    retrieval = FakeRetrieval(
        EvidenceResult(EvidenceStrength.NONE, (), "NO_SCOPED_EVIDENCE")
    )
    result = await AskService(
        conn,
        retrieval=retrieval,
        provider=FakeProvider(),
        usage=FakeUsage(),
        prompt_service=FakePrompts(),
    ).ask(
        request(chapter_id=None),
        uid="student",
        tier=AccountTier.ANONYMOUS,
    )
    assert retrieval.calls == 1
    assert result.terminal_status == TerminalStatus.NO_ANSWER
    assert result.error_code == "NO_ACTIVE_CORPUS"


@pytest.mark.asyncio
async def test_strong_retrieval_persists_pending_grounded_candidate(conn):
    evidence = EvidenceResult(
        EvidenceStrength.STRONG,
        (
            RetrievedEvidence(
                citation=Citation(
                    citation_id="chunk-1",
                    content="Force is a push or pull.",
                    chapter_id="chapter-1",
                    topic_no="1.1",
                    topic_title="Force",
                    page_start=4,
                    page_end=4,
                ),
                fused_rank=1,
                contributions=(
                    ChannelContribution(RetrievalChannel.DENSE, 1),
                    ChannelContribution(RetrievalChannel.LEXICAL, 1),
                ),
            ),
        ),
        "strong",
    )
    prompts = FakePrompts()
    retrieval = FakeRetrieval(evidence)
    provider = FakeProvider()
    usage = FakeUsage()
    service = AskService(
        conn,
        retrieval=retrieval,
        provider=provider,
        usage=usage,
        prompt_service=prompts,
    )
    ask_request = request()
    result = await service.ask(ask_request, uid="student", tier=AccountTier.ANONYMOUS)
    assert result.answer_source == AnswerSource.SYLLABUS_GROUNDED
    assert prompts.keys == [PromptKey.ASK_GROUNDED]
    review_status = await conn.fetchval(
        """SELECT a.review_status FROM ai_answers a
           JOIN ai_requests r ON r.id=a.request_id
           WHERE r.client_request_id=$1""",
        result.request_id,
    )
    assert review_status == "pending"
    replay = await service.ask(ask_request, uid="student", tier=AccountTier.ANONYMOUS)
    assert replay.blocks == result.blocks
    assert replay.citations == result.citations
    assert provider.calls == retrieval.calls == usage.reserved == 1


@pytest.mark.asyncio
async def test_strong_but_irrelevant_evidence_becomes_labelled_general_answer(conn):
    await conn.execute(
        """UPDATE ask_source_policies SET allow_general=TRUE
           WHERE class_id IS NULL AND subject_id IS NULL"""
    )
    evidence = EvidenceResult(
        EvidenceStrength.STRONG,
        (
            RetrievedEvidence(
                citation=Citation(
                    citation_id="chunk-1",
                    content="Matter can be solid, liquid, or gas.",
                    chapter_id="chapter-1",
                    topic_no="1.1",
                    topic_title="Matter",
                    page_start=4,
                    page_end=4,
                ),
                fused_rank=1,
                contributions=(
                    ChannelContribution(RetrievalChannel.DENSE, 1),
                    ChannelContribution(RetrievalChannel.EXPECTED_QUESTION, 1),
                ),
            ),
        ),
        "strong",
    )
    prompts = FakePrompts()
    result = await AskService(
        conn,
        retrieval=FakeRetrieval(evidence),
        provider=FakeProvider(
            {
                "blocks": [{"type": "paragraph", "text": "Tokyo is the capital."}],
                "cited_chunk_ids": [],
            }
        ),
        usage=FakeUsage(),
        prompt_service=prompts,
    ).ask(
        request(question="What is the capital of Japan?"),
        uid="student",
        tier=AccountTier.ANONYMOUS,
    )

    assert result.answer_source == AnswerSource.GENERAL_KNOWLEDGE
    assert result.general_ai_label
    assert result.citations == []
    assert result.visuals == []
    assert prompts.keys == [PromptKey.ASK_GROUNDED]


@pytest.mark.asyncio
async def test_weak_retrieval_general_fallback_is_labelled_and_reference_free(conn):
    await conn.execute(
        """UPDATE ask_source_policies SET allow_general=TRUE
           WHERE class_id IS NULL AND subject_id IS NULL"""
    )
    provider = FakeProvider(
        {
            "blocks": [{"type": "paragraph", "text": "General answer"}],
            "cited_chunk_ids": [],
        }
    )
    prompts = FakePrompts()
    result = await AskService(
        conn,
        retrieval=FakeRetrieval(EvidenceResult(EvidenceStrength.WEAK, (), "weak")),
        provider=provider,
        usage=FakeUsage(),
        prompt_service=prompts,
    ).ask(request(), uid="student", tier=AccountTier.ANONYMOUS)
    assert result.answer_source == AnswerSource.GENERAL_KNOWLEDGE
    assert result.citations == []
    assert result.visuals == []
    assert result.general_ai_label
    assert prompts.keys == [PromptKey.ASK_GENERAL]


@pytest.mark.asyncio
async def test_provider_failure_refunds_and_keeps_no_reusable_answer(conn):
    usage = FakeUsage()
    provider = FakeProvider(error=DeepSeekProviderError(ProviderErrorCode.UNAVAILABLE))
    evidence = EvidenceResult(
        EvidenceStrength.STRONG,
        (
            RetrievedEvidence(
                citation=Citation(
                    citation_id="chunk-1",
                    content="Force is a push.",
                    chapter_id="chapter-1",
                    topic_no=None,
                    topic_title=None,
                    page_start=None,
                    page_end=None,
                ),
                fused_rank=1,
                contributions=(
                    ChannelContribution(RetrievalChannel.DENSE, 1),
                    ChannelContribution(RetrievalChannel.LEXICAL, 1),
                ),
            ),
        ),
        "strong",
    )
    with pytest.raises(AskServiceError, match="provider_unavailable"):
        await AskService(
            conn,
            retrieval=FakeRetrieval(evidence),
            provider=provider,
            usage=usage,
            prompt_service=FakePrompts(),
        ).ask(request(), uid="student", tier=AccountTier.ANONYMOUS)
    assert usage.refunded == 1
    assert await conn.fetchval("SELECT COUNT(*) FROM ai_answers") == 0


@pytest.mark.asyncio
async def test_admin_authored_revision_is_immediately_approved_with_actor(conn):
    revision_id = await create_approved(conn, "State Newton's first law.", "chapter-2")
    row = await conn.fetchrow(
        """SELECT review_status,approved_by,approved_at,source
           FROM question_bank_revisions WHERE id=$1::uuid""",
        revision_id,
    )
    assert row["review_status"] == "approved"
    assert row["approved_by"] == "admin"
    assert row["approved_at"] is not None
    assert row["source"] == "admin_authored"


@pytest.mark.asyncio
async def test_candidate_approval_retains_and_links_original_candidate(conn):
    asks = AskRepository(conn)
    candidate_request = await asks.create_pending(
        client_request_id=str(uuid.uuid4()),
        uid_hash="c" * 64,
        board_id="punjab",
        class_id="class-9",
        subject_id="physics",
        chapter_id="chapter-1",
        answer_mode="short",
        answer_style="exam_style",
        raw_question="What is inertia?",
        normalized_question="what is inertia",
        question_hash=question_hash("what is inertia"),
        usage_business_date=datetime(2026, 7, 30).date(),
    )
    answer = await asks.complete(
        ai_request_id=str(candidate_request["id"]),
        answer_source="general_knowledge",
        blocks=[{"type": "paragraph", "text": "Inertia is resistance to change."}],
        citations=[],
        visual_ids=[],
        prompt_version="test:1",
        corpus_version_id=None,
        provider="fake",
        model="fake",
    )
    revision_id = await create_approved(conn, "What is inertia?", "chapter-1")
    await asks.approve_candidate(
        answer_id=str(answer["id"]),
        revision_id=revision_id,
        actor_id="admin",
    )
    linked = await conn.fetchrow(
        """SELECT review_status,approved_revision_id
           FROM ai_answers WHERE id=$1""",
        answer["id"],
    )
    assert linked["review_status"] == "approved"
    assert str(linked["approved_revision_id"]) == revision_id
    assert (
        await conn.fetchval("SELECT COUNT(*) FROM ai_answers WHERE id=$1", answer["id"])
        == 1
    )
