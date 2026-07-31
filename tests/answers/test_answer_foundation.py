from datetime import datetime, timezone

import pytest

from app.schemas.ask import (
    AnswerSource,
    CitationDto,
    EquationBlock,
    ParagraphBlock,
    VisualDto,
    VisualRefBlock,
)
from app.services.answers.context import assemble_context
from app.services.answers.normalization import normalize_question, question_hash
from app.services.answers.validation import (
    AnswerValidationError,
    validate_generated_answer,
)
from app.services.retrieval.evidence import (
    ChannelContribution,
    Citation,
    RetrievalChannel,
    RetrievedEvidence,
)
from app.services.usage.models import pakistan_business_window


def test_question_normalization_is_deterministic_and_versionable():
    first = normalize_question("  What’s  Acceleration?! ")
    second = normalize_question("WHAT S acceleration")
    assert first == second == "what s acceleration"
    assert len(question_hash(first)) == 64


def test_pakistan_midnight_window_crosses_utc_date_boundary():
    window = pakistan_business_window(
        datetime(2026, 7, 30, 18, 59, 59, tzinfo=timezone.utc)
    )
    assert window.business_date.isoformat() == "2026-07-30"
    assert window.resets_at.isoformat() == "2026-07-30T19:00:00+00:00"
    assert window.ttl_seconds == 1

    next_window = pakistan_business_window(
        datetime(2026, 7, 30, 19, 0, 0, tzinfo=timezone.utc)
    )
    assert next_window.business_date.isoformat() == "2026-07-31"
    assert next_window.resets_at.isoformat() == "2026-07-31T19:00:00+00:00"


def test_context_uses_four_unique_parents_and_character_budget():
    results = tuple(
        RetrievedEvidence(
            citation=Citation(
                citation_id=f"chunk-{index}",
                content="x" * 10,
                chapter_id="one",
                topic_no=None,
                topic_title=None,
                page_start=None,
                page_end=None,
            ),
            fused_rank=index,
            contributions=(ChannelContribution(RetrievalChannel.DENSE, index),),
        )
        for index in range(1, 7)
    )
    context = assemble_context(results, max_chunks=4, max_characters=25)
    assert [item.citation_id for item in context] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert sum(len(item.content) for item in context) == 25


def test_mixed_invalid_citation_rejects_entire_answer():
    allowed = {
        "chunk-1": CitationDto(citation_id="chunk-1"),
    }
    with pytest.raises(AnswerValidationError, match="ANSWER_CITATION_NOT_ALLOWED"):
        validate_generated_answer(
            source=AnswerSource.SYLLABUS_GROUNDED,
            blocks=[ParagraphBlock(type="paragraph", text="Answer")],
            citation_ids=["chunk-1", "invented"],
            allowed_citations=allowed,
            allowed_visuals={},
        )


def test_mixed_invalid_visual_rejects_entire_answer():
    allowed = {
        "Visual_1": VisualDto(
            visual_id="Visual_1",
            title="Diagram",
            description="Reviewed",
            display_policy="llm_decide",
            display_order=0,
        )
    }
    with pytest.raises(AnswerValidationError, match="ANSWER_VISUAL_NOT_ALLOWED"):
        validate_generated_answer(
            source=AnswerSource.SYLLABUS_GROUNDED,
            blocks=[
                VisualRefBlock(type="visual_ref", visual_id="Visual_1"),
                VisualRefBlock(type="visual_ref", visual_id="invented"),
            ],
            citation_ids=[],
            allowed_citations={},
            allowed_visuals=allowed,
        )


def test_general_answer_forbids_citations_and_visuals():
    with pytest.raises(
        AnswerValidationError, match="GENERAL_ANSWER_HAS_TEXTBOOK_REFERENCES"
    ):
        validate_generated_answer(
            source=AnswerSource.GENERAL_KNOWLEDGE,
            blocks=[VisualRefBlock(type="visual_ref", visual_id="Visual_1")],
            citation_ids=["chunk-1"],
            allowed_citations={"chunk-1": CitationDto(citation_id="chunk-1")},
            allowed_visuals={
                "Visual_1": VisualDto(
                    visual_id="Visual_1",
                    title="Diagram",
                    description="Reviewed",
                    display_policy="llm_decide",
                    display_order=0,
                )
            },
        )


def test_always_visual_is_backend_guaranteed_and_unsafe_latex_rejected():
    visual = VisualDto(
        visual_id="Visual_1",
        title="Diagram",
        description="Reviewed",
        display_policy="always",
        display_order=2,
    )
    result = validate_generated_answer(
        source=AnswerSource.SYLLABUS_GROUNDED,
        blocks=[ParagraphBlock(type="paragraph", text="Answer")],
        citation_ids=[],
        allowed_citations={},
        allowed_visuals={visual.visual_id: visual},
    )
    assert result.blocks[-1] == VisualRefBlock(type="visual_ref", visual_id="Visual_1")

    with pytest.raises(AnswerValidationError, match="ANSWER_EQUATION_UNSAFE"):
        validate_generated_answer(
            source=AnswerSource.GENERAL_KNOWLEDGE,
            blocks=[EquationBlock(type="equation", latex=r"\input{secret}")],
            citation_ids=[],
            allowed_citations={},
            allowed_visuals={},
        )
