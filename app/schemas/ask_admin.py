"""Strict local-admin operation DTOs for Module 4 Run 2 interfaces."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.ask import (
    AnswerBlock,
    AnswerMode,
    AnswerStyle,
    EquationBlock,
    StrictModel,
    VisualRefBlock,
)


class McqOptionInput(StrictModel):
    key: str = Field(min_length=1, max_length=20)
    text: str = Field(min_length=1, max_length=1000)
    is_correct: bool = False


class ApprovedQuestionInput(StrictModel):
    board_id: str = Field(min_length=1, max_length=120)
    class_id: str = Field(min_length=1, max_length=120)
    subject_id: str = Field(min_length=1, max_length=120)
    chapter_id: str | None = Field(default=None, min_length=1, max_length=120)
    answer_mode: AnswerMode
    answer_style: Literal[AnswerStyle.EXAM_STYLE] = AnswerStyle.EXAM_STYLE
    difficulty: Literal["easy", "medium", "hard"]
    marks: float = Field(gt=0, le=1000)
    question: str = Field(min_length=1, max_length=4000)
    blocks: list[AnswerBlock] = Field(min_length=1, max_length=120)
    mcq_options: list[McqOptionInput] = Field(default_factory=list, max_length=12)
    citation_ids: list[UUID] = Field(default_factory=list, max_length=20)
    visual_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_mcq(self):
        if self.answer_mode == AnswerMode.MCQ:
            if len(self.mcq_options) < 2:
                raise ValueError("MCQ_OPTIONS_REQUIRED")
            if sum(item.is_correct for item in self.mcq_options) != 1:
                raise ValueError("MCQ_ONE_CORRECT_OPTION_REQUIRED")
        elif self.mcq_options:
            raise ValueError("MCQ_OPTIONS_FORBIDDEN")
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("CITATION_LINK_DUPLICATE")
        if len(self.visual_ids) != len(set(self.visual_ids)):
            raise ValueError("VISUAL_LINK_DUPLICATE")
        visual_refs = [
            item.visual_id for item in self.blocks if isinstance(item, VisualRefBlock)
        ]
        if len(visual_refs) != len(set(visual_refs)):
            raise ValueError("VISUAL_BLOCK_DUPLICATE")
        if set(visual_refs) != set(self.visual_ids):
            raise ValueError("VISUAL_BLOCK_LINK_MISMATCH")
        if any(
            isinstance(item, EquationBlock)
            and any(
                command in item.latex.lower()
                for command in (
                    "\\input",
                    "\\include",
                    "\\write18",
                    "\\openout",
                    "\\usepackage",
                    "\\href",
                    "\\url",
                )
            )
            for item in self.blocks
        ):
            raise ValueError("ANSWER_EQUATION_UNSAFE")
        return self


class AskAdminRequest(StrictModel):
    operation: Literal[
        "prompt_history",
        "prompt_create_draft",
        "prompt_update_draft",
        "prompt_test_draft",
        "prompt_activate",
        "prompt_rollback",
        "candidate_list",
        "candidate_inspect",
        "candidate_approve",
        "candidate_reject",
        "candidate_retention_preview",
        "candidate_retention_cleanup",
        "bank_list",
        "bank_create",
        "bank_import",
        "bank_view",
        "bank_history",
        "bank_archive",
        "bank_add_variation",
        "bank_set_variation_active",
        "bank_requeue_embedding",
        "bank_set_visuals",
    ]
    prompt_id: UUID | None = None
    prompt_key: Literal["ask_grounded", "ask_general"] | None = None
    answer_mode: AnswerMode | None = None
    board_id: str | None = Field(default=None, max_length=120)
    class_id: str | None = Field(default=None, max_length=120)
    subject_id: str | None = Field(default=None, max_length=120)
    chapter_id: str | None = Field(default=None, max_length=120)
    content: str | None = Field(default=None, max_length=20000)
    question: str | None = Field(default=None, max_length=4000)
    candidate_id: UUID | None = None
    question_id: UUID | None = None
    revision_id: UUID | None = None
    variation_id: UUID | None = None
    rejection_reason: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, max_length=1000)
    variation: str | None = Field(default=None, max_length=4000)
    active: bool | None = None
    source_feature: Literal["single_question", "multiple_question"] | None = None
    answer_source: (
        Literal["approved_bank", "syllabus_grounded", "general_knowledge"] | None
    ) = None
    provider: str | None = Field(default=None, max_length=120)
    bank_source: str | None = Field(default=None, max_length=120)
    age_days: int | None = Field(default=None, ge=0, le=3650)
    visual_ids: list[str] = Field(default_factory=list, max_length=20)
    approved_question: ApprovedQuestionInput | None = None
    import_key: str | None = Field(default=None, max_length=200)
    import_questions: list[ApprovedQuestionInput] = Field(
        default_factory=list, max_length=500
    )
    limit: int = Field(default=50, ge=1, le=100)
