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


DEFAULT_IMPORT_MARKS: dict[str, float] = {"mcq": 1, "short": 2, "long": 4}


class BlueprintSectionInput(StrictModel):
    key: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    title: str = Field(min_length=1, max_length=160)
    type: Literal["mcq", "short", "long"]
    select_count: int = Field(ge=1, le=100)
    attempt_count: int = Field(ge=1, le=100)
    marks_each: float = Field(gt=0, le=1000)
    difficulty_distribution: dict[Literal["easy", "medium", "hard"], int] = Field(default_factory=dict)
    chapter_distribution: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_quotas(self):
        if self.attempt_count > self.select_count:
            raise ValueError("BLUEPRINT_ATTEMPT_COUNT_INVALID")
        for chapter_id, count in self.chapter_distribution.items():
            if not chapter_id or len(chapter_id) > 120 or count < 1:
                raise ValueError("BLUEPRINT_CHAPTER_INVALID")
        if any(count < 1 for count in self.difficulty_distribution.values()):
            raise ValueError("BLUEPRINT_DIFFICULTY_INVALID")
        if self.difficulty_distribution and sum(self.difficulty_distribution.values()) != self.select_count:
            raise ValueError("BLUEPRINT_DIFFICULTY_TOTAL_INVALID")
        if self.chapter_distribution and sum(self.chapter_distribution.values()) != self.select_count:
            raise ValueError("BLUEPRINT_CHAPTER_TOTAL_INVALID")
        return self


class BoardPaperBlueprintInput(StrictModel):
    duration_minutes: int = Field(ge=1, le=600)
    sections: list[BlueprintSectionInput] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_section_keys(self):
        keys = [section.key for section in self.sections]
        if len(keys) != len(set(keys)):
            raise ValueError("BLUEPRINT_SECTION_KEY_DUPLICATE")
        return self


class BulkImportQuestionInput(StrictModel):
    """Human-friendly local-admin JSON input, normalized into the bank contract."""

    question: str = Field(min_length=1, max_length=4000)
    type: Literal["mcq", "short", "long"]
    difficulty: Literal["easy", "medium", "hard"]
    marks: float | None = Field(default=None, gt=0, le=1000)
    options: list[str] = Field(default_factory=list, max_length=12)
    correct_answer: str | None = Field(default=None, max_length=1000)
    answer_blocks: list[AnswerBlock] = Field(default_factory=list, max_length=120)
    visual_ids: list[str] = Field(default_factory=list, max_length=20)

    @property
    def resolved_marks(self) -> float:
        return self.marks if self.marks is not None else DEFAULT_IMPORT_MARKS[self.type]

    @model_validator(mode="after")
    def validate_question(self):
        if len(self.visual_ids) != len(set(self.visual_ids)):
            raise ValueError("VISUAL_LINK_DUPLICATE")
        visual_refs = [
            item.visual_id
            for item in self.answer_blocks
            if isinstance(item, VisualRefBlock)
        ]
        if len(visual_refs) != len(set(visual_refs)):
            raise ValueError("VISUAL_BLOCK_DUPLICATE")
        if set(visual_refs) - set(self.visual_ids):
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
            for item in self.answer_blocks
        ):
            raise ValueError("ANSWER_EQUATION_UNSAFE")

        if self.type == "mcq":
            if len(self.options) < 2 or any(not item.strip() for item in self.options):
                raise ValueError("MCQ_OPTIONS_REQUIRED")
            if len(set(self.options)) != len(self.options):
                raise ValueError("MCQ_OPTIONS_DUPLICATE")
            if self.correct_answer is None or self.correct_answer not in self.options:
                raise ValueError("MCQ_CORRECT_ANSWER_INVALID")
        else:
            if not self.answer_blocks:
                raise ValueError("ANSWER_BLOCKS_REQUIRED")
            if self.options or self.correct_answer is not None:
                raise ValueError("MCQ_FIELDS_FORBIDDEN")
        return self

    def as_approved_question(
        self,
        *,
        board_id: str,
        class_id: str,
        subject_id: str,
        chapter_id: str,
    ) -> ApprovedQuestionInput:
        blocks = [item.model_dump() for item in self.answer_blocks]
        block_visual_ids = {
            item["visual_id"] for item in blocks if item["type"] == "visual_ref"
        }
        # The answer renderer uses visual_ref blocks.  Keep JSON concise by
        # adding canonical references for any selected existing visuals.
        blocks.extend(
            {"type": "visual_ref", "visual_id": visual_id}
            for visual_id in self.visual_ids
            if visual_id not in block_visual_ids
        )
        if self.type == "mcq" and not blocks:
            # MCQs retain their correct answer in the same block format used by
            # existing renderer/reuse code; choices live in the MCQ table.
            blocks = [{"type": "paragraph", "text": self.correct_answer}]
        return ApprovedQuestionInput(
            board_id=board_id,
            class_id=class_id,
            subject_id=subject_id,
            chapter_id=chapter_id,
            answer_mode=AnswerMode(self.type),
            answer_style=AnswerStyle.EXAM_STYLE,
            difficulty=self.difficulty,
            marks=self.resolved_marks,
            question=self.question,
            blocks=blocks,
            mcq_options=[
                McqOptionInput(
                    key=str(index + 1),
                    text=option,
                    is_correct=option == self.correct_answer,
                )
                for index, option in enumerate(self.options)
            ],
            visual_ids=self.visual_ids,
        )


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
        "source_policy_get",
        "source_policy_set_semantic_threshold",
        "blueprint_get",
        "blueprint_preview",
        "blueprint_save",
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
    semantic_similarity_threshold: float | None = Field(default=None, ge=0.80, le=0.99)
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
    import_questions: list[BulkImportQuestionInput] = Field(
        default_factory=list, max_length=500
    )
    blueprint_name: str | None = Field(default=None, min_length=1, max_length=160)
    blueprint: BoardPaperBlueprintInput | None = None
    blueprint_active: bool | None = None
    selection_seed: str | None = Field(default=None, min_length=1, max_length=200)
    limit: int = Field(default=50, ge=1, le=100)

    @model_validator(mode="after")
    def validate_prompt_management_scope(self):
        if self.operation not in {"prompt_history", "prompt_create_draft"}:
            return self
        if self.prompt_key is None or self.answer_mode is None:
            return self
        if self.answer_mode == AnswerMode.MCQ:
            raise ValueError("PROMPT_MCQ_CONFIGURATION_UNSUPPORTED")
        exact_scope = bool(self.board_id and self.class_id and self.subject_id)
        subject_global_scope = bool(
            self.subject_id and self.board_id is None and self.class_id is None
        )
        if not (exact_scope or subject_global_scope):
            raise ValueError("PROMPT_SCOPE_REQUIRES_EXACT_OR_SUBJECT_GLOBAL")
        return self

    @model_validator(mode="after")
    def validate_semantic_threshold_scope(self):
        if self.operation not in {
            "source_policy_get",
            "source_policy_set_semantic_threshold",
        }:
            return self
        if not self.subject_id:
            raise ValueError("SEMANTIC_THRESHOLD_SUBJECT_REQUIRED")
        if self.operation == "source_policy_set_semantic_threshold" and (
            self.semantic_similarity_threshold is None
        ):
            raise ValueError("SEMANTIC_THRESHOLD_REQUIRED")
        return self

    @model_validator(mode="after")
    def validate_bank_import_scope(self):
        if self.operation != "bank_import":
            return self
        if not all((self.board_id, self.class_id, self.subject_id, self.chapter_id)):
            raise ValueError("IMPORT_SCOPE_REQUIRES_BOARD_CLASS_SUBJECT_CHAPTER")
        return self

    @model_validator(mode="after")
    def validate_blueprint_request(self):
        if self.operation not in {"blueprint_get", "blueprint_preview", "blueprint_save"}:
            return self
        if not all((self.board_id, self.class_id, self.subject_id)):
            raise ValueError("BLUEPRINT_SCOPE_REQUIRED")
        if self.operation in {"blueprint_preview", "blueprint_save"} and self.blueprint is None:
            raise ValueError("BLUEPRINT_REQUIRED")
        if self.operation == "blueprint_save" and (
            self.blueprint_name is None or self.blueprint_active is None
        ):
            raise ValueError("BLUEPRINT_SAVE_FIELDS_REQUIRED")
        return self
