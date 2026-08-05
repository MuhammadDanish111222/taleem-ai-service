"""Strict Module 4 Ask contracts shared by the internal API and services."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnswerMode(StrEnum):
    SHORT = "short"
    LONG = "long"
    MCQ = "mcq"


class AnswerStyle(StrEnum):
    EXAM_STYLE = "exam_style"


class AnswerSource(StrEnum):
    APPROVED_BANK = "approved_bank"
    SYLLABUS_GROUNDED = "syllabus_grounded"
    GENERAL_KNOWLEDGE = "general_knowledge"


class TerminalStatus(StrEnum):
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"


class AskRequest(StrictModel):
    request_id: UUID
    board_id: str = Field(min_length=1, max_length=120)
    class_id: str = Field(min_length=1, max_length=120)
    subject_id: str = Field(min_length=1, max_length=120)
    chapter_id: str | None = Field(default=None, min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=2000)
    answer_mode: Literal[AnswerMode.SHORT, AnswerMode.LONG]
    answer_style: Literal[AnswerStyle.EXAM_STYLE]

    @field_validator("question")
    @classmethod
    def reject_embedded_file_payloads(cls, value: str) -> str:
        lowered = value.lower()
        forbidden = (
            "data:image/",
            "data:application/pdf",
            "base64,",
            "%pdf-",
        )
        if any(marker in lowered for marker in forbidden):
            raise ValueError("ASK_TEXT_ONLY")
        return value


class UsageRequest(StrictModel):
    feature: Literal["single_question"] = "single_question"


class ParagraphBlock(StrictModel):
    type: Literal["paragraph"]
    text: str = Field(min_length=1, max_length=12000)


class HeadingBlock(StrictModel):
    type: Literal["heading"]
    text: str = Field(min_length=1, max_length=300)
    level: Literal[2, 3]


class BulletListBlock(StrictModel):
    type: Literal["bullet_list"]
    items: list[str] = Field(min_length=1, max_length=40)

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 2000 for item in value):
            raise ValueError("ANSWER_BULLET_ITEM_INVALID")
        return value


class EquationBlock(StrictModel):
    type: Literal["equation"]
    latex: str = Field(min_length=1, max_length=4000)


class VisualRefBlock(StrictModel):
    type: Literal["visual_ref"]
    visual_id: str = Field(min_length=1, max_length=160)


AnswerBlock = Annotated[
    Union[
        ParagraphBlock,
        HeadingBlock,
        BulletListBlock,
        EquationBlock,
        VisualRefBlock,
    ],
    Field(discriminator="type"),
]


class CitationDto(StrictModel):
    citation_id: str = Field(min_length=1, max_length=160)
    chapter_id: str | None = Field(default=None, max_length=120)
    topic_no: str | None = Field(default=None, max_length=120)
    topic_title: str | None = Field(default=None, max_length=300)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class VisualDto(StrictModel):
    visual_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1, max_length=4000)
    display_policy: Literal["always", "llm_decide"]
    display_order: int = Field(ge=0)


class UsageDto(StrictModel):
    feature: Literal["single_question"] = "single_question"
    used: int = Field(ge=0)
    limit: int | None = Field(default=None, ge=0)
    remaining: int | None = Field(default=None, ge=0)
    resets_at: datetime


class AskResponse(StrictModel):
    request_id: UUID
    answer_source: AnswerSource | None
    answer_mode: Literal[AnswerMode.SHORT, AnswerMode.LONG]
    answer_style: Literal[AnswerStyle.EXAM_STYLE]
    blocks: list[AnswerBlock] = Field(default_factory=list, max_length=120)
    citations: list[CitationDto] = Field(default_factory=list, max_length=20)
    visuals: list[VisualDto] = Field(default_factory=list, max_length=20)
    general_ai_label: str | None = Field(default=None, max_length=160)
    prompt_version: str | None = Field(default=None, max_length=120)
    corpus_version: str | None = Field(default=None, max_length=120)
    approved_revision_id: UUID | None = None
    usage: UsageDto
    terminal_status: TerminalStatus
    error_code: str | None = Field(default=None, max_length=120)


GENERAL_AI_LABEL = "General AI answer — not verified from your selected textbook."
