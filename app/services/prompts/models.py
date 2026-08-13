"""Immutable prompt-domain records and value objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PromptKey(StrEnum):
    ASK_GROUNDED = "ask_grounded"
    ASK_GENERAL = "ask_general"


class AnswerMode(StrEnum):
    SHORT = "short"
    LONG = "long"
    MCQ = "mcq"


class PromptStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    RETIRED = "retired"


class PromptConfigurationError(LookupError):
    """No active prompt is configured for a student's required prompt family."""

    def __init__(
        self, *, prompt_key: PromptKey, answer_mode: AnswerMode, scope: "PromptScope"
    ) -> None:
        self.prompt_key = prompt_key
        self.answer_mode = answer_mode
        self.scope = scope
        super().__init__(
            "PROMPT_CONFIGURATION_MISSING:"
            f"{prompt_key.value}:{answer_mode.value}:"
            f"{scope.board_id}/{scope.class_id}/{scope.subject_id}"
        )


@dataclass(frozen=True, slots=True)
class PromptScope:
    """A persisted prompt scope; student resolution uses exact then subject-global."""

    board_id: str | None = None
    class_id: str | None = None
    subject_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.board_id, self.class_id, self.subject_id):
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError("PROMPT_SCOPE_VALUE_INVALID")
        if self.board_id is not None and (
            self.class_id is None or self.subject_id is None
        ):
            raise ValueError("PROMPT_SCOPE_BOARD_REQUIRES_CLASS_AND_SUBJECT")
        if self.class_id is not None and self.subject_id is None:
            raise ValueError("PROMPT_SCOPE_CLASS_REQUIRES_SUBJECT")

    def resolution_chain(self) -> tuple[PromptScope, ...]:
        """Return the only permitted student resolution order.

        Legacy class/subject and application-global rows remain representable for
        history, but are deliberately never candidates for student traffic.
        """

        if self.board_id is None or self.class_id is None or self.subject_id is None:
            raise ValueError("PROMPT_RESOLUTION_REQUIRES_FULL_STUDENT_SCOPE")
        return (self, PromptScope(subject_id=self.subject_id))


@dataclass(frozen=True, slots=True)
class PromptRecord:
    id: str
    prompt_key: PromptKey
    answer_mode: AnswerMode
    scope: PromptScope
    version: int
    content: str
    status: PromptStatus
    created_by: str
    created_at: datetime
    activated_by: str | None = None
    activated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.id or self.version < 1 or not self.created_by:
            raise ValueError("PROMPT_RECORD_INVALID")
        if type(self.content) is not str or not self.content.strip():
            raise ValueError("PROMPT_CONTENT_REQUIRED")


@dataclass(frozen=True, slots=True)
class PromptActivation:
    active: PromptRecord
    retired: PromptRecord | None


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    record: PromptRecord
    system_prompt: str
