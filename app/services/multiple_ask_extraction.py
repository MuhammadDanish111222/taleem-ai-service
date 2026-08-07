"""Deterministic, conservative splitting of printed Multiple Ask papers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

AnswerMode = Literal["short", "long", "mcq", "not_clear"]

_ARABIC_START = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*)?(?P<label>\d{1,3})\s*[.)]\s*(?P<text>.*)$",
    re.I,
)
_DECIMAL_START = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*)?(?P<label>\d{1,3}\.\d{1,3})(?:\s*[.)])?\s*(?P<text>.*)$",
    re.I,
)
_LETTERED_START = re.compile(
    r"^\s*(?P<label>\d{1,3}\s*\(\s*[a-z]\s*\))\s*(?P<text>.*)$", re.I
)
_ROMAN_START = re.compile(
    r"^\s*(?:\(\s*)?(?P<label>i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)(?:\s*\))?\s*[.)]\s*(?P<text>.*)$",
    re.I,
)
_OPTION = re.compile(r"^\s*\(?\s*([A-Da-d])\s*\)?\s*[.)]\s*(.*\S)\s*$")
_LONG_CUES = re.compile(
    r"\b(explain|describe|discuss|compare|derive|prove|how|write\s+(?:a\s+)?detailed\s+note)\b",
    re.I,
)
_UNREADABLE_CUES = re.compile(
    r"\b(illegible|unclear|unreadable)\b|\[\s*illegible", re.I
)
_GROUP_CUES = re.compile(
    r"\b(any\s+\w+\s+parts?|following\s+parts?|short\s+answers?|attempt\s+any|answer\s+any)\b",
    re.I,
)
_SECTION = re.compile(r"^\s*(SECTION[-\s]*[IVXLC]+)\s*$", re.I)
_DECORATION = re.compile(
    r"^\s*(?:PAGE\s*\d+|TIME\s*(?:ALLOWED)?|TOTAL\s+MARKS?|PAPER\s+CODE|ROLL\s+NO)\b",
    re.I,
)
_MARKS_SUFFIX = re.compile(r"\s*(?:\(?\s*\d+\s*marks?\s*\)?|\[\s*\d+\s*\])\s*$", re.I)


class QuestionLimitExceeded(ValueError):
    def __init__(self, count: int, limit: int):
        self.count, self.limit = count, limit
        super().__init__(f"MULTIPLE_ASK_TOO_MANY_QUESTIONS:{count}>{limit}")


@dataclass(frozen=True)
class ExtractedQuestion:
    source_order: int
    display_label: str
    section_context: str | None
    question_text: str
    answer_mode: AnswerMode
    mcq_options: tuple[dict[str, str], ...]
    unclear_reason: str | None
    source_locator: dict[str, Any]


@dataclass
class _Candidate:
    page_number: int
    kind: Literal["arabic", "roman", "lettered"]
    label: str
    lines: list[str]
    section_context: str | None


def _start(line: str) -> tuple[Literal["arabic", "roman", "lettered"], str, str] | None:
    for kind, pattern in (
        ("lettered", _LETTERED_START),
        ("arabic", _DECIMAL_START),
        ("arabic", _ARABIC_START),
        ("roman", _ROMAN_START),
    ):
        match = pattern.match(line)
        if match is not None:
            return (
                kind,
                re.sub(r"\s+", "", match.group("label")),
                match.group("text").strip(),
            )
    return None


def _split_inline_starts(line: str) -> list[str]:
    """Recover common OCR joins while preserving the words that were read."""
    pieces: list[str] = []
    remainder = line
    while remainder:
        if _start(remainder) is None:
            pieces.append(remainder)
            break
        next_index: int | None = None
        for index in range(1, len(remainder)):
            if (
                remainder[index - 1].isspace()
                and not remainder[index].isspace()
                and _start(remainder[index:])
            ):
                next_index = index
                break
        if next_index is None:
            pieces.append(remainder)
            break
        pieces.append(remainder[:next_index].rstrip())
        remainder = remainder[next_index:]
    return pieces


def _is_container(candidates: list[_Candidate], index: int) -> bool:
    candidate = candidates[index]
    if candidate.kind != "arabic" or not _GROUP_CUES.search(" ".join(candidate.lines)):
        return False
    # Only the immediately following subparts belong to this container.  Do
    # not scan past the next numbered question: an OCR page can contain a
    # later, unrelated Roman list.
    for following in candidates[index + 1 :]:
        if following.kind == "arabic":
            return False
        if following.kind in {"roman", "lettered"}:
            return True
    return False


def _question_from(
    candidate: _Candidate, source_order: int, label: str, group_context: str | None
) -> ExtractedQuestion:
    question_lines: list[str] = []
    options: list[dict[str, str]] = []
    for line in candidate.lines:
        if _DECORATION.match(line):
            continue
        option = _OPTION.match(line)
        if option is not None:
            options.append({"label": option.group(1).upper(), "text": option.group(2)})
            continue
        cleaned = _MARKS_SUFFIX.sub("", line).strip()
        if not cleaned:
            continue
        if options:
            options[-1]["text"] = f"{options[-1]['text']} {cleaned}".strip()
        else:
            question_lines.append(cleaned)
    question_text = "\n".join(question_lines).strip()
    expected_labels = ["A", "B", "C", "D"]
    complete_mcq = (
        len(options) == 4 and [item["label"] for item in options] == expected_labels
    )
    if not question_text or _UNREADABLE_CUES.search(question_text):
        mode: AnswerMode = "not_clear"
        reason = "QUESTION_TEXT_UNCLEAR"
    elif options and not complete_mcq:
        mode = "not_clear"
        reason = "MCQ_OPTIONS_INCOMPLETE"
    elif complete_mcq:
        mode = "mcq"
        reason = None
    elif group_context and re.search(r"\bshort\s+answers?\b", group_context, re.I):
        # The paper's section instruction is stronger evidence than a word
        # such as "why" inside one individual short question.
        mode = "short"
        reason = None
    elif _LONG_CUES.search(question_text):
        mode = "long"
        reason = None
    else:
        mode = "short"
        reason = None
    return ExtractedQuestion(
        source_order=source_order,
        display_label=label,
        section_context=group_context or candidate.section_context,
        question_text=question_text,
        answer_mode=mode,
        mcq_options=tuple(options) if mode == "mcq" else (),
        unclear_reason=reason,
        source_locator={"page_number": candidate.page_number, "display_label": label},
    )


def extract_ordered_questions(
    source_text: str, *, max_questions: int = 60
) -> list[ExtractedQuestion]:
    """Extract explicit paper questions without silently truncating a paper."""
    candidates: list[_Candidate] = []
    page_number = 1
    section_context: str | None = None
    current: _Candidate | None = None
    for raw_line in source_text.split("\n"):
        for fragment_index, fragment in enumerate(raw_line.split("\f")):
            if fragment_index:
                page_number += 1
            section = _SECTION.match(fragment)
            if section is not None:
                section_context = section.group(1).upper()
                continue
            for line in _split_inline_starts(fragment):
                marker = _start(line)
                if marker is not None:
                    if current is not None:
                        candidates.append(current)
                    kind, label, text = marker
                    current = _Candidate(
                        page_number, kind, label, [text], section_context
                    )
                elif current is not None:
                    current.lines.append(line.strip())
    if current is not None:
        candidates.append(current)

    output: list[tuple[_Candidate, str, str | None]] = []
    group_label: str | None = None
    group_context: str | None = None
    for index, candidate in enumerate(candidates):
        if _is_container(candidates, index):
            group_label = candidate.label
            group_context = " ".join(candidate.lines).strip()
            continue
        if candidate.kind == "arabic":
            group_label, group_context = None, None
            output.append((candidate, candidate.label, None))
        elif candidate.kind == "lettered" and re.match(r"^\d", candidate.label):
            # `5(a)` is a standalone long-question part, not a child of a
            # previous "write short answers" group.
            group_label, group_context = None, None
            output.append((candidate, candidate.label, None))
        elif candidate.kind == "roman" and group_label:
            output.append(
                (candidate, f"{group_label}({candidate.label.lower()})", group_context)
            )
        else:
            output.append((candidate, candidate.label, group_context))
    if len(output) > max_questions:
        raise QuestionLimitExceeded(len(output), max_questions)
    return [
        _question_from(candidate, source_order, label, context)
        for source_order, (candidate, label, context) in enumerate(output)
    ]
