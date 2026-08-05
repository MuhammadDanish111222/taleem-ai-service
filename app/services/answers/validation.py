"""Deterministic validation for model-produced answer blocks and references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.schemas.ask import (
    AnswerBlock,
    AnswerSource,
    BulletListBlock,
    CitationDto,
    EquationBlock,
    HeadingBlock,
    ParagraphBlock,
    VisualDto,
    VisualRefBlock,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DANGEROUS_LATEX = re.compile(
    r"\\(?:input|include|write18|openout|read|usepackage|href|url)\b",
    re.IGNORECASE,
)


class AnswerValidationError(ValueError):
    """Stable validation failure that is safe to expose as an error code."""


@dataclass(frozen=True)
class ValidatedAnswer:
    blocks: tuple[AnswerBlock, ...]
    citations: tuple[CitationDto, ...]
    visuals: tuple[VisualDto, ...]


def _clean_text(value: str, *, maximum: int, code: str) -> str:
    cleaned = _CONTROL.sub("", value).strip()
    if not cleaned or len(cleaned) > maximum:
        raise AnswerValidationError(code)
    return cleaned


def validate_generated_answer(
    *,
    source: AnswerSource,
    blocks: Iterable[AnswerBlock],
    citation_ids: Iterable[str],
    allowed_citations: dict[str, CitationDto],
    allowed_visuals: dict[str, VisualDto],
    include_all_allowed_visuals: bool = False,
) -> ValidatedAnswer:
    """Validate the entire response; mixed valid/invalid references fail atomically."""
    citation_id_list = list(citation_ids)
    block_list = list(blocks)
    if not block_list:
        raise AnswerValidationError("ANSWER_BLOCKS_EMPTY")
    if len(block_list) > 120 or len(citation_id_list) > 20:
        raise AnswerValidationError("ANSWER_STRUCTURE_TOO_LARGE")
    if len(citation_id_list) != len(set(citation_id_list)):
        raise AnswerValidationError("ANSWER_CITATION_DUPLICATE")

    unknown_citations = set(citation_id_list) - set(allowed_citations)
    if unknown_citations:
        raise AnswerValidationError("ANSWER_CITATION_NOT_ALLOWED")

    referenced_visual_ids: list[str] = []
    sanitized: list[AnswerBlock] = []
    for block in block_list:
        if isinstance(block, ParagraphBlock):
            sanitized.append(
                ParagraphBlock(
                    type="paragraph",
                    text=_clean_text(
                        block.text,
                        maximum=12000,
                        code="ANSWER_PARAGRAPH_INVALID",
                    ),
                )
            )
        elif isinstance(block, HeadingBlock):
            sanitized.append(
                HeadingBlock(
                    type="heading",
                    text=_clean_text(
                        block.text,
                        maximum=300,
                        code="ANSWER_HEADING_INVALID",
                    ),
                    level=block.level,
                )
            )
        elif isinstance(block, BulletListBlock):
            sanitized.append(
                BulletListBlock(
                    type="bullet_list",
                    items=[
                        _clean_text(
                            item,
                            maximum=2000,
                            code="ANSWER_BULLET_ITEM_INVALID",
                        )
                        for item in block.items
                    ],
                )
            )
        elif isinstance(block, EquationBlock):
            latex = _clean_text(
                block.latex, maximum=4000, code="ANSWER_EQUATION_INVALID"
            )
            if _DANGEROUS_LATEX.search(latex):
                raise AnswerValidationError("ANSWER_EQUATION_UNSAFE")
            sanitized.append(EquationBlock(type="equation", latex=latex))
        elif isinstance(block, VisualRefBlock):
            referenced_visual_ids.append(block.visual_id)
            sanitized.append(block)
        else:  # pragma: no cover - Pydantic prevents this at the API boundary.
            raise AnswerValidationError("ANSWER_BLOCK_TYPE_INVALID")

    if len(referenced_visual_ids) != len(set(referenced_visual_ids)):
        raise AnswerValidationError("ANSWER_VISUAL_DUPLICATE")
    if set(referenced_visual_ids) - set(allowed_visuals):
        raise AnswerValidationError("ANSWER_VISUAL_NOT_ALLOWED")

    if source == AnswerSource.GENERAL_KNOWLEDGE and (
        citation_id_list or referenced_visual_ids
    ):
        raise AnswerValidationError("GENERAL_ANSWER_HAS_TEXTBOOK_REFERENCES")
    if source == AnswerSource.SYLLABUS_GROUNDED and not citation_id_list:
        raise AnswerValidationError("GROUNDED_ANSWER_HAS_NO_CITATION")

    selected_visuals = tuple(
        allowed_visuals[visual_id] for visual_id in referenced_visual_ids
    )
    if source == AnswerSource.SYLLABUS_GROUNDED:
        always_visuals = sorted(
            (
                visual
                for visual in allowed_visuals.values()
                if (include_all_allowed_visuals or visual.display_policy == "always")
                and visual.visual_id not in referenced_visual_ids
            ),
            key=lambda visual: (visual.display_order, visual.visual_id),
        )
        for visual in always_visuals:
            sanitized.append(
                VisualRefBlock(type="visual_ref", visual_id=visual.visual_id)
            )
        selected_visuals += tuple(always_visuals)

    return ValidatedAnswer(
        blocks=tuple(sanitized),
        citations=tuple(allowed_citations[item] for item in citation_id_list),
        visuals=selected_visuals,
    )
