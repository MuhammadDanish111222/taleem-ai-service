"""Bounded deterministic retrieval-context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.services.retrieval.evidence import RetrievedEvidence


@dataclass(frozen=True)
class ContextChunk:
    citation_id: str
    content: str
    chapter_id: str | None
    topic_title: str | None


def assemble_context(
    results: Iterable[RetrievedEvidence],
    *,
    max_chunks: int = 4,
    max_characters: int = 12000,
) -> tuple[ContextChunk, ...]:
    if max_chunks < 1 or max_characters < 1:
        raise ValueError("CONTEXT_LIMIT_INVALID")
    assembled: list[ContextChunk] = []
    seen: set[str] = set()
    remaining = max_characters
    for result in results:
        citation = result.citation
        if citation.citation_id in seen or len(assembled) >= max_chunks:
            continue
        content = citation.content.strip()
        if not content:
            continue
        if len(content) > remaining:
            content = content[:remaining].rstrip()
        if not content:
            break
        assembled.append(
            ContextChunk(
                citation_id=citation.citation_id,
                content=content,
                chapter_id=citation.chapter_id,
                topic_title=citation.topic_title,
            )
        )
        seen.add(citation.citation_id)
        remaining -= len(content)
        if remaining <= 0:
            break
    return tuple(assembled)
