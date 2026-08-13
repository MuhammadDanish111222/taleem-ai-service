"""Small, conservative lexical matcher for approved-bank questions only."""

from __future__ import annotations

from collections.abc import Iterable

from app.services.answers.normalization import normalize_question

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "define",
        "definition",
        "describe",
        "do",
        "explain",
        "for",
        "give",
        "how",
        "in",
        "is",
        "it",
        "of",
        "state",
        "the",
        "to",
        "what",
        "which",
        "why",
        "with",
        "write",
    }
)
_MIN_SCORE = 0.80
_AMBIGUITY_MARGIN = 0.15


def _stem(token: str) -> str:
    """Only fold ordinary English plurals; avoid broad or surprising stemming."""

    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_tokens(value: str) -> frozenset[str]:
    return frozenset(
        _stem(token)
        for token in normalize_question(value).split()
        if token not in _STOP_WORDS and len(token) > 1
    )


def lexical_score(question: str, candidate: str) -> float:
    """Return a high score only when the meaningful question terms agree.

    This deliberately favors false negatives: every term from the smaller
    question must be present and the larger question cannot add much new intent.
    """

    question_tokens = meaningful_tokens(question)
    candidate_tokens = meaningful_tokens(candidate)
    if not question_tokens or not candidate_tokens:
        return 0.0
    overlap = len(question_tokens & candidate_tokens)
    if overlap != min(len(question_tokens), len(candidate_tokens)):
        return 0.0
    return overlap / max(len(question_tokens), len(candidate_tokens))


def select_lexical_match(
    question: str, candidates: Iterable[tuple[str, str, str | None]]
) -> str | None:
    """Return an unambiguous revision id, otherwise safely return no match.

    ``candidates`` contain revision id, approved question/variation text, and
    chapter.  Variations for the same revision compete as one candidate.
    """

    best_by_revision: dict[str, tuple[float, str | None]] = {}
    for revision_id, text, chapter_id in candidates:
        score = lexical_score(question, text)
        existing = best_by_revision.get(revision_id)
        if existing is None or score > existing[0]:
            best_by_revision[revision_id] = (score, chapter_id)
    ranked = sorted(
        (
            (score, revision_id, chapter_id)
            for revision_id, (score, chapter_id) in best_by_revision.items()
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < _MIN_SCORE:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < _AMBIGUITY_MARGIN:
        return None
    return ranked[0][1]
