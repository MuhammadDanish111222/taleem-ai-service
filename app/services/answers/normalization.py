"""The single authoritative question-normalization implementation."""

from __future__ import annotations

import hashlib
import re
import unicodedata

NORMALIZATION_VERSION = 1
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _NON_WORD.sub(" ", normalized)
    return _SPACE.sub(" ", normalized).strip()


def question_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
