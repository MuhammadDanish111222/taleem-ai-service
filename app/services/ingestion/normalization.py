"""Deterministic normalization shared by JSONL validation and persistence."""

import re
import unicodedata


def normalize_expected_question(question: str) -> str:
    """Normalizes expected questions for duplicate detection and unique storage."""
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", question).strip()
    ).casefold()
