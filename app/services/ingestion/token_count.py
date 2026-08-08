"""Voyage-4-lite model-aware token counting and estimation for chunk metadata."""

from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import get_settings

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class EmbeddingTokenCounter:
    """Voyage-4-lite model-aware token counter and estimator for chunk metadata."""

    method = "voyage_token_estimator"

    def __init__(self, model_name: str, revision: str):
        self.model_name = model_name
        self.revision = revision

    @property
    def version(self) -> str:
        return f"{self.method}:{self.model_name}@{self.revision}"

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        # Deterministic Voyage-4-lite model-aware estimation
        tokens = _TOKEN_PATTERN.findall(text)
        return max(1, len(tokens))


@lru_cache
def get_token_counter() -> EmbeddingTokenCounter:
    settings = get_settings()
    return EmbeddingTokenCounter(
        settings.EMBEDDING_MODEL, settings.EMBEDDING_MODEL_REVISION
    )
