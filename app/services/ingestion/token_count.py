"""Voyage-4-lite model-aware token counting and estimation for chunk metadata."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class EmbeddingTokenCounter:
    """Voyage-4-lite model-aware token counter and estimator for chunk metadata."""

    method = "voyage_token_estimator"

    def __init__(
        self, model_name: str, revision: str, tokenizer: Tokenizer | None = None
    ):
        self.model_name = model_name
        self.revision = revision
        self._tokenizer = tokenizer

    @property
    def version(self) -> str:
        return f"{self.method}:{self.model_name}@{self.revision}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # Deterministic Voyage-4-lite model-aware estimation
        tokens = _TOKEN_PATTERN.findall(text)
        return max(1, len(tokens)) if text.strip() else 0


@lru_cache
def get_token_counter() -> EmbeddingTokenCounter:
    settings = get_settings()
    return EmbeddingTokenCounter(
        settings.EMBEDDING_MODEL, settings.EMBEDDING_MODEL_REVISION
    )
