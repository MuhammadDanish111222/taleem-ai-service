"""Embedding-tokenizer based, deterministic token counting for ingestion."""

from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


class EmbeddingTokenCounter:
    """Caches only the configured embedding tokenizer, never the embedding model."""

    method = "huggingface_auto_tokenizer"

    def __init__(
        self, model_name: str, revision: str, tokenizer: Tokenizer | None = None
    ):
        self.model_name = model_name
        self.revision = revision
        self._tokenizer = tokenizer

    @property
    def version(self) -> str:
        return f"{self.method}:{self.model_name}@{self.revision}"

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            # Deferred import keeps startup and offline unit tests independent of transformers.
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                use_fast=True,
            )
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))


@lru_cache
def get_token_counter() -> EmbeddingTokenCounter:
    settings = get_settings()
    return EmbeddingTokenCounter(
        settings.EMBEDDING_MODEL, settings.EMBEDDING_MODEL_REVISION
    )
