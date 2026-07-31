"""Pinned BGE embedding provider used only by background workers."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from typing import Sequence

MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Hugging Face commit for the published v1.5 repository state.  Do not replace
# this immutable revision with `main`: corpus versions must be reproducible.
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
EMBEDDING_DIMENSIONS = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DOCUMENT_INPUT_FORMAT = "topic-heading-approved-visuals-v2"
QUERY_INPUT_FORMAT = "bge-query-instruction-v1"


@dataclass(frozen=True)
class BGEEmbeddingConfiguration:
    model: str = MODEL_NAME
    revision: str = MODEL_REVISION
    dimensions: int = EMBEDDING_DIMENSIONS
    normalize: bool = True
    query_instruction: str = QUERY_INSTRUCTION
    document_input_format: str = DOCUMENT_INPUT_FORMAT
    query_input_format: str = QUERY_INPUT_FORMAT

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def format_chunk_embedding_input(
    *,
    topic_no: object,
    topic_title: object,
    chunk_text: object,
    approved_visuals: Sequence[dict[str, object]] = (),
    approved_visual: dict[str, object] | None = None,
) -> str:
    """Builds the only allowed chunk embedding text.

    Only already-reviewed metadata is included, in deterministic logical-ID
    order supplied by the repository. IDs, paths, storage keys, and all other
    metadata are never part of this input.
    """
    lines = []
    if number := _clean_text(topic_no):
        lines.append(f"Topic {number}")
    if title := _clean_text(topic_title):
        lines.append(f"Title: {title}")
    text = _clean_text(chunk_text)
    if not text:
        raise ValueError("Chunk embedding text must not be blank.")
    lines.append(f"Content: {text}")
    # Compatibility argument is intentionally ignored unless it carries the
    # same explicit approval/link proof used by the old Phase 3D test fixture.
    visuals = sorted(
        approved_visuals,
        key=lambda item: _clean_text(item.get("visual_id")),
    )
    if (
        approved_visual
        and approved_visual.get("is_linked")
        and approved_visual.get("is_approved")
    ):
        visuals.append(approved_visual)
    for approved_visual in visuals:
        if visual_title := _clean_text(approved_visual.get("title")):
            lines.append(f"Visual: {visual_title}")
        if visual_description := _clean_text(approved_visual.get("description")):
            lines.append(f"Visual description: {visual_description}")
    return "\n".join(lines)


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BGEEmbeddingProvider:
    """Minimal Transformers/PyTorch implementation using BGE CLS pooling."""

    def __init__(self, configuration: BGEEmbeddingConfiguration | None = None):
        self.configuration = configuration or BGEEmbeddingConfiguration()
        if (
            self.configuration.model != MODEL_NAME
            or self.configuration.revision != MODEL_REVISION
            or self.configuration.dimensions != EMBEDDING_DIMENSIONS
            or not self.configuration.normalize
        ):
            raise ValueError(
                "Phase 3D permits only the pinned BAAI/bge-base-en-v1.5 configuration."
            )
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "BGE embedding requires the declared torch and transformers dependencies."
                ) from exc
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.configuration.model, revision=self.configuration.revision
            )
            self._model = AutoModel.from_pretrained(
                self.configuration.model, revision=self.configuration.revision
            )
            self._model.eval()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        assert self._tokenizer is not None and self._model is not None
        with self._inference_lock:
            batch = self._tokenizer(
                list(texts), padding=True, truncation=True, return_tensors="pt"
            )
            with self._torch.no_grad():
                model_output = self._model(**batch)
                vectors = model_output.last_hidden_state[:, 0]
                vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
            result = [[float(value) for value in row] for row in vectors.tolist()]
        self._validate_vectors(result)
        return result

    def _validate_vectors(self, vectors: Sequence[Sequence[float]]) -> None:
        if any(len(vector) != self.configuration.dimensions for vector in vectors):
            raise ValueError(
                "Embedding provider returned a vector with an invalid dimension."
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [
            f"{self.configuration.query_instruction}{_clean_text(text)}"
            for text in texts
        ]
        return self._embed(formatted)
