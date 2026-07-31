"""Pinned BGE embedding provider for bulk and on-demand query inference."""

from __future__ import annotations

import hashlib
import json
import os
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
# Qdrant's Apache-2.0 quantized ONNX port of this exact BGE model keeps the
# public query path within Railway's 1 GB memory limit. Pin both repository and
# immutable revision independently from the source-model provenance above.
ONNX_MODEL_REPO = "Qdrant/bge-base-en-v1.5-onnx-Q"
ONNX_MODEL_REVISION = "738cad1c108e2f23649db9e44b2eab988626493b"
ONNX_MODEL_FILENAME = "model_optimized.onnx"
SUPPORTED_INFERENCE_RUNTIMES = frozenset({"torch", "onnx"})


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


def resolve_inference_runtime(value: str | None = None) -> str:
    """Use low-memory ONNX for Railway queries and PyTorch for local bulk work."""
    selected = (value or os.getenv("BGE_INFERENCE_RUNTIME", "")).strip().lower()
    if not selected:
        selected = (
            "onnx"
            if os.getenv("WORKER_MODE", "").strip() == "railway_public"
            else "torch"
        )
    if selected not in SUPPORTED_INFERENCE_RUNTIMES:
        raise ValueError("BGE_INFERENCE_RUNTIME must be 'torch' or 'onnx'.")
    return selected


class BGEEmbeddingProvider:
    """Pinned BGE CLS-pooling provider with Torch and low-memory ONNX runtimes."""

    def __init__(
        self,
        configuration: BGEEmbeddingConfiguration | None = None,
        *,
        inference_runtime: str | None = None,
    ):
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
        self.inference_runtime = resolve_inference_runtime(inference_runtime)
        self._tokenizer = None
        self._model = None
        self._onnx_session = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def configuration_fingerprint(self) -> str:
        return self.configuration.fingerprint()

    def _runtime_is_loaded(self) -> bool:
        if self.inference_runtime == "onnx":
            return self._onnx_session is not None and self._tokenizer is not None
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> None:
        if self._runtime_is_loaded():
            return
        with self._load_lock:
            if self._runtime_is_loaded():
                return
            if self.inference_runtime == "onnx":
                self._load_onnx()
                return
            self._load_torch()

    def _load_torch(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGE Torch inference requires the declared torch and transformers dependencies."
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.configuration.model, revision=self.configuration.revision
        )
        self._model = AutoModel.from_pretrained(
            self.configuration.model, revision=self.configuration.revision
        )
        self._model.eval()

    def _load_onnx(self) -> None:
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGE ONNX inference requires the declared onnxruntime, huggingface-hub, "
                "and transformers dependencies."
            ) from exc

        model_path = hf_hub_download(
            repo_id=ONNX_MODEL_REPO,
            filename=ONNX_MODEL_FILENAME,
            revision=ONNX_MODEL_REVISION,
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.configuration.model, revision=self.configuration.revision
        )
        self._onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _embed_torch(self, texts: Sequence[str]) -> list[list[float]]:
        assert self._tokenizer is not None and self._model is not None
        batch = self._tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        with self._torch.no_grad():
            model_output = self._model(**batch)
            vectors = model_output.last_hidden_state[:, 0]
            vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
        return [[float(value) for value in row] for row in vectors.tolist()]

    def _embed_onnx(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            import numpy
        except ImportError as exc:
            raise RuntimeError("BGE ONNX inference requires numpy.") from exc

        assert self._tokenizer is not None and self._onnx_session is not None
        batch = self._tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="np"
        )
        model_inputs = self._onnx_session.get_inputs()
        inputs = {
            item.name: numpy.asarray(batch[item.name], dtype=numpy.int64)
            for item in model_inputs
            if item.name in batch
        }
        if {item.name for item in model_inputs}.difference(inputs):
            raise RuntimeError("BGE ONNX tokenizer output does not match model inputs.")
        outputs = self._onnx_session.get_outputs()
        output_name = next(
            (item.name for item in outputs if item.name == "last_hidden_state"),
            outputs[0].name,
        )
        hidden_state = self._onnx_session.run([output_name], inputs)[0]
        vectors = numpy.asarray(hidden_state[:, 0, :], dtype=numpy.float32)
        norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
        if numpy.any(norms == 0):
            raise RuntimeError("BGE ONNX returned a zero-length vector.")
        vectors = vectors / norms
        return vectors.tolist()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        with self._inference_lock:
            result = (
                self._embed_onnx(texts)
                if self.inference_runtime == "onnx"
                else self._embed_torch(texts)
            )
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
