from contextlib import nullcontext
from types import SimpleNamespace

import numpy
import pytest

from app.providers.embeddings.bge import (
    EMBEDDING_DIMENSIONS,
    BGEEmbeddingProvider,
    resolve_inference_runtime,
)


class _FakeVectors:
    def tolist(self):
        value = 1 / (EMBEDDING_DIMENSIONS**0.5)
        return [[value] * EMBEDDING_DIMENSIONS]


class _FakeHiddenState:
    def __init__(self, vectors):
        self.vectors = vectors
        self.selection = None

    def __getitem__(self, key):
        self.selection = key
        return self.vectors


def test_bge_uses_cls_pooling_and_l2_normalizes_768_dimensions():
    provider = BGEEmbeddingProvider(inference_runtime="torch")
    vectors = _FakeVectors()
    hidden_state = _FakeHiddenState(vectors)
    normalize_calls = []

    provider._tokenizer = lambda *_args, **_kwargs: {"input_ids": "fixture"}
    provider._model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden_state)
    provider._torch = SimpleNamespace(
        no_grad=nullcontext,
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                normalize=lambda value, p, dim: (
                    normalize_calls.append((value, p, dim)) or value
                )
            )
        ),
    )

    result = provider._embed(["fixture document"])

    assert hidden_state.selection == (slice(None), 0)
    assert normalize_calls == [(vectors, 2, 1)]
    assert len(result) == 1
    assert len(result[0]) == 768
    assert sum(value * value for value in result[0]) == pytest.approx(1.0)


def test_railway_public_defaults_to_low_memory_onnx(monkeypatch):
    monkeypatch.delenv("BGE_INFERENCE_RUNTIME", raising=False)
    monkeypatch.setenv("WORKER_MODE", "railway_public")

    assert resolve_inference_runtime() == "onnx"


def test_explicit_runtime_override_is_validated(monkeypatch):
    monkeypatch.setenv("WORKER_MODE", "railway_public")

    assert resolve_inference_runtime("torch") == "torch"
    with pytest.raises(ValueError, match="BGE_INFERENCE_RUNTIME"):
        resolve_inference_runtime("unknown")


class _FakeOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, output_names, inputs):
        assert output_names == ["last_hidden_state"]
        assert set(inputs) == {"input_ids", "attention_mask"}
        hidden = numpy.zeros((1, 2, EMBEDDING_DIMENSIONS), dtype=numpy.float32)
        hidden[:, 0, :] = 1.0
        return [hidden]


def test_onnx_runtime_uses_cls_pooling_and_l2_normalizes():
    provider = BGEEmbeddingProvider(inference_runtime="onnx")
    provider._tokenizer = lambda *_args, **_kwargs: {
        "input_ids": numpy.ones((1, 2), dtype=numpy.int64),
        "attention_mask": numpy.ones((1, 2), dtype=numpy.int64),
    }
    provider._onnx_session = _FakeOnnxSession()

    result = provider._embed(["fixture document"])

    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIMENSIONS
    assert sum(value * value for value in result[0]) == pytest.approx(1.0)
