import sys
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


class _FakeEncoding:
    ids = [101, 102]
    attention_mask = [1, 1]
    type_ids = [0, 0]


class _FakeOnnxTokenizer:
    def encode_batch(self, texts):
        assert texts == ["fixture document"]
        return [_FakeEncoding()]


def test_onnx_runtime_uses_cls_pooling_and_l2_normalizes():
    provider = BGEEmbeddingProvider(inference_runtime="onnx")
    provider._tokenizer = _FakeOnnxTokenizer()
    provider._onnx_session = _FakeOnnxSession()

    result = provider._embed(["fixture document"])

    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIMENSIONS
    assert sum(value * value for value in result[0]) == pytest.approx(1.0)


def test_onnx_loader_does_not_import_torch_or_transformers(monkeypatch, tmp_path):
    imported = []

    class _Options:
        def add_session_config_entry(self, *_args):
            return None

    class _Tokenizer:
        def enable_truncation(self, **_kwargs):
            return None

        def enable_padding(self, **_kwargs):
            return None

    fake_onnx = SimpleNamespace(
        SessionOptions=_Options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC="basic"),
        InferenceSession=lambda *_args, **_kwargs: object(),
    )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(tmp_path / kwargs["filename"])
    )
    fake_tokenizers = SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda _path: _Tokenizer())
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    original_import = __import__

    def recording_import(name, *args, **kwargs):
        imported.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", recording_import)
    provider = BGEEmbeddingProvider(inference_runtime="onnx")

    provider._load_onnx()

    assert not any(name == "torch" or name.startswith("transformers") for name in imported)
