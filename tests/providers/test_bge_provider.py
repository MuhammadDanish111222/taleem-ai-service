from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.providers.embeddings.bge import EMBEDDING_DIMENSIONS, BGEEmbeddingProvider


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
    provider = BGEEmbeddingProvider()
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
