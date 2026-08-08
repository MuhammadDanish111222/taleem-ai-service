from __future__ import annotations

import json
from pathlib import Path

from app.providers.embeddings.voyage import VoyageEmbeddingConfiguration
from scripts.evaluate_semantic_reuse import evaluate


class FakeLockedProvider:
    configuration = VoyageEmbeddingConfiguration()
    configuration_fingerprint = configuration.fingerprint()

    def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_queries(self, texts):
        return [[0.0, 1.0] for _ in texts]


def test_semantic_harness_covers_required_safety_categories_without_policy_write():
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures" / "module4_semantic_cases.json"
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    report = evaluate(payload, provider=FakeLockedProvider())

    categories = {item["category"] for item in report["observations"]}
    assert {
        "positive_paraphrase",
        "punctuation_case",
        "closely_related_different",
        "same_terminology_different_answer",
        "cross_chapter_hard_negative",
        "cross_subject_hard_negative",
        "short_ambiguous",
    } <= categories
    assert report["model"] == "voyage-4-lite"
    assert report["policy_changed"] is False
    assert report["threshold_reports"]
