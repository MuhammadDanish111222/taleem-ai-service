"""Offline precision-first semantic-reuse evaluation with the locked Voyage model.

This harness never changes policy or writes to PostgreSQL. Its report is
evidence for an administrator; enabling semantic reuse remains a separate,
explicit decision after evaluating real approved-bank data.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
from pathlib import Path
from typing import Any

from app.providers.embeddings.voyage import VoyageEmbeddingProvider

REQUIRED_CATEGORIES = {
    "positive_paraphrase",
    "punctuation_case",
    "closely_related_different",
    "same_terminology_different_answer",
    "cross_chapter_hard_negative",
    "cross_subject_hard_negative",
    "short_ambiguous",
}


def _distance(left: list[float], right: list[float]) -> float:
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


def _same_candidate_scope(case: dict[str, Any], item: dict[str, Any]) -> bool:
    return (
        case["board_id"] == item["board_id"]
        and case["class_id"] == item["class_id"]
        and case["subject_id"] == item["subject_id"]
        and case["answer_mode"] == item["answer_mode"]
        and (
            case.get("chapter_id") is None
            or case.get("chapter_id") == item.get("chapter_id")
        )
    )


def evaluate(
    payload: dict[str, Any], provider: VoyageEmbeddingProvider | None = None
) -> dict[str, Any]:
    approved = payload.get("approved")
    cases = payload.get("cases")
    if not isinstance(approved, list) or not approved:
        raise ValueError("SEMANTIC_APPROVED_FIXTURE_REQUIRED")
    if not isinstance(cases, list) or not cases:
        raise ValueError("SEMANTIC_CASE_FIXTURE_REQUIRED")
    categories = {case.get("category") for case in cases}
    missing = sorted(REQUIRED_CATEGORIES - categories)
    if missing:
        raise ValueError(f"SEMANTIC_CASE_CATEGORIES_MISSING:{','.join(missing)}")

    provider = provider or VoyageEmbeddingProvider()
    doc_fn = provider.embed_documents
    query_fn = provider.embed_queries
    if inspect.iscoroutinefunction(doc_fn):
        document_vectors = asyncio.run(doc_fn([item["text"] for item in approved]))
    else:
        document_vectors = doc_fn([item["text"] for item in approved])
    if inspect.iscoroutinefunction(query_fn):
        query_vectors = asyncio.run(query_fn([case["query"] for case in cases]))
    else:
        query_vectors = query_fn([case["query"] for case in cases])

    observations: list[dict[str, Any]] = []
    for case, query_vector in zip(cases, query_vectors, strict=True):
        candidates = [
            (item, vector)
            for item, vector in zip(approved, document_vectors, strict=True)
            if _same_candidate_scope(case, item)
        ]
        ranked = sorted(
            (
                (_distance(query_vector, vector), item["revision_id"])
                for item, vector in candidates
            ),
            key=lambda row: (row[0], row[1]),
        )
        nearest = ranked[0] if ranked else (math.inf, None)
        observations.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_revision_id": case.get("expected_revision_id"),
                "nearest_revision_id": nearest[1],
                "nearest_distance": nearest[0],
            }
        )

    finite = sorted(
        {
            item["nearest_distance"]
            for item in observations
            if math.isfinite(item["nearest_distance"])
        }
    )
    threshold_reports = []
    for threshold in finite:
        tp = fp = fn = 0
        false_positive_cases = []
        for item in observations:
            accepted = item["nearest_distance"] <= threshold
            expected = item["expected_revision_id"]
            correct = item["nearest_revision_id"] == expected
            if expected is not None and accepted and correct:
                tp += 1
            elif accepted:
                fp += 1
                false_positive_cases.append(item["case_id"])
            elif expected is not None:
                fn += 1
        threshold_reports.append(
            {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": (tp / (tp + fp)) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_cases": false_positive_cases,
            }
        )
    return {
        "model": provider.configuration.model,
        "revision": provider.configuration.revision,
        "configuration_fingerprint": provider.configuration_fingerprint,
        "approved_count": len(approved),
        "case_count": len(cases),
        "threshold_reports": threshold_reports,
        "observations": observations,
        "policy_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        type=Path,
        nargs="?",
        default=Path("tests/fixtures/module4_semantic_cases.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.fixture.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
