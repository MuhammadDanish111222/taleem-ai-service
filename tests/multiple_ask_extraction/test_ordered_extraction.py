"""Fixture contract for Module 5 Run 2 deterministic question extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services.multiple_ask_extraction import (
    QuestionLimitExceeded,
    extract_ordered_questions,
)

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "ordered_extraction_cases.json").read_text(
        encoding="utf-8"
    )
)


def _as_dict(item: Any) -> dict[str, Any]:
    """Keep the test focused on the stable extractor result contract."""
    return {
        "source_order": item.source_order,
        "question_text": item.question_text,
        "answer_mode": item.answer_mode,
        "mcq_options": list(item.mcq_options),
        "unclear_reason": item.unclear_reason,
        "source_locator": {"page_number": item.source_locator["page_number"]},
    }


def test_numbered_questions_preserve_order_modes_options_and_locator():
    case = _CASES["standard_numbered"]

    actual = [_as_dict(item) for item in extract_ordered_questions(case["source_text"])]

    assert actual == [
        {**expected, "unclear_reason": None} for expected in case["expected"]
    ]


def test_page_breaks_produce_page_ordered_locators_and_unclear_item():
    case = _CASES["page_order_and_unclear"]

    actual = [_as_dict(item) for item in extract_ordered_questions(case["source_text"])]

    assert actual == [
        {**expected, "unclear_reason": expected.get("unclear_reason")}
        for expected in case["expected"]
    ]
    assert [item["source_order"] for item in actual] == [0, 1, 2]
    assert [item["source_locator"]["page_number"] for item in actual] == [1, 2, 2]


def test_extraction_is_deterministic_and_does_not_correct_ocr_content():
    case = _CASES["conservative_ocr_text"]

    first = extract_ordered_questions(case["source_text"])
    second = extract_ordered_questions(case["source_text"])

    assert first == second
    assert [item.question_text for item in first] == case["expected_question_text"]


def test_extraction_rejects_instead_of_silently_dropping_questions_over_the_cap():
    source_text = "\n".join(f"{index}. What is item {index}?" for index in range(1, 62))
    with pytest.raises(QuestionLimitExceeded) as caught:
        extract_ordered_questions(source_text)
    assert (caught.value.count, caught.value.limit) == (61, 60)


def test_objective_paper_supports_decimal_and_whole_numbered_mcqs():
    items = extract_ordered_questions(
        "SECTION-I\n1.1 Which quantity is a vector? (1 mark)\n(A) Mass\n(B) Time\n(C) Velocity\n(D) Speed\n"
        "2. Which instrument measures current?\n(A) Ammeter\n(B) Voltmeter\n(C) Barometer\n(D) Thermometer\n"
        "17. Which is a scalar?\n(A) Force\n(B) Displacement\n(C) Speed\n(D) Acceleration"
    )

    assert [item.question_text for item in items] == [
        "Which quantity is a vector?",
        "Which instrument measures current?",
        "Which is a scalar?",
    ]
    assert all(item.answer_mode == "mcq" for item in items)
    assert [option["label"] for option in items[0].mcq_options] == ["A", "B", "C", "D"]


def test_subjective_grouping_keeps_every_roman_and_lettered_answerable_subpart():
    items = extract_ordered_questions(
        "SECTION-II\n2. Write short answers of any eight parts.\n"
        "i. Define momentum.\nii. State Newton's second law.\niii. Why is friction useful?\n"
        "xii. Name the SI unit of power.\n"
        "5(a) Explain the law of conservation of energy. (4 marks)\n"
        "5(b) Describe a simple pendulum.\n6(a) What is acceleration?\n6(b) How does mass affect inertia?"
    )

    assert [item.question_text for item in items] == [
        "Define momentum.",
        "State Newton's second law.",
        "Why is friction useful?",
        "Name the SI unit of power.",
        "Explain the law of conservation of energy.",
        "Describe a simple pendulum.",
        "What is acceleration?",
        "How does mass affect inertia?",
    ]
    assert [item.answer_mode for item in items] == [
        "short",
        "short",
        "short",
        "short",
        "short",
        "short",
        "short",
        "short",
    ]
    assert [item.display_label for item in items[:4]] == [
        "2(i)",
        "2(ii)",
        "2(iii)",
        "2(xii)",
    ]
    assert items[0].section_context == "Write short answers of any eight parts."


def test_joined_roman_subparts_and_wrapped_options_remain_distinct_items():
    items = extract_ordered_questions(
        "2. Write short answers of any two parts i. Define work. ii. Define power.\n"
        "3. Choose the correct answer.\n(A) First option wraps\non this line\n(B) Second option\n(C) Third option\n(D) Fourth option"
    )

    assert [item.question_text for item in items] == [
        "Define work.",
        "Define power.",
        "Choose the correct answer.",
    ]
    assert items[2].mcq_options[0] == {
        "label": "A",
        "text": "First option wraps on this line",
    }


def test_incomplete_or_misordered_mcq_options_are_not_answerable_mcqs():
    items = extract_ordered_questions(
        "1. Choose correctly.\n(A) One\n(B) Two\n(D) Four\n"
        "2. Another choice.\n(A) One\n(C) Three\n(B) Two\n(D) Four"
    )
    assert [item.answer_mode for item in items] == ["not_clear", "not_clear"]
    assert [item.unclear_reason for item in items] == [
        "MCQ_OPTIONS_INVALID",
        "MCQ_OPTIONS_INVALID",
    ]


def test_stage4_section_priority_and_no_section_fallbacks():
    items = extract_ordered_questions(
        "SECTION A\nMULTIPLE CHOICE QUESTIONS\n1. Which state has highest kinetic energy?\n"
        "SHORT ANSWER QUESTIONS\n2. Explain graphite conduction.\n"
        "LONG QUESTIONS\n3. Define allotropy.\n"
        "SECTION B\n4. Explain electrolysis.\n"
        "5. Carbon form?\nA. Graphite\nB. Diamond"
    )
    assert [item.answer_mode for item in items] == [
        "mcq",
        "short",
        "long",
        "short",
        "mcq",
    ]
    assert items[0].mcq_options == ()
    assert items[1].section_context == "SECTION A — SHORT ANSWER QUESTIONS"


@pytest.mark.parametrize("labels", ["AB", "ABC", "ABCD", "ABCDE"])
def test_stage4_dynamic_ordered_mcq_options(labels: str):
    source = "1. Select.\n" + "\n".join(f"{label}. option {label}" for label in labels)
    item = extract_ordered_questions(source)[0]
    assert item.answer_mode == "mcq"
    assert [option["label"] for option in item.mcq_options] == list(labels)


@pytest.mark.parametrize(
    "source",
    [
        "1. Select.\nA. one\nC. three",
        "1. Select.\nA. one\nB. two\nB. duplicate\nC. three",
        "1. Select.\nA. \nB. two",
    ],
)
def test_stage4_malformed_options_are_not_guessed(source: str):
    item = extract_ordered_questions(source)[0]
    assert item.answer_mode == "not_clear"
    assert item.unclear_reason == "MCQ_OPTIONS_INVALID"
