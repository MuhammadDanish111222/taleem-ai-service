"""Stage 1 JSON-import normalization tests (no database required)."""

import pytest
from pydantic import ValidationError

from app.schemas.ask_admin import AskAdminRequest, BulkImportQuestionInput


def _question(**overrides):
    value = {
        "question": "Define an atom.",
        "type": "short",
        "difficulty": "easy",
        "answer_blocks": [
            {"type": "paragraph", "text": "An atom retains an element's properties."}
        ],
    }
    value.update(overrides)
    return value


def test_bulk_import_defaults_marks_and_applies_selected_scope():
    short = BulkImportQuestionInput(**_question())
    long = BulkImportQuestionInput(**_question(type="long"))
    mcq = BulkImportQuestionInput(
        **_question(
            type="mcq",
            options=["Proton", "Electron"],
            correct_answer="Electron",
            answer_blocks=[],
        )
    )

    assert (mcq.resolved_marks, short.resolved_marks, long.resolved_marks) == (1, 2, 4)
    explicit = BulkImportQuestionInput(**_question(marks=6))
    assert explicit.resolved_marks == 6

    normalized = short.as_approved_question(
        board_id="punjab",
        class_id="class-9",
        subject_id="chemistry",
        chapter_id="atoms",
    )
    assert (
        normalized.board_id,
        normalized.class_id,
        normalized.subject_id,
        normalized.chapter_id,
    ) == ("punjab", "class-9", "chemistry", "atoms")
    assert normalized.blocks[0].type == "paragraph"


def test_bulk_import_rejects_invalid_item_and_incomplete_scope_before_any_insert():
    valid = _question()
    invalid = _question(
        question="Broken MCQ",
        type="mcq",
        options=["A", "B"],
        correct_answer="C",
        answer_blocks=[],
    )
    with pytest.raises(ValidationError, match="MCQ_CORRECT_ANSWER_INVALID"):
        AskAdminRequest(
            operation="bank_import",
            board_id="punjab",
            class_id="class-9",
            subject_id="chemistry",
            chapter_id="atoms",
            import_key="batch-1",
            import_questions=[valid, invalid],
        )
    with pytest.raises(
        ValidationError, match="IMPORT_SCOPE_REQUIRES_BOARD_CLASS_SUBJECT_CHAPTER"
    ):
        AskAdminRequest(
            operation="bank_import",
            board_id="punjab",
            class_id="class-9",
            subject_id="chemistry",
            import_key="batch-1",
            import_questions=[valid],
        )


@pytest.mark.parametrize(
    "options,correct_answer",
    [([], "A"), (["A"], "A"), (["A", ""], "A"), (["A", "A"], "A")],
)
def test_bulk_import_rejects_malformed_mcq_options(options, correct_answer):
    with pytest.raises(ValidationError):
        BulkImportQuestionInput(
            **_question(
                type="mcq",
                options=options,
                correct_answer=correct_answer,
                answer_blocks=[],
            )
        )


def test_bulk_import_normalizes_mcq_and_explicit_visual_roles_to_bank_contract():
    value = BulkImportQuestionInput(
        **_question(
            question="Which particle has a negative charge?",
            type="mcq",
            options=["Proton", "Electron"],
            correct_answer="Electron",
            answer_blocks=[],
            question_visual_ids=["atom-diagram"],
            answer_visual_ids=["annotated-atom"],
        )
    ).as_approved_question(
        board_id="punjab",
        class_id="class-9",
        subject_id="chemistry",
        chapter_id="atoms",
    )
    assert [
        (option.key, option.text, option.is_correct) for option in value.mcq_options
    ] == [
        ("1", "Proton", False),
        ("2", "Electron", True),
    ]
    assert [block.type for block in value.blocks] == ["visual_ref"]
    assert value.question_visual_ids == ["atom-diagram"]
    assert value.answer_visual_ids == ["annotated-atom"]


def test_bulk_import_rejects_legacy_ambiguous_visual_ids():
    with pytest.raises(ValidationError):
        BulkImportQuestionInput(**_question(visual_ids=["legacy-visual"]))
