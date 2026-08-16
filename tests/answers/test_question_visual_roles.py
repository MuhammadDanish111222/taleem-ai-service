"""Role-separation invariants that do not require PostgreSQL."""

import pytest

from app.api.v1 import ask_admin
from app.schemas.ask_admin import ApprovedQuestionInput


def approved(*, question_visual_ids=None, answer_visual_ids=None):
    return ApprovedQuestionInput(
        board_id="punjab", class_id="class-9", subject_id="physics", chapter_id="motion",
        answer_mode="short", difficulty="easy", marks=2, question="Define momentum.",
        blocks=[{"type": "visual_ref", "visual_id": "answer-diagram"}],
        question_visual_ids=question_visual_ids or [], answer_visual_ids=answer_visual_ids or ["answer-diagram"],
    )


@pytest.mark.asyncio
async def test_generated_candidates_cannot_create_question_visuals(monkeypatch):
    async def visual_rows(*args, **kwargs):
        return ["row"]

    async def citations(*args, **kwargs):
        return []

    class Bank:
        async def create_approved_revision(self, **kwargs):
            return "revision-1"

    monkeypatch.setattr(ask_admin, "_visual_row_ids", visual_rows)
    monkeypatch.setattr(ask_admin, "_validated_citation_ids", citations)
    with pytest.raises(ValueError, match="CANDIDATE_QUESTION_VISUALS_FORBIDDEN"):
        await ask_admin._create_approved(object(), Bank(), approved(question_visual_ids=["question-diagram"]), actor_id="admin", source="generated_candidate")


@pytest.mark.asyncio
async def test_admin_authored_questions_keep_roles_separate(monkeypatch):
    async def visual_rows(_conn, ids, **_kwargs):
        return [f"row:{item}" for item in ids]

    async def citations(*args, **kwargs):
        return []

    class Bank:
        received = None

        async def create_approved_revision(self, **kwargs):
            self.received = kwargs
            return "revision-1"

    monkeypatch.setattr(ask_admin, "_visual_row_ids", visual_rows)
    monkeypatch.setattr(ask_admin, "_validated_citation_ids", citations)
    bank = Bank()
    assert await ask_admin._create_approved(object(), bank, approved(question_visual_ids=["question-diagram"]), actor_id="admin", source="admin_authored") == "revision-1"
    assert bank.received["question_visual_row_ids"] == ["row:question-diagram"]
    assert bank.received["answer_visual_row_ids"] == ["row:answer-diagram"]


def test_stage3_migration_filters_student_paper_visuals_to_question_role():
    migration = open("migrations/0020_question_answer_visual_roles.sql", encoding="utf-8").read()
    assert "link.role='question'" in migration
    assert "DEFAULT 'answer'" in migration
