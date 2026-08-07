"""The Multiple Ask paper capacity remains a server-enforced safety ceiling."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_extracted_question_capacity_defaults_to_and_cannot_exceed_sixty():
    assert Settings().MULTIPLE_ASK_MAX_EXTRACTED_QUESTIONS == 60
    assert (
        Settings(
            MULTIPLE_ASK_MAX_EXTRACTED_QUESTIONS=12
        ).MULTIPLE_ASK_MAX_EXTRACTED_QUESTIONS
        == 12
    )
    with pytest.raises(ValidationError):
        Settings(MULTIPLE_ASK_MAX_EXTRACTED_QUESTIONS=61)
