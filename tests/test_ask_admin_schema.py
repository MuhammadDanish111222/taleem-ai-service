import pytest
from pydantic import ValidationError

from app.schemas.ask_admin import AskAdminRequest


def test_semantic_threshold_admin_request_is_bounded_and_scoped():
    value = AskAdminRequest(
        operation="source_policy_set_semantic_threshold",
        subject_id="physics",
        class_id="class-9",
        semantic_similarity_threshold=0.82,
    )
    assert value.semantic_similarity_threshold == 0.82

    with pytest.raises(ValidationError):
        AskAdminRequest(
            operation="source_policy_set_semantic_threshold",
            subject_id="physics",
            semantic_similarity_threshold=0.79,
        )
    with pytest.raises(ValidationError):
        AskAdminRequest(
            operation="source_policy_set_semantic_threshold",
            semantic_similarity_threshold=0.82,
        )
