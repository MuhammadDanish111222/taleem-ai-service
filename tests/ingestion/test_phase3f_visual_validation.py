import json

import pytest

from app.services.ingestion.jsonl_chunks import (
    VALID_VISUAL_TYPES,
    validate_and_parse_jsonl,
)
from app.services.ingestion.token_count import EmbeddingTokenCounter

COUNTER = EmbeddingTokenCounter("voyage-4-lite", "voyage-4-lite-512-v1")


def _row(visuals):
    return json.dumps(
        {
            "board_id": "board",
            "class_id": "class",
            "subject_id": "subject",
            "chapter_id": "chapter",
            "topic_no": "1",
            "topic_title": "Forces",
            "chunk_order": 1,
            "content_type": "explanation",
            "chunk_text": "Force changes motion.",
            "expected_questions": ["What is force?"],
            "visuals": visuals,
        }
    )


@pytest.mark.asyncio
async def test_jsonl_visuals_are_normalized_without_exposing_storage_key_in_errors():
    chunks, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "force-diagram",
                    "visual_type": "diagram",
                    "title": "Force diagram",
                    "description": "A block with arrows.",
                    "storage_key": "server-only-drive-key",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert not errors
    assert chunks[0]["visuals"] == [
        {
            "visual_id": "force-diagram",
            "visual_type": "diagram",
            "title": "Force diagram",
            "description": "A block with arrows.",
            "storage_key": "server-only-drive-key",
        }
    ]


@pytest.mark.asyncio
async def test_jsonl_rejects_duplicate_visual_ids_without_echoing_key_or_description():
    _, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "same",
                    "visual_type": "diagram",
                    "title": "A",
                    "description": "secret description",
                    "storage_key": "drive-secret",
                },
                {
                    "visual_id": "same",
                    "visual_type": "figure",
                    "title": "B",
                    "description": "another secret",
                    "storage_key": "another-drive-secret",
                },
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    encoded = json.dumps(errors)
    assert any(error["reason"] == "duplicate_visual_id_in_chunk" for error in errors)
    assert "drive-secret" not in encoded
    assert "secret description" not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("visual_type", sorted(VALID_VISUAL_TYPES))
async def test_jsonl_accepts_each_supported_visual_type(visual_type: str):
    _, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": f"visual-{visual_type}",
                    "visual_type": visual_type,
                    "title": "Supported visual",
                    "description": "A valid visual type.",
                    "storage_key": "server-only-drive-key",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )

    assert not errors


@pytest.mark.asyncio
async def test_jsonl_validator_preserves_review_status_and_display_policy():
    """A. Enriched visual with review_status=approved and display_policy=always_show:
    After validate_and_parse_jsonl(), both fields are present and preserved."""
    chunks, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "force-diagram",
                    "visual_type": "diagram",
                    "title": "Force diagram",
                    "description": "A block with arrows.",
                    "storage_key": "server-only-drive-key",
                    "review_status": "approved",
                    "display_policy": "always_show",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert not errors
    assert len(chunks) == 1
    visual = chunks[0]["visuals"][0]
    assert visual["review_status"] == "approved"
    assert visual["display_policy"] == "always_show"


@pytest.mark.asyncio
async def test_jsonl_validator_omits_absent_lifecycle_fields():
    """Generic visual without review_status/display_policy:
    After validate_and_parse_jsonl(), the fields remain absent in the output."""
    chunks, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "force-diagram",
                    "visual_type": "diagram",
                    "title": "Force diagram",
                    "description": "A block with arrows.",
                    "storage_key": "server-only-drive-key",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert not errors
    assert len(chunks) == 1
    visual = chunks[0]["visuals"][0]
    assert "review_status" not in visual
    assert "display_policy" not in visual


@pytest.mark.asyncio
async def test_jsonl_validator_rejects_invalid_review_status():
    """D. Invalid review_status is safely rejected with clear error."""
    _, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "force-diagram",
                    "visual_type": "diagram",
                    "title": "Force diagram",
                    "description": "A block with arrows.",
                    "storage_key": "server-only-drive-key",
                    "review_status": "invalid_status",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert any(
        error.get("field") == "visuals.review_status"
        and error.get("reason") == "invalid_review_status_enum"
        for error in errors
    )


@pytest.mark.asyncio
async def test_jsonl_validator_rejects_invalid_display_policy():
    """D. Invalid display_policy is safely rejected with clear error."""
    _, errors = await validate_and_parse_jsonl(
        _row(
            [
                {
                    "visual_id": "force-diagram",
                    "visual_type": "diagram",
                    "title": "Force diagram",
                    "description": "A block with arrows.",
                    "storage_key": "server-only-drive-key",
                    "display_policy": "show_on_mondays",
                }
            ]
        ),
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert any(
        error.get("field") == "visuals.display_policy"
        and error.get("reason") == "invalid_display_policy_enum"
        for error in errors
    )


@pytest.mark.asyncio
async def test_validator_to_repository_flow_paired_vs_generic():
    """B & C. End-to-end real validator output fed into RagRepository.replace_chapter_chunks:
    - B. Paired imported visual (approved + llm_decide) -> validator preserves both -> repository stores approved + llm_decide.
    - C. Generic visual without lifecycle fields -> repository defaults to pending + llm_decide (not silently auto-approved).
    """
    from app.repositories.rag_repository import RagRepository

    # 1. Validated paired import chunk (has explicit approved + llm_decide)
    paired_jsonl = _row(
        [
            {
                "visual_id": "paired-v1",
                "visual_type": "figure",
                "title": "Paired Visual",
                "description": "From trusted import.",
                "storage_key": "drive-key-paired",
                "review_status": "approved",
                "display_policy": "llm_decide",
            }
        ]
    )
    paired_chunks, paired_errs = await validate_and_parse_jsonl(
        paired_jsonl,
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert not paired_errs

    # 2. Validated generic import chunk (no lifecycle fields)
    generic_jsonl = _row(
        [
            {
                "visual_id": "generic-v2",
                "visual_type": "diagram",
                "title": "Generic Visual",
                "description": "Generic chunk without explicit approval.",
                "storage_key": "drive-key-generic",
            }
        ]
    )
    generic_chunks, generic_errs = await validate_and_parse_jsonl(
        generic_jsonl,
        None,
        allow_mock_validation_for_tests=True,
        token_counter=COUNTER,
    )
    assert not generic_errs

    # Mock DB to record visual inserts
    visual_inserts = []

    class MockConn:
        async def fetchrow(self, query, *args):
            if "FOR UPDATE" in query:
                return {"status": "building"}
            if "INSERT INTO rag_chunks" in query:
                return {
                    "id": "c-1",
                    "document_version_id": "doc-1",
                    "corpus_version_id": "cv-1",
                    "chunk_index": 0,
                    "content": "test",
                    "chapter_id": "chapter",
                    "topic_no": "1",
                    "topic_title": "Forces",
                    "content_type": "explanation",
                    "content_hash": "abc",
                }
            if "expected_chunk_count" in query or "embedded_chunk_count" in query:
                return {
                    "expected_chunk_count": 1,
                    "embedded_chunk_count": 0,
                    "expected_question_count": 0,
                    "embedded_question_count": 0,
                }
            return None

        async def fetch(self, query, *args):
            return []

        async def execute(self, query, *args):
            if "INSERT INTO rag_visuals" in query:
                visual_inserts.append(args)
            return "INSERT 1"

    repo = RagRepository(MockConn())  # type: ignore

    # Execute replace_chapter_chunks with paired import
    await repo.replace_chapter_chunks(
        corpus_version_id="cv-1",
        document_version_id="doc-1",
        chunks=paired_chunks,
    )
    assert len(visual_inserts) == 1
    # Args order: chunk_id, visual_id, visual_type, storage_key, title, description, display_policy, review_status, visual_text_hash
    paired_insert = visual_inserts[0]
    assert paired_insert[1] == "paired-v1"
    assert paired_insert[6] == "llm_decide"
    assert paired_insert[7] == "approved", "Paired import must be inserted as approved"

    # Execute replace_chapter_chunks with generic import
    visual_inserts.clear()
    await repo.replace_chapter_chunks(
        corpus_version_id="cv-1",
        document_version_id="doc-1",
        chunks=generic_chunks,
    )
    assert len(visual_inserts) == 1
    generic_insert = visual_inserts[0]
    assert generic_insert[1] == "generic-v2"
    assert generic_insert[6] == "llm_decide"
    assert generic_insert[7] == "pending", "Generic import must default to pending (not silently approved)"

