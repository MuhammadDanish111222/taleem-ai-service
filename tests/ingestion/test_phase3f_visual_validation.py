import json

import pytest

from app.services.ingestion.jsonl_chunks import (
    VALID_VISUAL_TYPES,
    validate_and_parse_jsonl,
)
from app.services.ingestion.token_count import EmbeddingTokenCounter


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text if not character.isspace()]


COUNTER = EmbeddingTokenCounter(
    "BAAI/bge-base-en-v1.5", "test", tokenizer=FakeTokenizer()
)


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
