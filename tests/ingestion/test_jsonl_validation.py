"""Unit tests for JSONL Chunk Parser and Validator.

Environment: Verified against a mocked Firestore client (unittest.mock.MagicMock).
"""

import json
from unittest.mock import MagicMock

import pytest

from app.services.ingestion.jsonl_chunks import validate_and_parse_jsonl


class FakeTokenCounter:
    method = "test_tokenizer"
    version = "test_tokenizer:fixture@v1"

    def count(self, text: str) -> int:
        return len(text.split())


FAKE_COUNTER = FakeTokenCounter()


def create_mock_firestore(active_chain: bool = True):
    """Creates a mock firestore client simulating a 4-level catalogue hierarchy."""
    mock_db = MagicMock()

    def get_doc(doc_id, active_flag=active_chain):
        doc_mock = MagicMock()
        doc_mock.exists = True
        doc_mock.to_dict.return_value = {"active": active_flag}
        return doc_mock

    board_mock = MagicMock()
    board_mock.get.return_value = get_doc("fbise")

    class_mock = MagicMock()
    class_mock.get.return_value = get_doc("class_9")

    subject_mock = MagicMock()
    subject_mock.get.return_value = get_doc("physics")

    chapter_mock = MagicMock()
    chapter_mock.get.return_value = get_doc("ch_1")

    mock_db.collection.return_value.document.return_value = board_mock
    board_mock.collection.return_value.document.return_value = class_mock
    class_mock.collection.return_value.document.return_value = subject_mock
    subject_mock.collection.return_value.document.return_value = chapter_mock

    return mock_db


@pytest.mark.asyncio
async def test_valid_jsonl_parsing():
    """Valid multi-row JSONL fixture produces parsed chunks with correct field mapping."""
    mock_db = create_mock_firestore(active_chain=True)
    raw_jsonl = (
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.1","topic_title":"Physical Quantities","chunk_order":0,'
        '"content_type":"explanation","chunk_text":"Physics is the study of matter and energy.",'
        '"expected_questions":["What is physics?"],"page_range":[1,5],"language":"en"}\n'
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.2","topic_title":"SI Units","chunk_order":1,'
        '"content_type":"definition","chunk_text":"SI units are standard units of measurement.",'
        '"expected_questions":["Define SI units."],"page_range":[6,10],"language":"en"}'
    )

    chunks, errors = await validate_and_parse_jsonl(
        raw_jsonl, mock_db, token_counter=FAKE_COUNTER
    )

    assert errors == []
    assert len(chunks) == 2

    c0 = chunks[0]
    assert c0["board_id"] == "fbise"
    assert c0["class_id"] == "class_9"
    assert c0["subject_id"] == "physics"
    assert c0["chapter_id"] == "ch_1"
    assert c0["topic_no"] == "1.1"
    assert c0["topic_title"] == "Physical Quantities"
    assert c0["chunk_order"] == 0
    assert c0["content_type"] == "explanation"
    assert c0["chunk_text"] == "Physics is the study of matter and energy."
    assert c0["expected_questions"] == ["What is physics?"]
    assert c0["page_start"] == 1
    assert c0["page_end"] == 5
    assert len(c0["content_hash"]) == 64
    assert c0["token_count"] == 8
    assert c0["language"] == "en"

    c1 = chunks[1]
    assert c1["chunk_order"] == 1
    assert c1["content_type"] == "definition"
    assert c1["page_start"] == 6
    assert c1["page_end"] == 10


@pytest.mark.asyncio
async def test_inactive_chapter_rejection():
    """Inactive node in Firestore hierarchy causes all-or-nothing rejection."""
    mock_db = create_mock_firestore(active_chain=False)
    raw_jsonl = (
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.1","topic_title":"Physical Quantities","chunk_order":0,'
        '"content_type":"explanation","chunk_text":"Sample text","expected_questions":[],"page_range":[1,2]}'
    )

    chunks, errors = await validate_and_parse_jsonl(
        raw_jsonl, mock_db, token_counter=FAKE_COUNTER
    )

    assert chunks == []
    assert len(errors) == 1
    assert errors[0]["field"] == "chapter_id"
    assert errors[0]["reason"] == "unknown_or_inactive_catalogue_hierarchy"


@pytest.mark.asyncio
async def test_malformed_fields_rejection():
    """Invalid content_type enum, duplicate chunk_order, and bad page_range are rejected."""
    raw_jsonl = (
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.1","topic_title":"Topic 1","chunk_order":0,'
        '"content_type":"invalid_type","chunk_text":"Text 1","page_range":"1-5"}\n'
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.1","topic_title":"Topic 1","chunk_order":0,'
        '"content_type":"explanation","chunk_text":"Text 2","page_range":[1,5]}'
    )

    chunks, errors = await validate_and_parse_jsonl(
        raw_jsonl,
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )

    assert chunks == []
    assert len(errors) >= 2

    # Verify per-row error format and absence of raw chunk_text in errors
    for err in errors:
        assert "row" in err
        assert "field" in err
        assert "reason" in err
        err_str = json.dumps(err)
        assert "Text 1" not in err_str
        assert "Text 2" not in err_str


@pytest.mark.asyncio
async def test_error_log_sanitization():
    """Asserts that raw chunk text is sanitized and absent from validation error outputs."""
    raw_jsonl = (
        '{"board_id":"fbise","class_id":"class_9","subject_id":"physics","chapter_id":"ch_1",'
        '"topic_no":"1.1","topic_title":"Topic 1","chunk_order":0,'
        '"content_type":"explanation","chunk_text":"SECRET_SENSITIVE_TEXT_12345"}'
    )
    # Missing required field chapter_id will fail JSON schema, check raw text not in error dict
    raw_jsonl_bad = raw_jsonl.replace('"chapter_id":"ch_1",', "")

    chunks, errors = await validate_and_parse_jsonl(
        raw_jsonl_bad,
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )

    assert chunks == []
    assert len(errors) > 0
    errors_json = json.dumps(errors)
    assert "SECRET_SENSITIVE_TEXT_12345" not in errors_json


@pytest.mark.asyncio
async def test_rejects_empty_mixed_scope_and_invalid_expected_questions_without_source_text():
    empty_chunks, empty_errors = await validate_and_parse_jsonl(
        " \n", allow_mock_validation_for_tests=True, token_counter=FAKE_COUNTER
    )
    assert empty_chunks == []
    assert empty_errors[0]["code"] == "EMPTY_JSONL"

    rows = [
        {
            "board_id": "fbise",
            "class_id": "class_9",
            "subject_id": "physics",
            "chapter_id": "ch_1",
            "topic_no": "1",
            "topic_title": "One",
            "chunk_order": 0,
            "content_type": "explanation",
            "chunk_text": "safe one",
            "expected_questions": ["What is force?"],
        },
        {
            "board_id": "fbise",
            "class_id": "class_9",
            "subject_id": "physics",
            "chapter_id": "ch_2",
            "topic_no": "2",
            "topic_title": "Two",
            "chunk_order": 0,
            "content_type": "explanation",
            "chunk_text": "safe two",
            "expected_questions": ["What is mass?"],
        },
    ]
    chunks, errors = await validate_and_parse_jsonl(
        "\n".join(json.dumps(row) for row in rows),
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )
    assert chunks == []
    assert any(
        error["code"] == "JSONL_SCOPE_MISMATCH" and error["row"] == 2
        for error in errors
    )

    bad_questions = dict(
        rows[0], expected_questions=["  ", "What is force?", " what  is  force? "]
    )
    chunks, errors = await validate_and_parse_jsonl(
        json.dumps(bad_questions),
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )
    assert chunks == []
    assert {error["reason"] for error in errors} >= {
        "blank_or_non_string_question",
        "duplicate_question_in_chunk",
    }


@pytest.mark.asyncio
async def test_rejects_duplicate_chunk_order_and_requires_expected_questions_array():
    base = {
        "board_id": "fbise",
        "class_id": "class_9",
        "subject_id": "physics",
        "chapter_id": "ch_1",
        "topic_no": "1",
        "topic_title": "One",
        "chunk_order": 0,
        "content_type": "explanation",
        "chunk_text": "safe",
    }
    with_questions = dict(base, expected_questions=[])
    chunks, errors = await validate_and_parse_jsonl(
        "\n".join(
            [json.dumps(with_questions), json.dumps(dict(with_questions, topic_no="2"))]
        ),
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )
    assert chunks == []
    assert any(error["reason"] == "duplicate_chunk_order_in_batch" for error in errors)

    chunks, errors = await validate_and_parse_jsonl(
        json.dumps(base),
        firestore_db=None,
        allow_mock_validation_for_tests=True,
        token_counter=FAKE_COUNTER,
    )
    assert chunks == []
    assert any(
        error["field"] == "expected_questions" and error["reason"] == "must_be_array"
        for error in errors
    )
