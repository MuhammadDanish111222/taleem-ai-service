"""Unit tests for JSONL Chunk Parser and Validator.

Environment: Verified against a mocked Firestore client (unittest.mock.MagicMock).
"""

import json
import pytest
from unittest.mock import MagicMock
from app.services.ingestion.jsonl_chunks import validate_and_parse_jsonl


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

    chunks, errors = await validate_and_parse_jsonl(raw_jsonl, mock_db)

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

    chunks, errors = await validate_and_parse_jsonl(raw_jsonl, mock_db)

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

    chunks, errors = await validate_and_parse_jsonl(raw_jsonl, firestore_db=None)

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
    raw_jsonl_bad = raw_jsonl.replace('"chapter_id":"ch_1",', '')

    chunks, errors = await validate_and_parse_jsonl(raw_jsonl_bad, firestore_db=None)

    assert chunks == []
    assert len(errors) > 0
    errors_json = json.dumps(errors)
    assert "SECRET_SENSITIVE_TEXT_12345" not in errors_json
