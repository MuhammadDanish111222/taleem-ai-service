import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.v1.internal import _multiple_ask_status_response
from app.schemas.ask import AnswerMode
from app.services.multiple_ask_answers import MultipleAskAnswerService


class _Connection:
    @asynccontextmanager
    async def transaction(self):
        yield self


def _item(index: int) -> dict:
    return {
        "id": f"00000000-0000-0000-0000-00000000000{index}",
        "source_text": f"Question {index}",
        "normalized_question": f"question {index}",
        "answer_mode": "mcq",
        "mcq_options": [
            {"label": "A", "text": "one"},
            {"label": "B", "text": "two"},
            {"label": "C", "text": "three"},
            {"label": "D", "text": "four"},
        ],
    }


def test_polling_status_never_serializes_raw_source_or_storage_reference():
    record = {
        "id": "00000000-0000-0000-0000-000000000099",
        "workflow_status": "completed",
        "input_kind": "text",
        "board_id": "b",
        "class_id": "c",
        "subject_id": "s",
        "chapter_id": None,
        "created_at": None,
        "updated_at": None,
        "retention_expires_at": None,
        "terminal_error_code": None,
        "queue_status": "succeeded",
        "queue_stage": None,
        "queue_progress": 100,
        "items": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "item_index": 0,
                "display_label": "1",
                "section_context": None,
                "item_status": "answered",
                "normalized_question": "safe question",
                "answer_mode": "mcq",
                "mcq_options": [],
                "unclear_reason": None,
                "source_locator": {"page_number": 1},
                "extraction_version": 1,
                "correction_version": 0,
                "corrected_at": None,
                "answer_source": "general_knowledge",
                "persisted_answer_source": "general_knowledge",
                "answer_blocks": [{"type": "paragraph", "text": "Answer"}],
                "citation_sources": [],
                "visual_ids": [],
                "approved_revision_id": None,
                "source_text": "RAW_SOURCE_SECRET",
                "storage_object_key": "private/secret.pdf",
            }
        ],
    }
    body = json.dumps(_multiple_ask_status_response(record))
    assert "RAW_SOURCE_SECRET" not in body
    assert "private/secret.pdf" not in body


@pytest.mark.asyncio
async def test_short_answer_with_evidence_expands_topics_with_scope_fields():
    """Verify non-approved question with retrieval evidence expands topics without AttributeError."""
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    service._repo = SimpleNamespace(
        mark_item_answering=AsyncMock(return_value=True),
        complete_answer_item=AsyncMock(),
    )
    candidate_request = {
        "id": "req-123",
        "status": "pending",
        "source_feature": "multiple_ask",
    }
    service._asks = SimpleNamespace(
        complete=AsyncMock(return_value={"id": "ans-456"})
    )
    service._fail_item = AsyncMock()

    prompt_service = SimpleNamespace(
        resolve_active=AsyncMock(
            return_value=SimpleNamespace(
                system_prompt="grounded prompt",
                record=SimpleNamespace(id="p-1", version=1),
            )
        )
    )
    provider = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                document={
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": "Evaporation is a surface phenomenon.",
                        }
                    ],
                    "cited_chunk_ids": ["chunk-1"],
                },
                provider="deepseek",
                model="deepseek-chat",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                latency_ms=150,
            )
        )
    )
    expand_topics_mock = AsyncMock(
        return_value=[
            SimpleNamespace(
                citation=SimpleNamespace(
                    citation_id="chunk-1",
                    chapter_id="chap-1",
                    topic_no="1.1",
                    topic_title="Evaporation",
                    page_start=10,
                    page_end=12,
                    content="Evaporation is a surface process...",
                    visuals=[],
                ),
                chunk_text="Evaporation is a surface process...",
            )
        ]
    )
    service._ask = SimpleNamespace(
        _expand_answer_topics=expand_topics_mock,
        _prompts=prompt_service,
        _provider=provider,
    )

    job = {
        "id": "job-1",
        "board_id": "b1",
        "class_id": "c1",
        "subject_id": "s1",
        "chapter_id": None,
        "uid_hash": "uid-hash",
    }
    item = {
        "id": "item-1",
        "source_text": "Define evaporation.",
        "normalized_question": "define evaporation",
        "answer_mode": "short",
    }
    evidence = SimpleNamespace(
        results=[
            SimpleNamespace(
                citation=SimpleNamespace(
                    citation_id="chunk-1",
                    chapter_id="chap-1",
                    topic_no="1.1",
                    topic_title="Evaporation",
                    page_start=10,
                    page_end=12,
                    content="Evaporation is a surface process...",
                    visuals=[],
                ),
                chunk_text="Evaporation is a surface process...",
                fused_rank=1,
            )
        ]
    )
    active = {"id": "corp-1"}
    policy = {"allow_general": True, "semantic_reuse_enabled": False}

    await service._answer_single_short_or_long(
        job, item, candidate_request, AnswerMode.SHORT, evidence, active, policy
    )

    # Verify _expand_answer_topics received request object with all 4 required scope attributes
    assert expand_topics_mock.await_count == 1
    passed_request = expand_topics_mock.await_args.kwargs["request"]
    assert passed_request.board_id == "b1"
    assert passed_request.class_id == "c1"
    assert passed_request.subject_id == "s1"
    assert passed_request.answer_mode.value == "short"

    # Verify provider was called
    assert provider.generate.await_count == 1
    # Verify complete was called with grounded source
    assert service._asks.complete.await_count == 1
    complete_args = service._asks.complete.await_args.kwargs
    assert complete_args["answer_source"] == "syllabus_grounded"
    assert service._repo.complete_answer_item.await_count == 1


@pytest.mark.asyncio
async def test_mcq_answer_with_options_passed_to_prompt():
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    service._repo = SimpleNamespace(
        mark_item_answering=AsyncMock(return_value=True),
        complete_answer_item=AsyncMock(),
    )
    candidate_request = {"id": "req-mcq", "status": "pending"}
    service._asks = SimpleNamespace(
        complete=AsyncMock(return_value={"id": "ans-mcq"})
    )
    service._fail_item = AsyncMock()

    prompt_service = SimpleNamespace(
        resolve_active=AsyncMock(
            return_value=SimpleNamespace(
                system_prompt="mcq prompt",
                record=SimpleNamespace(id="p-mcq", version=1),
            )
        )
    )
    provider = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                document={
                    "blocks": [
                        {"type": "paragraph", "text": "Correct answer is B."}
                    ],
                    "cited_chunk_ids": [],
                },
                provider="deepseek",
                model="deepseek-chat",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
                latency_ms=100,
            )
        )
    )
    service._ask = SimpleNamespace(
        _prompts=prompt_service,
        _provider=provider,
    )

    job = {
        "id": "job-mcq",
        "board_id": "b1",
        "class_id": "c1",
        "subject_id": "s1",
        "chapter_id": None,
    }
    item = _item(1)
    evidence = SimpleNamespace(results=[])
    policy = {"allow_general": True, "semantic_reuse_enabled": False}

    await service._answer_single_mcq(
        job, item, candidate_request, evidence, None, policy
    )

    assert provider.generate.await_count == 1
    sent_prompt = json.loads(provider.generate.await_args.kwargs["user_prompt"])
    assert sent_prompt["answer_mode"] == "mcq"
    assert sent_prompt["mcq_options"] == item["mcq_options"]
    assert service._asks.complete.await_count == 1
    assert service._repo.complete_answer_item.await_count == 1
