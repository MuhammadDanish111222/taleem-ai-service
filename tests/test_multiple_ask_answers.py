import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.v1.internal import _multiple_ask_status_response
from app.services.multiple_ask_answers import MultipleAskAnswerService


class _Connection:
    @asynccontextmanager
    async def transaction(self):
        yield self


def _item(index: int) -> dict:
    return {
        "id": f"00000000-0000-0000-0000-00000000000{index}",
        "source_text": f"Question {index}",
        "mcq_options": [
            {"label": "A", "text": "one"},
            {"label": "B", "text": "two"},
            {"label": "C", "text": "three"},
            {"label": "D", "text": "four"},
        ],
    }


def test_mcq_groups_are_ordered_and_single_batch_under_normal_limit():
    items = [_item(1), _item(2), _item(3)]
    assert MultipleAskAnswerService._mcq_groups(items) == [items]
    assert MultipleAskAnswerService._mcq_groups(items, limit=120) == [
        [items[0]],
        [items[1]],
        [items[2]],
    ]


@pytest.mark.asyncio
async def test_mcq_batch_uses_general_knowledge_without_citations_or_visuals():
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    service._repo = SimpleNamespace(
        mark_item_answering=AsyncMock(return_value=True),
        complete_answer_item=AsyncMock(),
    )
    service._asks = SimpleNamespace(
        complete=AsyncMock(side_effect=[{"id": "a1"}, {"id": "a2"}])
    )
    generation = SimpleNamespace(
        document={
            "results": [
                {
                    "item_id": _item(1)["id"],
                    "selected_option": "B",
                    "explanation": "Because.",
                },
                {
                    "item_id": _item(2)["id"],
                    "selected_option": "C",
                    "explanation": "Therefore.",
                },
            ]
        },
        provider="fake",
        model="fake-model",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        latency_ms=1,
    )
    provider = SimpleNamespace(generate=AsyncMock(return_value=generation))
    prompt = SimpleNamespace(
        resolve_active=AsyncMock(
            return_value=SimpleNamespace(
                system_prompt="general", record=SimpleNamespace(id="prompt", version=1)
            )
        )
    )
    service._ask = SimpleNamespace(_prompts=prompt, _provider=provider)
    service._candidate_request = AsyncMock(side_effect=[{"id": "r1"}, {"id": "r2"}])
    service._existing_completion = AsyncMock(return_value=False)
    service._fail_item = AsyncMock()

    await service._answer_mcq_group(
        {"id": "job", "board_id": "b", "class_id": "c", "subject_id": "s"},
        [_item(1), _item(2)],
    )

    assert provider.generate.await_count == 1
    sent = json.loads(provider.generate.await_args.kwargs["user_prompt"])
    assert [row["item_id"] for row in sent["items"]] == [_item(1)["id"], _item(2)["id"]]
    for call in service._asks.complete.await_args_list:
        assert call.kwargs["answer_source"] == "general_knowledge"
        assert call.kwargs["citations"] == []
        assert call.kwargs["visual_ids"] == []
    assert service._fail_item.await_count == 0


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
