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

    async def fetchrow(self, *args, **kwargs):
        return None


def _item(index: int) -> dict:
    return {
        "id": f"00000000-0000-0000-0000-00000000000{index}",
        "source_text": f"Question {index}",
        "normalized_question": f"question {index}",
        "answer_mode": "mcq",
        "item_status": "ready_to_answer",
        "mcq_options": [
            {"label": "A", "text": "one"},
            {"label": "B", "text": "two"},
            {"label": "C", "text": "three"},
            {"label": "D", "text": "four"},
        ],
    }


def test_polling_status_exposes_question_text_not_normalized_or_storage_reference():
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
                "normalized_question": "internal normalized question",
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
                "source_text": "What is the fourth state of matter?",
                "storage_object_key": "private/secret.pdf",
            }
        ],
    }
    response = _multiple_ask_status_response(record)
    body = json.dumps(response)
    assert "What is the fourth state of matter?" in body
    assert "internal normalized question" in body
    assert "private/secret.pdf" not in body
    assert (
        response["items"][0]["question_text"] == "What is the fourth state of matter?"
    )
    assert (
        response["items"][0]["question_text"]
        != response["items"][0]["normalized_question"]
    )


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
    service._asks = SimpleNamespace(complete=AsyncMock(return_value={"id": "ans-456"}))
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
@pytest.mark.parametrize("count", [1, 5, 20])
async def test_all_mcqs_use_one_general_short_deepseek_call(count: int):
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    service._repo = SimpleNamespace(
        get_mcq_batch=AsyncMock(return_value=None),
        save_mcq_batch=AsyncMock(
            side_effect=lambda **kwargs: {
                "results": kwargs["results"],
                "prompt_version": kwargs["prompt_version"],
                "provider": kwargs["provider"],
                "model": kwargs["model"],
            }
        ),
        complete_answer_item=AsyncMock(),
    )
    candidate_request = {"id": "req-mcq", "status": "pending"}
    service._asks = SimpleNamespace(complete=AsyncMock(return_value={"id": "ans-mcq"}))
    service._candidate_request = AsyncMock(return_value=candidate_request)
    service._existing_completion = AsyncMock(return_value=False)

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
                    "results": [
                        {
                            "item_id": _item(index)["id"],
                            "selected_option": "B",
                            "answer_text": None,
                            "explanation": "B is correct.",
                        }
                        for index in range(1, count + 1)
                    ]
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
    items = [_item(index) for index in range(1, count + 1)]

    await service._answer_mcq_batch(job, items, epoch=1)

    assert provider.generate.await_count == 1
    sent_prompt = json.loads(provider.generate.await_args.kwargs["user_prompt"])
    assert sent_prompt["items"] == [
        {
            "item_id": item["id"],
            "question": item["source_text"],
            "options": item["mcq_options"],
        }
        for item in items
    ]
    assert (
        prompt_service.resolve_active.await_args.kwargs["prompt_key"].value
        == "ask_general"
    )
    assert (
        prompt_service.resolve_active.await_args.kwargs["answer_mode"].value == "short"
    )
    assert service._asks.complete.await_count == count
    assert service._repo.complete_answer_item.await_count == count


@pytest.mark.asyncio
async def test_mixed_paper_batches_mcqs_once_then_answers_written_items_in_paper_order(
    monkeypatch,
):
    """The MCQ-only path must remain isolated when a paper also has written items."""
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    mcq_one, mcq_two, short_item, long_item = (
        _item(1),
        _item(2),
        _item(3),
        _item(4),
    )
    short_item.update(
        answer_mode="short",
        source_text="Define velocity.",
        normalized_question="define velocity",
    )
    long_item.update(
        answer_mode="long",
        source_text="Explain acceleration.",
        normalized_question="explain acceleration",
    )
    refreshed = [
        {**mcq_one, "item_status": "answered"},
        {**mcq_two, "item_status": "answered"},
        short_item,
        long_item,
    ]
    job = {
        "id": "job-1",
        "workflow_status": "answering",
        "answer_epoch": 1,
        "board_id": "b1",
        "class_id": "c1",
        "subject_id": "s1",
        "chapter_id": None,
    }
    service._repo = SimpleNamespace(
        lock_answer_context=AsyncMock(return_value=job),
        lock_job_items=AsyncMock(
            side_effect=[[mcq_one, mcq_two, short_item, long_item], refreshed]
        ),
        mark_item_answering=AsyncMock(return_value=True),
        finish_answers=AsyncMock(return_value="completed"),
    )
    service._answer_mcq_batch = AsyncMock()
    service._candidate_request = AsyncMock(
        side_effect=[{"id": "req-short"}, {"id": "req-long"}]
    )
    service._existing_completion = AsyncMock(return_value=False)
    service._answer_single_short_or_long = AsyncMock()
    service._ask = SimpleNamespace(
        find_approved_without_embedding=AsyncMock(return_value=(None, None)),
        _source_policy=AsyncMock(return_value={"semantic_reuse_enabled": False}),
        _retrieval=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(results=[]))
        ),
    )

    class _NoActiveCorpus:
        def __init__(self, _conn):
            pass

        async def get_active_corpus_version(self, *_args):
            return None

    monkeypatch.setattr(
        "app.services.multiple_ask_answers.RagRepository", _NoActiveCorpus
    )
    monkeypatch.setattr(
        "app.services.multiple_ask_answers.get_settings",
        lambda: SimpleNamespace(MULTIPLE_ASK_ANSWER_BATCH_SIZE=2),
    )

    assert await service.answer(session_id="session-1", epoch=1) == "completed"
    assert service._answer_mcq_batch.await_count == 1
    assert [item["id"] for item in service._answer_mcq_batch.await_args.args[1]] == [
        mcq_one["id"],
        mcq_two["id"],
    ]
    assert [
        call.args[1]["id"]
        for call in service._answer_single_short_or_long.await_args_list
    ] == [
        short_item["id"],
        long_item["id"],
    ]
    assert [
        call.args[0] for call in service._ask._retrieval.retrieve.await_args_list
    ] == [
        short_item["source_text"],
        long_item["source_text"],
    ]


@pytest.mark.parametrize("option_count", [2, 3, 4, 5])
def test_mcq_result_accepts_actual_dynamic_option_labels(option_count: int):
    item = _item(1)
    item["mcq_options"] = [
        {"label": chr(65 + index), "text": f"choice {index}"}
        for index in range(option_count)
    ]
    result = MultipleAskAnswerService._validate_mcq_results(
        {
            "results": [
                {
                    "item_id": item["id"],
                    "selected_option": chr(64 + option_count),
                    "answer_text": None,
                    "explanation": "Brief reason.",
                }
            ]
        },
        [item],
    )
    assert result[0]["correct_answer_text"] == f"choice {option_count - 1}"


def test_mcq_result_rejects_invalid_option_and_accepts_zero_option_answer():
    item = _item(1)
    with pytest.raises(Exception, match="MULTIPLE_ASK_MCQ_OPTION_INVALID"):
        MultipleAskAnswerService._validate_mcq_results(
            {
                "results": [
                    {
                        "item_id": item["id"],
                        "selected_option": "Z",
                        "answer_text": None,
                        "explanation": "Brief reason.",
                    }
                ]
            },
            [item],
        )
    item["mcq_options"] = []
    result = MultipleAskAnswerService._validate_mcq_results(
        {
            "results": [
                {
                    "item_id": item["id"],
                    "selected_option": None,
                    "answer_text": "Plasma",
                    "explanation": "It is ionized matter.",
                }
            ]
        },
        [item],
    )
    assert result[0]["correct_answer_text"] == "Plasma"


@pytest.mark.asyncio
async def test_persisted_mcq_batch_resumes_without_a_provider_call():
    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    item = _item(1)
    persisted_result = {
        "item_id": item["id"],
        "selected_option": "B",
        "correct_answer_text": "two",
        "explanation": "B is correct.",
    }
    service._repo = SimpleNamespace(
        get_mcq_batch=AsyncMock(
            return_value={
                "results": [persisted_result],
                "prompt_version": "p:1",
                "provider": "deepseek",
                "model": "deepseek-chat",
            }
        ),
        complete_answer_item=AsyncMock(),
    )
    provider = SimpleNamespace(generate=AsyncMock())
    service._ask = SimpleNamespace(_provider=provider)
    service._asks = SimpleNamespace(complete=AsyncMock(return_value={"id": "ans-1"}))
    service._candidate_request = AsyncMock(return_value={"id": "req-1"})
    service._existing_completion = AsyncMock(return_value=False)

    await service._answer_mcq_batch(
        {"id": "job-1", "board_id": "b", "class_id": "c", "subject_id": "s"},
        [item],
        epoch=1,
    )

    assert provider.generate.await_count == 0
    assert (
        service._repo.complete_answer_item.await_args.kwargs["mcq_result"]
        == persisted_result
    )


@pytest.mark.parametrize(
    "results,error",
    [
        ([], "MISMATCH"),
        (
            [{"item_id": "unknown", "selected_option": "A", "explanation": "x"}],
            "UNKNOWN",
        ),
        (
            [
                {"item_id": _item(1)["id"], "selected_option": "A", "explanation": "x"},
                {"item_id": _item(1)["id"], "selected_option": "A", "explanation": "x"},
            ],
            "MISMATCH",
        ),
        (
            [{"item_id": _item(1)["id"], "selected_option": "A", "explanation": " "}],
            "EXPLANATION",
        ),
    ],
)
def test_mcq_result_validation_rejects_malformed_results(results, error):
    with pytest.raises(Exception, match=f"MULTIPLE_ASK_MCQ_.*{error}"):
        MultipleAskAnswerService._validate_mcq_results({"results": results}, [_item(1)])


def test_mcq_result_validation_rejects_duplicate_and_provider_artifacts():
    first, second = _item(1), _item(2)
    duplicate = {
        "item_id": first["id"],
        "selected_option": "A",
        "answer_text": None,
        "explanation": "Brief.",
    }
    with pytest.raises(Exception, match="MULTIPLE_ASK_MCQ_DUPLICATE_ITEM"):
        MultipleAskAnswerService._validate_mcq_results(
            {"results": [duplicate, duplicate]}, [first, second]
        )
    with pytest.raises(Exception, match="MULTIPLE_ASK_MCQ_RESPONSE_INVALID"):
        MultipleAskAnswerService._validate_mcq_results(
            {"results": [duplicate], "citations": []}, [first]
        )


@pytest.mark.asyncio
async def test_retryable_deepseek_error_re_raises_for_queue_backoff():
    from app.providers.llm.deepseek import DeepSeekProviderError, ProviderErrorCode

    service = object.__new__(MultipleAskAnswerService)
    service._conn = _Connection()
    service._repo = SimpleNamespace(
        lock_answer_context=AsyncMock(
            return_value={
                "id": "job-1",
                "workflow_status": "answering",
                "answer_epoch": 1,
                "board_id": "b1",
                "class_id": "c1",
                "subject_id": "s1",
                "chapter_id": None,
            }
        ),
        lock_job_items=AsyncMock(
            return_value=[
                {
                    "id": "item-1",
                    "source_text": "Q1",
                    "normalized_question": "q1",
                    "item_status": "ready_to_answer",
                    "answer_mode": "short",
                }
            ]
        ),
        mark_item_answering=AsyncMock(return_value=True),
        finish_answers=AsyncMock(return_value="completed"),
    )
    service._candidate_request = AsyncMock(return_value={"id": "req-1"})
    service._existing_completion = AsyncMock(return_value=False)
    service._fail_item = AsyncMock()

    retryable_err = DeepSeekProviderError(ProviderErrorCode.TIMEOUT, retryable=True)
    service._answer_single_short_or_long = AsyncMock(side_effect=retryable_err)
    service._ask = SimpleNamespace(
        _source_policy=AsyncMock(return_value={"semantic_reuse_enabled": False}),
        find_approved_without_embedding=AsyncMock(return_value=(None, None)),
        _bank=SimpleNamespace(
            find_exact=AsyncMock(return_value=None),
            find_exact_variation=AsyncMock(return_value=None),
        ),
        _retrieval=SimpleNamespace(
            retrieve=AsyncMock(return_value=SimpleNamespace(results=[]))
        ),
    )

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await service.answer(session_id="sess-1", epoch=1)

    assert exc_info.value.retryable is True
    assert service._fail_item.await_count == 0
