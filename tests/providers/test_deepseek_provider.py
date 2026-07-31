import asyncio

import pytest

from app.providers.llm.deepseek import (
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekConfig,
    DeepSeekProvider,
    DeepSeekProviderError,
    ProviderErrorCode,
    TransportResponse,
)


def _success_response() -> TransportResponse:
    return TransportResponse(
        status_code=200,
        body={
            "id": "safe-provider-id",
            "model": DEFAULT_DEEPSEEK_MODEL,
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"blocks":[{"type":"paragraph","text":"Answer"}],'
                            '"cited_chunk_ids":[]}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "prompt_cache_hit_tokens": 3,
                "reasoning_tokens": 0,
                "completion_tokens": 7,
            },
        },
    )


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post_json(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class HangingTransport:
    def __init__(self):
        self.calls = 0

    async def post_json(self, **_kwargs):
        self.calls += 1
        await asyncio.sleep(1)
        return _success_response()


class AttemptRecorder:
    def __init__(self):
        self.rows = []

    async def record_attempt(self, **fields):
        self.rows.append(fields)


@pytest.mark.asyncio
async def test_deepseek_sends_only_text_messages_and_requests_non_thinking_json():
    transport = FakeTransport([_success_response()])
    recorder = AttemptRecorder()
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key"),
        transport=transport,
        attempt_recorder=recorder,
    )

    result = await provider.generate(
        system_prompt="Return JSON for this teaching task.",
        user_prompt="What is photosynthesis?",
        ai_request_id="00000000-0000-0000-0000-000000000001",
    )

    payload = transport.calls[0]["payload"]
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "Return JSON for this teaching task."},
        {"role": "user", "content": "What is photosynthesis?"},
    ]
    assert result.document["blocks"][0]["text"] == "Answer"
    assert result.usage.prompt_tokens == 11
    assert result.usage.cache_tokens == 3
    assert recorder.rows[0]["status"] == "completed"
    assert "answer" not in recorder.rows[0]
    assert "prompt" not in recorder.rows[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        b"binary",
        {"type": "image_url", "image_url": "https://example.test/a.png"},
        ["text", "image"],
        "data:image/png;base64,AAAA",
        "Please inspect https://example.test/diagram.webp",
        "![diagram](https://example.test/diagram)",
        "A" * 300,
    ],
)
async def test_deepseek_rejects_non_text_and_image_payloads_before_transport(invalid):
    transport = FakeTransport([_success_response()])
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key"), transport=transport
    )

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await provider.generate(
            system_prompt="Return a JSON object.",
            user_prompt=invalid,
        )

    assert exc_info.value.code is ProviderErrorCode.INVALID_TEXT_INPUT
    assert transport.calls == []


@pytest.mark.asyncio
async def test_missing_key_is_provider_unavailable_without_transport_call():
    transport = FakeTransport([_success_response()])
    provider = DeepSeekProvider(DeepSeekConfig(api_key=""), transport=transport)

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await provider.generate(
            system_prompt="Return a JSON object.",
            user_prompt="A text-only question",
        )

    assert exc_info.value.code is ProviderErrorCode.UNAVAILABLE
    assert str(exc_info.value) == "provider_unavailable"
    assert transport.calls == []


@pytest.mark.asyncio
async def test_combined_provider_input_is_bounded_before_transport():
    transport = FakeTransport([_success_response()])
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key", max_input_characters=1_000),
        transport=transport,
    )

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await provider.generate(
            system_prompt="Return one JSON object. " + ("s" * 600),
            user_prompt="u" * 600,
        )

    assert exc_info.value.code is ProviderErrorCode.INVALID_TEXT_INPUT
    assert transport.calls == []


@pytest.mark.asyncio
async def test_retry_is_bounded_and_attempt_errors_are_sanitized():
    secret_body = {"error": {"message": "secret prompt and key"}}
    transport = FakeTransport(
        [
            TransportResponse(status_code=500, body=secret_body),
            TransportResponse(status_code=429, body=secret_body),
            _success_response(),
        ]
    )
    recorder = AttemptRecorder()
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key", max_retries=2),
        transport=transport,
        attempt_recorder=recorder,
    )

    result = await provider.generate(
        system_prompt="Return a JSON object.",
        user_prompt="A text-only question",
    )

    assert result.provider_request_id == "safe-provider-id"
    assert len(transport.calls) == 3
    assert [row["error_code"] for row in recorder.rows] == [
        "provider_failure",
        "provider_rate_limited",
        None,
    ]
    assert "secret prompt" not in repr(recorder.rows)


@pytest.mark.asyncio
async def test_timeout_is_strict_and_does_not_exceed_retry_bound():
    transport = HangingTransport()
    recorder = AttemptRecorder()
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key", timeout_seconds=0.1, max_retries=1),
        transport=transport,
        attempt_recorder=recorder,
    )

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await provider.generate(
            system_prompt="Return a JSON object.",
            user_prompt="A text-only question",
        )

    assert exc_info.value.code is ProviderErrorCode.TIMEOUT
    assert transport.calls == 2
    assert [row["error_code"] for row in recorder.rows] == [
        "provider_timeout",
        "provider_timeout",
    ]


@pytest.mark.asyncio
async def test_invalid_json_is_never_returned_as_structured_output():
    response = TransportResponse(
        status_code=200,
        body={
            "choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]
        },
    )
    provider = DeepSeekProvider(
        DeepSeekConfig(api_key="fixture-key", max_retries=0),
        transport=FakeTransport([response]),
    )

    with pytest.raises(DeepSeekProviderError) as exc_info:
        await provider.generate(
            system_prompt="Return a JSON object.",
            user_prompt="A text-only question",
        )

    assert exc_info.value.code is ProviderErrorCode.BAD_RESPONSE
    assert str(exc_info.value) == "provider_bad_response"
