"""Strictly text-only DeepSeek chat-completions provider.

The module deliberately owns no application settings singleton. Configuration,
transport, and attempt persistence are injected so callers can keep secrets and
database transactions in their existing composition roots.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_IMAGE_MARKERS = (
    re.compile(r"data\s*:\s*image/", re.IGNORECASE),
    re.compile(r"!\[[^\]]*\]\s*\(", re.IGNORECASE),
    re.compile(r"<\s*(?:img|picture|source)\b", re.IGNORECASE),
    re.compile(
        r"https?://[^\s<>'\"]+\.(?:avif|bmp|gif|heic|jpe?g|png|svg|tiff?|webp)"
        r"(?:[?#][^\s<>'\"]*)?",
        re.IGNORECASE,
    ),
)
_BASE64_MARKER = re.compile(r"(?:^|[\s,;:])(?:[A-Za-z0-9+/]{256,}={0,2})(?:$|[\s,;:])")


class ProviderErrorCode(StrEnum):
    """Stable error codes safe to expose across an internal API boundary."""

    UNAVAILABLE = "provider_unavailable"
    INVALID_TEXT_INPUT = "invalid_text_input"
    TIMEOUT = "provider_timeout"
    RATE_LIMITED = "provider_rate_limited"
    REJECTED = "provider_rejected"
    BAD_RESPONSE = "provider_bad_response"
    FAILURE = "provider_failure"


class DeepSeekProviderError(RuntimeError):
    """Sanitized provider failure.

    The original exception and response body are intentionally neither retained
    nor chained, preventing accidental prompt, answer, evidence, or secret
    disclosure through logs and API error serialization.
    """

    def __init__(self, code: ProviderErrorCode, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class DeepSeekConfig:
    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout_seconds: float = 15.0
    max_retries: int = 2
    max_output_tokens: int = 2_048
    max_input_characters: int = 32_000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("DEEPSEEK_MODEL_MUST_NOT_BE_BLANK")
        if not self.base_url.startswith("https://"):
            raise ValueError("DEEPSEEK_BASE_URL_MUST_USE_HTTPS")
        if not 0.1 <= self.timeout_seconds <= 60:
            raise ValueError("DEEPSEEK_TIMEOUT_OUT_OF_RANGE")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("DEEPSEEK_RETRIES_OUT_OF_RANGE")
        if not 1 <= self.max_output_tokens <= 8_192:
            raise ValueError("DEEPSEEK_MAX_OUTPUT_TOKENS_OUT_OF_RANGE")
        if not 1_000 <= self.max_input_characters <= 200_000:
            raise ValueError("DEEPSEEK_MAX_INPUT_CHARACTERS_OUT_OF_RANGE")

    @classmethod
    def from_environment(cls) -> DeepSeekConfig:
        """Build configuration without printing or otherwise exposing values."""

        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        )


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: Mapping[str, Any] | None


class JsonTransport(Protocol):
    async def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse: ...


class ProviderAttemptRecorder(Protocol):
    async def record_attempt(self, **fields: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    cache_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StructuredGeneration:
    document: Mapping[str, Any]
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: int
    provider_request_id: str | None
    finish_reason: str | None


class UrllibJsonTransport:
    """Minimal production transport with no dependency on a provider SDK."""

    async def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        return await asyncio.to_thread(
            self._post_json_sync,
            url,
            headers,
            payload,
            timeout_seconds,
        )

    @staticmethod
    def _post_json_sync(
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.status
                raw_body = response.read()
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            raw_body = exc.read()
        except (OSError, TimeoutError):
            raise DeepSeekProviderError(
                ProviderErrorCode.FAILURE, retryable=True
            ) from None

        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            body = None
        return TransportResponse(status_code=status_code, body=body)


class DeepSeekProvider:
    """Bounded, non-thinking, structured-JSON DeepSeek provider."""

    provider_name = "deepseek"

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        transport: JsonTransport | None = None,
        attempt_recorder: ProviderAttemptRecorder | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibJsonTransport()
        self._attempt_recorder = attempt_recorder

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        ai_request_id: str | None = None,
        trace_id: str | None = None,
    ) -> StructuredGeneration:
        """Generate one JSON object from two text-only messages."""

        _validate_text(system_prompt)
        _validate_text(user_prompt)
        if len(system_prompt) + len(user_prompt) > self._config.max_input_characters:
            raise DeepSeekProviderError(ProviderErrorCode.INVALID_TEXT_INPUT)
        if not self._config.api_key.strip():
            raise DeepSeekProviderError(ProviderErrorCode.UNAVAILABLE)

        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": self._config.max_output_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        final_error: DeepSeekProviderError | None = None

        for attempt_no in range(1, self._config.max_retries + 2):
            started = time.monotonic()
            try:
                async with asyncio.timeout(self._config.timeout_seconds):
                    response = await self._transport.post_json(
                        url=url,
                        headers=headers,
                        payload=payload,
                        timeout_seconds=self._config.timeout_seconds,
                    )
                result = _parse_response(response, self._config.model, started)
            except TimeoutError:
                final_error = DeepSeekProviderError(
                    ProviderErrorCode.TIMEOUT, retryable=True
                )
                await self._record_attempt(
                    attempt_no=attempt_no,
                    started=started,
                    ai_request_id=ai_request_id,
                    trace_id=trace_id,
                    status="failed",
                    error_code=final_error.code.value,
                    retryable=True,
                )
            except DeepSeekProviderError as exc:
                final_error = exc
                await self._record_attempt(
                    attempt_no=attempt_no,
                    started=started,
                    ai_request_id=ai_request_id,
                    trace_id=trace_id,
                    status="failed",
                    error_code=exc.code.value,
                    retryable=exc.retryable,
                )
            except Exception:
                final_error = DeepSeekProviderError(
                    ProviderErrorCode.FAILURE, retryable=True
                )
                await self._record_attempt(
                    attempt_no=attempt_no,
                    started=started,
                    ai_request_id=ai_request_id,
                    trace_id=trace_id,
                    status="failed",
                    error_code=final_error.code.value,
                    retryable=True,
                )
            else:
                await self._record_attempt(
                    attempt_no=attempt_no,
                    started=started,
                    ai_request_id=ai_request_id,
                    trace_id=trace_id,
                    status="completed",
                    result=result,
                )
                return result

            if final_error is None or not final_error.retryable:
                break

        if final_error is None:
            final_error = DeepSeekProviderError(ProviderErrorCode.FAILURE)
        raise final_error from None

    async def _record_attempt(
        self,
        *,
        attempt_no: int,
        started: float,
        ai_request_id: str | None,
        trace_id: str | None,
        status: str,
        error_code: str | None = None,
        retryable: bool | None = None,
        result: StructuredGeneration | None = None,
    ) -> None:
        if self._attempt_recorder is None:
            return
        usage = result.usage if result else TokenUsage()
        fields = {
            "provider": self.provider_name,
            "model": self._config.model,
            "status": status,
            "attempt_no": attempt_no,
            "ai_request_id": ai_request_id,
            "provider_request_id": (
                result.provider_request_id if result is not None else None
            ),
            "finish_reason": result.finish_reason if result is not None else None,
            "prompt_tokens": usage.prompt_tokens,
            "cache_tokens": usage.cache_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_ms": max(0, int((time.monotonic() - started) * 1_000)),
            "error_code": error_code,
            "retryable": retryable,
            "trace_id": trace_id,
        }
        try:
            await self._attempt_recorder.record_attempt(**fields)
        except Exception:
            # Accounting must be retried/reconciled by orchestration, but its
            # failure must not cause a second paid provider call.
            return


def _validate_text(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise DeepSeekProviderError(ProviderErrorCode.INVALID_TEXT_INPUT)
    if "\x00" in value:
        raise DeepSeekProviderError(ProviderErrorCode.INVALID_TEXT_INPUT)
    if any(pattern.search(value) for pattern in _IMAGE_MARKERS):
        raise DeepSeekProviderError(ProviderErrorCode.INVALID_TEXT_INPUT)
    if _BASE64_MARKER.search(value):
        raise DeepSeekProviderError(ProviderErrorCode.INVALID_TEXT_INPUT)


def _parse_response(
    response: TransportResponse,
    requested_model: str,
    started: float,
) -> StructuredGeneration:
    if response.status_code == 429:
        raise DeepSeekProviderError(ProviderErrorCode.RATE_LIMITED, retryable=True)
    if response.status_code in {408, 409} or response.status_code >= 500:
        raise DeepSeekProviderError(ProviderErrorCode.FAILURE, retryable=True)
    if response.status_code < 200 or response.status_code >= 300:
        raise DeepSeekProviderError(ProviderErrorCode.REJECTED)
    body = response.body
    try:
        if not isinstance(body, Mapping):
            raise TypeError
        choices = body["choices"]
        choice = choices[0]
        message = choice["message"]
        content = message["content"]
        if type(content) is not str or not content.strip():
            raise TypeError
        document = json.loads(content)
        if not isinstance(document, dict):
            raise TypeError
        usage_body = body.get("usage") or {}
        if not isinstance(usage_body, Mapping):
            raise TypeError
        usage = TokenUsage(
            prompt_tokens=_safe_nonnegative_int(usage_body.get("prompt_tokens")),
            cache_tokens=_safe_nonnegative_int(
                usage_body.get("prompt_cache_hit_tokens")
            ),
            reasoning_tokens=_safe_nonnegative_int(usage_body.get("reasoning_tokens")),
            completion_tokens=_safe_nonnegative_int(
                usage_body.get("completion_tokens")
            ),
        )
        provider_request_id = body.get("id")
        finish_reason = choice.get("finish_reason")
        returned_model = body.get("model")
        return StructuredGeneration(
            document=document,
            provider="deepseek",
            model=returned_model if type(returned_model) is str else requested_model,
            usage=usage,
            latency_ms=max(0, int((time.monotonic() - started) * 1_000)),
            provider_request_id=(
                provider_request_id if type(provider_request_id) is str else None
            ),
            finish_reason=finish_reason if type(finish_reason) is str else None,
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
        raise DeepSeekProviderError(
            ProviderErrorCode.BAD_RESPONSE, retryable=True
        ) from None


def _safe_nonnegative_int(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    return 0
