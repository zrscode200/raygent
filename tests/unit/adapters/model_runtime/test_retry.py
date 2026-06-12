from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import cast

import pytest

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.adapters.model_runtime import (
    ProtocolModelProvider,
    ProviderPayloadError,
    ProviderRetryPolicy,
    ProviderTransportRequest,
    classify_retry_decision,
)
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    ModelMessage,
    ModelRequest,
    ModelStreamEvent,
    ProviderError,
    TextContentBlock,
    TokenCountRequest,
)


def _request() -> ModelRequest:
    return ModelRequest(
        model="claude-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(TextContentBlock(text="hello"),),
                )
            ),
        ),
    )


def _anthropic_response(text: str, *, request_id: str = "req_1") -> ProviderResponse:
    return {
        "id": "msg_1",
        "request_id": request_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


class QueueTransport:
    def __init__(
        self,
        *,
        complete_steps: tuple[ProviderResponse | BaseException, ...] = (),
        stream_steps: tuple[tuple[ProviderEvent | BaseException, ...] | BaseException, ...] = (),
        token_steps: tuple[int | BaseException, ...] = (),
    ) -> None:
        self.complete_steps = list(complete_steps)
        self.stream_steps = list(stream_steps)
        self.token_steps = list(token_steps)
        self.complete_requests: list[ProviderTransportRequest] = []
        self.stream_requests: list[ProviderTransportRequest] = []
        self.token_requests: list[ProviderTransportRequest] = []

    async def complete(self, request: ProviderTransportRequest) -> ProviderResponse:
        self.complete_requests.append(request)
        step = self.complete_steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    async def stream(
        self,
        request: ProviderTransportRequest,
    ) -> AsyncIterator[ProviderEvent]:
        self.stream_requests.append(request)
        step = self.stream_steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        for event in step:
            if isinstance(event, BaseException):
                raise event
            await asyncio.sleep(0)
            yield event

    async def count_tokens(self, request: ProviderTransportRequest) -> int:
        self.token_requests.append(request)
        step = self.token_steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.mark.asyncio
async def test_complete_retries_transient_error_and_succeeds() -> None:
    transport = QueueTransport(
        complete_steps=(
            ProviderPayloadError(
                {
                    "type": "api_error",
                    "status_code": 503,
                    "message": "temporarily unavailable",
                }
            ),
            _anthropic_response("after retry", request_id="req_retry"),
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=2),
    )

    response = await provider.complete(_request())

    assert response.provider_request_id == "req_retry"
    assert len(transport.complete_requests) == 2
    attempts = [
        cast(Mapping[str, object], thaw_json(request.metadata))["attempt"]
        for request in transport.complete_requests
    ]
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_complete_stops_after_retry_exhaustion() -> None:
    error = ProviderPayloadError(
        {
            "type": "api_error",
            "status_code": 503,
            "message": "temporarily unavailable",
        }
    )
    transport = QueueTransport(complete_steps=(error, error))
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=2),
    )

    with pytest.raises(ProviderPayloadError):
        await provider.complete(_request())

    assert len(transport.complete_requests) == 2


@pytest.mark.asyncio
async def test_rate_limit_does_not_retry_unless_policy_allows_it() -> None:
    error = ProviderPayloadError(
        {
            "type": "rate_limit_error",
            "status_code": 429,
            "message": "slow down",
            "retry_after_s": 2.5,
        }
    )
    transport = QueueTransport(
        complete_steps=(error, _anthropic_response("should not be used"))
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=2),
    )

    with pytest.raises(ProviderPayloadError):
        await provider.complete(_request())

    classified = provider.classify_error(error)
    assert classified.kind == "rate_limit"
    assert classified.retry_after_s == 2.5
    assert len(transport.complete_requests) == 1


@pytest.mark.asyncio
async def test_rate_limit_retry_uses_provider_retry_after_when_enabled() -> None:
    sleeps: list[float] = []

    async def record_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    transport = QueueTransport(
        complete_steps=(
            ProviderPayloadError(
                {
                    "type": "rate_limit_error",
                    "status_code": 429,
                    "message": "slow down",
                    "retry_after": "1.75",
                }
            ),
            _anthropic_response("after rate retry"),
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(
            max_attempts=2,
            retry_rate_limit=True,
        ),
        retry_sleep=record_sleep,
    )

    response = await provider.complete(_request())

    assert response.provider_request_id == "req_1"
    assert sleeps == [1.75]
    assert len(transport.complete_requests) == 2


@pytest.mark.asyncio
async def test_auth_config_errors_do_not_retry() -> None:
    transport = QueueTransport(
        complete_steps=(
            ProviderPayloadError(
                {
                    "type": "authentication_error",
                    "status_code": 401,
                    "message": "invalid api key",
                }
            ),
            _anthropic_response("should not be used"),
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=3),
    )

    with pytest.raises(ProviderPayloadError):
        await provider.complete(_request())

    assert len(transport.complete_requests) == 1


@pytest.mark.asyncio
async def test_stream_provider_error_event_propagates_without_runtime_retry() -> None:
    transport = QueueTransport(
        stream_steps=(
            (
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "status_code": 529,
                        "message": "overloaded",
                    },
                },
            ),
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(
            max_attempts=2,
            retry_stream_transport_errors=True,
            fallback_stream_to_complete=True,
        ),
    )

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == ["provider_error"]
    assert events[0].provider_error is not None
    assert events[0].provider_error.kind == "server_overload"
    assert len(transport.stream_requests) == 1
    assert len(transport.complete_requests) == 0


@pytest.mark.asyncio
async def test_stream_transport_error_can_fallback_to_complete_before_yield() -> None:
    transport = QueueTransport(
        stream_steps=(
            ProviderPayloadError(
                {
                    "type": "overloaded_error",
                    "status_code": 529,
                    "message": "stream endpoint overloaded",
                }
            ),
        ),
        complete_steps=(_anthropic_response("non-streaming replacement"),),
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(fallback_stream_to_complete=True),
    )

    events = [event async for event in provider.stream(_request())]
    response = assemble_model_stream(events)

    assert [event.type for event in events] == [
        "streaming_transport_fallback_started",
        "streaming_transport_fallback_completed",
    ]
    assert response.api_message.message.content == (
        TextContentBlock(text="non-streaming replacement"),
    )
    assert len(transport.stream_requests) == 1
    assert len(transport.complete_requests) == 1


@pytest.mark.asyncio
async def test_stream_transport_error_retries_before_visible_event() -> None:
    transport = QueueTransport(
        stream_steps=(
            ProviderPayloadError(
                {
                    "type": "api_error",
                    "status_code": 503,
                    "message": "temporarily unavailable",
                }
            ),
            (
                {
                    "type": "message_start",
                    "request_id": "req_retry_stream",
                    "message": {"id": "msg_retry"},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "after stream retry"},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop"},
            ),
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(
            max_attempts=2,
            retry_stream_transport_errors=True,
        ),
    )

    events = [event async for event in provider.stream(_request())]
    response = assemble_model_stream(events)

    assert response.api_message.message.content == (
        TextContentBlock(text="after stream retry"),
    )
    assert len(transport.stream_requests) == 2
    attempts = [
        cast(Mapping[str, object], thaw_json(request.metadata))["attempt"]
        for request in transport.stream_requests
    ]
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_stream_transport_error_after_visible_event_does_not_fallback() -> None:
    error = ProviderPayloadError(
        {
            "type": "overloaded_error",
            "status_code": 529,
            "message": "stream endpoint overloaded",
        }
    )
    transport = QueueTransport(
        stream_steps=(
            (
                {
                    "type": "message_start",
                    "request_id": "req_partial",
                    "message": {"id": "msg_partial"},
                },
                error,
            ),
        ),
        complete_steps=(_anthropic_response("should not be used"),),
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(
            max_attempts=2,
            retry_stream_transport_errors=True,
            fallback_stream_to_complete=True,
        ),
    )

    events: list[ModelStreamEvent] = []
    with pytest.raises(ProviderPayloadError):
        async for event in provider.stream(_request()):
            events.append(event)

    assert [event.type for event in events] == ["message_start"]
    assert len(transport.stream_requests) == 1
    assert len(transport.complete_requests) == 0


@pytest.mark.asyncio
async def test_count_tokens_retries_transient_error_and_succeeds() -> None:
    transport = QueueTransport(
        token_steps=(
            ProviderPayloadError(
                {
                    "type": "api_error",
                    "status_code": 503,
                    "message": "temporarily unavailable",
                }
            ),
            42,
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=2),
    )

    result = await provider.count_tokens(
        TokenCountRequest(model="claude-test", messages=_request().messages)
    )

    assert result == 42
    assert len(transport.token_requests) == 2
    attempts = [
        cast(Mapping[str, object], thaw_json(request.metadata))["attempt"]
        for request in transport.token_requests
    ]
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_count_tokens_stops_after_retry_exhaustion() -> None:
    error = ProviderPayloadError(
        {
            "type": "api_error",
            "status_code": 503,
            "message": "temporarily unavailable",
        }
    )
    transport = QueueTransport(token_steps=(error, error))
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        retry_policy=ProviderRetryPolicy(max_attempts=2),
    )

    with pytest.raises(ProviderPayloadError):
        await provider.count_tokens(
            TokenCountRequest(model="claude-test", messages=_request().messages)
        )

    assert len(transport.token_requests) == 2


def test_retry_decision_is_pure_and_policy_bounded() -> None:
    decision = classify_retry_decision(
        ProviderError(kind="transient", message="temporary", retryable=True),
        operation="complete",
        attempt=1,
        policy=ProviderRetryPolicy(max_attempts=3, base_delay_s=0.5),
    )
    exhausted = classify_retry_decision(
        ProviderError(kind="transient", message="temporary", retryable=True),
        operation="complete",
        attempt=3,
        policy=ProviderRetryPolicy(max_attempts=3, base_delay_s=0.5),
    )

    assert decision.should_retry is True
    assert decision.delay_s == 0.5
    assert exhausted.should_retry is False
    assert exhausted.reason == "max_attempts_exhausted"
