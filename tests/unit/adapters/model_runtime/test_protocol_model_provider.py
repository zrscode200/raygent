from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
    PreparedModelRequest,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.adapters.model_runtime import (
    ProtocolModelProvider,
    ProviderPayloadError,
    ProviderTransportRequest,
)
from raygent_harness.core.messages import message_param_from_api_message, thaw_json
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    CachePolicy,
    ModelBudgetSnapshot,
    ModelCapabilities,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelResolveContext,
    ModelSampling,
    ModelStreamEvent,
    PermissionContextSnapshot,
    TaskBudgetInfo,
    TextContentBlock,
    TokenCountRequest,
    TokenCountResult,
)


def _request(*, abort_event: asyncio.Event | None = None) -> ModelRequest:
    return ModelRequest(
        model="model-alias",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(TextContentBlock(text="hello"),),
                )
            ),
        ),
        sampling=ModelSampling(max_tokens=123),
        fallback_model="fallback-model",
        effort="high",
        agent_id="agent_1",
        query_source="user",
        tool_choice="auto",
        max_output_tokens_override=456,
        task_budget=TaskBudgetInfo(total=1000, remaining=750),
        abort_event=abort_event,
        permission_context=PermissionContextSnapshot(
            mode="default",
            always_allow_rules={"project": ("Read(*)",)},
            should_avoid_permission_prompts=True,
        ),
        allowed_agent_types=("general",),
        mcp_tool_names=("mcp__search",),
        has_pending_mcp_servers=True,
        cache_policy=CachePolicy(skip_cache_write=True, cache_scope="ephemeral"),
        budget=ModelBudgetSnapshot(
            requested_model="model-alias",
            effective_model="anthropic-real",
            context_window=200000,
            default_max_output_tokens=32000,
            upper_max_output_tokens=64000,
            requested_max_tokens=456,
            effective_max_tokens=456,
            input_token_count=12,
            provider_input_token_count=12,
            token_count_fallback_used=False,
        ),
    )


def _body(prepared: PreparedModelRequest) -> Mapping[str, object]:
    body = thaw_json(prepared.body)
    assert isinstance(body, Mapping)
    return cast(Mapping[str, object], body)


class FakeTransport:
    def __init__(
        self,
        *,
        response: ProviderResponse | None = None,
        events: tuple[ProviderEvent, ...] = (),
        token_count: int | TokenCountResult = 0,
        abort_after_events: int | None = None,
    ) -> None:
        self.response = response or {}
        self.events = events
        self.token_count = token_count
        self.abort_after_events = abort_after_events
        self.complete_requests: list[ProviderTransportRequest] = []
        self.stream_requests: list[ProviderTransportRequest] = []
        self.token_requests: list[ProviderTransportRequest] = []

    async def complete(self, request: ProviderTransportRequest) -> ProviderResponse:
        self.complete_requests.append(request)
        return self.response

    async def stream(
        self,
        request: ProviderTransportRequest,
    ) -> AsyncIterator[ProviderEvent]:
        self.stream_requests.append(request)
        for index, event in enumerate(self.events, start=1):
            if (
                self.abort_after_events is not None
                and index > self.abort_after_events
                and request.abort_event is not None
            ):
                request.abort_event.set()
            await asyncio.sleep(0)
            yield event

    async def count_tokens(self, request: ProviderTransportRequest) -> int | TokenCountResult:
        self.token_requests.append(request)
        return self.token_count


@pytest.mark.asyncio
async def test_complete_uses_non_stream_request_and_protocol_parser() -> None:
    transport = FakeTransport(
        response={
            "id": "msg_1",
            "request_id": "req_1",
            "content": [{"type": "text", "text": "hello back"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        models=(
            ModelInfo(
                model="anthropic-real",
                aliases=("model-alias",),
                capabilities=ModelCapabilities(supports_streaming=True),
            ),
        ),
    )

    response = await provider.complete(_request())

    assert message_param_from_api_message(response.api_message) == {
        "role": "assistant",
        "id": "msg_1",
        "content": "hello back",
    }
    assert response.provider_request_id == "req_1"
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2
    assert len(transport.complete_requests) == 1
    transport_request = transport.complete_requests[0]
    prepared = transport_request.prepared_request
    assert thaw_json(prepared.options) == {"operation": "messages", "mode": "complete"}
    body = _body(prepared)
    assert "stream" not in body
    assert body["max_tokens"] == 456
    metadata = thaw_json(transport_request.metadata)
    assert isinstance(metadata, Mapping)
    assert metadata["operation"] == "complete"
    assert metadata["query_source"] == "user"
    assert metadata["agent_id"] == "agent_1"
    assert metadata["fallback_model"] == "fallback-model"
    assert metadata["tool_count"] == 0
    assert metadata["mcp_tool_count"] == 1
    assert metadata["has_pending_mcp_servers"] is True
    assert metadata["allowed_agent_types"] == ["general"]
    assert metadata["cache_policy"] == {
        "skip_cache_write": True,
        "cache_scope": "ephemeral",
    }
    permission = cast(Mapping[str, object], metadata["permission_context"])
    assert permission == {
        "mode": "default",
        "always_allow_rule_source_count": 1,
        "always_deny_rule_source_count": 0,
        "always_ask_rule_source_count": 0,
        "should_avoid_permission_prompts": True,
        "is_bypass_permissions_mode_available": False,
        "is_auto_mode_available": False,
    }
    raw_metadata = cast(Mapping[str, object], transport_request.metadata)
    with pytest.raises(TypeError):
        cast(Any, raw_metadata)["operation"] = "mutated"


def test_openai_stateful_parser_preserves_incremental_tool_arguments() -> None:
    adapter = OpenAIResponsesAdapter()
    parser = adapter.create_stream_parser()
    events: list[ModelStreamEvent] = []
    for provider_event in (
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Search",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"query"',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": ':"raygent"}',
        },
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1"}},
        {"type": "response.completed", "response": {"id": "resp_1"}},
    ):
        events.extend(parser.feed(provider_event))
    events.extend(parser.finish())

    response = assemble_model_stream(events)

    assert response.tool_uses[0].id == "call_1"
    assert cast(Mapping[str, object], response.tool_uses[0].input) == {"query": "raygent"}


@pytest.mark.asyncio
async def test_stream_uses_stateful_parser_and_preserves_abort_responsiveness() -> None:
    abort_event = asyncio.Event()
    transport = FakeTransport(
        events=(
            {"type": "response.created", "response": {"id": "resp_abort"}},
            {"type": "response.output_text.delta", "item_id": "txt", "delta": "stale"},
        ),
        abort_after_events=1,
    )
    provider = ProtocolModelProvider(adapter=OpenAIResponsesAdapter(), transport=transport)

    yielded: list[ModelStreamEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in provider.stream(_request(abort_event=abort_event)):
            yielded.append(event)

    assert [event.type for event in yielded] == ["message_start"]
    assert len(transport.stream_requests) == 1
    transport_request = transport.stream_requests[0]
    prepared = transport_request.prepared_request
    assert thaw_json(prepared.options) == {"operation": "responses", "mode": "stream"}
    assert _body(prepared)["stream"] is True
    metadata = thaw_json(transport_request.metadata)
    assert isinstance(metadata, Mapping)
    assert metadata["operation"] == "stream"


@pytest.mark.asyncio
async def test_count_tokens_returns_exact_count_and_preserves_registry_lookup() -> None:
    transport = FakeTransport(token_count=17)
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        models=(ModelInfo(model="real-model", aliases=("model-alias",), context_window=321),),
        timeout_s=9.5,
    )
    token_request = TokenCountRequest(
        model="real-model",
        messages=_request().messages,
    )

    assert await provider.count_tokens(token_request) == 17
    assert len(transport.token_requests) == 1
    token_transport_request = transport.token_requests[0]
    assert token_transport_request.timeout_s == 9.5
    token_metadata = thaw_json(token_transport_request.metadata)
    assert isinstance(token_metadata, Mapping)
    assert token_metadata["operation"] == "count_tokens"
    assert token_metadata["message_count"] == 1
    assert token_metadata["tool_count"] == 0
    assert provider.resolve_model("model-alias", ModelResolveContext()) == "real-model"
    assert provider.model_info("model-alias").context_window == 321


@pytest.mark.asyncio
async def test_count_tokens_accepts_exact_result_metadata_without_raw_content() -> None:
    transport = FakeTransport(
        token_count=TokenCountResult(
            token_count=23,
            provider_request_id="count_req_1",
            safe_metadata={"cache": "miss"},
        )
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=transport,
        timeout_s=3.0,
    )
    token_request = TokenCountRequest(
        model="claude-test",
        messages=_request().messages,
        system_prompt="SECRET_SYSTEM",
        effort="high",
        media_context={"images": 1},
    )

    result = await provider.count_tokens(token_request)

    assert isinstance(result, TokenCountResult)
    assert result.token_count == 23
    assert result.provider_request_id == "count_req_1"
    assert thaw_json(result.safe_metadata) == {"cache": "miss"}
    transport_request = transport.token_requests[0]
    raw_metadata = thaw_json(transport_request.metadata)
    assert isinstance(raw_metadata, Mapping)
    metadata = cast(Mapping[str, object], raw_metadata)
    assert metadata["operation"] == "count_tokens"
    assert metadata["system_prompt_char_count"] == len("SECRET_SYSTEM")
    assert metadata["has_media_context"] is True
    assert "SECRET_SYSTEM" not in str(metadata)


def test_provider_payload_error_delegates_to_protocol_classifier() -> None:
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=FakeTransport(),
    )

    classified = provider.classify_error(
        ProviderPayloadError(
            {
                "type": "invalid_request_error",
                "status_code": 400,
                "message": "prompt is too long: 137500 tokens > 135000 maximum",
            }
        )
    )

    assert classified.kind == "context_overflow"
    assert classified.actual_tokens == 137_500
    assert classified.limit_tokens == 135_000
