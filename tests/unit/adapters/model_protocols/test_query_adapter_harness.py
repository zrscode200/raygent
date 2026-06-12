from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
)
from raygent_harness.adapters.model_protocols.base import (
    ModelProtocolAdapter,
    PreparedModelRequest,
    ProviderEvent,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam, message_param_from_api_message, thaw_json
from raygent_harness.core.model_registry import count_message_tokens
from raygent_harness.core.model_types import (
    ModelCapabilities,
    ModelInfo,
    ModelToolSpec,
)
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKResult,
    SDKUserMessage,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptScope,
    load_session_replay,
)
from tests.fakes import AdapterBackedFakeModelProvider


def _openai_adapter() -> ModelProtocolAdapter:
    return cast(ModelProtocolAdapter, OpenAIResponsesAdapter())


def _anthropic_adapter() -> ModelProtocolAdapter:
    return cast(ModelProtocolAdapter, AnthropicMessagesAdapter())


class ExampleInput(BaseModel):
    value: int = 0


async def _example_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="tool done")


def _example_tool() -> Any:
    return build_tool(
        ToolSpec(
            name="Example",
            description="Example tool",
            input_model=ExampleInput,
            call=_example_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _config(*, tools: Sequence[Any] = ()) -> QueryConfig:
    return QueryConfig(
        model="gpt-test",
        session_id="s",
        system_prompt="system",
        tools=tuple(tools),
        experiments={"streaming_tool_execution": True},
    )


async def _run_engine(
    provider: AdapterBackedFakeModelProvider,
    *,
    prompt: str | MessageParam = "hi",
    config: QueryConfig | None = None,
    transcript_store: JsonlTranscriptStore | None = None,
) -> tuple[QueryEngine, list[Any]]:
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        transcript_store=transcript_store,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    engine = QueryEngine(config or _config(), deps, _ctx())
    events = [event async for event in engine.submit_message(prompt)]
    return engine, events


def _prepared_body(prepared: PreparedModelRequest) -> Mapping[str, object]:
    raw = thaw_json(prepared.body)
    assert isinstance(raw, Mapping)
    return cast(Mapping[str, object], raw)


def _openai_tool_call_stream() -> tuple[ProviderEvent, ...]:
    return (
        {"type": "response.created", "response": {"id": "resp_tool"}},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Example",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"value":7}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Example",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_tool",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        },
    )


def _openai_text_stream(text: str) -> tuple[ProviderEvent, ...]:
    return (
        {"type": "response.created", "response": {"id": "resp_done"}},
        {"type": "response.output_text.delta", "item_id": "text_1", "delta": text},
        {"type": "response.output_text.done", "item_id": "text_1"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_done",
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        },
    )


@pytest.mark.asyncio
async def test_query_streaming_tool_execution_uses_adapter_normalized_events() -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        provider_event_batches=(
            _openai_tool_call_stream(),
            _openai_text_stream("done"),
        ),
        model_infos={
            "gpt-test": ModelInfo(
                model="gpt-test",
                capabilities=ModelCapabilities(supports_streaming=True),
            )
        },
    )

    engine, events = await _run_engine(provider, config=_config(tools=(_example_tool(),)))

    assert [request.model for request in provider.stream_requests] == [
        "gpt-test",
        "gpt-test",
    ]
    assert [prepared.protocol_id for prepared in provider.prepared_requests] == [
        "openai_responses",
        "openai_responses",
    ]
    first_body = _prepared_body(provider.prepared_requests[0])
    assert cast(list[dict[str, object]], first_body["tools"])[0]["name"] == "Example"

    second_body = _prepared_body(provider.prepared_requests[1])
    second_input = cast(list[dict[str, object]], second_body["input"])
    assert any(
        item.get("type") == "function_call_output"
        and item.get("call_id") == "call_1"
        and item.get("output") == "tool done"
        for item in second_input
    )

    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    user_events = [event for event in events if isinstance(event, SDKUserMessage)]
    assert len(assistant_events) == 2
    assert len(user_events) == 1
    first_assistant = assistant_events[0].message
    assert first_assistant.get("id") == "resp_tool"
    first_content = cast(list[dict[str, object]], first_assistant["content"])
    assert first_content[0]["type"] == "tool_use"
    assert first_content[0]["id"] == "call_1"
    assert first_content[0]["input"] == {"value": 7}
    assert user_events[0].message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "tool done",
            }
        ],
    }
    assert assistant_events[-1].message == {
        "role": "assistant",
        "content": "done",
        "id": "resp_done",
    }
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert engine._messages[-1] == assistant_events[-1].message  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_query_complete_path_returns_adapter_assembled_model_response() -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        complete_event_batches=(_openai_text_stream("complete answer"),),
    )

    engine, events = await _run_engine(
        provider,
        config=QueryConfig(
            model="gpt-test",
            session_id="s",
            system_prompt="system",
        ),
    )

    assert len(provider.complete_requests) == 1
    assert provider.stream_requests == []
    assert provider.prepared_requests[0].protocol_id == "openai_responses"
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events == [
        SDKAssistantMessage(
            session_id="s",
            message={
                "role": "assistant",
                "content": "complete answer",
                "id": "resp_done",
            },
        )
    ]
    assert engine._messages[-1] == assistant_events[0].message  # pyright: ignore[reportPrivateUsage]
    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"


@pytest.mark.asyncio
async def test_adapter_token_count_lowers_tools_thinking_and_media_context() -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_anthropic_adapter(),
        token_counts=(321,),
    )
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "media_type": "image/png",
                    "source": {"type": "base64", "data": "abc"},
                },
            ],
        }
    ]
    tool = ModelToolSpec(
        name="Inspect",
        description="inspect image",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    count = await count_message_tokens(
        provider=provider,
        model="claude-test",
        messages=messages,
        system_prompt="system prompt",
        tools=(tool,),
        thinking={"type": "enabled", "budget_tokens": 2048},
        effort="high",
        media_context={"images": 1},
        provider_options={"anthropic": {"headers": {"anthropic-beta": "test"}}},
        fallback_estimator=lambda _messages: 0,
    )

    assert count == 321
    assert len(provider.prepared_token_requests) == 1
    prepared = provider.prepared_token_requests[0]
    body = _prepared_body(prepared)
    options = cast(Mapping[str, object], thaw_json(prepared.options))
    assert options["operation"] == "count_tokens"
    assert "provider_options" in options
    assert "stream" not in body
    assert "max_tokens" not in body
    assert body["system"] == [{"type": "text", "text": "system prompt"}]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert cast(list[dict[str, object]], body["tools"])[0]["name"] == "Inspect"
    content = cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], body["messages"])[0]["content"],
    )
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "data": "abc"},
    }


def _openai_error_stream(
    *,
    code: str,
    message: str,
    status: int = 400,
) -> tuple[ProviderEvent, ...]:
    return (
        {
            "type": "response.failed",
            "response": {
                "id": "resp_error",
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": message,
                    "status_code": status,
                },
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message", "expected_reason", "api_error"),
    [
        (
            "context_length_exceeded",
            "This model's maximum context length is 128000 tokens. "
            "However, your messages resulted in 130000 tokens.",
            "error_during_execution",
            "context_overflow",
        ),
        (
            "image_too_large",
            "Image input exceeds the maximum allowed size.",
            "error_during_execution",
            "media_overflow",
        ),
    ],
)
async def test_adapter_provider_errors_enter_terminal_recovery_ladder(
    code: str,
    message: str,
    expected_reason: str,
    api_error: str,
) -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        provider_event_batches=(_openai_error_stream(code=code, message=message),),
    )

    engine, events = await _run_engine(provider)

    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert len(assistant_events) == 1
    assert assistant_events[0].message.get("isApiErrorMessage") is True
    assert assistant_events[0].message.get("apiError") == api_error
    assert engine._messages[-1].get("apiError") == api_error  # pyright: ignore[reportPrivateUsage]
    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == expected_reason
    assert result.is_error is True


@pytest.mark.asyncio
async def test_adapter_max_output_error_recovers_and_retries_with_resume_instruction() -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        provider_event_batches=(
            _openai_error_stream(
                code="max_output_tokens",
                message="Response hit max_output_tokens.",
            ),
            _openai_text_stream("resumed"),
        ),
    )

    engine, events = await _run_engine(provider)

    assert len(provider.stream_requests) == 2
    retry_messages = [
        message_param_from_api_message(message)
        for message in provider.stream_requests[1].messages
    ]
    assert retry_messages[-2].get("apiError") == "max_output_tokens"
    assert "Resume directly" in str(retry_messages[-1]["content"])
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[-1].message == {
        "role": "assistant",
        "content": "resumed",
        "id": "resp_done",
    }
    assert engine._messages[-1] == assistant_events[-1].message  # pyright: ignore[reportPrivateUsage]
    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"


def _openai_reasoning_hosted_tool_stream() -> tuple[ProviderEvent, ...]:
    return (
        {"type": "response.created", "response": {"id": "resp_rich"}},
        {
            "type": "response.reasoning_summary.delta",
            "item_id": "rs_1",
            "delta": "checked memory",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "checked memory"}],
                "encrypted_content": "sealed-reasoning",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"query": "raygent"},
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "text_1",
            "delta": "answer",
        },
        {"type": "response.output_text.done", "item_id": "text_1"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_rich",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 31,
                    "output_tokens_details": {"reasoning_tokens": 3},
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_query_transcript_preserves_adapter_reasoning_media_and_tool_result_content(
    tmp_path: Path,
) -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        provider_event_batches=(_openai_reasoning_hosted_tool_stream(),),
        model_infos={
            "gpt-test": ModelInfo(
                model="gpt-test",
                capabilities=ModelCapabilities(supports_images=True),
            )
        },
    )
    store = JsonlTranscriptStore(tmp_path)
    prompt: MessageParam = {
        "role": "user",
        "content": [
            {"type": "text", "text": "read this image"},
            {
                "type": "image",
                "media_type": "image/png",
                "source": {"type": "base64", "data": "abc"},
            },
        ],
    }

    engine, events = await _run_engine(
        provider,
        prompt=prompt,
        transcript_store=store,
    )

    prepared_body = _prepared_body(provider.prepared_requests[0])
    input_items = cast(list[dict[str, object]], prepared_body["input"])
    user_content = cast(list[dict[str, object]], input_items[1]["content"])
    assert user_content[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc",
    }
    assert not any(isinstance(event, SDKUserMessage) for event in events)

    assistant = next(event for event in events if isinstance(event, SDKAssistantMessage))
    blocks = cast(list[dict[str, object]], assistant.message["content"])
    assert [block["type"] for block in blocks] == [
        "thinking",
        "server_tool_use",
        "tool_result",
        "text",
    ]
    thinking_metadata = cast(Mapping[str, object], blocks[0]["provider_metadata"])
    assert cast(Mapping[str, object], thinking_metadata["openai"])[
        "reasoningEncryptedContent"
    ] == "sealed-reasoning"
    assert blocks[1]["provider_executed"] is True
    assert blocks[2]["tool_use_id"] == "ws_1"

    assert engine._messages == [prompt, assistant.message]  # pyright: ignore[reportPrivateUsage]
    replay = await load_session_replay(store, TranscriptScope(session_id="s"))
    assert replay.messages == engine._messages  # pyright: ignore[reportPrivateUsage]
    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"
