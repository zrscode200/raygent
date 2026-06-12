from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
)
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_request_normalization import (
    ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER,
    TOOL_REFERENCES_REMOVED_PLACEHOLDER,
    UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER,
    normalize_model_request_for_provider,
)
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelCapabilities,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelSampling,
    ModelToolSpec,
    TextContentBlock,
    ThinkingContentBlock,
    TokenCountRequest,
    ToolResultContentBlock,
    ToolUseContentBlock,
    Usage,
)


def _body(prepared: object) -> Mapping[str, object]:
    raw = thaw_json(cast(Any, prepared).body)
    assert isinstance(raw, Mapping)
    return cast(Mapping[str, object], raw)


def _sample_request() -> ModelRequest:
    return ModelRequest(
        model="gpt-test",
        system_prompt="system prompt",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        TextContentBlock(text="see image"),
                        MediaContentBlock(
                            media_kind="image",
                            media_type="image/png",
                            data={"source": {"type": "base64", "data": "iVBORw0KGgo="}},
                        ),
                    ),
                )
            ),
            ApiMessage(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        TextContentBlock(text="I will search."),
                        ThinkingContentBlock(
                            text="reasoning summary",
                            redacted=True,
                            provider_metadata={
                                "openai": {
                                    "itemId": "rs_1",
                                    "reasoningEncryptedContent": "sealed",
                                }
                            },
                        ),
                        ToolUseContentBlock(
                            id="call_1",
                            name="Search",
                            input={"query": "raygent"},
                        ),
                    ),
                )
            ),
            ApiMessage(
                message=ModelMessage(
                    role="tool",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="call_1",
                            content=cast(
                                FrozenJson,
                                [
                                    {"type": "text", "text": "found"},
                                    {
                                        "type": "image",
                                        "media_type": "image/png",
                                        "source": {"type": "base64", "data": "abc"},
                                    },
                                ],
                            ),
                        ),
                    ),
                )
            ),
        ),
        tools=(
            ModelToolSpec(
                name="Search",
                description="search docs",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ),
        sampling=ModelSampling(max_tokens=512, temperature=0.4, top_p=0.8),
        tool_choice="tool:Search",
        effort="high",
        provider_options={
            "openai": {
                "headers": {"openai-beta": "responses=v1"},
                "store": False,
                "prompt_cache_key": "cache-key",
                "encrypted_reasoning": True,
                "reasoning_summary": "auto",
                "text_verbosity": "low",
            }
        },
    )


def test_prepare_request_lowers_messages_tools_options_and_reasoning() -> None:
    adapter = OpenAIResponsesAdapter()

    prepared = adapter.prepare_request(_sample_request())
    body = _body(prepared)

    assert prepared.protocol_id == "openai_responses"
    assert prepared.model == "gpt-test"
    assert thaw_json(prepared.headers) == {"openai-beta": "responses=v1"}
    assert body["stream"] is True
    assert body["max_output_tokens"] == 512
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.8
    assert body["tool_choice"] == {"type": "function", "name": "Search"}
    assert body["store"] is False
    assert body["prompt_cache_key"] == "cache-key"
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["text"] == {"verbosity": "low"}
    assert body["tools"] == [
        {
            "type": "function",
            "name": "Search",
            "description": "search docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    input_items = cast(list[dict[str, object]], body["input"])
    assert input_items[0] == {"role": "system", "content": "system prompt"}
    assert input_items[1] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "see image"},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
        ],
    }
    assert input_items[2] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I will search."}],
    }
    assert input_items[3] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "reasoning summary"}],
        "encrypted_content": "sealed",
    }
    assert input_items[4] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "Search",
        "arguments": '{"query":"raygent"}',
    }
    assert input_items[5] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "found"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ],
    }


def test_prepare_request_uses_api_bound_tool_reference_stripping() -> None:
    adapter = OpenAIResponsesAdapter()
    request = ModelRequest(
        model="gpt-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        ToolUseContentBlock(
                            id="call_search",
                            name="ToolSearch",
                            input={"query": "Read"},
                            provider_metadata={"caller": {"type": "tool_search"}},
                        ),
                    ),
                )
            ),
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="call_search",
                            content=cast(
                                FrozenJson,
                                [{"type": "tool_reference", "tool_name": "Read"}],
                            ),
                        ),
                    ),
                )
            ),
        ),
    )

    normalized = normalize_model_request_for_provider(
        request,
        model_info=ModelInfo(model="gpt-test"),
    )
    body = _body(adapter.prepare_request(normalized))

    input_items = cast(list[dict[str, object]], body["input"])
    assert input_items[0]["type"] == "function_call"
    assert input_items[1] == {
        "type": "function_call_output",
        "call_id": "call_search",
        "output": [
            {
                "type": "input_text",
                "text": TOOL_REFERENCES_REMOVED_PLACEHOLDER,
            }
        ],
    }


def test_prepare_request_lowers_leading_orphan_tool_role_placeholder_as_user() -> None:
    adapter = OpenAIResponsesAdapter()
    request = ModelRequest(
        model="gpt-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="tool",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="call_orphan",
                            content="stale",
                        ),
                    ),
                )
            ),
        ),
    )

    normalized = normalize_model_request_for_provider(
        request,
        model_info=ModelInfo(model="gpt-test"),
    )
    body = _body(adapter.prepare_request(normalized))

    input_items = cast(list[dict[str, object]], body["input"])
    assert input_items == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER,
                }
            ],
        }
    ]


def test_stream_events_assemble_text_reasoning_tool_call_and_usage() -> None:
    adapter = OpenAIResponsesAdapter()
    provider_events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_text.delta", "item_id": "msg_text", "delta": "I will "},
        {"type": "response.output_text.delta", "item_id": "msg_text", "delta": "search."},
        {"type": "response.output_text.done", "item_id": "msg_text"},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Search",
            },
        },
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"query"'},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": ':"raygent"}',
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Search",
                "arguments": '{"query":"raygent"}',
            },
        },
        {
            "type": "response.reasoning_summary.delta",
            "item_id": "rs_1",
            "delta": "plan",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "ignored"}],
                "encrypted_content": "sealed-by-provider",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "usage": {
                    "input_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 11,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 25,
                },
            },
        },
    ]

    events = list(adapter.stream_events(provider_events))
    response = assemble_model_stream(events)

    assert response.provider_request_id == "resp_1"
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(
        input_tokens=10,
        output_tokens=11,
        cache_read_input_tokens=4,
        reasoning_tokens=3,
        total_tokens=25,
        provider_metadata={
            "openai": {
                "input_tokens": 14,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 11,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 25,
            }
        },
    )
    text = cast(TextContentBlock, response.api_message.message.content[0])
    tool = cast(ToolUseContentBlock, response.api_message.message.content[1])
    thinking = cast(ThinkingContentBlock, response.api_message.message.content[2])

    assert text.text == "I will search."
    assert tool.id == "call_1"
    assert tool.name == "Search"
    assert cast(Mapping[str, object], tool.input)["query"] == "raygent"
    assert thinking.text == "plan"
    assert thinking.redacted is True
    assert thinking.provider_metadata is not None
    assert cast(Mapping[str, object], thinking.provider_metadata) == {
        "openai": {
            "itemId": "rs_1",
            "reasoningEncryptedContent": "sealed-by-provider",
        }
    }
    assert response.tool_uses[0].id == "call_1"
    reasoning_delta_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "content_block_delta"
        and isinstance(event.delta, Mapping)
        and event.delta.get("type") == "thinking_delta"
    )
    reasoning_stop_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "content_block_stop" and event.identity.content_block_index == 2
    )
    assert reasoning_delta_index < reasoning_stop_index


def test_stream_events_done_only_function_call_reports_tool_use_stop_reason() -> None:
    adapter = OpenAIResponsesAdapter()
    provider_events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Search",
                "arguments": '{"query":"raygent"}',
            },
        },
        {"type": "response.completed", "response": {"id": "resp_1"}},
    ]

    response = assemble_model_stream(adapter.stream_events(provider_events))

    assert response.stop_reason == "tool_use"
    assert response.tool_uses[0].id == "call_1"
    assert response.tool_uses[0].name == "Search"
    assert cast(Mapping[str, object], response.tool_uses[0].input)["query"] == "raygent"


def test_stream_events_preserve_hosted_tool_as_provider_executed() -> None:
    adapter = OpenAIResponsesAdapter()
    provider_events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"query": "raygent"},
                "status": "completed",
                "results": [{"title": "Raygent"}],
            },
        },
        {"type": "response.completed", "response": {"id": "resp_1"}},
    ]

    response = assemble_model_stream(adapter.stream_events(provider_events))

    tool = cast(ToolUseContentBlock, response.api_message.message.content[0])
    result = cast(ToolResultContentBlock, response.api_message.message.content[1])
    assert tool.provider_executed is True
    assert tool.name == "web_search"
    assert response.tool_uses == ()
    assert result.tool_use_id == "ws_1"
    assert result.provider_metadata is not None


def test_classify_error_maps_openai_recovery_categories() -> None:
    adapter = OpenAIResponsesAdapter()

    context = adapter.classify_error(
        {
            "status_code": 400,
            "code": "context_length_exceeded",
            "message": (
                "This model's maximum context length is 128000 tokens. "
                "You requested 130000 tokens."
            ),
        }
    )
    media = adapter.classify_error(
        {"status": 400, "code": "invalid_image", "message": "image input too large"}
    )
    unsupported_image = adapter.classify_error(
        {
            "status": 400,
            "code": "invalid_image",
            "message": "unsupported image format",
        }
    )
    rate = adapter.classify_error(
        {"status": 429, "code": "rate_limit_exceeded", "message": "slow down"}
    )
    overload = adapter.classify_error(
        {"status": 503, "code": "server_overloaded", "message": "capacity"}
    )

    assert context.kind == "context_overflow"
    assert context.actual_tokens == 130000
    assert context.limit_tokens == 128000
    assert context.api_error is not None
    assert media.kind == "media_overflow"
    assert media.api_error is not None
    assert unsupported_image.kind == "fatal_unknown"
    assert unsupported_image.api_error is None
    assert rate.kind == "rate_limit"
    assert rate.retryable is True
    assert overload.kind == "server_overload"
    assert overload.safe_to_fallback is True


def test_classify_error_preserves_retry_after_from_outer_headers() -> None:
    adapter = OpenAIResponsesAdapter()

    error = adapter.classify_error(
        {
            "status": 429,
            "headers": {"retry-after": "4"},
            "error": {
                "code": "rate_limit_exceeded",
                "message": "slow down",
            },
        }
    )

    assert error.kind == "rate_limit"
    assert error.retry_after_s == 4.0


def test_prepare_request_rejects_unsupported_user_media() -> None:
    adapter = OpenAIResponsesAdapter()
    request = ModelRequest(
        model="gpt-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        MediaContentBlock(
                            media_kind="document",
                            media_type="application/pdf",
                            data={"source": {"type": "file_id", "file_id": "file_1"}},
                        ),
                    ),
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="only supports images"):
        adapter.prepare_request(request)


def test_prepare_request_accepts_api_bound_unsupported_media_placeholder() -> None:
    adapter = OpenAIResponsesAdapter()
    request = ModelRequest(
        model="gpt-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        MediaContentBlock(
                            media_kind="document",
                            media_type="application/pdf",
                            data={"source": {"type": "base64", "data": "pdf"}},
                        ),
                    ),
                )
            ),
        ),
    )
    normalized = normalize_model_request_for_provider(
        request,
        model_info=ModelInfo(
            model="gpt-test",
            capabilities=ModelCapabilities(),
        ),
    )

    body = _body(adapter.prepare_request(normalized))

    input_items = cast(list[dict[str, object]], body["input"])
    assert input_items[0] == {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER,
            }
        ],
    }


def test_prepare_token_count_lowers_generation_free_body() -> None:
    adapter = OpenAIResponsesAdapter()
    request = TokenCountRequest(
        model="gpt-test",
        system_prompt="system prompt",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(TextContentBlock(text="count me"),),
                )
            ),
        ),
        tools=(
            ModelToolSpec(
                name="Search",
                description="search docs",
                input_schema={"type": "object"},
            ),
        ),
        effort="low",
        media_context={"images": 1},
        provider_options={"openai": {"headers": {"x-test": "1"}, "reasoning_summary": "auto"}},
    )

    prepared = adapter.prepare_token_count(request)
    body = _body(prepared)

    assert "stream" not in body
    assert "max_output_tokens" not in body
    assert body["model"] == "gpt-test"
    input_items = cast(list[Mapping[str, object]], body["input"])
    assert input_items[0] == {"role": "system", "content": "system prompt"}
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["media_context"] == {"images": 1}
    assert thaw_json(prepared.headers) == {"x-test": "1"}
    options = thaw_json(prepared.options)
    assert isinstance(options, Mapping)
    assert options["operation"] == "count_tokens"


def test_same_raygent_request_lowers_to_materially_different_protocol_bodies() -> None:
    request = _sample_request()

    openai_body = _body(OpenAIResponsesAdapter().prepare_request(request))
    anthropic_body = _body(AnthropicMessagesAdapter().prepare_request(request))

    assert "input" in openai_body
    assert "messages" in anthropic_body
    openai_tools = cast(list[dict[str, object]], openai_body["tools"])
    anthropic_tools = cast(list[dict[str, object]], anthropic_body["tools"])

    assert "tools" in openai_body and openai_tools[0]["type"] == "function"
    assert "tools" in anthropic_body and "input_schema" in anthropic_tools[0]
    assert openai_body != anthropic_body
