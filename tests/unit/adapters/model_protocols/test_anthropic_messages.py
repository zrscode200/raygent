from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from raygent_harness.adapters.model_protocols import AnthropicMessagesAdapter
from raygent_harness.core.messages import (
    message_param_from_api_message,
    model_response_from_message_param,
    thaw_json,
)
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelMessage,
    ModelRequest,
    ModelSampling,
    ModelToolSpec,
    TextContentBlock,
    ThinkingContentBlock,
    TokenCountRequest,
    ToolResultContentBlock,
    ToolUseContentBlock,
)


def _body(prepared: object) -> Mapping[str, object]:
    raw = thaw_json(cast(Any, prepared).body)
    assert isinstance(raw, Mapping)
    return cast(Mapping[str, object], raw)


def test_prepare_request_lowers_messages_tools_sampling_and_thinking() -> None:
    adapter = AnthropicMessagesAdapter()
    request = ModelRequest(
        model="claude-test",
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
                            data={
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgo=",
                                }
                            },
                        ),
                    ),
                )
            ),
            ApiMessage(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        ThinkingContentBlock(text="plan", signature="sig_1"),
                        ToolUseContentBlock(
                            id="toolu_1",
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
                            tool_use_id="toolu_1",
                            content=cast(FrozenJson, [
                                {"type": "text", "text": "found"},
                                {
                                    "type": "image",
                                    "media_type": "image/png",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ]),
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
        sampling=ModelSampling(
            max_tokens=1024,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            stop_sequences=("STOP",),
        ),
        tool_choice="tool:Search",
        provider_options={
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "headers": {"anthropic-beta": "tools-2024"},
            }
        },
    )

    prepared = adapter.prepare_request(request)
    body = _body(prepared)

    assert prepared.protocol_id == "anthropic_messages"
    assert prepared.model == "claude-test"
    assert thaw_json(prepared.headers) == {"anthropic-beta": "tools-2024"}
    assert body["system"] == [{"type": "text", "text": "system prompt"}]
    assert body["stream"] is True
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["stop_sequences"] == ["STOP"]
    assert body["tool_choice"] == {"type": "tool", "name": "Search"}
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["tools"] == [
        {
            "name": "Search",
            "description": "search docs",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    messages = cast(list[dict[str, object]], body["messages"])
    assert messages[0]["role"] == "user"
    assert cast(list[dict[str, object]], messages[0]["content"])[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo=",
        },
    }
    assert cast(list[dict[str, object]], messages[1]["content"])[0] == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig_1",
    }
    assert cast(list[dict[str, object]], messages[2]["content"])[0]["content"] == [
        {"type": "text", "text": "found"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc",
            },
        },
    ]


def test_prepare_request_lowers_structured_tool_result_json_to_text() -> None:
    adapter = AnthropicMessagesAdapter()
    request = ModelRequest(
        model="claude-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="tool",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="toolu_json",
                            content={"status": "ok", "count": 2},
                        ),
                    ),
                )
            ),
        ),
    )

    body = _body(adapter.prepare_request(request))
    messages = cast(list[dict[str, object]], body["messages"])
    tool_result = cast(list[dict[str, object]], messages[0]["content"])[0]

    assert tool_result["content"] == '{"count":2,"status":"ok"}'


def test_prepare_request_preserves_provider_native_tool_references() -> None:
    adapter = AnthropicMessagesAdapter()
    request = ModelRequest(
        model="claude-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        ToolUseContentBlock(
                            id="toolu_search",
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
                            tool_use_id="toolu_search",
                            content=cast(
                                FrozenJson,
                                [
                                    {"type": "text", "text": "selected"},
                                    {"type": "tool_reference", "tool_name": "Read"},
                                ],
                            ),
                            provider_metadata={"cache_reference": "toolu_search"},
                        ),
                    ),
                )
            ),
        ),
    )

    body = _body(adapter.prepare_request(request))
    messages = cast(list[dict[str, object]], body["messages"])
    tool_use = cast(list[dict[str, object]], messages[0]["content"])[0]
    tool_result = cast(list[dict[str, object]], messages[1]["content"])[0]

    assert tool_use["caller"] == {"type": "tool_search"}
    assert tool_result["cache_reference"] == "toolu_search"
    assert tool_result["content"] == [
        {"type": "text", "text": "selected"},
        {"type": "tool_reference", "tool_name": "Read"},
    ]


def test_prepare_request_lowers_nested_document_tool_result_content() -> None:
    adapter = AnthropicMessagesAdapter()
    request = ModelRequest(
        model="claude-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="toolu_pdf",
                            content=cast(
                                FrozenJson,
                                [
                                    {
                                        "type": "text",
                                        "text": "PDF file read: /tmp/paper.pdf (15 B)",
                                    },
                                    {
                                        "type": "document",
                                        "media_type": "application/pdf",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "application/pdf",
                                            "data": "JVBERi0xLjQ=",
                                        },
                                    },
                                ],
                            ),
                        ),
                    ),
                )
            ),
        ),
    )

    body = _body(adapter.prepare_request(request))
    messages = cast(list[dict[str, object]], body["messages"])
    tool_result = cast(list[dict[str, object]], messages[0]["content"])[0]

    assert tool_result["content"] == [
        {"type": "text", "text": "PDF file read: /tmp/paper.pdf (15 B)"},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQ=",
            },
        },
    ]


def test_stream_events_assemble_to_equivalent_model_response_shape() -> None:
    adapter = AnthropicMessagesAdapter()
    provider_events = [
        {
            "type": "message_start",
            "request_id": "req_1",
            "message": {
                "id": "msg_1",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 2, "debug": True},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "I will search."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "plan"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "sig_1"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Search"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"raygent"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {
                "output_tokens": 7,
                "cache_creation_input_tokens": 0,
                "debug": False,
            },
        },
        {"type": "message_stop"},
    ]

    events = list(adapter.stream_events(provider_events))
    response = assemble_model_stream(events)
    expected = model_response_from_message_param(
        {
            "role": "assistant",
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "I will search."},
                {"type": "thinking", "text": "plan", "signature": "sig_1"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Search",
                    "input": {"query": "raygent"},
                },
            ],
        },
        stop_reason="tool_use",
        provider_request_id="req_1",
    )

    assert [
        event.identity.content_block_index
        for event in events
        if event.type == "content_block_start"
    ] == [0, 1, 2]
    assert [
        event.identity.content_block_index
        for event in events
        if event.type == "content_block_delta"
    ] == [0, 1, 1, 2]
    assert response.provider_request_id == expected.provider_request_id
    assert response.stop_reason == expected.stop_reason
    assert message_param_from_api_message(response.api_message) == (
        message_param_from_api_message(expected.api_message)
    )
    assert response.usage.input_tokens == 10
    assert response.usage.cache_read_input_tokens == 2
    assert response.usage.output_tokens == 7
    assert response.usage.effective_total_tokens == 19
    assert cast(Mapping[str, object], response.usage.provider_metadata)["anthropic"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0,
        "output_tokens": 7,
        "debug": False,
    }


def test_server_tool_use_is_not_exposed_for_client_execution() -> None:
    adapter = AnthropicMessagesAdapter()
    provider_events = [
        {
            "type": "message_start",
            "request_id": "req_1",
            "message": {"id": "msg_1"},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srv_1",
                "name": "web_search",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"raygent"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    response = assemble_model_stream(adapter.stream_events(provider_events))
    tool_use = cast(ToolUseContentBlock, response.api_message.message.content[0])

    assert tool_use.provider_executed is True
    assert tool_use.provider_metadata is not None
    assert cast(Mapping[str, object], tool_use.provider_metadata)["type"] == (
        "server_tool_use"
    )
    assert cast(Mapping[str, object], tool_use.input) == {"query": "raygent"}
    assert response.tool_uses == ()


def test_classify_error_preserves_prompt_and_media_recovery_details() -> None:
    adapter = AnthropicMessagesAdapter()

    prompt = adapter.classify_error(
        {
            "type": "invalid_request_error",
            "status_code": 400,
            "message": "prompt is too long: 137500 tokens > 135000 maximum",
        }
    )
    media = adapter.classify_error(
        {
            "type": "invalid_request_error",
            "status_code": 400,
            "message": "image exceeds 5 MB maximum",
        }
    )

    assert prompt.kind == "context_overflow"
    assert prompt.api_error is not None
    assert prompt.api_error.raw_details is not None
    assert prompt.api_error.actual_tokens == 137_500
    assert prompt.api_error.limit_tokens == 135_000
    assert media.kind == "media_overflow"
    assert media.api_error is not None
    assert media.api_error.raw_details is not None


def test_classify_error_preserves_retry_after_from_headers() -> None:
    adapter = AnthropicMessagesAdapter()

    error = adapter.classify_error(
        {
            "type": "rate_limit_error",
            "status_code": 429,
            "message": "slow down",
            "headers": {"retry-after": "3.25"},
        }
    )

    assert error.kind == "rate_limit"
    assert error.retry_after_s == 3.25


def test_prepare_token_count_lowers_messages_tools_and_thinking_without_generation() -> None:
    adapter = AnthropicMessagesAdapter()
    message = ApiMessage(
        message=ModelMessage(
            role="user",
            content=(TextContentBlock(text="count me"),),
        )
    )
    request = TokenCountRequest(
        model="claude-test",
        messages=(message,),
        system_prompt="system prompt",
        tools=(
            ModelToolSpec(
                name="Search",
                description="search docs",
                input_schema={"type": "object"},
            ),
        ),
        thinking={"type": "enabled", "budget_tokens": 1024},
        provider_options={"anthropic": {"headers": {"anthropic-beta": "counting"}}},
    )

    prepared = adapter.prepare_token_count(request)
    body = _body(prepared)

    assert thaw_json(prepared.options) == {
        "operation": "count_tokens",
        "provider_options": {
            "anthropic": {"headers": {"anthropic-beta": "counting"}}
        },
    }
    assert thaw_json(prepared.headers) == {"anthropic-beta": "counting"}
    assert body["model"] == "claude-test"
    assert "stream" not in body
    assert "max_tokens" not in body
    assert "provider_options" not in body
    assert body["system"] == [{"type": "text", "text": "system prompt"}]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body["tools"] == [
        {
            "name": "Search",
            "description": "search docs",
            "input_schema": {"type": "object"},
        }
    ]


def test_prepare_token_count_can_derive_thinking_from_provider_options() -> None:
    adapter = AnthropicMessagesAdapter()
    message = ApiMessage(
        message=ModelMessage(
            role="user",
            content=(TextContentBlock(text="count me"),),
        )
    )
    request = TokenCountRequest(
        model="claude-test",
        messages=(message,),
        provider_options={
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            }
        },
    )

    body = _body(adapter.prepare_token_count(request))

    assert "provider_options" not in body
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
