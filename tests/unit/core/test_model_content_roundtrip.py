from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from raygent_harness.core.messages import (
    MessageParam,
    message_param_from_model_message,
    model_message_from_message_param,
)
from raygent_harness.core.model_types import (
    MediaContentBlock,
    ModelStreamEvent,
    StreamIdentity,
    ThinkingContentBlock,
    ToolResultContentBlock,
    Usage,
)


def test_tool_result_preserves_structured_content_round_trip() -> None:
    message: MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    {"type": "text", "text": "created chart"},
                    {
                        "type": "image",
                        "media_type": "image/png",
                        "source": {
                            "type": "base64",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                ],
                "provider_metadata": {"cache_control": {"type": "ephemeral"}},
            }
        ],
    }

    model_message = model_message_from_message_param(message)
    block = cast(ToolResultContentBlock, model_message.content[0])
    content = cast(tuple[object, ...], block.content)

    assert cast(Mapping[str, object], content[0])["text"] == "created chart"
    assert cast(Mapping[str, object], content[1])["media_type"] == "image/png"
    assert block.provider_metadata is not None

    raw = message_param_from_model_message(model_message)
    raw_content = cast(list[dict[str, Any]], raw["content"])
    tool_result = raw_content[0]

    assert tool_result["content"] == cast(list[dict[str, Any]], message["content"])[0][
        "content"
    ]
    assert tool_result["provider_metadata"] == {
        "cache_control": {"type": "ephemeral"},
    }


def test_reasoning_and_media_metadata_survive_normalized_round_trip() -> None:
    message: MessageParam = {
        "role": "assistant",
        "content": [
            {
                "type": "encrypted_thinking",
                "text": "",
                "encrypted_content": "sealed-by-provider",
                "redacted": True,
            },
            {
                "type": "document",
                "media_type": "application/pdf",
                "source": {"type": "file_id", "file_id": "file_123"},
                "provider_metadata": {"provider_block_id": "block_1"},
            },
        ],
    }

    model_message = model_message_from_message_param(message)
    thinking = cast(ThinkingContentBlock, model_message.content[0])
    media = cast(MediaContentBlock, model_message.content[1])

    assert thinking.redacted is True
    assert thinking.provider_metadata is not None
    assert cast(Mapping[str, object], thinking.provider_metadata)["type"] == (
        "encrypted_thinking"
    )
    assert cast(Mapping[str, object], thinking.provider_metadata)["encrypted_content"] == (
        "sealed-by-provider"
    )
    assert media.media_kind == "document"
    assert media.provider_metadata is not None

    raw = message_param_from_model_message(model_message)
    raw_blocks = cast(list[dict[str, Any]], raw["content"])

    assert raw_blocks[0] == {
        "type": "thinking",
        "text": "",
        "redacted": True,
        "provider_metadata": {
            "type": "encrypted_thinking",
            "text": "",
            "encrypted_content": "sealed-by-provider",
            "redacted": True,
        },
    }
    assert raw_blocks[1]["type"] == "document"
    assert raw_blocks[1]["source"] == {"type": "file_id", "file_id": "file_123"}
    assert raw_blocks[1]["provider_metadata"] == {"provider_block_id": "block_1"}


def test_usage_preserves_reasoning_total_and_provider_metadata() -> None:
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=3,
        total_tokens=21,
        provider_metadata={"cache": {"read": 2}},
    )
    event = ModelStreamEvent.message_delta(
        StreamIdentity(message_id="msg_1"),
        usage=usage,
    )

    assert event.usage is not None
    assert event.usage.effective_total_tokens == 21
    assert cast(Mapping[str, object], event.usage.provider_metadata)["cache"] == {
        "read": 2,
    }


def test_usage_fallback_total_treats_reasoning_as_output_breakdown() -> None:
    usage = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=3)

    assert usage.effective_total_tokens == 15
