from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from raygent_harness.core.messages import model_response_from_message_param
from raygent_harness.core.model_stream import (
    ModelStreamAssembler,
    ModelStreamAssemblyError,
    assemble_model_stream,
)
from raygent_harness.core.model_types import (
    ModelRequest,
    ModelStreamEvent,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
    Usage,
)
from tests.fakes import FakeModelProvider


def _identity(
    *,
    block_index: int | None = None,
    message_id: str = "msg_1",
    request_id: str = "req_1",
    attempt_id: str = "attempt_1",
) -> StreamIdentity:
    return StreamIdentity(
        message_id=message_id,
        content_block_index=block_index,
        provider_request_id=request_id,
        attempt_id=attempt_id,
    )


def test_stream_assembler_builds_final_response_from_normalized_events() -> None:
    events = [
        ModelStreamEvent.message_start(
            _identity(),
            usage=Usage(input_tokens=11, cache_read_input_tokens=3),
        ),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=TextContentBlock(text="provider start text ignored"),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "text_delta", "text": "I will "},
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "text_delta", "text": "search."},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.content_block_start(
            _identity(block_index=1),
            block=ToolUseContentBlock(id="toolu_1", name="Search", input={}),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=1),
            delta={"type": "input_json_delta", "partial_json": '{"query": "raygent"}'},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=1)),
        ModelStreamEvent.message_delta(
            _identity(),
            usage=Usage(output_tokens=7),
            stop_reason="tool_use",
        ),
        ModelStreamEvent.message_stop(_identity()),
    ]

    response = assemble_model_stream(events)

    assert response.provider_request_id == "req_1"
    assert response.stop_reason == "tool_use"
    assert response.usage.input_tokens == 11
    assert response.usage.cache_read_input_tokens == 3
    assert response.usage.output_tokens == 7
    assert response.api_message.message.id == "msg_1"

    text = cast(TextContentBlock, response.api_message.message.content[0])
    tool = cast(ToolUseContentBlock, response.api_message.message.content[1])
    assert text.text == "I will search."
    assert tool.id == "toolu_1"
    assert tool.name == "Search"
    assert cast(Mapping[str, object], tool.input)["query"] == "raygent"
    assert response.tool_uses[0].id == "toolu_1"
    assert response.tool_uses[0].index == 1

    metadata = cast(Mapping[str, object], response.raw_metadata)
    assert metadata["attempt_id"] == "attempt_1"


def test_stream_assembler_updates_usage_and_stop_reason_from_message_stop() -> None:
    events = [
        ModelStreamEvent.message_start(_identity(), usage=Usage(input_tokens=5)),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=ThinkingContentBlock(text="ignored", signature=None),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "thinking_delta", "thinking": "reason"},
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "signature_delta", "signature": "sig_1"},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(
            _identity(),
            usage=Usage(output_tokens=4),
            stop_reason="end_turn",
        ),
    ]

    response = assemble_model_stream(events)

    thinking = cast(ThinkingContentBlock, response.api_message.message.content[0])
    assert thinking.text == "reason"
    assert thinking.signature == "sig_1"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 4
    assert response.stop_reason == "end_turn"


def test_stream_assembler_preserves_reasoning_replay_metadata() -> None:
    events = [
        ModelStreamEvent.message_start(_identity()),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=ThinkingContentBlock(
                text="ignored",
                signature=None,
                redacted=True,
                provider_metadata={
                    "type": "encrypted_thinking",
                    "encrypted_content": "sealed-by-provider",
                },
            ),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "thinking_delta", "thinking": ""},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(_identity(), stop_reason="end_turn"),
    ]

    response = assemble_model_stream(events)

    thinking = cast(ThinkingContentBlock, response.api_message.message.content[0])
    assert thinking.redacted is True
    assert thinking.provider_metadata is not None
    assert cast(Mapping[str, object], thinking.provider_metadata) == {
        "type": "encrypted_thinking",
        "encrypted_content": "sealed-by-provider",
    }


def test_streaming_transport_fallback_can_complete_with_replacement_response() -> None:
    replacement = model_response_from_message_param(
        {"role": "assistant", "content": "non-streaming replacement"},
        stop_reason="end_turn",
        provider_request_id="req_fallback",
    )
    identity = StreamIdentity(provider_request_id="req_1", attempt_id="attempt_1")
    assembler = ModelStreamAssembler()

    started = assembler.apply(
        ModelStreamEvent.streaming_transport_fallback_started(
            identity,
            reason="stream endpoint unavailable",
        )
    )
    completed = assembler.apply(
        ModelStreamEvent.streaming_transport_fallback_completed(
            identity,
            reason="non-streaming retry completed",
            replacement_response=replacement,
        )
    )

    assert started.streaming_transport_fallback is not None
    assert started.model_fallback is None
    assert completed.response is replacement
    assert assembler.response() is replacement
    assert assembler.model_fallback is None


def test_streaming_transport_fallback_discards_partial_stream_attempt() -> None:
    assembler = ModelStreamAssembler()
    assembler.apply(ModelStreamEvent.message_start(_identity(message_id="orphan")))
    assembler.apply(
        ModelStreamEvent.content_block_start(
            _identity(block_index=0, message_id="orphan"),
            block=TextContentBlock(text=""),
        )
    )
    assembler.apply(
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0, message_id="orphan"),
            delta={"type": "text_delta", "text": "orphaned"},
        )
    )

    fallback_identity = StreamIdentity(provider_request_id="req_2", attempt_id="attempt_2")
    assembler.apply(
        ModelStreamEvent.streaming_transport_fallback_started(
            fallback_identity,
            reason="stream watchdog",
        )
    )
    replacement_identity = _identity(message_id="replacement")
    assembler.apply(ModelStreamEvent.message_start(replacement_identity))
    assembler.apply(
        ModelStreamEvent.content_block_start(
            _identity(block_index=0, message_id="replacement"),
            block=TextContentBlock(text=""),
        )
    )
    assembler.apply(
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0, message_id="replacement"),
            delta={"type": "text_delta", "text": "replacement"},
        )
    )
    assembler.apply(
        ModelStreamEvent.content_block_stop(_identity(block_index=0, message_id="replacement"))
    )
    assembler.apply(ModelStreamEvent.message_stop(replacement_identity))

    response = assembler.response()
    text = cast(TextContentBlock, response.api_message.message.content[0])
    assert response.api_message.message.id == "replacement"
    assert text.text == "replacement"


def test_model_fallback_event_is_distinct_from_transport_fallback() -> None:
    identity = StreamIdentity(provider_request_id="req_1", attempt_id="attempt_1")
    assembler = ModelStreamAssembler()

    update = assembler.apply(
        ModelStreamEvent.model_fallback_triggered(
            identity,
            original_model="primary",
            fallback_model="fallback",
            reason="server overload",
        )
    )

    assert update.model_fallback is not None
    assert update.streaming_transport_fallback is None
    assert assembler.streaming_transport_fallback is None
    assert assembler.model_fallback is not None
    assert assembler.model_fallback.fallback_model == "fallback"
    with pytest.raises(ModelStreamAssemblyError, match="model fallback"):
        assembler.response()


def test_provider_error_event_blocks_response_assembly() -> None:
    identity = StreamIdentity(provider_request_id="req_1", attempt_id="attempt_1")
    assembler = ModelStreamAssembler()
    provider_error = ProviderError(kind="transient", message="stream failed", retryable=True)

    update = assembler.apply(
        ModelStreamEvent.provider_error_event(identity, provider_error=provider_error)
    )

    assert update.provider_error is provider_error
    assert assembler.provider_error is provider_error
    with pytest.raises(ModelStreamAssemblyError, match="provider_error"):
        assembler.response()


def test_stream_assembler_rejects_invalid_content_block_sequence() -> None:
    assembler = ModelStreamAssembler()

    with pytest.raises(ModelStreamAssemblyError, match="before message_start"):
        assembler.apply(
            ModelStreamEvent.content_block_start(
                _identity(block_index=0),
                block=TextContentBlock(text=""),
            )
        )

    assembler.apply(ModelStreamEvent.message_start(_identity()))

    with pytest.raises(ModelStreamAssemblyError, match="unopened block"):
        assembler.apply(
            ModelStreamEvent.content_block_delta(
                _identity(block_index=0),
                delta={"type": "text_delta", "text": "orphan"},
            )
        )

    with pytest.raises(ModelStreamAssemblyError, match="content_block_index"):
        assembler.apply(
            ModelStreamEvent.content_block_start(
                _identity(block_index=None),
                block=TextContentBlock(text=""),
            )
        )


@pytest.mark.parametrize(
    ("identity", "expected_field"),
    [
        (_identity(block_index=0, message_id="other"), "message_id"),
        (_identity(block_index=0, request_id="other"), "provider_request_id"),
        (_identity(block_index=0, attempt_id="other"), "attempt_id"),
    ],
)
def test_stream_assembler_rejects_message_identity_mismatch(
    identity: StreamIdentity,
    expected_field: str,
) -> None:
    assembler = ModelStreamAssembler()
    assembler.apply(ModelStreamEvent.message_start(_identity()))

    with pytest.raises(ModelStreamAssemblyError, match=expected_field):
        assembler.apply(
            ModelStreamEvent.content_block_start(
                identity,
                block=TextContentBlock(text=""),
            )
        )


@pytest.mark.parametrize(
    ("block", "delta", "expected"),
    [
        (
            TextContentBlock(text=""),
            {"type": "input_json_delta", "partial_json": "{}"},
            "input JSON delta",
        ),
        (
            ToolUseContentBlock(id="toolu_1", name="Search", input={}),
            {"type": "text_delta", "text": "wrong"},
            "text delta",
        ),
        (
            TextContentBlock(text=""),
            {"type": "thinking_delta", "thinking": "wrong"},
            "thinking delta",
        ),
        (
            ToolUseContentBlock(id="toolu_1", name="Search", input={}),
            {"type": "signature_delta", "signature": "sig"},
            "signature delta",
        ),
    ],
)
def test_stream_assembler_rejects_delta_block_type_mismatch(
    block: TextContentBlock | ToolUseContentBlock,
    delta: dict[str, str],
    expected: str,
) -> None:
    assembler = ModelStreamAssembler()
    assembler.apply(ModelStreamEvent.message_start(_identity()))
    assembler.apply(
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=block,
        )
    )

    with pytest.raises(ModelStreamAssemblyError, match=expected):
        assembler.apply(
            ModelStreamEvent.content_block_delta(
                _identity(block_index=0),
                delta=delta,
            )
        )


@pytest.mark.asyncio
async def test_fake_provider_stream_yields_normalized_events() -> None:
    event = ModelStreamEvent.message_start(_identity(), usage=Usage(input_tokens=1))
    provider = FakeModelProvider(stream_events=(event,))
    request = ModelRequest(model="model-1", messages=())

    events = [item async for item in provider.stream(request)]

    assert events == [event]
    assert provider.stream_requests == [request]
