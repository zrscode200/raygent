from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import (
    MessageParam,
    message_param_from_api_message,
    model_response_from_message_param,
)
from raygent_harness.core.model_types import (
    ModelCapabilities,
    ModelInfo,
    ModelStreamEvent,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
    Usage,
)
from raygent_harness.core.observability import (
    KernelEventBus,
    RecordingKernelEventSink,
)
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query import (
    MemoryRecallMessage,
    StreamRequestStart,
    ToolProgressEvent,
    ToolResultMessage,
    query,
)
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKMessageDelta,
    SDKReasoningAvailable,
    SDKResult,
    SDKStreamEvent,
    SDKStreamEventCallback,
    SDKStreamOptions,
    SDKSystemInit,
    SDKUserMessage,
)
from raygent_harness.core.state import State
from raygent_harness.core.stop_hooks import HookContext, HookContinue, HookResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolProgress,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.core.tool_orchestration import TOOL_CANCEL_MESSAGE
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    StreamEventEntry,
    TombstoneEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    load_session_replay,
)
from tests.fakes import FakeModelProvider


class SequencedStreamProvider(FakeModelProvider):
    def __init__(self, stream_batches: Sequence[Sequence[ModelStreamEvent]]) -> None:
        super().__init__()
        self.stream_batches = tuple(tuple(batch) for batch in stream_batches)

    def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        index = len(self.stream_requests) - 1
        batch = self.stream_batches[index] if index < len(self.stream_batches) else ()
        return self._stream_batch(batch)

    async def _stream_batch(
        self,
        batch: Sequence[ModelStreamEvent],
    ) -> AsyncIterator[ModelStreamEvent]:
        for event in batch:
            yield event


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


class ExampleInput(BaseModel):
    value: int = 0


def _example_tool(
    call: Any | None = None,
    *,
    interrupt_behavior: Literal["cancel", "block"] = "block",
):
    async def default_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(content="tool done")

    return build_tool(
        ToolSpec(
            name="Example",
            description="Example tool",
            input_model=ExampleInput,
            call=call or default_call,
            is_read_only=True,
            is_concurrency_safe=True,
            interrupt_behavior=interrupt_behavior,
        )
    )


def _agent_placeholder_tool():
    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(content="agent launched")

    return build_tool(
        ToolSpec(
            name="Agent",
            description="Agent placeholder",
            input_model=ExampleInput,
            call=call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _memory_recall_message(content: str = "memory context") -> MessageParam:
    return {
        "role": "user",
        "content": content,
        "raygentMessageKind": "memory_recall",
        "raygentMemoryRecall": {
            "type": "relevant_memories",
            "memories": [{"path": "/tmp/memory.md", "content_bytes": len(content)}],
        },
    }


@dataclass
class RecordingMemoryPrefetch:
    messages_to_return: tuple[MessageParam, ...] = ()
    consumed_on_iteration_value: int | None = None
    cancel_count: int = 0

    @property
    def settled_at(self) -> float | None:
        return 1.0

    @property
    def consumed_on_iteration(self) -> int | None:
        return self.consumed_on_iteration_value

    async def consume_if_ready(
        self,
        *,
        ctx: ToolUseContext,
        iteration: int,
    ) -> tuple[MessageParam, ...]:
        _ = ctx
        self.consumed_on_iteration_value = iteration
        return self.messages_to_return

    def cancel(self) -> None:
        self.cancel_count += 1


def _empty_recall_start_calls() -> list[
    tuple[tuple[MessageParam, ...], QueryConfig, ToolUseContext]
]:
    return []


@dataclass
class RecordingMemoryRecallProvider:
    prefetch: RecordingMemoryPrefetch
    start_calls: list[tuple[tuple[MessageParam, ...], QueryConfig, ToolUseContext]] = field(
        default_factory=_empty_recall_start_calls
    )

    def start(
        self,
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> RecordingMemoryPrefetch:
        self.start_calls.append((tuple(messages), config, ctx))
        return self.prefetch


def _config(*, streaming: bool = True, tools: Sequence[Any] = ()) -> QueryConfig:
    return QueryConfig(
        model="model-1",
        session_id="s",
        tools=tuple(tools),
        experiments={"streaming_tool_execution": streaming} if streaming else {},
    )


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


def _text_stream(text: str) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(
            _identity(),
            usage=Usage(input_tokens=3),
        ),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=TextContentBlock(text=""),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "text_delta", "text": text},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(
            _identity(),
            usage=Usage(output_tokens=5),
            stop_reason="end_turn",
        ),
    )


def _unknown_text_shape_stream(text: str) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(_identity()),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=TextContentBlock(text=""),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "provider_text_delta", "text": text},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(_identity(), stop_reason="end_turn"),
    )


def _text_stream_without_message_id(text: str) -> tuple[ModelStreamEvent, ...]:
    identity = StreamIdentity(
        message_id=None,
        content_block_index=None,
        provider_request_id="provider-request-secret",
        attempt_id="attempt-secret",
    )
    block_identity = StreamIdentity(
        message_id=None,
        content_block_index=0,
        provider_request_id="provider-request-secret",
        attempt_id="attempt-secret",
    )
    return (
        ModelStreamEvent.message_start(identity),
        ModelStreamEvent.content_block_start(
            block_identity,
            block=TextContentBlock(text=""),
        ),
        ModelStreamEvent.content_block_delta(
            block_identity,
            delta={"type": "text_delta", "text": text},
        ),
        ModelStreamEvent.content_block_stop(block_identity),
        ModelStreamEvent.message_stop(identity, stop_reason="end_turn"),
    )


def _reasoning_stream() -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(_identity()),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=ThinkingContentBlock(
                text="secret plan",
                signature="secret-signature",
                redacted=True,
                provider_metadata={"encrypted_content": "sealed-reasoning"},
            ),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "thinking_delta", "thinking": "hidden continuation"},
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "signature_delta", "signature": "secret-signature-2"},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.content_block_start(
            _identity(block_index=1),
            block=TextContentBlock(text=""),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=1),
            delta={"type": "text_delta", "text": "visible answer"},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=1)),
        ModelStreamEvent.message_stop(
            _identity(),
            usage=Usage(output_tokens=2, reasoning_tokens=7),
            stop_reason="end_turn",
        ),
    )


def _tool_use_stream() -> tuple[ModelStreamEvent, ...]:
    return (
        *_tool_use_stream_events(
            tool_use_id="toolu_1",
            message_id="msg_tool",
            value=1,
        ),
        ModelStreamEvent.message_stop(
            _identity(message_id="msg_tool"),
            usage=Usage(output_tokens=4),
            stop_reason="tool_use",
        ),
    )


def _tool_use_stream_events(
    *,
    tool_use_id: str,
    message_id: str,
    value: int,
) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(
            _identity(message_id=message_id),
            usage=Usage(input_tokens=3),
        ),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0, message_id=message_id),
            block=ToolUseContentBlock(id=tool_use_id, name="Example", input={}),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0, message_id=message_id),
            delta={"type": "input_json_delta", "partial_json": f'{{"value": {value}}}'},
        ),
        ModelStreamEvent.content_block_stop(
            _identity(block_index=0, message_id=message_id)
        ),
    )


async def _run_engine(
    provider: FakeModelProvider,
    *,
    config: QueryConfig | None = None,
    transcript_store: JsonlTranscriptStore | None = None,
    observability: KernelEventBus | None = None,
    stream_callback: SDKStreamEventCallback | None = None,
    stream_options: SDKStreamOptions | None = None,
) -> tuple[QueryEngine, list[Any]]:
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        transcript_store=transcript_store,
        observability=observability or KernelEventBus(),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    engine = QueryEngine(config or _config(), deps, _ctx())
    events = [
        event
        async for event in engine.submit_message(
            "hi",
            stream_callback=stream_callback,
            stream_options=stream_options,
        )
    ]
    return engine, events


@pytest.mark.asyncio
async def test_sdk_stream_callback_emits_bounded_text_deltas_only_via_callback() -> None:
    provider = FakeModelProvider(stream_events=_text_stream("streamed"))
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        stream_callback=stream_events.append,
        stream_options=SDKStreamOptions(max_text_delta_chars=4),
    )

    assert [event for event in events if isinstance(event, SDKMessageDelta)] == []
    assert stream_events == [
        SDKMessageDelta(
            session_id="s",
            turn_id="turn-1",
            stream_id="msg_1",
            text="stre",
            index=0,
            is_final=False,
        ),
        SDKMessageDelta(
            session_id="s",
            turn_id="turn-1",
            stream_id="msg_1",
            text="amed",
            index=0,
            is_final=False,
        ),
    ]
    assert isinstance(events[-1], SDKResult)
    assert events[-1].result == "streamed"


@pytest.mark.asyncio
async def test_sdk_stream_id_fallback_does_not_expose_provider_request_ids() -> None:
    provider = FakeModelProvider(stream_events=_text_stream_without_message_id("safe"))
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        stream_callback=stream_events.append,
    )

    assert stream_events == [
        SDKMessageDelta(
            session_id="s",
            turn_id="turn-1",
            stream_id="turn-1",
            text="safe",
            index=0,
            is_final=False,
        )
    ]
    assert "provider-request-secret" not in repr(stream_events)
    assert "attempt-secret" not in repr(stream_events)
    assert isinstance(events[-1], SDKResult)


@pytest.mark.asyncio
async def test_sdk_stream_callback_respects_text_delta_option() -> None:
    provider = FakeModelProvider(stream_events=_text_stream("suppressed"))
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        stream_callback=stream_events.append,
        stream_options=SDKStreamOptions(text_deltas=False),
    )

    assert stream_events == []
    assert isinstance(events[-1], SDKResult)
    assert events[-1].result == "suppressed"


@pytest.mark.asyncio
async def test_sdk_stream_callback_ignores_unknown_text_shapes() -> None:
    provider = FakeModelProvider(stream_events=_unknown_text_shape_stream("private"))
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        stream_callback=stream_events.append,
    )

    assert stream_events == []
    assert isinstance(events[-1], SDKResult)
    assert events[-1].result == "private"


@pytest.mark.asyncio
async def test_sdk_stream_callback_emits_reasoning_availability_without_content() -> None:
    provider = FakeModelProvider(stream_events=_reasoning_stream())
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        stream_callback=stream_events.append,
    )

    reasoning_events = [
        event for event in stream_events if isinstance(event, SDKReasoningAvailable)
    ]
    assert reasoning_events == [
        SDKReasoningAvailable(
            session_id="s",
            turn_id="turn-1",
            stream_id="msg_1",
            char_count=len("secret plan"),
        ),
        SDKReasoningAvailable(
            session_id="s",
            turn_id="turn-1",
            stream_id="msg_1",
            char_count=len("hidden continuation"),
        ),
        SDKReasoningAvailable(
            session_id="s",
            turn_id="turn-1",
            stream_id="msg_1",
            token_count=7,
        ),
    ]
    assert [
        event.text for event in stream_events if isinstance(event, SDKMessageDelta)
    ] == ["visible answer"]
    assert "secret plan" not in repr(reasoning_events)
    assert "hidden continuation" not in repr(reasoning_events)
    assert "secret-signature" not in repr(reasoning_events)
    assert isinstance(events[-1], SDKResult)


@pytest.mark.asyncio
async def test_sdk_stream_callback_suppresses_tool_input_json_deltas() -> None:
    provider = FakeModelProvider(stream_events=_tool_use_stream())
    stream_events: list[SDKStreamEvent] = []

    _engine, events = await _run_engine(
        provider,
        config=_config(tools=(_example_tool(),)),
        stream_callback=stream_events.append,
    )

    assert stream_events == []
    assert "partial_json" not in repr(stream_events)
    assert "value" not in repr(stream_events)
    assert isinstance(events[-1], SDKResult)


@pytest.mark.asyncio
async def test_sdk_stream_callback_exception_propagates_without_sdk_error_result() -> None:
    provider = FakeModelProvider(stream_events=_text_stream("boom"))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus(),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    engine = QueryEngine(_config(), deps, _ctx())
    yielded_events: list[Any] = []

    def broken_callback(_event: SDKStreamEvent) -> None:
        raise RuntimeError("stream callback failed")

    with pytest.raises(RuntimeError, match="stream callback failed"):
        async for event in engine.submit_message(
            "hi",
            stream_callback=broken_callback,
        ):
            yielded_events.append(event)

    assert [type(event) for event in yielded_events] == [SDKSystemInit]


@pytest.mark.asyncio
async def test_streaming_no_tool_response_matches_complete_path() -> None:
    provider = FakeModelProvider(stream_events=_text_stream("streamed answer"))

    engine, events = await _run_engine(provider)

    assert len(provider.stream_requests) == 1
    assert provider.requests == []
    assert isinstance(events[0], SDKSystemInit)
    assert isinstance(events[-1], SDKResult)
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert len(assistant_events) == 1
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": "streamed answer",
        "id": "msg_1",
    }
    assert events[-1].subtype == "success"
    assert events[-1].usage.input_tokens == 3
    assert events[-1].usage.output_tokens == 5
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "streamed answer", "id": "msg_1"},
    ]


@pytest.mark.asyncio
async def test_streaming_gate_off_uses_complete_path() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "complete answer"},),
        stream_events=_text_stream("should not stream"),
    )

    _engine, events = await _run_engine(provider, config=_config(streaming=False))

    assert len(provider.requests) == 1
    assert provider.stream_requests == []
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": "complete answer",
    }


@pytest.mark.asyncio
async def test_streaming_gate_falls_back_to_complete_when_model_cannot_stream() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "non-streaming answer"},),
        stream_events=_text_stream("should not stream"),
        model_infos={
            "model-1": ModelInfo(
                model="model-1",
                capabilities=ModelCapabilities(supports_streaming=False),
            )
        },
    )

    _engine, events = await _run_engine(provider)

    assert len(provider.requests) == 1
    assert provider.stream_requests == []
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": "non-streaming answer",
    }


@pytest.mark.asyncio
async def test_streaming_gate_disables_overlap_for_fork_agent_tool() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "complete answer"},),
        stream_events=_text_stream("should not stream"),
        model_infos={
            "model-1": ModelInfo(
                model="model-1",
                capabilities=ModelCapabilities(supports_streaming=True),
            )
        },
    )
    config = QueryConfig(
        model="model-1",
        session_id="s",
        tools=(_agent_placeholder_tool(),),
        experiments={
            "streaming_tool_execution": True,
            "fork_subagent": True,
        },
    )

    _engine, events = await _run_engine(provider, config=config)

    assert len(provider.requests) == 1
    assert provider.stream_requests == []
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": "complete answer",
    }


@pytest.mark.asyncio
async def test_stream_events_are_persisted_without_mutating_messages(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    provider = FakeModelProvider(stream_events=_text_stream("persist me"))

    engine, events = await _run_engine(provider, transcript_store=store)
    entries = await store.read_entries(TranscriptScope(session_id="s"))

    assert isinstance(events[-1], SDKResult)
    assert [
        entry.type
        for entry in entries
        if isinstance(entry, StreamEventEntry | TranscriptMessageEntry)
    ] == [
        "message",
        "stream_event",
        "stream_event",
        "stream_event",
        "stream_event",
        "stream_event",
        "message",
    ]
    stream_entries = [entry for entry in entries if isinstance(entry, StreamEventEntry)]
    assert [entry.event["type"] for entry in stream_entries] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    ]
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "persist me", "id": "msg_1"},
    ]


@pytest.mark.asyncio
async def test_streaming_transport_fallback_persists_tombstone(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    sink = RecordingKernelEventSink()
    secret_reason = "SECRET stream endpoint failed"
    replacement = model_response_from_message_param(
        {"role": "assistant", "content": "replacement answer"},
        stop_reason="end_turn",
        provider_request_id="req_replacement",
    )
    provider = FakeModelProvider(
        stream_events=(
            ModelStreamEvent.message_start(_identity(message_id="orphan")),
            ModelStreamEvent.content_block_start(
                _identity(block_index=0, message_id="orphan"),
                block=TextContentBlock(text=""),
            ),
            ModelStreamEvent.content_block_delta(
                _identity(block_index=0, message_id="orphan"),
                delta={"type": "text_delta", "text": "orphaned"},
            ),
            ModelStreamEvent.streaming_transport_fallback_started(
                _identity(message_id="orphan"),
                reason=secret_reason,
            ),
            ModelStreamEvent.streaming_transport_fallback_completed(
                _identity(message_id="orphan"),
                reason="non-streaming retry completed",
                replacement_response=replacement,
            ),
        )
    )

    engine, events = await _run_engine(
        provider,
        transcript_store=store,
        observability=KernelEventBus([sink]),
    )
    entries = await store.read_entries(TranscriptScope(session_id="s"))

    assert isinstance(events[-1], SDKResult)
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": "replacement answer",
    }
    tombstones = [entry for entry in entries if isinstance(entry, TombstoneEntry)]
    assert len(tombstones) == 1
    assert tombstones[0].target_message_id == "orphan"
    assert tombstones[0].reason == secret_reason
    assert tombstones[0].event is not None
    assert tombstones[0].event["type"] == "streaming_transport_fallback_started"
    transcript_tombstone_events = [
        event
        for event in sink.by_type("transcript.appended")
        if event.data["entry_type"] == "tombstone"
    ]
    assert len(transcript_tombstone_events) == 1
    tombstone_event = transcript_tombstone_events[0]
    assert tombstone_event.data["reason_category"] == "provider_supplied"
    assert tombstone_event.data["reason_char_count"] == len(secret_reason)
    assert secret_reason not in "\n".join(str(event.data) for event in sink.events)
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "replacement answer"},
    ]


@pytest.mark.asyncio
async def test_stream_provider_error_enters_existing_recovery_ladder() -> None:
    provider = FakeModelProvider(
        stream_events=(
            ModelStreamEvent.provider_error_event(
                _identity(),
                provider_error=ProviderError(kind="rate_limit", message="slow down"),
            ),
        )
    )

    engine, events = await _run_engine(provider)

    assert len(provider.stream_requests) == 1
    assert provider.requests == []
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert len(assistant_events) == 1
    assert assistant_events[0].message.get("isApiErrorMessage") is True
    assert assistant_events[0].message.get("apiError") == "rate_limit"
    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"
    assert result.is_error is True
    assert engine._messages[-1].get("isApiErrorMessage") is True  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_model_fallback_stream_event_retries_with_fallback_model() -> None:
    provider = SequencedStreamProvider(
        (
            (
                ModelStreamEvent.model_fallback_triggered(
                    _identity(request_id="req_primary", attempt_id="attempt_primary"),
                    original_model="model-1",
                    fallback_model="model-2",
                    reason="primary overloaded",
                ),
            ),
            _text_stream("fallback answer"),
        )
    )

    engine, events = await _run_engine(provider)

    assert [request.model for request in provider.stream_requests] == [
        "model-1",
        "model-2",
    ]
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[-1].message == {
        "role": "assistant",
        "content": "fallback answer",
        "id": "msg_1",
    }
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert engine._messages[-1] == {  # pyright: ignore[reportPrivateUsage]
        "role": "assistant",
        "content": "fallback answer",
        "id": "msg_1",
    }


@pytest.mark.asyncio
async def test_streamed_tool_use_uses_streaming_executor_not_non_overlap_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_orchestrate(
        _assistant_message: MessageParam,
        _state: Any,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[object]:
        raise AssertionError("streaming tool path should skip non-overlap orchestration")
        yield  # pragma: no cover

    monkeypatch.setattr(query_mod, "_orchestrate_tools", fake_orchestrate)
    provider = SequencedStreamProvider((_tool_use_stream(), _text_stream("done")))

    engine, events = await _run_engine(provider, config=_config(tools=(_example_tool(),)))

    assert len(provider.stream_requests) == 2
    assistant_events = [event for event in events if isinstance(event, SDKAssistantMessage)]
    assert assistant_events[0].message == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Example",
                "input": {"value": 1},
            }
        ],
        "id": "msg_tool",
    }
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    tool_result: MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "tool done",
            }
        ],
    }
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        assistant_events[0].message,
        tool_result,
        {"role": "assistant", "content": "done", "id": "msg_1"},
    ]


@pytest.mark.asyncio
async def test_streamed_tool_starts_before_message_stop() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    observed_start_before_stop = False

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        started.set()
        await release.wait()
        yield ToolResult(content="overlapped")

    class WaitingProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            index = len(self.stream_requests) - 1
            if index > 0:
                return self._stream_batch(_text_stream("done"))
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            nonlocal observed_start_before_stop
            for event in _tool_use_stream()[:-1]:
                yield event
            await asyncio.wait_for(started.wait(), timeout=1.0)
            observed_start_before_stop = True
            release.set()
            yield _tool_use_stream()[-1]

    provider = WaitingProvider(())

    _engine, events = await _run_engine(
        provider,
        config=_config(tools=(_example_tool(call),)),
    )

    assert observed_start_before_stop is True
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"


@pytest.mark.asyncio
async def test_streamed_tool_progress_yields_before_final_assistant_message() -> None:
    progress_seen = asyncio.Event()
    finish = asyncio.Event()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolProgress(message="working")
        progress_seen.set()
        await finish.wait()
        yield ToolResult(content="done")

    class ProgressProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            index = len(self.stream_requests) - 1
            if index > 0:
                return self._stream_batch(_text_stream("finished"))
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            for event in _tool_use_stream()[:-1]:
                yield event
            await asyncio.wait_for(progress_seen.wait(), timeout=1.0)
            finish.set()
            yield _tool_use_stream()[-1]

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=ProgressProvider(()),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "hi"}]),
            _config(tools=(_example_tool(call),)),
            deps,
            _ctx(),
        )
    ]

    progress_index = next(
        index for index, event in enumerate(events) if isinstance(event, ToolProgressEvent)
    )
    assistant_index = next(
        index
        for index, event in enumerate(events)
        if getattr(event, "type", None) == "assistant_message"
    )
    assert progress_index < assistant_index


@pytest.mark.asyncio
async def test_streaming_tool_path_preserves_stop_hook_transcript_site() -> None:
    seen_hook_messages: list[list[MessageParam]] = []

    async def inspect_hook(hc: HookContext) -> HookResult:
        seen_hook_messages.append(list(hc.messages))
        return HookContinue()

    provider = SequencedStreamProvider((_tool_use_stream(), _text_stream("done")))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        stop_hooks=[inspect_hook],  # pyright: ignore[reportArgumentType]
    )

    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "hi"}]),
            _config(tools=(_example_tool(),)),
            deps,
            _ctx(),
        )
    ]

    assert isinstance(events[-1], query_mod.TerminalEvent)
    assert events[-1].terminal.reason == "completed"
    assert seen_hook_messages == [
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Example",
                        "input": {"value": 1},
                    }
                ],
                "id": "msg_tool",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "tool done",
                    }
                ],
            },
            {"role": "assistant", "content": "done", "id": "msg_1"},
        ]
    ]


@pytest.mark.asyncio
async def test_streaming_tool_path_consumes_memory_recall_before_next_model_call() -> None:
    memory_message = _memory_recall_message()
    prefetch = RecordingMemoryPrefetch(messages_to_return=(memory_message,))
    recall_provider = RecordingMemoryRecallProvider(prefetch)
    provider = SequencedStreamProvider((_tool_use_stream(), _text_stream("done")))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        memory_recall_provider=recall_provider,
    )

    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "hi"}]),
            _config(tools=(_example_tool(),)),
            deps,
            _ctx(),
        )
    ]

    tool_result_index = next(
        index for index, event in enumerate(events) if isinstance(event, ToolResultMessage)
    )
    memory_index = next(
        index for index, event in enumerate(events) if isinstance(event, MemoryRecallMessage)
    )
    second_stream_index = [
        index for index, event in enumerate(events) if isinstance(event, StreamRequestStart)
    ][1]
    assert tool_result_index < memory_index < second_stream_index
    assert prefetch.consumed_on_iteration == 1
    assert prefetch.cancel_count == 1
    assert recall_provider.start_calls[0][0] == (
        {"role": "user", "content": "hi"},
    )
    assert message_param_from_api_message(provider.stream_requests[1].messages[-1]) == (
        memory_message
    )

    terminal = next(
        event.terminal
        for event in events
        if isinstance(event, query_mod.TerminalEvent)
    )
    assert terminal.final_state is not None
    assert terminal.final_state.messages[-2] == memory_message
    assert terminal.final_state.messages[-1] == {
        "role": "assistant",
        "content": "done",
        "id": "msg_1",
    }


@pytest.mark.asyncio
async def test_streaming_transport_fallback_discards_stale_tool_results(
    tmp_path: Path,
) -> None:
    old_tool_started = asyncio.Event()
    release_old_tool = asyncio.Event()

    async def call(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        value = getattr(input_, "value", 0)
        if value == 99:
            old_tool_started.set()
            await release_old_tool.wait()
            yield ToolResult(content="stale old attempt")
            return
        yield ToolResult(content="replacement tool result")

    replacement_assistant: MessageParam = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_replacement",
                "name": "Example",
                "input": {"value": 1},
            }
        ],
        "id": "replacement_msg",
    }
    replacement = model_response_from_message_param(
        replacement_assistant,
        stop_reason="tool_use",
        provider_request_id="req_replacement",
    )

    class FallbackProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            if len(self.stream_requests) > 1:
                return self._stream_batch(_text_stream("done"))
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            for event in _tool_use_stream_events(
                tool_use_id="toolu_old",
                message_id="orphan_tool_msg",
                value=99,
            ):
                yield event
            await asyncio.wait_for(old_tool_started.wait(), timeout=1.0)
            yield ModelStreamEvent.streaming_transport_fallback_started(
                _identity(message_id="orphan_tool_msg"),
                reason="stream endpoint failed",
            )
            yield ModelStreamEvent.streaming_transport_fallback_completed(
                _identity(message_id="orphan_tool_msg"),
                reason="non-streaming retry completed",
                replacement_response=replacement,
            )

    store = JsonlTranscriptStore(tmp_path)
    engine, events = await _run_engine(
        FallbackProvider(()),
        config=_config(tools=(_example_tool(call),)),
        transcript_store=store,
    )
    release_old_tool.set()
    await asyncio.sleep(0)

    tool_results = [
        event.message
        for event in events
        if isinstance(event, SDKUserMessage)
        and isinstance(event.message.get("content"), list)
    ]
    assert tool_results == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_replacement",
                    "content": "replacement tool result",
                }
            ],
        }
    ]
    assert "stale old attempt" not in str(engine._messages)  # pyright: ignore[reportPrivateUsage]

    replay = await load_session_replay(store, TranscriptScope(session_id="s"))
    assert replay.messages == engine._messages  # pyright: ignore[reportPrivateUsage]
    assert replay.warnings == ()


@pytest.mark.asyncio
async def test_model_fallback_discards_accepted_streamed_tool_use() -> None:
    old_tool_started = asyncio.Event()
    release_old_tool = asyncio.Event()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        old_tool_started.set()
        await release_old_tool.wait()
        yield ToolResult(content="old model result")

    class ModelFallbackProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            if len(self.stream_requests) > 1:
                return self._stream_batch(_text_stream("fallback answer"))
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            for event in _tool_use_stream_events(
                tool_use_id="toolu_old",
                message_id="model_fallback_orphan",
                value=1,
            ):
                yield event
            await asyncio.wait_for(old_tool_started.wait(), timeout=1.0)
            yield ModelStreamEvent.model_fallback_triggered(
                _identity(
                    message_id="model_fallback_orphan",
                    request_id="req_primary",
                    attempt_id="attempt_primary",
                ),
                original_model="model-1",
                fallback_model="model-2",
                reason="primary overloaded",
            )

    provider = ModelFallbackProvider(())
    engine, events = await _run_engine(
        provider,
        config=_config(tools=(_example_tool(call),)),
    )
    release_old_tool.set()
    await asyncio.sleep(0)

    assert [request.model for request in provider.stream_requests] == ["model-1", "model-2"]
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "fallback answer", "id": "msg_1"},
    ]
    assert not any(
        isinstance(event, SDKUserMessage)
        and isinstance(event.message.get("content"), list)
        for event in events
    )


@pytest.mark.asyncio
async def test_provider_error_discards_accepted_streamed_tool_use() -> None:
    old_tool_started = asyncio.Event()
    release_old_tool = asyncio.Event()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        old_tool_started.set()
        await release_old_tool.wait()
        yield ToolResult(content="old provider-error result")

    class ProviderErrorAfterToolUse(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            for event in _tool_use_stream_events(
                tool_use_id="toolu_old",
                message_id="provider_error_orphan",
                value=1,
            ):
                yield event
            await asyncio.wait_for(old_tool_started.wait(), timeout=1.0)
            yield ModelStreamEvent.provider_error_event(
                _identity(message_id="provider_error_orphan"),
                provider_error=ProviderError(
                    kind="rate_limit",
                    message="rate limited after partial stream",
                ),
            )

    engine, events = await _run_engine(
        ProviderErrorAfterToolUse(()),
        config=_config(tools=(_example_tool(call),)),
    )
    release_old_tool.set()
    await asyncio.sleep(0)

    assert "old provider-error result" not in str(engine._messages)  # pyright: ignore[reportPrivateUsage]
    assert not any(
        isinstance(event, SDKUserMessage)
        and isinstance(event.message.get("content"), list)
        for event in events
    )
    assert engine._messages[-1].get("isApiErrorMessage") is True  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_abort_during_no_tool_stream_skips_stop_hooks() -> None:
    ctx = _ctx()
    stop_hook_called = False

    async def stop_hook(_hook_ctx: HookContext) -> HookResult:
        nonlocal stop_hook_called
        stop_hook_called = True
        return HookContinue()

    class AbortNoToolProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.message_start(_identity(message_id="abort_text"))
            yield ModelStreamEvent.content_block_start(
                _identity(block_index=0, message_id="abort_text"),
                block=TextContentBlock(text=""),
            )
            yield ModelStreamEvent.content_block_delta(
                _identity(block_index=0, message_id="abort_text"),
                delta={"type": "text_delta", "text": "partial answer"},
            )
            yield ModelStreamEvent.content_block_stop(
                _identity(block_index=0, message_id="abort_text")
            )
            ctx.abort_event.set()
            yield ModelStreamEvent.message_stop(
                _identity(message_id="abort_text"),
                usage=Usage(output_tokens=4),
                stop_reason="end_turn",
            )

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=AbortNoToolProvider(()),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        stop_hooks=[stop_hook],  # pyright: ignore[reportArgumentType]
    )
    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "hi"}]),
            _config(),
            deps,
            ctx,
        )
    ]

    terminal = events[-1]
    assert isinstance(terminal, query_mod.TerminalEvent)
    assert terminal.terminal.reason == "aborted_streaming"
    assert stop_hook_called is False
    assert terminal.terminal.final_state is not None
    assert terminal.terminal.final_state.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial answer", "id": "abort_text"},
    ]


@pytest.mark.asyncio
async def test_abort_after_streamed_assistant_yield_skips_stop_hooks() -> None:
    ctx = _ctx()
    stop_hook_called = False

    async def stop_hook(_hook_ctx: HookContext) -> HookResult:
        nonlocal stop_hook_called
        stop_hook_called = True
        return HookContinue()

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(stream_events=_text_stream("answer")),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        stop_hooks=[stop_hook],  # pyright: ignore[reportArgumentType]
    )
    events: list[Any] = []
    async for event in query(
        State(messages=[{"role": "user", "content": "hi"}]),
        _config(),
        deps,
        ctx,
    ):
        events.append(event)
        if getattr(event, "type", None) == "assistant_message":
            ctx.abort_event.set()

    terminal = events[-1]
    assert isinstance(terminal, query_mod.TerminalEvent)
    assert terminal.terminal.reason == "aborted_streaming"
    assert stop_hook_called is False


@pytest.mark.asyncio
async def test_abort_during_streamed_tool_drains_results_as_aborted_streaming() -> None:
    tool_started = asyncio.Event()
    ctx = _ctx()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        tool_started.set()
        await asyncio.Event().wait()
        yield ToolResult(content="should not appear")

    class AbortProvider(SequencedStreamProvider):
        def stream(self, request: Any) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            if len(self.stream_requests) > 1:
                raise AssertionError("abort should terminate the turn")
            return self._first_stream()

        async def _first_stream(self) -> AsyncIterator[ModelStreamEvent]:
            for event in _tool_use_stream_events(
                tool_use_id="toolu_abort",
                message_id="abort_msg",
                value=1,
            ):
                yield event
            await asyncio.wait_for(tool_started.wait(), timeout=1.0)
            ctx.abort_event.set()
            yield ModelStreamEvent.message_stop(
                _identity(message_id="abort_msg"),
                usage=Usage(output_tokens=4),
                stop_reason="tool_use",
            )

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=AbortProvider(()),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "hi"}]),
            _config(
                tools=(
                    _example_tool(
                        call,
                        interrupt_behavior="cancel",
                    ),
                )
            ),
            deps,
            ctx,
        )
    ]

    terminal = events[-1]
    assert isinstance(terminal, query_mod.TerminalEvent)
    assert terminal.terminal.reason == "aborted_streaming"
    tool_result = next(event for event in events if isinstance(event, ToolResultMessage))
    assert tool_result.message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_abort",
                "content": TOOL_CANCEL_MESSAGE,
                "is_error": True,
            }
        ],
    }
    assert terminal.terminal.final_state is not None
    assert terminal.terminal.final_state.messages == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abort",
                    "name": "Example",
                    "input": {"value": 1},
                }
            ],
            "id": "abort_msg",
        },
        tool_result.message,
    ]


@pytest.mark.asyncio
async def test_abort_after_streamed_tool_assistant_yield_remains_aborted_streaming() -> None:
    ctx = _ctx()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        await asyncio.Event().wait()
        yield ToolResult(content="should not appear")

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=SequencedStreamProvider((_tool_use_stream(),)),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    events: list[Any] = []
    async for event in query(
        State(messages=[{"role": "user", "content": "hi"}]),
        _config(tools=(_example_tool(call, interrupt_behavior="cancel"),)),
        deps,
        ctx,
    ):
        events.append(event)
        if getattr(event, "type", None) == "assistant_message":
            ctx.abort_event.set()

    terminal = events[-1]
    assert isinstance(terminal, query_mod.TerminalEvent)
    assert terminal.terminal.reason == "aborted_streaming"
    tool_result = next(event for event in events if isinstance(event, ToolResultMessage))
    assert tool_result.message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": TOOL_CANCEL_MESSAGE,
                "is_error": True,
            }
        ],
    }
