from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import (
    JsonlKernelEventSink,
    KernelEvent,
    KernelEventBus,
    KernelEventContext,
    NoopKernelEventBus,
    RecordingKernelEventSink,
    freeze_event_data,
    redacted_payload,
)
from raygent_harness.core.task import AppStateStore


def test_event_bus_assigns_stable_sequence_timestamps_and_immutable_payload() -> None:
    sink = RecordingKernelEventSink()
    bus = KernelEventBus(
        [sink],
        clock=lambda: 123.25,
        id_factory=lambda sequence: f"event-{sequence}",
    )
    context = KernelEventContext(
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
        turn_id="turn-1",
        span_id="span-1",
        source="query",
    )

    first = bus.emit(
        "query.turn.started",
        context=context,
        data={"message_count": 3, "nested": {"tool_count": 2}},
    )
    second = bus.emit("query.turn.completed", context=context)

    assert first.id == "event-1"
    assert first.sequence == 1
    assert first.created_at == 123.25
    assert first.source == "query"
    assert first.session_id == "session-1"
    assert first.runtime_session_id == "runtime-1"
    assert first.agent_id == "agent-1"
    assert first.turn_id == "turn-1"
    assert first.span_id == "span-1"
    assert first.data["message_count"] == 3
    assert second.id == "event-2"
    assert second.sequence == 2
    assert sink.events == [first, second]
    assert sink.event_types == ("query.turn.started", "query.turn.completed")

    with pytest.raises(TypeError):
        cast(Any, first.data)["message_count"] = 4
    with pytest.raises(TypeError):
        cast(Any, first.data["nested"])["tool_count"] = 3
    with pytest.raises(FrozenInstanceError):
        cast(Any, first).type = "mutated"


def test_event_bus_is_fail_soft_and_continues_to_other_sinks() -> None:
    class FailingSink:
        def emit(self, event: KernelEvent) -> None:
            assert event.type == "model.request.started"
            raise RuntimeError("sink offline")

    recording = RecordingKernelEventSink()
    bus = KernelEventBus([FailingSink(), recording], id_factory=lambda sequence: str(sequence))

    event = bus.emit("model.request.started")

    assert recording.events == [event]
    assert len(bus.sink_errors) == 1
    error = bus.sink_errors[0]
    assert error.event_id == "1"
    assert error.event_type == "model.request.started"
    assert error.sink == "FailingSink"
    assert error.error_type == "RuntimeError"
    assert error.message == "sink offline"
    assert bus.dropped_sink_error_count == 0

    bus.clear_sink_errors()
    assert bus.sink_errors == ()
    assert bus.dropped_sink_error_count == 0


def test_event_bus_bounds_sink_error_retention() -> None:
    class FailingSink:
        def emit(self, event: KernelEvent) -> None:
            raise RuntimeError(f"failed {event.sequence}")

    bus = KernelEventBus(
        [FailingSink()],
        id_factory=lambda sequence: str(sequence),
        max_sink_errors=2,
    )

    for _ in range(4):
        bus.emit("query.iteration.started")

    assert [error.event_id for error in bus.sink_errors] == ["3", "4"]
    assert [error.message for error in bus.sink_errors] == ["failed 3", "failed 4"]
    assert bus.dropped_sink_error_count == 2


def test_event_bus_can_drop_all_sink_errors_when_configured() -> None:
    class FailingSink:
        def emit(self, event: KernelEvent) -> None:
            raise RuntimeError(f"failed {event.sequence}")

    bus = KernelEventBus([FailingSink()], max_sink_errors=0)

    bus.emit("query.iteration.started")

    assert bus.sink_errors == ()
    assert bus.dropped_sink_error_count == 1


def test_late_attached_sinks_do_not_receive_backlog() -> None:
    sink = RecordingKernelEventSink()
    bus = KernelEventBus(id_factory=lambda sequence: f"evt-{sequence}")

    before_attach = bus.emit("query.turn.started")
    bus.add_sink(sink)
    after_attach = bus.emit("query.turn.completed")
    bus.remove_sink(sink)
    after_detach = bus.emit("query.terminal")

    assert before_attach.sequence == 1
    assert after_attach.sequence == 2
    assert after_detach.sequence == 3
    assert sink.events == [after_attach]


def test_context_helpers_preserve_parent_child_correlation() -> None:
    parent = KernelEventContext(
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="parent-agent",
        turn_id="turn-1",
        span_id="parent-span",
        source="query",
    )

    child_agent = parent.for_child_agent("child-agent")
    child_span = child_agent.for_child_span("tool-span", source="tool")
    iteration = child_span.with_iteration(3)

    assert child_agent.agent_id == "child-agent"
    assert child_agent.parent_agent_id == "parent-agent"
    assert child_agent.parent_span_id == "parent-span"
    assert child_span.span_id == "tool-span"
    assert child_span.parent_span_id == "parent-span"
    assert child_span.source == "tool"
    assert iteration.iteration == 3
    assert iteration.session_id == "session-1"
    assert iteration.runtime_session_id == "runtime-1"
    assert iteration.turn_id == "turn-1"


def test_recording_sink_supports_lightweight_eval_filters() -> None:
    sink = RecordingKernelEventSink()
    bus = KernelEventBus([sink])

    first_context = KernelEventContext(agent_id="agent-1", span_id="span-1")
    second_context = KernelEventContext(agent_id="agent-2", span_id="span-2")
    first = bus.emit("tool.call.started", context=first_context, data={"index": 1})
    second = bus.emit("tool.call.completed", context=first_context, data={"index": 2})
    third = bus.emit("tool.call.started", context=second_context, data={"index": 3})

    assert sink.by_type("tool.call.started") == (first, third)
    assert sink.by_agent_id("agent-1") == (first, second)
    assert sink.by_span_id("span-2") == (third,)
    assert sink.matching(lambda event: event.sequence % 2 == 0) == (second,)


def test_jsonl_sink_records_sanitized_event_envelopes(tmp_path: Path) -> None:
    path = tmp_path / "events" / "trace.jsonl"
    bus = KernelEventBus(
        [JsonlKernelEventSink(path)],
        clock=lambda: 42.0,
        id_factory=lambda sequence: f"evt-{sequence}",
    )

    bus.emit(
        "memory.recall.completed",
        context=KernelEventContext(
            session_id="s",
            agent_id="agent-1",
            turn_id="turn-1",
            source="memory",
        ),
        data={
            "status": "completed",
            "counts": {"surfaced": 2},
            "paths": ("memory.md",),
        },
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "agent_id": "agent-1",
            "content_policy": "metadata_only",
            "created_at": 42.0,
            "data": {
                "counts": {"surfaced": 2},
                "paths": ["memory.md"],
                "status": "completed",
            },
            "id": "evt-1",
            "iteration": None,
            "parent_agent_id": None,
            "parent_span_id": None,
            "runtime_session_id": None,
            "sequence": 1,
            "session_id": "s",
            "source": "memory",
            "span_id": None,
            "turn_id": "turn-1",
            "type": "memory.recall.completed",
            "version": 1,
        }
    ]


def test_redaction_and_freeze_helpers_are_json_safe() -> None:
    data = freeze_event_data(
        {
            "prompt": redacted_payload("raw_prompt", char_count=12),
            "tool_names": ["Read", "Edit"],
        }
    )

    prompt = cast(dict[str, object], data["prompt"])
    assert prompt["redacted"] is True
    assert prompt["reason"] == "raw_prompt"
    assert prompt["char_count"] == 12
    assert data["tool_names"] == ("Read", "Edit")

    with pytest.raises(TypeError):
        freeze_event_data({"bad": object()})


def test_query_deps_default_observability_is_noop_bus() -> None:
    deps = QueryDeps(task_store=AppStateStore())

    assert isinstance(deps.observability, NoopKernelEventBus)
    event = deps.observability.emit("query.turn.started")
    assert event.type == "query.turn.started"
    assert deps.observability.sink_errors == ()
