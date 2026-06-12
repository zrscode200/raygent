"""Provider-neutral kernel observability primitives.

The event bus is intentionally observation-only: sinks receive immutable facts
about kernel behavior, but sink failures and late attachment never affect query,
tool, task, or memory semantics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Literal, Protocol, cast

from raygent_harness.core.model_types import FrozenJson, freeze_json

ContentPolicy = Literal["metadata_only", "redacted", "content_opt_in"]
EventData = Mapping[str, FrozenJson]


def _empty_event_data() -> EventData:
    return MappingProxyType({})


def _empty_event_list() -> list[KernelEvent]:
    return []


def freeze_event_data(data: Mapping[str, object] | None = None) -> EventData:
    """Freeze a JSON-like event payload as an immutable object mapping."""

    if data is None:
        return _empty_event_data()
    frozen = freeze_json(data)
    if not isinstance(frozen, Mapping):
        raise TypeError("Kernel event data must freeze to a JSON object")
    return cast(EventData, frozen)


def metadata_only_data(data: Mapping[str, object] | None = None) -> EventData:
    """Declare a payload as safe metadata, not raw model/tool/user content."""

    return freeze_event_data(data)


def redacted_payload(
    reason: str = "content_redacted",
    *,
    char_count: int | None = None,
    byte_count: int | None = None,
) -> FrozenJson:
    """Return a structured redaction marker for unsafe content fields."""

    data: dict[str, object] = {
        "redacted": True,
        "reason": reason,
    }
    if char_count is not None:
        data["char_count"] = char_count
    if byte_count is not None:
        data["byte_count"] = byte_count
    return freeze_json(data)


@dataclass(frozen=True, slots=True)
class KernelEventContext:
    """Correlation metadata threaded through kernel event producers."""

    session_id: str | None = None
    runtime_session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    turn_id: str | None = None
    iteration: int | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    source: str = "core"

    def for_child_agent(self, agent_id: str) -> KernelEventContext:
        """Return context for a child agent while preserving parent lineage."""

        return replace(
            self,
            agent_id=agent_id,
            parent_agent_id=self.agent_id,
            span_id=None,
            parent_span_id=self.span_id,
        )

    def for_child_span(
        self,
        span_id: str,
        *,
        source: str | None = None,
    ) -> KernelEventContext:
        """Return context for a nested span under the current span."""

        return replace(
            self,
            span_id=span_id,
            parent_span_id=self.span_id or self.parent_span_id,
            source=source or self.source,
        )

    def with_iteration(self, iteration: int) -> KernelEventContext:
        """Return context for a query-loop iteration."""

        return replace(self, iteration=iteration)

    def with_source(self, source: str) -> KernelEventContext:
        """Return context with a more specific producer namespace."""

        return replace(self, source=source)


@dataclass(frozen=True, slots=True)
class KernelEvent:
    """Immutable provider-neutral event envelope emitted by the harness kernel."""

    id: str
    type: str
    sequence: int
    created_at: float
    source: str
    data: EventData = field(default_factory=_empty_event_data)
    version: int = 1
    session_id: str | None = None
    runtime_session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    turn_id: str | None = None
    iteration: int | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    content_policy: ContentPolicy = "metadata_only"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("KernelEvent.id must be non-empty")
        if not self.type:
            raise ValueError("KernelEvent.type must be non-empty")
        if self.sequence < 1:
            raise ValueError("KernelEvent.sequence must be >= 1")
        if self.version < 1:
            raise ValueError("KernelEvent.version must be >= 1")
        object.__setattr__(self, "data", freeze_event_data(self.data))


class KernelEventSink(Protocol):
    """Observation sink. Implementations must not mutate kernel behavior."""

    def emit(self, event: KernelEvent) -> None:
        """Record or forward one event."""
        ...


@dataclass(frozen=True, slots=True)
class KernelEventSinkError:
    """Best-effort record of a sink failure swallowed by the event bus."""

    event_id: str
    event_type: str
    sink: str
    error_type: str
    message: str


class KernelEventBus:
    """Synchronous fail-soft fanout bus for kernel observability events."""

    def __init__(
        self,
        sinks: Iterable[KernelEventSink] = (),
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[int], str] | None = None,
        max_sink_errors: int = 128,
    ) -> None:
        if max_sink_errors < 0:
            raise ValueError("max_sink_errors must be >= 0")
        self._sinks: list[KernelEventSink] = list(sinks)
        self._clock = clock
        self._id_factory = id_factory or _default_event_id
        self._sequence = 0
        self._max_sink_errors = max_sink_errors
        self._sink_errors: list[KernelEventSinkError] = []
        self._dropped_sink_error_count = 0
        self._lock = RLock()

    def add_sink(self, sink: KernelEventSink) -> None:
        """Attach a sink for future events.

        Core intentionally does not replay prior events on late attachment.
        Adapters that need startup buffering should wrap a sink outside core.
        """

        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: KernelEventSink) -> None:
        """Detach one sink instance if present."""

        with self._lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                return

    @property
    def sequence(self) -> int:
        """Last allocated event sequence."""

        with self._lock:
            return self._sequence

    @property
    def sink_errors(self) -> tuple[KernelEventSinkError, ...]:
        """Sink failures observed so far. Failures are never re-raised."""

        with self._lock:
            return tuple(self._sink_errors)

    @property
    def dropped_sink_error_count(self) -> int:
        """Number of older sink errors discarded from the bounded buffer."""

        with self._lock:
            return self._dropped_sink_error_count

    def clear_sink_errors(self) -> None:
        with self._lock:
            self._sink_errors.clear()
            self._dropped_sink_error_count = 0

    def emit(
        self,
        event_type: str,
        *,
        context: KernelEventContext | None = None,
        data: Mapping[str, object] | None = None,
        content_policy: ContentPolicy = "metadata_only",
        version: int = 1,
        source: str | None = None,
    ) -> KernelEvent:
        """Create and fan out an immutable event.

        Returns the created event for tests/eval harnesses. Sink exceptions are
        captured in `sink_errors` and never escape this method.
        """

        event_context = context or KernelEventContext()
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            sink_snapshot = tuple(self._sinks)

        event = KernelEvent(
            id=self._id_factory(sequence),
            type=event_type,
            sequence=sequence,
            created_at=self._clock(),
            source=source or event_context.source,
            data=freeze_event_data(data),
            version=version,
            session_id=event_context.session_id,
            runtime_session_id=event_context.runtime_session_id,
            agent_id=event_context.agent_id,
            parent_agent_id=event_context.parent_agent_id,
            turn_id=event_context.turn_id,
            iteration=event_context.iteration,
            span_id=event_context.span_id,
            parent_span_id=event_context.parent_span_id,
            content_policy=content_policy,
        )

        for sink in sink_snapshot:
            try:
                sink.emit(event)
            except Exception as exc:  # pragma: no cover - exercised by behavior.
                self._record_sink_error(event, sink, exc)

        return event

    def _record_sink_error(
        self,
        event: KernelEvent,
        sink: KernelEventSink,
        exc: Exception,
    ) -> None:
        error = KernelEventSinkError(
            event_id=event.id,
            event_type=event.type,
            sink=type(sink).__name__,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        with self._lock:
            if self._max_sink_errors == 0:
                self._dropped_sink_error_count += 1
                return
            if len(self._sink_errors) >= self._max_sink_errors:
                self._sink_errors.pop(0)
                self._dropped_sink_error_count += 1
            self._sink_errors.append(error)


class NoopKernelEventBus(KernelEventBus):
    """Default event bus with no attached sinks."""

    def __init__(self) -> None:
        super().__init__(())


@dataclass
class RecordingKernelEventSink:
    """In-memory sink for tests, evals, and local debugging."""

    events: list[KernelEvent] = field(default_factory=_empty_event_list)

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(event.type for event in self.events)

    def by_type(self, event_type: str) -> tuple[KernelEvent, ...]:
        return tuple(event for event in self.events if event.type == event_type)

    def by_agent_id(self, agent_id: str | None) -> tuple[KernelEvent, ...]:
        return tuple(event for event in self.events if event.agent_id == agent_id)

    def by_span_id(self, span_id: str | None) -> tuple[KernelEvent, ...]:
        return tuple(event for event in self.events if event.span_id == span_id)

    def matching(
        self,
        predicate: Callable[[KernelEvent], bool],
    ) -> tuple[KernelEvent, ...]:
        return tuple(event for event in self.events if predicate(event))


@dataclass(frozen=True, slots=True)
class JsonlKernelEventSink:
    """Append sanitized kernel events to JSONL outside the model transcript."""

    path: str | Path
    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def emit(self, event: KernelEvent) -> None:
        payload = _event_to_jsonable_dict(event)
        path = Path(self.path)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                handle.write("\n")


def _default_event_id(sequence: int) -> str:
    return f"evt_{sequence:012d}"


def _event_to_jsonable_dict(event: KernelEvent) -> dict[str, object]:
    return {
        "id": event.id,
        "type": event.type,
        "sequence": event.sequence,
        "created_at": event.created_at,
        "source": event.source,
        "version": event.version,
        "session_id": event.session_id,
        "runtime_session_id": event.runtime_session_id,
        "agent_id": event.agent_id,
        "parent_agent_id": event.parent_agent_id,
        "turn_id": event.turn_id,
        "iteration": event.iteration,
        "span_id": event.span_id,
        "parent_span_id": event.parent_span_id,
        "content_policy": event.content_policy,
        "data": _jsonable(event.data),
    }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, tuple | list):
        items = cast(Sequence[object], value)
        return [_jsonable(item) for item in items]
    raise TypeError(f"Kernel event contains non-JSON value: {type(value).__name__}")


__all__ = [
    "ContentPolicy",
    "EventData",
    "JsonlKernelEventSink",
    "KernelEvent",
    "KernelEventBus",
    "KernelEventContext",
    "KernelEventSink",
    "KernelEventSinkError",
    "NoopKernelEventBus",
    "RecordingKernelEventSink",
    "freeze_event_data",
    "metadata_only_data",
    "redacted_payload",
]
