"""Goal-runner event models and sink helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal, Protocol, cast

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.goals.models import GoalState, GoalStatus

GoalEventType = Literal[
    "goal_created",
    "goal_updated",
    "goal_resumed",
    "goal_continuation_started",
    "goal_turn_started",
    "goal_turn_accounted",
    "goal_budget_limited",
    "goal_usage_limited",
    "goal_blocked",
    "goal_completed",
    "goal_paused",
    "goal_cancelled",
    "goal_failed",
]
_GOAL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "goal_created",
        "goal_updated",
        "goal_resumed",
        "goal_continuation_started",
        "goal_turn_started",
        "goal_turn_accounted",
        "goal_budget_limited",
        "goal_usage_limited",
        "goal_blocked",
        "goal_completed",
        "goal_paused",
        "goal_cancelled",
        "goal_failed",
    }
)
_GOAL_STATUSES: frozenset[str] = frozenset(
    {
        "active",
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
        "cancelled",
        "failed",
    }
)


def _empty_payload() -> Mapping[str, FrozenJson]:
    return {}


@dataclass(frozen=True, slots=True)
class GoalEvent:
    """Redacted, provider-neutral goal event.

    The payload is metadata-only. Raw transcript, prompt, tool arguments, file
    contents, and provider payloads must not be embedded here.
    """

    id: str
    type: GoalEventType
    sequence: int
    goal_id: str
    session_id: str
    status: GoalStatus
    created_at: float
    data: Mapping[str, FrozenJson] = field(default_factory=_empty_payload)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("GoalEvent.id must be non-empty")
        if self.sequence < 1:
            raise ValueError("GoalEvent.sequence must be >= 1")
        if self.version < 1:
            raise ValueError("GoalEvent.version must be >= 1")
        frozen = freeze_json(self.data)
        if not isinstance(frozen, Mapping):
            raise TypeError("GoalEvent.data must be a JSON object")
        object.__setattr__(self, "data", cast(Mapping[str, FrozenJson], frozen))


class GoalEventSink(Protocol):
    """Observation sink for goal events."""

    def emit(self, event: GoalEvent) -> None:
        """Record or forward one goal event."""
        ...


class InMemoryGoalEventSink:
    """Thread-safe in-memory sink for tests and lightweight embedders."""

    def __init__(self) -> None:
        self._events: list[GoalEvent] = []
        self._lock = RLock()

    def emit(self, event: GoalEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[GoalEvent, ...]:
        with self._lock:
            return tuple(self._events)


class GoalEventEmitter:
    """Small fail-soft goal event fanout with process-local sequence ids."""

    def __init__(
        self,
        sinks: Iterable[GoalEventSink] = (),
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sinks: list[GoalEventSink] = list(sinks)
        self._clock = clock
        self._sequence = 0
        self._lock = RLock()

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def add_sink(self, sink: GoalEventSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def emit(
        self,
        event_type: GoalEventType,
        *,
        state: GoalState,
        data: Mapping[str, object] | None = None,
    ) -> GoalEvent:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            sinks = tuple(self._sinks)
        event = GoalEvent(
            id=f"goal_event_{sequence:06d}",
            type=event_type,
            sequence=sequence,
            goal_id=state.goal_id,
            session_id=state.session_id,
            status=state.status,
            created_at=self._clock(),
            data=goal_event_payload(state, data=data),
        )
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                continue
        return event


def goal_event_payload(
    state: GoalState,
    *,
    data: Mapping[str, object] | None = None,
) -> Mapping[str, FrozenJson]:
    """Build a redacted metadata-only payload for a goal state."""

    payload: dict[str, object] = {
        "goal_id": state.goal_id,
        "session_id": state.session_id,
        "status": state.status,
        "objective_char_count": len(state.spec.objective),
        "success_criteria_count": len(state.spec.success_criteria),
        "constraints_count": len(state.spec.constraints),
        "non_goals_count": len(state.spec.non_goals),
        "expected_outputs_count": len(state.spec.expected_outputs),
        "acceptance_checks_count": len(state.spec.acceptance_checks),
        "turn_count": state.turn_count,
        "tokens_used": state.tokens_used,
        "token_budget": state.token_budget,
        "time_used_s": state.time_used_s,
        "blocked_turn_count": state.blocked_turn_count,
        "summary_char_count": len(state.summary) if state.summary is not None else 0,
        "checkpoint_count": len(state.checkpoints),
        "plan_step_count": len(state.plan_steps),
        "artifact_count": len(state.artifacts),
        "pending_task_count": len(state.pending_task_ids),
        "last_reason_char_count": len(state.last_reason)
        if state.last_reason is not None
        else 0,
    }
    if data:
        payload.update(data)
    frozen = freeze_json(payload)
    if not isinstance(frozen, Mapping):
        raise TypeError("Goal event payload must freeze to a JSON object")
    return cast(Mapping[str, FrozenJson], frozen)


def goal_event_to_dict(event: GoalEvent) -> dict[str, object]:
    """Return a JSON-serializable goal event snapshot."""

    return {
        "id": event.id,
        "type": event.type,
        "sequence": event.sequence,
        "goal_id": event.goal_id,
        "session_id": event.session_id,
        "status": event.status,
        "created_at": event.created_at,
        "data": _jsonable(event.data),
        "version": event.version,
    }


def goal_event_from_dict(data: Mapping[str, object]) -> GoalEvent:
    """Rehydrate a goal event from a JSON-like dictionary."""

    return GoalEvent(
        id=_required_str(data, "id"),
        type=_required_event_type(data),
        sequence=_required_int(data, "sequence"),
        goal_id=_required_str(data, "goal_id"),
        session_id=_required_str(data, "session_id"),
        status=_required_goal_status(data),
        created_at=_required_float(data, "created_at"),
        data=_event_data(data.get("data", {})),
        version=_optional_int(data.get("version"), default=1),
    )


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"GoalEvent field {key!r} must be a non-empty string")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"GoalEvent field {key!r} must be an integer")
    return value


def _optional_int(value: object, *, default: int) -> int:
    return value if isinstance(value, int) else default


def _required_event_type(data: Mapping[str, object]) -> GoalEventType:
    value = _required_str(data, "type")
    if value not in _GOAL_EVENT_TYPES:
        raise ValueError(f"GoalEvent field 'type' is unknown: {value}")
    return cast(GoalEventType, value)


def _required_goal_status(data: Mapping[str, object]) -> GoalStatus:
    value = _required_str(data, "status")
    if value not in _GOAL_STATUSES:
        raise ValueError(f"GoalEvent field 'status' is unknown: {value}")
    return cast(GoalStatus, value)


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"GoalEvent field {key!r} must be numeric")


def _event_data(value: object) -> Mapping[str, FrozenJson]:
    if isinstance(value, Mapping):
        frozen = freeze_json(cast(Mapping[str, object], value))
        if isinstance(frozen, Mapping):
            return cast(Mapping[str, FrozenJson], frozen)
    raise ValueError("GoalEvent field 'data' must be an object")


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, tuple | list):
        items = cast(tuple[object, ...] | list[object], value)
        return [_jsonable(item) for item in items]
    return value
