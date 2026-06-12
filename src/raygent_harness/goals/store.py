"""Goal state persistence protocols and in-memory implementation."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

from raygent_harness.goals.events import (
    GoalEvent,
    GoalEventEmitter,
    GoalEventType,
    goal_event_from_dict,
    goal_event_to_dict,
)
from raygent_harness.goals.models import (
    GoalState,
    GoalStatus,
    goal_state_from_dict,
    goal_state_to_dict,
)

DEFAULT_GOAL_STORE_DIR = ".raygent/goals"
_MAX_SAFE_COMPONENT_PREFIX_CHARS = 96
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


class GoalStoreError(Exception):
    """Base exception for goal store failures."""


class GoalNotFoundError(GoalStoreError):
    """Raised when a requested goal id does not exist."""


class GoalAlreadyExistsError(GoalStoreError):
    """Raised when attempting to create a duplicate goal id."""


class GoalStateConflictError(GoalStoreError):
    """Raised when an update precondition fails."""


class GoalStore(Protocol):
    """Durable goal-state store.

    Implementations may be memory-only, JSONL-backed, database-backed, or
    product-provided. The kernel uses this protocol instead of selecting a DB.
    """

    def create(self, state: GoalState) -> GoalState:
        """Persist a new goal state."""
        ...

    def get(self, goal_id: str) -> GoalState | None:
        """Return one goal state by id."""
        ...

    def get_active_for_session(self, session_id: str) -> GoalState | None:
        """Return the currently active goal for a session, if any."""
        ...

    def update(
        self,
        goal_id: str,
        updater: Callable[[GoalState], GoalState],
    ) -> GoalState:
        """Atomically update one goal state."""
        ...

    def list_for_session(self, session_id: str) -> tuple[GoalState, ...]:
        """Return all goal states for a session."""
        ...

    def append_event(self, event: GoalEvent) -> None:
        """Append a goal event."""
        ...

    def list_events(self, goal_id: str | None = None) -> tuple[GoalEvent, ...]:
        """Return events, optionally filtered by goal id."""
        ...


@dataclass
class InMemoryGoalStore:
    """Thread-safe in-memory goal store for tests and lightweight embeddings."""

    event_emitter: GoalEventEmitter | None = None
    _states: dict[str, GoalState] = field(default_factory=dict[str, GoalState])
    _events: list[GoalEvent] = field(default_factory=list[GoalEvent])
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def create(self, state: GoalState) -> GoalState:
        with self._lock:
            if state.goal_id in self._states:
                raise GoalAlreadyExistsError(f"Goal already exists: {state.goal_id}")
            if self.get_active_for_session(state.session_id) is not None:
                raise GoalStateConflictError(
                    f"Session already has an active goal: {state.session_id}"
                )
            self._states[state.goal_id] = state
            self._emit_locked("goal_created", state=state)
            return state

    def get(self, goal_id: str) -> GoalState | None:
        with self._lock:
            return self._states.get(goal_id)

    def get_active_for_session(self, session_id: str) -> GoalState | None:
        with self._lock:
            for state in self._states.values():
                if state.session_id == session_id and state.status == "active":
                    return state
            return None

    def update(
        self,
        goal_id: str,
        updater: Callable[[GoalState], GoalState],
    ) -> GoalState:
        with self._lock:
            current = self._states.get(goal_id)
            if current is None:
                raise GoalNotFoundError(f"Goal not found: {goal_id}")
            updated = updater(current)
            if updated.goal_id != goal_id:
                raise GoalStateConflictError("Goal updater cannot change goal_id")
            active = self.get_active_for_session(updated.session_id)
            if (
                updated.status == "active"
                and active is not None
                and active.goal_id != goal_id
            ):
                raise GoalStateConflictError(
                    f"Session already has an active goal: {updated.session_id}"
                )
            self._states[goal_id] = updated
            event_type = _event_type_for_transition(current.status, updated.status)
            self._emit_locked(event_type, state=updated)
            return updated

    def list_for_session(self, session_id: str) -> tuple[GoalState, ...]:
        with self._lock:
            return tuple(
                state for state in self._states.values() if state.session_id == session_id
            )

    def append_event(self, event: GoalEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(self, goal_id: str | None = None) -> tuple[GoalEvent, ...]:
        with self._lock:
            if goal_id is None:
                return tuple(self._events)
            return tuple(event for event in self._events if event.goal_id == goal_id)

    def _emit_locked(self, event_type: GoalEventType, *, state: GoalState) -> None:
        if self.event_emitter is None:
            return
        event = self.event_emitter.emit(event_type, state=state)
        self._events.append(event)


class JsonGoalStore:
    """Filesystem-backed goal store.

    State lives under
    `<base>/<safe-session-id>/states/<safe-goal-id>.json`. Events are stored as
    JSONL under `<base>/<safe-session-id>/events.jsonl`.

    The store is intentionally small and product-neutral: it provides durable
    local persistence for kernels/tests, while products can still inject a DB or
    service-backed `GoalStore` implementation. It is a single-writer local
    store; multi-process or distributed writers should use a transactional
    product-provided `GoalStore`.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
        event_emitter: GoalEventEmitter | None = None,
    ) -> None:
        self.base_dir = resolve_goal_store_base_dir(
            base_dir,
            project_root=project_root,
        )
        self.event_emitter = event_emitter
        self._lock = RLock()

    def state_dir(self, session_id: str) -> Path:
        path = (self.base_dir / safe_goal_component(session_id) / "states").resolve()
        if not _is_relative_to(path, self.base_dir):
            raise ValueError(f"goal state directory escapes base directory: {path}")
        return path

    def state_path(self, session_id: str, goal_id: str) -> Path:
        state_dir = self.state_dir(session_id)
        path = state_dir / f"{safe_goal_component(goal_id)}.json"
        if path.parent != state_dir:
            raise ValueError(f"goal state path escapes base directory: {path}")
        return path

    def events_path(self, session_id: str) -> Path:
        path = (self.base_dir / safe_goal_component(session_id) / "events.jsonl").resolve()
        if not _is_relative_to(path, self.base_dir):
            raise ValueError(f"goal event path escapes base directory: {path}")
        return path

    def create(self, state: GoalState) -> GoalState:
        with self._lock:
            if self.get(state.goal_id) is not None:
                raise GoalAlreadyExistsError(f"Goal already exists: {state.goal_id}")
            if self.get_active_for_session(state.session_id) is not None:
                raise GoalStateConflictError(
                    f"Session already has an active goal: {state.session_id}"
                )
            _write_goal_state(self.state_path(state.session_id, state.goal_id), state)
            self._emit_locked("goal_created", state=state)
            return state

    def get(self, goal_id: str) -> GoalState | None:
        with self._lock:
            for path in self._state_paths_for_goal(goal_id):
                state = _read_goal_state(path)
                if state is not None and state.goal_id == goal_id:
                    return state
            return None

    def get_active_for_session(self, session_id: str) -> GoalState | None:
        with self._lock:
            for state in self.list_for_session(session_id):
                if state.status == "active":
                    return state
            return None

    def update(
        self,
        goal_id: str,
        updater: Callable[[GoalState], GoalState],
    ) -> GoalState:
        with self._lock:
            current = self.get(goal_id)
            if current is None:
                raise GoalNotFoundError(f"Goal not found: {goal_id}")
            updated = updater(current)
            if updated.goal_id != goal_id:
                raise GoalStateConflictError("Goal updater cannot change goal_id")
            active = self.get_active_for_session(updated.session_id)
            if (
                updated.status == "active"
                and active is not None
                and active.goal_id != goal_id
            ):
                raise GoalStateConflictError(
                    f"Session already has an active goal: {updated.session_id}"
                )
            _write_goal_state(self.state_path(updated.session_id, updated.goal_id), updated)
            if updated.session_id != current.session_id:
                old_path = self.state_path(current.session_id, current.goal_id)
                with contextlib.suppress(FileNotFoundError):
                    old_path.unlink()
            event_type = _event_type_for_transition(current.status, updated.status)
            self._emit_locked(event_type, state=updated)
            return updated

    def list_for_session(self, session_id: str) -> tuple[GoalState, ...]:
        with self._lock:
            state_dir = self.state_dir(session_id)
            if not state_dir.exists():
                return ()
            states: list[GoalState] = []
            for path in sorted(state_dir.glob("*.json")):
                state = _read_goal_state(path)
                if state is None or state.session_id != session_id:
                    continue
                states.append(state)
            return tuple(states)

    def append_event(self, event: GoalEvent) -> None:
        with self._lock:
            _append_goal_event(self.events_path(event.session_id), event)

    def list_events(self, goal_id: str | None = None) -> tuple[GoalEvent, ...]:
        with self._lock:
            events: list[GoalEvent] = []
            for path in sorted(self.base_dir.glob("*/events.jsonl")):
                events.extend(_read_goal_events(path))
            if goal_id is None:
                return tuple(events)
            return tuple(event for event in events if event.goal_id == goal_id)

    def _state_paths_for_goal(self, goal_id: str) -> tuple[Path, ...]:
        safe_goal = safe_goal_component(goal_id)
        return tuple(sorted(self.base_dir.glob(f"*/states/{safe_goal}.json")))

    def _emit_locked(self, event_type: GoalEventType, *, state: GoalState) -> None:
        if self.event_emitter is None:
            return
        event = self.event_emitter.emit(event_type, state=state)
        _append_goal_event(self.events_path(state.session_id), event)


def _event_type_for_transition(
    old_status: GoalStatus,
    new_status: GoalStatus,
) -> GoalEventType:
    if old_status == new_status:
        return "goal_updated"
    event_types: dict[GoalStatus, GoalEventType] = {
        "active": "goal_resumed",
        "paused": "goal_paused",
        "blocked": "goal_blocked",
        "usage_limited": "goal_usage_limited",
        "budget_limited": "goal_budget_limited",
        "complete": "goal_completed",
        "cancelled": "goal_cancelled",
        "failed": "goal_failed",
    }
    return event_types.get(new_status, "goal_updated")


def resolve_goal_store_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / DEFAULT_GOAL_STORE_DIR).resolve()


def safe_goal_component(value: str) -> str:
    if value == "":
        raise ValueError("goal path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    prefix = normalized[:_MAX_SAFE_COMPONENT_PREFIX_CHARS].rstrip("._")
    if prefix == "":
        return f"id-{digest}"
    return f"{prefix}-{digest}"


def _write_goal_state(path: Path, state: GoalState) -> None:
    payload = json.dumps(
        _jsonable(goal_state_to_dict(state)),
        sort_keys=True,
        separators=(",", ":"),
    )
    _atomic_write_text(path, payload + "\n")


def _read_goal_state(path: Path) -> GoalState | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return goal_state_from_dict(cast(dict[str, object], raw))
    except Exception:
        return None


def _append_goal_event(path: Path, event: GoalEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(goal_event_to_dict(event)),
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()


def _read_goal_events(path: Path) -> tuple[GoalEvent, ...]:
    if not path.exists():
        return ()
    events: list[GoalEvent] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                events.append(goal_event_from_dict(cast(dict[str, object], raw)))
            except Exception:
                continue
    except OSError:
        return ()
    return tuple(events)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, tuple | list):
        items = cast(tuple[object, ...] | list[object], value)
        return [_jsonable(item) for item in items]
    return value
