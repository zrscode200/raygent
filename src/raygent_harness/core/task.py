"""Task vtable + TaskStateBase + registry.

Task is a
narrow lifecycle wrapper for long-running or backgroundable work — NOT the
agent loop. The vtable is tiny: `{ name, type, kill }`. Everything else
(register, complete, fail, update-progress, notify) is bespoke per type.

Per-type state lives in its own module (tasks/local_bash.py, tasks/local_agent.py, ...)
and extends TaskStateBase. AppState-equivalent holds the tasks dict keyed by task_id.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, runtime_checkable

from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    NoopKernelEventBus,
)

TaskType = Literal[
    "local_bash",
    "local_agent",
    "remote_agent",
    "in_process_teammate",
    # Future: "local_workflow", "monitor_mcp", "dream"
]

TaskStatus = Literal["pending", "running", "completed", "failed", "killed"]

TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {"completed", "failed", "killed"}
)


def is_terminal_task_status(status: TaskStatus) -> bool:
    """Guards eviction + UI paths so dead tasks don't receive injected messages."""
    return status in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# TaskStateBase — shared across all task types.
# ---------------------------------------------------------------------------


@dataclass
class TaskStateBase:
    """Fields every task type shares. Subclasses add type-specific fields.

    Modeled as a mutable dataclass so updates flow via a state-store pattern
    (see AppStateStore.update_task). The store enforces immutable-from-outside
    semantics by cloning on update. Internally the dataclass stays mutable so
    we don't pay Pydantic's revalidation cost per progress tick.
    """

    # Identity
    id: str
    type: TaskType
    description: str

    # Lifecycle
    status: TaskStatus
    start_time: float
    """Monotonic-ish timestamp (time.time()). For ordering + duration only."""

    end_time: float | None = None
    total_paused_ms: int = 0
    """Accumulated time the task was paused/suspended (future backgrounding)."""

    # Provenance
    tool_use_id: str | None = None
    """If this task was spawned by a tool call, the tool_use_id that started it."""

    # Disk-backed output / transcript reference
    output_file: str | None = None
    """Path or opaque string for per-task output/transcript storage."""

    output_offset: int = 0
    """Byte offset of next unread output. Used by TaskOutput reader."""

    # Notification gate
    notified: bool = False
    """Atomic check-and-set flag. Prevents duplicate notifications when multiple
    paths (explicit kill, async completion, bulk kill) can mark a task terminal."""


# ---------------------------------------------------------------------------
# Task vtable — the narrow interface, one per type.
# ---------------------------------------------------------------------------


@runtime_checkable
class Task(Protocol):
    """The Task vtable. Intentionally tiny — kill is the only cross-cutting op.

    Registration, progress updates, completion, and notification are per-type
    functions exported from each task's module. Forcing them onto the vtable
    would add polymorphism without simplifying any caller.
    """

    name: str
    type: TaskType

    async def kill(self, task_id: str, store: AppStateStore) -> None:
        """Cancel the task. Must be idempotent — callable on already-terminal
        tasks without raising. Sets status to 'killed' if currently running."""
        ...


# ---------------------------------------------------------------------------
# TaskNotification — structured model-facing signal from a task.
# Shared queue on AppStateStore.
# ---------------------------------------------------------------------------


NotificationKind = Literal["completed", "stalled", "error"]
"""v1 kinds. `"progress"` deliberately omitted — background tasks emitting
progress every few seconds would blow the next iteration's context. Add
later with throttling if a concrete use case surfaces."""


NotificationPriority = Literal["now", "next", "later"]
"""Tri-state ordering matching reference's `QueuePriority` at
- `now`: process ahead of user input (reserved — used for system-urgent signals).
- `next`: next iteration. Default for task completions.
- `later`: lowest; pending-notification default at
  messages.
"""


NotificationMode = Literal["inline", "background"]
"""Reserved field. `"background"` = drained at iteration top as a synthetic
user message (current behavior). `"inline"` reserved for future mid-turn
injection. Today always `"background"`."""


@dataclass(frozen=True)
class TaskNotification:
    """Model-facing signal from a task.

    Tasks enqueue completion, stall, and error facts. The query loop drains
    them at iteration boundaries and folds them into synthetic user messages.
    This is separate from user-facing notification sinks.
    """

    task_id: str
    message: str
    """Human-readable summary to fold into the synthetic user message."""

    kind: NotificationKind
    tool_use_id: str | None = None
    """Originating tool_use_id, when the task was spawned by a tool call."""

    priority: NotificationPriority = "later"
    """FIFO within a priority tier. Default `later` avoids starving user input."""

    mode: NotificationMode = "background"
    """Reserved. Always `background` today."""

    agent_id: str | None = None
    """Target agent id. None targets the main thread."""

    created_at: float = field(default_factory=time.time)
    """Wall-clock/source timestamp for telemetry.

    Queue drain order is priority plus FIFO enqueue order, not timestamp order.
    Recovery paths may preserve source timestamps without changing FIFO.
    """

    dedupe_key: str | None = None
    """Optional stable key for replayed/offline notifications.

    Live task drivers normally leave this unset and continue to rely on the
    per-task `notified` flag. Recovery paths set it when the same terminal fact
    may be rediscovered across process restarts, so the shared queue can skip
    duplicate model-facing notifications.
    """


# ---------------------------------------------------------------------------
# AgentRouteRecord — resumable named/raw-id route metadata.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentRouteRecord:
    """Session-local metadata needed to route/resume addressable agents.

    Active task state may be disposable, but this record preserves the facts
    needed to re-register a background local_agent with the same id.
    """

    agent_id: str
    task_id: str
    task_type: TaskType
    name: str | None = None
    parent_agent_id: str | None = None
    parent_session_id: str = ""
    runtime_session_id: str | None = None
    agent_type: str | None = None
    description: str = ""
    model: str | None = None
    system_prompt: str | None = None
    tool_names: tuple[str, ...] = ()
    permission_mode: str | None = None
    cwd: str | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    worktree_slug: str | None = None
    worktree_created_at: float | None = None
    worktree_touched_at: float | None = None
    worktree_cleanup_policy: str | None = None
    transcript_path: str | None = None
    is_sidechain: bool = True
    content_replacement_replay: bool = True
    route_registered_at: float = 0.0
    """Stable registration timestamp used to restore duplicate-name
    latest-wins routing after a process restart. Terminal metadata refreshes
    must preserve this value instead of using completion time."""


# ---------------------------------------------------------------------------
# AppStateStore — small in-process state container.
# ---------------------------------------------------------------------------


_PRIORITY_ORDER: dict[NotificationPriority, int] = {
    "now": 0,
    "next": 1,
    "later": 2,
}


@dataclass
class AppStateStore:
    """Holds the tasks dict + shared notification queue + atomic update helpers.

    This is the Python-side equivalent of `setAppState(prev => {...})`. The TS
    version is a React reducer; here we keep it a plain object with a lock for
    the atomic-update contract (matters for atomic check-and-set on `notified`
    when multiple concurrent paths race to fire notifications).

    Single-threaded asyncio: we rely on no-await between read and write within
    `update_task` / `enqueue_notification` / `drain_notifications` to get
    atomicity. If we later move to threads or multiple event loops, each of
    those methods gets a proper lock.
    """

    tasks: dict[str, TaskStateBase] = field(default_factory=dict[str, TaskStateBase])
    agent_name_registry: dict[str, str] = field(default_factory=dict[str, str])
    """Session-local name -> task_id registry for addressable agents/teammates.

    Reference `AppStateStore.agentNameRegistry` maps human names to agent IDs so
    `SendMessage` can route to an already-running background agent. Raygent stores
    task IDs because task state carries the provider-neutral identity details.
    """

    agent_route_records: dict[str, AgentRouteRecord] = field(
        default_factory=dict[str, AgentRouteRecord]
    )
    """task_id/agent_id -> resumable route metadata.

    Active TeamCreate teammates can remove their name registry entry on stop.
    Ordinary named local agents must keep enough route metadata to support
    stopped/evicted sidechain resume, so their record persists independently of
    active task state.
    """

    notifications: list[TaskNotification] = field(
        default_factory=list[TaskNotification]
    )
    """Shared task-notification queue. All tasks enqueue here; the agent
    loop drains at iteration top. No `consumed: bool` tombstone — drain
    state (`commandQueue`)."""

    notification_dedupe_keys: set[str] = field(default_factory=set[str])
    """Stable replay keys already accepted into or through the queue.

    This is deliberately separate from `TaskStateBase.notified`: live task
    drivers own `notified`, while restart/replay paths need a cross-task queue
    guard for facts rediscovered from durable stores.
    """

    observability: KernelEventBus = field(default_factory=NoopKernelEventBus)
    """Observation-only bus for task lifecycle and notification facts.

    QueryDeps wires this to its own bus in `__post_init__`. Direct task users
    can set it explicitly when they need task events without a full query turn.
    """

    def update_task(
        self,
        task_id: str,
        updater: Callable[[TaskStateBase], TaskStateBase | None],
    ) -> None:
        """Apply updater to the task with task_id. No-op if not found.

        Updater may return None or the same object to signal "no change" — we
        still assign so the reference stays consistent. For atomic read-modify-
        write (e.g., check notified + set notified in one step), the updater
        callback is where the check happens.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return
        previous_status = task.status
        updated = updater(task)
        if updated is not None:
            self.tasks[task_id] = updated
            if (
                previous_status not in TERMINAL_STATUSES
                and updated.status in TERMINAL_STATUSES
            ):
                self._emit_task_terminal(updated)

    def register_task(self, task: TaskStateBase) -> None:
        """Insert a new task. Caller is responsible for uniqueness of id."""
        self.tasks[task.id] = task
        self._emit_task_event("task.registered", task)

    def remove_task(self, task_id: str) -> None:
        self.tasks.pop(task_id, None)

    # ----- Notification queue operations -----

    def enqueue_notification(self, notification: TaskNotification) -> bool:
        """Append a notification. Drain order is (priority, FIFO-within-
        priority).

        Returns True when the notification was accepted. Returns False only
        when `notification.dedupe_key` has already been accepted before.
        """
        if notification.dedupe_key is not None:
            context = KernelEventContext(agent_id=notification.agent_id, source="task")
            if notification.dedupe_key in self.notification_dedupe_keys:
                self.observability.emit(
                    "task.notification.skipped_duplicate",
                    context=context,
                    data=_notification_payload(notification),
                )
                return False
            self.notification_dedupe_keys.add(notification.dedupe_key)

        self.notifications.append(notification)
        context = KernelEventContext(agent_id=notification.agent_id, source="task")
        data = _notification_payload(notification)
        self.observability.emit(
            "task.notification.enqueued",
            context=context,
            data=data,
        )
        if notification.kind == "stalled":
            self.observability.emit("task.stalled", context=context, data=data)
        return True

    def drain_notifications(
        self,
        agent_id: str | None,
    ) -> list[TaskNotification]:
        """Atomically remove and return all notifications for the given
        agent_id, sorted by (priority, FIFO). Non-matching notifications
        stay in the queue.

        `agent_id=None` means "main thread" — drains notifications with
        `agent_id is None`. Subagents pass their own id and only see their
        own notifications. Mirrors the filter-at-dequeue pattern at

        Returned list is in drain order — caller appends them in this
        order as one synthetic user message (or one per, if preferred).
        """
        if not self.notifications:
            return []

        matched: list[TaskNotification] = []
        remaining: list[TaskNotification] = []
        for n in self.notifications:
            if n.agent_id == agent_id:
                matched.append(n)
            else:
                remaining.append(n)

        self.notifications = remaining
        # Python's stable sort preserves the original queue order within a
        # priority tier, matching reference `dequeue()`'s first-best scan.
        matched.sort(key=lambda n: _PRIORITY_ORDER[n.priority])
        if matched:
            self.observability.emit(
                "task.notification.drained",
                context=KernelEventContext(agent_id=agent_id, source="task"),
                data={
                    "agent_id": agent_id,
                    "notification_count": len(matched),
                    "task_ids": tuple(n.task_id for n in matched),
                    "priorities": tuple(n.priority for n in matched),
                    "kinds": tuple(n.kind for n in matched),
                },
            )
        return matched

    def emit_task_progress(
        self,
        task_id: str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Emit a metadata-only task progress fact for existing task state."""

        task = self.tasks.get(task_id)
        if task is None:
            return
        payload = _task_payload(task)
        if data:
            payload.update(data)
        self.observability.emit(
            "task.progress",
            context=_task_event_context(task),
            data=payload,
        )

    def _emit_task_terminal(self, task: TaskStateBase) -> None:
        event_type = {
            "completed": "task.completed",
            "failed": "task.failed",
            "killed": "task.killed",
        }.get(task.status)
        if event_type is None:
            return
        self._emit_task_event(event_type, task)

    def _emit_task_event(self, event_type: str, task: TaskStateBase) -> None:
        self.observability.emit(
            event_type,
            context=_task_event_context(task),
            data=_task_payload(task),
        )


def _task_event_context(task: TaskStateBase) -> KernelEventContext:
    agent_id = getattr(task, "agent_id", None)
    if not isinstance(agent_id, str):
        agent_id = getattr(task, "parent_agent_id", None)
    parent_agent_id = getattr(task, "parent_agent_id", None)
    if not isinstance(parent_agent_id, str):
        parent_agent_id = None
    return KernelEventContext(
        agent_id=agent_id if isinstance(agent_id, str) else None,
        parent_agent_id=parent_agent_id,
        source="task",
    )


def _task_payload(task: TaskStateBase) -> dict[str, object]:
    duration_ms = (
        int((task.end_time - task.start_time) * 1000)
        if task.end_time is not None
        else None
    )
    payload: dict[str, object] = {
        "task_id": task.id,
        "task_type": task.type,
        "status": task.status,
        "description_char_count": len(task.description),
        "tool_use_id": task.tool_use_id,
        "notified": task.notified,
        "duration_ms": duration_ms,
    }
    for attr in (
        "agent_id",
        "parent_agent_id",
        "agent_type",
        "is_idle",
        "shutdown_requested",
    ):
        value = getattr(task, attr, None)
        if isinstance(value, str | bool) or value is None:
            payload[attr] = value
    remote_id = getattr(task, "remote_id", None)
    if isinstance(remote_id, str):
        payload["remote_id_present"] = bool(remote_id)
        payload["remote_id_char_count"] = len(remote_id)
    return payload


def _notification_payload(notification: TaskNotification) -> dict[str, object]:
    return {
        "task_id": notification.task_id,
        "kind": notification.kind,
        "tool_use_id": notification.tool_use_id,
        "priority": notification.priority,
        "mode": notification.mode,
        "agent_id": notification.agent_id,
        "message_char_count": len(notification.message),
        "dedupe_key_present": notification.dedupe_key is not None,
    }


# ---------------------------------------------------------------------------
# Task registry — one Task impl per type.
# ---------------------------------------------------------------------------


_REGISTRY: dict[TaskType, Task] = {}


def register_task_impl(task_impl: Task) -> None:
    """Called at module-init by each task type module."""
    _REGISTRY[task_impl.type] = task_impl


def get_task_by_type(task_type: TaskType) -> Task | None:
    return _REGISTRY.get(task_type)


def get_all_tasks() -> list[Task]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# ID generation — per-type prefix for disambiguation.
# ---------------------------------------------------------------------------

_TASK_ID_PREFIX: dict[TaskType, str] = {
    "local_bash": "b",
    "local_agent": "a",
    "remote_agent": "r",
    "in_process_teammate": "t",
}
"""Single-char prefix per type. Uses a stable per-type convention. Reserved:
s (main-session), r (remote), t (teammate), w (workflow), m (monitor), d (dream)."""


def generate_task_id(task_type: TaskType) -> str:
    """Generate a per-type prefixed task ID."""
    import secrets

    prefix = _TASK_ID_PREFIX.get(task_type)
    if prefix is None:
        msg = f"No ID prefix configured for task type: {task_type}"
        raise ValueError(msg)
    return prefix + secrets.token_hex(8)


# ---------------------------------------------------------------------------
# Internal stop-dispatch helper — type-agnostic lookup + validate-running +
# kill via the registry, with typed errors. NOT a public API: callers must
# go through `core/tasks/stop_task.py`, which adds per-type post-kill
# policy (e.g., suppress LocalBashTask "exit 137" notifications). Direct
# use of this helper bypasses that policy and leaks notifications.
# ---------------------------------------------------------------------------


class StopTaskError(Exception):
    """Base class for stop-task failures."""

    code: ClassVar[Literal["not_found", "not_running", "unsupported_type"]]


class TaskNotFoundError(StopTaskError):
    code: ClassVar[Literal["not_found", "not_running", "unsupported_type"]] = "not_found"


class TaskNotRunningError(StopTaskError):
    code: ClassVar[Literal["not_found", "not_running", "unsupported_type"]] = "not_running"


class UnsupportedTaskTypeError(StopTaskError):
    code: ClassVar[Literal["not_found", "not_running", "unsupported_type"]] = "unsupported_type"


@dataclass
class StopTaskResult:
    task_id: str
    task_type: TaskType
    description: str


async def dispatch_stop_task(task_id: str, store: AppStateStore) -> StopTaskResult:
    """Internal: look up, validate running, dispatch to `task_impl.kill`.

    Raises TaskNotFoundError / TaskNotRunningError / UnsupportedTaskTypeError.

    NOT a public API. Callers must use `core.tasks.stop_task.stop_task`,
    which adds per-type post-kill policy (bash notification suppression).
    Direct use bypasses that policy.
    """
    task = store.tasks.get(task_id)
    if task is None:
        msg = f"No task found with ID: {task_id}"
        raise TaskNotFoundError(msg)

    if task.status != "running":
        msg = f"Task {task_id} is not running (status: {task.status})"
        raise TaskNotRunningError(msg)

    task_impl = get_task_by_type(task.type)
    if task_impl is None:
        msg = f"Unsupported task type: {task.type}"
        raise UnsupportedTaskTypeError(msg)

    await task_impl.kill(task_id, store)

    return StopTaskResult(
        task_id=task_id,
        task_type=task.type,
        description=task.description,
    )


# ---------------------------------------------------------------------------
# Helpers for the atomic check-and-set `notified` pattern.
# ---------------------------------------------------------------------------


def mark_notified_if_unset(
    store: AppStateStore, task_id: str
) -> bool:
    """Atomic check-and-set. Returns True iff this call set the flag.

    Callers use this to gate notification enqueues:
        if mark_notified_if_unset(store, task_id):
            enqueue_notification(...)

    Without the atomic version, two concurrent paths (bulk kill + async
    completion) each read unset, each set, each enqueue a duplicate.
    """
    acquired = False

    def update(task: TaskStateBase) -> TaskStateBase:
        nonlocal acquired
        if task.notified:
            return task
        acquired = True
        task.notified = True
        return task

    store.update_task(task_id, update)
    return acquired


__all__ = [
    "TERMINAL_STATUSES",
    "AppStateStore",
    "NotificationKind",
    "NotificationMode",
    "NotificationPriority",
    "StopTaskError",
    "StopTaskResult",
    "Task",
    "TaskNotFoundError",
    "TaskNotRunningError",
    "TaskNotification",
    "TaskStateBase",
    "TaskStatus",
    "TaskType",
    "UnsupportedTaskTypeError",
    "generate_task_id",
    "get_all_tasks",
    "get_task_by_type",
    "is_terminal_task_status",
    "mark_notified_if_unset",
    "register_task_impl",
]
