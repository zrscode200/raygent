"""Headless coordinator runtime state.

The coordinator runtime is a model-facing coordination ledger, not a task
launcher or UI mode. Tools keep owning side effects; this service records work,
blackboard entries, and task-notification facts so the query loop can later
render a bounded coordinator digest.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, cast

from raygent_harness.core.messages import (
    MessageParam,
    RaygentCoordinatorRuntimeMetadata,
    user_message,
)
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    NoopKernelEventBus,
)
from raygent_harness.core.task import TaskNotification

CoordinatorWorkStatus = Literal[
    "planned",
    "launched",
    "running",
    "idle",
    "completed",
    "failed",
    "killed",
    "blocked",
]

CoordinatorWorkKind = Literal[
    "research",
    "implementation",
    "verification",
    "synthesis",
    "other",
]

CoordinatorBlackboardKind = Literal[
    "fact",
    "decision",
    "risk",
    "worker_result",
    "open_question",
    "synthesis_note",
    "system",
]

CoordinatorBlackboardSource = Literal["user", "coordinator", "agent", "task", "system"]
COORDINATOR_RUNTIME_SNAPSHOT_SCHEMA_VERSION = 1

_WORK_STATUS_VALUES: tuple[CoordinatorWorkStatus, ...] = (
    "planned",
    "launched",
    "running",
    "idle",
    "completed",
    "failed",
    "killed",
    "blocked",
)
_WORK_KIND_VALUES: tuple[CoordinatorWorkKind, ...] = (
    "research",
    "implementation",
    "verification",
    "synthesis",
    "other",
)
_BLACKBOARD_KIND_VALUES: tuple[CoordinatorBlackboardKind, ...] = (
    "fact",
    "decision",
    "risk",
    "worker_result",
    "open_question",
    "synthesis_note",
    "system",
)
_BLACKBOARD_SOURCE_VALUES: tuple[CoordinatorBlackboardSource, ...] = (
    "user",
    "coordinator",
    "agent",
    "task",
    "system",
)


@dataclass(frozen=True, slots=True)
class CoordinatorRuntimeConfig:
    """Caps for stored coordinator state and rendered model context."""

    max_stored_work_items: int = 128
    max_stored_blackboard_entries: int = 256
    max_stored_entry_chars: int = 4_000
    max_rendered_work_items: int = 12
    max_rendered_blackboard_entries: int = 20
    max_total_rendered_chars: int = 8_000
    max_entry_chars: int = 1_500
    include_task_notification_excerpt: bool = False
    task_notification_excerpt_chars: int = 240

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_stored_work_items", self.max_stored_work_items),
            ("max_stored_blackboard_entries", self.max_stored_blackboard_entries),
            ("max_stored_entry_chars", self.max_stored_entry_chars),
            ("max_rendered_work_items", self.max_rendered_work_items),
            ("max_rendered_blackboard_entries", self.max_rendered_blackboard_entries),
            ("max_total_rendered_chars", self.max_total_rendered_chars),
            ("max_entry_chars", self.max_entry_chars),
            ("task_notification_excerpt_chars", self.task_notification_excerpt_chars),
        ):
            if value < 1:
                raise ValueError(f"{field_name} must be >= 1")
        min_total_chars = len("<coordinator_runtime>\n</coordinator_runtime>")
        if self.max_total_rendered_chars < min_total_chars:
            raise ValueError(
                "max_total_rendered_chars must be >= "
                f"{min_total_chars} to preserve coordinator wrapper tags"
            )


@dataclass(frozen=True, slots=True)
class CoordinatorWorkItem:
    id: str
    kind: CoordinatorWorkKind
    title: str
    status: CoordinatorWorkStatus
    agent_name: str | None = None
    agent_type: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    depends_on: tuple[str, ...] = ()
    priority: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    result_summary: str | None = None
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("CoordinatorWorkItem.id must be non-empty")
        if not self.title:
            raise ValueError("CoordinatorWorkItem.title must be non-empty")


@dataclass(frozen=True, slots=True)
class CoordinatorBlackboardEntry:
    id: str
    kind: CoordinatorBlackboardKind
    content: str
    source: CoordinatorBlackboardSource
    work_item_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    priority: int = 0
    created_at: float = 0.0
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("CoordinatorBlackboardEntry.id must be non-empty")
        if not self.content:
            raise ValueError("CoordinatorBlackboardEntry.content must be non-empty")


@dataclass(frozen=True, slots=True)
class CoordinatorNotificationIngestionResult:
    processed_count: int
    skipped_duplicate_count: int
    work_item_ids: tuple[str, ...]
    blackboard_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoordinatorAgentLaunchResult:
    work_item: CoordinatorWorkItem


@dataclass(frozen=True, slots=True)
class CoordinatorSendMessageResult:
    work_items: tuple[CoordinatorWorkItem, ...]
    blackboard_entry: CoordinatorBlackboardEntry


@dataclass(frozen=True, slots=True)
class CoordinatorTaskStopResult:
    work_item: CoordinatorWorkItem
    blackboard_entry: CoordinatorBlackboardEntry


@dataclass(frozen=True, slots=True)
class CoordinatorRuntimeSnapshot:
    work_items: tuple[CoordinatorWorkItem, ...] = ()
    blackboard_entries: tuple[CoordinatorBlackboardEntry, ...] = ()
    processed_notification_count: int = 0
    processed_notification_keys: tuple[str, ...] = ()
    schema_version: int = COORDINATOR_RUNTIME_SNAPSHOT_SCHEMA_VERSION
    session_id: str | None = None
    runtime_session_id: str | None = None
    work_id_counter: int = 0
    blackboard_id_counter: int = 0
    config: CoordinatorRuntimeConfig = field(default_factory=CoordinatorRuntimeConfig)


class CoordinatorRuntimeSnapshotDecodeError(ValueError):
    """Raised when a persisted coordinator snapshot cannot be decoded safely."""


class CoordinatorRuntimeProtocol(Protocol):
    """Narrow protocol that later core/query wiring can depend on."""

    def has_processed_task_notification(self, notification: TaskNotification) -> bool:
        """Return True when this task-notification fact was already ingested."""
        ...

    def record_task_notifications(
        self,
        notifications: Sequence[TaskNotification],
    ) -> CoordinatorNotificationIngestionResult:
        ...

    def record_agent_launch(
        self,
        *,
        agent_id: str,
        agent_type: str,
        description: str,
        prompt_chars: int,
        task_id: str | None = None,
        agent_name: str | None = None,
        team_name: str | None = None,
        mode: str = "background",
        status: str = "running",
        result_summary: str | None = None,
        error_summary: str | None = None,
    ) -> CoordinatorAgentLaunchResult:
        ...

    def record_send_message(
        self,
        *,
        sender: str,
        target: str,
        summary: str | None,
        message_chars: int,
        recipient_task_ids: Sequence[str] = (),
        recipient_agent_ids: Sequence[str] = (),
        team_name: str | None = None,
    ) -> CoordinatorSendMessageResult:
        ...

    def record_task_stop(
        self,
        *,
        task_id: str,
        task_type: str,
        description: str | None = None,
    ) -> CoordinatorTaskStopResult:
        ...

    def render_context(self) -> MessageParam | None:
        ...


class CoordinatorRuntime:
    """In-memory coordinator work ledger and blackboard."""

    def __init__(
        self,
        *,
        config: CoordinatorRuntimeConfig | None = None,
        observability: KernelEventBus | None = None,
        event_context: KernelEventContext | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or CoordinatorRuntimeConfig()
        self._observability = observability or NoopKernelEventBus()
        self._event_context = event_context or KernelEventContext(source="coordinator")
        self._clock = clock
        self._work_items: dict[str, CoordinatorWorkItem] = {}
        self._blackboard_entries: dict[str, CoordinatorBlackboardEntry] = {}
        self._task_work_index: dict[str, str] = {}
        self._agent_work_index: dict[str, str] = {}
        self._seen_notifications: set[str] = set()
        self._work_id_counter = 0
        self._blackboard_id_counter = 0

    @property
    def work_items(self) -> tuple[CoordinatorWorkItem, ...]:
        return tuple(self._ordered_work_items())

    @property
    def blackboard_entries(self) -> tuple[CoordinatorBlackboardEntry, ...]:
        return tuple(self._ordered_blackboard_entries())

    def snapshot(self) -> CoordinatorRuntimeSnapshot:
        return CoordinatorRuntimeSnapshot(
            work_items=self.work_items,
            blackboard_entries=self.blackboard_entries,
            processed_notification_count=len(self._seen_notifications),
            processed_notification_keys=tuple(sorted(self._seen_notifications)),
            session_id=self._event_context.session_id,
            runtime_session_id=self._event_context.runtime_session_id,
            work_id_counter=self._work_id_counter,
            blackboard_id_counter=self._blackboard_id_counter,
            config=self.config,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CoordinatorRuntimeSnapshot,
        *,
        config: CoordinatorRuntimeConfig | None = None,
        observability: KernelEventBus | None = None,
        event_context: KernelEventContext | None = None,
        clock: Callable[[], float] = time.time,
    ) -> CoordinatorRuntime:
        """Reconstruct runtime state from a durable coordinator snapshot."""

        if snapshot.schema_version != COORDINATOR_RUNTIME_SNAPSHOT_SCHEMA_VERSION:
            raise CoordinatorRuntimeSnapshotDecodeError(
                "Unsupported coordinator runtime snapshot schema version: "
                f"{snapshot.schema_version}"
            )

        context = event_context or KernelEventContext(source="coordinator")
        context = replace(
            context,
            session_id=snapshot.session_id or context.session_id,
            runtime_session_id=snapshot.runtime_session_id or context.runtime_session_id,
            source="coordinator",
        )
        runtime = cls(
            config=config or snapshot.config,
            observability=observability,
            event_context=context,
            clock=clock,
        )
        runtime._restore_from_snapshot(snapshot)
        runtime._observability.emit(
            "coordinator.snapshot.restored",
            context=runtime._context(),
            data={
                "schema_version": snapshot.schema_version,
                "work_item_count": len(runtime._work_items),
                "blackboard_entry_count": len(runtime._blackboard_entries),
                "processed_notification_key_count": len(runtime._seen_notifications),
                "processed_notification_count": snapshot.processed_notification_count,
                "work_id_counter": runtime._work_id_counter,
                "blackboard_id_counter": runtime._blackboard_id_counter,
                "session_id_present": snapshot.session_id is not None,
                "runtime_session_id_present": snapshot.runtime_session_id is not None,
            },
        )
        return runtime

    def add_work_item(
        self,
        *,
        kind: CoordinatorWorkKind,
        title: str,
        status: CoordinatorWorkStatus = "planned",
        id: str | None = None,
        agent_name: str | None = None,
        agent_type: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        depends_on: Sequence[str] = (),
        priority: int = 0,
        result_summary: str | None = None,
        error_summary: str | None = None,
    ) -> CoordinatorWorkItem:
        now = self._clock()
        item = CoordinatorWorkItem(
            id=id or self._next_work_id(),
            kind=kind,
            title=_truncate_text(title.strip(), self.config.max_stored_entry_chars).text,
            status=status,
            agent_name=agent_name,
            agent_type=agent_type,
            task_id=task_id,
            agent_id=agent_id,
            depends_on=tuple(depends_on),
            priority=priority,
            created_at=now,
            updated_at=now,
            result_summary=(
                _truncate_text(result_summary.strip(), self.config.max_stored_entry_chars).text
                if result_summary is not None
                else None
            ),
            error_summary=(
                _truncate_text(error_summary.strip(), self.config.max_stored_entry_chars).text
                if error_summary is not None
                else None
            ),
        )
        if item.id in self._work_items:
            raise ValueError(f"Duplicate coordinator work item id: {item.id}")

        self._work_items[item.id] = item
        self._index_work_item(item)
        self._enforce_work_item_cap()
        self._emit_work_item_event("coordinator.work_item.added", item)
        return item

    def update_work_item(
        self,
        work_item_id: str,
        *,
        status: CoordinatorWorkStatus | None = None,
        result_summary: str | None = None,
        error_summary: str | None = None,
        priority: int | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> CoordinatorWorkItem | None:
        item = self._work_items.get(work_item_id)
        if item is None:
            return None

        updated = replace(
            item,
            status=status or item.status,
            result_summary=(
                _truncate_text(result_summary.strip(), self.config.max_stored_entry_chars).text
                if result_summary is not None
                else item.result_summary
            ),
            error_summary=(
                _truncate_text(error_summary.strip(), self.config.max_stored_entry_chars).text
                if error_summary is not None
                else item.error_summary
            ),
            priority=item.priority if priority is None else priority,
            task_id=task_id if task_id is not None else item.task_id,
            agent_id=agent_id if agent_id is not None else item.agent_id,
            updated_at=self._clock(),
        )
        self._remove_work_indexes(item)
        self._work_items[work_item_id] = updated
        self._index_work_item(updated)
        self._emit_work_item_event("coordinator.work_item.updated", updated)
        return updated

    def add_blackboard_entry(
        self,
        *,
        kind: CoordinatorBlackboardKind,
        content: str,
        source: CoordinatorBlackboardSource,
        id: str | None = None,
        work_item_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        priority: int = 0,
    ) -> CoordinatorBlackboardEntry:
        bounded = _truncate_text(content.strip(), self.config.max_stored_entry_chars)
        entry = CoordinatorBlackboardEntry(
            id=id or self._next_blackboard_id(),
            kind=kind,
            content=bounded.text,
            source=source,
            work_item_id=work_item_id,
            task_id=task_id,
            agent_id=agent_id,
            priority=priority,
            created_at=self._clock(),
            truncated=bounded.truncated,
        )
        if entry.id in self._blackboard_entries:
            raise ValueError(f"Duplicate coordinator blackboard entry id: {entry.id}")

        self._blackboard_entries[entry.id] = entry
        self._enforce_blackboard_cap()
        self._emit_blackboard_event(entry)
        return entry

    def record_task_notifications(
        self,
        notifications: Sequence[TaskNotification],
    ) -> CoordinatorNotificationIngestionResult:
        processed = 0
        skipped = 0
        work_item_ids: list[str] = []
        blackboard_entry_ids: list[str] = []

        for notification in notifications:
            key = _notification_key(notification)
            if key in self._seen_notifications:
                skipped += 1
                continue
            self._seen_notifications.add(key)
            processed += 1

            status = _notification_status(notification)
            work_item = self._upsert_task_work_item(notification, status)
            work_item_ids.append(work_item.id)
            entry = self.add_blackboard_entry(
                kind="worker_result" if status == "completed" else "risk",
                content=self._notification_blackboard_content(notification, status),
                source="task",
                work_item_id=work_item.id,
                task_id=notification.task_id,
                agent_id=notification.agent_id,
                priority=work_item.priority,
            )
            blackboard_entry_ids.append(entry.id)

        self._observability.emit(
            "coordinator.task_notifications.recorded",
            context=self._context(),
            data={
                "notification_count": len(notifications),
                "processed_count": processed,
                "skipped_duplicate_count": skipped,
                "work_item_ids": tuple(work_item_ids),
                "blackboard_entry_ids": tuple(blackboard_entry_ids),
            },
        )
        return CoordinatorNotificationIngestionResult(
            processed_count=processed,
            skipped_duplicate_count=skipped,
            work_item_ids=tuple(work_item_ids),
            blackboard_entry_ids=tuple(blackboard_entry_ids),
        )

    def has_processed_task_notification(self, notification: TaskNotification) -> bool:
        """Check de-duplication state without mutating the coordinator ledger."""

        return _notification_key(notification) in self._seen_notifications

    def record_agent_launch(
        self,
        *,
        agent_id: str,
        agent_type: str,
        description: str,
        prompt_chars: int,
        task_id: str | None = None,
        agent_name: str | None = None,
        team_name: str | None = None,
        mode: str = "background",
        status: str = "running",
        result_summary: str | None = None,
        error_summary: str | None = None,
    ) -> CoordinatorAgentLaunchResult:
        """Record AgentTool work only after the launch/result succeeded."""

        work_status = _coerce_work_status(status, fallback="running")
        existing_id = self._find_work_item_id(task_id=task_id, agent_id=agent_id)
        if existing_id is not None:
            updated = self.update_work_item(
                existing_id,
                status=work_status,
                result_summary=result_summary,
                error_summary=error_summary,
                task_id=task_id,
                agent_id=agent_id,
            )
            if updated is not None:
                return CoordinatorAgentLaunchResult(work_item=updated)

        item = self.add_work_item(
            id=self._unique_work_id(f"cw_agent_{task_id or agent_id}"),
            kind="other",
            title=_agent_launch_title(
                description=description,
                agent_type=agent_type,
                mode=mode,
                agent_name=agent_name,
                team_name=team_name,
                prompt_chars=prompt_chars,
            ),
            status=work_status,
            agent_name=agent_name,
            agent_type=agent_type,
            task_id=task_id,
            agent_id=agent_id,
            result_summary=result_summary,
            error_summary=error_summary,
        )
        return CoordinatorAgentLaunchResult(work_item=item)

    def record_send_message(
        self,
        *,
        sender: str,
        target: str,
        summary: str | None,
        message_chars: int,
        recipient_task_ids: Sequence[str] = (),
        recipient_agent_ids: Sequence[str] = (),
        team_name: str | None = None,
    ) -> CoordinatorSendMessageResult:
        """Record successful SendMessage routing without storing raw message text."""

        work_items: list[CoordinatorWorkItem] = []
        recipient_count = max(len(recipient_task_ids), len(recipient_agent_ids))
        for index, task_id in enumerate(recipient_task_ids):
            agent_id = recipient_agent_ids[index] if index < len(recipient_agent_ids) else None
            work_items.append(
                self._upsert_routed_message_work_item(
                    task_id=task_id,
                    agent_id=agent_id,
                    target=target,
                    summary=summary,
                )
            )
        for index, agent_id in enumerate(recipient_agent_ids[len(recipient_task_ids) :]):
            work_items.append(
                self._upsert_routed_message_work_item(
                    task_id=None,
                    agent_id=agent_id,
                    target=target,
                    summary=summary,
                    offset=index + len(recipient_task_ids),
                )
            )

        entry = self.add_blackboard_entry(
            kind="system",
            content=_send_message_blackboard_content(
                sender=sender,
                target=target,
                summary=summary,
                message_chars=message_chars,
                recipient_count=recipient_count,
                team_name=team_name,
            ),
            source="coordinator",
            work_item_id=work_items[0].id if work_items else None,
            task_id=work_items[0].task_id if work_items else None,
            agent_id=work_items[0].agent_id if work_items else None,
            priority=1,
        )
        return CoordinatorSendMessageResult(
            work_items=tuple(work_items),
            blackboard_entry=entry,
        )

    def record_task_stop(
        self,
        *,
        task_id: str,
        task_type: str,
        description: str | None = None,
    ) -> CoordinatorTaskStopResult:
        """Record successful TaskStop after the underlying task kill succeeds."""

        existing_id = self._find_work_item_id(task_id=task_id, agent_id=None)
        error_summary = f"TaskStop stopped task_type={task_type}."
        if existing_id is not None:
            item = self.update_work_item(
                existing_id,
                status="killed",
                error_summary=error_summary,
            )
            assert item is not None
        else:
            item = self.add_work_item(
                id=self._unique_work_id(f"cw_task_{task_id}"),
                kind="other",
                title=(description or f"Task {task_id}").strip(),
                status="killed",
                task_id=task_id,
                agent_type=task_type,
                error_summary=error_summary,
            )

        entry = self.add_blackboard_entry(
            kind="risk",
            content=_task_stop_blackboard_content(
                task_id=task_id,
                task_type=task_type,
                description=description,
            ),
            source="coordinator",
            work_item_id=item.id,
            task_id=task_id,
            agent_id=item.agent_id,
            priority=1,
        )
        return CoordinatorTaskStopResult(work_item=item, blackboard_entry=entry)

    def render_context(self) -> MessageParam | None:
        ordered_work = self._ordered_work_items()
        ordered_blackboard = self._ordered_blackboard_entries()
        if not ordered_work and not ordered_blackboard:
            return None

        rendered = _render_runtime_context(
            ordered_work=ordered_work,
            ordered_blackboard=ordered_blackboard,
            config=self.config,
        )

        metadata: RaygentCoordinatorRuntimeMetadata = {
            "type": "coordinator_runtime",
            "work_item_count": len(ordered_work),
            "blackboard_entry_count": len(ordered_blackboard),
            "rendered_work_item_count": len(rendered.work_items),
            "rendered_blackboard_entry_count": len(rendered.blackboard_entries),
            "dropped_work_item_count": len(ordered_work) - len(rendered.work_items),
            "dropped_blackboard_entry_count": (
                len(ordered_blackboard) - len(rendered.blackboard_entries)
            ),
            "truncated": rendered.truncated,
            "rendered_char_count": len(rendered.content),
            "max_total_chars": self.config.max_total_rendered_chars,
            "max_work_items": self.config.max_rendered_work_items,
            "max_blackboard_entries": self.config.max_rendered_blackboard_entries,
            "max_entry_chars": self.config.max_entry_chars,
            "work_item_ids": [item.id for item in rendered.work_items],
            "blackboard_entry_ids": [entry.id for entry in rendered.blackboard_entries],
        }
        message = user_message(rendered.content)
        message["raygentMessageKind"] = "coordinator_runtime"
        message["raygentCoordinatorRuntime"] = metadata
        self._observability.emit(
            "coordinator.context.rendered",
            context=self._context(),
            data={
                "work_item_count": len(ordered_work),
                "blackboard_entry_count": len(ordered_blackboard),
                "rendered_work_item_count": len(rendered.work_items),
                "rendered_blackboard_entry_count": len(rendered.blackboard_entries),
                "dropped_work_item_count": len(ordered_work) - len(rendered.work_items),
                "dropped_blackboard_entry_count": (
                    len(ordered_blackboard) - len(rendered.blackboard_entries)
                ),
                "truncated": rendered.truncated,
                "rendered_char_count": len(rendered.content),
            },
        )
        return message

    def _upsert_task_work_item(
        self,
        notification: TaskNotification,
        status: CoordinatorWorkStatus,
    ) -> CoordinatorWorkItem:
        existing_id = self._task_work_index.get(notification.task_id)
        result_summary = (
            f"Task notification recorded; message_chars={len(notification.message)}."
            if status == "completed"
            else None
        )
        error_summary = (
            f"Task notification recorded; message_chars={len(notification.message)}."
            if status != "completed"
            else None
        )
        if existing_id is not None:
            updated = self.update_work_item(
                existing_id,
                status=status,
                result_summary=result_summary,
                error_summary=error_summary,
            )
            if updated is not None:
                return updated

        return self.add_work_item(
            id=self._unique_work_id(f"cw_task_{notification.task_id}"),
            kind="other",
            title=f"Task {notification.task_id}",
            status=status,
            task_id=notification.task_id,
            agent_id=notification.agent_id,
            result_summary=result_summary,
            error_summary=error_summary,
        )

    def _upsert_routed_message_work_item(
        self,
        *,
        task_id: str | None,
        agent_id: str | None,
        target: str,
        summary: str | None,
        offset: int = 0,
    ) -> CoordinatorWorkItem:
        existing_id = self._find_work_item_id(task_id=task_id, agent_id=agent_id)
        result_summary = _routed_message_summary(
            target=target,
            summary=summary,
        )
        if existing_id is not None:
            updated = self.update_work_item(
                existing_id,
                status="running",
                result_summary=result_summary,
                task_id=task_id,
                agent_id=agent_id,
            )
            if updated is not None:
                return updated

        stable_id = task_id or agent_id or f"{target}_{offset}"
        return self.add_work_item(
            id=self._unique_work_id(f"cw_route_{stable_id}"),
            kind="other",
            title=f"Routed message to {target}",
            status="running",
            task_id=task_id,
            agent_id=agent_id,
            result_summary=result_summary,
        )

    def _notification_blackboard_content(
        self,
        notification: TaskNotification,
        status: CoordinatorWorkStatus,
    ) -> str:
        parts = [
            f"Task {notification.task_id} signaled status={status}",
            f"kind={notification.kind}",
            f"message_chars={len(notification.message)}",
        ]
        if notification.tool_use_id is not None:
            parts.append(f"tool_use_id={notification.tool_use_id}")
        if self.config.include_task_notification_excerpt:
            excerpt = _truncate_text(
                notification.message.strip(),
                self.config.task_notification_excerpt_chars,
            ).text
            parts.append(f"excerpt={excerpt}")
        return "; ".join(parts) + "."

    def _ordered_work_items(self) -> tuple[CoordinatorWorkItem, ...]:
        return tuple(
            sorted(
                self._work_items.values(),
                key=lambda item: (
                    -item.priority,
                    _status_order(item.status),
                    -item.updated_at,
                    item.created_at,
                    item.id,
                ),
            )
        )

    def _ordered_blackboard_entries(self) -> tuple[CoordinatorBlackboardEntry, ...]:
        return tuple(
            sorted(
                self._blackboard_entries.values(),
                key=lambda entry: (-entry.priority, entry.created_at, entry.id),
            )
        )

    def _find_work_item_id(
        self,
        *,
        task_id: str | None,
        agent_id: str | None,
    ) -> str | None:
        if task_id is not None:
            work_item_id = self._task_work_index.get(task_id)
            if work_item_id is not None:
                return work_item_id
        if agent_id is not None:
            return self._agent_work_index.get(agent_id)
        return None

    def _next_work_id(self) -> str:
        self._work_id_counter += 1
        return f"cw_{self._work_id_counter:06d}"

    def _next_blackboard_id(self) -> str:
        self._blackboard_id_counter += 1
        return f"cb_{self._blackboard_id_counter:06d}"

    def _unique_work_id(self, preferred: str) -> str:
        candidate = _safe_id(preferred)
        if candidate not in self._work_items:
            return candidate
        while True:
            next_id = self._next_work_id()
            if next_id not in self._work_items:
                return next_id

    def _index_work_item(self, item: CoordinatorWorkItem) -> None:
        if item.task_id is not None:
            self._task_work_index[item.task_id] = item.id
        if item.agent_id is not None:
            self._agent_work_index[item.agent_id] = item.id

    def _remove_work_indexes(self, item: CoordinatorWorkItem) -> None:
        if item.task_id is not None and self._task_work_index.get(item.task_id) == item.id:
            self._task_work_index.pop(item.task_id, None)
        if item.agent_id is not None and self._agent_work_index.get(item.agent_id) == item.id:
            self._agent_work_index.pop(item.agent_id, None)

    def _enforce_work_item_cap(self) -> None:
        while len(self._work_items) > self.config.max_stored_work_items:
            victim = min(self._work_items.values(), key=_work_retention_key)
            self._remove_work_indexes(victim)
            self._work_items.pop(victim.id, None)

    def _enforce_blackboard_cap(self) -> None:
        while len(self._blackboard_entries) > self.config.max_stored_blackboard_entries:
            victim = min(self._blackboard_entries.values(), key=_blackboard_retention_key)
            self._blackboard_entries.pop(victim.id, None)

    def _context(self, *, agent_id: str | None = None) -> KernelEventContext:
        return replace(
            self._event_context,
            agent_id=agent_id if agent_id is not None else self._event_context.agent_id,
            source="coordinator",
        )

    def _emit_work_item_event(self, event_type: str, item: CoordinatorWorkItem) -> None:
        self._observability.emit(
            event_type,
            context=self._context(agent_id=item.agent_id),
            data={
                "work_item_id": item.id,
                "kind": item.kind,
                "status": item.status,
                "priority": item.priority,
                "task_id": item.task_id,
                "agent_id": item.agent_id,
                "agent_name_present": item.agent_name is not None,
                "agent_type": item.agent_type,
                "title_char_count": len(item.title),
                "result_summary_char_count": (
                    len(item.result_summary) if item.result_summary is not None else 0
                ),
                "error_summary_char_count": (
                    len(item.error_summary) if item.error_summary is not None else 0
                ),
                "depends_on_count": len(item.depends_on),
            },
        )

    def _emit_blackboard_event(self, entry: CoordinatorBlackboardEntry) -> None:
        self._observability.emit(
            "coordinator.blackboard.entry.added",
            context=self._context(agent_id=entry.agent_id),
            data={
                "blackboard_entry_id": entry.id,
                "kind": entry.kind,
                "source": entry.source,
                "priority": entry.priority,
                "work_item_id": entry.work_item_id,
                "task_id": entry.task_id,
                "agent_id": entry.agent_id,
                "content_char_count": len(entry.content),
                "truncated": entry.truncated,
            },
        )

    def _restore_from_snapshot(self, snapshot: CoordinatorRuntimeSnapshot) -> None:
        self._work_items = {}
        self._blackboard_entries = {}
        self._task_work_index = {}
        self._agent_work_index = {}

        for item in snapshot.work_items:
            if item.id in self._work_items:
                raise CoordinatorRuntimeSnapshotDecodeError(
                    f"Duplicate coordinator work item id in snapshot: {item.id}"
                )
            self._work_items[item.id] = item
            self._index_work_item(item)

        for entry in snapshot.blackboard_entries:
            if entry.id in self._blackboard_entries:
                raise CoordinatorRuntimeSnapshotDecodeError(
                    f"Duplicate coordinator blackboard entry id in snapshot: {entry.id}"
                )
            self._blackboard_entries[entry.id] = entry

        self._seen_notifications = set(snapshot.processed_notification_keys)
        self._work_id_counter = max(
            snapshot.work_id_counter,
            _max_prefixed_numeric_id(self._work_items, "cw_"),
        )
        self._blackboard_id_counter = max(
            snapshot.blackboard_id_counter,
            _max_prefixed_numeric_id(self._blackboard_entries, "cb_"),
        )
        self._enforce_work_item_cap()
        self._enforce_blackboard_cap()


@dataclass(frozen=True, slots=True)
class _BoundedText:
    text: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _RenderedCoordinatorContext:
    content: str
    work_items: tuple[CoordinatorWorkItem, ...]
    blackboard_entries: tuple[CoordinatorBlackboardEntry, ...]
    truncated: bool


def _truncate_text(text: str, max_chars: int) -> _BoundedText:
    if len(text) <= max_chars:
        return _BoundedText(text=text, truncated=False)
    marker = f"...[truncated {len(text) - max_chars} chars]"
    if len(marker) >= max_chars:
        return _BoundedText(text=marker[:max_chars], truncated=True)
    budget = max(0, max_chars - len(marker))
    return _BoundedText(text=text[:budget].rstrip() + marker, truncated=True)


def _render_runtime_context(
    *,
    ordered_work: Sequence[CoordinatorWorkItem],
    ordered_blackboard: Sequence[CoordinatorBlackboardEntry],
    config: CoordinatorRuntimeConfig,
) -> _RenderedCoordinatorContext:
    lines = ["<coordinator_runtime>"]
    rendered_work: list[CoordinatorWorkItem] = []
    rendered_blackboard: list[CoordinatorBlackboardEntry] = []
    truncated = False

    work_result = _try_render_section(
        existing_lines=lines,
        open_tag="<work_items>",
        close_tag="</work_items>",
        candidates=ordered_work,
        max_candidates=config.max_rendered_work_items,
        max_chars=config.max_total_rendered_chars,
        render_line=lambda item: _render_work_item(item, config.max_entry_chars),
        omitted_label="work item",
    )
    if work_result is not None:
        lines.extend(work_result.lines)
        rendered_work.extend(work_result.rendered_items)
        truncated = truncated or work_result.truncated
    elif ordered_work:
        truncated = True

    blackboard_result = _try_render_section(
        existing_lines=lines,
        open_tag="<blackboard>",
        close_tag="</blackboard>",
        candidates=ordered_blackboard,
        max_candidates=config.max_rendered_blackboard_entries,
        max_chars=config.max_total_rendered_chars,
        render_line=lambda entry: _render_blackboard_entry(entry, config.max_entry_chars),
        omitted_label="blackboard entry",
    )
    if blackboard_result is not None:
        lines.extend(blackboard_result.lines)
        rendered_blackboard.extend(blackboard_result.rendered_items)
        truncated = truncated or blackboard_result.truncated
    elif ordered_blackboard:
        truncated = True

    lines.append("</coordinator_runtime>")
    content = "\n".join(lines)
    if len(content) > config.max_total_rendered_chars:
        raise RuntimeError("coordinator runtime render exceeded configured budget")

    return _RenderedCoordinatorContext(
        content=content,
        work_items=tuple(rendered_work),
        blackboard_entries=tuple(rendered_blackboard),
        truncated=(
            truncated
            or len(rendered_work) < len(ordered_work)
            or len(rendered_blackboard) < len(ordered_blackboard)
        ),
    )


@dataclass(frozen=True, slots=True)
class _RenderedSection[T]:
    lines: tuple[str, ...]
    rendered_items: tuple[T, ...]
    truncated: bool


def _try_render_section[T](
    *,
    existing_lines: Sequence[str],
    open_tag: str,
    close_tag: str,
    candidates: Sequence[T],
    max_candidates: int,
    max_chars: int,
    render_line: Callable[[T], str],
    omitted_label: str,
) -> _RenderedSection[T] | None:
    if not candidates:
        return None

    section_lines = [open_tag]
    rendered_items: list[T] = []
    truncated = len(candidates) > max_candidates
    limited = candidates[:max_candidates]

    if not _would_fit(existing_lines, [open_tag, close_tag], max_chars):
        return None

    for item in limited:
        line = render_line(item)
        if not _would_fit(existing_lines, [*section_lines, line, close_tag], max_chars):
            truncated = True
            break
        section_lines.append(line)
        rendered_items.append(item)

    omitted_count = len(candidates) - len(rendered_items)
    if omitted_count:
        marker = f"... {omitted_count} additional {omitted_label}(s) omitted by render cap."
        if _would_fit(existing_lines, [*section_lines, marker, close_tag], max_chars):
            section_lines.append(marker)
        truncated = True

    section_lines.append(close_tag)
    return _RenderedSection(
        lines=tuple(section_lines),
        rendered_items=tuple(rendered_items),
        truncated=truncated,
    )


def _would_fit(
    existing_lines: Sequence[str],
    additional_lines: Sequence[str],
    max_chars: int,
) -> bool:
    return (
        len(
            "\n".join(
                [
                    *existing_lines,
                    *additional_lines,
                    "</coordinator_runtime>",
                ]
            )
        )
        <= max_chars
    )


def _render_work_item(item: CoordinatorWorkItem, max_chars: int) -> str:
    parts = [
        f"- id={item.id}",
        f"status={item.status}",
        f"kind={item.kind}",
    ]
    if item.task_id is not None:
        parts.append(f"task_id={item.task_id}")
    if item.agent_name is not None:
        parts.append(f"agent_name={item.agent_name}")
    if item.agent_type is not None:
        parts.append(f"agent_type={item.agent_type}")
    title = _truncate_text(item.title, max_chars).text
    line = " ".join(parts) + f" title={title}"
    summaries: list[str] = []
    if item.result_summary:
        summaries.append("result=" + _truncate_text(item.result_summary, max_chars).text)
    if item.error_summary:
        summaries.append("error=" + _truncate_text(item.error_summary, max_chars).text)
    if summaries:
        line += " " + " ".join(summaries)
    return _truncate_text(line, max_chars).text


def _render_blackboard_entry(entry: CoordinatorBlackboardEntry, max_chars: int) -> str:
    parts = [
        f"- id={entry.id}",
        f"kind={entry.kind}",
        f"source={entry.source}",
    ]
    if entry.work_item_id is not None:
        parts.append(f"work_item_id={entry.work_item_id}")
    if entry.task_id is not None:
        parts.append(f"task_id={entry.task_id}")
    content = _truncate_text(entry.content, max_chars).text
    return _truncate_text(" ".join(parts) + f" content={content}", max_chars).text


def _status_order(status: CoordinatorWorkStatus) -> int:
    return {
        "blocked": 0,
        "failed": 1,
        "killed": 2,
        "running": 3,
        "launched": 4,
        "planned": 5,
        "idle": 6,
        "completed": 7,
    }[status]


def _work_retention_key(item: CoordinatorWorkItem) -> tuple[int, int, float, float, str]:
    terminal = item.status in ("completed", "failed", "killed")
    return (
        item.priority,
        0 if terminal else 1,
        item.updated_at,
        item.created_at,
        item.id,
    )


def _blackboard_retention_key(
    entry: CoordinatorBlackboardEntry,
) -> tuple[int, float, str]:
    return (entry.priority, entry.created_at, entry.id)


def _notification_status(notification: TaskNotification) -> CoordinatorWorkStatus:
    match = re.search(r"<status>\s*([^<]+?)\s*</status>", notification.message)
    if match:
        status = match.group(1).strip().lower()
        if status in (
            "planned",
            "launched",
            "running",
            "idle",
            "completed",
            "failed",
            "killed",
            "blocked",
        ):
            return status
    if notification.kind == "completed":
        return "completed"
    if notification.kind == "stalled":
        return "blocked"
    return "failed"


def _coerce_work_status(
    status: str,
    *,
    fallback: CoordinatorWorkStatus,
) -> CoordinatorWorkStatus:
    normalized = status.strip().lower()
    if normalized in (
        "planned",
        "launched",
        "running",
        "idle",
        "completed",
        "failed",
        "killed",
        "blocked",
    ):
        return normalized
    return fallback


def _agent_launch_title(
    *,
    description: str,
    agent_type: str,
    mode: str,
    agent_name: str | None,
    team_name: str | None,
    prompt_chars: int,
) -> str:
    title = description.strip() or f"{agent_type} agent"
    parts = [title, f"agent_type={agent_type}", f"mode={mode}", f"prompt_chars={prompt_chars}"]
    if agent_name is not None:
        parts.append(f"name={agent_name}")
    if team_name is not None:
        parts.append(f"team={team_name}")
    return "; ".join(parts)


def _routed_message_summary(
    *,
    target: str,
    summary: str | None,
) -> str:
    if summary is None or not summary.strip():
        return f"Message routed to {target}."
    return f"Message routed to {target}; summary={summary.strip()}."


def _send_message_blackboard_content(
    *,
    sender: str,
    target: str,
    summary: str | None,
    message_chars: int,
    recipient_count: int,
    team_name: str | None,
) -> str:
    parts = [
        f"SendMessage routed sender={sender}",
        f"target={target}",
        f"recipient_count={recipient_count}",
        f"message_chars={message_chars}",
    ]
    if team_name is not None:
        parts.append(f"team={team_name}")
    if summary is not None and summary.strip():
        parts.append(f"summary={summary.strip()}")
    return "; ".join(parts) + "."


def _task_stop_blackboard_content(
    *,
    task_id: str,
    task_type: str,
    description: str | None,
) -> str:
    parts = [
        f"TaskStop stopped task={task_id}",
        f"task_type={task_type}",
    ]
    if description is not None and description.strip():
        parts.append(f"description_chars={len(description.strip())}")
    return "; ".join(parts) + "."


def _notification_key(notification: TaskNotification) -> str:
    if notification.dedupe_key is not None:
        return notification.dedupe_key
    digest = hashlib.sha256(notification.message.encode("utf-8")).hexdigest()[:16]
    return (
        f"{notification.task_id}:{notification.kind}:{notification.tool_use_id}:"
        f"{notification.agent_id}:{notification.created_at:.9f}:{digest}"
    )


def coordinator_runtime_snapshot_to_dict(
    snapshot: CoordinatorRuntimeSnapshot,
) -> dict[str, object]:
    """Encode a coordinator runtime snapshot as JSON-safe metadata."""

    return {
        "schema_version": snapshot.schema_version,
        "session_id": snapshot.session_id,
        "runtime_session_id": snapshot.runtime_session_id,
        "config": _config_to_dict(snapshot.config),
        "work_items": [_work_item_to_dict(item) for item in snapshot.work_items],
        "blackboard_entries": [
            _blackboard_entry_to_dict(entry) for entry in snapshot.blackboard_entries
        ],
        "processed_notification_count": snapshot.processed_notification_count,
        "processed_notification_keys": list(snapshot.processed_notification_keys),
        "work_id_counter": snapshot.work_id_counter,
        "blackboard_id_counter": snapshot.blackboard_id_counter,
    }


def coordinator_runtime_snapshot_from_dict(
    raw: object,
) -> CoordinatorRuntimeSnapshot:
    """Decode a JSON object into a validated coordinator runtime snapshot."""

    data = _require_mapping(raw, "coordinator runtime snapshot")
    schema_version = _required_int(data, "schema_version")
    if schema_version != COORDINATOR_RUNTIME_SNAPSHOT_SCHEMA_VERSION:
        raise CoordinatorRuntimeSnapshotDecodeError(
            "Unsupported coordinator runtime snapshot schema version: "
            f"{schema_version}"
        )
    config_raw = data.get("config")
    config = (
        _config_from_dict(_require_mapping(config_raw, "config"))
        if config_raw is not None
        else CoordinatorRuntimeConfig()
    )
    processed_count = _optional_int(
        data,
        "processed_notification_count",
        default=0,
    )
    processed_key_values = [
        _required_str(item, "processed_notification_keys[]")
        for item in _optional_list(data, "processed_notification_keys")
    ]
    processed_keys = tuple(processed_key_values)
    if len(set(processed_keys)) != len(processed_keys):
        raise CoordinatorRuntimeSnapshotDecodeError(
            "processed_notification_keys must not contain duplicates"
        )
    if processed_count != len(processed_keys):
        raise CoordinatorRuntimeSnapshotDecodeError(
            "processed_notification_count must match processed_notification_keys length"
        )
    return CoordinatorRuntimeSnapshot(
        work_items=tuple(
            _work_item_from_dict(item) for item in _optional_list(data, "work_items")
        ),
        blackboard_entries=tuple(
            _blackboard_entry_from_dict(item)
            for item in _optional_list(data, "blackboard_entries")
        ),
        processed_notification_count=processed_count,
        processed_notification_keys=processed_keys,
        schema_version=schema_version,
        session_id=_optional_str(data, "session_id"),
        runtime_session_id=_optional_str(data, "runtime_session_id"),
        work_id_counter=_optional_int(data, "work_id_counter", default=0),
        blackboard_id_counter=_optional_int(data, "blackboard_id_counter", default=0),
        config=config,
    )


def _config_to_dict(config: CoordinatorRuntimeConfig) -> dict[str, object]:
    return {
        "max_stored_work_items": config.max_stored_work_items,
        "max_stored_blackboard_entries": config.max_stored_blackboard_entries,
        "max_stored_entry_chars": config.max_stored_entry_chars,
        "max_rendered_work_items": config.max_rendered_work_items,
        "max_rendered_blackboard_entries": config.max_rendered_blackboard_entries,
        "max_total_rendered_chars": config.max_total_rendered_chars,
        "max_entry_chars": config.max_entry_chars,
        "include_task_notification_excerpt": config.include_task_notification_excerpt,
        "task_notification_excerpt_chars": config.task_notification_excerpt_chars,
    }


def _config_from_dict(data: dict[str, object]) -> CoordinatorRuntimeConfig:
    defaults = CoordinatorRuntimeConfig()
    return CoordinatorRuntimeConfig(
        max_stored_work_items=_optional_int(
            data,
            "max_stored_work_items",
            default=defaults.max_stored_work_items,
        ),
        max_stored_blackboard_entries=_optional_int(
            data,
            "max_stored_blackboard_entries",
            default=defaults.max_stored_blackboard_entries,
        ),
        max_stored_entry_chars=_optional_int(
            data,
            "max_stored_entry_chars",
            default=defaults.max_stored_entry_chars,
        ),
        max_rendered_work_items=_optional_int(
            data,
            "max_rendered_work_items",
            default=defaults.max_rendered_work_items,
        ),
        max_rendered_blackboard_entries=_optional_int(
            data,
            "max_rendered_blackboard_entries",
            default=defaults.max_rendered_blackboard_entries,
        ),
        max_total_rendered_chars=_optional_int(
            data,
            "max_total_rendered_chars",
            default=defaults.max_total_rendered_chars,
        ),
        max_entry_chars=_optional_int(
            data,
            "max_entry_chars",
            default=defaults.max_entry_chars,
        ),
        include_task_notification_excerpt=_optional_bool(
            data,
            "include_task_notification_excerpt",
            default=defaults.include_task_notification_excerpt,
        ),
        task_notification_excerpt_chars=_optional_int(
            data,
            "task_notification_excerpt_chars",
            default=defaults.task_notification_excerpt_chars,
        ),
    )


def _work_item_to_dict(item: CoordinatorWorkItem) -> dict[str, object]:
    return {
        "id": item.id,
        "kind": item.kind,
        "title": item.title,
        "status": item.status,
        "agent_name": item.agent_name,
        "agent_type": item.agent_type,
        "task_id": item.task_id,
        "agent_id": item.agent_id,
        "depends_on": list(item.depends_on),
        "priority": item.priority,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "result_summary": item.result_summary,
        "error_summary": item.error_summary,
    }


def _work_item_from_dict(raw: object) -> CoordinatorWorkItem:
    data = _require_mapping(raw, "work item")
    kind = _required_literal(data, "kind", _WORK_KIND_VALUES)
    status = _required_literal(data, "status", _WORK_STATUS_VALUES)
    return CoordinatorWorkItem(
        id=_required_str_field(data, "id"),
        kind=cast(CoordinatorWorkKind, kind),
        title=_required_str_field(data, "title"),
        status=cast(CoordinatorWorkStatus, status),
        agent_name=_optional_str(data, "agent_name"),
        agent_type=_optional_str(data, "agent_type"),
        task_id=_optional_str(data, "task_id"),
        agent_id=_optional_str(data, "agent_id"),
        depends_on=tuple(
            _required_str(item, "depends_on[]")
            for item in _optional_list(data, "depends_on")
        ),
        priority=_optional_int(data, "priority", default=0),
        created_at=_optional_float(data, "created_at", default=0.0),
        updated_at=_optional_float(data, "updated_at", default=0.0),
        result_summary=_optional_str(data, "result_summary"),
        error_summary=_optional_str(data, "error_summary"),
    )


def _blackboard_entry_to_dict(entry: CoordinatorBlackboardEntry) -> dict[str, object]:
    return {
        "id": entry.id,
        "kind": entry.kind,
        "content": entry.content,
        "source": entry.source,
        "work_item_id": entry.work_item_id,
        "task_id": entry.task_id,
        "agent_id": entry.agent_id,
        "priority": entry.priority,
        "created_at": entry.created_at,
        "truncated": entry.truncated,
    }


def _blackboard_entry_from_dict(raw: object) -> CoordinatorBlackboardEntry:
    data = _require_mapping(raw, "blackboard entry")
    kind = _required_literal(data, "kind", _BLACKBOARD_KIND_VALUES)
    source = _required_literal(data, "source", _BLACKBOARD_SOURCE_VALUES)
    return CoordinatorBlackboardEntry(
        id=_required_str_field(data, "id"),
        kind=cast(CoordinatorBlackboardKind, kind),
        content=_required_str_field(data, "content"),
        source=cast(CoordinatorBlackboardSource, source),
        work_item_id=_optional_str(data, "work_item_id"),
        task_id=_optional_str(data, "task_id"),
        agent_id=_optional_str(data, "agent_id"),
        priority=_optional_int(data, "priority", default=0),
        created_at=_optional_float(data, "created_at", default=0.0),
        truncated=_optional_bool(data, "truncated", default=False),
    )


def _require_mapping(raw: object, name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be an object")
    return cast(dict[str, object], raw)


def _required_str(raw: object, name: str) -> str:
    if not isinstance(raw, str) or raw == "":
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be a non-empty string")
    return raw


def _required_str_field(data: dict[str, object], name: str) -> str:
    return _required_str(data.get(name), name)


def _optional_str(data: dict[str, object], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    return _required_str(value, name)


def _required_int(data: dict[str, object], name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be an integer")
    return value


def _optional_int(data: dict[str, object], name: str, *, default: int) -> int:
    value = data.get(name)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be an integer")
    return value


def _optional_float(data: dict[str, object], name: str, *, default: float) -> float:
    value = data.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be numeric")
    return float(value)


def _optional_bool(data: dict[str, object], name: str, *, default: bool) -> bool:
    value = data.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be a boolean")
    return value


def _optional_list(data: dict[str, object], name: str) -> list[object]:
    value = data.get(name)
    if value is None:
        return []
    if not isinstance(value, list):
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} must be a list")
    return cast(list[object], value)


def _required_literal[T](
    data: dict[str, object],
    name: str,
    allowed: tuple[T, ...],
) -> T:
    value = data.get(name)
    if value not in allowed:
        raise CoordinatorRuntimeSnapshotDecodeError(f"{name} has unsupported value")
    return cast(T, value)


def _max_prefixed_numeric_id(items: Mapping[str, object], prefix: str) -> int:
    maximum = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item_id in items:
        match = pattern.match(item_id)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return safe or "coordinator_item"


__all__ = [
    "CoordinatorAgentLaunchResult",
    "CoordinatorBlackboardEntry",
    "CoordinatorBlackboardKind",
    "CoordinatorBlackboardSource",
    "CoordinatorNotificationIngestionResult",
    "CoordinatorRuntime",
    "CoordinatorRuntimeConfig",
    "CoordinatorRuntimeProtocol",
    "CoordinatorRuntimeSnapshot",
    "CoordinatorRuntimeSnapshotDecodeError",
    "CoordinatorSendMessageResult",
    "CoordinatorTaskStopResult",
    "CoordinatorWorkItem",
    "CoordinatorWorkKind",
    "CoordinatorWorkStatus",
    "coordinator_runtime_snapshot_from_dict",
    "coordinator_runtime_snapshot_to_dict",
]
