"""Replay recovered task facts into the model-facing notification queue.

Live tasks enqueue `TaskNotification` directly after they transition terminal.
Runtime recovery needs the same model-facing path for facts discovered while the
embedding process was offline, with additional de-duplication against durable
stores and coordinator snapshots.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
)
from raygent_harness.core.task import (
    AppStateStore,
    NotificationKind,
    NotificationMode,
    NotificationPriority,
    TaskNotification,
)
from raygent_harness.services.remote_agent import (
    RemoteAgentPersistenceRecord,
    RemoteAgentRestoreResult,
)

TaskNotificationReplaySource = Literal[
    "runtime_recovery",
    "remote_agent_restore",
    "local_agent_resume",
    "task_output",
    "manual",
]


class TaskNotificationReplayCoordinator(Protocol):
    """Coordinator-runtime subset needed to avoid replaying ingested facts."""

    def has_processed_task_notification(
        self,
        notification: TaskNotification,
        /,
    ) -> bool:
        """Return True when the coordinator snapshot already ingested this fact."""
        ...


@dataclass(frozen=True, slots=True)
class TaskNotificationReplayRecord:
    """Durable/offline task fact normalized for replay.

    `dedupe_key` must be stable across resume attempts for the same recovered
    fact. The message is intentionally model-facing and may include a result;
    observability emitted by this module records only ids, counts, and source
    metadata.
    """

    task_id: str
    message: str
    kind: NotificationKind
    dedupe_key: str
    tool_use_id: str | None = None
    priority: NotificationPriority = "later"
    mode: NotificationMode = "background"
    agent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    source: TaskNotificationReplaySource = "runtime_recovery"
    explicitly_stopped: bool = False

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("TaskNotificationReplayRecord.task_id must be non-empty")
        if not self.message:
            raise ValueError("TaskNotificationReplayRecord.message must be non-empty")
        if not self.dedupe_key:
            raise ValueError("TaskNotificationReplayRecord.dedupe_key must be non-empty")

    def to_notification(self) -> TaskNotification:
        return TaskNotification(
            task_id=self.task_id,
            message=self.message,
            kind=self.kind,
            tool_use_id=self.tool_use_id,
            priority=self.priority,
            mode=self.mode,
            agent_id=self.agent_id,
            created_at=self.created_at,
            dedupe_key=self.dedupe_key,
        )


@dataclass(frozen=True, slots=True)
class TaskNotificationReplayResult:
    """Summary of one replay pass."""

    enqueued_notifications: tuple[TaskNotification, ...] = ()
    skipped_duplicate_count: int = 0
    skipped_explicit_stop_count: int = 0
    skipped_coordinator_processed_count: int = 0
    warning_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def enqueued_count(self) -> int:
        return len(self.enqueued_notifications)


def replay_task_notifications(
    *,
    store: AppStateStore,
    records: Sequence[TaskNotificationReplayRecord],
    coordinator_runtime: TaskNotificationReplayCoordinator | None = None,
    observability: KernelEventBus | None = None,
    event_context: KernelEventContext | None = None,
) -> TaskNotificationReplayResult:
    """Replay recovered facts into `store.notifications`.

    This function is synchronous because `AppStateStore` queue operations are
    synchronous. It still accepts explicit observability/context parameters so
    runtime recovery can report replay as part of its own resume span.
    """

    bus = observability or store.observability
    context = event_context or KernelEventContext(source="task_notification_replay")
    enqueued: list[TaskNotification] = []
    skipped_duplicate = 0
    skipped_explicit_stop = 0
    skipped_coordinator_processed = 0
    warnings: list[str] = []

    for record in records:
        if record.explicitly_stopped:
            skipped_explicit_stop += 1
            _emit_replay_skipped(
                bus,
                context,
                record,
                reason="explicit_stop",
            )
            continue

        notification = record.to_notification()
        if coordinator_runtime is not None:
            try:
                if coordinator_runtime.has_processed_task_notification(notification):
                    skipped_coordinator_processed += 1
                    _emit_replay_skipped(
                        bus,
                        context,
                        record,
                        reason="coordinator_processed",
                    )
                    continue
            except Exception as exc:
                warnings.append(
                    "coordinator_processed_check_failed:"
                    f"{type(exc).__name__}"
                )

        accepted = store.enqueue_notification(notification)
        if accepted:
            enqueued.append(notification)
        else:
            skipped_duplicate += 1
            _emit_replay_skipped(
                bus,
                context,
                record,
                reason="duplicate",
            )

    bus.emit(
        "task_notification_replay.completed",
        context=context,
        data={
            "record_count": len(records),
            "enqueued_count": len(enqueued),
            "skipped_duplicate_count": skipped_duplicate,
            "skipped_explicit_stop_count": skipped_explicit_stop,
            "skipped_coordinator_processed_count": skipped_coordinator_processed,
            "warning_count": len(warnings),
            "sources": tuple(record.source for record in records),
        },
    )
    return TaskNotificationReplayResult(
        enqueued_notifications=tuple(enqueued),
        skipped_duplicate_count=skipped_duplicate,
        skipped_explicit_stop_count=skipped_explicit_stop,
        skipped_coordinator_processed_count=skipped_coordinator_processed,
        warning_count=len(warnings),
        warnings=tuple(warnings),
    )


def remote_agent_restore_replay_record(
    record: RemoteAgentPersistenceRecord,
    result: RemoteAgentRestoreResult,
) -> TaskNotificationReplayRecord | None:
    """Build a replay record for terminal remote-agent restore results.

    `archived` and `gone` are cleanup facts, not model-facing terminal results.
    Explicit stops are already suppressed by the remote-agent restore path when
    a terminal Raygent task state exists in the store.
    """

    if result.status not in {"completed", "failed"}:
        return None
    final_status = cast(Literal["completed", "failed"], result.status)
    message = _build_remote_agent_restore_notification_message(
        record,
        final_status=final_status,
        final_message=result.message,
        error=result.error,
    )
    return TaskNotificationReplayRecord(
        task_id=record.task_id,
        message=message,
        kind="completed" if final_status == "completed" else "error",
        dedupe_key=remote_agent_terminal_dedupe_key(
            task_id=record.task_id,
            remote_id=record.remote_id,
            final_status=final_status,
        ),
        tool_use_id=record.tool_use_id,
        priority="later",
        agent_id=record.parent_agent_id,
        created_at=record.updated_at or record.start_time or time.time(),
        source="remote_agent_restore",
    )


def remote_agent_terminal_dedupe_key(
    *,
    task_id: str,
    remote_id: str,
    final_status: Literal["completed", "failed"],
) -> str:
    """Stable key shared by live and restore remote terminal notifications."""

    digest = hashlib.sha256(f"{task_id}\n{remote_id}".encode()).hexdigest()[:16]
    return f"remote_agent_terminal:{task_id}:{final_status}:{digest}"


def _emit_replay_skipped(
    bus: KernelEventBus,
    context: KernelEventContext,
    record: TaskNotificationReplayRecord,
    *,
    reason: str,
) -> None:
    bus.emit(
        "task_notification_replay.skipped",
        context=context,
        data={
            "task_id": record.task_id,
            "kind": record.kind,
            "priority": record.priority,
            "agent_id": record.agent_id,
            "source": record.source,
            "reason": reason,
            "dedupe_key_present": bool(record.dedupe_key),
        },
    )


def _build_remote_agent_restore_notification_message(
    record: RemoteAgentPersistenceRecord,
    *,
    final_status: Literal["completed", "failed"],
    final_message: str,
    error: str | None,
) -> str:
    parts = [
        "<task_notification>",
        f"<task_id>{record.task_id}</task_id>",
        "<task_type>remote_agent</task_type>",
        f"<status>{final_status}</status>",
    ]
    summary = (
        f'Remote agent "{record.description}" completed while offline'
        if final_status == "completed"
        else f'Remote agent "{record.description}" failed while offline'
        + (f": {error}" if error else "")
    )
    parts.append(f"<summary>{summary}</summary>")
    if final_message:
        tag = "result" if final_status == "completed" else "partial_result"
        parts.append(f"<{tag}>{final_message}</{tag}>")
    parts.append("</task_notification>")
    return "\n".join(parts)


__all__ = [
    "TaskNotificationReplayCoordinator",
    "TaskNotificationReplayRecord",
    "TaskNotificationReplayResult",
    "TaskNotificationReplaySource",
    "remote_agent_restore_replay_record",
    "remote_agent_terminal_dedupe_key",
    "replay_task_notifications",
]
