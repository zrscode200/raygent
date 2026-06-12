from __future__ import annotations

from raygent_harness.coordinator import CoordinatorRuntime
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.task import AppStateStore, TaskNotification
from raygent_harness.services.task_notification_replay import (
    TaskNotificationReplayRecord,
    replay_task_notifications,
)


def _record(
    task_id: str,
    *,
    agent_id: str | None = None,
    priority: str = "later",
    created_at: float = 1.0,
    dedupe_key: str | None = None,
    explicitly_stopped: bool = False,
) -> TaskNotificationReplayRecord:
    return TaskNotificationReplayRecord(
        task_id=task_id,
        message=f"<task_notification><task_id>{task_id}</task_id></task_notification>",
        kind="completed",
        priority=priority,  # type: ignore[arg-type]
        agent_id=agent_id,
        created_at=created_at,
        dedupe_key=dedupe_key or f"replay:{task_id}",
        explicitly_stopped=explicitly_stopped,
    )


def test_replay_preserves_queue_priority_order_and_agent_filtering() -> None:
    store = AppStateStore()
    replay_result = replay_task_notifications(
        store=store,
        records=(
            _record("restored-later", created_at=20.0),
            _record("child-now", agent_id="child", priority="now", created_at=5.0),
            _record("restored-now", priority="now", created_at=10.0),
        ),
    )
    store.enqueue_notification(
        TaskNotification(
            task_id="live-later",
            message="live",
            kind="completed",
            priority="later",
            created_at=15.0,
        )
    )

    assert replay_result.enqueued_count == 3
    assert [n.task_id for n in store.drain_notifications(None)] == [
        "restored-now",
        "restored-later",
        "live-later",
    ]
    assert [n.task_id for n in store.drain_notifications("child")] == ["child-now"]


def test_replay_suppresses_explicit_stops_and_duplicate_replay_keys() -> None:
    sink = RecordingKernelEventSink()
    store = AppStateStore(observability=KernelEventBus([sink]))

    result = replay_task_notifications(
        store=store,
        records=(
            _record("stopped", explicitly_stopped=True),
            _record("dup", dedupe_key="stable-dup"),
            _record("dup", dedupe_key="stable-dup"),
        ),
    )

    assert result.enqueued_count == 1
    assert result.skipped_explicit_stop_count == 1
    assert result.skipped_duplicate_count == 1
    assert [n.task_id for n in store.drain_notifications(None)] == ["dup"]
    assert "SECRET" not in str([event.data for event in sink.events])


def test_replay_suppresses_facts_already_ingested_by_coordinator_snapshot() -> None:
    runtime = CoordinatorRuntime()
    store = AppStateStore()
    record = _record("coordinated", dedupe_key="stable-coordinator-key")
    runtime.record_task_notifications([record.to_notification()])
    restored_runtime = CoordinatorRuntime.from_snapshot(runtime.snapshot())

    result = replay_task_notifications(
        store=store,
        records=(record,),
        coordinator_runtime=restored_runtime,
    )

    assert result.enqueued_count == 0
    assert result.skipped_coordinator_processed_count == 1
    assert store.drain_notifications(None) == []
