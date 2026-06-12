"""AppStateStore notification queue invariants — drain filter, priority
ordering, atomic mark_notified_if_unset."""

from __future__ import annotations

import time

from raygent_harness.core.task import (
    AppStateStore,
    TaskNotification,
    mark_notified_if_unset,
)
from raygent_harness.core.tasks.local_bash import LocalBashState


def _notif(
    *,
    task_id: str,
    agent_id: str | None,
    priority: str = "next",
    message: str = "x",
) -> TaskNotification:
    return TaskNotification(
        task_id=task_id,
        message=message,
        kind="completed",
        tool_use_id=None,
        priority=priority,  # type: ignore[arg-type]
        agent_id=agent_id,
    )


def test_drain_by_agent_id_filters_correctly() -> None:
    store = AppStateStore()
    store.enqueue_notification(_notif(task_id="t1", agent_id="parent"))
    store.enqueue_notification(_notif(task_id="t2", agent_id="other"))
    store.enqueue_notification(_notif(task_id="t3", agent_id=None))
    store.enqueue_notification(_notif(task_id="t4", agent_id="parent"))

    parent_drained = store.drain_notifications("parent")
    assert {n.task_id for n in parent_drained} == {"t1", "t4"}

    # Remaining queue still has the other two.
    main_drained = store.drain_notifications(None)
    assert [n.task_id for n in main_drained] == ["t3"]

    other_drained = store.drain_notifications("other")
    assert [n.task_id for n in other_drained] == ["t2"]


def test_drain_orders_by_priority_then_fifo() -> None:
    store = AppStateStore()
    # Enqueue out of priority order; created_at is monotonic via time.monotonic.
    n_low = _notif(task_id="low", agent_id=None, priority="next")
    n_high1 = _notif(task_id="high1", agent_id=None, priority="now")
    n_high2 = _notif(task_id="high2", agent_id=None, priority="now")
    store.enqueue_notification(n_low)
    # Force monotonic ordering of created_at since they're set in __init__.
    time.sleep(0.001)
    n_high1 = _notif(task_id="high1", agent_id=None, priority="now")
    store.enqueue_notification(n_high1)
    time.sleep(0.001)
    n_high2 = _notif(task_id="high2", agent_id=None, priority="now")
    store.enqueue_notification(n_high2)

    drained = store.drain_notifications(None)
    # immediate priority comes before next; within priority, FIFO by created_at.
    assert [n.task_id for n in drained] == ["high1", "high2", "low"]


def test_enqueue_notification_suppresses_only_stable_dedupe_duplicates() -> None:
    store = AppStateStore()
    keyed = TaskNotification(
        task_id="deduped",
        message="first",
        kind="completed",
        dedupe_key="stable-key-1",
    )
    duplicate = TaskNotification(
        task_id="deduped",
        message="second",
        kind="completed",
        dedupe_key="stable-key-1",
    )
    unkeyed_a = TaskNotification(
        task_id="unkeyed",
        message="first",
        kind="completed",
    )
    unkeyed_b = TaskNotification(
        task_id="unkeyed",
        message="second",
        kind="completed",
    )

    assert store.enqueue_notification(keyed) is True
    assert store.enqueue_notification(duplicate) is False
    assert store.enqueue_notification(unkeyed_a) is True
    assert store.enqueue_notification(unkeyed_b) is True

    drained = store.drain_notifications(None)
    assert [notification.message for notification in drained] == [
        "first",
        "first",
        "second",
    ]


def test_mark_notified_if_unset_is_atomic_and_idempotent() -> None:
    store = AppStateStore()
    state = LocalBashState(
        id="t1",
        type="local_bash",
        description="x",
        status="running",
        start_time=time.time(),
        command="echo",
    )
    store.register_task(state)

    # First call wins — returns True, flag flips.
    assert mark_notified_if_unset(store, "t1") is True
    assert store.tasks["t1"].notified is True

    # Subsequent calls are no-ops — return False.
    assert mark_notified_if_unset(store, "t1") is False
    assert mark_notified_if_unset(store, "t1") is False
