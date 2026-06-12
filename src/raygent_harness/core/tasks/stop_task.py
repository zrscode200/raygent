"""Public stop entry point — the only supported way to stop a task.

Wraps the internal type-agnostic dispatcher in `core.task` and adds
per-type post-kill notification policy:

- **LocalBashTask:** SIGTERM termination produces an "exit code 137" status
  notification that is operational noise. After a successful kill, mark
  the task `notified=True` so the bash driver's later terminal-path
  `mark_notified_if_unset` returns False and suppresses its enqueue.
- **LocalAgentTask:** the driver's terminal notification carries the
  partial result via `<partial_result>`. Do NOT suppress — it is the
  payload, not noise.

Direct use of `core.task.dispatch_stop_task` bypasses this policy and
will leak bash kill notifications. The dispatcher is intentionally
excluded from `core.task.__all__` to keep this module the only
documented public entry point.

event-queue emit; SDK/product event surfaces live outside this wrapper.
"""

from __future__ import annotations

from raygent_harness.core.task import (
    AppStateStore,
    StopTaskResult,
    dispatch_stop_task,
    mark_notified_if_unset,
)
from raygent_harness.core.tasks.local_bash import LocalBashState


async def stop_task(task_id: str, store: AppStateStore) -> StopTaskResult:
    """Stop a running task, applying per-type post-kill notification policy.

    Raises `TaskNotFoundError` / `TaskNotRunningError` /
    `UnsupportedTaskTypeError` from the internal dispatcher — propagated
    unchanged so callers can branch on `error.code`.
    """
    # Snapshot type BEFORE dispatch — `dispatch_stop_task` runs
    # `task_impl.kill` which may transition state; we want the pre-kill
    # type to decide policy.
    task_before = store.tasks.get(task_id)

    result = await dispatch_stop_task(task_id, store)

    if isinstance(task_before, LocalBashState):
        # `LocalBashTask.kill` flips status="killed" synchronously and
        # returns; the bash driver continues in a separate asyncio task
        # toward its terminal path. Since `mark_notified_if_unset` is
        # synchronous and runs before the next await point, the wrapper
        # wins the race against the driver — it sets notified=True first,
        # the driver's later `mark_notified_if_unset` returns False, and
        # the driver's `enqueue_notification` is gated off.
        mark_notified_if_unset(store, task_id)

    return result


__all__ = ["stop_task"]
