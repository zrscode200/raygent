"""Stall watchdogs for local_bash tasks.

(`startStallWatchdog`). Key design properties:

- **Signal-only, no kill.** The watchdog emits a `TaskNotification`
  (`kind="stalled"`) when a task stops producing output past the
  threshold. It does NOT kill the task. Hard-kill paths belong to
  separate mechanisms (absolute timeout, runaway-output size guard).
- **One-shot per stall detection.** After firing the notification the
  watchdog cancels itself (matches reference's `cancelled=true;
  wants continuous monitoring after a fire, it restarts the watchdog.
- **Regex prompt-pattern gating deferred** (reference does this at
  validated patterns land in a follow-up.

Shape: `StallWatchdog` is a `Protocol` returning `asyncio.Task[None]`.
Caller owns the cancel — when the task terminates, caller calls
`task.cancel()`. This matches Python asyncio idioms better than
returning a cancel callable (reference shape).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from raygent_harness.core.task import TaskNotification

if TYPE_CHECKING:
    from raygent_harness.core.task import AppStateStore


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


DEFAULT_STALL_CHECK_INTERVAL_S = 5.0
"""Polling cadence. Reference `STALL_CHECK_INTERVAL_MS=5_000`."""

DEFAULT_STALL_THRESHOLD_S = 45.0
"""Quiescence window. No output growth for this long → stall signal.
Reference `STALL_THRESHOLD_MS=45_000`."""


# ---------------------------------------------------------------------------
# LastOutputObserver — abstraction over "when did the task last produce
# output." Decouples watchdog from how output is captured (stdout pipe,
# disk file, etc.).
# ---------------------------------------------------------------------------


class LastOutputObserver(Protocol):
    """Returns the wall-clock timestamp of the most recent output from
    the task. Watchdog subtracts from `time.time()` to decide stall.

    Reference uses file size change as the observable (`stat(outputPath
    abstraction so v1 can use an in-memory timestamp field on
    `LocalBashState` and a future v2 can switch to disk-based observation
    without touching watchdog logic.
    """

    def last_output_at(self) -> float: ...


# ---------------------------------------------------------------------------
# StallWatchdog protocol — what LocalBashTask consumes.
# ---------------------------------------------------------------------------


class StallWatchdog(Protocol):
    """Start-and-return-task contract. Caller owns the cancel lifecycle.

    we return `asyncio.Task[None]` so callers use `.cancel()` — matches
    Python asyncio lifecycle idioms.
    """

    def start(
        self,
        task_id: str,
        description: str,
        observer: LastOutputObserver,
        store: AppStateStore,
        *,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> asyncio.Task[None]: ...


# ---------------------------------------------------------------------------
# QuiescenceWatchdog — the only concrete impl in v1.
# ---------------------------------------------------------------------------


@dataclass
class QuiescenceWatchdog:
    """Poll `observer.last_output_at()` on interval. When gap exceeds
    `threshold_s`, enqueue a `TaskNotification(kind="stalled", ...)` and
    terminate. Exactly one notification per watchdog lifetime.
    """

    check_interval_s: float = DEFAULT_STALL_CHECK_INTERVAL_S
    threshold_s: float = DEFAULT_STALL_THRESHOLD_S

    def start(
        self,
        task_id: str,
        description: str,
        observer: LastOutputObserver,
        store: AppStateStore,
        *,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> asyncio.Task[None]:
        return asyncio.create_task(
            self._run(
                task_id=task_id,
                description=description,
                observer=observer,
                store=store,
                tool_use_id=tool_use_id,
                agent_id=agent_id,
            ),
            name=f"stall-watchdog:{task_id}",
        )

    async def _run(
        self,
        *,
        task_id: str,
        description: str,
        observer: LastOutputObserver,
        store: AppStateStore,
        tool_use_id: str | None,
        agent_id: str | None,
    ) -> None:
        """Polling loop. Exits on: (a) stall detected and fired, or (b)
        task cancelled externally (CancelledError propagates).

        We intentionally do not catch CancelledError — caller's cancel
        is how this watchdog ends in the normal path.
        """
        while True:
            await asyncio.sleep(self.check_interval_s)
            now = time.time()
            gap = now - observer.last_output_at()
            if gap < self.threshold_s:
                continue

            store.enqueue_notification(
                TaskNotification(
                    task_id=task_id,
                    message=(
                        f'Background task "{description}" produced no output '
                        f"for {gap:.0f}s (threshold {self.threshold_s:.0f}s). "
                        "The task may be stalled or waiting on input. Consider "
                        "stopping it and re-running with non-interactive flags."
                    ),
                    kind="stalled",
                    tool_use_id=tool_use_id,
                    priority="next",
                    agent_id=agent_id,
                ),
            )
            # One-shot: fire and exit. Matches reference at
            # `cancelled=true` and clears the interval after firing.
            return


__all__ = [
    "DEFAULT_STALL_CHECK_INTERVAL_S",
    "DEFAULT_STALL_THRESHOLD_S",
    "LastOutputObserver",
    "QuiescenceWatchdog",
    "StallWatchdog",
]
