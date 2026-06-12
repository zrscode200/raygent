"""Subprocess-backed local bash task implementation.

The task separates timeout/resource protection, runaway-output protection, and
stall notification. Output is written through TaskOutputStore while state keeps
only a bounded recent-output tail.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from raygent_harness.core.stall_watchdog import (
    QuiescenceWatchdog,
    StallWatchdog,
)
from raygent_harness.core.task import (
    TERMINAL_STATUSES,
    AppStateStore,
    TaskNotFoundError,
    TaskNotification,
    TaskStateBase,
    TaskType,
    UnsupportedTaskTypeError,
    mark_notified_if_unset,
    register_task_impl,
)
from raygent_harness.services.task_output import (
    DEFAULT_MAX_READ_BYTES,
    FileTaskOutputStore,
    TaskOutputReadResult,
    TaskOutputStore,
    read_task_output_file_range,
    read_task_output_file_tail,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------


DEFAULT_TIMEOUT_S = 120.0
"""Default hard timeout for foreground local bash tasks."""

DEFAULT_MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB
"""Runaway-output SIGKILL threshold, separate from stall detection."""

DEFAULT_OUTPUT_TAIL_BYTES = 64 * 1024
"""Bounded compatibility cache retained in LocalBashState."""

_default_output_store_instance: FileTaskOutputStore | None = None

# ---------------------------------------------------------------------------
# LocalBashState — per-task state record.
# ---------------------------------------------------------------------------


@dataclass
class LocalBashState(TaskStateBase):
    """State for a local_bash task. Extends TaskStateBase.

    `output_file` on TaskStateBase points at the file-backed task output.
    `output_buffer` is only a bounded recent-output tail for compatibility
    with older callers/tests. `last_output_at` gets bumped on every accepted
    write; the stall watchdog reads it via `_StateObserver` below.
    """

    command: str = ""
    """The shell command string."""

    cwd: str | None = None
    """Working directory. None means inherit from parent process."""

    timeout_s: float = DEFAULT_TIMEOUT_S
    """Absolute timeout before hard SIGTERM."""

    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    """Output-size SIGKILL threshold. Separate from stall detection."""

    output_buffer: list[bytes] = field(default_factory=list[bytes])
    """Bounded recent-output tail (stdout + stderr merged).

    Bytes, not str, because subprocess output is not guaranteed UTF-8. The
    authoritative full output is in `output_file` / `TaskOutputStore`.
    """

    output_bytes: int = 0
    """Running total of accepted output bytes written to TaskOutputStore."""

    output_tail_bytes: int = DEFAULT_OUTPUT_TAIL_BYTES
    """Maximum bytes retained in `output_buffer`."""

    output_truncated: bool = False
    """True when the output cap fired and over-cap chunks were dropped."""

    output_error: str | None = None
    """Sanitized output-store error when file-backed writes fail closed."""

    last_output_at: float = 0.0
    """Wall clock of most recent write to `output_buffer`. Stall watchdog
    polls this field."""

    exit_code: int | None = None
    """Populated on completion. None while running."""

    killed_by_timeout: bool = False
    killed_by_size: bool = False
    """Distinguish the two hard-kill paths for telemetry / error message
    disambiguation."""

    agent_id: str | None = None
    """Which agent owns this shell. Set when a subagent spawns a bash task
    so its parent's `kill_shell_tasks_for_agent(agent_id)` cleanup can
    shells on subagent exit by filtering tasks where the task's
    `agentId == subagent.agentId`. Threading via notification alone
    (the prior shape) was not enough — cleanup needs to filter the
    store, which means the field has to live on state."""


# ---------------------------------------------------------------------------
# _StateObserver — adapts LocalBashState to the LastOutputObserver
# protocol expected by the stall watchdog.
# ---------------------------------------------------------------------------


@dataclass
class _StateObserver:
    """Reads `last_output_at` from the AppStateStore on every poll (not
    from a snapshot). This matters: a captured-at-start reference to
    LocalBashState would be a different object after any
    `store.update_task(...)` swap.
    """

    store: AppStateStore
    task_id: str

    def last_output_at(self) -> float:
        task = self.store.tasks.get(self.task_id)
        if not isinstance(task, LocalBashState):
            # Task was removed or replaced with a different type. Return
            # `time.time()` to signal "no stall" — the real task state is
            # gone so the watchdog would have nothing to flag anyway.
            return time.time()
        return task.last_output_at


# ---------------------------------------------------------------------------
# LocalBashTask — the vtable impl.
# ---------------------------------------------------------------------------


@dataclass
class LocalBashTask:
    """Task-vtable implementation for `type == "local_bash"`.

    Instances are registered at module import via `register_task_impl`.
    The default instance uses `QuiescenceWatchdog`; tests / consumers
    can construct a `LocalBashTask(watchdog=...)` and register their
    own instance.

    `name` / `type` are regular (not ClassVar) fields because the `Task`
    protocol expects instance attributes. Functionally treated as
    constants — don't mutate post-construction.
    """

    name: str = "local_bash"
    type: TaskType = "local_bash"
    watchdog: StallWatchdog = field(default_factory=QuiescenceWatchdog)

    # ----- vtable: kill -----

    async def kill(self, task_id: str, store: AppStateStore) -> None:
        """Idempotent cancel. No-op when the task is already terminal
        (matches Task protocol contract). Process-tree-safe — signals
        the whole process group, not just the parent PID."""
        task = store.tasks.get(task_id)
        if not isinstance(task, LocalBashState):
            return
        if task.status in TERMINAL_STATUSES:
            return
        proc = _process_for(task_id)
        if proc is not None:
            _kill_process_tree(proc, signal.SIGTERM)
        # Update state to "killed". Exit code resolution happens in the
        # watcher; we set status eagerly so subsequent kill() calls no-op.
        store.update_task(
            task_id,
            lambda t: _replace(
                t,
                status="killed",
                end_time=time.time(),
            ),
        )


# ---------------------------------------------------------------------------
# Process registry — a process-wide map task_id → asyncio.subprocess.Process.
# Kept outside the state store because Process objects aren't serializable and
# don't belong in the immutable-state model. Access via `_process_for(task_id)`.
# ---------------------------------------------------------------------------


_PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_DRIVER_TASKS: dict[str, asyncio.Task[None]] = {}
"""Strong-reference registry for the `_drive` coroutine of each task.
Without this, asyncio would GC the driver task while the subprocess is
still running (RUF006). Keyed by task_id; entries removed in `_drive`
just before the terminal notification fires."""


def _process_for(task_id: str) -> asyncio.subprocess.Process | None:
    return _PROCESSES.get(task_id)


def _kill_process_tree(
    proc: asyncio.subprocess.Process,
    sig: int = signal.SIGTERM,
) -> None:
    """Process-tree-safe kill. Signals the entire process group rather
    than just the parent PID — children spawned by the shell command
    would otherwise orphan.

    npm package (`treeKill(this.#childProcess.pid, 'SIGKILL')`). The
    Python equivalent is `os.killpg(pgid, sig)` when the process was
    started in its own session (so PID == PGID).

    POSIX-only. On Windows, asyncio doesn't expose process groups in
    the same shape; `proc.terminate()` falls back to per-process
    terminate which is best-effort. Marked TBD.
    """
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


# ---------------------------------------------------------------------------
# spawn_local_bash — the public entry point.
# ---------------------------------------------------------------------------


async def spawn_local_bash(
    command: str,
    store: AppStateStore,
    *,
    description: str = "",
    cwd: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    tool_use_id: str | None = None,
    agent_id: str | None = None,
    watchdog: StallWatchdog | None = None,
    output_store: TaskOutputStore | None = None,
) -> LocalBashState:
    """Register a LocalBashState, spawn the subprocess, start the stall
    watchdog + size guard, and return the initial state.

    Caller can `await run_until_done(task_id, store)` to block on
    completion, or leave the task running and consume completion via
    the notification queue.
    """
    task_id = _generate_id()
    effective_output_store = output_store or _default_output_store()
    output_initialized = False

    # Fail closed before launching the subprocess. If output storage cannot be
    # safely initialized, running the command would either lose output or tempt
    # an unsafe in-memory fallback.
    output_reference = await effective_output_store.init_task_output(task_id)
    output_initialized = True

    # Initialize output first, then spawn. If create_subprocess_shell raises
    # (bad cwd, missing binary, etc.), nothing has been registered and the
    # empty output file is cleaned up. Reference does the equivalent:
    # shell command exists.
    #
    # `start_new_session=True` puts the child in its own process group
    # so kill paths can signal the whole tree (see _kill_process_tree).
    # POSIX-only; on Windows, asyncio doesn't accept the kwarg
    # consistently. Build the kwargs dict so the cross-platform branch
    # is one obvious site.
    spawn_kwargs: dict[str, object] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
        "cwd": cwd,
    }
    if sys.platform != "win32":
        spawn_kwargs["start_new_session"] = True

    try:
        proc = await asyncio.create_subprocess_shell(command, **spawn_kwargs)  # type: ignore[arg-type]
    except Exception:
        if output_initialized:
            with contextlib.suppress(Exception):
                await effective_output_store.cleanup_task_output(task_id)
        raise

    state = LocalBashState(
        id=task_id,
        type="local_bash",
        description=description or command,
        status="running",
        start_time=time.time(),
        last_output_at=time.time(),
        command=command,
        cwd=cwd,
        timeout_s=timeout_s,
        max_output_bytes=max_output_bytes,
        tool_use_id=tool_use_id,
        output_file=output_reference.path,
        agent_id=agent_id,
    )
    store.register_task(state)
    _PROCESSES[task_id] = proc

    effective_watchdog = watchdog or QuiescenceWatchdog()
    observer = _StateObserver(store=store, task_id=task_id)
    watchdog_task = effective_watchdog.start(
        task_id=task_id,
        description=state.description,
        observer=observer,
        store=store,
        tool_use_id=tool_use_id,
        agent_id=agent_id,
    )

    # Fire-and-forget driver. Caller awaits via run_until_done or
    # consumes completion via the notification queue. Strong reference
    # held in `_DRIVER_TASKS` so the task isn't GC'd mid-run.
    driver = asyncio.create_task(
        _drive(
            task_id=task_id,
            proc=proc,
            store=store,
            watchdog_task=watchdog_task,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            agent_id=agent_id,
            tool_use_id=tool_use_id,
            output_store=effective_output_store,
        ),
        name=f"local-bash-driver:{task_id}",
    )
    _DRIVER_TASKS[task_id] = driver

    return state


async def read_local_bash_output(
    task_id: str,
    store: AppStateStore,
    *,
    output_store: TaskOutputStore | None = None,
    offset: int | None = None,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
    tail: bool = True,
) -> TaskOutputReadResult:
    """Read local_bash output through the headless task-output store.

    `tail=True` reads the bounded end of the output by default. Pass
    `tail=False` with an `offset` for delta/range-style reads.
    """

    task = store.tasks.get(task_id)
    if task is None:
        msg = f"No task found with ID: {task_id}"
        raise TaskNotFoundError(msg)
    if not isinstance(task, LocalBashState):
        msg = f"Task {task_id} is not a local_bash task"
        raise UnsupportedTaskTypeError(msg)

    if output_store is not None:
        await output_store.flush_task_output(task_id)
        if tail:
            return await output_store.read_tail(task_id, max_bytes=max_bytes)
        return await output_store.read_range(
            task_id,
            offset=0 if offset is None else offset,
            max_bytes=max_bytes,
        )

    if task.output_file is not None:
        if tail:
            return await read_task_output_file_tail(
                task_id,
                task.output_file,
                max_bytes=max_bytes,
            )
        return await read_task_output_file_range(
            task_id,
            task.output_file,
            0 if offset is None else offset,
            max_bytes=max_bytes,
        )

    effective_output_store = _default_output_store()
    await effective_output_store.flush_task_output(task_id)
    if tail:
        return await effective_output_store.read_tail(task_id, max_bytes=max_bytes)
    return await effective_output_store.read_range(
        task_id,
        offset=0 if offset is None else offset,
        max_bytes=max_bytes,
    )


async def kill_shell_tasks_for_agent(
    agent_id: str,
    store: AppStateStore,
) -> None:
    """Kill all running local_bash tasks owned by `agent_id`.

    Snapshots matching task IDs before awaiting any kill so that
    concurrent store mutations during the cleanup don't cause us to
    skip or double-kill a task. Filters to `status == "running"`
    upfront — terminal tasks are no-ops anyway, but skipping them
    keeps the await loop tight.

    `killShellTasksForAgent` in the subagent's `finally` so a
    `run_in_background` shell loop can't outlive its parent agent
    as a PPID=1 zombie.
    """
    impl = LocalBashTask()
    matching_ids: list[str] = [
        task.id
        for task in store.tasks.values()
        if isinstance(task, LocalBashState)
        and task.agent_id == agent_id
        and task.status == "running"
    ]
    for task_id in matching_ids:
        await impl.kill(task_id, store)


def suppress_shell_notifications_for_agent(
    agent_id: str,
    store: AppStateStore,
) -> None:
    """Pre-mark running local_bash tasks owned by `agent_id` as notified.

    Child query runners call this before cleanup kills so terminal
    SIGTERM/SIGKILL noise from child-owned shells does not orphan notifications
    addressed to an agent that is already exiting.
    """

    for task in list(store.tasks.values()):
        if (
            isinstance(task, LocalBashState)
            and task.agent_id == agent_id
            and task.status == "running"
            and not task.notified
        ):
            store.update_task(task.id, lambda t: _replace(t, notified=True))


async def cleanup_shell_tasks_for_agent(
    agent_id: str,
    store: AppStateStore,
) -> None:
    """Suppress, kill, and drain local_bash tasks owned by `agent_id`."""

    suppress_shell_notifications_for_agent(agent_id, store)
    await kill_shell_tasks_for_agent(agent_id, store)
    store.drain_notifications(agent_id)


async def run_until_done(task_id: str, store: AppStateStore) -> LocalBashState:
    """Await until the task reaches a terminal status. Returns the final
    LocalBashState. Useful for synchronous tool-call patterns that want
    to await completion inline rather than consume the notification.
    """
    while True:
        task = store.tasks.get(task_id)
        if isinstance(task, LocalBashState) and task.status in TERMINAL_STATUSES:
            return task
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# _drive — the subprocess lifecycle coroutine.
# ---------------------------------------------------------------------------


async def _drive(
    *,
    task_id: str,
    proc: asyncio.subprocess.Process,
    store: AppStateStore,
    watchdog_task: asyncio.Task[None],
    timeout_s: float,
    max_output_bytes: int,
    agent_id: str | None,
    tool_use_id: str | None,
    output_store: TaskOutputStore,
) -> None:
    """Own the process lifecycle: reader + size-guard + timeout + completion.

    Returns when the process exits (for any reason) and the terminal
    notification is enqueued. Handles all three hard-kill paths:
    external `.kill()` (status set by LocalBashTask.kill), absolute
    timeout (SIGTERM), runaway output (SIGKILL).
    """
    reader_task = asyncio.create_task(
        _read_output(task_id, proc, store, max_output_bytes, output_store),
        name=f"local-bash-reader:{task_id}",
    )

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except TimeoutError:
        store.update_task(
            task_id,
            lambda t: _replace(t, killed_by_timeout=True),
        )
        if proc.returncode is None:
            _kill_process_tree(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                _kill_process_tree(proc, signal.SIGKILL)
                await proc.wait()

    exit_code = proc.returncode if proc.returncode is not None else -1

    watchdog_task.cancel()
    # The subprocess has exited, so stdout should reach EOF. Prefer awaiting
    # the reader so the last chunks are written before terminal notification.
    # Fall back to cancellation if the transport does not settle promptly.
    try:
        await asyncio.wait_for(reader_task, timeout=2.0)
    except TimeoutError:
        reader_task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await reader_task
    except Exception as exc:
        _record_output_error(task_id, store, exc)

    with contextlib.suppress(Exception, asyncio.CancelledError):
        await watchdog_task

    try:
        await output_store.flush_task_output(task_id)
    except Exception as exc:
        _record_output_error(task_id, store, exc)

    _PROCESSES.pop(task_id, None)
    _DRIVER_TASKS.pop(task_id, None)

    # Resolve final status. The hard-kill paths (size-kill, timeout-kill)
    # both collapse to "killed"; `killed_by_size` / `killed_by_timeout`
    # are flags that disambiguate the cause for the completion message,
    # not separate status values.
    task = store.tasks.get(task_id)
    if not isinstance(task, LocalBashState):
        return

    final_status = _resolve_final_status(task, exit_code)

    # Preserve any `end_time` already set by an explicit `kill()` call —
    # that timestamp marks when the user-visible cancellation happened,
    # not when the driver got around to reaping the process. Reference
    # `task.status === 'killed'` branch returns the unchanged task
    # rather than overwriting endTime.
    now = time.time()
    store.update_task(
        task_id,
        lambda t: _replace(
            t,
            status=final_status,
            end_time=t.end_time if t.end_time is not None else now,
            exit_code=exit_code,
        ),
    )

    # Enqueue terminal notification (atomic check-and-set via
    # mark_notified_if_unset prevents duplicates if an explicit kill
    # fired first).
    if mark_notified_if_unset(store, task_id):
        store.enqueue_notification(
            TaskNotification(
                task_id=task_id,
                message=_build_completion_message(
                    task=task,
                    final_status=final_status,
                    exit_code=exit_code,
                ),
                kind="completed" if final_status == "completed" else "error",
                tool_use_id=tool_use_id,
                # (the MONITOR_TOOL feature bumps it to 'next' but we
                # don't have that gate). Default `later` keeps user
                # input from being starved by terminal notifications.
                priority="later",
                agent_id=agent_id,
            ),
        )
    with contextlib.suppress(Exception):
        await output_store.evict_task_output(task_id)


def _resolve_final_status(
    task: LocalBashState,
    exit_code: int,
) -> str:
    if task.status == "killed":
        return "killed"
    if task.output_error is not None:
        return "failed"
    if task.killed_by_size or task.killed_by_timeout:
        return "killed"
    return "completed" if exit_code == 0 else "failed"


async def _read_output(
    task_id: str,
    proc: asyncio.subprocess.Process,
    store: AppStateStore,
    max_output_bytes: int,
    output_store: TaskOutputStore,
) -> None:
    """Drain proc.stdout into the task-output store. Updates `last_output_at`
    on every accepted chunk — that's what the stall watchdog polls.

    Enforces `max_output_bytes` AT THE READER, not via a polling
    works because content is buffered to a file fd outside the JS runtime.
    Raygent writes each accepted chunk to `TaskOutputStore` and keeps only
    a bounded state tail, so successful long-output commands do not live in
    task state.

    On overflow: SIGKILL the process tree, mark `killed_by_size`, drop
    the over-cap chunk so output never exceeds the cap.
    """
    stream = proc.stdout
    if stream is None:
        return
    discarding = False
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        if discarding:
            # Cap already fired; keep draining so the asyncio subprocess
            # transport can close the pipe cleanly (otherwise `proc.wait()`
            # in the driver waits indefinitely on a buffered read end).
            # Memory stays bounded — chunks are dropped on the floor.
            continue
        task = store.tasks.get(task_id)
        if not isinstance(task, LocalBashState):
            return
        if task.output_bytes + len(chunk) > max_output_bytes:
            store.update_task(
                task_id,
                lambda t: _replace(t, killed_by_size=True, output_truncated=True),
            )
            if proc.returncode is None:
                _kill_process_tree(proc, signal.SIGKILL)
            discarding = True
            continue

        try:
            await output_store.append_task_output(task_id, chunk)
        except Exception as exc:
            _record_output_error(task_id, store, exc)
            if proc.returncode is None:
                _kill_process_tree(proc, signal.SIGKILL)
            return

        store.update_task(task_id, _append_output(chunk))
        store.emit_task_progress(
            task_id,
            {
                "progress_type": "output",
                "chunk_byte_count": len(chunk),
                "output_bytes": task.output_bytes + len(chunk),
            },
        )


def _append_output(chunk: bytes) -> Callable[[TaskStateBase], TaskStateBase]:
    """Build an updater that appends `chunk` to LocalBashState's bounded
    compatibility tail and bumps `last_output_at`. Non-LocalBashState targets
    pass through unchanged.
    """

    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, LocalBashState):
            return t
        return _replace(
            t,
            output_buffer=_bounded_tail([*t.output_buffer, chunk], t.output_tail_bytes),
            output_bytes=t.output_bytes + len(chunk),
            last_output_at=time.time(),
        )

    return update


def _build_completion_message(
    *,
    task: LocalBashState,
    final_status: str,
    exit_code: int,
) -> str:
    output_suffix = _output_file_suffix(task)
    if task.output_error is not None:
        return (
            f'Task "{task.description}" failed: output store error '
            f"({task.output_error}).{output_suffix}"
        )
    if final_status == "completed":
        return f'Task "{task.description}" completed (exit 0).{output_suffix}'
    if task.killed_by_size:
        return (
            f'Task "{task.description}" killed: output exceeded '
            f"{task.max_output_bytes:,} bytes.{output_suffix}"
        )
    if task.killed_by_timeout:
        return (
            f'Task "{task.description}" killed: exceeded timeout of '
            f"{task.timeout_s:.0f}s.{output_suffix}"
        )
    if final_status == "killed":
        return f'Task "{task.description}" killed (exit {exit_code}).{output_suffix}'
    return f'Task "{task.description}" failed (exit {exit_code}).{output_suffix}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    """Per-type prefix 'b' matches the reservation in core/task.py's
    `_TASK_ID_PREFIX`."""
    return "b" + secrets.token_hex(8)


def _output_file_suffix(task: LocalBashState) -> str:
    if task.output_file is None:
        return ""
    return f" Output file: {task.output_file}."


def _default_output_store() -> FileTaskOutputStore:
    global _default_output_store_instance
    if _default_output_store_instance is None:
        _default_output_store_instance = FileTaskOutputStore()
    return _default_output_store_instance


def _record_output_error(
    task_id: str,
    store: AppStateStore,
    exc: Exception,
) -> None:
    message = str(exc).replace("\n", " ")[:240] or exc.__class__.__name__
    store.update_task(
        task_id,
        lambda t: _replace(t, output_error=message) if isinstance(t, LocalBashState) else t,
    )


def _bounded_tail(chunks: list[bytes], max_bytes: int) -> list[bytes]:
    if max_bytes <= 0:
        return []
    retained: list[bytes] = []
    total = 0
    for chunk in reversed(chunks):
        if total >= max_bytes:
            break
        remaining = max_bytes - total
        if len(chunk) <= remaining:
            retained.append(chunk)
            total += len(chunk)
            continue
        retained.append(chunk[-remaining:])
        total += remaining
        break
    retained.reverse()
    return retained


def _replace(task: TaskStateBase, **changes: object) -> TaskStateBase:
    """Thin wrapper so we don't have to import `dataclasses.replace`
    everywhere. Preserves the mutable-via-replace pattern used across
    the task framework.
    """
    from dataclasses import replace

    return replace(task, **changes)


# ---------------------------------------------------------------------------
# Module-init: register with the Task registry so `get_task_by_type(
# "local_bash")` returns an instance.
# ---------------------------------------------------------------------------


register_task_impl(LocalBashTask())


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_OUTPUT_TAIL_BYTES",
    "DEFAULT_TIMEOUT_S",
    "LocalBashState",
    "LocalBashTask",
    "cleanup_shell_tasks_for_agent",
    "kill_shell_tasks_for_agent",
    "read_local_bash_output",
    "run_until_done",
    "spawn_local_bash",
    "suppress_shell_notifications_for_agent",
]
