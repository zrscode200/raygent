"""LocalBashTask integration tests — happy path, exit-code failure,
timeout SIGTERM, runaway-output SIGKILL, stall-watchdog signal-without-
kill, kill_shell_tasks_for_agent filtering.

Per group-1 scope decision (d): assert behavior, not platform-specific
signal codes. All tests use tiny timeouts/thresholds + the autouse
registry-cleanup fixture in tests/conftest.py.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from raygent_harness.core.stall_watchdog import QuiescenceWatchdog
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks.local_bash import (
    DEFAULT_OUTPUT_TAIL_BYTES,
    LocalBashState,
    kill_shell_tasks_for_agent,
    read_local_bash_output,
    run_until_done,
    spawn_local_bash,
)
from raygent_harness.services.task_output import (
    FileTaskOutputStore,
    TaskOutputReadResult,
    TaskOutputReference,
)


@pytest.fixture
def output_store(tmp_path: Path) -> FileTaskOutputStore:
    return FileTaskOutputStore(tmp_path, session_id="s")


@pytest.mark.asyncio
async def test_happy_completion_emits_completed_notification(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "echo hello",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "completed"
    assert final.exit_code == 0
    assert b"hello" in b"".join(final.output_buffer)
    output = await read_local_bash_output(
        state.id,
        store,
        output_store=output_store,
    )
    assert b"hello" in output.content

    notifs = store.drain_notifications("a1")
    assert len(notifs) == 1
    assert notifs[0].kind == "completed"
    assert notifs[0].task_id == state.id


@pytest.mark.asyncio
async def test_nonzero_exit_emits_error_notification(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "false",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "failed"
    assert final.exit_code != 0

    notifs = store.drain_notifications("a1")
    assert len(notifs) == 1
    assert notifs[0].kind == "error"


@pytest.mark.asyncio
async def test_timeout_kills_process_and_flags_killed_by_timeout(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "sleep 5",
        store,
        timeout_s=0.2,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "killed"
    assert final.killed_by_timeout is True


@pytest.mark.asyncio
async def test_runaway_output_kills_via_size_guard(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    # Generate output far above the cap. `yes` floods stdout; the guard
    # should land before the timeout (default 600s).
    state = await spawn_local_bash(
        "yes",
        store,
        max_output_bytes=512,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "killed"
    assert final.killed_by_size is True


@pytest.mark.asyncio
async def test_runaway_output_buffer_does_not_exceed_cap(
    output_store: FileTaskOutputStore,
) -> None:
    """Reader-path enforcement bounds RSS at ~max_output_bytes + one
    chunk. Pre-fix the polling guard could let `yes` grow buffer arbi-
    trarily large between ticks. Verify the buffered total stays bounded.
    """
    store = AppStateStore()
    cap = 512
    state = await spawn_local_bash(
        "yes",
        store,
        max_output_bytes=cap,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "killed"
    assert final.killed_by_size is True
    assert final.output_truncated is True
    assert final.output_bytes <= cap
    assert sum(len(chunk) for chunk in final.output_buffer) <= min(
        cap,
        DEFAULT_OUTPUT_TAIL_BYTES,
    )
    assert await output_store.size(state.id) <= cap


@pytest.mark.asyncio
async def test_explicit_kill_preserves_first_end_time(
    output_store: FileTaskOutputStore,
) -> None:
    """`LocalBashTask.kill()` stamps `end_time` synchronously when it
    flips status. The driver must NOT overwrite it when it later writes
    the terminal status."""
    from raygent_harness.core.tasks.local_bash import LocalBashTask

    store = AppStateStore()
    state = await spawn_local_bash(
        "sleep 5",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    await asyncio.sleep(0.05)

    impl = LocalBashTask()
    await impl.kill(state.id, store)

    immediate = store.tasks[state.id]
    assert isinstance(immediate, LocalBashState)
    assert immediate.status == "killed"
    kill_time = immediate.end_time
    assert kill_time is not None

    final = await run_until_done(state.id, store)
    assert final.status == "killed"
    # Driver-set end_time must NOT clobber the kill-set one.
    assert final.end_time == kill_time


@pytest.mark.asyncio
async def test_stall_watchdog_emits_signal_without_killing_process(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    # Tiny watchdog: any quiet 0.1s window after 0.05s of polling fires.
    # The bash command sleeps 1s producing no output → watchdog fires
    # before the process exits. Process keeps running and exits cleanly
    # on its own.
    watchdog = QuiescenceWatchdog(check_interval_s=0.05, threshold_s=0.1)
    state = await spawn_local_bash(
        "sleep 1",
        store,
        watchdog=watchdog,
        agent_id="a1",
        output_store=output_store,
    )

    # Wait long enough for the watchdog to fire but not for the process
    # to finish.
    await asyncio.sleep(0.3)

    # Watchdog signal should be in the queue, distinct from completion.
    mid_notifs = store.drain_notifications("a1")
    stalled = [n for n in mid_notifs if n.kind == "stalled"]
    assert len(stalled) == 1, (
        f"expected one stalled notification, got kinds={[n.kind for n in mid_notifs]}"
    )

    # Process is NOT killed by the watchdog — completes normally.
    final = await run_until_done(state.id, store)
    assert final.status == "completed"
    assert final.killed_by_timeout is False
    assert final.killed_by_size is False


@pytest.mark.asyncio
async def test_kill_shell_tasks_for_agent_filters_by_agent_id(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    s_a1_one = await spawn_local_bash(
        "sleep 5",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    s_a1_two = await spawn_local_bash(
        "sleep 5",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    s_a2 = await spawn_local_bash(
        "sleep 5",
        store,
        agent_id="a2",
        output_store=output_store,
    )
    await asyncio.sleep(0.05)

    await kill_shell_tasks_for_agent("a1", store)

    a1_one = store.tasks[s_a1_one.id]
    a1_two = store.tasks[s_a1_two.id]
    a2 = store.tasks[s_a2.id]
    assert isinstance(a1_one, LocalBashState)
    assert isinstance(a1_two, LocalBashState)
    assert isinstance(a2, LocalBashState)

    assert a1_one.status == "killed"
    assert a1_two.status == "killed"
    assert a2.status == "running"

    # Drain a2 so the conftest cleanup doesn't have to fight a watchdog
    # that's still polling into a queue.
    await kill_shell_tasks_for_agent("a2", store)


@pytest.mark.asyncio
async def test_large_success_output_keeps_only_bounded_state_tail(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    byte_count = DEFAULT_OUTPUT_TAIL_BYTES + 32_768
    command = (
        f"{shlex.quote(sys.executable)} -c "
        f"{shlex.quote(f'import sys; sys.stdout.buffer.write(b\"x\"*{byte_count})')}"
    )
    state = await spawn_local_bash(
        command,
        store,
        max_output_bytes=byte_count + 4096,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)

    assert final.status == "completed"
    assert final.output_bytes == byte_count
    assert sum(len(chunk) for chunk in final.output_buffer) == DEFAULT_OUTPUT_TAIL_BYTES

    output = await read_local_bash_output(
        state.id,
        store,
        output_store=output_store,
        max_bytes=byte_count,
    )
    assert output.bytes_total == byte_count
    assert output.content == b"x" * byte_count


@pytest.mark.asyncio
async def test_read_local_bash_output_supports_range_reads(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "printf abcdefghij",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)
    assert final.status == "completed"

    output = await read_local_bash_output(
        state.id,
        store,
        output_store=output_store,
        tail=False,
        offset=2,
        max_bytes=3,
    )
    assert output.content == b"cde"
    assert output.next_offset == 5
    assert output.truncated_after is True


@pytest.mark.asyncio
async def test_read_local_bash_output_uses_recorded_output_file_without_store(
    output_store: FileTaskOutputStore,
) -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "printf stored-output",
        store,
        agent_id="a1",
        output_store=output_store,
    )
    final = await run_until_done(state.id, store)
    assert final.status == "completed"

    output = await read_local_bash_output(state.id, store)
    assert output.content == b"stored-output"
    assert output.bytes_total == len(b"stored-output")


class _FailingAppendOutputStore:
    async def init_task_output(self, task_id: str) -> TaskOutputReference:
        return TaskOutputReference(task_id=task_id, path=f"/virtual/{task_id}.output")

    async def append_task_output(self, task_id: str, chunk: bytes) -> None:
        _ = task_id, chunk
        raise OSError("disk full")

    async def flush_task_output(self, task_id: str) -> None:
        _ = task_id

    async def evict_task_output(self, task_id: str) -> None:
        _ = task_id

    async def cleanup_task_output(self, task_id: str) -> None:
        _ = task_id

    async def read_tail(
        self,
        task_id: str,
        *,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> TaskOutputReadResult:
        _ = max_bytes
        return TaskOutputReadResult(
            task_id=task_id,
            content=b"",
            start_offset=0,
            bytes_read=0,
            bytes_total=0,
            next_offset=0,
        )

    async def read_range(
        self,
        task_id: str,
        *,
        offset: int,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> TaskOutputReadResult:
        _ = offset, max_bytes
        return await self.read_tail(task_id, max_bytes=0)

    async def size(self, task_id: str) -> int:
        _ = task_id
        return 0


@pytest.mark.asyncio
async def test_output_store_append_failure_fails_closed() -> None:
    store = AppStateStore()
    state = await spawn_local_bash(
        "echo hello",
        store,
        agent_id="a1",
        output_store=_FailingAppendOutputStore(),
    )
    final = await run_until_done(state.id, store)

    assert final.status == "failed"
    assert final.output_error == "disk full"

    notifs = store.drain_notifications("a1")
    assert len(notifs) == 1
    assert notifs[0].kind == "error"
    assert "output store error" in notifs[0].message
