from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.task import AppStateStore, TaskNotification
from raygent_harness.core.tasks.remote_agent import (
    RemoteAgentState,
    RemoteAgentTask,
    restore_remote_agents,
    spawn_remote_agent,
)
from raygent_harness.services.remote_agent import (
    JsonRemoteAgentPersistenceStore,
    RemoteAgentLaunchRequest,
    RemoteAgentLaunchResult,
    RemoteAgentPersistenceRecord,
    RemoteAgentPollRequest,
    RemoteAgentPollResult,
    RemoteAgentRestoreRequest,
    RemoteAgentRestoreResult,
    RemoteAgentStopRequest,
    safe_remote_agent_component,
)
from raygent_harness.services.task_notification_replay import (
    remote_agent_restore_replay_record,
)


class RestorableBackend:
    def __init__(
        self,
        *,
        polls: Sequence[RemoteAgentPollResult | Exception] = (),
        restores: Sequence[RemoteAgentRestoreResult | Exception] = (),
    ) -> None:
        self.launches: list[RemoteAgentLaunchRequest] = []
        self.poll_requests: list[RemoteAgentPollRequest] = []
        self.stop_requests: list[RemoteAgentStopRequest] = []
        self.restore_requests: list[RemoteAgentRestoreRequest] = []
        self.polls = list(polls)
        self.restores = list(restores)

    async def launch(
        self,
        request: RemoteAgentLaunchRequest,
    ) -> RemoteAgentLaunchResult:
        self.launches.append(request)
        return RemoteAgentLaunchResult(
            remote_id=f"remote-{request.task_id}",
            title=request.description,
            session_url=f"https://example.test/sessions/{request.task_id}",
            metadata={"launch": "yes"},
        )

    async def poll(self, request: RemoteAgentPollRequest) -> RemoteAgentPollResult:
        self.poll_requests.append(request)
        if self.polls:
            result = self.polls.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return RemoteAgentPollResult(status="running")

    async def stop(self, request: RemoteAgentStopRequest) -> None:
        self.stop_requests.append(request)

    async def restore(
        self,
        request: RemoteAgentRestoreRequest,
    ) -> RemoteAgentRestoreResult:
        self.restore_requests.append(request)
        if self.restores:
            result = self.restores.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return RemoteAgentRestoreResult(status="running")


class FailingSaveStore:
    async def save(self, record: RemoteAgentPersistenceRecord) -> None:
        _ = record
        raise OSError("save unavailable")

    async def list_records(self) -> tuple[RemoteAgentPersistenceRecord, ...]:
        return ()

    async def delete(self, task_id: str) -> None:
        _ = task_id


class BlockingSaveStore:
    def __init__(self) -> None:
        self.save_started = asyncio.Event()

    async def save(self, record: RemoteAgentPersistenceRecord) -> None:
        _ = record
        self.save_started.set()
        await asyncio.Event().wait()

    async def list_records(self) -> tuple[RemoteAgentPersistenceRecord, ...]:
        return ()

    async def delete(self, task_id: str) -> None:
        _ = task_id


class FailingDeleteStore:
    def __init__(self) -> None:
        self.records: dict[str, RemoteAgentPersistenceRecord] = {}
        self.delete_calls: list[str] = []

    async def save(self, record: RemoteAgentPersistenceRecord) -> None:
        self.records[record.task_id] = record

    async def list_records(self) -> tuple[RemoteAgentPersistenceRecord, ...]:
        return tuple(self.records.values())

    async def delete(self, task_id: str) -> None:
        self.delete_calls.append(task_id)
        raise OSError("delete unavailable")


class BlockingDeleteStore(FailingDeleteStore):
    def __init__(self) -> None:
        super().__init__()
        self.delete_started = asyncio.Event()

    async def delete(self, task_id: str) -> None:
        self.delete_calls.append(task_id)
        self.delete_started.set()
        await asyncio.Event().wait()


async def _drain_notification(
    store: AppStateStore,
    agent_id: str | None,
) -> TaskNotification:
    for _ in range(100):
        notifications = store.drain_notifications(agent_id)
        if notifications:
            return notifications[0]
        await asyncio.sleep(0.01)
    raise AssertionError("notification was not enqueued")


async def _wait_records(
    persistence: JsonRemoteAgentPersistenceStore,
    count: int,
) -> tuple[RemoteAgentPersistenceRecord, ...]:
    for _ in range(100):
        records = await persistence.list_records()
        if len(records) == count:
            return records
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} persistence records")


async def _wait_delete_call(store: FailingDeleteStore, task_id: str) -> None:
    for _ in range(100):
        if task_id in store.delete_calls:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"delete was not called for {task_id}")


async def _wait_failing_store_record(
    store: FailingDeleteStore,
    task_id: str,
) -> RemoteAgentPersistenceRecord:
    for _ in range(100):
        record = store.records.get(task_id)
        if record is not None:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"record was not saved for {task_id}")


async def _wait_event(
    sink: RecordingKernelEventSink,
    event_type: str,
    *,
    operation: str | None = None,
) -> None:
    for _ in range(100):
        for event in sink.events:
            if event.type != event_type:
                continue
            if operation is not None and event.data.get("operation") != operation:
                continue
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"event {event_type!r} was not emitted")


@pytest.mark.asyncio
async def test_remote_agent_launch_persists_identity_without_prompt(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    backend = RestorableBackend(polls=(RemoteAgentPollResult(status="running"),))
    deps = QueryDeps(
        task_store=AppStateStore(),
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    task_id = await spawn_remote_agent(
        prompt="raw prompt must not be persisted",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
        tool_use_id="toolu_remote",
        model="model-a",
        cwd="/repo",
        poll_interval_s=60.0,
    )

    records = await _wait_records(persistence, 1)
    assert len(records) == 1
    record = records[0]
    assert record.task_id == task_id
    assert record.remote_id == f"remote-{task_id}"
    assert record.description == "remote worker"
    assert record.parent_agent_id == "parent"
    assert record.tool_use_id == "toolu_remote"
    assert record.agent_type == "worker"
    assert record.model == "model-a"
    assert record.cwd == "/repo"
    assert record.metadata == {"launch": "yes"}
    assert not hasattr(record, "prompt")
    assert not hasattr(record, "final_message")


@pytest.mark.asyncio
async def test_remote_agent_persistence_save_is_nonblocking_for_launch() -> None:
    persistence = BlockingSaveStore()
    backend = RestorableBackend(polls=(RemoteAgentPollResult(status="running"),))
    store = AppStateStore()
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    task_id = await asyncio.wait_for(
        spawn_remote_agent(
            prompt="do work",
            description="remote worker",
            agent_type="worker",
            parent_agent_id="parent",
            parent_deps=deps,
            poll_interval_s=60.0,
        ),
        timeout=0.1,
    )

    await asyncio.wait_for(persistence.save_started.wait(), timeout=0.1)
    task = store.tasks[task_id]
    assert isinstance(task, RemoteAgentState)
    assert task.status == "running"


@pytest.mark.asyncio
async def test_remote_agent_persistence_save_failure_does_not_fail_launch_or_poll() -> None:
    sink = RecordingKernelEventSink()
    store = AppStateStore()
    backend = RestorableBackend(
        polls=(RemoteAgentPollResult(status="completed", message="remote done"),)
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=FailingSaveStore(),
        observability=KernelEventBus([sink]),
    )

    task_id = await spawn_remote_agent(
        prompt="do work",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
    )
    notification = await _drain_notification(store, "parent")
    task = store.tasks[task_id]

    assert isinstance(task, RemoteAgentState)
    assert task.status == "completed"
    assert "remote done" in notification.message
    await _wait_event(sink, "remote_agent.persistence.failed", operation="save")


@pytest.mark.asyncio
async def test_remote_agent_persistence_delete_failure_does_not_block_completion() -> None:
    sink = RecordingKernelEventSink()
    persistence = FailingDeleteStore()
    store = AppStateStore()
    backend = RestorableBackend(
        polls=(RemoteAgentPollResult(status="completed", message="remote done"),)
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
        observability=KernelEventBus([sink]),
    )

    task_id = await spawn_remote_agent(
        prompt="do work",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
    )
    notification = await _drain_notification(store, "parent")
    task = store.tasks[task_id]

    assert isinstance(task, RemoteAgentState)
    assert task.status == "completed"
    assert "remote done" in notification.message
    await _wait_delete_call(persistence, task_id)
    await _wait_event(sink, "remote_agent.persistence.failed", operation="delete")


@pytest.mark.asyncio
async def test_live_remote_terminal_notification_key_matches_restore_replay_key() -> None:
    persistence = FailingDeleteStore()
    store = AppStateStore()
    backend = RestorableBackend(
        polls=(RemoteAgentPollResult(status="completed", message="done"),)
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    task_id = await spawn_remote_agent(
        prompt="remote prompt",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
        tool_use_id="toolu_remote",
        poll_interval_s=0.01,
    )
    notification = await _drain_notification(store, "parent")
    record = await _wait_failing_store_record(persistence, task_id)
    replay_record = remote_agent_restore_replay_record(
        record,
        RemoteAgentRestoreResult(status="completed", message="done"),
    )

    assert replay_record is not None
    assert notification.dedupe_key is not None
    assert notification.dedupe_key == replay_record.dedupe_key


@pytest.mark.asyncio
async def test_remote_agent_persistence_delete_is_nonblocking_for_notification() -> None:
    persistence = BlockingDeleteStore()
    store = AppStateStore()
    backend = RestorableBackend(
        polls=(RemoteAgentPollResult(status="completed", message="remote done"),)
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    task_id = await spawn_remote_agent(
        prompt="do work",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
    )
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=0.1,
    )

    assert task_id in notification.message
    assert "remote done" in notification.message
    await asyncio.wait_for(persistence.delete_started.wait(), timeout=0.1)


@pytest.mark.asyncio
async def test_remote_agent_persistence_delete_failure_does_not_block_explicit_stop() -> None:
    sink = RecordingKernelEventSink()
    persistence = FailingDeleteStore()
    store = AppStateStore()
    backend = RestorableBackend(polls=(RemoteAgentPollResult(status="running"),))
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
        observability=KernelEventBus([sink]),
    )

    task_id = await spawn_remote_agent(
        prompt="do work",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
        poll_interval_s=60.0,
    )
    await RemoteAgentTask().kill(task_id, store)
    await _wait_delete_call(persistence, task_id)
    task = store.tasks[task_id]

    assert isinstance(task, RemoteAgentState)
    assert task.status == "killed"
    assert task.notified is True
    assert backend.stop_requests[0].task_id == task_id
    assert persistence.delete_calls == [task_id]
    assert store.drain_notifications("parent") == []
    await _wait_event(sink, "remote_agent.persistence.failed", operation="delete")


@pytest.mark.asyncio
async def test_remote_agent_poll_exception_keeps_running_and_preserves_sidecar(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    store = AppStateStore()
    sink = RecordingKernelEventSink()
    backend = RestorableBackend(
        polls=(
            RuntimeError("network blip"),
            RemoteAgentPollResult(status="running"),
        )
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
        observability=KernelEventBus([sink]),
    )

    task_id = await spawn_remote_agent(
        prompt="do work",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
        poll_interval_s=0.01,
    )
    await _wait_event(sink, "remote_agent.poll.failed")
    task = store.tasks[task_id]
    records = await _wait_records(persistence, 1)

    assert isinstance(task, RemoteAgentState)
    assert task.status == "running"
    assert records[0].task_id == task_id
    assert store.drain_notifications("parent") == []


@pytest.mark.asyncio
async def test_restore_remote_agents_reconstructs_running_task_and_restarts_polling(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_restore",
            remote_id="remote-r_restore",
            description="restored remote",
            parent_agent_id="parent",
            tool_use_id="toolu_restore",
            agent_type="worker",
            metadata={"cursor": "1"},
            start_time=123.0,
        )
    )
    store = AppStateStore()
    backend = RestorableBackend(
        restores=(RemoteAgentRestoreResult(status="running", metadata={"cursor": "2"}),),
        polls=(RemoteAgentPollResult(status="completed", message="restored done"),),
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    restored = await restore_remote_agents(deps=deps, poll_interval_s=0.01)
    notification = await _drain_notification(store, "parent")
    task = store.tasks["r_restore"]

    assert restored == ("r_restore",)
    assert isinstance(task, RemoteAgentState)
    assert task.status == "completed"
    assert task.prompt == ""
    assert backend.restore_requests[0].remote_id == "remote-r_restore"
    assert backend.poll_requests[0].metadata == {"cursor": "2"}
    assert notification.agent_id == "parent"
    assert notification.tool_use_id == "toolu_restore"
    assert "restored done" in notification.message
    assert await _wait_records(persistence, 0) == ()


@pytest.mark.asyncio
async def test_restore_remote_agents_keeps_record_on_recoverable_backend_failure(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_restore",
            remote_id="remote-r_restore",
            description="restored remote",
            parent_agent_id="parent",
        )
    )
    store = AppStateStore()
    backend = RestorableBackend(restores=(RuntimeError("login required"),))
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    restored = await restore_remote_agents(deps=deps)

    assert restored == ()
    assert store.tasks == {}
    assert [record.task_id for record in await persistence.list_records()] == [
        "r_restore"
    ]


@pytest.mark.asyncio
async def test_restore_remote_agents_removes_known_terminal_records(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_gone",
            remote_id="remote-r_gone",
            description="gone remote",
        )
    )
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_archived",
            remote_id="remote-r_archived",
            description="archived remote",
        )
    )
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_completed",
            remote_id="remote-r_completed",
            description="completed remote",
        )
    )
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_failed",
            remote_id="remote-r_failed",
            description="failed remote",
        )
    )
    backend = RestorableBackend(
        restores=(
            RemoteAgentRestoreResult(status="gone"),
            RemoteAgentRestoreResult(status="archived"),
            RemoteAgentRestoreResult(status="completed"),
            RemoteAgentRestoreResult(status="failed"),
        )
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    restored = await restore_remote_agents(deps=deps)

    assert restored == ()
    assert await _wait_records(persistence, 0) == ()


@pytest.mark.asyncio
async def test_restored_remote_agent_explicit_stop_suppresses_notification(
    tmp_path: Path,
) -> None:
    persistence = JsonRemoteAgentPersistenceStore(tmp_path, session_id="session-a")
    await persistence.save(
        RemoteAgentPersistenceRecord(
            task_id="r_restore",
            remote_id="remote-r_restore",
            description="restored remote",
            parent_agent_id="parent",
        )
    )
    store = AppStateStore()
    backend = RestorableBackend(
        restores=(RemoteAgentRestoreResult(status="running"),),
        polls=(RemoteAgentPollResult(status="running"),),
    )
    deps = QueryDeps(
        task_store=store,
        remote_agent_backend=backend,
        remote_agent_persistence_store=persistence,
    )

    restored = await restore_remote_agents(deps=deps, poll_interval_s=60.0)
    await RemoteAgentTask().kill("r_restore", store)
    await asyncio.sleep(0)
    task = store.tasks["r_restore"]

    assert restored == ("r_restore",)
    assert isinstance(task, RemoteAgentState)
    assert task.status == "killed"
    assert task.notified is True
    assert backend.stop_requests[0].remote_id == "remote-r_restore"
    assert store.drain_notifications("parent") == []


def test_remote_agent_persistence_store_sanitizes_session_path_components(
    tmp_path: Path,
) -> None:
    store = JsonRemoteAgentPersistenceStore(tmp_path, session_id="..")

    assert safe_remote_agent_component("..").startswith("id-")
    assert store.record_dir.is_relative_to(tmp_path.resolve())


@pytest.mark.asyncio
async def test_remote_agent_persistence_store_skips_symlink_records(
    tmp_path: Path,
) -> None:
    store = JsonRemoteAgentPersistenceStore(tmp_path / "records", session_id="s")
    store.record_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(
        (
            '{"task_id":"r_outside","remote_id":"remote-outside",'
            '"description":"outside"}\n'
        ),
        encoding="utf-8",
    )
    link = store.record_dir / "r_link.json"
    try:
        os.symlink(outside, link)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"symlink unavailable: {exc}")

    records = await store.list_records()

    assert records == ()
