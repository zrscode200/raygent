from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from raygent_harness.coordinator import (
    CoordinatorRuntime,
    JsonCoordinatorRuntimeSnapshotStore,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    RecordingKernelEventSink,
)
from raygent_harness.core.query_engine import SDKResult, SDKSystemInit
from raygent_harness.core.task import (
    AgentRouteRecord,
    AppStateStore,
    TaskNotification,
    TaskStateBase,
)
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.agent_routes import JsonAgentRouteRecordStore
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
)
from raygent_harness.services.runtime_recovery import (
    RuntimeRecoveryRequest,
    RuntimeRecoveryWarning,
    resume_runtime_session,
)
from raygent_harness.services.task_notification_replay import (
    TaskNotificationReplayRecord,
    remote_agent_terminal_dedupe_key,
)
from raygent_harness.services.task_output import FileTaskOutputStore
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
    TranscriptSearchRequest,
    TranscriptSearchScope,
    TranscriptSearchService,
    transcript_entry_to_json,
)
from raygent_harness.services.worktree import (
    WorktreeCleanupResult,
    WorktreeInfo,
)
from tests.fakes import FakeModelProvider


class _RecordingWorktreeManager:
    def __init__(self, *, changed: bool = True) -> None:
        self.changed = changed
        self.has_change_checks: list[WorktreeInfo] = []

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        raise NotImplementedError(f"unexpected create_agent_worktree: {slug} {cwd}")

    async def has_changes(self, info: WorktreeInfo) -> bool:
        self.has_change_checks.append(info)
        return self.changed

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        _ = info, keep
        return WorktreeCleanupResult(kept=True, reason="kept")


class _RestorableRemoteBackend:
    def __init__(
        self,
        *,
        restores: Sequence[RemoteAgentRestoreResult | Exception] = (),
    ) -> None:
        self.restores = list(restores)
        self.restore_requests: list[RemoteAgentRestoreRequest] = []
        self.poll_requests: list[RemoteAgentPollRequest] = []
        self.stop_requests: list[RemoteAgentStopRequest] = []

    async def launch(
        self,
        request: RemoteAgentLaunchRequest,
    ) -> RemoteAgentLaunchResult:
        raise NotImplementedError(f"unexpected launch: {request.task_id}")

    async def poll(self, request: RemoteAgentPollRequest) -> RemoteAgentPollResult:
        self.poll_requests.append(request)
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


class _FailingEnqueueTaskStore(AppStateStore):
    def enqueue_notification(self, notification: TaskNotification) -> bool:
        _ = notification
        raise RuntimeError("enqueue unavailable")


def _ctx(
    *,
    session_id: str = "session-1",
    runtime_session_id: str | None = "runtime-1",
) -> ToolUseContext:
    return ToolUseContext(
        session_id=session_id,
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="chain-1", depth=0),
        observability_context=KernelEventContext(
            session_id=session_id,
            runtime_session_id=runtime_session_id,
            source="test",
        ),
    )


def _route_record(
    worktree_path: Path,
    *,
    task_id: str = "agent-1",
    name: str = "researcher",
    route_registered_at: float = 10.0,
) -> AgentRouteRecord:
    return AgentRouteRecord(
        agent_id=task_id,
        task_id=task_id,
        task_type="local_agent",
        name=name,
        parent_agent_id=None,
        parent_session_id="session-1",
        runtime_session_id="stale-runtime",
        agent_type="worker",
        description="research worker",
        model="model-1",
        system_prompt="worker system",
        tool_names=("Read", "Write"),
        permission_mode="default",
        cwd="/repo",
        worktree_path=str(worktree_path),
        worktree_branch="raygent/agent-1",
        worktree_slug="agent-1",
        worktree_created_at=1.0,
        worktree_touched_at=2.0,
        worktree_cleanup_policy="keep",
        transcript_path=f"/repo/transcripts/session-1/subagents/{task_id}.jsonl",
        is_sidechain=True,
        content_replacement_replay=True,
        route_registered_at=route_registered_at,
    )


async def _write_replay_transcript(
    transcript_store: JsonlTranscriptStore,
    *,
    session_id: str = "session-1",
    runtime_session_id: str | None = "runtime-1",
) -> None:
    scope = TranscriptScope(
        session_id=session_id,
        runtime_session_id=runtime_session_id,
    )
    await transcript_store.append_many(
        scope,
        (
            TranscriptMessageEntry(
                entry_id="m1",
                session_id=session_id,
                runtime_session_id=runtime_session_id,
                message={"role": "user", "content": "before"},
            ),
            TranscriptMessageEntry(
                entry_id="m2",
                parent_entry_id="m1",
                session_id=session_id,
                runtime_session_id=runtime_session_id,
                message={"role": "assistant", "content": "answer before"},
            ),
        ),
    )


async def _wait_remote_records(
    store: JsonRemoteAgentPersistenceStore,
    count: int,
) -> tuple[RemoteAgentPersistenceRecord, ...]:
    for _ in range(100):
        records = await store.list_records()
        if len(records) == count:
            return records
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} remote-agent records")


@pytest.mark.asyncio
async def test_resume_runtime_session_restores_state_and_rebinds_next_turn(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)

    coordinator = CoordinatorRuntime(
        event_context=KernelEventContext(
            session_id="session-1",
            runtime_session_id="runtime-1",
            source="coordinator",
        )
    )
    coordinator.add_work_item(kind="research", title="recover", status="running")
    snapshot_store = JsonCoordinatorRuntimeSnapshotStore(tmp_path / "snapshots")
    await snapshot_store.save(coordinator.snapshot())

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    route_store = JsonAgentRouteRecordStore(tmp_path / "routes")
    await route_store.save(_route_record(worktree_path))

    task_output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="session-1")
    output_ref = await task_output_store.init_task_output("bash-1")
    await task_output_store.append_task_output("bash-1", b"abc")

    remote_store = JsonRemoteAgentPersistenceStore(
        tmp_path / "remote",
        session_id="session-1",
    )
    await remote_store.save(
        RemoteAgentPersistenceRecord(
            task_id="remote-1",
            remote_id="remote-id-1",
            description="remote worker",
            parent_agent_id=None,
            metadata={"cursor": "1"},
            start_time=10.0,
        )
    )
    remote_backend = _RestorableRemoteBackend(
        restores=(RemoteAgentRestoreResult(status="running", metadata={"cursor": "2"}),)
    )

    task_store = AppStateStore()
    task_store.register_task(
        TaskStateBase(
            id="bash-1",
            type="local_bash",
            description="bash task",
            status="running",
            start_time=1.0,
            output_file=output_ref.path,
        )
    )
    worktree_manager = _RecordingWorktreeManager(changed=True)
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=task_store,
        transcript_store=transcript_store,
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "after resume"},)
        ),
        worktree_manager=worktree_manager,
        remote_agent_backend=remote_backend,
        remote_agent_persistence_store=remote_store,
        agent_route_record_store=route_store,
        observability=KernelEventBus((sink,)),
    )
    config = QueryConfig(model="model-1", session_id="session-1")
    offline_notification = TaskNotificationReplayRecord(
        task_id="agent-1",
        message=(
            "<task_notification><task_id>agent-1</task_id>"
            "<status>completed</status><result>offline route result</result>"
            "</task_notification>"
        ),
        kind="completed",
        dedupe_key="local_agent_resume:agent-1:offline-result",
        source="local_agent_resume",
        created_at=50.0,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=config,
            deps=deps,
            ctx=_ctx(),
            coordinator_snapshot_store=snapshot_store,
            task_output_store=task_output_store,
            remote_poll_interval_s=60.0,
            offline_task_notifications=(offline_notification,),
        )
    )

    assert result.transcript_scope == TranscriptScope(
        session_id="session-1",
        runtime_session_id="runtime-1",
    )
    assert result.last_message_entry_id == "m2"
    assert result.coordinator_runtime_restored is True
    assert deps.coordinator_runtime is not None
    assert task_store.agent_name_registry["researcher"] == "agent-1"
    assert task_store.agent_route_records["agent-1"].runtime_session_id == "runtime-1"
    assert result.restored_route_records[0].runtime_session_id == "runtime-1"
    assert result.restored_remote_task_ids == ("remote-1",)
    assert remote_backend.restore_requests[0].remote_id == "remote-id-1"
    if remote_backend.poll_requests:
        assert remote_backend.poll_requests[0].metadata == {"cursor": "2"}

    assert result.worktree_statuses[0].exists is True
    assert result.worktree_statuses[0].has_changes is True
    assert worktree_manager.has_change_checks[0].cleanup_policy == "keep"
    bash_output = next(
        status for status in result.task_output_statuses if status.task_id == "bash-1"
    )
    assert bash_output.output_file == output_ref.path
    assert bash_output.available is True
    assert bash_output.byte_count == 3
    assert result.task_notification_replay.enqueued_count == 1
    recovered_notification = task_store.drain_notifications(None)[0]
    assert recovered_notification.task_id == "agent-1"
    assert recovered_notification.dedupe_key == "local_agent_resume:agent-1:offline-result"
    assert "offline route result" in recovered_notification.message

    search_result = await TranscriptSearchService(transcript_store).search(
        TranscriptSearchRequest(
            query="answer before",
            scope=TranscriptSearchScope(
                session_id=result.transcript_scope.session_id,
                runtime_session_id=result.transcript_scope.runtime_session_id,
            ),
            max_results=1,
        )
    )
    assert [match.entry_id for match in search_result.matches] == ["m2"]
    assert "answer before" in search_result.matches[0].snippet

    engine = result.build_query_engine(
        QueryConfig(model="model-1", session_id="fresh-session"),
        deps,
        _ctx(session_id="fresh-session", runtime_session_id=None),
    )
    events = [event async for event in engine.submit_message("again")]

    assert isinstance(events[0], SDKSystemInit)
    assert events[0].session_id == "session-1"
    assert isinstance(events[-1], SDKResult)
    entries = await transcript_store.read_entries(
        TranscriptScope(session_id="session-1", runtime_session_id="runtime-1")
    )
    appended_user = next(
        entry
        for entry in entries
        if isinstance(entry, TranscriptMessageEntry)
        and entry.message == {"role": "user", "content": "again"}
    )
    assert appended_user.parent_entry_id == "m2"
    assert appended_user.session_id == "session-1"
    assert appended_user.runtime_session_id == "runtime-1"

    completed_events = [
        event for event in sink.events if event.type == "runtime_recovery.completed"
    ]
    assert len(completed_events) == 1
    assert completed_events[0].data["message_count"] == 2
    assert completed_events[0].data["restored_route_count"] == 1
    assert completed_events[0].data["restored_remote_task_count"] == 1
    assert completed_events[0].data["offline_task_notification_count"] == 1
    assert completed_events[0].data["replayed_task_notification_count"] == 1
    assert "before" not in str(completed_events[0].data)


@pytest.mark.asyncio
async def test_resume_runtime_session_skips_replay_already_processed_by_coordinator(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    replay_record = TaskNotificationReplayRecord(
        task_id="remote-processed",
        message=(
            "<task_notification><task_id>remote-processed</task_id>"
            "<status>completed</status></task_notification>"
        ),
        kind="completed",
        dedupe_key="remote_agent_restore:remote-processed:stable",
        created_at=100.0,
    )
    coordinator = CoordinatorRuntime(
        event_context=KernelEventContext(
            session_id="session-1",
            runtime_session_id="runtime-1",
            source="coordinator",
        )
    )
    coordinator.record_task_notifications([replay_record.to_notification()])
    snapshot_store = JsonCoordinatorRuntimeSnapshotStore(tmp_path / "snapshots")
    await snapshot_store.save(coordinator.snapshot())
    task_store = AppStateStore()
    deps = QueryDeps(task_store=task_store, transcript_store=transcript_store)

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            coordinator_snapshot_store=snapshot_store,
            offline_task_notifications=(replay_record,),
            restore_remote_agents_enabled=False,
        )
    )

    assert result.coordinator_runtime_restored is True
    assert result.task_notification_replay.enqueued_count == 0
    assert (
        result.task_notification_replay.skipped_coordinator_processed_count == 1
    )
    assert task_store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_resume_runtime_session_replays_terminal_remote_restore_fact(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    remote_store = JsonRemoteAgentPersistenceStore(
        tmp_path / "remote",
        session_id="session-1",
    )
    await remote_store.save(
        RemoteAgentPersistenceRecord(
            task_id="remote-done",
            remote_id="remote-id-done",
            description="offline remote",
            parent_agent_id="parent",
            tool_use_id="toolu_remote",
            start_time=10.0,
            updated_at=20.0,
        )
    )
    remote_backend = _RestorableRemoteBackend(
        restores=(
            RemoteAgentRestoreResult(
                status="completed",
                message="offline completion",
            ),
        )
    )
    task_store = AppStateStore()
    deps = QueryDeps(
        task_store=task_store,
        transcript_store=transcript_store,
        remote_agent_backend=remote_backend,
        remote_agent_persistence_store=remote_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            restore_remote_agents_enabled=True,
        )
    )

    assert result.restored_remote_task_ids == ()
    assert result.task_notification_replay.enqueued_count == 1
    notification = task_store.drain_notifications("parent")[0]
    assert notification.task_id == "remote-done"
    assert notification.tool_use_id == "toolu_remote"
    assert notification.dedupe_key is not None
    assert notification.dedupe_key.startswith("remote_agent_terminal:")
    assert "offline completion" in notification.message
    assert await _wait_remote_records(remote_store, 0) == ()


@pytest.mark.asyncio
async def test_resume_runtime_session_suppresses_remote_fact_seen_live_before_restart(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    live_notification = TaskNotificationReplayRecord(
        task_id="remote-done",
        message=(
            "<task_notification><task_id>remote-done</task_id>"
            "<status>completed</status><result>live completion</result>"
            "</task_notification>"
        ),
        kind="completed",
        dedupe_key=remote_agent_terminal_dedupe_key(
            task_id="remote-done",
            remote_id="remote-id-done",
            final_status="completed",
        ),
        created_at=10.0,
    ).to_notification()
    coordinator = CoordinatorRuntime(
        event_context=KernelEventContext(
            session_id="session-1",
            runtime_session_id="runtime-1",
            source="coordinator",
        )
    )
    coordinator.record_task_notifications([live_notification])
    snapshot_store = JsonCoordinatorRuntimeSnapshotStore(tmp_path / "snapshots")
    await snapshot_store.save(coordinator.snapshot())
    remote_store = JsonRemoteAgentPersistenceStore(
        tmp_path / "remote",
        session_id="session-1",
    )
    await remote_store.save(
        RemoteAgentPersistenceRecord(
            task_id="remote-done",
            remote_id="remote-id-done",
            description="offline remote",
            parent_agent_id="parent",
            start_time=10.0,
            updated_at=20.0,
        )
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=transcript_store,
        remote_agent_backend=_RestorableRemoteBackend(
            restores=(
                RemoteAgentRestoreResult(
                    status="completed",
                    message="restored completion",
                ),
            )
        ),
        remote_agent_persistence_store=remote_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            coordinator_snapshot_store=snapshot_store,
        )
    )

    assert result.task_notification_replay.enqueued_count == 0
    assert (
        result.task_notification_replay.skipped_coordinator_processed_count == 1
    )
    assert deps.task_store.drain_notifications("parent") == []
    assert await _wait_remote_records(remote_store, 0) == ()


@pytest.mark.asyncio
async def test_resume_runtime_session_preserves_remote_sidecar_when_replay_fails(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    remote_store = JsonRemoteAgentPersistenceStore(
        tmp_path / "remote",
        session_id="session-1",
    )
    await remote_store.save(
        RemoteAgentPersistenceRecord(
            task_id="remote-failed-replay",
            remote_id="remote-id-failed-replay",
            description="offline remote",
            parent_agent_id="parent",
            start_time=10.0,
            updated_at=20.0,
        )
    )
    deps = QueryDeps(
        task_store=_FailingEnqueueTaskStore(),
        transcript_store=transcript_store,
        remote_agent_backend=_RestorableRemoteBackend(
            restores=(
                RemoteAgentRestoreResult(
                    status="completed",
                    message="offline completion",
                ),
            )
        ),
        remote_agent_persistence_store=remote_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
        )
    )

    assert result.task_notification_replay.enqueued_count == 0
    assert result.task_notification_replay.warning_count == 1
    assert RuntimeRecoveryWarning(
        source="task_notification",
        reason="offline_notification_replay_failed",
        error_type="RuntimeError",
    ) in result.warnings
    assert [record.task_id for record in await _wait_remote_records(remote_store, 1)] == [
        "remote-failed-replay"
    ]


@pytest.mark.asyncio
async def test_resume_runtime_session_remote_archived_and_gone_are_cleanup_only(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    remote_store = JsonRemoteAgentPersistenceStore(
        tmp_path / "remote",
        session_id="session-1",
    )
    for task_id, status in (
        ("remote-archived", "archived"),
        ("remote-gone", "gone"),
    ):
        await remote_store.save(
            RemoteAgentPersistenceRecord(
                task_id=task_id,
                remote_id=f"remote-id-{status}",
                description=f"{status} remote",
            )
        )
    task_store = AppStateStore()
    deps = QueryDeps(
        task_store=task_store,
        transcript_store=transcript_store,
        remote_agent_backend=_RestorableRemoteBackend(
            restores=(
                RemoteAgentRestoreResult(status="archived"),
                RemoteAgentRestoreResult(status="gone"),
            )
        ),
        remote_agent_persistence_store=remote_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
        )
    )

    assert result.task_notification_replay.enqueued_count == 0
    assert task_store.drain_notifications(None) == []
    assert await _wait_remote_records(remote_store, 0) == ()


@pytest.mark.asyncio
async def test_resume_runtime_session_missing_optional_stores_is_fail_soft(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=transcript_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            restore_remote_agents_enabled=False,
        )
    )

    assert result.replay.messages == [
        {"role": "user", "content": "before"},
        {"role": "assistant", "content": "answer before"},
    ]
    assert result.coordinator_runtime_restored is False
    assert result.restored_route_records == ()
    assert result.restored_remote_task_ids == ()
    assert result.warnings == ()


@pytest.mark.asyncio
async def test_resume_runtime_session_surfaces_corrupt_route_warnings(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    route_store = JsonAgentRouteRecordStore(tmp_path / "routes")
    record_dir = route_store.record_dir("session-1")
    record_dir.mkdir(parents=True)
    (record_dir / "bad.json").write_text("{bad", encoding="utf-8")
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=transcript_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            agent_route_record_store=route_store,
            restore_remote_agents_enabled=False,
        )
    )

    assert result.restored_route_records == ()
    assert RuntimeRecoveryWarning(
        source="route_record",
        reason="skipped route record bad.json: JSONDecodeError",
    ) in result.warnings


@pytest.mark.asyncio
async def test_resume_runtime_session_restores_duplicate_names_latest_wins(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    await _write_replay_transcript(transcript_store)
    route_store = JsonAgentRouteRecordStore(tmp_path / "routes")
    missing_worktree = tmp_path / "missing-worktree"
    await route_store.save(
        _route_record(
            missing_worktree,
            task_id="agent-z-old",
            name="researcher",
            route_registered_at=1.0,
        )
    )
    await route_store.save(
        _route_record(
            missing_worktree,
            task_id="agent-a-new",
            name="researcher",
            route_registered_at=2.0,
        )
    )
    task_store = AppStateStore()
    deps = QueryDeps(
        task_store=task_store,
        transcript_store=transcript_store,
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="session-1"),
            deps=deps,
            ctx=_ctx(),
            agent_route_record_store=route_store,
            restore_remote_agents_enabled=False,
        )
    )

    assert [record.task_id for record in result.restored_route_records] == [
        "agent-z-old",
        "agent-a-new",
    ]
    assert task_store.agent_name_registry["researcher"] == "agent-a-new"


@pytest.mark.asyncio
async def test_resume_runtime_session_can_load_from_explicit_jsonl_path(
    tmp_path: Path,
) -> None:
    explicit_path = tmp_path / "outside-session.jsonl"
    explicit_path.write_text(
        "\n".join(
            (
                transcript_entry_to_json(
                    TranscriptMessageEntry(
                        entry_id="m1",
                        session_id="embedded-session",
                        runtime_session_id="embedded-runtime",
                        message={"role": "user", "content": "from explicit path"},
                    )
                ),
                "{bad",
            )
        ),
        encoding="utf-8",
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=JsonlTranscriptStore(tmp_path / "append-target"),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "explicit done"},)
        ),
    )

    result = await resume_runtime_session(
        RuntimeRecoveryRequest(
            config=QueryConfig(model="model-1", session_id="wrong-session"),
            deps=deps,
            ctx=_ctx(session_id="wrong-session", runtime_session_id=None),
            transcript_path=str(explicit_path),
            restore_remote_agents_enabled=False,
        )
    )

    assert result.transcript_path == str(explicit_path)
    assert result.transcript_scope == TranscriptScope(
        session_id="embedded-session",
        runtime_session_id="embedded-runtime",
    )
    assert result.replay.messages == [
        {"role": "user", "content": "from explicit path"}
    ]
    assert RuntimeRecoveryWarning(
        source="transcript",
        reason="skipped transcript line 2: TranscriptDecodeError",
    ) in result.warnings

    engine = result.build_query_engine(
        QueryConfig(model="model-1", session_id="still-wrong"),
        deps,
        _ctx(session_id="still-wrong", runtime_session_id=None),
    )
    events = [event async for event in engine.submit_message("continue explicit")]

    assert isinstance(events[-1], SDKResult)
    assert "continue explicit" in explicit_path.read_text(encoding="utf-8")
