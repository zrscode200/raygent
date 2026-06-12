from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from raygent_harness.coordinator import (
    CoordinatorRuntime,
    CoordinatorRuntimeConfig,
    CoordinatorRuntimeSnapshot,
    JsonCoordinatorRuntimeSnapshotStore,
)
from raygent_harness.coordinator.runtime import (
    CoordinatorBlackboardEntry,
    CoordinatorWorkItem,
    coordinator_runtime_snapshot_to_dict,
)
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    RecordingKernelEventSink,
)
from raygent_harness.core.task import TaskNotification


def _clock() -> Callable[[], float]:
    now = 0.0

    def tick() -> float:
        nonlocal now
        now += 1.0
        return now

    return tick


def _session_context() -> KernelEventContext:
    return KernelEventContext(
        session_id="session-1",
        runtime_session_id="runtime-1",
        source="coordinator",
    )


def _notification() -> TaskNotification:
    return TaskNotification(
        task_id="task-1",
        message=(
            "<task-notification><task-id>task-1</task-id>"
            "<status>completed</status><result>SECRET RESULT</result>"
            "</task-notification>"
        ),
        kind="completed",
        tool_use_id="tool-1",
        created_at=10.0,
    )


@pytest.mark.asyncio
async def test_snapshot_store_round_trips_and_restores_rendered_digest(
    tmp_path: Path,
) -> None:
    runtime = CoordinatorRuntime(
        event_context=_session_context(),
        clock=_clock(),
    )
    runtime.add_work_item(kind="research", title="Collect anchors", status="running")
    runtime.add_blackboard_entry(
        kind="decision",
        content="Use injected coordinator snapshot store.",
        source="coordinator",
    )
    runtime.record_task_notifications([_notification()])
    before = runtime.render_context()
    assert before is not None

    store = JsonCoordinatorRuntimeSnapshotStore(base_dir=tmp_path)
    await store.save(runtime.snapshot())
    loaded = await store.load("session-1", runtime_session_id="runtime-1")

    assert loaded.warnings == ()
    assert loaded.snapshot is not None
    assert loaded.snapshot.session_id == "session-1"
    assert loaded.snapshot.runtime_session_id == "runtime-1"
    assert loaded.snapshot.processed_notification_keys

    restored = CoordinatorRuntime.from_snapshot(loaded.snapshot, clock=_clock())
    after = restored.render_context()

    assert after is not None
    assert after["content"] == before["content"]
    assert after.get("raygentCoordinatorRuntime") == before.get(
        "raygentCoordinatorRuntime"
    )


def test_snapshot_restore_preserves_notification_dedupe_keys() -> None:
    runtime = CoordinatorRuntime(event_context=_session_context(), clock=_clock())
    notification = _notification()
    first = runtime.record_task_notifications([notification])
    snapshot = runtime.snapshot()

    restored = CoordinatorRuntime.from_snapshot(snapshot, clock=_clock())
    second = restored.record_task_notifications([notification])

    assert first.processed_count == 1
    assert second.processed_count == 0
    assert second.skipped_duplicate_count == 1
    assert len(restored.snapshot().work_items) == 1
    assert len(restored.snapshot().blackboard_entries) == 1


def test_snapshot_restore_preserves_stable_replay_notification_keys() -> None:
    runtime = CoordinatorRuntime(event_context=_session_context(), clock=_clock())
    notification = TaskNotification(
        task_id="remote-1",
        message="offline result",
        kind="completed",
        created_at=10.0,
        dedupe_key="remote_agent_restore:remote-1:stable",
    )
    same_fact_new_timestamp = TaskNotification(
        task_id="remote-1",
        message="offline result rediscovered",
        kind="completed",
        created_at=20.0,
        dedupe_key="remote_agent_restore:remote-1:stable",
    )
    first = runtime.record_task_notifications([notification])

    restored = CoordinatorRuntime.from_snapshot(runtime.snapshot(), clock=_clock())
    second = restored.record_task_notifications([same_fact_new_timestamp])

    assert first.processed_count == 1
    assert restored.has_processed_task_notification(same_fact_new_timestamp) is True
    assert second.processed_count == 0
    assert second.skipped_duplicate_count == 1


def test_snapshot_restore_preserves_counters_and_mutation_indexes() -> None:
    runtime = CoordinatorRuntime(event_context=_session_context(), clock=_clock())
    first_work = runtime.add_work_item(
        kind="research",
        title="numeric id",
        status="planned",
    )
    first_entry = runtime.add_blackboard_entry(
        kind="fact",
        content="numeric entry",
        source="coordinator",
    )
    launch = runtime.record_agent_launch(
        agent_id="agent-1",
        task_id="agent-1",
        agent_type="worker",
        description="worker",
        prompt_chars=12,
    )

    restored = CoordinatorRuntime.from_snapshot(runtime.snapshot(), clock=_clock())
    next_work = restored.add_work_item(
        kind="verification",
        title="post restore",
        status="running",
    )
    next_entry = restored.add_blackboard_entry(
        kind="risk",
        content="post restore entry",
        source="coordinator",
    )
    routed = restored.record_send_message(
        sender="team-lead",
        target="@agent-1",
        summary="follow-up",
        message_chars=20,
        recipient_task_ids=("agent-1",),
        recipient_agent_ids=("agent-1",),
    )
    stopped = restored.record_task_stop(task_id="agent-1", task_type="local_agent")
    completed = restored.record_task_notifications(
        [
            TaskNotification(
                task_id="agent-1",
                message="<status>completed</status>",
                kind="completed",
                created_at=20.0,
            )
        ]
    )
    snapshot = restored.snapshot()
    agent_items = [item for item in snapshot.work_items if item.task_id == "agent-1"]

    assert first_work.id == "cw_000001"
    assert first_entry.id == "cb_000001"
    assert next_work.id == "cw_000002"
    assert next_entry.id == "cb_000002"
    assert routed.work_items[0].id == launch.work_item.id
    assert stopped.work_item.id == launch.work_item.id
    assert completed.work_item_ids == (launch.work_item.id,)
    assert len(agent_items) == 1
    assert agent_items[0].status == "completed"


def test_snapshot_restore_applies_storage_caps_after_load() -> None:
    snapshot = CoordinatorRuntimeSnapshot(
        config=CoordinatorRuntimeConfig(
            max_stored_work_items=1,
            max_stored_blackboard_entries=1,
        ),
        work_items=(
            CoordinatorWorkItem(
                id="cw_000001",
                kind="research",
                title="low",
                status="completed",
                priority=0,
                created_at=1.0,
                updated_at=1.0,
            ),
            CoordinatorWorkItem(
                id="cw_000002",
                kind="verification",
                title="high",
                status="running",
                priority=5,
                created_at=2.0,
                updated_at=2.0,
            ),
        ),
        blackboard_entries=(
            CoordinatorBlackboardEntry(
                id="cb_000001",
                kind="fact",
                content="low",
                source="coordinator",
                priority=0,
                created_at=1.0,
            ),
            CoordinatorBlackboardEntry(
                id="cb_000002",
                kind="decision",
                content="high",
                source="coordinator",
                priority=5,
                created_at=2.0,
            ),
        ),
    )

    restored = CoordinatorRuntime.from_snapshot(snapshot)

    restored_snapshot = restored.snapshot()
    assert [item.id for item in restored_snapshot.work_items] == ["cw_000002"]
    assert [entry.id for entry in restored_snapshot.blackboard_entries] == ["cb_000002"]


@pytest.mark.asyncio
async def test_snapshot_store_load_is_fail_soft_for_corrupt_json(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    store = JsonCoordinatorRuntimeSnapshotStore(
        base_dir=tmp_path,
        observability=KernelEventBus([sink]),
    )
    path = store.path_for("session-1")
    path.parent.mkdir(parents=True)
    path.write_text("{not json SECRET RAW", encoding="utf-8")

    result = await store.load("session-1")

    assert result.snapshot is None
    assert result.warnings == (
        "coordinator snapshot load failed: CoordinatorRuntimeSnapshotDecodeError",
    )
    assert sink.event_types == ("coordinator.snapshot.load_failed",)
    assert "SECRET RAW" not in str(dict(sink.events[0].data))


@pytest.mark.asyncio
async def test_snapshot_store_load_is_fail_soft_for_unknown_schema_version(
    tmp_path: Path,
) -> None:
    store = JsonCoordinatorRuntimeSnapshotStore(base_dir=tmp_path)
    path = store.path_for("session-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 999, "work_items": [], "blackboard_entries": []}),
        encoding="utf-8",
    )

    result = await store.load("session-1")

    assert result.snapshot is None
    assert result.warnings == (
        "coordinator snapshot load failed: CoordinatorRuntimeSnapshotDecodeError",
    )


@pytest.mark.asyncio
async def test_snapshot_store_rejects_count_only_notification_state(
    tmp_path: Path,
) -> None:
    store = JsonCoordinatorRuntimeSnapshotStore(base_dir=tmp_path)
    path = store.path_for("session-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-1",
                "runtime_session_id": None,
                "config": {},
                "work_items": [],
                "blackboard_entries": [],
                "processed_notification_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = await store.load("session-1")

    assert result.snapshot is None
    assert result.warnings == (
        "coordinator snapshot load failed: CoordinatorRuntimeSnapshotDecodeError",
    )


@pytest.mark.asyncio
async def test_snapshot_store_rejects_mismatched_snapshot_identity(
    tmp_path: Path,
) -> None:
    runtime = CoordinatorRuntime(
        event_context=KernelEventContext(
            session_id="actual-session",
            runtime_session_id="actual-runtime",
            source="coordinator",
        ),
        clock=_clock(),
    )
    store = JsonCoordinatorRuntimeSnapshotStore(base_dir=tmp_path)
    path = store.path_for("requested-session", runtime_session_id="requested-runtime")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(coordinator_runtime_snapshot_to_dict(runtime.snapshot())),
        encoding="utf-8",
    )

    result = await store.load(
        "requested-session",
        runtime_session_id="requested-runtime",
    )

    assert result.snapshot is None
    assert result.warnings == (
        "coordinator snapshot load failed: CoordinatorRuntimeSnapshotDecodeError",
    )


@pytest.mark.asyncio
async def test_snapshot_events_are_metadata_only_without_raw_content(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    bus = KernelEventBus([sink])
    runtime = CoordinatorRuntime(
        event_context=_session_context(),
        observability=bus,
        clock=_clock(),
    )
    runtime.add_work_item(
        kind="research",
        title="SECRET TITLE",
        status="running",
        result_summary="SECRET SUMMARY",
    )
    runtime.add_blackboard_entry(
        kind="fact",
        content="SECRET BLACKBOARD",
        source="coordinator",
    )
    snapshot = runtime.snapshot()
    sink.clear()

    store = JsonCoordinatorRuntimeSnapshotStore(
        base_dir=tmp_path,
        observability=bus,
        event_context=_session_context(),
    )
    await store.save(snapshot)
    loaded = await store.load("session-1", runtime_session_id="runtime-1")
    assert loaded.snapshot is not None
    CoordinatorRuntime.from_snapshot(
        loaded.snapshot,
        observability=bus,
        event_context=_session_context(),
    )

    assert sink.event_types == (
        "coordinator.snapshot.saved",
        "coordinator.snapshot.loaded",
        "coordinator.snapshot.restored",
    )
    for event in sink.events:
        assert event.content_policy == "metadata_only"
        payload = str(dict(event.data))
        assert "SECRET TITLE" not in payload
        assert "SECRET SUMMARY" not in payload
        assert "SECRET BLACKBOARD" not in payload
