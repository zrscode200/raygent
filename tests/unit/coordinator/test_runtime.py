from __future__ import annotations

from collections.abc import Callable

from raygent_harness.coordinator.runtime import (
    CoordinatorRuntime,
    CoordinatorRuntimeConfig,
)
from raygent_harness.core.messages import MessageParam, RaygentCoordinatorRuntimeMetadata
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.task import TaskNotification


def _clock() -> Callable[[], float]:
    now = 0.0

    def tick() -> float:
        nonlocal now
        now += 1.0
        return now

    return tick


def _runtime_metadata(message: MessageParam) -> RaygentCoordinatorRuntimeMetadata:
    metadata = message.get("raygentCoordinatorRuntime")
    assert metadata is not None
    return metadata


def test_runtime_records_work_and_blackboard_with_stable_rendering() -> None:
    runtime = CoordinatorRuntime(clock=_clock())
    low = runtime.add_work_item(
        kind="research",
        title="Collect reference anchors",
        status="planned",
        priority=0,
    )
    high = runtime.add_work_item(
        kind="verification",
        title="Verify implementation",
        status="running",
        priority=3,
        task_id="a_high",
    )
    blocked = runtime.add_work_item(
        kind="implementation",
        title="Patch runtime",
        status="blocked",
        priority=3,
    )
    entry = runtime.add_blackboard_entry(
        kind="decision",
        content="Coordinator runtime is service-only in wave 1.",
        source="coordinator",
        work_item_id=blocked.id,
        priority=2,
    )

    message = runtime.render_context()

    assert message is not None
    content = str(message["content"])
    metadata = _runtime_metadata(message)
    assert message.get("raygentMessageKind") == "coordinator_runtime"
    assert metadata["work_item_count"] == 3
    assert metadata["blackboard_entry_count"] == 1
    assert metadata["work_item_ids"] == [
        blocked.id,
        high.id,
        low.id,
    ]
    assert content.index(f"id={blocked.id}") < content.index(f"id={high.id}")
    assert content.index(f"id={high.id}") < content.index(f"id={low.id}")
    assert f"id={entry.id}" in content
    assert "<coordinator_runtime>" in content
    assert "</coordinator_runtime>" in content


def test_rendering_enforces_count_entry_and_total_caps() -> None:
    runtime = CoordinatorRuntime(
        config=CoordinatorRuntimeConfig(
            max_rendered_work_items=1,
            max_rendered_blackboard_entries=1,
            max_total_rendered_chars=420,
            max_entry_chars=80,
        ),
        clock=_clock(),
    )
    runtime.add_work_item(
        kind="research",
        title="A" * 500,
        status="running",
        priority=1,
    )
    runtime.add_work_item(
        kind="verification",
        title="second item should be count-capped",
        status="running",
        priority=0,
    )
    runtime.add_blackboard_entry(
        kind="fact",
        content="B" * 500,
        source="agent",
        priority=1,
    )
    runtime.add_blackboard_entry(
        kind="risk",
        content="second entry should be count-capped",
        source="agent",
        priority=0,
    )

    message = runtime.render_context()

    assert message is not None
    metadata = _runtime_metadata(message)
    content = str(message["content"])
    assert metadata["truncated"] is True
    assert metadata["dropped_work_item_count"] == 1
    assert metadata["dropped_blackboard_entry_count"] == 1
    assert metadata["rendered_char_count"] <= 420
    assert "...[truncated" in content
    assert "omitted by render cap" in content


def test_tiny_render_caps_still_bound_output_length() -> None:
    runtime = CoordinatorRuntime(
        config=CoordinatorRuntimeConfig(
            max_total_rendered_chars=60,
            max_entry_chars=8,
        ),
        clock=_clock(),
    )
    runtime.add_blackboard_entry(
        kind="fact",
        content="C" * 500,
        source="agent",
    )

    message = runtime.render_context()

    assert message is not None
    metadata = _runtime_metadata(message)
    content = str(message["content"])
    assert len(content) <= 60
    assert content.startswith("<coordinator_runtime>")
    assert content.endswith("</coordinator_runtime>")
    assert metadata["rendered_char_count"] <= 60
    assert metadata["truncated"] is True


def test_total_cap_reports_only_entries_actually_rendered() -> None:
    runtime = CoordinatorRuntime(
        config=CoordinatorRuntimeConfig(
            max_total_rendered_chars=180,
            max_entry_chars=80,
        ),
        clock=_clock(),
    )
    work = runtime.add_work_item(
        kind="verification",
        title="short work",
        status="running",
    )
    entry = runtime.add_blackboard_entry(
        kind="fact",
        content="blackboard entry should not fit once work section is rendered",
        source="agent",
    )

    message = runtime.render_context()

    assert message is not None
    metadata = _runtime_metadata(message)
    content = str(message["content"])
    assert f"id={work.id}" in content
    assert f"id={entry.id}" not in content
    assert metadata["work_item_ids"] == [work.id]
    assert metadata["blackboard_entry_ids"] == []
    assert metadata["rendered_blackboard_entry_count"] == 0
    assert metadata["dropped_blackboard_entry_count"] == 1
    assert metadata["truncated"] is True
    assert content.endswith("</coordinator_runtime>")


def test_storage_caps_evict_low_priority_state_deterministically() -> None:
    runtime = CoordinatorRuntime(
        config=CoordinatorRuntimeConfig(
            max_stored_work_items=2,
            max_stored_blackboard_entries=2,
        ),
        clock=_clock(),
    )
    low = runtime.add_work_item(
        kind="research",
        title="low priority terminal",
        status="completed",
        priority=0,
    )
    keep_running = runtime.add_work_item(
        kind="implementation",
        title="running survives",
        status="running",
        priority=0,
    )
    keep_high = runtime.add_work_item(
        kind="verification",
        title="high priority survives",
        status="completed",
        priority=5,
    )
    low_entry = runtime.add_blackboard_entry(
        kind="fact",
        content="low",
        source="coordinator",
        priority=0,
    )
    keep_entry = runtime.add_blackboard_entry(
        kind="decision",
        content="keep",
        source="coordinator",
        priority=1,
    )
    latest_entry = runtime.add_blackboard_entry(
        kind="risk",
        content="latest",
        source="coordinator",
        priority=0,
    )

    snapshot = runtime.snapshot()

    assert {item.id for item in snapshot.work_items} == {keep_running.id, keep_high.id}
    assert low.id not in {item.id for item in snapshot.work_items}
    assert {entry.id for entry in snapshot.blackboard_entries} == {
        keep_entry.id,
        latest_entry.id,
    }
    assert low_entry.id not in {entry.id for entry in snapshot.blackboard_entries}


def test_task_notification_ingestion_is_idempotent_and_redacts_result_by_default() -> None:
    runtime = CoordinatorRuntime(clock=_clock())
    notification = TaskNotification(
        task_id="a_secret",
        message=(
            "<task_notification><task_id>a_secret</task_id>"
            "<status>completed</status><result>SECRET RAW RESULT</result>"
            "</task_notification>"
        ),
        kind="completed",
        tool_use_id="tool_1",
        created_at=10.0,
    )

    first = runtime.record_task_notifications([notification])
    second = runtime.record_task_notifications([notification])
    message = runtime.render_context()

    assert first.processed_count == 1
    assert first.skipped_duplicate_count == 0
    assert second.processed_count == 0
    assert second.skipped_duplicate_count == 1
    assert len(runtime.snapshot().work_items) == 1
    assert len(runtime.snapshot().blackboard_entries) == 1
    assert message is not None
    content = str(message["content"])
    assert "a_secret" in content
    assert "status=completed" in content
    assert "message_chars=" in content
    assert "SECRET RAW RESULT" not in content


def test_tool_action_recorders_update_work_items_and_blackboard() -> None:
    runtime = CoordinatorRuntime(clock=_clock())

    launch = runtime.record_agent_launch(
        agent_id="a_worker",
        task_id="a_worker",
        agent_type="worker",
        description="research worker",
        prompt_chars=42,
        mode="background",
        status="running",
    )
    routed = runtime.record_send_message(
        sender="team-lead",
        target="@researcher",
        summary="follow-up",
        message_chars=len("SECRET MESSAGE BODY"),
        recipient_task_ids=("a_worker",),
        recipient_agent_ids=("a_worker",),
        team_name="team",
    )
    stopped = runtime.record_task_stop(
        task_id="a_worker",
        task_type="local_agent",
        description="research worker",
    )
    message = runtime.render_context()

    assert launch.work_item.id == "cw_agent_a_worker"
    assert routed.work_items[0].id == launch.work_item.id
    assert stopped.work_item.id == launch.work_item.id
    assert runtime.snapshot().work_items[0].status == "killed"
    assert routed.blackboard_entry.kind == "system"
    assert stopped.blackboard_entry.kind == "risk"
    assert message is not None
    content = str(message["content"])
    assert "summary=follow-up" in content
    assert "message_chars=19" in content
    assert "SECRET MESSAGE BODY" not in content


def test_observability_payloads_are_metadata_only_without_raw_content() -> None:
    sink = RecordingKernelEventSink()
    runtime = CoordinatorRuntime(
        observability=KernelEventBus([sink]),
        clock=_clock(),
    )

    runtime.add_work_item(
        kind="research",
        title="SECRET TITLE",
        status="planned",
        result_summary="SECRET RESULT SUMMARY",
    )
    runtime.add_blackboard_entry(
        kind="fact",
        content="SECRET BLACKBOARD CONTENT",
        source="agent",
    )
    runtime.record_task_notifications(
        [
            TaskNotification(
                task_id="a1",
                message="<result>SECRET TASK RESULT</result>",
                kind="completed",
                created_at=1.0,
            )
        ]
    )
    runtime.render_context()

    assert sink.events
    for event in sink.events:
        assert event.content_policy == "metadata_only"
        payload = str(dict(event.data))
        assert "SECRET TITLE" not in payload
        assert "SECRET RESULT SUMMARY" not in payload
        assert "SECRET BLACKBOARD CONTENT" not in payload
        assert "SECRET TASK RESULT" not in payload
