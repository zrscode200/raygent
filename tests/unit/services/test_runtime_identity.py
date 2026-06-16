from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from raygent_harness.core.observability import KernelEvent
from raygent_harness.core.state import CompactBoundary
from raygent_harness.core.task import TaskStateBase
from raygent_harness.goals.models import GoalArtifact, GoalSpec, create_goal_state
from raygent_harness.goals.runtime import GoalRuntime
from raygent_harness.services.runtime_identity import (
    ArtifactDescriptor,
    GoalRuntimeDescriptor,
    RuntimeHandlesLike,
    RuntimeIdentitySnapshotOptions,
    RuntimeIdentityValidationError,
    RuntimeLifecycleDescriptor,
    RuntimeObjectReference,
    RuntimeProvenance,
    SessionDescriptor,
    TaskOutputDescriptor,
    TranscriptEntryDescriptor,
    describe_goal_artifact,
    describe_goal_runtime,
    describe_goal_state,
    describe_kernel_event,
    describe_runtime_handles,
    describe_runtime_recovery_result,
    describe_runtime_session,
    describe_task_output_read_result,
    describe_task_output_reference,
    describe_task_state,
    describe_transcript_entry,
    describe_transcript_scope,
    describe_transcript_search_match,
    runtime_lifecycle_category_for,
    runtime_lifecycle_descriptor_from_dict,
    runtime_lifecycle_descriptor_to_dict,
    runtime_object_descriptor_from_dict,
    runtime_object_descriptor_to_dict,
    runtime_object_ref,
    runtime_object_reference_from_dict,
    runtime_object_reference_to_dict,
    runtime_provenance_from_dict,
    runtime_provenance_to_dict,
)
from raygent_harness.services.runtime_recovery import (
    RuntimeRecoveryResult,
    RuntimeRecoveryWarning,
)
from raygent_harness.services.task_output import (
    TaskOutputReadResult,
    TaskOutputReference,
)
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    SessionReplay,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.services.transcript.search import TranscriptSearchMatch


def test_runtime_object_reference_round_trips_and_validates_kind() -> None:
    ref = RuntimeObjectReference(
        kind="session",
        object_id="session-1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
    )

    encoded = runtime_object_reference_to_dict(ref)

    assert runtime_object_reference_from_dict(encoded) == ref
    with pytest.raises(RuntimeIdentityValidationError):
        RuntimeObjectReference(kind="tool_call", object_id="toolu-1")  # type: ignore[arg-type]
    with pytest.raises(RuntimeIdentityValidationError):
        RuntimeObjectReference(kind="session", object_id="")


def test_runtime_provenance_uses_tool_use_identifier_only_reference() -> None:
    provenance = RuntimeProvenance(
        session_ref=RuntimeObjectReference(
            kind="session",
            object_id="session-1",
            session_id="session-1",
        ),
        tool_use_ref=RuntimeObjectReference(
            kind="tool_use",
            object_id="toolu-1",
            session_id="session-1",
        ),
        turn_id="turn-1",
        iteration=2,
        source="query",
    )

    encoded = runtime_provenance_to_dict(provenance)

    assert runtime_provenance_from_dict(encoded) == provenance
    assert "tool_use_ref" in encoded
    assert "tool_call_ref" not in encoded
    with pytest.raises(RuntimeIdentityValidationError):
        RuntimeProvenance(
            tool_use_ref=RuntimeObjectReference(
                kind="task",
                object_id="task-1",
            )
        )


def test_runtime_lifecycle_mapping_preserves_native_status_boundary() -> None:
    assert runtime_lifecycle_category_for("task", "killed") == "cancelled"
    assert runtime_lifecycle_category_for("task", "pending") == "running"
    assert runtime_lifecycle_category_for("goal", "usage_limited") == "limited"
    assert runtime_lifecycle_category_for("goal", "budget_limited") == "limited"
    assert runtime_lifecycle_category_for("event", None) == "completed"
    assert runtime_lifecycle_category_for("transcript_entry", "message") == "unknown"
    assert runtime_lifecycle_category_for("recovery", "recovered") == "unknown"

    lifecycle = RuntimeLifecycleDescriptor(
        native_status="usage_limited",
        category=runtime_lifecycle_category_for("goal", "usage_limited"),
        reason="token budget",
    )

    assert runtime_lifecycle_descriptor_from_dict(
        runtime_lifecycle_descriptor_to_dict(lifecycle)
    ) == lifecycle


def test_session_descriptor_hides_raw_paths_by_default() -> None:
    descriptor = SessionDescriptor(
        ref=RuntimeObjectReference(
            kind="session",
            object_id="session-1",
            session_id="session-1",
        ),
        cwd_path_present=True,
        transcript_path_present=True,
        output_dir_path_present=True,
        task_store_present=True,
        task_output_store_present=True,
        transcript_store_present=True,
        observability_present=True,
        goal_runtime_attached=True,
        metadata={"path_policy": "hidden"},
    )

    encoded = runtime_object_descriptor_to_dict(descriptor)
    serialized = json.dumps(encoded, sort_keys=True)

    assert runtime_object_descriptor_from_dict(encoded) == descriptor
    assert "cwd_path_present" in encoded
    assert "transcript_path_present" in encoded
    assert "/tmp/session-1" not in serialized
    assert "transcript.jsonl" not in serialized


def test_transcript_entry_descriptor_separates_message_only_fields() -> None:
    ref = RuntimeObjectReference(
        kind="transcript_entry",
        object_id="tr-1",
        session_id="session-1",
    )

    non_message = TranscriptEntryDescriptor(
        ref=ref,
        entry_type="compact_boundary",
        source_path_present=True,
    )
    encoded = runtime_object_descriptor_to_dict(non_message)

    assert runtime_object_descriptor_from_dict(encoded) == non_message
    assert encoded["message_fields_present"] is False
    assert encoded["role"] is None

    with pytest.raises(RuntimeIdentityValidationError):
        TranscriptEntryDescriptor(
            ref=ref,
            entry_type="compact_boundary",
            role="assistant",
        )

    with pytest.raises(RuntimeIdentityValidationError):
        TranscriptEntryDescriptor(
            ref=ref,
            entry_type="compact_boundary",
            role="assistant",
            message_fields_present=True,
        )

    message = TranscriptEntryDescriptor(
        ref=ref,
        entry_type="message",
        role="assistant",
        parent_entry_id="tr-parent",
        logical_parent_entry_id="tr-logical-parent",
        provider_message_id_present=True,
        is_sidechain=True,
        cwd_path_present=True,
        version_present=True,
        message_fields_present=True,
    )

    assert runtime_object_descriptor_from_dict(
        runtime_object_descriptor_to_dict(message)
    ) == message


def test_task_output_descriptor_records_path_presence_not_path_value() -> None:
    descriptor = TaskOutputDescriptor(
        ref=RuntimeObjectReference(
            kind="task_output",
            object_id="task-1",
            session_id="session-1",
        ),
        store_kind="filesystem",
        path_present=True,
        bytes_total=120,
        start_offset=20,
        bytes_read=80,
        next_offset=100,
        truncated_before=True,
        truncated_after=False,
    )

    encoded = runtime_object_descriptor_to_dict(descriptor)
    serialized = json.dumps(encoded, sort_keys=True)

    assert runtime_object_descriptor_from_dict(encoded) == descriptor
    assert encoded["path_present"] is True
    assert "path" not in encoded
    assert ".raygent/tasks/task-1.output" not in serialized


def test_goal_runtime_descriptor_uses_public_safe_facts_only() -> None:
    descriptor = GoalRuntimeDescriptor(
        ref=RuntimeObjectReference(
            kind="goal_runtime",
            object_id="session-1:goal_runtime",
            session_id="session-1",
        ),
        attached=True,
        config_present=True,
        store_kind="JsonGoalStore",
        active_goal_supplied=False,
    )

    encoded = runtime_object_descriptor_to_dict(descriptor)

    assert runtime_object_descriptor_from_dict(encoded) == descriptor
    assert "installed" not in json.dumps(encoded, sort_keys=True)
    assert "private" not in json.dumps(encoded, sort_keys=True)


def test_metadata_is_json_only_and_bounded() -> None:
    ref = RuntimeObjectReference(kind="artifact", object_id="artifact-1")

    with pytest.raises(TypeError):
        ArtifactDescriptor(
            ref=ref,
            artifact_kind="task_output",
            metadata=cast(Any, {"bad": object()}),
        )

    with pytest.raises(RuntimeIdentityValidationError):
        ArtifactDescriptor(
            ref=ref,
            artifact_kind="task_output",
            metadata={"huge": "x" * 25_000},
        )

    with pytest.raises(RuntimeIdentityValidationError):
        ArtifactDescriptor(
            ref=ref,
            artifact_kind="task_output",
            metadata_only=False,
        )


def test_builder_runtime_object_ref_and_handles_hide_paths() -> None:
    ref = runtime_object_ref(
        "session",
        "session-1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
    )
    assert ref.kind == "session"
    assert ref.runtime_session_id == "runtime-1"

    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=object(),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=object(),
            transcript_scope=TranscriptScope(session_id="session-1"),
            observability=object(),
            abort_event=object(),
            goal_runtime=object(),
        ),
    )

    descriptor = describe_runtime_handles(handles)
    encoded = runtime_object_descriptor_to_dict(descriptor)
    serialized = json.dumps(encoded, sort_keys=True)

    assert descriptor.cwd_path_present is True
    assert descriptor.transcript_path_present is True
    assert descriptor.output_dir_path_present is True
    assert descriptor.goal_runtime_attached is True
    assert "/tmp/raygent-project" not in serialized
    assert "/tmp/raygent-output" not in serialized


def test_builder_transcript_scope_emits_supplied_refs_only() -> None:
    descriptors = describe_transcript_scope(
        TranscriptScope(
            session_id="session-1",
            runtime_session_id="runtime-1",
            agent_id="agent-1",
            is_sidechain=True,
        )
    )

    assert tuple(descriptor.ref.kind for descriptor in descriptors) == (
        "session",
        "runtime_session",
        "agent",
    )
    assert descriptors[1].provenance.session_ref is not None
    assert descriptors[2].provenance.runtime_session_ref is not None


def test_builder_transcript_entry_never_copies_message_content_or_non_message_lineage() -> None:
    message = TranscriptMessageEntry(
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
        parent_entry_id="tr-parent",
        logical_parent_entry_id="tr-logical-parent",
        cwd="/tmp/secret-cwd",
        version="v1",
        message={
            "role": "assistant",
            "content": "secret transcript body",
        },
        provider_message_id="provider-1",
        created_at=10.0,
    )
    message_descriptor = describe_transcript_entry(message)
    message_serialized = json.dumps(
        runtime_object_descriptor_to_dict(message_descriptor),
        sort_keys=True,
    )

    assert message_descriptor.message_fields_present is True
    assert message_descriptor.role == "assistant"
    assert message_descriptor.provider_message_id_present is True
    assert message_descriptor.cwd_path_present is True
    assert "secret transcript body" not in message_serialized
    assert "/tmp/secret-cwd" not in message_serialized

    boundary = CompactBoundaryEntry(
        session_id="session-1",
        agent_id="agent-1",
        boundary=CompactBoundary(
            message_index=1,
            kind="microcompact",
            summary="secret summary",
        ),
        created_at=11.0,
    )
    boundary_descriptor = describe_transcript_entry(boundary)

    assert boundary_descriptor.entry_type == "compact_boundary"
    assert boundary_descriptor.message_fields_present is False
    assert boundary_descriptor.role is None
    assert boundary_descriptor.is_sidechain is None


def test_builder_transcript_search_match_never_copies_snippet_or_path() -> None:
    match = TranscriptSearchMatch(
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
        entry_id="tr-1",
        role="user",
        snippet="secret search snippet",
        score=42,
        order=3,
        created_at=20.0,
        source_path="/tmp/transcript.jsonl",
        snippet_truncated=True,
    )

    descriptor = describe_transcript_search_match(match)
    encoded = runtime_object_descriptor_to_dict(descriptor)
    serialized = json.dumps(encoded, sort_keys=True)

    assert descriptor.source_path_present is True
    assert descriptor.metadata["snippet_char_count"] == len(match.snippet)
    assert "score" not in descriptor.metadata
    assert "order" not in descriptor.metadata
    assert "secret search snippet" not in serialized
    assert "/tmp/transcript.jsonl" not in serialized


def test_builder_task_and_task_output_descriptors_avoid_paths_and_bytes() -> None:
    task = TaskStateBase(
        id="task-1",
        type="local_bash",
        description="run tests",
        status="killed",
        start_time=1.0,
        end_time=2.0,
        tool_use_id="toolu-1",
        output_file="/tmp/task-output.txt",
        output_offset=12,
        notified=True,
    )
    task_descriptor = describe_task_state(task)
    task_serialized = json.dumps(
        runtime_object_descriptor_to_dict(task_descriptor),
        sort_keys=True,
    )

    assert task_descriptor.lifecycle.category == "cancelled"
    assert task_descriptor.provenance.tool_use_ref is not None
    assert task_descriptor.output_file_present is True
    assert "/tmp/task-output.txt" not in task_serialized

    output_reference = TaskOutputReference(
        task_id="task-1",
        path="/tmp/task-output.txt",
        store_kind="filesystem",
    )
    reference_descriptor = describe_task_output_reference(output_reference)
    reference_serialized = json.dumps(
        runtime_object_descriptor_to_dict(reference_descriptor),
        sort_keys=True,
    )

    assert reference_descriptor.path_present is True
    assert "/tmp/task-output.txt" not in reference_serialized

    read_result = TaskOutputReadResult(
        task_id="task-1",
        content=b"secret task output bytes",
        start_offset=5,
        bytes_read=24,
        bytes_total=100,
        next_offset=29,
        truncated_before=True,
        truncated_after=True,
    )
    read_descriptor = describe_task_output_read_result(
        read_result,
        store_kind="filesystem",
        path_present=True,
    )
    read_serialized = json.dumps(
        runtime_object_descriptor_to_dict(read_descriptor),
        sort_keys=True,
    )

    assert read_descriptor.bytes_total == 100
    assert read_descriptor.bytes_read == 24
    assert "secret task output bytes" not in read_serialized


def test_builder_kernel_event_descriptor_never_copies_event_data() -> None:
    event = KernelEvent(
        id="event-1",
        type="tool.completed",
        sequence=7,
        created_at=30.0,
        source="tool",
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
        parent_agent_id="parent-agent",
        turn_id="turn-1",
        iteration=2,
        span_id="span-1",
        parent_span_id="span-parent",
        content_policy="content_opt_in",
        data={"token": "secret event data"},
    )

    descriptor = describe_kernel_event(event)
    serialized = json.dumps(
        runtime_object_descriptor_to_dict(descriptor),
        sort_keys=True,
    )

    assert descriptor.lifecycle.category == "completed"
    assert descriptor.data_present is True
    assert descriptor.metadata["data_key_count"] == 1
    assert descriptor.provenance.turn_id == "turn-1"
    assert "secret event data" not in serialized


def test_builder_goal_descriptors_avoid_objective_artifact_uri_and_description() -> None:
    artifact = GoalArtifact(
        artifact_id="artifact-1",
        kind="task_output",
        uri="file:///tmp/secret-artifact.txt",
        description="secret artifact description",
        metadata={"safe": True},
    )
    goal = replace(
        create_goal_state(
            goal_id="goal-1",
            session_id="session-1",
            spec=GoalSpec(objective="secret objective"),
            now=40.0,
        ),
        artifacts=(artifact,),
        pending_task_ids=("task-1",),
        summary="secret summary",
    )

    goal_descriptor = describe_goal_state(goal)
    artifact_descriptor = describe_goal_artifact(artifact, goal=goal)
    runtime_descriptor = describe_goal_runtime(
        None,
        session_id="session-1",
        active_goal=goal,
    )
    serialized = json.dumps(
        {
            "goal": runtime_object_descriptor_to_dict(goal_descriptor),
            "artifact": runtime_object_descriptor_to_dict(artifact_descriptor),
            "runtime": runtime_object_descriptor_to_dict(runtime_descriptor),
        },
        sort_keys=True,
    )

    assert goal_descriptor.native_goal_status == "active"
    assert goal_descriptor.artifact_refs[0].object_id == "artifact-1"
    assert artifact_descriptor.uri_present is True
    assert artifact_descriptor.metadata["description_present"] is True
    assert runtime_descriptor.active_goal_supplied is True
    assert "secret objective" not in serialized
    assert "secret summary" not in serialized
    assert "secret artifact description" not in serialized
    assert "file:///tmp/secret-artifact.txt" not in serialized

    fake_runtime = cast(
        GoalRuntime,
        SimpleNamespace(session_id="session-1", store=object()),
    )
    with pytest.raises(RuntimeIdentityValidationError):
        describe_goal_runtime(fake_runtime, session_id="session-2")
    with pytest.raises(RuntimeIdentityValidationError):
        describe_goal_runtime(None, session_id="session-2", active_goal=goal)
    with pytest.raises(RuntimeIdentityValidationError):
        describe_goal_artifact(artifact, goal=goal, session_id="session-2")
    with pytest.raises(RuntimeIdentityValidationError):
        describe_goal_artifact(
            GoalArtifact(artifact_id="artifact-2", kind="task_output"),
            goal=goal,
        )


def test_builder_runtime_recovery_result_avoids_paths() -> None:
    result = RuntimeRecoveryResult(
        replay=SessionReplay(
            session_id="session-1",
            messages=[],
            runtime_session_id="runtime-1",
            last_message_entry_id="tr-last",
        ),
        transcript_scope=TranscriptScope(
            session_id="session-1",
            runtime_session_id="runtime-1",
            agent_id="agent-1",
        ),
        transcript_path="/tmp/transcript.jsonl",
        last_message_entry_id="tr-last",
        coordinator_runtime_restored=True,
        restored_agent_names=("agent-name",),
        restored_remote_task_ids=("remote-task-1",),
        warnings=(RuntimeRecoveryWarning(source="transcript", reason="missing"),),
    )

    descriptor = describe_runtime_recovery_result(result)
    serialized = json.dumps(
        runtime_object_descriptor_to_dict(descriptor),
        sort_keys=True,
    )

    assert descriptor.transcript_path_present is True
    assert descriptor.warning_count == 1
    assert descriptor.restored_agent_name_count == 1
    assert descriptor.restored_remote_task_count == 1
    assert "/tmp/transcript.jsonl" not in serialized


def test_runtime_identity_snapshot_from_session_like_is_bounded_and_path_safe() -> None:
    first_task = TaskStateBase(
        id="task-1",
        type="local_bash",
        description="first task",
        status="running",
        start_time=1.0,
        output_file="/tmp/task-1.output",
    )
    second_task = TaskStateBase(
        id="task-2",
        type="local_agent",
        description="second task",
        status="pending",
        start_time=2.0,
        output_file="/tmp/task-2.output",
    )
    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(
                tasks={
                    "task-1": first_task,
                    "task-2": second_task,
                }
            ),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=object(),
            transcript_scope=TranscriptScope(
                session_id="session-1",
                runtime_session_id="runtime-1",
                agent_id="agent-1",
                is_sidechain=True,
            ),
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )
    session_like = SimpleNamespace(handles=handles)

    snapshot = describe_runtime_session(
        cast(Any, session_like),
        options=RuntimeIdentitySnapshotOptions(max_tasks=1),
    )
    serialized = json.dumps(
        [runtime_object_descriptor_to_dict(item) for item in snapshot.descriptors],
        sort_keys=True,
    )

    assert snapshot.session_id == "session-1"
    assert snapshot.truncated is True
    assert "tasks_truncated" in snapshot.warnings
    assert any(item.ref.kind == "session" for item in snapshot.descriptors)
    assert any(item.ref.kind == "runtime_session" for item in snapshot.descriptors)
    assert any(item.ref.kind == "agent" for item in snapshot.descriptors)
    assert sum(1 for item in snapshot.descriptors if item.ref.kind == "task") == 1
    assert "/tmp/raygent-project" not in serialized
    assert "/tmp/raygent-output" not in serialized
    assert "/tmp/task-1.output" not in serialized
    assert "/tmp/task-2.output" not in serialized


def test_runtime_identity_snapshot_includes_only_supplied_bounded_facts() -> None:
    artifact = GoalArtifact(
        artifact_id="artifact-1",
        kind="task_output",
        uri="file:///tmp/secret-artifact.txt",
        description="secret artifact description",
    )
    active_goal = replace(
        create_goal_state(
            goal_id="goal-1",
            session_id="session-1",
            spec=GoalSpec(objective="secret objective"),
            now=5.0,
        ),
        artifacts=(artifact,),
    )
    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(tasks={}),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=None,
            transcript_scope=None,
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )
    match = TranscriptSearchMatch(
        session_id="session-1",
        entry_id="tr-1",
        role="assistant",
        snippet="secret snippet",
        score=10,
        order=2,
        created_at=7.0,
        source_path="/tmp/transcript.jsonl",
    )
    read_result = TaskOutputReadResult(
        task_id="task-1",
        content=b"secret output bytes",
        start_offset=0,
        bytes_read=19,
        bytes_total=19,
        next_offset=19,
    )
    event = KernelEvent(
        id="event-1",
        type="task.completed",
        sequence=1,
        created_at=8.0,
        source="task",
        session_id="session-1",
        data={"secret": "event payload"},
    )

    snapshot = describe_runtime_session(
        handles,
        active_goal=active_goal,
        transcript_search_matches=(match,),
        task_output_read_results=(read_result,),
        kernel_events=(event,),
        goal_artifacts=(artifact,),
        options=RuntimeIdentitySnapshotOptions(max_supplied_items=1),
    )
    serialized = json.dumps(
        [runtime_object_descriptor_to_dict(item) for item in snapshot.descriptors],
        sort_keys=True,
    )

    assert snapshot.truncated is False
    assert any(item.ref.kind == "goal" for item in snapshot.descriptors)
    assert any(item.ref.kind == "artifact" for item in snapshot.descriptors)
    assert any(item.ref.kind == "task_output" for item in snapshot.descriptors)
    assert any(item.ref.kind == "event" for item in snapshot.descriptors)
    assert "secret objective" not in serialized
    assert "secret snippet" not in serialized
    assert "secret output bytes" not in serialized
    assert "event payload" not in serialized
    assert "file:///tmp/secret-artifact.txt" not in serialized
    assert "/tmp/transcript.jsonl" not in serialized


def test_runtime_identity_snapshot_enforces_descriptor_and_supplied_item_bounds() -> None:
    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(tasks={}),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=None,
            transcript_scope=None,
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )
    matches = tuple(
        TranscriptSearchMatch(
            session_id="session-1",
            entry_id=f"tr-{index}",
            role="user",
            snippet=f"snippet-{index}",
            score=index,
            order=index,
            created_at=float(index),
        )
        for index in range(3)
    )

    snapshot = describe_runtime_session(
        handles,
        transcript_search_matches=matches,
        options=RuntimeIdentitySnapshotOptions(
            include_goal_runtime=False,
            max_supplied_items=1,
            max_descriptors=1,
        ),
    )

    assert snapshot.truncated is True
    assert snapshot.warnings == ("transcript_search_matches_truncated", "descriptors_truncated")
    assert len(snapshot.descriptors) == 1
    with pytest.raises(RuntimeIdentityValidationError):
        RuntimeIdentitySnapshotOptions(max_tasks=-1)
    with pytest.raises(RuntimeIdentityValidationError):
        RuntimeIdentitySnapshotOptions(max_descriptors=0)


def test_runtime_identity_snapshot_rejects_cross_session_supplied_facts() -> None:
    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(tasks={}),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=object(),
            transcript_scope=TranscriptScope(session_id="session-2"),
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )

    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles)

    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(tasks={}),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=None,
            transcript_scope=None,
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )
    other_goal = create_goal_state(
        goal_id="goal-2",
        session_id="session-2",
        spec=GoalSpec(objective="other objective"),
    )
    other_entry = TranscriptMessageEntry(
        session_id="session-2",
        message={"role": "user", "content": "other session body"},
    )
    other_match = TranscriptSearchMatch(
        session_id="session-2",
        entry_id="tr-other",
        role="user",
        snippet="other session snippet",
        score=1,
        order=1,
        created_at=1.0,
    )
    other_event = KernelEvent(
        id="event-other",
        type="task.completed",
        sequence=1,
        created_at=1.0,
        source="task",
        session_id="session-2",
    )
    other_recovery = RuntimeRecoveryResult(
        replay=SessionReplay(session_id="session-2", messages=[]),
        transcript_scope=TranscriptScope(session_id="session-2"),
        transcript_path=None,
        last_message_entry_id=None,
    )

    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles, active_goal=other_goal)
    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles, transcript_entries=(other_entry,))
    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles, transcript_search_matches=(other_match,))
    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles, kernel_events=(other_event,))
    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(handles, runtime_recovery_results=(other_recovery,))
    with pytest.raises(RuntimeIdentityValidationError):
        describe_runtime_session(
            handles,
            goal_artifacts=(GoalArtifact(artifact_id="artifact-1", kind="task_output"),),
        )


def test_runtime_identity_snapshot_task_bounds_do_not_walk_entire_store() -> None:
    class _CountingTaskMapping(dict[str, TaskStateBase]):
        iterated = 0

        def values(self) -> Iterator[TaskStateBase]:  # type: ignore[override]
            for value in super().values():
                self.iterated += 1
                yield value

    tasks = _CountingTaskMapping(
        {
            f"task-{index}": TaskStateBase(
                id=f"task-{index}",
                type="local_bash",
                description=f"task {index}",
                status="running",
                start_time=float(index),
            )
            for index in range(10)
        }
    )
    handles = cast(
        RuntimeHandlesLike,
        SimpleNamespace(
            session_id="session-1",
            cwd="/tmp/raygent-project",
            task_store=SimpleNamespace(tasks=tasks),
            output_dir=Path("/tmp/raygent-output"),
            task_output_store=object(),
            transcript_store=None,
            transcript_scope=None,
            observability=object(),
            abort_event=object(),
            goal_runtime=None,
        ),
    )

    snapshot = describe_runtime_session(
        handles,
        options=RuntimeIdentitySnapshotOptions(max_tasks=2),
    )

    assert tasks.iterated == 3
    assert snapshot.truncated is True
    assert "tasks_truncated" in snapshot.warnings
    assert sum(1 for item in snapshot.descriptors if item.ref.kind == "task") == 2


def test_runtime_identity_snapshot_keeps_bounded_no_read_boundary() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "raygent_harness"
        / "services"
        / "runtime_identity"
        / "snapshot.py"
    ).read_text()

    banned_calls = (
        ".read_entries(",
        ".read_result(",
        ".read_tail(",
        ".read_range(",
        ".size(",
        ".get_active_for_session(",
        ".list_for_session(",
        ".install(",
        ".start(",
        ".resume(",
        ".cancel(",
        "run_until_result(",
        "subprocess",
        "urllib",
        "requests",
    )
    for banned in banned_calls:
        assert banned not in source


def test_runtime_identity_builders_keep_pure_adapter_boundary() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "raygent_harness"
        / "services"
        / "runtime_identity"
        / "builders.py"
    ).read_text()

    banned_calls = (
        ".read_entries(",
        ".read_result(",
        ".read_tail(",
        ".read_range(",
        ".size(",
        ".append_task_output(",
        ".init_task_output(",
        ".cleanup_task_output(",
        ".install(",
        ".start(",
        ".resume(",
        ".cancel(",
        "run_until_result(",
        "subprocess",
        "urllib",
        "requests",
    )
    for banned in banned_calls:
        assert banned not in source


def test_runtime_identity_models_keep_data_contract_import_boundary() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "raygent_harness"
        / "services"
        / "runtime_identity"
        / "models.py"
    ).read_text()

    banned_imports = (
        "raygent_harness.sdk",
        "raygent_harness.core.model_provider",
        "raygent_harness.core.tool",
        "raygent_harness.services.task_output",
        "raygent_harness.services.transcript",
        "subprocess",
        "pathlib",
        "urllib",
        "requests",
    )
    for banned in banned_imports:
        assert banned not in source
