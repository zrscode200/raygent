from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.services.runtime_identity import (
    ArtifactDescriptor,
    GoalRuntimeDescriptor,
    RuntimeIdentityValidationError,
    RuntimeLifecycleDescriptor,
    RuntimeObjectReference,
    RuntimeProvenance,
    SessionDescriptor,
    TaskOutputDescriptor,
    TranscriptEntryDescriptor,
    runtime_lifecycle_category_for,
    runtime_lifecycle_descriptor_from_dict,
    runtime_lifecycle_descriptor_to_dict,
    runtime_object_descriptor_from_dict,
    runtime_object_descriptor_to_dict,
    runtime_object_reference_from_dict,
    runtime_object_reference_to_dict,
    runtime_provenance_from_dict,
    runtime_provenance_to_dict,
)


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
