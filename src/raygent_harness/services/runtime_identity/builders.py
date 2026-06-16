"""Pure builders for runtime identity descriptors.

The functions in this module adapt already-supplied Raygent runtime facts into
runtime identity descriptors. They do not read stores, invoke tools, mutate
runtime state, or expose raw local paths/content by default.
"""

from __future__ import annotations

from typing import Protocol

from raygent_harness.core.observability import KernelEvent
from raygent_harness.core.task import TaskStateBase
from raygent_harness.goals.models import GoalArtifact, GoalState
from raygent_harness.goals.runtime import GoalRuntime
from raygent_harness.services.runtime_identity.models import (
    ArtifactDescriptor,
    EventDescriptor,
    GoalDescriptor,
    GoalRuntimeDescriptor,
    RecoveryDescriptor,
    RuntimeIdentityValidationError,
    RuntimeLifecycleDescriptor,
    RuntimeObjectDescriptor,
    RuntimeObjectKind,
    RuntimeObjectReference,
    RuntimeProvenance,
    SessionDescriptor,
    TaskDescriptor,
    TaskOutputDescriptor,
    TranscriptEntryDescriptor,
    runtime_lifecycle_category_for,
)
from raygent_harness.services.runtime_recovery import RuntimeRecoveryResult
from raygent_harness.services.task_output.store import (
    TaskOutputReadResult,
    TaskOutputReference,
)
from raygent_harness.services.transcript.models import (
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.services.transcript.search import TranscriptSearchMatch


class RuntimeHandlesLike(Protocol):
    """Minimal session handle shape consumed by runtime identity builders."""

    @property
    def session_id(self) -> str: ...

    @property
    def cwd(self) -> str: ...

    @property
    def task_store(self) -> object: ...

    @property
    def output_dir(self) -> object: ...

    @property
    def task_output_store(self) -> object: ...

    @property
    def transcript_store(self) -> object | None: ...

    @property
    def transcript_scope(self) -> TranscriptScope | None: ...

    @property
    def observability(self) -> object: ...

    @property
    def abort_event(self) -> object: ...

    @property
    def goal_runtime(self) -> GoalRuntime | None: ...


def runtime_object_ref(
    kind: RuntimeObjectKind,
    object_id: str,
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
    agent_id: str | None = None,
) -> RuntimeObjectReference:
    """Build a runtime object reference without adding authority semantics."""

    return RuntimeObjectReference(
        kind=kind,
        object_id=object_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )


def describe_runtime_handles(handles: RuntimeHandlesLike) -> SessionDescriptor:
    """Describe an SDK/runtime handle bundle without exposing concrete paths."""

    session_ref = _required_session_ref(handles.session_id)
    return SessionDescriptor(
        ref=session_ref,
        provenance=RuntimeProvenance(session_ref=session_ref, source="runtime_handles"),
        lifecycle=RuntimeLifecycleDescriptor(
            category=runtime_lifecycle_category_for("session", None),
        ),
        cwd_path_present=bool(handles.cwd),
        transcript_path_present=(
            handles.transcript_store is not None and handles.transcript_scope is not None
        ),
        output_dir_path_present=handles.output_dir is not None,
        task_store_present=handles.task_store is not None,
        task_output_store_present=handles.task_output_store is not None,
        transcript_store_present=handles.transcript_store is not None,
        observability_present=handles.observability is not None,
        goal_runtime_attached=handles.goal_runtime is not None,
    )


def describe_transcript_scope(
    scope: TranscriptScope,
) -> tuple[RuntimeObjectDescriptor, ...]:
    """Describe session/runtime-session/agent facts present in a transcript scope."""

    session_ref = _required_session_ref(scope.session_id)
    provenance = RuntimeProvenance(session_ref=session_ref, source="transcript_scope")
    descriptors: list[RuntimeObjectDescriptor] = [
        SessionDescriptor(
            ref=session_ref,
            provenance=provenance,
            lifecycle=RuntimeLifecycleDescriptor(
                category=runtime_lifecycle_category_for("session", None),
            ),
            metadata={
                "scope_kind": "transcript",
                "is_sidechain": scope.is_sidechain,
            },
        )
    ]
    if scope.runtime_session_id is not None:
        runtime_ref = _required_runtime_session_ref(
            scope.runtime_session_id,
            session_id=scope.session_id,
            agent_id=scope.agent_id,
        )
        descriptors.append(
            RuntimeObjectDescriptor(
                ref=runtime_ref,
                provenance=RuntimeProvenance(
                    session_ref=session_ref,
                    runtime_session_ref=runtime_ref,
                    source="transcript_scope",
                ),
                metadata={
                    "scope_kind": "transcript",
                    "is_sidechain": scope.is_sidechain,
                },
            )
        )
    if scope.agent_id is not None:
        agent_ref = _required_agent_ref(
            scope.agent_id,
            session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
        )
        descriptors.append(
            RuntimeObjectDescriptor(
                ref=agent_ref,
                provenance=RuntimeProvenance(
                    session_ref=session_ref,
                    runtime_session_ref=_runtime_session_ref(
                        scope.runtime_session_id,
                        session_id=scope.session_id,
                        agent_id=scope.agent_id,
                    )
                    if scope.runtime_session_id is not None
                    else None,
                    agent_ref=agent_ref,
                    source="transcript_scope",
                ),
                metadata={
                    "scope_kind": "transcript",
                    "is_sidechain": scope.is_sidechain,
                },
            )
        )
    return tuple(descriptors)


def describe_transcript_entry(entry: TranscriptEntry) -> TranscriptEntryDescriptor:
    """Describe one transcript entry without copying raw message/event content."""

    agent_id = _optional_string_attr(entry, "agent_id")
    runtime_session_id = (
        entry.runtime_session_id if isinstance(entry, TranscriptMessageEntry) else None
    )
    entry_ref = runtime_object_ref(
        "transcript_entry",
        entry.entry_id,
        session_id=entry.session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )
    provenance = _provenance_from_ids(
        session_id=entry.session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
        source="transcript_entry",
    )
    lifecycle = RuntimeLifecycleDescriptor(
        native_status=entry.type,
        category=runtime_lifecycle_category_for("transcript_entry", entry.type),
        updated_at=entry.created_at,
    )
    if isinstance(entry, TranscriptMessageEntry):
        role = entry.message.get("role")
        return TranscriptEntryDescriptor(
            ref=entry_ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=entry.created_at,
            updated_at=entry.created_at,
            entry_type=entry.type,
            role=role or None,
            parent_entry_id=entry.parent_entry_id,
            logical_parent_entry_id=entry.logical_parent_entry_id,
            provider_message_id_present=entry.provider_message_id is not None,
            is_sidechain=entry.is_sidechain,
            cwd_path_present=entry.cwd is not None,
            version_present=entry.version is not None,
            message_fields_present=True,
        )
    return TranscriptEntryDescriptor(
        ref=entry_ref,
        provenance=provenance,
        lifecycle=lifecycle,
        created_at=entry.created_at,
        updated_at=entry.created_at,
        entry_type=entry.type,
    )


def describe_transcript_search_match(
    match: TranscriptSearchMatch,
) -> TranscriptEntryDescriptor:
    """Describe a bounded transcript search match without storing the snippet."""

    return TranscriptEntryDescriptor(
        ref=runtime_object_ref(
            "transcript_entry",
            match.entry_id,
            session_id=match.session_id,
            runtime_session_id=match.runtime_session_id,
            agent_id=match.agent_id,
        ),
        provenance=_provenance_from_ids(
            session_id=match.session_id,
            runtime_session_id=match.runtime_session_id,
            agent_id=match.agent_id,
            source="transcript_search_match",
        ),
        lifecycle=RuntimeLifecycleDescriptor(
            native_status="message",
            category=runtime_lifecycle_category_for("transcript_entry", "message"),
            updated_at=match.created_at,
        ),
        created_at=match.created_at,
        updated_at=match.created_at,
        metadata={
            "snippet_char_count": len(match.snippet),
            "snippet_truncated": match.snippet_truncated,
        },
        entry_type="message",
        role=match.role or None,
        is_sidechain=match.is_sidechain,
        source_path_present=match.source_path is not None,
        message_fields_present=True,
    )


def describe_task_state(
    task: TaskStateBase,
    *,
    session_id: str | None = None,
) -> TaskDescriptor:
    """Describe task state without exposing output paths or output content."""

    agent_id = _optional_string_attr(task, "agent_id")
    parent_agent_id = _optional_string_attr(task, "parent_agent_id")
    return TaskDescriptor(
        ref=runtime_object_ref("task", task.id, session_id=session_id, agent_id=agent_id),
        provenance=_provenance_from_ids(
            session_id=session_id,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            tool_use_id=task.tool_use_id,
            source="task_state",
        ),
        lifecycle=RuntimeLifecycleDescriptor(
            native_status=task.status,
            category=runtime_lifecycle_category_for("task", task.status),
            started_at=task.start_time,
            ended_at=task.end_time,
            updated_at=task.end_time,
        ),
        created_at=task.start_time,
        updated_at=task.end_time,
        metadata={
            "output_offset": task.output_offset,
            "total_paused_ms": task.total_paused_ms,
            "notified": task.notified,
        },
        task_type=task.type,
        description=task.description,
        tool_use_id=task.tool_use_id,
        output_file_present=task.output_file is not None,
        output_reference_present=task.output_file is not None,
    )


def describe_task_output_reference(
    reference: TaskOutputReference,
    *,
    session_id: str | None = None,
) -> TaskOutputDescriptor:
    """Describe a task-output reference without exposing its concrete path."""

    return TaskOutputDescriptor(
        ref=runtime_object_ref(
            "task_output",
            reference.task_id,
            session_id=session_id,
        ),
        provenance=_provenance_from_ids(
            session_id=session_id,
            task_id=reference.task_id,
            source="task_output_reference",
        ),
        store_kind=reference.store_kind,
        path_present=reference.path is not None,
    )


def describe_task_output_read_result(
    result: TaskOutputReadResult,
    *,
    store_kind: str = "bounded_read",
    path_present: bool = False,
    session_id: str | None = None,
) -> TaskOutputDescriptor:
    """Describe bounded task-output read metadata without storing bytes."""

    return TaskOutputDescriptor(
        ref=runtime_object_ref("task_output", result.task_id, session_id=session_id),
        provenance=_provenance_from_ids(
            session_id=session_id,
            task_id=result.task_id,
            source="task_output_read_result",
        ),
        store_kind=store_kind,
        path_present=path_present,
        bytes_total=result.bytes_total,
        start_offset=result.start_offset,
        bytes_read=result.bytes_read,
        next_offset=result.next_offset,
        truncated_before=result.truncated_before,
        truncated_after=result.truncated_after,
    )


def describe_kernel_event(event: KernelEvent) -> EventDescriptor:
    """Describe one immutable kernel event without copying event data."""

    return EventDescriptor(
        ref=runtime_object_ref(
            "event",
            event.id,
            session_id=event.session_id,
            runtime_session_id=event.runtime_session_id,
            agent_id=event.agent_id,
        ),
        provenance=RuntimeProvenance(
            session_ref=_session_ref(event.session_id),
            runtime_session_ref=_runtime_session_ref(
                event.runtime_session_id,
                session_id=event.session_id,
                agent_id=event.agent_id,
            ),
            agent_ref=_agent_ref(
                event.agent_id,
                session_id=event.session_id,
                runtime_session_id=event.runtime_session_id,
            ),
            parent_agent_ref=_agent_ref(
                event.parent_agent_id,
                session_id=event.session_id,
                runtime_session_id=event.runtime_session_id,
            ),
            turn_id=event.turn_id,
            iteration=event.iteration,
            span_id=event.span_id,
            parent_span_id=event.parent_span_id,
            source=event.source,
        ),
        lifecycle=RuntimeLifecycleDescriptor(
            native_status=event.type,
            category=runtime_lifecycle_category_for("event", event.type),
            updated_at=event.created_at,
        ),
        created_at=event.created_at,
        updated_at=event.created_at,
        metadata={"data_key_count": len(event.data)},
        event_type=event.type,
        sequence=event.sequence,
        event_source=event.source,
        content_policy=event.content_policy,
        data_present=bool(event.data),
    )


def describe_goal_runtime(
    goal_runtime: GoalRuntime | None,
    *,
    session_id: str | None = None,
    active_goal: GoalState | None = None,
) -> GoalRuntimeDescriptor:
    """Describe public goal-runtime attachment facts without lifecycle actions."""

    resolved_session_id = session_id
    if goal_runtime is not None:
        if session_id is not None and session_id != goal_runtime.session_id:
            raise RuntimeIdentityValidationError(
                "session_id must match goal_runtime.session_id"
            )
        resolved_session_id = resolved_session_id or goal_runtime.session_id
    if resolved_session_id is None:
        raise RuntimeIdentityValidationError(
            "session_id is required when goal_runtime is None"
        )
    if active_goal is not None and active_goal.session_id != resolved_session_id:
        raise RuntimeIdentityValidationError(
            "active_goal must belong to the described session"
        )
    session_ref = _required_session_ref(resolved_session_id)
    goal_ref = (
        _goal_ref(active_goal.goal_id, session_id=resolved_session_id)
        if active_goal is not None
        else None
    )
    return GoalRuntimeDescriptor(
        ref=runtime_object_ref(
            "goal_runtime",
            f"{resolved_session_id}:goal_runtime",
            session_id=resolved_session_id,
        ),
        provenance=RuntimeProvenance(
            session_ref=session_ref,
            goal_ref=goal_ref,
            source="goal_runtime",
        ),
        attached=goal_runtime is not None,
        config_present=goal_runtime is not None,
        store_kind=type(goal_runtime.store).__name__
        if goal_runtime is not None
        else None,
        active_goal_supplied=active_goal is not None,
    )


def describe_goal_state(state: GoalState) -> GoalDescriptor:
    """Describe goal state without including objective text or policy bodies."""

    artifact_refs = tuple(
        _artifact_ref(artifact.artifact_id, session_id=state.session_id)
        for artifact in state.artifacts
    )
    return GoalDescriptor(
        ref=_required_goal_ref(state.goal_id, session_id=state.session_id),
        provenance=RuntimeProvenance(
            session_ref=_required_session_ref(state.session_id),
            source="goal_state",
        ),
        lifecycle=RuntimeLifecycleDescriptor(
            native_status=state.status,
            category=runtime_lifecycle_category_for("goal", state.status),
            started_at=state.created_at,
            updated_at=state.updated_at,
            reason=state.last_reason,
        ),
        created_at=state.created_at,
        updated_at=state.updated_at,
        metadata={
            "token_budget_present": state.token_budget is not None,
            "time_used_s": state.time_used_s,
            "blocked_turn_count": state.blocked_turn_count,
            "summary_present": state.summary is not None,
            "checkpoint_count": len(state.checkpoints),
            "plan_step_count": len(state.plan_steps),
            "artifact_count": len(state.artifacts),
        },
        native_goal_status=state.status,
        turn_count=state.turn_count,
        tokens_used=state.tokens_used,
        pending_task_ids=state.pending_task_ids,
        artifact_refs=artifact_refs,
    )


def describe_goal_artifact(
    artifact: GoalArtifact,
    *,
    goal: GoalState | None = None,
    session_id: str | None = None,
) -> ArtifactDescriptor:
    """Describe a metadata-only goal artifact without exposing URI text."""

    if goal is not None and session_id is not None and session_id != goal.session_id:
        raise RuntimeIdentityValidationError("session_id must match goal.session_id")
    if goal is not None and not any(
        existing.artifact_id == artifact.artifact_id for existing in goal.artifacts
    ):
        raise RuntimeIdentityValidationError(
            "artifact must be present in the supplied goal"
        )
    resolved_session_id = session_id or (goal.session_id if goal is not None else None)
    goal_ref = (
        _goal_ref(goal.goal_id, session_id=goal.session_id) if goal is not None else None
    )
    return ArtifactDescriptor(
        ref=_artifact_ref(artifact.artifact_id, session_id=resolved_session_id),
        provenance=RuntimeProvenance(
            session_ref=_session_ref(resolved_session_id),
            goal_ref=goal_ref,
            source="goal_artifact",
        ),
        metadata={
            "description_present": artifact.description is not None,
            "source_metadata_present": bool(artifact.metadata),
        },
        artifact_kind=artifact.kind,
        uri_present=artifact.uri is not None,
        metadata_only=True,
    )


def describe_runtime_recovery_result(
    result: RuntimeRecoveryResult,
) -> RecoveryDescriptor:
    """Describe a recovery result without exposing transcript or worktree paths."""

    scope = result.transcript_scope
    session_ref = _required_session_ref(scope.session_id)
    return RecoveryDescriptor(
        ref=runtime_object_ref(
            "recovery",
            f"{scope.session_id}:runtime_recovery",
            session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
            agent_id=scope.agent_id,
        ),
        provenance=RuntimeProvenance(
            session_ref=session_ref,
            runtime_session_ref=_runtime_session_ref(
                scope.runtime_session_id,
                session_id=scope.session_id,
                agent_id=scope.agent_id,
            ),
            agent_ref=_agent_ref(
                scope.agent_id,
                session_id=scope.session_id,
                runtime_session_id=scope.runtime_session_id,
            ),
            source="runtime_recovery",
        ),
        metadata={
            "coordinator_runtime_restored": result.coordinator_runtime_restored,
            "transcript_message_count": len(result.replay.messages),
            "compact_boundary_count": len(result.replay.compact_boundaries),
            "content_replacement_count": len(result.replay.content_replacements),
        },
        transcript_path_present=result.transcript_path is not None,
        last_message_entry_id=result.last_message_entry_id,
        warning_count=len(result.warnings),
        restored_route_count=len(result.restored_route_records),
        restored_agent_name_count=len(result.restored_agent_names),
        restored_remote_task_count=len(result.restored_remote_task_ids),
        worktree_status_count=len(result.worktree_statuses),
        task_output_status_count=len(result.task_output_statuses),
        improvement_chain_status=(
            result.improvement_chain_recovery.status
            if result.improvement_chain_recovery is not None
            else None
        ),
    )


def _provenance_from_ids(
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
    agent_id: str | None = None,
    parent_agent_id: str | None = None,
    goal_id: str | None = None,
    transcript_entry_id: str | None = None,
    tool_use_id: str | None = None,
    task_id: str | None = None,
    event_id: str | None = None,
    turn_id: str | None = None,
    iteration: int | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    source: str | None = None,
) -> RuntimeProvenance:
    return RuntimeProvenance(
        session_ref=_session_ref(session_id),
        runtime_session_ref=_runtime_session_ref(
            runtime_session_id,
            session_id=session_id,
            agent_id=agent_id,
        ),
        agent_ref=_agent_ref(
            agent_id,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
        ),
        parent_agent_ref=_agent_ref(
            parent_agent_id,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
        ),
        goal_ref=_goal_ref(goal_id, session_id=session_id),
        transcript_entry_ref=_transcript_entry_ref(
            transcript_entry_id,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
            agent_id=agent_id,
        ),
        tool_use_ref=_tool_use_ref(tool_use_id, session_id=session_id),
        task_ref=_task_ref(task_id, session_id=session_id),
        event_ref=_event_ref(
            event_id,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
            agent_id=agent_id,
        ),
        turn_id=turn_id,
        iteration=iteration,
        span_id=span_id,
        parent_span_id=parent_span_id,
        source=source,
    )


def _session_ref(session_id: str | None) -> RuntimeObjectReference | None:
    if session_id is None:
        return None
    return _required_session_ref(session_id)


def _required_session_ref(session_id: str) -> RuntimeObjectReference:
    return runtime_object_ref("session", session_id, session_id=session_id)


def _runtime_session_ref(
    runtime_session_id: str | None,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> RuntimeObjectReference | None:
    if runtime_session_id is None:
        return None
    return _required_runtime_session_ref(
        runtime_session_id,
        session_id=session_id,
        agent_id=agent_id,
    )


def _required_runtime_session_ref(
    runtime_session_id: str,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
) -> RuntimeObjectReference:
    return runtime_object_ref(
        "runtime_session",
        runtime_session_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )


def _agent_ref(
    agent_id: str | None,
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
) -> RuntimeObjectReference | None:
    if agent_id is None:
        return None
    return _required_agent_ref(
        agent_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
    )


def _required_agent_ref(
    agent_id: str,
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
) -> RuntimeObjectReference:
    return runtime_object_ref(
        "agent",
        agent_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )


def _goal_ref(
    goal_id: str | None,
    *,
    session_id: str | None = None,
) -> RuntimeObjectReference | None:
    if goal_id is None:
        return None
    return _required_goal_ref(goal_id, session_id=session_id)


def _required_goal_ref(
    goal_id: str,
    *,
    session_id: str | None = None,
) -> RuntimeObjectReference:
    return runtime_object_ref("goal", goal_id, session_id=session_id)


def _transcript_entry_ref(
    entry_id: str | None,
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
    agent_id: str | None = None,
) -> RuntimeObjectReference | None:
    if entry_id is None:
        return None
    return runtime_object_ref(
        "transcript_entry",
        entry_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )


def _tool_use_ref(
    tool_use_id: str | None,
    *,
    session_id: str | None = None,
) -> RuntimeObjectReference | None:
    if tool_use_id is None:
        return None
    return runtime_object_ref("tool_use", tool_use_id, session_id=session_id)


def _task_ref(
    task_id: str | None,
    *,
    session_id: str | None = None,
) -> RuntimeObjectReference | None:
    if task_id is None:
        return None
    return runtime_object_ref("task", task_id, session_id=session_id)


def _artifact_ref(
    artifact_id: str,
    *,
    session_id: str | None = None,
) -> RuntimeObjectReference:
    return runtime_object_ref("artifact", artifact_id, session_id=session_id)


def _event_ref(
    event_id: str | None,
    *,
    session_id: str | None = None,
    runtime_session_id: str | None = None,
    agent_id: str | None = None,
) -> RuntimeObjectReference | None:
    if event_id is None:
        return None
    return runtime_object_ref(
        "event",
        event_id,
        session_id=session_id,
        runtime_session_id=runtime_session_id,
        agent_id=agent_id,
    )


def _optional_string_attr(value: object, name: str) -> str | None:
    raw = getattr(value, name, None)
    return raw if isinstance(raw, str) else None


__all__ = (
    "RuntimeHandlesLike",
    "describe_goal_artifact",
    "describe_goal_runtime",
    "describe_goal_state",
    "describe_kernel_event",
    "describe_runtime_handles",
    "describe_runtime_recovery_result",
    "describe_task_output_read_result",
    "describe_task_output_reference",
    "describe_task_state",
    "describe_transcript_entry",
    "describe_transcript_scope",
    "describe_transcript_search_match",
    "runtime_object_ref",
)
