"""Bounded one-session runtime identity snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from raygent_harness.core.observability import KernelEvent
from raygent_harness.core.task import TaskStateBase
from raygent_harness.goals.models import GoalArtifact, GoalState
from raygent_harness.services.runtime_identity.builders import (
    RuntimeHandlesLike,
    describe_goal_artifact,
    describe_goal_runtime,
    describe_goal_state,
    describe_kernel_event,
    describe_runtime_handles,
    describe_runtime_recovery_result,
    describe_task_output_read_result,
    describe_task_output_reference,
    describe_task_state,
    describe_transcript_entry,
    describe_transcript_scope,
    describe_transcript_search_match,
)
from raygent_harness.services.runtime_identity.models import (
    RuntimeDescriptor,
    RuntimeIdentityValidationError,
    RuntimeObjectReference,
)
from raygent_harness.services.runtime_recovery import RuntimeRecoveryResult
from raygent_harness.services.task_output.store import (
    TaskOutputReadResult,
    TaskOutputReference,
)
from raygent_harness.services.transcript.models import TranscriptEntry
from raygent_harness.services.transcript.search import TranscriptSearchMatch


class RuntimeIdentitySessionLike(Protocol):
    """Minimal session shape consumed by `describe_runtime_session`."""

    @property
    def handles(self) -> RuntimeHandlesLike:
        """Inspectable runtime handles for the session."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeIdentitySnapshotOptions:
    """Bounds and include flags for one runtime identity snapshot."""

    include_transcript_scope: bool = True
    include_tasks: bool = True
    include_goal_runtime: bool = True
    include_active_goal: bool = True
    max_tasks: int = 50
    max_supplied_items: int = 50
    max_descriptors: int = 256

    def __post_init__(self) -> None:
        _require_non_negative(self.max_tasks, "max_tasks")
        _require_non_negative(self.max_supplied_items, "max_supplied_items")
        if self.max_descriptors < 1:
            raise RuntimeIdentityValidationError("max_descriptors must be >= 1")


@dataclass(frozen=True, slots=True)
class RuntimeIdentitySnapshot:
    """Bounded descriptor snapshot for one supplied runtime session."""

    session_id: str
    descriptors: tuple[RuntimeDescriptor, ...]
    warnings: tuple[str, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise RuntimeIdentityValidationError("RuntimeIdentitySnapshot.session_id")
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def describe_runtime_session(
    session_or_handles: RuntimeIdentitySessionLike | RuntimeHandlesLike,
    *,
    options: RuntimeIdentitySnapshotOptions | None = None,
    active_goal: GoalState | None = None,
    transcript_entries: Sequence[TranscriptEntry] = (),
    transcript_search_matches: Sequence[TranscriptSearchMatch] = (),
    task_output_references: Sequence[TaskOutputReference] = (),
    task_output_read_results: Sequence[TaskOutputReadResult] = (),
    kernel_events: Sequence[KernelEvent] = (),
    goal_artifacts: Sequence[GoalArtifact] = (),
    runtime_recovery_results: Sequence[RuntimeRecoveryResult] = (),
) -> RuntimeIdentitySnapshot:
    """Describe one supplied runtime session without reading stores.

    Optional sequences must already be bounded by the caller; this helper
    applies defensive count bounds and records warnings when it truncates.
    """

    resolved_options = options or RuntimeIdentitySnapshotOptions()
    handles = _resolve_handles(session_or_handles)
    descriptors: list[RuntimeDescriptor] = []
    warnings: list[str] = []
    seen_refs: set[tuple[str, str, str | None, str | None, str | None]] = set()

    truncated = _append_descriptor(
        descriptors,
        describe_runtime_handles(handles),
        seen_refs,
        max_descriptors=resolved_options.max_descriptors,
        warnings=warnings,
    )

    if handles.transcript_scope is not None:
        _require_session_match(
            handles.transcript_scope.session_id,
            handles.session_id,
            "handles.transcript_scope.session_id",
        )
    if active_goal is not None:
        _require_session_match(
            active_goal.session_id,
            handles.session_id,
            "active_goal.session_id",
        )
    if resolved_options.include_transcript_scope and handles.transcript_scope is not None:
        truncated = _append_descriptors(
            descriptors,
            describe_transcript_scope(handles.transcript_scope),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warning_prefix="transcript_scope",
            warnings=warnings,
        ) or truncated

    if resolved_options.include_tasks:
        truncated = _append_task_store_descriptors(
            descriptors,
            handles.task_store,
            session_id=handles.session_id,
            max_tasks=resolved_options.max_tasks,
            max_descriptors=resolved_options.max_descriptors,
            seen_refs=seen_refs,
            warnings=warnings,
        ) or truncated

    if resolved_options.include_goal_runtime:
        goal_for_runtime = (
            active_goal
            if resolved_options.include_active_goal and handles.goal_runtime is not None
            else None
        )
        truncated = _append_descriptor(
            descriptors,
            describe_goal_runtime(
                handles.goal_runtime,
                session_id=handles.session_id,
                active_goal=goal_for_runtime,
            ),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    if resolved_options.include_active_goal and active_goal is not None:
        truncated = _append_descriptor(
            descriptors,
            describe_goal_state(active_goal),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated

    supplied_groups: tuple[tuple[str, Sequence[object]], ...] = (
        ("transcript_entries", transcript_entries),
        ("transcript_search_matches", transcript_search_matches),
        ("task_output_references", task_output_references),
        ("task_output_read_results", task_output_read_results),
        ("kernel_events", kernel_events),
        ("goal_artifacts", goal_artifacts),
        ("runtime_recovery_results", runtime_recovery_results),
    )
    for name, items in supplied_groups:
        if len(items) > resolved_options.max_supplied_items:
            warnings.append(f"{name}_truncated")
            truncated = True

    for entry in tuple(transcript_entries)[: resolved_options.max_supplied_items]:
        _require_session_match(entry.session_id, handles.session_id, "transcript_entry")
        truncated = _append_descriptor(
            descriptors,
            describe_transcript_entry(entry),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for match in tuple(transcript_search_matches)[: resolved_options.max_supplied_items]:
        _require_session_match(
            match.session_id,
            handles.session_id,
            "transcript_search_match",
        )
        truncated = _append_descriptor(
            descriptors,
            describe_transcript_search_match(match),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for reference in tuple(task_output_references)[: resolved_options.max_supplied_items]:
        truncated = _append_descriptor(
            descriptors,
            describe_task_output_reference(reference, session_id=handles.session_id),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for result in tuple(task_output_read_results)[: resolved_options.max_supplied_items]:
        truncated = _append_descriptor(
            descriptors,
            describe_task_output_read_result(result, session_id=handles.session_id),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for event in tuple(kernel_events)[: resolved_options.max_supplied_items]:
        if event.session_id is not None:
            _require_session_match(event.session_id, handles.session_id, "kernel_event")
        truncated = _append_descriptor(
            descriptors,
            describe_kernel_event(event),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for artifact in tuple(goal_artifacts)[: resolved_options.max_supplied_items]:
        if active_goal is None:
            raise RuntimeIdentityValidationError(
                "active_goal is required when supplying goal_artifacts"
            )
        truncated = _append_descriptor(
            descriptors,
            describe_goal_artifact(artifact, goal=active_goal),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated
    for result in tuple(runtime_recovery_results)[: resolved_options.max_supplied_items]:
        _require_session_match(
            result.transcript_scope.session_id,
            handles.session_id,
            "runtime_recovery_result",
        )
        truncated = _append_descriptor(
            descriptors,
            describe_runtime_recovery_result(result),
            seen_refs,
            max_descriptors=resolved_options.max_descriptors,
            warnings=warnings,
        ) or truncated

    return RuntimeIdentitySnapshot(
        session_id=handles.session_id,
        descriptors=tuple(descriptors),
        warnings=tuple(warnings),
        truncated=truncated,
    )


def _resolve_handles(
    session_or_handles: RuntimeIdentitySessionLike | RuntimeHandlesLike,
) -> RuntimeHandlesLike:
    candidate = getattr(session_or_handles, "handles", session_or_handles)
    return cast(RuntimeHandlesLike, candidate)


def _append_task_store_descriptors(
    descriptors: list[RuntimeDescriptor],
    task_store: object,
    *,
    session_id: str,
    max_tasks: int,
    max_descriptors: int,
    seen_refs: set[tuple[str, str, str | None, str | None, str | None]],
    warnings: list[str],
) -> bool:
    tasks = getattr(task_store, "tasks", None)
    if tasks is None:
        warnings.append("task_store_tasks_unavailable")
        return False
    if not isinstance(tasks, Mapping):
        warnings.append("task_store_tasks_unavailable")
        return False
    task_mapping = cast(Mapping[object, object], tasks)
    task_values = _bounded_task_values(task_mapping, max_tasks=max_tasks)
    truncated = False
    if len(task_values) > max_tasks:
        warnings.append("tasks_truncated")
        truncated = True
    for task in task_values[:max_tasks]:
        if not isinstance(task, TaskStateBase):
            warnings.append("task_store_item_unsupported")
            continue
        descriptor = describe_task_state(task, session_id=session_id)
        truncated = _append_descriptor(
            descriptors,
            descriptor,
            seen_refs,
            max_descriptors=max_descriptors,
            warnings=warnings,
        ) or truncated
    return truncated


def _append_descriptors(
    descriptors: list[RuntimeDescriptor],
    new_descriptors: Sequence[RuntimeDescriptor],
    seen_refs: set[tuple[str, str, str | None, str | None, str | None]],
    *,
    max_descriptors: int,
    warning_prefix: str,
    warnings: list[str],
) -> bool:
    truncated = False
    for descriptor in new_descriptors:
        truncated = _append_descriptor(
            descriptors,
            descriptor,
            seen_refs,
            max_descriptors=max_descriptors,
            warnings=warnings,
            warning_prefix=warning_prefix,
        ) or truncated
    return truncated


def _append_descriptor(
    descriptors: list[RuntimeDescriptor],
    descriptor: RuntimeDescriptor,
    seen_refs: set[tuple[str, str, str | None, str | None, str | None]],
    *,
    max_descriptors: int,
    warnings: list[str],
    warning_prefix: str = "descriptors",
) -> bool:
    ref_key = _ref_key(descriptor.ref)
    if ref_key in seen_refs:
        return False
    if len(descriptors) >= max_descriptors:
        warning = f"{warning_prefix}_truncated"
        if warning not in warnings:
            warnings.append(warning)
        return True
    descriptors.append(descriptor)
    seen_refs.add(ref_key)
    return False


def _ref_key(
    ref: RuntimeObjectReference,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        ref.kind,
        ref.object_id,
        ref.session_id,
        ref.runtime_session_id,
        ref.agent_id,
    )


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise RuntimeIdentityValidationError(f"{name} must be >= 0")


def _require_session_match(
    supplied_session_id: str,
    expected_session_id: str,
    source: str,
) -> None:
    if supplied_session_id != expected_session_id:
        raise RuntimeIdentityValidationError(
            f"{source} must match snapshot session_id"
        )


def _bounded_task_values(
    task_mapping: Mapping[object, object],
    *,
    max_tasks: int,
) -> tuple[object, ...]:
    if max_tasks < 0:
        raise RuntimeIdentityValidationError("max_tasks must be >= 0")
    limit = max_tasks + 1
    values: list[object] = []
    for task in task_mapping.values():
        values.append(task)
        if len(values) >= limit:
            break
    return tuple(values)


__all__ = (
    "RuntimeIdentitySessionLike",
    "RuntimeIdentitySnapshot",
    "RuntimeIdentitySnapshotOptions",
    "describe_runtime_session",
)
