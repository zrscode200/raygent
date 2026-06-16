"""Runtime identity, provenance, and descriptor data contracts.

This package owns stable, JSON-compatible facts about Raygent runtime objects.
The data contracts deliberately avoid reading stores, exposing raw local paths
by default, or inventing lifecycle/provenance that source objects do not own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from raygent_harness.core.model_types import FrozenJson, freeze_json

RUNTIME_IDENTITY_SCHEMA_VERSION = 1
DEFAULT_MAX_RUNTIME_DESCRIPTOR_METADATA_CHARS = 20_000

RuntimeObjectKind = Literal[
    "session",
    "runtime_session",
    "agent",
    "goal_runtime",
    "goal",
    "task",
    "tool_use",
    "transcript_entry",
    "task_output",
    "artifact",
    "recovery",
    "event",
]
RuntimeLifecycleCategory = Literal[
    "unknown",
    "live",
    "running",
    "completed",
    "failed",
    "cancelled",
    "closed",
    "recoverable",
    "unrecoverable",
    "blocked",
    "paused",
    "limited",
]
RuntimePathDisclosure = Literal["hidden", "local_debug"]
RuntimeObjectDescriptorType = Literal[
    "runtime_object",
    "session",
    "transcript_entry",
    "task",
    "task_output",
    "goal_runtime",
    "goal",
    "artifact",
    "recovery",
    "event",
]

_OBJECT_KINDS: frozenset[str] = frozenset(
    {
        "session",
        "runtime_session",
        "agent",
        "goal_runtime",
        "goal",
        "task",
        "tool_use",
        "transcript_entry",
        "task_output",
        "artifact",
        "recovery",
        "event",
    }
)
_LIFECYCLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "unknown",
        "live",
        "running",
        "completed",
        "failed",
        "cancelled",
        "closed",
        "recoverable",
        "unrecoverable",
        "blocked",
        "paused",
        "limited",
    }
)
_DESCRIPTOR_TYPES: frozenset[str] = frozenset(
    {
        "runtime_object",
        "session",
        "transcript_entry",
        "task",
        "task_output",
        "goal_runtime",
        "goal",
        "artifact",
        "recovery",
        "event",
    }
)


class RuntimeIdentityValidationError(ValueError):
    """Raised when runtime identity data violates the descriptor contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RuntimeObjectReference:
    """Stable opaque reference to one Raygent-owned runtime object."""

    kind: RuntimeObjectKind
    object_id: str
    session_id: str | None = None
    runtime_session_id: str | None = None
    agent_id: str | None = None
    schema_version: int = RUNTIME_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_literal(self.kind, _OBJECT_KINDS, "RuntimeObjectReference.kind")
        _require_non_empty(self.object_id, "RuntimeObjectReference.object_id")
        _require_optional_non_empty(self.session_id, "RuntimeObjectReference.session_id")
        _require_optional_non_empty(
            self.runtime_session_id,
            "RuntimeObjectReference.runtime_session_id",
        )
        _require_optional_non_empty(self.agent_id, "RuntimeObjectReference.agent_id")
        _require_positive(self.schema_version, "RuntimeObjectReference.schema_version")


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Lineage facts attached to a descriptor.

    Fields are optional because builders must never guess missing provenance.
    """

    session_ref: RuntimeObjectReference | None = None
    runtime_session_ref: RuntimeObjectReference | None = None
    agent_ref: RuntimeObjectReference | None = None
    parent_agent_ref: RuntimeObjectReference | None = None
    goal_ref: RuntimeObjectReference | None = None
    transcript_entry_ref: RuntimeObjectReference | None = None
    tool_use_ref: RuntimeObjectReference | None = None
    task_ref: RuntimeObjectReference | None = None
    event_ref: RuntimeObjectReference | None = None
    turn_id: str | None = None
    iteration: int | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        _require_reference_kind(self.session_ref, "session", "session_ref")
        _require_reference_kind(
            self.runtime_session_ref,
            "runtime_session",
            "runtime_session_ref",
        )
        _require_reference_kind(self.agent_ref, "agent", "agent_ref")
        _require_reference_kind(self.parent_agent_ref, "agent", "parent_agent_ref")
        _require_reference_kind(self.goal_ref, "goal", "goal_ref")
        _require_reference_kind(
            self.transcript_entry_ref,
            "transcript_entry",
            "transcript_entry_ref",
        )
        _require_reference_kind(self.tool_use_ref, "tool_use", "tool_use_ref")
        _require_reference_kind(self.task_ref, "task", "task_ref")
        _require_reference_kind(self.event_ref, "event", "event_ref")
        _require_optional_non_empty(self.turn_id, "RuntimeProvenance.turn_id")
        _require_optional_non_empty(self.span_id, "RuntimeProvenance.span_id")
        _require_optional_non_empty(
            self.parent_span_id,
            "RuntimeProvenance.parent_span_id",
        )
        _require_optional_non_empty(self.source, "RuntimeProvenance.source")
        if self.iteration is not None and self.iteration < 0:
            raise RuntimeIdentityValidationError("RuntimeProvenance.iteration must be >= 0")


@dataclass(frozen=True, slots=True)
class RuntimeLifecycleDescriptor:
    """Native lifecycle status plus a coarse cross-object category."""

    native_status: str | None = None
    category: RuntimeLifecycleCategory = "unknown"
    started_at: float | None = None
    ended_at: float | None = None
    updated_at: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_optional_non_empty(
            self.native_status,
            "RuntimeLifecycleDescriptor.native_status",
        )
        _require_literal(
            self.category,
            _LIFECYCLE_CATEGORIES,
            "RuntimeLifecycleDescriptor.category",
        )
        _require_optional_non_negative(
            self.started_at,
            "RuntimeLifecycleDescriptor.started_at",
        )
        _require_optional_non_negative(
            self.ended_at,
            "RuntimeLifecycleDescriptor.ended_at",
        )
        _require_optional_non_negative(
            self.updated_at,
            "RuntimeLifecycleDescriptor.updated_at",
        )
        _require_optional_non_empty(self.reason, "RuntimeLifecycleDescriptor.reason")


@dataclass(frozen=True, slots=True)
class RuntimeObjectDescriptor:
    """Base descriptor for one kernel object."""

    ref: RuntimeObjectReference
    provenance: RuntimeProvenance = field(default_factory=RuntimeProvenance)
    lifecycle: RuntimeLifecycleDescriptor = field(
        default_factory=RuntimeLifecycleDescriptor
    )
    created_at: float | None = None
    updated_at: float | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_optional_non_negative(self.created_at, "created_at")
        _require_optional_non_negative(self.updated_at, "updated_at")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "metadata",
                DEFAULT_MAX_RUNTIME_DESCRIPTOR_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class SessionDescriptor(RuntimeObjectDescriptor):
    """Descriptor for a Raygent session handle boundary."""

    cwd_path_present: bool = False
    transcript_path_present: bool = False
    output_dir_path_present: bool = False
    task_store_present: bool = False
    task_output_store_present: bool = False
    transcript_store_present: bool = False
    observability_present: bool = False
    goal_runtime_attached: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "session", "SessionDescriptor.ref")


@dataclass(frozen=True, slots=True)
class TranscriptEntryDescriptor(RuntimeObjectDescriptor):
    """Descriptor for a transcript entry without raw message content."""

    entry_type: str = ""
    role: str | None = None
    parent_entry_id: str | None = None
    logical_parent_entry_id: str | None = None
    provider_message_id_present: bool = False
    is_sidechain: bool | None = None
    source_path_present: bool = False
    cwd_path_present: bool = False
    version_present: bool = False
    message_fields_present: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "transcript_entry", "TranscriptEntryDescriptor.ref")
        _require_non_empty(
            self.entry_type,
            "TranscriptEntryDescriptor.entry_type",
        )
        _require_optional_non_empty(self.role, "TranscriptEntryDescriptor.role")
        _require_optional_non_empty(
            self.parent_entry_id,
            "TranscriptEntryDescriptor.parent_entry_id",
        )
        _require_optional_non_empty(
            self.logical_parent_entry_id,
            "TranscriptEntryDescriptor.logical_parent_entry_id",
        )
        has_message_only_fields = (
            self.role is not None
            or self.parent_entry_id is not None
            or self.logical_parent_entry_id is not None
            or self.provider_message_id_present
            or self.is_sidechain is not None
            or self.cwd_path_present
            or self.version_present
        )
        if self.entry_type != "message" and (
            self.message_fields_present or has_message_only_fields
        ):
            raise RuntimeIdentityValidationError(
                "message-only transcript fields require entry_type='message'"
            )
        if not self.message_fields_present and has_message_only_fields:
            raise RuntimeIdentityValidationError(
                "message-only transcript fields require message_fields_present=True"
            )


@dataclass(frozen=True, slots=True)
class TaskDescriptor(RuntimeObjectDescriptor):
    """Descriptor for a Raygent task state."""

    task_type: str = ""
    description: str = ""
    tool_use_id: str | None = None
    output_file_present: bool = False
    output_reference_present: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "task", "TaskDescriptor.ref")
        _require_non_empty(self.task_type, "TaskDescriptor.task_type")
        _require_optional_non_empty(self.tool_use_id, "TaskDescriptor.tool_use_id")


@dataclass(frozen=True, slots=True)
class TaskOutputDescriptor(RuntimeObjectDescriptor):
    """Descriptor for bounded task-output metadata."""

    store_kind: str = ""
    path_present: bool = False
    bytes_total: int | None = None
    start_offset: int | None = None
    bytes_read: int | None = None
    next_offset: int | None = None
    truncated_before: bool = False
    truncated_after: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "task_output", "TaskOutputDescriptor.ref")
        _require_non_empty(self.store_kind, "TaskOutputDescriptor.store_kind")
        _require_optional_int_min(self.bytes_total, 0, "TaskOutputDescriptor.bytes_total")
        _require_optional_int_min(
            self.start_offset,
            0,
            "TaskOutputDescriptor.start_offset",
        )
        _require_optional_int_min(self.bytes_read, 0, "TaskOutputDescriptor.bytes_read")
        _require_optional_int_min(self.next_offset, 0, "TaskOutputDescriptor.next_offset")


@dataclass(frozen=True, slots=True)
class GoalRuntimeDescriptor(RuntimeObjectDescriptor):
    """Descriptor for public/safe goal-runtime attachment facts."""

    attached: bool = False
    config_present: bool = False
    store_kind: str | None = None
    active_goal_supplied: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "goal_runtime", "GoalRuntimeDescriptor.ref")
        _require_optional_non_empty(
            self.store_kind,
            "GoalRuntimeDescriptor.store_kind",
        )


@dataclass(frozen=True, slots=True)
class GoalDescriptor(RuntimeObjectDescriptor):
    """Descriptor for metadata-only goal state facts."""

    native_goal_status: str = ""
    turn_count: int | None = None
    tokens_used: int | None = None
    pending_task_ids: tuple[str, ...] = ()
    artifact_refs: tuple[RuntimeObjectReference, ...] = ()

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "goal", "GoalDescriptor.ref")
        _require_non_empty(self.native_goal_status, "GoalDescriptor.native_goal_status")
        _require_optional_int_min(self.turn_count, 0, "GoalDescriptor.turn_count")
        _require_optional_int_min(self.tokens_used, 0, "GoalDescriptor.tokens_used")
        object.__setattr__(
            self,
            "pending_task_ids",
            _string_tuple(self.pending_task_ids, "GoalDescriptor.pending_task_ids"),
        )
        for ref in self.artifact_refs:
            _require_ref_kind(ref, "artifact", "GoalDescriptor.artifact_refs")
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor(RuntimeObjectDescriptor):
    """Descriptor for a Raygent-owned or Raygent-recorded artifact reference."""

    artifact_kind: str = ""
    uri_present: bool = False
    metadata_only: bool = True

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "artifact", "ArtifactDescriptor.ref")
        _require_non_empty(self.artifact_kind, "ArtifactDescriptor.artifact_kind")
        if not self.metadata_only:
            raise RuntimeIdentityValidationError(
                "ArtifactDescriptor must remain metadata_only in SDK-RUNTIME-001A"
            )


@dataclass(frozen=True, slots=True)
class RecoveryDescriptor(RuntimeObjectDescriptor):
    """Descriptor for runtime recovery summary facts."""

    transcript_path_present: bool = False
    last_message_entry_id: str | None = None
    warning_count: int = 0
    restored_route_count: int = 0
    restored_agent_name_count: int = 0
    restored_remote_task_count: int = 0
    worktree_status_count: int = 0
    task_output_status_count: int = 0
    improvement_chain_status: str | None = None

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "recovery", "RecoveryDescriptor.ref")
        _require_optional_non_empty(
            self.last_message_entry_id,
            "RecoveryDescriptor.last_message_entry_id",
        )
        _require_int_min(self.warning_count, 0, "RecoveryDescriptor.warning_count")
        _require_int_min(
            self.restored_route_count,
            0,
            "RecoveryDescriptor.restored_route_count",
        )
        _require_int_min(
            self.restored_agent_name_count,
            0,
            "RecoveryDescriptor.restored_agent_name_count",
        )
        _require_int_min(
            self.restored_remote_task_count,
            0,
            "RecoveryDescriptor.restored_remote_task_count",
        )
        _require_int_min(
            self.worktree_status_count,
            0,
            "RecoveryDescriptor.worktree_status_count",
        )
        _require_int_min(
            self.task_output_status_count,
            0,
            "RecoveryDescriptor.task_output_status_count",
        )
        _require_optional_non_empty(
            self.improvement_chain_status,
            "RecoveryDescriptor.improvement_chain_status",
        )


@dataclass(frozen=True, slots=True)
class EventDescriptor(RuntimeObjectDescriptor):
    """Descriptor for a supplied immutable kernel event object."""

    event_type: str = ""
    sequence: int | None = None
    event_source: str = ""
    content_policy: str | None = None
    data_present: bool = False

    def __post_init__(self) -> None:
        RuntimeObjectDescriptor.__post_init__(self)
        _require_ref_kind(self.ref, "event", "EventDescriptor.ref")
        _require_non_empty(self.event_type, "EventDescriptor.event_type")
        _require_optional_int_min(self.sequence, 1, "EventDescriptor.sequence")
        _require_non_empty(self.event_source, "EventDescriptor.event_source")
        _require_optional_non_empty(self.content_policy, "EventDescriptor.content_policy")


type RuntimeDescriptor = (
    RuntimeObjectDescriptor
    | SessionDescriptor
    | TranscriptEntryDescriptor
    | TaskDescriptor
    | TaskOutputDescriptor
    | GoalRuntimeDescriptor
    | GoalDescriptor
    | ArtifactDescriptor
    | RecoveryDescriptor
    | EventDescriptor
)


def runtime_lifecycle_category_for(
    kind: RuntimeObjectKind,
    native_status: str | None,
) -> RuntimeLifecycleCategory:
    """Return a coarse category while preserving native status elsewhere."""

    _require_literal(kind, _OBJECT_KINDS, "kind")
    status = native_status
    if status is None:
        return "completed" if kind == "event" else "unknown"
    normalized = status.strip().lower()
    if not normalized:
        return "completed" if kind == "event" else "unknown"
    if kind == "task":
        return cast(
            RuntimeLifecycleCategory,
            {
                "pending": "running",
                "running": "running",
                "completed": "completed",
                "failed": "failed",
                "killed": "cancelled",
            }.get(normalized, "unknown"),
        )
    if kind == "goal":
        return cast(
            RuntimeLifecycleCategory,
            {
                "active": "running",
                "paused": "paused",
                "blocked": "blocked",
                "usage_limited": "limited",
                "budget_limited": "limited",
                "complete": "completed",
                "cancelled": "cancelled",
                "failed": "failed",
            }.get(normalized, "unknown"),
        )
    if kind == "session":
        if normalized == "closed":
            return "closed"
        if normalized in {"active", "live", "running"}:
            return "live"
        return "unknown"
    if kind == "event":
        return "completed"
    if kind == "recovery" and normalized in {"recoverable", "unrecoverable"}:
        return cast(RuntimeLifecycleCategory, normalized)
    return "unknown"


def runtime_object_reference_to_dict(
    ref: RuntimeObjectReference,
) -> dict[str, object]:
    return {
        "schema_version": ref.schema_version,
        "kind": ref.kind,
        "object_id": ref.object_id,
        "session_id": ref.session_id,
        "runtime_session_id": ref.runtime_session_id,
        "agent_id": ref.agent_id,
    }


def runtime_object_reference_from_dict(
    data: Mapping[str, object],
) -> RuntimeObjectReference:
    return RuntimeObjectReference(
        kind=cast(
            RuntimeObjectKind,
            _literal_from_object(data["kind"], _OBJECT_KINDS, "kind"),
        ),
        object_id=_str_from_object(data["object_id"], "object_id"),
        session_id=_optional_str(data.get("session_id")),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        agent_id=_optional_str(data.get("agent_id")),
        schema_version=_int_from_object(
            data.get("schema_version", RUNTIME_IDENTITY_SCHEMA_VERSION),
            "schema_version",
        ),
    )


def runtime_provenance_to_dict(provenance: RuntimeProvenance) -> dict[str, object]:
    return {
        "session_ref": _optional_ref_to_dict(provenance.session_ref),
        "runtime_session_ref": _optional_ref_to_dict(provenance.runtime_session_ref),
        "agent_ref": _optional_ref_to_dict(provenance.agent_ref),
        "parent_agent_ref": _optional_ref_to_dict(provenance.parent_agent_ref),
        "goal_ref": _optional_ref_to_dict(provenance.goal_ref),
        "transcript_entry_ref": _optional_ref_to_dict(
            provenance.transcript_entry_ref
        ),
        "tool_use_ref": _optional_ref_to_dict(provenance.tool_use_ref),
        "task_ref": _optional_ref_to_dict(provenance.task_ref),
        "event_ref": _optional_ref_to_dict(provenance.event_ref),
        "turn_id": provenance.turn_id,
        "iteration": provenance.iteration,
        "span_id": provenance.span_id,
        "parent_span_id": provenance.parent_span_id,
        "source": provenance.source,
    }


def runtime_provenance_from_dict(data: Mapping[str, object]) -> RuntimeProvenance:
    return RuntimeProvenance(
        session_ref=_optional_ref_from_object(data.get("session_ref"), "session_ref"),
        runtime_session_ref=_optional_ref_from_object(
            data.get("runtime_session_ref"),
            "runtime_session_ref",
        ),
        agent_ref=_optional_ref_from_object(data.get("agent_ref"), "agent_ref"),
        parent_agent_ref=_optional_ref_from_object(
            data.get("parent_agent_ref"),
            "parent_agent_ref",
        ),
        goal_ref=_optional_ref_from_object(data.get("goal_ref"), "goal_ref"),
        transcript_entry_ref=_optional_ref_from_object(
            data.get("transcript_entry_ref"),
            "transcript_entry_ref",
        ),
        tool_use_ref=_optional_ref_from_object(
            data.get("tool_use_ref"),
            "tool_use_ref",
        ),
        task_ref=_optional_ref_from_object(data.get("task_ref"), "task_ref"),
        event_ref=_optional_ref_from_object(data.get("event_ref"), "event_ref"),
        turn_id=_optional_str(data.get("turn_id")),
        iteration=_optional_int(data.get("iteration"), "iteration"),
        span_id=_optional_str(data.get("span_id")),
        parent_span_id=_optional_str(data.get("parent_span_id")),
        source=_optional_str(data.get("source")),
    )


def runtime_lifecycle_descriptor_to_dict(
    lifecycle: RuntimeLifecycleDescriptor,
) -> dict[str, object]:
    return {
        "native_status": lifecycle.native_status,
        "category": lifecycle.category,
        "started_at": lifecycle.started_at,
        "ended_at": lifecycle.ended_at,
        "updated_at": lifecycle.updated_at,
        "reason": lifecycle.reason,
    }


def runtime_lifecycle_descriptor_from_dict(
    data: Mapping[str, object],
) -> RuntimeLifecycleDescriptor:
    return RuntimeLifecycleDescriptor(
        native_status=_optional_str(data.get("native_status")),
        category=cast(
            RuntimeLifecycleCategory,
            _literal_from_object(
                data.get("category", "unknown"),
                _LIFECYCLE_CATEGORIES,
                "category",
            ),
        ),
        started_at=_optional_float(data.get("started_at"), "started_at"),
        ended_at=_optional_float(data.get("ended_at"), "ended_at"),
        updated_at=_optional_float(data.get("updated_at"), "updated_at"),
        reason=_optional_str(data.get("reason")),
    )


def runtime_object_descriptor_to_dict(
    descriptor: RuntimeDescriptor,
) -> dict[str, object]:
    data = _base_descriptor_to_dict(descriptor)
    descriptor_type = _descriptor_type_for(descriptor)
    data["descriptor_type"] = descriptor_type
    if isinstance(descriptor, SessionDescriptor):
        data.update(
            {
                "cwd_path_present": descriptor.cwd_path_present,
                "transcript_path_present": descriptor.transcript_path_present,
                "output_dir_path_present": descriptor.output_dir_path_present,
                "task_store_present": descriptor.task_store_present,
                "task_output_store_present": descriptor.task_output_store_present,
                "transcript_store_present": descriptor.transcript_store_present,
                "observability_present": descriptor.observability_present,
                "goal_runtime_attached": descriptor.goal_runtime_attached,
            }
        )
    elif isinstance(descriptor, TranscriptEntryDescriptor):
        data.update(
            {
                "entry_type": descriptor.entry_type,
                "role": descriptor.role,
                "parent_entry_id": descriptor.parent_entry_id,
                "logical_parent_entry_id": descriptor.logical_parent_entry_id,
                "provider_message_id_present": (
                    descriptor.provider_message_id_present
                ),
                "is_sidechain": descriptor.is_sidechain,
                "source_path_present": descriptor.source_path_present,
                "cwd_path_present": descriptor.cwd_path_present,
                "version_present": descriptor.version_present,
                "message_fields_present": descriptor.message_fields_present,
            }
        )
    elif isinstance(descriptor, TaskDescriptor):
        data.update(
            {
                "task_type": descriptor.task_type,
                "description": descriptor.description,
                "tool_use_id": descriptor.tool_use_id,
                "output_file_present": descriptor.output_file_present,
                "output_reference_present": descriptor.output_reference_present,
            }
        )
    elif isinstance(descriptor, TaskOutputDescriptor):
        data.update(
            {
                "store_kind": descriptor.store_kind,
                "path_present": descriptor.path_present,
                "bytes_total": descriptor.bytes_total,
                "start_offset": descriptor.start_offset,
                "bytes_read": descriptor.bytes_read,
                "next_offset": descriptor.next_offset,
                "truncated_before": descriptor.truncated_before,
                "truncated_after": descriptor.truncated_after,
            }
        )
    elif isinstance(descriptor, GoalRuntimeDescriptor):
        data.update(
            {
                "attached": descriptor.attached,
                "config_present": descriptor.config_present,
                "store_kind": descriptor.store_kind,
                "active_goal_supplied": descriptor.active_goal_supplied,
            }
        )
    elif isinstance(descriptor, GoalDescriptor):
        data.update(
            {
                "native_goal_status": descriptor.native_goal_status,
                "turn_count": descriptor.turn_count,
                "tokens_used": descriptor.tokens_used,
                "pending_task_ids": list(descriptor.pending_task_ids),
                "artifact_refs": [
                    runtime_object_reference_to_dict(ref)
                    for ref in descriptor.artifact_refs
                ],
            }
        )
    elif isinstance(descriptor, ArtifactDescriptor):
        data.update(
            {
                "artifact_kind": descriptor.artifact_kind,
                "uri_present": descriptor.uri_present,
                "metadata_only": descriptor.metadata_only,
            }
        )
    elif isinstance(descriptor, RecoveryDescriptor):
        data.update(
            {
                "transcript_path_present": descriptor.transcript_path_present,
                "last_message_entry_id": descriptor.last_message_entry_id,
                "warning_count": descriptor.warning_count,
                "restored_route_count": descriptor.restored_route_count,
                "restored_agent_name_count": descriptor.restored_agent_name_count,
                "restored_remote_task_count": descriptor.restored_remote_task_count,
                "worktree_status_count": descriptor.worktree_status_count,
                "task_output_status_count": descriptor.task_output_status_count,
                "improvement_chain_status": descriptor.improvement_chain_status,
            }
        )
    elif isinstance(descriptor, EventDescriptor):
        data.update(
            {
                "event_type": descriptor.event_type,
                "sequence": descriptor.sequence,
                "event_source": descriptor.event_source,
                "content_policy": descriptor.content_policy,
                "data_present": descriptor.data_present,
            }
        )
    return data


def runtime_object_descriptor_from_dict(
    data: Mapping[str, object],
) -> RuntimeDescriptor:
    descriptor_type = cast(
        RuntimeObjectDescriptorType,
        _literal_from_object(
            data.get("descriptor_type", "runtime_object"),
            _DESCRIPTOR_TYPES,
            "descriptor_type",
        ),
    )
    ref = runtime_object_reference_from_dict(_mapping_field(data, "ref"))
    provenance = runtime_provenance_from_dict(
        _mapping_field(data, "provenance", default={})
    )
    lifecycle = runtime_lifecycle_descriptor_from_dict(
        _mapping_field(data, "lifecycle", default={})
    )
    created_at = _optional_float(data.get("created_at"), "created_at")
    updated_at = _optional_float(data.get("updated_at"), "updated_at")
    metadata = _metadata_from_object(data.get("metadata", {}), "metadata")
    if descriptor_type == "runtime_object":
        return RuntimeObjectDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )
    if descriptor_type == "session":
        return SessionDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            cwd_path_present=_bool_field(data, "cwd_path_present"),
            transcript_path_present=_bool_field(data, "transcript_path_present"),
            output_dir_path_present=_bool_field(data, "output_dir_path_present"),
            task_store_present=_bool_field(data, "task_store_present"),
            task_output_store_present=_bool_field(data, "task_output_store_present"),
            transcript_store_present=_bool_field(data, "transcript_store_present"),
            observability_present=_bool_field(data, "observability_present"),
            goal_runtime_attached=_bool_field(data, "goal_runtime_attached"),
        )
    if descriptor_type == "transcript_entry":
        return TranscriptEntryDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            entry_type=_str_from_object(data["entry_type"], "entry_type"),
            role=_optional_str(data.get("role")),
            parent_entry_id=_optional_str(data.get("parent_entry_id")),
            logical_parent_entry_id=_optional_str(
                data.get("logical_parent_entry_id")
            ),
            provider_message_id_present=_bool_field(
                data,
                "provider_message_id_present",
            ),
            is_sidechain=_optional_bool(data.get("is_sidechain"), "is_sidechain"),
            source_path_present=_bool_field(data, "source_path_present"),
            cwd_path_present=_bool_field(data, "cwd_path_present"),
            version_present=_bool_field(data, "version_present"),
            message_fields_present=_bool_field(data, "message_fields_present"),
        )
    if descriptor_type == "task":
        return TaskDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            task_type=_str_from_object(data["task_type"], "task_type"),
            description=_str_from_object(data.get("description", ""), "description"),
            tool_use_id=_optional_str(data.get("tool_use_id")),
            output_file_present=_bool_field(data, "output_file_present"),
            output_reference_present=_bool_field(data, "output_reference_present"),
        )
    if descriptor_type == "task_output":
        return TaskOutputDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            store_kind=_str_from_object(data["store_kind"], "store_kind"),
            path_present=_bool_field(data, "path_present"),
            bytes_total=_optional_int(data.get("bytes_total"), "bytes_total"),
            start_offset=_optional_int(data.get("start_offset"), "start_offset"),
            bytes_read=_optional_int(data.get("bytes_read"), "bytes_read"),
            next_offset=_optional_int(data.get("next_offset"), "next_offset"),
            truncated_before=_bool_field(data, "truncated_before"),
            truncated_after=_bool_field(data, "truncated_after"),
        )
    if descriptor_type == "goal_runtime":
        return GoalRuntimeDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            attached=_bool_field(data, "attached"),
            config_present=_bool_field(data, "config_present"),
            store_kind=_optional_str(data.get("store_kind")),
            active_goal_supplied=_bool_field(data, "active_goal_supplied"),
        )
    if descriptor_type == "goal":
        return GoalDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            native_goal_status=_str_from_object(
                data["native_goal_status"],
                "native_goal_status",
            ),
            turn_count=_optional_int(data.get("turn_count"), "turn_count"),
            tokens_used=_optional_int(data.get("tokens_used"), "tokens_used"),
            pending_task_ids=_string_tuple(
                data.get("pending_task_ids", ()),
                "pending_task_ids",
            ),
            artifact_refs=tuple(
                runtime_object_reference_from_dict(item)
                for item in _mapping_sequence(data.get("artifact_refs", ()), "artifact_refs")
            ),
        )
    if descriptor_type == "artifact":
        return ArtifactDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            artifact_kind=_str_from_object(data["artifact_kind"], "artifact_kind"),
            uri_present=_bool_field(data, "uri_present"),
            metadata_only=_bool_field(data, "metadata_only", default=True),
        )
    if descriptor_type == "recovery":
        return RecoveryDescriptor(
            ref=ref,
            provenance=provenance,
            lifecycle=lifecycle,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            transcript_path_present=_bool_field(data, "transcript_path_present"),
            last_message_entry_id=_optional_str(data.get("last_message_entry_id")),
            warning_count=_int_from_object(
                data.get("warning_count", 0),
                "warning_count",
            ),
            restored_route_count=_int_from_object(
                data.get("restored_route_count", 0),
                "restored_route_count",
            ),
            restored_agent_name_count=_int_from_object(
                data.get("restored_agent_name_count", 0),
                "restored_agent_name_count",
            ),
            restored_remote_task_count=_int_from_object(
                data.get("restored_remote_task_count", 0),
                "restored_remote_task_count",
            ),
            worktree_status_count=_int_from_object(
                data.get("worktree_status_count", 0),
                "worktree_status_count",
            ),
            task_output_status_count=_int_from_object(
                data.get("task_output_status_count", 0),
                "task_output_status_count",
            ),
            improvement_chain_status=_optional_str(data.get("improvement_chain_status")),
        )
    return EventDescriptor(
        ref=ref,
        provenance=provenance,
        lifecycle=lifecycle,
        created_at=created_at,
        updated_at=updated_at,
        metadata=metadata,
        event_type=_str_from_object(data["event_type"], "event_type"),
        sequence=_optional_int(data.get("sequence"), "sequence"),
        event_source=_str_from_object(data["event_source"], "event_source"),
        content_policy=_optional_str(data.get("content_policy")),
        data_present=_bool_field(data, "data_present"),
    )


def _base_descriptor_to_dict(descriptor: RuntimeObjectDescriptor) -> dict[str, object]:
    return {
        "ref": runtime_object_reference_to_dict(descriptor.ref),
        "provenance": runtime_provenance_to_dict(descriptor.provenance),
        "lifecycle": runtime_lifecycle_descriptor_to_dict(descriptor.lifecycle),
        "created_at": descriptor.created_at,
        "updated_at": descriptor.updated_at,
        "metadata": _metadata_to_dict(descriptor.metadata),
    }


def _descriptor_type_for(descriptor: RuntimeDescriptor) -> RuntimeObjectDescriptorType:
    if isinstance(descriptor, SessionDescriptor):
        return "session"
    if isinstance(descriptor, TranscriptEntryDescriptor):
        return "transcript_entry"
    if isinstance(descriptor, TaskDescriptor):
        return "task"
    if isinstance(descriptor, TaskOutputDescriptor):
        return "task_output"
    if isinstance(descriptor, GoalRuntimeDescriptor):
        return "goal_runtime"
    if isinstance(descriptor, GoalDescriptor):
        return "goal"
    if isinstance(descriptor, ArtifactDescriptor):
        return "artifact"
    if isinstance(descriptor, RecoveryDescriptor):
        return "recovery"
    if isinstance(descriptor, EventDescriptor):
        return "event"
    return "runtime_object"


def _optional_ref_to_dict(ref: RuntimeObjectReference | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return runtime_object_reference_to_dict(ref)


def _optional_ref_from_object(
    value: object,
    name: str,
) -> RuntimeObjectReference | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return runtime_object_reference_from_dict(cast(Mapping[str, object], value))


def _require_ref_kind(
    ref: RuntimeObjectReference,
    expected: RuntimeObjectKind,
    name: str,
) -> None:
    if ref.kind != expected:
        raise RuntimeIdentityValidationError(
            f"{name} must reference kind={expected!r}, got {ref.kind!r}"
        )


def _require_reference_kind(
    ref: RuntimeObjectReference | None,
    expected: RuntimeObjectKind,
    name: str,
) -> None:
    if ref is not None:
        _require_ref_kind(ref, expected, name)


def _require_literal(value: str, allowed: frozenset[str], name: str) -> str:
    if value not in allowed:
        raise RuntimeIdentityValidationError(f"{name} must be one of {sorted(allowed)}")
    return value


def _literal_from_object(value: object, allowed: frozenset[str], name: str) -> str:
    text = _str_from_object(value, name)
    return _require_literal(text, allowed, name)


def _require_non_empty(value: str, name: str) -> None:
    if value == "":
        raise RuntimeIdentityValidationError(f"{name} must be non-empty")


def _require_optional_non_empty(value: str | None, name: str) -> None:
    if value == "":
        raise RuntimeIdentityValidationError(f"{name} must be non-empty when present")


def _require_positive(value: int, name: str) -> None:
    if value < 1:
        raise RuntimeIdentityValidationError(f"{name} must be >= 1")


def _require_int_min(value: int, minimum: int, name: str) -> None:
    if value < minimum:
        raise RuntimeIdentityValidationError(f"{name} must be >= {minimum}")


def _require_optional_int_min(value: int | None, minimum: int, name: str) -> None:
    if value is not None:
        _require_int_min(value, minimum, name)


def _require_optional_non_negative(value: float | None, name: str) -> None:
    if value is not None and value < 0:
        raise RuntimeIdentityValidationError(f"{name} must be >= 0")


def _str_from_object(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be a string or None")
    return value


def _int_from_object(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _int_from_object(value, name)


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _bool_field(
    data: Mapping[str, object],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean or None")
    return value


def _mapping_field(
    data: Mapping[str, object],
    key: str,
    *,
    default: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    value = data.get(key, default)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return cast(Mapping[str, object], value)


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    sequence = cast(Sequence[object], value)
    result: list[Mapping[str, object]] = []
    for item in sequence:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} items must be mappings")
        result.append(cast(Mapping[str, object], item))
    return tuple(result)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    sequence = cast(Sequence[object], value)
    result: list[str] = []
    for item in sequence:
        if not isinstance(item, str):
            raise TypeError(f"{name} items must be strings")
        _require_non_empty(item, f"{name} item")
        result.append(item)
    return tuple(result)


def _metadata_from_object(value: object, name: str) -> Mapping[str, FrozenJson]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze_metadata(
        cast(Mapping[str, object], value),
        name,
        DEFAULT_MAX_RUNTIME_DESCRIPTOR_METADATA_CHARS,
    )


def _freeze_metadata(
    metadata: Mapping[str, object],
    name: str,
    max_chars: int,
) -> Mapping[str, FrozenJson]:
    frozen = freeze_json(metadata)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{name} must serialize to a JSON object")
    frozen_mapping = cast(Mapping[str, FrozenJson], frozen)
    if _json_text_chars(frozen_mapping) > max_chars:
        raise RuntimeIdentityValidationError(f"{name} exceeds {max_chars} chars")
    return frozen_mapping


def _metadata_to_dict(metadata: Mapping[str, FrozenJson]) -> dict[str, object]:
    return {key: _json_ready(value) for key, value in metadata.items()}


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_ready(item) for key, item in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [_json_ready(item) for item in sequence]
    raise TypeError(f"Expected JSON-like value, got {type(value).__name__}")


def _json_text_chars(value: Mapping[str, FrozenJson]) -> int:
    return len(
        json.dumps(
            _metadata_to_dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


__all__ = (
    "DEFAULT_MAX_RUNTIME_DESCRIPTOR_METADATA_CHARS",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "ArtifactDescriptor",
    "EventDescriptor",
    "GoalDescriptor",
    "GoalRuntimeDescriptor",
    "RecoveryDescriptor",
    "RuntimeDescriptor",
    "RuntimeIdentityValidationError",
    "RuntimeLifecycleCategory",
    "RuntimeLifecycleDescriptor",
    "RuntimeObjectDescriptor",
    "RuntimeObjectDescriptorType",
    "RuntimeObjectKind",
    "RuntimeObjectReference",
    "RuntimePathDisclosure",
    "RuntimeProvenance",
    "SessionDescriptor",
    "TaskDescriptor",
    "TaskOutputDescriptor",
    "TranscriptEntryDescriptor",
    "runtime_lifecycle_category_for",
    "runtime_lifecycle_descriptor_from_dict",
    "runtime_lifecycle_descriptor_to_dict",
    "runtime_object_descriptor_from_dict",
    "runtime_object_descriptor_to_dict",
    "runtime_object_reference_from_dict",
    "runtime_object_reference_to_dict",
    "runtime_provenance_from_dict",
    "runtime_provenance_to_dict",
)
