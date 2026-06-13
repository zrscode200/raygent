"""Optional runtime bridge contracts for bounded self-improvement.

The RSI-006A layer defines data and protocol seams only. It does not wire the
query loop, SDK factory, goal runtime, tools, concrete filesystem writers,
shell runners, Git, network, or CI into self-improvement behavior.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, Protocol, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.core.observability import KernelEventBus, KernelEventContext
from raygent_harness.improvement import (
    ImprovementEvidence,
    ImprovementEvidenceBounds,
    ImprovementEvidenceSource,
    ImprovementRequiredPermission,
    improvement_evidence_from_dict,
    improvement_evidence_text_chars,
    improvement_evidence_to_dict,
)
from raygent_harness.services.task_output import TaskOutputStore
from raygent_harness.services.transcript import (
    TranscriptSearchCompactMode,
    TranscriptSearchMatch,
    TranscriptSearchOrder,
    TranscriptSearchRequest,
    TranscriptSearchScope,
    TranscriptSearchService,
)

IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION = 1

DEFAULT_MAX_EVIDENCE_COLLECTION_ITEMS = 12
DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS = 1_000
DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS = 1_000
DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS = 4_000
DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_METADATA_CHARS = 12_000
DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS = 20
DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_CHARS = 12_000

DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS = 20_000
DEFAULT_MAX_IMPROVEMENT_RUNTIME_PAYLOAD_REF_CHARS = 1_000
DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS = 1_000
DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS = 20
DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS = 4_000
DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS = 50
TRANSCRIPT_SEARCH_MIN_SNIPPET_CHARS = 16

ImprovementRuntimeRecordKind = Literal[
    "evidence_collected",
    "proposal_recorded",
    "gate_evaluated",
    "candidate_planned",
    "worktree_allocated",
    "materialization_recorded",
    "verification_recorded",
    "outcome_derived",
    "archive_recorded",
    "promotion_recorded",
    "runtime_blocked",
    "runtime_not_enabled",
]
ImprovementRuntimeTransitionStatus = Literal["completed", "blocked", "not_enabled"]
ImprovementRuntimeRecoveryStatus = Literal["not_found", "recovered", "blocked"]
ImprovementTaskOutputReadMode = Literal["tail", "range"]
ImprovementRuntimePermissionStatus = Literal[
    "not_required",
    "required",
    "approved",
    "blocked",
]

_RUNTIME_RECORD_KINDS: frozenset[str] = frozenset(
    {
        "evidence_collected",
        "proposal_recorded",
        "gate_evaluated",
        "candidate_planned",
        "worktree_allocated",
        "materialization_recorded",
        "verification_recorded",
        "outcome_derived",
        "archive_recorded",
        "promotion_recorded",
        "runtime_blocked",
        "runtime_not_enabled",
    }
)
_RUNTIME_TRANSITION_STATUSES: frozenset[str] = frozenset(
    {"completed", "blocked", "not_enabled"}
)
_RUNTIME_RECOVERY_STATUSES: frozenset[str] = frozenset(
    {"not_found", "recovered", "blocked"}
)
_EVIDENCE_SOURCES: frozenset[str] = frozenset(
    {
        "transcript",
        "observability",
        "task_output",
        "verification",
        "user_report",
        "cost_usage",
        "other",
    }
)
_TASK_OUTPUT_READ_MODES: frozenset[str] = frozenset({"tail", "range"})
_PERMISSION_STATUSES: frozenset[str] = frozenset(
    {"not_required", "required", "approved", "blocked"}
)
_REQUIRED_PERMISSIONS: frozenset[str] = frozenset(
    {
        "none",
        "human_review",
        "model_provider",
        "filesystem_mutation",
        "shell",
        "worktree",
        "commit",
        "network",
        "external_service",
    }
)
_UNSAFE_RUNTIME_PERMISSION_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "permission_report",
        "permission_reports",
        "permission_summary",
        "approved_permissions",
        "supplied_approved_permissions",
        "approval",
        "approvals",
        "approver",
        "approver_identity",
    }
)
_RAW_OBSERVABILITY_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "content",
        "output",
        "tool_input",
        "tool_result",
        "transcript",
    }
)
_REDACTION_MARKER_KEYS: frozenset[str] = frozenset(
    {
        "redacted",
        "summary",
        "reason",
        "kind",
        "digest",
        "chars",
        "bytes",
        "items",
    }
)


class ImprovementRuntimeValidationError(ValueError):
    """Raised when runtime bridge data violates the RSI-006A contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementEvidenceCollectionBounds:
    """Named source-collection bounds before concrete evidence adapters land."""

    max_items: int = DEFAULT_MAX_EVIDENCE_COLLECTION_ITEMS
    max_excerpt_chars: int = DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS
    max_source_uri_chars: int = DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS
    max_item_metadata_chars: int = DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS
    max_total_metadata_chars: int = DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_METADATA_CHARS
    max_warnings: int = DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS
    max_total_chars: int = DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_CHARS
    proposal_evidence_bounds: ImprovementEvidenceBounds = field(
        default_factory=ImprovementEvidenceBounds
    )

    def __post_init__(self) -> None:
        _require_minimum(self.max_items, 1, "ImprovementEvidenceCollectionBounds.max_items")
        _require_minimum(
            self.max_excerpt_chars,
            1,
            "ImprovementEvidenceCollectionBounds.max_excerpt_chars",
        )
        _require_minimum(
            self.max_source_uri_chars,
            1,
            "ImprovementEvidenceCollectionBounds.max_source_uri_chars",
        )
        _require_minimum(
            self.max_item_metadata_chars,
            1,
            "ImprovementEvidenceCollectionBounds.max_item_metadata_chars",
        )
        _require_minimum(
            self.max_total_metadata_chars,
            self.max_item_metadata_chars,
            "ImprovementEvidenceCollectionBounds.max_total_metadata_chars",
        )
        _require_minimum(
            self.max_warnings,
            0,
            "ImprovementEvidenceCollectionBounds.max_warnings",
        )
        _require_minimum(
            self.max_total_chars,
            self.proposal_evidence_bounds.max_item_text_chars,
            "ImprovementEvidenceCollectionBounds.max_total_chars",
        )
        if self.max_items > self.proposal_evidence_bounds.max_items:
            raise ImprovementRuntimeValidationError(
                "ImprovementEvidenceCollectionBounds.max_items must not exceed "
                "proposal_evidence_bounds.max_items"
            )
        if self.max_total_chars > self.proposal_evidence_bounds.max_total_text_chars:
            raise ImprovementRuntimeValidationError(
                "ImprovementEvidenceCollectionBounds.max_total_chars must not exceed "
                "proposal_evidence_bounds.max_total_text_chars"
            )


@dataclass(frozen=True, slots=True)
class ImprovementEvidenceCollectionRequest:
    """Request passed to caller-owned evidence source adapters."""

    request_id: str
    session_id: str
    runtime_session_id: str | None = None
    source_kinds: tuple[ImprovementEvidenceSource, ...] = ()
    query: str | None = None
    bounds: ImprovementEvidenceCollectionBounds = field(
        default_factory=ImprovementEvidenceCollectionBounds
    )
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.request_id,
            "ImprovementEvidenceCollectionRequest.request_id",
        )
        _require_non_empty(
            self.session_id,
            "ImprovementEvidenceCollectionRequest.session_id",
        )
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementEvidenceCollectionRequest.runtime_session_id",
            )
        object.__setattr__(
            self,
            "source_kinds",
            _source_kind_tuple(self.source_kinds, "source_kinds"),
        )
        if self.query is not None:
            _require_non_empty(self.query, "ImprovementEvidenceCollectionRequest.query")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementEvidenceCollectionRequest.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementEvidenceCollectionResult:
    """Bounded evidence returned by one caller-owned source adapter."""

    request_id: str
    session_id: str
    evidence: tuple[ImprovementEvidence, ...]
    source_id: str | None = None
    runtime_session_id: str | None = None
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.request_id,
            "ImprovementEvidenceCollectionResult.request_id",
        )
        _require_non_empty(
            self.session_id,
            "ImprovementEvidenceCollectionResult.session_id",
        )
        if self.source_id is not None:
            _require_non_empty(
                self.source_id,
                "ImprovementEvidenceCollectionResult.source_id",
            )
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementEvidenceCollectionResult.runtime_session_id",
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementEvidenceCollectionResult.warnings",
                max_count=DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementEvidenceCollectionResult.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class BoundedImprovementEvidenceCollection:
    """Validated source evidence plus collection accounting."""

    evidence: tuple[ImprovementEvidence, ...]
    total_text_chars: int
    total_metadata_chars: int
    warnings: tuple[str, ...] = ()
    truncated: bool = False


class ImprovementEvidenceSourceAdapter(Protocol):
    """Caller-owned source adapter for bounded improvement evidence."""

    async def collect(
        self,
        request: ImprovementEvidenceCollectionRequest,
    ) -> ImprovementEvidenceCollectionResult:
        """Collect bounded evidence without invoking Raygent tools."""
        ...


@dataclass(frozen=True, slots=True)
class TranscriptSearchImprovementEvidenceAdapter:
    """Convert bounded transcript search matches into improvement evidence."""

    search_service: TranscriptSearchService
    roles: tuple[str, ...] = ("user", "assistant")
    compact_mode: TranscriptSearchCompactMode = "active"
    order: TranscriptSearchOrder = "newest_first"
    include_main: bool = True
    sidechain_agent_ids: tuple[str, ...] = ()
    include_all_sidechains: bool = False
    source_id: str = "transcript_search"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "roles",
            _bounded_string_tuple(
                self.roles,
                "TranscriptSearchImprovementEvidenceAdapter.roles",
                max_count=20,
                max_chars=80,
            ),
        )
        object.__setattr__(
            self,
            "sidechain_agent_ids",
            _bounded_string_tuple(
                self.sidechain_agent_ids,
                "TranscriptSearchImprovementEvidenceAdapter.sidechain_agent_ids",
                max_count=50,
                max_chars=200,
            ),
        )
        _require_literal(
            self.compact_mode,
            frozenset({"active", "full"}),
            "TranscriptSearchImprovementEvidenceAdapter.compact_mode",
        )
        _require_literal(
            self.order,
            frozenset({"newest_first", "oldest_first"}),
            "TranscriptSearchImprovementEvidenceAdapter.order",
        )
        _require_non_empty(
            self.source_id,
            "TranscriptSearchImprovementEvidenceAdapter.source_id",
        )

    async def collect(
        self,
        request: ImprovementEvidenceCollectionRequest,
    ) -> ImprovementEvidenceCollectionResult:
        """Search transcript snippets only when the caller requested transcripts."""

        if not _source_requested(request, "transcript"):
            return _collection_result(request, source_id=self.source_id)
        if request.query is None or not request.query.strip():
            return _collection_result(
                request,
                source_id=self.source_id,
                warnings=("transcript evidence query is required",),
            )

        search_budget = _transcript_search_budget(request.bounds)
        if isinstance(search_budget, str):
            return _collection_result(
                request,
                source_id=self.source_id,
                warnings=(search_budget,),
            )
        max_snippet_chars, max_total_snippet_chars = search_budget
        result = await self.search_service.search(
            TranscriptSearchRequest(
                query=request.query,
                scope=TranscriptSearchScope(
                    session_id=request.session_id,
                    runtime_session_id=request.runtime_session_id,
                    include_main=self.include_main,
                    sidechain_agent_ids=self.sidechain_agent_ids,
                    include_all_sidechains=self.include_all_sidechains,
                ),
                max_results=request.bounds.max_items,
                max_snippet_chars=max_snippet_chars,
                max_total_snippet_chars=max_total_snippet_chars,
                compact_mode=self.compact_mode,
                order=self.order,
                roles=self.roles,
            )
        )

        evidence = tuple(_transcript_match_to_evidence(match) for match in result.matches)
        warnings = list(result.warnings)
        if result.dropped_match_count:
            warnings.append(
                f"transcript search dropped {result.dropped_match_count} matches"
            )
        if result.truncated:
            warnings.append("transcript search results were truncated")
        return _collection_result(
            request,
            source_id=self.source_id,
            evidence=evidence,
            warnings=tuple(warnings),
            truncated=result.truncated,
            metadata={
                "scanned_entry_count": result.scanned_entry_count,
                "matched_entry_count": result.matched_entry_count,
                "dropped_match_count": result.dropped_match_count,
                "scope_count": len(result.scopes_searched),
                "read_count": len(result.read_stats),
            },
        )


@dataclass(frozen=True, slots=True)
class ImprovementTaskOutputEvidenceTarget:
    """One explicit task-output read target for improvement evidence."""

    task_id: str
    mode: ImprovementTaskOutputReadMode = "tail"
    offset: int | None = None
    max_bytes: int | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_bounded_non_empty(
            self.task_id,
            "ImprovementTaskOutputEvidenceTarget.task_id",
            DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS,
            "DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS",
        )
        _require_literal(
            self.mode,
            _TASK_OUTPUT_READ_MODES,
            "ImprovementTaskOutputEvidenceTarget.mode",
        )
        if self.mode == "range":
            if self.offset is None:
                raise ImprovementRuntimeValidationError(
                    "ImprovementTaskOutputEvidenceTarget.offset is required for range reads"
                )
            _require_minimum(
                self.offset,
                0,
                "ImprovementTaskOutputEvidenceTarget.offset",
            )
        elif self.offset is not None:
            raise ImprovementRuntimeValidationError(
                "ImprovementTaskOutputEvidenceTarget.offset is only valid for range reads"
            )
        if self.max_bytes is not None:
            _require_minimum(
                self.max_bytes,
                1,
                "ImprovementTaskOutputEvidenceTarget.max_bytes",
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementTaskOutputEvidenceTarget.metadata",
                DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskOutputImprovementEvidenceAdapter:
    """Read explicit bounded task-output targets into improvement evidence."""

    store: TaskOutputStore
    targets: tuple[ImprovementTaskOutputEvidenceTarget, ...]
    source_id: str = "task_output"

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not targets:
            raise ImprovementRuntimeValidationError(
                "TaskOutputImprovementEvidenceAdapter requires explicit targets"
            )
        object.__setattr__(self, "targets", targets)
        _require_non_empty(
            self.source_id,
            "TaskOutputImprovementEvidenceAdapter.source_id",
        )

    async def collect(
        self,
        request: ImprovementEvidenceCollectionRequest,
    ) -> ImprovementEvidenceCollectionResult:
        """Read only construction-time task-output targets."""

        if not _source_requested(request, "task_output"):
            return _collection_result(request, source_id=self.source_id)

        evidence: list[ImprovementEvidence] = []
        warnings: list[str] = []
        truncated = False
        for target in self.targets:
            max_bytes = _task_output_max_bytes(target, request.bounds)
            if target.mode == "tail":
                read = await self.store.read_tail(target.task_id, max_bytes=max_bytes)
            else:
                read = await self.store.read_range(
                    target.task_id,
                    offset=target.offset or 0,
                    max_bytes=max_bytes,
                )
            if read.bytes_read == 0:
                warnings.append(f"task output empty or unavailable: {target.task_id}")
                continue
            excerpt, excerpt_truncated = _truncate_text(
                read.content.decode("utf-8", errors="replace"),
                request.bounds.max_excerpt_chars,
            )
            if not excerpt.strip():
                warnings.append(f"task output empty or unavailable: {target.task_id}")
                continue
            truncated = (
                truncated
                or read.truncated_before
                or read.truncated_after
                or excerpt_truncated
            )
            safe_task_id = _safe_uri_component(target.task_id)
            evidence.append(
                ImprovementEvidence(
                    evidence_id=(
                        f"iev_task_output_{safe_task_id}_"
                        f"{read.start_offset}_{read.next_offset}"
                    ),
                    source="task_output",
                    summary=f"Task output {target.mode} read for {target.task_id}",
                    excerpt=excerpt,
                    source_uri=(
                        f"task-output://{safe_task_id}?"
                        f"start={read.start_offset}&next={read.next_offset}"
                    ),
                    created_at=0.0,
                    metadata={
                        "task_id": target.task_id,
                        "read_mode": target.mode,
                        "start_offset": read.start_offset,
                        "next_offset": read.next_offset,
                        "bytes_read": read.bytes_read,
                        "bytes_total": read.bytes_total,
                        "truncated_before": read.truncated_before,
                        "truncated_after": read.truncated_after or excerpt_truncated,
                        "target_metadata": target.metadata,
                    },
                )
            )
        return _collection_result(
            request,
            source_id=self.source_id,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            truncated=truncated,
        )


@dataclass(frozen=True, slots=True)
class ImprovementObservabilitySnapshot:
    """Caller-supplied metadata snapshot for observability evidence."""

    event_id: str
    event_type: str
    summary: str
    created_at: float | None = None
    source_uri: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_bounded_non_empty(
            self.event_id,
            "ImprovementObservabilitySnapshot.event_id",
            DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS,
            "DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS",
        )
        _require_bounded_non_empty(
            self.event_type,
            "ImprovementObservabilitySnapshot.event_type",
            DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS,
            "DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS",
        )
        _require_non_empty(self.summary, "ImprovementObservabilitySnapshot.summary")
        if self.created_at is not None and self.created_at < 0:
            raise ValueError("ImprovementObservabilitySnapshot.created_at must be >= 0")
        if self.source_uri is not None:
            _require_bounded_non_empty(
                self.source_uri,
                "ImprovementObservabilitySnapshot.source_uri",
                DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS,
                "DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS",
            )
        _reject_raw_observability_metadata(
            self.metadata,
            "ImprovementObservabilitySnapshot.metadata",
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementObservabilitySnapshot.metadata",
                DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservabilitySnapshotImprovementEvidenceAdapter:
    """Convert caller-supplied observability snapshots into evidence."""

    snapshots: tuple[ImprovementObservabilitySnapshot, ...]
    source_id: str = "observability"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        _require_non_empty(
            self.source_id,
            "ObservabilitySnapshotImprovementEvidenceAdapter.source_id",
        )

    async def collect(
        self,
        request: ImprovementEvidenceCollectionRequest,
    ) -> ImprovementEvidenceCollectionResult:
        """Return metadata-only observability snapshots selected by the caller."""

        if not _source_requested(request, "observability"):
            return _collection_result(request, source_id=self.source_id)

        evidence = tuple(
            ImprovementEvidence(
                evidence_id=f"iev_observability_{_safe_uri_component(snapshot.event_id)}",
                source="observability",
                summary=snapshot.summary,
                source_uri=snapshot.source_uri
                or (
                    "observability://"
                    f"{_safe_uri_component(snapshot.event_type)}/"
                    f"{_safe_uri_component(snapshot.event_id)}"
                ),
                created_at=snapshot.created_at if snapshot.created_at is not None else 0.0,
                metadata={
                    "event_id": snapshot.event_id,
                    "event_type": snapshot.event_type,
                    "snapshot_metadata": snapshot.metadata,
                },
            )
            for snapshot in self.snapshots
        )
        return _collection_result(
            request,
            source_id=self.source_id,
            evidence=evidence,
            truncated=False,
            metadata={"snapshot_count": len(self.snapshots)},
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRecord:
    """Immutable envelope for one record in an explicit improvement chain."""

    record_id: str
    record_kind: ImprovementRuntimeRecordKind
    session_id: str
    sequence: int
    payload: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)
    payload_ref: str | None = None
    schema_version: int = IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION
    runtime_session_id: str | None = None
    run_id: str | None = None
    proposal_id: str | None = None
    candidate_id: str | None = None
    stage_id: str | None = None
    warnings: tuple[str, ...] = ()
    stop_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.record_id, "ImprovementRuntimeRecord.record_id")
        _require_literal(
            self.record_kind,
            _RUNTIME_RECORD_KINDS,
            "ImprovementRuntimeRecord.record_kind",
        )
        _require_non_empty(self.session_id, "ImprovementRuntimeRecord.session_id")
        _require_minimum(self.sequence, 1, "ImprovementRuntimeRecord.sequence")
        for field_name, value in (
            ("runtime_session_id", self.runtime_session_id),
            ("run_id", self.run_id),
            ("proposal_id", self.proposal_id),
            ("candidate_id", self.candidate_id),
            ("stage_id", self.stage_id),
        ):
            if value is not None:
                _require_non_empty(value, f"ImprovementRuntimeRecord.{field_name}")
        frozen_payload = _freeze_metadata(
            self.payload,
            "ImprovementRuntimeRecord.payload",
            DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
        )
        if self.payload_ref is not None:
            _require_bounded_non_empty(
                self.payload_ref,
                "ImprovementRuntimeRecord.payload_ref",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_PAYLOAD_REF_CHARS,
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_PAYLOAD_REF_CHARS",
            )
        if not frozen_payload and self.payload_ref is None:
            raise ImprovementRuntimeValidationError(
                "ImprovementRuntimeRecord requires payload or payload_ref"
            )
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementRuntimeRecord.warnings",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        if self.stop_reason is not None:
            _require_bounded_non_empty(
                self.stop_reason,
                "ImprovementRuntimeRecord.stop_reason",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS,
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS",
            )
        if self.created_at < 0:
            raise ValueError("ImprovementRuntimeRecord.created_at must be >= 0")
        _reject_unsafe_runtime_permission_metadata(
            self.metadata,
            "ImprovementRuntimeRecord.metadata",
            allow_permission_summary=True,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeRecord.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRecordQuery:
    """Selector for loading bounded improvement-chain records."""

    session_id: str | None = None
    run_id: str | None = None
    proposal_id: str | None = None
    candidate_id: str | None = None
    max_records: int = DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS

    def __post_init__(self) -> None:
        selectors = (
            self.session_id,
            self.run_id,
            self.proposal_id,
            self.candidate_id,
        )
        if not any(value is not None for value in selectors):
            raise ImprovementRuntimeValidationError(
                "ImprovementRuntimeRecordQuery requires at least one selector"
            )
        for field_name, value in (
            ("session_id", self.session_id),
            ("run_id", self.run_id),
            ("proposal_id", self.proposal_id),
            ("candidate_id", self.candidate_id),
        ):
            if value is not None:
                _require_non_empty(value, f"ImprovementRuntimeRecordQuery.{field_name}")
        _require_minimum(
            self.max_records,
            1,
            "ImprovementRuntimeRecordQuery.max_records",
        )
        if self.max_records > DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS:
            raise ImprovementRuntimeValidationError(
                "ImprovementRuntimeRecordQuery.max_records must not exceed "
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS"
            )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeChainSummary:
    """Bounded recovery-friendly summary of a stopped improvement chain."""

    session_id: str
    record_count: int
    status: ImprovementRuntimeTransitionStatus
    schema_version: int = IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION
    runtime_session_id: str | None = None
    run_id: str | None = None
    proposal_id: str | None = None
    candidate_id: str | None = None
    last_record_id: str | None = None
    last_sequence: int | None = None
    last_record_kind: ImprovementRuntimeRecordKind | None = None
    next_record_kind: ImprovementRuntimeRecordKind | None = None
    blocked_reason: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.session_id, "ImprovementRuntimeChainSummary.session_id")
        _require_minimum(
            self.record_count,
            0,
            "ImprovementRuntimeChainSummary.record_count",
        )
        _require_literal(
            self.status,
            _RUNTIME_TRANSITION_STATUSES,
            "ImprovementRuntimeChainSummary.status",
        )
        for field_name, value in (
            ("runtime_session_id", self.runtime_session_id),
            ("run_id", self.run_id),
            ("proposal_id", self.proposal_id),
            ("candidate_id", self.candidate_id),
            ("last_record_id", self.last_record_id),
        ):
            if value is not None:
                _require_non_empty(value, f"ImprovementRuntimeChainSummary.{field_name}")
        if self.last_sequence is not None:
            _require_minimum(
                self.last_sequence,
                1,
                "ImprovementRuntimeChainSummary.last_sequence",
            )
        for field_name, value in (
            ("last_record_kind", self.last_record_kind),
            ("next_record_kind", self.next_record_kind),
        ):
            if value is not None:
                _require_literal(
                    value,
                    _RUNTIME_RECORD_KINDS,
                    f"ImprovementRuntimeChainSummary.{field_name}",
                )
        if self.blocked_reason is not None:
            _require_bounded_non_empty(
                self.blocked_reason,
                "ImprovementRuntimeChainSummary.blocked_reason",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS,
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS",
            )
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementRuntimeChainSummary.warnings",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        _reject_unsafe_runtime_permission_metadata(
            self.metadata,
            "ImprovementRuntimeChainSummary.metadata",
            allow_permission_summary=True,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeChainSummary.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimePermissionRequirement:
    """Advisory permission requirement for an explicit improvement stage."""

    stage_id: str
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    record_kind: ImprovementRuntimeRecordKind | None = None
    reason: str | None = None
    source_record_id: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.stage_id,
            "ImprovementRuntimePermissionRequirement.stage_id",
        )
        object.__setattr__(
            self,
            "required_permissions",
            _permission_tuple(
                self.required_permissions,
                "ImprovementRuntimePermissionRequirement.required_permissions",
                allow_empty=False,
            ),
        )
        if self.record_kind is not None:
            _require_literal(
                self.record_kind,
                _RUNTIME_RECORD_KINDS,
                "ImprovementRuntimePermissionRequirement.record_kind",
            )
        if self.reason is not None:
            _require_bounded_non_empty(
                self.reason,
                "ImprovementRuntimePermissionRequirement.reason",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS,
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS",
            )
        if self.source_record_id is not None:
            _require_non_empty(
                self.source_record_id,
                "ImprovementRuntimePermissionRequirement.source_record_id",
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimePermissionRequirement.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimePermissionSummary:
    """Sanitized permission metadata safe for durable runtime records."""

    stage_id: str
    status: ImprovementRuntimePermissionStatus
    required_permission_count: int
    missing_permission_count: int
    extra_permission_count: int
    requirement_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.stage_id, "ImprovementRuntimePermissionSummary.stage_id")
        _require_literal(
            self.status,
            _PERMISSION_STATUSES,
            "ImprovementRuntimePermissionSummary.status",
        )
        _require_minimum(
            self.required_permission_count,
            0,
            "ImprovementRuntimePermissionSummary.required_permission_count",
        )
        _require_minimum(
            self.missing_permission_count,
            0,
            "ImprovementRuntimePermissionSummary.missing_permission_count",
        )
        _require_minimum(
            self.extra_permission_count,
            0,
            "ImprovementRuntimePermissionSummary.extra_permission_count",
        )
        object.__setattr__(
            self,
            "requirement_labels",
            _bounded_string_tuple(
                self.requirement_labels,
                "ImprovementRuntimePermissionSummary.requirement_labels",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimePermissionReport:
    """Return-only advisory preflight report for improvement permissions."""

    request_id: str
    session_id: str
    stage_id: str
    status: ImprovementRuntimePermissionStatus
    requirements: tuple[ImprovementRuntimePermissionRequirement, ...] = ()
    supplied_approved_permissions: tuple[ImprovementRequiredPermission, ...] = ()
    missing_permissions: tuple[ImprovementRequiredPermission, ...] = ()
    extra_permissions: tuple[ImprovementRequiredPermission, ...] = ()
    runtime_session_id: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "ImprovementRuntimePermissionReport.request_id")
        _require_non_empty(self.session_id, "ImprovementRuntimePermissionReport.session_id")
        _require_non_empty(self.stage_id, "ImprovementRuntimePermissionReport.stage_id")
        _require_literal(
            self.status,
            _PERMISSION_STATUSES,
            "ImprovementRuntimePermissionReport.status",
        )
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementRuntimePermissionReport.runtime_session_id",
            )
        requirements = tuple(self.requirements)
        for requirement in requirements:
            if requirement.stage_id != self.stage_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimePermissionReport.requirements stage_id "
                    "must match report stage_id"
                )
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(
            self,
            "supplied_approved_permissions",
            _permission_tuple(
                self.supplied_approved_permissions,
                "ImprovementRuntimePermissionReport.supplied_approved_permissions",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "missing_permissions",
            _permission_tuple(
                self.missing_permissions,
                "ImprovementRuntimePermissionReport.missing_permissions",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "extra_permissions",
            _permission_tuple(
                self.extra_permissions,
                "ImprovementRuntimePermissionReport.extra_permissions",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementRuntimePermissionReport.warnings",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimePermissionReport.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )

    def to_summary(self) -> ImprovementRuntimePermissionSummary:
        """Return durable-safe metadata that omits approval-adjacent facts."""

        return improvement_runtime_permission_report_to_summary(self)


@dataclass(frozen=True, slots=True)
class ImprovementRuntimePermissionPolicy:
    """Evaluate advisory permission facts without authorizing later stages."""

    allow_extra_permissions: bool = False

    def evaluate(
        self,
        *,
        request_id: str,
        session_id: str,
        stage_id: str,
        requirements: Sequence[ImprovementRuntimePermissionRequirement] = (),
        approved_permissions: Sequence[ImprovementRequiredPermission] = (),
        runtime_session_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementRuntimePermissionReport:
        """Return a preflight report; stage services still validate approvals."""

        _require_non_empty(stage_id, "ImprovementRuntimePermissionPolicy.stage_id")
        requirement_items = tuple(requirements)
        for requirement in requirement_items:
            if requirement.stage_id != stage_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimePermissionPolicy requirements stage_id "
                    "must match stage_id"
                )
        supplied = _permission_tuple(
            approved_permissions,
            "ImprovementRuntimePermissionPolicy.approved_permissions",
            allow_empty=True,
        )
        required = _permission_set_from_requirements(requirement_items)
        warnings: list[str] = []

        if "none" in required and len(required) > 1:
            warnings.append("permission requirement none cannot be combined")
            status: ImprovementRuntimePermissionStatus = "blocked"
            required_without_none = tuple(
                permission for permission in sorted(required) if permission != "none"
            )
            missing = _permission_tuple(
                required_without_none,
                "ImprovementRuntimePermissionPolicy.missing_permissions",
                allow_empty=True,
            )
            extra = _extra_permissions(supplied, required)
        elif not required or required == {"none"}:
            extra = _extra_permissions(supplied, {"none"})
            missing = ()
            status = "not_required" if not extra else "blocked"
        else:
            missing = _missing_permissions(required, supplied)
            extra = _extra_permissions(supplied, required)
            if not supplied:
                status = "required"
            elif missing or (extra and not self.allow_extra_permissions):
                status = "blocked"
            else:
                status = "approved"

        if extra and self.allow_extra_permissions:
            warnings.append("extra approved permissions were ignored by advisory policy")

        return ImprovementRuntimePermissionReport(
            request_id=request_id,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
            stage_id=stage_id,
            status=status,
            requirements=requirement_items,
            supplied_approved_permissions=supplied,
            missing_permissions=missing,
            extra_permissions=extra,
            warnings=tuple(warnings),
            metadata=_empty_metadata() if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeObservabilityEvent:
    """Metadata-only transition event for explicit improvement runtime calls."""

    request_id: str
    session_id: str
    status: ImprovementRuntimeTransitionStatus
    runtime_session_id: str | None = None
    record_ids: tuple[str, ...] = ()
    last_record_kind: ImprovementRuntimeRecordKind | None = None
    evidence_count: int = 0
    warning_count: int = 0
    truncated: bool = False
    stage_id: str | None = None
    permission_status: ImprovementRuntimePermissionStatus | None = None
    permission_required_count: int = 0
    permission_missing_count: int = 0
    permission_extra_count: int = 0

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "ImprovementRuntimeObservabilityEvent.request_id")
        _require_non_empty(self.session_id, "ImprovementRuntimeObservabilityEvent.session_id")
        _require_literal(
            self.status,
            _RUNTIME_TRANSITION_STATUSES,
            "ImprovementRuntimeObservabilityEvent.status",
        )
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementRuntimeObservabilityEvent.runtime_session_id",
            )
        object.__setattr__(
            self,
            "record_ids",
            _bounded_string_tuple(
                self.record_ids,
                "ImprovementRuntimeObservabilityEvent.record_ids",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_PAYLOAD_REF_CHARS,
            ),
        )
        if self.last_record_kind is not None:
            _require_literal(
                self.last_record_kind,
                _RUNTIME_RECORD_KINDS,
                "ImprovementRuntimeObservabilityEvent.last_record_kind",
            )
        _require_minimum(
            self.evidence_count,
            0,
            "ImprovementRuntimeObservabilityEvent.evidence_count",
        )
        _require_minimum(
            self.warning_count,
            0,
            "ImprovementRuntimeObservabilityEvent.warning_count",
        )
        if self.stage_id is not None:
            _require_non_empty(self.stage_id, "ImprovementRuntimeObservabilityEvent.stage_id")
        if self.permission_status is not None:
            _require_literal(
                self.permission_status,
                _PERMISSION_STATUSES,
                "ImprovementRuntimeObservabilityEvent.permission_status",
            )
        _require_minimum(
            self.permission_required_count,
            0,
            "ImprovementRuntimeObservabilityEvent.permission_required_count",
        )
        _require_minimum(
            self.permission_missing_count,
            0,
            "ImprovementRuntimeObservabilityEvent.permission_missing_count",
        )
        _require_minimum(
            self.permission_extra_count,
            0,
            "ImprovementRuntimeObservabilityEvent.permission_extra_count",
        )


class ImprovementRuntimeObservabilitySink(Protocol):
    """Caller-owned sink for metadata-only improvement transition events."""

    def emit_transition(self, event: ImprovementRuntimeObservabilityEvent) -> None:
        """Emit one metadata-only transition event."""
        ...


@dataclass(frozen=True, slots=True)
class KernelEventImprovementRuntimeObserver:
    """Emit improvement runtime transitions through a kernel event bus."""

    event_bus: KernelEventBus
    event_type: str = "improvement_runtime.transition"
    source: str = "improvement_runtime"

    def __post_init__(self) -> None:
        _require_non_empty(
            self.event_type,
            "KernelEventImprovementRuntimeObserver.event_type",
        )
        _require_non_empty(self.source, "KernelEventImprovementRuntimeObserver.source")

    def emit_transition(self, event: ImprovementRuntimeObservabilityEvent) -> None:
        """Emit bounded metadata; `KernelEventBus` swallows sink failures."""

        self.event_bus.emit(
            self.event_type,
            context=KernelEventContext(
                session_id=event.session_id,
                runtime_session_id=event.runtime_session_id,
                source=self.source,
            ),
            data=improvement_runtime_observability_event_to_dict(event),
            content_policy="metadata_only",
            source=self.source,
        )


class ImprovementRecordStore(Protocol):
    """Caller-owned store for immutable improvement-chain envelopes."""

    async def append_record(
        self,
        record: ImprovementRuntimeRecord,
    ) -> ImprovementRuntimeRecord:
        """Persist one immutable improvement-chain record."""
        ...

    async def load_records(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> tuple[ImprovementRuntimeRecord, ...]:
        """Load bounded records matching a chain selector."""
        ...

    async def summarize_chain(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> ImprovementRuntimeChainSummary | None:
        """Return a bounded recovery summary for the selected chain."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRecoveryRequest:
    """Read-only request to recover a stopped explicit improvement chain."""

    request_id: str
    record_store: ImprovementRecordStore
    query: ImprovementRuntimeRecordQuery
    expected_session_id: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.request_id,
            "ImprovementRuntimeRecoveryRequest.request_id",
        )
        if self.expected_session_id is not None:
            _require_non_empty(
                self.expected_session_id,
                "ImprovementRuntimeRecoveryRequest.expected_session_id",
            )
        _reject_unsafe_runtime_permission_metadata(
            self.metadata,
            "ImprovementRuntimeRecoveryRequest.metadata",
            allow_permission_summary=False,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeRecoveryRequest.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRecoveryResult:
    """Bounded read-only recovery state for one explicit improvement chain."""

    request_id: str
    status: ImprovementRuntimeRecoveryStatus
    query: ImprovementRuntimeRecordQuery
    session_id: str | None = None
    expected_session_id: str | None = None
    runtime_session_id: str | None = None
    run_id: str | None = None
    proposal_id: str | None = None
    candidate_id: str | None = None
    summary: ImprovementRuntimeChainSummary | None = None
    records: tuple[ImprovementRuntimeRecord, ...] = ()
    last_record_id: str | None = None
    last_sequence: int | None = None
    last_record_kind: ImprovementRuntimeRecordKind | None = None
    last_completed_record_id: str | None = None
    last_completed_sequence: int | None = None
    last_completed_record_kind: ImprovementRuntimeRecordKind | None = None
    next_record_kind: ImprovementRuntimeRecordKind | None = None
    permission_summary: ImprovementRuntimePermissionSummary | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.request_id,
            "ImprovementRuntimeRecoveryResult.request_id",
        )
        _require_literal(
            self.status,
            _RUNTIME_RECOVERY_STATUSES,
            "ImprovementRuntimeRecoveryResult.status",
        )
        for field_name, value in (
            ("session_id", self.session_id),
            ("expected_session_id", self.expected_session_id),
            ("runtime_session_id", self.runtime_session_id),
            ("run_id", self.run_id),
            ("proposal_id", self.proposal_id),
            ("candidate_id", self.candidate_id),
            ("last_record_id", self.last_record_id),
            ("last_completed_record_id", self.last_completed_record_id),
        ):
            if value is not None:
                _require_non_empty(
                    value,
                    f"ImprovementRuntimeRecoveryResult.{field_name}",
                )
        for field_name, value in (
            ("last_sequence", self.last_sequence),
            ("last_completed_sequence", self.last_completed_sequence),
        ):
            if value is not None:
                _require_minimum(
                    value,
                    1,
                    f"ImprovementRuntimeRecoveryResult.{field_name}",
                )
        for field_name, value in (
            ("last_record_kind", self.last_record_kind),
            ("last_completed_record_kind", self.last_completed_record_kind),
            ("next_record_kind", self.next_record_kind),
        ):
            if value is not None:
                _require_literal(
                    value,
                    _RUNTIME_RECORD_KINDS,
                    f"ImprovementRuntimeRecoveryResult.{field_name}",
                )
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementRuntimeRecoveryResult.warnings",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        _reject_unsafe_runtime_permission_metadata(
            self.metadata,
            "ImprovementRuntimeRecoveryResult.metadata",
            allow_permission_summary=False,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeRecoveryResult.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRecoveryService:
    """Read-only recovery helper for caller-owned improvement record stores."""

    async def recover(
        self,
        request: ImprovementRuntimeRecoveryRequest,
    ) -> ImprovementRuntimeRecoveryResult:
        """Load and summarize a stopped explicit improvement chain."""

        query_conflict = _expected_session_query_conflict(request)
        if query_conflict is not None:
            return _blocked_recovery_result(request, query_conflict)

        try:
            records = tuple(await request.record_store.load_records(request.query))
            summary = await request.record_store.summarize_chain(request.query)
        except Exception as exc:  # pragma: no cover - exact store errors are caller-owned.
            return _blocked_recovery_result(
                request,
                "improvement record store recovery failed",
                metadata={"error_type": type(exc).__name__},
            )

        try:
            _validate_recovery_records(request.query, records)
            if summary is not None:
                _validate_recovery_summary(request.query, summary)
            _validate_recovery_session_linkage(request, records, summary)
            ordered_records, ordering_warnings = _ordered_recovery_records(records)
            if not ordered_records and summary is None:
                return ImprovementRuntimeRecoveryResult(
                    request_id=request.request_id,
                    status="not_found",
                    query=request.query,
                    session_id=request.expected_session_id or request.query.session_id,
                    expected_session_id=request.expected_session_id,
                    warnings=("no improvement runtime records matched the query",),
                    metadata=request.metadata,
                )
            resolved_summary = summary or _summary_from_recovery_records(ordered_records)
            last_record = ordered_records[-1] if ordered_records else None
            last_completed = _last_completed_recovery_record(
                ordered_records,
                resolved_summary,
            )
            session_id = _recovery_session_id(
                request,
                ordered_records,
                resolved_summary,
            )
            return ImprovementRuntimeRecoveryResult(
                request_id=request.request_id,
                status="recovered",
                query=request.query,
                session_id=session_id,
                expected_session_id=request.expected_session_id,
                runtime_session_id=_recovery_runtime_session_id(
                    ordered_records,
                    resolved_summary,
                ),
                run_id=_recovery_selector_value(
                    "run_id",
                    request.query.run_id,
                    ordered_records,
                    resolved_summary,
                ),
                proposal_id=_recovery_selector_value(
                    "proposal_id",
                    request.query.proposal_id,
                    ordered_records,
                    resolved_summary,
                ),
                candidate_id=_recovery_selector_value(
                    "candidate_id",
                    request.query.candidate_id,
                    ordered_records,
                    resolved_summary,
                ),
                summary=resolved_summary,
                records=ordered_records,
                last_record_id=(
                    last_record.record_id
                    if last_record is not None
                    else resolved_summary.last_record_id
                ),
                last_sequence=(
                    last_record.sequence
                    if last_record is not None
                    else resolved_summary.last_sequence
                ),
                last_record_kind=(
                    last_record.record_kind
                    if last_record is not None
                    else resolved_summary.last_record_kind
                ),
                last_completed_record_id=(
                    last_completed.record_id
                    if isinstance(last_completed, ImprovementRuntimeRecord)
                    else resolved_summary.last_record_id
                    if last_completed is resolved_summary
                    else None
                ),
                last_completed_sequence=(
                    last_completed.sequence
                    if isinstance(last_completed, ImprovementRuntimeRecord)
                    else resolved_summary.last_sequence
                    if last_completed is resolved_summary
                    else None
                ),
                last_completed_record_kind=(
                    last_completed.record_kind
                    if isinstance(last_completed, ImprovementRuntimeRecord)
                    else resolved_summary.last_record_kind
                    if last_completed is resolved_summary
                    else None
                ),
                next_record_kind=resolved_summary.next_record_kind,
                permission_summary=_permission_summary_from_recovery_record(last_record),
                warnings=_combined_recovery_warnings(
                    ordering_warnings,
                    ordered_records,
                    resolved_summary,
                ),
                metadata=request.metadata,
            )
        except ImprovementRuntimeValidationError as exc:
            return _blocked_recovery_result(request, str(exc))


async def recover_improvement_runtime_chain(
    request: ImprovementRuntimeRecoveryRequest,
) -> ImprovementRuntimeRecoveryResult:
    """Recover a stopped explicit improvement chain from an injected store."""

    return await ImprovementRuntimeRecoveryService().recover(request)


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeBridgeConfig:
    """Explicit opt-in configuration for the improvement runtime bridge."""

    enabled: bool = False
    record_store: ImprovementRecordStore | None = None
    evidence_sources: tuple[ImprovementEvidenceSourceAdapter, ...] = ()
    observability_sink: ImprovementRuntimeObservabilitySink | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_sources", tuple(self.evidence_sources))
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeBridgeConfig.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeRequest:
    """One explicit bridge transition request."""

    request_id: str
    session_id: str
    collection_request: ImprovementEvidenceCollectionRequest | None = None
    enabled: bool = True
    runtime_session_id: str | None = None
    record_store: ImprovementRecordStore | None = None
    evidence_sources: tuple[ImprovementEvidenceSourceAdapter, ...] = ()
    permission_report: ImprovementRuntimePermissionReport | None = None
    record_sequence: int = 1
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "ImprovementRuntimeRequest.request_id")
        _require_non_empty(self.session_id, "ImprovementRuntimeRequest.session_id")
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementRuntimeRequest.runtime_session_id",
            )
        if self.collection_request is not None:
            if self.collection_request.session_id != self.session_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeRequest.collection_request.session_id must "
                    "match session_id"
                )
            collection_runtime_session_id = self.collection_request.runtime_session_id
            if self.runtime_session_id is None and collection_runtime_session_id is not None:
                object.__setattr__(
                    self,
                    "runtime_session_id",
                    collection_runtime_session_id,
                )
            elif self.runtime_session_id is not None and (
                collection_runtime_session_id != self.runtime_session_id
            ):
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeRequest.collection_request.runtime_session_id "
                    "must match runtime_session_id"
                )
        object.__setattr__(self, "evidence_sources", tuple(self.evidence_sources))
        if self.permission_report is not None:
            if self.permission_report.request_id != self.request_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeRequest.permission_report.request_id must "
                    "match request_id"
                )
            if self.permission_report.session_id != self.session_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeRequest.permission_report.session_id must "
                    "match session_id"
                )
            if self.permission_report.runtime_session_id != self.runtime_session_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeRequest.permission_report.runtime_session_id "
                    "must match runtime_session_id"
                )
        _require_minimum(
            self.record_sequence,
            1,
            "ImprovementRuntimeRequest.record_sequence",
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeRequest.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeTransitionResult:
    """Result for one explicit bridge transition."""

    request_id: str
    session_id: str
    status: ImprovementRuntimeTransitionStatus
    runtime_session_id: str | None = None
    records: tuple[ImprovementRuntimeRecord, ...] = ()
    summary: ImprovementRuntimeChainSummary | None = None
    evidence: tuple[ImprovementEvidence, ...] = ()
    permission_report: ImprovementRuntimePermissionReport | None = None
    warnings: tuple[str, ...] = ()
    blocked_reason: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.request_id,
            "ImprovementRuntimeTransitionResult.request_id",
        )
        _require_non_empty(
            self.session_id,
            "ImprovementRuntimeTransitionResult.session_id",
        )
        _require_literal(
            self.status,
            _RUNTIME_TRANSITION_STATUSES,
            "ImprovementRuntimeTransitionResult.status",
        )
        if self.runtime_session_id is not None:
            _require_non_empty(
                self.runtime_session_id,
                "ImprovementRuntimeTransitionResult.runtime_session_id",
            )
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.permission_report is not None:
            if self.permission_report.request_id != self.request_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeTransitionResult.permission_report.request_id "
                    "must match request_id"
                )
            if self.permission_report.session_id != self.session_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeTransitionResult.permission_report.session_id "
                    "must match session_id"
                )
            if self.permission_report.runtime_session_id != self.runtime_session_id:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRuntimeTransitionResult.permission_report."
                    "runtime_session_id must match runtime_session_id"
                )
        object.__setattr__(
            self,
            "warnings",
            _bounded_string_tuple(
                self.warnings,
                "ImprovementRuntimeTransitionResult.warnings",
                max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
            ),
        )
        if self.blocked_reason is not None:
            _require_bounded_non_empty(
                self.blocked_reason,
                "ImprovementRuntimeTransitionResult.blocked_reason",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS,
                "DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS",
            )
        if self.status == "blocked" and self.blocked_reason is None:
            raise ImprovementRuntimeValidationError(
                "blocked ImprovementRuntimeTransitionResult requires blocked_reason"
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeTransitionResult.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImprovementRuntimeBridge:
    """Explicit service-layer bridge for one caller-requested transition."""

    config: ImprovementRuntimeBridgeConfig = field(
        default_factory=ImprovementRuntimeBridgeConfig
    )
    clock: Callable[[], float] = time.time
    record_id_factory: Callable[[], str] | None = None

    async def collect_evidence(
        self,
        request: ImprovementRuntimeRequest,
    ) -> ImprovementRuntimeTransitionResult:
        """Collect evidence through injected adapters and return one transition result."""

        if not self.config.enabled or not request.enabled:
            return self._emit_transition_observation(
                _runtime_result(
                    request,
                    status="not_enabled",
                    blocked_reason="improvement runtime bridge is not enabled",
                    record_kind="runtime_not_enabled",
                    sequence=request.record_sequence,
                    created_at=self.clock(),
                    record_id=self._new_record_id(),
                )
            )
        if request.collection_request is None:
            return await self._store_and_emit_transition_observation(
                request,
                _runtime_result(
                    request,
                    status="blocked",
                    blocked_reason="evidence collection request is required",
                    record_kind="runtime_blocked",
                    sequence=request.record_sequence,
                    created_at=self.clock(),
                    record_id=self._new_record_id(),
                ),
            )

        evidence_sources = request.evidence_sources or self.config.evidence_sources
        if not evidence_sources:
            return await self._store_and_emit_transition_observation(
                request,
                _runtime_result(
                    request,
                    status="blocked",
                    blocked_reason="no improvement evidence source adapters were supplied",
                    record_kind="runtime_blocked",
                    sequence=request.record_sequence,
                    created_at=self.clock(),
                    record_id=self._new_record_id(),
                ),
            )

        evidence: list[ImprovementEvidence] = []
        warnings: list[str] = []
        truncated = False
        for source in evidence_sources:
            result = await source.collect(request.collection_request)
            _validate_collection_result_linkage(request.collection_request, result)
            evidence.extend(result.evidence)
            warnings.extend(result.warnings)
            truncated = truncated or result.truncated

        bounded = validate_improvement_evidence_collection(
            evidence,
            warnings=warnings,
            truncated=truncated,
            bounds=request.collection_request.bounds,
        )
        record = ImprovementRuntimeRecord(
            record_id=self._new_record_id(),
            record_kind="evidence_collected",
            session_id=request.session_id,
            runtime_session_id=request.runtime_session_id,
            sequence=request.record_sequence,
            payload={
                "request_id": request.request_id,
                "collection_request_id": request.collection_request.request_id,
                "evidence_count": len(bounded.evidence),
                "truncated": bounded.truncated,
                "total_text_chars": bounded.total_text_chars,
                "total_metadata_chars": bounded.total_metadata_chars,
            },
            warnings=bounded.warnings,
            created_at=self.clock(),
            metadata=_runtime_record_metadata(
                request.metadata,
                request.permission_report,
            ),
        )
        record_store = request.record_store or self.config.record_store
        if record_store is not None:
            stored_record = await record_store.append_record(record)
            _validate_store_append_result(record, stored_record)
            record = stored_record

        summary = ImprovementRuntimeChainSummary(
            session_id=request.session_id,
            runtime_session_id=request.runtime_session_id,
            record_count=1,
            status="completed",
            last_record_id=record.record_id,
            last_sequence=record.sequence,
            last_record_kind=record.record_kind,
            warnings=bounded.warnings,
            metadata={"last_transition": "evidence_collected"},
        )
        return self._emit_transition_observation(
            ImprovementRuntimeTransitionResult(
                request_id=request.request_id,
                session_id=request.session_id,
                runtime_session_id=request.runtime_session_id,
                status="completed",
                records=(record,),
                summary=summary,
                evidence=bounded.evidence,
                permission_report=request.permission_report,
                warnings=bounded.warnings,
                metadata={
                    "evidence_count": len(bounded.evidence),
                    "truncated": bounded.truncated,
                },
            )
        )

    def _new_record_id(self) -> str:
        record_id = (
            self.record_id_factory()
            if self.record_id_factory is not None
            else f"irtr_{uuid4().hex}"
        )
        if not record_id.strip():
            raise ImprovementRuntimeValidationError(
                "record_id_factory returned an empty id"
            )
        return record_id

    async def _store_and_emit_transition_observation(
        self,
        request: ImprovementRuntimeRequest,
        result: ImprovementRuntimeTransitionResult,
    ) -> ImprovementRuntimeTransitionResult:
        record_store = request.record_store or self.config.record_store
        if record_store is None or not result.records:
            return self._emit_transition_observation(result)
        stored_records: list[ImprovementRuntimeRecord] = []
        for record in result.records:
            stored_record = await record_store.append_record(record)
            _validate_store_append_result(record, stored_record)
            stored_records.append(stored_record)
        summary = result.summary
        if summary is not None:
            summary = replace(
                summary,
                last_record_id=stored_records[-1].record_id,
                last_sequence=stored_records[-1].sequence,
                last_record_kind=stored_records[-1].record_kind,
            )
        return self._emit_transition_observation(
            replace(result, records=tuple(stored_records), summary=summary)
        )

    def _emit_transition_observation(
        self,
        result: ImprovementRuntimeTransitionResult,
    ) -> ImprovementRuntimeTransitionResult:
        sink = self.config.observability_sink
        if sink is None:
            return result
        event = improvement_runtime_observability_event_from_result(result)
        try:
            sink.emit_transition(event)
        except Exception as exc:  # pragma: no cover - exact exception is caller-owned.
            warning = (
                "improvement runtime observability observer failed: "
                f"{type(exc).__name__}"
            )
            return replace(
                result,
                warnings=_bounded_string_tuple(
                    (*result.warnings, warning),
                    "ImprovementRuntimeTransitionResult.warnings",
                    max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
                    max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
                ),
            )
        return result


def validate_improvement_evidence_collection(
    evidence: Sequence[ImprovementEvidence],
    *,
    warnings: Sequence[str] = (),
    truncated: bool = False,
    bounds: ImprovementEvidenceCollectionBounds | None = None,
) -> BoundedImprovementEvidenceCollection:
    """Validate evidence source output before proposal-generation use."""

    resolved_bounds = bounds or ImprovementEvidenceCollectionBounds()
    items = tuple(evidence)
    if len(items) > resolved_bounds.max_items:
        raise ImprovementRuntimeValidationError(
            "improvement evidence collection item count exceeds "
            f"{resolved_bounds.max_items}: {len(items)}"
        )
    warning_items = _bounded_string_tuple(
        warnings,
        "validate_improvement_evidence_collection.warnings",
        max_count=resolved_bounds.max_warnings,
        max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
    )

    total_text_chars = 0
    total_metadata_chars = 0
    for item in items:
        if item.excerpt is not None and len(item.excerpt) > resolved_bounds.max_excerpt_chars:
            raise ImprovementRuntimeValidationError(
                f"improvement evidence item {item.evidence_id!r} excerpt exceeds "
                "ImprovementEvidenceCollectionBounds.max_excerpt_chars"
            )
        if item.source_uri is not None and (
            len(item.source_uri) > resolved_bounds.max_source_uri_chars
        ):
            raise ImprovementRuntimeValidationError(
                f"improvement evidence item {item.evidence_id!r} source_uri exceeds "
                "ImprovementEvidenceCollectionBounds.max_source_uri_chars"
            )
        item_metadata_chars = _json_text_chars(item.metadata)
        if item_metadata_chars > resolved_bounds.max_item_metadata_chars:
            raise ImprovementRuntimeValidationError(
                f"improvement evidence item {item.evidence_id!r} metadata exceeds "
                "ImprovementEvidenceCollectionBounds.max_item_metadata_chars"
            )
        total_metadata_chars += item_metadata_chars
        total_text_chars += improvement_evidence_text_chars(item)

    if total_metadata_chars > resolved_bounds.max_total_metadata_chars:
        raise ImprovementRuntimeValidationError(
            "improvement evidence collection metadata exceeds "
            "ImprovementEvidenceCollectionBounds.max_total_metadata_chars"
        )
    if total_text_chars > resolved_bounds.max_total_chars:
        raise ImprovementRuntimeValidationError(
            "improvement evidence collection text exceeds "
            "ImprovementEvidenceCollectionBounds.max_total_chars"
        )
    return BoundedImprovementEvidenceCollection(
        evidence=items,
        total_text_chars=total_text_chars,
        total_metadata_chars=total_metadata_chars,
        warnings=warning_items,
        truncated=truncated,
    )


def improvement_runtime_record_to_dict(
    record: ImprovementRuntimeRecord,
) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "record_id": record.record_id,
        "record_kind": record.record_kind,
        "session_id": record.session_id,
        "runtime_session_id": record.runtime_session_id,
        "run_id": record.run_id,
        "proposal_id": record.proposal_id,
        "candidate_id": record.candidate_id,
        "stage_id": record.stage_id,
        "sequence": record.sequence,
        "payload": _metadata_to_dict(record.payload),
        "payload_ref": record.payload_ref,
        "warnings": list(record.warnings),
        "stop_reason": record.stop_reason,
        "created_at": record.created_at,
        "metadata": _metadata_to_dict(record.metadata),
    }


def improvement_runtime_record_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimeRecord:
    return ImprovementRuntimeRecord(
        record_id=str(data["record_id"]),
        record_kind=cast(
            ImprovementRuntimeRecordKind,
            _literal_from_object(
                data["record_kind"],
                _RUNTIME_RECORD_KINDS,
                "record_kind",
            ),
        ),
        session_id=str(data["session_id"]),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        run_id=_optional_str(data.get("run_id")),
        proposal_id=_optional_str(data.get("proposal_id")),
        candidate_id=_optional_str(data.get("candidate_id")),
        stage_id=_optional_str(data.get("stage_id")),
        sequence=_int_from_object(data["sequence"], "sequence"),
        payload=_metadata_from_object(data.get("payload", {}), "payload"),
        payload_ref=_optional_str(data.get("payload_ref")),
        schema_version=_int_from_object(
            data.get("schema_version", IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION),
            "schema_version",
        ),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        stop_reason=_optional_str(data.get("stop_reason")),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_runtime_chain_summary_to_dict(
    summary: ImprovementRuntimeChainSummary,
) -> dict[str, object]:
    return {
        "schema_version": summary.schema_version,
        "session_id": summary.session_id,
        "runtime_session_id": summary.runtime_session_id,
        "run_id": summary.run_id,
        "proposal_id": summary.proposal_id,
        "candidate_id": summary.candidate_id,
        "record_count": summary.record_count,
        "status": summary.status,
        "last_record_id": summary.last_record_id,
        "last_sequence": summary.last_sequence,
        "last_record_kind": summary.last_record_kind,
        "next_record_kind": summary.next_record_kind,
        "blocked_reason": summary.blocked_reason,
        "warnings": list(summary.warnings),
        "metadata": _metadata_to_dict(summary.metadata),
    }


def improvement_runtime_chain_summary_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimeChainSummary:
    return ImprovementRuntimeChainSummary(
        session_id=str(data["session_id"]),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        run_id=_optional_str(data.get("run_id")),
        proposal_id=_optional_str(data.get("proposal_id")),
        candidate_id=_optional_str(data.get("candidate_id")),
        record_count=_int_from_object(data["record_count"], "record_count"),
        status=cast(
            ImprovementRuntimeTransitionStatus,
            _literal_from_object(
                data["status"],
                _RUNTIME_TRANSITION_STATUSES,
                "status",
            ),
        ),
        schema_version=_int_from_object(
            data.get("schema_version", IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION),
            "schema_version",
        ),
        last_record_id=_optional_str(data.get("last_record_id")),
        last_sequence=_optional_int(data.get("last_sequence"), "last_sequence"),
        last_record_kind=_optional_record_kind(data.get("last_record_kind")),
        next_record_kind=_optional_record_kind(data.get("next_record_kind")),
        blocked_reason=_optional_str(data.get("blocked_reason")),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_evidence_collection_result_to_dict(
    result: ImprovementEvidenceCollectionResult,
) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "session_id": result.session_id,
        "runtime_session_id": result.runtime_session_id,
        "source_id": result.source_id,
        "evidence": [improvement_evidence_to_dict(item) for item in result.evidence],
        "warnings": list(result.warnings),
        "truncated": result.truncated,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_evidence_collection_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementEvidenceCollectionResult:
    return ImprovementEvidenceCollectionResult(
        request_id=str(data["request_id"]),
        session_id=str(data["session_id"]),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        source_id=_optional_str(data.get("source_id")),
        evidence=tuple(
            improvement_evidence_from_dict(item)
            for item in _mapping_sequence(data.get("evidence", ()), "evidence")
        ),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        truncated=bool(data.get("truncated", False)),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_runtime_permission_requirement_to_dict(
    requirement: ImprovementRuntimePermissionRequirement,
) -> dict[str, object]:
    return {
        "stage_id": requirement.stage_id,
        "record_kind": requirement.record_kind,
        "required_permissions": list(requirement.required_permissions),
        "reason": requirement.reason,
        "source_record_id": requirement.source_record_id,
        "metadata": _metadata_to_dict(requirement.metadata),
    }


def improvement_runtime_permission_requirement_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimePermissionRequirement:
    return ImprovementRuntimePermissionRequirement(
        stage_id=str(data["stage_id"]),
        record_kind=_optional_record_kind(data.get("record_kind")),
        required_permissions=_permission_tuple(
            _string_tuple(data.get("required_permissions", ()), "required_permissions"),
            "required_permissions",
            allow_empty=False,
        ),
        reason=_optional_str(data.get("reason")),
        source_record_id=_optional_str(data.get("source_record_id")),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_runtime_permission_summary_to_dict(
    summary: ImprovementRuntimePermissionSummary,
) -> dict[str, object]:
    return {
        "stage_id": summary.stage_id,
        "status": summary.status,
        "required_permission_count": summary.required_permission_count,
        "missing_permission_count": summary.missing_permission_count,
        "extra_permission_count": summary.extra_permission_count,
        "requirement_labels": list(summary.requirement_labels),
    }


def improvement_runtime_permission_summary_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimePermissionSummary:
    return ImprovementRuntimePermissionSummary(
        stage_id=str(data["stage_id"]),
        status=cast(
            ImprovementRuntimePermissionStatus,
            _literal_from_object(data["status"], _PERMISSION_STATUSES, "status"),
        ),
        required_permission_count=_int_from_object(
            data["required_permission_count"],
            "required_permission_count",
        ),
        missing_permission_count=_int_from_object(
            data["missing_permission_count"],
            "missing_permission_count",
        ),
        extra_permission_count=_int_from_object(
            data["extra_permission_count"],
            "extra_permission_count",
        ),
        requirement_labels=_string_tuple(
            data.get("requirement_labels", ()),
            "requirement_labels",
        ),
    )


def improvement_runtime_permission_report_to_dict(
    report: ImprovementRuntimePermissionReport,
) -> dict[str, object]:
    return {
        "request_id": report.request_id,
        "session_id": report.session_id,
        "runtime_session_id": report.runtime_session_id,
        "stage_id": report.stage_id,
        "status": report.status,
        "requirements": [
            improvement_runtime_permission_requirement_to_dict(requirement)
            for requirement in report.requirements
        ],
        "supplied_approved_permissions": list(report.supplied_approved_permissions),
        "missing_permissions": list(report.missing_permissions),
        "extra_permissions": list(report.extra_permissions),
        "warnings": list(report.warnings),
        "metadata": _metadata_to_dict(report.metadata),
    }


def improvement_runtime_permission_report_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimePermissionReport:
    return ImprovementRuntimePermissionReport(
        request_id=str(data["request_id"]),
        session_id=str(data["session_id"]),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        stage_id=str(data["stage_id"]),
        status=cast(
            ImprovementRuntimePermissionStatus,
            _literal_from_object(data["status"], _PERMISSION_STATUSES, "status"),
        ),
        requirements=tuple(
            improvement_runtime_permission_requirement_from_dict(item)
            for item in _mapping_sequence(data.get("requirements", ()), "requirements")
        ),
        supplied_approved_permissions=_permission_tuple(
            _string_tuple(
                data.get("supplied_approved_permissions", ()),
                "supplied_approved_permissions",
            ),
            "supplied_approved_permissions",
            allow_empty=True,
        ),
        missing_permissions=_permission_tuple(
            _string_tuple(data.get("missing_permissions", ()), "missing_permissions"),
            "missing_permissions",
            allow_empty=True,
        ),
        extra_permissions=_permission_tuple(
            _string_tuple(data.get("extra_permissions", ()), "extra_permissions"),
            "extra_permissions",
            allow_empty=True,
        ),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_runtime_permission_report_to_summary(
    report: ImprovementRuntimePermissionReport,
) -> ImprovementRuntimePermissionSummary:
    labels = tuple(_permission_requirement_label(item) for item in report.requirements)
    return ImprovementRuntimePermissionSummary(
        stage_id=report.stage_id,
        status=report.status,
        required_permission_count=len(_permission_set_from_requirements(report.requirements)),
        missing_permission_count=len(report.missing_permissions),
        extra_permission_count=len(report.extra_permissions),
        requirement_labels=labels,
    )


def improvement_runtime_recovery_result_to_dict(
    result: ImprovementRuntimeRecoveryResult,
) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "status": result.status,
        "query": _runtime_record_query_to_dict(result.query),
        "session_id": result.session_id,
        "expected_session_id": result.expected_session_id,
        "runtime_session_id": result.runtime_session_id,
        "run_id": result.run_id,
        "proposal_id": result.proposal_id,
        "candidate_id": result.candidate_id,
        "summary": None
        if result.summary is None
        else improvement_runtime_chain_summary_to_dict(result.summary),
        "records": [improvement_runtime_record_to_dict(record) for record in result.records],
        "last_record_id": result.last_record_id,
        "last_sequence": result.last_sequence,
        "last_record_kind": result.last_record_kind,
        "last_completed_record_id": result.last_completed_record_id,
        "last_completed_sequence": result.last_completed_sequence,
        "last_completed_record_kind": result.last_completed_record_kind,
        "next_record_kind": result.next_record_kind,
        "permission_summary": None
        if result.permission_summary is None
        else improvement_runtime_permission_summary_to_dict(result.permission_summary),
        "warnings": list(result.warnings),
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_runtime_recovery_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimeRecoveryResult:
    summary_data = data.get("summary")
    permission_summary_data = data.get("permission_summary")
    return ImprovementRuntimeRecoveryResult(
        request_id=str(data["request_id"]),
        status=cast(
            ImprovementRuntimeRecoveryStatus,
            _literal_from_object(data["status"], _RUNTIME_RECOVERY_STATUSES, "status"),
        ),
        query=_runtime_record_query_from_dict(
            cast(Mapping[str, object], data["query"])
        ),
        session_id=_optional_str(data.get("session_id")),
        expected_session_id=_optional_str(data.get("expected_session_id")),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        run_id=_optional_str(data.get("run_id")),
        proposal_id=_optional_str(data.get("proposal_id")),
        candidate_id=_optional_str(data.get("candidate_id")),
        summary=None
        if summary_data is None
        else improvement_runtime_chain_summary_from_dict(
            cast(Mapping[str, object], summary_data)
        ),
        records=tuple(
            improvement_runtime_record_from_dict(item)
            for item in _mapping_sequence(data.get("records", ()), "records")
        ),
        last_record_id=_optional_str(data.get("last_record_id")),
        last_sequence=_optional_int(data.get("last_sequence"), "last_sequence"),
        last_record_kind=_optional_record_kind(data.get("last_record_kind")),
        last_completed_record_id=_optional_str(data.get("last_completed_record_id")),
        last_completed_sequence=_optional_int(
            data.get("last_completed_sequence"),
            "last_completed_sequence",
        ),
        last_completed_record_kind=_optional_record_kind(
            data.get("last_completed_record_kind")
        ),
        next_record_kind=_optional_record_kind(data.get("next_record_kind")),
        permission_summary=None
        if permission_summary_data is None
        else improvement_runtime_permission_summary_from_dict(
            cast(Mapping[str, object], permission_summary_data)
        ),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        metadata=_metadata_from_object(data.get("metadata", {}), "metadata"),
    )


def improvement_runtime_observability_event_to_dict(
    event: ImprovementRuntimeObservabilityEvent,
) -> dict[str, object]:
    return {
        "request_id": event.request_id,
        "session_id": event.session_id,
        "runtime_session_id": event.runtime_session_id,
        "status": event.status,
        "record_ids": list(event.record_ids),
        "last_record_kind": event.last_record_kind,
        "evidence_count": event.evidence_count,
        "warning_count": event.warning_count,
        "truncated": event.truncated,
        "stage_id": event.stage_id,
        "permission_status": event.permission_status,
        "permission_required_count": event.permission_required_count,
        "permission_missing_count": event.permission_missing_count,
        "permission_extra_count": event.permission_extra_count,
    }


def improvement_runtime_observability_event_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimeObservabilityEvent:
    permission_status = data.get("permission_status")
    return ImprovementRuntimeObservabilityEvent(
        request_id=str(data["request_id"]),
        session_id=str(data["session_id"]),
        runtime_session_id=_optional_str(data.get("runtime_session_id")),
        status=cast(
            ImprovementRuntimeTransitionStatus,
            _literal_from_object(data["status"], _RUNTIME_TRANSITION_STATUSES, "status"),
        ),
        record_ids=_string_tuple(data.get("record_ids", ()), "record_ids"),
        last_record_kind=_optional_record_kind(data.get("last_record_kind")),
        evidence_count=_int_from_object(data["evidence_count"], "evidence_count"),
        warning_count=_int_from_object(data["warning_count"], "warning_count"),
        truncated=bool(data.get("truncated", False)),
        stage_id=_optional_str(data.get("stage_id")),
        permission_status=None
        if permission_status is None
        else cast(
            ImprovementRuntimePermissionStatus,
            _literal_from_object(
                permission_status,
                _PERMISSION_STATUSES,
                "permission_status",
            ),
        ),
        permission_required_count=_int_from_object(
            data.get("permission_required_count", 0),
            "permission_required_count",
        ),
        permission_missing_count=_int_from_object(
            data.get("permission_missing_count", 0),
            "permission_missing_count",
        ),
        permission_extra_count=_int_from_object(
            data.get("permission_extra_count", 0),
            "permission_extra_count",
        ),
    )


def improvement_runtime_observability_event_from_result(
    result: ImprovementRuntimeTransitionResult,
) -> ImprovementRuntimeObservabilityEvent:
    last_record = result.records[-1] if result.records else None
    permission_summary = (
        result.permission_report.to_summary()
        if result.permission_report is not None
        else None
    )
    truncated = result.metadata.get("truncated")
    return ImprovementRuntimeObservabilityEvent(
        request_id=result.request_id,
        session_id=result.session_id,
        runtime_session_id=result.runtime_session_id,
        status=result.status,
        record_ids=tuple(record.record_id for record in result.records),
        last_record_kind=None if last_record is None else last_record.record_kind,
        evidence_count=len(result.evidence),
        warning_count=len(result.warnings),
        truncated=truncated if isinstance(truncated, bool) else False,
        stage_id=(
            permission_summary.stage_id
            if permission_summary is not None
            else None if last_record is None else last_record.stage_id
        ),
        permission_status=None if permission_summary is None else permission_summary.status,
        permission_required_count=(
            0
            if permission_summary is None
            else permission_summary.required_permission_count
        ),
        permission_missing_count=(
            0 if permission_summary is None else permission_summary.missing_permission_count
        ),
        permission_extra_count=(
            0 if permission_summary is None else permission_summary.extra_permission_count
        ),
    )


def _collection_result(
    request: ImprovementEvidenceCollectionRequest,
    *,
    source_id: str,
    evidence: Sequence[ImprovementEvidence] = (),
    warnings: Sequence[str] = (),
    truncated: bool = False,
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementEvidenceCollectionResult:
    result = ImprovementEvidenceCollectionResult(
        request_id=request.request_id,
        session_id=request.session_id,
        runtime_session_id=request.runtime_session_id,
        source_id=source_id,
        evidence=tuple(evidence),
        warnings=tuple(warnings),
        truncated=truncated,
        metadata=_empty_metadata() if metadata is None else metadata,
    )
    _validate_collection_result_linkage(request, result)
    return result


def _source_requested(
    request: ImprovementEvidenceCollectionRequest,
    source: ImprovementEvidenceSource,
) -> bool:
    return not request.source_kinds or source in request.source_kinds


def _transcript_search_budget(
    bounds: ImprovementEvidenceCollectionBounds,
) -> tuple[int, int] | str:
    max_total_snippet_chars = min(
        bounds.max_total_chars // 2,
        bounds.proposal_evidence_bounds.max_total_text_chars // 2,
    )
    if max_total_snippet_chars < TRANSCRIPT_SEARCH_MIN_SNIPPET_CHARS:
        return (
            "transcript evidence bounds cannot satisfy "
            "TranscriptSearchRequest.max_snippet_chars >= 16"
        )
    max_snippet_chars = min(
        bounds.max_excerpt_chars,
        bounds.proposal_evidence_bounds.max_item_text_chars,
        max_total_snippet_chars,
    )
    if max_snippet_chars < TRANSCRIPT_SEARCH_MIN_SNIPPET_CHARS:
        return (
            "transcript evidence bounds cannot satisfy "
            "TranscriptSearchRequest.max_snippet_chars >= 16"
        )
    return max_snippet_chars, max_total_snippet_chars


def _transcript_match_to_evidence(
    match: TranscriptSearchMatch,
) -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id=f"iev_transcript_{_safe_uri_component(match.entry_id)}",
        source="transcript",
        summary=f"Transcript {match.role or 'message'} match from {match.entry_id}",
        excerpt=match.snippet,
        source_uri=(
            "transcript://"
            f"{_safe_uri_component(match.session_id)}/"
            f"{_safe_uri_component(match.entry_id)}"
        ),
        created_at=match.created_at,
        metadata={
            "session_id": match.session_id,
            "runtime_session_id": match.runtime_session_id,
            "entry_id": match.entry_id,
            "role": match.role,
            "score": match.score,
            "order": match.order,
            "agent_id": match.agent_id,
            "is_sidechain": match.is_sidechain,
            "snippet_truncated": match.snippet_truncated,
        },
    )


def _task_output_max_bytes(
    target: ImprovementTaskOutputEvidenceTarget,
    bounds: ImprovementEvidenceCollectionBounds,
) -> int:
    if target.max_bytes is None:
        return bounds.max_excerpt_chars
    return min(target.max_bytes, bounds.max_excerpt_chars)


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _safe_uri_component(value: str, *, max_chars: int = 120) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(char if char in allowed else "_" for char in value).strip("._-")
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if not cleaned:
        cleaned = f"id-{digest}"
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 13]}-{digest}"


def _reject_raw_observability_metadata(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            key_text = str(key)
            child_name = f"{name}.{key_text}"
            if key_text in _RAW_OBSERVABILITY_METADATA_KEYS:
                if _is_redaction_marker(item):
                    continue
                raise ImprovementRuntimeValidationError(
                    f"{child_name} must be a bounded redaction marker"
                )
            _reject_raw_observability_metadata(item, child_name)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(cast(Sequence[object], value)):
            _reject_raw_observability_metadata(item, f"{name}[{index}]")


def _is_redaction_marker(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    marker = cast(Mapping[object, object], value)
    if marker.get("redacted") is not True:
        return False
    for key, item in marker.items():
        key_text = str(key)
        if key_text not in _REDACTION_MARKER_KEYS:
            return False
        if key_text == "redacted":
            continue
        if isinstance(item, Mapping):
            return False
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            return False
    return True


def _expected_session_query_conflict(
    request: ImprovementRuntimeRecoveryRequest,
) -> str | None:
    if (
        request.expected_session_id is not None
        and request.query.session_id is not None
        and request.query.session_id != request.expected_session_id
    ):
        return (
            "ImprovementRuntimeRecoveryRequest.query.session_id must match "
            "expected_session_id"
        )
    return None


def _blocked_recovery_result(
    request: ImprovementRuntimeRecoveryRequest,
    reason: str,
    *,
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementRuntimeRecoveryResult:
    recovery_metadata: dict[str, object] = _metadata_to_dict(request.metadata)
    if metadata is not None:
        recovery_metadata.update(_metadata_to_dict(metadata))
    return ImprovementRuntimeRecoveryResult(
        request_id=request.request_id,
        status="blocked",
        query=request.query,
        session_id=request.expected_session_id or request.query.session_id,
        expected_session_id=request.expected_session_id,
        warnings=(reason,),
        metadata=_freeze_metadata(
            recovery_metadata,
            "ImprovementRuntimeRecoveryResult.metadata",
            DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
        ),
    )


def _validate_recovery_records(
    query: ImprovementRuntimeRecordQuery,
    records: Sequence[ImprovementRuntimeRecord],
) -> None:
    if len(records) > query.max_records:
        raise ImprovementRuntimeValidationError(
            "ImprovementRecordStore.load_records returned more records than "
            "ImprovementRuntimeRecordQuery.max_records"
        )
    for record in records:
        for field_name, expected in (
            ("session_id", query.session_id),
            ("run_id", query.run_id),
            ("proposal_id", query.proposal_id),
            ("candidate_id", query.candidate_id),
        ):
            if expected is not None and getattr(record, field_name) != expected:
                raise ImprovementRuntimeValidationError(
                    "ImprovementRecordStore.load_records returned a record "
                    f"outside query selector {field_name}"
                )


def _validate_recovery_summary(
    query: ImprovementRuntimeRecordQuery,
    summary: ImprovementRuntimeChainSummary,
) -> None:
    for field_name, expected in (
        ("session_id", query.session_id),
        ("run_id", query.run_id),
        ("proposal_id", query.proposal_id),
        ("candidate_id", query.candidate_id),
    ):
        if expected is not None and getattr(summary, field_name) != expected:
            raise ImprovementRuntimeValidationError(
                "ImprovementRecordStore.summarize_chain returned a summary "
                f"outside query selector {field_name}"
            )


def _validate_recovery_session_linkage(
    request: ImprovementRuntimeRecoveryRequest,
    records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary | None,
) -> None:
    expected = request.expected_session_id
    session_ids = {record.session_id for record in records}
    if summary is not None:
        session_ids.add(summary.session_id)
    if expected is not None and any(session_id != expected for session_id in session_ids):
        raise ImprovementRuntimeValidationError(
            "improvement runtime recovery session_id must match expected_session_id"
        )
    if len(session_ids) > 1:
        raise ImprovementRuntimeValidationError(
            "improvement runtime recovery records must belong to one session_id"
        )


def _ordered_recovery_records(
    records: Sequence[ImprovementRuntimeRecord],
) -> tuple[tuple[ImprovementRuntimeRecord, ...], tuple[str, ...]]:
    ordered = tuple(
        sorted(records, key=lambda item: (item.sequence, item.created_at, item.record_id))
    )
    duplicate_sequences = sorted(
        sequence
        for sequence in {record.sequence for record in ordered}
        if sum(1 for record in ordered if record.sequence == sequence) > 1
    )
    warnings = tuple(
        f"duplicate improvement runtime record sequence: {sequence}"
        for sequence in duplicate_sequences
    )
    return ordered, warnings


def _summary_from_recovery_records(
    ordered_records: Sequence[ImprovementRuntimeRecord],
) -> ImprovementRuntimeChainSummary:
    if not ordered_records:
        raise ImprovementRuntimeValidationError(
            "cannot derive recovery summary without records"
        )
    last = ordered_records[-1]
    status = _transition_status_from_record_kind(last.record_kind)
    return ImprovementRuntimeChainSummary(
        session_id=last.session_id,
        runtime_session_id=last.runtime_session_id,
        run_id=last.run_id,
        proposal_id=last.proposal_id,
        candidate_id=last.candidate_id,
        record_count=len(ordered_records),
        status=status,
        last_record_id=last.record_id,
        last_sequence=last.sequence,
        last_record_kind=last.record_kind,
        blocked_reason=last.stop_reason if status != "completed" else None,
        metadata={"summary_source": "loaded_records"},
    )


def _transition_status_from_record_kind(
    record_kind: ImprovementRuntimeRecordKind,
) -> ImprovementRuntimeTransitionStatus:
    if record_kind == "runtime_blocked":
        return "blocked"
    if record_kind == "runtime_not_enabled":
        return "not_enabled"
    return "completed"


def _last_completed_recovery_record(
    ordered_records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary,
) -> ImprovementRuntimeRecord | ImprovementRuntimeChainSummary | None:
    for record in reversed(ordered_records):
        if _transition_status_from_record_kind(record.record_kind) == "completed":
            return record
    if (
        not ordered_records
        and summary.last_record_kind is not None
        and _transition_status_from_record_kind(summary.last_record_kind) == "completed"
    ):
        return summary
    return None


def _recovery_session_id(
    request: ImprovementRuntimeRecoveryRequest,
    ordered_records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary,
) -> str | None:
    if request.expected_session_id is not None:
        return request.expected_session_id
    if request.query.session_id is not None:
        return request.query.session_id
    if ordered_records:
        return ordered_records[0].session_id
    return summary.session_id


def _recovery_runtime_session_id(
    ordered_records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary,
) -> str | None:
    for record in reversed(ordered_records):
        if record.runtime_session_id is not None:
            return record.runtime_session_id
    return summary.runtime_session_id


def _recovery_selector_value(
    field_name: str,
    query_value: str | None,
    ordered_records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary,
) -> str | None:
    if query_value is not None:
        return query_value
    for record in reversed(ordered_records):
        value = getattr(record, field_name)
        if value is not None:
            return cast(str, value)
    return cast(str | None, getattr(summary, field_name))


def _permission_summary_from_recovery_record(
    record: ImprovementRuntimeRecord | None,
) -> ImprovementRuntimePermissionSummary | None:
    if record is None:
        return None
    value = record.metadata.get("permission_summary")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ImprovementRuntimeValidationError(
            "ImprovementRuntimeRecord.metadata.permission_summary must be a mapping"
        )
    return improvement_runtime_permission_summary_from_dict(cast(Mapping[str, object], value))


def _combined_recovery_warnings(
    ordering_warnings: Sequence[str],
    ordered_records: Sequence[ImprovementRuntimeRecord],
    summary: ImprovementRuntimeChainSummary,
) -> tuple[str, ...]:
    warnings: list[str] = list(ordering_warnings)
    warnings.extend(summary.warnings)
    for record in ordered_records:
        warnings.extend(record.warnings)
    return _bounded_string_tuple(
        warnings,
        "ImprovementRuntimeRecoveryResult.warnings",
        max_count=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
        max_chars=DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS,
    )


def _runtime_result(
    request: ImprovementRuntimeRequest,
    *,
    status: ImprovementRuntimeTransitionStatus,
    blocked_reason: str,
    record_kind: ImprovementRuntimeRecordKind,
    sequence: int,
    created_at: float,
    record_id: str,
) -> ImprovementRuntimeTransitionResult:
    record = ImprovementRuntimeRecord(
        record_id=record_id,
        record_kind=record_kind,
        session_id=request.session_id,
        runtime_session_id=request.runtime_session_id,
        sequence=sequence,
        payload={"request_id": request.request_id, "status": status},
        stop_reason=blocked_reason,
        created_at=created_at,
        metadata=_runtime_record_metadata(request.metadata, request.permission_report),
    )
    summary = ImprovementRuntimeChainSummary(
        session_id=request.session_id,
        runtime_session_id=request.runtime_session_id,
        record_count=1,
        status=status,
        last_record_id=record.record_id,
        last_sequence=record.sequence,
        last_record_kind=record.record_kind,
        blocked_reason=blocked_reason,
        metadata={"last_transition": status},
    )
    return ImprovementRuntimeTransitionResult(
        request_id=request.request_id,
        session_id=request.session_id,
        runtime_session_id=request.runtime_session_id,
        status=status,
        records=(record,),
        summary=summary,
        warnings=(),
        blocked_reason=blocked_reason,
        permission_report=request.permission_report,
    )


def _validate_collection_result_linkage(
    request: ImprovementEvidenceCollectionRequest,
    result: ImprovementEvidenceCollectionResult,
) -> None:
    if result.request_id != request.request_id:
        raise ImprovementRuntimeValidationError(
            "ImprovementEvidenceCollectionResult.request_id must match request_id"
        )
    if result.session_id != request.session_id:
        raise ImprovementRuntimeValidationError(
            "ImprovementEvidenceCollectionResult.session_id must match session_id"
        )
    if result.runtime_session_id != request.runtime_session_id:
        raise ImprovementRuntimeValidationError(
            "ImprovementEvidenceCollectionResult.runtime_session_id must match "
            "runtime_session_id"
        )
    validate_improvement_evidence_collection(
        result.evidence,
        warnings=result.warnings,
        truncated=result.truncated,
        bounds=request.bounds,
    )


def _validate_store_append_result(
    expected: ImprovementRuntimeRecord,
    actual: ImprovementRuntimeRecord,
) -> None:
    if actual != expected:
        raise ImprovementRuntimeValidationError(
            "ImprovementRecordStore.append_record returned a different runtime record"
        )


def _runtime_record_query_to_dict(
    query: ImprovementRuntimeRecordQuery,
) -> dict[str, object]:
    return {
        "session_id": query.session_id,
        "run_id": query.run_id,
        "proposal_id": query.proposal_id,
        "candidate_id": query.candidate_id,
        "max_records": query.max_records,
    }


def _runtime_record_query_from_dict(
    data: Mapping[str, object],
) -> ImprovementRuntimeRecordQuery:
    return ImprovementRuntimeRecordQuery(
        session_id=_optional_str(data.get("session_id")),
        run_id=_optional_str(data.get("run_id")),
        proposal_id=_optional_str(data.get("proposal_id")),
        candidate_id=_optional_str(data.get("candidate_id")),
        max_records=_int_from_object(
            data.get("max_records", DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS),
            "max_records",
        ),
    )


def _require_schema_version(schema_version: int) -> None:
    if schema_version != IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION:
        raise ImprovementRuntimeValidationError(
            "improvement runtime schema_version must be "
            f"{IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION}"
        )


def _require_minimum(value: int, minimum: int, name: str) -> None:
    if value < minimum:
        raise ImprovementRuntimeValidationError(f"{name} must be >= {minimum}")


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ImprovementRuntimeValidationError(f"{name} must be non-empty")


def _require_bounded_non_empty(
    value: str,
    name: str,
    max_chars: int,
    max_name: str,
) -> None:
    _require_non_empty(value, name)
    if len(value) > max_chars:
        raise ImprovementRuntimeValidationError(f"{name} exceeds {max_name}")


def _require_literal(value: str, allowed: frozenset[str], name: str) -> None:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ImprovementRuntimeValidationError(f"{name} must be one of: {expected}")


def _literal_from_object(value: object, allowed: frozenset[str], name: str) -> str:
    text = str(value)
    _require_literal(text, allowed, name)
    return text


def _source_kind_tuple(
    values: Sequence[str],
    name: str,
) -> tuple[ImprovementEvidenceSource, ...]:
    return cast(
        tuple[ImprovementEvidenceSource, ...],
        tuple(_literal_from_object(value, _EVIDENCE_SOURCES, name) for value in values),
    )


def _permission_tuple(
    values: Sequence[str],
    name: str,
    *,
    allow_empty: bool,
) -> tuple[ImprovementRequiredPermission, ...]:
    items = tuple(
        cast(
            ImprovementRequiredPermission,
            _literal_from_object(item, _REQUIRED_PERMISSIONS, name),
        )
        for item in values
    )
    if not allow_empty and not items:
        raise ImprovementRuntimeValidationError(f"{name} must not be empty")
    duplicate_items = sorted(item for item in set(items) if items.count(item) > 1)
    if duplicate_items:
        joined = ", ".join(duplicate_items)
        raise ImprovementRuntimeValidationError(f"{name} contains duplicates: {joined}")
    if "none" in items and len(items) > 1:
        raise ImprovementRuntimeValidationError(f"{name} cannot mix none")
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _permission_set_from_requirements(
    requirements: Sequence[ImprovementRuntimePermissionRequirement],
) -> set[str]:
    result: set[str] = set()
    for requirement in requirements:
        result.update(requirement.required_permissions)
    return result


def _missing_permissions(
    required: set[str],
    supplied: Sequence[ImprovementRequiredPermission],
) -> tuple[ImprovementRequiredPermission, ...]:
    return _sorted_permission_tuple(required.difference(supplied))


def _extra_permissions(
    supplied: Sequence[ImprovementRequiredPermission],
    required: set[str],
) -> tuple[ImprovementRequiredPermission, ...]:
    return _sorted_permission_tuple(set(supplied).difference(required))


def _sorted_permission_tuple(
    values: Iterable[str],
) -> tuple[ImprovementRequiredPermission, ...]:
    return cast(tuple[ImprovementRequiredPermission, ...], tuple(sorted(values)))


def _permission_requirement_label(
    requirement: ImprovementRuntimePermissionRequirement,
) -> str:
    if requirement.record_kind is None:
        return requirement.stage_id
    return f"{requirement.stage_id}:{requirement.record_kind}"


def _runtime_record_metadata(
    metadata: Mapping[str, FrozenJson],
    permission_report: ImprovementRuntimePermissionReport | None,
) -> Mapping[str, FrozenJson]:
    _reject_unsafe_runtime_permission_metadata(
        metadata,
        "ImprovementRuntimeRequest.metadata",
        allow_permission_summary=False,
    )
    data = _metadata_to_dict(metadata)
    if permission_report is not None:
        data["permission_summary"] = improvement_runtime_permission_summary_to_dict(
            permission_report.to_summary()
        )
    return _freeze_metadata(
        data,
        "ImprovementRuntimeRecord.metadata",
        DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
    )


def _reject_unsafe_runtime_permission_metadata(
    value: object,
    name: str,
    *,
    allow_permission_summary: bool,
) -> None:
    if isinstance(value, Mapping):
        for key, item in cast(Mapping[object, object], value).items():
            key_text = str(key)
            child_name = f"{name}.{key_text}"
            if key_text == "permission_summary" and allow_permission_summary:
                _validate_permission_summary_metadata(item, child_name)
                continue
            if key_text in _UNSAFE_RUNTIME_PERMISSION_METADATA_KEYS:
                raise ImprovementRuntimeValidationError(
                    f"{child_name} is reserved for sanitized permission summaries"
                )
            _reject_unsafe_runtime_permission_metadata(
                item,
                child_name,
                allow_permission_summary=allow_permission_summary,
            )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(cast(Sequence[object], value)):
            _reject_unsafe_runtime_permission_metadata(
                item,
                f"{name}[{index}]",
                allow_permission_summary=allow_permission_summary,
            )


def _validate_permission_summary_metadata(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ImprovementRuntimeValidationError(
            f"{name} must be a sanitized permission summary"
        )
    mapping = cast(Mapping[object, object], value)
    keys = {str(key) for key in mapping}
    expected = {
        "stage_id",
        "status",
        "required_permission_count",
        "missing_permission_count",
        "extra_permission_count",
        "requirement_labels",
    }
    if keys != expected:
        raise ImprovementRuntimeValidationError(
            f"{name} must be a sanitized permission summary"
        )
    improvement_runtime_permission_summary_from_dict(cast(Mapping[str, object], value))


def _bounded_string_tuple(
    values: Sequence[str],
    name: str,
    *,
    max_count: int,
    max_chars: int,
) -> tuple[str, ...]:
    items = tuple(values)
    if len(items) > max_count:
        raise ImprovementRuntimeValidationError(f"{name} exceeds max count {max_count}")
    for item in items:
        _require_bounded_non_empty(
            item,
            name,
            max_chars,
            "DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS",
        )
    return items


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    return tuple(str(item) for item in cast(Sequence[object], value))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    _require_non_empty(text, "optional string")
    return text


def _optional_record_kind(value: object) -> ImprovementRuntimeRecordKind | None:
    if value is None:
        return None
    return cast(
        ImprovementRuntimeRecordKind,
        _literal_from_object(value, _RUNTIME_RECORD_KINDS, "record_kind"),
    )


def _int_from_object(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _int_from_object(value, name)


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _metadata_from_object(value: object, name: str) -> Mapping[str, FrozenJson]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze_metadata(
        cast(Mapping[str, object], value),
        name,
        DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
    )


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
        raise ImprovementRuntimeValidationError(f"{name} exceeds {max_chars} chars")
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
    "DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_ITEMS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_SOURCE_URI_CHARS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_CHARS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_TOTAL_METADATA_CHARS",
    "DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_PAYLOAD_REF_CHARS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_STOP_REASON_CHARS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS",
    "DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNING_CHARS",
    "IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION",
    "TRANSCRIPT_SEARCH_MIN_SNIPPET_CHARS",
    "BoundedImprovementEvidenceCollection",
    "ImprovementEvidenceCollectionBounds",
    "ImprovementEvidenceCollectionRequest",
    "ImprovementEvidenceCollectionResult",
    "ImprovementEvidenceSourceAdapter",
    "ImprovementObservabilitySnapshot",
    "ImprovementRecordStore",
    "ImprovementRuntimeBridge",
    "ImprovementRuntimeBridgeConfig",
    "ImprovementRuntimeChainSummary",
    "ImprovementRuntimeObservabilityEvent",
    "ImprovementRuntimeObservabilitySink",
    "ImprovementRuntimePermissionPolicy",
    "ImprovementRuntimePermissionReport",
    "ImprovementRuntimePermissionRequirement",
    "ImprovementRuntimePermissionStatus",
    "ImprovementRuntimePermissionSummary",
    "ImprovementRuntimeRecord",
    "ImprovementRuntimeRecordKind",
    "ImprovementRuntimeRecordQuery",
    "ImprovementRuntimeRecoveryRequest",
    "ImprovementRuntimeRecoveryResult",
    "ImprovementRuntimeRecoveryService",
    "ImprovementRuntimeRecoveryStatus",
    "ImprovementRuntimeRequest",
    "ImprovementRuntimeTransitionResult",
    "ImprovementRuntimeTransitionStatus",
    "ImprovementRuntimeValidationError",
    "ImprovementTaskOutputEvidenceTarget",
    "ImprovementTaskOutputReadMode",
    "KernelEventImprovementRuntimeObserver",
    "ObservabilitySnapshotImprovementEvidenceAdapter",
    "TaskOutputImprovementEvidenceAdapter",
    "TranscriptSearchImprovementEvidenceAdapter",
    "improvement_evidence_collection_result_from_dict",
    "improvement_evidence_collection_result_to_dict",
    "improvement_runtime_chain_summary_from_dict",
    "improvement_runtime_chain_summary_to_dict",
    "improvement_runtime_observability_event_from_dict",
    "improvement_runtime_observability_event_from_result",
    "improvement_runtime_observability_event_to_dict",
    "improvement_runtime_permission_report_from_dict",
    "improvement_runtime_permission_report_to_dict",
    "improvement_runtime_permission_report_to_summary",
    "improvement_runtime_permission_requirement_from_dict",
    "improvement_runtime_permission_requirement_to_dict",
    "improvement_runtime_permission_summary_from_dict",
    "improvement_runtime_permission_summary_to_dict",
    "improvement_runtime_record_from_dict",
    "improvement_runtime_record_to_dict",
    "improvement_runtime_recovery_result_from_dict",
    "improvement_runtime_recovery_result_to_dict",
    "recover_improvement_runtime_chain",
    "validate_improvement_evidence_collection",
)
