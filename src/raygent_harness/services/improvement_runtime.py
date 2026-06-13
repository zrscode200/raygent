"""Optional runtime bridge contracts for bounded self-improvement.

The RSI-006A layer defines data and protocol seams only. It does not wire the
query loop, SDK factory, goal runtime, tools, concrete filesystem writers,
shell runners, Git, network, or CI into self-improvement behavior.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, Protocol, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement import (
    ImprovementEvidence,
    ImprovementEvidenceBounds,
    ImprovementEvidenceSource,
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
ImprovementTaskOutputReadMode = Literal["tail", "range"]

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
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(
                self.metadata,
                "ImprovementRuntimeChainSummary.metadata",
                DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
            ),
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
class ImprovementRuntimeBridgeConfig:
    """Explicit opt-in configuration for the improvement runtime bridge."""

    enabled: bool = False
    record_store: ImprovementRecordStore | None = None
    evidence_sources: tuple[ImprovementEvidenceSourceAdapter, ...] = ()
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
            return _runtime_result(
                request,
                status="not_enabled",
                blocked_reason="improvement runtime bridge is not enabled",
                record_kind="runtime_not_enabled",
                sequence=request.record_sequence,
                created_at=self.clock(),
                record_id=self._new_record_id(),
            )
        if request.collection_request is None:
            return _runtime_result(
                request,
                status="blocked",
                blocked_reason="evidence collection request is required",
                record_kind="runtime_blocked",
                sequence=request.record_sequence,
                created_at=self.clock(),
                record_id=self._new_record_id(),
            )

        evidence_sources = request.evidence_sources or self.config.evidence_sources
        if not evidence_sources:
            return _runtime_result(
                request,
                status="blocked",
                blocked_reason="no improvement evidence source adapters were supplied",
                record_kind="runtime_blocked",
                sequence=request.record_sequence,
                created_at=self.clock(),
                record_id=self._new_record_id(),
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
            metadata=request.metadata,
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
        return ImprovementRuntimeTransitionResult(
            request_id=request.request_id,
            session_id=request.session_id,
            runtime_session_id=request.runtime_session_id,
            status="completed",
            records=(record,),
            summary=summary,
            evidence=bounded.evidence,
            warnings=bounded.warnings,
            metadata={
                "evidence_count": len(bounded.evidence),
                "truncated": bounded.truncated,
            },
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
        metadata=request.metadata,
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
    "ImprovementRuntimeRecord",
    "ImprovementRuntimeRecordKind",
    "ImprovementRuntimeRecordQuery",
    "ImprovementRuntimeRequest",
    "ImprovementRuntimeTransitionResult",
    "ImprovementRuntimeTransitionStatus",
    "ImprovementRuntimeValidationError",
    "ImprovementTaskOutputEvidenceTarget",
    "ImprovementTaskOutputReadMode",
    "ObservabilitySnapshotImprovementEvidenceAdapter",
    "TaskOutputImprovementEvidenceAdapter",
    "TranscriptSearchImprovementEvidenceAdapter",
    "improvement_evidence_collection_result_from_dict",
    "improvement_evidence_collection_result_to_dict",
    "improvement_runtime_chain_summary_from_dict",
    "improvement_runtime_chain_summary_to_dict",
    "improvement_runtime_record_from_dict",
    "improvement_runtime_record_to_dict",
    "validate_improvement_evidence_collection",
)
