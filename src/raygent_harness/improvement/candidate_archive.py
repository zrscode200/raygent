"""Injected archive-store records for improvement patch candidates.

The RSI-004B layer persists archive-recommended candidate outcomes through a
caller-owned store protocol. It records the store reference, but it does not
ship a concrete archive backend, search archives, clean worktrees, promote
branches, execute shell commands, commit, or integrate product goals.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.candidate_outcome import (
    DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS,
    DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES,
    DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS,
    DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS,
    DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
    ImprovementPatchCandidateArchiveDecision,
    ImprovementPatchCandidateOutcome,
    ImprovementPatchCandidateOutcomeDecision,
)
from raygent_harness.improvement.models import ImprovementRequiredPermission

DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS = 1_000
DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS = 100
DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS = 20_000

ImprovementPatchCandidateArchiveStatus = Literal["archived"]

_ARCHIVE_STATUSES: frozenset[str] = frozenset({"archived"})
_OUTCOME_DECISIONS: frozenset[str] = frozenset(
    {"promotable", "reject", "needs_review"}
)
_REQUIRED_ARCHIVE_APPROVAL_PERMISSIONS: frozenset[str] = frozenset(
    {"filesystem_mutation"}
)
_APPROVAL_PERMISSIONS: frozenset[str] = frozenset(
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
_RAW_STORAGE_MARKERS: tuple[str, ...] = (
    "diff --git ",
    "Traceback (most recent call last)",
    "-----BEGIN ",
    "@@ ",
)
_COPIED_FILE_PREFIXES: tuple[str, ...] = (
    "#!",
    "class ",
    "def ",
    "from ",
    "import ",
    "package ",
    "module ",
)
_ARCHIVE_STORE_METADATA_KEY = "archive_store_metadata"


class ImprovementPatchCandidateArchiveError(ValueError):
    """Raised when a candidate archive cannot be produced."""


class ImprovementPatchCandidateArchiveValidationError(
    ImprovementPatchCandidateArchiveError
):
    """Raised when archive persistence data violates the RSI-004B contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveApproval:
    """Call-time authority to archive one candidate through an injected store."""

    approved_permissions: tuple[ImprovementRequiredPermission, ...]
    reason: str
    approved_by: str | None = None
    approved: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.approved:
            raise ImprovementPatchCandidateArchiveValidationError(
                "ImprovementPatchCandidateArchiveApproval.approved must be true"
            )
        object.__setattr__(
            self,
            "approved_permissions",
            _approval_permission_tuple(self.approved_permissions),
        )
        _require_non_empty(
            self.reason,
            "ImprovementPatchCandidateArchiveApproval.reason",
        )
        if self.approved_by is not None:
            _require_non_empty(
                self.approved_by,
                "ImprovementPatchCandidateArchiveApproval.approved_by",
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateArchiveApproval.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveRequest:
    """Normalized archive request passed to an injected archive store."""

    archive_decision_id: str
    outcome_id: str
    materialization_id: str
    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    base_revision: str
    patch_digest: str
    evaluation_id: str
    outcome_decision: ImprovementPatchCandidateOutcomeDecision
    archive_recommended: bool
    archive_reason: str
    summary: str
    failure_symptoms: tuple[str, ...] = ()
    artifact_references: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _validate_archive_request_fields(self)


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveStoreResult:
    """Reference returned by a caller-owned archive store."""

    storage_key: str
    storage_kind: str
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_storage_reference(
            self.storage_key,
            "ImprovementPatchCandidateArchiveStoreResult.storage_key",
            DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS,
            "DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS",
        )
        _require_storage_reference(
            self.storage_kind,
            "ImprovementPatchCandidateArchiveStoreResult.storage_kind",
            DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS,
            "DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ImprovementPatchCandidateArchiveStore(Protocol):
    """Caller-owned persistence seam for RSI-004B archive records."""

    async def archive(
        self,
        request: ImprovementPatchCandidateArchiveRequest,
    ) -> ImprovementPatchCandidateArchiveStoreResult:
        """Persist the supplied normalized archive request."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveRecord:
    """Serializable record for one persisted candidate archive reference."""

    archive_id: str
    archive_decision_id: str
    outcome_id: str
    materialization_id: str
    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    base_revision: str
    patch_digest: str
    evaluation_id: str
    outcome_decision: ImprovementPatchCandidateOutcomeDecision
    archive_recommended: bool
    archive_reason: str
    summary: str
    failure_symptoms: tuple[str, ...]
    artifact_references: tuple[str, ...]
    storage_key: str
    storage_kind: str
    archive_digest: str
    status: ImprovementPatchCandidateArchiveStatus = "archived"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.archive_id,
            "ImprovementPatchCandidateArchiveRecord.archive_id",
        )
        _require_literal(
            self.status,
            _ARCHIVE_STATUSES,
            "ImprovementPatchCandidateArchiveRecord.status",
        )
        _require_storage_reference(
            self.storage_key,
            "ImprovementPatchCandidateArchiveRecord.storage_key",
            DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS,
            "DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS",
        )
        _require_storage_reference(
            self.storage_kind,
            "ImprovementPatchCandidateArchiveRecord.storage_kind",
            DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS,
            "DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        request = _archive_request_from_record(self)
        expected_digest = _archive_digest(
            request,
            storage_key=self.storage_key,
            storage_kind=self.storage_kind,
        )
        if self.archive_digest != expected_digest:
            raise ImprovementPatchCandidateArchiveValidationError(
                "ImprovementPatchCandidateArchiveRecord.archive_digest does not "
                "match archive record identity"
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateArchiveRecord.created_at must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiver:
    """Archive one rejected or review-needed candidate through an injected store."""

    clock: Callable[[], float] = time.time
    archive_id_factory: Callable[[], str] | None = None

    async def archive(
        self,
        outcome: ImprovementPatchCandidateOutcome,
        archive_decision: ImprovementPatchCandidateArchiveDecision,
        *,
        archive_store: ImprovementPatchCandidateArchiveStore | None,
        approval: ImprovementPatchCandidateArchiveApproval | None,
        archive_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateArchiveRecord:
        """Invoke an injected archive store and return the immutable record."""

        _validate_outcome_archive_decision_linkage(outcome, archive_decision)
        _validate_archiveable_outcome(outcome, archive_decision)
        if archive_store is None:
            raise ImprovementPatchCandidateArchiveValidationError(
                "ImprovementPatchCandidateArchiver requires an injected "
                "ImprovementPatchCandidateArchiveStore"
            )
        if approval is None:
            raise ImprovementPatchCandidateArchiveValidationError(
                "ImprovementPatchCandidateArchiver requires explicit call-time "
                "approval"
            )
        _validate_approval(approval)

        request_metadata = _request_metadata(
            archive_decision.metadata,
            metadata or {},
        )
        request = ImprovementPatchCandidateArchiveRequest(
            archive_decision_id=archive_decision.archive_decision_id,
            outcome_id=archive_decision.outcome_id,
            materialization_id=archive_decision.materialization_id,
            allocation_id=archive_decision.allocation_id,
            candidate_id=archive_decision.candidate_id,
            run_id=archive_decision.run_id,
            proposal_id=archive_decision.proposal_id,
            gate_evaluation_id=archive_decision.gate_evaluation_id,
            base_revision=archive_decision.base_revision,
            patch_digest=archive_decision.patch_digest,
            evaluation_id=archive_decision.evaluation_id,
            outcome_decision=archive_decision.outcome_decision,
            archive_recommended=archive_decision.archive_recommended,
            archive_reason=archive_decision.archive_reason,
            summary=archive_decision.summary,
            failure_symptoms=archive_decision.failure_symptoms,
            artifact_references=archive_decision.artifact_references,
            metadata=request_metadata,
        )
        store_result = await archive_store.archive(request)
        record_metadata = _record_metadata(request.metadata, store_result.metadata)
        archive_digest = _archive_digest(
            request,
            storage_key=store_result.storage_key,
            storage_kind=store_result.storage_kind,
        )

        return ImprovementPatchCandidateArchiveRecord(
            archive_id=archive_id or self._new_archive_id(),
            archive_decision_id=request.archive_decision_id,
            outcome_id=request.outcome_id,
            materialization_id=request.materialization_id,
            allocation_id=request.allocation_id,
            candidate_id=request.candidate_id,
            run_id=request.run_id,
            proposal_id=request.proposal_id,
            gate_evaluation_id=request.gate_evaluation_id,
            base_revision=request.base_revision,
            patch_digest=request.patch_digest,
            evaluation_id=request.evaluation_id,
            outcome_decision=request.outcome_decision,
            archive_recommended=request.archive_recommended,
            archive_reason=request.archive_reason,
            summary=request.summary,
            failure_symptoms=request.failure_symptoms,
            artifact_references=request.artifact_references,
            storage_key=store_result.storage_key,
            storage_kind=store_result.storage_kind,
            archive_digest=archive_digest,
            status="archived",
            created_at=self.clock(),
            metadata=record_metadata,
        )

    def _new_archive_id(self) -> str:
        archive_id = (
            self.archive_id_factory()
            if self.archive_id_factory is not None
            else f"ipca_{uuid4().hex}"
        )
        if not archive_id.strip():
            raise ImprovementPatchCandidateArchiveValidationError(
                "archive_id_factory returned an empty id"
            )
        return archive_id


def improvement_patch_candidate_archive_request_to_dict(
    request: ImprovementPatchCandidateArchiveRequest,
) -> dict[str, object]:
    return {
        "archive_decision_id": request.archive_decision_id,
        "outcome_id": request.outcome_id,
        "materialization_id": request.materialization_id,
        "allocation_id": request.allocation_id,
        "candidate_id": request.candidate_id,
        "run_id": request.run_id,
        "proposal_id": request.proposal_id,
        "gate_evaluation_id": request.gate_evaluation_id,
        "base_revision": request.base_revision,
        "patch_digest": request.patch_digest,
        "evaluation_id": request.evaluation_id,
        "outcome_decision": request.outcome_decision,
        "archive_recommended": request.archive_recommended,
        "archive_reason": request.archive_reason,
        "summary": request.summary,
        "failure_symptoms": list(request.failure_symptoms),
        "artifact_references": list(request.artifact_references),
        "metadata": _metadata_to_dict(request.metadata),
    }


def improvement_patch_candidate_archive_request_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateArchiveRequest:
    return ImprovementPatchCandidateArchiveRequest(
        archive_decision_id=str(data["archive_decision_id"]),
        outcome_id=str(data["outcome_id"]),
        materialization_id=str(data["materialization_id"]),
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        base_revision=str(data["base_revision"]),
        patch_digest=str(data["patch_digest"]),
        evaluation_id=str(data["evaluation_id"]),
        outcome_decision=cast(
            ImprovementPatchCandidateOutcomeDecision,
            _literal_from_object(
                data["outcome_decision"],
                _OUTCOME_DECISIONS,
                "outcome_decision",
            ),
        ),
        archive_recommended=bool(data["archive_recommended"]),
        archive_reason=str(data["archive_reason"]),
        summary=str(data["summary"]),
        failure_symptoms=_string_tuple(
            data.get("failure_symptoms", ()),
            "failure_symptoms",
        ),
        artifact_references=_string_tuple(
            data.get("artifact_references", ()),
            "artifact_references",
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_archive_store_result_to_dict(
    result: ImprovementPatchCandidateArchiveStoreResult,
) -> dict[str, object]:
    return {
        "storage_key": result.storage_key,
        "storage_kind": result.storage_kind,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_patch_candidate_archive_store_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateArchiveStoreResult:
    return ImprovementPatchCandidateArchiveStoreResult(
        storage_key=str(data["storage_key"]),
        storage_kind=str(data["storage_kind"]),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_archive_record_to_dict(
    record: ImprovementPatchCandidateArchiveRecord,
) -> dict[str, object]:
    return {
        "archive_id": record.archive_id,
        "archive_decision_id": record.archive_decision_id,
        "outcome_id": record.outcome_id,
        "materialization_id": record.materialization_id,
        "allocation_id": record.allocation_id,
        "candidate_id": record.candidate_id,
        "run_id": record.run_id,
        "proposal_id": record.proposal_id,
        "gate_evaluation_id": record.gate_evaluation_id,
        "base_revision": record.base_revision,
        "patch_digest": record.patch_digest,
        "evaluation_id": record.evaluation_id,
        "outcome_decision": record.outcome_decision,
        "archive_recommended": record.archive_recommended,
        "archive_reason": record.archive_reason,
        "summary": record.summary,
        "failure_symptoms": list(record.failure_symptoms),
        "artifact_references": list(record.artifact_references),
        "storage_key": record.storage_key,
        "storage_kind": record.storage_kind,
        "archive_digest": record.archive_digest,
        "status": record.status,
        "created_at": record.created_at,
        "metadata": _metadata_to_dict(record.metadata),
    }


def improvement_patch_candidate_archive_record_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateArchiveRecord:
    return ImprovementPatchCandidateArchiveRecord(
        archive_id=str(data["archive_id"]),
        archive_decision_id=str(data["archive_decision_id"]),
        outcome_id=str(data["outcome_id"]),
        materialization_id=str(data["materialization_id"]),
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        base_revision=str(data["base_revision"]),
        patch_digest=str(data["patch_digest"]),
        evaluation_id=str(data["evaluation_id"]),
        outcome_decision=cast(
            ImprovementPatchCandidateOutcomeDecision,
            _literal_from_object(
                data["outcome_decision"],
                _OUTCOME_DECISIONS,
                "outcome_decision",
            ),
        ),
        archive_recommended=bool(data["archive_recommended"]),
        archive_reason=str(data["archive_reason"]),
        summary=str(data["summary"]),
        failure_symptoms=_string_tuple(
            data.get("failure_symptoms", ()),
            "failure_symptoms",
        ),
        artifact_references=_string_tuple(
            data.get("artifact_references", ()),
            "artifact_references",
        ),
        storage_key=str(data["storage_key"]),
        storage_kind=str(data["storage_kind"]),
        archive_digest=str(data["archive_digest"]),
        status=cast(
            ImprovementPatchCandidateArchiveStatus,
            _literal_from_object(
                data.get("status", "archived"),
                _ARCHIVE_STATUSES,
                "status",
            ),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_archive_request_fields(
    request: ImprovementPatchCandidateArchiveRequest,
) -> None:
    _require_non_empty(
        request.archive_decision_id,
        "ImprovementPatchCandidateArchiveRequest.archive_decision_id",
    )
    _require_non_empty(
        request.outcome_id,
        "ImprovementPatchCandidateArchiveRequest.outcome_id",
    )
    _require_non_empty(
        request.materialization_id,
        "ImprovementPatchCandidateArchiveRequest.materialization_id",
    )
    _require_non_empty(
        request.allocation_id,
        "ImprovementPatchCandidateArchiveRequest.allocation_id",
    )
    _require_non_empty(
        request.candidate_id,
        "ImprovementPatchCandidateArchiveRequest.candidate_id",
    )
    _require_non_empty(
        request.run_id,
        "ImprovementPatchCandidateArchiveRequest.run_id",
    )
    _require_non_empty(
        request.proposal_id,
        "ImprovementPatchCandidateArchiveRequest.proposal_id",
    )
    _require_non_empty(
        request.gate_evaluation_id,
        "ImprovementPatchCandidateArchiveRequest.gate_evaluation_id",
    )
    _require_non_empty(
        request.base_revision,
        "ImprovementPatchCandidateArchiveRequest.base_revision",
    )
    _require_non_empty(
        request.patch_digest,
        "ImprovementPatchCandidateArchiveRequest.patch_digest",
    )
    _require_non_empty(
        request.evaluation_id,
        "ImprovementPatchCandidateArchiveRequest.evaluation_id",
    )
    _require_literal(
        request.outcome_decision,
        _OUTCOME_DECISIONS,
        "ImprovementPatchCandidateArchiveRequest.outcome_decision",
    )
    if request.outcome_decision == "promotable":
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveRequest cannot archive promotable outcomes"
        )
    if not request.archive_recommended:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveRequest.archive_recommended must be true"
        )
    _require_bounded_non_empty(
        request.archive_reason,
        "ImprovementPatchCandidateArchiveRequest.archive_reason",
        DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
        "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
    )
    _require_bounded_non_empty(
        request.summary,
        "ImprovementPatchCandidateArchiveRequest.summary",
        DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
        "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
    )
    object.__setattr__(
        request,
        "failure_symptoms",
        _bounded_unique_string_tuple(
            request.failure_symptoms,
            "ImprovementPatchCandidateArchiveRequest.failure_symptoms",
            DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS,
            DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS,
            "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS",
            "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS",
        ),
    )
    object.__setattr__(
        request,
        "artifact_references",
        _artifact_reference_tuple(
            request.artifact_references,
            "ImprovementPatchCandidateArchiveRequest.artifact_references",
        ),
    )
    metadata = _freeze_metadata(request.metadata)
    _reject_reserved_request_metadata_key(
        metadata,
        "ImprovementPatchCandidateArchiveRequest.metadata",
    )
    object.__setattr__(request, "metadata", metadata)


def _validate_outcome_archive_decision_linkage(
    outcome: ImprovementPatchCandidateOutcome,
    archive_decision: ImprovementPatchCandidateArchiveDecision,
) -> None:
    fields = (
        ("outcome_id", "outcome_id"),
        ("materialization_id", "materialization_id"),
        ("allocation_id", "allocation_id"),
        ("candidate_id", "candidate_id"),
        ("run_id", "run_id"),
        ("proposal_id", "proposal_id"),
        ("gate_evaluation_id", "gate_evaluation_id"),
        ("base_revision", "base_revision"),
        ("patch_digest", "patch_digest"),
        ("evaluation_id", "evaluation_id"),
        ("decision", "outcome_decision"),
    )
    for outcome_field, decision_field in fields:
        if getattr(outcome, outcome_field) != getattr(archive_decision, decision_field):
            raise ImprovementPatchCandidateArchiveValidationError(
                "ImprovementPatchCandidateArchiver linkage mismatch: "
                f"{decision_field}"
            )


def _validate_archiveable_outcome(
    outcome: ImprovementPatchCandidateOutcome,
    archive_decision: ImprovementPatchCandidateArchiveDecision,
) -> None:
    if outcome.decision == "promotable" or archive_decision.outcome_decision == (
        "promotable"
    ):
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiver cannot archive promotable outcomes"
        )
    if not outcome.archive_recommended:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiver requires an archive-recommended outcome"
        )
    if not archive_decision.archive_recommended:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiver requires an archive-recommended "
            "archive decision"
        )


def _validate_approval(approval: ImprovementPatchCandidateArchiveApproval) -> None:
    if not approval.approved:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved must be true"
        )
    missing = _REQUIRED_ARCHIVE_APPROVAL_PERMISSIONS.difference(
        approval.approved_permissions
    )
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved_permissions "
            f"missing: {joined}"
        )


def _approval_permission_tuple(
    permissions: Sequence[str],
) -> tuple[ImprovementRequiredPermission, ...]:
    items = tuple(
        cast(
            ImprovementRequiredPermission,
            _literal_from_object(item, _APPROVAL_PERMISSIONS, "approved_permissions"),
        )
        for item in permissions
    )
    if not items:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved_permissions "
            "must not be empty"
        )
    if "none" in items:
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved_permissions "
            "cannot include none"
        )
    missing = _REQUIRED_ARCHIVE_APPROVAL_PERMISSIONS.difference(items)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved_permissions "
            f"missing: {joined}"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateArchiveValidationError(
            "ImprovementPatchCandidateArchiveApproval.approved_permissions "
            f"contains duplicates: {joined}"
        )
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _archive_request_from_record(
    record: ImprovementPatchCandidateArchiveRecord,
) -> ImprovementPatchCandidateArchiveRequest:
    return ImprovementPatchCandidateArchiveRequest(
        archive_decision_id=record.archive_decision_id,
        outcome_id=record.outcome_id,
        materialization_id=record.materialization_id,
        allocation_id=record.allocation_id,
        candidate_id=record.candidate_id,
        run_id=record.run_id,
        proposal_id=record.proposal_id,
        gate_evaluation_id=record.gate_evaluation_id,
        base_revision=record.base_revision,
        patch_digest=record.patch_digest,
        evaluation_id=record.evaluation_id,
        outcome_decision=record.outcome_decision,
        archive_recommended=record.archive_recommended,
        archive_reason=record.archive_reason,
        summary=record.summary,
        failure_symptoms=record.failure_symptoms,
        artifact_references=record.artifact_references,
        metadata=_identity_metadata(record.metadata),
    )


def _archive_digest(
    request: ImprovementPatchCandidateArchiveRequest,
    *,
    storage_key: str,
    storage_kind: str,
) -> str:
    payload = {
        "request": improvement_patch_candidate_archive_request_to_dict(request),
        "storage_key": storage_key,
        "storage_kind": storage_kind,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _request_metadata(
    archive_decision_metadata: Mapping[str, FrozenJson],
    metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    archive_decision_data = _metadata_to_dict(archive_decision_metadata)
    caller_data = _metadata_to_dict(metadata)
    _reject_reserved_request_metadata_key(
        archive_decision_data,
        "ImprovementPatchCandidateArchiveDecision.metadata",
    )
    _reject_reserved_request_metadata_key(
        caller_data,
        "ImprovementPatchCandidateArchiver.metadata",
    )
    combined = dict(archive_decision_data)
    combined.update(caller_data)
    return _freeze_metadata(combined)


def _record_metadata(
    request_metadata: Mapping[str, FrozenJson],
    store_metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    combined = _metadata_to_dict(request_metadata)
    if store_metadata:
        combined[_ARCHIVE_STORE_METADATA_KEY] = _metadata_to_dict(store_metadata)
    return _freeze_metadata(combined)


def _identity_metadata(metadata: Mapping[str, FrozenJson]) -> Mapping[str, FrozenJson]:
    identity = {
        key: value
        for key, value in _metadata_to_dict(metadata).items()
        if key != _ARCHIVE_STORE_METADATA_KEY
    }
    return _freeze_metadata(identity)


def _reject_reserved_request_metadata_key(
    metadata: Mapping[str, object],
    name: str,
) -> None:
    if _ARCHIVE_STORE_METADATA_KEY in metadata:
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} cannot include reserved {_ARCHIVE_STORE_METADATA_KEY}"
        )


def _require_storage_reference(
    value: str,
    name: str,
    max_chars: int,
    max_chars_name: str,
) -> None:
    _require_bounded_non_empty(value, name, max_chars, max_chars_name)
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} must be a single-line storage reference"
        )
    stripped = value.strip()
    if any(marker in stripped for marker in _RAW_STORAGE_MARKERS):
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} must not contain raw output or diff payloads"
        )
    if stripped.startswith(_COPIED_FILE_PREFIXES):
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} must not look like copied file contents"
        )


def _artifact_reference_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = _bounded_unique_string_tuple(
        values,
        name,
        DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES,
        DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS,
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES",
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS",
    )
    for item in items:
        if "\n" in item or "\r" in item or "\x00" in item:
            raise ImprovementPatchCandidateArchiveValidationError(
                f"{name} must be single-line references"
            )
        if any(marker in item for marker in _RAW_STORAGE_MARKERS):
            raise ImprovementPatchCandidateArchiveValidationError(
                f"{name} must not contain raw output or diff payloads"
            )
    return items


def _bounded_unique_string_tuple(
    values: Sequence[str],
    name: str,
    max_items: int,
    max_chars: int,
    max_items_name: str,
    max_chars_name: str,
) -> tuple[str, ...]:
    items = tuple(values)
    if len(items) > max_items:
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} exceeds {max_items_name}"
        )
    for item in items:
        _require_bounded_non_empty(item, name, max_chars, max_chars_name)
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} contains duplicates: {joined}"
        )
    return items


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    return tuple(dict.fromkeys(duplicates))


def _require_bounded_non_empty(
    value: str,
    name: str,
    max_chars: int,
    max_chars_name: str,
) -> None:
    _require_non_empty(value, name)
    if len(value) > max_chars:
        raise ImprovementPatchCandidateArchiveValidationError(
            f"{name} exceeds {max_chars_name}"
        )


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_literal(value: str, allowed: frozenset[str], name: str) -> None:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {expected}")


def _literal_from_object(value: object, allowed: frozenset[str], name: str) -> str:
    text = str(value)
    _require_literal(text, allowed, name)
    return text


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, FrozenJson]:
    frozen = freeze_json(metadata)
    if not isinstance(frozen, Mapping):
        raise TypeError("metadata must serialize to a JSON object")
    frozen_mapping = cast(Mapping[str, FrozenJson], frozen)
    encoded = json.dumps(
        _metadata_to_dict(frozen_mapping),
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded) > DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS:
        raise ImprovementPatchCandidateArchiveValidationError(
            "metadata exceeds DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS"
        )
    return frozen_mapping


def _metadata_from_object(value: object) -> Mapping[str, FrozenJson]:
    return _freeze_metadata(_expect_mapping(value, "metadata"))


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


def _expect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    sequence = cast(Sequence[object], value)
    return tuple(str(item) for item in sequence)


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a number") from exc


__all__ = (
    "DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS",
    "DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS",
    "DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS",
    "ImprovementPatchCandidateArchiveApproval",
    "ImprovementPatchCandidateArchiveError",
    "ImprovementPatchCandidateArchiveRecord",
    "ImprovementPatchCandidateArchiveRequest",
    "ImprovementPatchCandidateArchiveStatus",
    "ImprovementPatchCandidateArchiveStore",
    "ImprovementPatchCandidateArchiveStoreResult",
    "ImprovementPatchCandidateArchiveValidationError",
    "ImprovementPatchCandidateArchiver",
    "improvement_patch_candidate_archive_record_from_dict",
    "improvement_patch_candidate_archive_record_to_dict",
    "improvement_patch_candidate_archive_request_from_dict",
    "improvement_patch_candidate_archive_request_to_dict",
    "improvement_patch_candidate_archive_store_result_from_dict",
    "improvement_patch_candidate_archive_store_result_to_dict",
)
