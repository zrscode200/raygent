"""Injected promotion-attempt records for improvement patch candidates.

The RSI-004C layer records one caller-owned promotion attempt for a promotable
candidate outcome through an injected promoter protocol. It records bounded
references only; it does not ship a concrete Git promoter, run shell commands,
commit, push, open pull requests, call CI, clean worktrees, search archives, or
integrate product goals.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.candidate_materialization import (
    ImprovementPatchCandidateEvaluationDecision,
)
from raygent_harness.improvement.candidate_outcome import (
    ImprovementPatchCandidateOutcome,
    ImprovementPatchCandidateOutcomeDecision,
)
from raygent_harness.improvement.models import ImprovementRequiredPermission

DEFAULT_MAX_PROMOTION_REF_CHARS = 1_000
DEFAULT_MAX_PROMOTION_KIND_CHARS = 100
DEFAULT_MAX_PROMOTED_FILES = 100
DEFAULT_MAX_PROMOTED_FILE_CHARS = 1_000
DEFAULT_MAX_PROMOTION_SUMMARY_CHARS = 4_000
DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS = 20_000

ImprovementPatchCandidatePromotionStatus = Literal["promotion_recorded"]

_PROMOTION_STATUSES: frozenset[str] = frozenset({"promotion_recorded"})
_OUTCOME_DECISIONS: frozenset[str] = frozenset(
    {"promotable", "reject", "needs_review"}
)
_EVALUATION_DECISIONS: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "needs_review"}
)
_CORE_PROMOTION_PERMISSIONS: tuple[ImprovementRequiredPermission, ...] = (
    "human_review",
    "filesystem_mutation",
    "commit",
)
_CORE_PROMOTION_PERMISSION_SET: frozenset[str] = frozenset(
    _CORE_PROMOTION_PERMISSIONS
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
_RAW_PROMOTION_MARKERS: tuple[str, ...] = (
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
_PROMOTION_RESULT_METADATA_KEY = "promotion_result_metadata"


class ImprovementPatchCandidatePromotionError(ValueError):
    """Raised when a candidate promotion attempt cannot be recorded."""


class ImprovementPatchCandidatePromotionValidationError(
    ImprovementPatchCandidatePromotionError
):
    """Raised when promotion-attempt data violates the RSI-004C contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePromotionApproval:
    """Call-time authority to attempt one injected candidate promotion."""

    approved_permissions: tuple[ImprovementRequiredPermission, ...]
    reason: str
    approved_by: str
    approved: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.approved:
            raise ImprovementPatchCandidatePromotionValidationError(
                "ImprovementPatchCandidatePromotionApproval.approved must be true"
            )
        object.__setattr__(
            self,
            "approved_permissions",
            _exact_core_permission_tuple(
                self.approved_permissions,
                "ImprovementPatchCandidatePromotionApproval.approved_permissions",
            ),
        )
        _require_non_empty(
            self.reason,
            "ImprovementPatchCandidatePromotionApproval.reason",
        )
        _require_non_empty(
            self.approved_by,
            "ImprovementPatchCandidatePromotionApproval.approved_by",
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidatePromotionApproval.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePromotionRequest:
    """Normalized request passed to an injected promoter."""

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
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision
    outcome_decision: ImprovementPatchCandidateOutcomeDecision
    summary: str
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _validate_promotion_request_fields(self)


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePromotionResult:
    """Bounded reference returned by a caller-owned promoter."""

    promotion_ref: str
    promotion_kind: str
    source_worktree_ref: str
    target_ref: str
    target_revision: str
    promoted_files: tuple[str, ...]
    summary: str
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_promotion_reference(
            self.promotion_ref,
            "ImprovementPatchCandidatePromotionResult.promotion_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.promotion_kind,
            "ImprovementPatchCandidatePromotionResult.promotion_kind",
            DEFAULT_MAX_PROMOTION_KIND_CHARS,
            "DEFAULT_MAX_PROMOTION_KIND_CHARS",
        )
        _require_promotion_reference(
            self.source_worktree_ref,
            "ImprovementPatchCandidatePromotionResult.source_worktree_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.target_ref,
            "ImprovementPatchCandidatePromotionResult.target_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.target_revision,
            "ImprovementPatchCandidatePromotionResult.target_revision",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        object.__setattr__(
            self,
            "promoted_files",
            _promoted_file_tuple(
                self.promoted_files,
                "ImprovementPatchCandidatePromotionResult.promoted_files",
            ),
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidatePromotionResult.summary",
            DEFAULT_MAX_PROMOTION_SUMMARY_CHARS,
            "DEFAULT_MAX_PROMOTION_SUMMARY_CHARS",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ImprovementPatchCandidatePromoter(Protocol):
    """Caller-owned promotion seam for RSI-004C records."""

    async def promote(
        self,
        request: ImprovementPatchCandidatePromotionRequest,
    ) -> ImprovementPatchCandidatePromotionResult:
        """Attempt promotion for the supplied normalized request."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePromotionRecord:
    """Serializable record for one injected candidate promotion attempt."""

    promotion_id: str
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
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision
    outcome_decision: ImprovementPatchCandidateOutcomeDecision
    summary: str
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    promotion_ref: str
    promotion_kind: str
    source_worktree_ref: str
    target_ref: str
    target_revision: str
    promoted_files: tuple[str, ...]
    promotion_digest: str
    status: ImprovementPatchCandidatePromotionStatus = "promotion_recorded"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.promotion_id,
            "ImprovementPatchCandidatePromotionRecord.promotion_id",
        )
        _require_literal(
            self.status,
            _PROMOTION_STATUSES,
            "ImprovementPatchCandidatePromotionRecord.status",
        )
        _require_promotion_reference(
            self.promotion_ref,
            "ImprovementPatchCandidatePromotionRecord.promotion_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.promotion_kind,
            "ImprovementPatchCandidatePromotionRecord.promotion_kind",
            DEFAULT_MAX_PROMOTION_KIND_CHARS,
            "DEFAULT_MAX_PROMOTION_KIND_CHARS",
        )
        _require_promotion_reference(
            self.source_worktree_ref,
            "ImprovementPatchCandidatePromotionRecord.source_worktree_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.target_ref,
            "ImprovementPatchCandidatePromotionRecord.target_ref",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        _require_promotion_reference(
            self.target_revision,
            "ImprovementPatchCandidatePromotionRecord.target_revision",
            DEFAULT_MAX_PROMOTION_REF_CHARS,
            "DEFAULT_MAX_PROMOTION_REF_CHARS",
        )
        object.__setattr__(
            self,
            "promoted_files",
            _promoted_file_tuple(
                self.promoted_files,
                "ImprovementPatchCandidatePromotionRecord.promoted_files",
            ),
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        request = _promotion_request_from_record(self)
        expected_digest = _promotion_digest(
            request,
            promotion_ref=self.promotion_ref,
            promotion_kind=self.promotion_kind,
            source_worktree_ref=self.source_worktree_ref,
            target_ref=self.target_ref,
            target_revision=self.target_revision,
            promoted_files=self.promoted_files,
        )
        if self.promotion_digest != expected_digest:
            raise ImprovementPatchCandidatePromotionValidationError(
                "ImprovementPatchCandidatePromotionRecord.promotion_digest does not "
                "match promotion record identity"
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidatePromotionRecord.created_at must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePromotionService:
    """Record one promotable candidate through an injected promoter."""

    clock: Callable[[], float] = time.time
    promotion_id_factory: Callable[[], str] | None = None

    async def promote(
        self,
        outcome: ImprovementPatchCandidateOutcome,
        *,
        promoter: ImprovementPatchCandidatePromoter | None,
        approval: ImprovementPatchCandidatePromotionApproval | None,
        promotion_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidatePromotionRecord:
        """Invoke an injected promoter and return the immutable promotion record."""

        _validate_promotable_outcome(outcome)
        if promoter is None:
            raise ImprovementPatchCandidatePromotionValidationError(
                "ImprovementPatchCandidatePromotionService requires an injected "
                "ImprovementPatchCandidatePromoter"
            )
        if approval is None:
            raise ImprovementPatchCandidatePromotionValidationError(
                "ImprovementPatchCandidatePromotionService requires explicit "
                "call-time approval"
            )
        _validate_approval(approval)

        request_metadata = _request_metadata(outcome.metadata, metadata or {})
        request = ImprovementPatchCandidatePromotionRequest(
            outcome_id=outcome.outcome_id,
            materialization_id=outcome.materialization_id,
            allocation_id=outcome.allocation_id,
            candidate_id=outcome.candidate_id,
            run_id=outcome.run_id,
            proposal_id=outcome.proposal_id,
            gate_evaluation_id=outcome.gate_evaluation_id,
            base_revision=outcome.base_revision,
            patch_digest=outcome.patch_digest,
            evaluation_id=outcome.evaluation_id,
            evaluation_decision=outcome.evaluation_decision,
            outcome_decision=outcome.decision,
            summary=outcome.summary,
            required_permissions=outcome.required_permissions,
            metadata=request_metadata,
        )
        result = await promoter.promote(request)
        record_metadata = _record_metadata(request.metadata, result.metadata)
        promotion_digest = _promotion_digest(
            request,
            promotion_ref=result.promotion_ref,
            promotion_kind=result.promotion_kind,
            source_worktree_ref=result.source_worktree_ref,
            target_ref=result.target_ref,
            target_revision=result.target_revision,
            promoted_files=result.promoted_files,
        )

        return ImprovementPatchCandidatePromotionRecord(
            promotion_id=promotion_id or self._new_promotion_id(),
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
            evaluation_decision=request.evaluation_decision,
            outcome_decision=request.outcome_decision,
            summary=request.summary,
            required_permissions=request.required_permissions,
            promotion_ref=result.promotion_ref,
            promotion_kind=result.promotion_kind,
            source_worktree_ref=result.source_worktree_ref,
            target_ref=result.target_ref,
            target_revision=result.target_revision,
            promoted_files=result.promoted_files,
            promotion_digest=promotion_digest,
            status="promotion_recorded",
            created_at=self.clock(),
            metadata=record_metadata,
        )

    def _new_promotion_id(self) -> str:
        promotion_id = (
            self.promotion_id_factory()
            if self.promotion_id_factory is not None
            else f"ipcp_{uuid4().hex}"
        )
        if not promotion_id.strip():
            raise ImprovementPatchCandidatePromotionValidationError(
                "promotion_id_factory returned an empty id"
            )
        return promotion_id


def improvement_patch_candidate_promotion_request_to_dict(
    request: ImprovementPatchCandidatePromotionRequest,
) -> dict[str, object]:
    return {
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
        "evaluation_decision": request.evaluation_decision,
        "outcome_decision": request.outcome_decision,
        "summary": request.summary,
        "required_permissions": list(request.required_permissions),
        "metadata": _metadata_to_dict(request.metadata),
    }


def improvement_patch_candidate_promotion_request_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidatePromotionRequest:
    return ImprovementPatchCandidatePromotionRequest(
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
        evaluation_decision=cast(
            ImprovementPatchCandidateEvaluationDecision,
            _literal_from_object(
                data["evaluation_decision"],
                _EVALUATION_DECISIONS,
                "evaluation_decision",
            ),
        ),
        outcome_decision=cast(
            ImprovementPatchCandidateOutcomeDecision,
            _literal_from_object(
                data["outcome_decision"],
                _OUTCOME_DECISIONS,
                "outcome_decision",
            ),
        ),
        summary=str(data["summary"]),
        required_permissions=cast(
            tuple[ImprovementRequiredPermission, ...],
            _string_tuple(
                data.get("required_permissions", ()),
                "required_permissions",
            ),
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_promotion_result_to_dict(
    result: ImprovementPatchCandidatePromotionResult,
) -> dict[str, object]:
    return {
        "promotion_ref": result.promotion_ref,
        "promotion_kind": result.promotion_kind,
        "source_worktree_ref": result.source_worktree_ref,
        "target_ref": result.target_ref,
        "target_revision": result.target_revision,
        "promoted_files": list(result.promoted_files),
        "summary": result.summary,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_patch_candidate_promotion_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidatePromotionResult:
    return ImprovementPatchCandidatePromotionResult(
        promotion_ref=str(data["promotion_ref"]),
        promotion_kind=str(data["promotion_kind"]),
        source_worktree_ref=str(data["source_worktree_ref"]),
        target_ref=str(data["target_ref"]),
        target_revision=str(data["target_revision"]),
        promoted_files=_string_tuple(data.get("promoted_files", ()), "promoted_files"),
        summary=str(data["summary"]),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_promotion_record_to_dict(
    record: ImprovementPatchCandidatePromotionRecord,
) -> dict[str, object]:
    return {
        "promotion_id": record.promotion_id,
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
        "evaluation_decision": record.evaluation_decision,
        "outcome_decision": record.outcome_decision,
        "summary": record.summary,
        "required_permissions": list(record.required_permissions),
        "promotion_ref": record.promotion_ref,
        "promotion_kind": record.promotion_kind,
        "source_worktree_ref": record.source_worktree_ref,
        "target_ref": record.target_ref,
        "target_revision": record.target_revision,
        "promoted_files": list(record.promoted_files),
        "promotion_digest": record.promotion_digest,
        "status": record.status,
        "created_at": record.created_at,
        "metadata": _metadata_to_dict(record.metadata),
    }


def improvement_patch_candidate_promotion_record_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidatePromotionRecord:
    return ImprovementPatchCandidatePromotionRecord(
        promotion_id=str(data["promotion_id"]),
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
        evaluation_decision=cast(
            ImprovementPatchCandidateEvaluationDecision,
            _literal_from_object(
                data["evaluation_decision"],
                _EVALUATION_DECISIONS,
                "evaluation_decision",
            ),
        ),
        outcome_decision=cast(
            ImprovementPatchCandidateOutcomeDecision,
            _literal_from_object(
                data["outcome_decision"],
                _OUTCOME_DECISIONS,
                "outcome_decision",
            ),
        ),
        summary=str(data["summary"]),
        required_permissions=cast(
            tuple[ImprovementRequiredPermission, ...],
            _string_tuple(
                data.get("required_permissions", ()),
                "required_permissions",
            ),
        ),
        promotion_ref=str(data["promotion_ref"]),
        promotion_kind=str(data["promotion_kind"]),
        source_worktree_ref=str(data["source_worktree_ref"]),
        target_ref=str(data["target_ref"]),
        target_revision=str(data["target_revision"]),
        promoted_files=_string_tuple(data.get("promoted_files", ()), "promoted_files"),
        promotion_digest=str(data["promotion_digest"]),
        status=cast(
            ImprovementPatchCandidatePromotionStatus,
            _literal_from_object(
                data.get("status", "promotion_recorded"),
                _PROMOTION_STATUSES,
                "status",
            ),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_promotion_request_fields(
    request: ImprovementPatchCandidatePromotionRequest,
) -> None:
    _require_non_empty(
        request.outcome_id,
        "ImprovementPatchCandidatePromotionRequest.outcome_id",
    )
    _require_non_empty(
        request.materialization_id,
        "ImprovementPatchCandidatePromotionRequest.materialization_id",
    )
    _require_non_empty(
        request.allocation_id,
        "ImprovementPatchCandidatePromotionRequest.allocation_id",
    )
    _require_non_empty(
        request.candidate_id,
        "ImprovementPatchCandidatePromotionRequest.candidate_id",
    )
    _require_non_empty(
        request.run_id,
        "ImprovementPatchCandidatePromotionRequest.run_id",
    )
    _require_non_empty(
        request.proposal_id,
        "ImprovementPatchCandidatePromotionRequest.proposal_id",
    )
    _require_non_empty(
        request.gate_evaluation_id,
        "ImprovementPatchCandidatePromotionRequest.gate_evaluation_id",
    )
    _require_non_empty(
        request.base_revision,
        "ImprovementPatchCandidatePromotionRequest.base_revision",
    )
    _require_non_empty(
        request.patch_digest,
        "ImprovementPatchCandidatePromotionRequest.patch_digest",
    )
    _require_non_empty(
        request.evaluation_id,
        "ImprovementPatchCandidatePromotionRequest.evaluation_id",
    )
    _require_literal(
        request.evaluation_decision,
        _EVALUATION_DECISIONS,
        "ImprovementPatchCandidatePromotionRequest.evaluation_decision",
    )
    if request.evaluation_decision != "pass":
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionRequest.evaluation_decision must be pass"
        )
    _require_literal(
        request.outcome_decision,
        _OUTCOME_DECISIONS,
        "ImprovementPatchCandidatePromotionRequest.outcome_decision",
    )
    if request.outcome_decision != "promotable":
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionRequest.outcome_decision must be "
            "promotable"
        )
    _require_bounded_non_empty(
        request.summary,
        "ImprovementPatchCandidatePromotionRequest.summary",
        DEFAULT_MAX_PROMOTION_SUMMARY_CHARS,
        "DEFAULT_MAX_PROMOTION_SUMMARY_CHARS",
    )
    object.__setattr__(
        request,
        "required_permissions",
        _exact_core_permission_tuple(
            request.required_permissions,
            "ImprovementPatchCandidatePromotionRequest.required_permissions",
        ),
    )
    metadata = _freeze_metadata(request.metadata)
    _reject_reserved_request_metadata_key(
        metadata,
        "ImprovementPatchCandidatePromotionRequest.metadata",
    )
    object.__setattr__(request, "metadata", metadata)


def _validate_promotable_outcome(
    outcome: ImprovementPatchCandidateOutcome,
) -> None:
    if outcome.decision != "promotable":
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionService requires a promotable outcome"
        )
    if outcome.evaluation_decision != "pass":
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionService requires a pass evaluation"
        )
    if outcome.archive_recommended:
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionService rejects archive-recommended "
            "outcomes"
        )
    if outcome.promotion_blockers:
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionService rejects outcomes with "
            "promotion_blockers"
        )
    _exact_core_permission_tuple(
        outcome.required_permissions,
        "ImprovementPatchCandidateOutcome.required_permissions",
    )


def _validate_approval(approval: ImprovementPatchCandidatePromotionApproval) -> None:
    if not approval.approved:
        raise ImprovementPatchCandidatePromotionValidationError(
            "ImprovementPatchCandidatePromotionApproval.approved must be true"
        )
    _exact_core_permission_tuple(
        approval.approved_permissions,
        "ImprovementPatchCandidatePromotionApproval.approved_permissions",
    )


def _exact_core_permission_tuple(
    permissions: Sequence[str],
    name: str,
) -> tuple[ImprovementRequiredPermission, ...]:
    items = tuple(
        cast(
            ImprovementRequiredPermission,
            _literal_from_object(item, _APPROVAL_PERMISSIONS, name),
        )
        for item in permissions
    )
    if not items:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not be empty"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} contains duplicates: {joined}"
        )
    item_set = frozenset(items)
    extra = item_set.difference(_CORE_PROMOTION_PERMISSION_SET)
    if extra:
        joined = ", ".join(sorted(extra))
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} contains unsupported promotion permissions: {joined}"
        )
    missing = _CORE_PROMOTION_PERMISSION_SET.difference(item_set)
    if missing:
        joined = ", ".join(
            permission for permission in _CORE_PROMOTION_PERMISSIONS if permission in missing
        )
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} missing required promotion permissions: {joined}"
        )
    return _CORE_PROMOTION_PERMISSIONS


def _promotion_request_from_record(
    record: ImprovementPatchCandidatePromotionRecord,
) -> ImprovementPatchCandidatePromotionRequest:
    return ImprovementPatchCandidatePromotionRequest(
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
        evaluation_decision=record.evaluation_decision,
        outcome_decision=record.outcome_decision,
        summary=record.summary,
        required_permissions=record.required_permissions,
        metadata=_identity_metadata(record.metadata),
    )


def _promotion_digest(
    request: ImprovementPatchCandidatePromotionRequest,
    *,
    promotion_ref: str,
    promotion_kind: str,
    source_worktree_ref: str,
    target_ref: str,
    target_revision: str,
    promoted_files: Sequence[str],
) -> str:
    payload = {
        "request": improvement_patch_candidate_promotion_request_to_dict(request),
        "promotion_ref": promotion_ref,
        "promotion_kind": promotion_kind,
        "source_worktree_ref": source_worktree_ref,
        "target_ref": target_ref,
        "target_revision": target_revision,
        "promoted_files": list(promoted_files),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _request_metadata(
    outcome_metadata: Mapping[str, FrozenJson],
    metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    outcome_data = _metadata_to_dict(outcome_metadata)
    caller_data = _metadata_to_dict(metadata)
    _reject_reserved_request_metadata_key(
        outcome_data,
        "ImprovementPatchCandidateOutcome.metadata",
    )
    _reject_reserved_request_metadata_key(
        caller_data,
        "ImprovementPatchCandidatePromotionService.metadata",
    )
    combined = dict(outcome_data)
    combined.update(caller_data)
    return _freeze_metadata(combined)


def _record_metadata(
    request_metadata: Mapping[str, FrozenJson],
    result_metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    combined = _metadata_to_dict(request_metadata)
    result_data = _metadata_to_dict(result_metadata)
    if result_data:
        combined[_PROMOTION_RESULT_METADATA_KEY] = result_data
    return _freeze_metadata(combined)


def _identity_metadata(metadata: Mapping[str, FrozenJson]) -> Mapping[str, FrozenJson]:
    identity = {
        key: value
        for key, value in _metadata_to_dict(metadata).items()
        if key != _PROMOTION_RESULT_METADATA_KEY
    }
    return _freeze_metadata(identity)


def _reject_reserved_request_metadata_key(
    metadata: Mapping[str, object],
    name: str,
) -> None:
    if _PROMOTION_RESULT_METADATA_KEY in metadata:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} cannot include reserved {_PROMOTION_RESULT_METADATA_KEY}"
        )


def _require_promotion_reference(
    value: str,
    name: str,
    max_chars: int,
    max_chars_name: str,
) -> None:
    _require_bounded_non_empty(value, name, max_chars, max_chars_name)
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must be a single-line promotion reference"
        )
    stripped = value.strip()
    if any(marker in stripped for marker in _RAW_PROMOTION_MARKERS):
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not contain raw output or diff payloads"
        )
    if stripped.startswith(_COPIED_FILE_PREFIXES):
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not look like copied file contents"
        )


def _promoted_file_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_normalize_relative_path(value, name) for value in values)
    if not items:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not be empty"
        )
    if len(items) > DEFAULT_MAX_PROMOTED_FILES:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} exceeds DEFAULT_MAX_PROMOTED_FILES"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} contains duplicate normalized paths: {joined}"
        )
    return items


def _normalize_relative_path(value: str, name: str) -> str:
    _require_non_empty(value, name)
    if "\x00" in value:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not contain NUL bytes"
        )
    if "\\" in value:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must use POSIX-style separators"
        )
    if value.startswith("/"):
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must be relative"
        )
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not normalize to an empty path"
        )
    if normalized == ".." or normalized.startswith("../"):
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} must not contain parent traversal"
        )
    if len(normalized) > DEFAULT_MAX_PROMOTED_FILE_CHARS:
        raise ImprovementPatchCandidatePromotionValidationError(
            f"{name} exceeds DEFAULT_MAX_PROMOTED_FILE_CHARS"
        )
    return normalized


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
        raise ImprovementPatchCandidatePromotionValidationError(
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
    if len(encoded) > DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS:
        raise ImprovementPatchCandidatePromotionValidationError(
            "metadata exceeds DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS"
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
    "DEFAULT_MAX_PROMOTED_FILES",
    "DEFAULT_MAX_PROMOTED_FILE_CHARS",
    "DEFAULT_MAX_PROMOTION_KIND_CHARS",
    "DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS",
    "DEFAULT_MAX_PROMOTION_REF_CHARS",
    "DEFAULT_MAX_PROMOTION_SUMMARY_CHARS",
    "ImprovementPatchCandidatePromoter",
    "ImprovementPatchCandidatePromotionApproval",
    "ImprovementPatchCandidatePromotionError",
    "ImprovementPatchCandidatePromotionRecord",
    "ImprovementPatchCandidatePromotionRequest",
    "ImprovementPatchCandidatePromotionResult",
    "ImprovementPatchCandidatePromotionService",
    "ImprovementPatchCandidatePromotionStatus",
    "ImprovementPatchCandidatePromotionValidationError",
    "improvement_patch_candidate_promotion_record_from_dict",
    "improvement_patch_candidate_promotion_record_to_dict",
    "improvement_patch_candidate_promotion_request_from_dict",
    "improvement_patch_candidate_promotion_request_to_dict",
    "improvement_patch_candidate_promotion_result_from_dict",
    "improvement_patch_candidate_promotion_result_to_dict",
)
