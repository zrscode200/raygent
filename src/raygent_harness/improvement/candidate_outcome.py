"""Outcome and archive-decision records for evaluated patch candidates.

The RSI-004A layer derives data-only outcome and archive recommendations from
RSI-003C materialization/evaluation records. It does not write archives,
promote branches, execute commands, commit, or integrate product goals.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.candidate_materialization import (
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationDecision,
    ImprovementPatchCandidateMaterialization,
)
from raygent_harness.improvement.models import ImprovementRequiredPermission

DEFAULT_MAX_OUTCOME_SUMMARY_CHARS = 4_000
DEFAULT_MAX_OUTCOME_BLOCKERS = 20
DEFAULT_MAX_OUTCOME_BLOCKER_CHARS = 1_000
DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS = 20
DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS = 1_000
DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES = 30
DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS = 1_000
DEFAULT_MAX_OUTCOME_METADATA_CHARS = 20_000

ImprovementPatchCandidateOutcomeDecision = Literal[
    "promotable",
    "reject",
    "needs_review",
]
ImprovementPatchCandidateOutcomeReason = Literal[
    "evaluation_passed",
    "evaluation_failed",
    "evaluation_warned",
    "evaluation_needs_review",
    "manual_rejection",
    "manual_review_required",
]

_OUTCOME_DECISIONS: frozenset[str] = frozenset(
    {"promotable", "reject", "needs_review"}
)
_OUTCOME_REASONS: frozenset[str] = frozenset(
    {
        "evaluation_passed",
        "evaluation_failed",
        "evaluation_warned",
        "evaluation_needs_review",
        "manual_rejection",
        "manual_review_required",
    }
)
_EVALUATION_DECISIONS: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "needs_review"}
)
_REQUIRED_PROMOTION_PERMISSIONS: frozenset[str] = frozenset(
    {"human_review", "filesystem_mutation", "commit"}
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
_DEFAULT_DECISION_BY_EVALUATION: Mapping[
    ImprovementPatchCandidateEvaluationDecision,
    ImprovementPatchCandidateOutcomeDecision,
] = MappingProxyType(
    {
        "pass": "promotable",
        "warn": "needs_review",
        "fail": "reject",
        "needs_review": "needs_review",
    }
)
_DEFAULT_REASON_BY_EVALUATION: Mapping[
    ImprovementPatchCandidateEvaluationDecision,
    ImprovementPatchCandidateOutcomeReason,
] = MappingProxyType(
    {
        "pass": "evaluation_passed",
        "warn": "evaluation_warned",
        "fail": "evaluation_failed",
        "needs_review": "evaluation_needs_review",
    }
)
_DEFAULT_ARCHIVE_RECOMMENDATION_BY_OUTCOME: Mapping[
    ImprovementPatchCandidateOutcomeDecision,
    bool,
] = MappingProxyType(
    {
        "promotable": False,
        "reject": True,
        "needs_review": True,
    }
)
_RAW_ARTIFACT_REFERENCE_MARKERS: tuple[str, ...] = (
    "diff --git ",
    "Traceback (most recent call last)",
    "-----BEGIN ",
    "@@ ",
)


class ImprovementPatchCandidateOutcomeError(ValueError):
    """Raised when candidate outcome data cannot be produced."""


class ImprovementPatchCandidateOutcomeValidationError(
    ImprovementPatchCandidateOutcomeError
):
    """Raised when outcome/archive-decision data violates the RSI-004A contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateOutcome:
    """Data-only outcome for one evaluated, materialized patch candidate."""

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
    decision: ImprovementPatchCandidateOutcomeDecision
    reason: ImprovementPatchCandidateOutcomeReason
    summary: str
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    archive_recommended: bool
    promotion_blockers: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.outcome_id, "ImprovementPatchCandidateOutcome.outcome_id")
        _require_non_empty(
            self.materialization_id,
            "ImprovementPatchCandidateOutcome.materialization_id",
        )
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateOutcome.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateOutcome.candidate_id",
        )
        _require_non_empty(self.run_id, "ImprovementPatchCandidateOutcome.run_id")
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateOutcome.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateOutcome.gate_evaluation_id",
        )
        _require_non_empty(
            self.base_revision,
            "ImprovementPatchCandidateOutcome.base_revision",
        )
        _require_non_empty(
            self.patch_digest,
            "ImprovementPatchCandidateOutcome.patch_digest",
        )
        _require_non_empty(
            self.evaluation_id,
            "ImprovementPatchCandidateOutcome.evaluation_id",
        )
        _require_literal(
            self.evaluation_decision,
            _EVALUATION_DECISIONS,
            "ImprovementPatchCandidateOutcome.evaluation_decision",
        )
        _require_literal(
            self.decision,
            _OUTCOME_DECISIONS,
            "ImprovementPatchCandidateOutcome.decision",
        )
        _require_literal(
            self.reason,
            _OUTCOME_REASONS,
            "ImprovementPatchCandidateOutcome.reason",
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidateOutcome.summary",
            DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
            "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
        )
        blockers = _bounded_unique_string_tuple(
            self.promotion_blockers,
            "ImprovementPatchCandidateOutcome.promotion_blockers",
            DEFAULT_MAX_OUTCOME_BLOCKERS,
            DEFAULT_MAX_OUTCOME_BLOCKER_CHARS,
            "DEFAULT_MAX_OUTCOME_BLOCKERS",
            "DEFAULT_MAX_OUTCOME_BLOCKER_CHARS",
        )
        object.__setattr__(self, "promotion_blockers", blockers)
        object.__setattr__(
            self,
            "required_permissions",
            _permission_tuple(self.required_permissions),
        )
        _validate_outcome_policy(
            evaluation_decision=self.evaluation_decision,
            decision=self.decision,
            reason=self.reason,
            required_permissions=self.required_permissions,
            archive_recommended=self.archive_recommended,
            promotion_blockers=blockers,
        )
        if self.created_at < 0:
            raise ValueError("ImprovementPatchCandidateOutcome.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateOutcomePolicy:
    """Derive one data-only outcome from materialization and supplied evaluation."""

    clock: Callable[[], float] = time.time
    outcome_id_factory: Callable[[], str] | None = None

    def decide(
        self,
        materialization: ImprovementPatchCandidateMaterialization,
        evaluation: ImprovementPatchCandidateEvaluation,
        *,
        outcome_id: str | None = None,
        decision: ImprovementPatchCandidateOutcomeDecision | None = None,
        reason: ImprovementPatchCandidateOutcomeReason | None = None,
        summary: str | None = None,
        promotion_blockers: Sequence[str] | None = None,
        required_permissions: Sequence[str] | None = None,
        archive_recommended: bool | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateOutcome:
        """Return a bounded outcome record and stop before promotion/archive writes."""

        _validate_materialization_evaluation_linkage(materialization, evaluation)
        outcome_decision = decision or _DEFAULT_DECISION_BY_EVALUATION[evaluation.decision]
        outcome_reason = reason or _default_reason_for_decision(
            evaluation.decision,
            outcome_decision,
        )
        blockers = tuple(
            promotion_blockers
            if promotion_blockers is not None
            else _default_promotion_blockers(evaluation.decision, evaluation.warnings)
        )
        permissions = tuple(
            required_permissions
            if required_permissions is not None
            else _default_required_permissions(outcome_decision)
        )
        archive_flag = (
            archive_recommended
            if archive_recommended is not None
            else _DEFAULT_ARCHIVE_RECOMMENDATION_BY_OUTCOME[outcome_decision]
        )
        return ImprovementPatchCandidateOutcome(
            outcome_id=outcome_id or self._new_outcome_id(),
            materialization_id=materialization.materialization_id,
            allocation_id=materialization.allocation_id,
            candidate_id=materialization.candidate_id,
            run_id=materialization.run_id,
            proposal_id=materialization.proposal_id,
            gate_evaluation_id=materialization.gate_evaluation_id,
            base_revision=materialization.base_revision,
            patch_digest=materialization.patch_digest,
            evaluation_id=evaluation.evaluation_id,
            evaluation_decision=evaluation.decision,
            decision=outcome_decision,
            reason=outcome_reason,
            summary=summary or _default_summary(evaluation.decision, outcome_decision),
            required_permissions=cast(tuple[ImprovementRequiredPermission, ...], permissions),
            archive_recommended=archive_flag,
            promotion_blockers=blockers,
            created_at=self.clock(),
            metadata=metadata or {},
        )

    def _new_outcome_id(self) -> str:
        outcome_id = (
            self.outcome_id_factory()
            if self.outcome_id_factory is not None
            else f"ipco_{uuid4().hex}"
        )
        if not outcome_id.strip():
            raise ImprovementPatchCandidateOutcomeValidationError(
                "outcome_id_factory returned an empty id"
            )
        return outcome_id


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveDecision:
    """Data-only recommendation for later archive handling of one outcome."""

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
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.archive_decision_id,
            "ImprovementPatchCandidateArchiveDecision.archive_decision_id",
        )
        _require_non_empty(
            self.outcome_id,
            "ImprovementPatchCandidateArchiveDecision.outcome_id",
        )
        _require_non_empty(
            self.materialization_id,
            "ImprovementPatchCandidateArchiveDecision.materialization_id",
        )
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateArchiveDecision.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateArchiveDecision.candidate_id",
        )
        _require_non_empty(
            self.run_id,
            "ImprovementPatchCandidateArchiveDecision.run_id",
        )
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateArchiveDecision.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateArchiveDecision.gate_evaluation_id",
        )
        _require_non_empty(
            self.base_revision,
            "ImprovementPatchCandidateArchiveDecision.base_revision",
        )
        _require_non_empty(
            self.patch_digest,
            "ImprovementPatchCandidateArchiveDecision.patch_digest",
        )
        _require_non_empty(
            self.evaluation_id,
            "ImprovementPatchCandidateArchiveDecision.evaluation_id",
        )
        _require_literal(
            self.outcome_decision,
            _OUTCOME_DECISIONS,
            "ImprovementPatchCandidateArchiveDecision.outcome_decision",
        )
        _validate_archive_recommendation(
            self.outcome_decision,
            self.archive_recommended,
        )
        _require_bounded_non_empty(
            self.archive_reason,
            "ImprovementPatchCandidateArchiveDecision.archive_reason",
            DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
            "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidateArchiveDecision.summary",
            DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
            "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
        )
        object.__setattr__(
            self,
            "failure_symptoms",
            _bounded_unique_string_tuple(
                self.failure_symptoms,
                "ImprovementPatchCandidateArchiveDecision.failure_symptoms",
                DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS,
                DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS,
                "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS",
                "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS",
            ),
        )
        object.__setattr__(
            self,
            "artifact_references",
            _artifact_reference_tuple(self.artifact_references),
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateArchiveDecision.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateArchiveDecisionPolicy:
    """Derive one bounded archive recommendation from an outcome."""

    clock: Callable[[], float] = time.time
    archive_decision_id_factory: Callable[[], str] | None = None

    def decide(
        self,
        outcome: ImprovementPatchCandidateOutcome,
        *,
        archive_decision_id: str | None = None,
        archive_reason: str | None = None,
        summary: str | None = None,
        failure_symptoms: Sequence[str] | None = None,
        artifact_references: Sequence[str] | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateArchiveDecision:
        """Return an archive recommendation record and stop before persistence."""

        archive_recommended = _DEFAULT_ARCHIVE_RECOMMENDATION_BY_OUTCOME[
            outcome.decision
        ]
        symptoms = tuple(
            failure_symptoms
            if failure_symptoms is not None
            else _default_failure_symptoms(outcome)
        )
        return ImprovementPatchCandidateArchiveDecision(
            archive_decision_id=(
                archive_decision_id or self._new_archive_decision_id()
            ),
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
            outcome_decision=outcome.decision,
            archive_recommended=archive_recommended,
            archive_reason=archive_reason or _default_archive_reason(outcome),
            summary=summary or outcome.summary,
            failure_symptoms=symptoms,
            artifact_references=tuple(artifact_references or ()),
            created_at=self.clock(),
            metadata=metadata or {},
        )

    def _new_archive_decision_id(self) -> str:
        archive_decision_id = (
            self.archive_decision_id_factory()
            if self.archive_decision_id_factory is not None
            else f"ipcad_{uuid4().hex}"
        )
        if not archive_decision_id.strip():
            raise ImprovementPatchCandidateOutcomeValidationError(
                "archive_decision_id_factory returned an empty id"
            )
        return archive_decision_id


def improvement_patch_candidate_outcome_to_dict(
    outcome: ImprovementPatchCandidateOutcome,
) -> dict[str, object]:
    return {
        "outcome_id": outcome.outcome_id,
        "materialization_id": outcome.materialization_id,
        "allocation_id": outcome.allocation_id,
        "candidate_id": outcome.candidate_id,
        "run_id": outcome.run_id,
        "proposal_id": outcome.proposal_id,
        "gate_evaluation_id": outcome.gate_evaluation_id,
        "base_revision": outcome.base_revision,
        "patch_digest": outcome.patch_digest,
        "evaluation_id": outcome.evaluation_id,
        "evaluation_decision": outcome.evaluation_decision,
        "decision": outcome.decision,
        "reason": outcome.reason,
        "summary": outcome.summary,
        "required_permissions": list(outcome.required_permissions),
        "archive_recommended": outcome.archive_recommended,
        "promotion_blockers": list(outcome.promotion_blockers),
        "created_at": outcome.created_at,
        "metadata": _metadata_to_dict(outcome.metadata),
    }


def improvement_patch_candidate_outcome_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateOutcome:
    return ImprovementPatchCandidateOutcome(
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
        decision=cast(
            ImprovementPatchCandidateOutcomeDecision,
            _literal_from_object(data["decision"], _OUTCOME_DECISIONS, "decision"),
        ),
        reason=cast(
            ImprovementPatchCandidateOutcomeReason,
            _literal_from_object(data["reason"], _OUTCOME_REASONS, "reason"),
        ),
        summary=str(data["summary"]),
        required_permissions=_permission_tuple(
            _string_tuple(data.get("required_permissions", ()), "required_permissions")
        ),
        archive_recommended=bool(data["archive_recommended"]),
        promotion_blockers=_string_tuple(
            data.get("promotion_blockers", ()),
            "promotion_blockers",
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_archive_decision_to_dict(
    decision: ImprovementPatchCandidateArchiveDecision,
) -> dict[str, object]:
    return {
        "archive_decision_id": decision.archive_decision_id,
        "outcome_id": decision.outcome_id,
        "materialization_id": decision.materialization_id,
        "allocation_id": decision.allocation_id,
        "candidate_id": decision.candidate_id,
        "run_id": decision.run_id,
        "proposal_id": decision.proposal_id,
        "gate_evaluation_id": decision.gate_evaluation_id,
        "base_revision": decision.base_revision,
        "patch_digest": decision.patch_digest,
        "evaluation_id": decision.evaluation_id,
        "outcome_decision": decision.outcome_decision,
        "archive_recommended": decision.archive_recommended,
        "archive_reason": decision.archive_reason,
        "summary": decision.summary,
        "failure_symptoms": list(decision.failure_symptoms),
        "artifact_references": list(decision.artifact_references),
        "created_at": decision.created_at,
        "metadata": _metadata_to_dict(decision.metadata),
    }


def improvement_patch_candidate_archive_decision_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateArchiveDecision:
    return ImprovementPatchCandidateArchiveDecision(
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
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_materialization_evaluation_linkage(
    materialization: ImprovementPatchCandidateMaterialization,
    evaluation: ImprovementPatchCandidateEvaluation,
) -> None:
    fields = (
        "materialization_id",
        "allocation_id",
        "candidate_id",
        "run_id",
        "proposal_id",
        "gate_evaluation_id",
    )
    for field_name in fields:
        if getattr(materialization, field_name) != getattr(evaluation, field_name):
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcomePolicy linkage mismatch: "
                f"{field_name}"
            )


def _validate_outcome_policy(
    *,
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision,
    decision: ImprovementPatchCandidateOutcomeDecision,
    reason: ImprovementPatchCandidateOutcomeReason,
    required_permissions: Sequence[str],
    archive_recommended: bool,
    promotion_blockers: Sequence[str],
) -> None:
    if decision == "promotable" and evaluation_decision != "pass":
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.promotable requires pass evaluation"
        )
    if evaluation_decision != "pass":
        expected = _DEFAULT_DECISION_BY_EVALUATION[evaluation_decision]
        if decision != expected:
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcome.decision does not match "
                "evaluation decision"
            )
    if decision == "promotable":
        if evaluation_decision != "pass":
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcome.promotable requires pass evaluation"
            )
        if reason != "evaluation_passed":
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcome.promotable requires "
                "evaluation_passed reason"
            )
        missing = _REQUIRED_PROMOTION_PERMISSIONS.difference(required_permissions)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcome.required_permissions missing: "
                f"{joined}"
            )
        if promotion_blockers:
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateOutcome.promotion_blockers must be "
                "empty for promotable outcomes"
            )
    elif not promotion_blockers:
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.promotion_blockers must not be "
            "empty for non-promotable outcomes"
        )
    _validate_archive_recommendation(decision, archive_recommended)
    if reason == "evaluation_failed" and decision != "reject":
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.evaluation_failed requires reject"
        )
    if reason in {"evaluation_warned", "evaluation_needs_review"} and decision != (
        "needs_review"
    ):
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.review reasons require needs_review"
        )
    if reason == "manual_rejection" and decision != "reject":
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.manual_rejection requires reject"
        )
    if reason == "manual_review_required" and decision != "needs_review":
        raise ImprovementPatchCandidateOutcomeValidationError(
            "ImprovementPatchCandidateOutcome.manual_review_required requires "
            "needs_review"
        )


def _validate_archive_recommendation(
    outcome_decision: ImprovementPatchCandidateOutcomeDecision,
    archive_recommended: bool,
) -> None:
    expected = _DEFAULT_ARCHIVE_RECOMMENDATION_BY_OUTCOME[outcome_decision]
    if archive_recommended != expected:
        raise ImprovementPatchCandidateOutcomeValidationError(
            "archive_recommended does not match outcome decision"
        )


def _default_reason_for_decision(
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision,
    outcome_decision: ImprovementPatchCandidateOutcomeDecision,
) -> ImprovementPatchCandidateOutcomeReason:
    if evaluation_decision == "pass" and outcome_decision == "reject":
        return "manual_rejection"
    if evaluation_decision == "pass" and outcome_decision == "needs_review":
        return "manual_review_required"
    return _DEFAULT_REASON_BY_EVALUATION[evaluation_decision]


def _default_promotion_blockers(
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision,
    warnings: Sequence[str],
) -> tuple[str, ...]:
    if evaluation_decision == "pass":
        return ()
    if warnings:
        return tuple(warnings)
    if evaluation_decision == "warn":
        return ("evaluation produced warnings",)
    if evaluation_decision == "fail":
        return ("evaluation failed",)
    return ("evaluation requires review",)


def _default_required_permissions(
    decision: ImprovementPatchCandidateOutcomeDecision,
) -> tuple[ImprovementRequiredPermission, ...]:
    if decision == "promotable":
        return ("human_review", "filesystem_mutation", "commit")
    return ("human_review",)


def _default_summary(
    evaluation_decision: ImprovementPatchCandidateEvaluationDecision,
    outcome_decision: ImprovementPatchCandidateOutcomeDecision,
) -> str:
    if outcome_decision == "promotable":
        return "Evaluation passed; candidate is eligible for later explicit promotion."
    if outcome_decision == "reject":
        return "Evaluation failed; candidate should be rejected and may be archived."
    if evaluation_decision == "warn":
        return "Evaluation warned; candidate needs review before any later action."
    return "Evaluation needs review; candidate needs review before any later action."


def _default_archive_reason(outcome: ImprovementPatchCandidateOutcome) -> str:
    if outcome.decision == "promotable":
        return "Promotable candidates are not archived by default."
    return "Rejected or review-needed candidates are archive candidates by default."


def _default_failure_symptoms(
    outcome: ImprovementPatchCandidateOutcome,
) -> tuple[str, ...]:
    if outcome.decision == "promotable":
        return ()
    return outcome.promotion_blockers


def _permission_tuple(
    permissions: Sequence[str],
) -> tuple[ImprovementRequiredPermission, ...]:
    items = tuple(
        cast(
            ImprovementRequiredPermission,
            _literal_from_object(item, _REQUIRED_PERMISSIONS, "required_permissions"),
        )
        for item in permissions
    )
    if not items:
        raise ImprovementPatchCandidateOutcomeValidationError(
            "required_permissions must not be empty"
        )
    if "none" in items and len(items) > 1:
        raise ImprovementPatchCandidateOutcomeValidationError(
            "required_permissions cannot combine none with other permissions"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateOutcomeValidationError(
            f"required_permissions contains duplicates: {joined}"
        )
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _artifact_reference_tuple(values: Sequence[str]) -> tuple[str, ...]:
    items = _bounded_unique_string_tuple(
        values,
        "ImprovementPatchCandidateArchiveDecision.artifact_references",
        DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES,
        DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS,
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES",
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS",
    )
    for item in items:
        if "\n" in item or "\r" in item or "\x00" in item:
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateArchiveDecision.artifact_references "
                "must be single-line references"
            )
        if any(marker in item for marker in _RAW_ARTIFACT_REFERENCE_MARKERS):
            raise ImprovementPatchCandidateOutcomeValidationError(
                "ImprovementPatchCandidateArchiveDecision.artifact_references "
                "must not contain raw output or diff payloads"
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
        raise ImprovementPatchCandidateOutcomeValidationError(
            f"{name} exceeds {max_items_name}"
        )
    for item in items:
        _require_bounded_non_empty(item, name, max_chars, max_chars_name)
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateOutcomeValidationError(
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
        raise ImprovementPatchCandidateOutcomeValidationError(
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
    if len(encoded) > DEFAULT_MAX_OUTCOME_METADATA_CHARS:
        raise ImprovementPatchCandidateOutcomeValidationError(
            "metadata exceeds DEFAULT_MAX_OUTCOME_METADATA_CHARS"
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
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


__all__ = (
    "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES",
    "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS",
    "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS",
    "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS",
    "DEFAULT_MAX_OUTCOME_BLOCKERS",
    "DEFAULT_MAX_OUTCOME_BLOCKER_CHARS",
    "DEFAULT_MAX_OUTCOME_METADATA_CHARS",
    "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
    "ImprovementPatchCandidateArchiveDecision",
    "ImprovementPatchCandidateArchiveDecisionPolicy",
    "ImprovementPatchCandidateOutcome",
    "ImprovementPatchCandidateOutcomeDecision",
    "ImprovementPatchCandidateOutcomeError",
    "ImprovementPatchCandidateOutcomePolicy",
    "ImprovementPatchCandidateOutcomeReason",
    "ImprovementPatchCandidateOutcomeValidationError",
    "improvement_patch_candidate_archive_decision_from_dict",
    "improvement_patch_candidate_archive_decision_to_dict",
    "improvement_patch_candidate_outcome_from_dict",
    "improvement_patch_candidate_outcome_to_dict",
)
