"""Bounded self-improvement proposal records.

These models are the proposal/control-plane layer for improvement candidates.
They intentionally do not execute tools, mutate files, create worktrees, or
promote candidates.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from raygent_harness.core.model_types import FrozenJson, freeze_json

ImprovementTargetKind = Literal[
    "prompt",
    "context_policy",
    "preset",
    "recipe",
    "tool_affordance",
    "evaluator",
    "documentation",
    "source_code",
    "other",
]
ImprovementEvidenceSource = Literal[
    "transcript",
    "observability",
    "task_output",
    "verification",
    "user_report",
    "cost_usage",
    "other",
]
ImprovementRequiredPermission = Literal[
    "none",
    "human_review",
    "model_provider",
    "filesystem_mutation",
    "shell",
    "worktree",
    "commit",
    "network",
    "external_service",
]
ImprovementRunStatus = Literal[
    "proposed",
    "rejected",
    "accepted_for_evaluation",
    "failed",
    "archived",
]

_TARGET_KINDS: frozenset[str] = frozenset(
    {
        "prompt",
        "context_policy",
        "preset",
        "recipe",
        "tool_affordance",
        "evaluator",
        "documentation",
        "source_code",
        "other",
    }
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
_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "proposed",
        "rejected",
        "accepted_for_evaluation",
        "failed",
        "archived",
    }
)


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementTarget:
    """Artifact or behavior surface that may be improved."""

    target_id: str
    kind: ImprovementTargetKind
    description: str
    owner: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.target_id, "ImprovementTarget.target_id")
        _require_literal(self.kind, _TARGET_KINDS, "ImprovementTarget.kind")
        _require_non_empty(self.description, "ImprovementTarget.description")
        if self.owner is not None:
            _require_non_empty(self.owner, "ImprovementTarget.owner")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementEvidence:
    """One bounded evidence item supporting an improvement proposal."""

    evidence_id: str
    source: ImprovementEvidenceSource
    summary: str
    excerpt: str | None = None
    source_uri: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.evidence_id, "ImprovementEvidence.evidence_id")
        _require_literal(self.source, _EVIDENCE_SOURCES, "ImprovementEvidence.source")
        _require_non_empty(self.summary, "ImprovementEvidence.summary")
        if self.excerpt is not None:
            _require_non_empty(self.excerpt, "ImprovementEvidence.excerpt")
        if self.source_uri is not None:
            _require_non_empty(self.source_uri, "ImprovementEvidence.source_uri")
        if self.created_at < 0:
            raise ValueError("ImprovementEvidence.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementDiagnosis:
    """Symptom-level analysis before a proposed change."""

    summary: str
    symptoms: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    confidence: float | None = None
    unknowns: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.summary, "ImprovementDiagnosis.summary")
        object.__setattr__(self, "symptoms", _non_empty_string_tuple(self.symptoms, "symptoms"))
        object.__setattr__(
            self,
            "hypotheses",
            _non_empty_string_tuple(self.hypotheses, "hypotheses"),
        )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ImprovementDiagnosis.confidence must be between 0 and 1")
        object.__setattr__(self, "unknowns", _non_empty_string_tuple(self.unknowns, "unknowns"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementEvaluationCheck:
    """A proposed verification check. It is data, not command execution."""

    name: str
    instruction: str
    required: bool = True
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "ImprovementEvaluationCheck.name")
        _require_non_empty(self.instruction, "ImprovementEvaluationCheck.instruction")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementEvaluationPlan:
    """Structured plan for evaluating a future candidate."""

    checks: tuple[ImprovementEvaluationCheck, ...] = ()
    non_regression_checks: tuple[ImprovementEvaluationCheck, ...] = ()
    cost_checks: tuple[ImprovementEvaluationCheck, ...] = ()
    manual_review_required: bool = True
    success_criteria: tuple[str, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(
            self,
            "non_regression_checks",
            tuple(self.non_regression_checks),
        )
        object.__setattr__(self, "cost_checks", tuple(self.cost_checks))
        object.__setattr__(
            self,
            "success_criteria",
            _non_empty_string_tuple(self.success_criteria, "success_criteria"),
        )
        if not self.checks and not self.success_criteria:
            raise ValueError(
                "ImprovementEvaluationPlan requires at least one check or success criterion"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    """Structured, reviewable candidate improvement proposal."""

    proposal_id: str
    target: ImprovementTarget
    diagnosis: ImprovementDiagnosis
    hypothesis: str
    proposed_change: str
    intended_behavior_change: str
    expected_benefit: str
    risks: tuple[str, ...]
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    evaluation_plan: ImprovementEvaluationPlan
    rollback_plan: str
    stop_condition: str
    evidence_ids: tuple[str, ...]
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.proposal_id, "ImprovementProposal.proposal_id")
        _require_non_empty(self.hypothesis, "ImprovementProposal.hypothesis")
        _require_non_empty(self.proposed_change, "ImprovementProposal.proposed_change")
        _require_non_empty(
            self.intended_behavior_change,
            "ImprovementProposal.intended_behavior_change",
        )
        _require_non_empty(self.expected_benefit, "ImprovementProposal.expected_benefit")
        object.__setattr__(self, "risks", _required_string_tuple(self.risks, "risks"))
        object.__setattr__(
            self,
            "required_permissions",
            _required_permission_tuple(self.required_permissions),
        )
        _require_non_empty(self.rollback_plan, "ImprovementProposal.rollback_plan")
        _require_non_empty(self.stop_condition, "ImprovementProposal.stop_condition")
        object.__setattr__(
            self,
            "evidence_ids",
            _required_string_tuple(self.evidence_ids, "evidence_ids"),
        )
        if self.created_at < 0:
            raise ValueError("ImprovementProposal.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementRun:
    """Record of one bounded improvement proposal cycle."""

    run_id: str
    status: ImprovementRunStatus
    target: ImprovementTarget
    evidence: tuple[ImprovementEvidence, ...]
    proposal: ImprovementProposal
    warnings: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "ImprovementRun.run_id")
        _require_literal(self.status, _RUN_STATUSES, "ImprovementRun.status")
        if not self.evidence:
            raise ValueError("ImprovementRun.evidence must not be empty")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "warnings", _non_empty_string_tuple(self.warnings, "warnings"))
        if self.proposal.target != self.target:
            raise ValueError("ImprovementRun.proposal.target must match target")
        evidence_ids = {item.evidence_id for item in self.evidence}
        unknown = tuple(
            evidence_id
            for evidence_id in self.proposal.evidence_ids
            if evidence_id not in evidence_ids
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                "ImprovementRun.proposal.evidence_ids reference unknown evidence: "
                f"{joined}"
            )
        if self.created_at < 0:
            raise ValueError("ImprovementRun.created_at must be >= 0")
        if self.updated_at < 0:
            raise ValueError("ImprovementRun.updated_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def improvement_target_to_dict(target: ImprovementTarget) -> dict[str, object]:
    return {
        "target_id": target.target_id,
        "kind": target.kind,
        "description": target.description,
        "owner": target.owner,
        "metadata": _metadata_to_dict(target.metadata),
    }


def improvement_target_from_dict(data: Mapping[str, object]) -> ImprovementTarget:
    return ImprovementTarget(
        target_id=str(data["target_id"]),
        kind=cast(
            ImprovementTargetKind,
            _literal_from_object(data["kind"], _TARGET_KINDS, "kind"),
        ),
        description=str(data["description"]),
        owner=_optional_str(data.get("owner")),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_evidence_to_dict(evidence: ImprovementEvidence) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "source": evidence.source,
        "summary": evidence.summary,
        "excerpt": evidence.excerpt,
        "source_uri": evidence.source_uri,
        "created_at": evidence.created_at,
        "metadata": _metadata_to_dict(evidence.metadata),
    }


def improvement_evidence_from_dict(data: Mapping[str, object]) -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id=str(data["evidence_id"]),
        source=cast(
            ImprovementEvidenceSource,
            _literal_from_object(data["source"], _EVIDENCE_SOURCES, "source"),
        ),
        summary=str(data["summary"]),
        excerpt=_optional_str(data.get("excerpt")),
        source_uri=_optional_str(data.get("source_uri")),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_diagnosis_to_dict(diagnosis: ImprovementDiagnosis) -> dict[str, object]:
    return {
        "summary": diagnosis.summary,
        "symptoms": list(diagnosis.symptoms),
        "hypotheses": list(diagnosis.hypotheses),
        "confidence": diagnosis.confidence,
        "unknowns": list(diagnosis.unknowns),
        "metadata": _metadata_to_dict(diagnosis.metadata),
    }


def improvement_diagnosis_from_dict(data: Mapping[str, object]) -> ImprovementDiagnosis:
    return ImprovementDiagnosis(
        summary=str(data["summary"]),
        symptoms=_string_tuple(data.get("symptoms", ()), "symptoms"),
        hypotheses=_string_tuple(data.get("hypotheses", ()), "hypotheses"),
        confidence=_optional_float(data.get("confidence")),
        unknowns=_string_tuple(data.get("unknowns", ()), "unknowns"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_evaluation_check_to_dict(
    check: ImprovementEvaluationCheck,
) -> dict[str, object]:
    return {
        "name": check.name,
        "instruction": check.instruction,
        "required": check.required,
        "metadata": _metadata_to_dict(check.metadata),
    }


def improvement_evaluation_check_from_dict(
    data: Mapping[str, object],
) -> ImprovementEvaluationCheck:
    return ImprovementEvaluationCheck(
        name=str(data["name"]),
        instruction=str(data["instruction"]),
        required=bool(data.get("required", True)),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_evaluation_plan_to_dict(
    plan: ImprovementEvaluationPlan,
) -> dict[str, object]:
    return {
        "checks": [improvement_evaluation_check_to_dict(check) for check in plan.checks],
        "non_regression_checks": [
            improvement_evaluation_check_to_dict(check)
            for check in plan.non_regression_checks
        ],
        "cost_checks": [
            improvement_evaluation_check_to_dict(check) for check in plan.cost_checks
        ],
        "manual_review_required": plan.manual_review_required,
        "success_criteria": list(plan.success_criteria),
        "metadata": _metadata_to_dict(plan.metadata),
    }


def improvement_evaluation_plan_from_dict(
    data: Mapping[str, object],
) -> ImprovementEvaluationPlan:
    return ImprovementEvaluationPlan(
        checks=tuple(
            improvement_evaluation_check_from_dict(item)
            for item in _mapping_sequence(data.get("checks", ()), "checks")
        ),
        non_regression_checks=tuple(
            improvement_evaluation_check_from_dict(item)
            for item in _mapping_sequence(
                data.get("non_regression_checks", ()),
                "non_regression_checks",
            )
        ),
        cost_checks=tuple(
            improvement_evaluation_check_from_dict(item)
            for item in _mapping_sequence(data.get("cost_checks", ()), "cost_checks")
        ),
        manual_review_required=bool(data.get("manual_review_required", True)),
        success_criteria=_string_tuple(data.get("success_criteria", ()), "success_criteria"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_proposal_to_dict(proposal: ImprovementProposal) -> dict[str, object]:
    return {
        "proposal_id": proposal.proposal_id,
        "target": improvement_target_to_dict(proposal.target),
        "diagnosis": improvement_diagnosis_to_dict(proposal.diagnosis),
        "hypothesis": proposal.hypothesis,
        "proposed_change": proposal.proposed_change,
        "intended_behavior_change": proposal.intended_behavior_change,
        "expected_benefit": proposal.expected_benefit,
        "risks": list(proposal.risks),
        "required_permissions": list(proposal.required_permissions),
        "evaluation_plan": improvement_evaluation_plan_to_dict(proposal.evaluation_plan),
        "rollback_plan": proposal.rollback_plan,
        "stop_condition": proposal.stop_condition,
        "evidence_ids": list(proposal.evidence_ids),
        "created_at": proposal.created_at,
        "metadata": _metadata_to_dict(proposal.metadata),
    }


def improvement_proposal_from_dict(data: Mapping[str, object]) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=str(data["proposal_id"]),
        target=improvement_target_from_dict(_expect_mapping(data["target"], "target")),
        diagnosis=improvement_diagnosis_from_dict(
            _expect_mapping(data["diagnosis"], "diagnosis")
        ),
        hypothesis=str(data["hypothesis"]),
        proposed_change=str(data["proposed_change"]),
        intended_behavior_change=str(data["intended_behavior_change"]),
        expected_benefit=str(data["expected_benefit"]),
        risks=_string_tuple(data.get("risks", ()), "risks"),
        required_permissions=_required_permission_tuple(
            _string_tuple(data.get("required_permissions", ()), "required_permissions")
        ),
        evaluation_plan=improvement_evaluation_plan_from_dict(
            _expect_mapping(data["evaluation_plan"], "evaluation_plan")
        ),
        rollback_plan=str(data["rollback_plan"]),
        stop_condition=str(data["stop_condition"]),
        evidence_ids=_string_tuple(data.get("evidence_ids", ()), "evidence_ids"),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_run_to_dict(run: ImprovementRun) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "status": run.status,
        "target": improvement_target_to_dict(run.target),
        "evidence": [improvement_evidence_to_dict(item) for item in run.evidence],
        "proposal": improvement_proposal_to_dict(run.proposal),
        "warnings": list(run.warnings),
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "metadata": _metadata_to_dict(run.metadata),
    }


def improvement_run_from_dict(data: Mapping[str, object]) -> ImprovementRun:
    return ImprovementRun(
        run_id=str(data["run_id"]),
        status=cast(
            ImprovementRunStatus,
            _literal_from_object(data["status"], _RUN_STATUSES, "status"),
        ),
        target=improvement_target_from_dict(_expect_mapping(data["target"], "target")),
        evidence=tuple(
            improvement_evidence_from_dict(item)
            for item in _mapping_sequence(data.get("evidence", ()), "evidence")
        ),
        proposal=improvement_proposal_from_dict(
            _expect_mapping(data["proposal"], "proposal")
        ),
        warnings=_string_tuple(data.get("warnings", ()), "warnings"),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        updated_at=_float_from_object(data.get("updated_at", 0.0), "updated_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
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


def _required_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = _non_empty_string_tuple(values, name)
    if not items:
        raise ValueError(f"ImprovementProposal.{name} must not be empty")
    return items


def _non_empty_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(values)
    for item in items:
        _require_non_empty(item, name)
    return items


def _required_permission_tuple(
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
        raise ValueError("ImprovementProposal.required_permissions must not be empty")
    if "none" in items and len(items) > 1:
        raise ValueError("ImprovementProposal.required_permissions cannot mix none")
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, FrozenJson]:
    frozen = freeze_json(metadata)
    if not isinstance(frozen, Mapping):
        raise TypeError("metadata must serialize to a JSON object")
    return cast(Mapping[str, FrozenJson], frozen)


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


def _mapping_sequence(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    sequence = cast(Sequence[object], value)
    return tuple(_expect_mapping(item, name) for item in sequence)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    sequence = cast(Sequence[object], value)
    return tuple(str(item) for item in sequence)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a number") from exc


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _float_from_object(value, "float")


__all__ = (
    "ImprovementDiagnosis",
    "ImprovementEvaluationCheck",
    "ImprovementEvaluationPlan",
    "ImprovementEvidence",
    "ImprovementEvidenceSource",
    "ImprovementProposal",
    "ImprovementRequiredPermission",
    "ImprovementRun",
    "ImprovementRunStatus",
    "ImprovementTarget",
    "ImprovementTargetKind",
    "improvement_diagnosis_from_dict",
    "improvement_diagnosis_to_dict",
    "improvement_evaluation_check_from_dict",
    "improvement_evaluation_check_to_dict",
    "improvement_evaluation_plan_from_dict",
    "improvement_evaluation_plan_to_dict",
    "improvement_evidence_from_dict",
    "improvement_evidence_to_dict",
    "improvement_proposal_from_dict",
    "improvement_proposal_to_dict",
    "improvement_run_from_dict",
    "improvement_run_to_dict",
    "improvement_target_from_dict",
    "improvement_target_to_dict",
)
