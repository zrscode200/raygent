"""Data-only patch candidate records for bounded improvement."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.gates import ImprovementGateEvaluation
from raygent_harness.improvement.models import (
    ImprovementEvaluationPlan,
    ImprovementRequiredPermission,
    ImprovementRun,
    ImprovementTarget,
    improvement_evaluation_plan_from_dict,
    improvement_evaluation_plan_to_dict,
    improvement_target_from_dict,
    improvement_target_to_dict,
)

ImprovementPatchCandidateStatus = Literal["planned"]

_CANDIDATE_STATUSES: frozenset[str] = frozenset({"planned"})
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
_SOURCE_CODE_REQUIRED_PERMISSIONS: frozenset[str] = frozenset(
    {"filesystem_mutation", "worktree"}
)


class ImprovementPatchCandidateError(ValueError):
    """Raised when a patch candidate plan cannot be produced."""


class ImprovementPatchCandidateValidationError(ImprovementPatchCandidateError):
    """Raised when patch candidate data violates the RSI-003A contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePlan:
    """Reviewable plan for a future isolated patch candidate."""

    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    target: ImprovementTarget
    base_revision: str
    summary: str
    planned_changes: tuple[str, ...]
    expected_files: tuple[str, ...]
    required_permissions: tuple[ImprovementRequiredPermission, ...]
    evaluation_plan: ImprovementEvaluationPlan
    rollback_plan: str
    status: ImprovementPatchCandidateStatus = "planned"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, "ImprovementPatchCandidatePlan.candidate_id")
        _require_non_empty(self.run_id, "ImprovementPatchCandidatePlan.run_id")
        _require_non_empty(self.proposal_id, "ImprovementPatchCandidatePlan.proposal_id")
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidatePlan.gate_evaluation_id",
        )
        _require_non_empty(self.base_revision, "ImprovementPatchCandidatePlan.base_revision")
        _require_non_empty(self.summary, "ImprovementPatchCandidatePlan.summary")
        object.__setattr__(
            self,
            "planned_changes",
            _required_string_tuple(self.planned_changes, "planned_changes"),
        )
        object.__setattr__(
            self,
            "expected_files",
            _required_string_tuple(self.expected_files, "expected_files"),
        )
        object.__setattr__(
            self,
            "required_permissions",
            _candidate_permission_tuple(self.required_permissions, self.target),
        )
        _require_non_empty(self.rollback_plan, "ImprovementPatchCandidatePlan.rollback_plan")
        _require_literal(
            self.status,
            _CANDIDATE_STATUSES,
            "ImprovementPatchCandidatePlan.status",
        )
        if self.created_at < 0:
            raise ValueError("ImprovementPatchCandidatePlan.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidatePlanner:
    """Pure planner for data-only patch candidate records."""

    clock: Callable[[], float] = time.time
    candidate_id_factory: Callable[[], str] | None = None

    def plan(
        self,
        run: ImprovementRun,
        gate_evaluation: ImprovementGateEvaluation,
        *,
        base_revision: str,
        summary: str,
        planned_changes: Sequence[str],
        expected_files: Sequence[str],
        required_permissions: Sequence[str] | None = None,
        rollback_plan: str | None = None,
        candidate_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidatePlan:
        """Create one planned candidate record from a passed proposal gate."""

        _validate_gate_linkage(run, gate_evaluation)
        if gate_evaluation.decision != "pass":
            raise ImprovementPatchCandidateValidationError(
                "ImprovementPatchCandidatePlan requires a passing gate evaluation"
            )
        candidate_permissions = _candidate_permission_tuple(
            tuple(
                required_permissions
                if required_permissions is not None
                else run.proposal.required_permissions
            ),
            run.target,
        )
        return ImprovementPatchCandidatePlan(
            candidate_id=candidate_id or self._new_candidate_id(),
            run_id=run.run_id,
            proposal_id=run.proposal.proposal_id,
            gate_evaluation_id=gate_evaluation.evaluation_id,
            target=run.target,
            base_revision=base_revision,
            summary=summary,
            planned_changes=tuple(planned_changes),
            expected_files=tuple(expected_files),
            required_permissions=candidate_permissions,
            evaluation_plan=run.proposal.evaluation_plan,
            rollback_plan=(
                run.proposal.rollback_plan if rollback_plan is None else rollback_plan
            ),
            status="planned",
            created_at=self.clock(),
            metadata=metadata or {},
        )

    def _new_candidate_id(self) -> str:
        candidate_id = (
            self.candidate_id_factory()
            if self.candidate_id_factory is not None
            else f"ipc_{uuid4().hex}"
        )
        if not candidate_id.strip():
            raise ImprovementPatchCandidateValidationError(
                "candidate_id_factory returned an empty id"
            )
        return candidate_id


def improvement_patch_candidate_plan_to_dict(
    plan: ImprovementPatchCandidatePlan,
) -> dict[str, object]:
    return {
        "candidate_id": plan.candidate_id,
        "run_id": plan.run_id,
        "proposal_id": plan.proposal_id,
        "gate_evaluation_id": plan.gate_evaluation_id,
        "target": improvement_target_to_dict(plan.target),
        "base_revision": plan.base_revision,
        "summary": plan.summary,
        "planned_changes": list(plan.planned_changes),
        "expected_files": list(plan.expected_files),
        "required_permissions": list(plan.required_permissions),
        "evaluation_plan": improvement_evaluation_plan_to_dict(plan.evaluation_plan),
        "rollback_plan": plan.rollback_plan,
        "status": plan.status,
        "created_at": plan.created_at,
        "metadata": _metadata_to_dict(plan.metadata),
    }


def improvement_patch_candidate_plan_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidatePlan:
    return ImprovementPatchCandidatePlan(
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        target=improvement_target_from_dict(_expect_mapping(data["target"], "target")),
        base_revision=str(data["base_revision"]),
        summary=str(data["summary"]),
        planned_changes=_string_tuple(data.get("planned_changes", ()), "planned_changes"),
        expected_files=_string_tuple(data.get("expected_files", ()), "expected_files"),
        required_permissions=_candidate_permission_tuple(
            _string_tuple(data.get("required_permissions", ()), "required_permissions"),
            improvement_target_from_dict(_expect_mapping(data["target"], "target")),
        ),
        evaluation_plan=improvement_evaluation_plan_from_dict(
            _expect_mapping(data["evaluation_plan"], "evaluation_plan")
        ),
        rollback_plan=str(data["rollback_plan"]),
        status=cast(
            ImprovementPatchCandidateStatus,
            _literal_from_object(data.get("status", "planned"), _CANDIDATE_STATUSES, "status"),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_gate_linkage(
    run: ImprovementRun,
    gate_evaluation: ImprovementGateEvaluation,
) -> None:
    if gate_evaluation.run_id != run.run_id:
        raise ImprovementPatchCandidateValidationError(
            "ImprovementGateEvaluation.run_id must match ImprovementRun.run_id"
        )
    if gate_evaluation.proposal_id != run.proposal.proposal_id:
        raise ImprovementPatchCandidateValidationError(
            "ImprovementGateEvaluation.proposal_id must match proposal.proposal_id"
        )


def _candidate_permission_tuple(
    permissions: Sequence[str],
    target: ImprovementTarget,
) -> tuple[ImprovementRequiredPermission, ...]:
    items = tuple(
        cast(
            ImprovementRequiredPermission,
            _literal_from_object(item, _REQUIRED_PERMISSIONS, "required_permissions"),
        )
        for item in permissions
    )
    if not items:
        raise ImprovementPatchCandidateValidationError(
            "ImprovementPatchCandidatePlan.required_permissions must not be empty"
        )
    if "none" in items:
        raise ImprovementPatchCandidateValidationError(
            "ImprovementPatchCandidatePlan.required_permissions cannot include none"
        )
    if target.kind == "source_code":
        missing = _SOURCE_CODE_REQUIRED_PERMISSIONS.difference(items)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ImprovementPatchCandidateValidationError(
                "ImprovementPatchCandidatePlan.required_permissions missing: "
                f"{joined}"
            )
    return cast(tuple[ImprovementRequiredPermission, ...], items)


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
        raise ValueError(f"ImprovementPatchCandidatePlan.{name} must not be empty")
    return items


def _non_empty_string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(values)
    for item in items:
        _require_non_empty(item, name)
    return items


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
    "ImprovementPatchCandidateError",
    "ImprovementPatchCandidatePlan",
    "ImprovementPatchCandidatePlanner",
    "ImprovementPatchCandidateStatus",
    "ImprovementPatchCandidateValidationError",
    "improvement_patch_candidate_plan_from_dict",
    "improvement_patch_candidate_plan_to_dict",
)
