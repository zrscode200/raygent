"""Evaluation gate records for bounded improvement proposals.

The gate layer evaluates supplied results for an existing improvement run. It
does not execute checks, call models, mutate files, create worktrees, or promote
candidates.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.models import ImprovementRun

ImprovementGateKind = Literal[
    "evidence_bounds",
    "evidence_freshness",
    "permission_boundary",
    "evaluation_plan",
    "non_regression",
    "cost_budget",
    "review_status",
    "rollback_plan",
    "no_mutation",
    "other",
]
ImprovementGateStatus = Literal[
    "pass",
    "warn",
    "fail",
    "needs_review",
    "not_applicable",
]
ImprovementGateDecision = Literal[
    "pass",
    "warn",
    "fail",
    "needs_review",
]

_GATE_KINDS: frozenset[str] = frozenset(
    {
        "evidence_bounds",
        "evidence_freshness",
        "permission_boundary",
        "evaluation_plan",
        "non_regression",
        "cost_budget",
        "review_status",
        "rollback_plan",
        "no_mutation",
        "other",
    }
)
_GATE_STATUSES: frozenset[str] = frozenset(
    {
        "pass",
        "warn",
        "fail",
        "needs_review",
        "not_applicable",
    }
)
_GATE_DECISIONS: frozenset[str] = frozenset(
    {
        "pass",
        "warn",
        "fail",
        "needs_review",
    }
)
_EVIDENCE_DEPENDENT_GATES: frozenset[str] = frozenset(
    {
        "evidence_bounds",
        "evidence_freshness",
        "non_regression",
        "cost_budget",
    }
)


class ImprovementGateError(ValueError):
    """Raised when an improvement gate evaluation cannot be produced."""


class ImprovementGateValidationError(ImprovementGateError):
    """Raised when supplied gate data violates the gate contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementGateResult:
    """One supplied gate result for an improvement proposal run."""

    gate_id: str
    kind: ImprovementGateKind
    status: ImprovementGateStatus
    summary: str
    evidence_ids: tuple[str, ...] = ()
    reviewer: str | None = None
    required: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.gate_id, "ImprovementGateResult.gate_id")
        _require_literal(self.kind, _GATE_KINDS, "ImprovementGateResult.kind")
        _require_literal(self.status, _GATE_STATUSES, "ImprovementGateResult.status")
        _require_non_empty(self.summary, "ImprovementGateResult.summary")
        object.__setattr__(
            self,
            "evidence_ids",
            _non_empty_string_tuple(self.evidence_ids, "evidence_ids"),
        )
        if self.kind in _EVIDENCE_DEPENDENT_GATES and not self.evidence_ids:
            raise ValueError(
                "ImprovementGateResult.evidence_ids must not be empty for "
                f"{self.kind} gates"
            )
        if self.reviewer is not None:
            _require_non_empty(self.reviewer, "ImprovementGateResult.reviewer")
        if self.created_at < 0:
            raise ValueError("ImprovementGateResult.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementGateEvaluation:
    """Aggregate gate decision for one improvement run and proposal."""

    evaluation_id: str
    run_id: str
    proposal_id: str
    results: tuple[ImprovementGateResult, ...]
    manual_review_required: bool = True
    decision: ImprovementGateDecision = field(init=False)
    warnings: tuple[str, ...] = field(init=False)
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.evaluation_id, "ImprovementGateEvaluation.evaluation_id")
        _require_non_empty(self.run_id, "ImprovementGateEvaluation.run_id")
        _require_non_empty(self.proposal_id, "ImprovementGateEvaluation.proposal_id")
        object.__setattr__(self, "results", tuple(self.results))
        _validate_unique_gate_ids(self.results)
        decision, warnings = _derive_decision(self.manual_review_required, self.results)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "warnings", warnings)
        if self.created_at < 0:
            raise ValueError("ImprovementGateEvaluation.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementGatePolicy:
    """Pure policy evaluator over supplied improvement gate results."""

    clock: Callable[[], float] = time.time
    evaluation_id_factory: Callable[[], str] | None = None

    def evaluate(
        self,
        run: ImprovementRun,
        results: Sequence[ImprovementGateResult],
        *,
        evaluation_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementGateEvaluation:
        """Validate gate results and derive a gate decision for ``run``.

        The policy consumes caller-supplied results only. It does not run
        commands, inspect files, call providers, request permissions, or promote
        candidates.
        """

        gate_results = tuple(results)
        _validate_unique_gate_ids(gate_results)
        _validate_gate_evidence_ids(run, gate_results)
        return ImprovementGateEvaluation(
            evaluation_id=evaluation_id or self._new_evaluation_id(),
            run_id=run.run_id,
            proposal_id=run.proposal.proposal_id,
            results=gate_results,
            manual_review_required=run.proposal.evaluation_plan.manual_review_required,
            created_at=self.clock(),
            metadata=metadata or {},
        )

    def _new_evaluation_id(self) -> str:
        evaluation_id = (
            self.evaluation_id_factory()
            if self.evaluation_id_factory is not None
            else f"ige_{uuid4().hex}"
        )
        if not evaluation_id.strip():
            raise ImprovementGateValidationError(
                "evaluation_id_factory returned an empty id"
            )
        return evaluation_id


def improvement_gate_result_to_dict(
    result: ImprovementGateResult,
) -> dict[str, object]:
    return {
        "gate_id": result.gate_id,
        "kind": result.kind,
        "status": result.status,
        "summary": result.summary,
        "evidence_ids": list(result.evidence_ids),
        "reviewer": result.reviewer,
        "required": result.required,
        "created_at": result.created_at,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_gate_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementGateResult:
    return ImprovementGateResult(
        gate_id=str(data["gate_id"]),
        kind=cast(
            ImprovementGateKind,
            _literal_from_object(data["kind"], _GATE_KINDS, "kind"),
        ),
        status=cast(
            ImprovementGateStatus,
            _literal_from_object(data["status"], _GATE_STATUSES, "status"),
        ),
        summary=str(data["summary"]),
        evidence_ids=_string_tuple(data.get("evidence_ids", ()), "evidence_ids"),
        reviewer=_optional_str(data.get("reviewer")),
        required=bool(data.get("required", True)),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_gate_evaluation_to_dict(
    evaluation: ImprovementGateEvaluation,
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation.evaluation_id,
        "run_id": evaluation.run_id,
        "proposal_id": evaluation.proposal_id,
        "decision": evaluation.decision,
        "manual_review_required": evaluation.manual_review_required,
        "results": [
            improvement_gate_result_to_dict(result) for result in evaluation.results
        ],
        "warnings": list(evaluation.warnings),
        "created_at": evaluation.created_at,
        "metadata": _metadata_to_dict(evaluation.metadata),
    }


def improvement_gate_evaluation_from_dict(
    data: Mapping[str, object],
) -> ImprovementGateEvaluation:
    claimed_decision = cast(
        ImprovementGateDecision,
        _literal_from_object(data["decision"], _GATE_DECISIONS, "decision"),
    )
    claimed_warnings = _string_tuple(data.get("warnings", ()), "warnings")
    evaluation = ImprovementGateEvaluation(
        evaluation_id=str(data["evaluation_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        results=tuple(
            improvement_gate_result_from_dict(item)
            for item in _mapping_sequence(data.get("results", ()), "results")
        ),
        manual_review_required=bool(data.get("manual_review_required", True)),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )
    if evaluation.decision != claimed_decision:
        raise ImprovementGateValidationError(
            "ImprovementGateEvaluation.decision does not match derived gate policy"
        )
    if evaluation.warnings != claimed_warnings:
        raise ImprovementGateValidationError(
            "ImprovementGateEvaluation.warnings do not match derived gate policy"
        )
    return evaluation


def _derive_decision(
    manual_review_required: bool,
    results: tuple[ImprovementGateResult, ...],
) -> tuple[ImprovementGateDecision, tuple[str, ...]]:
    warnings: list[str] = []
    required = tuple(result for result in results if result.required)
    required_statuses = {result.status for result in required}
    all_statuses = {result.status for result in results}

    if not results:
        warnings.append("no gate results supplied")
        return "needs_review", tuple(warnings)
    if "fail" in required_statuses:
        return "fail", tuple(warnings)
    if "needs_review" in all_statuses:
        return "needs_review", tuple(warnings)

    required_review_statuses = tuple(
        result.status
        for result in results
        if result.required and result.kind == "review_status"
    )
    if manual_review_required and not required_review_statuses:
        warnings.append("manual review is required but no required review gate was supplied")
        return "needs_review", tuple(warnings)
    if manual_review_required and all(
        status == "not_applicable" for status in required_review_statuses
    ):
        warnings.append("manual review is required but the review gate was not applicable")
        return "needs_review", tuple(warnings)

    optional_failures = tuple(
        result for result in results if not result.required and result.status == "fail"
    )
    if optional_failures:
        warnings.append("one or more optional gates failed")

    if "warn" in {result.status for result in results} or optional_failures:
        return "warn", tuple(warnings)
    return "pass", tuple(warnings)


def _validate_unique_gate_ids(results: Sequence[ImprovementGateResult]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for result in results:
        if result.gate_id in seen:
            duplicates.append(result.gate_id)
        seen.add(result.gate_id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementGateValidationError(
            f"ImprovementGateEvaluation.results contains duplicate gate ids: {joined}"
        )


def _validate_gate_evidence_ids(
    run: ImprovementRun,
    results: Sequence[ImprovementGateResult],
) -> None:
    known_evidence_ids = {item.evidence_id for item in run.evidence}
    unknown = tuple(
        evidence_id
        for result in results
        for evidence_id in result.evidence_ids
        if evidence_id not in known_evidence_ids
    )
    if unknown:
        joined = ", ".join(dict.fromkeys(unknown))
        raise ImprovementGateValidationError(
            "ImprovementGateResult.evidence_ids reference unknown evidence: "
            f"{joined}"
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


__all__ = (
    "ImprovementGateDecision",
    "ImprovementGateError",
    "ImprovementGateEvaluation",
    "ImprovementGateKind",
    "ImprovementGatePolicy",
    "ImprovementGateResult",
    "ImprovementGateStatus",
    "ImprovementGateValidationError",
    "improvement_gate_evaluation_from_dict",
    "improvement_gate_evaluation_to_dict",
    "improvement_gate_result_from_dict",
    "improvement_gate_result_to_dict",
)
