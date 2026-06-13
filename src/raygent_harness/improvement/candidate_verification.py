"""Data-only verification plan records for improvement patch candidates.

The RSI-005A layer plans bounded candidate verification from an existing patch
candidate plan and materialization. It does not invoke runners, execute shell
commands, mutate files, clean worktrees, promote candidates, or integrate
product goals.
"""

from __future__ import annotations

import json
import posixpath
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.candidate_materialization import (
    DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES,
    DEFAULT_MAX_MATERIALIZATION_PATH_CHARS,
    ImprovementPatchCandidateEvaluationKind,
    ImprovementPatchCandidateMaterialization,
)
from raygent_harness.improvement.candidates import ImprovementPatchCandidatePlan
from raygent_harness.improvement.models import ImprovementEvaluationCheck

DEFAULT_MAX_VERIFICATION_CHECKS = 30
DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS = 200
DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS = 4_000
DEFAULT_MAX_VERIFICATION_REF_CHARS = 1_000
DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS = 100
DEFAULT_MAX_VERIFICATION_RECORD_METADATA_CHARS = 20_000

ImprovementPatchCandidateVerificationPlanStatus = Literal["verification_planned"]
ImprovementPatchCandidateVerificationCheckSource = Literal[
    "checks",
    "non_regression_checks",
    "cost_checks",
]

_VERIFICATION_PLAN_STATUSES: frozenset[str] = frozenset({"verification_planned"})
_VERIFICATION_CHECK_SOURCES: frozenset[str] = frozenset(
    {"checks", "non_regression_checks", "cost_checks"}
)
_EVALUATION_KINDS: frozenset[str] = frozenset(
    {"static_review", "non_regression", "unit_tests", "manual_review", "other"}
)


class ImprovementPatchCandidateVerificationError(ValueError):
    """Raised when a candidate verification plan cannot be produced."""


class ImprovementPatchCandidateVerificationValidationError(
    ImprovementPatchCandidateVerificationError
):
    """Raised when verification-plan data violates the RSI-005A contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationCheck:
    """One bounded, data-only verification check for a materialized candidate."""

    check_id: str
    kind: ImprovementPatchCandidateEvaluationKind
    name: str
    instruction: str
    source_plan_section: ImprovementPatchCandidateVerificationCheckSource
    required: bool = True
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_bounded_non_empty(
            self.check_id,
            "ImprovementPatchCandidateVerificationCheck.check_id",
            DEFAULT_MAX_VERIFICATION_REF_CHARS,
            "DEFAULT_MAX_VERIFICATION_REF_CHARS",
        )
        _require_literal(
            self.kind,
            _EVALUATION_KINDS,
            "ImprovementPatchCandidateVerificationCheck.kind",
        )
        _require_bounded_non_empty(
            self.name,
            "ImprovementPatchCandidateVerificationCheck.name",
            DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS,
            "DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS",
        )
        _require_bounded_non_empty(
            self.instruction,
            "ImprovementPatchCandidateVerificationCheck.instruction",
            DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
            "DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
        )
        _require_literal(
            self.source_plan_section,
            _VERIFICATION_CHECK_SOURCES,
            "ImprovementPatchCandidateVerificationCheck.source_plan_section",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationPlan:
    """Serializable data-only plan for verifying one materialized candidate."""

    verification_plan_id: str
    materialization_id: str
    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    base_revision: str
    worktree_path: str
    worktree_slug: str
    patch_digest: str
    allowed_changed_files: tuple[str, ...]
    checks: tuple[ImprovementPatchCandidateVerificationCheck, ...]
    status: ImprovementPatchCandidateVerificationPlanStatus = "verification_planned"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.verification_plan_id,
            "ImprovementPatchCandidateVerificationPlan.verification_plan_id",
        )
        _require_non_empty(
            self.materialization_id,
            "ImprovementPatchCandidateVerificationPlan.materialization_id",
        )
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateVerificationPlan.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateVerificationPlan.candidate_id",
        )
        _require_non_empty(
            self.run_id,
            "ImprovementPatchCandidateVerificationPlan.run_id",
        )
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateVerificationPlan.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateVerificationPlan.gate_evaluation_id",
        )
        _require_non_empty(
            self.base_revision,
            "ImprovementPatchCandidateVerificationPlan.base_revision",
        )
        _require_non_empty(
            self.worktree_path,
            "ImprovementPatchCandidateVerificationPlan.worktree_path",
        )
        _require_non_empty(
            self.worktree_slug,
            "ImprovementPatchCandidateVerificationPlan.worktree_slug",
        )
        _require_non_empty(
            self.patch_digest,
            "ImprovementPatchCandidateVerificationPlan.patch_digest",
        )
        object.__setattr__(
            self,
            "allowed_changed_files",
            _changed_file_tuple(self.allowed_changed_files, "allowed_changed_files"),
        )
        object.__setattr__(self, "checks", _verification_check_tuple(self.checks))
        _require_literal(
            self.status,
            _VERIFICATION_PLAN_STATUSES,
            "ImprovementPatchCandidateVerificationPlan.status",
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateVerificationPlan.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationPlanner:
    """Pure planner for data-only candidate verification records."""

    clock: Callable[[], float] = time.time
    verification_plan_id_factory: Callable[[], str] | None = None

    def plan(
        self,
        candidate_plan: ImprovementPatchCandidatePlan,
        materialization: ImprovementPatchCandidateMaterialization,
        *,
        verification_plan_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateVerificationPlan:
        """Create a data-only verification plan for one materialized candidate."""

        _validate_plan_materialization_linkage(candidate_plan, materialization)
        checks = _checks_from_evaluation_plan(candidate_plan)
        return ImprovementPatchCandidateVerificationPlan(
            verification_plan_id=verification_plan_id
            or self._new_verification_plan_id(),
            materialization_id=materialization.materialization_id,
            allocation_id=materialization.allocation_id,
            candidate_id=materialization.candidate_id,
            run_id=materialization.run_id,
            proposal_id=materialization.proposal_id,
            gate_evaluation_id=materialization.gate_evaluation_id,
            base_revision=materialization.base_revision,
            worktree_path=materialization.worktree_path,
            worktree_slug=materialization.worktree_slug,
            patch_digest=materialization.patch_digest,
            allowed_changed_files=materialization.changed_files,
            checks=checks,
            status="verification_planned",
            created_at=self.clock(),
            metadata=metadata or {},
        )

    def _new_verification_plan_id(self) -> str:
        verification_plan_id = (
            self.verification_plan_id_factory()
            if self.verification_plan_id_factory is not None
            else f"ipcvp_{uuid4().hex}"
        )
        if not verification_plan_id.strip():
            raise ImprovementPatchCandidateVerificationValidationError(
                "verification_plan_id_factory returned an empty id"
            )
        return verification_plan_id


def improvement_patch_candidate_verification_check_to_dict(
    check: ImprovementPatchCandidateVerificationCheck,
) -> dict[str, object]:
    return {
        "check_id": check.check_id,
        "kind": check.kind,
        "name": check.name,
        "instruction": check.instruction,
        "source_plan_section": check.source_plan_section,
        "required": check.required,
        "metadata": _metadata_to_dict(check.metadata),
    }


def improvement_patch_candidate_verification_check_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationCheck:
    return ImprovementPatchCandidateVerificationCheck(
        check_id=str(data["check_id"]),
        kind=cast(
            ImprovementPatchCandidateEvaluationKind,
            _literal_from_object(data["kind"], _EVALUATION_KINDS, "kind"),
        ),
        name=str(data["name"]),
        instruction=str(data["instruction"]),
        source_plan_section=cast(
            ImprovementPatchCandidateVerificationCheckSource,
            _literal_from_object(
                data["source_plan_section"],
                _VERIFICATION_CHECK_SOURCES,
                "source_plan_section",
            ),
        ),
        required=bool(data.get("required", True)),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_verification_plan_to_dict(
    plan: ImprovementPatchCandidateVerificationPlan,
) -> dict[str, object]:
    return {
        "verification_plan_id": plan.verification_plan_id,
        "materialization_id": plan.materialization_id,
        "allocation_id": plan.allocation_id,
        "candidate_id": plan.candidate_id,
        "run_id": plan.run_id,
        "proposal_id": plan.proposal_id,
        "gate_evaluation_id": plan.gate_evaluation_id,
        "base_revision": plan.base_revision,
        "worktree_path": plan.worktree_path,
        "worktree_slug": plan.worktree_slug,
        "patch_digest": plan.patch_digest,
        "allowed_changed_files": list(plan.allowed_changed_files),
        "checks": [
            improvement_patch_candidate_verification_check_to_dict(check)
            for check in plan.checks
        ],
        "status": plan.status,
        "created_at": plan.created_at,
        "metadata": _metadata_to_dict(plan.metadata),
    }


def improvement_patch_candidate_verification_plan_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationPlan:
    return ImprovementPatchCandidateVerificationPlan(
        verification_plan_id=str(data["verification_plan_id"]),
        materialization_id=str(data["materialization_id"]),
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        base_revision=str(data["base_revision"]),
        worktree_path=str(data["worktree_path"]),
        worktree_slug=str(data["worktree_slug"]),
        patch_digest=str(data["patch_digest"]),
        allowed_changed_files=_string_tuple(
            data.get("allowed_changed_files", ()),
            "allowed_changed_files",
        ),
        checks=tuple(
            improvement_patch_candidate_verification_check_from_dict(item)
            for item in _mapping_sequence(data.get("checks", ()), "checks")
        ),
        status=cast(
            ImprovementPatchCandidateVerificationPlanStatus,
            _literal_from_object(
                data.get("status", "verification_planned"),
                _VERIFICATION_PLAN_STATUSES,
                "status",
            ),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_plan_materialization_linkage(
    candidate_plan: ImprovementPatchCandidatePlan,
    materialization: ImprovementPatchCandidateMaterialization,
) -> None:
    if candidate_plan.status != "planned":
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlanner requires a planned "
            "candidate"
        )
    if materialization.status != "materialized":
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlanner requires a materialized "
            "candidate"
        )
    fields = (
        "candidate_id",
        "run_id",
        "proposal_id",
        "gate_evaluation_id",
        "base_revision",
    )
    for field_name in fields:
        if getattr(candidate_plan, field_name) != getattr(materialization, field_name):
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationPlanner linkage mismatch: "
                f"{field_name}"
            )
    expected_files = _changed_file_tuple(candidate_plan.expected_files, "expected_files")
    changed_files = _changed_file_tuple(materialization.changed_files, "changed_files")
    outside = tuple(item for item in changed_files if item not in set(expected_files))
    if outside:
        joined = ", ".join(outside)
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlanner.changed_files outside "
            f"expected files: {joined}"
        )


def _checks_from_evaluation_plan(
    candidate_plan: ImprovementPatchCandidatePlan,
) -> tuple[ImprovementPatchCandidateVerificationCheck, ...]:
    checks: list[ImprovementPatchCandidateVerificationCheck] = []
    checks.extend(
        _verification_checks_from_section(
            candidate_plan.evaluation_plan.checks,
            source_plan_section="checks",
            default_kind="other",
        )
    )
    checks.extend(
        _verification_checks_from_section(
            candidate_plan.evaluation_plan.non_regression_checks,
            source_plan_section="non_regression_checks",
            default_kind="non_regression",
        )
    )
    checks.extend(
        _verification_checks_from_section(
            candidate_plan.evaluation_plan.cost_checks,
            source_plan_section="cost_checks",
            default_kind="other",
        )
    )
    return _verification_check_tuple(checks)


def _verification_checks_from_section(
    checks: Sequence[ImprovementEvaluationCheck],
    *,
    source_plan_section: ImprovementPatchCandidateVerificationCheckSource,
    default_kind: ImprovementPatchCandidateEvaluationKind,
) -> tuple[ImprovementPatchCandidateVerificationCheck, ...]:
    items: list[ImprovementPatchCandidateVerificationCheck] = []
    for index, check in enumerate(checks, start=1):
        items.append(
            ImprovementPatchCandidateVerificationCheck(
                check_id=f"ipcvchk_{source_plan_section}_{index}",
                kind=default_kind,
                name=check.name,
                instruction=check.instruction,
                source_plan_section=source_plan_section,
                required=check.required,
                metadata=check.metadata,
            )
        )
    return tuple(items)


def _verification_check_tuple(
    checks: Sequence[ImprovementPatchCandidateVerificationCheck],
) -> tuple[ImprovementPatchCandidateVerificationCheck, ...]:
    items = tuple(checks)
    if not items:
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlan.checks must not be empty"
        )
    if len(items) > DEFAULT_MAX_VERIFICATION_CHECKS:
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlan.checks exceeds "
            "DEFAULT_MAX_VERIFICATION_CHECKS"
        )
    duplicates = _duplicates(tuple(check.check_id for check in items))
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationPlan.checks contains duplicate "
            f"check ids: {joined}"
        )
    return items


def _changed_file_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_normalize_relative_path(value, name) for value in values)
    if not items:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not be empty"
        )
    if len(items) > DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} exceeds DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} contains duplicate normalized paths: {joined}"
        )
    return items


def _normalize_relative_path(value: str, name: str) -> str:
    if "\x00" in value:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not contain NUL bytes"
        )
    if value.startswith("/"):
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must be a relative path"
        )
    raw_path = value.replace("\\", "/")
    if raw_path.startswith("/"):
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must be a relative path"
        )
    raw_parts = raw_path.split("/")
    if ".." in raw_parts:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not traverse parent directories"
        )
    normalized = posixpath.normpath(raw_path)
    if normalized in {"", "."}:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not be empty"
        )
    if normalized == ".." or normalized.startswith("../"):
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not traverse parent directories"
        )
    if len(normalized) > DEFAULT_MAX_MATERIALIZATION_PATH_CHARS:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} exceeds DEFAULT_MAX_MATERIALIZATION_PATH_CHARS"
        )
    return normalized


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must be non-empty"
        )


def _require_bounded_non_empty(
    value: str,
    name: str,
    max_chars: int,
    max_chars_name: str,
) -> None:
    _require_non_empty(value, name)
    if len(value) > max_chars:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} exceeds {max_chars_name}"
        )
    if "\x00" in value:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not contain NUL bytes"
        )


def _require_literal(value: str, allowed: frozenset[str], name: str) -> None:
    if value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must be one of: {expected}"
        )


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
    if len(encoded) > DEFAULT_MAX_VERIFICATION_RECORD_METADATA_CHARS:
        raise ImprovementPatchCandidateVerificationValidationError(
            "metadata exceeds DEFAULT_MAX_VERIFICATION_RECORD_METADATA_CHARS"
        )
    return MappingProxyType(dict(frozen_mapping))


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


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a number") from exc


def _duplicates(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return tuple(duplicates)


__all__ = (
    "DEFAULT_MAX_VERIFICATION_CHECKS",
    "DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS",
    "DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
    "DEFAULT_MAX_VERIFICATION_RECORD_METADATA_CHARS",
    "DEFAULT_MAX_VERIFICATION_REF_CHARS",
    "DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS",
    "ImprovementPatchCandidateVerificationCheck",
    "ImprovementPatchCandidateVerificationCheckSource",
    "ImprovementPatchCandidateVerificationError",
    "ImprovementPatchCandidateVerificationPlan",
    "ImprovementPatchCandidateVerificationPlanStatus",
    "ImprovementPatchCandidateVerificationPlanner",
    "ImprovementPatchCandidateVerificationValidationError",
    "improvement_patch_candidate_verification_check_from_dict",
    "improvement_patch_candidate_verification_check_to_dict",
    "improvement_patch_candidate_verification_plan_from_dict",
    "improvement_patch_candidate_verification_plan_to_dict",
)
