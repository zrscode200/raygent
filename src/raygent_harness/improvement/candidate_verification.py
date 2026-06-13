"""Data-only verification plan and runner records for improvement candidates.

The RSI-005A/RSI-005B layer plans bounded candidate verification from an
existing patch candidate plan and materialization, then records caller-owned
verifier results through an injected protocol. It does not ship a runner,
execute shell commands, mutate files directly, clean worktrees, promote
candidates, or integrate product goals.
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
    DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS,
    DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES,
    DEFAULT_MAX_MATERIALIZATION_PATH_CHARS,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationKind,
    ImprovementPatchCandidateEvaluationResult,
    ImprovementPatchCandidateEvaluationStatus,
    ImprovementPatchCandidateMaterialization,
)
from raygent_harness.improvement.candidates import ImprovementPatchCandidatePlan
from raygent_harness.improvement.models import (
    ImprovementEvaluationCheck,
    ImprovementRequiredPermission,
)

DEFAULT_MAX_VERIFICATION_CHECKS = 30
DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS = 200
DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS = 4_000
DEFAULT_MAX_VERIFICATION_REF_CHARS = 1_000
DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS = 100
DEFAULT_MAX_VERIFICATION_RECORD_METADATA_CHARS = 20_000

ImprovementPatchCandidateVerificationPlanStatus = Literal["verification_planned"]
ImprovementPatchCandidateVerificationStatus = Literal["verification_recorded"]
ImprovementPatchCandidateVerificationCheckSource = Literal[
    "checks",
    "non_regression_checks",
    "cost_checks",
]

_VERIFICATION_PLAN_STATUSES: frozenset[str] = frozenset({"verification_planned"})
_VERIFICATION_CHECK_SOURCES: frozenset[str] = frozenset(
    {"checks", "non_regression_checks", "cost_checks"}
)
_VERIFICATION_STATUSES: frozenset[str] = frozenset({"verification_recorded"})
_EVALUATION_KINDS: frozenset[str] = frozenset(
    {"static_review", "non_regression", "unit_tests", "manual_review", "other"}
)
_EVALUATION_STATUSES: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "needs_review", "not_applicable"}
)
_LOCAL_VERIFICATION_PERMISSIONS: tuple[ImprovementRequiredPermission, ...] = (
    "filesystem_mutation",
    "shell",
)
_LOCAL_VERIFICATION_PERMISSION_SET: frozenset[str] = frozenset(
    _LOCAL_VERIFICATION_PERMISSIONS
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
_RAW_REFERENCE_MARKERS: tuple[str, ...] = (
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
_VERIFICATION_RESULT_METADATA_KEY = "verification_result_metadata"
_VERIFICATION_EVALUATION_METADATA_KEYS: frozenset[str] = frozenset(
    {"verification_id", "verification_digest"}
)


class ImprovementPatchCandidateVerificationError(ValueError):
    """Raised when a candidate verification plan cannot be produced."""


class ImprovementPatchCandidateVerificationValidationError(
    ImprovementPatchCandidateVerificationError
):
    """Raised when verification data violates the RSI-005 contract."""


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


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationApproval:
    """Call-time authority for one injected verification request."""

    approved_permissions: tuple[ImprovementRequiredPermission, ...]
    reason: str
    approved_by: str
    approved: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.approved:
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationApproval.approved must be true"
            )
        object.__setattr__(
            self,
            "approved_permissions",
            _exact_local_verification_permission_tuple(
                self.approved_permissions,
                "ImprovementPatchCandidateVerificationApproval.approved_permissions",
            ),
        )
        _require_non_empty(
            self.reason,
            "ImprovementPatchCandidateVerificationApproval.reason",
        )
        _require_non_empty(
            self.approved_by,
            "ImprovementPatchCandidateVerificationApproval.approved_by",
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateVerificationApproval.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationRequest:
    """Normalized request passed to an injected verifier."""

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
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _validate_verification_request_fields(self)


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationCheckResult:
    """One bounded result for a planned verification check."""

    check_id: str
    status: ImprovementPatchCandidateEvaluationStatus
    summary: str
    changed_files: tuple[str, ...] = ()
    output_excerpt: str | None = None
    output_reference: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_bounded_non_empty(
            self.check_id,
            "ImprovementPatchCandidateVerificationCheckResult.check_id",
            DEFAULT_MAX_VERIFICATION_REF_CHARS,
            "DEFAULT_MAX_VERIFICATION_REF_CHARS",
        )
        _require_literal(
            self.status,
            _EVALUATION_STATUSES,
            "ImprovementPatchCandidateVerificationCheckResult.status",
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidateVerificationCheckResult.summary",
            DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
            "DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
        )
        object.__setattr__(
            self,
            "changed_files",
            _changed_file_tuple_or_empty(
                self.changed_files,
                "ImprovementPatchCandidateVerificationCheckResult.changed_files",
            ),
        )
        if self.output_excerpt is not None:
            _require_bounded_non_empty(
                self.output_excerpt,
                "ImprovementPatchCandidateVerificationCheckResult.output_excerpt",
                DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS,
                "DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS",
            )
        if self.output_reference is not None:
            _require_verification_reference(
                self.output_reference,
                "ImprovementPatchCandidateVerificationCheckResult.output_reference",
                DEFAULT_MAX_VERIFICATION_REF_CHARS,
                "DEFAULT_MAX_VERIFICATION_REF_CHARS",
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationResult:
    """Bounded result returned by a caller-owned verifier."""

    runner_ref: str
    runner_kind: str
    results: tuple[ImprovementPatchCandidateVerificationCheckResult, ...]
    summary: str
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_verification_reference(
            self.runner_ref,
            "ImprovementPatchCandidateVerificationResult.runner_ref",
            DEFAULT_MAX_VERIFICATION_REF_CHARS,
            "DEFAULT_MAX_VERIFICATION_REF_CHARS",
        )
        _require_verification_reference(
            self.runner_kind,
            "ImprovementPatchCandidateVerificationResult.runner_kind",
            DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS,
            "DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS",
        )
        object.__setattr__(
            self,
            "results",
            _verification_check_result_tuple(self.results),
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidateVerificationResult.summary",
            DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
            "DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ImprovementPatchCandidateVerifier(Protocol):
    """Caller-owned verifier seam for RSI-005B records."""

    async def verify(
        self,
        request: ImprovementPatchCandidateVerificationRequest,
    ) -> ImprovementPatchCandidateVerificationResult:
        """Verify the supplied normalized request."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationRecord:
    """Serializable record for one injected candidate verification."""

    verification_id: str
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
    runner_ref: str
    runner_kind: str
    results: tuple[ImprovementPatchCandidateVerificationCheckResult, ...]
    summary: str
    verification_digest: str
    status: ImprovementPatchCandidateVerificationStatus = "verification_recorded"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.verification_id,
            "ImprovementPatchCandidateVerificationRecord.verification_id",
        )
        _require_literal(
            self.status,
            _VERIFICATION_STATUSES,
            "ImprovementPatchCandidateVerificationRecord.status",
        )
        _require_verification_reference(
            self.runner_ref,
            "ImprovementPatchCandidateVerificationRecord.runner_ref",
            DEFAULT_MAX_VERIFICATION_REF_CHARS,
            "DEFAULT_MAX_VERIFICATION_REF_CHARS",
        )
        _require_verification_reference(
            self.runner_kind,
            "ImprovementPatchCandidateVerificationRecord.runner_kind",
            DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS,
            "DEFAULT_MAX_VERIFICATION_RUNNER_KIND_CHARS",
        )
        object.__setattr__(
            self,
            "results",
            _verification_check_result_tuple(self.results),
        )
        _require_bounded_non_empty(
            self.summary,
            "ImprovementPatchCandidateVerificationRecord.summary",
            DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
            "DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
        )
        _require_non_empty(
            self.verification_digest,
            "ImprovementPatchCandidateVerificationRecord.verification_digest",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        request = _verification_request_from_record(self)
        _validate_verification_result_for_request(request, self.results)
        expected_digest = _verification_digest(
            request,
            runner_ref=self.runner_ref,
            runner_kind=self.runner_kind,
            results=self.results,
        )
        if self.verification_digest != expected_digest:
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationRecord.verification_digest "
                "does not match verification record identity"
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateVerificationRecord.created_at must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateVerificationService:
    """Run one verification plan through an injected verifier protocol."""

    clock: Callable[[], float] = time.time
    verification_id_factory: Callable[[], str] | None = None

    async def verify(
        self,
        plan: ImprovementPatchCandidateVerificationPlan,
        *,
        verifier: ImprovementPatchCandidateVerifier | None,
        approval: ImprovementPatchCandidateVerificationApproval | None,
        verification_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateVerificationRecord:
        """Invoke an injected verifier and return the immutable record."""

        if plan.status != "verification_planned":
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationService requires a "
                "verification_planned plan"
            )
        if verifier is None:
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationService requires an injected "
                "ImprovementPatchCandidateVerifier"
            )
        if approval is None:
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationService requires explicit "
                "call-time approval"
            )
        _validate_approval(approval)

        request_metadata = _request_metadata(plan.metadata, metadata or {})
        request = ImprovementPatchCandidateVerificationRequest(
            verification_plan_id=plan.verification_plan_id,
            materialization_id=plan.materialization_id,
            allocation_id=plan.allocation_id,
            candidate_id=plan.candidate_id,
            run_id=plan.run_id,
            proposal_id=plan.proposal_id,
            gate_evaluation_id=plan.gate_evaluation_id,
            base_revision=plan.base_revision,
            worktree_path=plan.worktree_path,
            worktree_slug=plan.worktree_slug,
            patch_digest=plan.patch_digest,
            allowed_changed_files=plan.allowed_changed_files,
            checks=plan.checks,
            metadata=request_metadata,
        )
        verifier_result = await verifier.verify(request)
        _validate_verification_result_for_request(request, verifier_result.results)
        record_metadata = _record_metadata(request.metadata, verifier_result.metadata)
        verification_digest = _verification_digest(
            request,
            runner_ref=verifier_result.runner_ref,
            runner_kind=verifier_result.runner_kind,
            results=verifier_result.results,
        )

        return ImprovementPatchCandidateVerificationRecord(
            verification_id=verification_id or self._new_verification_id(),
            verification_plan_id=request.verification_plan_id,
            materialization_id=request.materialization_id,
            allocation_id=request.allocation_id,
            candidate_id=request.candidate_id,
            run_id=request.run_id,
            proposal_id=request.proposal_id,
            gate_evaluation_id=request.gate_evaluation_id,
            base_revision=request.base_revision,
            worktree_path=request.worktree_path,
            worktree_slug=request.worktree_slug,
            patch_digest=request.patch_digest,
            allowed_changed_files=request.allowed_changed_files,
            checks=request.checks,
            runner_ref=verifier_result.runner_ref,
            runner_kind=verifier_result.runner_kind,
            results=verifier_result.results,
            summary=verifier_result.summary,
            verification_digest=verification_digest,
            status="verification_recorded",
            created_at=self.clock(),
            metadata=record_metadata,
        )

    def _new_verification_id(self) -> str:
        verification_id = (
            self.verification_id_factory()
            if self.verification_id_factory is not None
            else f"ipcv_{uuid4().hex}"
        )
        if not verification_id.strip():
            raise ImprovementPatchCandidateVerificationValidationError(
                "verification_id_factory returned an empty id"
            )
        return verification_id


def improvement_patch_candidate_verification_record_to_evaluation(
    record: ImprovementPatchCandidateVerificationRecord,
    *,
    evaluation_id: str | None = None,
    created_at: float | None = None,
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementPatchCandidateEvaluation:
    """Convert a verification record into the existing supplied evaluation model."""

    checks_by_id = {check.check_id: check for check in record.checks}
    result_items: list[ImprovementPatchCandidateEvaluationResult] = []
    for result in record.results:
        check = checks_by_id[result.check_id]
        result_items.append(
            ImprovementPatchCandidateEvaluationResult(
                result_id=result.check_id,
                kind=check.kind,
                status=result.status,
                summary=result.summary,
                changed_files=result.changed_files,
                output_excerpt=result.output_excerpt,
                output_reference=result.output_reference,
                required=check.required,
                created_at=record.created_at if created_at is None else created_at,
                metadata=_evaluation_result_metadata(record, check),
            )
        )
    evaluation_metadata: dict[str, object] = {
        "verification_id": record.verification_id,
        "verification_digest": record.verification_digest,
    }
    if metadata:
        caller_metadata = _metadata_to_dict(metadata)
        _reject_reserved_evaluation_metadata_key(
            caller_metadata,
            "improvement_patch_candidate_verification_record_to_evaluation.metadata",
        )
        evaluation_metadata.update(caller_metadata)
    return ImprovementPatchCandidateEvaluation(
        evaluation_id=evaluation_id or f"ipce_{record.verification_id}",
        materialization_id=record.materialization_id,
        allocation_id=record.allocation_id,
        candidate_id=record.candidate_id,
        run_id=record.run_id,
        proposal_id=record.proposal_id,
        gate_evaluation_id=record.gate_evaluation_id,
        results=tuple(result_items),
        created_at=record.created_at if created_at is None else created_at,
        metadata=_freeze_metadata(evaluation_metadata),
    )


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


def improvement_patch_candidate_verification_request_to_dict(
    request: ImprovementPatchCandidateVerificationRequest,
) -> dict[str, object]:
    return {
        "verification_plan_id": request.verification_plan_id,
        "materialization_id": request.materialization_id,
        "allocation_id": request.allocation_id,
        "candidate_id": request.candidate_id,
        "run_id": request.run_id,
        "proposal_id": request.proposal_id,
        "gate_evaluation_id": request.gate_evaluation_id,
        "base_revision": request.base_revision,
        "worktree_path": request.worktree_path,
        "worktree_slug": request.worktree_slug,
        "patch_digest": request.patch_digest,
        "allowed_changed_files": list(request.allowed_changed_files),
        "checks": [
            improvement_patch_candidate_verification_check_to_dict(check)
            for check in request.checks
        ],
        "metadata": _metadata_to_dict(request.metadata),
    }


def improvement_patch_candidate_verification_request_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationRequest:
    return ImprovementPatchCandidateVerificationRequest(
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
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_verification_check_result_to_dict(
    result: ImprovementPatchCandidateVerificationCheckResult,
) -> dict[str, object]:
    return {
        "check_id": result.check_id,
        "status": result.status,
        "summary": result.summary,
        "changed_files": list(result.changed_files),
        "output_excerpt": result.output_excerpt,
        "output_reference": result.output_reference,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_patch_candidate_verification_check_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationCheckResult:
    return ImprovementPatchCandidateVerificationCheckResult(
        check_id=str(data["check_id"]),
        status=cast(
            ImprovementPatchCandidateEvaluationStatus,
            _literal_from_object(data["status"], _EVALUATION_STATUSES, "status"),
        ),
        summary=str(data["summary"]),
        changed_files=_string_tuple(data.get("changed_files", ()), "changed_files"),
        output_excerpt=_optional_string(data.get("output_excerpt"), "output_excerpt"),
        output_reference=_optional_string(
            data.get("output_reference"),
            "output_reference",
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_verification_result_to_dict(
    result: ImprovementPatchCandidateVerificationResult,
) -> dict[str, object]:
    return {
        "runner_ref": result.runner_ref,
        "runner_kind": result.runner_kind,
        "results": [
            improvement_patch_candidate_verification_check_result_to_dict(item)
            for item in result.results
        ],
        "summary": result.summary,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_patch_candidate_verification_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationResult:
    return ImprovementPatchCandidateVerificationResult(
        runner_ref=str(data["runner_ref"]),
        runner_kind=str(data["runner_kind"]),
        results=tuple(
            improvement_patch_candidate_verification_check_result_from_dict(item)
            for item in _mapping_sequence(data.get("results", ()), "results")
        ),
        summary=str(data["summary"]),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_verification_record_to_dict(
    record: ImprovementPatchCandidateVerificationRecord,
) -> dict[str, object]:
    return {
        "verification_id": record.verification_id,
        "verification_plan_id": record.verification_plan_id,
        "materialization_id": record.materialization_id,
        "allocation_id": record.allocation_id,
        "candidate_id": record.candidate_id,
        "run_id": record.run_id,
        "proposal_id": record.proposal_id,
        "gate_evaluation_id": record.gate_evaluation_id,
        "base_revision": record.base_revision,
        "worktree_path": record.worktree_path,
        "worktree_slug": record.worktree_slug,
        "patch_digest": record.patch_digest,
        "allowed_changed_files": list(record.allowed_changed_files),
        "checks": [
            improvement_patch_candidate_verification_check_to_dict(check)
            for check in record.checks
        ],
        "runner_ref": record.runner_ref,
        "runner_kind": record.runner_kind,
        "results": [
            improvement_patch_candidate_verification_check_result_to_dict(result)
            for result in record.results
        ],
        "summary": record.summary,
        "verification_digest": record.verification_digest,
        "status": record.status,
        "created_at": record.created_at,
        "metadata": _metadata_to_dict(record.metadata),
    }


def improvement_patch_candidate_verification_record_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateVerificationRecord:
    return ImprovementPatchCandidateVerificationRecord(
        verification_id=str(data["verification_id"]),
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
        runner_ref=str(data["runner_ref"]),
        runner_kind=str(data["runner_kind"]),
        results=tuple(
            improvement_patch_candidate_verification_check_result_from_dict(item)
            for item in _mapping_sequence(data.get("results", ()), "results")
        ),
        summary=str(data["summary"]),
        verification_digest=str(data["verification_digest"]),
        status=cast(
            ImprovementPatchCandidateVerificationStatus,
            _literal_from_object(
                data.get("status", "verification_recorded"),
                _VERIFICATION_STATUSES,
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


def _validate_verification_request_fields(
    request: ImprovementPatchCandidateVerificationRequest,
) -> None:
    _require_non_empty(
        request.verification_plan_id,
        "ImprovementPatchCandidateVerificationRequest.verification_plan_id",
    )
    _require_non_empty(
        request.materialization_id,
        "ImprovementPatchCandidateVerificationRequest.materialization_id",
    )
    _require_non_empty(
        request.allocation_id,
        "ImprovementPatchCandidateVerificationRequest.allocation_id",
    )
    _require_non_empty(
        request.candidate_id,
        "ImprovementPatchCandidateVerificationRequest.candidate_id",
    )
    _require_non_empty(
        request.run_id,
        "ImprovementPatchCandidateVerificationRequest.run_id",
    )
    _require_non_empty(
        request.proposal_id,
        "ImprovementPatchCandidateVerificationRequest.proposal_id",
    )
    _require_non_empty(
        request.gate_evaluation_id,
        "ImprovementPatchCandidateVerificationRequest.gate_evaluation_id",
    )
    _require_non_empty(
        request.base_revision,
        "ImprovementPatchCandidateVerificationRequest.base_revision",
    )
    _require_non_empty(
        request.worktree_path,
        "ImprovementPatchCandidateVerificationRequest.worktree_path",
    )
    _require_non_empty(
        request.worktree_slug,
        "ImprovementPatchCandidateVerificationRequest.worktree_slug",
    )
    _require_non_empty(
        request.patch_digest,
        "ImprovementPatchCandidateVerificationRequest.patch_digest",
    )
    object.__setattr__(
        request,
        "allowed_changed_files",
        _changed_file_tuple(request.allowed_changed_files, "allowed_changed_files"),
    )
    object.__setattr__(request, "checks", _verification_check_tuple(request.checks))
    metadata = _freeze_metadata(request.metadata)
    _reject_reserved_request_metadata_key(
        metadata,
        "ImprovementPatchCandidateVerificationRequest.metadata",
    )
    object.__setattr__(request, "metadata", metadata)


def _verification_check_result_tuple(
    results: Sequence[ImprovementPatchCandidateVerificationCheckResult],
) -> tuple[ImprovementPatchCandidateVerificationCheckResult, ...]:
    items = tuple(results)
    if not items:
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationResult.results must not be empty"
        )
    duplicates = _duplicates(tuple(result.check_id for result in items))
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationResult.results contains duplicate "
            f"check ids: {joined}"
        )
    return items


def _validate_verification_result_for_request(
    request: ImprovementPatchCandidateVerificationRequest,
    results: Sequence[ImprovementPatchCandidateVerificationCheckResult],
) -> None:
    result_items = _verification_check_result_tuple(results)
    expected_check_ids = tuple(check.check_id for check in request.checks)
    expected_set = set(expected_check_ids)
    returned_set = {result.check_id for result in result_items}
    unknown = tuple(
        result.check_id for result in result_items if result.check_id not in expected_set
    )
    if unknown:
        joined = ", ".join(unknown)
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationResult.results contains unknown "
            f"check ids: {joined}"
        )
    missing = tuple(check_id for check_id in expected_check_ids if check_id not in returned_set)
    if missing:
        joined = ", ".join(missing)
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationResult.results missing check "
            f"ids: {joined}"
        )
    allowed = set(request.allowed_changed_files)
    for result in result_items:
        outside = tuple(item for item in result.changed_files if item not in allowed)
        if outside:
            joined = ", ".join(outside)
            raise ImprovementPatchCandidateVerificationValidationError(
                "ImprovementPatchCandidateVerificationResult.changed_files outside "
                f"allowed_changed_files: {joined}"
            )


def _validate_approval(
    approval: ImprovementPatchCandidateVerificationApproval,
) -> None:
    if not approval.approved:
        raise ImprovementPatchCandidateVerificationValidationError(
            "ImprovementPatchCandidateVerificationApproval.approved must be true"
        )
    _exact_local_verification_permission_tuple(
        approval.approved_permissions,
        "ImprovementPatchCandidateVerificationApproval.approved_permissions",
    )


def _exact_local_verification_permission_tuple(
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
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not be empty"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} contains duplicates: {joined}"
        )
    item_set = frozenset(items)
    extra = item_set.difference(_LOCAL_VERIFICATION_PERMISSION_SET)
    if extra:
        joined = ", ".join(sorted(extra))
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} contains unsupported verification permissions: {joined}"
        )
    missing = _LOCAL_VERIFICATION_PERMISSION_SET.difference(item_set)
    if missing:
        joined = ", ".join(
            permission
            for permission in _LOCAL_VERIFICATION_PERMISSIONS
            if permission in missing
        )
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} missing required verification permissions: {joined}"
        )
    return _LOCAL_VERIFICATION_PERMISSIONS


def _verification_request_from_record(
    record: ImprovementPatchCandidateVerificationRecord,
) -> ImprovementPatchCandidateVerificationRequest:
    return ImprovementPatchCandidateVerificationRequest(
        verification_plan_id=record.verification_plan_id,
        materialization_id=record.materialization_id,
        allocation_id=record.allocation_id,
        candidate_id=record.candidate_id,
        run_id=record.run_id,
        proposal_id=record.proposal_id,
        gate_evaluation_id=record.gate_evaluation_id,
        base_revision=record.base_revision,
        worktree_path=record.worktree_path,
        worktree_slug=record.worktree_slug,
        patch_digest=record.patch_digest,
        allowed_changed_files=record.allowed_changed_files,
        checks=record.checks,
        metadata=_identity_metadata(record.metadata),
    )


def _verification_digest(
    request: ImprovementPatchCandidateVerificationRequest,
    *,
    runner_ref: str,
    runner_kind: str,
    results: Sequence[ImprovementPatchCandidateVerificationCheckResult],
) -> str:
    results_by_id = {result.check_id: result for result in results}
    payload = {
        "request": improvement_patch_candidate_verification_request_to_dict(request),
        "runner_ref": runner_ref,
        "runner_kind": runner_kind,
        "results": [
            _verification_result_identity(results_by_id[check.check_id])
            for check in request.checks
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _verification_result_identity(
    result: ImprovementPatchCandidateVerificationCheckResult,
) -> dict[str, object]:
    return {
        "check_id": result.check_id,
        "status": result.status,
        "changed_files": list(result.changed_files),
        "output_reference": result.output_reference,
    }


def _request_metadata(
    plan_metadata: Mapping[str, FrozenJson],
    metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    plan_data = _metadata_to_dict(plan_metadata)
    caller_data = _metadata_to_dict(metadata)
    _reject_reserved_request_metadata_key(
        plan_data,
        "ImprovementPatchCandidateVerificationPlan.metadata",
    )
    _reject_reserved_request_metadata_key(
        caller_data,
        "ImprovementPatchCandidateVerificationService.metadata",
    )
    combined = dict(plan_data)
    combined.update(caller_data)
    return _freeze_metadata(combined)


def _record_metadata(
    request_metadata: Mapping[str, FrozenJson],
    result_metadata: Mapping[str, FrozenJson],
) -> Mapping[str, FrozenJson]:
    combined = _metadata_to_dict(request_metadata)
    result_data = _metadata_to_dict(result_metadata)
    if result_data:
        combined[_VERIFICATION_RESULT_METADATA_KEY] = result_data
    return _freeze_metadata(combined)


def _identity_metadata(metadata: Mapping[str, FrozenJson]) -> Mapping[str, FrozenJson]:
    identity = {
        key: value
        for key, value in _metadata_to_dict(metadata).items()
        if key != _VERIFICATION_RESULT_METADATA_KEY
    }
    return _freeze_metadata(identity)


def _reject_reserved_request_metadata_key(
    metadata: Mapping[str, object],
    name: str,
) -> None:
    if _VERIFICATION_RESULT_METADATA_KEY in metadata:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} cannot include reserved {_VERIFICATION_RESULT_METADATA_KEY}"
        )


def _reject_reserved_evaluation_metadata_key(
    metadata: Mapping[str, object],
    name: str,
) -> None:
    reserved = tuple(
        key for key in metadata if key in _VERIFICATION_EVALUATION_METADATA_KEYS
    )
    if reserved:
        joined = ", ".join(reserved)
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} cannot include reserved verification metadata keys: {joined}"
        )


def _require_verification_reference(
    value: str,
    name: str,
    max_chars: int,
    max_chars_name: str,
) -> None:
    _require_bounded_non_empty(value, name, max_chars, max_chars_name)
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must be a single-line verification reference"
        )
    stripped = value.strip()
    if any(marker in stripped for marker in _RAW_REFERENCE_MARKERS):
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not contain raw output or diff payloads"
        )
    if stripped.startswith(_COPIED_FILE_PREFIXES):
        raise ImprovementPatchCandidateVerificationValidationError(
            f"{name} must not look like copied file contents"
        )


def _changed_file_tuple_or_empty(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_normalize_relative_path(value, name) for value in values)
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


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _evaluation_result_metadata(
    record: ImprovementPatchCandidateVerificationRecord,
    check: ImprovementPatchCandidateVerificationCheck,
) -> Mapping[str, FrozenJson]:
    return _freeze_metadata(
        {
            "verification_id": record.verification_id,
            "verification_digest": record.verification_digest,
            "verification_check_id": check.check_id,
            "source_plan_section": check.source_plan_section,
        }
    )


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
    "ImprovementPatchCandidateVerificationApproval",
    "ImprovementPatchCandidateVerificationCheck",
    "ImprovementPatchCandidateVerificationCheckResult",
    "ImprovementPatchCandidateVerificationCheckSource",
    "ImprovementPatchCandidateVerificationError",
    "ImprovementPatchCandidateVerificationPlan",
    "ImprovementPatchCandidateVerificationPlanStatus",
    "ImprovementPatchCandidateVerificationPlanner",
    "ImprovementPatchCandidateVerificationRecord",
    "ImprovementPatchCandidateVerificationRequest",
    "ImprovementPatchCandidateVerificationResult",
    "ImprovementPatchCandidateVerificationService",
    "ImprovementPatchCandidateVerificationStatus",
    "ImprovementPatchCandidateVerificationValidationError",
    "ImprovementPatchCandidateVerifier",
    "improvement_patch_candidate_verification_check_from_dict",
    "improvement_patch_candidate_verification_check_result_from_dict",
    "improvement_patch_candidate_verification_check_result_to_dict",
    "improvement_patch_candidate_verification_check_to_dict",
    "improvement_patch_candidate_verification_plan_from_dict",
    "improvement_patch_candidate_verification_plan_to_dict",
    "improvement_patch_candidate_verification_record_from_dict",
    "improvement_patch_candidate_verification_record_to_dict",
    "improvement_patch_candidate_verification_record_to_evaluation",
    "improvement_patch_candidate_verification_request_from_dict",
    "improvement_patch_candidate_verification_request_to_dict",
    "improvement_patch_candidate_verification_result_from_dict",
    "improvement_patch_candidate_verification_result_to_dict",
)
