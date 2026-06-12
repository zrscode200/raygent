"""Materialization records for isolated improvement patch candidates.

The RSI-003C layer invokes a caller-owned materializer through a narrow
protocol. It records materialization and supplied evaluation data, but it does
not implement a filesystem writer, execute commands, commit, promote, or
archive candidates.
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
from raygent_harness.improvement.candidate_worktree import (
    ImprovementPatchCandidateWorktreeAllocation,
)
from raygent_harness.improvement.candidates import ImprovementPatchCandidatePlan
from raygent_harness.improvement.models import ImprovementRequiredPermission

DEFAULT_MAX_MATERIALIZATION_OPERATIONS = 20
DEFAULT_MAX_MATERIALIZATION_PATH_CHARS = 240
DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS = 200_000
DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES = 50
DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS = 20_000
DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS = 20_000

ImprovementPatchOperationKind = Literal["create_file", "replace_text"]
ImprovementPatchCandidateMaterializationStatus = Literal["materialized"]
ImprovementPatchCandidateEvaluationKind = Literal[
    "static_review",
    "non_regression",
    "unit_tests",
    "manual_review",
    "other",
]
ImprovementPatchCandidateEvaluationStatus = Literal[
    "pass",
    "warn",
    "fail",
    "needs_review",
    "not_applicable",
]
ImprovementPatchCandidateEvaluationDecision = Literal[
    "pass",
    "warn",
    "fail",
    "needs_review",
]

_OPERATION_KINDS: frozenset[str] = frozenset({"create_file", "replace_text"})
_MATERIALIZATION_STATUSES: frozenset[str] = frozenset({"materialized"})
_EVALUATION_KINDS: frozenset[str] = frozenset(
    {"static_review", "non_regression", "unit_tests", "manual_review", "other"}
)
_EVALUATION_STATUSES: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "needs_review", "not_applicable"}
)
_EVALUATION_DECISIONS: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "needs_review"}
)
_REQUIRED_APPROVAL_PERMISSIONS: frozenset[str] = frozenset({"filesystem_mutation"})
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


class ImprovementPatchCandidateMaterializationError(ValueError):
    """Raised when an improvement patch candidate cannot be materialized."""


class ImprovementPatchCandidateMaterializationValidationError(
    ImprovementPatchCandidateMaterializationError
):
    """Raised when materialization data violates the RSI-003C contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateMaterializationApproval:
    """Call-time authority to materialize one candidate in an allocated worktree."""

    approved_permissions: tuple[ImprovementRequiredPermission, ...]
    reason: str
    approved_by: str | None = None
    approved: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.approved:
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchCandidateMaterializationApproval.approved must be true"
            )
        object.__setattr__(
            self,
            "approved_permissions",
            _approval_permission_tuple(self.approved_permissions),
        )
        _require_non_empty(
            self.reason,
            "ImprovementPatchCandidateMaterializationApproval.reason",
        )
        if self.approved_by is not None:
            _require_non_empty(
                self.approved_by,
                "ImprovementPatchCandidateMaterializationApproval.approved_by",
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateMaterializationApproval.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchOperation:
    """Bounded create/replace operation for an injected materializer."""

    operation_id: str
    kind: ImprovementPatchOperationKind
    relative_path: str
    new_text: str
    old_text: str | None = None
    replace_all: bool = False
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.operation_id, "ImprovementPatchOperation.operation_id")
        _require_literal(self.kind, _OPERATION_KINDS, "ImprovementPatchOperation.kind")
        object.__setattr__(
            self,
            "relative_path",
            _normalize_relative_path(self.relative_path, "relative_path"),
        )
        if self.kind == "replace_text":
            if self.old_text is None:
                raise ImprovementPatchCandidateMaterializationValidationError(
                    "ImprovementPatchOperation.old_text is required for replace_text"
                )
            _require_non_empty(self.old_text, "ImprovementPatchOperation.old_text")
        elif self.old_text is not None:
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchOperation.old_text must be omitted for create_file"
            )
        _require_text_bound(self.new_text, "ImprovementPatchOperation.new_text")
        if self.old_text is not None:
            _require_text_bound(self.old_text, "ImprovementPatchOperation.old_text")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchMaterializationRequest:
    """Normalized request passed to an injected materializer."""

    worktree_path: str
    operations: tuple[ImprovementPatchOperation, ...]
    expected_files: tuple[str, ...]
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.worktree_path,
            "ImprovementPatchMaterializationRequest.worktree_path",
        )
        object.__setattr__(
            self,
            "operations",
            _operation_tuple(self.operations),
        )
        expected_files = _normalized_path_tuple(self.expected_files, "expected_files")
        object.__setattr__(self, "expected_files", expected_files)
        _validate_operation_paths(self.operations, expected_files)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchMaterializationResult:
    """Result returned by a caller-owned materializer."""

    changed_files: tuple[str, ...]
    summary: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        changed_files = _changed_file_tuple(self.changed_files, "changed_files")
        if not changed_files:
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchMaterializationResult.changed_files must not be empty"
            )
        object.__setattr__(self, "changed_files", changed_files)
        if self.summary is not None:
            _require_non_empty(self.summary, "ImprovementPatchMaterializationResult.summary")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ImprovementPatchMaterializer(Protocol):
    """Caller-owned materializer seam for RSI-003C."""

    async def materialize(
        self,
        request: ImprovementPatchMaterializationRequest,
    ) -> ImprovementPatchMaterializationResult:
        """Materialize the supplied operations in the allocated worktree."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateMaterialization:
    """Serializable record for one materialized patch candidate."""

    materialization_id: str
    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    base_revision: str
    worktree_path: str
    worktree_slug: str
    operations: tuple[ImprovementPatchOperation, ...]
    changed_files: tuple[str, ...]
    patch_digest: str
    status: ImprovementPatchCandidateMaterializationStatus = "materialized"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.materialization_id,
            "ImprovementPatchCandidateMaterialization.materialization_id",
        )
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateMaterialization.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateMaterialization.candidate_id",
        )
        _require_non_empty(self.run_id, "ImprovementPatchCandidateMaterialization.run_id")
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateMaterialization.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateMaterialization.gate_evaluation_id",
        )
        _require_non_empty(
            self.base_revision,
            "ImprovementPatchCandidateMaterialization.base_revision",
        )
        _require_non_empty(
            self.worktree_path,
            "ImprovementPatchCandidateMaterialization.worktree_path",
        )
        _require_non_empty(
            self.worktree_slug,
            "ImprovementPatchCandidateMaterialization.worktree_slug",
        )
        operations = _operation_tuple(self.operations)
        _validate_operation_ids(operations)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "changed_files",
            _changed_file_tuple(self.changed_files, "changed_files"),
        )
        _require_non_empty(
            self.patch_digest,
            "ImprovementPatchCandidateMaterialization.patch_digest",
        )
        _require_literal(
            self.status,
            _MATERIALIZATION_STATUSES,
            "ImprovementPatchCandidateMaterialization.status",
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateMaterialization.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateMaterializer:
    """Validate and materialize one planned candidate in an allocated worktree."""

    clock: Callable[[], float] = time.time
    materialization_id_factory: Callable[[], str] | None = None

    async def materialize(
        self,
        plan: ImprovementPatchCandidatePlan,
        allocation: ImprovementPatchCandidateWorktreeAllocation,
        *,
        operations: Sequence[ImprovementPatchOperation],
        materializer: ImprovementPatchMaterializer | None,
        approval: ImprovementPatchCandidateMaterializationApproval | None,
        materialization_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateMaterialization:
        """Invoke an injected materializer and record the resulting patch data."""

        _validate_plan_allocation_linkage(plan, allocation)
        if materializer is None:
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchCandidateMaterializer requires an injected "
                "ImprovementPatchMaterializer"
            )
        if approval is None:
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchCandidateMaterializer requires explicit call-time "
                "approval"
            )
        _validate_approval(approval)

        expected_files = _normalized_path_tuple(plan.expected_files, "expected_files")
        operation_tuple = _operation_tuple(operations)
        _validate_operation_ids(operation_tuple)
        _validate_operation_paths(operation_tuple, expected_files)
        request = ImprovementPatchMaterializationRequest(
            worktree_path=allocation.worktree_path,
            operations=operation_tuple,
            expected_files=expected_files,
        )
        result = await materializer.materialize(request)
        _validate_changed_files(result.changed_files, expected_files)
        patch_digest = _patch_digest(operation_tuple, result.changed_files)
        combined_metadata = _materialization_metadata(metadata or {}, result)

        return ImprovementPatchCandidateMaterialization(
            materialization_id=materialization_id or self._new_materialization_id(),
            allocation_id=allocation.allocation_id,
            candidate_id=plan.candidate_id,
            run_id=plan.run_id,
            proposal_id=plan.proposal_id,
            gate_evaluation_id=plan.gate_evaluation_id,
            base_revision=plan.base_revision,
            worktree_path=allocation.worktree_path,
            worktree_slug=allocation.worktree_slug,
            operations=operation_tuple,
            changed_files=result.changed_files,
            patch_digest=patch_digest,
            status="materialized",
            created_at=self.clock(),
            metadata=combined_metadata,
        )

    def _new_materialization_id(self) -> str:
        materialization_id = (
            self.materialization_id_factory()
            if self.materialization_id_factory is not None
            else f"ipcm_{uuid4().hex}"
        )
        if not materialization_id.strip():
            raise ImprovementPatchCandidateMaterializationValidationError(
                "materialization_id_factory returned an empty id"
            )
        return materialization_id


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateEvaluationResult:
    """One caller-supplied evaluation result for a materialized candidate."""

    result_id: str
    kind: ImprovementPatchCandidateEvaluationKind
    status: ImprovementPatchCandidateEvaluationStatus
    summary: str
    changed_files: tuple[str, ...] = ()
    output_excerpt: str | None = None
    output_reference: str | None = None
    required: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.result_id,
            "ImprovementPatchCandidateEvaluationResult.result_id",
        )
        _require_literal(
            self.kind,
            _EVALUATION_KINDS,
            "ImprovementPatchCandidateEvaluationResult.kind",
        )
        _require_literal(
            self.status,
            _EVALUATION_STATUSES,
            "ImprovementPatchCandidateEvaluationResult.status",
        )
        _require_non_empty(
            self.summary,
            "ImprovementPatchCandidateEvaluationResult.summary",
        )
        object.__setattr__(
            self,
            "changed_files",
            _changed_file_tuple(self.changed_files, "changed_files"),
        )
        if self.output_excerpt is not None:
            _require_non_empty(
                self.output_excerpt,
                "ImprovementPatchCandidateEvaluationResult.output_excerpt",
            )
            if len(self.output_excerpt) > DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS:
                raise ImprovementPatchCandidateMaterializationValidationError(
                    "ImprovementPatchCandidateEvaluationResult.output_excerpt "
                    "exceeds DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS"
                )
        if self.output_reference is not None:
            _require_non_empty(
                self.output_reference,
                "ImprovementPatchCandidateEvaluationResult.output_reference",
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateEvaluationResult.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateEvaluation:
    """Aggregate supplied-result evaluation for one materialized candidate."""

    evaluation_id: str
    materialization_id: str
    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    results: tuple[ImprovementPatchCandidateEvaluationResult, ...]
    decision: ImprovementPatchCandidateEvaluationDecision = field(init=False)
    warnings: tuple[str, ...] = field(init=False)
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.evaluation_id,
            "ImprovementPatchCandidateEvaluation.evaluation_id",
        )
        _require_non_empty(
            self.materialization_id,
            "ImprovementPatchCandidateEvaluation.materialization_id",
        )
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateEvaluation.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateEvaluation.candidate_id",
        )
        _require_non_empty(self.run_id, "ImprovementPatchCandidateEvaluation.run_id")
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateEvaluation.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateEvaluation.gate_evaluation_id",
        )
        object.__setattr__(self, "results", tuple(self.results))
        _validate_unique_result_ids(self.results)
        decision, warnings = _derive_evaluation_decision(self.results)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "warnings", warnings)
        if self.created_at < 0:
            raise ValueError("ImprovementPatchCandidateEvaluation.created_at must be >= 0")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def improvement_patch_operation_to_dict(
    operation: ImprovementPatchOperation,
) -> dict[str, object]:
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "relative_path": operation.relative_path,
        "old_text": operation.old_text,
        "new_text": operation.new_text,
        "replace_all": operation.replace_all,
        "metadata": _metadata_to_dict(operation.metadata),
    }


def improvement_patch_operation_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id=str(data["operation_id"]),
        kind=cast(
            ImprovementPatchOperationKind,
            _literal_from_object(data["kind"], _OPERATION_KINDS, "kind"),
        ),
        relative_path=str(data["relative_path"]),
        old_text=_optional_string(data.get("old_text"), "old_text"),
        new_text=str(data["new_text"]),
        replace_all=bool(data.get("replace_all", False)),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_materialization_to_dict(
    materialization: ImprovementPatchCandidateMaterialization,
) -> dict[str, object]:
    return {
        "materialization_id": materialization.materialization_id,
        "allocation_id": materialization.allocation_id,
        "candidate_id": materialization.candidate_id,
        "run_id": materialization.run_id,
        "proposal_id": materialization.proposal_id,
        "gate_evaluation_id": materialization.gate_evaluation_id,
        "base_revision": materialization.base_revision,
        "worktree_path": materialization.worktree_path,
        "worktree_slug": materialization.worktree_slug,
        "operations": [
            improvement_patch_operation_to_dict(operation)
            for operation in materialization.operations
        ],
        "changed_files": list(materialization.changed_files),
        "patch_digest": materialization.patch_digest,
        "status": materialization.status,
        "created_at": materialization.created_at,
        "metadata": _metadata_to_dict(materialization.metadata),
    }


def improvement_patch_candidate_materialization_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateMaterialization:
    return ImprovementPatchCandidateMaterialization(
        materialization_id=str(data["materialization_id"]),
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        base_revision=str(data["base_revision"]),
        worktree_path=str(data["worktree_path"]),
        worktree_slug=str(data["worktree_slug"]),
        operations=tuple(
            improvement_patch_operation_from_dict(item)
            for item in _mapping_sequence(data.get("operations", ()), "operations")
        ),
        changed_files=_string_tuple(data.get("changed_files", ()), "changed_files"),
        patch_digest=str(data["patch_digest"]),
        status=cast(
            ImprovementPatchCandidateMaterializationStatus,
            _literal_from_object(
                data.get("status", "materialized"),
                _MATERIALIZATION_STATUSES,
                "status",
            ),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_evaluation_result_to_dict(
    result: ImprovementPatchCandidateEvaluationResult,
) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "kind": result.kind,
        "status": result.status,
        "summary": result.summary,
        "changed_files": list(result.changed_files),
        "output_excerpt": result.output_excerpt,
        "output_reference": result.output_reference,
        "required": result.required,
        "created_at": result.created_at,
        "metadata": _metadata_to_dict(result.metadata),
    }


def improvement_patch_candidate_evaluation_result_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateEvaluationResult:
    return ImprovementPatchCandidateEvaluationResult(
        result_id=str(data["result_id"]),
        kind=cast(
            ImprovementPatchCandidateEvaluationKind,
            _literal_from_object(data["kind"], _EVALUATION_KINDS, "kind"),
        ),
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
        required=bool(data.get("required", True)),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def improvement_patch_candidate_evaluation_to_dict(
    evaluation: ImprovementPatchCandidateEvaluation,
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation.evaluation_id,
        "materialization_id": evaluation.materialization_id,
        "allocation_id": evaluation.allocation_id,
        "candidate_id": evaluation.candidate_id,
        "run_id": evaluation.run_id,
        "proposal_id": evaluation.proposal_id,
        "gate_evaluation_id": evaluation.gate_evaluation_id,
        "decision": evaluation.decision,
        "results": [
            improvement_patch_candidate_evaluation_result_to_dict(result)
            for result in evaluation.results
        ],
        "warnings": list(evaluation.warnings),
        "created_at": evaluation.created_at,
        "metadata": _metadata_to_dict(evaluation.metadata),
    }


def improvement_patch_candidate_evaluation_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateEvaluation:
    claimed_decision = cast(
        ImprovementPatchCandidateEvaluationDecision,
        _literal_from_object(data["decision"], _EVALUATION_DECISIONS, "decision"),
    )
    claimed_warnings = _string_tuple(data.get("warnings", ()), "warnings")
    evaluation = ImprovementPatchCandidateEvaluation(
        evaluation_id=str(data["evaluation_id"]),
        materialization_id=str(data["materialization_id"]),
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        results=tuple(
            improvement_patch_candidate_evaluation_result_from_dict(item)
            for item in _mapping_sequence(data.get("results", ()), "results")
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )
    if evaluation.decision != claimed_decision:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateEvaluation.decision does not match "
            "derived evaluation policy"
        )
    if evaluation.warnings != claimed_warnings:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateEvaluation.warnings do not match "
            "derived evaluation policy"
        )
    return evaluation


def _validate_plan_allocation_linkage(
    plan: ImprovementPatchCandidatePlan,
    allocation: ImprovementPatchCandidateWorktreeAllocation,
) -> None:
    if plan.status != "planned":
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializer requires a planned candidate"
        )
    if allocation.status != "allocated":
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializer requires an allocated worktree"
        )
    fields = (
        "candidate_id",
        "run_id",
        "proposal_id",
        "gate_evaluation_id",
        "base_revision",
    )
    for field_name in fields:
        if getattr(plan, field_name) != getattr(allocation, field_name):
            raise ImprovementPatchCandidateMaterializationValidationError(
                "ImprovementPatchCandidateMaterializer linkage mismatch: "
                f"{field_name}"
            )


def _validate_approval(
    approval: ImprovementPatchCandidateMaterializationApproval,
) -> None:
    if not approval.approved:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializationApproval.approved must be true"
        )
    missing = _REQUIRED_APPROVAL_PERMISSIONS.difference(approval.approved_permissions)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializationApproval.approved_permissions "
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
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializationApproval.approved_permissions "
            "must not be empty"
        )
    if "none" in items:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializationApproval.approved_permissions "
            "cannot include none"
        )
    missing = _REQUIRED_APPROVAL_PERMISSIONS.difference(items)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializationApproval.approved_permissions "
            f"missing: {joined}"
        )
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _operation_tuple(
    operations: Sequence[ImprovementPatchOperation],
) -> tuple[ImprovementPatchOperation, ...]:
    items = tuple(operations)
    if not items:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializer.operations must not be empty"
        )
    if len(items) > DEFAULT_MAX_MATERIALIZATION_OPERATIONS:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializer.operations exceeds "
            "DEFAULT_MAX_MATERIALIZATION_OPERATIONS"
        )
    return items


def _validate_operation_ids(operations: Sequence[ImprovementPatchOperation]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for operation in operations:
        if operation.operation_id in seen:
            duplicates.append(operation.operation_id)
        seen.add(operation.operation_id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateMaterializer.operations contains duplicate "
            f"operation ids: {joined}"
        )


def _validate_operation_paths(
    operations: Sequence[ImprovementPatchOperation],
    expected_files: Sequence[str],
) -> None:
    expected = set(expected_files)
    outside = tuple(
        operation.relative_path
        for operation in operations
        if operation.relative_path not in expected
    )
    if outside:
        joined = ", ".join(dict.fromkeys(outside))
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchOperation.relative_path outside expected files: "
            f"{joined}"
        )
    create_paths: set[str] = set()
    duplicate_creates: list[str] = []
    for operation in operations:
        if operation.kind != "create_file":
            continue
        if operation.relative_path in create_paths:
            duplicate_creates.append(operation.relative_path)
        create_paths.add(operation.relative_path)
    if duplicate_creates:
        joined = ", ".join(dict.fromkeys(duplicate_creates))
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchOperation.create_file duplicates relative paths: "
            f"{joined}"
        )


def _validate_changed_files(
    changed_files: Sequence[str],
    expected_files: Sequence[str],
) -> None:
    items = _changed_file_tuple(changed_files, "changed_files")
    expected = set(expected_files)
    outside = tuple(item for item in items if item not in expected)
    if outside:
        joined = ", ".join(dict.fromkeys(outside))
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchMaterializationResult.changed_files outside expected "
            f"files: {joined}"
        )


def _derive_evaluation_decision(
    results: tuple[ImprovementPatchCandidateEvaluationResult, ...],
) -> tuple[ImprovementPatchCandidateEvaluationDecision, tuple[str, ...]]:
    warnings: list[str] = []
    required_statuses = {result.status for result in results if result.required}
    all_statuses = {result.status for result in results}

    if not results:
        warnings.append("no evaluation results supplied")
        return "needs_review", tuple(warnings)
    if "fail" in required_statuses:
        return "fail", tuple(warnings)
    if "needs_review" in all_statuses:
        return "needs_review", tuple(warnings)
    optional_failures = tuple(
        result for result in results if not result.required and result.status == "fail"
    )
    if optional_failures:
        warnings.append("one or more optional evaluation results failed")
    if "warn" in all_statuses or optional_failures:
        return "warn", tuple(warnings)
    return "pass", tuple(warnings)


def _validate_unique_result_ids(
    results: Sequence[ImprovementPatchCandidateEvaluationResult],
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for result in results:
        if result.result_id in seen:
            duplicates.append(result.result_id)
        seen.add(result.result_id)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateMaterializationValidationError(
            "ImprovementPatchCandidateEvaluation.results contains duplicate "
            f"result ids: {joined}"
        )


def _patch_digest(
    operations: Sequence[ImprovementPatchOperation],
    changed_files: Sequence[str],
) -> str:
    payload = {
        "changed_files": list(changed_files),
        "operations": [
            improvement_patch_operation_to_dict(operation) for operation in operations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _materialization_metadata(
    metadata: Mapping[str, FrozenJson],
    result: ImprovementPatchMaterializationResult,
) -> Mapping[str, FrozenJson]:
    combined: dict[str, object] = _metadata_to_dict(metadata)
    if result.summary is not None:
        combined["materializer_summary"] = result.summary
    result_metadata = _metadata_to_dict(result.metadata)
    if result_metadata:
        combined["materializer_metadata"] = result_metadata
    return _freeze_metadata(combined)


def _normalized_path_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_normalize_relative_path(value, name) for value in values)
    if not items:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must not be empty"
        )
    duplicates = _duplicates(items)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} contains duplicate normalized paths: {joined}"
        )
    return items


def _changed_file_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_normalize_relative_path(value, name) for value in values)
    if len(items) > DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} exceeds DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES"
        )
    return items


def _normalize_relative_path(value: str, name: str) -> str:
    _require_non_empty(value, name)
    if "\x00" in value:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must not contain NUL bytes"
        )
    if "\\" in value:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must use POSIX-style separators"
        )
    if value.startswith("/"):
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must be relative"
        )
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must not normalize to an empty path"
        )
    if normalized == ".." or normalized.startswith("../"):
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} must not contain parent traversal"
        )
    if len(normalized) > DEFAULT_MAX_MATERIALIZATION_PATH_CHARS:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} exceeds DEFAULT_MAX_MATERIALIZATION_PATH_CHARS"
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


def _require_text_bound(value: str, name: str) -> None:
    if len(value) > DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS:
        raise ImprovementPatchCandidateMaterializationValidationError(
            f"{name} exceeds DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS"
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
    if len(encoded) > DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS:
        raise ImprovementPatchCandidateMaterializationValidationError(
            "metadata exceeds DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS"
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


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    _require_non_empty(text, name)
    return text


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a number") from exc


__all__ = (
    "DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS",
    "DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES",
    "DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS",
    "DEFAULT_MAX_MATERIALIZATION_OPERATIONS",
    "DEFAULT_MAX_MATERIALIZATION_PATH_CHARS",
    "DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS",
    "ImprovementPatchCandidateEvaluation",
    "ImprovementPatchCandidateEvaluationDecision",
    "ImprovementPatchCandidateEvaluationKind",
    "ImprovementPatchCandidateEvaluationResult",
    "ImprovementPatchCandidateEvaluationStatus",
    "ImprovementPatchCandidateMaterialization",
    "ImprovementPatchCandidateMaterializationApproval",
    "ImprovementPatchCandidateMaterializationError",
    "ImprovementPatchCandidateMaterializationStatus",
    "ImprovementPatchCandidateMaterializationValidationError",
    "ImprovementPatchCandidateMaterializer",
    "ImprovementPatchMaterializationRequest",
    "ImprovementPatchMaterializationResult",
    "ImprovementPatchMaterializer",
    "ImprovementPatchOperation",
    "ImprovementPatchOperationKind",
    "improvement_patch_candidate_evaluation_from_dict",
    "improvement_patch_candidate_evaluation_result_from_dict",
    "improvement_patch_candidate_evaluation_result_to_dict",
    "improvement_patch_candidate_evaluation_to_dict",
    "improvement_patch_candidate_materialization_from_dict",
    "improvement_patch_candidate_materialization_to_dict",
    "improvement_patch_operation_from_dict",
    "improvement_patch_operation_to_dict",
)
