"""Isolated worktree allocation for reviewed patch candidate plans."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.candidates import ImprovementPatchCandidatePlan
from raygent_harness.improvement.models import ImprovementRequiredPermission
from raygent_harness.services.worktree.manager import WorktreeManager
from raygent_harness.services.worktree.models import (
    WorktreeCleanupPolicy,
    WorktreeInfo,
)

ImprovementPatchCandidateWorktreeStatus = Literal["allocated"]

_ALLOCATION_STATUSES: frozenset[str] = frozenset({"allocated"})
_REQUIRED_APPROVAL_PERMISSIONS: frozenset[str] = frozenset(
    {"filesystem_mutation", "worktree"}
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
_CLEANUP_POLICIES: frozenset[str] = frozenset({"remove_if_clean", "keep", "manual"})
_SAFE_ALLOCATION_SLUG = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_UNSAFE_SLUG_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class ImprovementPatchCandidateWorktreeError(ValueError):
    """Raised when a patch candidate worktree cannot be allocated."""


class ImprovementPatchCandidateWorktreeValidationError(
    ImprovementPatchCandidateWorktreeError
):
    """Raised when worktree allocation data violates the RSI-003B contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateWorktreeApproval:
    """Call-time authority to allocate one candidate worktree."""

    approved_permissions: tuple[ImprovementRequiredPermission, ...]
    reason: str
    approved_by: str | None = None
    approved: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.approved:
            raise ImprovementPatchCandidateWorktreeValidationError(
                "ImprovementPatchCandidateWorktreeApproval.approved must be true"
            )
        object.__setattr__(
            self,
            "approved_permissions",
            _approval_permission_tuple(self.approved_permissions),
        )
        _require_non_empty(
            self.reason,
            "ImprovementPatchCandidateWorktreeApproval.reason",
        )
        if self.approved_by is not None:
            _require_non_empty(
                self.approved_by,
                "ImprovementPatchCandidateWorktreeApproval.approved_by",
            )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateWorktreeApproval.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateWorktreeAllocation:
    """Serializable record for one allocated candidate worktree."""

    allocation_id: str
    candidate_id: str
    run_id: str
    proposal_id: str
    gate_evaluation_id: str
    base_revision: str
    worktree_path: str
    worktree_branch: str | None
    worktree_slug: str
    worktree_head_commit: str | None
    git_root: str | None
    cleanup_policy: WorktreeCleanupPolicy
    worktree_created_at: float | None = None
    worktree_touched_at: float | None = None
    status: ImprovementPatchCandidateWorktreeStatus = "allocated"
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(
            self.allocation_id,
            "ImprovementPatchCandidateWorktreeAllocation.allocation_id",
        )
        _require_non_empty(
            self.candidate_id,
            "ImprovementPatchCandidateWorktreeAllocation.candidate_id",
        )
        _require_non_empty(
            self.run_id,
            "ImprovementPatchCandidateWorktreeAllocation.run_id",
        )
        _require_non_empty(
            self.proposal_id,
            "ImprovementPatchCandidateWorktreeAllocation.proposal_id",
        )
        _require_non_empty(
            self.gate_evaluation_id,
            "ImprovementPatchCandidateWorktreeAllocation.gate_evaluation_id",
        )
        _require_non_empty(
            self.base_revision,
            "ImprovementPatchCandidateWorktreeAllocation.base_revision",
        )
        _require_non_empty(
            self.worktree_path,
            "ImprovementPatchCandidateWorktreeAllocation.worktree_path",
        )
        if self.worktree_branch is not None:
            _require_non_empty(
                self.worktree_branch,
                "ImprovementPatchCandidateWorktreeAllocation.worktree_branch",
            )
        _require_safe_slug(self.worktree_slug, "worktree_slug")
        if self.worktree_head_commit is not None:
            _require_non_empty(
                self.worktree_head_commit,
                "ImprovementPatchCandidateWorktreeAllocation.worktree_head_commit",
            )
            if self.worktree_head_commit != self.base_revision:
                raise ImprovementPatchCandidateWorktreeValidationError(
                    "ImprovementPatchCandidateWorktreeAllocation.worktree_head_commit "
                    "conflicts with base_revision"
                )
        if self.git_root is not None:
            _require_non_empty(
                self.git_root,
                "ImprovementPatchCandidateWorktreeAllocation.git_root",
            )
        _require_literal(
            self.cleanup_policy,
            _CLEANUP_POLICIES,
            "ImprovementPatchCandidateWorktreeAllocation.cleanup_policy",
        )
        _require_literal(
            self.status,
            _ALLOCATION_STATUSES,
            "ImprovementPatchCandidateWorktreeAllocation.status",
        )
        _require_optional_non_negative(
            self.worktree_created_at,
            "ImprovementPatchCandidateWorktreeAllocation.worktree_created_at",
        )
        _require_optional_non_negative(
            self.worktree_touched_at,
            "ImprovementPatchCandidateWorktreeAllocation.worktree_touched_at",
        )
        if self.created_at < 0:
            raise ValueError(
                "ImprovementPatchCandidateWorktreeAllocation.created_at must be >= 0"
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ImprovementPatchCandidateWorktreeAllocator:
    """Allocate one isolated worktree for a reviewed patch candidate plan."""

    clock: Callable[[], float] = time.time
    allocation_id_factory: Callable[[], str] | None = None

    async def allocate(
        self,
        plan: ImprovementPatchCandidatePlan,
        *,
        manager: WorktreeManager | None,
        cwd: str,
        approval: ImprovementPatchCandidateWorktreeApproval | None,
        allocation_id: str | None = None,
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> ImprovementPatchCandidateWorktreeAllocation:
        """Create an isolated worktree allocation record, then stop."""

        _validate_plan(plan)
        if manager is None:
            raise ImprovementPatchCandidateWorktreeValidationError(
                "ImprovementPatchCandidateWorktreeAllocator requires an injected "
                "WorktreeManager"
            )
        if approval is None:
            raise ImprovementPatchCandidateWorktreeValidationError(
                "ImprovementPatchCandidateWorktreeAllocator requires explicit "
                "call-time approval"
            )
        _validate_approval(approval)
        _require_non_empty(cwd, "cwd")

        slug = _candidate_worktree_slug(plan.candidate_id)
        info = await manager.create_agent_worktree(slug, cwd=cwd)
        if not info.path.strip():
            detail = await _cleanup_after_failed_create(manager, info)
            raise ImprovementPatchCandidateWorktreeValidationError(
                _with_cleanup_detail("WorktreeInfo.path must be non-empty", detail)
            )
        if info.slug is not None and info.slug != slug:
            detail = await _cleanup_after_failed_create(manager, info)
            raise ImprovementPatchCandidateWorktreeValidationError(
                _with_cleanup_detail(
                    "WorktreeInfo.slug must match requested allocation slug",
                    detail,
                )
            )
        if info.head_commit is not None and info.head_commit != plan.base_revision:
            detail = await _cleanup_after_failed_create(manager, info)
            raise ImprovementPatchCandidateWorktreeValidationError(
                _with_cleanup_detail(
                    "WorktreeInfo.head_commit conflicts with candidate base_revision",
                    detail,
                )
            )

        try:
            return ImprovementPatchCandidateWorktreeAllocation(
                allocation_id=allocation_id or self._new_allocation_id(),
                candidate_id=plan.candidate_id,
                run_id=plan.run_id,
                proposal_id=plan.proposal_id,
                gate_evaluation_id=plan.gate_evaluation_id,
                base_revision=plan.base_revision,
                worktree_path=info.path,
                worktree_branch=info.branch,
                worktree_slug=info.slug or slug,
                worktree_head_commit=info.head_commit,
                git_root=info.git_root,
                cleanup_policy=info.cleanup_policy,
                worktree_created_at=info.created_at,
                worktree_touched_at=info.touched_at,
                status="allocated",
                created_at=self.clock(),
                metadata=metadata or {},
            )
        except Exception as exc:
            detail = await _cleanup_after_failed_create(manager, info)
            raise ImprovementPatchCandidateWorktreeValidationError(
                _with_cleanup_detail(
                    f"Failed to build worktree allocation record: {exc}",
                    detail,
                )
            ) from exc

    def _new_allocation_id(self) -> str:
        allocation_id = (
            self.allocation_id_factory()
            if self.allocation_id_factory is not None
            else f"ipcw_{uuid4().hex}"
        )
        if not allocation_id.strip():
            raise ImprovementPatchCandidateWorktreeValidationError(
                "allocation_id_factory returned an empty id"
            )
        return allocation_id


def improvement_patch_candidate_worktree_allocation_to_dict(
    allocation: ImprovementPatchCandidateWorktreeAllocation,
) -> dict[str, object]:
    return {
        "allocation_id": allocation.allocation_id,
        "candidate_id": allocation.candidate_id,
        "run_id": allocation.run_id,
        "proposal_id": allocation.proposal_id,
        "gate_evaluation_id": allocation.gate_evaluation_id,
        "base_revision": allocation.base_revision,
        "worktree_path": allocation.worktree_path,
        "worktree_branch": allocation.worktree_branch,
        "worktree_slug": allocation.worktree_slug,
        "worktree_head_commit": allocation.worktree_head_commit,
        "git_root": allocation.git_root,
        "cleanup_policy": allocation.cleanup_policy,
        "worktree_created_at": allocation.worktree_created_at,
        "worktree_touched_at": allocation.worktree_touched_at,
        "status": allocation.status,
        "created_at": allocation.created_at,
        "metadata": _metadata_to_dict(allocation.metadata),
    }


def improvement_patch_candidate_worktree_allocation_from_dict(
    data: Mapping[str, object],
) -> ImprovementPatchCandidateWorktreeAllocation:
    return ImprovementPatchCandidateWorktreeAllocation(
        allocation_id=str(data["allocation_id"]),
        candidate_id=str(data["candidate_id"]),
        run_id=str(data["run_id"]),
        proposal_id=str(data["proposal_id"]),
        gate_evaluation_id=str(data["gate_evaluation_id"]),
        base_revision=str(data["base_revision"]),
        worktree_path=str(data["worktree_path"]),
        worktree_branch=_optional_string(data.get("worktree_branch"), "worktree_branch"),
        worktree_slug=str(data["worktree_slug"]),
        worktree_head_commit=_optional_string(
            data.get("worktree_head_commit"),
            "worktree_head_commit",
        ),
        git_root=_optional_string(data.get("git_root"), "git_root"),
        cleanup_policy=cast(
            WorktreeCleanupPolicy,
            _literal_from_object(
                data.get("cleanup_policy", "remove_if_clean"),
                _CLEANUP_POLICIES,
                "cleanup_policy",
            ),
        ),
        worktree_created_at=_optional_float_from_object(
            data.get("worktree_created_at"),
            "worktree_created_at",
        ),
        worktree_touched_at=_optional_float_from_object(
            data.get("worktree_touched_at"),
            "worktree_touched_at",
        ),
        status=cast(
            ImprovementPatchCandidateWorktreeStatus,
            _literal_from_object(
                data.get("status", "allocated"),
                _ALLOCATION_STATUSES,
                "status",
            ),
        ),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _validate_plan(plan: ImprovementPatchCandidatePlan) -> None:
    if plan.status != "planned":
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeAllocator requires a planned candidate"
        )


def _validate_approval(approval: ImprovementPatchCandidateWorktreeApproval) -> None:
    if not approval.approved:
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeApproval.approved must be true"
        )
    missing = _REQUIRED_APPROVAL_PERMISSIONS.difference(approval.approved_permissions)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeApproval.approved_permissions "
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
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeApproval.approved_permissions "
            "must not be empty"
        )
    if "none" in items:
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeApproval.approved_permissions "
            "cannot include none"
        )
    missing = _REQUIRED_APPROVAL_PERMISSIONS.difference(items)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ImprovementPatchCandidateWorktreeValidationError(
            "ImprovementPatchCandidateWorktreeApproval.approved_permissions "
            f"missing: {joined}"
        )
    return cast(tuple[ImprovementRequiredPermission, ...], items)


def _candidate_worktree_slug(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8]
    sanitized = _UNSAFE_SLUG_CHARS.sub("-", candidate_id.strip()).strip("._-")
    if not sanitized:
        sanitized = "candidate"
    max_fragment_chars = 80 - len("ipc-") - len("-") - len(digest)
    fragment = sanitized[:max_fragment_chars].strip("._-") or "candidate"
    slug = f"ipc-{fragment}-{digest}"
    _require_safe_slug(slug, "worktree_slug")
    return slug


async def _cleanup_after_failed_create(
    manager: WorktreeManager,
    info: WorktreeInfo,
) -> str | None:
    try:
        result = await manager.cleanup(info)
    except Exception as exc:
        return f"cleanup failed: {type(exc).__name__}: {exc}"
    if result.kept:
        return f"cleanup kept worktree: {result.reason}"
    return None


def _with_cleanup_detail(message: str, detail: str | None) -> str:
    if detail is None:
        return message
    return f"{message}; {detail}"


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


def _require_safe_slug(value: str, name: str) -> None:
    _require_non_empty(value, name)
    if _SAFE_ALLOCATION_SLUG.fullmatch(value) is None:
        raise ValueError(f"{name} must match Raygent worktree slug constraints")


def _require_optional_non_negative(value: float | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    _require_non_empty(text, name)
    return text


def _optional_float_from_object(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _float_from_object(value, name)


def _float_from_object(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(f"{name} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(f"{name} must be a number") from exc


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


__all__ = (
    "ImprovementPatchCandidateWorktreeAllocation",
    "ImprovementPatchCandidateWorktreeAllocator",
    "ImprovementPatchCandidateWorktreeApproval",
    "ImprovementPatchCandidateWorktreeError",
    "ImprovementPatchCandidateWorktreeStatus",
    "ImprovementPatchCandidateWorktreeValidationError",
    "improvement_patch_candidate_worktree_allocation_from_dict",
    "improvement_patch_candidate_worktree_allocation_to_dict",
)
