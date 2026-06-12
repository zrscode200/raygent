from __future__ import annotations

import ast
import inspect
import json
import re
from dataclasses import dataclass, field

import pytest

import raygent_harness.improvement.candidate_worktree as candidate_worktree_module
from raygent_harness.improvement import (
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidateWorktreeAllocation,
    ImprovementPatchCandidateWorktreeAllocator,
    ImprovementPatchCandidateWorktreeApproval,
    ImprovementPatchCandidateWorktreeValidationError,
    ImprovementTarget,
    improvement_patch_candidate_worktree_allocation_from_dict,
    improvement_patch_candidate_worktree_allocation_to_dict,
)
from raygent_harness.services.worktree.manager import is_ephemeral_worktree_slug
from raygent_harness.services.worktree.models import (
    WorktreeCleanupResult,
    WorktreeInfo,
)


def _target(kind: str = "source_code") -> ImprovementTarget:
    return ImprovementTarget(
        target_id="src/raygent_harness/improvement/candidate_worktree.py",
        kind=kind,  # type: ignore[arg-type]
        description="Improvement candidate worktree allocation records",
        owner="kernel",
        metadata={"component": "improvement"},
    )


def _evaluation_plan() -> ImprovementEvaluationPlan:
    return ImprovementEvaluationPlan(
        checks=(
            ImprovementEvaluationCheck(
                name="candidate-worktree",
                instruction="Verify allocation stops before patch materialization.",
            ),
        ),
        success_criteria=("Allocation records are serializable and bounded.",),
    )


def _candidate_plan(
    *,
    candidate_id: str = "ipc_1",
    base_revision: str = "b5a011f",
) -> ImprovementPatchCandidatePlan:
    return ImprovementPatchCandidatePlan(
        candidate_id=candidate_id,
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        target=_target(),
        base_revision=base_revision,
        summary="Plan one isolated worktree allocation.",
        planned_changes=("Allocate a candidate worktree.",),
        expected_files=("src/raygent_harness/improvement/candidate_worktree.py",),
        required_permissions=("filesystem_mutation", "worktree"),
        evaluation_plan=_evaluation_plan(),
        rollback_plan="Clean up or ignore the allocated worktree record.",
        created_at=400.0,
        metadata={"phase": "rsi-003b"},
    )


def _approval() -> ImprovementPatchCandidateWorktreeApproval:
    return ImprovementPatchCandidateWorktreeApproval(
        approved_permissions=("filesystem_mutation", "worktree"),
        reason="RSI-003B local source implementation approval",
        approved_by="tester",
        created_at=410.0,
    )


@dataclass
class FakeWorktreeManager:
    head_commit: str | None
    returned_slug: str | None = None
    use_requested_slug: bool = True
    cleanup_result: WorktreeCleanupResult = field(
        default_factory=lambda: WorktreeCleanupResult(
            kept=False,
            reason="removed",
        )
    )
    cleanup_error: Exception | None = None

    def __post_init__(self) -> None:
        self.create_calls: list[tuple[str, str]] = []
        self.cleanup_calls: list[WorktreeInfo] = []

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        self.create_calls.append((slug, cwd))
        returned_slug = slug if self.use_requested_slug else self.returned_slug
        return WorktreeInfo(
            path=f"/tmp/raygent/{slug}",
            branch=f"worktree-{slug}",
            head_commit=self.head_commit,
            git_root="/repo",
            slug=returned_slug,
            created_at=111.0,
            touched_at=112.0,
            cleanup_policy="remove_if_clean",
        )

    async def has_changes(self, info: WorktreeInfo) -> bool:
        return False

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        self.cleanup_calls.append(info)
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return self.cleanup_result


def test_worktree_approval_requires_mutation_and_worktree_permissions() -> None:
    with pytest.raises(ImprovementPatchCandidateWorktreeValidationError, match="worktree"):
        ImprovementPatchCandidateWorktreeApproval(
            approved_permissions=("filesystem_mutation",),
            reason="approve filesystem only",
        )
    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match="filesystem_mutation",
    ):
        ImprovementPatchCandidateWorktreeApproval(
            approved_permissions=("worktree",),
            reason="approve worktree only",
        )
    with pytest.raises(ImprovementPatchCandidateWorktreeValidationError, match="none"):
        ImprovementPatchCandidateWorktreeApproval(
            approved_permissions=("none", "filesystem_mutation", "worktree"),
            reason="invalid approval",
        )


@pytest.mark.asyncio
async def test_allocator_requires_injected_manager_and_call_time_approval() -> None:
    plan = _candidate_plan()
    allocator = ImprovementPatchCandidateWorktreeAllocator()

    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match="WorktreeManager",
    ):
        await allocator.allocate(
            plan,
            manager=None,
            cwd="/repo",
            approval=_approval(),
        )

    manager = FakeWorktreeManager(head_commit=plan.base_revision)
    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match="approval",
    ):
        await allocator.allocate(
            plan,
            manager=manager,
            cwd="/repo",
            approval=None,
        )

    assert manager.create_calls == []


@pytest.mark.asyncio
async def test_allocator_passes_explicit_cwd_and_records_allocation() -> None:
    plan = _candidate_plan()
    manager = FakeWorktreeManager(head_commit=plan.base_revision)
    allocator = ImprovementPatchCandidateWorktreeAllocator(
        clock=lambda: 500.0,
        allocation_id_factory=lambda: "ipcw_1",
    )

    allocation = await allocator.allocate(
        plan,
        manager=manager,
        cwd="/caller/repo",
        approval=_approval(),
        metadata={"phase": "rsi-003b"},
    )

    assert manager.create_calls == [(allocation.worktree_slug, "/caller/repo")]
    assert manager.cleanup_calls == []
    assert allocation.allocation_id == "ipcw_1"
    assert allocation.candidate_id == plan.candidate_id
    assert allocation.run_id == plan.run_id
    assert allocation.proposal_id == plan.proposal_id
    assert allocation.gate_evaluation_id == plan.gate_evaluation_id
    assert allocation.base_revision == plan.base_revision
    assert allocation.worktree_path == f"/tmp/raygent/{allocation.worktree_slug}"
    assert allocation.worktree_branch == f"worktree-{allocation.worktree_slug}"
    assert allocation.worktree_head_commit == plan.base_revision
    assert allocation.git_root == "/repo"
    assert allocation.cleanup_policy == "remove_if_clean"
    assert allocation.worktree_created_at == 111.0
    assert allocation.worktree_touched_at == 112.0
    assert allocation.status == "allocated"
    assert allocation.created_at == 500.0
    assert allocation.metadata == {"phase": "rsi-003b"}

    snapshot = improvement_patch_candidate_worktree_allocation_to_dict(allocation)
    assert "approval" not in snapshot
    assert "approved_by" not in snapshot
    assert "approved_permissions" not in snapshot
    assert "reason" not in snapshot


@pytest.mark.asyncio
async def test_worktree_allocation_serializes_to_json_and_round_trips() -> None:
    plan = _candidate_plan()
    manager = FakeWorktreeManager(head_commit=plan.base_revision)
    allocation = await ImprovementPatchCandidateWorktreeAllocator(
        clock=lambda: 500.0,
        allocation_id_factory=lambda: "ipcw_1",
    ).allocate(
        plan,
        manager=manager,
        cwd="/repo",
        approval=_approval(),
        metadata={"nested": {"ok": True}},
    )

    snapshot = improvement_patch_candidate_worktree_allocation_to_dict(allocation)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_worktree_allocation_from_dict(snapshot)

    assert restored == allocation
    assert restored.status == "allocated"
    assert restored.metadata == {"nested": {"ok": True}}


def test_allocation_record_rejects_head_mismatch_on_construction_and_restore() -> None:
    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match=r"head_commit.*base_revision",
    ):
        ImprovementPatchCandidateWorktreeAllocation(
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="expected",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_branch="worktree-ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            worktree_head_commit="different",
            git_root="/repo",
            cleanup_policy="remove_if_clean",
        )

    snapshot = improvement_patch_candidate_worktree_allocation_to_dict(
        ImprovementPatchCandidateWorktreeAllocation(
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="expected",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_branch="worktree-ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            worktree_head_commit="expected",
            git_root="/repo",
            cleanup_policy="remove_if_clean",
        )
    )
    snapshot["worktree_head_commit"] = "different"

    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match=r"head_commit.*base_revision",
    ):
        improvement_patch_candidate_worktree_allocation_from_dict(snapshot)


@pytest.mark.asyncio
async def test_allocator_uses_safe_non_swept_candidate_slug() -> None:
    plan = _candidate_plan(
        candidate_id="job-../../Needs Fix " + ("x" * 120),
        base_revision="abc123",
    )
    manager = FakeWorktreeManager(head_commit=plan.base_revision)

    allocation = await ImprovementPatchCandidateWorktreeAllocator().allocate(
        plan,
        manager=manager,
        cwd="/repo",
        approval=_approval(),
    )

    slug = allocation.worktree_slug
    assert slug.startswith("ipc-")
    assert len(slug) <= 80
    assert re.fullmatch(r"[A-Za-z0-9._-]{1,80}", slug)
    assert "/" not in slug
    assert not is_ephemeral_worktree_slug(slug)
    assert manager.create_calls == [(slug, "/repo")]


@pytest.mark.asyncio
async def test_allocator_cleans_up_when_allocation_record_validation_fails() -> None:
    plan = _candidate_plan()
    manager = FakeWorktreeManager(
        head_commit=plan.base_revision,
        cleanup_result=WorktreeCleanupResult(
            kept=True,
            reason="changed",
            path="/tmp/raygent/ipc_1",
            branch="worktree-ipc_1",
        ),
    )

    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match=r"allocation_id_factory.*cleanup kept",
    ):
        await ImprovementPatchCandidateWorktreeAllocator(
            allocation_id_factory=lambda: " ",
        ).allocate(
            plan,
            manager=manager,
            cwd="/repo",
            approval=_approval(),
        )

    assert len(manager.cleanup_calls) == 1


@pytest.mark.asyncio
async def test_allocator_rejects_head_mismatch_and_attempts_cleanup_when_kept() -> None:
    plan = _candidate_plan(base_revision="expected")
    manager = FakeWorktreeManager(
        head_commit="different",
        cleanup_result=WorktreeCleanupResult(
            kept=True,
            reason="changed",
            path="/tmp/raygent/ipc_1",
            branch="worktree-ipc_1",
        ),
    )

    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match=r"head_commit.*cleanup kept",
    ):
        await ImprovementPatchCandidateWorktreeAllocator().allocate(
            plan,
            manager=manager,
            cwd="/repo",
            approval=_approval(),
        )

    assert len(manager.cleanup_calls) == 1
    assert manager.create_calls[0][1] == "/repo"


@pytest.mark.asyncio
async def test_allocator_rejects_head_mismatch_when_cleanup_fails() -> None:
    plan = _candidate_plan(base_revision="expected")
    manager = FakeWorktreeManager(
        head_commit="different",
        cleanup_error=RuntimeError("boom"),
    )

    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match=r"head_commit.*cleanup failed",
    ):
        await ImprovementPatchCandidateWorktreeAllocator().allocate(
            plan,
            manager=manager,
            cwd="/repo",
            approval=_approval(),
        )

    assert len(manager.cleanup_calls) == 1


@pytest.mark.asyncio
async def test_allocator_rejects_returned_slug_mismatch_and_cleans_up() -> None:
    plan = _candidate_plan()
    manager = FakeWorktreeManager(
        head_commit=plan.base_revision,
        returned_slug="other-slug",
        use_requested_slug=False,
    )

    with pytest.raises(ImprovementPatchCandidateWorktreeValidationError, match="slug"):
        await ImprovementPatchCandidateWorktreeAllocator().allocate(
            plan,
            manager=manager,
            cwd="/repo",
            approval=_approval(),
        )

    assert len(manager.cleanup_calls) == 1


def test_candidate_worktree_module_imports_only_allowed_worktree_seam() -> None:
    source = inspect.getsource(candidate_worktree_module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    imported_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    forbidden_modules = {
        "raygent_harness.core.permission_engine",
        "raygent_harness.core.streaming_tool_executor",
        "raygent_harness.core.tool_execution",
        "raygent_harness.core.tool_orchestration",
        "raygent_harness.sdk",
        "raygent_harness.services.remote_agent",
        "os",
        "pathlib",
        "subprocess",
    }
    forbidden_names = {
        "GitWorktreeManager",
        "PermissionHandler",
        "PermissionRequest",
        "Path",
        "create_raygent",
    }

    assert "raygent_harness.services.worktree.manager" in imported_modules
    assert "WorktreeManager" in imported_names
    assert "WorktreeInfo" in imported_names
    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
