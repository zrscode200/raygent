from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field

import pytest

import raygent_harness.improvement.candidate_materialization as materialization_module
from raygent_harness.improvement import (
    DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS,
    DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES,
    DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS,
    DEFAULT_MAX_MATERIALIZATION_OPERATIONS,
    DEFAULT_MAX_MATERIALIZATION_PATH_CHARS,
    DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationResult,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidateMaterializationApproval,
    ImprovementPatchCandidateMaterializationValidationError,
    ImprovementPatchCandidateMaterializer,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidateWorktreeAllocation,
    ImprovementPatchMaterializationRequest,
    ImprovementPatchMaterializationResult,
    ImprovementPatchOperation,
    ImprovementTarget,
    improvement_patch_candidate_evaluation_from_dict,
    improvement_patch_candidate_evaluation_to_dict,
    improvement_patch_candidate_materialization_from_dict,
    improvement_patch_candidate_materialization_to_dict,
    improvement_patch_operation_from_dict,
    improvement_patch_operation_to_dict,
)


def _target(kind: str = "source_code") -> ImprovementTarget:
    return ImprovementTarget(
        target_id="src/raygent_harness/improvement/candidate_materialization.py",
        kind=kind,  # type: ignore[arg-type]
        description="Improvement candidate materialization records",
        owner="kernel",
        metadata={"component": "improvement"},
    )


def _evaluation_plan() -> ImprovementEvaluationPlan:
    return ImprovementEvaluationPlan(
        checks=(
            ImprovementEvaluationCheck(
                name="candidate-materialization",
                instruction="Verify materialization stops before promotion.",
            ),
        ),
        success_criteria=("Materialization records are serializable and bounded.",),
    )


def _candidate_plan(
    *,
    candidate_id: str = "ipc_1",
    base_revision: str = "2011b23",
    expected_files: tuple[str, ...] = (
        "src/raygent_harness/improvement/candidate_materialization.py",
    ),
) -> ImprovementPatchCandidatePlan:
    return ImprovementPatchCandidatePlan(
        candidate_id=candidate_id,
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        target=_target(),
        base_revision=base_revision,
        summary="Plan one materialized candidate.",
        planned_changes=("Materialize a bounded create/replace operation.",),
        expected_files=expected_files,
        required_permissions=("filesystem_mutation", "worktree"),
        evaluation_plan=_evaluation_plan(),
        rollback_plan="Discard the allocated worktree.",
        created_at=400.0,
        metadata={"phase": "rsi-003c"},
    )


def _allocation(
    plan: ImprovementPatchCandidatePlan,
    *,
    candidate_id: str | None = None,
    base_revision: str | None = None,
) -> ImprovementPatchCandidateWorktreeAllocation:
    return ImprovementPatchCandidateWorktreeAllocation(
        allocation_id="ipcw_1",
        candidate_id=candidate_id or plan.candidate_id,
        run_id=plan.run_id,
        proposal_id=plan.proposal_id,
        gate_evaluation_id=plan.gate_evaluation_id,
        base_revision=base_revision or plan.base_revision,
        worktree_path="/tmp/raygent/ipc_1",
        worktree_branch="worktree-ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        worktree_head_commit=base_revision or plan.base_revision,
        git_root="/repo",
        cleanup_policy="remove_if_clean",
        created_at=405.0,
    )


def _approval() -> ImprovementPatchCandidateMaterializationApproval:
    return ImprovementPatchCandidateMaterializationApproval(
        approved_permissions=("filesystem_mutation",),
        reason="RSI-003C local materialization approval",
        approved_by="tester",
        created_at=410.0,
    )


def _operation(
    *,
    operation_id: str = "op_1",
    relative_path: str = "src/raygent_harness/improvement/candidate_materialization.py",
    kind: str = "replace_text",
    old_text: str | None = "old",
    new_text: str = "new",
) -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id=operation_id,
        kind=kind,  # type: ignore[arg-type]
        relative_path=relative_path,
        old_text=old_text,
        new_text=new_text,
        metadata={"source": "unit-test"},
    )


@dataclass
class FakeMaterializer:
    changed_files: tuple[str, ...] = (
        "src/raygent_harness/improvement/candidate_materialization.py",
    )
    summary: str | None = "fake materialization"
    metadata: dict[str, bool] = field(default_factory=lambda: {"fake": True})

    def __post_init__(self) -> None:
        self.requests: list[ImprovementPatchMaterializationRequest] = []

    async def materialize(
        self,
        request: ImprovementPatchMaterializationRequest,
    ) -> ImprovementPatchMaterializationResult:
        self.requests.append(request)
        return ImprovementPatchMaterializationResult(
            changed_files=self.changed_files,
            summary=self.summary,
            metadata=self.metadata,
        )


def _materialization() -> ImprovementPatchCandidateMaterialization:
    plan = _candidate_plan()
    operation = _operation()
    return ImprovementPatchCandidateMaterialization(
        materialization_id="ipcm_1",
        allocation_id="ipcw_1",
        candidate_id=plan.candidate_id,
        run_id=plan.run_id,
        proposal_id=plan.proposal_id,
        gate_evaluation_id=plan.gate_evaluation_id,
        base_revision=plan.base_revision,
        worktree_path="/tmp/raygent/ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        operations=(operation,),
        changed_files=(operation.relative_path,),
        patch_digest="sha256:" + "0" * 64,
        created_at=500.0,
        metadata={"phase": "rsi-003c"},
    )


def test_materialization_approval_requires_filesystem_mutation() -> None:
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="filesystem_mutation",
    ):
        ImprovementPatchCandidateMaterializationApproval(
            approved_permissions=("worktree",),
            reason="approve worktree only",
        )
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="none",
    ):
        ImprovementPatchCandidateMaterializationApproval(
            approved_permissions=("none", "filesystem_mutation"),
            reason="invalid approval",
        )
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="approved",
    ):
        ImprovementPatchCandidateMaterializationApproval(
            approved_permissions=("filesystem_mutation",),
            reason="denied",
            approved=False,
        )


@pytest.mark.asyncio
async def test_materializer_requires_injected_materializer_and_call_time_approval() -> None:
    plan = _candidate_plan()
    allocation = _allocation(plan)
    service = ImprovementPatchCandidateMaterializer()
    operation = _operation()

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="ImprovementPatchMaterializer",
    ):
        await service.materialize(
            plan,
            allocation,
            operations=(operation,),
            materializer=None,
            approval=_approval(),
        )

    fake = FakeMaterializer()
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="approval",
    ):
        await service.materialize(
            plan,
            allocation,
            operations=(operation,),
            materializer=fake,
            approval=None,
        )

    assert fake.requests == []


@pytest.mark.asyncio
async def test_materializer_invokes_injected_materializer_once_and_records_result() -> None:
    plan = _candidate_plan(
        expected_files=("./src/raygent_harness/improvement/candidate_materialization.py",)
    )
    allocation = _allocation(plan)
    fake = FakeMaterializer()
    operation = _operation(
        relative_path="src/raygent_harness/improvement/./candidate_materialization.py"
    )

    materialization = await ImprovementPatchCandidateMaterializer(
        clock=lambda: 500.0,
        materialization_id_factory=lambda: "ipcm_1",
    ).materialize(
        plan,
        allocation,
        operations=(operation,),
        materializer=fake,
        approval=_approval(),
        metadata={"phase": "rsi-003c"},
    )

    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.worktree_path == allocation.worktree_path
    assert request.expected_files == (operation.relative_path,)
    assert request.operations == (operation,)
    assert materialization.materialization_id == "ipcm_1"
    assert materialization.allocation_id == allocation.allocation_id
    assert materialization.candidate_id == plan.candidate_id
    assert materialization.run_id == plan.run_id
    assert materialization.proposal_id == plan.proposal_id
    assert materialization.gate_evaluation_id == plan.gate_evaluation_id
    assert materialization.base_revision == plan.base_revision
    assert materialization.worktree_path == allocation.worktree_path
    assert materialization.worktree_slug == allocation.worktree_slug
    assert materialization.operations == (operation,)
    assert materialization.changed_files == (operation.relative_path,)
    assert materialization.patch_digest.startswith("sha256:")
    assert materialization.status == "materialized"
    assert materialization.created_at == 500.0
    assert materialization.metadata == {
        "phase": "rsi-003c",
        "materializer_summary": "fake materialization",
        "materializer_metadata": {"fake": True},
    }

    snapshot = improvement_patch_candidate_materialization_to_dict(materialization)
    assert "approval" not in snapshot
    assert "approved_by" not in snapshot
    assert "approved_permissions" not in snapshot
    assert "reason" not in snapshot


@pytest.mark.asyncio
async def test_materialization_serializes_to_json_and_round_trips() -> None:
    plan = _candidate_plan()
    allocation = _allocation(plan)
    materialization = await ImprovementPatchCandidateMaterializer(
        clock=lambda: 500.0,
        materialization_id_factory=lambda: "ipcm_1",
    ).materialize(
        plan,
        allocation,
        operations=(_operation(),),
        materializer=FakeMaterializer(),
        approval=_approval(),
        metadata={"nested": {"ok": True}},
    )

    snapshot = improvement_patch_candidate_materialization_to_dict(materialization)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_materialization_from_dict(snapshot)

    assert restored == materialization
    assert restored.status == "materialized"
    assert restored.metadata["nested"] == {"ok": True}


def test_patch_operation_serializes_and_normalizes_paths() -> None:
    operation = _operation(relative_path="./src/../src/file.py")
    snapshot = improvement_patch_operation_to_dict(operation)
    restored = improvement_patch_operation_from_dict(snapshot)

    assert operation.relative_path == "src/file.py"
    assert restored == operation
    assert restored.metadata == {"source": "unit-test"}


def test_materialization_record_rejects_duplicate_operation_ids() -> None:
    operation = _operation()

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="duplicate operation ids",
    ):
        ImprovementPatchCandidateMaterialization(
            materialization_id="ipcm_duplicate",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="2011b23",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            operations=(operation, operation),
            changed_files=(operation.relative_path,),
            patch_digest="sha256:" + "0" * 64,
        )


@pytest.mark.asyncio
async def test_materializer_validates_plan_allocation_linkage_before_call() -> None:
    plan = _candidate_plan()
    fake = FakeMaterializer()

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="candidate_id",
    ):
        await ImprovementPatchCandidateMaterializer().materialize(
            plan,
            _allocation(plan, candidate_id="ipc_other"),
            operations=(_operation(),),
            materializer=fake,
            approval=_approval(),
        )

    assert fake.requests == []


@pytest.mark.asyncio
async def test_materializer_rejects_unsafe_expected_operation_and_changed_paths() -> None:
    operation = _operation()

    plan_with_escape = _candidate_plan(expected_files=("../escape.py",))
    fake = FakeMaterializer()
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="parent traversal",
    ):
        await ImprovementPatchCandidateMaterializer().materialize(
            plan_with_escape,
            _allocation(plan_with_escape),
            operations=(operation,),
            materializer=fake,
            approval=_approval(),
        )
    assert fake.requests == []

    plan = _candidate_plan()
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="outside expected files",
    ):
        await ImprovementPatchCandidateMaterializer().materialize(
            plan,
            _allocation(plan),
            operations=(_operation(relative_path="src/other.py"),),
            materializer=fake,
            approval=_approval(),
        )
    assert fake.requests == []

    changed_fake = FakeMaterializer(changed_files=("src/other.py",))
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="changed_files outside expected files",
    ):
        await ImprovementPatchCandidateMaterializer().materialize(
            plan,
            _allocation(plan),
            operations=(operation,),
            materializer=changed_fake,
            approval=_approval(),
        )
    assert len(changed_fake.requests) == 1


def test_materialization_bounds_are_enforced() -> None:
    path = "a" * (DEFAULT_MAX_MATERIALIZATION_PATH_CHARS + 1)
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="PATH_CHARS",
    ):
        _operation(relative_path=path)

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="TEXT_CHARS",
    ):
        _operation(new_text="x" * (DEFAULT_MAX_MATERIALIZATION_TEXT_CHARS + 1))

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="OPERATIONS",
    ):
        ImprovementPatchMaterializationRequest(
            worktree_path="/tmp/raygent/ipc_1",
            operations=tuple(
                _operation(operation_id=f"op_{index}")
                for index in range(DEFAULT_MAX_MATERIALIZATION_OPERATIONS + 1)
            ),
            expected_files=(
                "src/raygent_harness/improvement/candidate_materialization.py",
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="CHANGED_FILES",
    ):
        ImprovementPatchMaterializationResult(
            changed_files=tuple(
                f"src/file_{index}.py"
                for index in range(DEFAULT_MAX_MATERIALIZATION_CHANGED_FILES + 1)
            )
        )

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="METADATA_CHARS",
    ):
        ImprovementPatchOperation(
            operation_id="op_meta",
            kind="create_file",
            relative_path="src/file.py",
            new_text="",
            metadata={"large": "x" * (DEFAULT_MAX_MATERIALIZATION_METADATA_CHARS + 1)},
        )


def _evaluation_result(
    result_id: str,
    status: str,
    *,
    required: bool = True,
    kind: str = "unit_tests",
) -> ImprovementPatchCandidateEvaluationResult:
    return ImprovementPatchCandidateEvaluationResult(
        result_id=result_id,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        summary=f"{kind} is {status}",
        changed_files=("src/raygent_harness/improvement/candidate_materialization.py",),
        output_excerpt="pytest passed",
        required=required,
        created_at=600.0,
        metadata={"source": "unit-test"},
    )


def _evaluation(
    results: tuple[ImprovementPatchCandidateEvaluationResult, ...],
) -> ImprovementPatchCandidateEvaluation:
    materialization = _materialization()
    return ImprovementPatchCandidateEvaluation(
        evaluation_id="ipce_1",
        materialization_id=materialization.materialization_id,
        allocation_id=materialization.allocation_id,
        candidate_id=materialization.candidate_id,
        run_id=materialization.run_id,
        proposal_id=materialization.proposal_id,
        gate_evaluation_id=materialization.gate_evaluation_id,
        results=results,
        created_at=700.0,
        metadata={"phase": "evaluation"},
    )


def test_evaluation_derives_fail_warn_needs_review_and_no_result_decisions() -> None:
    assert _evaluation((_evaluation_result("res_pass", "pass"),)).decision == "pass"
    assert _evaluation((_evaluation_result("res_fail", "fail"),)).decision == "fail"

    optional_failure = _evaluation(
        (_evaluation_result("res_optional", "fail", required=False),)
    )
    assert optional_failure.decision == "warn"
    assert optional_failure.warnings == ("one or more optional evaluation results failed",)

    needs_review = _evaluation((_evaluation_result("res_review", "needs_review"),))
    assert needs_review.decision == "needs_review"

    no_results = _evaluation(())
    assert no_results.decision == "needs_review"
    assert no_results.warnings == ("no evaluation results supplied",)


def test_evaluation_rejects_duplicate_results_and_oversized_output() -> None:
    result = _evaluation_result("res_duplicate", "pass")
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="duplicate result ids",
    ):
        _evaluation((result, result))

    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="OUTPUT_EXCERPT",
    ):
        ImprovementPatchCandidateEvaluationResult(
            result_id="res_large",
            kind="unit_tests",
            status="pass",
            summary="large output",
            output_excerpt="x" * (DEFAULT_MAX_EVALUATION_OUTPUT_EXCERPT_CHARS + 1),
        )


def test_evaluation_serializes_and_rejects_policy_inconsistent_snapshot() -> None:
    evaluation = _evaluation((_evaluation_result("res_pass", "pass"),))
    snapshot = improvement_patch_candidate_evaluation_to_dict(evaluation)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_evaluation_from_dict(snapshot)

    assert restored == evaluation
    assert restored.decision == "pass"
    assert restored.results[0].metadata == {"source": "unit-test"}

    snapshot["decision"] = "fail"
    with pytest.raises(
        ImprovementPatchCandidateMaterializationValidationError,
        match="decision",
    ):
        improvement_patch_candidate_evaluation_from_dict(snapshot)


def test_candidate_materialization_module_uses_only_injected_materializer_seam() -> None:
    source = inspect.getsource(materialization_module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    called_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)

    forbidden_modules = {
        "raygent_harness.core.permission_engine",
        "raygent_harness.core.streaming_tool_executor",
        "raygent_harness.core.tool_execution",
        "raygent_harness.core.tool_orchestration",
        "raygent_harness.sdk",
        "raygent_harness.services.remote_agent",
        "raygent_harness.services.worktree.manager",
        "raygent_harness.tools.bash_tool",
        "raygent_harness.tools.file_edit_tool",
        "raygent_harness.tools.file_text_utils",
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
        "write_text",
    }

    assert "hashlib" in imported_modules
    assert "json" in imported_modules
    assert "posixpath" in imported_modules
    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
    assert "open" not in called_names
