from __future__ import annotations

import ast
import inspect
import json
from dataclasses import replace

import pytest

import raygent_harness.improvement.candidate_verification as verification_module
from raygent_harness.improvement import (
    DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS,
    DEFAULT_MAX_VERIFICATION_CHECKS,
    DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidateVerificationCheck,
    ImprovementPatchCandidateVerificationPlan,
    ImprovementPatchCandidateVerificationPlanner,
    ImprovementPatchCandidateVerificationValidationError,
    ImprovementPatchOperation,
    ImprovementTarget,
    improvement_patch_candidate_verification_check_from_dict,
    improvement_patch_candidate_verification_check_to_dict,
    improvement_patch_candidate_verification_plan_from_dict,
    improvement_patch_candidate_verification_plan_to_dict,
)


def _target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="src/raygent_harness/improvement/candidate_verification.py",
        kind="source_code",
        description="Improvement candidate verification records",
        owner="kernel",
        metadata={"component": "improvement"},
    )


def _evaluation_plan() -> ImprovementEvaluationPlan:
    return ImprovementEvaluationPlan(
        checks=(
            ImprovementEvaluationCheck(
                name="candidate-verification",
                instruction="Verify candidate verification records are bounded.",
            ),
        ),
        non_regression_checks=(
            ImprovementEvaluationCheck(
                name="candidate-materialization-regression",
                instruction="Ensure materialization records still round trip.",
            ),
        ),
        cost_checks=(
            ImprovementEvaluationCheck(
                name="verification-cost",
                instruction="Check verification stays within a local budget.",
                required=False,
            ),
        ),
        success_criteria=("Verification records are serializable and data-only.",),
    )


def _candidate_plan(
    *,
    expected_files: tuple[str, ...] = (
        "src/raygent_harness/improvement/candidate_verification.py",
        "tests/unit/improvement/test_candidate_verification.py",
    ),
) -> ImprovementPatchCandidatePlan:
    return ImprovementPatchCandidatePlan(
        candidate_id="ipc_1",
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        target=_target(),
        base_revision="7996d86",
        summary="Plan candidate verification records.",
        planned_changes=("Add data-only candidate verification planning.",),
        expected_files=expected_files,
        required_permissions=("filesystem_mutation", "worktree"),
        evaluation_plan=_evaluation_plan(),
        rollback_plan="Discard the allocated worktree.",
        created_at=400.0,
        metadata={"phase": "rsi-005a"},
    )


def _operation() -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id="op_1",
        kind="replace_text",
        relative_path="src/raygent_harness/improvement/candidate_verification.py",
        old_text="old",
        new_text="new",
    )


def _materialization(
    *,
    candidate_id: str = "ipc_1",
    run_id: str = "ir_1",
    proposal_id: str = "ip_1",
    gate_evaluation_id: str = "ige_1",
    base_revision: str = "7996d86",
    changed_files: tuple[str, ...] = (
        "src/raygent_harness/improvement/candidate_verification.py",
    ),
) -> ImprovementPatchCandidateMaterialization:
    operation = _operation()
    return ImprovementPatchCandidateMaterialization(
        materialization_id="ipcm_1",
        allocation_id="ipcw_1",
        candidate_id=candidate_id,
        run_id=run_id,
        proposal_id=proposal_id,
        gate_evaluation_id=gate_evaluation_id,
        base_revision=base_revision,
        worktree_path="/tmp/raygent/ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        operations=(operation,),
        changed_files=changed_files,
        patch_digest="sha256:" + "5" * 64,
        created_at=500.0,
        metadata={"phase": "rsi-003c"},
    )


def _verification_plan() -> ImprovementPatchCandidateVerificationPlan:
    return ImprovementPatchCandidateVerificationPlanner(
        clock=lambda: 600.0,
        verification_plan_id_factory=lambda: "ipcvp_1",
    ).plan(
        _candidate_plan(),
        _materialization(),
        metadata={"phase": "verification-plan"},
    )


def test_verification_planner_creates_data_only_plan_from_materialization() -> None:
    plan = _verification_plan()

    assert plan.verification_plan_id == "ipcvp_1"
    assert plan.materialization_id == "ipcm_1"
    assert plan.allocation_id == "ipcw_1"
    assert plan.candidate_id == "ipc_1"
    assert plan.run_id == "ir_1"
    assert plan.proposal_id == "ip_1"
    assert plan.gate_evaluation_id == "ige_1"
    assert plan.base_revision == "7996d86"
    assert plan.worktree_path == "/tmp/raygent/ipc_1"
    assert plan.worktree_slug == "ipc-ipc_1-a9b687a7"
    assert plan.patch_digest == "sha256:" + "5" * 64
    assert plan.allowed_changed_files == (
        "src/raygent_harness/improvement/candidate_verification.py",
    )
    assert plan.status == "verification_planned"
    assert plan.created_at == 600.0
    assert plan.metadata == {"phase": "verification-plan"}

    assert [check.source_plan_section for check in plan.checks] == [
        "checks",
        "non_regression_checks",
        "cost_checks",
    ]
    assert [check.kind for check in plan.checks] == [
        "other",
        "non_regression",
        "other",
    ]
    assert [check.required for check in plan.checks] == [True, True, False]


@pytest.mark.parametrize(
    ("field_name", "candidate_id", "run_id", "proposal_id", "gate_evaluation_id", "base_revision"),
    (
        ("candidate_id", "ipc_other", "ir_1", "ip_1", "ige_1", "7996d86"),
        ("run_id", "ipc_1", "ir_other", "ip_1", "ige_1", "7996d86"),
        ("proposal_id", "ipc_1", "ir_1", "ip_other", "ige_1", "7996d86"),
        ("gate_evaluation_id", "ipc_1", "ir_1", "ip_1", "ige_other", "7996d86"),
        ("base_revision", "ipc_1", "ir_1", "ip_1", "ige_1", "other"),
    ),
)
def test_verification_planner_rejects_materialization_linkage_mismatch(
    field_name: str,
    candidate_id: str,
    run_id: str,
    proposal_id: str,
    gate_evaluation_id: str,
    base_revision: str,
) -> None:
    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match=field_name,
    ):
        ImprovementPatchCandidateVerificationPlanner().plan(
            _candidate_plan(),
            _materialization(
                candidate_id=candidate_id,
                run_id=run_id,
                proposal_id=proposal_id,
                gate_evaluation_id=gate_evaluation_id,
                base_revision=base_revision,
            ),
        )


def test_verification_planner_rejects_changed_files_outside_expected_files() -> None:
    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="outside expected files",
    ):
        ImprovementPatchCandidateVerificationPlanner().plan(
            _candidate_plan(
                expected_files=(
                    "src/raygent_harness/improvement/candidate_verification.py",
                )
            ),
            _materialization(
                changed_files=("tests/unit/improvement/test_candidate_verification.py",)
            ),
        )


def test_verification_planner_rejects_evaluation_plan_without_checks() -> None:
    candidate_plan = replace(
        _candidate_plan(),
        evaluation_plan=ImprovementEvaluationPlan(
            success_criteria=("A success criterion is not an executable check.",)
        ),
    )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="checks must not be empty",
    ):
        ImprovementPatchCandidateVerificationPlanner().plan(
            candidate_plan,
            _materialization(),
        )


def test_verification_check_and_plan_round_trip_through_json() -> None:
    plan = _verification_plan()
    check_snapshot = improvement_patch_candidate_verification_check_to_dict(
        plan.checks[0]
    )
    plan_snapshot = improvement_patch_candidate_verification_plan_to_dict(plan)

    json.dumps(check_snapshot)
    json.dumps(plan_snapshot)

    assert improvement_patch_candidate_verification_check_from_dict(
        check_snapshot
    ) == plan.checks[0]
    assert improvement_patch_candidate_verification_plan_from_dict(plan_snapshot) == plan


def test_verification_plan_deserialization_rejects_policy_inconsistent_status() -> None:
    snapshot = improvement_patch_candidate_verification_plan_to_dict(
        _verification_plan()
    )
    snapshot["status"] = "verification_recorded"

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="status",
    ):
        improvement_patch_candidate_verification_plan_from_dict(snapshot)


def test_verification_plan_rejects_duplicate_checks() -> None:
    check = ImprovementPatchCandidateVerificationCheck(
        check_id="check_1",
        kind="other",
        name="check",
        instruction="Run a bounded check.",
        source_plan_section="checks",
    )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="duplicate",
    ):
        ImprovementPatchCandidateVerificationPlan(
            verification_plan_id="ipcvp_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="7996d86",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            patch_digest="sha256:" + "5" * 64,
            allowed_changed_files=(
                "src/raygent_harness/improvement/candidate_verification.py",
            ),
            checks=(check, check),
        )


@pytest.mark.parametrize(
    "path",
    (
        "/tmp/file.py",
        "\\tmp\\file.py",
        "../file.py",
        "src/../outside.py",
        "src/raygent_harness/\x00file.py",
    ),
)
def test_verification_plan_rejects_unsafe_allowed_changed_files(path: str) -> None:
    with pytest.raises(ImprovementPatchCandidateVerificationValidationError):
        ImprovementPatchCandidateVerificationPlan(
            verification_plan_id="ipcvp_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="7996d86",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            patch_digest="sha256:" + "5" * 64,
            allowed_changed_files=(path,),
            checks=(_verification_plan().checks[0],),
        )


def test_verification_check_enforces_named_bounds() -> None:
    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS",
    ):
        ImprovementPatchCandidateVerificationCheck(
            check_id="check_1",
            kind="other",
            name="x" * (DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS + 1),
            instruction="Run a bounded check.",
            source_plan_section="checks",
        )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS",
    ):
        ImprovementPatchCandidateVerificationCheck(
            check_id="check_1",
            kind="other",
            name="check",
            instruction="x" * (DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS + 1),
            source_plan_section="checks",
        )


def test_verification_plan_enforces_check_count_bound() -> None:
    checks = tuple(
        ImprovementPatchCandidateVerificationCheck(
            check_id=f"check_{index}",
            kind="other",
            name=f"check {index}",
            instruction="Run a bounded check.",
            source_plan_section="checks",
        )
        for index in range(DEFAULT_MAX_VERIFICATION_CHECKS + 1)
    )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="DEFAULT_MAX_VERIFICATION_CHECKS",
    ):
        ImprovementPatchCandidateVerificationPlan(
            verification_plan_id="ipcvp_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="7996d86",
            worktree_path="/tmp/raygent/ipc_1",
            worktree_slug="ipc-ipc_1-a9b687a7",
            patch_digest="sha256:" + "5" * 64,
            allowed_changed_files=(
                "src/raygent_harness/improvement/candidate_verification.py",
            ),
            checks=checks,
        )


def test_candidate_verification_module_is_data_only_and_imports_no_runner() -> None:
    source = inspect.getsource(verification_module)
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
        "os",
        "pathlib",
        "subprocess",
        "raygent_harness.core.permission_engine",
        "raygent_harness.core.streaming_tool_executor",
        "raygent_harness.core.tool_execution",
        "raygent_harness.core.tool_orchestration",
        "raygent_harness.improvement.candidate_archive",
        "raygent_harness.improvement.candidate_promotion",
        "raygent_harness.sdk",
        "raygent_harness.services.worktree.manager",
        "raygent_harness.tools.bash_tool",
        "raygent_harness.tools.file_edit_tool",
        "raygent_harness.tools.file_text_utils",
    }
    forbidden_names = {
        "CommandRunner",
        "GitWorktreeManager",
        "ImprovementPatchCandidateArchiver",
        "ImprovementPatchCandidatePromotionService",
        "Path",
        "PermissionHandler",
        "PermissionRequest",
        "create_raygent",
        "open",
        "write_text",
    }

    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
    assert called_names.isdisjoint(forbidden_names)


def test_candidate_verification_module_exports_public_rsi_005a_symbols() -> None:
    assert set(verification_module.__all__) == {
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
    }
