from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field, replace
from typing import cast

import pytest

import raygent_harness.improvement.candidate_verification as verification_module
from raygent_harness.improvement import (
    DEFAULT_MAX_VERIFICATION_CHECK_NAME_CHARS,
    DEFAULT_MAX_VERIFICATION_CHECKS,
    DEFAULT_MAX_VERIFICATION_INSTRUCTION_CHARS,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidateOutcomePolicy,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidateVerificationApproval,
    ImprovementPatchCandidateVerificationCheck,
    ImprovementPatchCandidateVerificationCheckResult,
    ImprovementPatchCandidateVerificationPlan,
    ImprovementPatchCandidateVerificationPlanner,
    ImprovementPatchCandidateVerificationRecord,
    ImprovementPatchCandidateVerificationRequest,
    ImprovementPatchCandidateVerificationResult,
    ImprovementPatchCandidateVerificationService,
    ImprovementPatchCandidateVerificationValidationError,
    ImprovementPatchOperation,
    ImprovementTarget,
    improvement_patch_candidate_verification_check_from_dict,
    improvement_patch_candidate_verification_check_result_from_dict,
    improvement_patch_candidate_verification_check_result_to_dict,
    improvement_patch_candidate_verification_check_to_dict,
    improvement_patch_candidate_verification_plan_from_dict,
    improvement_patch_candidate_verification_plan_to_dict,
    improvement_patch_candidate_verification_record_from_dict,
    improvement_patch_candidate_verification_record_to_dict,
    improvement_patch_candidate_verification_record_to_evaluation,
    improvement_patch_candidate_verification_request_from_dict,
    improvement_patch_candidate_verification_request_to_dict,
    improvement_patch_candidate_verification_result_from_dict,
    improvement_patch_candidate_verification_result_to_dict,
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


def _approval() -> ImprovementPatchCandidateVerificationApproval:
    return ImprovementPatchCandidateVerificationApproval(
        approved_permissions=("shell", "filesystem_mutation"),
        reason="Run local verification through injected fake verifier.",
        approved_by="tester",
        created_at=610.0,
        metadata={"approved": True},
    )


def _check_results(
    *,
    status: str = "pass",
    include_optional: bool = True,
) -> tuple[ImprovementPatchCandidateVerificationCheckResult, ...]:
    results = [
        ImprovementPatchCandidateVerificationCheckResult(
            check_id="ipcvchk_checks_1",
            status=status,  # type: ignore[arg-type]
            summary=f"primary check {status}",
            changed_files=(
                "src/raygent_harness/improvement/candidate_verification.py",
            ),
            output_excerpt="primary verification output",
            output_reference="task-output:verification-primary",
            metadata={"rank": 1},
        ),
        ImprovementPatchCandidateVerificationCheckResult(
            check_id="ipcvchk_non_regression_checks_1",
            status="pass",
            summary="non-regression check passed",
            output_reference="task-output:verification-non-regression",
        ),
    ]
    if include_optional:
        results.append(
            ImprovementPatchCandidateVerificationCheckResult(
                check_id="ipcvchk_cost_checks_1",
                status="warn",
                summary="optional cost check warned",
                output_reference="task-output:verification-cost",
            )
        )
    return tuple(results)


@dataclass
class FakeVerifier:
    result: ImprovementPatchCandidateVerificationResult = field(
        default_factory=lambda: ImprovementPatchCandidateVerificationResult(
            runner_ref="local:pytest",
            runner_kind="local_test",
            results=_check_results(),
            summary="Verification completed through fake verifier.",
            metadata={"runner": "fake"},
        )
    )
    requests: list[ImprovementPatchCandidateVerificationRequest] = field(
        default_factory=lambda: []
    )

    async def verify(
        self,
        request: ImprovementPatchCandidateVerificationRequest,
    ) -> ImprovementPatchCandidateVerificationResult:
        self.requests.append(request)
        return self.result


async def _verification_record(
    *,
    verifier: FakeVerifier | None = None,
    plan: ImprovementPatchCandidateVerificationPlan | None = None,
) -> tuple[ImprovementPatchCandidateVerificationRecord, FakeVerifier]:
    selected_verifier = verifier or FakeVerifier()
    record = await ImprovementPatchCandidateVerificationService(
        clock=lambda: 700.0,
        verification_id_factory=lambda: "ipcv_1",
    ).verify(
        plan or _verification_plan(),
        verifier=selected_verifier,
        approval=_approval(),
        metadata={"phase": "verification-record"},
    )
    return record, selected_verifier


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


def test_verification_approval_requires_exact_local_permissions() -> None:
    approval = ImprovementPatchCandidateVerificationApproval(
        approved_permissions=("filesystem_mutation", "shell"),
        reason="approved local verification",
        approved_by="tester",
    )

    assert approval.approved_permissions == ("filesystem_mutation", "shell")

    for permissions in (
        ("shell",),
        ("filesystem_mutation", "shell", "network"),
        ("filesystem_mutation", "shell", "commit"),
        ("filesystem_mutation", "shell", "worktree"),
        ("none",),
    ):
        with pytest.raises(ImprovementPatchCandidateVerificationValidationError):
            ImprovementPatchCandidateVerificationApproval(
                approved_permissions=permissions,  # type: ignore[arg-type]
                reason="invalid",
                approved_by="tester",
            )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="approved",
    ):
        ImprovementPatchCandidateVerificationApproval(
            approved_permissions=("filesystem_mutation", "shell"),
            reason="denied",
            approved_by="tester",
            approved=False,
        )


@pytest.mark.asyncio
async def test_verification_service_invokes_injected_verifier_once() -> None:
    record, verifier = await _verification_record()

    assert len(verifier.requests) == 1
    request = verifier.requests[0]
    assert request.verification_plan_id == "ipcvp_1"
    assert request.materialization_id == "ipcm_1"
    assert request.allocation_id == "ipcw_1"
    assert request.candidate_id == "ipc_1"
    assert request.run_id == "ir_1"
    assert request.proposal_id == "ip_1"
    assert request.gate_evaluation_id == "ige_1"
    assert request.base_revision == "7996d86"
    assert request.patch_digest == "sha256:" + "5" * 64
    assert request.allowed_changed_files == (
        "src/raygent_harness/improvement/candidate_verification.py",
    )
    assert [check.check_id for check in request.checks] == [
        "ipcvchk_checks_1",
        "ipcvchk_non_regression_checks_1",
        "ipcvchk_cost_checks_1",
    ]
    assert request.metadata == {
        "phase": "verification-record",
    }

    assert record.verification_id == "ipcv_1"
    assert record.status == "verification_recorded"
    assert record.runner_ref == "local:pytest"
    assert record.runner_kind == "local_test"
    assert record.summary == "Verification completed through fake verifier."
    assert record.metadata == {
        "phase": "verification-record",
        "verification_result_metadata": {"runner": "fake"},
    }
    assert record.verification_digest.startswith("sha256:")


@pytest.mark.asyncio
async def test_verification_service_requires_verifier_and_approval() -> None:
    service = ImprovementPatchCandidateVerificationService()
    plan = _verification_plan()

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="Verifier",
    ):
        await service.verify(plan, verifier=None, approval=_approval())

    verifier = FakeVerifier()
    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="approval",
    ):
        await service.verify(plan, verifier=verifier, approval=None)

    assert verifier.requests == []


@pytest.mark.asyncio
async def test_verification_service_rejects_reserved_request_metadata_before_call() -> None:
    verifier = FakeVerifier()

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="verification_result_metadata",
    ):
        await ImprovementPatchCandidateVerificationService().verify(
            _verification_plan(),
            verifier=verifier,
            approval=_approval(),
            metadata={"verification_result_metadata": {"reserved": True}},
        )

    assert verifier.requests == []


@pytest.mark.asyncio
async def test_verification_service_rejects_unknown_duplicate_and_missing_results() -> None:
    for results, match in (
        (
            (
                *_check_results(),
                ImprovementPatchCandidateVerificationCheckResult(
                    check_id="unknown",
                    status="pass",
                    summary="unknown check",
                ),
            ),
            "unknown",
        ),
        (_check_results(include_optional=False), "missing"),
    ):
        verifier = FakeVerifier(
            result=ImprovementPatchCandidateVerificationResult(
                runner_ref="local:pytest",
                runner_kind="local_test",
                results=results,
                summary="invalid results",
            )
        )
        with pytest.raises(
            ImprovementPatchCandidateVerificationValidationError,
            match=match,
        ):
            await _verification_record(verifier=verifier)

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="duplicate",
    ):
        ImprovementPatchCandidateVerificationResult(
            runner_ref="local:pytest",
            runner_kind="local_test",
            results=(
                *_check_results(),
                ImprovementPatchCandidateVerificationCheckResult(
                    check_id="ipcvchk_checks_1",
                    status="pass",
                    summary="duplicate check",
                ),
            ),
            summary="duplicate results",
        )


@pytest.mark.asyncio
async def test_verification_service_rejects_changed_files_outside_materialization() -> None:
    verifier = FakeVerifier(
        result=ImprovementPatchCandidateVerificationResult(
            runner_ref="local:pytest",
            runner_kind="local_test",
            results=(
                ImprovementPatchCandidateVerificationCheckResult(
                    check_id="ipcvchk_checks_1",
                    status="pass",
                    summary="primary check passed",
                    changed_files=(
                        "tests/unit/improvement/test_candidate_verification.py",
                    ),
                    output_reference="task-output:verification-primary",
                ),
                *_check_results()[1:],
            ),
            summary="invalid changed files",
        )
    )

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="allowed_changed_files",
    ):
        await _verification_record(verifier=verifier)


def test_verification_result_rejects_raw_references_and_unsafe_changed_files() -> None:
    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="raw output",
    ):
        ImprovementPatchCandidateVerificationResult(
            runner_ref="diff --git a/file b/file",
            runner_kind="local_test",
            results=_check_results(),
            summary="bad runner ref",
        )

    with pytest.raises(ImprovementPatchCandidateVerificationValidationError):
        ImprovementPatchCandidateVerificationCheckResult(
            check_id="ipcvchk_checks_1",
            status="pass",
            summary="unsafe changed file",
            changed_files=("../outside.py",),
        )


@pytest.mark.asyncio
async def test_verification_request_result_and_record_round_trip_through_json() -> None:
    record, verifier = await _verification_record()
    request_snapshot = improvement_patch_candidate_verification_request_to_dict(
        verifier.requests[0]
    )
    result_snapshot = improvement_patch_candidate_verification_result_to_dict(
        verifier.result
    )
    check_result_snapshot = (
        improvement_patch_candidate_verification_check_result_to_dict(
            verifier.result.results[0]
        )
    )
    record_snapshot = improvement_patch_candidate_verification_record_to_dict(record)

    json.dumps(request_snapshot)
    json.dumps(result_snapshot)
    json.dumps(check_result_snapshot)
    json.dumps(record_snapshot)

    assert improvement_patch_candidate_verification_request_from_dict(
        request_snapshot
    ) == verifier.requests[0]
    assert improvement_patch_candidate_verification_result_from_dict(
        result_snapshot
    ) == verifier.result
    assert improvement_patch_candidate_verification_check_result_from_dict(
        check_result_snapshot
    ) == verifier.result.results[0]
    assert improvement_patch_candidate_verification_record_from_dict(
        record_snapshot
    ) == record


@pytest.mark.asyncio
async def test_verification_digest_identity_excludes_descriptive_fields() -> None:
    record, _ = await _verification_record()
    snapshot = improvement_patch_candidate_verification_record_to_dict(record)
    snapshot["summary"] = "Changed descriptive summary."
    snapshot["created_at"] = 701.0
    result_snapshots = snapshot["results"]
    assert isinstance(result_snapshots, list)
    first_result = cast(dict[str, object], result_snapshots[0])
    assert isinstance(first_result, dict)
    first_result["summary"] = "Changed result summary."
    first_result["output_excerpt"] = "Changed output excerpt."
    first_result["metadata"] = {"changed": True}

    changed_record = improvement_patch_candidate_verification_record_from_dict(snapshot)

    assert changed_record.verification_digest == record.verification_digest
    assert changed_record.summary == "Changed descriptive summary."

    broken_snapshot = improvement_patch_candidate_verification_record_to_dict(record)
    broken_results = broken_snapshot["results"]
    assert isinstance(broken_results, list)
    broken_first_result = cast(dict[str, object], broken_results[0])
    assert isinstance(broken_first_result, dict)
    broken_first_result["output_reference"] = "task-output:changed-reference"

    with pytest.raises(
        ImprovementPatchCandidateVerificationValidationError,
        match="verification_digest",
    ):
        improvement_patch_candidate_verification_record_from_dict(broken_snapshot)


@pytest.mark.asyncio
async def test_verification_record_converts_to_existing_candidate_evaluation() -> None:
    record, _ = await _verification_record()

    evaluation = improvement_patch_candidate_verification_record_to_evaluation(
        record,
        evaluation_id="ipce_verified",
        created_at=710.0,
    )

    assert isinstance(evaluation, ImprovementPatchCandidateEvaluation)
    assert evaluation.evaluation_id == "ipce_verified"
    assert evaluation.materialization_id == record.materialization_id
    assert evaluation.allocation_id == record.allocation_id
    assert evaluation.candidate_id == record.candidate_id
    assert evaluation.run_id == record.run_id
    assert evaluation.proposal_id == record.proposal_id
    assert evaluation.gate_evaluation_id == record.gate_evaluation_id
    assert evaluation.decision == "warn"
    assert [result.required for result in evaluation.results] == [True, True, False]
    assert evaluation.metadata == {
        "verification_id": "ipcv_1",
        "verification_digest": record.verification_digest,
    }
    assert evaluation.results[0].metadata["verification_id"] == "ipcv_1"
    assert evaluation.results[0].metadata["source_plan_section"] == "checks"

    outcome = ImprovementPatchCandidateOutcomePolicy(
        clock=lambda: 720.0,
        outcome_id_factory=lambda: "ipco_verified",
    ).decide(
        _materialization(),
        evaluation,
        metadata={"phase": "verification-outcome"},
    )

    assert outcome.outcome_id == "ipco_verified"
    assert outcome.evaluation_id == "ipce_verified"
    assert outcome.evaluation_decision == "warn"
    assert outcome.decision == "needs_review"
    assert outcome.archive_recommended is True


@pytest.mark.asyncio
async def test_verification_record_preserves_needs_review_status_for_outcome_policy() -> None:
    verifier = FakeVerifier(
        result=ImprovementPatchCandidateVerificationResult(
            runner_ref="local:pytest",
            runner_kind="local_test",
            results=_check_results(status="needs_review"),
            summary="Verification needs manual review.",
        )
    )
    record, _ = await _verification_record(verifier=verifier)

    assert record.results[0].status == "needs_review"

    evaluation = improvement_patch_candidate_verification_record_to_evaluation(record)
    assert evaluation.decision == "needs_review"

    outcome = ImprovementPatchCandidateOutcomePolicy(
        clock=lambda: 720.0,
        outcome_id_factory=lambda: "ipco_needs_review",
    ).decide(_materialization(), evaluation)

    assert outcome.evaluation_decision == "needs_review"
    assert outcome.decision == "needs_review"


@pytest.mark.asyncio
async def test_verification_record_to_evaluation_rejects_reserved_metadata() -> None:
    record, _ = await _verification_record()

    for key in ("verification_id", "verification_digest"):
        with pytest.raises(
            ImprovementPatchCandidateVerificationValidationError,
            match=key,
        ):
            improvement_patch_candidate_verification_record_to_evaluation(
                record,
                metadata={key: "caller-owned"},
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


def test_candidate_verification_module_exports_public_rsi_005_symbols() -> None:
    assert set(verification_module.__all__) == {
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
    }
