from __future__ import annotations

import ast
import inspect
import json

import pytest

import raygent_harness.improvement.candidates as candidate_module
from raygent_harness.improvement import (
    ImprovementDiagnosis,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementGateEvaluation,
    ImprovementGatePolicy,
    ImprovementGateResult,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidatePlanner,
    ImprovementPatchCandidateValidationError,
    ImprovementProposal,
    ImprovementRun,
    ImprovementTarget,
    improvement_patch_candidate_plan_from_dict,
    improvement_patch_candidate_plan_to_dict,
)


def _target(kind: str = "source_code") -> ImprovementTarget:
    return ImprovementTarget(
        target_id="src/raygent_harness/improvement/candidates.py",
        kind=kind,  # type: ignore[arg-type]
        description="Improvement candidate planning records",
        owner="kernel",
        metadata={"component": "improvement"},
    )


def _evidence() -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id="ev_1",
        source="verification",
        summary="The proposal has reviewable gate evidence.",
        excerpt="Gate evaluation passed before candidate planning.",
        created_at=100.0,
    )


def _proposal(
    target: ImprovementTarget,
    evidence: ImprovementEvidence,
    *,
    permissions: tuple[str, ...] = ("filesystem_mutation", "worktree"),
) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="ip_1",
        target=target,
        diagnosis=ImprovementDiagnosis(
            summary="Patch candidate records need an explicit plan boundary.",
            symptoms=("later mutation stages need reviewable inputs",),
            hypotheses=("data-only plans keep mutation authority separate",),
            confidence=0.8,
        ),
        hypothesis="Candidate plans can safely prepare later isolated patches.",
        proposed_change="Add data-only patch candidate planning records.",
        intended_behavior_change="Later patch services consume explicit candidate plans.",
        expected_benefit="Clearer permission and review boundaries before mutation.",
        risks=("Candidate records could be mistaken for authority.",),
        required_permissions=permissions,  # type: ignore[arg-type]
        evaluation_plan=ImprovementEvaluationPlan(
            checks=(
                ImprovementEvaluationCheck(
                    name="candidate-plan",
                    instruction="Verify the plan stays data-only.",
                ),
            ),
            success_criteria=("Candidate plans are serializable and non-mutating.",),
        ),
        rollback_plan="Discard the candidate plan record.",
        stop_condition="Stop after planning one data-only candidate.",
        evidence_ids=(evidence.evidence_id,),
        created_at=101.0,
    )


def _run(*, permissions: tuple[str, ...] = ("filesystem_mutation", "worktree")) -> ImprovementRun:
    target = _target()
    evidence = _evidence()
    proposal = _proposal(target, evidence, permissions=permissions)
    return ImprovementRun(
        run_id="ir_1",
        status="proposed",
        target=target,
        evidence=(evidence,),
        proposal=proposal,
        created_at=102.0,
        updated_at=103.0,
    )


def _gate(
    gate_id: str,
    kind: str,
    status: str,
    *,
    evidence_ids: tuple[str, ...] = (),
    reviewer: str | None = None,
) -> ImprovementGateResult:
    return ImprovementGateResult(
        gate_id=gate_id,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        summary=f"{kind} is {status}",
        evidence_ids=evidence_ids,
        reviewer=reviewer,
        created_at=200.0,
    )


def _pass_evaluation(run: ImprovementRun) -> ImprovementGateEvaluation:
    return ImprovementGatePolicy(
        clock=lambda: 300.0,
        evaluation_id_factory=lambda: "ige_1",
    ).evaluate(
        run,
        (
            _gate("gate_evidence", "evidence_bounds", "pass", evidence_ids=("ev_1",)),
            _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            _gate("gate_mutation", "no_mutation", "pass"),
        ),
    )


def _candidate_plan() -> ImprovementPatchCandidatePlan:
    run = _run()
    return ImprovementPatchCandidatePlanner(
        clock=lambda: 400.0,
        candidate_id_factory=lambda: "ipc_1",
    ).plan(
        run,
        _pass_evaluation(run),
        base_revision="b5a011f",
        summary="Plan one data-only candidate.",
        planned_changes=("Add candidate records.",),
        expected_files=("src/raygent_harness/improvement/candidates.py",),
        metadata={"phase": "rsi-003a"},
    )


def test_patch_candidate_plan_serializes_to_json_and_round_trips() -> None:
    plan = _candidate_plan()

    snapshot = improvement_patch_candidate_plan_to_dict(plan)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_plan_from_dict(snapshot)

    assert restored == plan
    assert restored.status == "planned"
    assert restored.required_permissions == ("filesystem_mutation", "worktree")
    assert restored.metadata == {"phase": "rsi-003a"}


def test_planner_requires_passing_gate_and_preserves_linkage() -> None:
    plan = _candidate_plan()

    assert plan.candidate_id == "ipc_1"
    assert plan.run_id == "ir_1"
    assert plan.proposal_id == "ip_1"
    assert plan.gate_evaluation_id == "ige_1"
    assert plan.status == "planned"
    assert plan.evaluation_plan.checks[0].name == "candidate-plan"


def test_planner_rejects_failed_or_review_pending_gate() -> None:
    run = _run()
    failed = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_fail").evaluate(
        run,
        (
            _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            _gate("gate_mutation", "no_mutation", "fail"),
        ),
    )
    needs_review = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_nr").evaluate(
        run,
        (_gate("gate_mutation", "no_mutation", "pass"),),
    )
    planner = ImprovementPatchCandidatePlanner(candidate_id_factory=lambda: "ipc_1")

    for evaluation in (failed, needs_review):
        with pytest.raises(ImprovementPatchCandidateValidationError, match="passing gate"):
            planner.plan(
                run,
                evaluation,
                base_revision="b5a011f",
                summary="Plan one candidate.",
                planned_changes=("Add records.",),
                expected_files=("src/file.py",),
            )


def test_planner_rejects_mismatched_gate_linkage() -> None:
    run = _run()
    evaluation = _pass_evaluation(run)
    mismatched_run = ImprovementGateEvaluation(
        evaluation_id=evaluation.evaluation_id,
        run_id="ir_other",
        proposal_id=evaluation.proposal_id,
        results=evaluation.results,
        manual_review_required=False,
    )
    mismatched_proposal = ImprovementGateEvaluation(
        evaluation_id=evaluation.evaluation_id,
        run_id=evaluation.run_id,
        proposal_id="ip_other",
        results=evaluation.results,
        manual_review_required=False,
    )
    planner = ImprovementPatchCandidatePlanner(candidate_id_factory=lambda: "ipc_1")

    with pytest.raises(ImprovementPatchCandidateValidationError, match="run_id"):
        planner.plan(
            run,
            mismatched_run,
            base_revision="b5a011f",
            summary="Plan one candidate.",
            planned_changes=("Add records.",),
            expected_files=("src/file.py",),
        )
    with pytest.raises(ImprovementPatchCandidateValidationError, match="proposal_id"):
        planner.plan(
            run,
            mismatched_proposal,
            base_revision="b5a011f",
            summary="Plan one candidate.",
            planned_changes=("Add records.",),
            expected_files=("src/file.py",),
        )


def test_candidate_permissions_cannot_be_none_or_missing_source_requirements() -> None:
    run = _run(permissions=("none",))
    evaluation = _pass_evaluation(run)
    planner = ImprovementPatchCandidatePlanner(candidate_id_factory=lambda: "ipc_1")

    with pytest.raises(ImprovementPatchCandidateValidationError, match="cannot include none"):
        planner.plan(
            run,
            evaluation,
            base_revision="b5a011f",
            summary="Plan one candidate.",
            planned_changes=("Add records.",),
            expected_files=("src/file.py",),
        )

    with pytest.raises(ImprovementPatchCandidateValidationError, match="missing"):
        ImprovementPatchCandidatePlan(
            candidate_id="ipc_bad",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            target=_target(),
            base_revision="b5a011f",
            summary="Plan one candidate.",
            planned_changes=("Add records.",),
            expected_files=("src/file.py",),
            required_permissions=("filesystem_mutation",),
            evaluation_plan=run.proposal.evaluation_plan,
            rollback_plan="Discard record.",
        )


def test_planner_rejects_empty_rollback_override() -> None:
    run = _run()
    planner = ImprovementPatchCandidatePlanner(candidate_id_factory=lambda: "ipc_1")

    with pytest.raises(ValueError, match="rollback_plan"):
        planner.plan(
            run,
            _pass_evaluation(run),
            base_revision="b5a011f",
            summary="Plan one candidate.",
            planned_changes=("Add records.",),
            expected_files=("src/file.py",),
            rollback_plan=" ",
        )


def test_candidate_deserialization_rejects_non_planned_status() -> None:
    snapshot = improvement_patch_candidate_plan_to_dict(_candidate_plan())
    snapshot["status"] = "materialized"

    with pytest.raises(ValueError, match="status"):
        improvement_patch_candidate_plan_from_dict(snapshot)


def test_candidate_module_has_only_data_layer_imports() -> None:
    source = inspect.getsource(candidate_module)
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
        "raygent_harness.services.worktree",
        "raygent_harness.core.permission_engine",
        "raygent_harness.sdk",
        "subprocess",
        "os",
        "pathlib",
    }
    forbidden_names = {
        "WorktreeManager",
        "GitWorktreeManager",
        "PermissionHandler",
        "PermissionRequest",
        "create_raygent",
        "Path",
    }

    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
