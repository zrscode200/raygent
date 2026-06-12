from __future__ import annotations

import ast
import inspect
import json

import pytest

import raygent_harness.improvement.candidate_outcome as outcome_module
from raygent_harness.improvement import (
    DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS,
    DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES,
    DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS,
    DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS,
    DEFAULT_MAX_OUTCOME_BLOCKER_CHARS,
    DEFAULT_MAX_OUTCOME_BLOCKERS,
    DEFAULT_MAX_OUTCOME_METADATA_CHARS,
    DEFAULT_MAX_OUTCOME_SUMMARY_CHARS,
    ImprovementPatchCandidateArchiveDecisionPolicy,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationResult,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidateOutcome,
    ImprovementPatchCandidateOutcomePolicy,
    ImprovementPatchCandidateOutcomeValidationError,
    ImprovementPatchOperation,
    improvement_patch_candidate_archive_decision_from_dict,
    improvement_patch_candidate_archive_decision_to_dict,
    improvement_patch_candidate_outcome_from_dict,
    improvement_patch_candidate_outcome_to_dict,
)


def _operation() -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id="op_1",
        kind="replace_text",
        relative_path="src/raygent_harness/improvement/candidate_outcome.py",
        old_text="old",
        new_text="new",
    )


def _materialization() -> ImprovementPatchCandidateMaterialization:
    operation = _operation()
    return ImprovementPatchCandidateMaterialization(
        materialization_id="ipcm_1",
        allocation_id="ipcw_1",
        candidate_id="ipc_1",
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        base_revision="a7ede0b",
        worktree_path="/tmp/raygent/ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        operations=(operation,),
        changed_files=(operation.relative_path,),
        patch_digest="sha256:" + "1" * 64,
        created_at=500.0,
        metadata={"phase": "rsi-003c"},
    )


def _evaluation_result(
    result_id: str,
    status: str,
    *,
    required: bool = True,
) -> ImprovementPatchCandidateEvaluationResult:
    return ImprovementPatchCandidateEvaluationResult(
        result_id=result_id,
        kind="unit_tests",
        status=status,  # type: ignore[arg-type]
        summary=f"unit tests {status}",
        changed_files=("src/raygent_harness/improvement/candidate_outcome.py",),
        output_reference="task-output:test-candidate-outcome",
        required=required,
        created_at=600.0,
    )


def _evaluation(
    results: tuple[ImprovementPatchCandidateEvaluationResult, ...],
    *,
    materialization_id: str = "ipcm_1",
) -> ImprovementPatchCandidateEvaluation:
    materialization = _materialization()
    return ImprovementPatchCandidateEvaluation(
        evaluation_id="ipce_1",
        materialization_id=materialization_id,
        allocation_id=materialization.allocation_id,
        candidate_id=materialization.candidate_id,
        run_id=materialization.run_id,
        proposal_id=materialization.proposal_id,
        gate_evaluation_id=materialization.gate_evaluation_id,
        results=results,
        created_at=700.0,
    )


def _outcome(decision: str = "promotable") -> ImprovementPatchCandidateOutcome:
    materialization = _materialization()
    evaluation = _evaluation((_evaluation_result("res_pass", "pass"),))
    return ImprovementPatchCandidateOutcomePolicy(
        clock=lambda: 800.0,
        outcome_id_factory=lambda: "ipco_1",
    ).decide(
        materialization,
        evaluation,
        decision=decision,  # type: ignore[arg-type]
        summary=(
            "Manual rejection for later archive."
            if decision == "reject"
            else "Evaluation passed."
        ),
        promotion_blockers=(
            ("manual rejection",) if decision == "reject" else None
        ),
    )


def test_outcome_policy_derives_promotable_pass_outcome_and_round_trips() -> None:
    materialization = _materialization()
    evaluation = _evaluation((_evaluation_result("res_pass", "pass"),))

    outcome = ImprovementPatchCandidateOutcomePolicy(
        clock=lambda: 800.0,
        outcome_id_factory=lambda: "ipco_1",
    ).decide(materialization, evaluation, metadata={"phase": "rsi-004a"})

    assert outcome.outcome_id == "ipco_1"
    assert outcome.materialization_id == materialization.materialization_id
    assert outcome.allocation_id == materialization.allocation_id
    assert outcome.candidate_id == materialization.candidate_id
    assert outcome.run_id == materialization.run_id
    assert outcome.proposal_id == materialization.proposal_id
    assert outcome.gate_evaluation_id == materialization.gate_evaluation_id
    assert outcome.base_revision == materialization.base_revision
    assert outcome.patch_digest == materialization.patch_digest
    assert outcome.evaluation_id == evaluation.evaluation_id
    assert outcome.evaluation_decision == "pass"
    assert outcome.decision == "promotable"
    assert outcome.reason == "evaluation_passed"
    assert outcome.required_permissions == (
        "human_review",
        "filesystem_mutation",
        "commit",
    )
    assert outcome.archive_recommended is False
    assert outcome.promotion_blockers == ()
    assert outcome.created_at == 800.0

    snapshot = improvement_patch_candidate_outcome_to_dict(outcome)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_outcome_from_dict(snapshot)

    assert restored == outcome
    assert restored.metadata == {"phase": "rsi-004a"}


def test_outcome_policy_derives_fail_warn_and_needs_review_decisions() -> None:
    materialization = _materialization()
    policy = ImprovementPatchCandidateOutcomePolicy(outcome_id_factory=lambda: "ipco_1")

    failed = policy.decide(
        materialization,
        _evaluation((_evaluation_result("res_fail", "fail"),)),
    )
    assert failed.decision == "reject"
    assert failed.reason == "evaluation_failed"
    assert failed.archive_recommended is True
    assert failed.promotion_blockers == ("evaluation failed",)

    warned = policy.decide(
        materialization,
        _evaluation((_evaluation_result("res_warn", "warn"),)),
    )
    assert warned.decision == "needs_review"
    assert warned.reason == "evaluation_warned"
    assert warned.archive_recommended is True
    assert warned.promotion_blockers == ("evaluation produced warnings",)

    needs_review = policy.decide(
        materialization,
        _evaluation((_evaluation_result("res_review", "needs_review"),)),
    )
    assert needs_review.decision == "needs_review"
    assert needs_review.reason == "evaluation_needs_review"
    assert needs_review.promotion_blockers == ("evaluation requires review",)


def test_outcome_policy_validates_materialization_evaluation_linkage() -> None:
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="materialization_id",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation(
                (_evaluation_result("res_pass", "pass"),),
                materialization_id="ipcm_other",
            ),
        )


def test_outcome_rejects_policy_inconsistent_direct_records_and_snapshots() -> None:
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="promotable",
    ):
        ImprovementPatchCandidateOutcome(
            outcome_id="ipco_bad",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="a7ede0b",
            patch_digest="sha256:" + "1" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="fail",
            decision="promotable",
            reason="evaluation_passed",
            summary="bad outcome",
            required_permissions=("human_review", "filesystem_mutation", "commit"),
            archive_recommended=False,
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="required_permissions",
    ):
        ImprovementPatchCandidateOutcome(
            outcome_id="ipco_missing_permission",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="a7ede0b",
            patch_digest="sha256:" + "1" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="pass",
            decision="promotable",
            reason="evaluation_passed",
            summary="missing commit permission",
            required_permissions=("human_review", "filesystem_mutation"),
            archive_recommended=False,
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="promotion_blockers",
    ):
        ImprovementPatchCandidateOutcome(
            outcome_id="ipco_no_blocker",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="a7ede0b",
            patch_digest="sha256:" + "1" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="fail",
            decision="reject",
            reason="evaluation_failed",
            summary="missing blocker",
            required_permissions=("human_review",),
            archive_recommended=True,
        )

    outcome = _outcome()
    snapshot = improvement_patch_candidate_outcome_to_dict(outcome)
    snapshot["evaluation_decision"] = "fail"
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="promotable",
    ):
        improvement_patch_candidate_outcome_from_dict(snapshot)


def test_archive_decision_policy_copies_outcome_linkage_and_round_trips() -> None:
    outcome = _outcome("reject")

    archive_decision = ImprovementPatchCandidateArchiveDecisionPolicy(
        clock=lambda: 900.0,
        archive_decision_id_factory=lambda: "ipcad_1",
    ).decide(
        outcome,
        artifact_references=("task-output:test-candidate-outcome",),
        metadata={"phase": "archive-decision"},
    )

    assert archive_decision.archive_decision_id == "ipcad_1"
    assert archive_decision.outcome_id == outcome.outcome_id
    assert archive_decision.materialization_id == outcome.materialization_id
    assert archive_decision.allocation_id == outcome.allocation_id
    assert archive_decision.candidate_id == outcome.candidate_id
    assert archive_decision.run_id == outcome.run_id
    assert archive_decision.proposal_id == outcome.proposal_id
    assert archive_decision.gate_evaluation_id == outcome.gate_evaluation_id
    assert archive_decision.base_revision == outcome.base_revision
    assert archive_decision.patch_digest == outcome.patch_digest
    assert archive_decision.evaluation_id == outcome.evaluation_id
    assert archive_decision.outcome_decision == "reject"
    assert archive_decision.archive_recommended is True
    assert archive_decision.failure_symptoms == ("manual rejection",)
    assert archive_decision.artifact_references == (
        "task-output:test-candidate-outcome",
    )
    assert archive_decision.created_at == 900.0

    snapshot = improvement_patch_candidate_archive_decision_to_dict(archive_decision)
    json.dumps(snapshot)
    restored = improvement_patch_candidate_archive_decision_from_dict(snapshot)

    assert restored == archive_decision
    assert restored.metadata == {"phase": "archive-decision"}


def test_archive_decision_rejects_inconsistent_recommendations_and_bad_references() -> None:
    outcome = _outcome()
    archive = ImprovementPatchCandidateArchiveDecisionPolicy(
        archive_decision_id_factory=lambda: "ipcad_1",
    ).decide(outcome)

    snapshot = improvement_patch_candidate_archive_decision_to_dict(archive)
    snapshot["archive_recommended"] = True
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="archive_recommended",
    ):
        improvement_patch_candidate_archive_decision_from_dict(snapshot)

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="single-line",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            _outcome("reject"),
            artifact_references=("line one\nline two",),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="raw output",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            _outcome("reject"),
            artifact_references=("diff --git a/file.py b/file.py",),
        )


def test_outcome_and_archive_bounds_are_enforced() -> None:
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="SUMMARY_CHARS",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_pass", "pass"),)),
            summary="x" * (DEFAULT_MAX_OUTCOME_SUMMARY_CHARS + 1),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="BLOCKERS",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_fail", "fail"),)),
            promotion_blockers=tuple(
                f"blocker-{index}" for index in range(DEFAULT_MAX_OUTCOME_BLOCKERS + 1)
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="BLOCKER_CHARS",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_fail", "fail"),)),
            promotion_blockers=("x" * (DEFAULT_MAX_OUTCOME_BLOCKER_CHARS + 1),),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="METADATA_CHARS",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_pass", "pass"),)),
            metadata={"large": "x" * (DEFAULT_MAX_OUTCOME_METADATA_CHARS + 1)},
        )

    reject_outcome = _outcome("reject")
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="FAILURE_SYMPTOMS",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            failure_symptoms=tuple(
                f"symptom-{index}"
                for index in range(DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS + 1)
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="FAILURE_SYMPTOM_CHARS",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            failure_symptoms=("x" * (DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS + 1),),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="ARTIFACT_REFERENCES",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            artifact_references=tuple(
                f"task-output:{index}"
                for index in range(DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES + 1)
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="ARTIFACT_REFERENCE_CHARS",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            artifact_references=(
                "x" * (DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS + 1),
            ),
        )


def test_outcome_and_archive_reject_empty_or_duplicate_bounded_items() -> None:
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="duplicates",
    ):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_fail", "fail"),)),
            promotion_blockers=("same", "same"),
        )

    with pytest.raises(ValueError, match="non-empty"):
        ImprovementPatchCandidateOutcomePolicy().decide(
            _materialization(),
            _evaluation((_evaluation_result("res_fail", "fail"),)),
            promotion_blockers=("",),
        )

    reject_outcome = _outcome("reject")
    with pytest.raises(
        ImprovementPatchCandidateOutcomeValidationError,
        match="duplicates",
    ):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            failure_symptoms=("same", "same"),
        )

    with pytest.raises(ValueError, match="non-empty"):
        ImprovementPatchCandidateArchiveDecisionPolicy().decide(
            reject_outcome,
            artifact_references=("",),
        )


def test_candidate_outcome_module_is_data_only_and_imports_no_runtime_authority() -> None:
    source = inspect.getsource(outcome_module)
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
        "os",
        "pathlib",
        "subprocess",
        "raygent_harness.core.permission_engine",
        "raygent_harness.core.streaming_tool_executor",
        "raygent_harness.core.tool_execution",
        "raygent_harness.core.tool_orchestration",
        "raygent_harness.sdk",
        "raygent_harness.services.worktree.manager",
        "raygent_harness.tools.bash_tool",
        "raygent_harness.tools.file_edit_tool",
        "raygent_harness.tools.file_text_utils",
    }
    forbidden_names = {
        "GitWorktreeManager",
        "PermissionHandler",
        "PermissionRequest",
        "Path",
        "create_raygent",
        "write_text",
    }

    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)


def test_candidate_outcome_module_exports_only_public_rsi_004a_symbols() -> None:
    assert "json" not in outcome_module.__all__
    assert "time" not in outcome_module.__all__
    assert "Callable" not in outcome_module.__all__
    assert "Mapping" not in outcome_module.__all__
    assert "Sequence" not in outcome_module.__all__
    assert set(outcome_module.__all__) == {
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCES",
        "DEFAULT_MAX_ARCHIVE_ARTIFACT_REFERENCE_CHARS",
        "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOMS",
        "DEFAULT_MAX_ARCHIVE_FAILURE_SYMPTOM_CHARS",
        "DEFAULT_MAX_OUTCOME_BLOCKERS",
        "DEFAULT_MAX_OUTCOME_BLOCKER_CHARS",
        "DEFAULT_MAX_OUTCOME_METADATA_CHARS",
        "DEFAULT_MAX_OUTCOME_SUMMARY_CHARS",
        "ImprovementPatchCandidateArchiveDecision",
        "ImprovementPatchCandidateArchiveDecisionPolicy",
        "ImprovementPatchCandidateOutcome",
        "ImprovementPatchCandidateOutcomeDecision",
        "ImprovementPatchCandidateOutcomeError",
        "ImprovementPatchCandidateOutcomePolicy",
        "ImprovementPatchCandidateOutcomeReason",
        "ImprovementPatchCandidateOutcomeValidationError",
        "improvement_patch_candidate_archive_decision_from_dict",
        "improvement_patch_candidate_archive_decision_to_dict",
        "improvement_patch_candidate_outcome_from_dict",
        "improvement_patch_candidate_outcome_to_dict",
    }
