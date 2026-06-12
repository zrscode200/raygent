from __future__ import annotations

import json
from pathlib import Path

import pytest

from raygent_harness.improvement import (
    ImprovementDiagnosis,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementGateEvaluation,
    ImprovementGatePolicy,
    ImprovementGateResult,
    ImprovementGateValidationError,
    ImprovementProposal,
    ImprovementRun,
    ImprovementTarget,
    improvement_gate_evaluation_from_dict,
    improvement_gate_evaluation_to_dict,
)


def _target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="preset.project_reader",
        kind="preset",
        description="Project-reader preset behavior",
    )


def _evidence(evidence_id: str = "ev_1") -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id=evidence_id,
        source="verification",
        summary="Focused verification passed.",
        excerpt="tests/unit/improvement passed",
        created_at=100.0,
    )


def _proposal(
    target: ImprovementTarget,
    evidence: ImprovementEvidence,
    *,
    manual_review_required: bool = True,
) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="ip_1",
        target=target,
        diagnosis=ImprovementDiagnosis(
            summary="Proposal needs explicit gates before mutation.",
            symptoms=("model generation would otherwise self-report progress",),
            hypotheses=("gate records make review state explicit",),
        ),
        hypothesis="Gate records reduce unsafe continuation.",
        proposed_change="Add gate result and aggregate gate evaluation records.",
        intended_behavior_change="Improvement runs stop unless supplied gates pass.",
        expected_benefit="Safer future model-backed proposal generation.",
        risks=("Gate policy could pass with weak evidence.",),
        required_permissions=("none",),
        evaluation_plan=ImprovementEvaluationPlan(
            checks=(
                ImprovementEvaluationCheck(
                    name="gate-policy",
                    instruction="Verify required gates fail closed.",
                ),
            ),
            manual_review_required=manual_review_required,
            success_criteria=("Gate decisions are derived from supplied results.",),
        ),
        rollback_plan="Do not use the gate evaluation record.",
        stop_condition="Stop after deriving the gate decision.",
        evidence_ids=(evidence.evidence_id,),
        created_at=101.0,
    )


def _run(*, manual_review_required: bool = True) -> ImprovementRun:
    target = _target()
    evidence = _evidence()
    proposal = _proposal(
        target,
        evidence,
        manual_review_required=manual_review_required,
    )
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
    required: bool = True,
    reviewer: str | None = None,
) -> ImprovementGateResult:
    return ImprovementGateResult(
        gate_id=gate_id,
        kind=kind,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        summary=f"{kind} is {status}",
        evidence_ids=evidence_ids,
        required=required,
        reviewer=reviewer,
        created_at=200.0,
        metadata={"source": "unit-test"},
    )


def _pass_gates(evidence_id: str = "ev_1") -> tuple[ImprovementGateResult, ...]:
    return (
        _gate(
            "gate_evidence",
            "evidence_bounds",
            "pass",
            evidence_ids=(evidence_id,),
        ),
        _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
        _gate("gate_mutation", "no_mutation", "pass"),
    )


def test_gate_evaluation_serializes_to_json_and_round_trips() -> None:
    run = _run()
    evaluation = ImprovementGatePolicy(
        clock=lambda: 300.0,
        evaluation_id_factory=lambda: "ige_1",
    ).evaluate(run, _pass_gates(), metadata={"phase": "rsi-002a"})

    snapshot = improvement_gate_evaluation_to_dict(evaluation)
    json.dumps(snapshot)
    restored = improvement_gate_evaluation_from_dict(snapshot)

    assert restored == evaluation
    assert restored.decision == "pass"
    assert restored.run_id == run.run_id
    assert restored.proposal_id == run.proposal.proposal_id
    assert restored.results[0].metadata == {"source": "unit-test"}


def test_policy_derives_pass_when_required_gates_pass() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(),
        _pass_gates(),
    )

    assert evaluation.decision == "pass"
    assert evaluation.warnings == ()


def test_policy_derives_fail_for_required_failure() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(),
        (
            _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            _gate("gate_mutation", "no_mutation", "fail"),
        ),
    )

    assert evaluation.decision == "fail"


def test_policy_derives_needs_review_for_required_review_status() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(),
        (
            _gate("gate_review", "review_status", "needs_review"),
            _gate("gate_mutation", "no_mutation", "pass"),
        ),
    )

    assert evaluation.decision == "needs_review"


def test_policy_derives_needs_review_when_manual_review_gate_missing() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(),
        (
            _gate("gate_mutation", "no_mutation", "pass"),
            _gate("gate_rollback", "rollback_plan", "pass"),
        ),
    )

    assert evaluation.decision == "needs_review"
    assert evaluation.warnings == (
        "manual review is required but no required review gate was supplied",
    )


def test_policy_derives_warn_for_warn_status_and_optional_failure() -> None:
    warn_evaluation = ImprovementGatePolicy(
        evaluation_id_factory=lambda: "ige_warn",
    ).evaluate(
        _run(),
        (
            _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            _gate("gate_cost", "cost_budget", "warn", evidence_ids=("ev_1",)),
        ),
    )
    optional_fail_evaluation = ImprovementGatePolicy(
        evaluation_id_factory=lambda: "ige_optional_fail",
    ).evaluate(
        _run(),
        (
            _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            _gate("gate_other", "other", "fail", required=False),
        ),
    )

    assert warn_evaluation.decision == "warn"
    assert optional_fail_evaluation.decision == "warn"
    assert optional_fail_evaluation.warnings == ("one or more optional gates failed",)


def test_policy_derives_needs_review_for_optional_needs_review_gate() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(manual_review_required=False),
        (
            _gate("gate_mutation", "no_mutation", "pass"),
            _gate("gate_optional", "other", "needs_review", required=False),
        ),
    )

    assert evaluation.decision == "needs_review"


def test_policy_allows_not_applicable_required_gate() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(manual_review_required=False),
        (
            _gate("gate_review", "review_status", "not_applicable"),
            _gate("gate_mutation", "no_mutation", "pass"),
        ),
    )

    assert evaluation.decision == "pass"


def test_policy_needs_review_when_required_review_gate_is_not_applicable() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(manual_review_required=True),
        (
            _gate("gate_review", "review_status", "not_applicable"),
            _gate("gate_mutation", "no_mutation", "pass"),
        ),
    )

    assert evaluation.decision == "needs_review"
    assert evaluation.warnings == (
        "manual review is required but the review gate was not applicable",
    )


def test_policy_derives_needs_review_for_empty_gate_set() -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(manual_review_required=False),
        (),
    )

    assert evaluation.decision == "needs_review"
    assert evaluation.warnings == ("no gate results supplied",)


def test_gate_result_requires_evidence_for_evidence_dependent_gate() -> None:
    with pytest.raises(ValueError, match="evidence_ids"):
        _gate("gate_evidence", "evidence_bounds", "pass")


def test_policy_rejects_unknown_evidence_ids() -> None:
    with pytest.raises(ImprovementGateValidationError, match="unknown evidence"):
        ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
            _run(),
            (
                _gate(
                    "gate_evidence",
                    "evidence_bounds",
                    "pass",
                    evidence_ids=("ev_unknown",),
                ),
                _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
            ),
        )


def test_policy_rejects_duplicate_gate_ids() -> None:
    with pytest.raises(ImprovementGateValidationError, match="duplicate gate ids"):
        ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
            _run(),
            (
                _gate("gate_review", "review_status", "pass", reviewer="reviewer"),
                _gate("gate_review", "no_mutation", "pass"),
            ),
        )


def test_gate_evaluation_rejects_mismatched_literal_values() -> None:
    with pytest.raises(ValueError, match="kind"):
        _gate("gate_bad", "not_a_gate", "pass")

    with pytest.raises(ValueError, match="status"):
        _gate("gate_bad", "no_mutation", "unknown")

    with pytest.raises(ValueError, match="decision"):
        improvement_gate_evaluation_from_dict(
            {
                "evaluation_id": "ige_bad",
                "run_id": "ir_1",
                "proposal_id": "ip_1",
                "decision": "unknown",
                "results": [],
            }
        )


def test_gate_evaluation_rejects_policy_inconsistent_deserialization() -> None:
    with pytest.raises(ImprovementGateValidationError, match="derived gate policy"):
        improvement_gate_evaluation_from_dict(
            {
                "evaluation_id": "ige_bad",
                "run_id": "ir_1",
                "proposal_id": "ip_1",
                "decision": "pass",
                "manual_review_required": True,
                "results": [],
                "warnings": [],
            }
        )


def test_gate_evaluation_public_construction_derives_decision() -> None:
    evaluation = ImprovementGateEvaluation(
        evaluation_id="ige_1",
        run_id="ir_1",
        proposal_id="ip_1",
        results=(),
        manual_review_required=True,
    )

    assert evaluation.decision == "needs_review"
    assert evaluation.warnings == ("no gate results supplied",)


def test_gate_policy_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    evaluation = ImprovementGatePolicy(evaluation_id_factory=lambda: "ige_1").evaluate(
        _run(),
        _pass_gates(),
        metadata={"tmp": str(tmp_path)},
    )

    assert evaluation.decision == "pass"
    assert list(tmp_path.iterdir()) == []
