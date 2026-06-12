from __future__ import annotations

import json

import pytest

from raygent_harness.improvement import (
    ImprovementDiagnosis,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementProposal,
    ImprovementRun,
    ImprovementTarget,
    improvement_run_from_dict,
    improvement_run_to_dict,
)


def _target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="preset.project_reader",
        kind="preset",
        description="Project-reader preset behavior",
        owner="sdk",
        metadata={"component": "profiles"},
    )


def _evidence() -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id="ev_1",
        source="transcript",
        summary="User needed read-only project inspection.",
        excerpt="The session repeatedly asked for grep/read-only behavior.",
        source_uri="transcript://session/s_1#ev_1",
        created_at=100.0,
    )


def _proposal(target: ImprovementTarget, evidence: ImprovementEvidence) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="ip_1",
        target=target,
        diagnosis=ImprovementDiagnosis(
            summary="Preset explanation is not concrete enough.",
            symptoms=("user confusion about read-only surface",),
            hypotheses=("add clearer preset affordance text",),
            confidence=0.8,
        ),
        hypothesis="Clearer preset affordance text will reduce setup mistakes.",
        proposed_change="Clarify project_reader read-only behavior.",
        intended_behavior_change="Users choose read-only profiles before mutation.",
        expected_benefit="Lower risk of accidental filesystem authority.",
        risks=("Documentation may overpromise behavior.",),
        required_permissions=("none",),
        evaluation_plan=ImprovementEvaluationPlan(
            checks=(
                ImprovementEvaluationCheck(
                    name="doc-read",
                    instruction="Verify public docs describe read-only tools.",
                ),
            ),
            success_criteria=("No mutation-capable tools are implied.",),
        ),
        rollback_plan="Revert the wording-only proposal.",
        stop_condition="Stop after emitting one reviewable proposal.",
        evidence_ids=(evidence.evidence_id,),
        created_at=101.0,
    )


def test_improvement_run_serializes_to_json_and_round_trips() -> None:
    target = _target()
    evidence = _evidence()
    proposal = _proposal(target, evidence)
    run = ImprovementRun(
        run_id="ir_1",
        status="proposed",
        target=target,
        evidence=(evidence,),
        proposal=proposal,
        warnings=("proposal-only",),
        created_at=102.0,
        updated_at=103.0,
        metadata={"phase": "rsi-001"},
    )

    snapshot = improvement_run_to_dict(run)
    json.dumps(snapshot)
    restored = improvement_run_from_dict(snapshot)

    assert restored == run
    assert restored.proposal.stop_condition == "Stop after emitting one reviewable proposal."
    assert restored.evidence[0].metadata == {}


def test_improvement_proposal_requires_risks_and_evidence_ids() -> None:
    target = _target()
    evidence = _evidence()

    with pytest.raises(ValueError, match="risks"):
        ImprovementProposal(
            proposal_id="ip_bad",
            target=target,
            diagnosis=ImprovementDiagnosis(summary="Missing risk list"),
            hypothesis="A hypothesis",
            proposed_change="A change",
            intended_behavior_change="A behavior change",
            expected_benefit="A benefit",
            risks=(),
            required_permissions=("none",),
            evaluation_plan=ImprovementEvaluationPlan(
                checks=(ImprovementEvaluationCheck(name="check", instruction="Run it"),)
            ),
            rollback_plan="Roll back",
            stop_condition="Stop",
            evidence_ids=(evidence.evidence_id,),
        )

    with pytest.raises(ValueError, match="evidence_ids"):
        ImprovementProposal(
            proposal_id="ip_bad",
            target=target,
            diagnosis=ImprovementDiagnosis(summary="Missing evidence ids"),
            hypothesis="A hypothesis",
            proposed_change="A change",
            intended_behavior_change="A behavior change",
            expected_benefit="A benefit",
            risks=("risk",),
            required_permissions=("none",),
            evaluation_plan=ImprovementEvaluationPlan(
                checks=(ImprovementEvaluationCheck(name="check", instruction="Run it"),)
            ),
            rollback_plan="Roll back",
            stop_condition="Stop",
            evidence_ids=(),
        )


def test_improvement_proposal_requires_permission_statement() -> None:
    target = _target()
    evidence = _evidence()

    with pytest.raises(ValueError, match="required_permissions"):
        ImprovementProposal(
            proposal_id="ip_bad",
            target=target,
            diagnosis=ImprovementDiagnosis(summary="Missing permission statement"),
            hypothesis="A hypothesis",
            proposed_change="A change",
            intended_behavior_change="A behavior change",
            expected_benefit="A benefit",
            risks=("risk",),
            required_permissions=(),
            evaluation_plan=ImprovementEvaluationPlan(
                checks=(ImprovementEvaluationCheck(name="check", instruction="Run it"),)
            ),
            rollback_plan="Roll back",
            stop_condition="Stop",
            evidence_ids=(evidence.evidence_id,),
        )


def test_improvement_run_rejects_mismatched_proposal_target_and_evidence() -> None:
    target = _target()
    evidence = _evidence()
    other_target = ImprovementTarget(
        target_id="docs.other",
        kind="documentation",
        description="Other docs",
    )

    with pytest.raises(ValueError, match=r"proposal\.target"):
        ImprovementRun(
            run_id="ir_bad",
            status="proposed",
            target=other_target,
            evidence=(evidence,),
            proposal=_proposal(target, evidence),
        )

    bad_proposal = ImprovementProposal(
        proposal_id="ip_bad",
        target=target,
        diagnosis=ImprovementDiagnosis(summary="Bad evidence linkage"),
        hypothesis="A hypothesis",
        proposed_change="A change",
        intended_behavior_change="A behavior change",
        expected_benefit="A benefit",
        risks=("risk",),
        required_permissions=("none",),
        evaluation_plan=ImprovementEvaluationPlan(
            checks=(ImprovementEvaluationCheck(name="check", instruction="Run it"),)
        ),
        rollback_plan="Roll back",
        stop_condition="Stop",
        evidence_ids=("ev_unknown",),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        ImprovementRun(
            run_id="ir_bad",
            status="proposed",
            target=target,
            evidence=(evidence,),
            proposal=bad_proposal,
        )


def test_improvement_evaluation_plan_requires_check_or_success_criterion() -> None:
    with pytest.raises(ValueError, match="at least one check"):
        ImprovementEvaluationPlan()


def test_required_permissions_cannot_mix_none_with_authority() -> None:
    target = _target()
    evidence = _evidence()

    with pytest.raises(ValueError, match="cannot mix none"):
        ImprovementProposal(
            proposal_id="ip_bad",
            target=target,
            diagnosis=ImprovementDiagnosis(summary="Mixed permissions"),
            hypothesis="A hypothesis",
            proposed_change="A change",
            intended_behavior_change="A behavior change",
            expected_benefit="A benefit",
            risks=("risk",),
            required_permissions=("none", "shell"),
            evaluation_plan=ImprovementEvaluationPlan(
                checks=(ImprovementEvaluationCheck(name="check", instruction="Run it"),)
            ),
            rollback_plan="Roll back",
            stop_condition="Stop",
            evidence_ids=(evidence.evidence_id,),
        )
