from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from raygent_harness.improvement import (
    ImprovementDiagnosis,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementEvidenceBounds,
    ImprovementEvidenceValidationError,
    ImprovementProposal,
    ImprovementProposalRequest,
    ImprovementService,
    ImprovementTarget,
    ImprovementValidationError,
)


def _target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="tool_affordance.readonly",
        kind="tool_affordance",
        description="Read-only tool affordance",
    )


def _evidence(
    evidence_id: str = "ev_1",
    *,
    excerpt: str = "Observed behavior",
) -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id=evidence_id,
        source="observability",
        summary="Read-only affordance was unclear.",
        excerpt=excerpt,
        created_at=100.0,
    )


@dataclass
class _Generator:
    calls: int = 0

    async def propose(self, request: ImprovementProposalRequest) -> ImprovementProposal:
        self.calls += 1
        return ImprovementProposal(
            proposal_id=request.proposal_id or "ip_generated",
            target=request.target,
            diagnosis=ImprovementDiagnosis(
                summary="The affordance lacks a clear safety statement.",
                symptoms=("reader could infer mutation authority",),
                hypotheses=("explicit safety wording will reduce mistakes",),
                confidence=0.75,
            ),
            hypothesis="Make the read-only boundary explicit.",
            proposed_change="Add an explicit read-only affordance note.",
            intended_behavior_change="Users can choose the safe profile confidently.",
            expected_benefit="Reduced accidental broad-authority setup.",
            risks=("The proposal might duplicate existing docs.",),
            required_permissions=("none",),
            evaluation_plan=ImprovementEvaluationPlan(
                checks=(
                    ImprovementEvaluationCheck(
                        name="doc-contract",
                        instruction="Check that docs list read-only tools.",
                    ),
                ),
                non_regression_checks=(
                    ImprovementEvaluationCheck(
                        name="no-mutation",
                        instruction="Confirm no mutation behavior is proposed.",
                    ),
                ),
                success_criteria=("Proposal remains data-only.",),
            ),
            rollback_plan="Discard the proposal record.",
            stop_condition=request.stop_condition,
            evidence_ids=tuple(item.evidence_id for item in request.evidence),
            created_at=101.0,
        )


@dataclass
class _BadEvidenceGenerator:
    async def propose(self, request: ImprovementProposalRequest) -> ImprovementProposal:
        proposal = await _Generator().propose(request)
        return ImprovementProposal(
            proposal_id=proposal.proposal_id,
            target=proposal.target,
            diagnosis=proposal.diagnosis,
            hypothesis=proposal.hypothesis,
            proposed_change=proposal.proposed_change,
            intended_behavior_change=proposal.intended_behavior_change,
            expected_benefit=proposal.expected_benefit,
            risks=proposal.risks,
            required_permissions=proposal.required_permissions,
            evaluation_plan=proposal.evaluation_plan,
            rollback_plan=proposal.rollback_plan,
            stop_condition=proposal.stop_condition,
            evidence_ids=("ev_unknown",),
            created_at=proposal.created_at,
        )


@pytest.mark.asyncio
async def test_service_returns_proposal_run_without_filesystem_side_effects(
    tmp_path: Path,
) -> None:
    generator = _Generator()
    service = ImprovementService(
        generator=generator,
        clock=lambda: 200.0,
        run_id_factory=lambda: "ir_generated",
    )
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(),),
        stop_condition="Stop after producing one proposal.",
        proposal_id="ip_requested",
        metadata={"mode": "proposal-only"},
    )

    run = await service.propose(request)

    assert run.run_id == "ir_generated"
    assert run.status == "proposed"
    assert run.created_at == 200.0
    assert run.proposal.proposal_id == "ip_requested"
    assert run.proposal.required_permissions == ("none",)
    assert run.proposal.stop_condition == "Stop after producing one proposal."
    assert generator.calls == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_service_rejects_missing_evidence() -> None:
    service = ImprovementService(generator=_Generator())
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(),
        stop_condition="Stop after producing one proposal.",
    )

    with pytest.raises(ImprovementEvidenceValidationError, match="must not be empty"):
        await service.propose(request)


@pytest.mark.asyncio
async def test_service_rejects_unbounded_evidence() -> None:
    service = ImprovementService(generator=_Generator())
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(excerpt="x" * 20),),
        stop_condition="Stop after producing one proposal.",
        evidence_bounds=ImprovementEvidenceBounds(
            max_items=1,
            max_item_text_chars=10,
            max_total_text_chars=10,
        ),
    )

    with pytest.raises(ImprovementEvidenceValidationError, match="exceeds"):
        await service.propose(request)


def test_request_requires_stop_condition() -> None:
    with pytest.raises(ImprovementValidationError, match="stop_condition"):
        ImprovementProposalRequest(
            target=_target(),
            evidence=(_evidence(),),
            stop_condition=" ",
        )


@pytest.mark.asyncio
async def test_service_bounds_evidence_metadata() -> None:
    service = ImprovementService(generator=_Generator())
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(
            ImprovementEvidence(
                evidence_id="ev_metadata",
                source="transcript",
                summary="Small summary",
                metadata={"raw": "x" * 200},
            ),
        ),
        stop_condition="Stop after producing one proposal.",
        evidence_bounds=ImprovementEvidenceBounds(
            max_items=1,
            max_item_text_chars=180,
            max_total_text_chars=180,
        ),
    )

    with pytest.raises(ImprovementEvidenceValidationError, match="exceeds"):
        await service.propose(request)


@pytest.mark.asyncio
async def test_service_rejects_unknown_proposal_evidence_ids() -> None:
    service = ImprovementService(generator=_BadEvidenceGenerator())
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(),),
        stop_condition="Stop after producing one proposal.",
    )

    with pytest.raises(ImprovementValidationError, match="unknown evidence"):
        await service.propose(request)
