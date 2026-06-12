"""Proposal-only improvement service."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, cast
from uuid import uuid4

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.improvement.evidence import (
    ImprovementEvidenceBounds,
    validate_bounded_improvement_evidence,
)
from raygent_harness.improvement.models import (
    ImprovementEvidence,
    ImprovementProposal,
    ImprovementRun,
    ImprovementTarget,
)


class ImprovementServiceError(ValueError):
    """Raised when proposal-only improvement service validation fails."""


class ImprovementValidationError(ImprovementServiceError):
    """Raised when a request or returned proposal violates the RSI-001 contract."""


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return {}


@dataclass(frozen=True, slots=True)
class ImprovementProposalRequest:
    """Input for one proposal-only improvement cycle."""

    target: ImprovementTarget
    evidence: tuple[ImprovementEvidence, ...]
    stop_condition: str
    run_id: str | None = None
    proposal_id: str | None = None
    evidence_bounds: ImprovementEvidenceBounds = field(
        default_factory=ImprovementEvidenceBounds
    )
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.stop_condition.strip():
            raise ImprovementValidationError(
                "ImprovementProposalRequest.stop_condition must be non-empty"
            )
        if self.run_id is not None and not self.run_id.strip():
            raise ImprovementValidationError("ImprovementProposalRequest.run_id is empty")
        if self.proposal_id is not None and not self.proposal_id.strip():
            raise ImprovementValidationError(
                "ImprovementProposalRequest.proposal_id is empty"
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class ImprovementProposalGenerator(Protocol):
    """Injected proposal generator.

    Implementations may use heuristics or model calls, but they must return
    data. The service owns evidence bounds and run validation.
    """

    async def propose(self, request: ImprovementProposalRequest) -> ImprovementProposal:
        """Generate one proposal for a validated request."""
        ...


@dataclass(frozen=True, slots=True)
class ImprovementService:
    """Validate a bounded request and return a proposal-only run record."""

    generator: ImprovementProposalGenerator
    clock: Callable[[], float] = time.time
    run_id_factory: Callable[[], str] | None = None

    async def propose(self, request: ImprovementProposalRequest) -> ImprovementRun:
        """Create one proposal-only improvement run.

        This method only validates and records proposal data. It does not run
        tools, request permissions, write files, create worktrees, or promote a
        candidate.
        """

        bounded = validate_bounded_improvement_evidence(
            request.evidence,
            bounds=request.evidence_bounds,
        )
        normalized_request = replace(request, evidence=bounded.evidence)
        proposal = await self.generator.propose(normalized_request)
        _validate_proposal(proposal, normalized_request)
        now = self.clock()
        return ImprovementRun(
            run_id=request.run_id or self._new_run_id(),
            status="proposed",
            target=request.target,
            evidence=bounded.evidence,
            proposal=proposal,
            created_at=now,
            updated_at=now,
            metadata=request.metadata,
        )

    def _new_run_id(self) -> str:
        run_id = (
            self.run_id_factory()
            if self.run_id_factory is not None
            else f"ir_{uuid4().hex}"
        )
        if not run_id.strip():
            raise ImprovementValidationError("run_id_factory returned an empty id")
        return run_id


def _validate_proposal(
    proposal: ImprovementProposal,
    request: ImprovementProposalRequest,
) -> None:
    if request.proposal_id is not None and proposal.proposal_id != request.proposal_id:
        raise ImprovementValidationError(
            "ImprovementProposal.proposal_id does not match request.proposal_id"
        )
    if proposal.target != request.target:
        raise ImprovementValidationError("ImprovementProposal.target must match request target")
    if proposal.stop_condition != request.stop_condition:
        raise ImprovementValidationError(
            "ImprovementProposal.stop_condition must match request stop_condition"
        )

    known_evidence_ids = {item.evidence_id for item in request.evidence}
    unknown = tuple(
        evidence_id
        for evidence_id in proposal.evidence_ids
        if evidence_id not in known_evidence_ids
    )
    if unknown:
        raise ImprovementValidationError(
            "ImprovementProposal.evidence_ids reference unknown evidence: "
            + ", ".join(unknown)
        )

    if not proposal.risks:
        raise ImprovementValidationError("ImprovementProposal.risks must not be empty")
    if not proposal.rollback_plan.strip():
        raise ImprovementValidationError(
            "ImprovementProposal.rollback_plan must be non-empty"
        )


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, FrozenJson]:
    frozen = freeze_json(metadata)
    if not isinstance(frozen, Mapping):
        raise TypeError("metadata must serialize to a JSON object")
    return cast(Mapping[str, FrozenJson], frozen)


__all__ = (
    "ImprovementProposalGenerator",
    "ImprovementProposalRequest",
    "ImprovementService",
    "ImprovementServiceError",
    "ImprovementValidationError",
)
