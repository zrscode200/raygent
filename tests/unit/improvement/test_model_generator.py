from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from raygent_harness.core.model_types import (
    ApiMessage,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelStreamEvent,
    ObservableMessage,
    ProviderError,
    TextContentBlock,
    TokenCountRequest,
    TokenCountResult,
)
from raygent_harness.improvement import (
    ImprovementDiagnosis,
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementEvidenceBounds,
    ImprovementEvidenceValidationError,
    ImprovementModelGenerator,
    ImprovementModelGeneratorError,
    ImprovementProposal,
    ImprovementProposalRequest,
    ImprovementService,
    ImprovementTarget,
    ImprovementValidationError,
    improvement_proposal_to_dict,
)


def _empty_model_requests() -> list[ModelRequest]:
    return []


@dataclass
class _FakeProvider:
    text: str
    observable_text: str | None = None
    requests: list[ModelRequest] = field(default_factory=_empty_model_requests)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return _model_response(self.text, observable_text=self.observable_text)

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        return _unused_stream()

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        _ = request
        return 0

    def resolve_model(
        self,
        requested: str,
        context: ModelResolveContext,
    ) -> str:
        _ = context
        return requested

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(model=model)

    def classify_error(self, error: BaseException) -> ProviderError:
        return ProviderError(kind="fatal_unknown", message=str(error))


async def _unused_stream() -> AsyncIterator[ModelStreamEvent]:
    raise AssertionError("ImprovementModelGenerator must not stream")
    yield  # pragma: no cover


def _model_response(text: str, *, observable_text: str | None = None) -> ModelResponse:
    api_message = ModelMessage(
        role="assistant",
        content=(TextContentBlock(text=text),),
    )
    observable_message = ModelMessage(
        role="assistant",
        content=(TextContentBlock(text=observable_text or text),),
    )
    return ModelResponse(
        api_message=ApiMessage(api_message),
        observable_message=ObservableMessage(observable_message),
    )


def _target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="preset.project_reader",
        kind="preset",
        description="Project-reader preset behavior",
        owner="sdk",
        metadata={"component": "profiles"},
    )


def _evidence(evidence_id: str = "ev_1") -> ImprovementEvidence:
    return ImprovementEvidence(
        evidence_id=evidence_id,
        source="transcript",
        summary="User needed read-only project inspection.",
        excerpt="The session asked for grep/read-only behavior.",
        source_uri=f"transcript://session/s_1#{evidence_id}",
        created_at=100.0,
    )


def _proposal(
    *,
    proposal_id: str = "ip_generated",
    evidence_ids: tuple[str, ...] = ("ev_1",),
    target: ImprovementTarget | None = None,
    stop_condition: str = "Stop after producing one proposal.",
) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=proposal_id,
        target=target or _target(),
        diagnosis=ImprovementDiagnosis(
            summary="Preset explanation is not concrete enough.",
            symptoms=("user confusion about read-only surface",),
            hypotheses=("clearer affordance text reduces setup mistakes",),
            confidence=0.8,
        ),
        hypothesis="Clearer preset affordance text will reduce setup mistakes.",
        proposed_change="Clarify project_reader read-only behavior.",
        intended_behavior_change="Users choose read-only profiles before mutation.",
        expected_benefit="Lower risk of accidental filesystem authority.",
        risks=("The proposal might duplicate existing docs.",),
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
        rollback_plan="Discard the proposal record.",
        stop_condition=stop_condition,
        evidence_ids=evidence_ids,
        created_at=101.0,
    )


def _request(
    *,
    proposal_id: str | None = None,
    stop_condition: str = "Stop after producing one proposal.",
) -> ImprovementProposalRequest:
    return ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(),),
        stop_condition=stop_condition,
        proposal_id=proposal_id,
        metadata={"mode": "model-generator"},
    )


@pytest.mark.asyncio
async def test_model_generator_uses_tool_free_provider_request_and_service_validation(
    tmp_path: Path,
) -> None:
    proposal = _proposal(proposal_id="ip_requested")
    provider = _FakeProvider(json.dumps(improvement_proposal_to_dict(proposal)))
    generator = ImprovementModelGenerator(provider=provider, model="test-model")
    service = ImprovementService(
        generator=generator,
        clock=lambda: 200.0,
        run_id_factory=lambda: "ir_model",
    )

    run = await service.propose(_request(proposal_id="ip_requested"))

    assert run.run_id == "ir_model"
    assert run.proposal.proposal_id == "ip_requested"
    assert run.proposal.evidence_ids == ("ev_1",)
    assert len(provider.requests) == 1
    model_request = provider.requests[0]
    assert model_request.model == "test-model"
    assert model_request.tools == ()
    assert model_request.query_source == "improvement.proposal"
    assert model_request.system_prompt
    prompt_text = model_request.messages[0].message.content[0]
    assert isinstance(prompt_text, TextContentBlock)
    assert '"allowed_evidence_ids"' in prompt_text.text
    assert '"ev_1"' in prompt_text.text
    assert "Stop after producing one proposal." in prompt_text.text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_model_generator_accepts_json_object_with_surrounding_text() -> None:
    proposal = _proposal()
    provider = _FakeProvider(
        "Here is the proposal:\n"
        + json.dumps(improvement_proposal_to_dict(proposal))
        + "\nDone."
    )
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    generated = await generator.propose(_request())

    assert generated == proposal
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_model_generator_prefers_api_message_text_over_observable_text() -> None:
    proposal = _proposal()
    provider = _FakeProvider(
        json.dumps(improvement_proposal_to_dict(proposal)),
        observable_text="{}",
    )
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    generated = await generator.propose(_request())

    assert generated == proposal


@pytest.mark.asyncio
async def test_model_generator_rejects_unbounded_evidence_before_model_call() -> None:
    provider = _FakeProvider(json.dumps(improvement_proposal_to_dict(_proposal())))
    generator = ImprovementModelGenerator(provider=provider, model="test-model")
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(),),
        stop_condition="Stop after producing one proposal.",
        evidence_bounds=ImprovementEvidenceBounds(
            max_items=1,
            max_item_text_chars=10,
            max_total_text_chars=10,
        ),
    )

    with pytest.raises(ImprovementEvidenceValidationError, match="exceeds"):
        await generator.propose(request)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_model_generator_rejects_oversized_metadata_before_model_call() -> None:
    provider = _FakeProvider(json.dumps(improvement_proposal_to_dict(_proposal())))
    generator = ImprovementModelGenerator(
        provider=provider,
        model="test-model",
        max_metadata_prompt_chars=10,
    )
    request = ImprovementProposalRequest(
        target=_target(),
        evidence=(_evidence(),),
        stop_condition="Stop after producing one proposal.",
        metadata={"raw": "x" * 100},
    )

    with pytest.raises(ImprovementModelGeneratorError, match="metadata"):
        await generator.propose(request)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_model_generator_rejects_empty_assistant_text() -> None:
    provider = _FakeProvider(" ")
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    with pytest.raises(ImprovementModelGeneratorError, match=r"assistant text|empty"):
        await generator.propose(_request())


@pytest.mark.asyncio
async def test_model_generator_rejects_invalid_json() -> None:
    provider = _FakeProvider("{not json}")
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    with pytest.raises(ImprovementModelGeneratorError, match="invalid"):
        await generator.propose(_request())


@pytest.mark.asyncio
async def test_model_generator_rejects_non_object_json() -> None:
    provider = _FakeProvider("[]")
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    with pytest.raises(ImprovementModelGeneratorError, match="object"):
        await generator.propose(_request())


@pytest.mark.asyncio
async def test_model_generator_rejects_invalid_proposal_schema() -> None:
    provider = _FakeProvider("{}")
    generator = ImprovementModelGenerator(provider=provider, model="test-model")

    with pytest.raises(ImprovementModelGeneratorError, match="schema"):
        await generator.propose(_request())


@pytest.mark.asyncio
async def test_service_rejects_model_proposal_with_unknown_evidence_id() -> None:
    proposal = _proposal(evidence_ids=("ev_unknown",))
    provider = _FakeProvider(json.dumps(improvement_proposal_to_dict(proposal)))
    generator = ImprovementModelGenerator(provider=provider, model="test-model")
    service = ImprovementService(generator=generator)

    with pytest.raises(ImprovementValidationError, match="unknown evidence"):
        await service.propose(_request())
