"""Model-backed improvement proposal generator.

This generator asks an injected model provider for proposal JSON. It does not
execute tools, mutate files, create worktrees, or promote candidates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from raygent_harness.core.model_provider import ModelProvider
from raygent_harness.core.model_types import (
    ApiMessage,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSampling,
    TextContentBlock,
)
from raygent_harness.improvement.evidence import validate_bounded_improvement_evidence
from raygent_harness.improvement.models import (
    ImprovementProposal,
    improvement_evidence_to_dict,
    improvement_proposal_from_dict,
    improvement_target_to_dict,
)
from raygent_harness.improvement.service import ImprovementProposalRequest

_DEFAULT_MAX_METADATA_PROMPT_CHARS = 4_000

DEFAULT_IMPROVEMENT_MODEL_GENERATOR_SYSTEM_PROMPT = (
    "You generate one bounded Raygent improvement proposal as JSON. "
    "Return only one JSON object. Do not claim to execute tests, shell "
    "commands, file edits, worktrees, commits, promotion, archive search, "
    "or product /goal behavior."
)


class ImprovementModelGeneratorError(ValueError):
    """Raised when model output cannot be converted into a proposal."""


@dataclass(frozen=True, slots=True)
class ImprovementModelGenerator:
    """Generate `ImprovementProposal` data through a narrow model-provider seam."""

    provider: ModelProvider
    model: str
    system_prompt: str = DEFAULT_IMPROVEMENT_MODEL_GENERATOR_SYSTEM_PROMPT
    sampling: ModelSampling = field(
        default_factory=lambda: ModelSampling(max_tokens=4096, temperature=0.0)
    )
    query_source: str = "improvement.proposal"
    max_metadata_prompt_chars: int = _DEFAULT_MAX_METADATA_PROMPT_CHARS

    def __post_init__(self) -> None:
        _require_non_empty(self.model, "ImprovementModelGenerator.model")
        _require_non_empty(
            self.system_prompt,
            "ImprovementModelGenerator.system_prompt",
        )
        _require_non_empty(self.query_source, "ImprovementModelGenerator.query_source")
        if self.max_metadata_prompt_chars < 2:
            raise ValueError(
                "ImprovementModelGenerator.max_metadata_prompt_chars must be >= 2"
            )

    async def propose(self, request: ImprovementProposalRequest) -> ImprovementProposal:
        """Ask the model provider for one JSON proposal.

        This method performs one non-streaming model call with no tools. The
        surrounding `ImprovementService` still owns bounded-evidence and
        proposal invariants.
        """

        bounded = validate_bounded_improvement_evidence(
            request.evidence,
            bounds=request.evidence_bounds,
        )
        bounded_request = replace(request, evidence=bounded.evidence)
        model_request = ModelRequest(
            model=self.model,
            messages=(
                ApiMessage(
                    ModelMessage(
                        role="user",
                        content=(
                            TextContentBlock(
                                text=_render_prompt(
                                    bounded_request,
                                    max_metadata_chars=self.max_metadata_prompt_chars,
                                )
                            ),
                        ),
                    )
                ),
            ),
            system_prompt=self.system_prompt,
            tools=(),
            sampling=self.sampling,
            query_source=self.query_source,
        )
        response = await self.provider.complete(model_request)
        payload = _parse_json_object(_model_response_text(response))
        return _proposal_from_payload(payload)


def _render_prompt(
    request: ImprovementProposalRequest,
    *,
    max_metadata_chars: int,
) -> str:
    allowed_evidence_ids = [item.evidence_id for item in request.evidence]
    metadata = _bounded_json_prompt_value(
        _json_ready(request.metadata),
        max_chars=max_metadata_chars,
        name="metadata",
    )
    payload = {
        "target": improvement_target_to_dict(request.target),
        "evidence": [improvement_evidence_to_dict(item) for item in request.evidence],
        "proposal_id": request.proposal_id,
        "stop_condition": request.stop_condition,
        "allowed_evidence_ids": allowed_evidence_ids,
        "metadata": metadata,
        "required_json_keys": [
            "proposal_id",
            "target",
            "diagnosis",
            "hypothesis",
            "proposed_change",
            "intended_behavior_change",
            "expected_benefit",
            "risks",
            "required_permissions",
            "evaluation_plan",
            "rollback_plan",
            "stop_condition",
            "evidence_ids",
        ],
        "instructions": [
            "Return only one JSON object.",
            "Use the supplied target exactly.",
            "Use only allowed_evidence_ids.",
            "Keep required_permissions descriptive.",
            "Use ['none'] when no later authority is required.",
            "Include evaluation plan, rollback plan, risks, and stop condition.",
            "Do not claim executed checks, shell commands, file edits, worktrees, "
            "commits, promotion, archive search, or product /goal.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _model_response_text(response: ModelResponse) -> str:
    for message in (response.api_message.message, response.observable_message.message):
        text = _message_text(message)
        if text.strip():
            return text
    raise ImprovementModelGeneratorError("model response did not contain assistant text")


def _message_text(message: ModelMessage) -> str:
    chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContentBlock) and block.text
    ]
    return "\n".join(chunks)


def _parse_json_object(text: str) -> Mapping[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ImprovementModelGeneratorError("model response text is empty")

    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = None
    else:
        if not isinstance(loaded, Mapping):
            raise ImprovementModelGeneratorError("model response JSON must be an object")
        return cast(Mapping[str, object], loaded)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ImprovementModelGeneratorError("model response did not contain a JSON object")
    candidate = stripped[start : end + 1]
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ImprovementModelGeneratorError("model response JSON is invalid") from exc
    if not isinstance(loaded, Mapping):
        raise ImprovementModelGeneratorError("model response JSON must be an object")
    return cast(Mapping[str, object], loaded)


def _proposal_from_payload(payload: Mapping[str, object]) -> ImprovementProposal:
    try:
        return improvement_proposal_from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ImprovementModelGeneratorError(
            "model response JSON does not match ImprovementProposal schema"
        ) from exc


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_ready(item) for key, item in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return [_json_ready(item) for item in sequence]
    raise TypeError(f"Expected JSON-like value, got {type(value).__name__}")


def _bounded_json_prompt_value(value: object, *, max_chars: int, name: str) -> object:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(rendered) > max_chars:
        raise ImprovementModelGeneratorError(
            f"request {name} exceeds {max_chars} prompt characters"
        )
    return value


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


__all__ = (
    "DEFAULT_IMPROVEMENT_MODEL_GENERATOR_SYSTEM_PROMPT",
    "ImprovementModelGenerator",
    "ImprovementModelGeneratorError",
)
