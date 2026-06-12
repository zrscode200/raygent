from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import (
    api_message_from_message_param,
    assistant_message,
    observable_message_from_message_param,
)
from raygent_harness.core.model_types import ModelResponse
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.state import UsageTotals
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.goals import (
    GoalSpec,
    ModelProviderGoalCompletionEvaluator,
    create_goal_state,
    goal_progress_ledger_to_dict,
    parse_goal_completion_evaluation,
)
from raygent_harness.goals.evaluators import GoalProgressLedger
from tests.fakes import FakeModelProvider


@dataclass
class _Engine:
    _messages: list[dict[str, Any]]


@dataclass
class _Session:
    config: QueryConfig
    deps: QueryDeps
    ctx: ToolUseContext
    engine: _Engine

    @property
    def session_id(self) -> str:
        return self.config.session_id


def _session() -> _Session:
    return _Session(
        config=QueryConfig(model="main-model", session_id="s-evaluator"),
        deps=QueryDeps(task_store=AppStateStore()),
        ctx=ToolUseContext(
            session_id="s-evaluator",
            agent_id=None,
            abort_event=asyncio.Event(),
            rendered_system_prompt="",
            cwd=".",
            query_tracking=QueryTracking(chain_id="s-evaluator", depth=0),
        ),
        engine=_Engine(
            _messages=[
                {"role": "user", "content": "Please finish the objective"},
                {"role": "assistant", "content": "I ran the checks"},
            ]
        ),
    )


def _sdk_result(text: str = "latest result") -> SDKResult:
    return SDKResult(
        subtype="success",
        session_id="s-evaluator",
        is_error=False,
        num_turns=1,
        result=text,
        usage=UsageTotals(),
    )


def test_parse_goal_completion_evaluation_prefers_structured_status() -> None:
    evaluation = parse_goal_completion_evaluation(
        """
        <goal_completion_evaluation>
        <status>complete</status>
        <reason>All acceptance checks passed.</reason>
        </goal_completion_evaluation>
        """
    )

    assert evaluation.status == "complete"
    assert evaluation.reason == "All acceptance checks passed."


def test_parse_goal_completion_evaluation_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="missing <status>"):
        parse_goal_completion_evaluation("yes, probably done")

    with pytest.raises(ValueError, match="status must be"):
        parse_goal_completion_evaluation(
            "<goal_completion_evaluation>"
            "<status>maybe</status>"
            "<reason>Ambiguous.</reason>"
            "</goal_completion_evaluation>"
        )


@pytest.mark.asyncio
async def test_model_provider_completion_evaluator_uses_resolved_model_and_api_message() -> None:
    api = api_message_from_message_param(
        assistant_message(
            "<goal_completion_evaluation>"
            "<status>complete</status>"
            "<reason>API-bound result says complete.</reason>"
            "</goal_completion_evaluation>"
        )
    )
    observable = observable_message_from_message_param(
        assistant_message(
            "<goal_completion_evaluation>"
            "<status>incomplete</status>"
            "<reason>Observable text diverged.</reason>"
            "</goal_completion_evaluation>"
        )
    )
    provider = FakeModelProvider(
        responses=(ModelResponse(api_message=api, observable_message=observable),),
        resolved_models={"goal-eval": "resolved-goal-eval"},
    )
    evaluator = ModelProviderGoalCompletionEvaluator(
        provider=provider,
        model="goal-eval",
    )
    state = create_goal_state(
        goal_id="g_eval",
        session_id="s-evaluator",
        spec=GoalSpec(objective="ship evaluator"),
        now=1.0,
    )

    evaluation = await evaluator.evaluate(
        state=state,
        sdk_result=_sdk_result(),
        session=_session(),
    )

    assert evaluation.status == "complete"
    assert evaluation.reason == "API-bound result says complete."
    assert provider.resolve_requests[0][0] == "goal-eval"
    assert provider.resolve_requests[0][1].query_source == "goal_completion_evaluator"
    assert provider.requests[0].model == "resolved-goal-eval"
    assert provider.requests[0].query_source == "goal_completion_evaluator"
    assert provider.requests[0].tools == ()
    assert "ship evaluator" in str(provider.requests[0].messages[0].provider_payload)


def test_progress_ledger_serializes_provider_neutral_shape() -> None:
    ledger = GoalProgressLedger(
        request_satisfied=False,
        made_progress=False,
        loop_detected=True,
        reason="Repeated same failing command.",
        next_action="Try a different diagnostic.",
        facts=("test still fails",),
        plan=("inspect logs",),
    )

    snapshot = goal_progress_ledger_to_dict(ledger)

    assert snapshot["request_satisfied"] is False
    assert snapshot["made_progress"] is False
    assert snapshot["loop_detected"] is True
    assert snapshot["next_action"] == "Try a different diagnostic."
