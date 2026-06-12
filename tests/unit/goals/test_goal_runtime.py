from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import context_provider_kind
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import (
    MessageParam,
    assistant_message,
    message_param_from_api_message,
    model_response_from_message_param,
)
from raygent_harness.core.model_types import ModelResponse, Usage
from raygent_harness.core.query_engine import SDKResult, SDKSystemInit
from raygent_harness.core.state import PermissionDenial, UsageTotals
from raygent_harness.core.task import AppStateStore, TaskStateBase
from raygent_harness.core.tool import QueryTracking, Tool, ToolUseContext
from raygent_harness.goals import (
    GET_GOAL_TOOL_NAME,
    UPDATE_GOAL_TOOL_NAME,
    GoalAcceptanceCheck,
    GoalApprovalPolicy,
    GoalBlockingPolicy,
    GoalBudgetPolicy,
    GoalCompletionEvaluation,
    GoalCompletionPolicy,
    GoalContinuationPolicy,
    GoalEventEmitter,
    GoalPolicy,
    GoalProgressLedger,
    GoalRuntime,
    GoalRuntimeConfig,
    GoalRuntimeStateError,
    GoalSpec,
    InMemoryGoalEventSink,
    InMemoryGoalStore,
    JsonGoalStore,
    ModelProviderGoalCompletionEvaluator,
)
from raygent_harness.sdk import create_raygent
from tests.fakes import FakeModelProvider


def _metadata_mapping(state: object, key: str) -> Mapping[str, object]:
    value = cast(Any, state).metadata[key]
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


def _response(
    text: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> ModelResponse:
    return replace(
        model_response_from_message_param(assistant_message(text)),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


def _update_goal_response(
    *,
    status: str,
    reason: str,
    tool_use_id: str = "tu_goal",
) -> MessageParam:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": UPDATE_GOAL_TOOL_NAME,
                "input": {"status": status, "reason": reason},
            }
        ],
    }


@dataclass
class _MutableClock:
    value: float = 100.0

    def now(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


class _StubEngine:
    def __init__(self, config: QueryConfig, total_usage: UsageTotals | None = None) -> None:
        self._config = config
        self._ctx: ToolUseContext | None = None
        self._total_usage = total_usage or UsageTotals()


class _StubSession:
    def __init__(self, result: SDKResult, *, clock: _MutableClock) -> None:
        self.config = QueryConfig(model="model-1", session_id="stub-session")
        self.deps = QueryDeps(task_store=AppStateStore(), clock=clock)
        self.ctx = ToolUseContext(
            session_id="stub-session",
            agent_id=None,
            abort_event=asyncio.Event(),
            rendered_system_prompt="",
            cwd=".",
            query_tracking=QueryTracking(chain_id="stub-session", depth=0),
        )
        self.engine = _StubEngine(self.config)
        self.result = result
        self.calls = 0

    @property
    def session_id(self) -> str:
        return self.config.session_id

    async def run_until_result(self, prompt: str) -> SDKResult:
        _ = prompt
        self.calls += 1
        self.deps.clock.advance(2.0)  # type: ignore[attr-defined]
        cast(Any, self.engine)._total_usage = self.result.usage
        return self.result


@dataclass
class _CompletionEvaluator:
    evaluations: list[GoalCompletionEvaluation]
    name: str = "completion_eval"
    calls: int = 0

    async def evaluate(
        self,
        *,
        state: object,
        sdk_result: SDKResult,
        session: object,
    ) -> GoalCompletionEvaluation:
        _ = state, sdk_result, session
        self.calls += 1
        return self.evaluations.pop(0)


@dataclass
class _ProgressEvaluator:
    ledgers: list[GoalProgressLedger]
    name: str = "progress_eval"
    calls: int = 0

    async def evaluate(
        self,
        *,
        state: object,
        sdk_result: SDKResult,
        session: object,
    ) -> GoalProgressLedger:
        _ = state, sdk_result, session
        self.calls += 1
        return self.ledgers.pop(0)


@dataclass
class _FailingCompletionEvaluator:
    name: str = "failing_completion_eval"

    async def evaluate(
        self,
        *,
        state: object,
        sdk_result: SDKResult,
        session: object,
    ) -> GoalCompletionEvaluation:
        _ = state, sdk_result, session
        raise RuntimeError("evaluator unavailable")


@pytest.mark.asyncio
async def test_start_installs_goal_context_tools_and_catalog_wrapper(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(_response("ok"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-runtime",
        tools="none",
        context="none",
    )

    async def empty_catalog(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: object,
        /,
    ) -> tuple[Tool, ...]:
        return ()

    session.deps.tool_catalog_provider = empty_catalog  # type: ignore[assignment]
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 10.0)
    store = InMemoryGoalStore(event_emitter=emitter)
    runtime = GoalRuntime(
        session,
        store=store,
        config=GoalRuntimeConfig(event_emitter=emitter),
    )

    state = runtime.start(GoalSpec(objective="ship runtime"), goal_id="goal-runtime")

    assert state.session_id == "s-runtime"
    assert store.get_active_for_session("s-runtime") == state
    assert {tool.name for tool in session.config.tools} == {
        GET_GOAL_TOOL_NAME,
        UPDATE_GOAL_TOOL_NAME,
    }
    engine_config = cast(Any, session.engine)._config
    assert {tool.name for tool in engine_config.tools} == {
        GET_GOAL_TOOL_NAME,
        UPDATE_GOAL_TOOL_NAME,
    }
    assert {tool.name for tool in session.ctx.tools} == {
        GET_GOAL_TOOL_NAME,
        UPDATE_GOAL_TOOL_NAME,
    }
    assert any(
        context_provider_kind(provider_obj) == "goal"
        for provider_obj in session.deps.context_providers
    )

    events = [event async for event in session.submit_message("inspect surfaces")]
    init = cast(SDKSystemInit, events[0])
    assert isinstance(init, SDKSystemInit)
    assert set(init.tools) == {GET_GOAL_TOOL_NAME, UPDATE_GOAL_TOOL_NAME}
    assert [event.type for event in sink.events[:1]] == ["goal_created"]


@pytest.mark.asyncio
async def test_run_until_idle_drives_continuations_and_accounts_cumulative_usage(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _response("working", input_tokens=5, output_tokens=2),
            _response(
                "still working",
                input_tokens=3,
                output_tokens=4,
                cache_read_input_tokens=1,
            ),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-loop",
        tools="none",
        context="none",
    )
    store = InMemoryGoalStore()
    runtime = GoalRuntime(session, store=store)
    runtime.start(
        GoalSpec(objective="finish two continuation turns"),
        policy=GoalPolicy(
            continuation=GoalContinuationPolicy(max_idle_continuations=2),
        ),
        goal_id="goal-loop",
    )

    result = await runtime.run_until_idle()

    assert result.stop_reason == "max_continuations_reached"
    assert result.continuation_count == 2
    assert len(result.sdk_results) == 2
    assert len(provider.requests) == 2
    state = store.get("goal-loop")
    assert state is not None
    assert state.status == "active"
    assert state.turn_count == 2
    assert state.tokens_used == 15
    first_request = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    assert "raygent_goal_context" in str(first_request[0]["content"])
    assert "finish two continuation turns" in str(first_request[0]["content"])
    assert str(first_request[-1]["content"]).startswith("Continue working")


@pytest.mark.asyncio
async def test_goal_budget_limit_stops_after_accounted_turn(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(_response("spent", input_tokens=2),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-budget",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "one turn only",
        policy=GoalPolicy(budget=GoalBudgetPolicy(max_turns=1)),
        goal_id="goal-budget",
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "budget_limited"
    assert result.state is not None
    assert result.state.status == "budget_limited"
    assert result.state.turn_count == 1
    assert result.state.tokens_used == 2


@pytest.mark.asyncio
async def test_provider_usage_limit_error_marks_goal_usage_limited() -> None:
    clock = _MutableClock()
    sdk_result = SDKResult(
        subtype="error_max_budget_usd",
        session_id="stub-session",
        is_error=True,
        num_turns=1,
        usage=UsageTotals(input_tokens=3, output_tokens=1),
        errors=("provider usage limit",),
    )
    session = _StubSession(sdk_result, clock=clock)
    runtime = GoalRuntime(session)
    runtime.start("respect provider usage limits", goal_id="goal-usage")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "usage_limited"
    assert result.state is not None
    assert result.state.status == "usage_limited"
    assert result.state.last_reason == "provider usage limit"
    assert result.state.tokens_used == 4
    assert result.state.time_used_s == 2.0


@pytest.mark.asyncio
async def test_turn_error_uses_failed_policy(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(RuntimeError("provider down"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-error",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "fail after one continuation error",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(max_consecutive_errors=1)),
        goal_id="goal-error",
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "failed"
    assert result.state is not None
    assert result.state.status == "failed"
    assert result.state.turn_count == 1
    assert "provider down" in (result.state.last_reason or "")


def test_pause_resume_and_cancel_lifecycle(tmp_path: Path) -> None:
    session = create_raygent(
        provider=FakeModelProvider(),
        model="model-1",
        cwd=tmp_path,
        session_id="s-lifecycle",
    )
    runtime = GoalRuntime(session)
    runtime.start("exercise lifecycle", goal_id="goal-life")

    paused = runtime.pause(reason="user paused")
    assert paused.status == "paused"
    assert paused.last_reason == "user paused"

    resumed = runtime.resume(reason="user resumed")
    assert resumed.status == "active"
    assert resumed.last_reason == "user resumed"

    cancelled = runtime.cancel(reason="user cancelled")
    assert cancelled.status == "cancelled"
    assert cancelled.last_reason == "user cancelled"

    with pytest.raises(GoalRuntimeStateError, match="Cannot resume"):
        runtime.resume("goal-life")


@pytest.mark.asyncio
async def test_runtime_resumes_goal_from_json_store(tmp_path: Path) -> None:
    store_dir = tmp_path / "goals"
    store = JsonGoalStore(store_dir)
    first_session = create_raygent(
        provider=FakeModelProvider(),
        model="model-1",
        cwd=tmp_path,
        session_id="s-json-resume",
    )
    first_runtime = GoalRuntime(first_session, store=store)
    first_runtime.start("resume from durable store", goal_id="goal-json-resume")
    first_runtime.pause(reason="session ended")

    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="complete", reason="resumed and finished"),
            assistant_message("done"),
        )
    )
    resumed_session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-json-resume",
    )
    resumed_runtime = GoalRuntime(
        resumed_session,
        store=JsonGoalStore(store_dir),
    )

    resumed = resumed_runtime.resume()
    result = await resumed_runtime.run_idle_once()

    assert resumed.status == "active"
    assert result.stop_reason == "completed"
    assert result.state is not None
    assert result.state.status == "complete"
    assert JsonGoalStore(store_dir).get("goal-json-resume") == result.state


def test_runtime_rejects_install_while_session_turn_is_active(tmp_path: Path) -> None:
    session = create_raygent(
        provider=FakeModelProvider(),
        model="model-1",
        cwd=tmp_path,
        session_id="s-active",
    )
    session._active_turn = True  # pyright: ignore[reportPrivateUsage]
    runtime = GoalRuntime(session)

    with pytest.raises(GoalRuntimeStateError, match="active"):
        runtime.install()


def test_start_rejects_active_session_without_creating_goal(tmp_path: Path) -> None:
    session = create_raygent(
        provider=FakeModelProvider(),
        model="model-1",
        cwd=tmp_path,
        session_id="s-active-start",
    )
    store = InMemoryGoalStore()
    runtime = GoalRuntime(session, store=store)
    session._active_turn = True  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(GoalRuntimeStateError, match="active"):
        runtime.start("must not create a ghost goal", goal_id="goal-ghost")

    assert store.get("goal-ghost") is None


@pytest.mark.asyncio
async def test_idle_without_active_goal_does_not_install_goal_surfaces(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=FakeModelProvider(),
        model="model-1",
        cwd=tmp_path,
        session_id="s-no-active",
    )
    runtime = GoalRuntime(session)

    result = await runtime.run_idle_once()

    assert result.stop_reason == "no_active_goal"
    assert session.config.tools == ()
    assert session.deps.context_providers == ()


@pytest.mark.asyncio
async def test_installed_update_goal_complete_stops_runtime_continuation(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(
                status="complete",
                reason="all evidence checked",
            ),
            assistant_message("ack"),
            assistant_message("should not be requested"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-complete",
    )
    store = InMemoryGoalStore()
    runtime = GoalRuntime(session, store=store)
    runtime.start("finish through tool report", goal_id="goal-complete")

    result = await runtime.run_until_idle(max_continuations=3)

    state = store.get("goal-complete")
    assert state is not None
    assert result.stop_reason == "completed"
    assert result.continuation_count == 1
    assert state.status == "complete"
    assert state.last_reason == "all evidence checked"
    assert state.turn_count == 1
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_installed_update_goal_blocked_stops_runtime_continuation(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(
                status="blocked",
                reason="external dependency unavailable",
            ),
            assistant_message("ack"),
            assistant_message("should not be requested"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-blocked",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "block through tool report",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(blocked_audit_turns=1)),
        goal_id="goal-blocked",
    )

    result = await runtime.run_until_idle(max_continuations=3)

    assert result.stop_reason == "blocked"
    assert result.continuation_count == 1
    assert result.state is not None
    assert result.state.status == "blocked"
    assert result.state.last_reason == "external dependency unavailable"
    assert result.state.turn_count == 1
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_blocked_reports_require_configured_audit_threshold(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="blocked", reason="same dependency"),
            assistant_message("first report recorded"),
            _update_goal_response(
                status="blocked",
                reason="same dependency",
                tool_use_id="tu_goal_2",
            ),
            assistant_message("second report recorded"),
            _update_goal_response(
                status="blocked",
                reason="same dependency",
                tool_use_id="tu_goal_3",
            ),
            assistant_message("third report recorded"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-blocked-audit",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "require three blocked reports",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(blocked_audit_turns=3)),
        goal_id="goal-blocked-audit",
    )

    result = await runtime.run_until_idle(max_continuations=3)

    state = runtime.store.get("goal-blocked-audit")
    assert state is not None
    assert result.stop_reason == "blocked"
    assert result.continuation_count == 3
    assert state.status == "blocked"
    assert state.blocked_turn_count == 3
    assert len(provider.requests) == 6


@pytest.mark.asyncio
async def test_blocked_audit_resets_after_productive_turn(tmp_path: Path) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="blocked", reason="temporary gap"),
            assistant_message("first report recorded"),
            assistant_message("productive turn"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-blocked-reset",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "reset blocked audit",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(blocked_audit_turns=3)),
        goal_id="goal-blocked-reset",
    )

    first = await runtime.run_idle_once()
    second = await runtime.run_idle_once()

    state = runtime.store.get("goal-blocked-reset")
    assert state is not None
    assert first.stop_reason == "continued"
    assert second.stop_reason == "continued"
    assert state.status == "active"
    assert state.blocked_turn_count == 0
    assert state.turn_count == 2


@pytest.mark.asyncio
async def test_completion_is_rejected_while_required_task_is_unfinished(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="complete", reason="I think it is done"),
            assistant_message("completion rejected"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-pending-task",
    )
    session.task_store.register_task(
        TaskStateBase(
            id="a_required",
            type="local_agent",
            description="required child work",
            status="running",
            start_time=1.0,
        )
    )
    runtime = GoalRuntime(session)
    runtime.start("wait for child task", goal_id="goal-pending-task")
    runtime.store.update(
        "goal-pending-task",
        lambda current: replace(current, pending_task_ids=("a_required",)),
    )

    result = await runtime.run_idle_once()

    state = runtime.store.get("goal-pending-task")
    assert state is not None
    assert result.stop_reason == "continued"
    assert state.status == "active"
    assert state.turn_count == 1
    assert "required task a_required is not completed" in (state.last_reason or "")


@pytest.mark.asyncio
async def test_completion_allows_completed_required_task(tmp_path: Path) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="complete", reason="child task completed"),
            assistant_message("done"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-completed-task",
    )
    session.task_store.register_task(
        TaskStateBase(
            id="a_required",
            type="local_agent",
            description="required child work",
            status="completed",
            start_time=1.0,
            end_time=2.0,
        )
    )
    runtime = GoalRuntime(session)
    runtime.start("allow completed child", goal_id="goal-completed-task")
    runtime.store.update(
        "goal-completed-task",
        lambda current: replace(current, pending_task_ids=("a_required",)),
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "completed"
    assert result.state is not None
    assert result.state.status == "complete"


@pytest.mark.asyncio
async def test_completion_can_disable_model_self_report(tmp_path: Path) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="complete", reason="I think it is done"),
            assistant_message("completion rejected"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-no-model-complete",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "require external completion authority",
        policy=GoalPolicy(
            completion=GoalCompletionPolicy(allow_model_complete=False)
        ),
        goal_id="goal-no-model-complete",
    )

    result = await runtime.run_idle_once()

    state = runtime.store.get("goal-no-model-complete")
    assert state is not None
    assert result.stop_reason == "continued"
    assert state.status == "active"
    assert "model-reported completion is disabled" in (state.last_reason or "")


@pytest.mark.asyncio
async def test_completion_acceptance_check_policy_requires_evaluator_or_product_check(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _update_goal_response(status="complete", reason="looks complete"),
            assistant_message("completion rejected"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-acceptance-check",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        GoalSpec(
            objective="requires acceptance check",
            acceptance_checks=(GoalAcceptanceCheck(name="test", instruction="run test"),),
        ),
        policy=GoalPolicy(
            completion=GoalCompletionPolicy(require_acceptance_checks=True)
        ),
        goal_id="goal-acceptance-check",
    )

    result = await runtime.run_idle_once()

    state = runtime.store.get("goal-acceptance-check")
    assert state is not None
    assert result.stop_reason == "continued"
    assert state.status == "active"
    assert "acceptance checks require" in (state.last_reason or "")


@pytest.mark.asyncio
async def test_completion_evaluator_can_complete_goal_without_model_update(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("work finished"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-eval-complete",
    )
    evaluator = _CompletionEvaluator(
        evaluations=[
            GoalCompletionEvaluation(
                status="complete",
                reason="Evaluator verified the acceptance checks.",
            )
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(completion_evaluator=evaluator),
    )
    runtime.start(
        GoalSpec(
            objective="requires evaluator completion",
            acceptance_checks=(GoalAcceptanceCheck(name="tests", instruction="pass"),),
        ),
        policy=GoalPolicy(
            completion=GoalCompletionPolicy(
                allow_model_complete=False,
                require_acceptance_checks=True,
            )
        ),
        goal_id="goal-eval-complete",
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "completed"
    assert result.state is not None
    assert result.state.status == "complete"
    assert _metadata_mapping(result.state, "goal_completion_evaluation")["status"] == (
        "complete"
    )
    assert evaluator.calls == 1


@pytest.mark.asyncio
async def test_completion_evaluator_cannot_complete_unfinished_required_task(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("work finished"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-eval-pending-task",
    )
    session.task_store.register_task(
        TaskStateBase(
            id="required_task",
            type="local_agent",
            description="required child work",
            status="running",
            start_time=1.0,
        )
    )
    evaluator = _CompletionEvaluator(
        evaluations=[
            GoalCompletionEvaluation(
                status="complete",
                reason="Evaluator thinks it is done.",
            )
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(completion_evaluator=evaluator),
    )
    runtime.start("respect pending tasks", goal_id="goal-eval-pending-task")
    runtime.store.update(
        "goal-eval-pending-task",
        lambda current: replace(current, pending_task_ids=("required_task",)),
    )

    result = await runtime.run_idle_once()

    state = runtime.store.get("goal-eval-pending-task")
    assert state is not None
    assert result.stop_reason == "continued"
    assert state.status == "active"
    assert "required task required_task is not completed" in (state.last_reason or "")


@pytest.mark.asyncio
async def test_progress_evaluator_blocks_after_no_progress_threshold(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _response("same attempt"),
            _response("same attempt again"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-no-progress",
    )
    evaluator = _ProgressEvaluator(
        ledgers=[
            GoalProgressLedger(
                made_progress=False,
                reason="No new evidence after first turn.",
            ),
            GoalProgressLedger(
                made_progress=False,
                loop_detected=True,
                reason="Repeated the same failing action.",
            ),
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(progress_evaluator=evaluator),
    )
    runtime.start(
        "detect no progress",
        policy=GoalPolicy(
            blocking=GoalBlockingPolicy(max_consecutive_no_progress_turns=2)
        ),
        goal_id="goal-no-progress",
    )

    result = await runtime.run_until_idle(max_continuations=2)

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert result.state.metadata["no_progress_turn_count"] == 2
    assert evaluator.calls == 2


@pytest.mark.asyncio
async def test_progress_evaluator_resets_no_progress_after_productive_turn(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            _response("stalled"),
            _response("made progress"),
            _response("stalled again"),
        )
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-progress-reset",
    )
    evaluator = _ProgressEvaluator(
        ledgers=[
            GoalProgressLedger(made_progress=False, reason="No progress yet."),
            GoalProgressLedger(made_progress=True, reason="New evidence found."),
            GoalProgressLedger(made_progress=False, reason="Another temporary stall."),
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(progress_evaluator=evaluator),
    )
    runtime.start(
        "reset no-progress counter",
        policy=GoalPolicy(
            blocking=GoalBlockingPolicy(max_consecutive_no_progress_turns=2)
        ),
        goal_id="goal-progress-reset",
    )

    first = await runtime.run_idle_once()
    second = await runtime.run_idle_once()
    third = await runtime.run_idle_once()

    state = runtime.store.get("goal-progress-reset")
    assert state is not None
    assert first.stop_reason == "continued"
    assert second.stop_reason == "continued"
    assert third.stop_reason == "continued"
    assert state.status == "active"
    assert state.metadata["no_progress_turn_count"] == 1
    assert _metadata_mapping(state, "goal_progress_ledger")["reason"] == (
        "Another temporary stall."
    )


@pytest.mark.asyncio
async def test_progress_evaluator_can_complete_goal(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(_response("done"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-progress-complete",
    )
    evaluator = _ProgressEvaluator(
        ledgers=[
            GoalProgressLedger(
                request_satisfied=True,
                made_progress=True,
                reason="Ledger says request is satisfied.",
            )
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(progress_evaluator=evaluator),
    )
    runtime.start("complete by ledger", goal_id="goal-progress-complete")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "completed"
    assert result.state is not None
    assert result.state.status == "complete"


@pytest.mark.asyncio
async def test_evaluator_failure_blocks_by_default(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(_response("work"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-eval-failure",
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(completion_evaluator=_FailingCompletionEvaluator()),
    )
    runtime.start("fail evaluator", goal_id="goal-eval-failure")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert _metadata_mapping(result.state, "goal_evaluator_error")["error_type"] == (
        "RuntimeError"
    )


@pytest.mark.asyncio
async def test_named_completion_evaluator_missing_blocks_by_default(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("work"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-missing-eval",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "requires named evaluator",
        policy=GoalPolicy(
            completion=GoalCompletionPolicy(evaluator_name="external_eval")
        ),
        goal_id="goal-missing-eval",
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert _metadata_mapping(result.state, "goal_evaluator_error")["error_type"] == (
        "EvaluatorUnavailable"
    )


@pytest.mark.asyncio
async def test_evaluator_failure_can_be_ignored_by_policy(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(_response("work"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-eval-ignore",
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(
            completion_evaluator=_FailingCompletionEvaluator(),
            evaluator_failure_status="ignore",
        ),
    )
    runtime.start("ignore evaluator failure", goal_id="goal-eval-ignore")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "continued"
    assert result.state is not None
    assert result.state.status == "active"
    assert _metadata_mapping(result.state, "goal_evaluator_error")["status_policy"] == (
        "ignore"
    )


@pytest.mark.asyncio
async def test_ignored_completion_evaluator_failure_still_runs_progress_evaluator(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("work"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-eval-ignore-progress",
    )
    progress = _ProgressEvaluator(
        ledgers=[
            GoalProgressLedger(
                request_satisfied=True,
                made_progress=True,
                reason="Progress evaluator verified completion.",
            )
        ]
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(
            completion_evaluator=_FailingCompletionEvaluator(),
            progress_evaluator=progress,
            evaluator_failure_status="ignore",
        ),
    )
    runtime.start("ignore then progress", goal_id="goal-eval-ignore-progress")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "completed"
    assert result.state is not None
    assert result.state.status == "complete"
    assert _metadata_mapping(result.state, "goal_evaluator_error")["status_policy"] == (
        "ignore"
    )
    assert _metadata_mapping(result.state, "goal_progress_ledger")["request_satisfied"]
    assert progress.calls == 1


@pytest.mark.asyncio
async def test_model_provider_evaluator_malformed_output_blocks_by_default(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("work"),))
    evaluator_provider = FakeModelProvider(responses=(assistant_message("maybe done"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-malformed-eval",
    )
    runtime = GoalRuntime(
        session,
        config=GoalRuntimeConfig(
            completion_evaluator=ModelProviderGoalCompletionEvaluator(
                provider=evaluator_provider
            )
        ),
    )
    runtime.start("malformed evaluator output", goal_id="goal-malformed-eval")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert _metadata_mapping(result.state, "goal_evaluator_error")["error_type"] == (
        "ValueError"
    )
    assert evaluator_provider.requests[0].query_source == "goal_completion_evaluator"


@pytest.mark.asyncio
async def test_preexisting_permission_denial_does_not_block_clean_goal_turn(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(_response("clean continuation"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-stale-denial",
    )
    cast(Any, session.engine)._permission_denials.append(
        PermissionDenial(
            tool_use_id="tu_before_goal",
            tool_name="Write",
            tool_input={"file_path": "old"},
            reason="previously denied",
        )
    )
    runtime = GoalRuntime(session)
    runtime.start("ignore stale permission tombstones", goal_id="goal-stale-denial")

    result = await runtime.run_idle_once()

    state = runtime.store.get("goal-stale-denial")
    assert state is not None
    assert result.stop_reason == "continued"
    assert state.status == "active"
    assert state.last_reason is None
    assert result.sdk_results[0].permission_denials[0].tool_use_id == "tu_before_goal"


@pytest.mark.asyncio
async def test_permission_boundary_blocks_goal_by_default() -> None:
    clock = _MutableClock()
    sdk_result = SDKResult(
        subtype="success",
        session_id="stub-session",
        is_error=False,
        num_turns=1,
        usage=UsageTotals(input_tokens=1),
        permission_denials=(
            PermissionDenial(
                tool_use_id="tu_ask",
                tool_name="Write",
                tool_input={"file_path": "secret"},
                reason="Permission required",
            ),
        ),
    )
    session = _StubSession(sdk_result, clock=clock)
    runtime = GoalRuntime(session)
    runtime.start("wait for approval", goal_id="goal-approval")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert "permission/approval boundary" in (result.state.last_reason or "")


@pytest.mark.asyncio
async def test_permission_boundary_can_fail_by_policy() -> None:
    clock = _MutableClock()
    sdk_result = SDKResult(
        subtype="success",
        session_id="stub-session",
        is_error=False,
        num_turns=1,
        usage=UsageTotals(input_tokens=1),
        permission_denials=(
            PermissionDenial(
                tool_use_id="tu_ask",
                tool_name="Write",
                tool_input={"file_path": "secret"},
                reason="Permission required",
            ),
        ),
    )
    session = _StubSession(sdk_result, clock=clock)
    runtime = GoalRuntime(session)
    runtime.start(
        "fail on approval boundary",
        policy=GoalPolicy(
            approvals=GoalApprovalPolicy(unresolved_approval_status="failed")
        ),
        goal_id="goal-approval-failed",
    )

    result = await runtime.run_idle_once()

    assert result.stop_reason == "failed"
    assert result.state is not None
    assert result.state.status == "failed"
    assert "permission/approval boundary" in (result.state.last_reason or "")


@pytest.mark.asyncio
async def test_default_turn_error_marks_goal_blocked(tmp_path: Path) -> None:
    provider = FakeModelProvider(responses=(RuntimeError("provider down"),))
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-default-error",
    )
    runtime = GoalRuntime(session)
    runtime.start("default error blocks", goal_id="goal-default-error")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "blocked"
    assert result.state is not None
    assert result.state.status == "blocked"
    assert result.state.turn_count == 1


@pytest.mark.asyncio
async def test_consecutive_error_policy_fails_on_second_error(tmp_path: Path) -> None:
    provider = FakeModelProvider(
        responses=(RuntimeError("first"), RuntimeError("second")),
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-two-errors",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "fail on second continuation error",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(max_consecutive_errors=2)),
        goal_id="goal-two-errors",
    )

    first = await runtime.run_idle_once()
    resumed = runtime.resume("goal-two-errors")
    second = await runtime.run_idle_once()

    assert first.stop_reason == "blocked"
    assert resumed.status == "active"
    assert second.stop_reason == "failed"
    assert second.state is not None
    assert second.state.status == "failed"
    assert second.state.turn_count == 2


@pytest.mark.asyncio
async def test_successful_turn_resets_consecutive_error_count(tmp_path: Path) -> None:
    provider = FakeModelProvider(
        responses=(
            RuntimeError("first"),
            assistant_message("recovered"),
            RuntimeError("third"),
        ),
    )
    session = create_raygent(
        provider=provider,
        model="model-1",
        cwd=tmp_path,
        session_id="s-error-reset",
    )
    runtime = GoalRuntime(session)
    runtime.start(
        "reset after success",
        policy=GoalPolicy(blocking=GoalBlockingPolicy(max_consecutive_errors=2)),
        goal_id="goal-error-reset",
    )

    first = await runtime.run_idle_once()
    runtime.resume("goal-error-reset")
    recovered = await runtime.run_idle_once()
    third = await runtime.run_idle_once()

    assert first.stop_reason == "blocked"
    assert recovered.stop_reason == "continued"
    assert third.stop_reason == "blocked"
    assert third.state is not None
    assert third.state.status == "blocked"
    assert third.state.turn_count == 3


@pytest.mark.asyncio
async def test_runtime_events_use_store_emitter_when_config_omits_emitter(
    tmp_path: Path,
) -> None:
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 10.0)
    store = InMemoryGoalStore(event_emitter=emitter)
    session = create_raygent(
        provider=FakeModelProvider(responses=(assistant_message("working"),)),
        model="model-1",
        cwd=tmp_path,
        session_id="s-store-emitter",
    )
    runtime = GoalRuntime(session, store=store)
    runtime.start("emit through store emitter", goal_id="goal-events")

    result = await runtime.run_idle_once()

    assert result.stop_reason == "continued"
    assert [event.type for event in sink.events] == [
        "goal_created",
        "goal_continuation_started",
        "goal_turn_started",
        "goal_updated",
        "goal_turn_accounted",
    ]


@pytest.mark.asyncio
async def test_runtime_events_use_json_store_emitter_when_config_omits_emitter(
    tmp_path: Path,
) -> None:
    sink = InMemoryGoalEventSink()
    emitter = GoalEventEmitter(sinks=(sink,), clock=lambda: 10.0)
    store = JsonGoalStore(tmp_path / "goals", event_emitter=emitter)
    session = create_raygent(
        provider=FakeModelProvider(responses=(assistant_message("working"),)),
        model="model-1",
        cwd=tmp_path,
        session_id="s-json-store-emitter",
    )
    runtime = GoalRuntime(session, store=store)
    runtime.start("emit through json store emitter", goal_id="goal-json-events")

    result = await runtime.run_idle_once()

    expected = [
        "goal_created",
        "goal_continuation_started",
        "goal_turn_started",
        "goal_updated",
        "goal_turn_accounted",
    ]
    assert result.stop_reason == "continued"
    assert [event.type for event in sink.events] == expected
    assert [event.type for event in JsonGoalStore(tmp_path / "goals").list_events()] == expected
