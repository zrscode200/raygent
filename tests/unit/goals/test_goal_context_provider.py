from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import (
    context_provider_kind,
    filter_context_providers_by_kind,
    scope_context_fragments,
)
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam, message_param_from_api_message
from raygent_harness.core.query import LayerResult
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.goals import (
    GoalCheckpoint,
    GoalContextProvider,
    GoalSpec,
    InMemoryGoalStore,
    JsonGoalStore,
    create_goal_state,
)
from tests.fakes import FakeModelProvider


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _store() -> InMemoryGoalStore:
    store = InMemoryGoalStore()
    store.create(
        create_goal_state(
            goal_id="g_1",
            session_id="s",
            spec=GoalSpec(
                objective="Finish wave 2",
                success_criteria=("goal context visible",),
            ),
            now=1.0,
        )
    )
    return store


@pytest.mark.asyncio
async def test_goal_context_provider_returns_active_goal_context() -> None:
    provider = GoalContextProvider(_store())

    fragments = await provider(QueryConfig(model="model-1", session_id="s"), _ctx())

    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.id == "goal_context:g_1"
    assert fragment.channel == "user_context"
    assert fragment.kind == "goal"
    assert fragment.agent_scope == "main"
    assert "Finish wave 2" in fragment.content
    assert "success_criteria" in fragment.content
    assert context_provider_kind(provider) == "goal"
    assert filter_context_providers_by_kind(
        (provider,),
        omitted_kinds=("goal",),
    ) == ()


@pytest.mark.asyncio
async def test_goal_context_provider_is_session_scoped_and_main_agent_scoped() -> None:
    provider = GoalContextProvider(_store())
    child_ctx = ToolUseContext(
        session_id="s",
        agent_id="child",
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )
    other_session_ctx = ToolUseContext(
        session_id="other",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )

    child_fragments = await provider(QueryConfig(model="model-1", session_id="s"), child_ctx)
    other_fragments = await provider(
        QueryConfig(model="model-1", session_id="other"),
        other_session_ctx,
    )

    assert len(child_fragments) == 1
    assert scope_context_fragments(child_fragments, agent_id="child") == ()
    assert other_fragments == ()


@pytest.mark.asyncio
async def test_goal_context_is_model_visible_after_compaction_boundary() -> None:
    provider = GoalContextProvider(_store())
    model = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    seen_microcompact_inputs: list[list[MessageParam]] = []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        seen_microcompact_inputs.append(list(messages))
        return LayerResult(
            messages=[{"role": "user", "content": "[compacted goal work]"}]
        )

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=(provider,),
        microcompact=fake_microcompact,
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s"),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("work")]

    assert isinstance(events[-1], SDKResult)
    assert seen_microcompact_inputs == [[{"role": "user", "content": "work"}]]
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    assert len(request_messages) == 2
    context_message = request_messages[0]
    assert context_message["role"] == "user"
    assert "raygent_goal_context" in str(context_message["content"])
    assert "Finish wave 2" in str(context_message["content"])
    assert request_messages[1] == {"role": "user", "content": "[compacted goal work]"}
    assert all("Finish wave 2" not in str(message) for message in engine._messages)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_json_goal_state_survives_compacted_transcript_loss(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "goals"
    store = JsonGoalStore(store_dir)
    state = replace(
        create_goal_state(
            goal_id="g_json",
            session_id="s",
            spec=GoalSpec(
                objective="Durable compacted goal",
                success_criteria=("survives compaction",),
            ),
            now=1.0,
        ),
        summary="durable goal summary",
        checkpoints=(
            GoalCheckpoint(
                checkpoint_id="cp_1",
                summary="compaction-safe checkpoint",
                created_at=2.0,
            ),
        ),
    )
    store.create(state)
    provider = GoalContextProvider(store)
    model = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))

    async def fake_microcompact(
        _messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        return LayerResult(
            messages=[{"role": "user", "content": "[compacted transcript only]"}]
        )

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=(provider,),
        microcompact=fake_microcompact,
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s"),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("work before compact")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    assert "Durable compacted goal" in str(request_messages[0]["content"])
    assert "durable goal summary" in str(request_messages[0]["content"])
    assert request_messages[1] == {
        "role": "user",
        "content": "[compacted transcript only]",
    }
    assert all(
        "Durable compacted goal" not in str(message)
        for message in engine._messages  # pyright: ignore[reportPrivateUsage]
    )
    reloaded = JsonGoalStore(store_dir).get("g_json")
    assert reloaded is not None
    assert reloaded.summary == "durable goal summary"
    assert reloaded.checkpoints[0].summary == "compaction-safe checkpoint"


def test_goal_context_kind_is_filterable() -> None:
    provider = GoalContextProvider(_store())
    retained: Sequence[object] = filter_context_providers_by_kind(
        (provider,),
        omitted_kinds=("memory",),
    )

    assert retained == (provider,)
