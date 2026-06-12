from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks import in_process_teammate as teammate_mod
from raygent_harness.core.tasks.in_process_teammate import (
    InProcessTeammateState,
    InProcessTeammateTask,
    TeammateMessage,
    queue_pending_message,
    spawn_in_process_teammate,
)
from raygent_harness.core.tool import QueryTracking, ToolUseContext


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="parent-session",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="parent system",
        cwd="/repo",
        query_tracking=QueryTracking(chain_id="parent", depth=0),
    )


def _config() -> QueryConfig:
    return QueryConfig(
        model="m",
        system_prompt="parent system",
        session_id="parent-session",
    )


def _deps(store: AppStateStore) -> QueryDeps:
    return QueryDeps(task_store=store)


async def _wait_for(condition: Any) -> None:
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_in_process_teammate_retains_engine_across_routed_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            self.prompts: list[str] = []
            instances.append(self)

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            self.prompts.append(prompt)
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=len(self.prompts),
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    task_id = await spawn_in_process_teammate(
        name="Researcher",
        team_name="My Team",
        prompt="first prompt",
        parent_agent_id=None,
        parent_config=_config(),
        parent_deps=_deps(store),
        parent_ctx=_ctx(),
        description="research worker",
    )

    await _wait_for(lambda: cast(InProcessTeammateState, store.tasks[task_id]).is_idle)
    assert len(instances) == 1
    assert instances[0].prompts == ["first prompt"]
    first_notifications = store.drain_notifications(None)
    assert len(first_notifications) == 1
    assert "<status>idle</status>" in first_notifications[0].message

    assert queue_pending_message(task_id, "second prompt", store)
    await _wait_for(lambda: instances[0].prompts == ["first prompt", "second prompt"])
    await _wait_for(lambda: cast(InProcessTeammateState, store.tasks[task_id]).is_idle)

    task = cast(InProcessTeammateState, store.tasks[task_id])
    assert task.status == "running"
    assert task.identity is not None
    assert task.identity.agent_id == "researcher@my-team"
    assert task.pending_messages == ()
    assert instances[0].prompts == ["first prompt", "second prompt"]


@pytest.mark.asyncio
async def test_in_process_teammate_drops_parent_non_persistent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[QueryConfig] = []

    class CapturingEngine:
        def __init__(
            self,
            config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            captured_configs.append(config)

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="ok",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)
    store = AppStateStore()
    parent_context: MessageParam = {"role": "user", "content": "parent context"}
    config = replace(
        _config(),
        context_messages=(parent_context,),
        context_system_prompt="parent env context",
    )

    task_id = await spawn_in_process_teammate(
        name="Researcher",
        team_name="My Team",
        prompt="first prompt",
        parent_agent_id=None,
        parent_config=config,
        parent_deps=_deps(store),
        parent_ctx=_ctx(),
        description="research worker",
    )

    await _wait_for(lambda: cast(InProcessTeammateState, store.tasks[task_id]).is_idle)
    assert len(captured_configs) == 1
    assert captured_configs[0].context_messages == ()
    assert captured_configs[0].context_system_prompt == ""


@pytest.mark.asyncio
async def test_in_process_teammate_formats_teammate_messages_for_model_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            self.prompts: list[str] = []
            instances.append(self)

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            self.prompts.append(prompt)
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=len(self.prompts),
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    task_id = await spawn_in_process_teammate(
        name="Researcher",
        team_name="My Team",
        prompt="first prompt",
        parent_agent_id=None,
        parent_config=_config(),
        parent_deps=_deps(store),
        parent_ctx=_ctx(),
    )
    await _wait_for(lambda: cast(InProcessTeammateState, store.tasks[task_id]).is_idle)
    store.drain_notifications(None)

    assert queue_pending_message(
        task_id,
        TeammateMessage(
            sender="team-lead",
            content="check this",
            summary="review request",
        ),
        store,
    )
    await _wait_for(lambda: len(instances[0].prompts) == 2)

    assert instances[0].prompts[1] == (
        '<teammate-message teammate_id="team-lead" summary="review request">\n'
        "check this\n"
        "</teammate-message>"
    )


@pytest.mark.asyncio
async def test_in_process_teammate_kill_flips_status_and_notifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneTurnEngine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", OneTurnEngine)

    store = AppStateStore()
    task_id = await spawn_in_process_teammate(
        name="Researcher",
        team_name="My Team",
        prompt="first prompt",
        parent_agent_id=None,
        parent_config=_config(),
        parent_deps=_deps(store),
        parent_ctx=_ctx(),
    )
    await _wait_for(lambda: cast(InProcessTeammateState, store.tasks[task_id]).is_idle)
    store.drain_notifications(None)

    await InProcessTeammateTask().kill(task_id, store)
    await _wait_for(lambda: store.tasks[task_id].status == "killed")
    await _wait_for(lambda: len(store.notifications) == 1)

    notifications = store.drain_notifications(None)
    assert len(notifications) == 1
    assert "<status>killed</status>" in notifications[0].message
    assert "<name>researcher</name>" in notifications[0].message
