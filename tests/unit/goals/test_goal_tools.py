from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from raygent_harness.core.permissions import empty_tool_permission_context
from raygent_harness.core.tool import Tool, ToolResult, ToolUseContext
from raygent_harness.goals import (
    GET_GOAL_TOOL_NAME,
    UPDATE_GOAL_TOOL_NAME,
    GetGoalInput,
    GoalSpec,
    GoalStatus,
    InMemoryGoalStore,
    UpdateGoalInput,
    build_get_goal_tool,
    build_update_goal_tool,
    create_goal_state,
)


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )


async def _collect_result(tool: Tool, input_: BaseModel) -> ToolResult:
    events = [event async for event in tool.call(input_, _ctx())]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


def _seed_store() -> InMemoryGoalStore:
    store = InMemoryGoalStore()
    store.create(
        create_goal_state(
            goal_id="g_1",
            session_id="s",
            spec=GoalSpec(
                objective="Finish the goal runner",
                success_criteria=("tests pass",),
            ),
            now=1.0,
        )
    )
    return store


@pytest.mark.asyncio
async def test_get_goal_tool_inspects_active_goal() -> None:
    store = _seed_store()
    tool = build_get_goal_tool(store=store)
    input_ = GetGoalInput()

    result = await _collect_result(tool, input_)

    assert tool.name == GET_GOAL_TOOL_NAME
    assert tool.is_read_only(input_) is True
    assert tool.is_concurrency_safe(input_) is True
    assert tool.is_open_world(input_) is False
    assert result.is_error is False
    assert isinstance(result.content, str)
    assert "goal_id: g_1" in result.content
    assert "objective: Finish the goal runner" in result.content
    assert "success_criteria:" in result.content


@pytest.mark.asyncio
async def test_update_goal_tool_allows_only_complete_or_blocked() -> None:
    store = _seed_store()
    tool = build_update_goal_tool(store=store)
    valid = UpdateGoalInput(status="complete", reason="All acceptance checks passed.")
    invalid = UpdateGoalInput(status="paused", reason="I want to pause.")

    validation = await tool.validate_input(valid, _ctx())
    invalid_validation = await tool.validate_input(invalid, _ctx())
    permission = await tool.check_permissions(
        valid,
        _ctx(),
        empty_tool_permission_context(),
    )
    result = await _collect_result(tool, valid)

    assert tool.name == UPDATE_GOAL_TOOL_NAME
    assert tool.is_read_only(valid) is False
    assert tool.is_destructive(valid) is False
    assert tool.is_open_world(valid) is False
    assert validation.result == "ok"
    assert invalid_validation.result == "error"
    assert permission.behavior == "allow"
    assert result.is_error is False
    assert store.get("g_1").status == "complete"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_goal_tools_report_missing_active_goal() -> None:
    store = InMemoryGoalStore()
    get_tool = build_get_goal_tool(store=store)
    update_tool = build_update_goal_tool(store=store)

    get_result = await _collect_result(get_tool, GetGoalInput())
    update_result = await _collect_result(
        update_tool,
        UpdateGoalInput(status="blocked", reason="No work available."),
    )

    assert get_result.is_error is True
    assert update_result.is_error is True


@pytest.mark.asyncio
async def test_goal_tools_scope_explicit_goal_id_to_current_session() -> None:
    store = InMemoryGoalStore()
    store.create(
        create_goal_state(
            goal_id="g_current",
            session_id="s",
            spec=GoalSpec(objective="current session"),
            now=1.0,
        )
    )
    store.create(
        create_goal_state(
            goal_id="g_other",
            session_id="other",
            spec=GoalSpec(objective="other session"),
            now=2.0,
        )
    )
    get_tool = build_get_goal_tool(store=store)
    update_tool = build_update_goal_tool(store=store)

    get_result = await _collect_result(get_tool, GetGoalInput(goal_id="g_other"))
    update_result = await _collect_result(
        update_tool,
        UpdateGoalInput(
            goal_id="g_other",
            status="complete",
            reason="I know the id.",
        ),
    )

    assert get_result.is_error is True
    assert update_result.is_error is True
    assert store.get("g_other").status == "active"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "status",
    [
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
        "cancelled",
        "failed",
    ],
)
@pytest.mark.asyncio
async def test_update_goal_tool_rejects_non_active_goals(status: GoalStatus) -> None:
    store = _seed_store()
    store.update(
        "g_1",
        lambda current: current.with_status(
            status,
            reason="runtime controlled",
            now=2.0,
        ),
    )
    tool = build_update_goal_tool(store=store)

    result = await _collect_result(
        tool,
        UpdateGoalInput(
            goal_id="g_1",
            status="complete",
            reason="Trying to override runtime state.",
        ),
    )

    assert result.is_error is True
    assert store.get("g_1").status == status  # type: ignore[union-attr]
