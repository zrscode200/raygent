from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from raygent_harness.coordinator.team import TeamStateStore
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.tools.team_create_tool import (
    TEAM_CREATE_TOOL_NAME,
    TeamCreateInput,
    build_team_create_tool,
    create_team_create_catalog_provider,
)


class EmptyInput(BaseModel):
    pass


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _base_tool(name: str) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} tool",
            input_model=EmptyInput,
            call=_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx(
    *,
    tools: Sequence[Tool] = (),
    agent_id: str | None = None,
    discovered: Sequence[str] | None = None,
) -> ToolUseContext:
    if discovered is None:
        discovered = (TEAM_CREATE_TOOL_NAME,)
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd="/repo",
        tools=tuple(tools),
        query_tracking=QueryTracking(chain_id="c", depth=0),
        discovered_tool_names=frozenset(discovered),
    )


def _deps() -> QueryDeps:
    return QueryDeps(task_store=AppStateStore())


def _tool_use(input_: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        id="toolu_team",
        name=TEAM_CREATE_TOOL_NAME,
        input=input_,
        index=0,
    )


async def _run_team_create(
    *,
    tool: Tool,
    deps: QueryDeps,
    ctx: ToolUseContext,
    input_: dict[str, Any],
) -> ToolExecutionResult:
    results = [
        event
        async for event in run_tool_use(
            tool_use=_tool_use(input_),
            assistant_message={"role": "assistant", "content": []},
            tools=ctx.tools,
            deps=deps,
            ctx=ctx,
        )
        if isinstance(event, ToolExecutionResult)
    ]
    assert len(results) == 1
    return results[0]


def _content(result: ToolExecutionResult) -> str | list[dict[str, Any]]:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return block["content"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_team_create_writes_project_local_metadata(tmp_path: Path) -> None:
    store = TeamStateStore(base_dir=tmp_path / "teams")
    tool = build_team_create_tool(team_store=store, current_model="m")
    ctx = _ctx(tools=(tool,))

    result = await _run_team_create(
        tool=tool,
        deps=_deps(),
        ctx=ctx,
        input_={
            "team_name": "My Project",
            "description": "build feature",
            "agent_type": "coordinator",
        },
    )

    content = _content(result)
    assert isinstance(content, list)
    metadata = content[1]
    assert metadata == {
        "type": "team_created",
        "team_name": "my-project",
        "team_file_path": str(tmp_path / "teams" / "my-project" / "config.json"),
        "lead_agent_id": "team-lead@my-project",
    }
    assert store.current_team is not None
    assert store.current_team.team_name == "my-project"
    data = json.loads((tmp_path / "teams" / "my-project" / "config.json").read_text())
    assert data["team_name"] == "my-project"
    assert data["members"][0]["agent_id"] == "team-lead@my-project"
    assert data["members"][0]["agent_type"] == "coordinator"
    assert data["members"][0]["model"] == "m"


@pytest.mark.asyncio
async def test_team_create_rejects_empty_subagent_and_current_team(
    tmp_path: Path,
) -> None:
    store = TeamStateStore(base_dir=tmp_path / "teams")
    tool = build_team_create_tool(team_store=store, current_model="m")

    empty = await tool.validate_input(TeamCreateInput(team_name="  "), _ctx(tools=(tool,)))
    subagent = await tool.validate_input(
        TeamCreateInput(team_name="team"),
        _ctx(tools=(tool,), agent_id="a_child"),
    )
    first = await _run_team_create(
        tool=tool,
        deps=_deps(),
        ctx=_ctx(tools=(tool,)),
        input_={"team_name": "team"},
    )
    duplicate = await tool.validate_input(
        TeamCreateInput(team_name="other"),
        _ctx(tools=(tool,)),
    )

    assert isinstance(empty, ValidationError)
    assert "team_name is required" in empty.message
    assert isinstance(subagent, ValidationError)
    assert "main coordinator" in subagent.message
    assert not _is_error(first)
    assert isinstance(duplicate, ValidationError)
    assert 'Already leading team "team"' in duplicate.message


@pytest.mark.asyncio
async def test_team_create_uses_deterministic_suffix_for_existing_file(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "teams" / "team" / "config.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    store = TeamStateStore(base_dir=tmp_path / "teams")
    tool = build_team_create_tool(team_store=store, current_model="m")

    result = await _run_team_create(
        tool=tool,
        deps=_deps(),
        ctx=_ctx(tools=(tool,)),
        input_={"team_name": "team"},
    )

    content = _content(result)
    assert isinstance(content, list)
    assert content[1]["team_name"] == "team-2"
    assert (tmp_path / "teams" / "team-2" / "config.json").exists()


@pytest.mark.asyncio
async def test_team_create_catalog_provider_enabled_main_thread_only(
    tmp_path: Path,
) -> None:
    base = _base_tool("Read")
    store = TeamStateStore(base_dir=tmp_path / "teams")
    provider = create_team_create_catalog_provider(team_store=store, enabled=True)
    disabled = create_team_create_catalog_provider(team_store=store, enabled=False)
    config = QueryConfig(model="m", tools=(base,))

    main_tools = await provider(config, _ctx(tools=(base,)), ())
    child_tools = await provider(config, _ctx(tools=(base,), agent_id="a_child"), ())
    disabled_tools = await disabled(config, _ctx(tools=(base,)), ())

    assert main_tools is not None
    assert tuple(tool.name for tool in main_tools) == ("Read", TEAM_CREATE_TOOL_NAME)
    assert child_tools is not None
    assert tuple(tool.name for tool in child_tools) == ("Read",)
    assert disabled_tools is not None
    assert tuple(tool.name for tool in disabled_tools) == ("Read",)


def _is_error(result: ToolExecutionResult) -> bool:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return bool(block.get("is_error", False))
