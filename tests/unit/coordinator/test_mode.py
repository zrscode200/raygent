from __future__ import annotations

import asyncio

import pytest

from raygent_harness.coordinator.mode import (
    CoordinatorModeConfig,
    build_coordinator_user_context,
    create_coordinator_system_prompt_provider,
    create_coordinator_tool_catalog_provider,
    create_coordinator_user_context_provider,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.query_engine import QueryEngine
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.tools.task_stop_tool import TASK_STOP_TOOL_NAME
from raygent_harness.tools.team_create_tool import TEAM_CREATE_TOOL_NAME


def _ctx(*, agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


@pytest.mark.asyncio
async def test_coordinator_prompt_provider_injects_only_when_enabled_main_thread() -> None:
    provider = create_coordinator_system_prompt_provider(
        CoordinatorModeConfig(
            enabled=True,
            worker_tool_names=("Read", "Bash"),
            mcp_server_names=("github",),
            scratchpad_dir=".raygent/scratch",
            include_team_tools=True,
        )
    )
    base = QueryConfig(model="m", system_prompt="base", session_id="s")
    deps = QueryDeps(
        task_store=AppStateStore(),
        system_prompt_provider=provider,
    )
    engine = QueryEngine(base, deps, _ctx())

    turn_config = await engine._build_turn_config(_ctx())  # pyright: ignore[reportPrivateUsage]
    child_turn_config = await engine._build_turn_config(  # pyright: ignore[reportPrivateUsage]
        _ctx(agent_id="a_child")
    )
    disabled_provider = create_coordinator_system_prompt_provider(
        CoordinatorModeConfig(enabled=False, worker_tool_names=("Read",))
    )
    disabled = await disabled_provider(base, _ctx())

    assert turn_config.system_prompt.startswith("base\n\n## Coordinator Mode")
    assert "Use `Agent` to launch independent background workers" in turn_config.system_prompt
    assert "<task_notification>" in turn_config.system_prompt
    assert "Workers may use MCP-backed tools from: github" not in turn_config.system_prompt
    assert "Scratchpad directory: .raygent/scratch" not in turn_config.system_prompt
    assert "non-persistent user context" in turn_config.system_prompt
    assert "Named Agent launches create addressable headless teammates" in (
        turn_config.system_prompt
    )
    assert "use SendMessage to route follow-up work" in turn_config.system_prompt
    assert child_turn_config.system_prompt == "base"
    assert disabled is None


def test_coordinator_user_context_is_reference_shaped_and_headless() -> None:
    context = build_coordinator_user_context(
        CoordinatorModeConfig(
            enabled=True,
            worker_tool_names=("Write", "Read"),
            mcp_server_names=("github", "linear"),
            scratchpad_dir=".raygent/scratch",
        )
    )

    assert tuple(context) == ("workerToolsContext",)
    assert "Workers spawned via Agent can use: Read, Write." in context["workerToolsContext"]
    assert "github, linear" in context["workerToolsContext"]
    assert ".raygent/scratch" in context["workerToolsContext"]
    assert build_coordinator_user_context(CoordinatorModeConfig(enabled=False)) == {}


@pytest.mark.asyncio
async def test_coordinator_worker_tools_context_uses_non_persistent_user_lane() -> None:
    settings = CoordinatorModeConfig(
        enabled=True,
        worker_tool_names=("Write", "Read"),
        mcp_server_names=("github", "linear"),
        scratchpad_dir=".raygent/scratch",
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        system_prompt_provider=create_coordinator_system_prompt_provider(settings),
        context_providers=(create_coordinator_user_context_provider(settings),),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, _ctx())

    turn_config = await engine._build_turn_config(_ctx())  # pyright: ignore[reportPrivateUsage]
    child_turn_config = await engine._build_turn_config(  # pyright: ignore[reportPrivateUsage]
        _ctx(agent_id="a_child")
    )

    assert len(turn_config.context_messages) == 1
    content = str(turn_config.context_messages[0]["content"])
    assert "# workerToolsContext" in content
    assert "Workers spawned via Agent can use: Read, Write." in content
    assert "github, linear" in content
    assert ".raygent/scratch" in content
    assert "Workers spawned via Agent can use" not in turn_config.system_prompt
    assert child_turn_config.context_messages == ()


@pytest.mark.asyncio
async def test_coordinator_tool_catalog_provider_adds_main_thread_tools_only() -> None:
    deps = QueryDeps(task_store=AppStateStore())
    provider = create_coordinator_tool_catalog_provider(parent_deps=deps, enabled=True)
    disabled = create_coordinator_tool_catalog_provider(parent_deps=deps, enabled=False)
    config = QueryConfig(model="m", session_id="s")

    main_tools = await provider(config, _ctx(), ())
    child_tools = await provider(config, _ctx(agent_id="a_child"), ())
    disabled_tools = await disabled(config, _ctx(), ())

    assert main_tools is not None
    assert tuple(tool.name for tool in main_tools) == (
        TASK_STOP_TOOL_NAME,
        TEAM_CREATE_TOOL_NAME,
    )
    assert child_tools == ()
    assert disabled_tools == ()
