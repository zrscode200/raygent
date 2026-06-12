from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel

from raygent_harness.agents.models import AgentDefinition
from raygent_harness.agents.tool_pool import AGENT_TOOL_NAME, resolve_agent_tools
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)


class EmptyInput(BaseModel):
    pass


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _tool(name: str, *, aliases: tuple[str, ...] = ()) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            aliases=aliases,
            description=f"{name} tool",
            input_model=EmptyInput,
            call=_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def test_wildcard_resolves_all_filtered_tools_minus_disallowed() -> None:
    agent = AgentDefinition(
        agent_type="worker",
        description="worker",
        system_prompt="worker",
        tools=("*",),
        disallowed_tools=("Write",),
    )
    read = _tool("Read")
    write = _tool("Write")
    agent_tool = _tool(AGENT_TOOL_NAME)

    resolved = resolve_agent_tools(agent, (read, write, agent_tool), is_main_thread=False)

    assert resolved.has_wildcard
    assert tuple(tool.name for tool in resolved.resolved_tools) == ("Read",)


def test_async_wildcard_applies_reference_allowlist_and_mcp_exception() -> None:
    agent = AgentDefinition(
        agent_type="worker",
        description="worker",
        system_prompt="worker",
        tools=("*",),
    )
    read = _tool("Read")
    skill = _tool("Skill")
    mcp = _tool("mcp__github__search")
    task_output = _tool("TaskOutput")
    random = _tool("RandomDanger")

    resolved = resolve_agent_tools(
        agent,
        (read, skill, mcp, task_output, random),
        is_async=True,
        is_main_thread=False,
    )

    assert tuple(tool.name for tool in resolved.resolved_tools) == (
        "Read",
        "Skill",
        "mcp__github__search",
    )


def test_explicit_specs_track_valid_invalid_and_agent_type_metadata() -> None:
    agent = AgentDefinition(
        agent_type="coordinator",
        description="coord",
        system_prompt="coord",
        tools=("Read", "Agent(worker,reviewer)", "Missing"),
    )
    read = _tool("Read")
    agent_tool = _tool(AGENT_TOOL_NAME)

    subagent_resolved = resolve_agent_tools(
        agent,
        (read, agent_tool),
        is_main_thread=False,
    )
    main_resolved = resolve_agent_tools(
        agent,
        (read, agent_tool),
        is_main_thread=True,
    )

    assert subagent_resolved.valid_tools == ("Read", "Agent(worker,reviewer)")
    assert subagent_resolved.invalid_tools == ("Missing",)
    assert tuple(tool.name for tool in subagent_resolved.resolved_tools) == ("Read",)
    assert subagent_resolved.allowed_agent_types == ("worker", "reviewer")

    assert tuple(tool.name for tool in main_resolved.resolved_tools) == (
        "Read",
        AGENT_TOOL_NAME,
    )
    assert main_resolved.allowed_agent_types == ("worker", "reviewer")
