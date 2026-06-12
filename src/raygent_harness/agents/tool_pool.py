"""Agent-specific tool-pool resolution.


The important fidelity property is that workers do not blindly inherit the
parent's currently model-visible catalog. Agent definitions select from the
runtime tool pool, subtract disallowed tools, and can carry allowed-agent-type
metadata through `Agent(worker,reviewer)` specs for main-thread use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from raygent_harness.agents.models import AgentDefinition
from raygent_harness.core.permission_engine import permission_rule_value_from_string
from raygent_harness.core.permissions import PermissionMode
from raygent_harness.core.tool import Tool, find_tool_by_name, tool_matches_name

AGENT_TOOL_NAME = "Agent"
LEGACY_AGENT_TOOL_NAME = "Task"
EXIT_PLAN_MODE_TOOL_NAME = "ExitPlanMode"
EXIT_PLAN_MODE_V2_TOOL_NAME = "ExitPlanModeV2"

ALL_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        AGENT_TOOL_NAME,
        LEGACY_AGENT_TOOL_NAME,
        "AskUserQuestion",
        "EnterPlanMode",
        EXIT_PLAN_MODE_TOOL_NAME,
        EXIT_PLAN_MODE_V2_TOOL_NAME,
        "Stop",
        "TaskOutput",
        "TaskStop",
        "Workflow",
    }
)

ASYNC_AGENT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "Bash",
        "Edit",
        "EnterWorktree",
        "ExitWorktree",
        "Glob",
        "Grep",
        "NotebookEdit",
        "Read",
        "Skill",
        "SyntheticOutput",
        "TodoWrite",
        "ToolSearch",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)
"""Reference async-agent allowlist, translated to Raygent tool names.

MCP tools are allowed separately by `mcp__server__tool` prefix, matching the
reference early-return in `filterToolsForAgent`.
"""


@dataclass(frozen=True)
class ResolvedAgentTools:
    has_wildcard: bool
    valid_tools: tuple[str, ...]
    invalid_tools: tuple[str, ...]
    resolved_tools: tuple[Tool, ...]
    allowed_agent_types: tuple[str, ...] | None = None


def resolve_agent_tools(
    agent_definition: AgentDefinition,
    available_tools: Sequence[Tool],
    *,
    is_async: bool = True,
    is_main_thread: bool = False,
) -> ResolvedAgentTools:
    """Resolve an agent definition's tool specs against available tools.

    `is_async` is present for parity with the reference resolver. Raygent's v1
    filter only needs the main-thread distinction and agent disallow list.
    """

    filtered_available = (
        tuple(available_tools)
        if is_main_thread
        else _filter_tools_for_agent(
            available_tools,
            permission_mode=agent_definition.permission_mode,
            is_async=is_async,
        )
    )
    allowed_available = _subtract_disallowed_tools(
        filtered_available,
        agent_definition.disallowed_tools,
    )

    raw_tool_specs = agent_definition.tools
    has_wildcard = raw_tool_specs is None or tuple(raw_tool_specs) == ("*",)
    if has_wildcard:
        return ResolvedAgentTools(
            has_wildcard=True,
            valid_tools=(),
            invalid_tools=(),
            resolved_tools=tuple(allowed_available),
        )

    valid: list[str] = []
    invalid: list[str] = []
    resolved: list[Tool] = []
    seen_tool_names: set[str] = set()
    allowed_agent_types: tuple[str, ...] | None = None
    tool_specs = tuple(raw_tool_specs or ())

    for spec in tool_specs:
        parsed = permission_rule_value_from_string(spec)
        tool_name = parsed.tool_name

        if tool_name in {AGENT_TOOL_NAME, LEGACY_AGENT_TOOL_NAME}:
            if parsed.rule_content:
                allowed_agent_types = _parse_allowed_agent_types(parsed.rule_content)
            if not is_main_thread:
                valid.append(spec)
                continue

        tool = find_tool_by_name(allowed_available, tool_name)
        if tool is None:
            invalid.append(spec)
            continue
        valid.append(spec)
        if tool.name not in seen_tool_names:
            resolved.append(tool)
            seen_tool_names.add(tool.name)

    return ResolvedAgentTools(
        has_wildcard=False,
        valid_tools=tuple(valid),
        invalid_tools=tuple(invalid),
        resolved_tools=tuple(resolved),
        allowed_agent_types=allowed_agent_types,
    )


def _filter_tools_for_agent(
    tools: Sequence[Tool],
    *,
    permission_mode: PermissionMode | None,
    is_async: bool,
) -> tuple[Tool, ...]:
    return tuple(
        tool
        for tool in tools
        if _tool_allowed_for_agent(
            tool,
            permission_mode=permission_mode,
            is_async=is_async,
        )
    )


def _tool_allowed_for_agent(
    tool: Tool,
    *,
    permission_mode: PermissionMode | None,
    is_async: bool,
) -> bool:
    if tool.name.startswith("mcp__"):
        return True
    if (
        _matches_any_tool_name(tool, {EXIT_PLAN_MODE_TOOL_NAME, EXIT_PLAN_MODE_V2_TOOL_NAME})
        and permission_mode == "plan"
    ):
        return True
    if _matches_any_tool_name(tool, ALL_AGENT_DISALLOWED_TOOLS):
        return False
    return not (is_async and not _matches_any_tool_name(tool, ASYNC_AGENT_ALLOWED_TOOLS))


def _subtract_disallowed_tools(
    tools: Sequence[Tool],
    disallowed_specs: Sequence[str],
) -> tuple[Tool, ...]:
    disallowed_names = {
        permission_rule_value_from_string(spec).tool_name for spec in disallowed_specs
    }
    return tuple(
        tool
        for tool in tools
        if not any(tool_matches_name(tool, name) for name in disallowed_names)
    )


def _matches_any_tool_name(tool: Tool, names: frozenset[str] | set[str]) -> bool:
    return any(tool_matches_name(tool, name) for name in names)


def _parse_allowed_agent_types(rule_content: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in rule_content.split(",")
        if item.strip()
    )


__all__ = [
    "AGENT_TOOL_NAME",
    "ALL_AGENT_DISALLOWED_TOOLS",
    "ASYNC_AGENT_ALLOWED_TOOLS",
    "LEGACY_AGENT_TOOL_NAME",
    "ResolvedAgentTools",
    "resolve_agent_tools",
]
