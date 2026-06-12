"""Agent definition registry helpers.


Chunk 6 deliberately starts with a small Raygent-owned built-in registry. File
and plugin loading use the same `AgentDefinition` model and can be layered on
later without changing AgentTool's call contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from raygent_harness.agents.models import AgentContextPolicy, AgentDefinition
from raygent_harness.core.permission_engine import get_rules
from raygent_harness.core.permissions import ToolPermissionContext

DEFAULT_AGENT_TYPE = "worker"

WORKER_AGENT = AgentDefinition(
    agent_type=DEFAULT_AGENT_TYPE,
    description=(
        "General-purpose worker for multi-step implementation, research, and "
        "verification tasks."
    ),
    system_prompt=(
        "You are a focused Raygent worker agent. Complete the delegated task "
        "fully, report concise findings, and avoid unrelated changes."
    ),
    tools=("*",),
    source="built-in",
)

EXPLORER_AGENT = AgentDefinition(
    agent_type="explorer",
    description=(
        "Read-only codebase exploration agent for finding files, tracing "
        "architecture, and answering implementation questions."
    ),
    system_prompt=(
        "You are a read-only Raygent explorer agent. Search and inspect the "
        "codebase, but do not create, edit, delete, or move files."
    ),
    tools=("*",),
    disallowed_tools=("Agent", "Task", "Edit", "Write", "NotebookEdit"),
    permission_mode="plan",
    context_policy=AgentContextPolicy.minimal(),
    source="built-in",
)

REVIEWER_AGENT = AgentDefinition(
    agent_type="reviewer",
    description=(
        "Independent review agent for checking behavioral correctness, "
        "fidelity, and missing tests."
    ),
    system_prompt=(
        "You are a Raygent reviewer agent. Prioritize concrete findings with "
        "file references, behavioral impact, and required fixes."
    ),
    tools=("*",),
    disallowed_tools=("Agent", "Task"),
    source="built-in",
)


def get_builtin_agent_definitions() -> tuple[AgentDefinition, ...]:
    """Return Raygent-owned built-in agents."""

    return (WORKER_AGENT, EXPLORER_AGENT, REVIEWER_AGENT)


def find_agent_definition(
    agent_type: str,
    agents: Sequence[AgentDefinition],
) -> AgentDefinition | None:
    for agent in agents:
        if agent.agent_type == agent_type:
            return agent
    return None


def filter_denied_agents(
    agents: Sequence[AgentDefinition],
    permission_context: ToolPermissionContext,
    *,
    tool_name: str,
) -> tuple[AgentDefinition, ...]:
    """Remove agents denied by `Agent(agent-type)` permission rules."""

    return tuple(
        agent
        for agent in agents
        if not is_agent_denied(agent.agent_type, permission_context, tool_name=tool_name)
    )


def is_agent_denied(
    agent_type: str,
    permission_context: ToolPermissionContext,
    *,
    tool_name: str,
) -> bool:
    return _matching_agent_rule(
        permission_context,
        behavior="deny",
        tool_name=tool_name,
        agent_type=agent_type,
    ) is not None


def _matching_agent_rule(
    permission_context: ToolPermissionContext,
    *,
    behavior: str,
    tool_name: str,
    agent_type: str,
) -> object | None:
    for rule in get_rules(permission_context, behavior):  # type: ignore[arg-type]
        value = rule.rule_value
        if value.tool_name != tool_name:
            continue
        if value.rule_content is None or value.rule_content == agent_type:
            return rule
    return None


__all__ = [
    "DEFAULT_AGENT_TYPE",
    "EXPLORER_AGENT",
    "REVIEWER_AGENT",
    "WORKER_AGENT",
    "filter_denied_agents",
    "find_agent_definition",
    "get_builtin_agent_definitions",
    "is_agent_denied",
]
