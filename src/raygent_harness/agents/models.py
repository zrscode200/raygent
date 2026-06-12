"""Agent definition data model.


Raygent keeps this as a headless, dependency-light subset of the reference
shape. Product/UI fields such as colors, telemetry, plugin provenance, and
worktree bookkeeping are intentionally absent or carried as raw metadata until
the coordinator layer needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from raygent_harness.core.context_providers import ContextKind
from raygent_harness.core.permissions import PermissionMode

AgentSource = Literal[
    "built-in",
    "project",
    "user",
    "policy",
    "plugin",
]

AgentIsolation = Literal["worktree", "remote"]


@dataclass(frozen=True)
class AgentContextPolicy:
    """Context-provider policy applied when this agent runs as a child."""

    omit_context_kinds: tuple[ContextKind, ...] = ()

    @classmethod
    def inherit(cls) -> AgentContextPolicy:
        return cls()

    @classmethod
    def omit_project_instructions(cls) -> AgentContextPolicy:
        return cls(omit_context_kinds=("project_instructions",))

    @classmethod
    def omit_git(cls) -> AgentContextPolicy:
        return cls(omit_context_kinds=("git",))

    @classmethod
    def minimal(cls) -> AgentContextPolicy:
        return cls(omit_context_kinds=("project_instructions", "git"))


@dataclass(frozen=True)
class AgentDefinition:
    """Headless agent definition used by AgentTool.

    `description` is the reference `whenToUse` field: it is what the parent
    model sees in the Agent tool prompt and should explain when to delegate to
    this agent.
    """

    agent_type: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    permission_mode: PermissionMode | None = None
    background: bool = False
    skills: tuple[str, ...] = ()
    mcp_servers: tuple[object, ...] = ()
    required_mcp_servers: tuple[str, ...] = ()
    isolation: AgentIsolation | None = None
    context_policy: AgentContextPolicy = field(default_factory=AgentContextPolicy)
    source: AgentSource = "project"
    initial_prompt: str | None = None


__all__ = [
    "AgentContextPolicy",
    "AgentDefinition",
    "AgentIsolation",
    "AgentSource",
]
