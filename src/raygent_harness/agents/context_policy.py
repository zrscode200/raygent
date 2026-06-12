"""Agent context-policy helpers."""

from __future__ import annotations

from dataclasses import replace

from raygent_harness.agents.models import AgentDefinition
from raygent_harness.core.context_providers import filter_context_providers_by_kind
from raygent_harness.core.deps import QueryDeps


def deps_for_agent_context_policy(
    deps: QueryDeps,
    agent: AgentDefinition | None,
) -> QueryDeps:
    """Return deps with context providers filtered for a child agent."""

    if agent is None or not agent.context_policy.omit_context_kinds:
        return deps
    omitted = agent.context_policy.omit_context_kinds
    return replace(
        deps,
        context_providers=filter_context_providers_by_kind(
            deps.context_providers,
            omitted_kinds=omitted,
        ),
        post_tool_context_providers=filter_context_providers_by_kind(
            deps.post_tool_context_providers,
            omitted_kinds=omitted,
        ),
    )


__all__ = ["deps_for_agent_context_policy"]
