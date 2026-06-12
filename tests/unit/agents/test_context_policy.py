from __future__ import annotations

from typing import Any

from raygent_harness.agents.context_policy import deps_for_agent_context_policy
from raygent_harness.agents.loader import EXPLORER_AGENT
from raygent_harness.agents.models import AgentContextPolicy, AgentDefinition
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import (
    ContextFragment,
    ContextKind,
    context_provider_kind,
    filter_context_providers_by_kind,
)
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import ToolUseContext


class TypedProvider:
    context_kind: ContextKind

    def __init__(self, kind: ContextKind) -> None:
        self.context_kind = kind

    async def __call__(
        self,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        *_args: Any,
    ) -> tuple[ContextFragment, ...]:
        return ()


class UntypedProvider:
    async def __call__(
        self,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        *_args: Any,
    ) -> tuple[ContextFragment, ...]:
        return ()


def _agent(policy: AgentContextPolicy) -> AgentDefinition:
    return AgentDefinition(
        agent_type="worker",
        description="worker",
        system_prompt="system",
        context_policy=policy,
    )


def test_context_provider_kind_preserves_unknown_as_custom() -> None:
    project = TypedProvider("project_instructions")
    goal = TypedProvider("goal")
    unknown = UntypedProvider()
    invalid = TypedProvider("custom")
    invalid.context_kind = "not-a-kind"  # type: ignore[assignment]

    assert context_provider_kind(project) == "project_instructions"
    assert context_provider_kind(goal) == "goal"
    assert context_provider_kind(unknown) == "custom"
    assert context_provider_kind(invalid) == "custom"


def test_filter_context_providers_by_kind_retains_custom_providers() -> None:
    project = TypedProvider("project_instructions")
    git = TypedProvider("git")
    custom = UntypedProvider()

    assert filter_context_providers_by_kind(
        (project, git, custom),
        omitted_kinds=("project_instructions", "git"),
    ) == (custom,)


def test_deps_for_agent_context_policy_filters_both_provider_stacks() -> None:
    project = TypedProvider("project_instructions")
    git = TypedProvider("git")
    environment = TypedProvider("environment")
    post_project = TypedProvider("project_instructions")
    post_custom = UntypedProvider()
    deps = QueryDeps(
        task_store=AppStateStore(),
        context_providers=(project, git, environment),
        post_tool_context_providers=(post_project, post_custom),
    )

    filtered = deps_for_agent_context_policy(
        deps,
        _agent(AgentContextPolicy.minimal()),
    )

    assert filtered is not deps
    assert filtered.task_store is deps.task_store
    assert filtered.context_providers == (environment,)
    assert filtered.post_tool_context_providers == (post_custom,)


def test_deps_for_agent_context_policy_inherit_returns_original_deps() -> None:
    deps = QueryDeps(task_store=AppStateStore())

    assert deps_for_agent_context_policy(deps, None) is deps
    assert deps_for_agent_context_policy(
        deps,
        _agent(AgentContextPolicy.inherit()),
    ) is deps


def test_builtin_explorer_uses_minimal_context_policy() -> None:
    assert EXPLORER_AGENT.context_policy.omit_context_kinds == (
        "project_instructions",
        "git",
    )
