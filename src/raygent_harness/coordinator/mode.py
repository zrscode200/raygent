"""Headless coordinator-mode prompt helpers.


The reference coordinator is primarily prompt/context shaping: the same agent
loop runs, but the model is instructed to fan out work through Agent, wait for
task notifications, continue workers, and stop bad workers. Raygent keeps v1
headless and exposes injectable prompt providers rather than a separate loop or
CLI mode switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raygent_harness.agents.tool_pool import ASYNC_AGENT_ALLOWED_TOOLS
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.tools.agent_tool import AGENT_TOOL_NAME
from raygent_harness.tools.task_stop_tool import (
    TASK_STOP_TOOL_NAME,
    create_task_stop_catalog_provider,
)

if TYPE_CHECKING:
    from raygent_harness.coordinator.team import TeamStateStore
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import (
        ContextProvider,
        QueryDeps,
        SystemPromptProvider,
        ToolCatalogProvider,
    )
    from raygent_harness.core.tool import ToolUseContext


@dataclass(frozen=True)
class CoordinatorModeConfig:
    """Static coordinator-mode settings for one harness session."""

    enabled: bool = False
    scratchpad_dir: str | None = None
    mcp_server_names: tuple[str, ...] = ()
    worker_tool_names: tuple[str, ...] | None = None
    include_team_tools: bool = False


def create_coordinator_system_prompt_provider(
    settings: CoordinatorModeConfig,
) -> SystemPromptProvider:
    """Return a QueryDeps-compatible provider for coordinator prompt injection.

    Coordinator prompt text is main-thread only. Subagents already receive their
    own agent-specific system prompt through AgentTool and should not be told to
    act as coordinators.
    """

    async def provider(
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str | None:
        if not settings.enabled or ctx.agent_id is not None:
            return None
        return build_coordinator_system_prompt(settings)

    return provider


def create_coordinator_user_context_provider(
    settings: CoordinatorModeConfig,
) -> ContextProvider:
    """Return non-persistent user-context fragments for static worker facts.

    Reference keeps worker-tools/MCP/scratchpad context in its user-context
    lane (`getCoordinatorUserContext`). Raygent's coordinator lane now supports
    the same split: role policy stays in the system prompt, static worker facts
    are visible for the current model call without becoming transcript history.
    """

    async def provider(
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        if not settings.enabled or ctx.agent_id is not None:
            return ()
        context = build_coordinator_user_context(settings)
        return tuple(
            ContextFragment(
                id=key,
                content=value,
                channel="user_context",
                source=key,
                priority=20,
                agent_scope="main",
            )
            for key, value in context.items()
        )

    return provider


def create_coordinator_tool_catalog_provider(
    *,
    parent_deps: QueryDeps,
    team_store: TeamStateStore | None = None,
    teams_dir: str | Path | None = None,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Append coordinator-owned tools in a reference-shaped order."""

    from raygent_harness.coordinator.team import TeamStateStore
    from raygent_harness.tools.send_message_tool import (
        create_send_message_catalog_provider,
    )
    from raygent_harness.tools.team_create_tool import (
        create_team_create_catalog_provider,
    )

    store = team_store or TeamStateStore(
        base_dir=Path(teams_dir) if teams_dir is not None else Path(".raygent/teams")
    )

    with_stop = create_task_stop_catalog_provider(
        parent_deps=parent_deps,
        enabled=enabled,
        upstream=upstream,
    )
    with_team_create = create_team_create_catalog_provider(
        team_store=store,
        enabled=enabled,
        upstream=with_stop,
    )
    return create_send_message_catalog_provider(
        team_store=store,
        task_store=parent_deps.task_store,
        enabled=enabled,
        upstream=with_team_create,
    )


def build_coordinator_system_prompt(settings: CoordinatorModeConfig) -> str:
    """Build Raygent-owned coordinator instructions."""

    team_note = (
        "\n- TeamCreate creates team metadata. Named Agent launches create "
        "addressable headless teammates; use SendMessage to route follow-up work. "
        "Raygent does not create panes or external teammate processes."
        if settings.include_team_tools
        else ""
    )

    return f"""## Coordinator Mode

You are the Raygent coordinator for this session. Your job is to help the user
complete the task by deciding what to do locally, what to delegate to workers,
and how to synthesize worker results.

Core rules:
- Use `{AGENT_TOOL_NAME}` to launch independent background workers for bounded
  research, implementation, or verification tasks.
- Do not launch a worker to merely poll another worker. Workers notify you when
  their task reaches a terminal state.
- After launching workers, briefly tell the user what was launched and stop the
  turn. Do not invent worker findings.
- Worker notifications arrive as user-role background signals. They may contain
  `<task_notification>` XML with `<task_id>`, `<status>`, `<summary>`,
  `<result>`, or `<partial_result>` tags.
- Treat task notifications as internal coordination signals, not as messages
  from the human user. Synthesize their relevant content for the user.
- Use `{TASK_STOP_TOOL_NAME}` to stop a worker that is no longer useful or was
  launched with the wrong objective.

Worker context:
- Background worker tool availability, MCP server names, and scratchpad details
  may be supplied as non-persistent user context for this model call.{team_note}

Delegation discipline:
- Write self-contained worker prompts. Include file paths, goals, constraints,
  and verification expectations.
- Parallelize independent read-only research. Avoid concurrent writes to the
  same files.
- When worker results conflict, inspect the code or run verification before
  deciding.
"""


def build_coordinator_user_context(settings: CoordinatorModeConfig) -> dict[str, str]:
    """Return reference-shaped user-context fragments for external renderers.

    QueryEngine can consume the same content through
    `create_coordinator_user_context_provider(...)`; external adapters may also
    render this mapping directly when they expose a native user-context surface.
    """

    if not settings.enabled:
        return {}
    tools = ", ".join(worker_tool_names(settings))
    content = f"Workers spawned via {AGENT_TOOL_NAME} can use: {tools}."
    if settings.mcp_server_names:
        content += (
            "\nWorkers also have MCP tools from connected servers: "
            + ", ".join(settings.mcp_server_names)
            + "."
        )
    if settings.scratchpad_dir:
        content += (
            f"\nScratchpad directory: {settings.scratchpad_dir}. Workers may use "
            "it for durable cross-worker knowledge."
        )
    return {"workerToolsContext": content}


def worker_tool_names(settings: CoordinatorModeConfig) -> tuple[str, ...]:
    """Return coordinator-visible worker tool names in stable display order."""

    if settings.worker_tool_names is not None:
        return tuple(sorted(settings.worker_tool_names))
    return tuple(sorted(ASYNC_AGENT_ALLOWED_TOOLS))


__all__ = [
    "CoordinatorModeConfig",
    "build_coordinator_system_prompt",
    "build_coordinator_user_context",
    "create_coordinator_system_prompt_provider",
    "create_coordinator_tool_catalog_provider",
    "create_coordinator_user_context_provider",
    "worker_tool_names",
]
