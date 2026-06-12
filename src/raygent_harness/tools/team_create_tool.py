"""Concrete model-callable TeamCreate wrapper.


Raygent v1 implements the headless metadata path only: create a project-local
team config file and remember the current session's team context. It does not
start tmux/iTerm panes, mailboxes, shared task-list tools, or teammate process
backends.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from raygent_harness.coordinator.team import (
    TeamAlreadyExistsError,
    TeamContext,
    TeamStateStore,
)
from raygent_harness.core.permissions import PermissionAllowDecision, PermissionResult
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


TEAM_CREATE_TOOL_NAME = "TeamCreate"
TEAM_CREATE_MAX_RESULT_SIZE_CHARS = 100_000


class TeamCreateInput(BaseModel):
    team_name: str = Field(description="Name for the new team to create.")
    description: str | None = Field(
        default=None,
        description="Team description or purpose.",
    )
    agent_type: str | None = Field(
        default=None,
        description="Type or role to record for the team lead.",
    )


def build_team_create_tool(
    *,
    team_store: TeamStateStore,
    current_model: str,
) -> Tool:
    """Build a concrete TeamCreate tool over the supplied team store."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.team_name.strip():
            return ValidationError(message="team_name is required for TeamCreate")
        if ctx.agent_id is not None:
            return ValidationError(
                message="TeamCreate is only available to the main coordinator."
            )
        if team_store.current_team is not None:
            return ValidationError(
                message=(
                    f'Already leading team "{team_store.current_team.team_name}". '
                    "Use the existing team context or end it before creating another."
                )
            )
        return ValidationOk()

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        try:
            team = team_store.create_team(
                team_name=parsed.team_name,
                description=parsed.description,
                agent_type=parsed.agent_type,
                model=current_model,
                cwd=ctx.cwd,
            )
        except TeamAlreadyExistsError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return

        yield ToolResult(content=_team_created_content(team))

    return build_tool(
        ToolSpec(
            name=TEAM_CREATE_TOOL_NAME,
            description="Create headless team metadata for coordinating agents.",
            search_hint="create a multi-agent coordination team",
            input_model=TeamCreateInput,
            call=call,
            prompt=TEAM_CREATE_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=False,
            is_open_world=False,
            should_defer=True,
            always_load=False,
            max_result_size_chars=TEAM_CREATE_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Creating team {_coerce_input(input_).team_name}"
            ),
        )
    )


def create_team_create_catalog_provider(
    *,
    team_store: TeamStateStore | None = None,
    teams_dir: str | Path | None = None,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends TeamCreate when enabled."""

    store = team_store or TeamStateStore(
        base_dir=Path(teams_dir) if teams_dir is not None else Path(".raygent/teams")
    )

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(
            tool for tool in tools if tool.name != TEAM_CREATE_TOOL_NAME
        )
        if not enabled or ctx.agent_id is not None:
            return without_existing
        return (
            *without_existing,
            build_team_create_tool(team_store=store, current_model=config.model),
        )

    return provider


TEAM_CREATE_PROMPT = """# TeamCreate

Create headless team metadata for a coordinator-led multi-agent effort.

Use TeamCreate when the user explicitly asks for a team/swarm/group of agents,
or when durable coordination metadata would help a complex task.

Raygent v1 creates:
- a project-local team config at `.raygent/teams/{team-name}/config.json`
- an in-session current-team context with a deterministic team lead id

Team workflow:
1. Use TeamCreate to establish the active coordinator team.
2. Use Agent with `name` to launch addressable in-process teammates.
3. Use SendMessage to route follow-up work to named teammates or broadcast with
   `to="*"`.

Raygent v1 does not create terminal panes, external worker processes, or
file-backed teammate mailboxes. Teammate routing is an in-process headless
kernel surface.
"""


def _coerce_input(input_: BaseModel) -> TeamCreateInput:
    if isinstance(input_, TeamCreateInput):
        return input_
    return TeamCreateInput.model_validate(input_.model_dump())


def _team_created_content(team: TeamContext) -> list[dict[str, Any]]:
    text = (
        f'Created team "{team.team_name}" with lead {team.lead_agent_id}. '
        "Team metadata was written to the project-local config file."
    )
    return [
        {"type": "text", "text": text},
        {
            "type": "team_created",
            "team_name": team.team_name,
            "team_file_path": team.team_file_path,
            "lead_agent_id": team.lead_agent_id,
        },
    ]


__all__ = [
    "TEAM_CREATE_MAX_RESULT_SIZE_CHARS",
    "TEAM_CREATE_PROMPT",
    "TEAM_CREATE_TOOL_NAME",
    "TeamCreateInput",
    "build_team_create_tool",
    "create_team_create_catalog_provider",
]
