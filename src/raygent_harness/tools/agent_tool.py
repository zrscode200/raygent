"""Concrete model-callable Agent wrapper.


Raygent implements the headless AgentTool kernel: background local agents,
foreground/synchronous child-query agents, gated fork-subagent execution,
optional worktree-isolated local agents, in-process teammate routing, and a
remote backend protocol seam. UI progress panes and concrete product backends
remain explicit adapter extensions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from raygent_harness.agents.context_policy import deps_for_agent_context_policy
from raygent_harness.agents.loader import (
    DEFAULT_AGENT_TYPE,
    filter_denied_agents,
    find_agent_definition,
    get_builtin_agent_definitions,
)
from raygent_harness.agents.models import AgentDefinition
from raygent_harness.agents.tool_pool import (
    AGENT_TOOL_NAME,
    LEGACY_AGENT_TOOL_NAME,
    resolve_agent_tools,
)
from raygent_harness.coordinator.team import sanitize_team_name
from raygent_harness.core.child_query import ChildQueryRequest, run_child_query
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.permission_engine import get_rules
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionBehavior,
    PermissionDenyDecision,
    PermissionResult,
    RulePermissionDecisionReason,
    ToolPermissionContext,
    empty_tool_permission_context,
)
from raygent_harness.core.task import generate_task_id
from raygent_harness.core.tasks.in_process_teammate import (
    InProcessTeammateState,
    spawn_in_process_teammate,
)
from raygent_harness.core.tasks.local_agent import spawn_local_agent
from raygent_harness.core.tasks.remote_agent import RemoteAgentState, spawn_remote_agent
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolPromptContext,
    ToolResult,
    ToolRuntimeContext,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.services.handoff import (
    HandoffClassificationRequest,
    classify_handoff_warning,
)
from raygent_harness.services.mcp import mcp_server_name_for_tool
from raygent_harness.services.worktree import WorktreeCleanupResult

if TYPE_CHECKING:
    from raygent_harness.coordinator.team import TeamStateStore
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.services.worktree import WorktreeInfo
    from raygent_harness.skills.models import SkillDefinition


AGENT_TOOL_MAX_RESULT_SIZE_CHARS = 100_000
FORK_SUBAGENT_TYPE = "fork"
FORK_QUERY_SOURCE = f"agent:builtin:{FORK_SUBAGENT_TYPE}"
FORK_BOILERPLATE_TAG = "forked_worker"
FORK_DIRECTIVE_PREFIX = "Directive: "
FORK_PLACEHOLDER_RESULT = "Fork started — processing in background"
_PARENT_PERMISSION_PRECEDENCE_MODES: frozenset[str] = frozenset(
    {"bypassPermissions", "acceptEdits", "auto"}
)


class AgentToolInput(BaseModel):
    prompt: str = Field(description="The task for the spawned agent to perform.")
    subagent_type: str | None = Field(
        default=None,
        description=(
            "Agent type to launch. If omitted, Raygent uses the default worker agent."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Short description of the delegated work.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for this spawned agent.",
    )
    run_in_background: bool | None = Field(
        default=None,
        description=(
            "When false, run the agent synchronously and return its result in "
            "this tool call."
        ),
    )
    name: str | None = Field(
        default=None,
        description="Optional teammate name for SendMessage routing.",
    )
    team_name: str | None = Field(
        default=None,
        description="Optional team name; defaults to the active TeamCreate context.",
    )
    isolation: Literal["worktree", "remote"] | None = Field(
        default=None,
        description="Optional isolated-agent backend.",
    )


def build_agent_tool(
    *,
    parent_config: QueryConfig,
    parent_deps: QueryDeps,
    agent_definitions: Sequence[AgentDefinition] | None = None,
    allowed_agent_types: Sequence[str] | None = None,
    default_agent_type: str = DEFAULT_AGENT_TYPE,
    team_store: TeamStateStore | None = None,
) -> Tool:
    """Build a concrete Agent tool over the supplied agent definitions."""

    agents = tuple(agent_definitions or get_builtin_agent_definitions())
    allowed = tuple(allowed_agent_types or ())

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        error = _validate_static_input(parsed)
        if error is not None:
            return ValidationError(message=error)
        route_error = _validate_named_route(
            parsed,
            ctx,
            team_store=team_store,
            store=parent_deps.task_store,
        )
        if route_error is not None:
            return ValidationError(message=route_error)

        agent_type = _requested_agent_type(parsed, parent_config, default_agent_type)
        if _should_use_fork_path(parsed, parent_config):
            effective_isolation = _effective_isolation(parsed, None)
            if effective_isolation == "remote":
                return ValidationError(
                    message="Agent remote isolation requires an explicit subagent_type."
                )
            isolation_error = _validate_isolation(effective_isolation, parent_deps)
            if isolation_error is not None:
                return ValidationError(message=isolation_error)
            return ValidationOk()
        candidates = _agents_matching_allowed_types(
            agents,
            allowed_agent_types=allowed,
        )
        selected = find_agent_definition(agent_type, candidates)
        if selected is None:
            return ValidationError(
                message=_unknown_agent_message(
                    agent_type,
                    _available_agents(
                        agents,
                        ctx,
                        allowed_agent_types=allowed,
                        filter_denied=True,
                    ),
                )
            )
        if (
            parsed.name is not None or parsed.team_name is not None
        ) and _effective_isolation(parsed, selected) is not None:
            return ValidationError(
                message="Named/team agent routing does not support isolation in this wave."
            )
        isolation_error = _validate_isolation(
            _effective_isolation(parsed, selected),
            parent_deps,
        )
        if isolation_error is not None:
            return ValidationError(message=isolation_error)
        if (
            _effective_isolation(parsed, selected) == "remote"
            and parsed.run_in_background is False
        ):
            return ValidationError(
                message="Agent remote isolation requires background execution."
            )
        missing_mcp = _missing_required_mcp_servers(selected, ctx.tools)
        if missing_mcp:
            return ValidationError(
                message=(
                    f"Agent '{selected.agent_type}' requires MCP servers matching: "
                    f"{', '.join(missing_mcp)}"
                )
            )
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        _ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        agent_type = _requested_agent_type(parsed, parent_config, default_agent_type)
        is_fork_path = _should_use_fork_path(parsed, parent_config)
        updated_input = _normalized_input(parsed, agent_type, is_fork_path=is_fork_path)

        deny_rule = _matching_agent_rule(permission_context, "deny", agent_type)
        if deny_rule is not None:
            return PermissionDenyDecision(
                message=(
                    f"Agent type '{agent_type}' has been denied by permission "
                    f"rule '{AGENT_TOOL_NAME}({agent_type})'."
                ),
                decision_reason=RulePermissionDecisionReason(rule=deny_rule),
            )

        ask_rule = _matching_agent_rule(permission_context, "ask", agent_type)
        if ask_rule is not None:
            return PermissionAskDecision(
                message=f"Launch agent: {agent_type}",
                updated_input=updated_input,
                decision_reason=RulePermissionDecisionReason(rule=ask_rule),
            )

        return PermissionAllowDecision(updated_input=updated_input)

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        agent_type = _requested_agent_type(parsed, parent_config, default_agent_type)
        is_fork_path = _should_use_fork_path(parsed, parent_config)
        selected = (
            _fork_agent_definition()
            if is_fork_path
            else find_agent_definition(
                agent_type,
                _available_agents(
                    agents,
                    ctx,
                    allowed_agent_types=allowed,
                    filter_denied=True,
                ),
            )
        )
        if selected is None:
            yield ToolResult(
                content=_unknown_agent_message(agent_type, agents),
                is_error=True,
            )
            return

        if is_fork_path and _is_recursive_fork(ctx):
            yield ToolResult(
                content=(
                    "Fork is not available inside a forked worker. Complete "
                    "the task directly using your tools."
                ),
                is_error=True,
            )
            return

        effective_isolation = _effective_isolation(parsed, selected)
        if (
            parsed.name is not None or parsed.team_name is not None
        ) and effective_isolation is not None:
            yield ToolResult(
                content="Named/team agent routing does not support isolation in this wave.",
                is_error=True,
            )
            return
        isolation_error = _validate_isolation(effective_isolation, parent_deps)
        if isolation_error is not None:
            yield ToolResult(content=isolation_error, is_error=True)
            return

        should_run_background = (
            (not is_fork_path)
            and (parsed.run_in_background is not False or selected.background is True)
        ) or (
            is_fork_path and parsed.run_in_background is True
        )
        is_async_tool_pool = should_run_background and not is_fork_path
        resolved = resolve_agent_tools(
            selected,
            ctx.tools or parent_config.tools,
            is_async=is_async_tool_pool,
            is_main_thread=False,
        )
        child_permission_context = _child_permission_context(
            parent_deps,
            ctx,
            selected,
            default_mode="acceptEdits",
        )
        parent_effective_model = _current_effective_model(parent_config, ctx)
        child_model = _resolve_model(
            requested_model=parsed.model,
            agent_model=selected.model,
            parent_model=parent_effective_model,
        )
        child_parent_deps = deps_for_agent_context_policy(parent_deps, selected)
        child_prompt = (
            parsed.prompt if is_fork_path else _build_child_prompt(selected, parsed.prompt)
        )
        description = parsed.description or selected.description
        if effective_isolation == "remote":
            if not should_run_background:
                yield ToolResult(
                    content="Agent remote isolation requires background execution.",
                    is_error=True,
                )
                return
            task_id = await spawn_remote_agent(
                prompt=child_prompt,
                description=description,
                agent_type=selected.agent_type,
                parent_agent_id=ctx.agent_id,
                parent_deps=parent_deps,
                tool_use_id=ctx.tool_use_id,
                model=child_model,
                cwd=ctx.cwd,
                parent_observability_context=ctx.observability_context,
            )
            coordinator_work_item_id = _record_coordinator_agent_launch(
                deps=parent_deps,
                ctx=ctx,
                agent_id=task_id,
                agent_type=selected.agent_type,
                description=description,
                prompt=child_prompt,
                task_id=task_id,
                mode="remote",
                status="running",
            )
            yield ToolResult(
                content=_remote_launch_content(
                    task_id=task_id,
                    agent_type=selected.agent_type,
                    description=description,
                    prompt=child_prompt,
                    store=parent_deps.task_store,
                    invalid_tools=resolved.invalid_tools,
                    coordinator_work_item_id=coordinator_work_item_id,
                )
            )
            return
        child_agent_id = generate_task_id("local_agent")
        worktree_info = await _create_worktree_if_needed(
            child_agent_id=child_agent_id,
            isolation=effective_isolation,
            ctx=ctx,
            deps=parent_deps,
        )

        if should_run_background and parsed.name is not None:
            if not parsed.name.strip():
                yield ToolResult(
                    content="Named/team agent routing requires a non-empty name.",
                    is_error=True,
                )
                return
            if team_store is None or team_store.current_team is None:
                if parsed.team_name is not None:
                    yield ToolResult(
                        content="Named/team agent routing requires an active team context.",
                        is_error=True,
                    )
                    return
                task_id = await spawn_local_agent(
                    prompt=child_prompt,
                    parent_agent_id=ctx.agent_id,
                    parent_config=parent_config,
                    parent_deps=child_parent_deps,
                    parent_ctx=ctx,
                    description=description,
                    tool_use_id=ctx.tool_use_id,
                    child_system_prompt=selected.system_prompt,
                    child_model=child_model,
                    child_tools=resolved.resolved_tools,
                    child_permission_context=child_permission_context,
                    child_cwd=worktree_info.path if worktree_info is not None else None,
                    child_agent_type=selected.agent_type,
                    name=parsed.name,
                    task_id=child_agent_id,
                    worktree_info=worktree_info,
                    worktree_manager=parent_deps.worktree_manager,
                )
                coordinator_work_item_id = _record_coordinator_agent_launch(
                    deps=parent_deps,
                    ctx=ctx,
                    agent_id=task_id,
                    agent_type=selected.agent_type,
                    description=description,
                    prompt=child_prompt,
                    task_id=task_id,
                    agent_name=parsed.name.strip(),
                    mode="background",
                    status="running",
                )
                yield ToolResult(
                    content=_async_launch_content(
                        agent_id=task_id,
                        agent_type=selected.agent_type,
                        description=description,
                        prompt=child_prompt,
                        invalid_tools=resolved.invalid_tools,
                        worktree_info=worktree_info,
                        coordinator_work_item_id=coordinator_work_item_id,
                        agent_name=parsed.name.strip(),
                    )
                )
                return
            team_name = _effective_team_name(parsed, team_store)
            if team_name is None:
                yield ToolResult(
                    content="Named agent routing requires an active team context.",
                    is_error=True,
                )
                return
            normalized_team_name = sanitize_team_name(team_name)
            if normalized_team_name != team_store.current_team.team_name:
                yield ToolResult(
                    content=f'Team "{normalized_team_name}" is not the current team.',
                    is_error=True,
                )
                return
            teammate_name = team_store.unique_member_name(parsed.name)
            task_id = await spawn_in_process_teammate(
                name=teammate_name,
                team_name=team_name,
                prompt=child_prompt,
                parent_agent_id=ctx.agent_id,
                parent_config=parent_config,
                parent_deps=child_parent_deps,
                parent_ctx=ctx,
                description=description,
                agent_type=selected.agent_type,
                team_store=team_store,
                tool_use_id=ctx.tool_use_id,
                child_system_prompt=selected.system_prompt,
                child_model=child_model,
                child_tools=resolved.resolved_tools,
                child_permission_context=child_permission_context,
            )
            team_store.add_member(
                agent_id=_teammate_agent_id(task_id, parent_deps.task_store),
                name=teammate_name,
                agent_type=selected.agent_type,
                model=child_model,
                cwd=ctx.cwd,
                team_name=team_name,
            )
            teammate_agent_id = _teammate_agent_id(task_id, parent_deps.task_store)
            coordinator_work_item_id = _record_coordinator_agent_launch(
                deps=parent_deps,
                ctx=ctx,
                agent_id=teammate_agent_id,
                agent_type=selected.agent_type,
                description=description,
                prompt=child_prompt,
                task_id=task_id,
                agent_name=teammate_name,
                team_name=team_name,
                mode="teammate",
                status="running",
            )
            yield ToolResult(
                content=_teammate_launch_content(
                    task_id=task_id,
                    agent_id=teammate_agent_id,
                    name=teammate_name,
                    team_name=team_name,
                    agent_type=selected.agent_type,
                    description=description,
                    prompt=child_prompt,
                    invalid_tools=resolved.invalid_tools,
                    coordinator_work_item_id=coordinator_work_item_id,
                )
            )
            return

        if should_run_background:
            background_prompt: str | MessageParam = child_prompt
            background_initial_messages: tuple[MessageParam, ...] = ()
            background_system_prompt = selected.system_prompt
            background_tools = resolved.resolved_tools
            background_query_source: str | None = None
            if is_fork_path:
                fork_messages = _fork_prompt_messages(ctx, parsed.prompt)
                if worktree_info is not None:
                    fork_messages = (
                        *fork_messages,
                        cast(
                            "MessageParam",
                            _user_message(_worktree_notice(ctx.cwd, worktree_info.path)),
                        ),
                    )
                background_initial_messages = fork_messages[:-1]
                background_prompt = fork_messages[-1]
                background_system_prompt = ctx.rendered_system_prompt
                background_tools = tuple(ctx.tools or parent_config.tools)
                background_query_source = FORK_QUERY_SOURCE
            task_id = await spawn_local_agent(
                prompt=background_prompt,
                parent_agent_id=ctx.agent_id,
                parent_config=parent_config,
                parent_deps=child_parent_deps,
                parent_ctx=ctx,
                description=description,
                tool_use_id=ctx.tool_use_id,
                child_system_prompt=background_system_prompt,
                child_model=child_model,
                child_tools=background_tools,
                child_permission_context=child_permission_context,
                child_cwd=worktree_info.path if worktree_info is not None else None,
                child_agent_type=selected.agent_type,
                task_id=child_agent_id,
                initial_messages=background_initial_messages,
                display_prompt=child_prompt,
                child_query_source=background_query_source,
                worktree_info=worktree_info,
                worktree_manager=parent_deps.worktree_manager,
            )
            coordinator_work_item_id = _record_coordinator_agent_launch(
                deps=parent_deps,
                ctx=ctx,
                agent_id=task_id,
                agent_type=selected.agent_type,
                description=description,
                prompt=child_prompt,
                task_id=task_id,
                mode="fork_background" if is_fork_path else "background",
                status="running",
            )
            yield ToolResult(
                content=_async_launch_content(
                    agent_id=task_id,
                    agent_type=selected.agent_type,
                    description=description,
                    prompt=child_prompt,
                    invalid_tools=resolved.invalid_tools,
                    worktree_info=worktree_info,
                    coordinator_work_item_id=coordinator_work_item_id,
                    is_fork=is_fork_path,
                )
            )
            return

        prompt_messages = (
            _fork_prompt_messages(ctx, parsed.prompt)
            if is_fork_path
            else (cast("MessageParam", _user_message(child_prompt)),)
        )
        if is_fork_path and worktree_info is not None:
            prompt_messages = (
                *prompt_messages,
                cast(
                    "MessageParam",
                    _user_message(_worktree_notice(ctx.cwd, worktree_info.path)),
                ),
            )
        child_system_prompt = (
            ctx.rendered_system_prompt
            if is_fork_path
            else selected.system_prompt
        )
        child_tools = (
            tuple(ctx.tools or parent_config.tools)
            if is_fork_path
            else resolved.resolved_tools
        )
        worktree_result: WorktreeCleanupResult | None = None
        try:
            result = await run_child_query(
                ChildQueryRequest(
                    prompt_messages=prompt_messages,
                    parent_config=parent_config,
                    parent_deps=child_parent_deps,
                    parent_ctx=ctx,
                    agent_id=child_agent_id,
                    agent_type=selected.agent_type,
                    system_prompt=child_system_prompt,
                    model=child_model,
                    tools=child_tools,
                    permission_context=child_permission_context,
                    cwd=worktree_info.path if worktree_info is not None else None,
                    query_source=FORK_QUERY_SOURCE if is_fork_path else None,
                    transcript_label=(
                        f"agent:{FORK_SUBAGENT_TYPE}"
                        if is_fork_path
                        else f"agent:{selected.agent_type}"
                    ),
                )
            )
        finally:
            worktree_result = await _cleanup_worktree_if_needed(
                worktree_info=worktree_info,
                deps=parent_deps,
            )
        if result.subtype == "error_aborted" and ctx.abort_event.is_set():
            raise asyncio.CancelledError()
        result_text = result.final_message or "\n".join(result.errors)
        if not result_text:
            result_text = "Agent execution completed"
        handoff_warning = await _handoff_warning_for_sync_result(
            deps=parent_deps,
            agent_id=child_agent_id,
            agent_type=selected.agent_type,
            description=description,
            result_text=result_text,
            is_error=result.is_error,
            messages=result.messages,
            tool_names=tuple(tool.name for tool in child_tools),
            permission_mode=child_permission_context.mode,
        )
        if handoff_warning:
            result_text = f"{handoff_warning}\n\n{result_text}".strip()
        coordinator_work_item_id = _record_coordinator_agent_launch(
            deps=parent_deps,
            ctx=ctx,
            agent_id=child_agent_id,
            agent_type=selected.agent_type,
            description=description,
            prompt=child_prompt,
            mode="fork" if is_fork_path else "foreground",
            status="failed" if result.is_error else "completed",
            result_summary=(
                f"Agent result recorded; result_chars={len(result_text)}."
                if not result.is_error
                else None
            ),
            error_summary=(
                f"Agent result recorded; result_chars={len(result_text)}."
                if result.is_error
                else None
            ),
        )
        yield ToolResult(
            content=_sync_result_content(
                agent_id=child_agent_id,
                agent_type=selected.agent_type,
                description=description,
                prompt=child_prompt,
                result=result_text,
                status="error" if result.is_error else "completed",
                is_fork=is_fork_path,
                worktree_result=worktree_result,
                coordinator_work_item_id=coordinator_work_item_id,
            ),
            is_error=result.is_error,
        )
        return

    async def prompt(ctx: ToolPromptContext | ToolUseContext | None = None) -> str:
        permission_context = (
            ctx.permission_context
            if isinstance(ctx, ToolUseContext)
            else (
                ctx.permission_context
                if isinstance(ctx, ToolPromptContext)
                else empty_tool_permission_context()
            )
        )
        tools = ctx.tools if ctx is not None else ()
        prompt_agents = _available_agents_from_parts(
            agents,
            tools=tuple(tools),
            permission_context=permission_context,
            allowed_agent_types=allowed,
            filter_denied=True,
        )
        return _agent_prompt(prompt_agents)

    return build_tool(
        ToolSpec(
            name=AGENT_TOOL_NAME,
            aliases=(LEGACY_AGENT_TOOL_NAME,),
            description="Launch a background subagent.",
            search_hint="delegate work to a subagent",
            input_model=AgentToolInput,
            call=call,
            prompt=prompt,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=True,
            should_defer=False,
            always_load=True,
            max_result_size_chars=AGENT_TOOL_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                _coerce_input(input_).description or "Running agent"
            ),
        )
    )


def create_agent_catalog_provider(
    *,
    parent_deps: QueryDeps,
    agent_definitions: Sequence[AgentDefinition] | None = None,
    allowed_agent_types: Sequence[str] | None = None,
    team_store: TeamStateStore | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends a fresh Agent tool.

    The tool closes over the per-turn `QueryConfig`, which lets the spawned
    child inherit sampling/fallback settings while replacing only
    agent-specific fields.
    """

    agents = tuple(agent_definitions or get_builtin_agent_definitions())

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing_agent = tuple(
            tool
            for tool in tools
            if tool.name not in {AGENT_TOOL_NAME, LEGACY_AGENT_TOOL_NAME}
        )
        if ctx.agent_id is not None:
            if ctx.query_source == FORK_QUERY_SOURCE:
                return tuple(tools)
            return without_existing_agent
        visible_agents = _available_agents(
            agents,
            replace(ctx, tools=without_existing_agent),
            allowed_agent_types=tuple(allowed_agent_types or ()),
            filter_denied=True,
        )
        if not visible_agents:
            return without_existing_agent
        return (
            *without_existing_agent,
            build_agent_tool(
                parent_config=config,
                parent_deps=parent_deps,
                agent_definitions=agents,
                allowed_agent_types=allowed_agent_types,
                team_store=team_store,
            ),
        )

    return provider


def _record_coordinator_agent_launch(
    *,
    deps: QueryDeps,
    ctx: ToolUseContext,
    agent_id: str,
    agent_type: str,
    description: str,
    prompt: str,
    task_id: str | None = None,
    agent_name: str | None = None,
    team_name: str | None = None,
    mode: str,
    status: str,
    result_summary: str | None = None,
    error_summary: str | None = None,
) -> str | None:
    runtime_deps = ctx.runtime.deps if ctx.runtime is not None else deps
    runtime = runtime_deps.coordinator_runtime
    if runtime is None:
        return None
    try:
        result = runtime.record_agent_launch(
            agent_id=agent_id,
            agent_type=agent_type,
            description=description,
            prompt_chars=len(prompt),
            task_id=task_id,
            agent_name=agent_name,
            team_name=team_name,
            mode=mode,
            status=status,
            result_summary=result_summary,
            error_summary=error_summary,
        )
    except Exception as exc:
        _emit_coordinator_runtime_failure(
            deps=runtime_deps,
            ctx=ctx,
            operation="record_agent_launch",
            exc=exc,
        )
        return None
    return _coordinator_work_item_id(result)


def _coordinator_work_item_id(result: object) -> str | None:
    work_item = getattr(result, "work_item", None)
    work_item_id = getattr(work_item, "id", None)
    return work_item_id if isinstance(work_item_id, str) else None


def _emit_coordinator_runtime_failure(
    *,
    deps: QueryDeps,
    ctx: ToolUseContext,
    operation: str,
    exc: Exception,
) -> None:
    event_context = (
        ctx.observability_context.with_source("coordinator")
        if ctx.observability_context is not None
        else KernelEventContext(
            session_id=ctx.session_id,
            agent_id=ctx.agent_id,
            source="coordinator",
        )
    )
    deps.observability.emit(
        "coordinator.runtime.integration_failed",
        context=event_context,
        data={
            "operation": operation,
            "error_type": type(exc).__name__,
        },
    )


def _coerce_input(input_: BaseModel) -> AgentToolInput:
    if isinstance(input_, AgentToolInput):
        return input_
    return AgentToolInput.model_validate(input_.model_dump())


def _validate_static_input(input_: AgentToolInput) -> str | None:
    if not input_.prompt.strip():
        return "Agent prompt cannot be empty."
    return None


def _validate_named_route(
    input_: AgentToolInput,
    ctx: ToolUseContext,
    *,
    team_store: TeamStateStore | None,
    store: Any,
) -> str | None:
    if input_.name is None and input_.team_name is None:
        return None
    if input_.name is None or not input_.name.strip():
        return "Named/team agent routing requires a non-empty name."
    if ctx.agent_id is not None:
        return "Named/team agent routing is only available to the main coordinator."
    if input_.run_in_background is False:
        return "Named/team agent routing requires a background agent."
    if input_.isolation is not None:
        return "Named/team agent routing does not support isolation in this wave."
    if team_store is None or team_store.current_team is None:
        if input_.team_name is not None:
            return "Named/team agent routing requires an active TeamCreate context."
        return None
    team_name = _effective_team_name(input_, team_store)
    if team_name is None:
        return "Named/team agent routing requires an active TeamCreate context."
    normalized_team = sanitize_team_name(team_name)
    if normalized_team != team_store.current_team.team_name:
        return f'Team "{normalized_team}" is not the current team.'
    return None


def _effective_team_name(
    input_: AgentToolInput,
    team_store: TeamStateStore | None,
) -> str | None:
    if input_.team_name is not None and input_.team_name.strip():
        return input_.team_name.strip()
    if team_store is None or team_store.current_team is None:
        return None
    return team_store.current_team.team_name


def _effective_isolation(
    input_: AgentToolInput,
    selected: AgentDefinition | None,
) -> Literal["worktree", "remote"] | None:
    return input_.isolation or (selected.isolation if selected is not None else None)


def _validate_isolation(
    isolation: Literal["worktree", "remote"] | None,
    deps: QueryDeps,
) -> str | None:
    if isolation == "remote" and deps.remote_agent_backend is None:
        return (
            "Agent remote isolation requires QueryDeps.remote_agent_backend to be "
            "configured."
        )
    if isolation == "worktree" and deps.worktree_manager is None:
        return (
            "Agent worktree isolation requires QueryDeps.worktree_manager to be "
            "configured."
        )
    return None


def _requested_agent_type(
    input_: AgentToolInput,
    config: QueryConfig,
    default_agent_type: str,
) -> str:
    if _should_use_fork_path(input_, config):
        return FORK_SUBAGENT_TYPE
    return (input_.subagent_type or default_agent_type).strip()


def _should_use_fork_path(input_: AgentToolInput, config: QueryConfig) -> bool:
    if input_.name is not None or input_.team_name is not None:
        return False
    return input_.subagent_type is None and config.experiments.get(
        "fork_subagent", False
    )


def _available_agents(
    agents: Sequence[AgentDefinition],
    ctx: ToolUseContext,
    *,
    allowed_agent_types: Sequence[str],
    filter_denied: bool,
) -> tuple[AgentDefinition, ...]:
    return _available_agents_from_parts(
        agents,
        tools=ctx.tools,
        permission_context=ctx.permission_context,
        allowed_agent_types=allowed_agent_types,
        filter_denied=filter_denied,
    )


def _available_agents_from_parts(
    agents: Sequence[AgentDefinition],
    *,
    tools: Sequence[Tool],
    permission_context: ToolPermissionContext,
    allowed_agent_types: Sequence[str],
    filter_denied: bool,
) -> tuple[AgentDefinition, ...]:
    result = tuple(
        agent
        for agent in _agents_matching_allowed_types(
            agents,
            allowed_agent_types=allowed_agent_types,
        )
        if not _missing_required_mcp_servers(agent, tools)
    )
    if filter_denied:
        result = filter_denied_agents(
            result,
            permission_context,
            tool_name=AGENT_TOOL_NAME,
        )
    return result


def _agents_matching_allowed_types(
    agents: Sequence[AgentDefinition],
    *,
    allowed_agent_types: Sequence[str],
) -> tuple[AgentDefinition, ...]:
    return tuple(
        agent
        for agent in agents
        if not allowed_agent_types or agent.agent_type in allowed_agent_types
    )


def _unknown_agent_message(
    agent_type: str,
    agents: Sequence[AgentDefinition],
) -> str:
    available = ", ".join(agent.agent_type for agent in agents) or "none"
    return f"Agent type '{agent_type}' not found. Available agents: {available}"


def _missing_required_mcp_servers(
    agent: AgentDefinition,
    tools: Sequence[Tool],
) -> tuple[str, ...]:
    if not agent.required_mcp_servers:
        return ()
    servers = _mcp_servers_with_tools(tools)
    return tuple(
        pattern
        for pattern in agent.required_mcp_servers
        if not any(pattern.lower() in server.lower() for server in servers)
    )


def _mcp_servers_with_tools(tools: Sequence[Tool]) -> tuple[str, ...]:
    servers: list[str] = []
    for tool in tools:
        server = mcp_server_name_for_tool(tool.name)
        if server is None:
            continue
        if server not in servers:
            servers.append(server)
    return tuple(servers)


def _matching_agent_rule(
    context: ToolPermissionContext,
    behavior: PermissionBehavior,
    agent_type: str,
) -> Any | None:
    for rule in get_rules(context, behavior):
        value = rule.rule_value
        if value.tool_name not in {AGENT_TOOL_NAME, LEGACY_AGENT_TOOL_NAME}:
            continue
        if value.rule_content is None or value.rule_content == agent_type:
            return rule
    return None


def _normalized_input(
    input_: AgentToolInput,
    agent_type: str,
    *,
    is_fork_path: bool,
) -> dict[str, object]:
    data = input_.model_dump()
    if not is_fork_path:
        data["subagent_type"] = agent_type
    return data


def _active_permission_context(
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> ToolPermissionContext:
    return deps.permission_context_for(ctx)


def _child_permission_context(
    deps: QueryDeps,
    ctx: ToolUseContext,
    selected: AgentDefinition,
    *,
    default_mode: str,
) -> ToolPermissionContext:
    permission_context = _active_permission_context(deps, ctx)
    mode = selected.permission_mode or default_mode
    if permission_context.mode in _PARENT_PERMISSION_PRECEDENCE_MODES:
        return permission_context
    return replace(permission_context, mode=cast(Any, mode))


def _current_effective_model(
    config: QueryConfig,
    ctx: ToolUseContext,
) -> str:
    runtime = _runtime_context(ctx)
    if ctx.model_override is not None:
        override = ctx.model_override.strip()
        if override == "inherit":
            runtime_model = runtime.effective_model if runtime is not None else None
            return runtime_model or config.model
        return ctx.model_override
    runtime_model = runtime.effective_model if runtime is not None else None
    return runtime_model or config.model


def _runtime_context(ctx: ToolUseContext) -> ToolRuntimeContext | None:
    return ctx.runtime


def _resolve_model(
    *,
    requested_model: str | None,
    agent_model: str | None,
    parent_model: str,
) -> str:
    model = requested_model or agent_model
    if model is None or model == "inherit":
        return parent_model
    return model


async def _create_worktree_if_needed(
    *,
    child_agent_id: str,
    isolation: Literal["worktree", "remote"] | None,
    ctx: ToolUseContext,
    deps: QueryDeps,
) -> WorktreeInfo | None:
    if isolation is None:
        return None
    if isolation == "remote":
        raise RuntimeError("Agent remote isolation is not implemented.")
    manager = deps.worktree_manager
    if manager is None:
        raise RuntimeError("Agent worktree isolation requires a worktree manager.")
    slug = f"agent-{child_agent_id[:8]}"
    info = await manager.create_agent_worktree(slug, cwd=ctx.cwd)
    if info.owner_task_id is None or info.slug is None:
        return replace(
            info,
            slug=info.slug or slug,
            owner_task_id=info.owner_task_id or child_agent_id,
        )
    return info


async def _cleanup_worktree_if_needed(
    *,
    worktree_info: WorktreeInfo | None,
    deps: QueryDeps,
) -> WorktreeCleanupResult | None:
    if worktree_info is None:
        return None
    manager = deps.worktree_manager
    if manager is None:
        return WorktreeCleanupResult(
            kept=True,
            reason="cleanup_failed",
            path=worktree_info.path,
            branch=worktree_info.branch,
        )
    try:
        return await manager.cleanup(worktree_info)
    except Exception:
        return WorktreeCleanupResult(
            kept=True,
            reason="cleanup_failed",
            path=worktree_info.path,
            branch=worktree_info.branch,
        )


def _worktree_notice(parent_cwd: str, worktree_cwd: str) -> str:
    return (
        "You've inherited the conversation context above from a parent agent "
        f"working in {parent_cwd}. You are operating in an isolated git "
        f"worktree at {worktree_cwd} — same repository, same relative file "
        "structure, separate working copy. Paths in the inherited context "
        "refer to the parent's working directory; translate them to your "
        "worktree root. Re-read files before editing if the parent may have "
        "modified them since they appear in the context. Your changes stay in "
        "this worktree and will not affect the parent's files."
    )


def _build_child_prompt(agent: AgentDefinition, prompt: str) -> str:
    if agent.initial_prompt is None or not agent.initial_prompt.strip():
        return prompt
    return f"{agent.initial_prompt.strip()}\n\n{prompt}"


def _fork_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        agent_type=FORK_SUBAGENT_TYPE,
        description="Implicit fork that inherits the current conversation context.",
        system_prompt="",
        tools=("*",),
        model="inherit",
        permission_mode="bubble",
        source="built-in",
    )


def _fork_prompt_messages(
    ctx: ToolUseContext,
    directive: str,
) -> tuple[MessageParam, ...]:
    inherited = tuple(ctx.messages)
    assistant_message = ctx.current_assistant_message
    if assistant_message is None:
        return (
            *inherited,
            cast("MessageParam", _user_message(_fork_child_message(directive))),
        )
    return (
        *inherited,
        _clone_message(assistant_message),
        cast(
            "MessageParam",
            {
                "role": "user",
                "content": (
                    *_fork_placeholder_tool_results(assistant_message),
                    {"type": "text", "text": _fork_child_message(directive)},
                ),
            },
        ),
    )


def _fork_placeholder_tool_results(
    assistant_message: MessageParam,
) -> tuple[dict[str, Any], ...]:
    content = assistant_message.get("content")
    if not isinstance(content, list):
        return ()
    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") != "tool_use":
            continue
        tool_use_id = block.get("id")
        if not isinstance(tool_use_id, str):
            continue
        blocks.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [{"type": "text", "text": FORK_PLACEHOLDER_RESULT}],
            }
        )
    return tuple(blocks)


def _fork_child_message(directive: str) -> str:
    return f"""<{FORK_BOILERPLATE_TAG}>
STOP. READ THIS FIRST.

You are a forked worker process. You are NOT the main agent.

RULES:
1. Do not spawn sub-agents; execute directly using your tools.
2. Stay strictly within your directive's scope.
3. Report structured facts once at the end.
4. Your response must begin with "Scope:".

Output format:
Scope: <assigned scope in one sentence>
Result: <answer or key findings>
Key files: <relevant file paths>
Files changed: <only if you modified files>
Issues: <only if there are issues to flag>
</{FORK_BOILERPLATE_TAG}>

{FORK_DIRECTIVE_PREFIX}{directive}"""


def _is_recursive_fork(ctx: ToolUseContext) -> bool:
    if ctx.query_source == FORK_QUERY_SOURCE:
        return True
    return any(_message_contains_fork_boilerplate(message) for message in ctx.messages)


def _message_contains_fork_boilerplate(message: MessageParam) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return f"<{FORK_BOILERPLATE_TAG}>" in content
    for block in content:
        text = block.get("text")
        if isinstance(text, str) and f"<{FORK_BOILERPLATE_TAG}>" in text:
            return True
        value = block.get("content")
        if isinstance(value, str) and f"<{FORK_BOILERPLATE_TAG}>" in value:
            return True
    return False


def _user_message(content: str) -> dict[str, object]:
    return {"role": "user", "content": content}


def _clone_message(message: MessageParam) -> MessageParam:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, list):
        return cast(
            "MessageParam",
            {
                "role": role,
                "content": [
                    dict(block) for block in content
                ],
            },
        )
    return cast("MessageParam", {"role": role, "content": content})


async def _handoff_warning_for_sync_result(
    *,
    deps: QueryDeps,
    agent_id: str,
    agent_type: str,
    description: str,
    result_text: str,
    is_error: bool,
    messages: Sequence[MessageParam],
    tool_names: tuple[str, ...],
    permission_mode: str | None,
) -> str | None:
    if is_error:
        return None
    return await classify_handoff_warning(
        deps.handoff_classifier,
        HandoffClassificationRequest(
            task_id=agent_id,
            task_type="local_agent",
            agent_type=agent_type,
            description=description,
            final_status="completed",
            final_message=result_text,
            messages=tuple(messages),
            tool_names=tool_names,
            permission_mode=permission_mode,
            total_tool_use_count=_count_tool_uses(messages),
        ),
        timeout_s=deps.handoff_classifier_timeout_s,
    )


def _count_tool_uses(messages: Sequence[MessageParam]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(1 for block in content if block.get("type") == "tool_use")
    return count


def _sync_result_content(
    *,
    agent_id: str,
    agent_type: str,
    description: str,
    prompt: str,
    result: str,
    status: Literal["completed", "error"],
    is_fork: bool,
    worktree_result: WorktreeCleanupResult | None = None,
    coordinator_work_item_id: str | None = None,
) -> list[dict[str, Any]]:
    label = "forked agent" if is_fork else "agent"
    text = f"{label.title()} {agent_id} ({agent_type}) {status}.\n\nResult:\n{result}"
    worktree_text = _worktree_result_text(worktree_result)
    if worktree_text:
        text = f"{text}\n\n{worktree_text}"
    metadata: dict[str, Any] = {
        "type": "agent_result",
        "status": status,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "description": description,
        "prompt": prompt,
        "result": result,
        "is_fork": is_fork,
    }
    metadata.update(_worktree_result_metadata(worktree_result))
    if coordinator_work_item_id is not None:
        metadata["coordinator_work_item_id"] = coordinator_work_item_id
    return [
        {"type": "text", "text": text},
        metadata,
    ]


def _async_launch_content(
    *,
    agent_id: str,
    agent_type: str,
    description: str,
    prompt: str,
    invalid_tools: Sequence[str],
    worktree_info: WorktreeInfo | None = None,
    coordinator_work_item_id: str | None = None,
    agent_name: str | None = None,
    is_fork: bool = False,
) -> list[dict[str, Any]]:
    label = "forked agent" if is_fork else "agent"
    text = (
        f"Launched {label} {agent_id} ({agent_type}) in the background. "
        "You will receive a task notification when it completes."
    )
    if worktree_info is not None:
        text = f"{text} Worktree: {worktree_info.path}"
    data: dict[str, Any] = {
        "type": "agent_launch",
        "status": "async_launched",
        "agent_id": agent_id,
        "agent_type": agent_type,
        "description": description,
        "prompt": prompt,
        "is_fork": is_fork,
    }
    data.update(_worktree_info_metadata(worktree_info))
    if coordinator_work_item_id is not None:
        data["coordinator_work_item_id"] = coordinator_work_item_id
    if agent_name is not None:
        data["agent_name"] = agent_name
    if invalid_tools:
        data["invalid_tools"] = list(invalid_tools)
    return [
        {"type": "text", "text": text},
        data,
    ]


def _teammate_launch_content(
    *,
    task_id: str,
    agent_id: str,
    name: str,
    team_name: str,
    agent_type: str,
    description: str,
    prompt: str,
    invalid_tools: Sequence[str],
    coordinator_work_item_id: str | None = None,
) -> list[dict[str, Any]]:
    text = (
        f"Launched teammate {name} ({agent_type}) for team {team_name}. "
        "Use SendMessage to route follow-up work while it is idle."
    )
    data: dict[str, Any] = {
        "type": "teammate_launch",
        "status": "async_launched",
        "task_id": task_id,
        "agent_id": agent_id,
        "name": name,
        "team_name": team_name,
        "agent_type": agent_type,
        "description": description,
        "prompt": prompt,
    }
    if coordinator_work_item_id is not None:
        data["coordinator_work_item_id"] = coordinator_work_item_id
    if invalid_tools:
        data["invalid_tools"] = list(invalid_tools)
    return [{"type": "text", "text": text}, data]


def _remote_launch_content(
    *,
    task_id: str,
    agent_type: str,
    description: str,
    prompt: str,
    store: Any,
    invalid_tools: Sequence[str],
    coordinator_work_item_id: str | None = None,
) -> list[dict[str, Any]]:
    task = store.tasks.get(task_id)
    session_url = task.session_url if isinstance(task, RemoteAgentState) else None
    text = (
        f"Launched remote agent {task_id} ({agent_type}) in the background. "
        "You will receive a task notification when it completes."
    )
    if session_url:
        text = f"{text} Session: {session_url}"
    data: dict[str, Any] = {
        "type": "agent_launch",
        "status": "remote_launched",
        "agent_id": task_id,
        "agent_type": agent_type,
        "description": description,
        "prompt": prompt,
        "is_remote": True,
    }
    if coordinator_work_item_id is not None:
        data["coordinator_work_item_id"] = coordinator_work_item_id
    if session_url:
        data["session_url"] = session_url
    if invalid_tools:
        data["invalid_tools"] = list(invalid_tools)
    return [{"type": "text", "text": text}, data]


def _teammate_agent_id(task_id: str, store: Any) -> str:
    task = store.tasks.get(task_id)
    if isinstance(task, InProcessTeammateState) and task.identity is not None:
        return task.identity.agent_id
    return task_id


def _worktree_info_metadata(info: WorktreeInfo | None) -> dict[str, Any]:
    if info is None:
        return {}
    data: dict[str, Any] = {
        "worktree_path": info.path,
    }
    if info.branch:
        data["worktree_branch"] = info.branch
    if info.slug:
        data["worktree_slug"] = info.slug
    if info.owner_task_id:
        data["worktree_owner_task_id"] = info.owner_task_id
    if info.created_at is not None:
        data["worktree_created_at"] = info.created_at
    if info.touched_at is not None:
        data["worktree_touched_at"] = info.touched_at
    data["worktree_cleanup_policy"] = info.cleanup_policy
    return data


def _worktree_result_metadata(result: WorktreeCleanupResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    data: dict[str, Any] = {
        "worktree_kept": result.kept,
        "worktree_cleanup_reason": result.reason,
    }
    if result.path:
        data["worktree_path"] = result.path
    if result.branch:
        data["worktree_branch"] = result.branch
    return data


def _worktree_result_text(result: WorktreeCleanupResult | None) -> str:
    if result is None:
        return ""
    if result.path:
        branch = f" on branch {result.branch}" if result.branch else ""
        return f"Worktree kept at {result.path}{branch}."
    return "Worktree removed after clean completion."


def _agent_prompt(agents: Sequence[AgentDefinition]) -> str:
    lines = [
        "Launch a background agent to handle complex, multi-step tasks autonomously.",
        "",
        "Available agent types:",
    ]
    if not agents:
        lines.append("- none")
    else:
        for agent in sorted(agents, key=lambda item: item.agent_type):
            lines.append(
                f"- {agent.agent_type}: {agent.description} "
                f"(Tools: {_tools_description(agent)})"
            )
    lines.extend(
        [
            "",
            "Usage notes:",
            "- Always include a concise task prompt with enough context.",
            "- By default Raygent launches background agents and reports completion "
            "through task notifications.",
            "- Set run_in_background=false only when the result must be returned "
            "in the same tool call.",
            "- The agent starts fresh unless your prompt includes the needed context.",
            "- In coordinator team mode, set name to launch an addressable teammate; "
            "follow-up work can be routed with SendMessage.",
        ]
    )
    return "\n".join(lines)


def _tools_description(agent: AgentDefinition) -> str:
    tools = tuple(agent.tools or ())
    disallowed = agent.disallowed_tools
    has_allowlist = len(tools) > 0 and tools != ("*",)
    if has_allowlist and disallowed:
        denied_names = set(disallowed)
        effective = tuple(tool for tool in tools if tool not in denied_names)
        return ", ".join(effective) if effective else "None"
    if has_allowlist:
        return ", ".join(tools)
    if disallowed:
        return f"All tools except {', '.join(disallowed)}"
    return "All tools"


__all__ = [
    "AGENT_TOOL_MAX_RESULT_SIZE_CHARS",
    "AGENT_TOOL_NAME",
    "AgentToolInput",
    "build_agent_tool",
    "create_agent_catalog_provider",
]
