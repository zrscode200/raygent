"""Concrete model-callable Skill wrapper.


Raygent preserves the reference inline SkillTool shape for normal skills and
routes `context: fork` skills through a synchronous child QueryEngine loop.
Forked skills return a same-turn tool result instead of a background task
notification.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from raygent_harness.agents.context_policy import deps_for_agent_context_policy
from raygent_harness.agents.loader import (
    DEFAULT_AGENT_TYPE,
    find_agent_definition,
    get_builtin_agent_definitions,
)
from raygent_harness.agents.models import AgentDefinition
from raygent_harness.agents.tool_pool import resolve_agent_tools
from raygent_harness.core.child_query import ChildQueryRequest, run_child_query
from raygent_harness.core.permission_engine import get_rules
from raygent_harness.core.permissions import (
    AddPermissionRules,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionCommandMetadata,
    PermissionDenyDecision,
    PermissionMetadata,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
    PermissionRuleValue,
    RulePermissionDecisionReason,
    ToolPermissionContext,
)
from raygent_harness.core.task import generate_task_id
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolContextModifier,
    ToolResult,
    ToolRuntimeContext,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.skills.models import SkillDefinition

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.messages import MessageParam


SKILL_TOOL_NAME = "Skill"
SKILL_TOOL_MAX_RESULT_SIZE_CHARS = 100_000
_PARENT_PERMISSION_PRECEDENCE_MODES: frozenset[str] = frozenset(
    {"bypassPermissions", "acceptEdits", "auto"}
)


class SkillToolInput(BaseModel):
    skill: str = Field(
        description='The skill name, for example "commit", "review-pr", or "pdf".'
    )
    args: str | None = Field(
        default=None,
        description="Optional arguments for the skill.",
    )


SKILL_TOOL_PROMPT = """Execute a skill within the main conversation.

When the user's request matches an available skill, invoke this tool before
answering normally. Skills provide specialized instructions and domain
knowledge.

When users reference a slash command such as "/commit" or "/review-pr", they
are referring to a skill. Pass the skill name without the leading slash; this
tool also accepts the leading slash for compatibility.

Available skills:
{skills}

After the tool returns, follow the returned skill instructions directly."""


def build_skill_tool(
    skills: Sequence[SkillDefinition],
    *,
    agent_definitions: Sequence[AgentDefinition] | None = None,
    default_agent_type: str = DEFAULT_AGENT_TYPE,
) -> Tool:
    """Build a concrete Skill tool over the turn-visible skill set."""

    skill_catalog = tuple(skills)
    prompt_skills = tuple(skill for skill in skill_catalog if is_model_invocable_skill(skill))
    agents = tuple(agent_definitions or get_builtin_agent_definitions())

    async def validate_input(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        command_name = normalize_skill_name(parsed.skill)
        if not command_name:
            return ValidationError(message=f"Invalid skill format: {parsed.skill}")

        skill = find_skill(command_name, skill_catalog)
        if skill is None:
            return ValidationError(message=f"Unknown skill: {command_name}")
        if skill.disable_model_invocation:
            return ValidationError(
                message=(
                    f"Skill {command_name} cannot be used with {SKILL_TOOL_NAME} "
                    "tool due to disable-model-invocation"
                )
            )
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        _ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        command_name = normalize_skill_name(parsed.skill)
        skill = find_skill(command_name, skill_catalog)
        updated_input = {"skill": command_name, "args": parsed.args}

        deny_rule = _matching_skill_rule(permission_context, "deny", command_name)
        if deny_rule is not None:
            return PermissionDenyDecision(
                message="Skill execution blocked by permission rules",
                decision_reason=RulePermissionDecisionReason(rule=deny_rule),
            )

        ask_rule = _matching_skill_rule(permission_context, "ask", command_name)
        if ask_rule is not None:
            return PermissionAskDecision(
                message=f"Execute skill: {command_name}",
                updated_input=updated_input,
                decision_reason=RulePermissionDecisionReason(rule=ask_rule),
                suggestions=_permission_suggestions(command_name),
                metadata=_permission_metadata(skill),
            )

        allow_rule = _matching_skill_rule(permission_context, "allow", command_name)
        if allow_rule is not None:
            return PermissionAllowDecision(
                updated_input=updated_input,
                decision_reason=RulePermissionDecisionReason(rule=allow_rule),
            )

        if skill is not None and skill_has_only_safe_properties(skill):
            return PermissionAllowDecision(updated_input=updated_input)

        return PermissionAskDecision(
            message=f"Execute skill: {command_name}",
            updated_input=updated_input,
            suggestions=_permission_suggestions(command_name),
            metadata=_permission_metadata(skill),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        command_name = normalize_skill_name(parsed.skill)
        skill = find_skill(command_name, skill_catalog)
        if skill is None:
            yield ToolResult(
                content=f"Unknown skill: {command_name}",
                is_error=True,
            )
            return

        rendered = skill.render_prompt(
            args=parsed.args or "",
            session_id=ctx.session_id,
        )
        if skill.context == "fork":
            async for event in _run_forked_skill(
                command_name=command_name,
                skill=skill,
                rendered_prompt=rendered,
                ctx=ctx,
                agents=agents,
                default_agent_type=default_agent_type,
            ):
                yield event
            return
        yield ToolResult(
            content=f"Launching skill: {command_name}",
            additional_messages=(_skill_prompt_message(rendered),),
            context_modifier=_build_context_modifier(skill),
        )

    async def prompt(_ctx: object | None = None) -> str:
        return SKILL_TOOL_PROMPT.format(skills=_format_available_skills(prompt_skills))

    return build_tool(
        ToolSpec(
            name=SKILL_TOOL_NAME,
            description="Execute a loaded prompt skill.",
            search_hint="invoke slash-command skills",
            input_model=SkillToolInput,
            call=call,
            prompt=prompt,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=True,
            is_open_world=True,
            should_defer=False,
            always_load=True,
            max_result_size_chars=SKILL_TOOL_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Executing skill: {normalize_skill_name(_coerce_input(input_).skill)}"
            ),
        )
    )


def create_skill_catalog_provider(
    *,
    upstream: ToolCatalogProvider | None = None,
    agent_definitions: Sequence[AgentDefinition] | None = None,
    default_agent_type: str = DEFAULT_AGENT_TYPE,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends Skill when skills are loaded."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing_skill = tuple(tool for tool in tools if tool.name != SKILL_TOOL_NAME)
        if not any(is_model_invocable_skill(skill) for skill in skills):
            return without_existing_skill
        return (
            *without_existing_skill,
            build_skill_tool(
                tuple(skills),
                agent_definitions=agent_definitions,
                default_agent_type=default_agent_type,
            ),
        )

    return provider


async def _run_forked_skill(
    *,
    command_name: str,
    skill: SkillDefinition,
    rendered_prompt: str,
    ctx: ToolUseContext,
    agents: Sequence[AgentDefinition],
    default_agent_type: str,
) -> AsyncIterator[ToolCallEvent]:
    runtime = _runtime_context(ctx)
    if runtime is None:
        yield ToolResult(
            content=(
                "Forked skill execution requires an active query runtime context."
            ),
            is_error=True,
        )
        return

    child_agent_id = generate_task_id("local_agent")
    selected_agent = _select_forked_skill_agent(
        skill,
        agents,
        default_agent_type=default_agent_type,
    )
    child_tools: tuple[Tool, ...] | None = None
    parent_effective_model = _current_effective_model(runtime, ctx)
    child_model: str | None = parent_effective_model
    child_system_prompt: str | None = None
    agent_type: str | None = skill.agent

    if selected_agent is not None:
        agent_type = selected_agent.agent_type
        child_system_prompt = selected_agent.system_prompt
        if skill.model is None:
            child_model = _resolve_agent_model(
                selected_agent.model,
                parent_effective_model,
            )
        resolved_tools = resolve_agent_tools(
            selected_agent,
            ctx.tools or runtime.config.tools,
            is_async=False,
            is_main_thread=False,
        )
        child_tools = resolved_tools.resolved_tools

    child_permission_context = runtime.deps.permission_context_for(ctx)
    child_permission_context = _with_agent_permission_mode(
        child_permission_context,
        selected_agent,
    )
    child_permission_context = _with_allowed_tool_rules(
        child_permission_context,
        skill.allowed_tools,
    )
    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=(cast("MessageParam", _skill_prompt_message(rendered_prompt)),),
            parent_config=runtime.config,
            parent_deps=deps_for_agent_context_policy(runtime.deps, selected_agent),
            parent_ctx=ctx,
            agent_id=child_agent_id,
            agent_type=agent_type,
            system_prompt=child_system_prompt,
            model=child_model,
            model_override=skill.model,
            effort=skill.effort,
            tools=child_tools,
            permission_context=child_permission_context,
            transcript_label=f"skill:{command_name}",
        )
    )
    if result.subtype == "error_aborted" and ctx.abort_event.is_set():
        raise asyncio.CancelledError()
    result_text = result.final_message or "\n".join(result.errors)
    if not result_text:
        result_text = "Skill execution completed"
    yield ToolResult(
        content=_forked_skill_result_content(
            command_name=command_name,
            result=result_text,
        ),
        is_error=result.is_error,
    )


def _runtime_context(ctx: ToolUseContext) -> ToolRuntimeContext | None:
    return ctx.runtime


def _current_effective_model(
    runtime: ToolRuntimeContext,
    ctx: ToolUseContext,
) -> str:
    if ctx.model_override is not None:
        override = ctx.model_override.strip()
        if override == "inherit":
            return runtime.effective_model or runtime.config.model
        return ctx.model_override
    return runtime.effective_model or runtime.config.model


def _resolve_agent_model(
    agent_model: str | None,
    parent_effective_model: str,
) -> str:
    if agent_model is None:
        return parent_effective_model
    if agent_model.strip() == "inherit":
        return parent_effective_model
    return agent_model


def _with_agent_permission_mode(
    permission_context: ToolPermissionContext,
    selected_agent: AgentDefinition | None,
) -> ToolPermissionContext:
    if selected_agent is None or selected_agent.permission_mode is None:
        return permission_context
    if permission_context.mode in _PARENT_PERMISSION_PRECEDENCE_MODES:
        return permission_context
    return replace(permission_context, mode=selected_agent.permission_mode)


def _select_forked_skill_agent(
    skill: SkillDefinition,
    agents: Sequence[AgentDefinition],
    *,
    default_agent_type: str,
) -> AgentDefinition | None:
    requested = skill.agent or default_agent_type
    return (
        find_agent_definition(requested, agents)
        or find_agent_definition(default_agent_type, agents)
        or (agents[0] if agents else None)
    )


def _forked_skill_result_content(
    *,
    command_name: str,
    result: str,
) -> str:
    return (
        f'Skill "{command_name}" completed (forked execution).\n\n'
        f"Result:\n{result}"
    )


def normalize_skill_name(raw: str) -> str:
    trimmed = raw.strip()
    return trimmed[1:] if trimmed.startswith("/") else trimmed


def find_skill(
    command_name: str,
    skills: Sequence[SkillDefinition],
) -> SkillDefinition | None:
    for skill in skills:
        if skill.name == command_name:
            return skill
        if skill.display_name == command_name:
            return skill
        if command_name in skill.aliases:
            return skill
    return None


def skill_has_only_safe_properties(skill: SkillDefinition) -> bool:
    """Return whether the skill can be auto-allowed without user approval.

    The reference uses an allowlist of prompt-command property keys. Raygent's
    smaller SkillDefinition shape maps the meaningful unsafe fields to the same
    policy: extra tool grants, hooks, and shell settings require approval.
    """

    if skill.allowed_tools:
        return False
    if skill.shell is not None:
        return False
    hooks = skill.hooks
    return hooks is None or len(hooks) == 0


def is_model_invocable_skill(skill: SkillDefinition) -> bool:
    """Whether the model should see this skill advertised in the Skill prompt."""

    return not skill.disable_model_invocation


def _coerce_input(input_: BaseModel) -> SkillToolInput:
    if isinstance(input_, SkillToolInput):
        return input_
    return SkillToolInput.model_validate(input_.model_dump())


def _matching_skill_rule(
    context: ToolPermissionContext,
    behavior: str,
    command_name: str,
) -> PermissionRule | None:
    for rule in get_rules(context, behavior):  # type: ignore[arg-type]
        value = rule.rule_value
        if value.tool_name != SKILL_TOOL_NAME:
            continue
        if _rule_content_matches(value.rule_content, command_name):
            return rule
    return None


def _rule_content_matches(rule_content: str | None, command_name: str) -> bool:
    if rule_content is None:
        return True
    normalized = normalize_skill_name(rule_content)
    if normalized == command_name:
        return True
    if normalized.endswith(":*"):
        prefix = normalized[:-2]
        return command_name.startswith(prefix)
    return False


def _permission_suggestions(command_name: str) -> tuple[AddPermissionRules, ...]:
    return (
        AddPermissionRules(
            destination="localSettings",
            behavior="allow",
            rules=(PermissionRuleValue(tool_name=SKILL_TOOL_NAME, rule_content=command_name),),
        ),
        AddPermissionRules(
            destination="localSettings",
            behavior="allow",
            rules=(
                PermissionRuleValue(
                    tool_name=SKILL_TOOL_NAME,
                    rule_content=f"{command_name}:*",
                ),
            ),
        ),
    )


def _permission_metadata(skill: SkillDefinition | None) -> PermissionMetadata | None:
    if skill is None:
        return None
    return PermissionMetadata(
        command=PermissionCommandMetadata(
            name=skill.name,
            description=skill.description,
        )
    )


def _format_available_skills(skills: Sequence[SkillDefinition]) -> str:
    if not skills:
        return "- No skills are currently loaded."
    lines: list[str] = []
    for skill in sorted(skills, key=lambda item: item.name):
        description = skill.description
        if skill.when_to_use:
            description = f"{description} - {skill.when_to_use}"
        lines.append(f"- {skill.name}: {description}")
    return "\n".join(lines)


def _skill_prompt_message(rendered_prompt: str) -> dict[str, Any]:
    return {"role": "user", "content": rendered_prompt}


def _build_context_modifier(skill: SkillDefinition) -> ToolContextModifier | None:
    if not skill.allowed_tools and skill.model is None and skill.effort is None:
        return None

    def modify(ctx: ToolUseContext) -> ToolUseContext:
        next_ctx = ctx
        if skill.allowed_tools:
            next_ctx = replace(
                next_ctx,
                permission_context=_with_allowed_tool_rules(
                    next_ctx.permission_context,
                    skill.allowed_tools,
                ),
            )
        if skill.model is not None:
            next_ctx = replace(next_ctx, model_override=skill.model)
        if skill.effort is not None:
            next_ctx = replace(next_ctx, reasoning_effort_override=skill.effort)
        return next_ctx

    return modify


def _with_allowed_tool_rules(
    permission_context: ToolPermissionContext,
    allowed_tools: Sequence[str],
) -> ToolPermissionContext:
    allow_rules: dict[PermissionRuleSource, tuple[str, ...]] = dict(
        permission_context.always_allow_rules
    )
    existing = allow_rules.get("command", ())
    allow_rules["command"] = _dedupe_preserve_order((*existing, *allowed_tools))
    return replace(permission_context, always_allow_rules=allow_rules)


def _dedupe_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


__all__ = [
    "SKILL_TOOL_MAX_RESULT_SIZE_CHARS",
    "SKILL_TOOL_NAME",
    "SKILL_TOOL_PROMPT",
    "SkillToolInput",
    "build_skill_tool",
    "create_skill_catalog_provider",
    "find_skill",
    "is_model_invocable_skill",
    "normalize_skill_name",
    "skill_has_only_safe_properties",
]
