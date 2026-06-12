"""Restricted tool and permission policy for memory extraction children.

"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from raygent_harness.core.permission_engine import permission_rule_value_from_string
from raygent_harness.core.permissions import (
    AdditionalWorkingDirectory,
    OtherPermissionDecisionReason,
    PermissionAllowDecision,
    PermissionDenyDecision,
    PermissionResult,
    PermissionRuleSource,
    ToolPermissionContext,
    ToolPermissionRulesBySource,
)
from raygent_harness.core.tool import (
    ToolCallEvent,
    ToolSpec,
    ValidationError,
    ValidationResult,
    build_tool,
    find_tool_by_name,
)
from raygent_harness.memdir.paths import (
    MemorySettings,
    get_auto_mem_path,
    is_auto_mem_path,
)
from raygent_harness.services.extract_memories.prompts import (
    BASH_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
)
from raygent_harness.tools.bash_tool import validate_restricted_bash_command
from raygent_harness.tools.file_permissions import expand_file_path

if TYPE_CHECKING:
    from raygent_harness.core.tool import Tool, ToolUseContext


READ_ONLY_EXTRACTION_TOOL_NAMES = (
    FILE_READ_TOOL_NAME,
    GREP_TOOL_NAME,
    GLOB_TOOL_NAME,
    BASH_TOOL_NAME,
)
WRITE_EXTRACTION_TOOL_NAMES = (FILE_WRITE_TOOL_NAME, FILE_EDIT_TOOL_NAME)
EXTRACTION_TOOL_NAMES = (*READ_ONLY_EXTRACTION_TOOL_NAMES, *WRITE_EXTRACTION_TOOL_NAMES)


@dataclass(frozen=True)
class ExtractionToolPolicy:
    """Restricted child tool pool plus the permission context it requires."""

    tools: tuple[Tool, ...]
    tool_names: tuple[str, ...]
    missing_tool_names: tuple[str, ...]
    missing_required_tool_names: tuple[str, ...]
    filtered_tool_count: int
    permission_context: ToolPermissionContext

    @property
    def is_usable(self) -> bool:
        """Whether the child can both inspect and save memories."""
        return not self.missing_required_tool_names


def build_extraction_tool_policy(
    tools: Sequence[Tool],
    *,
    settings: MemorySettings,
    parent_permission_context: ToolPermissionContext | None = None,
) -> ExtractionToolPolicy:
    """Select and wrap the tools an extraction child may see.

    Tool absence is the first deny layer: Agent/Task/Skill/MCP/remote/team tools
    never enter the child catalog. Write-capable tools that do enter are wrapped
    with an auto-memory path guard before delegating to their normal behavior.
    """

    parent_tools = tuple(tools)
    selected: list[Tool] = []
    selected_names: list[str] = []
    missing: list[str] = []
    seen_tool_ids: set[int] = set()

    for canonical_name in EXTRACTION_TOOL_NAMES:
        tool = find_tool_by_name(parent_tools, canonical_name)
        if tool is None:
            missing.append(canonical_name)
            continue
        tool_id = id(tool)
        if tool_id in seen_tool_ids:
            continue
        selected.append(
            wrap_extraction_tool(
                tool,
                canonical_name=canonical_name,
                settings=settings,
            )
        )
        selected_names.append(canonical_name)
        seen_tool_ids.add(tool_id)

    selected_tool_names = tuple(selected_names)
    return ExtractionToolPolicy(
        tools=tuple(selected),
        tool_names=selected_tool_names,
        missing_tool_names=tuple(missing),
        missing_required_tool_names=_missing_required_tool_names(selected_tool_names),
        filtered_tool_count=max(0, len(parent_tools) - len(selected)),
        permission_context=build_extraction_permission_context(
            settings=settings,
            parent_permission_context=parent_permission_context,
        ),
    )


def build_extraction_permission_context(
    *,
    settings: MemorySettings,
    parent_permission_context: ToolPermissionContext | None = None,
) -> ToolPermissionContext:
    """Build a non-bypass child permission context for extraction.

    Reference allows Read/Grep/Glob without ordinary permission prompts and
    constrains Write/Edit to the auto-memory directory. Raygent preserves
    explicit read denies and suspicious-path checks, but drops inherited read
    asks and adds a broad read allow so the non-interactive extraction child
    does not silently lose recall/discovery context.
    """

    parent = parent_permission_context or ToolPermissionContext()
    memory_dir = get_auto_mem_path(settings)
    session_allow = cast(
        "ToolPermissionRulesBySource",
        {
            "session": (
                f"{FILE_EDIT_TOOL_NAME}({memory_dir}/**)",
            )
        },
    )
    return ToolPermissionContext(
        mode="default",
        always_allow_rules=_merge_rule_maps(
            _filter_rule_map(parent.always_allow_rules, READ_ONLY_EXTRACTION_TOOL_NAMES),
            session_allow,
        ),
        always_deny_rules=_filter_rule_map(
            parent.always_deny_rules,
            READ_ONLY_EXTRACTION_TOOL_NAMES,
        ),
        additional_working_directories=_with_extraction_read_root(
            parent.additional_working_directories,
        ),
        should_avoid_permission_prompts=True,
        await_automated_checks_before_dialog=parent.await_automated_checks_before_dialog,
    )

def wrap_extraction_tool(
    tool: Tool,
    *,
    canonical_name: str,
    settings: MemorySettings,
) -> Tool:
    """Wrap an allowed tool with extraction-specific guards."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        guard = _validate_extraction_input(
            input_,
            ctx=ctx,
            tool_name=canonical_name,
            tool=tool,
            settings=settings,
        )
        if guard is not None:
            return guard
        return await tool.validate_input(input_, ctx)

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        deny = _deny_extraction_permission(
            input_,
            ctx=ctx,
            tool_name=canonical_name,
            tool=tool,
            settings=settings,
        )
        if deny is not None:
            return deny
        if canonical_name in (FILE_READ_TOOL_NAME, GREP_TOOL_NAME, GLOB_TOOL_NAME):
            read_context = _broad_read_permission_context(permission_context)
            permission = await tool.check_permissions(input_, ctx, read_context)
            if not isinstance(permission, PermissionAllowDecision):
                return permission
            return PermissionAllowDecision(updated_input=input_.model_dump())
        if canonical_name == BASH_TOOL_NAME and _is_extraction_bash_read_only(
            tool,
            input_,
        ):
            return PermissionAllowDecision(updated_input=input_.model_dump())
        return await tool.check_permissions(input_, ctx, permission_context)

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        call_ctx = (
            replace(
                ctx,
                permission_context=_broad_read_permission_context(ctx.permission_context),
            )
            if canonical_name in (FILE_READ_TOOL_NAME, GREP_TOOL_NAME, GLOB_TOOL_NAME)
            else ctx
        )
        async for event in tool.call(input_, call_ctx):
            yield event

    return build_tool(
        ToolSpec(
            name=tool.name,
            aliases=tool.aliases,
            description=tool.description,
            search_hint=tool.search_hint,
            input_model=tool.input_model,
            input_schema=tool.input_schema,
            call=call,
            prompt=tool.prompt,
            validate_input=validate_input,
            check_permissions=check_permissions,
            describe=tool.describe,
            get_activity_description=tool.get_activity_description,
            is_enabled=tool.is_enabled,
            is_concurrency_safe=tool.is_concurrency_safe,
            is_read_only=lambda input_: _is_extraction_read_only(
                input_,
                canonical_name=canonical_name,
                tool=tool,
            ),
            is_destructive=lambda input_: _is_extraction_destructive(
                input_,
                canonical_name=canonical_name,
                tool=tool,
            ),
            is_open_world=tool.is_open_world,
            requires_user_interaction=tool.requires_user_interaction,
            interrupt_behavior=tool.interrupt_behavior,
            should_defer=False,
            always_load=True,
            max_result_size_chars=tool.max_result_size_chars,
        )
    )


def _validate_extraction_input(
    input_: BaseModel,
    *,
    ctx: ToolUseContext,
    tool_name: str,
    tool: Tool,
    settings: MemorySettings,
) -> ValidationError | None:
    if tool_name in WRITE_EXTRACTION_TOOL_NAMES:
        path = _input_path(input_)
        if path is None:
            return None
        if not _path_is_auto_memory(path, ctx=ctx, settings=settings):
            return ValidationError(
                message=(
                    f"{tool_name} is restricted to the auto-memory directory "
                    "during memory extraction."
                )
            )
    if tool_name == BASH_TOOL_NAME and not _is_extraction_bash_read_only(tool, input_):
        return ValidationError(
            message="Bash is restricted to read-only commands during memory extraction."
        )
    return None


def _deny_extraction_permission(
    input_: BaseModel,
    *,
    ctx: ToolUseContext,
    tool_name: str,
    tool: Tool,
    settings: MemorySettings,
) -> PermissionDenyDecision | None:
    if tool_name in WRITE_EXTRACTION_TOOL_NAMES:
        path = _input_path(input_)
        if path is None:
            return None
        if not _path_is_auto_memory(path, ctx=ctx, settings=settings):
            return _deny(
                f"{tool_name} is restricted to the auto-memory directory "
                "during memory extraction."
            )
    if tool_name == BASH_TOOL_NAME and not _is_extraction_bash_read_only(tool, input_):
        return _deny("Bash is restricted to read-only commands during memory extraction.")
    return None


def _deny(message: str) -> PermissionDenyDecision:
    return PermissionDenyDecision(
        message=message,
        decision_reason=OtherPermissionDecisionReason(reason=message),
    )


def _input_path(input_: BaseModel) -> str | None:
    raw = input_.model_dump()
    value = raw.get("file_path", raw.get("path"))
    return value if isinstance(value, str) and value else None


def _path_is_auto_memory(
    path: str,
    *,
    ctx: ToolUseContext,
    settings: MemorySettings,
) -> bool:
    return is_auto_mem_path(Path(expand_file_path(path, cwd=ctx.cwd)), settings)


def _tool_reports_read_only(tool: Tool, input_: BaseModel) -> bool:
    try:
        return tool.is_read_only(input_)
    except Exception:
        return False


def _is_extraction_bash_read_only(tool: Tool, input_: BaseModel) -> bool:
    if _tool_reports_read_only(tool, input_):
        return True
    raw = input_.model_dump()
    if raw.get("run_in_background") is True:
        return False
    command = raw.get("command")
    if not isinstance(command, str):
        return False
    return validate_restricted_bash_command(command).allowed


def _is_extraction_read_only(
    input_: BaseModel,
    *,
    canonical_name: str,
    tool: Tool,
) -> bool:
    if canonical_name in WRITE_EXTRACTION_TOOL_NAMES:
        return False
    if canonical_name == BASH_TOOL_NAME:
        return _is_extraction_bash_read_only(tool, input_)
    return tool.is_read_only(input_)


def _is_extraction_destructive(
    input_: BaseModel,
    *,
    canonical_name: str,
    tool: Tool,
) -> bool:
    if canonical_name in WRITE_EXTRACTION_TOOL_NAMES:
        return True
    if canonical_name == BASH_TOOL_NAME:
        return not _is_extraction_bash_read_only(tool, input_)
    return tool.is_destructive(input_)


def _filter_rule_map(
    rules: ToolPermissionRulesBySource,
    tool_names: Sequence[str],
) -> ToolPermissionRulesBySource:
    allowed = frozenset(tool_names)
    filtered: dict[PermissionRuleSource, tuple[str, ...]] = {}
    for source, source_rules in rules.items():
        kept = tuple(
            rule
            for rule in source_rules
            if permission_rule_value_from_string(rule).tool_name in allowed
        )
        if kept:
            filtered[source] = kept
    return cast("ToolPermissionRulesBySource", filtered)


def _merge_rule_maps(
    *rule_maps: ToolPermissionRulesBySource,
) -> ToolPermissionRulesBySource:
    merged: dict[PermissionRuleSource, tuple[str, ...]] = {}
    for rule_map in rule_maps:
        for source, rules in rule_map.items():
            existing = merged.get(source, ())
            merged[source] = (*existing, *tuple(rules))
    return cast("ToolPermissionRulesBySource", merged)


def _broad_read_permission_context(
    permission_context: ToolPermissionContext,
) -> ToolPermissionContext:
    """Allow extraction reads broadly while preserving explicit denies."""

    return ToolPermissionContext(
        mode=permission_context.mode,
        additional_working_directories=_with_extraction_read_root(
            permission_context.additional_working_directories,
        ),
        always_allow_rules=permission_context.always_allow_rules,
        always_deny_rules=permission_context.always_deny_rules,
        should_avoid_permission_prompts=permission_context.should_avoid_permission_prompts,
        await_automated_checks_before_dialog=(
            permission_context.await_automated_checks_before_dialog
        ),
    )


def _with_extraction_read_root(
    additional_working_directories: Mapping[str, AdditionalWorkingDirectory],
) -> dict[str, AdditionalWorkingDirectory]:
    roots = dict(additional_working_directories)
    roots.setdefault("/", AdditionalWorkingDirectory(path="/", source="session"))
    return roots


def _missing_required_tool_names(tool_names: Sequence[str]) -> tuple[str, ...]:
    available = frozenset(tool_names)
    missing: list[str] = []
    if FILE_READ_TOOL_NAME not in available:
        missing.append(FILE_READ_TOOL_NAME)
    if not any(name in available for name in WRITE_EXTRACTION_TOOL_NAMES):
        missing.append(f"{FILE_WRITE_TOOL_NAME}|{FILE_EDIT_TOOL_NAME}")
    return tuple(missing)


__all__ = [
    "EXTRACTION_TOOL_NAMES",
    "READ_ONLY_EXTRACTION_TOOL_NAMES",
    "WRITE_EXTRACTION_TOOL_NAMES",
    "ExtractionToolPolicy",
    "build_extraction_permission_context",
    "build_extraction_tool_policy",
    "wrap_extraction_tool",
]
