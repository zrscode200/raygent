"""Pure permission engine and PreToolUse hook-decision resolution.

This ports the reference permission path at the kernel level:


No UI queues, settings persistence, classifier execution, telemetry, or concrete
tool orchestration live here. Those are adapter/execution layers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from raygent_harness.core.permissions import (
    ModePermissionDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionBehavior,
    PermissionDecision,
    PermissionDenyDecision,
    PermissionPassthrough,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
    PermissionRuleValue,
    RulePermissionDecisionReason,
    ToolPermissionContext,
)
from raygent_harness.core.tool import (
    Tool,
    ToolDescriptionContext,
    ToolUseContext,
)

PERMISSION_RULE_SOURCES: tuple[PermissionRuleSource, ...] = (
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
    "cliArg",
    "command",
    "session",
)


@dataclass(frozen=True)
class PermissionRequest:
    """Request handed to an injected permission handler."""

    tool: Tool
    input: BaseModel
    tool_use_context: ToolUseContext
    permission_context: ToolPermissionContext
    decision: PermissionAskDecision
    description: str
    tool_use_id: str | None = None


class PermissionHandler(Protocol):
    """Headless adapter seam for ask/interactive permission decisions."""

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        ...


@dataclass(frozen=True)
class ResolvedPermission:
    decision: PermissionDecision
    input: BaseModel


def permission_rule_value_from_string(rule_string: str) -> PermissionRuleValue:
    """Parse `Tool` or `Tool(content)` rule strings.

    Mirrors `permissionRuleValueFromString` for the format relevant to engine
    matching. Legacy product-name normalization is intentionally omitted; Raygent
    uses tool aliases for renamed-tool compatibility.
    """

    open_index = _find_first_unescaped(rule_string, "(")
    if open_index == -1:
        return PermissionRuleValue(tool_name=rule_string)

    close_index = _find_last_unescaped(rule_string, ")")
    if close_index == -1 or close_index <= open_index:
        return PermissionRuleValue(tool_name=rule_string)
    if close_index != len(rule_string) - 1:
        return PermissionRuleValue(tool_name=rule_string)

    tool_name = rule_string[:open_index]
    raw_content = rule_string[open_index + 1 : close_index]
    if not tool_name:
        return PermissionRuleValue(tool_name=rule_string)
    if raw_content in {"", "*"}:
        return PermissionRuleValue(tool_name=tool_name)

    return PermissionRuleValue(
        tool_name=tool_name,
        rule_content=_unescape_rule_content(raw_content),
    )


def permission_rule_value_to_string(rule_value: PermissionRuleValue) -> str:
    if not rule_value.rule_content:
        return rule_value.tool_name
    return f"{rule_value.tool_name}({_escape_rule_content(rule_value.rule_content)})"


def create_permission_request_message(
    tool_name: str,
    decision_reason: object | None = None,
) -> str:
    """Human-readable default ask message.

    The reference has product-specific wording and analytics context. Raygent
    keeps this minimal while preserving reason-specific messages used by tests
    and future SDK surfaces.
    """

    if isinstance(decision_reason, RulePermissionDecisionReason):
        rule = permission_rule_value_to_string(decision_reason.rule.rule_value)
        return f"Permission rule '{rule}' requires approval for this {tool_name} command"
    if isinstance(decision_reason, ModePermissionDecisionReason):
        return (
            f"Current permission mode ({decision_reason.mode}) requires approval "
            f"for this {tool_name} command"
        )
    return f"Permission required to use {tool_name}."


def get_rules(
    context: ToolPermissionContext,
    behavior: PermissionBehavior,
) -> tuple[PermissionRule, ...]:
    source_rules = {
        "allow": context.always_allow_rules,
        "deny": context.always_deny_rules,
        "ask": context.always_ask_rules,
    }[behavior]
    rules: list[PermissionRule] = []
    for source in PERMISSION_RULE_SOURCES:
        for rule_string in source_rules.get(source, ()):
            rules.append(
                PermissionRule(
                    source=source,
                    rule_behavior=behavior,
                    rule_value=permission_rule_value_from_string(rule_string),
                )
            )
    return tuple(rules)


def tool_always_allowed_rule(
    context: ToolPermissionContext,
    tool: Tool,
) -> PermissionRule | None:
    return _find_whole_tool_rule(get_rules(context, "allow"), tool)


def get_deny_rule_for_tool(
    context: ToolPermissionContext,
    tool: Tool,
) -> PermissionRule | None:
    return _find_whole_tool_rule(get_rules(context, "deny"), tool)


def get_ask_rule_for_tool(
    context: ToolPermissionContext,
    tool: Tool,
) -> PermissionRule | None:
    return _find_whole_tool_rule(get_rules(context, "ask"), tool)


async def check_rule_based_permissions(
    *,
    tool: Tool,
    input: BaseModel,
    tool_use_context: ToolUseContext,
    permission_context: ToolPermissionContext,
) -> PermissionAskDecision | PermissionDenyDecision | None:
    """Reference step-1 rule checks respected even under bypass/hook allow."""

    deny_rule = get_deny_rule_for_tool(permission_context, tool)
    if deny_rule is not None:
        return PermissionDenyDecision(
            message=f"Permission to use {tool.name} has been denied.",
            decision_reason=RulePermissionDecisionReason(rule=deny_rule),
        )

    ask_rule = get_ask_rule_for_tool(permission_context, tool)
    if ask_rule is not None:
        return PermissionAskDecision(
            message=create_permission_request_message(tool.name),
            decision_reason=RulePermissionDecisionReason(rule=ask_rule),
        )

    tool_result = await _check_tool_permissions(
        tool,
        input,
        tool_use_context,
        permission_context,
    )
    if isinstance(tool_result, PermissionDenyDecision):
        return tool_result
    if isinstance(tool_result, PermissionAskDecision) and (
        _is_rule_ask(tool_result) or _is_safety_check_ask(tool_result)
    ):
        return tool_result
    return None


async def can_use_tool(
    *,
    tool: Tool,
    input: BaseModel,
    tool_use_context: ToolUseContext,
    permission_context: ToolPermissionContext,
    handler: PermissionHandler | None = None,
    tool_use_id: str | None = None,
    force_decision: PermissionResult | None = None,
    is_non_interactive_session: bool = False,
    tools: Sequence[Tool] = (),
) -> ResolvedPermission:
    """Resolve whether a parsed, validated tool call may run."""

    _raise_if_aborted(tool_use_context)
    decision = (
        _to_permission_decision(force_decision, tool)
        if force_decision is not None
        else await _evaluate_permission(
            tool=tool,
            input=input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
        )
    )
    _raise_if_aborted(tool_use_context)

    if decision.behavior == "ask" and permission_context.mode == "dontAsk":
        decision = PermissionDenyDecision(
            message=f"Permission to use {tool.name} was denied by dontAsk mode.",
            decision_reason=ModePermissionDecisionReason(mode="dontAsk"),
        )

    if decision.behavior == "ask" and permission_context.should_avoid_permission_prompts:
        decision = PermissionDenyDecision(
            message=decision.message,
            decision_reason=decision.decision_reason
            or ModePermissionDecisionReason(mode=permission_context.mode),
        )

    if decision.behavior == "ask" and handler is not None:
        _raise_if_aborted(tool_use_context)
        description = await tool.describe(
            input,
            ToolDescriptionContext(
                is_non_interactive_session=is_non_interactive_session,
                permission_context=permission_context,
                tools=tools or (tool,),
            ),
        )
        _raise_if_aborted(tool_use_context)
        decision = await handler.ask(
            PermissionRequest(
                tool=tool,
                input=input,
                tool_use_context=tool_use_context,
                permission_context=permission_context,
                decision=decision,
                description=description,
                tool_use_id=tool_use_id,
            )
        )

    return ResolvedPermission(decision=decision, input=input)


async def resolve_hook_permission_decision(
    *,
    hook_permission_result: PermissionResult | None,
    tool: Tool,
    input: BaseModel,
    tool_use_context: ToolUseContext,
    permission_context: ToolPermissionContext,
    handler: PermissionHandler | None = None,
    tool_use_id: str | None = None,
    require_can_use_tool: bool = False,
    is_non_interactive_session: bool = False,
    tools: Sequence[Tool] = (),
) -> ResolvedPermission:
    """Resolve PreToolUse hook permission output into a final decision."""

    _raise_if_aborted(tool_use_context)
    if isinstance(hook_permission_result, PermissionAllowDecision):
        hook_input = _coerce_input(tool, hook_permission_result.updated_input, input)
        interaction_satisfied = (
            tool.requires_user_interaction()
            and hook_permission_result.updated_input is not None
        )
        if (
            (tool.requires_user_interaction() and not interaction_satisfied)
            or require_can_use_tool
        ):
            return await can_use_tool(
                tool=tool,
                input=hook_input,
                tool_use_context=tool_use_context,
                permission_context=permission_context,
                handler=handler,
                tool_use_id=tool_use_id,
                is_non_interactive_session=is_non_interactive_session,
                tools=tools,
            )

        rule_check = await check_rule_based_permissions(
            tool=tool,
            input=hook_input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
        )
        if rule_check is None:
            return ResolvedPermission(
                decision=hook_permission_result,
                input=hook_input,
            )
        if rule_check.behavior == "deny":
            return ResolvedPermission(decision=rule_check, input=hook_input)
        return await can_use_tool(
            tool=tool,
            input=hook_input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
            handler=handler,
            tool_use_id=tool_use_id,
            is_non_interactive_session=is_non_interactive_session,
            tools=tools,
        )

    if isinstance(hook_permission_result, PermissionDenyDecision):
        return ResolvedPermission(decision=hook_permission_result, input=input)

    if isinstance(hook_permission_result, PermissionAskDecision):
        ask_input = _coerce_input(tool, hook_permission_result.updated_input, input)
        return await can_use_tool(
            tool=tool,
            input=ask_input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
            handler=handler,
            tool_use_id=tool_use_id,
            force_decision=hook_permission_result,
            is_non_interactive_session=is_non_interactive_session,
            tools=tools,
        )

    return await can_use_tool(
        tool=tool,
        input=input,
        tool_use_context=tool_use_context,
        permission_context=permission_context,
        handler=handler,
        tool_use_id=tool_use_id,
        is_non_interactive_session=is_non_interactive_session,
        tools=tools,
    )


async def _evaluate_permission(
    *,
    tool: Tool,
    input: BaseModel,
    tool_use_context: ToolUseContext,
    permission_context: ToolPermissionContext,
) -> PermissionDecision:
    deny_rule = get_deny_rule_for_tool(permission_context, tool)
    if deny_rule is not None:
        return PermissionDenyDecision(
            message=f"Permission to use {tool.name} has been denied.",
            decision_reason=RulePermissionDecisionReason(rule=deny_rule),
        )

    ask_rule = get_ask_rule_for_tool(permission_context, tool)
    if ask_rule is not None:
        return PermissionAskDecision(
            message=create_permission_request_message(tool.name),
            decision_reason=RulePermissionDecisionReason(rule=ask_rule),
        )

    tool_result = await _check_tool_permissions(
        tool,
        input,
        tool_use_context,
        permission_context,
    )
    if isinstance(tool_result, PermissionDenyDecision):
        return tool_result
    if tool.requires_user_interaction() and isinstance(tool_result, PermissionAskDecision):
        return tool_result
    if isinstance(tool_result, PermissionAskDecision) and (
        _is_rule_ask(tool_result) or _is_safety_check_ask(tool_result)
    ):
        return tool_result

    should_bypass = permission_context.mode == "bypassPermissions" or (
        permission_context.mode == "plan"
        and permission_context.is_bypass_permissions_mode_available
    )
    if should_bypass:
        return PermissionAllowDecision(
            updated_input=_updated_input_or_fallback(tool_result, input),
            decision_reason=ModePermissionDecisionReason(mode=permission_context.mode),
        )

    allow_rule = tool_always_allowed_rule(permission_context, tool)
    if allow_rule is not None:
        return PermissionAllowDecision(
            updated_input=_updated_input_or_fallback(tool_result, input),
            decision_reason=RulePermissionDecisionReason(rule=allow_rule),
        )

    return _to_permission_decision(tool_result, tool)


def _to_permission_decision(
    result: PermissionResult,
    tool: Tool,
) -> PermissionDecision:
    if isinstance(result, PermissionPassthrough):
        return PermissionAskDecision(
            message=create_permission_request_message(tool.name, result.decision_reason),
            decision_reason=result.decision_reason,
            suggestions=result.suggestions,
            blocked_path=result.blocked_path,
            pending_classifier_check=result.pending_classifier_check,
        )
    return result


async def _check_tool_permissions(
    tool: Tool,
    input: BaseModel,
    tool_use_context: ToolUseContext,
    permission_context: ToolPermissionContext,
) -> PermissionResult:
    """Run tool-specific permission checks with reference-style resilience."""

    _raise_if_aborted(tool_use_context)
    try:
        result = await tool.check_permissions(
            input,
            tool_use_context,
            permission_context,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return PermissionPassthrough(message=create_permission_request_message(tool.name))
    _raise_if_aborted(tool_use_context)
    return result


def _raise_if_aborted(tool_use_context: ToolUseContext) -> None:
    if tool_use_context.abort_event.is_set():
        raise asyncio.CancelledError()


def _updated_input_or_fallback(
    result: PermissionResult,
    fallback: BaseModel,
) -> Mapping[str, object]:
    if isinstance(result, PermissionAllowDecision | PermissionAskDecision):
        return result.updated_input or fallback.model_dump()
    return fallback.model_dump()


def _is_rule_ask(result: PermissionResult) -> bool:
    return (
        isinstance(result, PermissionAskDecision)
        and isinstance(result.decision_reason, RulePermissionDecisionReason)
        and result.decision_reason.rule.rule_behavior == "ask"
    )


def _is_safety_check_ask(result: PermissionResult) -> bool:
    return (
        isinstance(result, PermissionAskDecision)
        and getattr(result.decision_reason, "type", None) == "safetyCheck"
    )


def _coerce_input(
    tool: Tool,
    updated_input: Mapping[str, object] | None,
    fallback: BaseModel,
) -> BaseModel:
    if updated_input is None:
        return fallback
    return tool.input_model.model_validate(dict(updated_input))


def _find_whole_tool_rule(
    rules: Sequence[PermissionRule],
    tool: Tool,
) -> PermissionRule | None:
    for rule in rules:
        if _tool_matches_rule(tool, rule):
            return rule
    return None


def _tool_matches_rule(tool: Tool, rule: PermissionRule) -> bool:
    if rule.rule_value.rule_content is not None:
        return False
    rule_name = rule.rule_value.tool_name
    if rule_name == tool.name or rule_name in tool.aliases:
        return True
    if rule_name.endswith("__*"):
        return tool.name.startswith(rule_name[:-1])
    return tool.name.startswith(f"{rule_name}__")


def _find_first_unescaped(value: str, char: str) -> int:
    for index, current in enumerate(value):
        if current == char and _preceding_backslashes(value, index) % 2 == 0:
            return index
    return -1


def _find_last_unescaped(value: str, char: str) -> int:
    for index in range(len(value) - 1, -1, -1):
        if value[index] == char and _preceding_backslashes(value, index) % 2 == 0:
            return index
    return -1


def _preceding_backslashes(value: str, index: int) -> int:
    count = 0
    cursor = index - 1
    while cursor >= 0 and value[cursor] == "\\":
        count += 1
        cursor -= 1
    return count


def _escape_rule_content(content: str) -> str:
    return content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _unescape_rule_content(content: str) -> str:
    return content.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")


__all__ = [
    "PERMISSION_RULE_SOURCES",
    "PermissionHandler",
    "PermissionRequest",
    "ResolvedPermission",
    "can_use_tool",
    "check_rule_based_permissions",
    "create_permission_request_message",
    "get_ask_rule_for_tool",
    "get_deny_rule_for_tool",
    "get_rules",
    "permission_rule_value_from_string",
    "permission_rule_value_to_string",
    "resolve_hook_permission_decision",
    "tool_always_allowed_rule",
]
