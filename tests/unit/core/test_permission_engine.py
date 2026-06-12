from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from raygent_harness.core.permission_engine import (
    PermissionRequest,
    can_use_tool,
    create_permission_request_message,
    permission_rule_value_from_string,
    permission_rule_value_to_string,
    resolve_hook_permission_decision,
)
from raygent_harness.core.permissions import (
    ModePermissionDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDecision,
    PermissionDenyDecision,
    PermissionMode,
    PermissionPassthrough,
    PermissionResult,
    PermissionRule,
    PermissionRuleValue,
    RulePermissionDecisionReason,
    SafetyCheckPermissionDecisionReason,
    ToolPermissionContext,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolDescriptionContext,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)


class ExampleInput(BaseModel):
    command: str


type CheckPermissionsFn = Callable[
    [BaseModel, ToolUseContext, ToolPermissionContext], Awaitable[PermissionResult]
]


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _tool_ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )


def _aborted_tool_ctx() -> ToolUseContext:
    ctx = _tool_ctx()
    ctx.abort_event.set()
    return ctx


def _permission_context(
    *,
    mode: PermissionMode = "default",
    allow: tuple[str, ...] = (),
    deny: tuple[str, ...] = (),
    ask: tuple[str, ...] = (),
    avoid_prompts: bool = False,
    bypass_available: bool = False,
) -> ToolPermissionContext:
    return ToolPermissionContext(
        mode=mode,
        always_allow_rules={"session": allow},
        always_deny_rules={"session": deny},
        always_ask_rules={"session": ask},
        should_avoid_permission_prompts=avoid_prompts,
        is_bypass_permissions_mode_available=bypass_available,
    )


async def _passthrough(
    _input: BaseModel,
    _ctx: ToolUseContext,
    _permission_context: ToolPermissionContext,
) -> PermissionPassthrough:
    return PermissionPassthrough(message="ask host")


async def _ask(
    _input: BaseModel,
    _ctx: ToolUseContext,
    _permission_context: ToolPermissionContext,
) -> PermissionAskDecision:
    return PermissionAskDecision(message="ask host")


async def _safety_ask(
    _input: BaseModel,
    _ctx: ToolUseContext,
    _permission_context: ToolPermissionContext,
) -> PermissionAskDecision:
    return PermissionAskDecision(
        message="safety",
        decision_reason=SafetyCheckPermissionDecisionReason(
            reason="sensitive",
            classifier_approvable=False,
        ),
    )


async def _raises_permission_error(
    _input: BaseModel,
    _ctx: ToolUseContext,
    _permission_context: ToolPermissionContext,
) -> PermissionResult:
    raise RuntimeError("permission bug")


def _build_tool(
    *,
    name: str = "Example",
    check_permissions: CheckPermissionsFn = _passthrough,
    requires_user_interaction: bool = False,
    describe_suffix: str = "",
) -> Tool:
    async def describe(
        input_: BaseModel,
        ctx: ToolDescriptionContext,
    ) -> str:
        parsed = input_
        assert isinstance(parsed, ExampleInput)
        assert ctx.permission_context.mode
        return f"describe {parsed.command}{describe_suffix}"

    return build_tool(
        ToolSpec(
            name=name,
            description="example",
            input_model=ExampleInput,
            call=_call,
            check_permissions=check_permissions,
            requires_user_interaction=requires_user_interaction,
            describe=describe,
        )
    )


@dataclass
class RecordingHandler:
    decision: PermissionDecision
    requests: list[PermissionRequest]

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        self.requests.append(request)
        return self.decision


def test_permission_rule_value_parser_matches_reference_format() -> None:
    parsed = permission_rule_value_from_string(r'Bash(python -c "print\(1\)")')

    assert parsed == PermissionRuleValue(
        tool_name="Bash",
        rule_content='python -c "print(1)"',
    )
    assert (
        permission_rule_value_to_string(parsed)
        == r'Bash(python -c "print\(1\)")'
    )
    assert permission_rule_value_from_string("Bash(*)") == PermissionRuleValue(
        tool_name="Bash"
    )


@pytest.mark.asyncio
async def test_can_use_tool_applies_deny_ask_allow_rules_in_reference_order() -> None:
    input_ = ExampleInput(command="run")
    tool = _build_tool()

    denied = await can_use_tool(
        tool=tool,
        input=input_,
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(deny=("Example",), allow=("Example",)),
    )
    assert isinstance(denied.decision, PermissionDenyDecision)
    assert isinstance(denied.decision.decision_reason, RulePermissionDecisionReason)

    asked = await can_use_tool(
        tool=tool,
        input=input_,
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(ask=("Example",), allow=("Example",)),
    )
    assert isinstance(asked.decision, PermissionAskDecision)

    allowed = await can_use_tool(
        tool=tool,
        input=input_,
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(allow=("Example",)),
    )
    assert isinstance(allowed.decision, PermissionAllowDecision)
    assert isinstance(allowed.decision.decision_reason, RulePermissionDecisionReason)


@pytest.mark.asyncio
async def test_can_use_tool_bypass_mode_allows_non_immune_asks() -> None:
    result = await can_use_tool(
        tool=_build_tool(check_permissions=_ask),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(mode="bypassPermissions"),
    )

    assert isinstance(result.decision, PermissionAllowDecision)
    assert result.decision.updated_input == {"command": "run"}


@pytest.mark.asyncio
async def test_can_use_tool_bypass_mode_preserves_safety_ask() -> None:
    result = await can_use_tool(
        tool=_build_tool(check_permissions=_safety_ask),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(mode="bypassPermissions"),
    )

    assert isinstance(result.decision, PermissionAskDecision)
    assert isinstance(
        result.decision.decision_reason,
        SafetyCheckPermissionDecisionReason,
    )


@pytest.mark.asyncio
async def test_tool_specific_permission_check_receives_permission_context() -> None:
    observed: list[PermissionMode] = []

    async def context_aware_check(
        _input: BaseModel,
        _ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionPassthrough:
        observed.append(permission_context.mode)
        assert permission_context.always_allow_rules["session"] == ("Example",)
        return PermissionPassthrough(message="ask host")

    result = await can_use_tool(
        tool=_build_tool(check_permissions=context_aware_check),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(
            mode="acceptEdits",
            allow=("Example",),
        ),
    )

    assert observed == ["acceptEdits"]
    assert isinstance(result.decision, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_permission_check_errors_fall_back_to_passthrough() -> None:
    default_result = await can_use_tool(
        tool=_build_tool(check_permissions=_raises_permission_error),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(),
    )
    bypass_result = await can_use_tool(
        tool=_build_tool(check_permissions=_raises_permission_error),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(mode="bypassPermissions"),
    )

    assert isinstance(default_result.decision, PermissionAskDecision)
    assert isinstance(bypass_result.decision, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_can_use_tool_dont_ask_and_avoid_prompts_deny_asks() -> None:
    dont_ask = await can_use_tool(
        tool=_build_tool(check_permissions=_ask),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(mode="dontAsk"),
    )
    avoid = await can_use_tool(
        tool=_build_tool(check_permissions=_ask),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(avoid_prompts=True),
    )

    assert isinstance(dont_ask.decision, PermissionDenyDecision)
    assert isinstance(dont_ask.decision.decision_reason, ModePermissionDecisionReason)
    assert isinstance(avoid.decision, PermissionDenyDecision)


@pytest.mark.asyncio
async def test_can_use_tool_handler_gets_dynamic_description() -> None:
    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "ok"}),
        requests=[],
    )
    tool = _build_tool(check_permissions=_ask, describe_suffix="!")

    result = await can_use_tool(
        tool=tool,
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(),
        handler=handler,
        tool_use_id="tu_1",
        is_non_interactive_session=True,
        tools=(tool,),
    )

    assert isinstance(result.decision, PermissionAllowDecision)
    assert handler.requests[0].description == "describe run!"
    assert handler.requests[0].tool_use_id == "tu_1"


@pytest.mark.asyncio
async def test_can_use_tool_aborts_before_permission_check_or_handler() -> None:
    calls: list[str] = []

    async def record_check(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionAskDecision:
        calls.append("check")
        return PermissionAskDecision(message="ask")

    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "ok"}),
        requests=[],
    )

    with pytest.raises(asyncio.CancelledError):
        await can_use_tool(
            tool=_build_tool(check_permissions=record_check),
            input=ExampleInput(command="run"),
            tool_use_context=_aborted_tool_ctx(),
            permission_context=_permission_context(),
            handler=handler,
        )

    assert calls == []
    assert handler.requests == []


@pytest.mark.asyncio
async def test_can_use_tool_aborts_before_handler_after_permission_check() -> None:
    ctx = _tool_ctx()

    async def aborting_check(
        _input: BaseModel,
        check_ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionAskDecision:
        check_ctx.abort_event.set()
        return PermissionAskDecision(message="ask")

    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "ok"}),
        requests=[],
    )

    with pytest.raises(asyncio.CancelledError):
        await can_use_tool(
            tool=_build_tool(check_permissions=aborting_check),
            input=ExampleInput(command="run"),
            tool_use_context=ctx,
            permission_context=_permission_context(),
            handler=handler,
        )

    assert handler.requests == []


@pytest.mark.asyncio
async def test_hook_allow_does_not_bypass_deny_rule() -> None:
    result = await resolve_hook_permission_decision(
        hook_permission_result=PermissionAllowDecision(),
        tool=_build_tool(),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(deny=("Example",)),
    )

    assert isinstance(result.decision, PermissionDenyDecision)


@pytest.mark.asyncio
async def test_hook_allow_with_ask_rule_re_enters_can_use_tool_handler() -> None:
    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "approved"}),
        requests=[],
    )

    result = await resolve_hook_permission_decision(
        hook_permission_result=PermissionAllowDecision(),
        tool=_build_tool(),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(ask=("Example",)),
        handler=handler,
    )

    assert isinstance(result.decision, PermissionAllowDecision)
    assert handler.requests[0].decision.behavior == "ask"


@pytest.mark.asyncio
async def test_interactive_hook_allow_requires_can_use_tool_unless_updated_input() -> None:
    tool = _build_tool(requires_user_interaction=True, check_permissions=_ask)
    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "handled"}),
        requests=[],
    )

    no_update = await resolve_hook_permission_decision(
        hook_permission_result=PermissionAllowDecision(),
        tool=tool,
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(),
        handler=handler,
    )
    with_update = await resolve_hook_permission_decision(
        hook_permission_result=PermissionAllowDecision(
            updated_input={"command": "from-hook"}
        ),
        tool=tool,
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(),
        handler=handler,
    )

    assert isinstance(no_update.decision, PermissionAllowDecision)
    assert len(handler.requests) == 1
    assert isinstance(with_update.input, ExampleInput)
    assert with_update.input.command == "from-hook"
    assert isinstance(with_update.decision, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_hook_ask_forces_handler_decision_with_updated_input() -> None:
    handler = RecordingHandler(
        decision=PermissionAllowDecision(updated_input={"command": "approved"}),
        requests=[],
    )

    result = await resolve_hook_permission_decision(
        hook_permission_result=PermissionAskDecision(
            message="hook asks",
            updated_input={"command": "hook-input"},
        ),
        tool=_build_tool(),
        input=ExampleInput(command="run"),
        tool_use_context=_tool_ctx(),
        permission_context=_permission_context(),
        handler=handler,
    )

    assert isinstance(result.decision, PermissionAllowDecision)
    assert isinstance(result.input, ExampleInput)
    assert result.input.command == "hook-input"
    assert handler.requests[0].decision.message == "hook asks"


def test_permission_request_message_uses_rule_reason() -> None:
    rule = RulePermissionDecisionReason(
        rule=PermissionRule(
            source="session",
            rule_behavior="ask",
            rule_value=PermissionRuleValue("Example"),
        )
    )

    assert "Permission rule" in create_permission_request_message("Example", rule)
