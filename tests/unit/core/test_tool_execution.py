from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.permissions import (
    HookPermissionDecisionReason,
    PermissionAllowDecision,
    PermissionDenyDecision,
    ToolPermissionContext,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallError,
    ToolCallEvent,
    ToolProgress,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    build_tool,
)
from raygent_harness.core.tool_execution import (
    ToolExecutionProgress,
    ToolExecutionResult,
    run_tool_use,
)
from raygent_harness.core.tool_hooks import PreToolUseContext, PreToolUseResult


class ExampleInput(BaseModel):
    command: str
    flag: bool = False


def _ctx(*, aborted: bool = False) -> ToolUseContext:
    abort_event = asyncio.Event()
    if aborted:
        abort_event.set()
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=abort_event,
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _deps(
    *,
    permission_context: ToolPermissionContext | None = None,
    pre_hooks: list[Any] | None = None,
) -> QueryDeps:
    return QueryDeps(
        task_store=AppStateStore(),
        permission_context=permission_context or ToolPermissionContext(mode="bypassPermissions"),
        pre_tool_use_hooks=pre_hooks or [],
    )


def _tool(
    *,
    name: str = "Example",
    aliases: tuple[str, ...] = (),
    validate_error: str | None = None,
    validate_raises: Exception | None = None,
    events: list[ToolCallEvent] | None = None,
    raise_error: BaseException | None = None,
    seen_inputs: list[ExampleInput] | None = None,
    check_permissions: Any | None = None,
) -> Tool:
    async def validate(input_: BaseModel, _ctx: ToolUseContext) -> ValidationOk | ValidationError:
        if validate_raises is not None:
            raise validate_raises
        if validate_error is not None:
            return ValidationError(message=validate_error)
        return ValidationOk()

    async def call(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        assert isinstance(input_, ExampleInput)
        if seen_inputs is not None:
            seen_inputs.append(input_)
        if raise_error is not None:
            raise raise_error
        for event in events or [ToolResult(content="ok")]:
            yield event

    return build_tool(
        ToolSpec(
            name=name,
            aliases=aliases,
            description=f"{name} tool",
            input_model=ExampleInput,
            call=call,
            validate_input=validate,
            check_permissions=check_permissions,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


async def _collect(
    tool_use: ToolUseBlock,
    tools: tuple[Tool, ...],
    deps: QueryDeps,
    ctx: ToolUseContext | None = None,
) -> list[ToolExecutionProgress | ToolExecutionResult]:
    return [
        event
        async for event in run_tool_use(
            tool_use=tool_use,
            assistant_message={"role": "assistant", "content": []},
            tools=tools,
            deps=deps,
            ctx=ctx or _ctx(),
        )
    ]


def _tool_use(
    *,
    name: str = "Example",
    id_: str = "toolu_1",
    input_: object = {"command": "echo hi"},
) -> ToolUseBlock:
    return ToolUseBlock(id=id_, name=name, input=input_, index=0)


def _first_block(result: ToolExecutionResult) -> dict[str, Any]:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    return block


def _content(result: ToolExecutionResult) -> str | list[dict[str, Any]]:
    return _first_block(result)["content"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_unknown_tool_returns_model_visible_error_result() -> None:
    events = await _collect(_tool_use(name="Missing"), (), _deps())

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    block = _first_block(result)
    assert block["tool_use_id"] == "toolu_1"
    assert block["is_error"] is True
    assert "No such tool available: Missing" in str(block["content"])


@pytest.mark.asyncio
async def test_input_parse_error_returns_tool_result_error() -> None:
    events = await _collect(_tool_use(input_={"flag": True}), (_tool(),), _deps())

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert "InputValidationError" in str(_content(result))
    assert _first_block(result)["is_error"] is True


@pytest.mark.asyncio
async def test_validate_input_error_returns_tool_result_error() -> None:
    events = await _collect(
        _tool_use(),
        (_tool(validate_error="command rejected"),),
        _deps(),
    )

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert "command rejected" in str(_content(result))
    assert _first_block(result)["is_error"] is True


@pytest.mark.asyncio
async def test_validate_input_exception_returns_tool_result_error() -> None:
    events = await _collect(
        _tool_use(),
        (_tool(validate_raises=RuntimeError("validator exploded")),),
        _deps(),
    )

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert "validator exploded" in str(_content(result))
    assert _first_block(result)["is_error"] is True


@pytest.mark.asyncio
async def test_permission_denial_returns_error_and_tombstone() -> None:
    async def deny(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionDenyDecision:
        return PermissionDenyDecision(
            message="denied by policy",
            decision_reason=HookPermissionDecisionReason(hook_name="test"),
        )

    events = await _collect(_tool_use(), (_tool(check_permissions=deny),), _deps())

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert _first_block(result)["is_error"] is True
    assert _content(result) == "denied by policy"
    assert len(result.permission_denials) == 1
    denial = result.permission_denials[0]
    assert denial.tool_use_id == "toolu_1"
    assert denial.tool_name == "Example"
    assert denial.tool_input == {"command": "echo hi", "flag": False}
    assert denial.reason == "denied by policy"


@pytest.mark.asyncio
async def test_permission_updated_input_reaches_tool_call() -> None:
    seen: list[ExampleInput] = []

    async def allow_updated(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionAllowDecision:
        return PermissionAllowDecision(updated_input={"command": "rewritten", "flag": True})

    events = await _collect(
        _tool_use(),
        (_tool(seen_inputs=seen, check_permissions=allow_updated),),
        _deps(),
    )

    assert isinstance(events[-1], ToolExecutionResult)
    assert seen == [ExampleInput(command="rewritten", flag=True)]
    assert _content(events[-1]) == "ok"


@pytest.mark.asyncio
async def test_invalid_permission_updated_input_returns_tool_result_error() -> None:
    async def allow_invalid_update(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionAllowDecision:
        return PermissionAllowDecision(updated_input={"flag": True})

    events = await _collect(
        _tool_use(),
        (_tool(check_permissions=allow_invalid_update),),
        _deps(),
    )

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert "InputValidationError" in str(_content(result))
    assert _first_block(result)["is_error"] is True


@pytest.mark.asyncio
async def test_success_and_progress_events_are_preserved() -> None:
    events = await _collect(
        _tool_use(),
        (
            _tool(
                events=[
                    ToolProgress(message="halfway", data={"pct": 50}),
                    ToolResult(content=[{"type": "text", "text": "done"}]),
                ]
            ),
        ),
        _deps(),
    )

    assert isinstance(events[0], ToolExecutionProgress)
    assert events[0].message == "halfway"
    assert events[0].data == {"pct": 50}
    assert isinstance(events[1], ToolExecutionResult)
    assert _content(events[1]) == [{"type": "text", "text": "done"}]
    assert "is_error" not in _first_block(events[1])


@pytest.mark.asyncio
async def test_tool_call_error_and_exception_become_error_results() -> None:
    call_error_events = await _collect(
        _tool_use(),
        (_tool(events=[ToolCallError(message="recoverable boom")]),),
        _deps(),
    )
    exception_events = await _collect(
        _tool_use(),
        (_tool(raise_error=RuntimeError("unexpected boom")),),
        _deps(),
    )

    for events, expected in (
        (call_error_events, "recoverable boom"),
        (exception_events, "unexpected boom"),
    ):
        result = events[0]
        assert isinstance(result, ToolExecutionResult)
        assert _first_block(result)["is_error"] is True
        assert expected in str(_content(result))


@pytest.mark.asyncio
async def test_cancelled_error_propagates() -> None:
    with pytest.raises(asyncio.CancelledError):
        await _collect(
            _tool_use(),
            (_tool(raise_error=asyncio.CancelledError()),),
            _deps(),
        )


@pytest.mark.asyncio
async def test_hook_allow_cannot_bypass_deny_rule() -> None:
    async def allow_hook(_context: PreToolUseContext) -> PreToolUseResult:
        return PreToolUseResult(
            permission_result=PermissionAllowDecision(),
        )

    deps = _deps(
        permission_context=ToolPermissionContext(
            mode="bypassPermissions",
            always_deny_rules={"session": ("Danger",)},
        ),
        pre_hooks=[allow_hook],
    )
    events = await _collect(
        _tool_use(name="Danger"),
        (_tool(name="Danger"),),
        deps,
    )

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert _first_block(result)["is_error"] is True
    assert "denied" in str(_content(result))
    assert len(result.permission_denials) == 1


@pytest.mark.asyncio
async def test_raising_pre_tool_hook_stops_before_tool_call() -> None:
    seen: list[ExampleInput] = []

    async def raising_hook(_context: PreToolUseContext) -> PreToolUseResult:
        raise RuntimeError("hook exploded")

    events = await _collect(
        _tool_use(),
        (_tool(seen_inputs=seen),),
        _deps(pre_hooks=[raising_hook]),
    )

    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert _first_block(result)["is_error"] is True
    assert "PreToolUse hook error: hook exploded" in str(_content(result))
    assert result.should_prevent_continuation is True
    assert seen == []


@pytest.mark.asyncio
async def test_hook_updated_input_reaches_normal_permission_and_tool_call() -> None:
    seen: list[ExampleInput] = []

    async def update_hook(_context: PreToolUseContext) -> PreToolUseResult:
        return PreToolUseResult(updated_input={"command": "hooked", "flag": True})

    events = await _collect(
        _tool_use(),
        (_tool(seen_inputs=seen),),
        _deps(pre_hooks=[update_hook]),
    )

    assert isinstance(events[-1], ToolExecutionResult)
    assert seen == [ExampleInput(command="hooked", flag=True)]
    assert _content(events[-1]) == "ok"
