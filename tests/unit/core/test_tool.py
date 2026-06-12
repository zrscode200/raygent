from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from raygent_harness.core.permissions import empty_tool_permission_context
from raygent_harness.core.tool import (
    PermissionAsk,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    build_tool,
    find_tool_by_name,
)


class ExampleInput(BaseModel):
    command: str
    destructive: bool = False


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


async def _validate_ok(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> ValidationOk:
    return ValidationOk()


async def _validate_error(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> ValidationError:
    return ValidationError(message="invalid")


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )


def _example(input_: BaseModel) -> ExampleInput:
    assert isinstance(input_, ExampleInput)
    return input_


@pytest.mark.asyncio
async def test_build_tool_defaults_are_fail_closed_and_blocking() -> None:
    tool = build_tool(
        ToolSpec(
            name="example",
            description="example tool",
            input_model=ExampleInput,
            call=_call,
        )
    )
    input_ = ExampleInput(command="cat file")

    assert tool.aliases == ()
    assert tool.search_hint is None
    assert tool.is_enabled() is True
    assert tool.is_concurrency_safe(input_) is False
    assert tool.is_read_only(input_) is False
    assert tool.is_destructive(input_) is True
    assert tool.is_open_world(input_) is True
    assert tool.requires_user_interaction() is False
    assert tool.interrupt_behavior() == "block"
    assert tool.max_result_size_chars == 50_000
    assert await tool.prompt() == "example tool"
    assert isinstance(await tool.validate_input(input_, _ctx()), ValidationOk)

    permission = await tool.check_permissions(
        input_,
        _ctx(),
        empty_tool_permission_context(),
    )
    assert isinstance(permission, PermissionAsk)


def test_build_tool_accepts_input_sensitive_capability_predicates() -> None:
    tool = build_tool(
        ToolSpec(
            name="bash-ish",
            description="example tool",
            input_model=ExampleInput,
            call=_call,
            aliases=("shell",),
            search_hint="run commands",
            prompt="Full prompt text",
            is_enabled=lambda: False,
            is_concurrency_safe=lambda input_: _example(input_).command.startswith("cat "),
            is_read_only=lambda input_: _example(input_).command.startswith("cat "),
            is_destructive=lambda input_: _example(input_).destructive,
            is_open_world=lambda input_: "curl" in _example(input_).command,
            requires_user_interaction=lambda: True,
            interrupt_behavior=lambda: "cancel",
            max_result_size_chars=30_000,
        )
    )

    read_input = ExampleInput(command="cat file")
    write_input = ExampleInput(command="rm file", destructive=True)
    network_input = ExampleInput(command="curl https://example.com")

    assert tool.aliases == ("shell",)
    assert tool.search_hint == "run commands"
    assert tool.is_enabled() is False
    assert tool.is_concurrency_safe(read_input) is True
    assert tool.is_read_only(read_input) is True
    assert tool.is_concurrency_safe(write_input) is False
    assert tool.is_read_only(write_input) is False
    assert tool.is_destructive(write_input) is True
    assert tool.is_open_world(network_input) is True
    assert tool.requires_user_interaction() is True
    assert tool.interrupt_behavior() == "cancel"
    assert tool.max_result_size_chars == 30_000
    assert asyncio.run(tool.prompt()) == "Full prompt text"

    assert find_tool_by_name([tool], "bash-ish") is tool
    assert find_tool_by_name([tool], "shell") is tool
    assert find_tool_by_name([tool], "missing") is None


def test_build_tool_accepts_constant_capability_overrides() -> None:
    tool = build_tool(
        ToolSpec(
            name="read",
            description="example tool",
            input_model=ExampleInput,
            call=_call,
            aliases=("view",),
            search_hint="inspect files",
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            is_enabled=True,
            requires_user_interaction=False,
            interrupt_behavior="block",
            max_result_size_chars=float("inf"),
        )
    )
    input_ = ExampleInput(command="anything")

    assert tool.aliases == ("view",)
    assert tool.search_hint == "inspect files"
    assert tool.is_enabled() is True
    assert tool.is_concurrency_safe(input_) is True
    assert tool.is_read_only(input_) is True
    assert tool.is_destructive(input_) is False
    assert tool.is_open_world(input_) is False
    assert tool.requires_user_interaction() is False
    assert tool.interrupt_behavior() == "block"
    assert tool.max_result_size_chars == float("inf")


@pytest.mark.asyncio
async def test_build_tool_awaits_async_validation_success() -> None:
    tool = build_tool(
        ToolSpec(
            name="validated",
            description="example tool",
            input_model=ExampleInput,
            call=_call,
            validate_input=_validate_ok,
        )
    )

    result = await tool.validate_input(ExampleInput(command="cat file"), _ctx())
    assert isinstance(result, ValidationOk)


@pytest.mark.asyncio
async def test_build_tool_awaits_async_validation_error() -> None:
    tool = build_tool(
        ToolSpec(
            name="validated",
            description="example tool",
            input_model=ExampleInput,
            call=_call,
            validate_input=_validate_error,
        )
    )

    result = await tool.validate_input(ExampleInput(command="rm file"), _ctx())
    assert isinstance(result, ValidationError)
    assert result.message == "invalid"
