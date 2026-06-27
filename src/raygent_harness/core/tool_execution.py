"""Single tool-use execution lifecycle.

This is the one-tool execution substrate: lookup, parse, validate, PreToolUse
hooks, permission resolution, call, progress, and model-visible result mapping.
Batch scheduling and query-loop integration are intentionally left to the next
chunk.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.observability import KernelEventContext, redacted_payload
from raygent_harness.core.state import PermissionDenial
from raygent_harness.core.tool import (
    Tool,
    ToolContextModifier,
    ToolProgress,
    ToolResult,
    find_tool_by_name,
    tool_visible_to_model,
)
from raygent_harness.core.tool import (
    ValidationError as ToolValidationError,
)
from raygent_harness.core.tool_hooks import (
    PostToolUseContext,
    PostToolUseFailureContext,
    PreToolUseContext,
    PreToolUseHookOutcome,
)

if TYPE_CHECKING:
    from raygent_harness.core.deps import QueryDeps
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.tool import ToolUseContext


ToolExecutionStatus = Literal[
    "completed",
    "failed",
    "denied",
    "aborted",
    "interrupted",
    "validation_error",
    "unknown_tool",
    "error",
]


@dataclass(frozen=True)
class ToolExecutionProgress:
    type: Literal["progress"] = "progress"
    tool_use_id: str = ""
    tool_name: str = ""
    message: str = ""
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    type: Literal["result"] = "result"
    message: MessageParam = field(
        default_factory=lambda: cast(
            "MessageParam",
            {"role": "user", "content": ""},
        )
    )
    tool_use_id: str = ""
    tool_name: str = ""
    status: ToolExecutionStatus = "completed"
    error_type: str | None = None
    permission_denials: tuple[PermissionDenial, ...] = ()
    pre_messages: tuple[MessageParam, ...] = ()
    """Messages emitted by PreToolUse hooks before the tool_result message."""

    additional_messages: tuple[MessageParam, ...] = ()
    """Tool-provided `newMessages` emitted after the tool_result message."""

    discovered_tool_names: tuple[str, ...] = ()
    """Deferred tools selected by the primary ToolSearch result."""

    context_modifier: ToolContextModifier | None = None
    should_prevent_continuation: bool = False
    prevent_reason: str | None = None


ToolExecutionEvent = ToolExecutionProgress | ToolExecutionResult


def deferred_tool_not_selected_result(
    *,
    tool_use_id: str,
    tool_name: str,
) -> ToolExecutionResult:
    """Model-visible error for raw calls to hidden deferred tools."""

    return _result(
        tool_use_id,
        _tool_error(
            f"Error: Tool {tool_name} is deferred and must be selected with "
            "ToolSearch before it can be used."
        ),
        is_error=True,
        tool_name=tool_name,
        status="unknown_tool",
    )


async def run_tool_use(
    *,
    tool_use: ToolUseBlock,
    assistant_message: MessageParam,
    tools: Sequence[Tool],
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> AsyncIterator[ToolExecutionEvent]:
    """Execute one normalized `tool_use` block.

    Unknown tools, parse failures, validation failures, permission denials, and
    execution exceptions are returned as model-visible `tool_result` errors.
    Cooperative cancellation (`CancelledError` or an already-set abort event)
    is re-raised so the query loop can produce the correct abort terminal.
    """
    tool = find_tool_by_name(tools, tool_use.name)
    if tool is None:
        _emit_tool_event(
            deps,
            ctx,
            "tool.call.failed",
            tool_use_id=tool_use.id,
            data=_tool_call_payload(
                tool_use=tool_use,
                stage="lookup",
                failure_reason="unknown_tool",
            ),
        )
        yield _result(
            tool_use.id,
            _tool_error(f"Error: No such tool available: {tool_use.name}"),
            is_error=True,
            tool_name=tool_use.name,
            status="unknown_tool",
        )
        return
    if not tool_visible_to_model(tool, ctx.discovered_tool_names):
        _emit_tool_event(
            deps,
            ctx,
            "tool.call.failed",
            tool_use_id=tool_use.id,
            data=_tool_call_payload(
                tool_use=tool_use,
                tool=tool,
                stage="lookup",
                failure_reason="deferred_tool_not_selected",
            ),
        )
        yield deferred_tool_not_selected_result(
            tool_use_id=tool_use.id,
            tool_name=tool_use.name,
        )
        return
    execution_ctx = replace(
        ctx,
        tool_use_id=tool_use.id,
        current_assistant_message=assistant_message,
    )
    _emit_tool_event(
        deps,
        execution_ctx,
        "tool.call.started",
        tool_use_id=tool_use.id,
        data=_tool_call_payload(tool_use=tool_use, tool=tool),
    )

    parsed = _parse_tool_input(tool, tool_use)
    if isinstance(parsed, ToolExecutionResult):
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=parsed,
            failure_reason="input_parse_failed",
        )
        yield parsed
        await _run_failure_hooks(tool, tool_use, assistant_message, deps, execution_ctx, parsed)
        return

    try:
        validation = await tool.validate_input(parsed, execution_ctx)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result = _result(
            tool_use.id,
            _tool_error(f"Error validating tool ({tool.name}): {exc}"),
            is_error=True,
            tool_name=tool.name,
            status="validation_error",
            error_type=type(exc).__name__,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="validation_exception",
            error_type=type(exc).__name__,
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return
    if isinstance(validation, ToolValidationError):
        result = _result(
            tool_use.id,
            _tool_error(validation.message),
            is_error=True,
            tool_name=tool.name,
            status="validation_error",
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="validation_rejected",
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return

    try:
        hook_outcome = await _run_pre_tool_use_hooks(
            tool=tool,
            tool_use=tool_use,
            assistant_message=assistant_message,
            deps=deps,
            ctx=execution_ctx,
            parsed=parsed,
        )
    except PydanticValidationError as exc:
        result = _result(
            tool_use.id,
            _tool_error(f"InputValidationError: {exc}"),
            is_error=True,
            tool_name=tool.name,
            status="validation_error",
            error_type=type(exc).__name__,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="pre_tool_use_invalid_updated_input",
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return
    parsed = hook_outcome.input

    if hook_outcome.stop:
        reason = hook_outcome.stop_reason or "Execution stopped by PreToolUse hook"
        result = ToolExecutionResult(
            message=_tool_result_message(tool_use.id, _tool_error(reason), is_error=True),
            tool_use_id=tool_use.id,
            tool_name=tool.name,
            status="interrupted",
            pre_messages=hook_outcome.additional_messages,
            should_prevent_continuation=hook_outcome.should_prevent_continuation,
            prevent_reason=reason,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="pre_tool_use_stopped",
            prevent_continuation=hook_outcome.should_prevent_continuation,
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return

    _emit_tool_event(
        deps,
        execution_ctx,
        "permission.requested",
        tool_use_id=tool_use.id,
        data={
            **_tool_call_payload(tool_use=tool_use, tool=tool, parsed=parsed),
            "permission_mode": execution_ctx.permission_context.mode,
            "hook_permission_result": _permission_result_kind(
                hook_outcome.permission_result
            ),
        },
    )
    try:
        resolved = await deps.resolve_hook_tool_permission(
            hook_permission_result=hook_outcome.permission_result,
            tool=tool,
            input=parsed,
            tool_use_context=execution_ctx,
            tool_use_id=tool_use.id,
            tools=tools,
        )
    except PydanticValidationError as exc:
        result = _result(
            tool_use.id,
            _tool_error(f"InputValidationError: {exc}"),
            is_error=True,
            tool_name=tool.name,
            status="validation_error",
            error_type=type(exc).__name__,
            pre_messages=hook_outcome.additional_messages,
            should_prevent_continuation=hook_outcome.should_prevent_continuation,
            prevent_reason=hook_outcome.stop_reason,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "permission.decided",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="permission_invalid_updated_input",
            decision_behavior="error",
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="permission_invalid_updated_input",
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return
    parsed = resolved.input
    _emit_tool_event(
        deps,
        execution_ctx,
        "permission.decided",
        tool_use_id=tool_use.id,
        data={
            **_tool_call_payload(tool_use=tool_use, tool=tool, parsed=parsed),
            "decision_behavior": resolved.decision.behavior,
            "decision_reason_type": _permission_reason_type(resolved.decision),
            "updated_input": getattr(resolved.decision, "updated_input", None)
            is not None,
        },
    )

    if resolved.decision.behavior != "allow":
        denial = _permission_denial(tool_use.id, tool, parsed, resolved.decision.message)
        result = ToolExecutionResult(
            message=_tool_result_message(
                tool_use.id,
                resolved.decision.message,
                is_error=True,
            ),
            tool_use_id=tool_use.id,
            tool_name=tool.name,
            status="denied",
            permission_denials=(denial,),
            pre_messages=hook_outcome.additional_messages,
            should_prevent_continuation=hook_outcome.should_prevent_continuation,
            prevent_reason=hook_outcome.stop_reason,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="permission_denied",
            decision_behavior=resolved.decision.behavior,
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )
        return

    if resolved.decision.updated_input is not None:
        try:
            parsed = tool.input_model.model_validate(dict(resolved.decision.updated_input))
        except PydanticValidationError as exc:
            result = _result(
                tool_use.id,
                _tool_error(f"InputValidationError: {exc}"),
                is_error=True,
                tool_name=tool.name,
                status="validation_error",
                error_type=type(exc).__name__,
                pre_messages=hook_outcome.additional_messages,
                should_prevent_continuation=hook_outcome.should_prevent_continuation,
                prevent_reason=hook_outcome.stop_reason,
            )
            _emit_tool_result_event(
                deps,
                execution_ctx,
                "tool.call.failed",
                tool_use=tool_use,
                tool=tool,
                result=result,
                failure_reason="permission_updated_input_invalid",
            )
            yield result
            await _run_failure_hooks(
                tool,
                tool_use,
                assistant_message,
                deps,
                execution_ctx,
                result,
                parsed,
            )
            return

    try:
        saw_result = False
        async for event in tool.call(parsed, execution_ctx):
            if isinstance(event, ToolProgress):
                progress_data = event.data or {}
                _emit_tool_event(
                    deps,
                    execution_ctx,
                    "tool.call.progress",
                    tool_use_id=tool_use.id,
                    data={
                        **_tool_call_payload(
                            tool_use=tool_use,
                            tool=tool,
                            parsed=parsed,
                        ),
                        "message_char_count": len(event.message),
                        "data_present": bool(progress_data),
                        "data_key_count": len(progress_data),
                    },
                )
                yield ToolExecutionProgress(
                    tool_use_id=tool_use.id,
                    tool_name=tool.name,
                    message=event.message,
                    data=event.data,
                )
                continue
            if isinstance(event, ToolResult):
                saw_result = True
                additional_messages = tuple(
                    cast("MessageParam", message)
                    for message in event.additional_messages
                )
                result = ToolExecutionResult(
                    message=_tool_result_message(
                        tool_use.id,
                        event.content,
                        is_error=event.is_error,
                    ),
                    tool_use_id=tool_use.id,
                    tool_name=tool.name,
                    status="failed" if event.is_error else "completed",
                    pre_messages=hook_outcome.additional_messages,
                    additional_messages=additional_messages,
                    discovered_tool_names=_trusted_discovered_tool_names(
                        tool.name,
                        event,
                    ),
                    context_modifier=event.context_modifier,
                    should_prevent_continuation=hook_outcome.should_prevent_continuation,
                    prevent_reason=hook_outcome.stop_reason,
                )
                _emit_tool_result_event(
                    deps,
                    execution_ctx,
                    "tool.call.failed" if event.is_error else "tool.call.completed",
                    tool_use=tool_use,
                    tool=tool,
                    result=result,
                    failure_reason="tool_returned_error" if event.is_error else None,
                )
                yield result
                await _run_post_tool_use_hooks(
                    tool,
                    tool_use,
                    assistant_message,
                    deps,
                    execution_ctx,
                    parsed,
                    result.message,
                )
                return
            saw_result = True
            result = _result(
                tool_use.id,
                _tool_error(f"Error calling tool ({tool.name}): {event.message}"),
                is_error=True,
                tool_name=tool.name,
                status="failed",
                error_type=type(event).__name__,
                pre_messages=hook_outcome.additional_messages,
                should_prevent_continuation=hook_outcome.should_prevent_continuation,
                prevent_reason=hook_outcome.stop_reason,
            )
            _emit_tool_result_event(
                deps,
                execution_ctx,
                "tool.call.failed",
                tool_use=tool_use,
                tool=tool,
                result=result,
                failure_reason="tool_call_error_event",
            )
            yield result
            await _run_failure_hooks(
                tool,
                tool_use,
                assistant_message,
                deps,
                execution_ctx,
                result,
                parsed,
            )
            return
        if not saw_result:
            result = _result(
                tool_use.id,
                _tool_error(f"Error calling tool ({tool.name}): tool produced no result"),
                is_error=True,
                tool_name=tool.name,
                status="failed",
                pre_messages=hook_outcome.additional_messages,
                should_prevent_continuation=hook_outcome.should_prevent_continuation,
                prevent_reason=hook_outcome.stop_reason,
            )
            _emit_tool_result_event(
                deps,
                execution_ctx,
                "tool.call.failed",
                tool_use=tool_use,
                tool=tool,
                result=result,
                failure_reason="tool_produced_no_result",
            )
            yield result
            await _run_failure_hooks(
                tool,
                tool_use,
                assistant_message,
                deps,
                execution_ctx,
                result,
                parsed,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result = _result(
            tool_use.id,
            _tool_error(f"Error calling tool ({tool.name}): {exc}"),
            is_error=True,
            tool_name=tool.name,
            status="error",
            error_type=type(exc).__name__,
            pre_messages=hook_outcome.additional_messages,
            should_prevent_continuation=hook_outcome.should_prevent_continuation,
            prevent_reason=hook_outcome.stop_reason,
        )
        _emit_tool_result_event(
            deps,
            execution_ctx,
            "tool.call.failed",
            tool_use=tool_use,
            tool=tool,
            result=result,
            failure_reason="tool_call_exception",
            error_type=type(exc).__name__,
        )
        yield result
        await _run_failure_hooks(
            tool,
            tool_use,
            assistant_message,
            deps,
            execution_ctx,
            result,
            parsed,
        )


def _parse_tool_input(tool: Tool, tool_use: ToolUseBlock) -> BaseModel | ToolExecutionResult:
    try:
        return tool.input_model.model_validate(tool_use.input)
    except PydanticValidationError as exc:
        return _result(
            tool_use.id,
            _tool_error(f"InputValidationError: {exc}"),
            is_error=True,
            tool_name=tool.name,
            status="validation_error",
            error_type=type(exc).__name__,
        )


async def _run_pre_tool_use_hooks(
    *,
    tool: Tool,
    tool_use: ToolUseBlock,
    assistant_message: MessageParam,
    deps: QueryDeps,
    ctx: ToolUseContext,
    parsed: BaseModel,
) -> PreToolUseHookOutcome:
    current_input = parsed
    permission_result = None
    additional_messages: list[MessageParam] = []
    should_prevent_continuation = False
    stop_reason: str | None = None
    stop = False
    errors: list[str] = []

    for index, hook in enumerate(deps.pre_tool_use_hooks):
        started_at = time.time()
        _emit_hook_event(
            deps,
            ctx,
            "hook.pre_tool_use.started",
            tool_use=tool_use,
            tool=tool,
            index=index,
            data={"input_field_count": len(current_input.model_dump())},
        )
        try:
            result = await hook(
                PreToolUseContext(
                    tool=tool,
                    tool_use=tool_use,
                    input=current_input,
                    tool_use_context=ctx,
                    assistant_message=assistant_message,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            errors.append(message)
            _emit_hook_event(
                deps,
                ctx,
                "hook.pre_tool_use.failed",
                tool_use=tool_use,
                tool=tool,
                index=index,
                started_at=started_at,
                data={
                    "error_type": type(exc).__name__,
                    "message_char_count": len(message),
                },
            )
            return PreToolUseHookOutcome(
                input=current_input,
                should_prevent_continuation=True,
                stop_reason=f"PreToolUse hook error: {message}",
                stop=True,
                errors=tuple(errors),
            )
        if result is None:
            _emit_hook_event(
                deps,
                ctx,
                "hook.pre_tool_use.completed",
                tool_use=tool_use,
                tool=tool,
                index=index,
                started_at=started_at,
                data={"returned_result": False},
            )
            continue
        if result.updated_input is not None:
            current_input = tool.input_model.model_validate(dict(result.updated_input))
        if result.permission_result is not None:
            permission_result = result.permission_result
        additional_messages.extend(result.additional_messages)
        should_prevent_continuation = (
            should_prevent_continuation or result.should_prevent_continuation
        )
        stop_reason = result.stop_reason or stop_reason
        stop = stop or result.stop
        _emit_hook_event(
            deps,
            ctx,
            "hook.pre_tool_use.completed",
            tool_use=tool_use,
            tool=tool,
            index=index,
            started_at=started_at,
            data={
                "returned_result": True,
                "updated_input": result.updated_input is not None,
                "permission_result": _permission_result_kind(result.permission_result),
                "additional_message_count": len(result.additional_messages),
                "should_prevent_continuation": result.should_prevent_continuation,
                "stop": result.stop,
                "stop_reason_char_count": (
                    len(result.stop_reason) if result.stop_reason else 0
                ),
            },
        )

    return PreToolUseHookOutcome(
        input=current_input,
        permission_result=permission_result,
        additional_messages=tuple(additional_messages),
        should_prevent_continuation=should_prevent_continuation,
        stop_reason=stop_reason,
        stop=stop,
        errors=tuple(errors),
    )


async def _run_post_tool_use_hooks(
    tool: Tool,
    tool_use: ToolUseBlock,
    assistant_message: MessageParam,
    deps: QueryDeps,
    ctx: ToolUseContext,
    parsed: BaseModel,
    result_message: MessageParam,
) -> None:
    for index, hook in enumerate(deps.post_tool_use_hooks):
        started_at = time.time()
        _emit_hook_event(
            deps,
            ctx,
            "hook.post_tool_use.started",
            tool_use=tool_use,
            tool=tool,
            index=index,
        )
        try:
            await hook(
                PostToolUseContext(
                    tool=tool,
                    tool_use=tool_use,
                    input=parsed,
                    tool_use_context=ctx,
                    assistant_message=assistant_message,
                    result_message=result_message,
                )
            )
        except Exception as exc:
            _emit_hook_event(
                deps,
                ctx,
                "hook.post_tool_use.failed",
                tool_use=tool_use,
                tool=tool,
                index=index,
                started_at=started_at,
                data={"error_type": type(exc).__name__},
            )
            continue
        _emit_hook_event(
            deps,
            ctx,
            "hook.post_tool_use.completed",
            tool_use=tool_use,
            tool=tool,
            index=index,
            started_at=started_at,
        )


async def _run_failure_hooks(
    tool: Tool | None,
    tool_use: ToolUseBlock,
    assistant_message: MessageParam,
    deps: QueryDeps,
    ctx: ToolUseContext,
    result: ToolExecutionResult,
    parsed: BaseModel | None = None,
) -> None:
    message = _tool_result_text(result.message)
    for index, hook in enumerate(deps.post_tool_use_failure_hooks):
        started_at = time.time()
        _emit_hook_event(
            deps,
            ctx,
            "hook.post_tool_use_failure.started",
            tool_use=tool_use,
            tool=tool,
            index=index,
            data={"error_message_char_count": len(message)},
        )
        try:
            await hook(
                PostToolUseFailureContext(
                    tool=tool,
                    tool_use=tool_use,
                    input=parsed,
                    tool_use_context=ctx,
                    assistant_message=assistant_message,
                    error_message=message,
                )
            )
        except Exception as exc:
            _emit_hook_event(
                deps,
                ctx,
                "hook.post_tool_use_failure.failed",
                tool_use=tool_use,
                tool=tool,
                index=index,
                started_at=started_at,
                data={"error_type": type(exc).__name__},
            )
            continue
        _emit_hook_event(
            deps,
            ctx,
            "hook.post_tool_use_failure.completed",
            tool_use=tool_use,
            tool=tool,
            index=index,
            started_at=started_at,
        )


def _result(
    tool_use_id: str,
    content: str | list[dict[str, Any]],
    *,
    is_error: bool,
    tool_name: str = "",
    status: ToolExecutionStatus | None = None,
    error_type: str | None = None,
    pre_messages: tuple[MessageParam, ...] = (),
    additional_messages: tuple[MessageParam, ...] = (),
    should_prevent_continuation: bool = False,
    prevent_reason: str | None = None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        message=_tool_result_message(tool_use_id, content, is_error=is_error),
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        status=status or ("failed" if is_error else "completed"),
        error_type=error_type,
        pre_messages=pre_messages,
        additional_messages=additional_messages,
        should_prevent_continuation=should_prevent_continuation,
        prevent_reason=prevent_reason,
    )


def _trusted_discovered_tool_names(
    tool_name: str,
    result: ToolResult,
) -> tuple[str, ...]:
    if tool_name != "ToolSearch" or result.is_error:
        return ()
    return tuple(name for name in result.discovered_tool_names if name)


def _tool_result_message(
    tool_use_id: str,
    content: str | list[dict[str, Any]],
    *,
    is_error: bool,
) -> MessageParam:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return cast("MessageParam", {"role": "user", "content": [block]})


def _tool_error(message: str) -> str:
    return f"<tool_use_error>{message}</tool_use_error>"


def _permission_denial(
    tool_use_id: str,
    tool: Tool,
    input: BaseModel,
    reason: str,
) -> PermissionDenial:
    return PermissionDenial(
        tool_use_id=tool_use_id,
        tool_name=tool.name,
        tool_input=dict(input.model_dump()),
        reason=reason,
    )


def _tool_result_text(message: MessageParam) -> str:
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return str(content)
    first = content[0]
    return str(first.get("content", ""))


def _tool_event_context(
    ctx: ToolUseContext,
    tool_use_id: str,
) -> KernelEventContext:
    base = ctx.observability_context or KernelEventContext(
        session_id=ctx.session_id,
        agent_id=ctx.agent_id,
    )
    return base.for_child_span(f"tool:{tool_use_id}", source="tool")


def _emit_tool_event(
    deps: QueryDeps,
    ctx: ToolUseContext,
    event_type: str,
    *,
    tool_use_id: str,
    data: dict[str, object],
) -> None:
    deps.observability.emit(
        event_type,
        context=_tool_event_context(ctx, tool_use_id),
        data=data,
    )


def _emit_tool_result_event(
    deps: QueryDeps,
    ctx: ToolUseContext,
    event_type: str,
    *,
    tool_use: ToolUseBlock,
    tool: Tool,
    result: ToolExecutionResult,
    failure_reason: str | None = None,
    error_type: str | None = None,
    decision_behavior: str | None = None,
    prevent_continuation: bool | None = None,
) -> None:
    payload = {
        **_tool_call_payload(tool_use=tool_use, tool=tool),
        **_tool_result_payload(result),
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    if error_type is not None:
        payload["error_type"] = error_type
    if decision_behavior is not None:
        payload["decision_behavior"] = decision_behavior
    if prevent_continuation is not None:
        payload["should_prevent_continuation"] = prevent_continuation
    _emit_tool_event(
        deps,
        ctx,
        event_type,
        tool_use_id=tool_use.id,
        data=payload,
    )


def _emit_hook_event(
    deps: QueryDeps,
    ctx: ToolUseContext,
    event_type: str,
    *,
    tool_use: ToolUseBlock,
    tool: Tool | None,
    index: int,
    started_at: float | None = None,
    data: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "tool_use_id": tool_use.id,
        "tool_name": tool.name if tool is not None else tool_use.name,
        "hook_index": index,
    }
    if started_at is not None:
        payload["duration_ms"] = int((time.time() - started_at) * 1000)
    if data:
        payload.update(data)
    _emit_tool_event(
        deps,
        ctx,
        event_type,
        tool_use_id=tool_use.id,
        data=payload,
    )


def _tool_call_payload(
    *,
    tool_use: ToolUseBlock,
    tool: Tool | None = None,
    parsed: BaseModel | None = None,
    stage: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    input_shape = _tool_input_shape(tool_use.input)
    payload: dict[str, object] = {
        "tool_use_id": tool_use.id,
        "tool_name": tool.name if tool is not None else tool_use.name,
        "tool_found": tool is not None,
        "tool_index": tool_use.index,
        "input_type": input_shape["input_type"],
        "input_key_count": input_shape["input_key_count"],
        "input": redacted_payload("tool_input_redacted"),
    }
    if parsed is not None:
        payload["parsed_field_count"] = len(parsed.model_dump())
    if stage is not None:
        payload["stage"] = stage
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    return payload


def _tool_input_shape(input_: object) -> dict[str, object]:
    if isinstance(input_, Mapping):
        mapping = cast(Mapping[object, object], input_)
        return {
            "input_type": "object",
            "input_key_count": len(mapping),
        }
    if isinstance(input_, list):
        items = cast(list[object], input_)
        return {
            "input_type": "array",
            "input_key_count": None,
            "input_item_count": len(items),
        }
    return {
        "input_type": type(input_).__name__,
        "input_key_count": None,
    }


def _tool_result_payload(result: ToolExecutionResult) -> dict[str, object]:
    text = _tool_result_text(result.message)
    return {
        "is_error": _tool_result_is_error(result),
        "result_char_count": len(text),
        "result": redacted_payload("tool_result_redacted", char_count=len(text)),
        "permission_denial_count": len(result.permission_denials),
        "pre_message_count": len(result.pre_messages),
        "additional_message_count": len(result.additional_messages),
        "should_prevent_continuation": result.should_prevent_continuation,
        "prevent_reason_char_count": (
            len(result.prevent_reason) if result.prevent_reason else 0
        ),
    }


def _tool_result_is_error(result: ToolExecutionResult) -> bool:
    content = result.message.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("is_error") is True for block in content)


def _permission_result_kind(result: object | None) -> str | None:
    if result is None:
        return None
    behavior = getattr(result, "behavior", None)
    return str(behavior) if behavior is not None else type(result).__name__


def _permission_reason_type(decision: object) -> str | None:
    reason = getattr(decision, "decision_reason", None)
    if reason is None:
        return None
    reason_type = getattr(reason, "type", None)
    return str(reason_type) if reason_type is not None else type(reason).__name__


__all__ = [
    "ToolExecutionEvent",
    "ToolExecutionProgress",
    "ToolExecutionResult",
    "run_tool_use",
]
