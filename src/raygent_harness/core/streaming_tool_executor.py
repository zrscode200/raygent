"""Streaming tool execution scheduler.

This module is the headless counterpart to the streaming
executor: tools are accepted as `tool_use` blocks arrive from the model stream,
safe tools can start immediately, unsafe tools preserve exclusive ordering, and
completed safe tool results can surface before earlier still-running safe tools.

The one-tool lifecycle still lives in `tool_execution.run_tool_use`; this module
only owns queueing, overlap, cancellation/discard synthesis, and live context
updates.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.observability import KernelEventContext, redacted_payload
from raygent_harness.core.tool import find_tool_by_name
from raygent_harness.core.tool_execution import (
    ToolExecutionProgress,
    ToolExecutionResult,
    run_tool_use,
)
from raygent_harness.core.tool_orchestration import TOOL_CANCEL_MESSAGE

if TYPE_CHECKING:
    from raygent_harness.core.deps import QueryDeps
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.state import PermissionDenial
    from raygent_harness.core.tool import Tool, ToolUseContext


BASH_TOOL_NAME = "Bash"
STREAMING_FALLBACK_MESSAGE = (
    "<tool_use_error>Error: Streaming fallback - tool execution discarded"
    "</tool_use_error>"
)
SIBLING_ERROR_MESSAGE = (
    "<tool_use_error>Cancelled: parallel tool call errored</tool_use_error>"
)


@dataclass(frozen=True)
class StreamingToolProgressUpdate:
    """Progress from an in-flight streamed tool call."""

    progress: ToolExecutionProgress
    sequence: int
    type: Literal["progress"] = "progress"


@dataclass(frozen=True)
class StreamingToolResultUpdate:
    """A completed streamed tool call result."""

    result: ToolExecutionResult
    sequence: int
    tool_use_id: str
    type: Literal["result"] = "result"


StreamingToolUpdate = StreamingToolProgressUpdate | StreamingToolResultUpdate


@dataclass
class _TrackedTool:
    sequence: int
    block: ToolUseBlock
    assistant_message: MessageParam
    is_concurrency_safe: bool
    status: Literal["queued", "executing", "completed", "yielded"] = "queued"
    pending_progress: list[ToolExecutionProgress] = field(
        default_factory=list[ToolExecutionProgress]
    )
    result: ToolExecutionResult | None = None
    task: asyncio.Task[None] | None = None


class StreamingToolExecutor:
    """Queue and execute streamed tool-use blocks.

    `drain_completed()` is non-blocking and returns currently available
    progress/results. `drain_remaining()` waits until all accepted tool-use
    blocks have a model-visible result or have been discarded.
    """

    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        deps: QueryDeps,
        ctx: ToolUseContext,
        max_concurrency: int,
    ) -> None:
        self._tools = tuple(tools)
        self._deps = deps
        self._current_context = ctx
        self._max_concurrency = max(1, max_concurrency)
        self._records: list[_TrackedTool] = []
        self._next_sequence = 0
        self._discarded = False
        self._bash_error_seen = False
        self._bash_error_description = ""
        self._available = asyncio.Event()
        self._tool_result_messages: list[MessageParam] = []
        self._permission_denials: list[PermissionDenial] = []
        self._should_prevent_continuation = False
        self._prevent_reason: str | None = None

    @property
    def updated_context(self) -> ToolUseContext:
        return self._current_context

    @property
    def tool_result_messages(self) -> tuple[MessageParam, ...]:
        return tuple(self._tool_result_messages)

    @property
    def permission_denials(self) -> tuple[PermissionDenial, ...]:
        return tuple(self._permission_denials)

    @property
    def should_prevent_continuation(self) -> bool:
        return self._should_prevent_continuation

    @property
    def prevent_reason(self) -> str | None:
        return self._prevent_reason

    def add_tool(self, block: ToolUseBlock, assistant_message: MessageParam) -> None:
        """Add one streamed tool-use block and start it if scheduling allows."""
        sequence = self._next_sequence
        self._next_sequence += 1
        self._emit_tool_event(
            "tool.call.scheduled",
            block,
            {
                "sequence": sequence,
                "streaming": True,
                **_tool_input_shape(block.input),
            },
        )

        if self._discarded:
            self._emit_tool_event(
                "tool.call.failed",
                block,
                {
                    "sequence": sequence,
                    "streaming": True,
                    "failure_reason": "streaming_fallback_discarded",
                    "result": redacted_payload("tool_result_redacted"),
                },
            )
            self._records.append(
                _TrackedTool(
                    sequence=sequence,
                    block=block,
                    assistant_message=assistant_message,
                    is_concurrency_safe=True,
                    status="completed",
                    result=_synthetic_result(
                        block.id,
                        STREAMING_FALLBACK_MESSAGE,
                        tool_name=block.name,
                        status="interrupted",
                    ),
                )
            )
            self._available.set()
            return

        tool = find_tool_by_name(self._tools, block.name)
        if tool is None:
            self._emit_tool_event(
                "tool.call.failed",
                block,
                {
                    "sequence": sequence,
                    "streaming": True,
                    "failure_reason": "unknown_tool",
                    "result": redacted_payload("tool_result_redacted"),
                },
            )
            self._records.append(
                _TrackedTool(
                    sequence=sequence,
                    block=block,
                    assistant_message=assistant_message,
                    is_concurrency_safe=True,
                    status="completed",
                    result=_synthetic_result(
                        block.id,
                        f"<tool_use_error>Error: No such tool available: "
                        f"{block.name}</tool_use_error>",
                        tool_name=block.name,
                        status="unknown_tool",
                    ),
                )
            )
            self._available.set()
            return

        record = _TrackedTool(
            sequence=sequence,
            block=block,
            assistant_message=assistant_message,
            is_concurrency_safe=_is_concurrency_safe(block, self._tools),
        )
        self._records.append(record)
        self._process_queue()

    def discard(self, reason: str = "streaming_fallback") -> None:
        """Discard all non-yielded tools from an invalidated streaming attempt."""
        self._discarded = True
        message = STREAMING_FALLBACK_MESSAGE
        if reason != "streaming_fallback":
            message = f"<tool_use_error>Error: {reason}</tool_use_error>"
        for record in self._records:
            if record.status == "yielded":
                continue
            self._complete_with_synthetic(record, message)
        self._available.set()

    async def drain_completed(self) -> AsyncIterator[StreamingToolUpdate]:
        """Yield currently available progress and completed results."""
        self._apply_user_abort_policy()
        self._process_queue()
        for update in self._pop_available_updates():
            yield update

    async def drain_remaining(self) -> AsyncIterator[StreamingToolUpdate]:
        """Wait for and yield all remaining progress/results."""
        while self._has_unfinished_tools():
            self._available.clear()
            yielded = False
            async for update in self.drain_completed():
                yielded = True
                yield update
            if yielded or not self._has_unfinished_tools():
                continue
            await self._available.wait()

        async for update in self.drain_completed():
            yield update

    def _process_queue(self) -> None:
        if self._discarded or self._current_context.abort_event.is_set():
            return

        for record in self._records:
            if record.status != "queued":
                continue
            if self._bash_error_seen:
                self._complete_with_synthetic(record, self._sibling_error_message())
                continue
            if self._can_execute(record):
                self._start(record)
                continue
            break

    def _can_execute(self, record: _TrackedTool) -> bool:
        executing = [item for item in self._records if item.status == "executing"]
        if len(executing) >= self._max_concurrency:
            return False
        return not executing or (
            record.is_concurrency_safe
            and all(item.is_concurrency_safe for item in executing)
        )

    def _start(self, record: _TrackedTool) -> None:
        record.status = "executing"
        record.task = asyncio.create_task(
            self._run_record(record),
            name=f"streaming-tool:{record.block.id}",
        )

    async def _run_record(self, record: _TrackedTool) -> None:
        try:
            async for event in run_tool_use(
                tool_use=record.block,
                assistant_message=record.assistant_message,
                tools=self._tools,
                deps=self._deps,
                ctx=self._current_context,
            ):
                if record.status != "executing":
                    return
                if isinstance(event, ToolExecutionProgress):
                    record.pending_progress.append(event)
                    self._available.set()
                    continue
                record.result = event
                if (
                    record.block.name == BASH_TOOL_NAME
                    and _result_is_error(event)
                ):
                    self._bash_error_seen = True
                    self._bash_error_description = _tool_description(record.block)
                    self._cancel_siblings(record)
                if (
                    not record.is_concurrency_safe
                    and event.context_modifier is not None
                ):
                    self._current_context = event.context_modifier(
                        self._current_context
                    )
                record.status = "completed"
                self._available.set()
                self._process_queue()
                return
            if record.status == "executing":
                record.result = _synthetic_result(
                    record.block.id,
                    "<tool_use_error>Error calling tool: tool produced no result"
                    "</tool_use_error>",
                    tool_name=record.block.name,
                    status="failed",
                )
                record.status = "completed"
                self._available.set()
                self._process_queue()
        except asyncio.CancelledError:
            if record.status == "executing":
                self._complete_with_synthetic(
                    record,
                    TOOL_CANCEL_MESSAGE,
                    status="aborted",
                )
        except Exception as exc:
            if record.status == "executing":
                record.result = _synthetic_result(
                    record.block.id,
                    f"<tool_use_error>Error calling tool ({record.block.name}): "
                    f"{exc}</tool_use_error>",
                    tool_name=record.block.name,
                    status="error",
                )
                record.status = "completed"
                self._available.set()
                self._process_queue()

    def _cancel_siblings(self, source: _TrackedTool) -> None:
        for record in self._records:
            if record is source or record.status == "yielded":
                continue
            if record.status in {"queued", "executing"}:
                self._complete_with_synthetic(record, self._sibling_error_message())

    def _complete_with_synthetic(
        self,
        record: _TrackedTool,
        message: str,
        *,
        status: Literal["aborted", "interrupted", "failed"] = "interrupted",
    ) -> None:
        if record.status == "yielded":
            return
        task = record.task
        if task is not None and not task.done():
            task.cancel()
        record.pending_progress.clear()
        record.result = _synthetic_result(
            record.block.id,
            message,
            tool_name=record.block.name,
            status=status,
        )
        record.status = "completed"
        self._emit_tool_event(
            "tool.call.failed",
            record.block,
            {
                "sequence": record.sequence,
                "streaming": True,
                "failure_reason": "synthetic_result",
                "result": redacted_payload(
                    "tool_result_redacted",
                    char_count=len(message),
                ),
            },
        )
        self._available.set()

    def _apply_user_abort_policy(self) -> None:
        if self._discarded or not self._current_context.abort_event.is_set():
            return
        for record in self._records:
            if record.status in {"completed", "yielded"}:
                continue
            if record.status == "queued":
                self._complete_with_synthetic(
                    record,
                    TOOL_CANCEL_MESSAGE,
                    status="aborted",
                )
                continue
            if record.status == "executing" and _interrupt_behavior(
                record.block,
                self._tools,
            ) == "cancel":
                self._complete_with_synthetic(
                    record,
                    TOOL_CANCEL_MESSAGE,
                    status="aborted",
                )

    def _pop_available_updates(self) -> list[StreamingToolUpdate]:
        updates: list[StreamingToolUpdate] = []

        for record in self._records:
            while record.pending_progress:
                updates.append(
                    StreamingToolProgressUpdate(
                        progress=record.pending_progress.pop(0),
                        sequence=record.sequence,
                    )
                )

        for record in self._records:
            if record.status == "yielded":
                continue
            if record.status == "executing" and not record.is_concurrency_safe:
                break
            if record.status != "completed" or record.result is None:
                continue
            record.status = "yielded"
            self._record_result(record.result)
            updates.append(
                StreamingToolResultUpdate(
                    result=record.result,
                    sequence=record.sequence,
                    tool_use_id=record.block.id,
                )
            )

        return updates

    def _sibling_error_message(self) -> str:
        if not self._bash_error_description:
            return SIBLING_ERROR_MESSAGE
        return (
            "<tool_use_error>Cancelled: parallel tool call "
            f"{self._bash_error_description} errored</tool_use_error>"
        )

    def _record_result(self, result: ToolExecutionResult) -> None:
        messages = (*result.pre_messages, result.message, *result.additional_messages)
        self._tool_result_messages.extend(messages)
        self._permission_denials.extend(result.permission_denials)
        self._should_prevent_continuation = (
            self._should_prevent_continuation or result.should_prevent_continuation
        )
        self._prevent_reason = self._prevent_reason or result.prevent_reason

    def _has_unfinished_tools(self) -> bool:
        return any(record.status != "yielded" for record in self._records)

    def _emit_tool_event(
        self,
        event_type: str,
        block: ToolUseBlock,
        data: dict[str, object],
    ) -> None:
        base = self._current_context.observability_context or KernelEventContext(
            session_id=self._current_context.session_id,
            agent_id=self._current_context.agent_id,
        )
        payload = {
            "tool_use_id": block.id,
            "tool_name": block.name,
            "tool_index": block.index,
            **data,
        }
        self._deps.observability.emit(
            event_type,
            context=base.for_child_span(f"tool:{block.id}", source="tool"),
            data=payload,
        )


def _is_concurrency_safe(
    tool_use: ToolUseBlock,
    tools: Sequence[Tool],
) -> bool:
    tool = find_tool_by_name(tools, tool_use.name)
    if tool is None:
        return False
    try:
        parsed = tool.input_model.model_validate(tool_use.input)
        return bool(tool.is_concurrency_safe(parsed))
    except Exception:
        return False


def _interrupt_behavior(
    tool_use: ToolUseBlock,
    tools: Sequence[Tool],
) -> Literal["cancel", "block"]:
    tool = find_tool_by_name(tools, tool_use.name)
    if tool is None:
        return "block"
    try:
        return tool.interrupt_behavior()
    except Exception:
        return "block"


def _result_is_error(result: ToolExecutionResult) -> bool:
    content = result.message.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("is_error") is True for block in content)


def _tool_description(tool_use: ToolUseBlock) -> str:
    input_ = tool_use.input
    if not isinstance(input_, Mapping):
        return tool_use.name
    input_mapping = cast(Mapping[str, object], input_)
    summary = (
        input_mapping.get("command")
        or input_mapping.get("file_path")
        or input_mapping.get("pattern")
    )
    if isinstance(summary, str) and summary:
        truncated = f"{summary[:40]}\u2026" if len(summary) > 40 else summary
        return f"{tool_use.name}({truncated})"
    return tool_use.name


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


def _synthetic_result(
    tool_use_id: str,
    content: str,
    *,
    tool_name: str = "",
    status: Literal[
        "failed",
        "aborted",
        "interrupted",
        "unknown_tool",
        "error",
    ] = "failed",
) -> ToolExecutionResult:
    return ToolExecutionResult(
        message=_tool_result_message(tool_use_id, content, is_error=True),
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        status=status,
    )


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


__all__ = [
    "BASH_TOOL_NAME",
    "SIBLING_ERROR_MESSAGE",
    "STREAMING_FALLBACK_MESSAGE",
    "StreamingToolExecutor",
    "StreamingToolProgressUpdate",
    "StreamingToolResultUpdate",
    "StreamingToolUpdate",
]
