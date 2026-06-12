"""Batch tool-use orchestration.

This module sits above ``tool_execution.run_tool_use``. It preserves the
Raygent split: one file owns the single-tool lifecycle; this file
owns partitioning, bounded concurrency, result ordering, and the final batch
outcome consumed by ``query()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.state import PermissionDenial
from raygent_harness.core.tool import find_tool_by_name
from raygent_harness.core.tool_execution import (
    ToolExecutionEvent,
    ToolExecutionProgress,
    ToolExecutionResult,
    run_tool_use,
)

if TYPE_CHECKING:
    from raygent_harness.core.deps import QueryDeps
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.tool import Tool, ToolContextModifier, ToolUseContext


DEFAULT_MAX_TOOL_USE_CONCURRENCY = 10
TOOL_CANCEL_MESSAGE = (
    "The user doesn't want to take this action right now. STOP what you are doing "
    "and wait for the user to tell you how to proceed."
)


@dataclass(frozen=True)
class ToolBatch:
    """A partition of adjacent tool uses with one scheduling mode."""

    is_concurrency_safe: bool
    blocks: tuple[ToolUseBlock, ...]


@dataclass(frozen=True)
class ToolOrchestrationOutcome:
    """Final outcome for one assistant tool-use turn."""

    type: Literal["outcome"] = "outcome"
    tool_result_messages: tuple[MessageParam, ...] = ()
    permission_denials: tuple[PermissionDenial, ...] = ()
    updated_context: ToolUseContext | None = None
    should_prevent_continuation: bool = False
    prevent_reason: str | None = None
    aborted: bool = False
    abort_reason: str | None = None


ToolOrchestrationEvent = (
    ToolExecutionProgress | ToolExecutionResult | ToolOrchestrationOutcome
)


def partition_tool_calls(
    tool_uses: Sequence[ToolUseBlock],
    tools: Sequence[Tool],
) -> tuple[ToolBatch, ...]:
    """Partition tool calls into serial unsafe calls and safe adjacent batches.

    invalid input or throwing predicates as unsafe. Raygent mirrors that
    fail-closed behavior with Pydantic validation and ``Tool.is_concurrency_safe``.
    """
    batches: list[ToolBatch] = []

    for tool_use in tool_uses:
        is_safe = _is_concurrency_safe(tool_use, tools)
        if is_safe and batches and batches[-1].is_concurrency_safe:
            previous = batches[-1]
            batches[-1] = ToolBatch(
                is_concurrency_safe=True,
                blocks=(*previous.blocks, tool_use),
            )
            continue
        batches.append(ToolBatch(is_concurrency_safe=is_safe, blocks=(tool_use,)))

    return tuple(batches)


async def run_tools(
    *,
    tool_uses: Sequence[ToolUseBlock],
    assistant_message: MessageParam,
    tools: Sequence[Tool],
    deps: QueryDeps,
    ctx: ToolUseContext,
    max_concurrency: int = DEFAULT_MAX_TOOL_USE_CONCURRENCY,
) -> AsyncIterator[ToolOrchestrationEvent]:
    """Run a batch of tool uses and yield progress/result events.

    Concurrent batches overlap execution but buffer final result messages until
    the batch settles, then yield them in assistant tool-use order. This keeps
    the next model-visible transcript deterministic while still surfacing
    progress as it arrives.
    """
    current_context = ctx
    result_messages: list[MessageParam] = []
    permission_denials: list[PermissionDenial] = []
    should_prevent_continuation = False
    prevent_reason: str | None = None
    safe_max_concurrency = max(1, max_concurrency)
    completed_tool_use_ids: set[str] = set()
    batches = partition_tool_calls(tool_uses, tools)

    _emit_tool_batch_event(
        deps,
        ctx,
        "tool.batch.started",
        {
            "tool_call_count": len(tool_uses),
            "batch_count": len(batches),
            "max_concurrency": safe_max_concurrency,
        },
    )
    for batch_index, batch in enumerate(batches):
        for sequence, block in enumerate(batch.blocks):
            _emit_tool_batch_event(
                deps,
                ctx,
                "tool.call.scheduled",
                {
                    "tool_use_id": block.id,
                    "tool_name": block.name,
                    "tool_index": block.index,
                    "batch_index": batch_index,
                    "batch_sequence": sequence,
                    "concurrency_safe": batch.is_concurrency_safe,
                    **_tool_input_shape(block.input),
                },
            )

    try:
        for batch in batches:
            if batch.is_concurrency_safe:
                event_stream = _run_concurrent_batch(
                    batch.blocks,
                    assistant_message,
                    tools,
                    deps,
                    current_context,
                    max_concurrency=safe_max_concurrency,
                )
                queued_context_modifiers: list[ToolContextModifier] = []
                async for event in event_stream:
                    if isinstance(event, ToolExecutionProgress):
                        yield event
                        continue

                    flattened_messages = _messages_from_result(event)
                    result_messages.extend(flattened_messages)
                    completed_tool_use_ids.update(_tool_result_ids(flattened_messages))
                    permission_denials.extend(event.permission_denials)
                    should_prevent_continuation = (
                        should_prevent_continuation or event.should_prevent_continuation
                    )
                    prevent_reason = prevent_reason or event.prevent_reason
                    if event.context_modifier is not None:
                        queued_context_modifiers.append(event.context_modifier)
                    yield event
                for modifier in queued_context_modifiers:
                    current_context = modifier(current_context)
            else:
                for block in batch.blocks:
                    async for event in run_tool_use(
                        tool_use=block,
                        assistant_message=assistant_message,
                        tools=tools,
                        deps=deps,
                        ctx=current_context,
                    ):
                        if isinstance(event, ToolExecutionProgress):
                            yield event
                            continue

                        flattened_messages = _messages_from_result(event)
                        result_messages.extend(flattened_messages)
                        completed_tool_use_ids.update(_tool_result_ids(flattened_messages))
                        permission_denials.extend(event.permission_denials)
                        should_prevent_continuation = (
                            should_prevent_continuation
                            or event.should_prevent_continuation
                        )
                        prevent_reason = prevent_reason or event.prevent_reason
                        if event.context_modifier is not None:
                            current_context = event.context_modifier(current_context)
                        yield event
                    if current_context.abort_event.is_set():
                        break

            if current_context.abort_event.is_set():
                break
    except asyncio.CancelledError:
        if not current_context.abort_event.is_set():
            raise

        for result in _cancelled_results(
            tool_uses,
            completed_tool_use_ids,
        ):
            result_messages.append(result.message)
            yield result
        outcome = ToolOrchestrationOutcome(
            tool_result_messages=tuple(result_messages),
            permission_denials=tuple(permission_denials),
            updated_context=current_context,
            should_prevent_continuation=should_prevent_continuation,
            prevent_reason=prevent_reason,
            aborted=True,
            abort_reason="abort signaled during tool execution",
        )
        _emit_tool_batch_event(
            deps,
            current_context,
            "tool.batch.completed",
            _batch_outcome_payload(outcome, result_messages, permission_denials),
        )
        yield outcome
        return

    if current_context.abort_event.is_set():
        for result in _cancelled_results(
            tool_uses,
            completed_tool_use_ids,
        ):
            result_messages.append(result.message)
            yield result

    outcome = ToolOrchestrationOutcome(
        tool_result_messages=tuple(result_messages),
        permission_denials=tuple(permission_denials),
        updated_context=current_context,
        should_prevent_continuation=should_prevent_continuation,
        prevent_reason=prevent_reason,
        aborted=current_context.abort_event.is_set(),
        abort_reason="abort signaled during tool execution"
        if current_context.abort_event.is_set()
        else None,
    )
    _emit_tool_batch_event(
        deps,
        current_context,
        "tool.batch.completed",
        _batch_outcome_payload(outcome, result_messages, permission_denials),
    )
    yield outcome


def _cancelled_results(
    tool_uses: Sequence[ToolUseBlock],
    completed_tool_use_ids: set[str],
) -> Iterator[ToolExecutionResult]:
    for message in _cancelled_tool_result_messages(tool_uses, completed_tool_use_ids):
        completed_tool_use_ids.update(_tool_result_ids((message,)))
        yield ToolExecutionResult(message=message)


def _cancelled_tool_result_messages(
    tool_uses: Sequence[ToolUseBlock],
    completed_tool_use_ids: set[str],
) -> tuple[MessageParam, ...]:
    return tuple(
        _cancelled_tool_result_message(block.id)
        for block in tool_uses
        if block.id not in completed_tool_use_ids
    )


def _cancelled_tool_result_message(tool_use_id: str) -> MessageParam:
    return cast(
        "MessageParam",
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": TOOL_CANCEL_MESSAGE,
                    "is_error": True,
                }
            ],
        },
    )


def _tool_result_ids(messages: Sequence[MessageParam]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if isinstance(tool_use_id, str):
                ids.add(tool_use_id)
    return ids


async def _run_concurrent_batch(
    blocks: Sequence[ToolUseBlock],
    assistant_message: MessageParam,
    tools: Sequence[Tool],
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    max_concurrency: int,
) -> AsyncIterator[ToolExecutionEvent]:
    queue: asyncio.Queue[tuple[int, ToolExecutionEvent | None]] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(index: int, block: ToolUseBlock) -> None:
        async with semaphore:
            try:
                async for event in run_tool_use(
                    tool_use=block,
                    assistant_message=assistant_message,
                    tools=tools,
                    deps=deps,
                    ctx=ctx,
                ):
                    await queue.put((index, event))
            finally:
                await queue.put((index, None))

    tasks = [
        asyncio.create_task(worker(index, block))
        for index, block in enumerate(blocks)
    ]
    results: dict[int, ToolExecutionResult] = {}
    completed = 0

    try:
        while completed < len(tasks):
            index, event = await queue.get()
            if event is None:
                completed += 1
                continue
            if isinstance(event, ToolExecutionProgress):
                yield event
                continue
            results[index] = event

        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for index in range(len(blocks)):
        result = results.get(index)
        if result is not None:
            yield result


def _is_concurrency_safe(
    tool_use: ToolUseBlock,
    tools: Sequence[Tool],
) -> bool:
    tool = find_tool_by_name(tools, tool_use.name)
    if tool is None:
        return False

    try:
        parsed = tool.input_model.model_validate(tool_use.input)
    except Exception:
        return False

    try:
        return bool(tool.is_concurrency_safe(parsed))
    except Exception:
        return False


def _messages_from_result(result: ToolExecutionResult) -> tuple[MessageParam, ...]:
    return (*result.pre_messages, result.message, *result.additional_messages)


def _tool_batch_context(ctx: ToolUseContext) -> KernelEventContext:
    base = ctx.observability_context or KernelEventContext(
        session_id=ctx.session_id,
        agent_id=ctx.agent_id,
    )
    return base.with_source("tool")


def _emit_tool_batch_event(
    deps: QueryDeps,
    ctx: ToolUseContext,
    event_type: str,
    data: dict[str, object],
) -> None:
    deps.observability.emit(
        event_type,
        context=_tool_batch_context(ctx),
        data=data,
    )


def _batch_outcome_payload(
    outcome: ToolOrchestrationOutcome,
    result_messages: Sequence[MessageParam],
    permission_denials: Sequence[object],
) -> dict[str, object]:
    return {
        "tool_result_message_count": len(result_messages),
        "permission_denial_count": len(permission_denials),
        "should_prevent_continuation": outcome.should_prevent_continuation,
        "prevent_reason_char_count": (
            len(outcome.prevent_reason) if outcome.prevent_reason else 0
        ),
        "aborted": outcome.aborted,
        "abort_reason_char_count": len(outcome.abort_reason) if outcome.abort_reason else 0,
    }


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


__all__ = [
    "DEFAULT_MAX_TOOL_USE_CONCURRENCY",
    "TOOL_CANCEL_MESSAGE",
    "ToolBatch",
    "ToolOrchestrationEvent",
    "ToolOrchestrationOutcome",
    "partition_tool_calls",
    "run_tools",
]
