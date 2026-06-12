from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.permissions import (
    HookPermissionDecisionReason,
    PermissionDenyDecision,
    ToolPermissionContext,
)
from raygent_harness.core.streaming_tool_executor import (
    STREAMING_FALLBACK_MESSAGE,
    StreamingToolExecutor,
    StreamingToolProgressUpdate,
    StreamingToolResultUpdate,
    StreamingToolUpdate,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolProgress,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.core.tool_hooks import PreToolUseContext, PreToolUseResult
from raygent_harness.core.tool_orchestration import TOOL_CANCEL_MESSAGE


class EmptyInput(BaseModel):
    pass


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _deps(*, pre_hooks: list[Any] | None = None) -> QueryDeps:
    return QueryDeps(
        task_store=AppStateStore(),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        pre_tool_use_hooks=pre_hooks or [],
    )


def _tool_use(
    name: str,
    id_: str,
    index: int = 0,
    input_: dict[str, Any] | None = None,
) -> ToolUseBlock:
    return ToolUseBlock(id=id_, name=name, input=input_ or {}, index=index)


def _assistant_message() -> MessageParam:
    return {"role": "assistant", "content": []}


def _tool(
    name: str,
    *,
    concurrency_safe: bool | Callable[[BaseModel], bool] = True,
    interrupt_behavior: str = "block",
    input_model: type[BaseModel] = EmptyInput,
    call: Callable[[BaseModel, ToolUseContext], AsyncIterator[ToolCallEvent]]
    | None = None,
    check_permissions: Any | None = None,
) -> Tool:
    async def default_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(content=f"{name}-done")

    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} tool",
            input_model=input_model,
            call=call or default_call,
            is_read_only=cast(Any, concurrency_safe),
            is_concurrency_safe=cast(Any, concurrency_safe),
            interrupt_behavior=cast(Any, interrupt_behavior),
            check_permissions=check_permissions,
        )
    )


async def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


async def _drain_completed(
    executor: StreamingToolExecutor,
) -> list[StreamingToolUpdate]:
    return [update async for update in executor.drain_completed()]


async def _drain_remaining(
    executor: StreamingToolExecutor,
) -> list[StreamingToolUpdate]:
    return [update async for update in executor.drain_remaining()]


async def _drain_until(
    executor: StreamingToolExecutor,
    predicate: Callable[[list[StreamingToolUpdate]], bool],
    *,
    timeout_s: float = 1.0,
) -> list[StreamingToolUpdate]:
    collected: list[StreamingToolUpdate] = []
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        collected.extend(await _drain_completed(executor))
        if predicate(collected):
            return collected
        await asyncio.sleep(0.005)
    raise AssertionError("expected streaming tool update did not arrive")


def _result_updates(
    updates: list[StreamingToolUpdate],
) -> list[StreamingToolResultUpdate]:
    return [update for update in updates if isinstance(update, StreamingToolResultUpdate)]


def _progress_updates(
    updates: list[StreamingToolUpdate],
) -> list[StreamingToolProgressUpdate]:
    return [update for update in updates if isinstance(update, StreamingToolProgressUpdate)]


def _content(update: StreamingToolResultUpdate) -> str:
    content = update.result.message["content"]
    assert isinstance(content, list)
    return str(content[0]["content"])


@pytest.mark.asyncio
async def test_safe_tools_start_immediately_and_overlap() -> None:
    starts: list[str] = []
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    def make_call(name: str, release: asyncio.Event) -> Callable[
        [BaseModel, ToolUseContext],
        AsyncIterator[ToolCallEvent],
    ]:
        async def call(
            _input: BaseModel,
            _ctx: ToolUseContext,
        ) -> AsyncIterator[ToolCallEvent]:
            starts.append(name)
            await release.wait()
            yield ToolResult(content=f"{name}-done")

        return call

    executor = StreamingToolExecutor(
        tools=(
            _tool("SafeA", call=make_call("SafeA", release_a)),
            _tool("SafeB", call=make_call("SafeB", release_b)),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )

    executor.add_tool(_tool_use("SafeA", "tu_a", 0), _assistant_message())
    executor.add_tool(_tool_use("SafeB", "tu_b", 1), _assistant_message())

    await _wait_for(lambda: {"SafeA", "SafeB"}.issubset(starts))
    release_a.set()
    release_b.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        "SafeA-done",
        "SafeB-done",
    ]


@pytest.mark.asyncio
async def test_max_concurrency_caps_safe_tool_width() -> None:
    starts: list[str] = []
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    def make_call(name: str, release: asyncio.Event) -> Callable[
        [BaseModel, ToolUseContext],
        AsyncIterator[ToolCallEvent],
    ]:
        async def call(
            _input: BaseModel,
            _ctx: ToolUseContext,
        ) -> AsyncIterator[ToolCallEvent]:
            starts.append(name)
            await release.wait()
            yield ToolResult(content=f"{name}-done")

        return call

    async def safe_c_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("SafeC")
        yield ToolResult(content="SafeC-done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("SafeA", call=make_call("SafeA", release_a)),
            _tool("SafeB", call=make_call("SafeB", release_b)),
            _tool("SafeC", call=safe_c_call),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=1,
    )
    executor.add_tool(_tool_use("SafeA", "tu_a", 0), _assistant_message())
    executor.add_tool(_tool_use("SafeB", "tu_b", 1), _assistant_message())
    executor.add_tool(_tool_use("SafeC", "tu_c", 2), _assistant_message())

    await _wait_for(lambda: starts == ["SafeA"])
    release_a.set()
    await _wait_for(lambda: starts == ["SafeA", "SafeB"])
    assert "SafeC" not in starts
    release_b.set()
    updates = await _drain_remaining(executor)

    assert starts == ["SafeA", "SafeB", "SafeC"]
    assert [_content(update) for update in _result_updates(updates)] == [
        "SafeA-done",
        "SafeB-done",
        "SafeC-done",
    ]


@pytest.mark.asyncio
async def test_progress_and_completed_safe_results_are_immediate() -> None:
    starts: list[str] = []
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    async def safe_a(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("SafeA")
        await release_a.wait()
        yield ToolResult(content="SafeA-done")

    async def safe_b(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("SafeB")
        yield ToolProgress(message="SafeB-progress")
        await release_b.wait()
        yield ToolResult(content="SafeB-done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("SafeA", call=safe_a),
            _tool("SafeB", call=safe_b),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("SafeA", "tu_a", 0), _assistant_message())
    executor.add_tool(_tool_use("SafeB", "tu_b", 1), _assistant_message())

    await _wait_for(lambda: {"SafeA", "SafeB"}.issubset(starts))
    early_updates = await _drain_until(
        executor,
        lambda updates: bool(_progress_updates(updates)),
    )
    progress = _progress_updates(early_updates)
    assert progress[0].progress.message == "SafeB-progress"
    assert _result_updates(early_updates) == []

    release_b.set()
    early_result_updates = await _drain_until(
        executor,
        lambda updates: len(_result_updates(updates)) == 1,
    )
    assert [_content(update) for update in _result_updates(early_result_updates)] == [
        "SafeB-done"
    ]

    release_a.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        "SafeA-done",
    ]
    assert executor.tool_result_messages == (
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_b",
                    "content": "SafeB-done",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_a",
                    "content": "SafeA-done",
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_executing_unsafe_tool_blocks_later_completed_result() -> None:
    starts: list[str] = []
    release_unsafe = asyncio.Event()

    async def unsafe_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Unsafe")
        await release_unsafe.wait()
        yield ToolResult(content="unsafe done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("Unsafe", concurrency_safe=False, call=unsafe_call),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Unsafe", "tu_unsafe", 0), _assistant_message())
    executor.add_tool(_tool_use("Missing", "tu_missing", 1), _assistant_message())

    await _wait_for(lambda: starts == ["Unsafe"])
    assert await _drain_completed(executor) == []

    release_unsafe.set()
    updates = await _drain_remaining(executor)

    results = _result_updates(updates)
    assert [_content(update) for update in results] == [
        "unsafe done",
        "<tool_use_error>Error: No such tool available: Missing</tool_use_error>",
    ]


@pytest.mark.asyncio
async def test_unsafe_tool_blocks_following_safe_tool_from_leapfrogging() -> None:
    starts: list[str] = []
    release_safe_a = asyncio.Event()
    release_unsafe_b = asyncio.Event()
    release_safe_c = asyncio.Event()

    def make_call(name: str, release: asyncio.Event) -> Callable[
        [BaseModel, ToolUseContext],
        AsyncIterator[ToolCallEvent],
    ]:
        async def call(
            _input: BaseModel,
            _ctx: ToolUseContext,
        ) -> AsyncIterator[ToolCallEvent]:
            starts.append(name)
            await release.wait()
            yield ToolResult(content=f"{name}-done")

        return call

    executor = StreamingToolExecutor(
        tools=(
            _tool("SafeA", concurrency_safe=True, call=make_call("SafeA", release_safe_a)),
            _tool(
                "UnsafeB",
                concurrency_safe=False,
                call=make_call("UnsafeB", release_unsafe_b),
            ),
            _tool("SafeC", concurrency_safe=True, call=make_call("SafeC", release_safe_c)),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("SafeA", "tu_a", 0), _assistant_message())
    executor.add_tool(_tool_use("UnsafeB", "tu_b", 1), _assistant_message())
    executor.add_tool(_tool_use("SafeC", "tu_c", 2), _assistant_message())

    await _wait_for(lambda: starts == ["SafeA"])
    release_safe_a.set()
    await _wait_for(lambda: starts == ["SafeA", "UnsafeB"])
    assert "SafeC" not in starts

    release_unsafe_b.set()
    await _wait_for(lambda: starts == ["SafeA", "UnsafeB", "SafeC"])
    release_safe_c.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        "SafeA-done",
        "UnsafeB-done",
        "SafeC-done",
    ]


@pytest.mark.asyncio
async def test_unknown_tool_completes_immediately_with_model_visible_error() -> None:
    executor = StreamingToolExecutor(
        tools=(),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )

    executor.add_tool(_tool_use("Missing", "tu_missing"), _assistant_message())
    updates = await _drain_completed(executor)

    results = _result_updates(updates)
    assert len(results) == 1
    assert "No such tool available: Missing" in _content(results[0])


@pytest.mark.asyncio
async def test_invalid_input_and_throwing_safe_predicate_are_treated_unsafe() -> None:
    class RequiredInput(BaseModel):
        required: str

    starts: list[str] = []
    release_safe_a = asyncio.Event()
    release_throwing = asyncio.Event()

    async def safe_a_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("SafeA")
        await release_safe_a.wait()
        yield ToolResult(content="SafeA-done")

    async def throwing_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Throwing")
        await release_throwing.wait()
        yield ToolResult(content="Throwing-done")

    async def safe_c_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("SafeC")
        yield ToolResult(content="SafeC-done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("SafeA", concurrency_safe=True, call=safe_a_call),
            _tool("Strict", concurrency_safe=True, input_model=RequiredInput),
            _tool(
                "Throwing",
                concurrency_safe=lambda _input: (_ for _ in ()).throw(
                    RuntimeError("predicate exploded")
                ),
                call=throwing_call,
            ),
            _tool("SafeC", concurrency_safe=True, call=safe_c_call),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("SafeA", "tu_a", 0), _assistant_message())
    executor.add_tool(_tool_use("Strict", "tu_strict", 1), _assistant_message())
    executor.add_tool(_tool_use("Throwing", "tu_throwing", 2), _assistant_message())
    executor.add_tool(_tool_use("SafeC", "tu_c", 3), _assistant_message())

    await _wait_for(lambda: starts == ["SafeA"])
    assert await _drain_completed(executor) == []

    release_safe_a.set()
    await _wait_for(lambda: starts == ["SafeA", "Throwing"])
    assert "SafeC" not in starts
    release_throwing.set()
    await _wait_for(lambda: starts == ["SafeA", "Throwing", "SafeC"])
    updates = await _drain_remaining(executor)

    contents = [_content(update) for update in _result_updates(updates)]
    assert contents[0] == "SafeA-done"
    assert "InputValidationError" in contents[1]
    assert contents[2:] == ["Throwing-done", "SafeC-done"]


@pytest.mark.asyncio
async def test_unsafe_context_modifier_updates_later_tool_context() -> None:
    seen_model_overrides: list[str | None] = []

    async def modifier_tool(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(
            content="modified",
            context_modifier=lambda ctx: replace(ctx, model_override="unsafe-model"),
        )

    async def reader_tool(
        _input: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        seen_model_overrides.append(ctx.model_override)
        yield ToolResult(content="read")

    executor = StreamingToolExecutor(
        tools=(
            _tool("Modifier", concurrency_safe=False, call=modifier_tool),
            _tool("Reader", concurrency_safe=False, call=reader_tool),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Modifier", "tu_modifier", 0), _assistant_message())
    executor.add_tool(_tool_use("Reader", "tu_reader", 1), _assistant_message())

    await _drain_remaining(executor)

    assert seen_model_overrides == ["unsafe-model"]


@pytest.mark.asyncio
async def test_safe_context_modifier_is_ignored_for_streaming_overlap() -> None:
    seen_model_overrides: list[str | None] = []

    async def modifier_tool(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(
            content="modified",
            context_modifier=lambda ctx: replace(ctx, model_override="safe-model"),
        )

    async def reader_tool(
        _input: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        seen_model_overrides.append(ctx.model_override)
        yield ToolResult(content="read")

    executor = StreamingToolExecutor(
        tools=(
            _tool("Modifier", concurrency_safe=True, call=modifier_tool),
            _tool("Reader", concurrency_safe=False, call=reader_tool),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Modifier", "tu_modifier", 0), _assistant_message())
    executor.add_tool(_tool_use("Reader", "tu_reader", 1), _assistant_message())

    await _drain_remaining(executor)

    assert seen_model_overrides == [None]


@pytest.mark.asyncio
async def test_aggregates_messages_denials_and_prevention_metadata() -> None:
    pre_message: MessageParam = {"role": "user", "content": "pre"}
    additional_message: dict[str, Any] = {"role": "user", "content": "additional"}

    async def hook(context: PreToolUseContext) -> PreToolUseResult | None:
        if context.tool.name != "Rich":
            return None
        return PreToolUseResult(
            additional_messages=(pre_message,),
            should_prevent_continuation=True,
            stop_reason="review required",
        )

    async def rich_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(
            content="rich result",
            additional_messages=(additional_message,),
        )

    async def deny(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionDenyDecision:
        return PermissionDenyDecision(
            message="denied by policy",
            decision_reason=HookPermissionDecisionReason(hook_name="test"),
        )

    executor = StreamingToolExecutor(
        tools=(
            _tool("Rich", concurrency_safe=False, call=rich_call),
            _tool("Denied", concurrency_safe=False, check_permissions=deny),
        ),
        deps=_deps(pre_hooks=[hook]),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Rich", "tu_rich", 0), _assistant_message())
    executor.add_tool(_tool_use("Denied", "tu_denied", 1), _assistant_message())

    await _drain_remaining(executor)

    assert executor.tool_result_messages == (
        pre_message,
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_rich",
                    "content": "rich result",
                }
            ],
        },
        cast(MessageParam, additional_message),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_denied",
                    "content": "denied by policy",
                    "is_error": True,
                }
            ],
        },
    )
    assert len(executor.permission_denials) == 1
    assert executor.permission_denials[0].tool_use_id == "tu_denied"
    assert executor.should_prevent_continuation is True
    assert executor.prevent_reason == "review required"


@pytest.mark.asyncio
async def test_bash_error_cancels_running_sibling_with_synthetic_result() -> None:
    starts: list[str] = []
    bash_release = asyncio.Event()
    sibling_release = asyncio.Event()

    async def bash_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Bash")
        await bash_release.wait()
        yield ToolResult(content="bash failed", is_error=True)

    async def sibling_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Sibling")
        await sibling_release.wait()
        yield ToolResult(content="sibling done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("Bash", concurrency_safe=True, call=bash_call),
            _tool("Sibling", concurrency_safe=True, call=sibling_call),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(
        _tool_use("Bash", "tu_bash", 0, {"command": "npm test"}),
        _assistant_message(),
    )
    executor.add_tool(_tool_use("Sibling", "tu_sibling", 1), _assistant_message())

    await _wait_for(lambda: {"Bash", "Sibling"}.issubset(starts))
    bash_release.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        "bash failed",
        "<tool_use_error>Cancelled: parallel tool call Bash(npm test) errored</tool_use_error>",
    ]


@pytest.mark.asyncio
async def test_non_bash_error_does_not_cancel_sibling() -> None:
    starts: list[str] = []
    error_release = asyncio.Event()
    sibling_release = asyncio.Event()

    async def error_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Read")
        await error_release.wait()
        yield ToolResult(content="read failed", is_error=True)

    async def sibling_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Sibling")
        await sibling_release.wait()
        yield ToolResult(content="sibling done")

    executor = StreamingToolExecutor(
        tools=(
            _tool("Read", concurrency_safe=True, call=error_call),
            _tool("Sibling", concurrency_safe=True, call=sibling_call),
        ),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Read", "tu_read", 0), _assistant_message())
    executor.add_tool(_tool_use("Sibling", "tu_sibling", 1), _assistant_message())

    await _wait_for(lambda: {"Read", "Sibling"}.issubset(starts))
    error_release.set()
    first_updates = await _drain_until(
        executor,
        lambda updates: bool(_result_updates(updates)),
    )
    assert [_content(update) for update in _result_updates(first_updates)] == [
        "read failed",
    ]

    sibling_release.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == ["sibling done"]


@pytest.mark.asyncio
async def test_discard_suppresses_stale_in_flight_result() -> None:
    starts: list[str] = []
    release = asyncio.Event()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Slow")
        await release.wait()
        yield ToolResult(content="stale")

    executor = StreamingToolExecutor(
        tools=(_tool("Slow", concurrency_safe=True, call=call),),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Slow", "tu_slow"), _assistant_message())

    await _wait_for(lambda: starts == ["Slow"])
    executor.discard()
    updates = await _drain_remaining(executor)
    release.set()
    await asyncio.sleep(0)

    assert [_content(update) for update in _result_updates(updates)] == [
        STREAMING_FALLBACK_MESSAGE,
    ]
    assert await _drain_completed(executor) == []


@pytest.mark.asyncio
async def test_tool_added_after_discard_gets_fallback_result_without_hanging() -> None:
    executor = StreamingToolExecutor(
        tools=(_tool("Later", concurrency_safe=True),),
        deps=_deps(),
        ctx=_ctx(),
        max_concurrency=10,
    )

    executor.discard()
    executor.add_tool(_tool_use("Later", "tu_later"), _assistant_message())
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        STREAMING_FALLBACK_MESSAGE,
    ]


@pytest.mark.asyncio
async def test_user_abort_cancels_cancel_interrupt_tool() -> None:
    starts: list[str] = []
    release = asyncio.Event()
    ctx = _ctx()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Cancelable")
        await release.wait()
        yield ToolResult(content="should not appear")

    executor = StreamingToolExecutor(
        tools=(
            _tool(
                "Cancelable",
                concurrency_safe=True,
                interrupt_behavior="cancel",
                call=call,
            ),
        ),
        deps=_deps(),
        ctx=ctx,
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Cancelable", "tu_cancel"), _assistant_message())

    await _wait_for(lambda: starts == ["Cancelable"])
    ctx.abort_event.set()
    updates = await _drain_completed(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        TOOL_CANCEL_MESSAGE,
    ]


@pytest.mark.asyncio
async def test_user_abort_waits_for_block_interrupt_tool() -> None:
    starts: list[str] = []
    release = asyncio.Event()
    ctx = _ctx()

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        starts.append("Blocking")
        await release.wait()
        yield ToolResult(content="finished despite abort")

    executor = StreamingToolExecutor(
        tools=(
            _tool(
                "Blocking",
                concurrency_safe=True,
                interrupt_behavior="block",
                call=call,
            ),
        ),
        deps=_deps(),
        ctx=ctx,
        max_concurrency=10,
    )
    executor.add_tool(_tool_use("Blocking", "tu_block"), _assistant_message())

    await _wait_for(lambda: starts == ["Blocking"])
    ctx.abort_event.set()
    assert await _drain_completed(executor) == []

    release.set()
    updates = await _drain_remaining(executor)

    assert [_content(update) for update in _result_updates(updates)] == [
        "finished despite abort",
    ]
