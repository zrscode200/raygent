from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel, model_validator

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.core.tool_execution import ToolExecutionResult
from raygent_harness.core.tool_orchestration import (
    TOOL_CANCEL_MESSAGE,
    ToolOrchestrationOutcome,
    partition_tool_calls,
    run_tools,
)


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


def _deps(*, max_concurrency: int = 10) -> QueryDeps:
    return QueryDeps(
        task_store=AppStateStore(),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        max_tool_use_concurrency=max_concurrency,
    )


def _tool_use(name: str, id_: str) -> ToolUseBlock:
    return ToolUseBlock(id=id_, name=name, input={}, index=0)


def _tool(
    name: str,
    *,
    concurrency_safe: bool = True,
    call: Any | None = None,
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
            input_model=EmptyInput,
            call=call or default_call,
            is_read_only=concurrency_safe,
            is_concurrency_safe=concurrency_safe,
        )
    )


async def _wait_for(predicate: Any, *, timeout_s: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _content(result: ToolExecutionResult) -> str:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    return cast(str, block["content"])


def test_hidden_deferred_partition_does_not_parse_or_evaluate_predicate() -> None:
    parse_seen: list[Any] = []
    predicate_seen: list[BaseModel] = []

    class CountingInput(BaseModel):
        parsed: ClassVar[list[Any]] = parse_seen

        @model_validator(mode="before")
        @classmethod
        def count_parse(cls, data: Any) -> Any:
            cls.parsed.append(data)
            return data

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(content="ok")

    def concurrency_safe(input_: BaseModel) -> bool:
        predicate_seen.append(input_)
        return True

    tool = build_tool(
        ToolSpec(
            name="Deferred",
            description="deferred",
            input_model=CountingInput,
            call=call,
            is_concurrency_safe=concurrency_safe,
            should_defer=True,
        )
    )

    batches = partition_tool_calls(
        (_tool_use("Deferred", "tu_deferred"),),
        (tool,),
        _ctx(),
    )

    assert [(batch.is_concurrency_safe, len(batch.blocks)) for batch in batches] == [
        (False, 1),
    ]
    assert parse_seen == []
    assert predicate_seen == []


@pytest.mark.asyncio
async def test_partition_treats_invalid_missing_and_throwing_predicates_as_unsafe() -> None:
    class RequiredInput(BaseModel):
        required: str

    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolResult(content="ok")

    throwing = build_tool(
        ToolSpec(
            name="Throwing",
            description="throws",
            input_model=EmptyInput,
            call=call,
            is_concurrency_safe=lambda _input: (_ for _ in ()).throw(
                RuntimeError("bad predicate")
            ),
        )
    )
    strict = build_tool(
        ToolSpec(
            name="Strict",
            description="strict",
            input_model=RequiredInput,
            call=call,
            is_concurrency_safe=True,
        )
    )
    safe = _tool("Safe", concurrency_safe=True)

    batches = partition_tool_calls(
        (
            _tool_use("Safe", "tu_safe"),
            _tool_use("Missing", "tu_missing"),
            _tool_use("Strict", "tu_strict"),
            _tool_use("Throwing", "tu_throwing"),
        ),
        (safe, strict, throwing),
        _ctx(),
    )

    assert [(batch.is_concurrency_safe, len(batch.blocks)) for batch in batches] == [
        (True, 1),
        (False, 1),
        (False, 1),
        (False, 1),
    ]


@pytest.mark.asyncio
async def test_safe_batches_overlap_but_unsafe_tools_serialize_boundaries() -> None:
    starts: list[str] = []
    first_safe_release = asyncio.Event()
    unsafe_release = asyncio.Event()
    second_safe_release = asyncio.Event()

    def make_call(name: str) -> Any:
        async def call(
            _input: BaseModel,
            _ctx: ToolUseContext,
        ) -> AsyncIterator[ToolCallEvent]:
            starts.append(name)
            if name in {"SafeA", "SafeB"}:
                await first_safe_release.wait()
            elif name == "UnsafeC":
                await unsafe_release.wait()
            else:
                await second_safe_release.wait()
            yield ToolResult(content=f"{name}-done")

        return call

    tools = (
        _tool("SafeA", concurrency_safe=True, call=make_call("SafeA")),
        _tool("SafeB", concurrency_safe=True, call=make_call("SafeB")),
        _tool("UnsafeC", concurrency_safe=False, call=make_call("UnsafeC")),
        _tool("SafeD", concurrency_safe=True, call=make_call("SafeD")),
        _tool("SafeE", concurrency_safe=True, call=make_call("SafeE")),
    )
    tool_uses = tuple(_tool_use(tool.name, f"tu_{tool.name}") for tool in tools)
    events: list[Any] = []

    async def collect() -> None:
        async for event in run_tools(
            tool_uses=tool_uses,
            assistant_message={"role": "assistant", "content": []},
            tools=tools,
            deps=_deps(),
            ctx=_ctx(),
        ):
            events.append(event)

    task = asyncio.create_task(collect())
    await _wait_for(lambda: {"SafeA", "SafeB"}.issubset(starts))
    assert "UnsafeC" not in starts

    first_safe_release.set()
    await _wait_for(lambda: "UnsafeC" in starts)
    assert "SafeD" not in starts
    assert "SafeE" not in starts

    unsafe_release.set()
    await _wait_for(lambda: {"SafeD", "SafeE"}.issubset(starts))
    second_safe_release.set()
    await task

    result_events = [event for event in events if isinstance(event, ToolExecutionResult)]
    assert [_content(event) for event in result_events] == [
        "SafeA-done",
        "SafeB-done",
        "UnsafeC-done",
        "SafeD-done",
        "SafeE-done",
    ]
    assert isinstance(events[-1], ToolOrchestrationOutcome)


@pytest.mark.asyncio
async def test_max_concurrency_caps_safe_batch_width() -> None:
    starts: list[str] = []
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    def make_call(name: str) -> Any:
        async def call(
            _input: BaseModel,
            _ctx: ToolUseContext,
        ) -> AsyncIterator[ToolCallEvent]:
            starts.append(name)
            if name == "SafeA":
                await release_a.wait()
            if name == "SafeB":
                await release_b.wait()
            yield ToolResult(content=f"{name}-done")

        return call

    tools = (
        _tool("SafeA", call=make_call("SafeA")),
        _tool("SafeB", call=make_call("SafeB")),
        _tool("SafeC", call=make_call("SafeC")),
    )
    tool_uses = tuple(_tool_use(tool.name, f"tu_{tool.name}") for tool in tools)

    async def collect() -> list[Any]:
        return [
            event
            async for event in run_tools(
                tool_uses=tool_uses,
                assistant_message={"role": "assistant", "content": []},
                tools=tools,
                deps=_deps(max_concurrency=1),
                ctx=_ctx(),
                max_concurrency=1,
            )
        ]

    task = asyncio.create_task(collect())
    await _wait_for(lambda: starts == ["SafeA"])
    release_a.set()
    await _wait_for(lambda: starts == ["SafeA", "SafeB"])
    assert "SafeC" not in starts
    release_b.set()
    await task
    assert starts == ["SafeA", "SafeB", "SafeC"]


@pytest.mark.asyncio
async def test_cooperative_cancel_yields_aborted_outcome_with_synthetic_results() -> None:
    async def cancelling_call(
        _input: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        ctx.abort_event.set()
        raise asyncio.CancelledError()
        yield ToolResult(content="unreachable")  # pragma: no cover

    tools = (
        _tool("SafeA", call=cancelling_call),
        _tool("SafeB"),
    )

    events = [
        event
        async for event in run_tools(
            tool_uses=(
                _tool_use("SafeA", "tu_a"),
                _tool_use("SafeB", "tu_b"),
            ),
            assistant_message={"role": "assistant", "content": []},
            tools=tools,
            deps=_deps(),
            ctx=_ctx(),
        )
    ]

    result_events = [event for event in events if isinstance(event, ToolExecutionResult)]
    outcome = events[-1]

    assert isinstance(outcome, ToolOrchestrationOutcome)
    assert outcome.aborted is True
    assert outcome.abort_reason == "abort signaled during tool execution"
    assert [_content(event) for event in result_events] == [
        TOOL_CANCEL_MESSAGE,
        TOOL_CANCEL_MESSAGE,
    ]
    assert outcome.tool_result_messages == tuple(event.message for event in result_events)


@pytest.mark.asyncio
async def test_abort_after_result_synthesizes_results_for_unrun_tools() -> None:
    async def aborting_call(
        _input: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        ctx.abort_event.set()
        yield ToolResult(content="first-done")

    tools = (
        _tool("UnsafeA", concurrency_safe=False, call=aborting_call),
        _tool("UnsafeB", concurrency_safe=False),
    )

    events = [
        event
        async for event in run_tools(
            tool_uses=(
                _tool_use("UnsafeA", "tu_a"),
                _tool_use("UnsafeB", "tu_b"),
            ),
            assistant_message={"role": "assistant", "content": []},
            tools=tools,
            deps=_deps(),
            ctx=_ctx(),
        )
    ]

    result_events = [event for event in events if isinstance(event, ToolExecutionResult)]
    outcome = events[-1]

    assert isinstance(outcome, ToolOrchestrationOutcome)
    assert outcome.aborted is True
    assert [_content(event) for event in result_events] == [
        "first-done",
        TOOL_CANCEL_MESSAGE,
    ]
    assert outcome.tool_result_messages == tuple(event.message for event in result_events)
