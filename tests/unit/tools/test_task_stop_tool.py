from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from raygent_harness.coordinator.runtime import CoordinatorRuntime
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.task import AgentRouteRecord, AppStateStore
from raygent_harness.core.tasks.local_agent import LocalAgentState
from raygent_harness.core.tasks.local_bash import spawn_local_bash
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.tools.task_stop_tool import (
    TASK_STOP_TOOL_NAME,
    TaskStopInput,
    build_task_stop_tool,
    create_task_stop_catalog_provider,
)


class EmptyInput(BaseModel):
    pass


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _base_tool(name: str) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} tool",
            input_model=EmptyInput,
            call=_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx(*, tools: Sequence[Tool] = (), agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        tools=tuple(tools),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _deps(store: AppStateStore) -> QueryDeps:
    return QueryDeps(task_store=store)


def _deps_with_coordinator(
    store: AppStateStore,
    runtime: CoordinatorRuntime,
) -> QueryDeps:
    return QueryDeps(task_store=store, coordinator_runtime=runtime)


def _tool_use(input_: dict[str, Any], *, name: str = TASK_STOP_TOOL_NAME) -> ToolUseBlock:
    return ToolUseBlock(
        id="toolu_stop",
        name=name,
        input=input_,
        index=0,
    )


async def _run_task_stop(
    *,
    tool: Tool,
    deps: QueryDeps,
    ctx: ToolUseContext,
    input_: dict[str, Any],
    tool_name: str = TASK_STOP_TOOL_NAME,
) -> ToolExecutionResult:
    results = [
        event
        async for event in run_tool_use(
            tool_use=_tool_use(input_, name=tool_name),
            assistant_message={"role": "assistant", "content": []},
            tools=ctx.tools,
            deps=deps,
            ctx=ctx,
        )
        if isinstance(event, ToolExecutionResult)
    ]
    assert len(results) == 1
    return results[0]


def _content(result: ToolExecutionResult) -> str | list[dict[str, Any]]:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return block["content"]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_task_stop_tool_stops_running_agent_task() -> None:
    store = AppStateStore()
    task = LocalAgentState(
        id="a_worker",
        type="local_agent",
        description="worker",
        status="running",
        start_time=1.0,
        parent_agent_id=None,
        prompt="work",
    )
    store.register_task(task)
    deps = _deps(store)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": "a_worker"},
    )

    content = _content(result)
    assert isinstance(content, list)
    assert content[1] == {
        "type": "task_stopped",
        "task_id": "a_worker",
        "task_type": "local_agent",
        "description": "worker",
    }
    assert store.tasks["a_worker"].status == "killed"


@pytest.mark.asyncio
async def test_task_stop_tool_preserves_named_local_agent_resume_record() -> None:
    store = AppStateStore()
    task = LocalAgentState(
        id="a_worker",
        type="local_agent",
        description="worker",
        status="running",
        start_time=1.0,
        parent_agent_id=None,
        prompt="work",
        name="researcher",
    )
    record = AgentRouteRecord(
        agent_id="a_worker",
        task_id="a_worker",
        task_type="local_agent",
        name="researcher",
        parent_session_id="s",
        transcript_path="/tmp/a_worker.jsonl",
    )
    store.register_task(task)
    store.agent_name_registry["researcher"] = "a_worker"
    store.agent_route_records["a_worker"] = record
    deps = _deps(store)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": "a_worker"},
    )

    assert not _is_error(result)
    assert store.tasks["a_worker"].status == "killed"
    assert store.agent_name_registry["researcher"] == "a_worker"
    assert store.agent_route_records["a_worker"] == record


@pytest.mark.asyncio
async def test_task_stop_tool_records_successful_stop_in_coordinator_runtime() -> None:
    store = AppStateStore()
    runtime = CoordinatorRuntime()
    task = LocalAgentState(
        id="a_worker",
        type="local_agent",
        description="worker",
        status="running",
        start_time=1.0,
        parent_agent_id=None,
        prompt="work",
    )
    store.register_task(task)
    runtime.record_agent_launch(
        agent_id="a_worker",
        task_id="a_worker",
        agent_type="worker",
        description="worker",
        prompt_chars=4,
        status="running",
    )
    deps = _deps_with_coordinator(store, runtime)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": "a_worker"},
    )

    content = _content(result)
    assert isinstance(content, list)
    assert content[1]["coordinator_work_item_id"] == "cw_agent_a_worker"
    assert content[1]["coordinator_blackboard_entry_id"] == "cb_000001"
    snapshot = runtime.snapshot()
    assert snapshot.work_items[0].status == "killed"
    assert snapshot.blackboard_entries[0].kind == "risk"
    assert "TaskStop stopped task=a_worker" in snapshot.blackboard_entries[0].content


@pytest.mark.asyncio
async def test_task_stop_tool_accepts_killshell_alias_and_shell_id() -> None:
    store = AppStateStore()
    task = LocalAgentState(
        id="legacy_shell",
        type="local_agent",
        description="legacy task",
        status="running",
        start_time=1.0,
        parent_agent_id=None,
        prompt="work",
    )
    store.register_task(task)
    deps = _deps(store)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"shell_id": "legacy_shell"},
        tool_name="KillShell",
    )

    assert not _is_error(result)
    assert store.tasks["legacy_shell"].status == "killed"


def test_task_stop_tool_reference_metadata() -> None:
    tool = build_task_stop_tool(deps=_deps(AppStateStore()))

    assert tool.aliases == ("KillShell",)
    assert tool.should_defer is True
    assert tool.always_load is False
    assert tool.max_result_size_chars == 100_000
    assert tool.is_concurrency_safe(TaskStopInput(task_id="task_1"))


@pytest.mark.asyncio
async def test_task_stop_tool_surfaces_typed_stop_errors() -> None:
    store = AppStateStore()
    deps = _deps(store)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": "missing"},
    )

    content = _content(result)
    assert isinstance(content, str)
    assert "TaskStop failed (not_found)" in content
    assert _is_error(result)


@pytest.mark.asyncio
async def test_task_stop_tool_error_does_not_record_coordinator_runtime() -> None:
    store = AppStateStore()
    runtime = CoordinatorRuntime()
    deps = _deps_with_coordinator(store, runtime)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": "missing"},
    )

    assert _is_error(result)
    assert runtime.snapshot().work_items == ()
    assert runtime.snapshot().blackboard_entries == ()


@pytest.mark.asyncio
async def test_task_stop_tool_preserves_bash_notification_suppression() -> None:
    store = AppStateStore()
    task = await spawn_local_bash("sleep 5", store, agent_id=None)
    await asyncio.sleep(0.05)
    deps = _deps(store)
    tool = build_task_stop_tool(deps=deps)

    result = await _run_task_stop(
        tool=tool,
        deps=deps,
        ctx=_ctx(tools=(tool,)),
        input_={"task_id": task.id},
    )
    await asyncio.sleep(0.3)

    assert not _is_error(result)
    assert store.tasks[task.id].status == "killed"
    assert store.tasks[task.id].notified is True
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_task_stop_tool_validates_main_thread_and_required_task_id() -> None:
    store = AppStateStore()
    tool = build_task_stop_tool(deps=_deps(store))

    empty = await tool.validate_input(TaskStopInput(task_id=" "), _ctx(tools=(tool,)))
    child = await tool.validate_input(
        TaskStopInput(task_id="a1"),
        _ctx(tools=(tool,), agent_id="a_child"),
    )

    assert isinstance(empty, ValidationError)
    assert "task_id is required" in empty.message
    assert isinstance(child, ValidationError)
    assert "main coordinator" in child.message


@pytest.mark.asyncio
async def test_task_stop_tool_direct_call_rejects_child_context_without_side_effects() -> None:
    store = AppStateStore()
    runtime = CoordinatorRuntime()
    task = LocalAgentState(
        id="a_worker",
        type="local_agent",
        description="worker",
        status="running",
        start_time=1.0,
        parent_agent_id=None,
        prompt="work",
    )
    store.register_task(task)
    deps = _deps_with_coordinator(store, runtime)
    tool = build_task_stop_tool(deps=deps)

    events = [
        event
        async for event in tool.call(
            TaskStopInput(task_id="a_worker"),
            _ctx(tools=(tool,), agent_id="a_child"),
        )
    ]

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "main coordinator" in str(result.content)
    assert store.tasks["a_worker"].status == "running"
    assert runtime.snapshot().work_items == ()
    assert runtime.snapshot().blackboard_entries == ()


@pytest.mark.asyncio
async def test_task_stop_catalog_provider_enabled_main_thread_only() -> None:
    base = _base_tool("Read")
    deps = _deps(AppStateStore())
    provider = create_task_stop_catalog_provider(parent_deps=deps, enabled=True)
    disabled = create_task_stop_catalog_provider(parent_deps=deps, enabled=False)
    config = QueryConfig(model="m", tools=(base,))

    main_tools = await provider(config, _ctx(tools=(base,)), ())
    child_tools = await provider(config, _ctx(tools=(base,), agent_id="a_child"), ())
    disabled_tools = await disabled(config, _ctx(tools=(base,)), ())

    assert main_tools is not None
    assert tuple(tool.name for tool in main_tools) == ("Read", TASK_STOP_TOOL_NAME)
    assert child_tools is not None
    assert tuple(tool.name for tool in child_tools) == ("Read",)
    assert disabled_tools is not None
    assert tuple(tool.name for tool in disabled_tools) == ("Read",)


def _is_error(result: ToolExecutionResult) -> bool:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return bool(block.get("is_error", False))
