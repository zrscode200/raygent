from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    RecordingKernelEventSink,
)
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.streaming_tool_executor import StreamingToolExecutor
from raygent_harness.core.task import AppStateStore, TaskNotification
from raygent_harness.core.tasks import local_agent as agent_mod
from raygent_harness.core.tasks.local_agent import run_until_done, spawn_local_agent
from raygent_harness.core.tasks.local_bash import LocalBashState
from raygent_harness.core.tasks.remote_agent import spawn_remote_agent
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
from raygent_harness.core.tool_orchestration import ToolOrchestrationOutcome, run_tools
from raygent_harness.services.remote_agent import (
    RemoteAgentLaunchRequest,
    RemoteAgentLaunchResult,
    RemoteAgentPollRequest,
    RemoteAgentPollResult,
)


class EchoInput(BaseModel):
    command: str


def _ctx(*, observability_context: KernelEventContext | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id="parent",
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        query_tracking=QueryTracking(chain_id="c", depth=0),
        observability_context=observability_context
        or KernelEventContext(
            session_id="s",
            agent_id="parent",
            turn_id="turn-1",
            source="query",
        ),
    )


def _tool() -> Tool:
    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        yield ToolProgress(
            message="SECRET_PROGRESS",
            data={"SECRET_PROGRESS_KEY": "secret"},
        )
        yield ToolResult(content="SECRET_RESULT")

    return build_tool(
        ToolSpec(
            name="Echo",
            description="Echo tool",
            input_model=EchoInput,
            call=call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


@pytest.mark.asyncio
async def test_tool_permission_and_hook_observability_is_metadata_only() -> None:
    sink = RecordingKernelEventSink()

    async def pre_hook(_context: PreToolUseContext) -> PreToolUseResult:
        return PreToolUseResult()

    async def post_hook(_context: Any) -> None:
        return None

    deps = QueryDeps(
        task_store=AppStateStore(),
        observability=KernelEventBus([sink]),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        pre_tool_use_hooks=[pre_hook],
        post_tool_use_hooks=[post_hook],
    )
    tool_use = ToolUseBlock(
        id="toolu_secret",
        name="Echo",
        input={"command": "SECRET_INPUT", "SECRET_KEY_NAME": "SECRET_VALUE"},
        index=0,
    )

    events = [
        event
        async for event in run_tools(
            tool_uses=(tool_use,),
            assistant_message={"role": "assistant", "content": []},
            tools=(_tool(),),
            deps=deps,
            ctx=_ctx(),
        )
    ]

    assert isinstance(events[-1], ToolOrchestrationOutcome)
    assert {
        "tool.batch.started",
        "tool.call.scheduled",
        "tool.call.started",
        "hook.pre_tool_use.started",
        "hook.pre_tool_use.completed",
        "permission.requested",
        "permission.decided",
        "tool.call.progress",
        "tool.call.completed",
        "hook.post_tool_use.started",
        "hook.post_tool_use.completed",
        "tool.batch.completed",
    }.issubset(set(sink.event_types))
    assert sink.event_types.index("permission.requested") < sink.event_types.index(
        "permission.decided"
    )
    assert sink.event_types.index("tool.call.completed") < sink.event_types.index(
        "tool.batch.completed"
    )

    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_INPUT" not in combined_payload
    assert "SECRET_KEY_NAME" not in combined_payload
    assert "SECRET_VALUE" not in combined_payload
    assert "SECRET_RESULT" not in combined_payload
    assert "SECRET_PROGRESS" not in combined_payload
    assert "SECRET_PROGRESS_KEY" not in combined_payload
    completed = sink.by_type("tool.call.completed")[0]
    assert completed.agent_id == "parent"
    assert completed.turn_id == "turn-1"
    assert completed.data["result_char_count"] == len("SECRET_RESULT")
    progress = sink.by_type("tool.call.progress")[0]
    assert progress.data["data_key_count"] == 1
    assert progress.data["data_present"] is True


@pytest.mark.asyncio
async def test_tool_observability_does_not_crash_on_non_object_input() -> None:
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        observability=KernelEventBus([sink]),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    tool_use = ToolUseBlock(id="toolu_bad", name="Echo", input=17, index=0)

    events = [
        event
        async for event in run_tools(
            tool_uses=(tool_use,),
            assistant_message={"role": "assistant", "content": []},
            tools=(_tool(),),
            deps=deps,
            ctx=_ctx(),
        )
    ]

    assert isinstance(events[-1], ToolOrchestrationOutcome)
    failed = sink.by_type("tool.call.failed")[0]
    assert failed.data["input_type"] == "int"
    assert failed.data["input_key_count"] is None


@pytest.mark.asyncio
async def test_streaming_tool_observability_uses_safe_input_shape() -> None:
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        observability=KernelEventBus([sink]),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    executor = StreamingToolExecutor(
        tools=(_tool(),),
        deps=deps,
        ctx=_ctx(),
        max_concurrency=1,
    )
    executor.add_tool(
        ToolUseBlock(
            id="toolu_stream",
            name="Echo",
            input={"SECRET_STREAM_KEY": "SECRET_STREAM_VALUE"},
            index=0,
        ),
        {"role": "assistant", "content": []},
    )

    updates = [update async for update in executor.drain_remaining()]

    assert updates
    scheduled = sink.by_type("tool.call.scheduled")[0]
    assert scheduled.data["input_type"] == "object"
    assert scheduled.data["input_key_count"] == 1
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_STREAM_KEY" not in combined_payload
    assert "SECRET_STREAM_VALUE" not in combined_payload


def test_task_store_emits_status_before_notification_and_drains_metadata_only() -> None:
    sink = RecordingKernelEventSink()
    store = AppStateStore(observability=KernelEventBus([sink]))
    state = LocalBashState(
        id="b1",
        type="local_bash",
        description="SECRET_COMMAND",
        status="running",
        start_time=time.time(),
        command="SECRET_COMMAND",
        agent_id="parent",
    )

    store.register_task(state)
    store.update_task(
        "b1",
        lambda task: replace(task, status="completed"),
    )
    store.enqueue_notification(
        TaskNotification(
            task_id="b1",
            message="SECRET_NOTIFICATION",
            kind="stalled",
            agent_id="parent",
        )
    )
    drained = store.drain_notifications("parent")

    assert [notification.task_id for notification in drained] == ["b1"]
    event_types = sink.event_types
    assert event_types.index("task.completed") < event_types.index(
        "task.notification.enqueued"
    )
    assert "task.stalled" in event_types
    assert event_types[-1] == "task.notification.drained"
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_COMMAND" not in combined_payload
    assert "SECRET_NOTIFICATION" not in combined_payload
    enqueued = sink.by_type("task.notification.enqueued")[0]
    assert enqueued.data["message_char_count"] == len("SECRET_NOTIFICATION")


class SecretRemoteBackend:
    def __init__(self, polls: Sequence[RemoteAgentPollResult]) -> None:
        self.launches: list[RemoteAgentLaunchRequest] = []
        self.polls = list(polls)
        self.poll_requests: list[RemoteAgentPollRequest] = []

    async def launch(
        self,
        request: RemoteAgentLaunchRequest,
    ) -> RemoteAgentLaunchResult:
        self.launches.append(request)
        return RemoteAgentLaunchResult(
            remote_id="SECRET_REMOTE_ID",
            title="SECRET_REMOTE_TITLE",
            metadata={"SECRET_LAUNCH_METADATA_KEY": "launch-value"},
        )

    async def poll(self, request: RemoteAgentPollRequest) -> RemoteAgentPollResult:
        self.poll_requests.append(request)
        if self.polls:
            return self.polls.pop(0)
        return RemoteAgentPollResult(status="running")

    async def stop(self, request: object) -> None:
        _ = request


@pytest.mark.asyncio
async def test_remote_agent_observability_redacts_backend_ids_and_metadata_keys() -> None:
    sink = RecordingKernelEventSink()
    store = AppStateStore()
    backend = SecretRemoteBackend(
        (
            RemoteAgentPollResult(
                status="completed",
                message="SECRET_REMOTE_RESULT",
                metadata={"SECRET_POLL_METADATA_KEY": "poll-value"},
            ),
        )
    )
    deps = QueryDeps(
        task_store=store,
        observability=KernelEventBus([sink]),
        remote_agent_backend=backend,
    )

    task_id = await spawn_remote_agent(
        prompt="SECRET_REMOTE_PROMPT",
        description="remote worker",
        agent_type="worker",
        parent_agent_id="parent",
        parent_deps=deps,
        parent_observability_context=_ctx().observability_context,
        poll_interval_s=0,
    )
    for _ in range(50):
        if store.tasks[task_id].status == "completed":
            break
        await asyncio.sleep(0)

    assert store.tasks[task_id].status == "completed"
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_REMOTE_ID" not in combined_payload
    assert "SECRET_REMOTE_TITLE" not in combined_payload
    assert "SECRET_REMOTE_RESULT" not in combined_payload
    assert "SECRET_REMOTE_PROMPT" not in combined_payload
    assert "SECRET_LAUNCH_METADATA_KEY" not in combined_payload
    assert "SECRET_POLL_METADATA_KEY" not in combined_payload
    started = sink.by_type("agent.child.started")[0]
    assert started.data["remote_id_present"] is True
    assert started.data["remote_id_char_count"] == len("SECRET_REMOTE_ID")
    assert started.data["metadata_key_count"] == 1
    registered = sink.by_type("task.registered")[0]
    assert registered.data["remote_id_present"] is True
    assert registered.data["remote_id_char_count"] == len("SECRET_REMOTE_ID")


class _SuccessEngine:
    def __init__(
        self,
        _config: Any,
        _deps: Any,
        _ctx: Any,
        *,
        transcript_scope: Any | None = None,
    ) -> None:
        _ = transcript_scope

    async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
        yield SDKResult(
            subtype="success",
            session_id="child",
            is_error=False,
            num_turns=1,
            result=f"answer to {prompt}",
        )


@pytest.mark.asyncio
async def test_local_agent_observability_inherits_parent_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _SuccessEngine)
    sink = RecordingKernelEventSink()
    store = AppStateStore()
    deps = QueryDeps(task_store=store, observability=KernelEventBus([sink]))
    config = QueryConfig(
        model="model-1",
        session_id="parent-session",
        agent_id="parent",
    )
    ctx = _ctx(
        observability_context=KernelEventContext(
            session_id="parent-session",
            agent_id="parent",
            turn_id="turn-9",
            source="query",
        )
    )

    task_id = await spawn_local_agent(
        prompt="SECRET_PROMPT",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
        child_agent_type="worker",
    )
    final = await run_until_done(task_id, store)

    assert final.status == "completed"
    started = sink.by_type("agent.child.started")[0]
    completed = sink.by_type("agent.child.completed")[0]
    assert started.agent_id == task_id
    assert started.parent_agent_id == "parent"
    assert started.turn_id == "turn-9"
    assert completed.agent_id == task_id
    assert completed.data["final_status"] == "completed"
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_PROMPT" not in combined_payload
