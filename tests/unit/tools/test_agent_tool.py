from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.agents.models import AgentContextPolicy, AgentDefinition
from raygent_harness.coordinator.runtime import CoordinatorRuntime
from raygent_harness.coordinator.team import TeamStateStore
from raygent_harness.core import child_query as child_query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment, ContextKind
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.model_types import ModelRequest
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKAssistantMessage, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks import in_process_teammate as teammate_mod
from raygent_harness.core.tasks import local_agent as agent_mod
from raygent_harness.core.tasks.in_process_teammate import (
    InProcessTeammateState,
    InProcessTeammateTask,
)
from raygent_harness.core.tasks.local_agent import (
    LocalAgentState,
    LocalAgentTask,
    run_until_done,
)
from raygent_harness.core.tasks.remote_agent import RemoteAgentState, RemoteAgentTask
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolRuntimeContext,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.services.handoff import (
    HandoffClassificationRequest,
    HandoffClassificationResult,
)
from raygent_harness.services.remote_agent import (
    RemoteAgentLaunchRequest,
    RemoteAgentLaunchResult,
    RemoteAgentPollRequest,
    RemoteAgentPollResult,
    RemoteAgentStopRequest,
)
from raygent_harness.services.transcript import JsonlTranscriptStore, get_agent_transcript
from raygent_harness.services.worktree import WorktreeCleanupResult, WorktreeInfo
from raygent_harness.tools.agent_tool import (
    AGENT_TOOL_NAME,
    AgentToolInput,
    build_agent_tool,
    create_agent_catalog_provider,
)
from tests.fakes import FakeModelProvider


class EmptyInput(BaseModel):
    pass


class RecordingContextProvider:
    context_kind: ContextKind

    def __init__(self, kind: ContextKind, marker: str) -> None:
        self.context_kind = kind
        self.marker = marker

    async def __call__(
        self,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        *_args: Any,
    ) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                id=self.marker,
                content=self.marker,
                channel="user_context",
                kind=self.context_kind,
            ),
        )


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
            check_permissions=lambda _input, _ctx, _pc: _allow(),
        )
    )


async def _allow() -> Any:
    from raygent_harness.core.permissions import PermissionAllowDecision

    return PermissionAllowDecision()


def _agent_definition(
    *,
    agent_type: str = "worker",
    system_prompt: str = "worker system",
    tools: tuple[str, ...] | None = ("*",),
    disallowed_tools: tuple[str, ...] = (),
    model: str | None = None,
    permission_mode: str | None = None,
    initial_prompt: str | None = None,
    required_mcp_servers: tuple[str, ...] = (),
    isolation: str | None = None,
    context_policy: AgentContextPolicy | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_type=agent_type,
        description=f"{agent_type} description",
        system_prompt=system_prompt,
        tools=tools,
        disallowed_tools=disallowed_tools,
        model=cast(Any, model),
        permission_mode=cast(Any, permission_mode),
        initial_prompt=initial_prompt,
        required_mcp_servers=required_mcp_servers,
        isolation=cast(Any, isolation),
        context_policy=context_policy or AgentContextPolicy.inherit(),
    )


def _ctx(
    *,
    tools: Sequence[Tool] = (),
    permission_context: ToolPermissionContext | None = None,
    agent_id: str | None = "parent",
) -> ToolUseContext:
    return ToolUseContext(
        session_id="parent-session",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="parent system",
        cwd=".",
        tools=tuple(tools),
        permission_context=permission_context or ToolPermissionContext(),
        query_tracking=QueryTracking(chain_id="parent", depth=0),
    )


def _config(tools: Sequence[Tool] = ()) -> QueryConfig:
    return QueryConfig(
        model="parent-model",
        system_prompt="parent system",
        tools=tuple(tools),
        session_id="parent-session",
        agent_id="parent",
    )


def _deps(
    *,
    store: AppStateStore | None = None,
    permission_context: ToolPermissionContext | None = None,
    model_provider: Any | None = None,
    worktree_manager: Any | None = None,
    remote_agent_backend: Any | None = None,
    handoff_classifier: Any | None = None,
    handoff_classifier_timeout_s: float | None = None,
    coordinator_runtime: Any | None = None,
    transcript_store: Any | None = None,
    context_providers: tuple[Any, ...] = (),
    post_tool_context_providers: tuple[Any, ...] = (),
) -> QueryDeps:
    kwargs: dict[str, Any] = {}
    if model_provider is not None:
        kwargs["model_provider"] = model_provider
    if worktree_manager is not None:
        kwargs["worktree_manager"] = worktree_manager
    if remote_agent_backend is not None:
        kwargs["remote_agent_backend"] = remote_agent_backend
    if handoff_classifier is not None:
        kwargs["handoff_classifier"] = handoff_classifier
    if handoff_classifier_timeout_s is not None:
        kwargs["handoff_classifier_timeout_s"] = handoff_classifier_timeout_s
    if coordinator_runtime is not None:
        kwargs["coordinator_runtime"] = coordinator_runtime
    if transcript_store is not None:
        kwargs["transcript_store"] = transcript_store
    return QueryDeps(
        task_store=store or AppStateStore(),
        permission_context=permission_context or ToolPermissionContext(),
        context_providers=cast(Any, context_providers),
        post_tool_context_providers=cast(Any, post_tool_context_providers),
        **kwargs,
    )


class FakeWorktreeManager:
    def __init__(
        self,
        *,
        cleanup_result: WorktreeCleanupResult | None = None,
    ) -> None:
        self.created: list[tuple[str, str]] = []
        self.cleaned: list[WorktreeInfo] = []
        self.cleanup_result = cleanup_result

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        self.created.append((slug, cwd))
        return WorktreeInfo(
            path=f"/tmp/raygent/{slug}",
            branch=f"worktree-{slug}",
            head_commit="abc123",
            git_root=cwd,
            slug=slug,
            created_at=123.0,
            touched_at=456.0,
            cleanup_policy="remove_if_clean",
        )

    async def has_changes(self, _info: WorktreeInfo) -> bool:
        return False

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        self.cleaned.append(info)
        if keep is True:
            return WorktreeCleanupResult(
                kept=True,
                reason="kept",
                path=info.path,
                branch=info.branch,
            )
        if self.cleanup_result is not None:
            return self.cleanup_result
        return WorktreeCleanupResult(kept=False, reason="removed")


class FakeRemoteBackend:
    def __init__(self, polls: Sequence[RemoteAgentPollResult]) -> None:
        self.launches: list[RemoteAgentLaunchRequest] = []
        self.poll_requests: list[RemoteAgentPollRequest] = []
        self.polls = list(polls)
        self.stop_requests: list[RemoteAgentStopRequest] = []

    async def launch(
        self,
        request: RemoteAgentLaunchRequest,
    ) -> RemoteAgentLaunchResult:
        self.launches.append(request)
        return RemoteAgentLaunchResult(
            remote_id=f"remote-{request.task_id}",
            title=request.description,
            session_url=f"https://example.test/sessions/{request.task_id}",
        )

    async def poll(self, request: RemoteAgentPollRequest) -> RemoteAgentPollResult:
        self.poll_requests.append(request)
        if self.polls:
            return self.polls.pop(0)
        return RemoteAgentPollResult(status="running")

    async def stop(self, request: RemoteAgentStopRequest) -> None:
        self.stop_requests.append(request)


class HangingStopRemoteBackend(FakeRemoteBackend):
    def __init__(self) -> None:
        super().__init__(polls=(RemoteAgentPollResult(status="running"),))
        self.stop_started = asyncio.Event()

    async def stop(self, request: RemoteAgentStopRequest) -> None:
        self.stop_requests.append(request)
        self.stop_started.set()
        await asyncio.Event().wait()


class WarningClassifier:
    def __init__(
        self,
        warning: str = "SECURITY WARNING: verify handoff",
        *,
        block: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        self.warning = warning
        self.block = block
        self.fail = fail
        self.requests: list[HandoffClassificationRequest] = []

    async def classify(
        self,
        request: HandoffClassificationRequest,
    ) -> HandoffClassificationResult:
        self.requests.append(request)
        if self.block is not None:
            await self.block.wait()
        if self.fail:
            raise RuntimeError("classifier unavailable")
        return HandoffClassificationResult(warning=self.warning, decision="blocked")


def _tool_use(input_: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        id="toolu_agent",
        name=AGENT_TOOL_NAME,
        input=input_,
        index=0,
    )


async def _run_agent_tool(
    *,
    tool: Tool,
    deps: QueryDeps,
    ctx: ToolUseContext,
    input_: dict[str, Any],
) -> list[ToolExecutionResult]:
    events = [
        event
        async for event in run_tool_use(
            tool_use=_tool_use(input_),
            assistant_message={"role": "assistant", "content": []},
            tools=ctx.tools,
            deps=deps,
            ctx=ctx,
        )
    ]
    return [event for event in events if isinstance(event, ToolExecutionResult)]


async def _wait_for_notification(
    store: AppStateStore,
    agent_id: str | None,
) -> None:
    while not store.drain_notifications(agent_id):
        await asyncio.sleep(0.01)


async def _drain_notification(
    store: AppStateStore,
    agent_id: str | None,
) -> Any:
    while True:
        notifications = store.drain_notifications(agent_id)
        if notifications:
            return notifications[0]
        await asyncio.sleep(0.01)


def _result_content(result: ToolExecutionResult) -> str | list[dict[str, Any]]:
    content = result.message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return block["content"]  # type: ignore[no-any-return]


def _launch_data(result: ToolExecutionResult) -> dict[str, Any]:
    content = _result_content(result)
    assert isinstance(content, list)
    launch = content[1]
    assert launch["type"] == "agent_launch"
    return launch


@pytest.mark.asyncio
async def test_agent_tool_unknown_and_denied_agents_are_model_visible_errors() -> None:
    store = AppStateStore()
    deps = _deps(store=store)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    unknown = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "do work", "subagent_type": "missing"},
    )
    assert "Agent type 'missing' not found" in str(_result_content(unknown[0]))

    denied_context = ToolPermissionContext(
        always_deny_rules={"localSettings": ("Agent(worker)",)}
    )
    denied_deps = _deps(store=store, permission_context=denied_context)
    denied = await _run_agent_tool(
        tool=agent_tool,
        deps=denied_deps,
        ctx=_ctx(tools=(agent_tool,)),
        input_={"prompt": "do work", "subagent_type": "worker"},
    )
    assert "has been denied" in str(_result_content(denied[0]))


@pytest.mark.asyncio
async def test_agent_tool_launches_background_local_agent_with_child_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[dict[str, Any]] = []

    class CapturingEngine:
        def __init__(
            self,
            config: QueryConfig,
            deps: QueryDeps,
            ctx: ToolUseContext,
        ) -> None:
            captures.append({"config": config, "deps": deps, "ctx": ctx, "prompt": None})

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            captures[-1]["prompt"] = prompt
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"finished {prompt}",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)

    read = _base_tool("Read")
    write = _base_tool("Write")
    store = AppStateStore()
    deps = _deps(store=store)
    agent = _agent_definition(
        tools=("Read", "Write", "Missing"),
        disallowed_tools=("Write",),
        system_prompt="child system",
        model="child-model",
        permission_mode="plan",
        initial_prompt="prep first",
    )
    agent_tool = build_agent_tool(
        parent_config=_config((read, write)),
        parent_deps=deps,
        agent_definitions=(agent,),
    )
    ctx = _ctx(tools=(read, write, agent_tool))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do implementation",
            "description": "implementation worker",
            "subagent_type": "worker",
            "model": "requested-model",
        },
    )
    launch = _launch_data(results[0])
    task_id = cast(str, launch["agent_id"])
    final = await run_until_done(task_id, store)

    assert launch["status"] == "async_launched"
    assert launch["agent_type"] == "worker"
    assert launch["description"] == "implementation worker"
    assert launch["invalid_tools"] == ["Write", "Missing"]
    assert final.status == "completed"
    assert captures[0]["prompt"] == "prep first\n\ndo implementation"

    child_config = cast(QueryConfig, captures[0]["config"])
    child_deps = cast(QueryDeps, captures[0]["deps"])
    child_ctx = cast(ToolUseContext, captures[0]["ctx"])
    assert child_config.agent_id == task_id
    assert child_config.model == "requested-model"
    assert child_config.system_prompt == "child system"
    assert tuple(tool.name for tool in child_config.tools) == ("Read",)
    assert child_deps.permission_context.mode == "plan"
    assert child_ctx.agent_id == task_id
    assert child_ctx.query_tracking is not None
    assert child_ctx.query_tracking.depth == 1

    notifications = store.drain_notifications("parent")
    assert len(notifications) == 1
    assert notifications[0].tool_use_id == "toolu_agent"
    assert notifications[0].agent_id == "parent"
    assert "<status>completed" in notifications[0].message


@pytest.mark.asyncio
async def test_agent_tool_background_filters_child_context_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[QueryDeps] = []

    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            captures.append(deps)

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="done",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    project = RecordingContextProvider("project_instructions", "project")
    git = RecordingContextProvider("git", "git")
    custom = RecordingContextProvider("custom", "custom")
    post_project = RecordingContextProvider("project_instructions", "post-project")
    post_custom = RecordingContextProvider("custom", "post-custom")
    deps = _deps(
        store=store,
        context_providers=(project, git, custom),
        post_tool_context_providers=(post_project, post_custom),
    )
    agent_tool = build_agent_tool(
        parent_config=_config((_base_tool("Read"),)),
        parent_deps=deps,
        agent_definitions=(
            _agent_definition(context_policy=AgentContextPolicy.minimal()),
        ),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "background", "subagent_type": "worker"},
    )
    await run_until_done(cast(str, _launch_data(results[0])["agent_id"]), store)

    assert captures
    child_deps = captures[0]
    assert child_deps.task_store is store
    assert child_deps.context_providers == (custom,)
    assert child_deps.post_tool_context_providers == (post_custom,)


@pytest.mark.asyncio
async def test_agent_tool_records_successful_launch_in_coordinator_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", BlockingEngine)

    runtime = CoordinatorRuntime()
    store = AppStateStore()
    deps = _deps(store=store, coordinator_runtime=runtime)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "coordinate worker",
            "description": "coordination worker",
            "subagent_type": "worker",
        },
    )

    launch = _launch_data(results[0])
    work_item_id = launch["coordinator_work_item_id"]
    task_id = cast(str, launch["agent_id"])
    snapshot = runtime.snapshot()

    assert work_item_id == f"cw_agent_{task_id}"
    assert len(snapshot.work_items) == 1
    assert snapshot.work_items[0].id == work_item_id
    assert snapshot.work_items[0].task_id == task_id
    assert snapshot.work_items[0].agent_id == task_id
    assert snapshot.work_items[0].agent_type == "worker"
    assert snapshot.work_items[0].status == "running"

    await LocalAgentTask().kill(task_id, store)
    final = await run_until_done(task_id, store)
    assert final.status == "killed"


@pytest.mark.asyncio
async def test_agent_tool_validation_failure_does_not_record_coordinator_work() -> None:
    runtime = CoordinatorRuntime()
    store = AppStateStore()
    deps = _deps(store=store, coordinator_runtime=runtime)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "do work", "subagent_type": "missing"},
    )

    assert "Agent type 'missing' not found" in str(_result_content(results[0]))
    assert runtime.snapshot().work_items == ()
    assert runtime.snapshot().blackboard_entries == ()


@pytest.mark.asyncio
async def test_agent_tool_remote_launch_polls_completion_and_notifies_parent() -> None:
    backend = FakeRemoteBackend(
        polls=(
            RemoteAgentPollResult(status="running", metadata={"phase": "started"}),
            RemoteAgentPollResult(status="completed", message="remote result"),
        )
    )
    store = AppStateStore()
    deps = _deps(store=store, remote_agent_backend=backend)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do remote work",
            "subagent_type": "worker",
            "isolation": "remote",
        },
    )
    launch = _launch_data(results[0])
    task_id = cast(str, launch["agent_id"])
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=1.0,
    )
    task = store.tasks[task_id]

    assert task_id.startswith("r")
    assert launch["status"] == "remote_launched"
    assert launch["is_remote"] is True
    assert launch["session_url"] == f"https://example.test/sessions/{task_id}"
    assert backend.launches[0].task_id == task_id
    assert backend.launches[0].prompt == "do remote work"
    assert isinstance(task, RemoteAgentState)
    assert task.status == "completed"
    assert task.metadata == {"phase": "started"}
    assert backend.poll_requests[0].task_id == task_id
    assert backend.poll_requests[0].remote_id == f"remote-{task_id}"
    assert notification.agent_id == "parent"
    assert notification.tool_use_id == "toolu_agent"
    assert "<task_type>remote_agent</task_type>" in notification.message
    assert "<result>remote result</result>" in notification.message


@pytest.mark.asyncio
async def test_remote_agent_stop_calls_backend_and_suppresses_notification() -> None:
    backend = FakeRemoteBackend(polls=(RemoteAgentPollResult(status="running"),))
    store = AppStateStore()
    deps = _deps(store=store, remote_agent_backend=backend)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "keep remote running",
            "subagent_type": "worker",
            "isolation": "remote",
        },
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    await RemoteAgentTask().kill(task_id, store)

    task = store.tasks[task_id]
    assert isinstance(task, RemoteAgentState)
    assert task.status == "killed"
    assert task.notified is True
    await asyncio.sleep(0)
    assert backend.stop_requests[0].task_id == task_id
    assert backend.stop_requests[0].remote_id == f"remote-{task_id}"
    assert store.drain_notifications("parent") == []


@pytest.mark.asyncio
async def test_remote_agent_stop_is_fire_and_forget_when_backend_hangs() -> None:
    backend = HangingStopRemoteBackend()
    store = AppStateStore()
    deps = _deps(store=store, remote_agent_backend=backend)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "keep remote running",
            "subagent_type": "worker",
            "isolation": "remote",
        },
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])

    await asyncio.wait_for(RemoteAgentTask().kill(task_id, store), timeout=0.1)
    await asyncio.wait_for(backend.stop_started.wait(), timeout=0.1)

    task = store.tasks[task_id]
    assert isinstance(task, RemoteAgentState)
    assert task.status == "killed"
    assert task.notified is True
    assert backend.stop_requests[0].task_id == task_id
    assert store.drain_notifications("parent") == []


@pytest.mark.asyncio
async def test_remote_agent_handoff_warning_prepends_notification() -> None:
    backend = FakeRemoteBackend(
        polls=(RemoteAgentPollResult(status="completed", message="remote result"),)
    )
    classifier = WarningClassifier("REMOTE WARNING")
    store = AppStateStore()
    deps = _deps(
        store=store,
        remote_agent_backend=backend,
        handoff_classifier=classifier,
    )
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do remote work",
            "subagent_type": "worker",
            "isolation": "remote",
        },
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=1.0,
    )

    assert classifier.requests[0].task_id == task_id
    assert classifier.requests[0].task_type == "remote_agent"
    assert classifier.requests[0].agent_type == "worker"
    assert "REMOTE WARNING" in notification.message
    assert "remote result" in notification.message


@pytest.mark.asyncio
async def test_background_agent_handoff_warning_prepends_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKAssistantMessage(
                session_id="child",
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_child",
                            "name": "Read",
                            "input": {},
                        }
                    ],
                },
            )
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="agent result",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)
    classifier = WarningClassifier()
    read = _base_tool("Read")
    store = AppStateStore()
    deps = _deps(store=store, handoff_classifier=classifier)
    agent_tool = build_agent_tool(
        parent_config=_config((read,)),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(read, agent_tool))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "do work", "subagent_type": "worker"},
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=1.0,
    )

    assert classifier.requests[0].task_id == task_id
    assert classifier.requests[0].agent_type == "worker"
    assert classifier.requests[0].tool_names == ("Read",)
    assert classifier.requests[0].permission_mode == "acceptEdits"
    assert classifier.requests[0].total_tool_use_count == 1
    assert classifier.requests[0].messages[0]["role"] == "assistant"
    assert "SECURITY WARNING: verify handoff" in notification.message
    assert "agent result" in notification.message


@pytest.mark.asyncio
async def test_foreground_agent_handoff_warning_prepends_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
            **_kwargs: Any,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKAssistantMessage(
                session_id="child",
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_child",
                            "name": "Read",
                            "input": {},
                        }
                    ],
                },
            )
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="foreground result",
            )

    monkeypatch.setattr(child_query_mod, "QueryEngine", CapturingEngine)
    classifier = WarningClassifier("SYNC WARNING")
    read = _base_tool("Read")
    store = AppStateStore()
    deps = _deps(store=store, handoff_classifier=classifier)
    agent_tool = build_agent_tool(
        parent_config=_config((read,)),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(read, agent_tool))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do foreground work",
            "subagent_type": "worker",
            "run_in_background": False,
        },
    )

    assert classifier.requests[0].task_type == "local_agent"
    assert classifier.requests[0].agent_type == "worker"
    assert classifier.requests[0].tool_names == ("Read",)
    assert classifier.requests[0].permission_mode == "acceptEdits"
    assert classifier.requests[0].total_tool_use_count == 1
    assert classifier.requests[0].messages[0]["role"] == "assistant"
    content = _result_content(results[0])
    assert isinstance(content, list)
    assert "SYNC WARNING" in content[0]["text"]
    assert "foreground result" in content[0]["text"]
    assert content[1]["result"].startswith("SYNC WARNING")


@pytest.mark.asyncio
async def test_handoff_classifier_timeout_is_fail_soft_after_status_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="agent result",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)
    classifier = WarningClassifier(block=asyncio.Event())
    store = AppStateStore()
    deps = _deps(
        store=store,
        handoff_classifier=classifier,
        handoff_classifier_timeout_s=0.01,
    )
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "do work", "subagent_type": "worker"},
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    final = await run_until_done(task_id, store)
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=1.0,
    )

    assert final.status == "completed"
    assert classifier.requests[0].task_id == task_id
    assert "SECURITY WARNING" not in notification.message
    assert "agent result" in notification.message


@pytest.mark.asyncio
async def test_handoff_classifier_failure_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="agent result",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)
    classifier = WarningClassifier(fail=True)
    store = AppStateStore()
    deps = _deps(store=store, handoff_classifier=classifier)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "do work", "subagent_type": "worker"},
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    notification = await asyncio.wait_for(
        _drain_notification(store, "parent"),
        timeout=1.0,
    )

    assert classifier.requests[0].task_id == task_id
    assert "SECURITY WARNING" not in notification.message
    assert "agent result" in notification.message


@pytest.mark.asyncio
async def test_agent_tool_named_background_launch_registers_teammate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    class CapturingEngine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    read = _base_tool("Read")
    store = AppStateStore()
    deps = _deps(store=store)
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    team_store.create_team(
        team_name="My Team",
        description=None,
        agent_type="coordinator",
        model="parent-model",
        cwd="/repo",
    )
    agent_tool = build_agent_tool(
        parent_config=replace(_config((read,)), experiments={"fork_subagent": True}),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
        team_store=team_store,
    )
    ctx = _ctx(tools=(read, agent_tool), agent_id=None)

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "research the topic",
            "description": "research worker",
            "name": "Researcher",
        },
    )

    content = _result_content(results[0])
    assert isinstance(content, list)
    launch = content[1]
    assert launch["type"] == "teammate_launch"
    assert launch["agent_type"] == "worker"
    assert launch["name"] == "researcher"
    task_id = cast(str, launch["task_id"])
    for _ in range(100):
        task = store.tasks[task_id]
        if isinstance(task, InProcessTeammateState) and task.is_idle:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("teammate did not become idle")

    task = cast(InProcessTeammateState, store.tasks[task_id])
    assert task.status == "running"
    assert task.identity is not None
    assert task.identity.agent_id == "researcher@my-team"
    assert task.identity.agent_type == "worker"
    assert store.agent_name_registry["researcher"] == task_id
    assert team_store.current_team is not None
    assert [member.name for member in team_store.current_team.members] == [
        "team-lead",
        "researcher",
    ]


@pytest.mark.asyncio
async def test_agent_tool_teammate_filters_child_context_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    captures: list[QueryDeps] = []

    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            captures.append(deps)

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    project = RecordingContextProvider("project_instructions", "project")
    git = RecordingContextProvider("git", "git")
    custom = RecordingContextProvider("custom", "custom")
    deps = _deps(
        store=store,
        context_providers=(project, git, custom),
    )
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    team_store.create_team(
        team_name="My Team",
        description=None,
        agent_type="coordinator",
        model="parent-model",
        cwd="/repo",
    )
    agent_tool = build_agent_tool(
        parent_config=replace(
            _config((_base_tool("Read"),)),
            experiments={"fork_subagent": True},
        ),
        parent_deps=deps,
        agent_definitions=(
            _agent_definition(context_policy=AgentContextPolicy.omit_git()),
        ),
        team_store=team_store,
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "research the topic",
            "description": "research worker",
            "name": "Researcher",
        },
    )

    content = _result_content(results[0])
    assert isinstance(content, list)
    assert content[1]["type"] == "teammate_launch"
    for _ in range(100):
        if captures:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("teammate engine did not start")

    assert captures
    assert captures[0].context_providers == (project, custom)


@pytest.mark.asyncio
async def test_agent_tool_named_teammate_cleanup_allows_relaunch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    class CapturingEngine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    deps = _deps(store=store)
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    team_store.create_team(
        team_name="My Team",
        description=None,
        agent_type="coordinator",
        model="parent-model",
        cwd="/repo",
    )
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
        team_store=team_store,
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    first = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "research the topic", "name": "Researcher"},
    )
    first_content = _result_content(first[0])
    assert isinstance(first_content, list)
    first_task_id = cast(str, first_content[1]["task_id"])
    for _ in range(100):
        task = store.tasks[first_task_id]
        if isinstance(task, InProcessTeammateState) and task.is_idle:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("teammate did not become idle")

    await InProcessTeammateTask().kill(first_task_id, store)
    for _ in range(100):
        if "researcher" not in store.agent_name_registry:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("teammate registry was not cleaned")

    assert team_store.current_team is not None
    assert [member.name for member in team_store.current_team.members] == ["team-lead"]
    data = json.loads(
        (tmp_path / "teams" / "my-team" / "config.json").read_text(encoding="utf-8")
    )
    assert [member["name"] for member in data["members"]] == ["team-lead"]

    second = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "research more", "name": "Researcher"},
    )
    second_content = _result_content(second[0])
    assert isinstance(second_content, list)
    assert second_content[1]["name"] == "researcher"


@pytest.mark.asyncio
async def test_agent_tool_named_duplicate_teammate_names_are_suffixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    class CapturingEngine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"done {prompt}",
            )

    monkeypatch.setattr(teammate_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    deps = _deps(store=store)
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    team_store.create_team(
        team_name="My Team",
        description=None,
        agent_type="coordinator",
        model="parent-model",
        cwd="/repo",
    )
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
        team_store=team_store,
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "research the topic", "name": "Researcher"},
    )
    duplicate = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "research more", "name": "Researcher"},
    )

    content = _result_content(duplicate[0])
    assert isinstance(content, list)
    assert content[1]["name"] == "researcher-2"
    assert content[1]["agent_id"] == "researcher-2@my-team"
    assert team_store.current_team is not None
    assert [member.name for member in team_store.current_team.members] == [
        "team-lead",
        "researcher",
        "researcher-2",
    ]


@pytest.mark.asyncio
async def test_agent_tool_named_background_without_team_registers_local_agent_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[QueryDeps] = []

    class BlockingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            captures.append(deps)

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", BlockingEngine)

    read = _base_tool("Read")
    store = AppStateStore()
    project = RecordingContextProvider("project_instructions", "project")
    git = RecordingContextProvider("git", "git")
    custom = RecordingContextProvider("custom", "custom")
    deps = _deps(
        store=store,
        context_providers=(project, git, custom),
    )
    agent_tool = build_agent_tool(
        parent_config=_config((read,)),
        parent_deps=deps,
        agent_definitions=(
            _agent_definition(
                model="child-model",
                context_policy=AgentContextPolicy.minimal(),
            ),
        ),
    )
    ctx = _ctx(tools=(read, agent_tool), agent_id=None)

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "research the topic",
            "description": "research worker",
            "name": "Researcher",
        },
    )

    launch = _launch_data(results[0])
    task_id = cast(str, launch["agent_id"])
    task = store.tasks[task_id]
    record = store.agent_route_records[task_id]

    assert isinstance(task, LocalAgentState)
    assert task.name == "Researcher"
    assert launch["agent_name"] == "Researcher"
    assert store.agent_name_registry["Researcher"] == task_id
    assert record.task_type == "local_agent"
    assert record.name == "Researcher"
    assert record.agent_type == "worker"
    assert record.model == "child-model"
    assert record.tool_names == ("Read",)
    assert record.parent_session_id == "parent-session"
    for _ in range(100):
        if captures:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("local agent engine did not start")

    assert captures
    assert captures[0].context_providers == (custom,)

    await LocalAgentTask().kill(task_id, store)
    final = await run_until_done(task_id, store)
    assert final.status == "killed"
    assert store.agent_name_registry["Researcher"] == task_id
    assert store.agent_route_records[task_id] == record


@pytest.mark.asyncio
async def test_agent_tool_duplicate_non_team_names_use_latest_wins_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingEngine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", BlockingEngine)

    store = AppStateStore()
    deps = _deps(store=store)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    first = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "first topic", "name": "Researcher"},
    )
    second = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "second topic", "name": "Researcher"},
    )
    first_id = cast(str, _launch_data(first[0])["agent_id"])
    second_id = cast(str, _launch_data(second[0])["agent_id"])

    assert first_id != second_id
    assert store.agent_name_registry["Researcher"] == second_id
    assert store.agent_route_records[first_id].name == "Researcher"
    assert store.agent_route_records[second_id].name == "Researcher"

    await LocalAgentTask().kill(first_id, store)
    await LocalAgentTask().kill(second_id, store)
    assert (await run_until_done(first_id, store)).status == "killed"
    assert (await run_until_done(second_id, store)).status == "killed"


@pytest.mark.asyncio
async def test_agent_tool_parent_abort_does_not_kill_background_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", BlockingEngine)

    store = AppStateStore()
    deps = _deps(store=store)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "keep running", "subagent_type": "worker"},
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])

    ctx.abort_event.set()
    await asyncio.sleep(0.05)
    running = store.tasks[task_id]
    assert isinstance(running, LocalAgentState)
    assert running.status == "running"

    await LocalAgentTask().kill(task_id, store)
    final = await run_until_done(task_id, store)
    assert final.status == "killed"


@pytest.mark.asyncio
async def test_agent_tool_direct_call_rejects_named_background_isolation() -> None:
    manager = FakeWorktreeManager()
    store = AppStateStore()
    deps = _deps(store=store, worktree_manager=manager)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(isolation="worktree"),),
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    direct_events = [
        event
        async for event in agent_tool.call(
            AgentToolInput(
                prompt="research the topic",
                name="Researcher",
                subagent_type="worker",
            ),
            ctx,
        )
    ]

    assert len(direct_events) == 1
    direct_result = direct_events[0]
    assert isinstance(direct_result, ToolResult)
    assert direct_result.is_error is True
    assert "does not support isolation" in str(direct_result.content)
    assert manager.created == []
    assert store.tasks == {}
    assert store.agent_route_records == {}
    assert store.agent_name_registry == {}


@pytest.mark.asyncio
async def test_agent_tool_background_worktree_threads_cwd_and_kept_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[dict[str, Any]] = []

    class CapturingEngine:
        def __init__(
            self,
            config: QueryConfig,
            deps: QueryDeps,
            ctx: ToolUseContext,
        ) -> None:
            captures.append({"config": config, "deps": deps, "ctx": ctx, "prompt": None})

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            captures[-1]["prompt"] = prompt
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="background isolated result",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)

    manager = FakeWorktreeManager(
        cleanup_result=WorktreeCleanupResult(
            kept=True,
            reason="changed",
            path="/tmp/raygent/kept-worktree",
            branch="worktree-kept",
        )
    )
    store = AppStateStore()
    deps = _deps(store=store, worktree_manager=manager)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do isolated background work",
            "subagent_type": "worker",
            "isolation": "worktree",
        },
    )
    launch = _launch_data(results[0])
    task_id = cast(str, launch["agent_id"])
    final = await run_until_done(task_id, store)

    assert launch["worktree_path"].startswith("/tmp/raygent/agent-a")
    assert launch["worktree_branch"].startswith("worktree-agent-a")
    assert launch["worktree_slug"].startswith("agent-a")
    assert launch["worktree_owner_task_id"] == task_id
    assert launch["worktree_created_at"] == 123.0
    assert launch["worktree_touched_at"] == 456.0
    assert launch["worktree_cleanup_policy"] == "remove_if_clean"
    assert len(manager.cleaned) == 1
    assert final.status == "completed"
    assert isinstance(final, LocalAgentState)
    assert final.worktree_path == "/tmp/raygent/kept-worktree"
    assert final.worktree_branch == "worktree-kept"
    assert final.worktree_slug == launch["worktree_slug"]
    assert final.worktree_cleanup_policy == "remove_if_clean"
    child_ctx = cast(ToolUseContext, captures[0]["ctx"])
    assert child_ctx.cwd == launch["worktree_path"]

    notifications = store.drain_notifications("parent")
    assert len(notifications) == 1
    assert "<worktreePath>/tmp/raygent/kept-worktree</worktreePath>" in notifications[
        0
    ].message
    assert "<worktreeBranch>worktree-kept</worktreeBranch>" in notifications[0].message


@pytest.mark.asyncio
async def test_agent_tool_background_worktree_status_precedes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
        ) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="background isolated result",
            )

    class SlowCleanupWorktreeManager(FakeWorktreeManager):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def cleanup(
            self,
            info: WorktreeInfo,
            *,
            keep: bool | None = None,
        ) -> WorktreeCleanupResult:
            self.cleaned.append(info)
            self.started.set()
            await self.release.wait()
            return WorktreeCleanupResult(kept=False, reason="removed")

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)

    manager = SlowCleanupWorktreeManager()
    store = AppStateStore()
    deps = _deps(store=store, worktree_manager=manager)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do isolated background work",
            "subagent_type": "worker",
            "isolation": "worktree",
        },
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])

    await asyncio.wait_for(manager.started.wait(), timeout=1.0)
    task = store.tasks[task_id]
    assert isinstance(task, LocalAgentState)
    assert task.status == "completed"
    assert store.drain_notifications("parent") == []

    manager.release.set()
    await asyncio.wait_for(_wait_for_notification(store, "parent"), timeout=1.0)
    final = store.tasks[task_id]
    assert isinstance(final, LocalAgentState)
    assert final.worktree_path is None
    assert final.worktree_branch is None
    assert final.worktree_slug is None
    assert final.worktree_created_at is None
    assert final.worktree_touched_at is None
    assert final.worktree_cleanup_policy is None
    route = store.agent_route_records[task_id]
    assert route.worktree_path is None
    assert route.worktree_branch is None
    assert route.worktree_slug is None
    assert route.worktree_cleanup_policy is None


@pytest.mark.asyncio
async def test_agent_tool_runs_foreground_child_query_without_task_notification() -> None:
    read = _base_tool("Read")
    store = AppStateStore()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "foreground result"},)
    )
    deps = _deps(store=store, model_provider=provider)
    agent = _agent_definition(
        tools=("Read", "Missing"),
        system_prompt="foreground system",
        model="child-model",
        permission_mode="plan",
        initial_prompt="prep first",
    )
    agent_tool = build_agent_tool(
        parent_config=_config((read,)),
        parent_deps=deps,
        agent_definitions=(agent,),
    )
    ctx = _ctx(tools=(read, agent_tool))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do foreground work",
            "subagent_type": "worker",
            "run_in_background": False,
        },
    )

    content = _result_content(results[0])
    assert isinstance(content, list)
    metadata = content[1]
    assert metadata["type"] == "agent_result"
    assert metadata["status"] == "completed"
    assert metadata["agent_type"] == "worker"
    assert metadata["result"] == "foreground result"
    assert metadata["is_fork"] is False
    assert store.tasks == {}
    assert store.drain_notifications("parent") == []

    request = provider.requests[0]
    assert request.model == "child-model"
    assert request.system_prompt == "foreground system"
    assert request.permission_context is not None
    assert request.permission_context.mode == "plan"
    assert tuple(tool.name for tool in request.tools) == ("Read",)
    assert request.messages[-1].provider_payload == {
        "role": "user",
        "content": "prep first\n\ndo foreground work",
    }


@pytest.mark.asyncio
async def test_agent_tool_foreground_filters_child_context_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[QueryDeps] = []

    class CapturingEngine:
        def __init__(
            self,
            _config: QueryConfig,
            deps: QueryDeps,
            _ctx: ToolUseContext,
            **_kwargs: Any,
        ) -> None:
            captures.append(deps)

        async def seed_messages(self, _messages: Sequence[Any]) -> None:
            pass

        async def submit_message(self, _message: Any) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="foreground result",
            )

    monkeypatch.setattr(child_query_mod, "QueryEngine", CapturingEngine)

    store = AppStateStore()
    project = RecordingContextProvider("project_instructions", "project")
    git = RecordingContextProvider("git", "git")
    custom = RecordingContextProvider("custom", "custom")
    deps = _deps(
        store=store,
        context_providers=(project, git, custom),
    )
    agent_tool = build_agent_tool(
        parent_config=_config((_base_tool("Read"),)),
        parent_deps=deps,
        agent_definitions=(
            _agent_definition(context_policy=AgentContextPolicy.omit_git()),
        ),
    )
    ctx = _ctx(tools=(agent_tool,))

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do foreground work",
            "subagent_type": "worker",
            "run_in_background": False,
        },
    )

    content = _result_content(results[0])
    assert isinstance(content, list)
    assert content[1]["status"] == "completed"
    assert captures
    assert captures[0].context_providers == (project, custom)


@pytest.mark.asyncio
async def test_agent_tool_foreground_worktree_sets_child_cwd_and_cleans_removed() -> None:
    manager = FakeWorktreeManager()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "worktree result"},)
    )
    seen_cwd: list[str] = []

    async def system_prompt_provider(
        _config: QueryConfig,
        ctx: ToolUseContext,
    ) -> str | None:
        seen_cwd.append(ctx.cwd)
        return None

    deps = _deps(model_provider=provider, worktree_manager=manager)
    deps.system_prompt_provider = system_prompt_provider
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(isolation="worktree"),),
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do isolated foreground work",
            "subagent_type": "worker",
            "run_in_background": False,
        },
    )

    assert manager.created
    slug, created_from_cwd = manager.created[0]
    worktree_path = f"/tmp/raygent/{slug}"
    assert slug.startswith("agent-a")
    assert created_from_cwd == "."
    assert seen_cwd == [worktree_path]
    assert len(manager.cleaned) == 1

    metadata = cast(list[dict[str, Any]], _result_content(results[0]))[1]
    assert metadata["type"] == "agent_result"
    assert metadata["worktree_kept"] is False
    assert metadata["worktree_cleanup_reason"] == "removed"
    assert "worktree_path" not in metadata
    assert deps.task_store.tasks == {}


@pytest.mark.asyncio
async def test_agent_tool_foreground_inherits_current_effective_model() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "foreground result"},)
    )
    deps = _deps(model_provider=provider)
    agent = _agent_definition(model="inherit")
    config = _config()
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(agent,),
    )
    ctx = _ctx(tools=(agent_tool,))
    ctx = replace(
        ctx,
        runtime=ToolRuntimeContext(
            config=config,
            deps=deps,
            effective_model="fallback-parent-model",
        ),
    )

    await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "do foreground work",
            "subagent_type": "worker",
            "run_in_background": False,
        },
    )

    assert provider.requests[0].model == "fallback-parent-model"


@pytest.mark.asyncio
async def test_agent_tool_fork_subagent_builds_inherited_prefix() -> None:
    read = _base_tool("Read")
    store = AppStateStore()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "fork result"},)
    )
    deps = _deps(store=store, model_provider=provider)
    config = _config((read,))
    config = replace(config, experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_agent_a",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate A"},
            },
            {
                "type": "tool_use",
                "id": "toolu_agent_b",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate B"},
            },
        ],
    }
    ctx = _ctx(tools=(read, agent_tool), agent_id=None)
    ctx = replace(
        ctx,
        messages=[{"role": "user", "content": "parent context"}],
    )

    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_agent_a",
                name=AGENT_TOOL_NAME,
                input={"prompt": "investigate A"},
                index=0,
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(read, agent_tool),
            deps=deps,
            ctx=ctx,
        )
    ]
    results = [event for event in events if isinstance(event, ToolExecutionResult)]

    metadata = cast(list[dict[str, Any]], _result_content(results[0]))[1]
    assert metadata["status"] == "completed"
    assert metadata["agent_type"] == "fork"
    assert metadata["is_fork"] is True
    assert store.tasks == {}

    request = provider.requests[0]
    assert request.system_prompt == "parent system"
    assert tuple(tool.name for tool in request.tools) == ("Read", AGENT_TOOL_NAME)
    assert request.permission_context is not None
    assert request.permission_context.mode == "bubble"
    payloads = [
        thaw_json(message.provider_payload)
        if message.provider_payload is not None
        else None
        for message in request.messages
    ]
    assert payloads[0] == {"role": "user", "content": "parent context"}
    assert payloads[1] == assistant_message
    fork_user = cast(dict[str, Any], payloads[2])
    assert fork_user["role"] == "user"
    fork_content = cast(list[dict[str, Any]], fork_user["content"])
    assert [block["tool_use_id"] for block in fork_content[:2]] == [
        "toolu_agent_a",
        "toolu_agent_b",
    ]
    assert "Directive: investigate A" in fork_content[2]["text"]


@pytest.mark.asyncio
async def test_agent_tool_fork_can_handoff_to_background_local_agent(
    tmp_path: Any,
) -> None:
    read = _base_tool("Read")
    store = AppStateStore()
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    runtime = CoordinatorRuntime()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "fork background result"},)
    )
    deps = _deps(
        store=store,
        model_provider=provider,
        transcript_store=transcript_store,
        coordinator_runtime=runtime,
    )
    config = replace(_config((read,)), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_agent_a",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate A"},
            },
            {
                "type": "tool_use",
                "id": "toolu_agent_b",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate B"},
            },
        ],
    }
    ctx = replace(
        _ctx(tools=(read, agent_tool), agent_id=None),
        messages=[{"role": "user", "content": "parent context"}],
    )

    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_agent_a",
                name=AGENT_TOOL_NAME,
                input={"prompt": "investigate A", "run_in_background": True},
                index=0,
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(read, agent_tool),
            deps=deps,
            ctx=ctx,
        )
    ]
    results = [event for event in events if isinstance(event, ToolExecutionResult)]

    launch = cast(list[dict[str, Any]], _result_content(results[0]))[1]
    task_id = cast(str, launch["agent_id"])
    final = await run_until_done(task_id, store)
    notification = await asyncio.wait_for(
        _drain_notification(store, None),
        timeout=1.0,
    )

    assert launch["status"] == "async_launched"
    assert launch["agent_type"] == "fork"
    assert launch["is_fork"] is True
    assert launch["coordinator_work_item_id"] == f"cw_agent_{task_id}"
    assert final.status == "completed"
    assert final.agent_type == "fork"
    assert final.prompt == "investigate A"
    assert final.runtime_session_id is not None
    assert notification.agent_id is None
    assert notification.task_id == task_id
    assert "<status>completed</status>" in notification.message

    request = provider.requests[0]
    assert request.system_prompt == "parent system"
    assert tuple(tool.name for tool in request.tools) == ("Read", AGENT_TOOL_NAME)
    assert request.permission_context is not None
    assert request.permission_context.mode == "bubble"
    payloads = [
        thaw_json(message.provider_payload)
        if message.provider_payload is not None
        else None
        for message in request.messages
    ]
    assert payloads[0] == {"role": "user", "content": "parent context"}
    assert payloads[1] == assistant_message
    fork_user = cast(dict[str, Any], payloads[2])
    assert fork_user["role"] == "user"
    fork_content = cast(list[dict[str, Any]], fork_user["content"])
    assert [block["tool_use_id"] for block in fork_content[:2]] == [
        "toolu_agent_a",
        "toolu_agent_b",
    ]
    assert "Directive: investigate A" in fork_content[2]["text"]

    replay = await get_agent_transcript(
        transcript_store,
        parent_session_id="parent-session",
        agent_id=task_id,
    )
    assert replay is not None
    assert replay.is_sidechain is True
    assert replay.runtime_session_id == final.runtime_session_id
    assert replay.messages[0] == {"role": "user", "content": "parent context"}
    assert replay.messages[1] == assistant_message
    assert replay.messages[-1] == {
        "role": "assistant",
        "content": "fork background result",
    }
    snapshot = runtime.snapshot()
    assert "mode=fork_background" in snapshot.work_items[0].title
    assert snapshot.work_items[0].status == "running"
    assert snapshot.work_items[0].task_id == task_id


@pytest.mark.asyncio
async def test_agent_tool_fork_background_survives_parent_abort_and_can_be_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    captured: dict[str, Any] = {}

    class BlockingEngine:
        def __init__(
            self,
            config: QueryConfig,
            _deps: QueryDeps,
            ctx: ToolUseContext,
        ) -> None:
            captured["config"] = config
            captured["ctx"] = ctx
            captured["seeded"] = ()

        async def seed_messages(self, messages: Sequence[Any]) -> None:
            captured["seeded"] = tuple(messages)

        async def submit_message(self, prompt: Any) -> AsyncIterator[Any]:
            captured["prompt"] = prompt
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", BlockingEngine)

    store = AppStateStore()
    deps = _deps(store=store)
    config = replace(_config(), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = replace(
        _ctx(tools=(agent_tool,), agent_id=None),
        messages=[{"role": "user", "content": "parent context"}],
    )

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "long fork", "run_in_background": True},
    )
    task_id = cast(str, _launch_data(results[0])["agent_id"])
    await asyncio.wait_for(started.wait(), timeout=1.0)

    ctx.abort_event.set()
    await asyncio.sleep(0.05)
    running = store.tasks[task_id]
    assert isinstance(running, LocalAgentState)
    assert running.status == "running"
    child_ctx = cast(ToolUseContext, captured["ctx"])
    assert child_ctx.query_source == "agent:builtin:fork"
    assert child_ctx.abort_event is not ctx.abort_event

    await LocalAgentTask().kill(task_id, store)
    final = await run_until_done(task_id, store)

    assert final.status == "killed"
    seeded = cast(tuple[dict[str, Any], ...], captured["seeded"])
    assert seeded[0] == {"role": "user", "content": "parent context"}
    submitted = cast(dict[str, Any], captured["prompt"])
    assert submitted["role"] == "user"
    assert "Directive: long fork" in str(submitted["content"])


@pytest.mark.asyncio
async def test_agent_tool_fork_worktree_appends_path_translation_notice() -> None:
    manager = FakeWorktreeManager(
        cleanup_result=WorktreeCleanupResult(
            kept=True,
            reason="changed",
            path="/tmp/raygent/fork-worktree",
            branch="worktree-fork",
        )
    )
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "fork worktree result"},)
    )
    deps = _deps(model_provider=provider, worktree_manager=manager)
    config = replace(_config(), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_agent_a",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate in worktree"},
            }
        ],
    }
    ctx = replace(
        _ctx(tools=(agent_tool,), agent_id=None),
        messages=[{"role": "user", "content": "parent file context"}],
    )

    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_agent_a",
                name=AGENT_TOOL_NAME,
                input={"prompt": "investigate in worktree", "isolation": "worktree"},
                index=0,
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(agent_tool,),
            deps=deps,
            ctx=ctx,
        )
    ]
    results = [event for event in events if isinstance(event, ToolExecutionResult)]

    metadata = cast(list[dict[str, Any]], _result_content(results[0]))[1]
    assert metadata["worktree_path"] == "/tmp/raygent/fork-worktree"
    assert metadata["worktree_branch"] == "worktree-fork"
    assert metadata["worktree_cleanup_reason"] == "changed"
    payloads = [
        thaw_json(message.provider_payload)
        if message.provider_payload is not None
        else None
        for message in provider.requests[0].messages
    ]
    notice = cast(dict[str, Any], payloads[-1])
    assert notice["role"] == "user"
    assert "isolated git worktree" in cast(str, notice["content"])
    assert "/tmp/raygent/agent-a" in cast(str, notice["content"])


@pytest.mark.asyncio
async def test_agent_tool_fork_background_worktree_preserves_notice_and_cleanup(
    tmp_path: Any,
) -> None:
    manager = FakeWorktreeManager(
        cleanup_result=WorktreeCleanupResult(
            kept=True,
            reason="changed",
            path="/tmp/raygent/fork-background-worktree",
            branch="worktree-fork-background",
        )
    )
    store = AppStateStore()
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "fork background worktree result"},)
    )
    deps = _deps(
        store=store,
        model_provider=provider,
        transcript_store=transcript_store,
        worktree_manager=manager,
    )
    config = replace(_config(), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_agent_a",
                "name": AGENT_TOOL_NAME,
                "input": {"prompt": "investigate in worktree"},
            }
        ],
    }
    ctx = replace(
        _ctx(tools=(agent_tool,), agent_id=None),
        messages=[{"role": "user", "content": "parent file context"}],
    )

    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_agent_a",
                name=AGENT_TOOL_NAME,
                input={
                    "prompt": "investigate in worktree",
                    "isolation": "worktree",
                    "run_in_background": True,
                },
                index=0,
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(agent_tool,),
            deps=deps,
            ctx=ctx,
        )
    ]
    results = [event for event in events if isinstance(event, ToolExecutionResult)]

    launch = cast(list[dict[str, Any]], _result_content(results[0]))[1]
    task_id = cast(str, launch["agent_id"])
    final = await run_until_done(task_id, store)
    notification = await asyncio.wait_for(
        _drain_notification(store, None),
        timeout=1.0,
    )

    assert launch["worktree_path"].startswith("/tmp/raygent/agent-")
    assert launch["is_fork"] is True
    assert manager.cleaned
    assert final.status == "completed"
    assert final.worktree_path == "/tmp/raygent/fork-background-worktree"
    assert "fork-background-worktree" in notification.message
    payloads = [
        thaw_json(message.provider_payload)
        if message.provider_payload is not None
        else None
        for message in provider.requests[0].messages
    ]
    notice = cast(dict[str, Any], payloads[-1])
    assert notice["role"] == "user"
    assert "isolated git worktree" in cast(str, notice["content"])
    assert "/tmp/raygent/agent-" in cast(str, notice["content"])


@pytest.mark.asyncio
async def test_agent_tool_fork_child_preserves_agent_when_catalog_provider_installed() -> None:
    read = _base_tool("Read")
    store = AppStateStore()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "fork result"},)
    )
    deps = _deps(store=store, model_provider=provider)
    config = replace(_config((read,)), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    deps.tool_catalog_provider = create_agent_catalog_provider(
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(read, agent_tool), agent_id=None)

    await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "fork with catalog provider"},
    )

    assert tuple(tool.name for tool in provider.requests[0].tools) == (
        "Read",
        AGENT_TOOL_NAME,
    )
    assert provider.requests[0].query_source == "agent:builtin:fork"


@pytest.mark.asyncio
async def test_agent_tool_omitted_type_uses_default_worker_when_fork_gate_off() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "default worker result"},)
    )
    deps = _deps(model_provider=provider)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(system_prompt="worker system"),),
    )
    ctx = _ctx(tools=(agent_tool,))

    await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "default worker", "run_in_background": False},
    )

    request = provider.requests[0]
    assert request.system_prompt == "worker system"
    assert all(
        "forked_worker" not in str(message.provider_payload)
        for message in request.messages
    )


@pytest.mark.asyncio
async def test_agent_tool_fork_recursive_guard_returns_error() -> None:
    provider = FakeModelProvider()
    deps = _deps(model_provider=provider)
    config = _config()
    config = replace(config, experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))
    ctx = replace(
        ctx,
        messages=[
            {
                "role": "user",
                "content": "<forked_worker>already forked</forked_worker>",
            }
        ],
    )

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "nested fork"},
    )

    assert "Fork is not available inside a forked worker" in str(
        _result_content(results[0])
    )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_agent_tool_fork_recursive_guard_uses_query_source() -> None:
    provider = FakeModelProvider()
    deps = _deps(model_provider=provider)
    config = replace(_config(), experiments={"fork_subagent": True})
    agent_tool = build_agent_tool(
        parent_config=config,
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = replace(
        _ctx(tools=(agent_tool,)),
        query_source="agent:builtin:fork",
        messages=[],
    )

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={"prompt": "nested fork after compact"},
    )

    assert "Fork is not available inside a forked worker" in str(
        _result_content(results[0])
    )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_agent_tool_foreground_parent_abort_reraises_cancelled() -> None:
    class AbortProvider(FakeModelProvider):
        async def complete(self, request: ModelRequest) -> Any:
            self.requests.append(request)
            assert request.abort_event is not None
            request.abort_event.set()
            raise asyncio.CancelledError()

    provider = AbortProvider()
    deps = _deps(model_provider=provider)
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )
    ctx = _ctx(tools=(agent_tool,))

    with pytest.raises(asyncio.CancelledError):
        await _run_agent_tool(
            tool=agent_tool,
            deps=deps,
            ctx=ctx,
            input_={
                "prompt": "abort",
                "subagent_type": "worker",
                "run_in_background": False,
            },
        )


@pytest.mark.asyncio
async def test_agent_catalog_provider_appends_agent_and_filters_denied() -> None:
    base = _base_tool("Read")
    deps = _deps()
    provider = create_agent_catalog_provider(
        parent_deps=deps,
        agent_definitions=(_agent_definition(agent_type="worker"),),
    )
    denied_context = ToolPermissionContext(
        always_deny_rules={"localSettings": ("Agent(worker)",)}
    )

    tools = await provider(
        _config((base,)),
        _ctx(tools=(base,), agent_id=None),
        (),
    )
    denied_tools = await provider(
        _config((base,)),
        _ctx(tools=(base,), permission_context=denied_context),
        (),
    )

    assert tools is not None
    assert tuple(tool.name for tool in tools) == ("Read", AGENT_TOOL_NAME)
    assert denied_tools is not None
    assert tuple(tool.name for tool in denied_tools) == ("Read",)


@pytest.mark.asyncio
async def test_agent_catalog_provider_does_not_readd_agent_inside_subagent() -> None:
    base = _base_tool("Read")
    deps = _deps()
    provider = create_agent_catalog_provider(
        parent_deps=deps,
        agent_definitions=(_agent_definition(agent_type="worker"),),
    )

    tools = await provider(
        _config((base,)),
        _ctx(tools=(base,), agent_id="local_agent_child"),
        (),
    )

    assert tools is not None
    assert tuple(tool.name for tool in tools) == ("Read",)


@pytest.mark.asyncio
async def test_child_query_engine_turn_config_does_not_readd_agent_tool() -> None:
    base = _base_tool("Read")
    deps = _deps()
    deps.tool_catalog_provider = create_agent_catalog_provider(
        parent_deps=deps,
        agent_definitions=(_agent_definition(agent_type="worker"),),
    )
    engine = QueryEngine(
        _config((base,)),
        deps,
        _ctx(tools=(base,), agent_id="local_agent_child"),
    )

    turn_config = await engine._build_turn_config(  # pyright: ignore[reportPrivateUsage]
        _ctx(tools=(base,), agent_id="local_agent_child")
    )

    assert tuple(tool.name for tool in turn_config.tools) == ("Read",)


@pytest.mark.asyncio
async def test_agent_tool_reports_missing_required_mcp_servers() -> None:
    deps = _deps()
    github_tool = _base_tool("mcp__github__search")
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(
            _agent_definition(
                agent_type="github-worker",
                required_mcp_servers=("github",),
            ),
        ),
    )

    missing = await agent_tool.validate_input(
        AgentToolInput(prompt="x", subagent_type="github-worker"),
        _ctx(tools=(agent_tool,)),
    )
    present = await agent_tool.validate_input(
        AgentToolInput(prompt="x", subagent_type="github-worker"),
        _ctx(tools=(agent_tool, github_tool)),
    )

    assert isinstance(missing, ValidationError)
    assert "requires MCP servers matching: github" in missing.message
    assert present.result == "ok"


@pytest.mark.asyncio
async def test_agent_tool_allows_foreground_and_non_team_named_background() -> None:
    deps = _deps()
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
    )

    foreground = await agent_tool.validate_input(
        AgentToolInput(prompt="x", run_in_background=False),
        _ctx(tools=(agent_tool,)),
    )
    named = await agent_tool.validate_input(
        AgentToolInput(prompt="x", name="worker-a"),
        _ctx(tools=(agent_tool,), agent_id=None),
    )
    explicit_team_name = await agent_tool.validate_input(
        AgentToolInput(prompt="x", name="worker-a", team_name="missing-team"),
        _ctx(tools=(agent_tool,), agent_id=None),
    )
    missing_worktree_manager = await agent_tool.validate_input(
        AgentToolInput(prompt="x", isolation="worktree"),
        _ctx(tools=(agent_tool,)),
    )
    remote = await agent_tool.validate_input(
        AgentToolInput(prompt="x", isolation="remote"),
        _ctx(tools=(agent_tool,)),
    )

    assert foreground.result == "ok"
    assert named.result == "ok"
    assert isinstance(explicit_team_name, ValidationError)
    assert "TeamCreate context" in explicit_team_name.message
    assert isinstance(missing_worktree_manager, ValidationError)
    assert "worktree isolation requires QueryDeps.worktree_manager" in (
        missing_worktree_manager.message
    )
    assert isinstance(remote, ValidationError)
    assert "remote isolation requires QueryDeps.remote_agent_backend" in remote.message


@pytest.mark.asyncio
async def test_agent_tool_explicit_team_name_without_active_team_does_not_spawn(
    tmp_path: Any,
) -> None:
    store = AppStateStore()
    deps = _deps(store=store)
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    agent_tool = build_agent_tool(
        parent_config=_config(),
        parent_deps=deps,
        agent_definitions=(_agent_definition(),),
        team_store=team_store,
    )
    ctx = _ctx(tools=(agent_tool,), agent_id=None)

    results = await _run_agent_tool(
        tool=agent_tool,
        deps=deps,
        ctx=ctx,
        input_={
            "prompt": "research",
            "name": "Researcher",
            "team_name": "Missing Team",
        },
    )

    assert "TeamCreate context" in str(_result_content(results[0]))
    assert store.tasks == {}
    assert store.agent_name_registry == {}

    direct_events = [
        event
        async for event in agent_tool.call(
            AgentToolInput(
                prompt="research",
                name="Researcher",
                team_name="Missing Team",
            ),
            ctx,
        )
    ]
    assert len(direct_events) == 1
    direct_result = direct_events[0]
    assert isinstance(direct_result, ToolResult)
    assert direct_result.is_error is True
    assert "active team context" in str(direct_result.content)
    assert store.tasks == {}
    assert store.agent_name_registry == {}
