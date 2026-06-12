from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.coordinator.runtime import CoordinatorRuntime
from raygent_harness.coordinator.team import TeamStateStore
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.task import AgentRouteRecord, AppStateStore
from raygent_harness.core.tasks import local_agent as local_agent_mod
from raygent_harness.core.tasks.in_process_teammate import (
    InProcessTeammateState,
    TeammateIdentity,
)
from raygent_harness.core.tasks.local_agent import LocalAgentState, run_until_done
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolRuntimeContext,
    ToolUseContext,
    ValidationError,
)
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.tools.send_message_tool import (
    SEND_MESSAGE_PROMPT,
    SendMessageInput,
    build_send_message_tool,
    create_send_message_catalog_provider,
)


def _ctx(agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd="/repo",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _ctx_with_runtime(
    *,
    task_store: AppStateStore,
    coordinator_runtime: CoordinatorRuntime,
    agent_id: str | None = None,
) -> ToolUseContext:
    deps = QueryDeps(
        task_store=task_store,
        coordinator_runtime=coordinator_runtime,
    )
    return replace(
        _ctx(agent_id=agent_id),
        runtime=ToolRuntimeContext(
            config=QueryConfig(model="m"),
            deps=deps,
            effective_model="m",
        ),
    )


def _register_teammate(
    store: AppStateStore,
    *,
    task_id: str,
    name: str,
    team_name: str = "team",
    status: str = "running",
) -> None:
    identity = TeammateIdentity(
        agent_id=f"{name}@{team_name}",
        name=name,
        team_name=team_name,
        parent_session_id="s",
        agent_type="worker",
        model="m",
    )
    store.register_task(
        InProcessTeammateState(
            id=task_id,
            type="in_process_teammate",
            description=name,
            status=cast(Any, status),
            start_time=1.0,
            identity=identity,
            prompt="initial",
            is_idle=True,
        )
    )
    store.agent_name_registry[name] = task_id


def _register_local_agent(
    store: AppStateStore,
    *,
    task_id: str,
    name: str | None = "researcher",
    status: str = "running",
    transcript_path: str | None = None,
    worktree_path: str | None = None,
    worktree_branch: str | None = None,
    worktree_slug: str | None = None,
) -> None:
    state = LocalAgentState(
        id=task_id,
        type="local_agent",
        description="research worker",
        status=cast(Any, status),
        start_time=1.0,
        parent_agent_id=None,
        prompt="initial",
        agent_type="worker",
        name=name,
        model="m",
        system_prompt="child system",
        tool_names=("Read",),
        permission_mode="acceptEdits",
        cwd="/repo",
        runtime_session_id=f"sub-{task_id}",
        transcript_path=transcript_path,
        worktree_path=worktree_path,
        worktree_branch=worktree_branch,
        worktree_slug=worktree_slug,
        worktree_created_at=123.0 if worktree_slug else None,
        worktree_touched_at=456.0 if worktree_slug else None,
        worktree_cleanup_policy="remove_if_clean" if worktree_slug else None,
    )
    store.register_task(state)
    store.agent_route_records[task_id] = AgentRouteRecord(
        agent_id=task_id,
        task_id=task_id,
        task_type="local_agent",
        name=name,
        parent_agent_id=None,
        parent_session_id="s",
        runtime_session_id=f"sub-{task_id}",
        agent_type="worker",
        description="research worker",
        model="m",
        system_prompt="child system",
        tool_names=("Read",),
        permission_mode="acceptEdits",
        cwd="/repo",
        worktree_path=worktree_path,
        worktree_branch=worktree_branch,
        worktree_slug=worktree_slug,
        worktree_created_at=123.0 if worktree_slug else None,
        worktree_touched_at=456.0 if worktree_slug else None,
        worktree_cleanup_policy="remove_if_clean" if worktree_slug else None,
        transcript_path=transcript_path,
    )
    if name is not None:
        store.agent_name_registry[name] = task_id


def test_send_message_prompt_mentions_local_agent_routing_scope() -> None:
    assert "ordinary non-team named local agents" in SEND_MESSAGE_PROMPT
    assert "raw local-agent task IDs" in SEND_MESSAGE_PROMPT
    assert "not broadcast to ordinary non-team local agents" in SEND_MESSAGE_PROMPT


async def _call_tool(
    tool_input: SendMessageInput,
    tool: Any,
    ctx: ToolUseContext,
) -> ToolResult:
    events: list[ToolCallEvent] = [event async for event in tool.call(tool_input, ctx)]
    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    return events[0]


def _team_store(tmp_path: Path) -> TeamStateStore:
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    team_store.create_team(
        team_name="team",
        description=None,
        agent_type="coordinator",
        model="m",
        cwd="/repo",
    )
    return team_store


def _install_capturing_local_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
    captures: list[dict[str, Any]],
) -> None:
    class CapturingEngine:
        def __init__(
            self,
            config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
            *,
            transcript_scope: Any | None = None,
        ) -> None:
            captures.append(
                {
                    "config": config,
                    "ctx": _ctx,
                    "transcript_scope": transcript_scope,
                    "prompt": None,
                    "replay_messages": None,
                }
            )

        @classmethod
        def from_replay(
            cls,
            config: QueryConfig,
            deps: QueryDeps,
            ctx: ToolUseContext,
            replay: Any,
            *,
            transcript_scope: Any | None = None,
        ) -> CapturingEngine:
            engine = cls(config, deps, ctx, transcript_scope=transcript_scope)
            captures[-1]["replay_messages"] = list(replay.messages)
            return engine

        async def submit_message(self, prompt: str) -> Any:
            captures[-1]["prompt"] = prompt
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result=f"resumed {prompt}",
            )

    monkeypatch.setattr(local_agent_mod, "QueryEngine", CapturingEngine)


@pytest.mark.asyncio
async def test_send_message_queues_direct_teammate_message(tmp_path: Path) -> None:
    task_store = AppStateStore()
    _register_teammate(task_store, task_id="t1", name="researcher")
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="researcher", message="check this", summary="review request"),
        tool,
        _ctx(),
    )

    assert not result.is_error
    task = cast(InProcessTeammateState, task_store.tasks["t1"])
    assert len(task.pending_messages) == 1
    assert task.pending_messages[0].sender == "team-lead"
    assert task.pending_messages[0].content == "check this"
    assert task.pending_messages[0].summary == "review request"
    assert isinstance(result.content, list)
    assert result.content[1]["task_id"] == "t1"
    assert result.content[1]["routing"]["target"] == "@researcher"


@pytest.mark.asyncio
async def test_send_message_queues_running_local_agent_without_team(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    _register_local_agent(task_store, task_id="la1", name="researcher")
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    tool = build_send_message_tool(
        team_store=team_store,
        task_store=task_store,
    )

    validation = await tool.validate_input(
        SendMessageInput(to="researcher", message="check this", summary="review request"),
        _ctx(),
    )
    result = await _call_tool(
        SendMessageInput(to="researcher", message="check this", summary="review request"),
        tool,
        _ctx(),
    )

    assert validation.result == "ok"
    assert not result.is_error
    task = cast(LocalAgentState, task_store.tasks["la1"])
    assert len(task.pending_messages) == 1
    assert task.pending_messages[0].sender == "team-lead"
    assert task.pending_messages[0].content == "check this"
    assert task.pending_messages[0].summary == "review request"
    assert isinstance(result.content, list)
    assert result.content[1]["task_id"] == "la1"
    assert result.content[1]["routing"]["target"] == "@researcher"


@pytest.mark.asyncio
async def test_send_message_queues_running_local_agent_by_raw_id_without_team(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    _register_local_agent(task_store, task_id="la1", name=None)
    tool = build_send_message_tool(
        team_store=TeamStateStore(base_dir=tmp_path / "teams"),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="la1", message="raw follow-up", summary="raw route"),
        tool,
        _ctx(),
    )

    assert not result.is_error
    task = cast(LocalAgentState, task_store.tasks["la1"])
    assert task.pending_messages[0].content == "raw follow-up"
    assert isinstance(result.content, list)
    assert result.content[1]["task_id"] == "la1"


@pytest.mark.asyncio
async def test_send_message_catalog_visible_for_local_agent_route_without_team(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    _register_local_agent(task_store, task_id="la1", name="researcher")
    team_store = TeamStateStore(base_dir=tmp_path / "teams")
    provider = create_send_message_catalog_provider(
        team_store=team_store,
        task_store=task_store,
    )

    tools = await provider(QueryConfig(model="m"), _ctx(), ())

    assert tools is not None
    assert [tool.name for tool in tools] == ["SendMessage"]


@pytest.mark.asyncio
async def test_send_message_resumes_terminal_local_agent_from_sidechain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []
    _install_capturing_local_agent_engine(monkeypatch, captures)

    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    scope = TranscriptScope(session_id="s", agent_id="la1", is_sidechain=True)
    await transcript_store.append_many(
        scope,
        [
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                runtime_session_id="sub-la1",
                agent_id="la1",
                is_sidechain=True,
                message={"role": "user", "content": "initial"},
            ),
        ],
    )
    task_store = AppStateStore()
    worktree_path = tmp_path / "agent-aabcdef0"
    worktree_path.mkdir()
    _register_local_agent(
        task_store,
        task_id="la1",
        name="researcher",
        status="completed",
        transcript_path=transcript_store.path_for(scope),
        worktree_path=str(worktree_path),
        worktree_branch="worktree-agent-aabcdef0",
        worktree_slug="agent-aabcdef0",
    )
    deps = QueryDeps(task_store=task_store, transcript_store=transcript_store)
    ctx = replace(
        _ctx(),
        runtime=ToolRuntimeContext(
            config=QueryConfig(model="m", session_id="s"),
            deps=deps,
            effective_model="m",
        ),
    )
    tool = build_send_message_tool(
        team_store=TeamStateStore(base_dir=tmp_path / "teams"),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="researcher", message="continue", summary="follow up"),
        tool,
        ctx,
    )
    final = await run_until_done("la1", task_store)

    assert not result.is_error
    assert isinstance(result.content, list)
    assert result.content[1]["task_id"] == "la1"
    assert "resumed it in the background" in result.content[0]["text"]
    assert captures[0]["prompt"] == "continue"
    assert captures[0]["replay_messages"] == [{"role": "user", "content": "initial"}]
    assert isinstance(captures[0]["transcript_scope"], TranscriptScope)
    assert final.status == "completed"
    assert final.final_message == "resumed continue"
    assert task_store.agent_name_registry["researcher"] == "la1"


@pytest.mark.asyncio
async def test_send_message_resumes_evicted_local_agent_by_raw_id_from_sidechain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []
    _install_capturing_local_agent_engine(monkeypatch, captures)

    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    scope = TranscriptScope(session_id="s", agent_id="la1", is_sidechain=True)
    await transcript_store.append_many(
        scope,
        [
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                runtime_session_id="sub-la1",
                agent_id="la1",
                is_sidechain=True,
                message={"role": "user", "content": "initial"},
            ),
        ],
    )
    task_store = AppStateStore()
    evicted_worktree_path = tmp_path / "evicted-agent-aabcdef0"
    evicted_worktree_path.mkdir()
    _register_local_agent(
        task_store,
        task_id="la1",
        name="researcher",
        status="completed",
        transcript_path=transcript_store.path_for(scope),
        worktree_path=str(evicted_worktree_path),
        worktree_branch="worktree-agent-aabcdef0",
        worktree_slug="agent-aabcdef0",
    )
    del task_store.tasks["la1"]
    deps = QueryDeps(task_store=task_store, transcript_store=transcript_store)
    ctx = replace(
        _ctx(),
        runtime=ToolRuntimeContext(
            config=QueryConfig(model="m", session_id="s"),
            deps=deps,
            effective_model="m",
        ),
    )
    tool = build_send_message_tool(
        team_store=TeamStateStore(base_dir=tmp_path / "teams"),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="la1", message="continue", summary="follow up"),
        tool,
        ctx,
    )
    final = await run_until_done("la1", task_store)

    assert not result.is_error
    assert isinstance(result.content, list)
    assert result.content[1]["task_id"] == "la1"
    assert "resumed it in the background" in result.content[0]["text"]
    assert captures[0]["prompt"] == "continue"
    assert captures[0]["replay_messages"] == [{"role": "user", "content": "initial"}]
    assert final.status == "completed"
    assert final.final_message == "resumed continue"
    assert task_store.agent_name_registry["researcher"] == "la1"
    assert final.worktree_path == str(evicted_worktree_path)
    assert final.worktree_branch == "worktree-agent-aabcdef0"
    assert final.worktree_slug == "agent-aabcdef0"
    assert final.worktree_created_at == 123.0
    assert final.worktree_touched_at == 456.0
    assert final.worktree_cleanup_policy == "remove_if_clean"


@pytest.mark.asyncio
async def test_send_message_resumes_evicted_local_agent_with_missing_worktree_from_parent_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captures: list[dict[str, Any]] = []
    _install_capturing_local_agent_engine(monkeypatch, captures)

    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    scope = TranscriptScope(session_id="s", agent_id="la1", is_sidechain=True)
    await transcript_store.append_many(
        scope,
        [
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                runtime_session_id="sub-la1",
                agent_id="la1",
                is_sidechain=True,
                message={"role": "user", "content": "initial"},
            ),
        ],
    )
    task_store = AppStateStore()
    missing_worktree_path = tmp_path / "missing-agent-aabcdef0"
    _register_local_agent(
        task_store,
        task_id="la1",
        name="researcher",
        status="completed",
        transcript_path=transcript_store.path_for(scope),
        worktree_path=str(missing_worktree_path),
        worktree_branch="worktree-agent-aabcdef0",
        worktree_slug="agent-aabcdef0",
    )
    del task_store.tasks["la1"]
    deps = QueryDeps(task_store=task_store, transcript_store=transcript_store)
    ctx = replace(
        _ctx(),
        runtime=ToolRuntimeContext(
            config=QueryConfig(model="m", session_id="s"),
            deps=deps,
            effective_model="m",
        ),
    )
    tool = build_send_message_tool(
        team_store=TeamStateStore(base_dir=tmp_path / "teams"),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="la1", message="continue", summary="follow up"),
        tool,
        ctx,
    )
    final = await run_until_done("la1", task_store)

    assert not result.is_error
    assert captures[0]["ctx"].cwd == "/repo"
    assert final.cwd == "/repo"
    assert final.worktree_path is None
    assert final.worktree_branch is None
    assert final.worktree_slug is None
    assert task_store.agent_route_records["la1"].cwd == "/repo"
    assert task_store.agent_route_records["la1"].worktree_path is None


@pytest.mark.asyncio
async def test_send_message_records_successful_route_in_coordinator_runtime(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    runtime = CoordinatorRuntime()
    _register_teammate(task_store, task_id="t1", name="researcher")
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(
            to="researcher",
            message="SECRET routed body",
            summary="review request",
        ),
        tool,
        _ctx_with_runtime(task_store=task_store, coordinator_runtime=runtime),
    )

    assert isinstance(result.content, list)
    assert result.content[1]["coordinator_work_item_ids"] == ["cw_route_t1"]
    assert result.content[1]["coordinator_blackboard_entry_id"] == "cb_000001"
    snapshot = runtime.snapshot()
    assert len(snapshot.work_items) == 1
    assert snapshot.work_items[0].task_id == "t1"
    assert snapshot.work_items[0].status == "running"
    assert len(snapshot.blackboard_entries) == 1
    assert "summary=review request" in snapshot.blackboard_entries[0].content
    assert "message_chars=18" in snapshot.blackboard_entries[0].content
    assert "SECRET routed body" not in snapshot.blackboard_entries[0].content


@pytest.mark.asyncio
async def test_send_message_broadcasts_to_running_team_except_sender(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    _register_teammate(task_store, task_id="t1", name="researcher")
    _register_teammate(task_store, task_id="t2", name="builder")
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="*", message="stand by", summary="team update"),
        tool,
        _ctx(agent_id="researcher@team"),
    )

    assert not result.is_error
    researcher = cast(InProcessTeammateState, task_store.tasks["t1"])
    builder = cast(InProcessTeammateState, task_store.tasks["t2"])
    assert researcher.pending_messages == ()
    assert len(builder.pending_messages) == 1
    assert builder.pending_messages[0].sender == "researcher"
    assert builder.pending_messages[0].content == "stand by"
    assert builder.pending_messages[0].summary == "team update"
    assert isinstance(result.content, list)
    assert result.content[1]["recipients"] == ["builder"]


@pytest.mark.asyncio
async def test_send_message_zero_recipient_broadcast_does_not_record_runtime(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    runtime = CoordinatorRuntime()
    _register_teammate(task_store, task_id="t1", name="researcher")
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="*", message="stand by", summary="team update"),
        tool,
        _ctx_with_runtime(
            task_store=task_store,
            coordinator_runtime=runtime,
            agent_id="researcher@team",
        ),
    )

    assert not result.is_error
    assert isinstance(result.content, list)
    assert result.content[1]["recipients"] == []
    assert "coordinator_work_item_ids" not in result.content[1]
    assert "coordinator_blackboard_entry_id" not in result.content[1]
    assert runtime.snapshot().work_items == ()
    assert runtime.snapshot().blackboard_entries == ()


@pytest.mark.asyncio
async def test_send_message_rejects_unknown_terminal_and_missing_summary(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    _register_teammate(task_store, task_id="t1", name="researcher", status="completed")
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    missing_summary = await tool.validate_input(
        SendMessageInput(to="researcher", message="check this"),
        _ctx(),
    )
    assert isinstance(missing_summary, ValidationError)

    unknown = await _call_tool(
        SendMessageInput(to="missing", message="check this", summary="review request"),
        tool,
        _ctx(),
    )
    terminal = await _call_tool(
        SendMessageInput(to="researcher", message="check this", summary="review request"),
        tool,
        _ctx(),
    )

    assert unknown.is_error
    assert "was not found" in str(unknown.content)
    assert terminal.is_error
    assert "is completed" in str(terminal.content)


@pytest.mark.asyncio
async def test_send_message_error_does_not_record_coordinator_runtime(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    runtime = CoordinatorRuntime()
    tool = build_send_message_tool(
        team_store=_team_store(tmp_path),
        task_store=task_store,
    )

    result = await _call_tool(
        SendMessageInput(to="missing", message="check this", summary="review request"),
        tool,
        _ctx_with_runtime(task_store=task_store, coordinator_runtime=runtime),
    )

    assert result.is_error
    assert runtime.snapshot().work_items == ()
    assert runtime.snapshot().blackboard_entries == ()
