"""LocalAgentTask tests — happy / kill / failure paths with a fake
QueryEngine patched in `local_agent.py`'s module namespace.

Per (b): patch `raygent_harness.core.tasks.local_agent.QueryEngine`
(NOT the global `core.query_engine.QueryEngine`) so the driver picks
up the fake.

The fake's `submit_message` is an async generator yielding canned
SDKResult values, matching the contract the driver consumes at
`local_agent.py:439`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.permissions import PermissionAllowDecision
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks import local_agent as agent_mod
from raygent_harness.core.tasks.local_agent import (
    LocalAgentState,
    resume_local_agent_background,
    run_until_done,
    spawn_local_agent,
)
from raygent_harness.core.tasks.local_bash import spawn_local_bash
from raygent_harness.core.tool import (
    ContentReplacementState,
    QueryTracking,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.services.agent_routes import JsonAgentRouteRecordStore
from raygent_harness.services.compact import PERSISTED_TOOL_RESULT_TAG
from raygent_harness.services.transcript import (
    ContentReplacementEntry,
    JsonlTranscriptStore,
    TranscriptScope,
    get_agent_transcript,
    transcript_path_for_scope,
)
from raygent_harness.services.worktree import WorktreeInfo
from tests.fakes import FakeModelProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Test fixtures: parent config / deps / ctx.
# ---------------------------------------------------------------------------


def _parent_setup() -> tuple[QueryConfig, QueryDeps, ToolUseContext, AppStateStore]:
    store = AppStateStore()
    config = QueryConfig(
        model="claude-opus-4-7",
        system_prompt="parent",
        agent_id="parent",
        session_id="parent-session",
    )
    deps = QueryDeps(
        task_store=store,
    )
    ctx = ToolUseContext(
        session_id="parent-session",
        agent_id="parent",
        abort_event=asyncio.Event(),
        rendered_system_prompt="parent",
        cwd=".",
        query_tracking=QueryTracking(chain_id="parent", depth=0),
    )
    return config, deps, ctx, store


async def _wait_route_records(
    route_store: JsonAgentRouteRecordStore,
    parent_session_id: str,
    *,
    count: int,
) -> Any:
    result: Any = None
    for _ in range(100):
        result = await route_store.list_records(parent_session_id)
        if len(result.records) == count:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(f"route records did not reach count {count}: {result!r}")


# ---------------------------------------------------------------------------
# Fake engines.
# ---------------------------------------------------------------------------


class _SuccessEngine:
    """Yields one assistant-equivalent (skipped here) + a successful SDKResult."""

    def __init__(
        self,
        _config: Any,
        _deps: Any,
        _ctx: Any,
        *,
        transcript_scope: Any | None = None,
    ) -> None:
        _ = transcript_scope
        pass

    async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
        yield SDKResult(
            subtype="success",
            session_id="child",
            is_error=False,
            num_turns=1,
            result=f"answer to: {prompt}",
        )


class _BlockingRouteStore:
    def __init__(self) -> None:
        self.save_started = asyncio.Event()
        self.save_calls = 0

    async def save(self, record: Any) -> None:
        _ = record
        self.save_calls += 1
        self.save_started.set()
        await asyncio.Event().wait()

    async def list_records(self, parent_session_id: str) -> Any:
        _ = parent_session_id
        return ()

    async def delete(self, parent_session_id: str, task_id: str) -> None:
        _ = parent_session_id, task_id


class _FailureEngine:
    """Yields a result with is_error=True and populated `errors`."""

    def __init__(
        self,
        _config: Any,
        _deps: Any,
        _ctx: Any,
        *,
        transcript_scope: Any | None = None,
    ) -> None:
        _ = transcript_scope
        pass

    async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
        yield SDKResult(
            subtype="error_max_turns",
            session_id="child",
            is_error=True,
            num_turns=10,
            result="",
            errors=("hit turn cap",),
        )


class _RaisingEngine:
    """Raises a non-Cancelled exception during submit_message."""

    def __init__(
        self,
        _config: Any,
        _deps: Any,
        _ctx: Any,
        *,
        transcript_scope: Any | None = None,
    ) -> None:
        _ = transcript_scope
        pass

    async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
        msg = "engine blew up"
        raise RuntimeError(msg)
        yield  # pragma: no cover  (unreachable; required for async gen)


class _BlockingEngine:
    """submit_message blocks forever — used to exercise the kill path."""

    def __init__(
        self,
        _config: Any,
        _deps: Any,
        _ctx: Any,
        *,
        transcript_scope: Any | None = None,
    ) -> None:
        _ = transcript_scope
        pass

    async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
        # Park here until cancelled by the driver's `kill()` flow.
        await asyncio.Event().wait()
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_completes_and_notifies_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _SuccessEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="hello",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)

    assert final.status == "completed"
    assert final.is_error is False
    assert final.final_message is not None
    assert "hello" in final.final_message

    # Notification routed to PARENT's drain key, not the child's.
    parent_notifs = store.drain_notifications("parent")
    assert len(parent_notifs) == 1
    assert parent_notifs[0].kind == "completed"
    assert "<status>completed" in parent_notifs[0].message

    child_notifs = store.drain_notifications(task_id)
    assert child_notifs == []


@pytest.mark.asyncio
async def test_spawn_local_agent_drops_parent_non_persistent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[QueryConfig] = []

    class CapturingEngine:
        def __init__(
            self,
            config: QueryConfig,
            _deps: QueryDeps,
            _ctx: ToolUseContext,
            *,
            transcript_scope: Any | None = None,
        ) -> None:
            _ = transcript_scope
            captured_configs.append(config)

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="ok",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingEngine)
    config, deps, ctx, store = _parent_setup()
    parent_context: MessageParam = {"role": "user", "content": "parent context"}
    config = replace(
        config,
        context_messages=(parent_context,),
        context_system_prompt="parent env context",
    )

    task_id = await spawn_local_agent(
        prompt="hello",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    await run_until_done(task_id, store)

    assert len(captured_configs) == 1
    assert captured_configs[0].context_messages == ()
    assert captured_configs[0].context_system_prompt == ""


@pytest.mark.asyncio
async def test_spawn_local_agent_persists_route_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _SuccessEngine)
    config, deps, ctx, store = _parent_setup()
    route_store = JsonAgentRouteRecordStore(tmp_path / "routes")
    sink = RecordingKernelEventSink()
    deps = replace(
        deps,
        agent_route_record_store=route_store,
        observability=KernelEventBus((sink,)),
    )

    task_id = await spawn_local_agent(
        prompt="SECRET CHILD PROMPT",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
        description="research worker",
        child_agent_type="worker",
        name="Researcher",
    )
    await run_until_done(task_id, store)
    load_result = await _wait_route_records(
        route_store,
        "parent-session",
        count=1,
    )

    assert load_result.warnings == ()
    assert len(load_result.records) == 1
    record = load_result.records[0]
    assert record.task_id == task_id
    assert record.name == "Researcher"
    assert record.agent_type == "worker"
    assert record.description == "research worker"
    assert record.parent_session_id == "parent-session"
    assert record.runtime_session_id is not None
    assert record.tool_names == ()
    assert store.agent_name_registry["Researcher"] == task_id
    assert any(event.type == "agent.route_record.saved" for event in sink.events)
    assert "SECRET CHILD PROMPT" not in str([event.data for event in sink.events])


@pytest.mark.asyncio
async def test_route_record_persistence_does_not_block_spawn_or_notification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _SuccessEngine)
    config, deps, ctx, store = _parent_setup()
    route_store = _BlockingRouteStore()
    deps = replace(deps, agent_route_record_store=route_store)

    task_id = await asyncio.wait_for(
        spawn_local_agent(
            prompt="hello",
            parent_agent_id="parent",
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            worktree_info=WorktreeInfo(
                path=str(tmp_path / "worktree"),
                branch="raygent/agent",
                cleanup_policy="keep",
            ),
            worktree_manager=None,
        ),
        timeout=1.0,
    )
    await asyncio.wait_for(route_store.save_started.wait(), timeout=1.0)
    final = await asyncio.wait_for(run_until_done(task_id, store), timeout=1.0)
    parent_notifs = store.drain_notifications("parent")

    assert final.status == "completed"
    assert route_store.save_calls >= 1
    assert len(parent_notifs) == 1
    assert parent_notifs[0].kind == "completed"


@pytest.mark.asyncio
async def test_failure_path_marks_failed_and_carries_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _FailureEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="x",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)

    assert final.status == "failed"
    assert final.is_error is True
    assert final.error is not None
    assert "hit turn cap" in final.error

    notifs = store.drain_notifications("parent")
    assert len(notifs) == 1
    assert notifs[0].kind == "error"
    assert "<status>failed" in notifs[0].message


@pytest.mark.asyncio
async def test_raising_engine_is_caught_and_terminals_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _RaisingEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="x",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)

    assert final.status == "failed"
    assert final.is_error is True
    assert final.error is not None
    assert "RuntimeError" in final.error
    assert "engine blew up" in final.error


@pytest.mark.asyncio
async def test_kill_flips_status_synchronously_and_suppresses_child_shells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_mod, "QueryEngine", _BlockingEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="x",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    # Spawn a bash task owned by the child agent (agent_id == task_id).
    # When the agent is killed, the cleanup must kill this shell AND
    # suppress its notification (per invariant 9).
    child_shell = await spawn_local_bash("sleep 5", store, agent_id=task_id)
    await asyncio.sleep(0.05)
    # Drain any startup-time entries so we measure leak post-kill.
    store.drain_notifications(task_id)

    impl = agent_mod.LocalAgentTask()
    await impl.kill(task_id, store)

    # Status flipped synchronously inside kill() — visible immediately,
    # before we wait for the driver.
    immediate = store.tasks[task_id]
    assert isinstance(immediate, LocalAgentState)
    assert immediate.status == "killed"

    # Let the driver finish its terminal path: cleanup + notification.
    for _ in range(50):
        if task_id not in agent_mod._DRIVER_TASKS:  # pyright: ignore[reportPrivateUsage]
            break
        await asyncio.sleep(0.05)

    final = store.tasks[task_id]
    assert isinstance(final, LocalAgentState)
    assert final.status == "killed"  # preserved by terminal updater
    assert final.is_error is True

    # Child shell got killed and its notification was suppressed.
    shell_state = store.tasks[child_shell.id]
    assert shell_state.status == "killed"
    assert shell_state.notified is True
    assert store.drain_notifications(task_id) == []

    # Parent gets exactly one notification with status=killed.
    parent_notifs = store.drain_notifications("parent")
    assert len(parent_notifs) == 1
    assert parent_notifs[0].kind == "error"
    assert "<status>killed" in parent_notifs[0].message


@pytest.mark.asyncio
async def test_kill_on_already_terminal_agent_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`LocalAgentTask.kill` must be idempotent: a second kill on an
    already-terminal task does not change state and does not enqueue
    a duplicate notification."""
    monkeypatch.setattr(agent_mod, "QueryEngine", _SuccessEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="hi",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)
    assert final.status == "completed"

    parent_notifs_before = store.drain_notifications("parent")
    assert len(parent_notifs_before) == 1
    completed_at = final.end_time

    # Already terminal — kill must be a no-op.
    impl = agent_mod.LocalAgentTask()
    await impl.kill(task_id, store)

    after = store.tasks[task_id]
    assert isinstance(after, LocalAgentState)
    assert after.status == "completed"
    assert after.end_time == completed_at
    # No fresh notification — already-drained queue stays empty.
    assert store.drain_notifications("parent") == []


@pytest.mark.asyncio
async def test_engine_receives_prompt_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local-agent driver must invoke `engine.submit_message(prompt)`
    EXACTLY ONCE per spawn — invariant 4 in the module docstring.
    Subagent state must NOT preload `messages=[user(prompt)]` and also have
    `submit_message` append the prompt again."""
    seen: list[str] = []

    class _OneShotEngine:
        def __init__(
            self,
            _config: Any,
            _deps: Any,
            _ctx: Any,
            *,
            transcript_scope: Any | None = None,
        ) -> None:
            _ = transcript_scope
            pass

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            seen.append(prompt)
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="ok",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", _OneShotEngine)
    config, deps, ctx, store = _parent_setup()

    task_id = await spawn_local_agent(
        prompt="exactly-once-marker",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)
    assert final.status == "completed"

    # Engine saw the prompt exactly once.
    assert seen == ["exactly-once-marker"]


@pytest.mark.asyncio
async def test_spawn_local_agent_writes_sidechain_transcript_under_parent_session(
    tmp_path: Path,
) -> None:
    config, deps, ctx, store = _parent_setup()
    transcript_store = JsonlTranscriptStore(tmp_path)
    deps = QueryDeps(
        task_store=store,
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "child answer"},),
        ),
        transcript_store=transcript_store,
    )

    task_id = await spawn_local_agent(
        prompt="child prompt",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)

    assert final.status == "completed"
    assert final.transcript_path is not None
    sidechain_scope = TranscriptScope(
        session_id="parent-session",
        agent_id=task_id,
        is_sidechain=True,
    )
    expected_path = Path(transcript_path_for_scope(tmp_path, sidechain_scope))
    assert Path(final.transcript_path) == expected_path
    assert expected_path.exists()
    assert not hasattr(final, "messages")

    replay = await get_agent_transcript(
        transcript_store,
        parent_session_id="parent-session",
        agent_id=task_id,
    )

    assert replay is not None
    assert replay.is_sidechain is True
    assert replay.agent_id == task_id
    assert replay.runtime_session_id is not None
    assert replay.runtime_session_id.startswith(f"sub-{task_id}-")
    assert replay.runtime_session_id not in str(expected_path)
    assert replay.messages == [
        {"role": "user", "content": "child prompt"},
        {"role": "assistant", "content": "child answer"},
    ]


@pytest.mark.asyncio
async def test_resume_fork_local_agent_preserves_query_source_and_tool_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class EmptyInput(BaseModel):
        pass

    async def noop_call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[Any]:
        yield ToolResult(content="ok")

    read_tool = build_tool(
        ToolSpec(
            name="Read",
            description="read",
            input_model=EmptyInput,
            call=noop_call,
            is_read_only=True,
            is_concurrency_safe=True,
            is_destructive=False,
            is_open_world=False,
        )
    )
    agent_tool = build_tool(
        ToolSpec(
            name="Agent",
            description="agent",
            input_model=EmptyInput,
            call=noop_call,
            is_read_only=True,
            is_concurrency_safe=True,
            is_destructive=False,
            is_open_world=True,
        )
    )
    config, _deps, ctx, store = _parent_setup()
    config = replace(config, tools=(read_tool, agent_tool))
    ctx.tools = (read_tool, agent_tool)
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    deps = QueryDeps(
        task_store=store,
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "fork done"},),
        ),
        transcript_store=transcript_store,
    )

    task_id = await spawn_local_agent(
        prompt={"role": "user", "content": "fork directive"},
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
        child_agent_type="fork",
        child_system_prompt="parent",
        child_tools=(read_tool, agent_tool),
        child_query_source="agent:builtin:fork",
        display_prompt="fork directive",
    )
    final = await run_until_done(task_id, store)
    assert final.status == "completed"

    captured: dict[str, Any] = {}

    class CapturingResumeEngine:
        @classmethod
        def from_replay(
            cls,
            config: QueryConfig,
            _deps: QueryDeps,
            resume_ctx: ToolUseContext,
            replay: Any,
            *,
            transcript_scope: Any | None = None,
        ) -> CapturingResumeEngine:
            _ = transcript_scope
            captured["config"] = config
            captured["ctx"] = resume_ctx
            captured["replay"] = replay
            return cls()

        async def submit_message(self, prompt: str) -> AsyncIterator[Any]:
            captured["prompt"] = prompt
            yield SDKResult(
                subtype="success",
                session_id="child",
                is_error=False,
                num_turns=1,
                result="resumed",
            )

    monkeypatch.setattr(agent_mod, "QueryEngine", CapturingResumeEngine)

    resumed_id = await resume_local_agent_background(
        target=task_id,
        prompt="resume after compacted history",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    resumed = await run_until_done(resumed_id, store)

    assert resumed.status == "completed"
    resumed_ctx = captured["ctx"]
    assert isinstance(resumed_ctx, ToolUseContext)
    assert resumed_ctx.query_source == "agent:builtin:fork"
    resumed_config = captured["config"]
    assert isinstance(resumed_config, QueryConfig)
    assert tuple(tool.name for tool in resumed_config.tools) == ("Read", "Agent")
    assert captured["prompt"] == "resume after compacted history"


@pytest.mark.asyncio
async def test_spawn_local_agent_persists_sidechain_replacement_records(
    tmp_path: Path,
) -> None:
    class EmptyInput(BaseModel):
        pass

    async def call_big_tool(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[Any]:
        yield ToolResult(content="x" * 200)

    async def allow_tool(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: Any,
    ) -> PermissionAllowDecision:
        return PermissionAllowDecision()

    big_tool = build_tool(
        ToolSpec(
            name="BigOutput",
            description="big output",
            input_model=EmptyInput,
            call=call_big_tool,
            check_permissions=allow_tool,
            is_read_only=True,
            is_concurrency_safe=True,
            is_destructive=False,
            is_open_world=False,
        )
    )
    config, deps, ctx, store = _parent_setup()
    config = QueryConfig(
        model="model-1",
        system_prompt=config.system_prompt,
        agent_id=config.agent_id,
        session_id=config.session_id,
        tools=(big_tool,),
    )
    ctx.tools = (big_tool,)
    ctx.content_replacement = ContentReplacementState(
        max_result_size_chars=12,
        replaced_outputs_dir=str(tmp_path / "outputs"),
    )
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    deps = QueryDeps(
        task_store=store,
        model_provider=FakeModelProvider(
            responses=(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_big",
                            "name": "BigOutput",
                            "input": {},
                        }
                    ],
                },
                {"role": "assistant", "content": "done"},
            ),
        ),
        transcript_store=transcript_store,
    )

    task_id = await spawn_local_agent(
        prompt="run big tool",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    final = await run_until_done(task_id, store)

    assert final.status == "completed"
    replay = await get_agent_transcript(
        transcript_store,
        parent_session_id="parent-session",
        agent_id=task_id,
    )

    assert replay is not None
    assert len(replay.content_replacements) == 1
    record = replay.content_replacements[0]
    assert record.tool_use_id == "toolu_big"
    assert record.replacement.startswith(PERSISTED_TOOL_RESULT_TAG)
    assert Path(record.path).exists()
    assert any(
        isinstance(entry, ContentReplacementEntry) and entry.agent_id == task_id
        for entry in await transcript_store.read_entries(
            TranscriptScope(
                session_id="parent-session",
                agent_id=task_id,
                is_sidechain=True,
            )
        )
    )
