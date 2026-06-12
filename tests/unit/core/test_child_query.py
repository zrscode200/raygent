from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.core.child_query import ChildQueryRequest, run_child_query
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.file_state import FileState
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks.local_bash import LocalBashState
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptScope,
    get_agent_transcript,
    transcript_path_for_scope,
)
from tests.fakes import FakeModelProvider


class EchoInput(BaseModel):
    text: str


async def _echo_call(
    input_: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    assert isinstance(input_, EchoInput)
    yield ToolResult(content=f"echo: {input_.text}")


def _echo_tool():
    return build_tool(
        ToolSpec(
            name="Echo",
            description="Echo text",
            input_model=EchoInput,
            call=_echo_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _parent_setup(
    *,
    tmp_path: Path,
    provider: FakeModelProvider,
) -> tuple[QueryConfig, QueryDeps, ToolUseContext, AppStateStore]:
    store = AppStateStore()
    config = QueryConfig(
        model="parent-model[1m]",
        system_prompt="parent system",
        session_id="parent-session",
    )
    deps = QueryDeps(
        task_store=store,
        model_provider=provider,
        transcript_store=JsonlTranscriptStore(tmp_path / "transcripts"),
    )
    ctx = ToolUseContext(
        session_id="parent-session",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="parent system",
        cwd=str(tmp_path),
        permission_context=ToolPermissionContext(
            always_allow_rules={"session": ("Read",)}
        ),
        query_tracking=QueryTracking(chain_id="root-chain", depth=0),
    )
    ctx.read_file_state.set(
        tmp_path / "README.md",
        FileState(content="parent read", timestamp=1, offset=None, limit=None),
    )
    return config, deps, ctx, store


@pytest.mark.asyncio
async def test_child_query_runs_synchronously_with_sidechain_and_effort(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "child answer"},),
        resolved_models={"child-model": "provider-child-model"},
    )
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)

    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
            system_prompt="child system",
            model="child-model",
            effort="high",
            cwd=str(tmp_path / "worktree"),
        )
    )

    assert result.agent_id == "local_agent_child"
    assert result.final_message == "child answer"
    assert result.is_error is False
    assert result.subtype == "success"
    assert result.transcript_path is not None
    assert provider.resolve_requests[0][0] == "child-model"
    assert provider.resolve_requests[0][1].agent_id == "local_agent_child"
    assert provider.resolve_requests[0][1].effort == "high"
    request = provider.requests[0]
    assert request.model == "provider-child-model"
    assert request.system_prompt == "child system"
    assert request.agent_id == "local_agent_child"
    assert request.effort == "high"
    assert request.abort_event is ctx.abort_event
    assert request.permission_context is not None
    assert request.permission_context.always_allow_rules == {"session": ("Read",)}

    scope = TranscriptScope(
        session_id="parent-session",
        agent_id="local_agent_child",
        is_sidechain=True,
    )
    assert Path(result.transcript_path) == Path(
        transcript_path_for_scope(tmp_path / "transcripts", scope)
    )
    assert deps.transcript_store is not None
    replay = await get_agent_transcript(
        deps.transcript_store,
        parent_session_id="parent-session",
        agent_id="local_agent_child",
    )
    assert replay is not None
    assert replay.is_sidechain is True
    assert replay.messages == [
        {"role": "user", "content": "child prompt"},
        {"role": "assistant", "content": "child answer"},
    ]


@pytest.mark.asyncio
async def test_child_query_seeds_initial_messages_into_one_model_turn(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)

    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=(
                {"role": "assistant", "content": "parent fork context"},
                {"role": "user", "content": "paired placeholder"},
                {"role": "user", "content": "child directive"},
            ),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
        )
    )

    assert result.is_error is False
    assert len(provider.requests) == 1
    request_payloads = [
        message.provider_payload for message in provider.requests[0].messages
    ]
    assert request_payloads == [
        {"role": "assistant", "content": "parent fork context"},
        {"role": "user", "content": "paired placeholder"},
        {"role": "user", "content": "child directive"},
    ]

    assert deps.transcript_store is not None
    replay = await get_agent_transcript(
        deps.transcript_store,
        parent_session_id="parent-session",
        agent_id="local_agent_child",
    )
    assert replay is not None
    assert replay.messages == [
        {"role": "assistant", "content": "parent fork context"},
        {"role": "user", "content": "paired placeholder"},
        {"role": "user", "content": "child directive"},
        {"role": "assistant", "content": "done"},
    ]


@pytest.mark.asyncio
async def test_child_query_drops_parent_non_persistent_context(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)
    parent_context: MessageParam = {"role": "user", "content": "parent context"}
    config = replace(
        config,
        context_messages=(parent_context,),
        context_system_prompt="parent env context",
    )

    await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
        )
    )

    request = provider.requests[0]
    assert request.system_prompt == "parent system"
    assert [message.provider_payload for message in request.messages] == [
        {"role": "user", "content": "child prompt"},
    ]


@pytest.mark.asyncio
async def test_child_query_clones_parent_read_state_and_uses_independent_abort(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)
    parent_key = tmp_path / "README.md"

    await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
            link_abort_to_parent=False,
        )
    )

    assert ctx.read_file_state.has(parent_key)
    request = provider.requests[0]
    assert request.abort_event is not None
    assert request.abort_event is not ctx.abort_event


@pytest.mark.asyncio
async def test_child_query_cleanup_suppresses_child_bash_notifications(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    config, deps, ctx, store = _parent_setup(tmp_path=tmp_path, provider=provider)
    store.register_task(
        LocalBashState(
            id="b_child",
            type="local_bash",
            description="child shell",
            status="running",
            start_time=0.0,
            command="sleep 100",
            agent_id="local_agent_child",
        )
    )

    await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
        )
    )

    child_shell = store.tasks["b_child"]
    assert isinstance(child_shell, LocalBashState)
    assert child_shell.status == "killed"
    assert child_shell.notified is True
    assert store.drain_notifications("local_agent_child") == []


@pytest.mark.asyncio
async def test_child_query_max_turns_caps_loop_and_cleans_shells(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "unused"},))
    config, deps, ctx, store = _parent_setup(tmp_path=tmp_path, provider=provider)
    store.register_task(
        LocalBashState(
            id="b_child",
            type="local_bash",
            description="child shell",
            status="running",
            start_time=0.0,
            command="sleep 100",
            agent_id="local_agent_child",
        )
    )

    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
            max_turns=0,
        )
    )

    assert result.is_error is True
    assert result.subtype == "error_max_turns"
    assert result.errors == ("max_turns <= 0 at turn entry",)
    assert provider.requests == []
    child_shell = store.tasks["b_child"]
    assert isinstance(child_shell, LocalBashState)
    assert child_shell.status == "killed"
    assert child_shell.notified is True


@pytest.mark.asyncio
async def test_child_query_positive_max_turns_allows_one_iteration_then_caps(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Echo",
                        "input": {"text": "hello"},
                    }
                ],
            },
            {"role": "assistant", "content": "should not be requested"},
        ),
    )
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)

    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
            tools=(_echo_tool(),),
            permission_context=ToolPermissionContext(mode="bypassPermissions"),
            max_turns=1,
        )
    )

    assert result.is_error is True
    assert result.subtype == "error_max_turns"
    assert result.errors == ("reached max_turns=1",)
    assert len(provider.requests) == 1
    assert result.messages[-1]["role"] == "user"
    assert result.messages[-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "echo: hello",
        }
    ]


@pytest.mark.asyncio
async def test_child_query_can_disable_sidechain_transcript(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    config, deps, ctx, _store = _parent_setup(tmp_path=tmp_path, provider=provider)

    result = await run_child_query(
        ChildQueryRequest(
            prompt_messages=({"role": "user", "content": "child prompt"},),
            parent_config=config,
            parent_deps=deps,
            parent_ctx=ctx,
            agent_id="local_agent_child",
            transcript_enabled=False,
        )
    )

    assert result.is_error is False
    assert result.transcript_path is None
    assert deps.transcript_store is not None
    replay = await get_agent_transcript(
        deps.transcript_store,
        parent_session_id="parent-session",
        agent_id="local_agent_child",
    )
    assert replay is None
