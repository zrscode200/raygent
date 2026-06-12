"""QueryEngine tests — submit_message event ordering.

Per item-11 review coverage gap: assert that `submit_message()` yields
SDKSystemInit FIRST, exactly one SDKResult LAST, both for clean and
exception paths. Test patches `_call_model` etc. so the engine drives
through `query()` without standing up the real model.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from raygent_harness import __version__ as RAYGENT_VERSION
from raygent_harness.coordinator.runtime import CoordinatorRuntime
from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.core.deps import (
    AgentTriggerDecision,
    AgentTriggerMatch,
    AgentTriggerPolicySpec,
    QueryDeps,
)
from raygent_harness.core.messages import (
    MessageParam,
    message_param_from_api_message,
    model_response_from_message_param,
)
from raygent_harness.core.model_request_normalization import (
    ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER,
)
from raygent_harness.core.model_types import Usage
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.query import (
    CompactBoundaryEvent,
    LayerResult,
    ToolOrchestrationComplete,
    ToolResultMessage,
)
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKCompactBoundary,
    SDKResult,
    SDKSystemInit,
    SDKUserMessage,
    _last_message_is_api_error,  # pyright: ignore[reportPrivateUsage]
)
from raygent_harness.core.stop_hooks import (
    ContinuationContextFragment,
    HookBlock,
    HookContext,
    HookContinue,
    HookContinueWithContext,
    HookPreventContinuation,
    HookResult,
)
from raygent_harness.core.task import AppStateStore, TaskNotification
from raygent_harness.core.tasks.local_agent import (
    LocalAgentPendingMessage,
    LocalAgentState,
)
from raygent_harness.core.tool import (
    ContentReplacementState,
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.services.compact import PERSISTED_TOOL_RESULT_TAG
from raygent_harness.services.compact.tool_result_budget import (
    ToolResultReplacementRecord,
)
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    replay_entries,
)
from tests.fakes import FakeModelProvider


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _engine(*, config: QueryConfig | None = None) -> QueryEngine:
    cfg = config or QueryConfig(model="claude-opus-4-7", session_id="s")
    deps = QueryDeps(task_store=AppStateStore())
    return QueryEngine(cfg, deps, _ctx())


def _engine_with_deps(deps: QueryDeps, *, config: QueryConfig | None = None) -> QueryEngine:
    cfg = config or QueryConfig(model="claude-opus-4-7", session_id="s")
    return QueryEngine(cfg, deps, _ctx())


def _engine_with_ctx(
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    config: QueryConfig | None = None,
) -> QueryEngine:
    cfg = config or QueryConfig(model="claude-opus-4-7", session_id="s")
    return QueryEngine(cfg, deps, ctx)


class _EmptyToolInput(BaseModel):
    pass


async def _unused_tool_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    if False:
        yield ToolResult(content="unused")


def _agent_named_tool(
    *,
    should_defer: bool = False,
    always_load: bool = False,
    is_enabled: bool = True,
) -> Any:
    return build_tool(
        ToolSpec(
            name="Agent",
            description="Launch a test agent.",
            input_model=_EmptyToolInput,
            call=_unused_tool_call,
            is_read_only=True,
            is_destructive=False,
            is_enabled=is_enabled,
            should_defer=should_defer,
            always_load=always_load,
        )
    )


def _tool_reference_message(tool_name: str = "Agent") -> MessageParam:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_tool_search",
                "content": [
                    {
                        "type": "tool_reference",
                        "tool_name": tool_name,
                    }
                ],
            }
        ],
    }


@dataclass
class RecordingMemoryPrefetch:
    messages_to_return: tuple[MessageParam, ...]
    settled_at_value: float | None = 1.0
    consumed_on_iteration_value: int | None = None
    consume_calls: list[int] = field(default_factory=list[int])
    cancel_count: int = 0

    @property
    def settled_at(self) -> float | None:
        return self.settled_at_value

    @property
    def consumed_on_iteration(self) -> int | None:
        return self.consumed_on_iteration_value

    async def consume_if_ready(
        self,
        *,
        ctx: ToolUseContext,
        iteration: int,
    ) -> tuple[MessageParam, ...]:
        _ = ctx
        self.consume_calls.append(iteration)
        self.consumed_on_iteration_value = iteration
        return self.messages_to_return

    def cancel(self) -> None:
        self.cancel_count += 1


def _empty_recall_start_calls() -> list[
    tuple[tuple[MessageParam, ...], QueryConfig, ToolUseContext]
]:
    return []


@dataclass
class RecordingMemoryRecallProvider:
    prefetch: RecordingMemoryPrefetch
    start_calls: list[
        tuple[tuple[MessageParam, ...], QueryConfig, ToolUseContext]
    ] = field(default_factory=_empty_recall_start_calls)

    def start(
        self,
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> RecordingMemoryPrefetch:
        self.start_calls.append((tuple(messages), config, ctx))
        return self.prefetch


@dataclass
class FailingCoordinatorRuntime:
    operation: str

    def has_processed_task_notification(
        self,
        _notification: TaskNotification,
        /,
    ) -> bool:
        return False

    def record_task_notifications(
        self,
        _notifications: Sequence[TaskNotification],
        /,
    ) -> object:
        if self.operation == "record":
            raise RuntimeError("coordinator secret should not leak")
        return object()

    def record_agent_launch(self, **_kwargs: object) -> object:
        return object()

    def record_send_message(self, **_kwargs: object) -> object:
        return object()

    def record_task_stop(self, **_kwargs: object) -> object:
        return object()

    def render_context(self) -> MessageParam | None:
        if self.operation == "render":
            raise RuntimeError("coordinator render secret should not leak")
        return {
            "role": "user",
            "content": "<coordinator_runtime>ok</coordinator_runtime>",
            "raygentMessageKind": "coordinator_runtime",
        }


@dataclass
class RecordingTranscriptStore:
    entries: list[TranscriptEntry] = field(default_factory=list[TranscriptEntry])
    operations: list[str] = field(default_factory=list[str])
    scopes: list[TranscriptScope] = field(default_factory=list[TranscriptScope])

    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
        self.operations.append(f"append:{entry.type}")
        self.scopes.append(scope)
        self.entries.append(entry)

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        for entry in entries:
            await self.append(scope, entry)

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        _ = scope
        return list(self.entries)

    async def flush(self, scope: TranscriptScope | None = None) -> None:
        _ = scope
        self.operations.append("flush")

    def path_for(self, scope: TranscriptScope) -> str:
        return f"/tmp/{scope.session_id}.jsonl"


def _memory_recall_message(content: str = "memory context") -> MessageParam:
    return {
        "role": "user",
        "content": content,
        "raygentMessageKind": "memory_recall",
        "raygentMemoryRecall": {
            "type": "relevant_memories",
            "memories": [{"path": "/tmp/memory.md", "content_bytes": len(content)}],
        },
    }


def test_last_message_is_api_error_detects_marked_assistant_message() -> None:
    messages: list[MessageParam] = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Prompt is too long"}],
            "isApiErrorMessage": True,
            "apiError": "context_overflow",
        },
    ]

    assert _last_message_is_api_error(messages) is True


def _patch_clean_response(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        return {"text": text}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)


@pytest.mark.asyncio
async def test_submit_message_event_ordering_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First yielded SDKMessage is SDKSystemInit; last is exactly one
    SDKResult; assistant message in between."""
    _patch_clean_response(monkeypatch, "the answer")
    engine = _engine()

    events: list[Any] = []
    async for msg in engine.submit_message("hi"):
        events.append(msg)

    assert isinstance(events[0], SDKSystemInit)
    assert isinstance(events[-1], SDKResult)
    # Exactly one SDKResult.
    result_events = [e for e in events if isinstance(e, SDKResult)]
    assert len(result_events) == 1
    # Assistant message landed between init and result.
    assistant_events = [e for e in events if isinstance(e, SDKAssistantMessage)]
    assert len(assistant_events) == 1
    init_idx = events.index(events[0])
    asst_idx = events.index(assistant_events[0])
    res_idx = events.index(result_events[0])
    assert init_idx < asst_idx < res_idx


@pytest.mark.asyncio
async def test_submit_message_records_user_prompt_before_query_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_model = False

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        nonlocal called_model
        called_model = True
        return {"text": "ok"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    store = RecordingTranscriptStore()
    engine = _engine_with_deps(
        QueryDeps(task_store=AppStateStore(), transcript_store=store)
    )

    stream = engine.submit_message("hi")
    init = await anext(stream)
    await stream.aclose()

    assert isinstance(init, SDKSystemInit)
    assert called_model is False
    assert len(store.entries) == 1
    first = store.entries[0]
    assert isinstance(first, TranscriptMessageEntry)
    assert first.message == {"role": "user", "content": "hi"}
    assert first.parent_entry_id is None
    assert first.version == RAYGENT_VERSION


@pytest.mark.asyncio
async def test_submit_message_flushes_transcript_before_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_response(monkeypatch, "the answer")
    store = RecordingTranscriptStore()
    engine = _engine_with_deps(
        QueryDeps(task_store=AppStateStore(), transcript_store=store)
    )

    stream = engine.submit_message("hi")
    init = await anext(stream)
    assistant = await anext(stream)
    assert isinstance(init, SDKSystemInit)
    assert isinstance(assistant, SDKAssistantMessage)
    assert "flush" not in store.operations

    result = await anext(stream)

    assert isinstance(result, SDKResult)
    assert store.operations[-1] == "flush"


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_source", ["ctx", "config"])
async def test_submit_message_does_not_implicitly_persist_subagent_transcript(
    monkeypatch: pytest.MonkeyPatch,
    agent_source: str,
) -> None:
    _patch_clean_response(monkeypatch, "ok")
    store = RecordingTranscriptStore()
    ctx = _ctx()
    config = QueryConfig(model="claude-opus-4-7", session_id="child-session")
    if agent_source == "ctx":
        ctx.agent_id = "local_agent_1"
    else:
        config = replace(config, agent_id="local_agent_1")
    engine = QueryEngine(
        config,
        QueryDeps(task_store=AppStateStore(), transcript_store=store),
        ctx,
    )

    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    assert store.entries == []
    assert store.operations == []


@pytest.mark.asyncio
async def test_submit_message_persists_memory_recall_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_message = _memory_recall_message()
    responses: list[dict[str, object]] = [{"tool": True}, {"text": "done"}]
    tool_result_message: MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "tool done",
            }
        ],
    }

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> dict[str, object]:
        return responses.pop(0)

    def fake_assistant(response: dict[str, object]) -> MessageParam:
        if response.get("tool") is True:
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Example",
                        "input": {},
                    }
                ],
            }
        return {"role": "assistant", "content": "done"}

    def fake_tool_uses(response: dict[str, object]) -> list[object]:
        return [object()] if response.get("tool") is True else []

    async def fake_orchestrate(
        _assistant_message: MessageParam,
        _state: Any,
        _config: QueryConfig,
        _deps: QueryDeps,
        ctx: ToolUseContext,
    ) -> AsyncIterator[object]:
        yield ToolResultMessage(message=tool_result_message)
        yield ToolOrchestrationComplete(
            tool_result_messages=(tool_result_message,),
            updated_context=ctx,
        )

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_observable_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)
    monkeypatch.setattr(query_mod, "_orchestrate_tools", fake_orchestrate)

    prefetch = RecordingMemoryPrefetch((memory_message,))
    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        memory_recall_provider=RecordingMemoryRecallProvider(prefetch),
        transcript_store=store,
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    memory_events = [
        event
        for event in events
        if isinstance(event, SDKUserMessage) and event.message == memory_message
    ]
    assert len(memory_events) == 1
    assert prefetch.consume_calls == [1]
    assert prefetch.cancel_count == 1
    assert any(
        message == memory_message
        for message in engine._messages  # pyright: ignore[reportPrivateUsage]
    )
    assert any(
        isinstance(entry, TranscriptMessageEntry) and entry.message == memory_message
        for entry in store.entries
    )


@pytest.mark.asyncio
async def test_submit_message_persists_task_notification_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_response(monkeypatch, "ok")
    task_store = AppStateStore()
    task_store.enqueue_notification(
        TaskNotification(
            task_id="task_1",
            kind="completed",
            message="background work done",
            priority="next",
        )
    )
    store = RecordingTranscriptStore()
    engine = _engine_with_deps(
        QueryDeps(task_store=task_store, transcript_store=store)
    )

    events = [msg async for msg in engine.submit_message("hi")]

    notification_events = [
        event
        for event in events
        if isinstance(event, SDKUserMessage)
        and "task=task_1 kind=completed" in str(event.message.get("content"))
    ]
    assert len(notification_events) == 1
    assert any(
        isinstance(entry, TranscriptMessageEntry)
        and "background work done" in str(entry.message.get("content"))
        for entry in store.entries
    )


@pytest.mark.asyncio
async def test_submit_message_drains_local_agent_queued_messages_before_model_call() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "child reply"},)
    )
    store = AppStateStore()
    store.register_task(
        LocalAgentState(
            id="la1",
            type="local_agent",
            description="worker",
            status="running",
            start_time=1.0,
            prompt="initial",
            pending_messages=(
                LocalAgentPendingMessage(
                    sender="team-lead",
                    content="queued follow-up",
                    summary="follow up",
                ),
            ),
        )
    )
    engine = QueryEngine(
        QueryConfig(model="m", session_id="s", agent_id="la1"),
        QueryDeps(task_store=store, model_provider=provider),
        ToolUseContext(
            session_id="sub-la1",
            agent_id="la1",
            abort_event=asyncio.Event(),
            rendered_system_prompt="",
            cwd="/repo",
        ),
    )

    events = [event async for event in engine.submit_message("initial prompt")]
    request_messages = [
        message_param_from_api_message(message)
        for message in provider.requests[0].messages
    ]
    user_events = [event for event in events if isinstance(event, SDKUserMessage)]
    task = store.tasks["la1"]

    assert isinstance(task, LocalAgentState)
    assert task.pending_messages == ()
    assert request_messages[0] == {"role": "user", "content": "initial prompt"}
    assert request_messages[1]["role"] == "user"
    assert request_messages[1]["content"] == "queued follow-up"
    assert request_messages[1].get("raygentMessageKind") == (
        "local_agent_queued_message"
    )
    assert len(user_events) == 1
    assert user_events[0].message == request_messages[1]
    assert engine._messages[1] == request_messages[1]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_submit_message_records_coordinator_runtime_after_task_notifications() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
    )
    task_store = AppStateStore()
    task_store.enqueue_notification(
        TaskNotification(
            task_id="task_1",
            kind="completed",
            message=(
                "<task_notification><task_id>task_1</task_id>"
                "<status>completed</status><result>SECRET RAW RESULT</result>"
                "</task_notification>"
            ),
            priority="next",
            tool_use_id="toolu_1",
        )
    )
    transcript_store = RecordingTranscriptStore()
    runtime = CoordinatorRuntime()
    deps = QueryDeps(
        task_store=task_store,
        model_provider=provider,
        transcript_store=transcript_store,
        coordinator_runtime=runtime,
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    request_messages = [
        message_param_from_api_message(api_message)
        for api_message in provider.requests[0].messages
    ]
    raw_index = next(
        index
        for index, message in enumerate(request_messages)
        if "task=task_1 kind=completed" in str(message.get("content"))
    )
    coordinator_index = next(
        index
        for index, message in enumerate(request_messages)
        if message.get("raygentMessageKind") == "coordinator_runtime"
    )
    coordinator_message = request_messages[coordinator_index]
    coordinator_content = str(coordinator_message.get("content"))

    assert raw_index < coordinator_index
    assert "SECRET RAW RESULT" in str(request_messages[raw_index].get("content"))
    assert "SECRET RAW RESULT" not in coordinator_content
    assert "task_1" in coordinator_content
    assert runtime.snapshot().processed_notification_count == 1

    user_event_messages = [
        event.message
        for event in events
        if isinstance(event, SDKUserMessage)
    ]
    assert user_event_messages[0] == request_messages[raw_index]
    assert user_event_messages[1] == coordinator_message

    transcript_messages = [
        entry.message
        for entry in transcript_store.entries
        if isinstance(entry, TranscriptMessageEntry)
    ]
    transcript_raw_index = transcript_messages.index(request_messages[raw_index])
    transcript_coordinator_index = transcript_messages.index(coordinator_message)
    assert transcript_raw_index < transcript_coordinator_index


@pytest.mark.parametrize(
    ("operation", "expected_failure_operation"),
    (
        ("record", "record_task_notifications"),
        ("render", "render_context"),
    ),
)
@pytest.mark.asyncio
async def test_submit_message_coordinator_runtime_failure_does_not_swallow_notification(
    operation: str,
    expected_failure_operation: str,
) -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
    )
    sink = RecordingKernelEventSink()
    task_store = AppStateStore()
    task_store.enqueue_notification(
        TaskNotification(
            task_id="task_1",
            kind="completed",
            message="background work done",
            priority="next",
        )
    )
    deps = QueryDeps(
        task_store=task_store,
        model_provider=provider,
        coordinator_runtime=FailingCoordinatorRuntime(operation=operation),
        observability=KernelEventBus([sink]),
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    request_messages = [
        message_param_from_api_message(api_message)
        for api_message in provider.requests[0].messages
    ]
    assert any(
        "background work done" in str(message.get("content"))
        for message in request_messages
    )
    assert not any(
        message.get("raygentMessageKind") == "coordinator_runtime"
        for message in request_messages
    )
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    failure_events = [
        event
        for event in sink.events
        if event.type == "coordinator.runtime.integration_failed"
    ]
    assert len(failure_events) == 1
    assert failure_events[0].data["operation"] == expected_failure_operation
    assert failure_events[0].data["error_type"] == "RuntimeError"
    assert "secret" not in str(failure_events[0].data).lower()


@pytest.mark.asyncio
async def test_submit_message_injects_existing_coordinator_runtime_context() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
    )
    runtime = CoordinatorRuntime()
    runtime.add_work_item(
        kind="research",
        title="Review coordinator runtime",
        status="running",
        task_id="task_1",
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        coordinator_runtime=runtime,
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    request_messages = [
        message_param_from_api_message(api_message)
        for api_message in provider.requests[0].messages
    ]
    coordinator_messages = [
        message
        for message in request_messages
        if message.get("raygentMessageKind") == "coordinator_runtime"
    ]
    assert len(coordinator_messages) == 1
    assert "Review coordinator runtime" in str(coordinator_messages[0].get("content"))
    assert any(
        isinstance(event, SDKUserMessage) and event.message == coordinator_messages[0]
        for event in events
    )


@pytest.mark.asyncio
async def test_submit_message_persists_stop_hook_block_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        nonlocal call_count
        call_count += 1
        return {"text": f"reply-{call_count}"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    hook_calls = 0

    async def block_once(_ctx: HookContext) -> HookResult:
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            return HookBlock("please add detail")
        return HookContinue()

    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=store,
        stop_hooks=[block_once],
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    assert call_count == 2
    stop_feedback_events = [
        event
        for event in events
        if isinstance(event, SDKUserMessage)
        and "please add detail" in str(event.message.get("content"))
    ]
    assert len(stop_feedback_events) == 1
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"

    messages = [
        entry.message
        for entry in store.entries
        if isinstance(entry, TranscriptMessageEntry)
    ]
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "please add detail"},
        {"role": "assistant", "content": "reply-2"},
    ]
    assert engine._messages == messages  # pyright: ignore[reportPrivateUsage]

    replay = replay_entries(store.entries, scope=TranscriptScope(session_id="s"))
    assert replay.messages == messages


@pytest.mark.asyncio
async def test_submit_message_persists_stop_hook_block_before_prevent_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_response(monkeypatch, "reply-1")

    async def block_hook(_ctx: HookContext) -> HookResult:
        return HookBlock("please add detail")

    async def prevent_hook(_ctx: HookContext) -> HookResult:
        return HookPreventContinuation("not done")

    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        transcript_store=store,
        stop_hooks=[block_hook, prevent_hook],
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    stop_feedback_events = [
        event
        for event in events
        if isinstance(event, SDKUserMessage)
        and "please add detail" in str(event.message.get("content"))
    ]
    assert len(stop_feedback_events) == 1
    assert isinstance(events[-1], SDKResult)
    assert events[-1].is_error is True
    assert events[-1].subtype == "error_during_execution"

    messages = [
        entry.message
        for entry in store.entries
        if isinstance(entry, TranscriptMessageEntry)
    ]
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "please add detail"},
    ]
    assert engine._messages == messages  # pyright: ignore[reportPrivateUsage]

    replay = replay_entries(store.entries, scope=TranscriptScope(session_id="s"))
    assert replay.messages == messages


@pytest.mark.asyncio
async def test_submit_message_persists_continuation_context_before_retry() -> None:
    hook_calls = 0

    async def context_once(_ctx: HookContext) -> HookResult:
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            return HookContinueWithContext(
                message="Add policy context.",
                fragments=(
                    ContinuationContextFragment(
                        id="ctx-1",
                        content="Retry with the verified input.",
                        source="policy",
                        reason="missing detail",
                    ),
                ),
            )
        return HookContinue()

    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=(
                {"role": "assistant", "content": "reply-1"},
                {"role": "assistant", "content": "reply-2"},
            )
        ),
        transcript_store=store,
        stop_hooks=[context_once],
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("hi")]

    context_events = [
        event
        for event in events
        if isinstance(event, SDKUserMessage)
        and event.message.get("raygentMessageKind") == "continuation_context"
    ]
    assert len(context_events) == 1
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"

    messages = [
        entry.message
        for entry in store.entries
        if isinstance(entry, TranscriptMessageEntry)
    ]
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1] == {"role": "assistant", "content": "reply-1"}
    assert messages[2] == context_events[0].message
    assert messages[2].get("raygentMessageKind") == "continuation_context"
    assert "Retry with the verified input." in str(messages[2]["content"])
    assert messages[3] == {"role": "assistant", "content": "reply-2"}
    assert engine._messages == messages  # pyright: ignore[reportPrivateUsage]

    replay = replay_entries(store.entries, scope=TranscriptScope(session_id="s"))
    assert replay.messages == messages


@pytest.mark.asyncio
async def test_submit_message_close_cancels_memory_recall_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_response(monkeypatch, "the answer")
    prefetch = RecordingMemoryPrefetch((_memory_recall_message(),))
    deps = QueryDeps(
        task_store=AppStateStore(),
        memory_recall_provider=RecordingMemoryRecallProvider(prefetch),
    )
    engine = _engine_with_deps(deps)

    stream = engine.submit_message("hi")
    init = await anext(stream)
    assistant = await anext(stream)
    assert isinstance(init, SDKSystemInit)
    assert isinstance(assistant, SDKAssistantMessage)

    await stream.aclose()

    assert prefetch.cancel_count == 1
    assert prefetch.consume_calls == []


@pytest.mark.asyncio
async def test_context_provider_adds_non_persistent_model_context() -> None:
    provider_calls: list[tuple[str, str | None]] = []

    async def context_provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        provider_calls.append((config.system_prompt, ctx.agent_id))
        return (
            ContextFragment(
                id="env",
                content="<env>\n  cwd: /repo\n</env>",
                channel="system",
                priority=0,
            ),
            ContextFragment(
                id="project-instructions",
                source="/repo/AGENTS.md",
                content="Prefer deterministic tests.",
                channel="user_context",
                priority=1,
            ),
        )

    async def system_prompt_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return "legacy system provider"

    async def memory_prompt_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return "memory mechanics"

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        transcript_store=store,
        context_providers=(context_provider,),
        system_prompt_provider=system_prompt_provider,
        memory_prompt_provider=memory_prompt_provider,
    )
    engine = _engine_with_deps(
        deps,
        config=QueryConfig(
            model="model-1",
            session_id="s",
            system_prompt="base prompt",
        ),
    )

    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    assert provider_calls == [("base prompt", None)]
    request = provider.requests[0]
    assert request.system_prompt == (
        "base prompt\n\n"
        "legacy system provider\n\n"
        "memory mechanics\n\n"
        "<env>\n  cwd: /repo\n</env>"
    )

    request_messages = [
        message_param_from_api_message(message) for message in request.messages
    ]
    assert len(request_messages) == 2
    assert request_messages[0]["role"] == "user"
    assert "/repo/AGENTS.md" in str(request_messages[0]["content"])
    assert "Prefer deterministic tests." in str(request_messages[0]["content"])
    assert request_messages[1] == {"role": "user", "content": "hi"}

    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    transcript_messages = [
        entry.message for entry in store.entries if isinstance(entry, TranscriptMessageEntry)
    ]
    assert transcript_messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_context_provider_is_fail_soft_and_agent_scoped() -> None:
    async def raising_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        raise RuntimeError("provider boom")

    async def scoped_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                id="main-only",
                content="main instructions",
                channel="user_context",
                agent_scope="main",
            ),
            ContextFragment(
                id="child-only",
                content="child instructions",
                channel="user_context",
                agent_scope="subagent",
            ),
        )

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    child_ctx = _ctx()
    child_ctx.agent_id = "agent-child"
    engine = _engine_with_ctx(
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=provider,
            context_providers=(raising_provider, scoped_provider),
        ),
        child_ctx,
        config=QueryConfig(model="model-1", session_id="s", agent_id="agent-child"),
    )

    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    assert "child instructions" in str(request_messages[0]["content"])
    assert "main instructions" not in str(request_messages[0]["content"])
    assert request_messages[1] == {"role": "user", "content": "hi"}


class FallbackTriggeredError(Exception):
    pass


@pytest.mark.asyncio
async def test_context_provider_snapshot_is_reused_across_model_fallback() -> None:
    calls = 0

    async def context_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        nonlocal calls
        calls += 1
        return (
            ContextFragment(id="env", content="env snapshot", channel="system"),
            ContextFragment(
                id="instructions",
                content="instruction snapshot",
                channel="user_context",
            ),
        )

    provider = FakeModelProvider(
        responses=(
            FallbackTriggeredError("fallback now"),
            {"role": "assistant", "content": "ok"},
        )
    )
    engine = _engine_with_deps(
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=provider,
            context_providers=(context_provider,),
        ),
        config=QueryConfig(
            model="primary-model",
            fallback_model="fallback-model",
            session_id="s",
            system_prompt="base",
        ),
    )

    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    assert calls == 1
    assert [request.model for request in provider.requests] == [
        "primary-model",
        "fallback-model",
    ]
    first_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    second_messages = [
        message_param_from_api_message(message) for message in provider.requests[1].messages
    ]
    assert first_messages == second_messages
    assert "instruction snapshot" in str(first_messages[0]["content"])
    assert provider.requests[0].system_prompt == "base\n\nenv snapshot"
    assert provider.requests[1].system_prompt == "base\n\nenv snapshot"


@pytest.mark.asyncio
async def test_agent_trigger_policy_injects_guidance_after_init() -> None:
    calls: list[tuple[MessageParam, tuple[MessageParam, ...], str | None]] = []

    async def policy(
        prompt: MessageParam,
        history: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        assert config.model == "model-1"
        calls.append((prompt, tuple(history), ctx.agent_id))
        return AgentTriggerDecision(
            matches=(
                AgentTriggerMatch(
                    id="frontend",
                    agent_name="frontend-agent",
                    reason="UI work is likely relevant",
                    prompt_hint="Review the UI request and decide whether to delegate.",
                    confidence=0.82,
                    source="test-policy",
                ),
            )
        )

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        transcript_store=store,
        agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
    )
    engine = _engine_with_deps(deps, config=QueryConfig(model="model-1", session_id="s"))

    events = [msg async for msg in engine.submit_message("build a settings page")]

    assert [type(event) for event in events] == [
        SDKSystemInit,
        SDKUserMessage,
        SDKAssistantMessage,
        SDKResult,
    ]
    trigger_event = events[1]
    assert isinstance(trigger_event, SDKUserMessage)
    assert trigger_event.message.get("raygentMessageKind") == "agent_trigger"
    assert "frontend-agent" in str(trigger_event.message["content"])
    assert "No agent-delegation tool is available" in str(trigger_event.message["content"])
    assert "available Agent/Task tool path" not in str(trigger_event.message["content"])
    metadata = trigger_event.message.get("raygentAgentTrigger")
    assert metadata is not None
    assert metadata["delegation_tool_available"] is False

    request_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    assert request_messages[0] == {
        "role": "user",
        "content": "build a settings page",
    }
    assert request_messages[1] == trigger_event.message
    assert calls == [
        (
            {"role": "user", "content": "build a settings page"},
            (),
            None,
        )
    ]
    transcript_messages = [
        entry.message for entry in store.entries if isinstance(entry, TranscriptMessageEntry)
    ]
    assert transcript_messages == [
        {"role": "user", "content": "build a settings page"},
        trigger_event.message,
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_agent_trigger_policy_mentions_delegation_only_when_agent_tool_available() -> None:
    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        return AgentTriggerDecision(
            matches=(AgentTriggerMatch(id="frontend", agent_name="frontend-agent"),)
        )

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
        agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
    )
    engine = _engine_with_deps(
        deps,
        config=QueryConfig(
            model="model-1",
            session_id="s",
            tools=(_agent_named_tool(),),
        ),
    )

    events = [msg async for msg in engine.submit_message("build a settings page")]

    trigger_event = next(event for event in events if isinstance(event, SDKUserMessage))
    assert "available Agent/Task tool path" in str(trigger_event.message["content"])
    assert "No agent-delegation tool is available" not in str(
        trigger_event.message["content"]
    )
    metadata = trigger_event.message.get("raygentAgentTrigger")
    assert metadata is not None
    assert metadata["delegation_tool_available"] is True
    completed = sink.by_type("agent_trigger.policy.completed")[-1]
    assert completed.data["delegation_tool_available"] is True


@pytest.mark.asyncio
async def test_agent_trigger_policy_respects_model_visible_delegation_tool_gate() -> None:
    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        return AgentTriggerDecision(
            matches=(AgentTriggerMatch(id="frontend", agent_name="frontend-agent"),)
        )

    async def run_case(
        *,
        tool: Any,
        seed_selected: bool = False,
    ) -> tuple[SDKUserMessage, RecordingKernelEventSink]:
        provider = FakeModelProvider(
            responses=({"role": "assistant", "content": "ok"},)
        )
        sink = RecordingKernelEventSink()
        deps = QueryDeps(
            task_store=AppStateStore(),
            model_provider=provider,
            observability=KernelEventBus([sink]),
            agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
        )
        engine = _engine_with_deps(
            deps,
            config=QueryConfig(
                model="model-1",
                session_id="s",
                tools=(tool,),
            ),
        )
        if seed_selected:
            await engine.seed_messages((_tool_reference_message("Agent"),))
        events = [msg async for msg in engine.submit_message("build a settings page")]
        trigger_event = next(event for event in events if isinstance(event, SDKUserMessage))
        return trigger_event, sink

    deferred_event, deferred_sink = await run_case(
        tool=_agent_named_tool(should_defer=True),
    )
    assert "No agent-delegation tool is available" in str(
        deferred_event.message["content"]
    )
    assert "available Agent/Task tool path" not in str(deferred_event.message["content"])
    deferred_metadata = deferred_event.message.get("raygentAgentTrigger")
    assert deferred_metadata is not None
    assert deferred_metadata["delegation_tool_available"] is False
    assert (
        deferred_sink.by_type("agent_trigger.policy.completed")[-1].data[
            "delegation_tool_available"
        ]
        is False
    )

    selected_event, selected_sink = await run_case(
        tool=_agent_named_tool(should_defer=True),
        seed_selected=True,
    )
    assert "available Agent/Task tool path" in str(selected_event.message["content"])
    selected_metadata = selected_event.message.get("raygentAgentTrigger")
    assert selected_metadata is not None
    assert selected_metadata["delegation_tool_available"] is True
    assert (
        selected_sink.by_type("agent_trigger.policy.completed")[-1].data[
            "delegation_tool_available"
        ]
        is True
    )

    always_loaded_event, _always_loaded_sink = await run_case(
        tool=_agent_named_tool(should_defer=True, always_load=True),
    )
    assert "available Agent/Task tool path" in str(always_loaded_event.message["content"])
    always_loaded_metadata = always_loaded_event.message.get("raygentAgentTrigger")
    assert always_loaded_metadata is not None
    assert (
        always_loaded_metadata["delegation_tool_available"] is True
    )

    disabled_event, _disabled_sink = await run_case(
        tool=_agent_named_tool(is_enabled=False),
    )
    assert "No agent-delegation tool is available" in str(
        disabled_event.message["content"]
    )
    disabled_metadata = disabled_event.message.get("raygentAgentTrigger")
    assert disabled_metadata is not None
    assert disabled_metadata["delegation_tool_available"] is False


@pytest.mark.asyncio
async def test_agent_trigger_policy_is_fail_soft_and_observable() -> None:
    sink = RecordingKernelEventSink()

    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        raise RuntimeError("policy saw secret prompt")

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
        agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
    )
    engine = _engine_with_deps(deps, config=QueryConfig(model="model-1", session_id="s"))

    events = [msg async for msg in engine.submit_message("secret prompt")]

    assert [type(event) for event in events] == [
        SDKSystemInit,
        SDKAssistantMessage,
        SDKResult,
    ]
    request_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    assert request_messages == [{"role": "user", "content": "secret prompt"}]
    completed = sink.by_type("agent_trigger.policy.completed")[-1]
    assert completed.data["failed"] is True
    assert completed.data["error_type"] == "RuntimeError"
    assert "secret prompt" not in str(completed.data)


@pytest.mark.asyncio
async def test_agent_trigger_policy_failure_debug_log_redacts_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="raygent_harness.core.query_engine")

    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        raise RuntimeError("policy saw SECRET_PROMPT_VALUE")

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
    )
    engine = _engine_with_deps(deps, config=QueryConfig(model="model-1", session_id="s"))

    events = [msg async for msg in engine.submit_message("SECRET_PROMPT_VALUE")]

    assert isinstance(events[-1], SDKResult)
    assert "RuntimeError" in caplog.text
    assert "SECRET_PROMPT_VALUE" not in caplog.text


@pytest.mark.asyncio
async def test_agent_trigger_policy_scope_defaults_to_main_and_can_opt_into_subagents() -> None:
    calls: list[str | None] = []

    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        calls.append(ctx.agent_id)
        return AgentTriggerDecision(
            matches=(AgentTriggerMatch(id="child", agent_name="child-agent"),)
        )

    child_ctx = _ctx()
    child_ctx.agent_id = "local_agent_1"
    skipped_provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "skipped"},)
    )
    skipped = _engine_with_ctx(
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=skipped_provider,
            agent_trigger_policy=AgentTriggerPolicySpec(policy=policy),
        ),
        child_ctx,
        config=QueryConfig(model="model-1", session_id="s", agent_id="local_agent_1"),
    )

    skipped_events = [msg async for msg in skipped.submit_message("hi")]
    assert isinstance(skipped_events[-1], SDKResult)
    assert calls == []
    skipped_messages = [
        message_param_from_api_message(message)
        for message in skipped_provider.requests[0].messages
    ]
    assert skipped_messages == [{"role": "user", "content": "hi"}]

    opted_provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "opted"},)
    )
    opted = _engine_with_ctx(
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=opted_provider,
            agent_trigger_policy=AgentTriggerPolicySpec(
                policy=policy,
                agent_scope="subagent",
            ),
        ),
        child_ctx,
        config=QueryConfig(model="model-1", session_id="s", agent_id="local_agent_1"),
    )

    opted_events = [msg async for msg in opted.submit_message("hi")]

    assert calls == ["local_agent_1"]
    trigger_events = [event for event in opted_events if isinstance(event, SDKUserMessage)]
    assert len(trigger_events) == 1
    assert trigger_events[0].message.get("raygentMessageKind") == "agent_trigger"


@pytest.mark.asyncio
async def test_agent_trigger_policy_bounds_messages_and_preserves_match_order() -> None:
    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        return AgentTriggerDecision(
            matches=(
                AgentTriggerMatch(id="a", agent_name="agent-a", reason="first"),
                AgentTriggerMatch(id="b", agent_name="agent-b", reason="second"),
            ),
            model_visible_messages=(
                {
                    "role": "assistant",
                    "content": "x" * 500,
                },
                {
                    "role": "user",
                    "content": "this extra message is over the count limit",
                },
            ),
            suppress_main_turn=True,
        )

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
        agent_trigger_policy=AgentTriggerPolicySpec(
            policy=policy,
            max_messages=2,
            max_message_chars=420,
        ),
    )
    engine = _engine_with_deps(deps, config=QueryConfig(model="model-1", session_id="s"))

    events = [msg async for msg in engine.submit_message("hi")]

    trigger_messages = [event.message for event in events if isinstance(event, SDKUserMessage)]
    assert len(trigger_messages) == 2
    assert all(message["role"] == "user" for message in trigger_messages)
    assert all(len(str(message["content"])) <= 420 for message in trigger_messages)
    first_content = str(trigger_messages[0]["content"])
    assert first_content.index("agent-a") < first_content.index("agent-b")
    second_content = str(trigger_messages[1]["content"])
    assert "agent trigger guidance truncated" in second_content
    metadata = trigger_messages[0].get("raygentAgentTrigger")
    assert metadata is not None
    assert metadata["suppress_main_turn_requested"] is True
    assert metadata["suppress_main_turn_applied"] is False
    assert metadata["dropped_message_count"] == 1
    assert metadata["matches"][0]["agent_name"] == "agent-a"
    completed = sink.by_type("agent_trigger.policy.completed")[-1]
    assert completed.data["injected_message_count"] == 2
    assert completed.data["dropped_message_count"] == 1
    assert completed.data["truncated_message_count"] == 1
    assert completed.data["suppress_main_turn_requested"] is True


@pytest.mark.asyncio
async def test_agent_trigger_policy_caps_rendered_match_count_before_building_message() -> None:
    async def policy(
        _prompt: MessageParam,
        _history: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        return AgentTriggerDecision(
            matches=tuple(
                AgentTriggerMatch(id=f"id-{index}", agent_name=f"agent-{index}")
                for index in range(100)
            )
        )

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        agent_trigger_policy=AgentTriggerPolicySpec(
            policy=policy,
            max_message_chars=8_000,
        ),
    )
    engine = _engine_with_deps(deps, config=QueryConfig(model="model-1", session_id="s"))

    events = [msg async for msg in engine.submit_message("hi")]

    trigger_messages = [event.message for event in events if isinstance(event, SDKUserMessage)]
    assert len(trigger_messages) == 1
    content = str(trigger_messages[0]["content"])
    assert "agent-0" in content
    assert "agent-63" in content
    assert "agent-64" not in content
    assert "36 additional matches omitted" in content
    metadata = trigger_messages[0].get("raygentAgentTrigger")
    assert metadata is not None
    assert metadata["match_count"] == 100
    assert len(metadata["matches"]) == 64


@pytest.mark.asyncio
async def test_submit_message_accumulates_provider_usage_across_turns() -> None:
    response_1 = replace(
        model_response_from_message_param({"role": "assistant", "content": "one"}),
        usage=Usage(input_tokens=7, output_tokens=3, cache_read_input_tokens=2),
    )
    response_2 = replace(
        model_response_from_message_param({"role": "assistant", "content": "two"}),
        usage=Usage(input_tokens=11, output_tokens=5, cache_creation_input_tokens=4),
    )
    provider = FakeModelProvider(responses=(response_1, response_2))
    engine = _engine_with_deps(
        QueryDeps(task_store=AppStateStore(), model_provider=provider),
        config=QueryConfig(model="model-1", session_id="s"),
    )

    first_events = [msg async for msg in engine.submit_message("hi")]
    second_events = [msg async for msg in engine.submit_message("again")]

    first_result = first_events[-1]
    second_result = second_events[-1]
    assert isinstance(first_result, SDKResult)
    assert isinstance(second_result, SDKResult)
    assert first_result.usage.input_tokens == 7
    assert first_result.usage.output_tokens == 3
    assert first_result.usage.cache_read_input_tokens == 2

    assert second_result.usage.input_tokens == 18
    assert second_result.usage.output_tokens == 8
    assert second_result.usage.cache_read_input_tokens == 2
    assert second_result.usage.cache_creation_input_tokens == 4


@pytest.mark.asyncio
async def test_submit_message_appends_memory_prompt_to_turn_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []
    provider_calls: list[tuple[str, str]] = []

    async def memory_prompt_provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str:
        provider_calls.append((config.system_prompt, ctx.rendered_system_prompt))
        return "memory mechanics"

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        config: QueryConfig,
        _deps: QueryDeps,
        ctx: ToolUseContext,
    ) -> Any:
        seen.append((config.system_prompt, ctx.rendered_system_prompt))
        return {"text": "ok"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    deps = QueryDeps(
        task_store=AppStateStore(),
        memory_prompt_provider=memory_prompt_provider,
    )
    engine = _engine_with_deps(
        deps,
        config=QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            system_prompt="base prompt",
        ),
    )

    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    assert provider_calls == [("base prompt", "")]
    assert seen == [
        ("base prompt\n\nmemory mechanics", "base prompt\n\nmemory mechanics")
    ]


@pytest.mark.asyncio
async def test_memory_prompt_provider_failure_blank_and_none_are_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_prompts: list[str] = []

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_prompts.append(config.system_prompt)
        return {"text": "ok"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    async def raising_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        raise RuntimeError("provider boom")

    async def blank_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return "   "

    async def none_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str | None:
        return None

    for provider in (raising_provider, blank_provider, none_provider):
        engine = _engine_with_deps(
            QueryDeps(
                task_store=AppStateStore(),
                memory_prompt_provider=provider,
            ),
            config=QueryConfig(
                model="claude-opus-4-7",
                session_id="s",
                system_prompt="base prompt",
            ),
        )
        events = [msg async for msg in engine.submit_message("hi")]
        assert isinstance(events[-1], SDKResult)

    assert seen_prompts == ["base prompt", "base prompt", "base prompt"]


@pytest.mark.asyncio
async def test_memory_extractor_runs_only_on_completed_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_calls: list[tuple[list[MessageParam], str | None]] = []

    async def extractor(
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> None:
        _ = config
        extraction_calls.append((list(messages), ctx.agent_id))

    _patch_clean_response(monkeypatch, "ok")
    completed_engine = _engine_with_deps(
        QueryDeps(
            task_store=AppStateStore(),
            memory_extractor=extractor,
        )
    )
    completed_events = [msg async for msg in completed_engine.submit_message("hi")]
    assert await completed_engine.drain_memory_extractions(timeout_s=1.0)

    assert isinstance(completed_events[-1], SDKResult)
    assert extraction_calls == [
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
            None,
        )
    ]

    extraction_calls.clear()

    async def raising_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        raise RuntimeError("model boom")

    monkeypatch.setattr(query_mod, "_call_model", raising_call)
    error_engine = _engine_with_deps(
        QueryDeps(
            task_store=AppStateStore(),
            memory_extractor=extractor,
        )
    )
    error_events = [msg async for msg in error_engine.submit_message("hi")]
    assert await error_engine.drain_memory_extractions(timeout_s=0.001)

    assert isinstance(error_events[-1], SDKResult)
    assert error_events[-1].is_error is True
    assert extraction_calls == []

    _patch_clean_response(monkeypatch, "ok")

    async def prevent_hook(_ctx: HookContext) -> HookResult:
        return HookPreventContinuation("not done")

    prevented_engine = _engine_with_deps(
        QueryDeps(
            task_store=AppStateStore(),
            stop_hooks=[prevent_hook],
            memory_extractor=extractor,
        )
    )
    prevented_events = [msg async for msg in prevented_engine.submit_message("hi")]
    assert await prevented_engine.drain_memory_extractions(timeout_s=0.001)

    assert isinstance(prevented_events[-1], SDKResult)
    assert prevented_events[-1].is_error is True
    assert extraction_calls == []


@pytest.mark.asyncio
async def test_memory_extractor_does_not_schedule_for_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_calls: list[str | None] = []

    async def extractor(
        _messages: Sequence[MessageParam],
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> None:
        extraction_calls.append(ctx.agent_id)

    _patch_clean_response(monkeypatch, "ok")
    ctx = ToolUseContext(
        session_id="s",
        agent_id="local_agent_1",
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=1),
    )
    engine = _engine_with_ctx(
        QueryDeps(
            task_store=AppStateStore(),
            memory_extractor=extractor,
        ),
        ctx,
    )

    events = [msg async for msg in engine.submit_message("hi")]
    assert await engine.drain_memory_extractions(timeout_s=0.001)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert extraction_calls == []


@pytest.mark.asyncio
async def test_memory_extraction_drain_timeout_does_not_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_response(monkeypatch, "ok")
    started = asyncio.Event()
    release = asyncio.Event()
    finished: list[bool] = []

    async def extractor(
        _messages: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> None:
        started.set()
        await release.wait()
        finished.append(True)

    engine = _engine_with_deps(
        QueryDeps(
            task_store=AppStateStore(),
            memory_extractor=extractor,
        )
    )
    events = [msg async for msg in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    await started.wait()
    assert not await engine.drain_memory_extractions(timeout_s=0.001)
    assert finished == []

    release.set()
    assert await engine.drain_memory_extractions(timeout_s=1.0)
    assert finished == [True]


@pytest.mark.asyncio
async def test_memory_extractor_receives_completed_turn_config() -> None:
    captured: list[tuple[tuple[MessageParam, ...], QueryConfig, ToolUseContext]] = []
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "answer"},),
        resolved_models={"base-model": "resolved-model"},
    )

    async def memory_prompt_provider(
        config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return f"memory prompt for {config.model}"

    async def extractor(
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> None:
        captured.append((tuple(messages), config, ctx))

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        memory_prompt_provider=memory_prompt_provider,
        memory_extractor=extractor,
    )
    engine = _engine_with_deps(
        deps,
        config=QueryConfig(
            model="base-model",
            session_id="s",
            system_prompt="base system",
        ),
    )

    events = [event async for event in engine.submit_message("hi")]
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert len(captured) == 1
    captured_messages, captured_config, captured_ctx = captured[0]
    model_request = provider.requests[0]
    assert captured_messages == (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer"},
    )
    assert captured_config.model == "base-model"
    assert captured_config.system_prompt == model_request.system_prompt
    assert captured_config.system_prompt == "base system\n\nmemory prompt for base-model"
    assert captured_config.tools == ()
    assert captured_ctx.agent_id is None
    assert model_request.model == "resolved-model"


@pytest.mark.asyncio
async def test_submit_message_event_ordering_exception_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the inner query() raises, submit_message must STILL yield a
    terminal SDKResult — and exactly one. SDKSystemInit must still come
    first.
    """

    async def raising_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        msg = "synthetic-call-failure"
        raise RuntimeError(msg)

    def fake_assistant(_response: Any) -> MessageParam:
        return {"role": "assistant", "content": ""}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", raising_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    engine = _engine()
    events: list[Any] = []
    async for msg in engine.submit_message("hi"):
        events.append(msg)

    assert isinstance(events[0], SDKSystemInit)
    result_events = [e for e in events if isinstance(e, SDKResult)]
    assert len(result_events) == 1
    final = result_events[0]
    assert final is events[-1]
    assert final.is_error is True
    # Either the inner recovery ladder mapped this to model_error, or the
    # outer except path produced error_during_execution. Both are valid
    # depending on classification — we just require an error subtype.
    assert final.subtype != "success"


@pytest.mark.asyncio
async def test_submit_message_reconciles_terminal_compacted_state_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compaction is a rewrite, not an append. The engine must replace its
    cross-turn log from `Terminal.final_state.messages` or the next turn will
    resurrect pre-compact history.
    """
    seen_model_inputs: list[list[MessageParam]] = []

    async def fake_call(
        msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_model_inputs.append(list(msgs))
        return {"text": f"reply-{len(seen_model_inputs)}"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        if any(m.get("content") == "first prompt" for m in messages):
            return LayerResult(
                messages=[
                    {
                        "role": "user",
                        "content": "[compacted summary of first prompt]",
                    }
                ]
            )
        return LayerResult(messages=messages)

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
    )
    engine = _engine_with_deps(deps)

    first_events = [msg async for msg in engine.submit_message("first prompt")]
    second_events = [msg async for msg in engine.submit_message("second prompt")]

    assert isinstance(first_events[-1], SDKResult)
    assert isinstance(second_events[-1], SDKResult)
    assert len(seen_model_inputs) == 2

    assert seen_model_inputs[0] == [
        {"role": "user", "content": "[compacted summary of first prompt]"}
    ]
    assert seen_model_inputs[1] == [
        {"role": "user", "content": "[compacted summary of first prompt]"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "second prompt"},
    ]
    assert {"role": "user", "content": "first prompt"} not in seen_model_inputs[1]


@pytest.mark.asyncio
async def test_submit_message_reconciles_compacted_state_after_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-error path must also preserve post-pipeline history.

    Before chunk-2 review fix, `query()` passed pre-pipeline `state` into
    `_handle_error(...)`, so `Terminal.final_state.messages` resurrected old
    history on error terminals. `QueryEngine` correctly reconciled from the
    terminal, but the terminal carried the wrong messages.
    """
    seen_model_inputs: list[list[MessageParam]] = []

    async def fake_call(
        msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_model_inputs.append(list(msgs))
        if len(seen_model_inputs) == 1:
            msg = "unrecoverable after compaction"
            raise RuntimeError(msg)
        return {"text": "second-ok"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        if any(m.get("content") == "first prompt" for m in messages):
            return LayerResult(
                messages=[
                    {
                        "role": "user",
                        "content": "[compacted summary before error]",
                    }
                ]
            )
        return LayerResult(messages=messages)

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
    )
    engine = _engine_with_deps(deps)

    first_events = [msg async for msg in engine.submit_message("first prompt")]
    second_events = [msg async for msg in engine.submit_message("second prompt")]

    assert isinstance(first_events[-1], SDKResult)
    assert first_events[-1].is_error is True
    assert isinstance(second_events[-1], SDKResult)
    assert len(seen_model_inputs) == 2

    assert seen_model_inputs[0] == [
        {"role": "user", "content": "[compacted summary before error]"}
    ]
    assert seen_model_inputs[1] == [
        {"role": "user", "content": "[compacted summary before error]"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "unrecoverable after compaction"}
            ],
            "isApiErrorMessage": True,
            "apiError": "fatal_unknown",
            "error": "fatal_unknown",
            "errorDetails": "unrecoverable after compaction",
        },
        {"role": "user", "content": "second prompt"},
    ]
    assert {"role": "user", "content": "first prompt"} not in seen_model_inputs[1]


@pytest.mark.asyncio
async def test_submit_message_preserves_compact_boundaries_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raygent keeps compact boundaries out of API-visible messages, so
    QueryEngine must carry them separately when reconstructing the next State.
    """
    seen_boundary_counts: list[int] = []

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        return {"text": f"reply-{len(seen_boundary_counts)}"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def fake_microcompact(
        messages: list[MessageParam],
        state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        seen_boundary_counts.append(len(state.compact_boundaries))
        if any(m.get("content") == "first prompt" for m in messages):
            return LayerResult(
                messages=[{"role": "user", "content": "[summary]"}],
                boundary=CompactBoundaryEvent(
                    kind="microcompact",
                    message_index=0,
                    summary="[summary]",
                ),
            )
        return LayerResult(messages=messages)

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
    )
    engine = _engine_with_deps(deps)

    first_events = [msg async for msg in engine.submit_message("first prompt")]
    second_events = [msg async for msg in engine.submit_message("second prompt")]

    assert [type(e) for e in first_events].count(SDKCompactBoundary) == 1
    summary_events = [
        e
        for e in first_events
        if isinstance(e, SDKUserMessage)
        and e.message == {"role": "user", "content": "[summary]"}
    ]
    assert len(summary_events) == 1
    assert isinstance(second_events[-1], SDKResult)
    assert seen_boundary_counts == [0, 1]


@pytest.mark.asyncio
async def test_submit_message_persists_compact_boundary_and_replays_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        return {"text": "after compact"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        if any(m.get("content") == "first prompt" for m in messages):
            return LayerResult(
                messages=[{"role": "user", "content": "[summary]"}],
                boundary=CompactBoundaryEvent(
                    kind="microcompact",
                    message_index=0,
                    summary="[summary]",
                ),
            )
        return LayerResult(messages=messages)

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    store = RecordingTranscriptStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
        transcript_store=store,
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("first prompt")]
    replay = replay_entries(store.entries, scope=TranscriptScope(session_id="s"))

    assert isinstance(events[-1], SDKResult)
    assert any(isinstance(entry, CompactBoundaryEntry) for entry in store.entries)
    assert replay.messages == [
        {"role": "user", "content": "[summary]"},
        {"role": "assistant", "content": "after compact"},
    ]
    assert replay.warnings == ()


@pytest.mark.asyncio
async def test_compact_boundary_append_failure_still_resets_transcript_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort transcript persistence must not corrupt replay topology.

    If the compact-boundary append fails but post-compact message writes succeed,
    those post-compact messages still need a fresh parent chain. Otherwise replay
    can walk back into stale pre-compact history.
    """

    class BoundaryFailingStore(RecordingTranscriptStore):
        async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
            if isinstance(entry, CompactBoundaryEntry):
                self.operations.append("append:compact_boundary:failed")
                raise OSError("boundary append failed")
            await super().append(scope, entry)

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        return {"text": "after compact"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: Any,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        if any(m.get("content") == "first prompt" for m in messages):
            return LayerResult(
                messages=[{"role": "user", "content": "[summary]"}],
                boundary=CompactBoundaryEvent(
                    kind="microcompact",
                    message_index=0,
                    summary="[summary]",
                ),
            )
        return LayerResult(messages=messages)

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    store = BoundaryFailingStore()
    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
        transcript_store=store,
    )
    engine = _engine_with_deps(deps)

    events = [msg async for msg in engine.submit_message("first prompt")]

    assert isinstance(events[-1], SDKResult)
    assert "append:compact_boundary:failed" in store.operations
    message_entries = [
        entry for entry in store.entries if isinstance(entry, TranscriptMessageEntry)
    ]
    assert [entry.message["content"] for entry in message_entries] == [
        "first prompt",
        "[summary]",
        "after compact",
    ]
    assert message_entries[1].parent_entry_id is None
    assert message_entries[2].parent_entry_id == message_entries[1].entry_id


@pytest.mark.asyncio
async def test_submit_message_persists_tool_result_replacement_records(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "after replacement"},)
    )
    store = RecordingTranscriptStore()
    ctx = _ctx()
    ctx.content_replacement = ContentReplacementState(
        max_result_size_chars=12,
        replaced_outputs_dir=str(tmp_path),
    )
    engine = _engine_with_ctx(
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=provider,
            transcript_store=store,
        ),
        ctx,
        config=QueryConfig(model="model-1", session_id="s"),
    )
    engine._messages = [  # pyright: ignore[reportPrivateUsage]
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_big",
                    "content": "x" * 200,
                }
            ],
        }
    ]

    events = [msg async for msg in engine.submit_message("continue")]

    replacement_entries = [
        entry for entry in store.entries if isinstance(entry, ContentReplacementEntry)
    ]
    assert isinstance(events[-1], SDKResult)
    assert len(replacement_entries) == 1
    assert replacement_entries[0].replacements[0].tool_use_id == "toolu_big"
    request_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    first_content = request_messages[0]["content"]
    assert first_content == ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER
    persisted_content = engine._messages[0]["content"]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(persisted_content, list)
    assert str(persisted_content[0]["content"]).startswith(PERSISTED_TOOL_RESULT_TAG)


@pytest.mark.asyncio
async def test_query_engine_from_replay_resumes_messages_and_transcript_parent() -> None:
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "two"},))
    store = RecordingTranscriptStore()
    replay = replay_entries(
        [
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                message={"role": "user", "content": "one"},
            ),
            TranscriptMessageEntry(
                entry_id="m2",
                parent_entry_id="m1",
                session_id="s",
                message={"role": "assistant", "content": "answer one"},
            ),
        ],
        scope=TranscriptScope(session_id="s"),
    )
    engine = QueryEngine.from_replay(
        QueryConfig(model="model-1", session_id="s"),
        QueryDeps(
            task_store=AppStateStore(),
            model_provider=provider,
            transcript_store=store,
        ),
        _ctx(),
        replay,
    )

    events = [msg async for msg in engine.submit_message("again")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in provider.requests[0].messages
    ]
    assert request_messages == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "again"},
    ]
    first_entry = store.entries[0]
    assert isinstance(first_entry, TranscriptMessageEntry)
    assert first_entry.message == {"role": "user", "content": "again"}
    assert first_entry.parent_entry_id == "m2"


def test_query_engine_from_replay_reconstructs_content_replacement_state(
    tmp_path: Path,
) -> None:
    record = ToolResultReplacementRecord(
        tool_use_id="toolu_1",
        replacement=f"{PERSISTED_TOOL_RESULT_TAG}\npath: {tmp_path / 'toolu_1.txt'}",
        path=str(tmp_path / "toolu_1.txt"),
        original_size_chars=200,
    )
    replay = replay_entries(
        [
            ContentReplacementEntry(
                entry_id="r1",
                session_id="s",
                replacements=(record,),
            ),
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                message={
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": record.replacement,
                        }
                    ],
                },
            ),
        ],
        scope=TranscriptScope(session_id="s"),
    )
    ctx = _ctx()
    ctx.content_replacement = ContentReplacementState(
        max_result_size_chars=10,
        replaced_outputs_dir=str(tmp_path),
    )

    engine = QueryEngine.from_replay(
        QueryConfig(model="model-1", session_id="s"),
        QueryDeps(task_store=AppStateStore()),
        ctx,
        replay,
    )

    state = engine._ctx.content_replacement  # pyright: ignore[reportPrivateUsage]
    assert state is not None
    assert state.seen_ids == {"toolu_1"}
    assert state.replacements == {"toolu_1": record.replacement}
