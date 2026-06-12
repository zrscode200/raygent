from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.model_registry import count_message_tokens
from raygent_harness.core.model_types import (
    ModelFallbackControl,
    ModelStreamEvent,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    TokenCountResult,
    Usage,
)
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    RecordingKernelEventSink,
)
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore, TaskNotification
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.memdir import (
    MemorySettings,
    create_relevant_memory_recall_provider,
    get_auto_mem_path,
)
from raygent_harness.memdir.relevance import MemorySelector
from raygent_harness.services.compact import create_autocompact_layer
from raygent_harness.services.extract_memories import ExtractionRunResult
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    load_session_replay,
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


async def _collect(engine: QueryEngine) -> list[Any]:
    return [event async for event in engine.submit_message("secret prompt")]


def _identity(
    *,
    block_index: int | None = None,
    message_id: str = "msg_1",
    request_id: str = "req_1",
    attempt_id: str = "attempt_1",
) -> StreamIdentity:
    return StreamIdentity(
        message_id=message_id,
        content_block_index=block_index,
        provider_request_id=request_id,
        attempt_id=attempt_id,
    )


def _text_stream(text: str) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(_identity(), usage=Usage(input_tokens=2)),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=TextContentBlock(text=f"raw block {text}"),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={
                "type": "text_delta",
                "text": text,
                "SECRET_DELTA_KEY": "SECRET_DELTA_VALUE",
            },
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(
            _identity(),
            usage=Usage(output_tokens=3),
            stop_reason="end_turn",
        ),
    )


@pytest.mark.asyncio
async def test_observability_emits_turn_surface_before_model_request() -> None:
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "answer"},)
        ),
        observability=KernelEventBus([sink]),
        context_providers=(_single_context_provider,),
        system_prompt_provider=_system_prompt_provider,
        memory_prompt_provider=_memory_prompt_provider,
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    event_types = sink.event_types
    assert event_types[0] == "query.turn.started"
    assert "query.turn.surface.started" in event_types
    assert "context.providers.completed" in event_types
    assert "system_prompt.provider.completed" in event_types
    assert "memory_prompt.provider.completed" in event_types
    assert "query.turn.surface.completed" in event_types
    assert "query.iteration.started" in event_types
    assert "model.request.started" in event_types
    assert event_types.index("query.turn.surface.completed") < event_types.index(
        "model.request.started"
    )
    assert event_types.index("query.terminal") < event_types.index(
        "query.turn.completed"
    )

    surface = sink.by_type("query.turn.surface.completed")[0]
    assert surface.session_id == "s"
    assert surface.turn_id == "turn-1"
    assert surface.data["context_message_count"] == 1
    assert surface.data["system_prompt_char_count"] == 32

    request = sink.by_type("model.request.started")[0]
    assert request.turn_id == "turn-1"
    assert request.data["message_count"] == 2
    assert request.data["request_mode"] == "complete"
    assert "secret prompt" not in str(request.data)


@pytest.mark.asyncio
async def test_observability_failure_events_do_not_change_terminal_result() -> None:
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(responses=(RuntimeError("provider down"),)),
        observability=KernelEventBus([sink]),
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.is_error is True
    assert "model.request.failed" in sink.event_types
    assert "query.terminal" in sink.event_types
    assert "query.turn.failed" in sink.event_types
    failed = sink.by_type("model.request.failed")[0]
    provider_error = _mapping(failed.data["provider_error"])
    assert provider_error["kind"] == "fatal_unknown"


@pytest.mark.asyncio
async def test_stream_observability_uses_sanitized_projection() -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(stream_events=_text_stream("secret streamed text"))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
    )
    config = QueryConfig(
        model="model-1",
        session_id="s",
        experiments={"streaming_tool_execution": True},
    )
    engine = QueryEngine(config, deps, _ctx())

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert "model.stream.started" in sink.event_types
    assert "model.stream.completed" in sink.event_types
    stream_events = sink.by_type("model.stream.event")
    assert len(stream_events) == 5
    combined_payload = "\n".join(str(event.data) for event in stream_events)
    assert "secret streamed text" not in combined_payload
    assert "SECRET_DELTA_KEY" not in combined_payload
    assert "SECRET_DELTA_VALUE" not in combined_payload
    assert "raw block" not in combined_payload
    delta_event = next(
        event
        for event in stream_events
        if event.data["stream_event_type"] == "content_block_delta"
    )
    delta = _mapping(delta_event.data["delta"])
    assert delta["key_count"] == 3
    assert delta["has_keys"] is True


@pytest.mark.asyncio
async def test_token_count_observability_events_are_optional() -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(
        token_counts=(
            TokenCountResult(
                token_count=17,
                provider_request_id="count_req_1",
                safe_metadata={"cache": "hit"},
            ),
        )
    )

    count = await count_message_tokens(
        provider=provider,
        model="model-1",
        messages=[{"role": "user", "content": "SECRET_PROMPT"}],
        system_prompt="SECRET_SYSTEM",
        fallback_estimator=lambda _messages: 5,
        observability=KernelEventBus([sink]),
        observability_context=KernelEventContext(session_id="s", source="model"),
    )

    assert count == 17
    assert sink.event_types == (
        "model.token_count.started",
        "model.token_count.completed",
    )
    assert sink.events[0].data["message_count"] == 1
    assert sink.events[0].data["system_prompt_char_count"] == len("SECRET_SYSTEM")
    assert sink.events[1].data["token_count"] == 17
    assert sink.events[1].data["fallback_used"] is False
    assert sink.events[1].data["provider_request_id"] == "count_req_1"
    assert sink.events[1].data["provider_metadata"] == {"cache": "hit"}
    assert "SECRET_PROMPT" not in str(sink.events)
    assert "SECRET_SYSTEM" not in str(sink.events)


@pytest.mark.asyncio
async def test_autocompact_token_count_events_emit_on_query_path() -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "after compact"},),
        token_counts=(10,),
    )

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "summary"

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
        autocompact=create_autocompact_layer(
            summarizer=summarizer,
            model_provider=provider,
            threshold_tokens=1,
            token_estimator=lambda _messages: 0,
            env={},
        ),
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert "model.token_count.started" in sink.event_types
    assert "model.token_count.completed" in sink.event_types
    token_started = sink.by_type("model.token_count.started")[0]
    assert token_started.turn_id == "turn-1"
    assert token_started.source == "model"
    assert token_started.data["model"] == "model-1"
    token_completed = sink.by_type("model.token_count.completed")[0]
    assert token_completed.data["token_count"] == 10
    assert token_completed.data["fallback_used"] is False


@pytest.mark.asyncio
async def test_memory_extraction_observability_emits_schedule_and_completion() -> None:
    sink = RecordingKernelEventSink()

    async def extractor(
        messages: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> ExtractionRunResult:
        return ExtractionRunResult(
            status="ran",
            new_message_count=len(messages),
            written_paths=(Path("SECRET-memory.md"),),
            memory_paths=(Path("SECRET-memory.md"),),
        )

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "answer"},)
        ),
        observability=KernelEventBus([sink]),
        memory_extractor=extractor,
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert "memory.extraction.scheduled" in sink.event_types
    assert "memory.extraction.completed" in sink.event_types
    scheduled = sink.by_type("memory.extraction.scheduled")[0]
    completed = sink.by_type("memory.extraction.completed")[0]
    assert scheduled.source == "memory"
    assert scheduled.turn_id == "turn-1"
    assert scheduled.data["message_count"] == 2
    assert completed.data["status"] == "ran"
    assert completed.data["new_message_count"] == 2
    assert completed.data["written_path_count"] == 1
    assert completed.data["memory_path_count"] == 1
    assert "SECRET-memory.md" not in str(completed.data)


@pytest.mark.asyncio
async def test_memory_extraction_observability_sanitizes_custom_status() -> None:
    sink = RecordingKernelEventSink()

    class SecretOutcome:
        status = "SECRET_STATUS /tmp/secret"
        new_message_count = 2

    async def extractor(
        _messages: Sequence[MessageParam],
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> object:
        return SecretOutcome()

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "answer"},)
        ),
        observability=KernelEventBus([sink]),
        memory_extractor=extractor,
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert isinstance(events[-1], SDKResult)
    completed = sink.by_type("memory.extraction.completed")[0]
    assert completed.data["status"] == "custom"
    assert completed.data["status_char_count"] == len("SECRET_STATUS /tmp/secret")
    assert "SECRET_STATUS" not in str(completed.data)
    assert "/tmp/secret" not in str(completed.data)


@pytest.mark.asyncio
async def test_compaction_and_transcript_observability_are_metadata_only(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "after compact"},),
        token_counts=(10,),
    )

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "SECRET_SUMMARY"

    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
        transcript_store=transcript_store,
        autocompact=create_autocompact_layer(
            summarizer=summarizer,
            model_provider=provider,
            threshold_tokens=1,
            token_estimator=lambda _messages: 0,
            env={},
        ),
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    compact = sink.by_type("compact.boundary")[0]
    assert compact.source == "compact"
    assert compact.data["kind"] == "autocompact"
    assert compact.data["summary_char_count"] == len("SECRET_SUMMARY")
    transcript_events = sink.by_type("transcript.appended")
    assert transcript_events
    assert any(event.data["entry_type"] == "compact_boundary" for event in transcript_events)
    combined_payload = "\n".join(
        str(event.data)
        for event in (*sink.by_type("compact.boundary"), *transcript_events)
    )
    assert "SECRET_SUMMARY" not in combined_payload
    assert "secret prompt" not in combined_payload
    assert "after compact" not in combined_payload

    replay_sink = RecordingKernelEventSink()
    replay = await load_session_replay(
        transcript_store,
        TranscriptScope(session_id="s"),
        observability=KernelEventBus([replay_sink]),
        observability_context=KernelEventContext(session_id="s", source="transcript"),
    )
    assert replay.messages
    replay_event = replay_sink.by_type("transcript.replay.completed")[0]
    assert replay_event.data["message_count"] == len(replay.messages)
    assert replay_event.data["compact_boundary_count"] == len(replay.compact_boundaries)


@pytest.mark.asyncio
async def test_wave4_end_to_end_event_order_and_correlation_smoke(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    memory_settings = MemorySettings(
        project_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "memory",
    )
    memory_path = get_auto_mem_path(memory_settings) / "recall.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "\n".join(
            [
                "---",
                "description: SECRET_MEMORY_DESCRIPTION",
                "---",
                "",
                "SECRET_MEMORY_BODY",
            ]
        ),
        encoding="utf-8",
    )

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "SECRET_COMPACT_SUMMARY"

    task_store = AppStateStore()
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Echo",
                        "input": {"value": "SECRET_TOOL_INPUT"},
                    }
                ],
            },
            {"role": "assistant", "content": "final answer"},
        )
    )
    deps = QueryDeps(
        task_store=task_store,
        model_provider=provider,
        observability=KernelEventBus([sink]),
        transcript_store=JsonlTranscriptStore(tmp_path / "transcripts"),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        memory_recall_provider=create_relevant_memory_recall_provider(
            memory_settings,
            selector=FixedMemorySelector(("recall.md",)),
        ),
        autocompact=create_autocompact_layer(
            summarizer=summarizer,
            threshold_tokens=1,
            token_estimator=lambda _messages: 10,
            env={},
        ),
    )
    task_store.enqueue_notification(
        TaskNotification(
            task_id="b_secret",
            message="SECRET_NOTIFICATION",
            kind="completed",
        )
    )
    engine = QueryEngine(
        QueryConfig(
            model="model-1",
            session_id="s",
            tools=(_echo_tool(),),
        ),
        deps,
        _ctx(),
    )

    events = await _collect(engine)

    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"
    event_types = sink.event_types
    for required in (
        "query.turn.started",
        "task.notification.drained",
        "memory.recall.started",
        "memory.recall.completed",
        "compact.boundary",
        "model.request.started",
        "tool.call.completed",
        "transcript.appended",
        "query.terminal",
        "query.turn.completed",
    ):
        assert required in event_types

    first_model_request = event_types.index("model.request.started")
    second_model_request = event_types.index(
        "model.request.started",
        first_model_request + 1,
    )
    assert event_types.index("query.iteration.started") < event_types.index(
        "task.notification.drained"
    )
    assert event_types.index("task.notification.drained") < event_types.index(
        "compact.boundary"
    )
    assert event_types.index("compact.boundary") < first_model_request
    assert event_types.index("tool.call.completed") < event_types.index(
        "memory.recall.completed"
    )
    assert event_types.index("memory.recall.completed") < second_model_request
    assert event_types.index("query.terminal") < event_types.index(
        "query.turn.completed"
    )

    correlated = [
        event
        for event in sink.events
        if event.type
        in {
            "query.turn.started",
            "memory.recall.started",
            "compact.boundary",
            "model.request.started",
            "tool.call.completed",
            "transcript.appended",
            "query.terminal",
        }
        and event.source != "task"
    ]
    assert correlated
    assert all(event.turn_id == "turn-1" for event in correlated)
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    for secret in (
        "SECRET_MEMORY_DESCRIPTION",
        "SECRET_MEMORY_BODY",
        "SECRET_COMPACT_SUMMARY",
        "SECRET_TOOL_INPUT",
        "SECRET_TOOL_OUTPUT",
        "SECRET_NOTIFICATION",
        "secret prompt",
    ):
        assert secret not in combined_payload


@pytest.mark.asyncio
async def test_transcript_append_failure_observability_is_fail_soft() -> None:
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "answer"},)
        ),
        observability=KernelEventBus([sink]),
        transcript_store=FailingTranscriptStore(),
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.subtype == "success"
    failures = sink.by_type("transcript.append_failed")
    assert len(failures) >= 2
    assert failures[0].data["entry_type"] == "message"
    assert failures[0].data["error_type"] == "RuntimeError"
    combined_payload = "\n".join(str(event.data) for event in failures)
    assert "secret prompt" not in combined_payload
    assert "answer" not in combined_payload


@pytest.mark.asyncio
async def test_sidechain_transcript_observability_uses_scope_agent_id(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    scope = TranscriptScope(
        session_id="parent-session",
        runtime_session_id="runtime-child",
        agent_id="agent-sidechain",
        is_sidechain=True,
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            responses=({"role": "assistant", "content": "answer"},)
        ),
        observability=KernelEventBus([sink]),
        transcript_store=JsonlTranscriptStore(tmp_path / "transcripts"),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="parent-session"),
        deps,
        _ctx(),
        transcript_scope=scope,
    )

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    appended = sink.by_type("transcript.appended")
    assert appended
    assert all(event.agent_id == "agent-sidechain" for event in appended)
    assert all(event.runtime_session_id == "runtime-child" for event in appended)
    assert all(event.data["agent_id_present"] is True for event in appended)


@pytest.mark.asyncio
async def test_replay_observability_uses_scope_agent_id_when_context_lacks_it(
    tmp_path: Path,
) -> None:
    sink = RecordingKernelEventSink()
    store = JsonlTranscriptStore(tmp_path / "transcripts")
    scope = TranscriptScope(
        session_id="parent-session",
        runtime_session_id="runtime-child",
        agent_id="agent-sidechain",
        is_sidechain=True,
    )
    await store.append(
        scope,
        TranscriptMessageEntry(
            session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
            agent_id=scope.agent_id,
            is_sidechain=True,
            message={"role": "user", "content": "secret replay prompt"},
        ),
    )

    replay = await load_session_replay(
        store,
        scope,
        observability=KernelEventBus([sink]),
        observability_context=KernelEventContext(
            session_id="parent-session",
            source="transcript",
        ),
    )

    assert replay.agent_id == "agent-sidechain"
    event = sink.by_type("transcript.replay.completed")[0]
    assert event.agent_id == "agent-sidechain"
    assert event.runtime_session_id == "runtime-child"
    assert event.data["agent_id_present"] is True
    assert "secret replay prompt" not in str(event.data)


@pytest.mark.asyncio
async def test_compact_failure_and_recovery_observability_emit_safe_facts() -> None:
    sink = RecordingKernelEventSink()

    async def failing_summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        raise RuntimeError("SECRET_COMPACT_FAILURE")

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(responses=(RuntimeError("SECRET_PROVIDER"),)),
        observability=KernelEventBus([sink]),
        autocompact=create_autocompact_layer(
            summarizer=failing_summarizer,
            threshold_tokens=1,
            token_estimator=lambda _messages: 10,
            env={},
        ),
    )
    engine = QueryEngine(QueryConfig(model="model-1", session_id="s"), deps, _ctx())

    events = await _collect(engine)

    result = events[-1]
    assert isinstance(result, SDKResult)
    assert result.is_error is True
    assert "compact.failed" in sink.event_types
    assert "recovery.exhausted" in sink.event_types
    failed = sink.by_type("compact.failed")[0]
    exhausted = sink.by_type("recovery.exhausted")[0]
    assert failed.data["kind"] == "autocompact"
    assert exhausted.data["terminal_reason"] == "model_error"
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "SECRET_COMPACT_FAILURE" not in combined_payload
    assert "SECRET_PROVIDER" not in combined_payload


@pytest.mark.asyncio
async def test_model_fallback_observability_redacts_provider_reason() -> None:
    secret = "SECRET_FALLBACK_REASON"
    sink = RecordingKernelEventSink()
    provider = SecretFallbackProvider(
        responses=(RuntimeError(secret), {"role": "assistant", "content": "fallback ok"}),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", fallback_model="model-2", session_id="s"),
        deps,
        _ctx(),
    )

    events = await _collect(engine)

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert "model.fallback.triggered" in sink.event_types
    assert "recovery.rung.selected" in sink.event_types
    all_payloads = "\n".join(str(event.data) for event in sink.events)
    assert secret not in all_payloads
    fallback_event = sink.by_type("model.fallback.triggered")[0]
    reason = _mapping(fallback_event.data["reason"])
    assert reason["redacted"] is True
    assert reason["char_count"] == len(secret)
    recovery_event = sink.by_type("recovery.rung.selected")[0]
    assert recovery_event.data["rung"] == "fallback_model"


async def _single_context_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    /,
) -> tuple[ContextFragment, ...]:
    return (
        ContextFragment(
            id="ctx-1",
            content="safe context",
            channel="user_context",
            source="test",
        ),
    )


async def _system_prompt_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    /,
) -> str:
    return "system addition"


async def _memory_prompt_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    /,
) -> str:
    return "memory addition"


class SecretFallbackProvider(FakeModelProvider):
    def classify_error(self, error: BaseException) -> ProviderError:
        return ProviderError(
            kind="model_fallback_triggered",
            message=str(error),
            model_fallback=ModelFallbackControl(
                original_model="model-1",
                fallback_model="model-2",
                reason=str(error),
            ),
            safe_to_fallback=True,
        )


class EchoInput(BaseModel):
    value: str


def _echo_tool() -> Tool:
    async def call(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        await asyncio.sleep(0.01)
        yield ToolResult(content="SECRET_TOOL_OUTPUT")

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


class FixedMemorySelector(MemorySelector):
    def __init__(self, selected: tuple[str, ...]) -> None:
        self._selected = selected

    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools, abort_event
        return list(self._selected)


class FailingTranscriptStore:
    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
        del scope, entry
        raise RuntimeError("SECRET_TRANSCRIPT_FAILURE")

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        del scope, entries
        raise RuntimeError("SECRET_TRANSCRIPT_FAILURE")

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        del scope
        return []

    async def flush(self, scope: TranscriptScope | None = None) -> None:
        del scope
        return None

    def path_for(self, scope: TranscriptScope) -> str | None:
        del scope
        return None


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)
