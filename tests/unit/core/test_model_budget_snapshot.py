from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import pytest

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig, SamplingParams
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_registry import CONTEXT_1M_WINDOW_TOKENS
from raygent_harness.core.model_types import (
    ModelCapabilities,
    ModelInfo,
    ModelStreamEvent,
    StreamIdentity,
    TextContentBlock,
    Usage,
)
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from tests.fakes import FakeModelProvider


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _identity(
    *,
    block_index: int | None = None,
    message_id: str = "msg_1",
    request_id: str = "req_1",
) -> StreamIdentity:
    return StreamIdentity(
        message_id=message_id,
        content_block_index=block_index,
        provider_request_id=request_id,
    )


def _text_stream(text: str) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.message_start(_identity(), usage=Usage(input_tokens=2)),
        ModelStreamEvent.content_block_start(
            _identity(block_index=0),
            block=TextContentBlock(text=""),
        ),
        ModelStreamEvent.content_block_delta(
            _identity(block_index=0),
            delta={"type": "text_delta", "text": text},
        ),
        ModelStreamEvent.content_block_stop(_identity(block_index=0)),
        ModelStreamEvent.message_stop(
            _identity(),
            usage=Usage(output_tokens=3),
            stop_reason="end_turn",
        ),
    )


@pytest.mark.asyncio
async def test_model_budget_snapshot_uses_resolved_window_and_output_limits() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
        token_counts=(321,),
        resolved_models={"alias[1m]": "provider-model[1m]"},
        model_infos={
            "provider-model[1m]": ModelInfo(
                model="provider-model[1m]",
                context_window=CONTEXT_1M_WINDOW_TOKENS,
                max_output_tokens_default=12_000,
                max_output_tokens_upper_limit=24_000,
            )
        },
    )

    await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "hello"}],
        "alias[1m]",
        QueryConfig(model="alias[1m]", system_prompt="system"),
        QueryDeps(model_provider=provider, task_store=AppStateStore()),
        _ctx(),
    )

    request = provider.requests[0]
    assert request.model == "provider-model[1m]"
    assert request.sampling.max_tokens == 12_000
    assert request.budget is not None
    assert request.budget.requested_model == "alias[1m]"
    assert request.budget.effective_model == "provider-model[1m]"
    assert request.budget.context_window == CONTEXT_1M_WINDOW_TOKENS
    assert request.budget.default_max_output_tokens == 12_000
    assert request.budget.upper_max_output_tokens == 24_000
    assert request.budget.requested_max_tokens == 8192
    assert request.budget.effective_max_tokens == 12_000
    assert request.budget.input_token_count == 321
    assert request.budget.provider_input_token_count == 321
    assert request.budget.token_count_fallback_used is False
    assert provider.token_requests[0].system_prompt == "system"
    assert provider.token_requests[0].model == "provider-model[1m]"


@pytest.mark.asyncio
async def test_model_budget_snapshot_records_deterministic_fallback() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
        token_counts=(RuntimeError("count unavailable"),),
    )

    await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "abcdefghijklmnop"}],
        "model-1",
        QueryConfig(model="model-1"),
        QueryDeps(model_provider=provider, task_store=AppStateStore()),
        _ctx(),
    )

    budget = provider.requests[0].budget
    assert budget is not None
    assert budget.token_count_fallback_used is True
    assert budget.token_count_error_type == "RuntimeError"
    assert budget.provider_input_token_count is None
    assert budget.fallback_input_token_count == budget.input_token_count
    assert budget.input_token_count is not None
    assert budget.input_token_count > 0


@pytest.mark.asyncio
async def test_budget_token_count_request_carries_child_effort_and_window_model() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
        token_counts=(123,),
        resolved_models={"skill-model[1m]": "provider-skill[1m]"},
    )
    ctx = _ctx()
    ctx = ToolUseContext(
        session_id=ctx.session_id,
        agent_id="child-agent",
        abort_event=ctx.abort_event,
        rendered_system_prompt=ctx.rendered_system_prompt,
        cwd=ctx.cwd,
        query_tracking=ctx.query_tracking,
        model_override="skill-model[1m]",
        reasoning_effort_override="high",
    )

    await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "child prompt"}],
        "skill-model[1m]",
        QueryConfig(model="parent-model"),
        QueryDeps(model_provider=provider, task_store=AppStateStore()),
        ctx,
    )

    assert provider.resolve_requests[0][0] == "skill-model[1m]"
    assert provider.resolve_requests[0][1].agent_id == "child-agent"
    assert provider.resolve_requests[0][1].effort == "high"
    token_request = provider.token_requests[0]
    assert token_request.model == "provider-skill[1m]"
    assert token_request.effort == "high"
    assert provider.requests[0].agent_id == "child-agent"
    assert provider.requests[0].effort == "high"


@pytest.mark.asyncio
async def test_budget_snapshot_observability_is_metadata_only() -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "SECRET_ANSWER"},),
        token_counts=(77,),
    )

    await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "SECRET_PROMPT"}],
        "model-1",
        QueryConfig(model="model-1", sampling=SamplingParams(max_tokens=123)),
        QueryDeps(
            model_provider=provider,
            task_store=AppStateStore(),
            observability=KernelEventBus([sink]),
        ),
        _ctx(),
    )

    request_event = sink.by_type("model.request.started")[0]
    budget = cast(Mapping[str, object], request_event.data["budget"])
    assert budget["requested_model"] == "model-1"
    assert budget["effective_model"] == "model-1"
    assert budget["input_token_count"] == 77
    assert budget["requested_max_tokens"] == 123
    assert budget["effective_max_tokens"] == 123
    assert "SECRET_PROMPT" not in str(request_event.data)
    assert "SECRET_ANSWER" not in str(request_event.data)


@pytest.mark.asyncio
async def test_streaming_path_exposes_same_budget_snapshot_metadata() -> None:
    sink = RecordingKernelEventSink()
    provider = FakeModelProvider(
        stream_events=_text_stream("streamed"),
        token_counts=(88,),
        model_infos={
            "model-1": ModelInfo(
                model="model-1",
                context_window=333_000,
                max_output_tokens_default=11_000,
                max_output_tokens_upper_limit=22_000,
                capabilities=ModelCapabilities(supports_streaming=True),
            )
        },
    )
    deps = QueryDeps(
        model_provider=provider,
        task_store=AppStateStore(),
        observability=KernelEventBus([sink]),
    )
    engine = QueryEngine(
        QueryConfig(
            model="model-1",
            session_id="s",
            experiments={"streaming_tool_execution": True},
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("SECRET_PROMPT")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert len(provider.stream_requests) == 1
    assert provider.requests == []
    budget = provider.stream_requests[0].budget
    assert budget is not None
    assert budget.input_token_count == 88
    assert budget.context_window == 333_000
    assert budget.effective_max_tokens == 11_000

    request_event = sink.by_type("model.request.started")[0]
    observed_budget = cast(Mapping[str, object], request_event.data["budget"])
    assert observed_budget["input_token_count"] == 88
    assert observed_budget["context_window"] == 333_000
    assert observed_budget["effective_max_tokens"] == 11_000
    assert request_event.data["request_mode"] == "stream"
    assert "SECRET_PROMPT" not in str(request_event.data)
