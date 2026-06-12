from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig, SamplingParams
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_adapter import ToolUseBlock, normalize_assistant_turn
from raygent_harness.core.model_types import ModelInfo, ModelToolSpec
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from tests.fakes import FakeModelProvider

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


class SearchInput(BaseModel):
    query: str


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def test_normalize_assistant_turn_extracts_text_and_tool_use_blocks() -> None:
    response = {
        "content": [
            {"type": "text", "text": "I will search."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Search",
                "input": {"query": "abc"},
            },
        ]
    }

    turn = normalize_assistant_turn(response)

    assert turn.message == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I will search."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Search",
                "input": {"query": "abc"},
            },
        ],
    }
    assert turn.tool_uses == (
        ToolUseBlock(id="toolu_1", name="Search", input={"query": "abc"}, index=1),
    )


def test_normalize_assistant_turn_accepts_legacy_text_fake() -> None:
    turn = normalize_assistant_turn({"text": "plain answer"})

    assert turn.message == {"role": "assistant", "content": "plain answer"}
    assert turn.tool_uses == ()


@pytest.mark.asyncio
async def test_call_model_builds_provider_neutral_model_request() -> None:
    visible_tool = build_tool(
        ToolSpec(
            name="Search",
            description="fallback description",
            input_model=SearchInput,
            call=_call,
            prompt="search prompt",
            is_enabled=True,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )
    deferred_tool = build_tool(
        ToolSpec(
            name="Deferred",
            description="hidden",
            input_model=SearchInput,
            call=_call,
            should_defer=True,
        )
    )
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": [{"type": "text", "text": "ok"}]},),
        resolved_models={"model-1": "provider-model-1"},
    )
    deps = QueryDeps(
        model_provider=provider,
        task_store=AppStateStore(),
    )

    ctx = _ctx()
    response = await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "hi"}],
        "model-1",
        QueryConfig(
            model="model-1",
            system_prompt="system",
            sampling=SamplingParams(
                max_tokens=123,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                stop_sequences=("STOP",),
            ),
            tools=(visible_tool, deferred_tool),
        ),
        deps,
        ctx,
    )

    assert query_mod._extract_assistant_message(response) == {  # pyright: ignore[reportPrivateUsage]
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
    }
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "provider-model-1"
    assert request.sampling.max_tokens == 123
    assert request.system_prompt == "system"
    assert request.sampling.temperature == 0.2
    assert request.sampling.top_p == 0.9
    assert request.sampling.top_k == 40
    assert request.sampling.stop_sequences == ("STOP",)
    assert request.abort_event is ctx.abort_event
    assert request.messages[0].provider_payload is not None
    assert request.tools == (
        ModelToolSpec(
            name="Search",
            description="search prompt",
            input_schema=SearchInput.model_json_schema(),
        ),
    )


@pytest.mark.asyncio
async def test_call_model_uses_provider_max_output_metadata_for_default_sampling() -> None:
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},),
        resolved_models={"model-1": "provider-model-1"},
        model_infos={
            "provider-model-1": ModelInfo(
                model="provider-model-1",
                max_output_tokens_default=32_000,
                max_output_tokens_upper_limit=64_000,
            )
        },
    )

    await query_mod._call_model(  # pyright: ignore[reportPrivateUsage]
        [{"role": "user", "content": "hi"}],
        "model-1",
        QueryConfig(model="model-1"),
        QueryDeps(model_provider=provider, task_store=AppStateStore()),
        _ctx(),
    )

    assert provider.requests[0].model == "provider-model-1"
    assert provider.requests[0].sampling.max_tokens == 32_000


@pytest.mark.asyncio
async def test_query_refreshes_context_messages_after_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from raygent_harness.core.query import LayerResult, TerminalEvent, query
    from raygent_harness.core.state import State

    seen_ctx_messages: list[list[MessageParam]] = []

    async def fake_microcompact(
        _messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        return LayerResult(
            messages=[{"role": "user", "content": "[summary]"}],
        )

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        ctx: ToolUseContext,
    ) -> dict[str, object]:
        seen_ctx_messages.append(list(ctx.messages))
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(query_mod, "_call_model", fake_call)

    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
    )
    events = [
        event
        async for event in query(
            State(messages=[{"role": "user", "content": "long"}]),
            QueryConfig(model="model-1"),
            deps,
            _ctx(),
        )
    ]

    assert seen_ctx_messages == [[{"role": "user", "content": "[summary]"}]]
    terminal = next(event for event in events if isinstance(event, TerminalEvent))
    assert terminal.terminal.reason == "completed"
