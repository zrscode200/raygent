"""Tests for proactive autocompact policy and layer behavior.

"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.model_types import ModelInfo
from raygent_harness.core.query import (
    CompactBoundaryEvent,
    TerminalEvent,
    ToolOrchestrationComplete,
    query,
)
from raygent_harness.core.state import AutoCompactTrackingState, State, UsageTotals
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.compact import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    CompactSummaryResult,
    PostCompactCleanupContext,
    compact_conversation,
    create_autocompact_layer,
    estimate_message_tokens,
    get_auto_compact_threshold,
    get_compact_user_summary_message,
    get_effective_context_window_size,
    register_post_compact_cleanup_hook,
    should_auto_compact,
)
from tests.fakes import FakeModelProvider

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def _msg(content: object) -> MessageParam:
    return cast("MessageParam", {"role": "user", "content": content})


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _ctx_with_notifications(notifications: list[str]) -> ToolUseContext:
    ctx = _ctx()
    ctx.add_notification = notifications.append
    return ctx


def _content_len_estimator(messages: list[MessageParam]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _compact_summary_message(summary: str) -> MessageParam:
    return _msg(
        get_compact_user_summary_message(
            summary,
            suppress_follow_up_questions=True,
        )
    )


def test_effective_context_window_reserves_summary_output_tokens() -> None:
    # Threshold contract: contextWindow - min(modelMaxOutput, MAX_OUTPUT_TOKENS_FOR_SUMMARY).
    assert (
        get_effective_context_window_size(
            "claude-test",
            context_window_size=200_000,
            max_output_tokens_for_model=8_192,
            env={},
        )
        == 191_808
    )
    assert (
        get_effective_context_window_size(
            "claude-test",
            context_window_size=200_000,
            max_output_tokens_for_model=40_000,
            env={},
        )
        == 200_000 - MAX_OUTPUT_TOKENS_FOR_SUMMARY
    )


def test_effective_context_window_uses_model_metadata() -> None:
    assert (
        get_effective_context_window_size(
            "provider-model",
            model_info=ModelInfo(
                model="provider-model",
                context_window=1_000_000,
                max_output_tokens_default=8_000,
            ),
            env={},
        )
        == 992_000
    )


def test_auto_compact_threshold_matches_reference_buffer_and_pct_override() -> None:
    effective = 180_000
    default_threshold = effective - AUTOCOMPACT_BUFFER_TOKENS
    assert (
        get_auto_compact_threshold(
            "claude-test",
            effective_context_window_size=effective,
            env={},
        )
        == default_threshold
    )
    # The reference percentage override is capped by the default threshold.
    assert (
        get_auto_compact_threshold(
            "claude-test",
            effective_context_window_size=effective,
            pct_override=50,
            env={},
        )
        == 90_000
    )
    assert (
        get_auto_compact_threshold(
            "claude-test",
            effective_context_window_size=effective,
            pct_override=99,
            env={},
        )
        == default_threshold
    )


def test_auto_compact_env_overrides_use_raygent_names_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "10_000")
    monkeypatch.setenv("RAYGENT_AUTO_COMPACT_WINDOW", "50_000")
    monkeypatch.setenv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "10")
    monkeypatch.setenv("RAYGENT_AUTOCOMPACT_PCT_OVERRIDE", "50")

    effective = get_effective_context_window_size(
        "model",
        context_window_size=200_000,
        max_output_tokens_for_model=1_000,
    )

    assert effective == 49_000
    assert (
        get_auto_compact_threshold(
            "model",
            effective_context_window_size=effective,
        )
        == 24_500
    )


def test_auto_compact_explicit_env_accepts_reference_compat_names() -> None:
    effective = get_effective_context_window_size(
        "model",
        context_window_size=200_000,
        max_output_tokens_for_model=1_000,
        env={"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "50_000"},
    )

    assert effective == 49_000
    assert (
        get_auto_compact_threshold(
            "model",
            effective_context_window_size=effective,
            env={"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "50"},
        )
        == 24_500
    )


def test_estimator_counts_structured_text_blocks_stably() -> None:
    text_only = estimate_message_tokens([_msg("abcd" * 10)])
    structured = estimate_message_tokens(
        [
            _msg(
                [
                    {"type": "text", "text": "abcd" * 10},
                    {"type": "tool_result", "content": "efgh" * 10},
                ]
            )
        ]
    )
    assert text_only > 0
    assert structured > text_only


def test_should_auto_compact_respects_threshold_and_snip_tokens_freed() -> None:
    messages = [_msg("x" * 100)]
    assert should_auto_compact(
        messages,
        "claude-test",
        token_estimator=_content_len_estimator,
        threshold_tokens=80,
        env={},
    )
    assert not should_auto_compact(
        messages,
        "claude-test",
        token_estimator=_content_len_estimator,
        threshold_tokens=80,
        snip_tokens_freed=30,
        env={},
    )


def test_should_auto_compact_honors_disable_and_recursion_guards() -> None:
    messages = [_msg("x" * 100)]
    assert not should_auto_compact(
        messages,
        "claude-test",
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={"DISABLE_AUTO_COMPACT": "true"},
    )
    assert not should_auto_compact(
        messages,
        "claude-test",
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        query_source="compact",
        env={},
    )
    assert not should_auto_compact(
        messages,
        "claude-test",
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        query_source="session_memory",
        env={},
    )


@pytest.mark.asyncio
async def test_compact_conversation_formats_summary_and_counts_tokens() -> None:
    seen_prompt: list[str] = []

    async def summarizer(
        _messages: list[MessageParam],
        prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        seen_prompt.append(prompt)
        return "<analysis>scratch</analysis><summary>keep this</summary>"

    messages = [_msg("x" * 50), _msg("y" * 50)]
    result = await compact_conversation(
        messages,
        QueryConfig(model="claude-test"),
        _ctx(),
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
    )

    assert seen_prompt == [
        "Summarize the conversation so future turns can continue with full "
        "context. Return <summary>...</summary> and omit irrelevant detail.\n\n"
        "Message count: 2"
    ]
    assert result.boundary.kind == "autocompact"
    assert result.boundary.message_index == 1
    expected_summary = _compact_summary_message(
        "<analysis>scratch</analysis><summary>keep this</summary>"
    )
    assert result.summary_messages == [expected_summary]
    assert result.pre_compact_token_count == 100
    assert result.post_compact_token_count == _content_len_estimator(
        [expected_summary]
    )


@pytest.mark.asyncio
async def test_compact_conversation_preserves_summary_usage() -> None:
    usage = UsageTotals(input_tokens=10, output_tokens=5, cost_usd=0.01)

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> CompactSummaryResult:
        return CompactSummaryResult(
            text="<summary>with usage</summary>",
            usage=usage,
        )

    result = await compact_conversation(
        [_msg("input")],
        QueryConfig(model="claude-test"),
        _ctx(),
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
    )

    assert result.summary_messages == [
        _compact_summary_message("<summary>with usage</summary>")
    ]
    assert result.compaction_usage is usage


@pytest.mark.asyncio
async def test_autocompact_layer_noops_without_summarizer() -> None:
    layer = create_autocompact_layer(
        summarizer=None,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={},
    )
    messages = [_msg("x" * 100)]

    result = await layer(messages, State(messages=messages), QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.boundary is None
    assert result.auto_compact_tracking is None


@pytest.mark.asyncio
async def test_autocompact_layer_uses_provider_token_count_for_threshold() -> None:
    called = False

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        nonlocal called
        called = True
        return "<summary>should not run</summary>"

    provider = FakeModelProvider(
        token_counts=(999,),
        model_infos={
            "provider-model-1": ModelInfo(
                model="provider-model-1",
                context_window=15_000,
                max_output_tokens_default=1_000,
            )
        },
        resolved_models={"model-1": "provider-model-1"},
    )
    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=lambda _messages: 50_000,
        model_provider=provider,
        env={},
    )

    result = await layer(
        [_msg("large")],
        State(),
        QueryConfig(model="model-1"),
        _ctx(),
    )

    assert result.boundary is None
    assert called is False
    assert len(provider.token_requests) == 1
    assert provider.token_requests[0].model == "provider-model-1"


@pytest.mark.asyncio
async def test_autocompact_layer_preserves_window_suffix_through_provider_resolution() -> None:
    provider = FakeModelProvider(
        token_counts=(986_000,),
        resolved_models={"sonnet[1m]": "provider-sonnet[1m]"},
        model_infos={
            "provider-sonnet[1m]": ModelInfo(
                model="provider-sonnet[1m]",
                context_window=1_000_000,
                max_output_tokens_default=8_000,
            )
        },
    )
    layer = create_autocompact_layer(
        summarizer=None,
        model_provider=provider,
        token_estimator=lambda _messages: 0,
        env={},
    )

    result = await layer(
        [_msg("large")],
        State(),
        QueryConfig(model="sonnet[1m]"),
        _ctx(),
    )

    assert result.boundary is None
    assert provider.resolve_requests[0][0] == "sonnet[1m]"
    assert provider.model_info_requests == ["provider-sonnet[1m]"]
    assert provider.token_requests[0].model == "provider-sonnet[1m]"


@pytest.mark.asyncio
async def test_autocompact_layer_noops_when_auto_compact_disabled() -> None:
    called = False

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        nonlocal called
        called = True
        return "<summary>should not run</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        model_provider=FakeModelProvider(token_counts=(10_000,)),
        env={"DISABLE_AUTO_COMPACT": "1"},
    )
    messages = [_msg("x" * 100)]

    result = await layer(messages, State(messages=messages), QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.auto_compact_tracking is None
    assert called is False


@pytest.mark.asyncio
async def test_autocompact_layer_guard_noop_does_not_call_provider() -> None:
    provider = FakeModelProvider(token_counts=(10_000,))
    layer = create_autocompact_layer(
        summarizer=None,
        model_provider=provider,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        query_source="compact",
        env={},
    )

    result = await layer(
        [_msg("x" * 100)],
        State(),
        QueryConfig(model="model-1"),
        _ctx(),
    )

    assert result.boundary is None
    assert provider.resolve_requests == []
    assert provider.model_info_requests == []
    assert provider.token_requests == []


@pytest.mark.asyncio
async def test_autocompact_layer_noops_for_compact_query_source() -> None:
    called = False

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        nonlocal called
        called = True
        return "<summary>should not run</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        query_source="compact",
        env={},
    )
    messages = [_msg("x" * 100)]

    result = await layer(messages, State(messages=messages), QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.auto_compact_tracking is None
    assert called is False


@pytest.mark.asyncio
async def test_autocompact_layer_uses_active_model_for_summary_config() -> None:
    seen_models: list[str] = []

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        seen_models.append(config.model)
        return "<summary>fallback-model summary</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={},
    )
    messages = [_msg("x" * 100)]
    state = State(messages=messages, active_model="claude-fallback")

    result = await layer(
        messages,
        state,
        QueryConfig(model="claude-primary"),
        _ctx(),
    )

    assert seen_models == ["claude-fallback"]
    assert result.messages == [
        _compact_summary_message("<summary>fallback-model summary</summary>")
    ]


@pytest.mark.asyncio
async def test_autocompact_layer_uses_context_model_override_for_summary_config() -> None:
    seen_models: list[str] = []

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        seen_models.append(config.model)
        return "<summary>skill-model summary</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={},
    )
    messages = [_msg("x" * 100)]
    ctx = _ctx()
    ctx.model_override = "claude-skill"

    result = await layer(
        messages,
        State(messages=messages),
        QueryConfig(model="claude-primary"),
        ctx,
    )

    assert seen_models == ["claude-skill"]
    assert result.messages == [
        _compact_summary_message("<summary>skill-model summary</summary>")
    ]


@pytest.mark.asyncio
async def test_autocompact_layer_uses_context_effort_and_window_suffix() -> None:
    provider = FakeModelProvider(
        token_counts=(123,),
        resolved_models={
            "skill-model": "provider-skill",
            "provider-skill[1m]": "provider-skill[1m]",
        },
        model_infos={
            "provider-skill": ModelInfo(
                model="provider-skill",
                context_window=1_000_000,
            )
        },
    )

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        assert config.model == "provider-skill[1m]"
        return "<summary>skill effort summary</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        model_provider=provider,
        env={},
    )
    messages = [_msg("x" * 100)]
    ctx = _ctx()
    ctx.model_override = "skill-model"
    ctx.reasoning_effort_override = "high"

    result = await layer(
        messages,
        State(messages=messages),
        QueryConfig(model="parent-model[1m]"),
        ctx,
    )

    assert result.messages == [
        _compact_summary_message("<summary>skill effort summary</summary>")
    ]
    assert [request[0] for request in provider.resolve_requests] == [
        "skill-model",
        "provider-skill[1m]",
    ]
    assert provider.resolve_requests[0][1].effort == "high"
    assert provider.token_requests[0].model == "provider-skill[1m]"
    assert provider.token_requests[0].effort == "high"


@pytest.mark.asyncio
async def test_autocompact_layer_success_returns_boundary_messages_and_tracking() -> None:
    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "<summary>short</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        turn_id_factory=lambda: "turn-1",
        env={},
    )
    messages = [_msg("x" * 60), _msg("y" * 60)]

    result = await layer(messages, State(messages=messages), QueryConfig(model="m"), _ctx())

    expected_summary = _compact_summary_message("<summary>short</summary>")
    assert result.messages == [expected_summary]
    assert isinstance(result.boundary, CompactBoundaryEvent)
    assert result.boundary.kind == "autocompact"
    assert result.boundary.message_index == 1
    assert result.tokens_freed == max(
        0,
        120 - _content_len_estimator([expected_summary]),
    )
    assert result.auto_compact_tracking == AutoCompactTrackingState(
        compacted=True,
        turn_counter=0,
        turn_id="turn-1",
        consecutive_failures=0,
    )


@pytest.mark.asyncio
async def test_autocompact_layer_runs_post_compact_cleanup_on_success() -> None:
    calls: list[PostCompactCleanupContext] = []

    def hook(ctx: PostCompactCleanupContext) -> None:
        calls.append(ctx)

    unregister = register_post_compact_cleanup_hook(hook)
    try:
        async def summarizer(
            _messages: list[MessageParam],
            _prompt: str,
            _config: QueryConfig,
            _ctx: ToolUseContext,
        ) -> str:
            return "<summary>cleaned</summary>"

        layer = create_autocompact_layer(
            summarizer=summarizer,
            token_estimator=_content_len_estimator,
            threshold_tokens=1,
            query_source="sdk",
            env={},
        )
        result = await layer(
            [_msg("large")],
            State(),
            QueryConfig(model="claude-opus-4-7"),
            _ctx(),
        )
    finally:
        unregister()

    assert result.boundary is not None
    assert len(calls) == 1
    assert calls[0].query_source == "sdk"
    assert calls[0].agent_id is None
    assert calls[0].is_main_thread is True


@pytest.mark.asyncio
async def test_autocompact_cleanup_hook_error_notifies_without_failing() -> None:
    notifications: list[str] = []

    def broken_hook(_ctx: PostCompactCleanupContext) -> None:
        raise RuntimeError("cleanup boom")

    unregister = register_post_compact_cleanup_hook(broken_hook)
    try:
        async def summarizer(
            _messages: list[MessageParam],
            _prompt: str,
            _config: QueryConfig,
            _ctx: ToolUseContext,
        ) -> str:
            return "<summary>cleaned</summary>"

        layer = create_autocompact_layer(
            summarizer=summarizer,
            token_estimator=_content_len_estimator,
            threshold_tokens=1,
            query_source="sdk",
            env={},
        )
        result = await layer(
            [_msg("large")],
            State(),
            QueryConfig(model="claude-opus-4-7"),
            _ctx_with_notifications(notifications),
        )
    finally:
        unregister()

    assert result.boundary is not None
    assert notifications == [
        "post-compact cleanup hook failed: broken_hook: cleanup boom"
    ]


@pytest.mark.asyncio
async def test_autocompact_layer_failure_increments_circuit_breaker() -> None:
    async def failing_summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        raise RuntimeError("summary failed")

    layer = create_autocompact_layer(
        summarizer=failing_summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={},
    )
    messages = [_msg("x" * 100)]
    state = State(
        messages=messages,
        auto_compact_tracking=AutoCompactTrackingState(consecutive_failures=2),
    )

    result = await layer(messages, state, QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.boundary is None
    assert result.auto_compact_tracking == AutoCompactTrackingState(
        consecutive_failures=MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
    )


@pytest.mark.asyncio
async def test_autocompact_layer_circuit_breaker_skips_summarizer() -> None:
    called = False

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        nonlocal called
        called = True
        return "<summary>should not run</summary>"

    layer = create_autocompact_layer(
        summarizer=summarizer,
        token_estimator=_content_len_estimator,
        threshold_tokens=1,
        env={},
    )
    messages = [_msg("x" * 100)]
    state = State(
        messages=messages,
        auto_compact_tracking=AutoCompactTrackingState(
            consecutive_failures=MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
        ),
    )

    result = await layer(messages, state, QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.boundary is None
    assert result.auto_compact_tracking is None
    assert called is False


@pytest.mark.asyncio
async def test_autocompact_layer_noop_does_not_increment_turn_counter() -> None:
    layer = create_autocompact_layer(
        summarizer=None,
        token_estimator=_content_len_estimator,
        threshold_tokens=10_000,
        env={},
    )
    messages = [_msg("below threshold")]
    state = State(
        messages=messages,
        auto_compact_tracking=AutoCompactTrackingState(
            compacted=True,
            turn_counter=2,
            turn_id="prior",
            consecutive_failures=0,
        ),
    )

    result = await layer(messages, state, QueryConfig(model="m"), _ctx())

    assert result.messages == messages
    assert result.auto_compact_tracking is None


@pytest.mark.asyncio
async def test_autocompact_layer_integrates_with_query_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_model_inputs: list[list[MessageParam]] = []

    async def fake_call(
        msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_model_inputs.append(list(msgs))
        return {"text": "ok"}

    def fake_assistant(_response: Any) -> MessageParam:
        return {"role": "assistant", "content": "ok"}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "<summary>autocompacted</summary>"

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    deps = QueryDeps(
        task_store=AppStateStore(),
        autocompact=create_autocompact_layer(
            summarizer=summarizer,
            token_estimator=lambda messages: (
                100 if any(m.get("content") == "x" * 50 for m in messages) else 0
            ),
            threshold_tokens=1,
            turn_id_factory=lambda: "turn-q",
            env={},
        ),
    )

    events: list[Any] = []
    async for ev in query(
        State(messages=[_msg("x" * 50)]),
        QueryConfig(model="claude-test"),
        deps,
        _ctx(),
    ):
        events.append(ev)

    expected_summary = _compact_summary_message("<summary>autocompacted</summary>")
    assert seen_model_inputs == [[expected_summary]]
    boundary_events = [ev for ev in events if isinstance(ev, CompactBoundaryEvent)]
    assert len(boundary_events) == 1
    terminal_events = [ev for ev in events if isinstance(ev, TerminalEvent)]
    assert len(terminal_events) == 1
    final = terminal_events[0].terminal.final_state
    assert final is not None
    assert final.messages == [
        expected_summary,
        cast("MessageParam", {"role": "assistant", "content": "ok"}),
    ]
    assert final.compact_boundaries[0].kind == "autocompact"
    assert final.auto_compact_tracking == AutoCompactTrackingState(
        compacted=True,
        turn_counter=0,
        turn_id="turn-q",
        consecutive_failures=0,
    )


@pytest.mark.asyncio
async def test_query_successful_tool_continue_increments_post_compact_turn_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_model_inputs: list[list[MessageParam]] = []
    call_count = {"n": 0}

    async def fake_call(
        msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_model_inputs.append(list(msgs))
        call_count["n"] += 1
        return {"text": f"reply-{call_count['n']}"}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return ["tool-use"] if call_count["n"] == 1 else []

    async def fake_orchestrate_tools(
        _assistant_message: Any,
        _state: State,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        yield ToolOrchestrationComplete(tool_result_messages=(_msg("tool result"),))

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> str:
        return "<summary>autocompacted</summary>"

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)
    monkeypatch.setattr(query_mod, "_orchestrate_tools", fake_orchestrate_tools)

    deps = QueryDeps(
        task_store=AppStateStore(),
        autocompact=create_autocompact_layer(
            summarizer=summarizer,
            token_estimator=lambda messages: (
                100 if any(m.get("content") == "x" * 50 for m in messages) else 0
            ),
            threshold_tokens=1,
            turn_id_factory=lambda: "turn-q",
            env={},
        ),
    )

    events: list[Any] = []
    async for ev in query(
        State(messages=[_msg("x" * 50)]),
        QueryConfig(model="claude-test"),
        deps,
        _ctx(),
    ):
        events.append(ev)

    assert len(seen_model_inputs) == 2
    terminal_events = [ev for ev in events if isinstance(ev, TerminalEvent)]
    assert len(terminal_events) == 1
    final = terminal_events[0].terminal.final_state
    assert final is not None
    assert final.auto_compact_tracking == AutoCompactTrackingState(
        compacted=True,
        turn_counter=1,
        turn_id="turn-q",
        consecutive_failures=0,
    )
