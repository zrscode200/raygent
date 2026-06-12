"""Tests for reactive context-overflow compaction.

single reactive compaction has a chance to produce post-compact messages.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.state import ErrorWatermark, State
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.compact import (
    PostCompactCleanupContext,
    create_reactive_compact,
    get_compact_user_summary_message,
    register_post_compact_cleanup_hook,
    try_reactive_compact,
)

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def _msg(content: str) -> MessageParam:
    return cast("MessageParam", {"role": "user", "content": content})


def _compact_summary_message(summary: str) -> MessageParam:
    return _msg(
        get_compact_user_summary_message(
            summary,
            suppress_follow_up_questions=True,
        )
    )


def _ctx(*, aborted: bool = False) -> ToolUseContext:
    abort_event = asyncio.Event()
    if aborted:
        abort_event.set()
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=abort_event,
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _ctx_with_notifications(notifications: list[str]) -> ToolUseContext:
    ctx = _ctx()
    ctx.add_notification = notifications.append
    return ctx


@pytest.mark.asyncio
async def test_try_reactive_compact_noops_without_summarizer() -> None:
    result = await try_reactive_compact(
        [_msg("large")],
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
        summarizer=None,
        env={},
    )
    assert result is None


@pytest.mark.asyncio
async def test_try_reactive_compact_success_formats_summary() -> None:
    async def summarizer(
        messages: list[MessageParam],
        prompt: str,
        config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        assert messages == [_msg("large")]
        assert "<summary>" in prompt
        assert config.model == "claude-opus-4-7"
        return "<analysis>scratch</analysis><summary>short</summary>"

    result = await try_reactive_compact(
        [_msg("large")],
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
        summarizer=summarizer,
        token_estimator=lambda _messages: 42,
        env={},
    )
    assert result is not None
    assert result.summary_messages == [
        _compact_summary_message("<summary>short</summary>")
    ]
    assert result.boundary.summary == "Summary:\nshort"
    assert result.pre_compact_token_count == 42
    assert result.post_compact_token_count == 42


@pytest.mark.asyncio
async def test_try_reactive_compact_runs_post_compact_cleanup_on_success() -> None:
    calls: list[PostCompactCleanupContext] = []

    def hook(ctx: PostCompactCleanupContext) -> None:
        calls.append(ctx)

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return "<summary>short</summary>"

    unregister = register_post_compact_cleanup_hook(hook)
    try:
        result = await try_reactive_compact(
            [_msg("large")],
            QueryConfig(model="claude-opus-4-7"),
            _ctx(),
            summarizer=summarizer,
            query_source="sdk",
            env={},
        )
    finally:
        unregister()

    assert result is not None
    assert len(calls) == 1
    assert calls[0].query_source == "sdk"
    assert calls[0].is_main_thread is True


@pytest.mark.asyncio
async def test_try_reactive_compact_cleanup_hook_error_notifies_without_failing() -> None:
    notifications: list[str] = []

    def broken_hook(_ctx: PostCompactCleanupContext) -> None:
        raise RuntimeError("cleanup boom")

    async def summarizer(
        _messages: list[MessageParam],
        _prompt: str,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str:
        return "<summary>short</summary>"

    unregister = register_post_compact_cleanup_hook(broken_hook)
    try:
        result = await try_reactive_compact(
            [_msg("large")],
            QueryConfig(model="claude-opus-4-7"),
            _ctx_with_notifications(notifications),
            summarizer=summarizer,
            query_source="sdk",
            env={},
        )
    finally:
        unregister()

    assert result is not None
    assert notifications == [
        "post-compact cleanup hook failed: broken_hook: cleanup boom"
    ]


@pytest.mark.asyncio
async def test_create_reactive_compact_skips_when_already_attempted() -> None:
    summarizer = MagicMock()
    compactor = create_reactive_compact(summarizer=summarizer, env={})
    result = await compactor(
        [_msg("large")],
        State(error_watermark=ErrorWatermark(tried_reduce_context=True)),
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
    )
    assert result is None
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_create_reactive_compact_skips_when_aborted_or_recursing() -> None:
    summarizer = MagicMock()
    aborted_compactor = create_reactive_compact(summarizer=summarizer, env={})
    assert (
        await aborted_compactor(
            [_msg("large")],
            State(),
            QueryConfig(model="claude-opus-4-7"),
            _ctx(aborted=True),
        )
        is None
    )

    recursive_compactor = create_reactive_compact(
        summarizer=summarizer,
        query_source="compact",
        env={},
    )
    assert (
        await recursive_compactor(
            [_msg("large")],
            State(),
            QueryConfig(model="claude-opus-4-7"),
            _ctx(),
        )
        is None
    )
    summarizer.assert_not_called()


@pytest.mark.asyncio
async def test_try_reactive_compact_respects_disable_env() -> None:
    summarizer = MagicMock()
    result = await try_reactive_compact(
        [_msg("large")],
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
        summarizer=summarizer,
        env={"DISABLE_AUTO_COMPACT": "1"},
    )
    assert result is None
    summarizer.assert_not_called()
