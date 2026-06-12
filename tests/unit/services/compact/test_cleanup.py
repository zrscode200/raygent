"""Tests for post-compact cleanup hook registry.

successful compaction and scopes main-thread-only resets away from subagents.
"""

from __future__ import annotations

import pytest

from raygent_harness.services.compact import (
    PostCompactCleanupContext,
    is_main_thread_compact,
    register_post_compact_cleanup_hook,
    run_post_compact_cleanup,
)


@pytest.mark.asyncio
async def test_run_post_compact_cleanup_calls_registered_hooks_in_order() -> None:
    calls: list[tuple[str, PostCompactCleanupContext]] = []

    def first(ctx: PostCompactCleanupContext) -> None:
        calls.append(("first", ctx))

    async def second(ctx: PostCompactCleanupContext) -> None:
        calls.append(("second", ctx))

    register_post_compact_cleanup_hook(first)
    register_post_compact_cleanup_hook(second)

    result = await run_post_compact_cleanup(query_source="sdk")

    assert result.called == ("first", "second")
    assert result.skipped == ()
    assert result.errors == ()
    assert [name for name, _ctx in calls] == ["first", "second"]
    assert all(ctx.is_main_thread for _name, ctx in calls)


@pytest.mark.asyncio
async def test_main_thread_only_hooks_are_skipped_for_subagents() -> None:
    calls: list[str] = []

    def any_compact(_ctx: PostCompactCleanupContext) -> None:
        calls.append("any")

    def main_only(_ctx: PostCompactCleanupContext) -> None:
        calls.append("main")

    register_post_compact_cleanup_hook(any_compact)
    register_post_compact_cleanup_hook(main_only, main_thread_only=True)

    result = await run_post_compact_cleanup(
        query_source="agent:child",
        agent_id="local_agent_123",
    )

    assert calls == ["any"]
    assert result.called == ("any_compact",)
    assert result.skipped == ("main_only",)
    assert result.context.is_main_thread is False


@pytest.mark.asyncio
async def test_cleanup_hook_errors_are_captured_and_later_hooks_still_run() -> None:
    calls: list[str] = []
    notifications: list[str] = []

    def broken(_ctx: PostCompactCleanupContext) -> None:
        calls.append("broken")
        raise RuntimeError("boom")

    def after(_ctx: PostCompactCleanupContext) -> None:
        calls.append("after")

    register_post_compact_cleanup_hook(broken)
    register_post_compact_cleanup_hook(after)

    result = await run_post_compact_cleanup(notify_error=notifications.append)

    assert calls == ["broken", "after"]
    assert result.called == ("after",)
    assert result.skipped == ()
    assert result.errors == ("broken: boom",)
    assert notifications == ["post-compact cleanup hook failed: broken: boom"]


def test_is_main_thread_compact_matches_reference_sources() -> None:
    assert is_main_thread_compact(query_source=None, agent_id=None) is True
    assert is_main_thread_compact(query_source="sdk", agent_id=None) is True
    assert (
        is_main_thread_compact(query_source="repl_main_thread:abc", agent_id=None)
        is True
    )
    assert is_main_thread_compact(query_source="agent:abc", agent_id=None) is False
    assert is_main_thread_compact(query_source=None, agent_id="agent_123") is False


@pytest.mark.asyncio
async def test_register_returns_unregister_callback() -> None:
    calls: list[str] = []

    def hook(_ctx: PostCompactCleanupContext) -> None:
        calls.append("called")

    unregister = register_post_compact_cleanup_hook(hook)
    unregister()

    result = await run_post_compact_cleanup()

    assert calls == []
    assert result.called == ()
