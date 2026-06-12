"""Reactive context-overflow compaction.

`reactiveCompact.tryReactiveCompact` after the API withholds a prompt-too-long
response. On success, the post-compact transcript becomes the next query state;
on failure, the withheld error surfaces and stop hooks do not run.

This module owns the compaction attempt; `core.query._handle_error` owns the
one-shot recovery policy and terminal mapping.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from raygent_harness.services.compact.auto_compact import (
    DEFAULT_DISABLED_QUERY_SOURCES,
    CompactSummarizer,
    TokenEstimator,
    compact_conversation,
    estimate_message_tokens,
)
from raygent_harness.services.compact.cleanup import run_post_compact_cleanup

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ReactiveCompactor
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.state import State
    from raygent_harness.core.tool import ToolUseContext
    from raygent_harness.services.compact.models import CompactionResult


async def try_reactive_compact(
    messages: list[MessageParam],
    config: QueryConfig,
    ctx: ToolUseContext,
    *,
    summarizer: CompactSummarizer | None,
    token_estimator: TokenEstimator = estimate_message_tokens,
    enabled: bool = True,
    already_attempted: bool = False,
    aborted: bool = False,
    query_source: str | None = None,
    disabled_query_sources: tuple[str, ...] = DEFAULT_DISABLED_QUERY_SOURCES,
    env: dict[str, str] | None = None,
) -> CompactionResult | None:
    """Try one reactive compaction and return None if recovery is unavailable.

    Fail-closed cases mirror the reference's guards:
    - no summarizer/feature disabled -> no recovery
    - already attempted -> no death spiral
    - aborted -> do not start a new summary call
    - compact/session_memory query sources -> avoid recursive compaction agents
    - DISABLE_COMPACT / DISABLE_AUTO_COMPACT -> user opted out of automatic
      recovery
    """
    if (
        not enabled
        or summarizer is None
        or already_attempted
        or aborted
        or query_source in disabled_query_sources
        or _env_truthy("DISABLE_COMPACT", env)
        or _env_truthy("DISABLE_AUTO_COMPACT", env)
    ):
        return None

    try:
        result = await compact_conversation(
            messages,
            config,
            ctx,
            summarizer=summarizer,
            token_estimator=token_estimator,
        )
        await run_post_compact_cleanup(
            query_source=query_source,
            agent_id=getattr(ctx, "agent_id", None),
            notify_error=getattr(ctx, "add_notification", None),
        )
        return result
    except Exception:
        return None


def create_reactive_compact(
    *,
    summarizer: CompactSummarizer | None = None,
    token_estimator: TokenEstimator = estimate_message_tokens,
    enabled: bool = True,
    query_source: str | None = None,
    disabled_query_sources: tuple[str, ...] = DEFAULT_DISABLED_QUERY_SOURCES,
    env: dict[str, str] | None = None,
) -> ReactiveCompactor:
    """Build a `QueryDeps.reactive_compact` callable.

    Runtime behavior is `Awaitable[CompactionResult | None]`.
    """

    async def reactive_compact(
        messages: list[MessageParam],
        state: State,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        return await try_reactive_compact(
            messages,
            config,
            ctx,
            summarizer=summarizer,
            token_estimator=token_estimator,
            enabled=enabled,
            already_attempted=state.error_watermark.tried_reduce_context,
            aborted=ctx.abort_event.is_set(),
            query_source=query_source,
            disabled_query_sources=disabled_query_sources,
            env=env,
        )

    return reactive_compact


def _env_truthy(name: str, env: dict[str, str] | None) -> bool:
    values = os.environ if env is None else env
    raw = values.get(name)
    return raw is not None and raw.lower() not in {"", "0", "false", "no", "off"}


__all__ = [
    "create_reactive_compact",
    "try_reactive_compact",
]
