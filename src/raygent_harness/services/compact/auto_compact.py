"""Proactive autocompact policy and layer factory.

  threshold calculation.
  and 3-failure circuit-breaker behavior.
  the query pipeline.

This is the v1 Python spine: deterministic token estimation, injected summary
generation, and reference-equivalent tracking semantics. It deliberately does
not call the production model directly; callers install the returned layer on
`QueryDeps.autocompact` with a summarizer implementation.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from raygent_harness.core.model_registry import (
    MODEL_CONTEXT_WINDOW_DEFAULT,
    count_message_tokens,
    get_model_output_limits,
    model_info_with_fallback,
    resolve_model_name,
    resolve_skill_model_override,
)
from raygent_harness.core.model_registry import (
    get_context_window_for_model as get_registry_context_window_for_model,
)
from raygent_harness.core.model_types import ModelResolveContext
from raygent_harness.core.query import CompactBoundaryEvent, LayerResult
from raygent_harness.core.state import AutoCompactTrackingState, UsageTotals
from raygent_harness.services.compact.cleanup import run_post_compact_cleanup
from raygent_harness.services.compact.models import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    CompactionResult,
    build_post_compact_messages,
)
from raygent_harness.services.compact.prompt import (
    format_compact_summary,
    get_compact_user_summary_message,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.model_provider import ModelProvider
    from raygent_harness.core.model_types import FrozenJson, ModelInfo, ModelToolSpec
    from raygent_harness.core.query import Layer
    from raygent_harness.core.tool import ToolUseContext


DEFAULT_CONTEXT_WINDOW_TOKENS = MODEL_CONTEXT_WINDOW_DEFAULT
"""Fallback context window for unknown models."""

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Back-compat shim for older tests/callers. Chunk 4 routes new metadata
    # through `core.model_registry`; this prefix map remains only as a final
    # deterministic fallback for callers that still use this module directly.
}

TokenEstimator = Callable[[list["MessageParam"]], int]
TurnIdFactory = Callable[[], str]
DEFAULT_DISABLED_QUERY_SOURCES = ("compact", "session_memory")
AUTO_COMPACT_WINDOW_ENV = "RAYGENT_AUTO_COMPACT_WINDOW"
LEGACY_AUTO_COMPACT_WINDOW_ENV = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"
AUTOCOMPACT_PCT_OVERRIDE_ENV = "RAYGENT_AUTOCOMPACT_PCT_OVERRIDE"
LEGACY_AUTOCOMPACT_PCT_OVERRIDE_ENV = "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"


@dataclass(frozen=True)
class CompactSummaryResult:
    """Raw summary text plus optional usage from the summarizer model call."""

    text: str
    usage: UsageTotals | None = None


class CompactSummarizer(Protocol):
    """Injected summary producer used by `compact_conversation`.

    The prompt string is passed explicitly so tests can assert what would be
    sent to a production model, while the service stays model-client agnostic.
    """

    async def __call__(
        self,
        messages: list[MessageParam],
        prompt: str,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str | CompactSummaryResult:
        """Return raw model text, usually containing `<summary>...</summary>`."""
        ...


def get_context_window_for_model(model: str) -> int:
    """Return the model context window used by autocompact policy.

    Provider-aware callers should pass model metadata through
    `get_effective_context_window_size(...)` / `create_autocompact_layer(...)`.
    This direct helper remains deterministic for pure policy tests and
    provider-free embedders.
    """
    suffix_window = get_registry_context_window_for_model(model)
    if suffix_window != DEFAULT_CONTEXT_WINDOW_TOKENS:
        return suffix_window
    normalized = model.lower()
    for prefix, window in MODEL_CONTEXT_WINDOWS.items():
        if normalized.startswith(prefix):
            return window
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def get_effective_context_window_size(
    model: str,
    *,
    context_window_size: int | None = None,
    max_output_tokens_for_model: int | None = None,
    model_info: ModelInfo | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Return context window minus summary-output reservation.

    Mirrors reference `getEffectiveContextWindowSize`:
    `contextWindow - min(modelMaxOutput, MAX_OUTPUT_TOKENS_FOR_SUMMARY)`.
    Also supports Raygent's `RAYGENT_AUTO_COMPACT_WINDOW` env cap. The
    reference env name remains accepted only when callers pass an explicit
    `env` mapping, so core does not read product globals by default.
    """
    window = (
        context_window_size
        if context_window_size is not None
        else get_registry_context_window_for_model(model, model_info=model_info)
    )
    override = _env_value(
        AUTO_COMPACT_WINDOW_ENV,
        legacy_name=LEGACY_AUTO_COMPACT_WINDOW_ENV,
        env=env,
    )
    if override:
        with _suppress_value_error():
            parsed = int(override)
            if parsed > 0:
                window = min(window, parsed)

    max_output = (
        max_output_tokens_for_model
        if max_output_tokens_for_model is not None
        else get_model_output_limits(model, model_info=model_info).default
    )
    reserved = min(max_output, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    return max(0, window - reserved)


def get_auto_compact_threshold(
    model: str,
    *,
    effective_context_window_size: int | None = None,
    context_window_size: int | None = None,
    max_output_tokens_for_model: int | None = None,
    model_info: ModelInfo | None = None,
    pct_override: float | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Return the token threshold where proactive autocompact should fire.

    Mirrors reference `getAutoCompactThreshold`: effective window minus the
    13K buffer, with optional `RAYGENT_AUTOCOMPACT_PCT_OVERRIDE` capped so it
    never raises the threshold above the default.
    """
    effective = (
        effective_context_window_size
        if effective_context_window_size is not None
        else get_effective_context_window_size(
            model,
            context_window_size=context_window_size,
            max_output_tokens_for_model=max_output_tokens_for_model,
            model_info=model_info,
            env=env,
        )
    )
    threshold = max(0, effective - AUTOCOMPACT_BUFFER_TOKENS)

    override = pct_override
    if override is None:
        raw = _env_value(
            AUTOCOMPACT_PCT_OVERRIDE_ENV,
            legacy_name=LEGACY_AUTOCOMPACT_PCT_OVERRIDE_ENV,
            env=env,
        )
        if raw:
            with _suppress_value_error():
                override = float(raw)

    if override is not None and 0 < override <= 100:
        percentage_threshold = int(effective * (override / 100))
        return min(percentage_threshold, threshold)

    return threshold


def estimate_message_tokens(messages: list[MessageParam]) -> int:
    """Deterministic, dependency-free token estimate for threshold policy.

    Partial fidelity: reference uses `tokenCountWithEstimation`, which mixes
    cached API usage with estimators. Until raygent has model usage metadata,
    this counts visible text at roughly 4 chars/token plus stable overhead for
    message/block structure.
    """
    total = 0
    for message in messages:
        total += 4  # role/id/JSON overhead
        total += _estimate_any_tokens(message.get("content"))
    return total


def should_auto_compact(
    messages: list[MessageParam],
    model: str,
    *,
    enabled: bool = True,
    token_estimator: TokenEstimator = estimate_message_tokens,
    threshold_tokens: int | None = None,
    snip_tokens_freed: int = 0,
    query_source: str | None = None,
    disabled_query_sources: tuple[str, ...] = DEFAULT_DISABLED_QUERY_SOURCES,
    context_window_size: int | None = None,
    max_output_tokens_for_model: int | None = None,
    model_info: ModelInfo | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Return True when current estimated usage reaches autocompact threshold."""
    if (
        not enabled
        or _env_truthy("DISABLE_COMPACT", env)
        or _env_truthy("DISABLE_AUTO_COMPACT", env)
        or query_source in disabled_query_sources
    ):
        return False

    token_count = max(0, token_estimator(messages) - snip_tokens_freed)
    threshold = (
        threshold_tokens
        if threshold_tokens is not None
        else get_auto_compact_threshold(
            model,
            context_window_size=context_window_size,
            max_output_tokens_for_model=max_output_tokens_for_model,
            model_info=model_info,
            env=env,
        )
    )
    return token_count >= threshold


async def compact_conversation(
    messages: list[MessageParam],
    config: QueryConfig,
    ctx: ToolUseContext,
    *,
    summarizer: CompactSummarizer,
    token_estimator: TokenEstimator = estimate_message_tokens,
    suppress_follow_up_questions: bool = True,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
) -> CompactionResult:
    """Run one summarizing compaction and return a `CompactionResult`.

    The summarizer returns raw model text; `format_compact_summary` strips the
    `<analysis>` block and normalizes `<summary>`. The model-visible summary
    is then wrapped with the reference continuation instruction. v1 compacts
    the full input into one user-role summary message and preserves no suffix.
    """
    summary_output = await summarizer(
        list(messages),
        _build_compact_prompt(messages),
        config,
        ctx,
    )
    if isinstance(summary_output, CompactSummaryResult):
        raw_summary = summary_output.text
        usage = summary_output.usage
    else:
        raw_summary = summary_output
        usage = None

    formatted_summary = format_compact_summary(raw_summary)
    if not formatted_summary:
        raise ValueError("compact summarizer produced an empty summary")

    summary_content = get_compact_user_summary_message(
        raw_summary,
        suppress_follow_up_questions=suppress_follow_up_questions,
        transcript_path=transcript_path,
        recent_messages_preserved=recent_messages_preserved,
    )
    summary_message = cast("MessageParam", {"role": "user", "content": summary_content})
    boundary = CompactBoundaryEvent(
        kind="autocompact",
        message_index=max(0, len(messages) - 1),
        summary=formatted_summary,
    )
    post_messages = [summary_message]
    return CompactionResult(
        boundary=boundary,
        summary_messages=post_messages,
        pre_compact_token_count=token_estimator(messages),
        post_compact_token_count=token_estimator(post_messages),
        true_post_compact_token_count=token_estimator(post_messages),
        compaction_usage=usage,
    )


def create_autocompact_layer(
    *,
    summarizer: CompactSummarizer | None = None,
    token_estimator: TokenEstimator = estimate_message_tokens,
    threshold_tokens: int | None = None,
    context_window_size: int | None = None,
    max_output_tokens_for_model: int | None = None,
    model_provider: ModelProvider | None = None,
    tools: tuple[ModelToolSpec, ...] = (),
    thinking: FrozenJson | None = None,
    effort: str | int | None = None,
    media_context: FrozenJson | None = None,
    provider_options: FrozenJson | None = None,
    snip_tokens_freed: int = 0,
    query_source: str | None = None,
    disabled_query_sources: tuple[str, ...] = DEFAULT_DISABLED_QUERY_SOURCES,
    enabled: bool = True,
    turn_id_factory: TurnIdFactory | None = None,
    env: dict[str, str] | None = None,
) -> Layer:
    """Build a `QueryDeps.autocompact` layer.

    Default fail-closed behavior: if disabled, below threshold, circuit-breaker
    tripped, or no summarizer is supplied, the layer returns messages unchanged.
    Failures increment `consecutive_failures`; successes reset it and return a
    boundary + post-compact messages.
    """
    make_turn_id = turn_id_factory or (lambda: uuid.uuid4().hex)

    async def autocompact_layer(
        messages: list[MessageParam],
        state: Any,
        config: QueryConfig,
        ctx: ToolUseContext,
    ) -> LayerResult:
        tracking = _tracking_from_state(state)
        if (
            not enabled
            or _env_truthy("DISABLE_COMPACT", env)
            or _env_truthy("DISABLE_AUTO_COMPACT", env)
            or query_source in disabled_query_sources
        ):
            return LayerResult(messages=messages)

        if tracking.consecutive_failures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES:
            return LayerResult(messages=messages)

        effective_effort = ctx.reasoning_effort_override or effort
        resolve_context = ModelResolveContext(
            agent_id=getattr(ctx, "agent_id", None),
            effort=effective_effort,
        )
        active_model = getattr(state, "active_model", None)
        if active_model is not None:
            effective_model = resolve_model_name(
                active_model,
                provider=model_provider,
                context=resolve_context,
            )
        elif ctx.model_override is not None:
            effective_model = resolve_skill_model_override(
                ctx.model_override,
                config.model,
                provider=model_provider,
                context=resolve_context,
            )
        else:
            effective_model = resolve_model_name(
                config.model,
                provider=model_provider,
                context=resolve_context,
            )
        effective_config = (
            replace(config, model=effective_model)
            if effective_model != config.model
            else config
        )
        resolved_model = effective_model
        metadata = model_info_with_fallback(
            resolved_model,
            provider=model_provider,
        )

        token_count = await _token_count_for_threshold(
            messages,
            resolved_model,
            model_provider=model_provider,
            token_estimator=token_estimator,
            tools=tools,
            thinking=thinking,
            effort=effective_effort,
            media_context=media_context,
            provider_options=provider_options,
            ctx=ctx,
        )
        if not _is_above_auto_compact_threshold(
            token_count=token_count,
            model=resolved_model,
            threshold_tokens=threshold_tokens,
            snip_tokens_freed=snip_tokens_freed,
            context_window_size=context_window_size,
            max_output_tokens_for_model=max_output_tokens_for_model,
            model_info=metadata,
            env=env,
        ):
            return LayerResult(messages=messages)

        if summarizer is None:
            return LayerResult(messages=messages)

        try:
            result = await compact_conversation(
                messages,
                effective_config,
                ctx,
                summarizer=summarizer,
                token_estimator=token_estimator,
            )
        except Exception:
            next_failures = tracking.consecutive_failures + 1
            failed_tracking = replace(
                tracking,
                consecutive_failures=next_failures,
            )
            return LayerResult(
                messages=messages,
                auto_compact_tracking=failed_tracking,
            )

        await run_post_compact_cleanup(
            query_source=query_source,
            agent_id=getattr(ctx, "agent_id", None),
            notify_error=getattr(ctx, "add_notification", None),
        )
        post_compact_messages = build_post_compact_messages(result)
        next_tracking = AutoCompactTrackingState(
            compacted=True,
            turn_counter=0,
            turn_id=make_turn_id(),
            consecutive_failures=0,
        )
        return LayerResult(
            messages=post_compact_messages,
            boundary=result.boundary,
            tokens_freed=max(
                0,
                (result.pre_compact_token_count or token_estimator(messages))
                - (
                    result.true_post_compact_token_count
                    or result.post_compact_token_count
                    or token_estimator(post_compact_messages)
                ),
            ),
            auto_compact_tracking=next_tracking,
        )

    return autocompact_layer


async def _token_count_for_threshold(
    messages: list[MessageParam],
    model: str,
    *,
    model_provider: ModelProvider | None,
    token_estimator: TokenEstimator,
    tools: tuple[ModelToolSpec, ...],
    thinking: FrozenJson | None,
    effort: str | int | None,
    media_context: FrozenJson | None,
    provider_options: FrozenJson | None,
    ctx: ToolUseContext,
) -> int:
    if model_provider is None:
        return max(0, token_estimator(messages))
    observability = ctx.runtime.deps.observability if ctx.runtime is not None else None
    return await count_message_tokens(
        provider=model_provider,
        model=model,
        messages=messages,
        tools=tools,
        thinking=thinking,
        effort=effort,
        media_context=media_context,
        provider_options=provider_options,
        fallback_estimator=token_estimator,
        observability=observability,
        observability_context=(
            ctx.observability_context.with_source("model")
            if ctx.observability_context is not None
            else None
        ),
    )


def _is_above_auto_compact_threshold(
    *,
    token_count: int,
    model: str,
    threshold_tokens: int | None,
    snip_tokens_freed: int,
    context_window_size: int | None,
    max_output_tokens_for_model: int | None,
    model_info: ModelInfo | None,
    env: dict[str, str] | None,
) -> bool:
    threshold = (
        threshold_tokens
        if threshold_tokens is not None
        else get_auto_compact_threshold(
            model,
            context_window_size=context_window_size,
            max_output_tokens_for_model=max_output_tokens_for_model,
            model_info=model_info,
            env=env,
        )
    )
    return max(0, token_count - snip_tokens_freed) >= threshold


class _suppress_value_error:
    """Tiny context manager to avoid importing `contextlib` for one exception."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> bool:
        return exc_type is ValueError


def _build_compact_prompt(messages: list[MessageParam]) -> str:
    """Render a minimal deterministic summary prompt.

    injectable seam and stable instruction wrapper; prompt exactness can evolve
    without changing the layer contract.
    """
    return (
        "Summarize the conversation so future turns can continue with full "
        "context. Return <summary>...</summary> and omit irrelevant detail.\n\n"
        f"Message count: {len(messages)}"
    )


def _tracking_from_state(state: Any) -> AutoCompactTrackingState:
    tracking = getattr(state, "auto_compact_tracking", None)
    if isinstance(tracking, AutoCompactTrackingState):
        return tracking
    return AutoCompactTrackingState()


def _estimate_any_tokens(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return _estimate_text_tokens(value)
    if isinstance(value, list):
        total = 0
        for item in cast("list[object]", value):
            total += 2
            total += _estimate_any_tokens(item)
        return total
    if isinstance(value, dict):
        fields = cast("dict[object, object]", value)
        text_fields: list[object] = []
        for key, field_value in fields.items():
            if (
                isinstance(key, str)
                and key in {"text", "content", "input", "name", "type"}
            ) or isinstance(field_value, str):
                text_fields.append(field_value)
        return 2 + sum(_estimate_any_tokens(field_value) for field_value in text_fields)
    return _estimate_text_tokens(json.dumps(value, sort_keys=True, default=str))


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _env_truthy(name: str, env: dict[str, str] | None) -> bool:
    values = os.environ if env is None else env
    raw = values.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(
    name: str,
    *,
    legacy_name: str,
    env: dict[str, str] | None,
) -> str | None:
    """Read Raygent env names by default, accepting reference names by injection."""

    values = os.environ if env is None else env
    if env is None:
        return values.get(name)
    return values.get(name) or values.get(legacy_name)


__all__ = [
    "DEFAULT_CONTEXT_WINDOW_TOKENS",
    "MODEL_CONTEXT_WINDOWS",
    "CompactSummarizer",
    "TokenEstimator",
    "compact_conversation",
    "create_autocompact_layer",
    "estimate_message_tokens",
    "get_auto_compact_threshold",
    "get_context_window_for_model",
    "get_effective_context_window_size",
    "should_auto_compact",
]
