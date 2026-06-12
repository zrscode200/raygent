"""Recovery ladder — `_handle_error` rung-by-rung.

Calls `_handle_error` directly with typed fake exceptions whose class
names match the suffix discrimination in `_classify_error`. One
representative path per branch (per group-1 scope decision: minimal,
not exhaustive). Tombstone-on-fallback is not tested — `TombstoneMessage`
is currently a placeholder; revisit when emission is wired.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps, ReactiveCompactor
from raygent_harness.core.media_budget import MEDIA_OVERFLOW_RETRY_REMOVED_PLACEHOLDER
from raygent_harness.core.model_types import (
    ProviderError,
    ProviderErrorKind,
    build_model_api_error_message,
)
from raygent_harness.core.query import (
    CompactBoundaryEvent,
    _handle_error,  # pyright: ignore[reportPrivateUsage]
)
from raygent_harness.core.state import AutoCompactTrackingState, ErrorWatermark, State
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.compact.models import CompactionResult
from tests.fakes import FakeModelProvider

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


# Fake error classes — names matter; `_classify_error` does substring
# match on `type(error).__name__.lower()`.
class FallbackTriggeredError(Exception):
    pass


class ContextOverflowError(Exception):
    pass


class MediaOverflowError(Exception):
    pass


class MaxOutputTokensError(Exception):
    pass


class TransientNetworkError(Exception):
    pass


class GenericError(Exception):
    pass


class ClassifiedErrorProvider(FakeModelProvider):
    def __init__(self, provider_error: ProviderError) -> None:
        super().__init__()
        self.provider_error = provider_error

    def classify_error(self, error: BaseException) -> ProviderError:
        _ = error
        return self.provider_error


def _config(*, fallback: str | None = "claude-haiku-4-5") -> QueryConfig:
    return QueryConfig(model="claude-opus-4-7", fallback_model=fallback)


def _state(*, watermark: ErrorWatermark | None = None) -> State:
    return State(error_watermark=watermark or ErrorWatermark())


def _msg(content: str) -> MessageParam:
    return cast("MessageParam", {"role": "user", "content": content})


def _image_msg(image_id: str = "img_1") -> MessageParam:
    return cast(
        "MessageParam",
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "media_type": "image/png",
                    "id": image_id,
                }
            ],
        },
    )


def _stub_deps_and_ctx(
    *,
    reactive_compact: ReactiveCompactor | None = None,
    provider_error: ProviderError | None = None,
) -> tuple[QueryDeps, ToolUseContext]:
    model_provider = (
        ClassifiedErrorProvider(provider_error)
        if provider_error is not None
        else None
    )
    if reactive_compact is not None and model_provider is not None:
        deps = QueryDeps(
            task_store=AppStateStore(),
            reactive_compact=reactive_compact,
            model_provider=model_provider,
        )
    elif reactive_compact is not None:
        deps = QueryDeps(
            task_store=AppStateStore(),
            reactive_compact=reactive_compact,
        )
    elif model_provider is not None:
        deps = QueryDeps(task_store=AppStateStore(), model_provider=model_provider)
    else:
        deps = QueryDeps(task_store=AppStateStore())
    ctx = ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )
    return deps, ctx


def _compaction_result(summary: str = "reactively compacted") -> CompactionResult:
    return CompactionResult(
        boundary=CompactBoundaryEvent(
            kind="autocompact",
            message_index=1,
            summary=summary,
        ),
        summary_messages=[_msg(summary)],
    )


def _provider_api_error(
    kind: ProviderErrorKind,
    message: str,
    *,
    public_message: str | None = None,
    raw_details: str | None = None,
) -> ProviderError:
    api_error = build_model_api_error_message(
        kind=kind,
        public_message=public_message or message,
        raw_details=raw_details,
    )
    return ProviderError(
        kind=kind,
        message=message,
        raw_details=raw_details,
        api_error=api_error,
    )


@pytest.mark.asyncio
async def test_fallback_swap_advances_watermark_and_swaps_model() -> None:
    deps, ctx = _stub_deps_and_ctx()
    new_state, terminal = await _handle_error(
        FallbackTriggeredError("primary down"),
        _state(),
        _config(fallback="claude-haiku-4-5"),
        deps,
        ctx,
    )
    assert terminal is None
    assert new_state.active_model == "claude-haiku-4-5"
    assert new_state.error_watermark.tried_fallback_model is True


@pytest.mark.asyncio
async def test_fallback_after_already_tried_terminals_as_exhausted() -> None:
    deps, ctx = _stub_deps_and_ctx()
    wm = ErrorWatermark(tried_fallback_model=True)
    new_state, terminal = await _handle_error(
        FallbackTriggeredError("still down"),
        _state(watermark=wm),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "fallback_exhausted"
    assert new_state.error_watermark.tried_fallback_model is True


@pytest.mark.asyncio
async def test_context_overflow_without_recovery_terminals_prompt_too_long() -> None:
    deps, ctx = _stub_deps_and_ctx()
    new_state, terminal = await _handle_error(
        ContextOverflowError("too long"),
        State(messages=[_msg("one"), _msg("two")]),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "prompt_too_long"
    assert new_state.error_watermark.tried_reduce_context is True
    assert new_state.messages[:2] == [_msg("one"), _msg("two")]
    assert new_state.messages[-1].get("isApiErrorMessage") is True
    assert new_state.messages[-1].get("apiError") == "context_overflow"


@pytest.mark.asyncio
async def test_context_overflow_without_recovery_surfaces_api_error_message() -> None:
    deps, ctx = _stub_deps_and_ctx(
        provider_error=_provider_api_error(
            "context_overflow",
            "provider says too long",
            public_message="Prompt is too long",
            raw_details="prompt is too long: 137500 tokens > 135000 maximum",
        )
    )
    new_state, terminal = await _handle_error(
        ContextOverflowError("too long"),
        State(messages=[_msg("one"), _msg("two")]),
        _config(),
        deps,
        ctx,
    )

    assert terminal is not None
    assert terminal.reason == "prompt_too_long"
    api_error = new_state.messages[-1]
    assert api_error["role"] == "assistant"
    assert api_error.get("isApiErrorMessage") is True
    assert api_error.get("apiError") == "context_overflow"
    assert api_error.get("errorDetails") == (
        "prompt is too long: 137500 tokens > 135000 maximum"
    )
    assert terminal.final_state is not None
    assert terminal.final_state.messages[-1] == api_error


@pytest.mark.asyncio
async def test_media_overflow_without_recovery_surfaces_image_error_message() -> None:
    deps, ctx = _stub_deps_and_ctx(
        provider_error=_provider_api_error(
            "media_overflow",
            "image exceeds maximum",
            raw_details="image exceeds 5 MB maximum",
        )
    )
    new_state, terminal = await _handle_error(
        MediaOverflowError("too large"),
        State(messages=[_msg("image")]),
        _config(),
        deps,
        ctx,
    )

    assert terminal is not None
    assert terminal.reason == "image_error"
    api_error = new_state.messages[-1]
    assert api_error["role"] == "assistant"
    assert api_error.get("isApiErrorMessage") is True
    assert api_error.get("apiError") == "media_overflow"
    assert api_error.get("errorDetails") == "image exceeds 5 MB maximum"


@pytest.mark.asyncio
async def test_media_overflow_downscopes_media_and_retries_without_compaction() -> None:
    called = False

    async def reactive_compact(
        _messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        nonlocal called
        called = True
        return _compaction_result("should not run")

    deps, ctx = _stub_deps_and_ctx(
        reactive_compact=reactive_compact,
        provider_error=_provider_api_error(
            "media_overflow",
            "image exceeds maximum",
            raw_details="image exceeds 5 MB maximum",
        ),
    )

    image_message = _image_msg("old-image")
    image_message["raygentMessageKind"] = "memory_recall"
    image_message["raygentMemoryRecall"] = {
        "type": "relevant_memories",
        "memories": [{"path": "memory.md", "content_bytes": 17}],
    }

    new_state, terminal = await _handle_error(
        MediaOverflowError("too large"),
        State(messages=[_msg("before"), image_message]),
        _config(),
        deps,
        ctx,
    )

    assert terminal is None
    assert called is False
    assert new_state.error_watermark.tried_media_downscope is True
    assert new_state.error_watermark.tried_reduce_context is False
    image_replacement = new_state.messages[-1]
    assert image_replacement["role"] == "user"
    assert image_replacement["content"] == (
        f"{MEDIA_OVERFLOW_RETRY_REMOVED_PLACEHOLDER} (image)"
    )
    assert image_replacement.get("raygentMessageKind") == "memory_recall"
    assert image_replacement.get("raygentMemoryRecall") == {
        "type": "relevant_memories",
        "memories": [{"path": "memory.md", "content_bytes": 17}],
    }


@pytest.mark.asyncio
async def test_media_overflow_after_downscope_terminals_image_error() -> None:
    deps, ctx = _stub_deps_and_ctx(
        provider_error=_provider_api_error(
            "media_overflow",
            "image still too large",
            raw_details="image exceeds 5 MB maximum",
        )
    )
    wm = ErrorWatermark(tried_media_downscope=True)

    new_state, terminal = await _handle_error(
        MediaOverflowError("still too large"),
        State(messages=[_msg("already downscoped")], error_watermark=wm),
        _config(),
        deps,
        ctx,
    )

    assert terminal is not None
    assert terminal.reason == "image_error"
    assert new_state.error_watermark.tried_media_downscope is True
    assert new_state.messages[-1].get("apiError") == "media_overflow"


@pytest.mark.asyncio
async def test_context_overflow_reactive_success_retries_from_post_compact_state() -> None:
    async def reactive_compact(
        messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        assert messages == [_msg("one"), _msg("two")]
        return _compaction_result("summary")

    deps, ctx = _stub_deps_and_ctx(reactive_compact=reactive_compact)
    tracking = AutoCompactTrackingState(compacted=True, turn_counter=3, turn_id="t")
    new_state, terminal = await _handle_error(
        ContextOverflowError("too long"),
        State(
            messages=[_msg("one"), _msg("two")],
            auto_compact_tracking=tracking,
        ),
        _config(),
        deps,
        ctx,
    )
    assert terminal is None
    assert new_state.messages == [_msg("summary")]
    assert len(new_state.compact_boundaries) == 1
    assert new_state.compact_boundaries[0].summary == "summary"
    assert new_state.error_watermark.tried_reduce_context is True
    # Reference resets autoCompactTracking on reactive success.
    assert new_state.auto_compact_tracking is None


@pytest.mark.asyncio
async def test_context_overflow_reactive_success_withholds_api_error_message() -> None:
    async def reactive_compact(
        _messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        return _compaction_result("summary")

    deps, ctx = _stub_deps_and_ctx(
        reactive_compact=reactive_compact,
        provider_error=_provider_api_error(
            "context_overflow",
            "provider says too long",
            public_message="Prompt is too long",
        ),
    )
    new_state, terminal = await _handle_error(
        ContextOverflowError("too long"),
        State(messages=[_msg("one"), _msg("two")]),
        _config(),
        deps,
        ctx,
    )

    assert terminal is None
    assert new_state.messages == [_msg("summary")]
    assert all(message.get("isApiErrorMessage") is not True for message in new_state.messages)


@pytest.mark.asyncio
async def test_context_overflow_reactive_uses_active_model_config() -> None:
    async def reactive_compact(
        _messages: list[MessageParam],
        _state: State,
        config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        assert config.model == "claude-haiku-4-5"
        return _compaction_result("summary")

    deps, ctx = _stub_deps_and_ctx(reactive_compact=reactive_compact)
    new_state, terminal = await _handle_error(
        ContextOverflowError("too long"),
        State(
            messages=[_msg("one"), _msg("two")],
            active_model="claude-haiku-4-5",
        ),
        _config(),
        deps,
        ctx,
    )
    assert terminal is None
    assert new_state.messages == [_msg("summary")]


@pytest.mark.asyncio
async def test_context_overflow_after_reduction_terminals_prompt_too_long() -> None:
    called = False

    async def reactive_compact(
        _messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        nonlocal called
        called = True
        return _compaction_result()

    deps, ctx = _stub_deps_and_ctx(reactive_compact=reactive_compact)
    wm = ErrorWatermark(tried_reduce_context=True)
    _, terminal = await _handle_error(
        ContextOverflowError("still too long"),
        _state(watermark=wm),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "prompt_too_long"
    assert called is False


@pytest.mark.asyncio
async def test_max_output_tokens_below_limit_increments_counter() -> None:
    deps, ctx = _stub_deps_and_ctx()
    wm = ErrorWatermark(max_output_tokens_recovery_count=1)
    new_state, terminal = await _handle_error(
        MaxOutputTokensError("hit"),
        _state(watermark=wm),
        _config(),
        deps,
        ctx,
    )
    assert terminal is None
    assert new_state.error_watermark.max_output_tokens_recovery_count == 2


@pytest.mark.asyncio
async def test_max_output_tokens_recovery_appends_api_error_and_resume_instruction() -> None:
    deps, ctx = _stub_deps_and_ctx(
        provider_error=_provider_api_error(
            "max_output_tokens",
            "response exceeded output cap",
        )
    )
    wm = ErrorWatermark(max_output_tokens_recovery_count=1)
    new_state, terminal = await _handle_error(
        MaxOutputTokensError("hit"),
        State(messages=[_msg("one")], error_watermark=wm),
        _config(),
        deps,
        ctx,
    )

    assert terminal is None
    assert new_state.error_watermark.max_output_tokens_recovery_count == 2
    assert len(new_state.messages) == 3
    api_error = new_state.messages[-2]
    recovery = new_state.messages[-1]
    assert api_error["role"] == "assistant"
    assert api_error.get("isApiErrorMessage") is True
    assert api_error.get("apiError") == "max_output_tokens"
    assert recovery["role"] == "user"
    assert "Resume directly" in str(recovery["content"])


@pytest.mark.asyncio
async def test_max_output_tokens_at_limit_completes_with_api_error_surface() -> None:
    deps, ctx = _stub_deps_and_ctx()
    # MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3 per query.py.
    wm = ErrorWatermark(max_output_tokens_recovery_count=3)
    _, terminal = await _handle_error(
        MaxOutputTokensError("hit again"),
        _state(watermark=wm),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "completed"


@pytest.mark.asyncio
async def test_max_output_tokens_at_limit_surfaces_api_error_message() -> None:
    deps, ctx = _stub_deps_and_ctx(
        provider_error=_provider_api_error(
            "max_output_tokens",
            "response exceeded output cap",
        )
    )
    wm = ErrorWatermark(max_output_tokens_recovery_count=3)
    new_state, terminal = await _handle_error(
        MaxOutputTokensError("hit again"),
        State(messages=[_msg("one")], error_watermark=wm),
        _config(),
        deps,
        ctx,
    )

    assert terminal is not None
    assert terminal.reason == "completed"
    api_error = new_state.messages[-1]
    assert api_error.get("isApiErrorMessage") is True
    assert api_error.get("apiError") == "max_output_tokens"
    assert terminal.final_state is not None
    assert terminal.final_state.messages[-1] == api_error


@pytest.mark.asyncio
async def test_transient_first_time_advances_rung() -> None:
    deps, ctx = _stub_deps_and_ctx()
    new_state, terminal = await _handle_error(
        TransientNetworkError("flaky"),
        _state(),
        _config(),
        deps,
        ctx,
    )
    assert terminal is None
    assert new_state.error_watermark.tried_transient_retry is True


@pytest.mark.asyncio
async def test_transient_after_retry_terminals_model_error() -> None:
    deps, ctx = _stub_deps_and_ctx()
    wm = ErrorWatermark(tried_transient_retry=True)
    _, terminal = await _handle_error(
        TransientNetworkError("flaky"),
        _state(watermark=wm),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "model_error"


@pytest.mark.asyncio
async def test_unrecoverable_error_terminals_immediately() -> None:
    deps, ctx = _stub_deps_and_ctx()
    _, terminal = await _handle_error(
        GenericError("boom"),
        _state(),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.reason == "model_error"


@pytest.mark.asyncio
async def test_terminal_carries_last_error_message() -> None:
    deps, ctx = _stub_deps_and_ctx()
    _, terminal = await _handle_error(
        GenericError("specific failure detail"),
        _state(),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.message is not None
    assert "specific failure detail" in terminal.message


@pytest.mark.asyncio
async def test_terminal_final_state_carries_updated_watermark() -> None:
    """Regression for item-11 review Medium #1: `_handle_error` returned
    `(new_state, terminal)` but built terminal from the OLD state, leaving
    `Terminal.final_state.error_watermark.last_error` unset. Now both
    must reflect the same `new_state`."""
    deps, ctx = _stub_deps_and_ctx()
    new_state, terminal = await _handle_error(
        GenericError("watermark-marker"),
        _state(),
        _config(),
        deps,
        ctx,
    )
    assert terminal is not None
    assert terminal.final_state is not None
    assert (
        terminal.final_state.error_watermark.last_error
        == new_state.error_watermark.last_error
    )
    assert terminal.final_state.error_watermark.last_error == "watermark-marker"


# Silence unused-import warnings for AsyncMock + replace which aren't used
# yet; kept available for future tests that need them.
_ = (AsyncMock, replace)
