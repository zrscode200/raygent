"""Tests for `services.compact.models` — pure data shapes, no wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from raygent_harness.core.query import CompactBoundaryEvent
from raygent_harness.core.state import UsageTotals
from raygent_harness.services.compact import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    CompactionResult,
    build_post_compact_messages,
)

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def _user_msg(text: str) -> MessageParam:
    return cast("MessageParam", {"role": "user", "content": text})


def _boundary() -> CompactBoundaryEvent:
    return CompactBoundaryEvent(kind="autocompact", message_index=0, summary="s")


def test_constants_match_reference_values() -> None:
    # Drift-detector: if we ever bump these without updating the reference
    # citation in models.py, the test should fail loud.
    assert MAX_OUTPUT_TOKENS_FOR_SUMMARY == 20_000
    assert AUTOCOMPACT_BUFFER_TOKENS == 13_000
    assert MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES == 3


def test_compaction_result_optional_fields_default_empty_or_none() -> None:
    result = CompactionResult(
        boundary=_boundary(),
        summary_messages=[_user_msg("summary")],
    )
    assert result.messages_to_keep == []
    assert result.attachments == []
    assert result.hook_results == []
    assert result.user_display_message is None
    assert result.pre_compact_token_count is None
    assert result.post_compact_token_count is None
    assert result.true_post_compact_token_count is None
    assert result.compaction_usage is None


def test_compaction_usage_slot_is_populated_when_supplied() -> None:
    # the autocompact summary call's tokens would be invisible to the
    # turn-level UsageTotals.
    usage = UsageTotals(input_tokens=100, output_tokens=200, cost_usd=0.01)
    result = CompactionResult(
        boundary=_boundary(),
        summary_messages=[_user_msg("summary")],
        compaction_usage=usage,
    )
    assert result.compaction_usage is usage
    assert usage.input_tokens == 100
    assert usage.output_tokens == 200


def test_compaction_result_default_factories_dont_alias() -> None:
    # Frozen dataclass + default_factory=list should produce a fresh list per
    # instance. Mutation on one must not leak into the next default.
    a = CompactionResult(boundary=_boundary(), summary_messages=[])
    a.messages_to_keep.append(_user_msg("leak?"))
    b = CompactionResult(boundary=_boundary(), summary_messages=[])
    assert b.messages_to_keep == []


def test_build_post_compact_messages_preserves_order() -> None:
    summary = _user_msg("summary")
    keep = _user_msg("keep")
    attach = _user_msg("attach")
    hook = _user_msg("hook")
    result = CompactionResult(
        boundary=_boundary(),
        summary_messages=[summary],
        messages_to_keep=[keep],
        attachments=[attach],
        hook_results=[hook],
    )
    assert build_post_compact_messages(result) == [summary, keep, attach, hook]


def test_build_post_compact_messages_excludes_boundary() -> None:
    # Raygent boundary contract: the boundary
    # marker is timeline-only in raygent, not the first message of the
    # post-compact API history.
    summary = _user_msg("summary")
    result = CompactionResult(boundary=_boundary(), summary_messages=[summary])
    assembled = build_post_compact_messages(result)
    assert assembled == [summary]
    # No element should be the boundary event itself (it's not a MessageParam
    # anyway, but pin the contract).
    assert not any(isinstance(m, CompactBoundaryEvent) for m in assembled)


def test_build_post_compact_messages_returns_fresh_list() -> None:
    # Callers may want to extend the assembled list (e.g., append a synthetic
    # marker); mutating the return value must not alias back into the frozen
    # CompactionResult.
    result = CompactionResult(
        boundary=_boundary(),
        summary_messages=[_user_msg("s")],
        messages_to_keep=[_user_msg("k")],
    )
    assembled = build_post_compact_messages(result)
    assembled.append(_user_msg("extra"))
    assert len(result.summary_messages) == 1
    assert len(result.messages_to_keep) == 1
