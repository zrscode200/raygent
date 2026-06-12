"""Pure data shapes for compaction.

Mirrors the reference's per-call result structure (`CompactionResult` in
in later chunks (`auto_compact.py`, `reactive.py`).

Representation divergence from the reference, by design:
- The reference's `CompactionResult.boundaryMarker` is a `SystemMessage` that
  also doubles as the first message in the post-compact API history. raygent
  keeps the boundary as a separate timeline event (`CompactBoundaryEvent`,
  persisted to `State.compact_boundaries`); the API-visible message list
  starts with the summary. See `build_post_compact_messages` for the API history
  assembly contract.

usage slot — `compactionUsage?: ReturnType<typeof getTokenUsage>` becomes
`compaction_usage: UsageTotals | None`. Without this, the autocompact layer's
own model call (chunk 3) would have no place to report tokens spent on the
summary, and the turn-level `UsageTotals` would undercount.

Why `AutoCompactTrackingState` is NOT here: it's a per-turn state field,
so it lives in `core/state.py` next to `State` to avoid a circular import
between `core.query` (which extends `LayerResult` to carry it) and
`services.compact`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.query import CompactBoundaryEvent
    from raygent_harness.core.state import UsageTotals


# ---------------------------------------------------------------------------
# Constants. Names match reference for cross-grep; values match too.
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
"""Tokens reserved for the summary output during a compaction call.

window for the *body* of the compaction request is
`context_window - MAX_OUTPUT_TOKENS_FOR_SUMMARY` (capped by the model's own
max-output limit when smaller)."""

AUTOCOMPACT_BUFFER_TOKENS = 13_000
"""Headroom under the effective context window before autocompact fires.

threshold is `effective_context_window - AUTOCOMPACT_BUFFER_TOKENS` — i.e.,
we trigger compaction with this many tokens still free, leaving room for the
final user turn that pushed us over the line."""

MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
"""Circuit-breaker limit for consecutive autocompact failures.

returns unchanged messages on every subsequent call — without it, sessions
where the context is irrecoverably over the limit would hammer the API with
doomed compaction attempts on every turn (the reference comment cites
~250K wasted API calls/day before the breaker was added)."""


# ---------------------------------------------------------------------------
# CompactionResult — what the summarizer produces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    """Output of one compaction pass.

    minus the `boundaryMarker: SystemMessage` field — raygent keeps the
    boundary as a separate `CompactBoundaryEvent` timeline event (carried
    here by reference) instead of inlining it as the first API message.

    Token counts are optional because deterministic estimation in v1 is
    coarse; downstream tracking only consumes them when the summarizer
    surfaces real values. Slots are kept so callers don't have to break
    their constructors when richer estimation arrives.
    """

    boundary: CompactBoundaryEvent
    """Timeline event the orchestrator yields and appends to
    `State.compact_boundaries`. NOT included in `build_post_compact_messages`
    output — see module docstring."""

    summary_messages: list[MessageParam]
    """The compaction summary, modeled as user-role messages so it slots into
    the message log without changing role-alternation invariants. Order:
    these come first in the post-compact transcript."""

    messages_to_keep: list[MessageParam] = field(default_factory=list["MessageParam"])
    """Tail of the pre-compact transcript preserved verbatim. Empty for
    full-replace compaction (the v1 default); populated for partial
    compactions when those land later."""

    attachments: list[MessageParam] = field(default_factory=list["MessageParam"])
    """Re-injected context (skill listings, memory files, etc.) that has to
    survive across the boundary. Empty until memory/context services provide
    attachments."""

    hook_results: list[MessageParam] = field(default_factory=list["MessageParam"])
    """Output of post-compact hooks the model needs to see (e.g., reloaded
    instructions). Empty in v1 — populated when the hook ecosystem lands."""

    user_display_message: str | None = None
    """Human-readable headline shown alongside timeline UIs. Not surfaced to
    the model."""

    pre_compact_token_count: int | None = None
    post_compact_token_count: int | None = None
    true_post_compact_token_count: int | None = None
    """Token accounting slots. v1 leaves these `None` because the
    deterministic estimator (chunk 3) is coarse; populate from real API
    usage once `_call_model` surfaces it."""

    compaction_usage: UsageTotals | None = None
    """Tokens / cost the summarizer's own model call consumed. Mirrors
    turn-level `UsageTotals` by the autocompact layer in chunk 3 — without
    a slot here, summary-call cost would be invisible to the SDK result."""


# ---------------------------------------------------------------------------
# Post-compact message assembly.
# ---------------------------------------------------------------------------


def build_post_compact_messages(result: CompactionResult) -> list[MessageParam]:
    """Assemble the API-visible message list that follows a compaction.

    Order: `summary_messages, messages_to_keep, attachments, hook_results`.
    as the first element; raygent keeps the boundary as a separate timeline
    event (`State.compact_boundaries` + `CompactBoundaryEvent` yield) so the
    list returned here is what the model sees, nothing more.

    Returns a fresh list; callers may mutate without aliasing back into the
    frozen result.
    """
    return [
        *result.summary_messages,
        *result.messages_to_keep,
        *result.attachments,
        *result.hook_results,
    ]


__all__ = [
    "AUTOCOMPACT_BUFFER_TOKENS",
    "MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES",
    "MAX_OUTPUT_TOKENS_FOR_SUMMARY",
    "CompactionResult",
    "build_post_compact_messages",
]
