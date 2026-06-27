"""State — the per-iteration, wholesale-replaced state of a turn.

Each iteration of the agent loop produces a NEW State; we never
mutate the prior one. Old states can be observed after the fact (telemetry,
debugging) without worrying they changed under us.

Why wholesale replacement (not mutation):
- Recovery ladder. When a step fails we need to roll back to a prior state.
  Mutation means rollback = deep-copy-before-every-attempt. Replacement means
  rollback = keep a reference to the last-good state.
- Compaction. Some compaction layers replay prior tool results against a
  reduced message list. If State were mutated in place, the replayed messages
  could diverge from what the model saw. Replacement forces explicit rewrites.
- Observability. Snapshot-per-iteration is what lets us emit timeline events
  the user can scrub through.

State is a dataclass, NOT frozen — dataclasses.replace() is the idiomatic way
to produce "new state from old" without the frozen-dataclass ergonomics hit
(frozen blocks __setattr__ which some Pydantic-style mutation patterns rely on).
The contract is by convention: produce via replace(), don't mutate fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


# ---------------------------------------------------------------------------
# Usage accumulator — totals across the turn.
# ---------------------------------------------------------------------------


@dataclass
class UsageTotals:
    """Token + cost totals for the turn. Monotonically increasing.

    Kept separate from State so we can emit usage deltas between iterations
    without diffing whole State objects.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    """Running sum of per-response cost. None if pricing not available."""


# ---------------------------------------------------------------------------
# Permission-denial tombstone — recorded when the user denies a tool call.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermissionDenial:
    """Recorded when a tool call was denied. Replayed to the model as context.

    Frozen because we only ever append these; a denial doesn't get revised.
    """

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    reason: str


# ---------------------------------------------------------------------------
# Compact boundary — marks where a compaction occurred in the message history.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactBoundary:
    """Marks the pre-compact message index where compaction happened.

    Raygent keeps boundary records out-of-band while `State.messages` is
    rewritten to the post-compact transcript. The index is therefore
    historical metadata from the pre-compact input, not a live index into the
    current `State.messages`. Frozen because a boundary is a historical fact,
    not editable state.
    """

    message_index: int
    """Index into the pre-compact input. Everything at or before this index
    was consumed and summarized; post-compact content now lives in
    `State.messages`."""

    kind: Literal["microcompact", "autocompact", "context_collapse", "snip"]
    """Which compaction layer produced this boundary."""

    summary: str
    """Human-readable summary of what got compacted. Shown in timeline UI."""


# ---------------------------------------------------------------------------
# Autocompact tracking — bookkeeping the autocompact layer reads/writes
# between iterations within a turn.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoCompactTrackingState:
    """State the autocompact layer threads across iterations.

    Lives in `core/state.py` (not `services/compact/`) to avoid a circular
    import: `core/query.py` extends `LayerResult` to carry this; if the
    type were in `services.compact`, the import direction would loop.

    Field semantics:
    - `compacted` flips True the first time a compaction succeeds. The
      layer reads this on the next iteration to populate the
      `RecompactionInfo.isRecompactionInChain` flag (reference
    - `turn_counter` / `turn_id` track which iteration produced the last
      compaction — diagnostic only; the layer compares against the
      current turn to compute `turnsSincePreviousCompact`
    - `consecutive_failures` is the circuit-breaker counter. Once it hits
      `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3`, the layer no-ops on
      every subsequent call to stop hammering doomed retries

    Frozen because the orchestrator replaces `State.auto_compact_tracking`
    wholesale when a layer returns a fresh tracking instance — never
    in-place mutation (matches the State invariant).
    """

    compacted: bool = False
    turn_counter: int = 0
    turn_id: str = ""
    consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# Per-iteration error watermark — prevents infinite retry loops.
# ---------------------------------------------------------------------------


@dataclass
class ErrorWatermark:
    """Tracks which recovery-ladder rungs have been tried since the last clean
    iteration. Survives across iterations; resets only when the loop completes
    an iteration cleanly (model call + tool execution without raising).

    Why not reset-per-iteration: a persistent error (e.g., the API is down)
    should escalate rung-by-rung across retries, not restart at rung 1 each
    time and loop forever. Recovery counters live on State and survive across
    iterations.
    """

    tried_transient_retry: bool = False
    """Harness-level retry of a transient error (rate-limit, timeout).
    Distinct from a provider SDK's built-in retry — this is for cases
    where we want to retry at the loop level after an intra-loop state
    change. Not combined with `max_output_tokens_recovery_count` because
    the two rungs target different failure modes: a transient retry
    shouldn't consume the max-output-tokens budget, and vice versa."""

    tried_reduce_context: bool = False
    """Attempted a reactive compaction in response to prompt-too-long.
    Compaction is handled by `services.compact`."""

    tried_media_downscope: bool = False
    """Attempted a media-specific retry after replacing media blocks with
    bounded placeholders. Kept separate from `tried_reduce_context` because
    media-size failures should not consume token/context compaction rungs."""

    tried_drop_tools: bool = False
    """Harness-specific rung (not in reference): retry with tools=[] as a
    last resort when a misbehaving tool keeps causing failures. Reserved —
    not yet wired."""

    tried_fallback_model: bool = False
    """Switched to `config.fallback_model` after the primary model failed.
    the turn (`active_model` also stays on fallback)."""

    max_output_tokens_recovery_count: int = 0
    """Counter, not a bool: reference allows up to N recovery attempts
    escalating to terminal. Each `max_output_tokens` error bumps this; the
    rung fires only while the count is below limit."""

    last_error: str | None = None
    """Message from the most recent error. Surfaced in terminal result if we
    exhaust the ladder."""


# ---------------------------------------------------------------------------
# State — the per-iteration snapshot.
# ---------------------------------------------------------------------------


@dataclass
class State:
    """Per-iteration snapshot. Produced via dataclasses.replace(prev, ...).

    Do not mutate in place. The loop reads fields, computes a new State, and
    yields it; any consumer holding an old reference sees the prior iteration
    unchanged.
    """

    # --- message history ---
    messages: list[MessageParam] = field(default_factory=list["MessageParam"])
    """The API-visible message history. Appends happen by replace(messages=[*old, new])."""

    # --- iteration counter ---
    iteration: int = 0
    """Which agent-loop turn we're on. Incremented once per assistant response."""

    # --- usage accumulator ---
    usage: UsageTotals = field(default_factory=UsageTotals)

    # --- recovery bookkeeping ---
    error_watermark: ErrorWatermark = field(default_factory=ErrorWatermark)
    """Survives across iterations. Reset only on clean iteration completion
    (successful model call + tool execution), via
    `replace(error_watermark=ErrorWatermark())` at the successful-continue
    site. Persistent errors climb rungs rather than restart at rung 1."""

    # --- active model override — set by recovery ladder after fallback swap ---
    active_model: str | None = None
    """When set, `_call_model` uses this instead of `config.model`. Set by
    None = use config.model."""

    # --- compaction ---
    compact_boundaries: tuple[CompactBoundary, ...] = ()
    """History of compactions, in order. Tuple because append-only + frozen items."""

    # --- permission denials — replayed into the next turn as context ---
    permission_denials: tuple[PermissionDenial, ...] = ()

    # --- trusted deferred-tool discovery ---
    discovered_tool_names: frozenset[str] = field(
        default_factory=lambda: frozenset[str]()
    )
    """Deferred tool names selected by engine-owned ToolSearch execution.

    Transcript messages alone are not authority for deferred-tool execution:
    callers can seed or replay structured messages. The query loop updates
    this set only from tool results it produced while executing a ToolSearch
    call, and QueryEngine carries it across live turns.
    """

    # --- in-flight tool call count — for concurrency-limit checks ---
    in_flight_tool_calls: int = 0

    # --- terminal flag — set when the loop is about to emit a terminal result ---
    is_terminal: bool = False
    """True in the final State yielded by query() before it exits."""

    # --- autocompact bookkeeping — read/written by the autocompact layer ---
    auto_compact_tracking: AutoCompactTrackingState | None = None
    """Threaded across iterations within a turn. `None` until the autocompact
    layer first runs; replaced wholesale (`replace(state, ...)`) when a layer
    returns a fresh tracking instance via `LayerResult.auto_compact_tracking`.
    Reference `AutoCompactTrackingState` is held by the loop closure
    it on State so the recovery ladder + cross-iteration replay observe the
    same value."""


__all__ = [
    "AutoCompactTrackingState",
    "CompactBoundary",
    "ErrorWatermark",
    "PermissionDenial",
    "State",
    "UsageTotals",
]
