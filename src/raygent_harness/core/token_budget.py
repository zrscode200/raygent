"""Token-budget tracker for adaptive turn continuation.

Distinct from `QueryConfig.budget` hard ceilings, this module implements
adaptive continuation: as the turn spends tokens, Raygent decides whether to
continue another iteration (still making progress) or stop (diminishing returns
or near the budget). The tracker is always constructible but only consulted
when configured by the caller.

Contract:

- One `BudgetTracker` per turn (created at `query()` entry).
- `check_token_budget(tracker, agent_id, budget, turn_tokens)` returns
  `BudgetContinue` or `BudgetStop`. `Continue` carries a nudge message to
  inject into the conversation; `Stop` carries an optional completion
  event.
- Subagents never continue via this mechanism — `agent_id is not None`

Integration note: the loop's `_call_model` must surface `turn_tokens` (a
running total of input+output) before this tracker can do useful work.
The types and logic are intentionally independent from provider adapters so
callers can wire budget checks at their chosen model-call boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Budget thresholds.
# ---------------------------------------------------------------------------

COMPLETION_THRESHOLD = 0.9
"""Fraction of budget that signals the turn is close to done."""

DIMINISHING_THRESHOLD = 500
"""Delta-token cutoff below which a continuation is treated as low progress."""

DIMINISHING_RETURN_MIN_CONTINUATIONS = 3
"""Minimum continuations before the diminishing-returns check can fire."""


# ---------------------------------------------------------------------------
# BudgetTracker — mutable per-turn state for the budget algorithm.
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Per-turn bookkeeping for the adaptive-continuation decision.

    tracker is held by the loop, mutated by `check_token_budget`, and not
    visible outside the turn. Keeping it mutable sidesteps a replace()
    ceremony on every check.
    """

    continuation_count: int = 0
    """How many times we've already told the model 'keep going.' Used
    for the diminishing-returns gate."""

    last_delta_tokens: int = 0
    """Tokens spent in the most recent check interval. Compared against
    `DIMINISHING_THRESHOLD` to detect stall."""

    last_global_turn_tokens: int = 0
    """Snapshot of `global_turn_tokens` at the last check, so we can
    compute delta without the caller tracking it."""

    started_at: float = 0.0
    """Wall-clock time (seconds) at turn start. Surfaced in completion
    event for duration telemetry."""


def create_budget_tracker() -> BudgetTracker:
    return BudgetTracker(started_at=time.time())


# ---------------------------------------------------------------------------
# Decision types — what `check_token_budget` returns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetContinue:
    """The model should be nudged to continue. Loop injects `nudge_message`
    into the conversation as the next user message.
    """

    action: Literal["continue"] = "continue"
    nudge_message: str = ""
    continuation_count: int = 0
    pct: int = 0
    """Percentage of budget consumed. 0-100 (rounded)."""

    turn_tokens: int = 0
    budget: int = 0


@dataclass(frozen=True)
class BudgetCompletionEvent:
    """Telemetry payload attached to a `BudgetStop`. None when the turn
    stopped *without* ever hitting a continuation — a normal short turn
    doesn't need to announce it finished under budget.
    """

    continuation_count: int
    pct: int
    turn_tokens: int
    budget: int
    diminishing_returns: bool
    duration_ms: int


@dataclass(frozen=True)
class BudgetStop:
    """The turn should stop. `completion_event` is populated when we had
    continuations or hit diminishing returns; `None` means a short turn
    stopped naturally (nothing interesting to report).
    """

    action: Literal["stop"] = "stop"
    completion_event: BudgetCompletionEvent | None = None


BudgetDecision = BudgetContinue | BudgetStop


# ---------------------------------------------------------------------------
# check_token_budget — the decision function. Mutates tracker.
# ---------------------------------------------------------------------------


def check_token_budget(
    tracker: BudgetTracker,
    agent_id: str | None,
    budget: int | None,
    global_turn_tokens: int,
) -> BudgetDecision:
    """Decide whether to continue or stop the turn based on token spend.

    when this is a subagent (`agent_id is not None`) or when the caller
    didn't set a budget. Otherwise:

    1. Compute pct of budget and delta since last check.
    2. Diminishing-returns check: `continuation_count >= 3` AND
       delta < 500 AND last_delta < 500 → stop.
    3. Not diminishing AND turn < 90% of budget → continue (bump
       counter, update snapshots, return nudge message).
    4. Otherwise stop. Emit completion event if we had continuations
       or hit diminishing returns; else stop silently.

    Mutates `tracker` in place per the reference pattern.
    """
    if agent_id is not None or budget is None or budget <= 0:
        return BudgetStop()

    turn_tokens = global_turn_tokens
    pct = round((turn_tokens / budget) * 100)
    delta_since_last_check = global_turn_tokens - tracker.last_global_turn_tokens

    is_diminishing = (
        tracker.continuation_count >= DIMINISHING_RETURN_MIN_CONTINUATIONS
        and delta_since_last_check < DIMINISHING_THRESHOLD
        and tracker.last_delta_tokens < DIMINISHING_THRESHOLD
    )

    if not is_diminishing and turn_tokens < budget * COMPLETION_THRESHOLD:
        tracker.continuation_count += 1
        tracker.last_delta_tokens = delta_since_last_check
        tracker.last_global_turn_tokens = global_turn_tokens
        return BudgetContinue(
            nudge_message=_build_nudge_message(pct, turn_tokens, budget),
            continuation_count=tracker.continuation_count,
            pct=pct,
            turn_tokens=turn_tokens,
            budget=budget,
        )

    if is_diminishing or tracker.continuation_count > 0:
        return BudgetStop(
            completion_event=BudgetCompletionEvent(
                continuation_count=tracker.continuation_count,
                pct=pct,
                turn_tokens=turn_tokens,
                budget=budget,
                diminishing_returns=is_diminishing,
                duration_ms=int((time.time() - tracker.started_at) * 1000),
            ),
        )

    return BudgetStop()


def _build_nudge_message(pct: int, turn_tokens: int, budget: int) -> str:
    """Nudge text injected into the conversation on `continue`. Reference
    we build an equivalent inline — the prompt wording lives here for
    auditability (not hidden in a utils module).
    """
    return (
        f"[system] You've used {turn_tokens:,} of {budget:,} turn tokens "
        f"({pct}%). There's budget left — continue the task."
    )


__all__ = [
    "COMPLETION_THRESHOLD",
    "DIMINISHING_RETURN_MIN_CONTINUATIONS",
    "DIMINISHING_THRESHOLD",
    "BudgetCompletionEvent",
    "BudgetContinue",
    "BudgetDecision",
    "BudgetStop",
    "BudgetTracker",
    "check_token_budget",
    "create_budget_tracker",
]
