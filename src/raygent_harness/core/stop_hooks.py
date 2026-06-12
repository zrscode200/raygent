"""Stop-hook contract — in-process hook evaluator.

doesn't own: shell-process spawning, settings.json loading, prompt-suggestion
side effects, memory extraction, auto-dream, job classifier. Those belong
to higher layers (the SDK consumer wiring QueryDeps) or later build groups.

What lives here:
- The hook callable signature (`StopHook`).
- The evaluation dispatcher (`evaluate_on_success`, `fire_on_failure`).
- Typed result shapes with three outcomes: `continue` (hook said ok),
  `block` (hook raised a blocking error to inject into the conversation),
  `prevent` (hook vetoed turn-end — loop should keep going).
- Per-hook async timeout.

What does NOT live here:
- Shell execution. A consumer that wants shell hooks writes a
  `StopHook` that shells out, catches its output, and returns a
  `HookResult`. The harness is agnostic.
- Hook registration from config files. The consumer hands hooks to
  `QueryDeps.stop_hooks` directly; the config-loading is theirs.

Success-path vs failure-path (mirrors reference's `handleStopHooks` vs
- **Success**: hooks run when the loop is about to emit `completed`.
  They can veto (prevent_continuation → loop keeps going). Used for
  "are we actually done?" checks.
- **Failure**: hooks run when the loop is about to emit an error-class
  terminal (prompt_too_long, hook_stopped). Fire-and-forget side
  effects only; they cannot veto (the turn is already terminating).
  Used for cleanup / notification.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from raygent_harness.core.messages import (
        MessageParam,
        RaygentContinuationContextFragmentMetadata,
    )
    from raygent_harness.core.tool import ToolUseContext


# ---------------------------------------------------------------------------
# Hook context — the bag every hook receives.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookContext:
    """Passed to every stop-hook invocation.

    Frozen: hooks should not mutate the context. If a hook needs to
    publish data back to the harness, it does so via its `HookResult`
    return value.
    """

    messages: list[MessageParam]
    """Snapshot of the conversation at hook-invocation time. Read-only by
    convention (list identity isn't frozen, but hooks should not append)."""

    tool_use_context: ToolUseContext
    """For hooks that need cwd, abort signal, session_id, etc."""

    phase: Literal["success", "failure"]
    """Which path invoked us. Hooks that should only fire on success
    (e.g., 'are we done?' checks) gate on this."""


# ---------------------------------------------------------------------------
# Hook results — what a StopHook callable returns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookContinue:
    """Hook ran successfully, no action needed."""

    status: Literal["continue"] = "continue"


@dataclass(frozen=True)
class HookBlock:
    """Hook raised a blocking error — inject `message` into the
    conversation as a synthetic user message so the model can see the
    feedback. Reference equivalent: `getStopHookMessage(blockingError)`
    """

    message: str
    status: Literal["block"] = "block"


@dataclass(frozen=True)
class ContinuationContextFragment:
    """Model-visible context a stop hook wants to add before retrying.

    This is the headless Raygent equivalent of the visible MoreRight lifecycle
    seam: a hook can say "not enough context yet" and supply bounded fragments
    for the next model call without hiding side effects outside the transcript.
    """

    id: str
    content: str
    source: str | None = None
    reason: str | None = None
    priority: int = 0
    max_chars: int | None = None


@dataclass(frozen=True)
class HookContinueWithContext:
    """Hook asks the loop to continue with typed synthetic context.

    Semantically this is block-only retry with richer metadata: the rendered
    context is appended as user-role synthetic messages, yielded to the caller,
    persisted by QueryEngine, and included in the next model request.
    """

    fragments: tuple[ContinuationContextFragment, ...]
    message: str | None = None
    status: Literal["context"] = "context"


@dataclass(frozen=True)
class HookPreventContinuation:
    """Hook vetoed turn-end. The loop should keep iterating instead of
    emitting `completed`.
    """

    reason: str = "Stop hook prevented continuation"
    status: Literal["prevent"] = "prevent"


@dataclass(frozen=True)
class HookError:
    """Hook itself errored (raised or timed out). Not a blocking error
    from the user's perspective — it's an operational issue the harness
    should surface without failing the turn.
    """

    message: str
    status: Literal["error"] = "error"


HookResult = (
    HookContinue
    | HookBlock
    | HookContinueWithContext
    | HookPreventContinuation
    | HookError
)


# ---------------------------------------------------------------------------
# Hook callable signature.
# ---------------------------------------------------------------------------


StopHook = Callable[[HookContext], Awaitable[HookResult]]
"""A registered stop-hook. Async; must return a `HookResult` within the
configured timeout. Consumers register hooks by appending to
`QueryDeps.stop_hooks`."""


# ---------------------------------------------------------------------------
# Aggregate result of a hook-evaluation pass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopHookEvaluation:
    """What `evaluate_on_success` returns after running all hooks.

    Captures enough for the loop to decide what to do:
    - `prevent_continuation`: any hook said "keep going."
    - `blocking_messages`: synthetic user messages to append before the
      next iteration.
    - `errors`: hook-execution failures (not blocking errors).
    """

    prevent_continuation: bool = False
    blocking_messages: tuple[MessageParam, ...] = ()
    continuation_messages: tuple[MessageParam, ...] = ()
    errors: tuple[str, ...] = ()
    prevent_reason: str | None = None
    """First-seen prevention reason. Used in telemetry / summary."""

    hooks_ran: int = 0
    continuation_fragment_count: int = 0
    continuation_input_char_count: int = 0
    continuation_rendered_char_count: int = 0
    continuation_truncated_fragment_count: int = 0
    continuation_dropped_empty_fragment_count: int = 0
    continuation_dropped_fragment_count: int = 0
    continuation_truncated_message_count: int = 0


# ---------------------------------------------------------------------------
# Evaluators — one per phase.
# ---------------------------------------------------------------------------


DEFAULT_HOOK_TIMEOUT_S = 30.0
"""Per-hook async timeout. Hit the timeout → `HookError`. Conservative
default so a misbehaving hook doesn't wedge the loop indefinitely."""

DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS: Final = 8_000
"""Default total content budget for one stop-hook evaluation pass."""

DEFAULT_CONTINUATION_CONTEXT_FRAGMENT_CHARS: Final = 4_000
"""Default per-fragment content budget before applying a fragment override."""

DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS: Final = 64
"""Default cap on rendered fragments for one continuation-context message."""


async def evaluate_on_success(
    hooks: list[StopHook],
    ctx: HookContext,
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S,
) -> StopHookEvaluation:
    """Run hooks on the clean-completion path. Collect results and return
    an aggregate. Hooks run sequentially in registration order.

    via a streaming generator). Sequential is simpler for v1; if hook
    wall-time becomes a problem, revisit with `asyncio.gather` + the
    same aggregation logic.

    Sequential is also more debuggable: a hook that breaks behavior
    can be isolated by disabling subsequent hooks; concurrent gather
    interleaves failures.
    """
    if ctx.phase != "success":
        msg = f"evaluate_on_success called with phase={ctx.phase}"
        raise ValueError(msg)

    return await _run_hooks(hooks, ctx, timeout_s)


async def fire_on_failure(
    hooks: list[StopHook],
    ctx: HookContext,
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S,
) -> StopHookEvaluation:
    """Run hooks on the error-class terminal path. Results are collected
    for telemetry but callers typically ignore `prevent_continuation`
    (the turn is already terminating — a hook can't un-terminate it).

    Separate function from `evaluate_on_success` because the semantics
    ARE different even if the machinery is the same: failure-path hooks
    should not be able to turn a `prompt_too_long` terminal into "keep
    prompt-too-long creates a "death spiral").
    """
    if ctx.phase != "failure":
        msg = f"fire_on_failure called with phase={ctx.phase}"
        raise ValueError(msg)

    result = await _run_hooks(hooks, ctx, timeout_s)
    # Strip prevent_continuation — callers of failure-path hooks must not
    # be able to un-terminate the turn. Preserve the rest for telemetry.
    if result.prevent_continuation:
        result = StopHookEvaluation(
            prevent_continuation=False,
            blocking_messages=result.blocking_messages,
            continuation_messages=result.continuation_messages,
            errors=(*result.errors, "prevent_continuation ignored on failure path"),
            prevent_reason=None,
            hooks_ran=result.hooks_ran,
            continuation_fragment_count=result.continuation_fragment_count,
            continuation_input_char_count=result.continuation_input_char_count,
            continuation_rendered_char_count=result.continuation_rendered_char_count,
            continuation_truncated_fragment_count=(
                result.continuation_truncated_fragment_count
            ),
            continuation_dropped_empty_fragment_count=(
                result.continuation_dropped_empty_fragment_count
            ),
            continuation_dropped_fragment_count=result.continuation_dropped_fragment_count,
            continuation_truncated_message_count=(
                result.continuation_truncated_message_count
            ),
        )
    return result


# ---------------------------------------------------------------------------
# Internal — shared runner.
# ---------------------------------------------------------------------------


@dataclass
class _Accum:
    prevent: bool = False
    prevent_reason: str | None = None
    blocking: list[MessageParam] = field(default_factory=list["MessageParam"])
    continuation: list[MessageParam] = field(default_factory=list["MessageParam"])
    errors: list[str] = field(default_factory=list[str])
    ran: int = 0
    continuation_fragment_count: int = 0
    continuation_input_char_count: int = 0
    continuation_rendered_char_count: int = 0
    continuation_truncated_fragment_count: int = 0
    continuation_dropped_empty_fragment_count: int = 0
    continuation_dropped_fragment_count: int = 0
    continuation_truncated_message_count: int = 0


async def _run_hooks(
    hooks: list[StopHook],
    ctx: HookContext,
    timeout_s: float,
) -> StopHookEvaluation:
    """Core dispatch. Each hook is awaited with a timeout; exceptions and
    timeouts become `HookError`s and don't abort the pass (other hooks
    still run).
    """
    acc = _Accum()

    for hook in hooks:
        acc.ran += 1
        try:
            result = await asyncio.wait_for(hook(ctx), timeout=timeout_s)
        except TimeoutError:
            acc.errors.append(f"hook timed out after {timeout_s}s")
            continue
        except Exception as err:
            acc.errors.append(f"hook raised: {err}")
            continue

        _fold(acc, result)

    return StopHookEvaluation(
        prevent_continuation=acc.prevent,
        blocking_messages=tuple(acc.blocking),
        continuation_messages=tuple(acc.continuation),
        errors=tuple(acc.errors),
        prevent_reason=acc.prevent_reason,
        hooks_ran=acc.ran,
        continuation_fragment_count=acc.continuation_fragment_count,
        continuation_input_char_count=acc.continuation_input_char_count,
        continuation_rendered_char_count=acc.continuation_rendered_char_count,
        continuation_truncated_fragment_count=acc.continuation_truncated_fragment_count,
        continuation_dropped_empty_fragment_count=(
            acc.continuation_dropped_empty_fragment_count
        ),
        continuation_dropped_fragment_count=acc.continuation_dropped_fragment_count,
        continuation_truncated_message_count=acc.continuation_truncated_message_count,
    )


def _fold(acc: _Accum, result: HookResult) -> None:
    """Fold one hook's result into the accumulator."""
    if isinstance(result, HookPreventContinuation):
        # First prevent wins the reason; subsequent prevents still count.
        if not acc.prevent:
            acc.prevent_reason = result.reason
        acc.prevent = True
        return

    if isinstance(result, HookBlock):
        acc.blocking.append({"role": "user", "content": result.message})
        return

    if isinstance(result, HookContinueWithContext):
        message, stats = _render_continuation_context_message(result)
        acc.continuation_fragment_count += stats.fragment_count
        acc.continuation_input_char_count += stats.input_char_count
        acc.continuation_rendered_char_count += stats.rendered_char_count
        acc.continuation_truncated_fragment_count += stats.truncated_fragment_count
        acc.continuation_dropped_empty_fragment_count += stats.dropped_empty_fragment_count
        acc.continuation_dropped_fragment_count += stats.dropped_fragment_count
        if stats.rendered_message_truncated:
            acc.continuation_truncated_message_count += 1
        if message is not None:
            acc.continuation.append(message)
        return

    if isinstance(result, HookError):
        acc.errors.append(result.message)
        return

    # HookContinue — no action.


@dataclass(frozen=True)
class _ContinuationRenderStats:
    fragment_count: int = 0
    input_char_count: int = 0
    rendered_char_count: int = 0
    truncated_fragment_count: int = 0
    dropped_empty_fragment_count: int = 0
    dropped_fragment_count: int = 0
    rendered_message_truncated: bool = False


def _render_continuation_context_message(
    result: HookContinueWithContext,
) -> tuple[MessageParam | None, _ContinuationRenderStats]:
    ordered_fragments = sorted(
        enumerate(result.fragments),
        key=lambda item: (item[1].priority, item[0]),
    )
    non_empty_fragments: list[tuple[ContinuationContextFragment, str]] = []
    dropped_empty_count = 0
    input_char_count = 0
    for _index, fragment in ordered_fragments:
        raw_content = fragment.content.strip()
        if not raw_content:
            dropped_empty_count += 1
            continue
        input_char_count += len(raw_content)
        non_empty_fragments.append((fragment, raw_content))

    selected_fragments = non_empty_fragments[:DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS]
    dropped_fragment_count = max(
        0,
        len(non_empty_fragments) - len(selected_fragments),
    )

    sections: list[str] = ["[continuation context]"]
    metadata_fragments: list[RaygentContinuationContextFragmentMetadata] = []
    truncated_count = 0
    rendered_message_truncated = False

    lead_message = result.message.strip() if result.message is not None else ""
    if lead_message:
        added, truncated = _append_bounded_section(
            sections,
            lead_message,
            marker="\n[lead message truncated]",
        )
        rendered_message_truncated = rendered_message_truncated or truncated
        if not added:
            rendered_message_truncated = True

    budget_exhausted = _rendered_sections_len(sections) >= (
        DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS
    )

    for index, (fragment, raw_content) in enumerate(selected_fragments):
        if budget_exhausted:
            dropped_fragment_count += len(selected_fragments) - index
            break
        per_fragment_budget = DEFAULT_CONTINUATION_CONTEXT_FRAGMENT_CHARS
        if fragment.max_chars is not None:
            per_fragment_budget = min(per_fragment_budget, max(0, fragment.max_chars))
        rendered_content = raw_content[:per_fragment_budget]
        truncated = len(raw_content) > len(rendered_content)

        lines = [_continuation_fragment_header(fragment)]
        if rendered_content:
            lines.append(rendered_content)
        if truncated:
            lines.append(f"[truncated to {len(rendered_content)} chars]")
        added, message_truncated = _append_bounded_section(
            sections,
            "\n".join(lines),
            marker="\n[continuation fragment truncated]",
        )
        if not added:
            dropped_fragment_count += len(selected_fragments) - index
            budget_exhausted = True
            break
        truncated = truncated or message_truncated
        rendered_message_truncated = rendered_message_truncated or message_truncated
        if truncated:
            truncated_count += 1
        metadata_fragments.append(
            cast(
                "RaygentContinuationContextFragmentMetadata",
                {
                    "id": _bounded_label(fragment.id),
                    "input_chars": len(raw_content),
                    "rendered_chars": len(rendered_content),
                    "truncated": truncated,
                    "source": _bounded_optional_label(fragment.source),
                    "reason": _bounded_optional_label(fragment.reason),
                },
            )
        )
        budget_exhausted = _rendered_sections_len(sections) >= (
            DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS
        )

    if len(sections) == 1 and not lead_message and not metadata_fragments:
        return (
            None,
            _ContinuationRenderStats(
                fragment_count=0,
                input_char_count=input_char_count,
                rendered_char_count=0,
                truncated_fragment_count=truncated_count,
                dropped_empty_fragment_count=dropped_empty_count,
                dropped_fragment_count=dropped_fragment_count,
                rendered_message_truncated=rendered_message_truncated,
            ),
        )

    content = "\n\n".join(sections)
    message: MessageParam = {
        "role": "user",
        "content": content,
        "raygentMessageKind": "continuation_context",
        "raygentContinuationContext": {
            "type": "continuation_context",
            "fragment_count": len(metadata_fragments),
            "input_char_count": input_char_count,
            "rendered_char_count": len(content),
            "truncated_fragment_count": truncated_count,
            "dropped_empty_fragment_count": dropped_empty_count,
            "dropped_fragment_count": dropped_fragment_count,
            "rendered_message_truncated": rendered_message_truncated,
            "total_budget_chars": DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS,
            "per_fragment_budget_chars": DEFAULT_CONTINUATION_CONTEXT_FRAGMENT_CHARS,
            "max_fragment_count": DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS,
            "fragments": metadata_fragments,
        },
    }
    return (
        message,
        _ContinuationRenderStats(
            fragment_count=len(metadata_fragments),
            input_char_count=input_char_count,
            rendered_char_count=len(content),
            truncated_fragment_count=truncated_count,
            dropped_empty_fragment_count=dropped_empty_count,
            dropped_fragment_count=dropped_fragment_count,
            rendered_message_truncated=rendered_message_truncated,
        ),
    )


def _continuation_fragment_header(fragment: ContinuationContextFragment) -> str:
    parts = [f"id={_bounded_label(fragment.id)}"]
    source = _bounded_optional_label(fragment.source)
    if source:
        parts.append(f"source={source}")
    reason = _bounded_optional_label(fragment.reason)
    if reason:
        parts.append(f"reason={reason}")
    return "[context " + " ".join(parts) + "]"


def _append_bounded_section(
    sections: list[str],
    section: str,
    *,
    marker: str,
    total_budget: int = DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS,
) -> tuple[bool, bool]:
    """Append one model-visible section without exceeding the total budget."""
    separator_len = 2 if sections else 0
    remaining = total_budget - _rendered_sections_len(sections) - separator_len
    if remaining <= 0:
        return False, True
    if len(section) <= remaining:
        sections.append(section)
        return True, False
    if len(marker) < remaining:
        sections.append(f"{section[: remaining - len(marker)]}{marker}")
    else:
        sections.append(section[:remaining])
    return True, True


def _rendered_sections_len(sections: list[str]) -> int:
    if not sections:
        return 0
    return sum(len(section) for section in sections) + (2 * (len(sections) - 1))


def _bounded_label(value: str, *, max_chars: int = 120) -> str:
    return _bounded_optional_label(value, max_chars=max_chars) or ""


def _bounded_optional_label(value: str | None, *, max_chars: int = 120) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars]}..."


__all__ = [
    "DEFAULT_CONTINUATION_CONTEXT_FRAGMENT_CHARS",
    "DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS",
    "DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS",
    "DEFAULT_HOOK_TIMEOUT_S",
    "ContinuationContextFragment",
    "HookBlock",
    "HookContext",
    "HookContinue",
    "HookContinueWithContext",
    "HookError",
    "HookPreventContinuation",
    "HookResult",
    "StopHook",
    "StopHookEvaluation",
    "evaluate_on_success",
    "fire_on_failure",
]
