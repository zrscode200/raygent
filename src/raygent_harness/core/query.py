"""query() — the per-turn async-generator agent loop.

The per-turn agent loop is exposed as one async generator. Raygent keeps the
loop body, event stream, recovery ladder, compaction pipeline, tool execution,
and terminal result construction in this module.

Shape contract (invariants this file is built to preserve):

1. **Async generator.** `query` yields events as they happen (assistant
   messages, tool-result messages, compact boundaries) and the *last event it
   yields* is always a `TerminalEvent` carrying the `Terminal` for the turn.
   Python async generators cannot return a value from within the body (unlike
   sync generators that stash it on `StopAsyncIteration.value`), so the
   terminal is part of the yield union, not the return. Callers dispatch on
   `isinstance(event, TerminalEvent)` to close out the turn. Python does not
   support returning the terminal from an async generator, so Raygent flattens
   the terminal into the yield stream.

2. **Immutable config, mutable-by-convention state.** `config: QueryConfig` is
   frozen (`core/config.py`) and never reassigned. `state: State`
   (`core/state.py`) is mutable *as a convenience for wholesale reassignment*
   but individual iterations produce new States via `dataclasses.replace`. At
   every branch that iterates, we assign `state = replace(state, ...)` — not
   `state.field = ...`. This preserves the recovery-rollback +
   compaction-replay invariants.

3. **Layered context pipeline (read-only message transforms before model call).**
   a. tool-result budget (oversize outputs → replacement markers)
   b. snip (HISTORY_SNIP gate)
   c. microcompact
   d. context collapse
   e. autocompact
   Each layer returns a new messages list; none mutate state. Later layers see
   the output of earlier ones.

4. **Recovery ladder persists across iterations.** `state.error_watermark`
   tracks which rungs have been tried since the last clean iteration.
   Rungs: fallback-model → reduce-context → max_output_tokens recovery →
   transient retry → terminal. The watermark resets ONLY at the successful-
   continue site (after model call + tool execution complete without
   raising). A persistent error therefore climbs rungs across iterations rather
   than restarting at rung 1.

5. **Abort is cooperative via `ctx.abort_event`.** We check it between stages,
   not inside tight loops. A set event means the next iteration boundary should
   return a `Terminal(reason="aborted_*")`.

This module is the *skeleton*. Stage functions (context pipeline layers, model
call, tool orchestration, stop hooks, recovery) are placeholders that raise
`NotImplementedError` — filled in by subsequent files in the build order.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import (
    ContextFragment,
    render_user_context_messages,
    scope_context_fragments,
)
from raygent_harness.core.media_budget import downscope_message_params_for_media_retry
from raygent_harness.core.messages import (
    MessageParam,
    api_message_from_message_param,
    message_param_from_api_message,
    message_param_from_model_api_error,
    message_param_from_model_message,
    thaw_json,
)
from raygent_harness.core.model_adapter import ToolUseBlock, normalize_assistant_turn
from raygent_harness.core.model_registry import (
    build_model_budget_snapshot,
    get_model_output_limits,
    model_info_with_fallback,
    resolve_skill_model_override,
)
from raygent_harness.core.model_request_normalization import (
    normalize_model_request_for_provider,
)
from raygent_harness.core.model_stream import ModelStreamAssembler
from raygent_harness.core.model_types import (
    FrozenJson,
    MediaContentBlock,
    ModelApiErrorMessage,
    ModelContentBlock,
    ModelFallbackControl,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelSampling,
    ModelStreamEvent,
    ModelToolSpec,
    PermissionContextSnapshot,
    ProviderError,
    StreamingTransportFallbackControl,
    TextContentBlock,
    ThinkingContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
    Usage,
    build_model_api_error_message,
)
from raygent_harness.core.observability import KernelEventContext, redacted_payload
from raygent_harness.core.state import (
    AutoCompactTrackingState,
    CompactBoundary,
    ErrorWatermark,
    PermissionDenial,
    State,
    UsageTotals,
)
from raygent_harness.core.stop_hooks import (
    HookContext,
    evaluate_on_success,
)
from raygent_harness.core.streaming_tool_executor import (
    StreamingToolExecutor,
    StreamingToolProgressUpdate,
)
from raygent_harness.core.task import TaskNotification
from raygent_harness.core.tool import Tool, ToolRuntimeContext, ToolUseContext
from raygent_harness.core.tool_execution import ToolExecutionProgress, ToolExecutionResult
from raygent_harness.core.tool_orchestration import (
    ToolOrchestrationOutcome,
    run_tools,
)

if TYPE_CHECKING:
    from raygent_harness.core.deps import MemoryRecallPrefetch, QueryDeps
    from raygent_harness.services.compact.tool_result_budget import (
        ToolResultReplacementRecord,
    )


# ---------------------------------------------------------------------------
# Terminal — the payload carried by the final `TerminalEvent` of each turn.
# One reason per loop-exit path; yielded (not returned) since Python async
# generators cannot carry a return value.
# ---------------------------------------------------------------------------

TerminalReason = Literal[
    "completed",
    "max_turns",
    "budget_exceeded",
    "blocking_limit",
    "prompt_too_long",
    "image_error",
    "model_error",
    "aborted_streaming",
    "aborted_tools",
    "hook_stopped",
    "stop_hook_prevented",
    "fallback_exhausted",
]
# `fallback_exhausted` distinguishes raw model failures from a fully exhausted
# fallback chain. `budget_exceeded` surfaces loop-level budget enforcement with
# the same terminal event shape as other turn-level limits.


@dataclass(frozen=True)
class Terminal:
    """The value returned by the `query()` generator when the turn ends.

    Frozen: a terminal is a historical fact; callers log it, inspect it, emit
    metrics. Mutating it would break those downstream readers.
    """

    reason: TerminalReason
    message: str | None = None
    """Human-readable detail. For `model_error`, the last API error string."""

    turn_count: int = 0
    """How many iterations ran before termination."""

    final_state: State | None = None
    """The last State produced. Useful for tests and observability; callers
    that only care about messages can read from here."""


# ---------------------------------------------------------------------------
# Query events — what the generator yields as the turn unfolds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamRequestStart:
    """Emitted at the top of each iteration, before the model call.

    """

    type: Literal["stream_request_start"] = "stream_request_start"
    iteration: int = 0


@dataclass(frozen=True)
class AssistantMessage:
    """A completed assistant message. Yielded after streaming completes."""

    type: Literal["assistant_message"] = "assistant_message"
    message: MessageParam = field(default_factory=lambda: {"role": "assistant", "content": ""})  # type: ignore[arg-type]
    api_message: MessageParam | None = None
    """Optional API-bound payload when `message` is observer-enriched."""


@dataclass(frozen=True)
class ToolResultMessage:
    """Yielded after all tool calls for the iteration resolve.

    Carries the synthetic user message that will be appended to history.
    """

    type: Literal["tool_result_message"] = "tool_result_message"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class ToolProgressEvent:
    """Progress emitted by a running tool call.

    Progress is timeline/observer-visible only; it is not appended to
    `State.messages` or fed to the next model call.
    """

    type: Literal["tool_progress"] = "tool_progress"
    tool_use_id: str = ""
    tool_name: str = ""
    message: str = ""
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolOrchestrationComplete:
    """Internal completion carrier for one tool-use batch.

    Progress/result messages may be yielded while tools run, but the loop also
    needs a final structured outcome so it can carry the actual tool-result
    messages into the next state. This prevents the placeholder trap where
    `_build_tool_results(response)` tries to infer tool results from the model
    response itself.
    """

    type: Literal["tool_orchestration_complete"] = "tool_orchestration_complete"
    tool_result_messages: tuple[MessageParam, ...] = ()
    permission_denials: tuple[PermissionDenial, ...] = ()
    updated_context: ToolUseContext | None = None
    should_prevent_continuation: bool = False
    prevent_reason: str | None = None
    aborted: bool = False
    abort_reason: str | None = None


@dataclass(frozen=True)
class ModelStreamEventMessage:
    """Internal normalized model stream event.

    Wave 1 keeps stream internals headless: QueryEngine persists these for
    replay/adapter reconstruction, but SDK translation absorbs them by default.
    The model-visible transcript still receives only the final assistant
    message assembled from the stream.
    """

    type: Literal["model_stream_event"] = "model_stream_event"
    event: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class CompactBoundaryEvent:
    """A compaction layer produced a boundary. Timeline-visible.

    `message_index` is a pre-compact input index, not a live index into the
    post-compact `State.messages`.
    """

    type: Literal["compact_boundary"] = "compact_boundary"
    kind: Literal["microcompact", "autocompact", "context_collapse", "snip"] = "microcompact"
    message_index: int = 0
    summary: str = ""


@dataclass(frozen=True)
class TombstoneMessage:
    """A stream/model fallback tombstone for an invalidated attempt.

    Replay and adapters need a durable marker explaining why partial
    output/tool state should not be treated as the live branch.
    """

    type: Literal["tombstone"] = "tombstone"
    target_entry_id: str | None = None
    target_message_id: str | None = None
    reason: str = ""
    event: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskNotificationsMessage:
    """Yielded when the iteration-top drain produced one or more
    notifications. Carries the synthetic user message that was appended
    to `state.messages` so consumers (QueryEngine) can persist it to
    the cross-turn log — without this, drained notifications live only
    in per-turn state and disappear next turn.

    The event is yielded to the consumer and folded into next-iteration
    messages. Reusing `ToolResultMessage` would be a semantic stretch because
    this is not a tool result; a distinct event type lets the consumer dispatch
    on what is really happening.
    """

    type: Literal["task_notifications"] = "task_notifications"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class CoordinatorRuntimeMessage:
    """Yielded when coordinator runtime state renders a model-visible digest.

    This is transcript-visible mutable coordination state, unlike static
    worker-tool context that belongs in `QueryConfig.context_messages`.
    QueryEngine persists it like task notifications so replay observes the same
    coordinator state snapshot the model saw.
    """

    type: Literal["coordinator_runtime"] = "coordinator_runtime"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class LocalAgentMessagesMessage:
    """Yielded when a running local_agent drains SendMessage queued prompts.

    Reference local agents drain `pendingMessages` into queued-command
    attachments at the next tool/model boundary. Raygent represents the same
    model-visible input as user-role synthetic messages so QueryEngine can
    persist them in the sidechain transcript exactly where the model saw them.
    """

    type: Literal["local_agent_messages"] = "local_agent_messages"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class PostCompactMessage:
    """Yielded when compaction rewrites API-visible history.

    Carries the synthetic user-role message(s) that replace pre-compact
    history. Without this event, the model can see the compacted summary while
    SDK/transcript consumers only learn about it later via terminal
    reconciliation.
    """

    type: Literal["post_compact_message"] = "post_compact_message"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class MemoryRecallMessage:
    """Yielded when query-time memory recall injects model-visible context.

    Reference relevant-memory attachments are yielded to the consumer and folded
    into the next model-call message list after tool results. Raygent carries the
    provider-neutral user message here so QueryEngine can persist the same
    synthetic context into the cross-turn transcript.
    """

    type: Literal["memory_recall_message"] = "memory_recall_message"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class StopHookMessage:
    """Yielded when a stop hook emits model-visible blocking feedback.

    The query loop carries the same messages into retry state so Raygent's live
    model state, QueryEngine message log, and transcript log stay aligned.
    """

    type: Literal["stop_hook_message"] = "stop_hook_message"
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]


@dataclass(frozen=True)
class ContentReplacementRecords:
    """Internal event carrying newly persisted tool-result replacements.

    Tool-result budgeting rewrites API-visible messages before the model call.
    QueryEngine consumes this event to append transcript replacement records,
    but SDK callers do not see it. Keeping it as a query event avoids importing
    transcript storage into the compact service.
    """

    type: Literal["content_replacement_records"] = "content_replacement_records"
    replacements: tuple[ToolResultReplacementRecord, ...] = ()


@dataclass(frozen=True)
class TerminalEvent:
    """Always the last event yielded by `query()`. Carries the `Terminal`.

    Python async generators cannot return a value; callers read the terminal
    by pattern-matching on this event. Every `query()` exit path yields
    exactly one `TerminalEvent` and then returns.
    """

    type: Literal["terminal"] = "terminal"
    terminal: Terminal = field(default_factory=lambda: Terminal(reason="completed"))


QueryEvent = (
    StreamRequestStart
    | AssistantMessage
    | ToolResultMessage
    | ToolProgressEvent
    | ToolOrchestrationComplete
    | ModelStreamEventMessage
    | TaskNotificationsMessage
    | CoordinatorRuntimeMessage
    | LocalAgentMessagesMessage
    | PostCompactMessage
    | MemoryRecallMessage
    | StopHookMessage
    | ContentReplacementRecords
    | CompactBoundaryEvent
    | TombstoneMessage
    | TerminalEvent
)
"""The yield union for `query()`.

**Deferred runtime surfaces:**
  background tool-result summarizer.

Product SDK partial-message surfaces remain deferred. `ModelStreamEventMessage`
is a headless internal event for persistence/adapter reconstruction, not a
user-facing SDK chunk.
"""


# ---------------------------------------------------------------------------
# Context pipeline — layer protocol + orchestrator.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerResult:
    """What a pipeline layer returns. Layers are pure over `messages`.

    Layers should not mutate `state`; if they produce a boundary event the
    orchestrator appends it to `state.compact_boundaries` via `replace()`.
    """

    messages: list[MessageParam]
    """The possibly-rewritten message list."""

    boundary: CompactBoundaryEvent | None = None
    """Optional compaction boundary marking where this layer compressed
    history. `None` when the layer was a no-op or trivial rewrite (e.g.,
    tool-result size caps don't produce boundaries)."""

    tokens_freed: int = 0
    """How many tokens the layer removed.

    Later layers can use this value when deciding whether enough headroom has
    already been recovered.
    """

    auto_compact_tracking: AutoCompactTrackingState | None = None
    """Updated tracking state from the autocompact layer (if it ran). The
    orchestrator applies via `replace(state, auto_compact_tracking=...)`
    when non-None. Other layers (microcompact, tool-result-budget) leave
    `{ wasCompacted, ..., consecutiveFailures }` — we collapse those into
    one updated tracking value the layer constructs internally."""

    content_replacements: tuple[ToolResultReplacementRecord, ...] = ()
    """New tool-result replacement records produced by this layer.

    QueryEngine persists these as separate transcript entries. They are not
    model-visible messages and are intentionally not surfaced to SDK callers.
    """


Layer = Callable[
    [list["MessageParam"], State, "QueryConfig", "ToolUseContext"],
    Awaitable[LayerResult],
]
"""Uniform context-pipeline layer signature.

Concrete layers can ignore inputs they do not need; the shared shape keeps the
orchestrator simple and makes layer order explicit.
"""


async def _run_context_pipeline(
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> tuple[
    list[MessageParam],
    list[CompactBoundaryEvent],
    AutoCompactTrackingState | None,
    tuple[ToolResultReplacementRecord, ...],
]:
    """Run context layers in model-input order.

    Ordering is intentionally stable: tool-result budget replacement first,
    microcompact second, and autocompact last. The function returns the model
    input messages, any boundary events, the latest tracking-state update, and
    any tool-result replacement records. It does not mutate state.
    """
    messages = list(state.messages)
    boundaries: list[CompactBoundaryEvent] = []
    tracking_update: AutoCompactTrackingState | None = None
    content_replacements: list[ToolResultReplacementRecord] = []

    for layer in _ordered_layers(deps):
        pre_layer_message_count = len(messages)
        result = await layer(messages, state, config, ctx)
        messages = result.messages
        if result.boundary is not None:
            boundaries.append(result.boundary)
            _emit_observability(
                deps,
                "compact.boundary",
                _observability_context(config, ctx, state=state, source="compact"),
                {
                    "kind": result.boundary.kind,
                    "message_index": result.boundary.message_index,
                    "pre_message_count": pre_layer_message_count,
                    "post_message_count": len(result.messages),
                    "tokens_freed": result.tokens_freed,
                    "summary_char_count": len(result.boundary.summary),
                },
            )
        if result.auto_compact_tracking is not None:
            previous_failures = (
                state.auto_compact_tracking.consecutive_failures
                if state.auto_compact_tracking is not None
                else 0
            )
            if (
                result.boundary is None
                and result.auto_compact_tracking.consecutive_failures
                > previous_failures
            ):
                _emit_observability(
                    deps,
                    "compact.failed",
                    _observability_context(
                        config,
                        ctx,
                        state=state,
                        source="compact",
                    ),
                    {
                        "kind": "autocompact",
                        "pre_message_count": pre_layer_message_count,
                        "consecutive_failures": (
                            result.auto_compact_tracking.consecutive_failures
                        ),
                    },
                )
            tracking_update = result.auto_compact_tracking
        content_replacements.extend(result.content_replacements)

    return messages, boundaries, tracking_update, tuple(content_replacements)


def _ordered_layers(deps: QueryDeps) -> list[Layer]:
    """Pipeline order. Centralized so the ordering invariant is one-line
    auditable. Don't reorder without updating the implementation rationale —
    the reference chose this order for cache-coherence and threshold-
    composition reasons.
    """
    return [
        _apply_tool_result_budget,
        deps.microcompact,
        deps.autocompact,
    ]


async def _apply_tool_result_budget(
    messages: list[MessageParam],
    _state: State,
    _config: QueryConfig,
    ctx: ToolUseContext,
) -> LayerResult:
    """Per-call output size cap. No-op in skeleton.

    Scans tool_result content
    blocks, writes oversized payloads to disk, replaces the in-message content
    with a marker. Produces no boundary because this is a per-result rewrite,
    not a history-compression step.
    """
    from raygent_harness.services.compact.tool_result_budget import (
        apply_tool_result_budget,
    )

    result = await apply_tool_result_budget(messages, ctx.content_replacement)
    return LayerResult(
        messages=result.messages,
        content_replacements=result.newly_replaced,
    )


# ---------------------------------------------------------------------------
# Task-notification drain — runs at iteration top, before context pipeline.
# ---------------------------------------------------------------------------


def _drain_task_notifications(
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> tuple[State, MessageParam | None, MessageParam | None]:
    """Pull all pending task notifications for this agent and fold them
    into `state.messages` as a synthetic user message. Returns
    `(new_state, drained_message_or_none, coordinator_context_or_none)`.
    The caller yields distinct query events so the consumer (QueryEngine) can
    persist both cross-turn in the exact order the model saw.

    Run at the TOP of each iteration, before the context pipeline — so
    compaction layers see the drained messages and can compress them
    with the rest of the conversation.

    Notifications arrive as attachments appended to the message list before
    the next model call and are also yielded so consumers can persist them.
    Scope is agent-filtered: main thread (`ctx.agent_id is None`) drains
    main-thread notifications; subagents drain their own.

    Not a `Layer` because Layer signature doesn't carry deps. The drain
    step is a distinct concern (event-pull, not message-rewrite) and
    keeping it as a dedicated inline step in `query()` is clearer than
    widening Layer's contract.
    """
    pending = deps.task_store.drain_notifications(agent_id=ctx.agent_id)
    message = _format_notifications_message(pending) if pending else None
    messages = [*state.messages]
    if message is not None:
        messages.append(message)

    coordinator_message = _record_and_render_coordinator_runtime(
        pending,
        state=state,
        config=config,
        deps=deps,
        ctx=ctx,
    )
    if coordinator_message is not None:
        messages.append(coordinator_message)

    if message is None and coordinator_message is None:
        return state, None, None
    return replace(state, messages=messages), message, coordinator_message


def _record_and_render_coordinator_runtime(
    notifications: Sequence[TaskNotification],
    *,
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> MessageParam | None:
    runtime = deps.coordinator_runtime
    if runtime is None:
        return None
    try:
        if notifications:
            runtime.record_task_notifications(tuple(notifications))
    except Exception as exc:
        _emit_coordinator_runtime_integration_failure(
            "record_task_notifications",
            exc,
            notification_count=len(notifications),
            state=state,
            config=config,
            deps=deps,
            ctx=ctx,
        )
        return None
    try:
        return runtime.render_context()
    except Exception as exc:
        _emit_coordinator_runtime_integration_failure(
            "render_context",
            exc,
            notification_count=len(notifications),
            state=state,
            config=config,
            deps=deps,
            ctx=ctx,
        )
        return None


def _emit_coordinator_runtime_integration_failure(
    operation: str,
    exc: Exception,
    *,
    notification_count: int,
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> None:
    _emit_observability(
        deps,
        "coordinator.runtime.integration_failed",
        _observability_context(config, ctx, state=state, source="coordinator"),
        {
            "operation": operation,
            "error_type": type(exc).__name__,
            "notification_count": notification_count,
            "iteration": state.iteration,
        },
    )


def _format_notifications_message(
    notifications: list[TaskNotification],
) -> MessageParam:
    """Fold one or more notifications into a single synthetic user
    message. One message per drain (not per notification) so the model
    sees a batched "here's what happened in the background" block.
    """
    lines: list[str] = ["[background task signals]"]
    for n in notifications:
        header = f"task={n.task_id} kind={n.kind}"
        if n.tool_use_id is not None:
            header += f" tool_use_id={n.tool_use_id}"
        lines.append(header)
        lines.append(n.message)
        lines.append("")  # blank line between entries
    return {"role": "user", "content": "\n".join(lines).rstrip()}


def _drain_local_agent_messages(
    state: State,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> tuple[State, tuple[MessageParam, ...]]:
    """Drain SendMessage prompts queued for the current local_agent.

    Dynamic import avoids a module cycle: `local_agent` imports QueryEngine for
    its driver, and QueryEngine imports this query module.
    """

    if ctx.agent_id is None:
        return state, ()
    try:
        from raygent_harness.core.tasks.local_agent import (
            drain_pending_messages_for_agent,
        )
    except Exception:
        return state, ()

    pending = drain_pending_messages_for_agent(deps.task_store, ctx.agent_id)
    if not pending:
        return state, ()
    messages = tuple(
        cast(
            "MessageParam",
            {
                "role": "user",
                "content": item.content,
                "raygentMessageKind": "local_agent_queued_message",
                "raygentLocalAgentMessage": {
                    "sender": item.sender,
                    "summary": item.summary,
                },
            },
        )
        for item in pending
    )
    return replace(state, messages=[*state.messages, *messages]), messages


def _start_memory_recall_prefetch(
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> MemoryRecallPrefetch | None:
    provider = deps.memory_recall_provider
    if provider is None:
        return None
    try:
        return provider.start(tuple(state.messages), config, ctx)
    except Exception:
        return None


async def _consume_memory_recall_if_ready(
    prefetch: MemoryRecallPrefetch | None,
    ctx: ToolUseContext,
    *,
    iteration: int,
) -> tuple[MessageParam, ...]:
    if prefetch is None:
        return ()
    if prefetch.consumed_on_iteration is not None:
        return ()
    if prefetch.settled_at is None:
        return ()
    try:
        return await prefetch.consume_if_ready(ctx=ctx, iteration=iteration)
    except asyncio.CancelledError:
        if ctx.abort_event.is_set():
            return ()
        raise
    except Exception:
        return ()


def _config_with_transient_context(
    config: QueryConfig,
    messages: Sequence[MessageParam],
) -> QueryConfig:
    """Return a per-request config carrying transient context messages."""

    if not messages:
        return config
    return replace(
        config,
        context_messages=(*config.context_messages, *messages),
    )


async def _collect_post_tool_transient_context(
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    read_cursor: int,
    already_attached_sources: set[str],
    state: State,
) -> tuple[tuple[MessageParam, ...], tuple[str, ...], int]:
    """Resolve non-persistent context from tool effects for later model calls."""

    read_paths = tuple(ctx.successful_text_read_paths[read_cursor:])
    next_cursor = len(ctx.successful_text_read_paths)
    if not deps.post_tool_context_providers:
        return (), (), next_cursor

    started_at = deps.clock.now()
    fragments: list[ContextFragment] = []
    failure_count = 0
    for provider in deps.post_tool_context_providers:
        try:
            provided = await provider(
                config,
                ctx,
                read_paths,
                tuple(already_attached_sources),
            )
        except Exception:
            failure_count += 1
            continue
        fragments.extend(provided)

    scoped = scope_context_fragments(fragments, agent_id=ctx.agent_id)
    filtered: list[ContextFragment] = []
    new_sources: list[str] = []
    seen = set(already_attached_sources)
    for fragment in scoped:
        if fragment.channel != "user_context" or not fragment.content.strip():
            continue
        source = _fragment_source_key(fragment)
        if source in seen:
            continue
        seen.add(source)
        filtered.append(fragment)
        new_sources.append(source)

    messages = render_user_context_messages(filtered)
    _emit_observability(
        deps,
        "context.post_tool.completed",
        _observability_context(config, ctx, state=state, source="context"),
        {
            "provider_count": len(deps.post_tool_context_providers),
            "read_path_count": len(read_paths),
            "fragment_count": len(fragments),
            "scoped_fragment_count": len(scoped),
            "attached_fragment_count": len(filtered),
            "message_count": len(messages),
            "failure_count": failure_count,
            "duration_s": deps.clock.now() - started_at,
        },
    )
    return messages, tuple(new_sources), next_cursor


def _fragment_source_key(fragment: ContextFragment) -> str:
    return fragment.source or fragment.id


async def _call_model(
    messages: list[MessageParam],
    model: str,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> ModelResponse:
    """Invoke the configured model provider.

    Single chokepoint per `why_not_patterns.md` — reliability/fallback policy
    wraps this function, not the loop body. `_model` is resolved by the
    loop via `_effective_model(config, state)` so the fallback-model rung
    takes effect without `_call_model` knowing about state at all.

    Returns a normalized response; recovery ladder decides what to do on error.
    """
    plan = await _build_model_call_plan(messages, model, config, deps, ctx)
    return await _complete_model_plan(
        plan,
        config,
        deps,
        ctx,
        request_mode="complete",
    )


async def _complete_model_plan(
    plan: _ModelCallPlan,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    request_mode: Literal["complete"],
) -> ModelResponse:
    """Invoke a prebuilt non-streaming model request with observability."""
    event_context = _observability_context(
        config,
        ctx,
        source="model",
        span_id=f"model-complete-{id(plan.request)}",
    )
    _emit_observability(
        deps,
        "model.request.started",
        event_context,
        _model_request_payload(plan.request, request_mode="complete"),
    )
    try:
        response = await deps.model_provider.complete(plan.request)
    except Exception as exc:
        provider_error = deps.model_provider.classify_error(exc)
        if provider_error.model_fallback is not None:
            _emit_observability(
                deps,
                "model.fallback.triggered",
                event_context,
                _model_fallback_observability_payload(provider_error.model_fallback),
            )
        _emit_observability(
            deps,
            "model.request.failed",
            event_context,
            {
                "model": plan.request.model,
                "request_mode": "complete",
                "error_type": type(exc).__name__,
                "provider_error": _provider_error_observability_payload(
                    provider_error
                ),
            },
        )
        raise
    _emit_observability(
        deps,
        "model.request.completed",
        event_context,
        {
            "model": plan.request.model,
            "request_mode": "complete",
            **_model_response_payload(response),
        },
    )
    return response


@dataclass(frozen=True)
class _ModelCallPlan:
    request: ModelRequest
    model_info: ModelInfo


@dataclass(frozen=True)
class _StreamingModelResponse:
    response: ModelResponse
    executor: StreamingToolExecutor | None = None
    buffered_tool_result_messages: tuple[MessageParam, ...] = ()
    abort_signaled_during_stream: bool = False


class _ProviderStreamEventError(RuntimeError):
    """Wrap provider-normalized stream control errors for the recovery ladder."""

    def __init__(self, provider_error: ProviderError) -> None:
        super().__init__(provider_error.message)
        self.provider_error = provider_error


async def _build_model_call_plan(
    messages: list[MessageParam],
    model: str,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> _ModelCallPlan:
    """Build the provider request once for complete or streaming paths."""
    permission_context = _permission_snapshot(deps, ctx)
    resolve_context = ModelResolveContext(
        permission_mode=permission_context.mode,
        query_source=ctx.query_source,
        agent_id=ctx.agent_id,
        effort=ctx.reasoning_effort_override,
    )
    if ctx.model_override is not None and model == ctx.model_override:
        resolved_model = resolve_skill_model_override(
            ctx.model_override,
            config.model,
            provider=deps.model_provider,
            context=resolve_context,
        )
    else:
        resolved_model = deps.model_provider.resolve_model(model, resolve_context)
    model_info = model_info_with_fallback(
        resolved_model,
        provider=deps.model_provider,
    )
    request_messages = [*config.context_messages, *messages]
    tool_specs = tuple(await _build_model_tool_specs(config.tools, ctx))
    effective_max_tokens = _resolve_model_max_tokens(
        requested_max_tokens=config.sampling.max_tokens,
        model=resolved_model,
        model_info=model_info,
    )
    request = ModelRequest(
        model=resolved_model,
        fallback_model=config.fallback_model,
        messages=tuple(
            api_message_from_message_param(message)
            for message in request_messages
        ),
        system_prompt=_append_model_system_context(
            config.system_prompt,
            config.context_system_prompt,
        ),
        tools=tool_specs,
        sampling=ModelSampling(
            max_tokens=effective_max_tokens,
            temperature=config.sampling.temperature,
            top_p=config.sampling.top_p,
            top_k=config.sampling.top_k,
            stop_sequences=config.sampling.stop_sequences,
        ),
        abort_event=ctx.abort_event,
        query_source=ctx.query_source,
        effort=ctx.reasoning_effort_override,
        agent_id=ctx.agent_id,
        permission_context=permission_context,
    )
    request = normalize_model_request_for_provider(request, model_info=model_info)
    normalized_request_messages = [
        message_param_from_api_message(message) for message in request.messages
    ]
    budget = await build_model_budget_snapshot(
        provider=deps.model_provider,
        requested_model=model,
        request=request,
        model_info=model_info,
        requested_max_tokens=config.sampling.max_tokens,
        messages=normalized_request_messages,
        fallback_estimator=_estimate_model_budget_tokens,
        observability=deps.observability,
        observability_context=_observability_context(
            config,
            ctx,
            source="model",
            span_id=f"model-budget-{id(request)}",
        ),
    )
    request = replace(request, budget=budget)
    return _ModelCallPlan(request=request, model_info=model_info)


def _append_model_system_context(base: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return base
    if not base:
        return addition
    return f"{base}\n\n{addition}"


def _streaming_tool_execution_enabled(
    config: QueryConfig,
    model_info: ModelInfo,
) -> bool:
    if config.experiments.get("fork_subagent", False) and any(
        tool.name in {"Agent", "Task"} for tool in config.tools
    ):
        # Fork AgentTool needs the complete assistant message so it can pair
        # every sibling tool_use with placeholder results. Streaming overlap
        # can only provide the completed prefix seen so far.
        return False
    return (
        config.experiments.get("streaming_tool_execution", False)
        and model_info.capabilities.supports_streaming
    )


async def _call_model_streaming(
    plan: _ModelCallPlan,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    effective_model: str,
) -> AsyncIterator[
    ModelStreamEventMessage
    | TombstoneMessage
    | ToolProgressEvent
    | ToolResultMessage
    | _StreamingModelResponse
]:
    """Consume provider stream events and assemble a final response.

    The headless query loop carries one final API-bound assistant message, so
    model-visible tool-result messages are buffered until that assistant event
    has been yielded. Tool progress can still surface during the model stream.
    """
    event_context = _observability_context(
        config,
        ctx,
        source="model",
        span_id=f"model-stream-{id(plan.request)}",
    )
    _emit_observability(
        deps,
        "model.request.started",
        event_context,
        _model_request_payload(plan.request, request_mode="stream"),
    )
    _emit_observability(
        deps,
        "model.stream.started",
        event_context,
        {
            "model": plan.request.model,
            "message_count": len(plan.request.messages),
            "tool_count": len(plan.request.tools),
        },
    )
    assembler = ModelStreamAssembler()
    response: ModelResponse | None = None
    streaming_executor: StreamingToolExecutor | None = _new_streaming_tool_executor(
        config,
        deps,
        ctx,
        effective_model=effective_model,
    )
    completed_blocks: dict[int, ModelContentBlock] = {}
    buffered_tool_result_messages: list[MessageParam] = []
    abort_signaled_during_stream = False
    failed_emitted = False

    try:
        async for stream_event in deps.model_provider.stream(plan.request):
            abort_signaled_during_stream = (
                abort_signaled_during_stream or ctx.abort_event.is_set()
            )
            payload = _model_stream_event_payload(stream_event)
            update = assembler.apply(stream_event)
            _emit_observability(
                deps,
                "model.stream.event",
                event_context,
                _model_stream_event_observability_payload(stream_event),
            )
            yield ModelStreamEventMessage(event=payload)

            if update.completed_block is not None:
                block_index = stream_event.identity.content_block_index
                if block_index is not None:
                    completed_blocks[block_index] = update.completed_block
                    if (
                        streaming_executor is not None
                        and isinstance(update.completed_block, ToolUseContentBlock)
                    ):
                        tool_use = ToolUseBlock(
                            id=update.completed_block.id,
                            name=update.completed_block.name,
                            input=thaw_json(update.completed_block.input),
                            index=block_index,
                        )
                        assistant_message = _assistant_message_from_stream_blocks(
                            completed_blocks,
                            message_id=stream_event.identity.message_id,
                        )
                        streaming_executor.add_tool(tool_use, assistant_message)

            if stream_event.type == "streaming_transport_fallback_started":
                fallback = update.streaming_transport_fallback
                if streaming_executor is not None:
                    streaming_executor.discard()
                streaming_executor = _new_streaming_tool_executor(
                    config,
                    deps,
                    ctx,
                    effective_model=effective_model,
                )
                buffered_tool_result_messages.clear()
                completed_blocks.clear()
                yield TombstoneMessage(
                    target_message_id=stream_event.identity.message_id,
                    reason=(
                        fallback.reason
                        if fallback is not None
                        else "streaming_transport_fallback_started"
                    ),
                    event=payload,
                )

            if (
                stream_event.type == "streaming_transport_fallback_completed"
                and update.response is not None
                and update.streaming_transport_fallback is not None
                and update.streaming_transport_fallback.replacement_response is not None
            ):
                # A provider-supplied replacement_response is already a complete
                # non-streaming response. Do not let the fresh post-fallback
                # executor make the query loop skip normal orchestration for its
                # tool_use blocks.
                if streaming_executor is not None:
                    streaming_executor.discard("streaming_transport_fallback")
                streaming_executor = None
                buffered_tool_result_messages.clear()
                completed_blocks.clear()

            if update.provider_error is not None:
                if streaming_executor is not None:
                    streaming_executor.discard("provider_error")
                    streaming_executor = None
                failed_emitted = True
                _emit_observability(
                    deps,
                    "model.request.failed",
                    event_context,
                    {
                        "model": plan.request.model,
                        "request_mode": "stream",
                        "provider_error": _provider_error_observability_payload(
                            update.provider_error
                        ),
                    },
                )
                raise _ProviderStreamEventError(update.provider_error)

            if update.model_fallback is not None:
                if streaming_executor is not None:
                    streaming_executor.discard("model_fallback")
                    streaming_executor = None
                failed_emitted = True
                provider_error = ProviderError(
                    kind="model_fallback_triggered",
                    message=update.model_fallback.reason,
                    model_fallback=update.model_fallback,
                    safe_to_fallback=True,
                )
                _emit_observability(
                    deps,
                    "model.fallback.triggered",
                    event_context,
                    _model_fallback_observability_payload(update.model_fallback),
                )
                _emit_observability(
                    deps,
                    "model.request.failed",
                    event_context,
                    {
                        "model": plan.request.model,
                        "request_mode": "stream",
                        "provider_error": _provider_error_observability_payload(
                            provider_error
                        ),
                    },
                )
                raise _ProviderStreamEventError(
                    provider_error
                )

            if streaming_executor is not None:
                async for tool_event in _drain_streaming_tool_completed(
                    streaming_executor,
                    emit_results=False,
                    buffered_tool_result_messages=buffered_tool_result_messages,
                ):
                    yield tool_event

            if update.response is not None:
                response = update.response
    except asyncio.CancelledError:
        if streaming_executor is not None:
            streaming_executor.discard("streaming_cancelled")
        if not failed_emitted:
            _emit_observability(
                deps,
                "model.request.failed",
                event_context,
                {
                    "model": plan.request.model,
                    "request_mode": "stream",
                    "error_type": "CancelledError",
                },
            )
        raise
    except Exception as exc:
        if streaming_executor is not None:
            streaming_executor.discard("streaming_error")
        if not failed_emitted:
            provider_error = deps.model_provider.classify_error(exc)
            if provider_error.model_fallback is not None:
                _emit_observability(
                    deps,
                    "model.fallback.triggered",
                    event_context,
                    _model_fallback_observability_payload(provider_error.model_fallback),
                )
            _emit_observability(
                deps,
                "model.request.failed",
                event_context,
                {
                    "model": plan.request.model,
                    "request_mode": "stream",
                    "error_type": type(exc).__name__,
                    "provider_error": _provider_error_observability_payload(
                        provider_error
                    ),
                },
            )
        raise

    if response is None:
        response = assembler.response()
    _emit_observability(
        deps,
        "model.stream.completed",
        event_context,
        {
            "model": plan.request.model,
            **_model_response_payload(response),
        },
    )
    _emit_observability(
        deps,
        "model.request.completed",
        event_context,
        {
            "model": plan.request.model,
            "request_mode": "stream",
            **_model_response_payload(response),
        },
    )
    yield _StreamingModelResponse(
        response=response,
        executor=streaming_executor,
        buffered_tool_result_messages=tuple(buffered_tool_result_messages),
        abort_signaled_during_stream=(
            abort_signaled_during_stream or ctx.abort_event.is_set()
        ),
    )


def _new_streaming_tool_executor(
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
    *,
    effective_model: str | None = None,
) -> StreamingToolExecutor:
    return StreamingToolExecutor(
        tools=config.tools,
        deps=deps,
        ctx=_with_tool_runtime(ctx, config, deps, effective_model=effective_model),
        max_concurrency=deps.max_tool_use_concurrency,
    )


def _with_tool_runtime(
    ctx: ToolUseContext,
    config: QueryConfig,
    deps: QueryDeps,
    *,
    effective_model: str | None = None,
) -> ToolUseContext:
    return replace(
        ctx,
        runtime=ToolRuntimeContext(
            config=config,
            deps=deps,
            effective_model=effective_model,
        ),
    )


def _model_stream_event_payload(event: ModelStreamEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event.type,
        "identity": {
            "message_id": event.identity.message_id,
            "content_block_index": event.identity.content_block_index,
            "provider_request_id": event.identity.provider_request_id,
            "attempt_id": event.identity.attempt_id,
        },
    }
    if event.block is not None:
        payload["block"] = _model_content_block_payload(event.block)
    if event.delta is not None:
        payload["delta"] = thaw_json(event.delta)
    if event.usage is not None:
        payload["usage"] = _usage_payload(event.usage)
    if event.stop_reason is not None:
        payload["stop_reason"] = event.stop_reason
    if event.provider_error is not None:
        payload["provider_error"] = _provider_error_payload(event.provider_error)
    if event.streaming_transport_fallback is not None:
        payload["streaming_transport_fallback"] = _streaming_fallback_payload(
            event.streaming_transport_fallback
        )
    if event.model_fallback is not None:
        payload["model_fallback"] = _model_fallback_payload(event.model_fallback)
    return payload


def _model_content_block_payload(block: ModelContentBlock) -> dict[str, Any]:
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseContentBlock):
        payload: dict[str, Any] = {
            "type": "server_tool_use" if block.provider_executed else "tool_use",
            "id": block.id,
            "name": block.name,
            "input": thaw_json(block.input),
        }
        if block.provider_executed:
            payload["provider_executed"] = True
        if block.provider_metadata is not None:
            payload["provider_metadata"] = thaw_json(block.provider_metadata)
        return payload
    if isinstance(block, ToolResultContentBlock):
        payload: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": thaw_json(block.content),
            "is_error": block.is_error,
        }
        if block.provider_metadata is not None:
            payload["provider_metadata"] = thaw_json(block.provider_metadata)
        return payload
    if isinstance(block, ThinkingContentBlock):
        payload: dict[str, Any] = {"type": "thinking", "text": block.text}
        if block.signature is not None:
            payload["signature"] = block.signature
        if block.redacted:
            payload["redacted"] = True
        if block.provider_metadata is not None:
            payload["provider_metadata"] = thaw_json(block.provider_metadata)
        return payload
    if isinstance(block, MediaContentBlock):
        payload = {
            "type": block.media_kind,
            "media_type": block.media_type,
            "data": thaw_json(block.data),
        }
        if block.provider_metadata is not None:
            payload["provider_metadata"] = thaw_json(block.provider_metadata)
        return payload
    return {
        "type": block.block_type,
        "payload": thaw_json(block.payload),
    }


def _usage_payload(usage: Usage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "cache_creation_input_tokens": int(usage.cache_creation_input_tokens),
        "cache_read_input_tokens": int(usage.cache_read_input_tokens),
    }
    if usage.reasoning_tokens:
        payload["reasoning_tokens"] = int(usage.reasoning_tokens)
    if usage.total_tokens is not None:
        payload["total_tokens"] = int(usage.total_tokens)
    if usage.provider_metadata:
        payload["provider_metadata"] = thaw_json(usage.provider_metadata)
    return payload


def _usage_observability_payload(usage: Usage) -> dict[str, object]:
    return {
        "input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "cache_creation_input_tokens": int(usage.cache_creation_input_tokens),
        "cache_read_input_tokens": int(usage.cache_read_input_tokens),
        "reasoning_tokens": int(usage.reasoning_tokens),
        "total_tokens": usage.total_tokens,
        "effective_total_tokens": usage.effective_total_tokens,
    }


def _observability_context(
    config: QueryConfig,
    ctx: ToolUseContext,
    *,
    state: State | None = None,
    source: str = "query",
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> KernelEventContext:
    base = ctx.observability_context or KernelEventContext(
        session_id=config.session_id or ctx.session_id,
        agent_id=ctx.agent_id or config.agent_id,
    )
    if state is not None:
        base = base.with_iteration(state.iteration)
    if source != base.source:
        base = base.with_source(source)
    if span_id is not None or parent_span_id is not None:
        base = replace(
            base,
            span_id=span_id if span_id is not None else base.span_id,
            parent_span_id=(
                parent_span_id if parent_span_id is not None else base.parent_span_id
            ),
        )
    return base


def _emit_observability(
    deps: QueryDeps,
    event_type: str,
    context: KernelEventContext,
    data: Mapping[str, object] | None = None,
) -> None:
    deps.observability.emit(event_type, context=context, data=data)


def _model_request_payload(
    request: ModelRequest,
    *,
    request_mode: Literal["stream", "complete"],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": request.model,
        "fallback_model": request.fallback_model,
        "request_mode": request_mode,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
        "tool_names": tuple(tool.name for tool in request.tools),
        "system_prompt_char_count": len(request.system_prompt),
        "max_tokens": request.sampling.max_tokens,
        "temperature": request.sampling.temperature,
        "top_p": request.sampling.top_p,
        "top_k": request.sampling.top_k,
        "stop_sequence_count": len(request.sampling.stop_sequences),
        "effort": request.effort,
        "query_source": request.query_source,
        "permission_mode": (
            request.permission_context.mode
            if request.permission_context is not None
            else None
        ),
    }
    if request.budget is not None:
        payload["budget"] = {
            "requested_model": request.budget.requested_model,
            "effective_model": request.budget.effective_model,
            "context_window": request.budget.context_window,
            "default_max_output_tokens": request.budget.default_max_output_tokens,
            "upper_max_output_tokens": request.budget.upper_max_output_tokens,
            "requested_max_tokens": request.budget.requested_max_tokens,
            "effective_max_tokens": request.budget.effective_max_tokens,
            "input_token_count": request.budget.input_token_count,
            "provider_input_token_count": request.budget.provider_input_token_count,
            "fallback_input_token_count": request.budget.fallback_input_token_count,
            "token_count_fallback_used": request.budget.token_count_fallback_used,
            "token_count_error_type": request.budget.token_count_error_type,
        }
    if request.media_budget is not None:
        payload["media_budget"] = {
            "max_media_items": request.media_budget.max_media_items,
            "original_media_items": request.media_budget.original_media_items,
            "retained_media_items": request.media_budget.retained_media_items,
            "stripped_media_items": request.media_budget.stripped_media_items,
            "top_level_media_items": request.media_budget.top_level_media_items,
            "nested_media_items": request.media_budget.nested_media_items,
            "mode": request.media_budget.mode,
        }
    return payload


def _model_response_payload(response: ModelResponse) -> dict[str, object]:
    return {
        "provider_request_id": response.provider_request_id,
        "stop_reason": response.stop_reason,
        "tool_use_count": len(response.tool_uses),
        "content_block_count": len(response.api_message.message.content),
        "usage": _usage_observability_payload(response.usage),
    }


def _provider_error_observability_payload(error: ProviderError) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": error.kind,
        "message_char_count": len(error.message),
        "retryable": error.retryable,
        "safe_to_fallback": error.safe_to_fallback,
    }
    if error.actual_tokens is not None:
        payload["actual_tokens"] = error.actual_tokens
    if error.limit_tokens is not None:
        payload["limit_tokens"] = error.limit_tokens
    if error.retry_after_s is not None:
        payload["retry_after_s"] = error.retry_after_s
    if error.status_code is not None:
        payload["status_code"] = error.status_code
    if error.model_fallback is not None:
        payload["model_fallback"] = _model_fallback_observability_payload(
            error.model_fallback
        )
    return payload


def _model_stream_event_observability_payload(
    event: ModelStreamEvent,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stream_event_type": event.type,
        "message_id": event.identity.message_id,
        "content_block_index": event.identity.content_block_index,
        "provider_request_id": event.identity.provider_request_id,
        "attempt_id": event.identity.attempt_id,
    }
    if event.block is not None:
        payload["block"] = _model_content_block_observability_payload(event.block)
    if event.delta is not None:
        payload["delta"] = _frozen_json_shape_payload(event.delta)
    if event.usage is not None:
        payload["usage"] = _usage_observability_payload(event.usage)
    if event.stop_reason is not None:
        payload["stop_reason"] = event.stop_reason
    if event.provider_error is not None:
        payload["provider_error"] = _provider_error_observability_payload(
            event.provider_error
        )
    if event.streaming_transport_fallback is not None:
        payload["streaming_transport_fallback"] = {
            "reason_char_count": len(event.streaming_transport_fallback.reason),
            "has_replacement_response": (
                event.streaming_transport_fallback.replacement_response is not None
            ),
        }
    if event.model_fallback is not None:
        payload["model_fallback"] = _model_fallback_observability_payload(
            event.model_fallback
        )
    return payload


def _model_content_block_observability_payload(
    block: ModelContentBlock,
) -> dict[str, object]:
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": redacted_payload(char_count=len(block.text))}
    if isinstance(block, ToolUseContentBlock):
        return {
            "type": "server_tool_use" if block.provider_executed else "tool_use",
            "id": block.id,
            "name": block.name,
            "provider_executed": block.provider_executed,
            "input": redacted_payload("tool_input_redacted"),
        }
    if isinstance(block, ToolResultContentBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "is_error": block.is_error,
            "content": redacted_payload("tool_result_content_redacted"),
        }
    if isinstance(block, ThinkingContentBlock):
        return {
            "type": "thinking",
            "thinking": redacted_payload("thinking_redacted", char_count=len(block.text)),
            "has_signature": block.signature is not None,
            "redacted": block.redacted,
        }
    if isinstance(block, MediaContentBlock):
        return {
            "type": block.media_kind,
            "media_type": block.media_type,
            "data": redacted_payload("media_data_redacted"),
        }
    return {
        "type": block.block_type,
        "payload": redacted_payload("unknown_block_payload_redacted"),
    }


def _frozen_json_shape_payload(value: FrozenJson) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            "json_kind": "object",
            "key_count": len(value),
            "has_keys": bool(value),
        }
    if isinstance(value, tuple):
        return {"json_kind": "array", "item_count": len(value)}
    if isinstance(value, str):
        return {"json_kind": "string", "char_count": len(value)}
    if isinstance(value, bool):
        return {"json_kind": "boolean"}
    if isinstance(value, int | float):
        return {"json_kind": "number"}
    return {"json_kind": "null"}


def _provider_error_payload(error: ProviderError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": error.kind,
        "message": error.message,
        "retryable": error.retryable,
        "safe_to_fallback": error.safe_to_fallback,
    }
    if error.raw_details is not None:
        payload["raw_details"] = error.raw_details
    if error.actual_tokens is not None:
        payload["actual_tokens"] = error.actual_tokens
    if error.limit_tokens is not None:
        payload["limit_tokens"] = error.limit_tokens
    if error.retry_after_s is not None:
        payload["retry_after_s"] = error.retry_after_s
    if error.status_code is not None:
        payload["status_code"] = error.status_code
    if error.model_fallback is not None:
        payload["model_fallback"] = _model_fallback_payload(error.model_fallback)
    return payload


def _streaming_fallback_payload(
    fallback: StreamingTransportFallbackControl,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"reason": fallback.reason}
    if fallback.replacement_response is not None:
        payload["replacement_provider_request_id"] = (
            fallback.replacement_response.provider_request_id
        )
        payload["replacement_stop_reason"] = fallback.replacement_response.stop_reason
    return payload


def _model_fallback_payload(fallback: ModelFallbackControl) -> dict[str, str]:
    return {
        "original_model": fallback.original_model,
        "fallback_model": fallback.fallback_model,
        "reason": fallback.reason,
    }


def _model_fallback_observability_payload(
    fallback: ModelFallbackControl,
) -> dict[str, object]:
    return {
        "original_model": fallback.original_model,
        "fallback_model": fallback.fallback_model,
        "reason": redacted_payload(
            "model_fallback_reason_redacted",
            char_count=len(fallback.reason),
        ),
    }


def _assistant_message_from_stream_blocks(
    completed_blocks: Mapping[int, ModelContentBlock],
    *,
    message_id: str | None,
) -> MessageParam:
    """Build the current API-bound assistant view for a streamed tool block."""
    blocks = tuple(completed_blocks[index] for index in sorted(completed_blocks))
    return message_param_from_model_message(
        ModelMessage(role="assistant", content=blocks, id=message_id)
    )


async def _drain_streaming_tool_completed(
    executor: StreamingToolExecutor,
    *,
    emit_results: bool,
    buffered_tool_result_messages: list[MessageParam] | None = None,
) -> AsyncIterator[ToolProgressEvent | ToolResultMessage]:
    """Map currently available streaming-tool updates to query events.

    Before the final assistant event has been yielded, result messages are
    buffered instead of emitted so QueryEngine never persists a user
    `tool_result` before its matching assistant `tool_use`.
    """
    async for update in executor.drain_completed():
        if isinstance(update, StreamingToolProgressUpdate):
            yield _tool_progress_event(update.progress)
            continue

        messages = _tool_result_messages_from_execution_result(update.result)
        if emit_results:
            for message in messages:
                yield ToolResultMessage(message=message)
        elif buffered_tool_result_messages is not None:
            buffered_tool_result_messages.extend(messages)


async def _drain_streaming_tool_remaining(
    executor: StreamingToolExecutor,
) -> AsyncIterator[ToolProgressEvent | ToolResultMessage]:
    """Drain the streaming executor after the final assistant message is visible."""
    async for update in executor.drain_remaining():
        if isinstance(update, StreamingToolProgressUpdate):
            yield _tool_progress_event(update.progress)
            continue
        for message in _tool_result_messages_from_execution_result(update.result):
            yield ToolResultMessage(message=message)


def _tool_progress_event(progress: ToolExecutionProgress) -> ToolProgressEvent:
    return ToolProgressEvent(
        tool_use_id=progress.tool_use_id,
        tool_name=progress.tool_name,
        message=progress.message,
        data=progress.data,
    )


def _tool_result_messages_from_execution_result(
    result: ToolExecutionResult,
) -> tuple[MessageParam, ...]:
    return (*result.pre_messages, result.message, *result.additional_messages)


def _streaming_tool_orchestration_complete(
    executor: StreamingToolExecutor,
) -> ToolOrchestrationComplete:
    return ToolOrchestrationComplete(
        tool_result_messages=executor.tool_result_messages,
        permission_denials=executor.permission_denials,
        updated_context=executor.updated_context,
        should_prevent_continuation=executor.should_prevent_continuation,
        prevent_reason=executor.prevent_reason,
        aborted=executor.updated_context.abort_event.is_set(),
        abort_reason=(
            "abort signaled during tool execution"
            if executor.updated_context.abort_event.is_set()
            else None
        ),
    )


def _resolve_model_max_tokens(
    *,
    requested_max_tokens: int,
    model: str,
    model_info: ModelInfo,
) -> int:
    """Use provider metadata for default max-output sizing.

    `SamplingParams.max_tokens` predates provider metadata and defaults to
    8192. Until config grows an explicit "unset" value, treat that legacy
    default as unspecified and let provider metadata choose the runtime default.
    Non-default caller values remain explicit overrides.
    """

    if requested_max_tokens != 8192:
        return requested_max_tokens
    if (
        model_info.max_output_tokens_default is None
        and model_info.max_output_tokens_upper_limit is None
    ):
        return requested_max_tokens
    limits = get_model_output_limits(model, model_info=model_info)
    return limits.default


def _estimate_model_budget_tokens(messages: list[MessageParam]) -> int:
    """Deterministic fallback for model-call budget snapshots.

    This mirrors the dependency-free estimator used by compaction without
    importing service-layer code into the query kernel.
    """

    total = 0
    for message in messages:
        total += 4
        total += _estimate_model_budget_value(message.get("content"))
    return total


def _estimate_model_budget_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, len(value) // 4)
    if isinstance(value, int | float | bool):
        return 1
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return sum(
            _estimate_model_budget_value(key) + _estimate_model_budget_value(item)
            for key, item in mapping.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[object], value)
        return sum(_estimate_model_budget_value(item) for item in sequence)
    return max(1, len(str(value)) // 4)


async def _orchestrate_tools(
    assistant_message: MessageParam,
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> AsyncIterator[QueryEvent]:
    """For each tool_use block: validate → check_permissions → call.

    Yields progress events. Produces the synthetic tool_result user message
    that gets appended to `state.messages` at the continue site.

    Permission denials are recorded as `PermissionDenial` in state for replay;
    tool exceptions are surfaced as error tool_results so the model can
    self-correct. See `reference_tasks_framework.md` for abort cascade rules.
    """
    tool_use_blocks = normalize_assistant_turn(assistant_message).tool_uses
    outcome: ToolOrchestrationOutcome | None = None

    async for event in run_tools(
        tool_uses=tool_use_blocks,
        assistant_message=assistant_message,
        tools=config.tools,
        deps=deps,
        ctx=_with_tool_runtime(
            ctx,
            config,
            deps,
            effective_model=_effective_model(config, state, ctx),
        ),
        max_concurrency=deps.max_tool_use_concurrency,
    ):
        if isinstance(event, ToolExecutionProgress):
            yield ToolProgressEvent(
                tool_use_id=event.tool_use_id,
                tool_name=event.tool_name,
                message=event.message,
                data=event.data,
            )
            continue
        if isinstance(event, ToolExecutionResult):
            for message in (*event.pre_messages, event.message, *event.additional_messages):
                yield ToolResultMessage(message=message)
            continue
        outcome = event

    if outcome is None:
        outcome = ToolOrchestrationOutcome(updated_context=ctx)

    yield ToolOrchestrationComplete(
        tool_result_messages=outcome.tool_result_messages,
        permission_denials=outcome.permission_denials,
        updated_context=outcome.updated_context,
        should_prevent_continuation=outcome.should_prevent_continuation,
        prevent_reason=outcome.prevent_reason,
        aborted=outcome.aborted,
        abort_reason=outcome.abort_reason,
    )


@dataclass(frozen=True)
class StopHookOutcome:
    """Result of running success-path stop hooks for one iteration.

    Three mutually exclusive outcomes — encoded so the loop body can't
    accidentally treat "prevent" as "block-only retry" or vice versa.

    - `terminal is not None` → hook vetoed the turn. Caller yields the
      `Terminal(reason="stop_hook_prevented", ...)` and returns.
    - `should_continue is True` → at least one hook returned `HookBlock`
      (no veto). Caller loops with `state` (which already includes the
      assistant message + the blocking messages).
    - Both False/None → clean completion. Caller yields
      `Terminal(reason="completed", ..., final_state=state)`.

    `state` always carries the assistant message that triggered the
    hook eval — the loop must build the terminal *and* the next-
    iteration carry-state from it, not from a pre-assistant snapshot.
    """

    state: State
    terminal: Terminal | None = None
    should_continue: bool = False
    blocking_messages: tuple[MessageParam, ...] = ()
    continuation_messages: tuple[MessageParam, ...] = ()


async def _evaluate_stop_hooks(
    state: State,
    messages_for_model: list[MessageParam],
    assistant_message: MessageParam,
    _config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> StopHookOutcome:
    """Run success-path stop hooks against the post-assistant transcript.

    The post-assistant transcript is `[*messages_for_model,
    assistant_message]` — i.e. the COMPACTED view the model just
    answered, not the pre-pipeline `state.messages`. If the context
    pipeline collapsed 10 messages to 1 summary, hooks must see the
    summary; the next-iteration carry-state must too — otherwise the
    next iteration re-feeds pre-compact history and compaction output
    is lost at the no-tool boundary.

    Outcomes:
      - hooks see `[*messages_for_model, assistant_message]`
      - prevent_continuation → `Terminal("stop_hook_prevented")`
      - blocking_errors only → continue with combined messages

    Failure-path hooks (`fire_on_failure`) are wired at the terminal
    emit site in the loop body, not here.
    """
    post_assistant_messages: list[MessageParam] = [
        *messages_for_model,
        assistant_message,
    ]
    state_with_assistant = replace(state, messages=post_assistant_messages)

    if not deps.stop_hooks:
        return StopHookOutcome(state=state_with_assistant)

    hook_ctx = HookContext(
        messages=list(post_assistant_messages),
        tool_use_context=ctx,
        phase="success",
    )
    hook_started_at = time.time()
    _emit_observability(
        deps,
        "hook.stop.started",
        _observability_context(_config, ctx, state=state_with_assistant, source="hook"),
        {
            "phase": "success",
            "hook_count": len(deps.stop_hooks),
            "message_count": len(post_assistant_messages),
        },
    )
    evaluation = await evaluate_on_success(deps.stop_hooks, hook_ctx)
    _emit_observability(
        deps,
        "hook.stop.completed",
        _observability_context(_config, ctx, state=state_with_assistant, source="hook"),
        {
            "phase": "success",
            "hook_count": len(deps.stop_hooks),
            "hooks_ran": evaluation.hooks_ran,
            "duration_ms": int((time.time() - hook_started_at) * 1000),
            "blocking_message_count": len(evaluation.blocking_messages),
            "continuation_message_count": len(evaluation.continuation_messages),
            "continuation_fragment_count": evaluation.continuation_fragment_count,
            "continuation_input_char_count": evaluation.continuation_input_char_count,
            "continuation_rendered_char_count": (
                evaluation.continuation_rendered_char_count
            ),
            "continuation_truncated_fragment_count": (
                evaluation.continuation_truncated_fragment_count
            ),
            "continuation_dropped_empty_fragment_count": (
                evaluation.continuation_dropped_empty_fragment_count
            ),
            "continuation_dropped_fragment_count": (
                evaluation.continuation_dropped_fragment_count
            ),
            "continuation_truncated_message_count": (
                evaluation.continuation_truncated_message_count
            ),
            "error_count": len(evaluation.errors),
            "prevent_continuation": evaluation.prevent_continuation,
            "prevent_reason_char_count": (
                len(evaluation.prevent_reason) if evaluation.prevent_reason else 0
            ),
        },
    )

    # Surface hook errors to the notification sink. Never escalates to
    # terminal — hook machinery failures should not fail the turn.
    for err in evaluation.errors:
        deps.notify(f"stop-hook error: {err}")

    # Prevent always wins, regardless of any blocking_messages also
    # produced this pass. Reference returns `stop_hook_prevented`
    if evaluation.prevent_continuation:
        # Still fold blocking_messages into final_state so observers see
        # what the hooks emitted, even though we're terminating.
        final_messages: list[MessageParam] = [
            *post_assistant_messages,
            *evaluation.blocking_messages,
            *evaluation.continuation_messages,
        ]
        prevented_state = replace(state, messages=final_messages)
        return StopHookOutcome(
            state=prevented_state,
            terminal=_build_terminal(
                "stop_hook_prevented",
                evaluation.prevent_reason,
                prevented_state,
            ),
            blocking_messages=tuple(evaluation.blocking_messages),
            continuation_messages=tuple(evaluation.continuation_messages),
        )

    # Block/context without prevent → continue with the appended messages so
    # the next iteration's model call sees the hook feedback/context.
    # Raygent's continuation-context extension deliberately reuses that
    # transcript-visible retry path.
    retry_messages = (
        *evaluation.blocking_messages,
        *evaluation.continuation_messages,
    )
    if retry_messages:
        new_state = replace(
            state,
            messages=[*post_assistant_messages, *retry_messages],
        )
        return StopHookOutcome(
            state=new_state,
            should_continue=True,
            blocking_messages=tuple(evaluation.blocking_messages),
            continuation_messages=tuple(evaluation.continuation_messages),
        )

    return StopHookOutcome(state=state_with_assistant)


ErrorClass = Literal[
    "transient",
    "context_overflow",
    "media_overflow",
    "model_fallback",
    "max_output_tokens",
    "unrecoverable",
]
"""Coarse bucketing of provider-normalized errors the recovery ladder sees."""


MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3
"""Ceiling on `max_output_tokens` recovery attempts per turn."""


MAX_OUTPUT_TOKENS_RECOVERY_MESSAGE = (
    "Output token limit hit. Resume directly - no apology, no recap of what you "
    "were doing. Pick up mid-thought if that is where the cut happened. Break "
    "remaining work into smaller pieces."
)
"""User-facing recovery instruction after a max-output-token interruption."""


def _classify_error(error: ProviderError) -> ErrorClass:
    """Map provider errors to query recovery rungs."""
    if error.kind == "model_fallback_triggered":
        return "model_fallback"
    if error.kind == "context_overflow":
        return "context_overflow"
    if error.kind == "media_overflow":
        return "media_overflow"
    if error.kind == "max_output_tokens":
        return "max_output_tokens"
    if error.kind in ("rate_limit", "server_overload", "transient"):
        return "transient"
    return "unrecoverable"


def _should_complete_with_api_error(provider_error: ProviderError) -> bool:
    """Return True for provider failures that are assistant-visible API errors.

    Provider errors such as rate-limit/auth/config can be materialized as
    assistant API-error messages and then exit the loop as `completed`,
    skipping success stop hooks. Context/media/max-output have dedicated
    recovery/terminal rungs, and true transient transport failures keep the
    one-shot retry path.
    """
    if provider_error.kind in (
        "context_overflow",
        "media_overflow",
        "max_output_tokens",
        "model_fallback_triggered",
        "user_abort",
        "transient",
    ):
        return False
    return provider_error.kind in ("rate_limit", "server_overload", "auth_config")


def _api_error_for_provider_error(
    provider_error: ProviderError,
) -> ModelApiErrorMessage | None:
    if provider_error.api_error is not None:
        return provider_error.api_error
    if provider_error.kind in ("model_fallback_triggered", "user_abort"):
        return None
    return build_model_api_error_message(
        kind=provider_error.kind,
        public_message=_api_error_public_message(provider_error),
        raw_details=provider_error.raw_details or provider_error.message,
        actual_tokens=provider_error.actual_tokens,
        limit_tokens=provider_error.limit_tokens,
        retry_after_s=provider_error.retry_after_s,
        status_code=provider_error.status_code,
    )


def _api_error_public_message(provider_error: ProviderError) -> str:
    if provider_error.kind == "context_overflow":
        return "Prompt is too long"
    return provider_error.message


def _append_api_error_to_state(
    state: State,
    provider_error: ProviderError,
    watermark: ErrorWatermark,
) -> State:
    api_error = _api_error_for_provider_error(provider_error)
    if api_error is None:
        return replace(state, error_watermark=watermark)
    message = message_param_from_model_api_error(api_error)
    return replace(
        state,
        messages=[*state.messages, message],
        error_watermark=watermark,
    )


def _max_output_tokens_recovery_message() -> MessageParam:
    return {"role": "user", "content": MAX_OUTPUT_TOKENS_RECOVERY_MESSAGE}


def _last_message_if_api_error(state: State) -> MessageParam | None:
    if not state.messages:
        return None
    message = state.messages[-1]
    return message if message.get("isApiErrorMessage") is True else None


async def _handle_error(
    error: Exception,
    state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> tuple[State, Terminal | None]:
    """Recovery ladder. Advances `state.error_watermark` one rung per call.

    Returns `(new_state, terminal_or_none)`:
      - `terminal is None` → caller `continue`s the loop with the new state.
        On the next iteration the loop retries, possibly with a mutated
        model (fallback) or rewritten history (context reduction, not yet
        wired).
      - `terminal is set` → caller yields `TerminalEvent(terminal)` and
        returns. Used when the ladder exhausts or the error class is
        unrecoverable.

    Watermark persists across iterations (see `ErrorWatermark` docstring).
    Reset at the successful-continue site in `query()`.
    """
    wm = state.error_watermark
    provider_error = (
        error.provider_error
        if isinstance(error, _ProviderStreamEventError)
        else deps.model_provider.classify_error(error)
    )
    err_msg = provider_error.message
    new_wm = replace(wm, last_error=err_msg)
    if provider_error.kind == "user_abort":
        aborted_state = replace(state, error_watermark=new_wm)
        return (
            aborted_state,
            _build_terminal("aborted_streaming", err_msg, aborted_state),
        )
    err_class = _classify_error(provider_error)
    recovery_context = _observability_context(
        config,
        ctx,
        state=state,
        source="recovery",
    )
    recovery_base_payload: dict[str, object] = {
        "error_class": err_class,
        "provider_error_kind": provider_error.kind,
        "error_message_char_count": len(err_msg),
        "message_count": len(state.messages),
    }

    if _should_complete_with_api_error(provider_error):
        api_error_state = _append_api_error_to_state(state, provider_error, new_wm)
        return (
            api_error_state,
            _build_terminal("completed", None, api_error_state),
        )

    # Rung 1: fallback-model swap. Highest priority because explicit fallback
    # triggers are recoverable by model swap, and the primary model usually will
    # not succeed on retry.
    fallback_model = (
        provider_error.model_fallback.fallback_model
        if provider_error.model_fallback is not None
        else config.fallback_model
    )
    if (
        err_class == "model_fallback"
        and fallback_model is not None
        and not wm.tried_fallback_model
    ):
        _emit_observability(
            deps,
            "recovery.rung.selected",
            recovery_context,
            {
                **recovery_base_payload,
                "rung": "fallback_model",
                "fallback_model": fallback_model,
            },
        )
        return (
            replace(
                state,
                error_watermark=replace(new_wm, tried_fallback_model=True),
                active_model=fallback_model,
            ),
            None,
        )

    # Rung 2a: media-specific retry. Reference treats image/PDF media-size
    # failures as distinct from prompt-too-long errors and can replace media
    # blocks with text markers before retrying. Do not consume the
    # prompt/context compaction rung for pure media rejection.
    if err_class == "media_overflow" and not wm.tried_media_downscope:
        _emit_observability(
            deps,
            "recovery.rung.selected",
            recovery_context,
            {
                **recovery_base_payload,
                "rung": "media_downscope",
            },
        )
        attempted_wm = replace(new_wm, tried_media_downscope=True)
        downscoped = downscope_message_params_for_media_retry(state.messages)
        if downscoped.changed and downscoped.snapshot is not None:
            _emit_observability(
                deps,
                "recovery.media_downscoped",
                recovery_context,
                {
                    **recovery_base_payload,
                    "original_media_items": (
                        downscoped.snapshot.original_media_items
                    ),
                    "retained_media_items": (
                        downscoped.snapshot.retained_media_items
                    ),
                    "stripped_media_items": (
                        downscoped.snapshot.stripped_media_items
                    ),
                },
            )
            return (
                replace(
                    state,
                    messages=list(downscoped.messages),
                    error_watermark=attempted_wm,
                ),
                None,
            )

        no_recovery_state = _append_api_error_to_state(
            state,
            provider_error,
            attempted_wm,
        )
        _emit_observability(
            deps,
            "recovery.exhausted",
            recovery_context,
            {
                **recovery_base_payload,
                "last_rung": "media_downscope",
                "terminal_reason": "image_error",
            },
        )
        return (
            no_recovery_state,
            _build_terminal("image_error", err_msg, no_recovery_state),
        )

    # Rung 2b: reactive context-reduction retry. Reference withholds the
    # prompt-too-long/media error, tries one reactive compaction, and either
    # retries from post-compact messages or surfaces prompt_too_long without
    if err_class == "context_overflow" and not wm.tried_reduce_context:
        _emit_observability(
            deps,
            "recovery.rung.selected",
            recovery_context,
            {
                **recovery_base_payload,
                "rung": "reduce_context",
            },
        )
        attempted_wm = replace(new_wm, tried_reduce_context=True)
        effective_config = (
            replace(config, model=state.active_model)
            if state.active_model is not None and state.active_model != config.model
            else config
        )
        compacted = await deps.reactive_compact(
            list(state.messages),
            state,
            effective_config,
            ctx,
        )
        if compacted is not None:
            from raygent_harness.services.compact.models import build_post_compact_messages

            compacted_state = replace(
                state,
                messages=build_post_compact_messages(compacted),
                compact_boundaries=(
                    *state.compact_boundaries,
                    _compact_boundary_from_event(compacted.boundary),
                ),
                error_watermark=attempted_wm,
                auto_compact_tracking=None,
            )
            return compacted_state, None

        no_recovery_state = _append_api_error_to_state(
            state,
            provider_error,
            attempted_wm,
        )
        terminal_reason: TerminalReason = "prompt_too_long"
        _emit_observability(
            deps,
            "recovery.exhausted",
            recovery_context,
            {
                **recovery_base_payload,
                "last_rung": "reduce_context",
                "terminal_reason": terminal_reason,
            },
        )
        return (no_recovery_state, _build_terminal(terminal_reason, err_msg, no_recovery_state))

    # Rung 3: max_output_tokens recovery. Reference injects a "you hit
    # max output, please wrap up" message and bumps a counter (`src/query.
    # ts:1188-1246`), allowing up to MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
    # attempts before terminating. Skeleton: advance
    # counter, don't inject. Real impl lands with `_call_model`.
    if (
        err_class == "max_output_tokens"
        and wm.max_output_tokens_recovery_count < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
    ):
        recovery_wm = replace(
            new_wm,
            max_output_tokens_recovery_count=(
                wm.max_output_tokens_recovery_count + 1
            ),
        )
        _emit_observability(
            deps,
            "recovery.rung.selected",
            recovery_context,
            {
                **recovery_base_payload,
                "rung": "max_output_tokens",
                "attempt_count": recovery_wm.max_output_tokens_recovery_count,
            },
        )
        state_with_api_error = _append_api_error_to_state(
            state,
            provider_error,
            recovery_wm,
        )
        return (
            replace(
                state_with_api_error,
                messages=[
                    *state_with_api_error.messages,
                    _max_output_tokens_recovery_message(),
                ],
            ),
            None,
        )

    # Rung 4: one harness-level retry for transient errors. Distinct from
    # a provider SDK's internal retry — this is loop-level, used when
    # we want another shot after an intra-loop state change (e.g., after
    # compaction reduced the context).
    if err_class == "transient" and not wm.tried_transient_retry:
        _emit_observability(
            deps,
            "recovery.rung.selected",
            recovery_context,
            {
                **recovery_base_payload,
                "rung": "transient_retry",
            },
        )
        return (
            replace(
                state,
                error_watermark=replace(new_wm, tried_transient_retry=True),
            ),
            None,
        )

    # Ladder exhausted → terminal. Precedence: classify by the CURRENT
    # error first; `fallback_exhausted` is a last-resort reason, not a
    # priority signal. A context-overflow after a prior fallback attempt
    # still surfaces as `prompt_too_long` — reference does the same at
    # fallback machinery is present).
    if err_class == "context_overflow":
        terminal_reason: TerminalReason = "prompt_too_long"
    elif err_class == "media_overflow":
        terminal_reason = "image_error"
    elif err_class == "max_output_tokens":
        terminal_reason = "completed"
    elif wm.tried_fallback_model:
        terminal_reason = "fallback_exhausted"
    else:
        terminal_reason = "model_error"

    new_state = _append_api_error_to_state(state, provider_error, new_wm)
    _emit_observability(
        deps,
        "recovery.exhausted",
        recovery_context,
        {
            **recovery_base_payload,
            "last_rung": (
                "fallback_model"
                if wm.tried_fallback_model
                else "max_output_tokens"
                if err_class == "max_output_tokens"
                else "media_downscope"
                if wm.tried_media_downscope
                else "transient_retry"
                if wm.tried_transient_retry
                else "none"
            ),
            "terminal_reason": terminal_reason,
        },
    )
    return (
        new_state,
        # Build the terminal from the SAME updated state we hand back, so
        # `Terminal.final_state.error_watermark.last_error` reflects the
        # error that drove termination. Building from `state` (pre-update)
        # leaves the watermark stale and breaks observability.
        _build_terminal(terminal_reason, err_msg, new_state),
    )




def _effective_model(config: QueryConfig, state: State, ctx: ToolUseContext) -> str:
    """Resolve which model `_call_model` should use this iteration.

    `state.active_model` wins if set (fallback rung activated), then
    tool-provided context model override (Skill), then `config.model`. Single
    function so the precedence rule is auditable.
    """
    if state.active_model:
        return state.active_model
    if ctx.model_override is not None:
        if ctx.model_override.strip() == "inherit":
            return config.model
        return ctx.model_override
    return config.model


def _add_response_usage(state: State, response: object) -> State:
    if not isinstance(response, ModelResponse):
        return state
    usage = response.usage
    if (
        usage.input_tokens == 0
        and usage.output_tokens == 0
        and usage.cache_creation_input_tokens == 0
        and usage.cache_read_input_tokens == 0
    ):
        return state
    return replace(
        state,
        usage=UsageTotals(
            input_tokens=state.usage.input_tokens + usage.input_tokens,
            output_tokens=state.usage.output_tokens + usage.output_tokens,
            cache_creation_input_tokens=(
                state.usage.cache_creation_input_tokens
                + usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                state.usage.cache_read_input_tokens + usage.cache_read_input_tokens
            ),
            cost_usd=state.usage.cost_usd,
        ),
    )


# ---------------------------------------------------------------------------
# query() — the outer agent-loop generator. Skeleton.
# ---------------------------------------------------------------------------


async def query(
    initial_state: State,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> AsyncGenerator[QueryEvent, None]:
    """Run one turn of the agent loop.

    Contract:
      - Yields `QueryEvent`s as they happen (stream start, messages, compaction).
      - **Last yielded event is always a `TerminalEvent`** carrying the
        `Terminal` for the turn. Every exit path goes through `_build_terminal()`
        and emits exactly one `TerminalEvent` before `return`.
      - On abort: yields a `TerminalEvent(Terminal(reason="aborted_*"))` and
        returns.

    The body is a `while True` with explicit continue-site state reassignment,
      1. Increment iteration counter. (Watermark is NOT reset here — it
         persists across iterations so persistent errors climb the ladder.)
      2. Run context pipeline (yield boundary events).
      3. Call model.
      4. Orchestrate tool calls (yield progress + tool_result message).
      5. Evaluate stop hooks.
      6. Either successful-continue (replace state.messages with the
         COMPACTED view + new history, AND reset the error watermark here
         since we made clean progress) or yield TerminalEvent and return.
    """
    state: State = initial_state

    # Turn-entry: budget + max-turns pre-check. These are cheap enough to run
    # before the first model call, and short-circuit the pathological case
    # where the caller passes max_turns=0.
    if config.budget.max_turns is not None and config.budget.max_turns <= 0:
        yield _terminal_event(
            _build_terminal("max_turns", "max_turns <= 0 at turn entry", state),
            deps,
            config,
            ctx,
        )
        return

    ctx = replace(
        ctx,
        runtime=ToolRuntimeContext(
            config=config,
            deps=deps,
            effective_model=_effective_model(config, state, ctx),
        ),
        successful_text_read_paths=[],
    )
    memory_recall_prefetch = _start_memory_recall_prefetch(state, config, deps, ctx)
    successful_text_read_cursor = 0
    transient_context_messages: tuple[MessageParam, ...] = ()
    attached_transient_context_sources: set[str] = set()

    try:
        while True:
            # Bump iteration only. Watermark persists across iterations — see
            # ErrorWatermark docstring. Reset is at the successful-continue
            # site below, once tool execution completes without raising.
            state = replace(state, iteration=state.iteration + 1)
            _emit_observability(
                deps,
                "query.iteration.started",
                _observability_context(config, ctx, state=state, source="query"),
                {
                    "iteration": state.iteration,
                    "message_count": len(state.messages),
                    "active_model": _effective_model(config, state, ctx),
                    "tool_count": len(config.tools),
                },
            )

            if ctx.abort_event.is_set():
                yield _terminal_event(
                    _build_terminal(
                        "aborted_streaming", "abort signaled at iteration top", state
                    ),
                    deps,
                    config,
                    ctx,
                )
                return

            yield StreamRequestStart(iteration=state.iteration)

            if (
                config.budget.max_turns is not None
                and state.iteration > config.budget.max_turns
            ):
                yield _terminal_event(
                    _build_terminal(
                        "max_turns", f"reached max_turns={config.budget.max_turns}", state
                    ),
                    deps,
                    config,
                    ctx,
                )
                return

            # Stage 0 — drain task notifications. Runs before the context
            # pipeline so compaction layers see the drained messages. Scope
            # is agent-filtered (ctx.agent_id): main thread drains main
            # notifications; subagents drain their own.
            # Yield as TaskNotificationsMessage so the consumer (QueryEngine)
            # persists drained content to the cross-turn message log; without
            # this, drained notifications would only live in per-turn state.
            state, drained_msg, coordinator_msg = _drain_task_notifications(
                state,
                config,
                deps,
                ctx,
            )
            if drained_msg is not None:
                yield TaskNotificationsMessage(message=drained_msg)
            if coordinator_msg is not None:
                yield CoordinatorRuntimeMessage(message=coordinator_msg)

            state, local_agent_messages = _drain_local_agent_messages(
                state,
                deps,
                ctx,
            )
            for local_agent_message in local_agent_messages:
                yield LocalAgentMessagesMessage(message=local_agent_message)

            ctx = cast(  # pyright: ignore[reportUnnecessaryCast]
                ToolUseContext,
                replace(
                    ctx,
                    runtime=ToolRuntimeContext(
                        config=config,
                        deps=deps,
                        effective_model=_effective_model(config, state, ctx),
                    ),
                ),
            )

            # Stage 1 — context pipeline. Produces messages_for_model, any
            # compact boundary events, and an optional autocompact tracking
            # update. Does not mutate state; the orchestrator appends
            # observed boundaries and applies the tracking update below.
            (
                messages_for_model,
                boundary_events,
                tracking_update,
                content_replacements,
            ) = await _run_context_pipeline(state, config, deps, ctx)
            ctx = cast(  # pyright: ignore[reportUnnecessaryCast]
                ToolUseContext,
                replace(
                    ctx,
                    messages=list(messages_for_model),
                    tools=tuple(config.tools),
                    runtime=ToolRuntimeContext(
                        config=config,
                        deps=deps,
                        effective_model=_effective_model(config, state, ctx),
                    ),
                ),
            )

            if content_replacements:
                yield ContentReplacementRecords(replacements=content_replacements)

            for ev in boundary_events:
                yield ev

            if boundary_events:
                for message in messages_for_model:
                    yield PostCompactMessage(message=message)

            if boundary_events:
                state = replace(
                    state,
                    compact_boundaries=(
                        *state.compact_boundaries,
                        *(
                            _compact_boundary_from_event(ev)
                            for ev in boundary_events
                        ),
                    ),
                )

            # Apply the autocompact tracking update so the next iteration's
            # autocompact layer sees the updated `compacted` flag and the
            # `consecutive_failures` circuit-breaker counter. Reference passes
            # it on State so recovery + iteration replay observe the same
            # source of truth.
            if tracking_update is not None:
                state = replace(state, auto_compact_tracking=tracking_update)

            # Stage 2 — model call. Model resolved through `_effective_model` so
            # the fallback-model rung takes effect here.
            model_config = _config_with_transient_context(
                config,
                transient_context_messages,
            )
            state_for_model = replace(state, messages=messages_for_model)
            streaming_response: _StreamingModelResponse | None = None
            try:
                response: ModelResponse
                if config.experiments.get("streaming_tool_execution", False):
                    plan = await _build_model_call_plan(
                        messages_for_model,
                        _effective_model(config, state, ctx),
                        model_config,
                        deps,
                        ctx,
                    )
                    if _streaming_tool_execution_enabled(model_config, plan.model_info):
                        async for ev in _call_model_streaming(
                            plan,
                            model_config,
                            deps,
                            ctx,
                            effective_model=_effective_model(config, state, ctx),
                        ):
                            if isinstance(ev, _StreamingModelResponse):
                                streaming_response = ev
                                continue
                            yield ev
                        if streaming_response is None:
                            raise RuntimeError("streaming model call produced no response")
                        response = streaming_response.response
                    else:
                        response = await _complete_model_plan(
                            plan,
                            model_config,
                            deps,
                            ctx,
                            request_mode="complete",
                        )
                else:
                    response = await _call_model(
                        messages_for_model,
                        _effective_model(config, state, ctx),
                        model_config,
                        deps,
                        ctx,
                    )
            except asyncio.CancelledError:
                if ctx.abort_event.is_set():
                    yield _terminal_event(
                        _build_terminal(
                            "aborted_streaming",
                            "abort signaled during model call",
                            state_for_model,
                        ),
                        deps,
                        config,
                        ctx,
                    )
                    return
                raise
            except Exception as err:
                previous_boundary_count = len(state_for_model.compact_boundaries)
                state, recovered_terminal = await _handle_error(
                    err, state_for_model, config, deps, ctx
                )
                for boundary in state.compact_boundaries[previous_boundary_count:]:
                    yield CompactBoundaryEvent(
                        kind=boundary.kind,
                        message_index=boundary.message_index,
                        summary=boundary.summary,
                    )
                if recovered_terminal is not None:
                    api_error_message = _last_message_if_api_error(state)
                    if api_error_message is not None:
                        yield AssistantMessage(message=api_error_message)
                    yield _terminal_event(recovered_terminal, deps, config, ctx)
                    return
                if len(state.compact_boundaries) > previous_boundary_count:
                    for message in state.messages:
                        yield PostCompactMessage(message=message)
                continue
            state_after_model = _add_response_usage(state_for_model, response)

            # Stage 3 — tool orchestration. If no tool_use blocks, the turn is done.
            assistant_message = _extract_assistant_message(response)
            yield AssistantMessage(
                message=_extract_observable_assistant_message(response),
                api_message=assistant_message,
            )
            streaming_abort_before_continuation = (
                streaming_response is not None
                and (
                    streaming_response.abort_signaled_during_stream
                    or ctx.abort_event.is_set()
                )
            )

            tool_use_blocks = _extract_tool_uses(response)
            if not tool_use_blocks:
                if streaming_abort_before_continuation:
                    aborted_state = replace(
                        state_after_model,
                        messages=[*messages_for_model, assistant_message],
                    )
                    yield _terminal_event(
                        _build_terminal(
                            "aborted_streaming",
                            "abort signaled during model stream",
                            aborted_state,
                        ),
                        deps,
                        config,
                        ctx,
                    )
                    return
                if assistant_message.get("isApiErrorMessage") is True:
                    api_error_state = replace(
                        state_after_model,
                        messages=[*messages_for_model, assistant_message],
                    )
                    yield _terminal_event(
                        _build_terminal("completed", None, api_error_state),
                        deps,
                        config,
                        ctx,
                    )
                    return

                # Stage 4a — stop hooks on clean completion. Hook context
                # AND the carry-forward state must be built from
                # `messages_for_model` (the COMPACTED view the model just
                # answered), not pre-pipeline `state.messages` — otherwise
                # compaction output is lost at the no-tool boundary.
                # Three outcomes:
                #   - terminal set        → veto, emit and return
                #   - should_continue     → block-only retry, loop
                #   - neither             → clean `completed`
                outcome = await _evaluate_stop_hooks(
                    state_after_model,
                    messages_for_model,
                    assistant_message,
                    config,
                    deps,
                    ctx,
                )
                for message in outcome.blocking_messages:
                    yield StopHookMessage(message=message)
                for message in outcome.continuation_messages:
                    yield StopHookMessage(message=message)
                if outcome.terminal is not None:
                    yield _terminal_event(outcome.terminal, deps, config, ctx)
                    return
                if outcome.should_continue:
                    state = replace(
                        outcome.state,
                        error_watermark=_reset_watermark_after_stop_hook_block(
                            outcome.state.error_watermark
                        ),
                    )
                    continue
                yield _terminal_event(
                    _build_terminal("completed", None, outcome.state),
                    deps,
                    config,
                    ctx,
                )
                return

            orchestration_complete: ToolOrchestrationComplete | None = None
            if streaming_response is not None and streaming_response.executor is not None:
                for message in streaming_response.buffered_tool_result_messages:
                    yield ToolResultMessage(message=message)
                async for ev in _drain_streaming_tool_remaining(
                    streaming_response.executor
                ):
                    yield ev
                orchestration_complete = _streaming_tool_orchestration_complete(
                    streaming_response.executor
                )
            else:
                async for ev in _orchestrate_tools(
                    assistant_message,
                    state_after_model,
                    config,
                    deps,
                    ctx,
                ):
                    if isinstance(ev, ToolOrchestrationComplete):
                        orchestration_complete = ev
                        continue
                    yield ev
            if orchestration_complete is None:
                raise RuntimeError("tool orchestration completed without outcome")
            if orchestration_complete.updated_context is not None:
                ctx = cast(ToolUseContext, orchestration_complete.updated_context)
            discovered_after_tools = _selected_tool_names_from_messages(
                orchestration_complete.tool_result_messages
            )
            if discovered_after_tools:
                ctx = cast(  # pyright: ignore[reportUnnecessaryCast]
                    ToolUseContext,
                    replace(
                        ctx,
                        discovered_tool_names=frozenset(
                            {*ctx.discovered_tool_names, *discovered_after_tools}
                        ),
                    ),
                )

            state_after_tools = replace(
                state_after_model,
                messages=[
                    *messages_for_model,
                    assistant_message,
                    *orchestration_complete.tool_result_messages,
                ],
                permission_denials=(
                    *state.permission_denials,
                    *orchestration_complete.permission_denials,
                ),
            )

            if orchestration_complete.aborted or ctx.abort_event.is_set():
                terminal_reason: TerminalReason = "aborted_tools"
                terminal_message = (
                    orchestration_complete.abort_reason
                    or "abort signaled during tool execution"
                )
                if streaming_abort_before_continuation:
                    terminal_reason = "aborted_streaming"
                    terminal_message = "abort signaled during model stream"
                yield _terminal_event(
                    _build_terminal(
                        terminal_reason,
                        terminal_message,
                        state_after_tools,
                    ),
                    deps,
                    config,
                    ctx,
                )
                return

            if orchestration_complete.should_prevent_continuation:
                yield _terminal_event(
                    _build_terminal(
                        "hook_stopped",
                        orchestration_complete.prevent_reason,
                        state_after_tools,
                    ),
                    deps,
                    config,
                    ctx,
                )
                return

            (
                new_transient_context_messages,
                new_transient_context_sources,
                successful_text_read_cursor,
            ) = await _collect_post_tool_transient_context(
                config,
                deps,
                ctx,
                read_cursor=successful_text_read_cursor,
                already_attached_sources=attached_transient_context_sources,
                state=state_after_tools,
            )
            if new_transient_context_messages:
                transient_context_messages = (
                    *transient_context_messages,
                    *new_transient_context_messages,
                )
                attached_transient_context_sources.update(
                    new_transient_context_sources
                )

            memory_recall_messages = await _consume_memory_recall_if_ready(
                memory_recall_prefetch,
                ctx,
                iteration=state_after_tools.iteration,
            )
            if memory_recall_messages:
                for message in memory_recall_messages:
                    yield MemoryRecallMessage(message=message)
                state_after_tools = replace(
                    state_after_tools,
                    messages=[*state_after_tools.messages, *memory_recall_messages],
                )

            # Successful continue site: model called, tools executed, no raise.
            #
            # Messages carried forward are the COMPACTED view (`messages_for_
            # model`), not the pre-compact `state.messages`. If the context
            # pipeline collapsed 10 messages to 1 summary, the next iteration
            # must see the summary — otherwise compaction output is lost on
            # the very next iteration and the loop re-compacts forever.
            #
            # Reset the recovery watermark here (not at the top of iteration)
            # so a persistent error across iterations climbs the ladder rather
            # than restarting at rung 1.
            #
            # Do NOT reset `active_model`. Once a fallback swap succeeds we
            # stay on fallback for the rest of the turn.
            state = replace(
                state_after_tools,
                error_watermark=ErrorWatermark(),
                auto_compact_tracking=_increment_auto_compact_turn_counter(
                    state_after_tools.auto_compact_tracking
                ),
            )

    finally:
        if memory_recall_prefetch is not None:
            memory_recall_prefetch.cancel()

# ---------------------------------------------------------------------------
# Helpers — thin shims to keep the loop body readable.
# ---------------------------------------------------------------------------


def _build_terminal(
    reason: TerminalReason,
    message: str | None,
    state: State,
) -> Terminal:
    """Build a `Terminal` for the current state. Every terminal — whether
    constructed in the loop body or returned from `_handle_error` /
    `_evaluate_stop_hooks` — should flow through here so `turn_count` and
    `final_state` stay consistent. Stage functions that already return a
    `Terminal` are expected to use this helper internally; the loop body
    does not re-wrap them, just forwards via `TerminalEvent`.
    """
    return Terminal(
        reason=reason,
        message=message,
        turn_count=state.iteration,
        final_state=replace(state, is_terminal=True),
    )


def _terminal_event(
    terminal: Terminal,
    deps: QueryDeps,
    config: QueryConfig,
    ctx: ToolUseContext,
) -> TerminalEvent:
    final_state = terminal.final_state
    _emit_observability(
        deps,
        "query.terminal",
        _observability_context(
            config,
            ctx,
            state=final_state,
            source="query",
        ),
        {
            "reason": terminal.reason,
            "message_char_count": len(terminal.message) if terminal.message else 0,
            "turn_count": terminal.turn_count,
            "is_error": terminal.reason != "completed",
            "final_message_count": (
                len(final_state.messages) if final_state is not None else 0
            ),
        },
    )
    return TerminalEvent(terminal=terminal)


async def _build_model_tool_specs(
    tools: Sequence[Tool],
    ctx: ToolUseContext,
) -> list[ModelToolSpec]:
    """Build provider-neutral schemas for currently model-visible tools."""
    selected_deferred_tool_names = {
        *ctx.discovered_tool_names,
        *_selected_tool_names_from_messages(ctx.messages),
    }
    specs: list[ModelToolSpec] = []
    for tool in tools:
        if not tool_visible_to_model(tool, selected_deferred_tool_names):
            continue
        try:
            description = await tool.prompt(ctx)
        except Exception:
            description = tool.description
        specs.append(
            ModelToolSpec(
                name=tool.name,
                description=description,
                input_schema=tool.input_schema or tool.input_model.model_json_schema(),
            )
        )
    return specs


def _permission_snapshot(
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> PermissionContextSnapshot:
    permission_context = deps.permission_context_for(ctx)
    return PermissionContextSnapshot(
        mode=permission_context.mode,
        always_allow_rules=_rules_to_json(permission_context.always_allow_rules),
        always_deny_rules=_rules_to_json(permission_context.always_deny_rules),
        always_ask_rules=_rules_to_json(permission_context.always_ask_rules),
        should_avoid_permission_prompts=permission_context.should_avoid_permission_prompts,
        is_bypass_permissions_mode_available=(
            permission_context.is_bypass_permissions_mode_available
        ),
        is_auto_mode_available=permission_context.is_auto_mode_available,
    )


def _rules_to_json(rules: Mapping[Any, Sequence[str]]) -> dict[str, FrozenJson]:
    return {str(source): tuple(values) for source, values in rules.items()}


def tool_visible_to_model(tool: Tool, selected_tool_names: set[str] | None = None) -> bool:
    """Model request visibility gate.

    Disabled tools and deferred tools are not sent to the model unless the tool
    explicitly opts into `always_load` or a prior ToolSearch result selected
    them. Raygent parses prior tool-reference result blocks to emulate the
    reference's client-side schema expansion path.
    """
    try:
        if not tool.is_enabled():
            return False
    except Exception:
        return False
    if selected_tool_names and (
        tool.name in selected_tool_names
        or any(alias in selected_tool_names for alias in tool.aliases)
    ):
        return True
    return tool.always_load or not tool.should_defer


def selected_tool_names_from_messages(messages: Sequence[MessageParam]) -> set[str]:
    from raygent_harness.tools.tool_search_tool import (
        selected_tool_names_from_messages as _selected_tool_names_from_messages_impl,
    )

    return _selected_tool_names_from_messages_impl(messages)


def _selected_tool_names_from_messages(messages: Sequence[MessageParam]) -> set[str]:
    """Backward-compatible private alias for older internal tests/helpers."""
    return selected_tool_names_from_messages(messages)


def _extract_assistant_message(response: Any) -> MessageParam:
    """Pull the API-bound assistant message out of a normalized response."""
    if isinstance(response, ModelResponse):
        return message_param_from_api_message(response.api_message)
    return normalize_assistant_turn(response).message


def _extract_observable_assistant_message(response: Any) -> MessageParam:
    """Pull the observer-visible assistant message for SDK/timeline yield.

    `AssistantMessage` events may be observable, but `State.messages` retains
    the API-bound message returned by `_extract_assistant_message`.
    """
    if isinstance(response, ModelResponse):
        return message_param_from_api_message(response.observable_message)
    return normalize_assistant_turn(response).message


def _extract_tool_uses(response: Any) -> list[ToolUseBlock]:
    """Filter normalized `tool_use` content blocks from the assistant response."""
    if isinstance(response, ModelResponse):
        return [
            ToolUseBlock(
                id=block.id,
                name=block.name,
                input=thaw_json(block.input),
                index=block.index,
            )
            for block in response.tool_uses
        ]
    return list(normalize_assistant_turn(response).tool_uses)


def _compact_boundary_from_event(event: CompactBoundaryEvent) -> CompactBoundary:
    """Shape-shim. `CompactBoundaryEvent` is what layers/consumers see on the
    wire; `CompactBoundary` (in state.py) is what gets persisted on State.
    Same data, different shape because the event carries a `type`
    discriminator for the yield union and the state record doesn't need it.

    The event-provided `message_index` is authoritative — the layer that
    produced the boundary computed it at the time of compaction, relative
    to the pre-compact messages being rewritten. The loop must not
    second-guess; using `len(state.messages)` here would break reactive
    compaction, where `state.messages` has already become the post-compact
    view by the time the boundary is persisted.
    """
    return CompactBoundary(
        message_index=event.message_index,
        kind=event.kind,
        summary=event.summary,
    )


def _increment_auto_compact_turn_counter(
    tracking: AutoCompactTrackingState | None,
) -> AutoCompactTrackingState | None:
    """Advance post-compact turn diagnostics at the successful continue site.

    Keeping this in the orchestrator, not the autocompact layer, avoids
    counting failed model calls or disabled no-op passes as successful
    post-compact turns.
    """
    if tracking is None or not tracking.compacted:
        return tracking
    return replace(tracking, turn_counter=tracking.turn_counter + 1)


def _reset_watermark_after_stop_hook_block(
    watermark: ErrorWatermark,
) -> ErrorWatermark:
    """Reset recovery counters for a clean assistant response, but preserve the
    reactive compaction death-spiral guard.

    Stop-hook blocking retries can still hit the same prompt-too-long condition
    after hooks inject more context. Successful tool-bearing continues reset the
    full watermark at their continue site.
    """
    return ErrorWatermark(tried_reduce_context=watermark.tried_reduce_context)


__all__ = [
    "AssistantMessage",
    "CompactBoundaryEvent",
    "ContentReplacementRecords",
    "CoordinatorRuntimeMessage",
    "Layer",
    "LayerResult",
    "LocalAgentMessagesMessage",
    "MemoryRecallMessage",
    "ModelStreamEventMessage",
    "PostCompactMessage",
    "QueryEvent",
    "StopHookMessage",
    "StopHookOutcome",
    "StreamRequestStart",
    "TaskNotificationsMessage",
    "Terminal",
    "TerminalEvent",
    "TerminalReason",
    "TombstoneMessage",
    "ToolProgressEvent",
    "ToolResultMessage",
    "query",
    "selected_tool_names_from_messages",
    "tool_visible_to_model",
]
