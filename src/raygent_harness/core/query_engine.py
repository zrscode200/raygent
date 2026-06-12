"""QueryEngine — conversation-scoped wrapper around per-turn `query()`.

One QueryEngine per conversation; each `submit_message()`
call is one turn. Cross-turn state (message log, usage totals, permission
denials, read-file cache) lives on the instance; per-turn state (iteration,
error watermark, compact boundaries) lives in the `State` threaded into
`query()`.

Shape contract:

- **One instance per conversation.** Constructing a new QueryEngine resets
  cross-turn state. If you need session-persistent history, hold the instance.
- **`submit_message()` yields `SDKMessage` events** as the turn unfolds and
  ends with a `SDKResult` message (success or typed error). The generator then
  returns normally. Callers typically `async for msg in engine.submit_message(...)`.
- **Every turn delegates to `query()`** and translates internal `QueryEvent`s
  into `SDKMessage`s via a switch on `type`. Non-translated events (progress,
  stream chunks, tool_use summaries) are either absorbed or surfaced based on
  the `SDKConfig` toggles.
- **Cross-turn mutation goes through `_append_messages()` / `_track_usage()`
  / `_track_denials()`.** Each is a narrow choke point so future persistence
  (transcript recording, usage telemetry, crash-recovery) wraps one function
  instead of being sprinkled through the yield-translation loop.

Scope skeleton — the terminal-result shape and the event-translation spine are
present; the wiring into `query()` (turn-entry State construction, terminal
decoding, ToolUseContext build) is stubbed with `NotImplementedError` until
the stage functions land (`_call_model`, `_orchestrate_tools`, etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from raygent_harness import __version__ as RAYGENT_VERSION
from raygent_harness.core.context_providers import (
    ContextFragment,
    render_system_context,
    render_user_context_messages,
    scope_context_fragments,
)
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.query import (
    AssistantMessage as _QueryAssistantMessage,
)
from raygent_harness.core.query import (
    CompactBoundaryEvent as _QueryCompactBoundary,
)
from raygent_harness.core.query import (
    ContentReplacementRecords as _QueryContentReplacementRecords,
)
from raygent_harness.core.query import (
    CoordinatorRuntimeMessage as _QueryCoordinatorRuntimeMessage,
)
from raygent_harness.core.query import (
    LocalAgentMessagesMessage as _QueryLocalAgentMessagesMessage,
)
from raygent_harness.core.query import (
    MemoryRecallMessage as _QueryMemoryRecallMessage,
)
from raygent_harness.core.query import (
    ModelStreamEventMessage as _QueryModelStreamEvent,
)
from raygent_harness.core.query import (
    PostCompactMessage as _QueryPostCompactMessage,
)
from raygent_harness.core.query import (
    StopHookMessage as _QueryStopHookMessage,
)
from raygent_harness.core.query import (
    StreamRequestStart as _QueryStreamStart,
)
from raygent_harness.core.query import (
    TaskNotificationsMessage as _QueryTaskNotificationsMessage,
)
from raygent_harness.core.query import (
    Terminal,
    TerminalReason,
    query,
    selected_tool_names_from_messages,
    tool_visible_to_model,
)
from raygent_harness.core.query import (
    TerminalEvent as _QueryTerminalEvent,
)
from raygent_harness.core.query import (
    TombstoneMessage as _QueryTombstone,
)
from raygent_harness.core.query import (
    ToolResultMessage as _QueryToolResultMessage,
)
from raygent_harness.core.state import (
    CompactBoundary,
    PermissionDenial,
    State,
    UsageTotals,
)
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    StreamEventEntry,
    TombstoneEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    content_replacement_state_from_replay,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import (
        AgentTriggerDecision,
        AgentTriggerMatch,
        AgentTriggerPolicySpec,
        QueryDeps,
    )
    from raygent_harness.core.tool import ToolUseContext
    from raygent_harness.services.transcript import SessionReplay
    from raygent_harness.skills.models import SkillDefinition

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK message types: what QueryEngine yields to external callers.
# ---------------------------------------------------------------------------


SDKResultSubtype = Literal[
    "success",
    "error_max_turns",
    "error_max_budget_usd",
    "error_during_execution",
    "error_aborted",
]
"""Terminal SDK result subtypes.

`error_aborted` is separate from generic execution failure because callers often
retry aborted turns differently from other errors.
"""


@dataclass(frozen=True)
class SDKSystemInit:
    """Emitted once per `submit_message()`, before the first model call.

    Carries the turn-entry config snapshot so SDK callers can correlate metrics.
    """

    type: Literal["system"] = "system"
    subtype: Literal["init"] = "init"
    session_id: str = ""
    model: str = ""
    permission_mode: str = "default"
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class SDKAssistantMessage:
    """One assistant message produced by the model."""

    type: Literal["assistant"] = "assistant"
    session_id: str = ""
    message: MessageParam = field(
        default_factory=lambda: {"role": "assistant", "content": ""}
    )  # type: ignore[arg-type]
    parent_tool_use_id: str | None = None


@dataclass(frozen=True)
class SDKUserMessage:
    """A user-role message (typically the synthetic tool_result message)."""

    type: Literal["user"] = "user"
    session_id: str = ""
    message: MessageParam = field(default_factory=lambda: {"role": "user", "content": ""})  # type: ignore[arg-type]
    parent_tool_use_id: str | None = None


@dataclass(frozen=True)
class SDKCompactBoundary:
    """Emitted when a compaction layer runs. Timeline-visible."""

    type: Literal["system"] = "system"
    subtype: Literal["compact_boundary"] = "compact_boundary"
    session_id: str = ""
    kind: Literal["microcompact", "autocompact", "context_collapse", "snip"] = "microcompact"
    summary: str = ""


@dataclass(frozen=True)
class SDKResult:
    """Terminal SDK message. Always the last yielded item per turn.

    """

    type: Literal["result"] = "result"
    subtype: SDKResultSubtype = "success"
    session_id: str = ""
    is_error: bool = False
    num_turns: int = 0
    result: str = ""
    """Final assistant text content. Empty on error subtypes."""

    usage: UsageTotals = field(default_factory=UsageTotals)
    permission_denials: tuple[PermissionDenial, ...] = ()
    errors: tuple[str, ...] = ()
    """Human-readable error messages when `subtype` is an error variant."""


SDKMessage = (
    SDKSystemInit
    | SDKAssistantMessage
    | SDKUserMessage
    | SDKCompactBoundary
    | SDKResult
)


# ---------------------------------------------------------------------------
# QueryEngine — the conversation container.
# ---------------------------------------------------------------------------


class QueryEngine:
    """One instance per conversation. Each `submit_message()` = one turn.

    Cross-turn state persists here. Per-turn state is built at `submit_message()`
    entry and threaded into `query()` as a fresh `State`.
    """

    def __init__(
        self,
        config: QueryConfig,
        deps: QueryDeps,
        ctx: ToolUseContext,
        transcript_scope: TranscriptScope | None = None,
    ) -> None:
        self._config: QueryConfig = config
        self._deps: QueryDeps = deps
        self._ctx: ToolUseContext = ctx
        self._transcript_scope: TranscriptScope | None = (
            transcript_scope
            if transcript_scope is not None
            else _default_transcript_scope(config, ctx)
        )
        self._transcript_parent_entry_id: str | None = None

        self._messages: list[MessageParam] = []
        """The conversation-wide message log. Mutated across turns, never
        re-ordered. Each turn appends the user prompt, then the query()
        generator appends assistant/tool_result messages as they land."""

        self._compact_boundaries: tuple[CompactBoundary, ...] = ()
        """Conversation-wide compact boundary history.

        Raygent keeps compact boundaries outside API-visible messages, unlike
        in-band compact-boundary marker. Preserve them separately across
        `submit_message()` turns so future layers/resume code can still reason
        about the latest compact point.
        """

        self._total_usage: UsageTotals = UsageTotals()
        """Accumulated across all turns. Surfaced in every SDKResult."""

        self._permission_denials: list[PermissionDenial] = []
        """Appended when a tool-permission gate says 'deny'. Replayed in the
        next turn's prompt context by the context pipeline."""

        self._discovered_tool_names: set[str] = set()
        """Deferred tool schemas discovered through ToolSearch results.

        Reference carries this through compact boundary metadata. Raygent keeps
        compaction boundaries out-of-band, so QueryEngine persists the set
        alongside the cross-turn transcript.
        """

        self._turn_index: int = 0
        """Number of completed turns. Used for telemetry and resume."""

        self._memory_extraction_tasks: set[asyncio.Task[None]] = set()
        """Fire-and-forget memory extraction tasks scheduled after completed
        turns. Kept strongly referenced until done so background work is not
        garbage-collected before it settles."""

    @classmethod
    def from_replay(
        cls,
        config: QueryConfig,
        deps: QueryDeps,
        ctx: ToolUseContext,
        replay: SessionReplay,
        *,
        transcript_scope: TranscriptScope | None = None,
    ) -> QueryEngine:
        """Construct a QueryEngine from a loaded transcript replay.

        Rebuilds API-visible messages, compact boundaries, content-replacement
        cache state, and deferred-tool discovery. Usage totals and product UI
        metadata intentionally remain outside replay scope.
        """

        scope = transcript_scope or TranscriptScope(
            session_id=replay.session_id,
            agent_id=replay.agent_id,
            is_sidechain=replay.is_sidechain,
            runtime_session_id=replay.runtime_session_id,
        )
        replay_ctx = ctx
        if replay_ctx.content_replacement is not None:
            replay_ctx = replace(
                replay_ctx,
                content_replacement=content_replacement_state_from_replay(
                    replay,
                    max_result_size_chars=(
                        replay_ctx.content_replacement.max_result_size_chars
                    ),
                    replaced_outputs_dir=(
                        replay_ctx.content_replacement.replaced_outputs_dir
                    ),
                ),
            )
        engine = cls(config, deps, replay_ctx, transcript_scope=scope)
        engine._messages = list(replay.messages)
        engine._compact_boundaries = replay.compact_boundaries
        engine._transcript_parent_entry_id = replay.last_message_entry_id
        engine._track_discovered_tool_names(*engine._messages)
        engine._emit_transcript_observability(
            "transcript.replay.completed",
            {
                "message_count": len(replay.messages),
                "compact_boundary_count": len(replay.compact_boundaries),
                "content_replacement_count": len(replay.content_replacements),
                "warning_count": len(replay.warnings),
                "is_sidechain": replay.is_sidechain,
                "agent_id_present": replay.agent_id is not None,
            },
        )
        return engine

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    @property
    def transcript_scope(self) -> TranscriptScope | None:
        """Transcript scope used by this conversation engine, when enabled."""

        return self._transcript_scope

    async def seed_messages(self, messages: Sequence[MessageParam]) -> None:
        """Seed conversation history before the next `submit_message()` turn.

        Synchronous child-query paths need the reference shape where fork
        context messages and the child prompt are passed as one initial message
        array to a single query loop. This method records the pre-prompt
        messages without creating additional model turns.
        """

        for message in messages:
            self._append_message(message)
            await self._record_transcript_message(message)

    async def submit_message(
        self,
        prompt: str | MessageParam,
    ) -> AsyncGenerator[SDKMessage, None]:
        """Run one turn. Yields SDK messages; ends with a terminal SDKResult.

          1. Clear turn-scoped per-instance state (discovered skills, etc.).
          2. Build/extend the system prompt and non-persistent context messages.
          3. Append the user prompt to `self._messages`.
          4. Yield `SDKSystemInit` announcing the turn.
          5. Build per-turn `State` with current message log.
          6. Delegate to `query()` and translate each `QueryEvent` to an
             `SDKMessage`, appending assistant/user messages to our log.
          7. On generator completion, inspect `Terminal` and emit `SDKResult`.
        """
        self._turn_index += 1
        event_context = self._observability_context(
            turn_id=f"turn-{self._turn_index}",
            source="query_engine",
        )
        self._emit_observability(
            "query.turn.started",
            event_context,
            {
                "turn_index": self._turn_index,
                "history_message_count": len(self._messages),
                "configured_model": self._config.model,
                "configured_tool_count": len(self._config.tools),
                "prompt_kind": "message" if isinstance(prompt, dict) else "text",
            },
        )

        pre_user_ctx = replace(
            self._ctx,
            messages=list(self._messages),
            permission_context=self._deps.permission_context,
            discovered_tool_names=frozenset(self._discovered_tool_names),
            observability_context=event_context,
        )
        self._emit_observability(
            "query.turn.surface.started",
            event_context,
            {
                "turn_index": self._turn_index,
                "base_tool_count": len(self._config.tools),
                "context_provider_count": len(self._deps.context_providers),
                "has_skill_provider": self._deps.skill_provider is not None,
                "has_tool_catalog_provider": (
                    self._deps.tool_catalog_provider is not None
                ),
                "has_system_prompt_provider": (
                    self._deps.system_prompt_provider is not None
                ),
                "has_memory_prompt_provider": (
                    self._deps.memory_prompt_provider is not None
                ),
            },
        )
        turn_config = await self._build_turn_config(pre_user_ctx, event_context)
        self._emit_observability(
            "query.turn.surface.completed",
            event_context,
            {
                "turn_index": self._turn_index,
                "model": turn_config.model,
                "tool_count": len(turn_config.tools),
                "tool_names": tuple(tool.name for tool in turn_config.tools),
                "system_prompt_char_count": len(turn_config.system_prompt),
                "context_system_prompt_char_count": len(
                    turn_config.context_system_prompt
                ),
                "context_message_count": len(turn_config.context_messages),
            },
        )
        turn_ctx = replace(
            pre_user_ctx,
            rendered_system_prompt=turn_config.system_prompt,
            tools=turn_config.tools,
            observability_context=event_context,
        )

        history_before_prompt = list(self._messages)
        user_msg = _coerce_user_message(prompt)
        self._append_message(user_msg)
        await self._record_transcript_message(user_msg)

        yield SDKSystemInit(
            session_id=turn_config.session_id,
            model=turn_config.model,
            permission_mode=self._deps.permission_context.mode,
            tools=tuple(t.name for t in turn_config.tools),
        )

        trigger_messages = await self._evaluate_agent_trigger_policy(
            prompt=user_msg,
            history=history_before_prompt,
            config=turn_config,
            ctx=replace(turn_ctx, messages=list(self._messages)),
            event_context=event_context,
        )
        for trigger_message in trigger_messages:
            self._append_message(trigger_message)
            await self._record_transcript_message(trigger_message)
            yield SDKUserMessage(
                session_id=turn_config.session_id,
                message=trigger_message,
            )

        state = State(
            messages=list(self._messages),
            compact_boundaries=self._compact_boundaries,
            permission_denials=tuple(self._permission_denials),
        )
        terminal: Terminal | None = None
        final_turn_count = state.iteration

        try:
            async with contextlib.aclosing(
                query(state, turn_config, self._deps, turn_ctx)
            ) as query_gen:
                async for event in query_gen:
                    # TerminalEvent is always the last event from query().
                    # Capture the Terminal and break — any further events would
                    # indicate an upstream contract violation. `aclosing`
                    # ensures query() finalizers still run on this break.
                    if isinstance(event, _QueryTerminalEvent):
                        terminal = event.terminal
                        final_turn_count = terminal.turn_count
                        self._reconcile_terminal_state(terminal)
                        self._schedule_memory_extraction(
                            terminal,
                            turn_ctx,
                            turn_config,
                        )
                        break
                    translated = await self._translate_event(event)
                    if translated is None:
                        continue
                    yield translated
        except GeneratorExit:
            # Caller closed the generator — honor it without emitting a terminal.
            raise
        except Exception as err:
            await self._flush_transcript()
            result = self._build_error_result(
                "error_during_execution",
                [str(err)],
                turn_count=state.iteration,
            )
            self._emit_observability(
                "query.turn.failed",
                event_context,
                {
                    "turn_index": self._turn_index,
                    "error_type": type(err).__name__,
                    "error_message_char_count": len(str(err)),
                    "turn_count": state.iteration,
                },
            )
            yield result
            return

        await self._flush_transcript()
        result = self._build_terminal_result(terminal, turn_count=final_turn_count)
        if result.is_error:
            self._emit_observability(
                "query.turn.failed",
                event_context,
                {
                    "turn_index": self._turn_index,
                    "terminal_reason": terminal.reason if terminal is not None else None,
                    "turn_count": final_turn_count,
                    "sdk_subtype": result.subtype,
                    "error_count": len(result.errors),
                },
            )
        else:
            self._emit_observability(
                "query.turn.completed",
                event_context,
                {
                    "turn_index": self._turn_index,
                    "terminal_reason": terminal.reason if terminal is not None else None,
                    "turn_count": final_turn_count,
                    "sdk_subtype": result.subtype,
                    "output_char_count": len(result.result),
                    "usage": _usage_totals_payload(result.usage),
                },
            )
        yield result

    async def _build_turn_config(
        self,
        ctx: ToolUseContext,
        event_context: KernelEventContext | None = None,
    ) -> QueryConfig:
        """Build per-turn config with optional tooling/memory seams applied.

        Reference assembles the system prompt inside QueryEngine and appends
        `loadMemoryPrompt()` output when memory is explicitly enabled
        injectable through `QueryDeps` so `core` stays independent of `memdir`.

        Skill loading also loads optional skills and lets an injected tool
        catalog provider expand/rewrite `config.tools` for this turn. Defaults
        are all no-op, so existing callers keep the exact frozen config shape.
        """
        if event_context is None:
            event_context = self._observability_context(
                turn_id=f"turn-{self._turn_index or 1}",
                source="query_engine",
            )
        turn_config = self._config
        turn_skills = await self._load_turn_skills(turn_config, ctx, event_context)
        turn_config = await self._build_tool_catalog(
            turn_config,
            turn_skills,
            ctx,
            event_context,
        )

        turn_config = await self._apply_context_providers(
            turn_config,
            ctx,
            event_context,
        )
        turn_config = await self._apply_system_prompt_provider(
            turn_config,
            ctx,
            event_context,
        )
        return await self._apply_memory_prompt_provider(
            turn_config,
            ctx,
            event_context,
        )

    async def _evaluate_agent_trigger_policy(
        self,
        *,
        prompt: MessageParam,
        history: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> tuple[MessageParam, ...]:
        """Evaluate optional agent-trigger policy and render bounded guidance."""

        spec = self._deps.agent_trigger_policy
        if spec is None:
            self._emit_agent_trigger_completed(
                event_context,
                provider_configured=False,
                evaluated=False,
            )
            return ()

        if not _agent_trigger_scope_allows(spec, ctx.agent_id):
            self._emit_agent_trigger_completed(
                event_context,
                provider_configured=True,
                evaluated=False,
                skipped_scope=True,
                scope=spec.agent_scope,
            )
            return ()

        started_at = self._deps.clock.now()
        self._emit_observability(
            "agent_trigger.policy.started",
            event_context,
            {
                "provider_configured": True,
                "scope": spec.agent_scope,
                "agent_id_present": ctx.agent_id is not None,
                "delegation_tool_available": _agent_delegation_tool_available(
                    config,
                    ctx,
                ),
                "history_message_count": len(history),
                "prompt_char_count": _message_content_char_count(prompt),
                "max_messages": max(0, spec.max_messages),
                "max_message_chars": max(0, spec.max_message_chars),
            },
        )
        try:
            decision = await spec.policy(prompt, history, config, ctx)
        except Exception as exc:
            _LOGGER.debug(
                "agent trigger policy failed: error_type=%s",
                type(exc).__name__,
            )
            self._emit_agent_trigger_completed(
                event_context,
                provider_configured=True,
                evaluated=True,
                failed=True,
                scope=spec.agent_scope,
                error_type=type(exc).__name__,
                duration_s=self._deps.clock.now() - started_at,
            )
            return ()

        delegation_tool_available = _agent_delegation_tool_available(config, ctx)
        rendered = _render_agent_trigger_messages(
            decision,
            spec,
            delegation_tool_available=delegation_tool_available,
        )
        self._emit_agent_trigger_completed(
            event_context,
            provider_configured=True,
            evaluated=True,
            scope=spec.agent_scope,
            delegation_tool_available=delegation_tool_available,
            match_count=len(decision.matches),
            injected_message_count=len(rendered.messages),
            input_char_count=rendered.input_char_count,
            rendered_char_count=rendered.rendered_char_count,
            truncated_message_count=rendered.truncated_message_count,
            dropped_message_count=rendered.dropped_message_count,
            suppress_main_turn_requested=decision.suppress_main_turn,
            suppress_main_turn_applied=False,
            duration_s=self._deps.clock.now() - started_at,
        )
        return rendered.messages

    def _emit_agent_trigger_completed(
        self,
        event_context: KernelEventContext,
        *,
        provider_configured: bool,
        evaluated: bool,
        failed: bool = False,
        skipped_scope: bool = False,
        scope: str | None = None,
        delegation_tool_available: bool = False,
        match_count: int = 0,
        injected_message_count: int = 0,
        input_char_count: int = 0,
        rendered_char_count: int = 0,
        truncated_message_count: int = 0,
        dropped_message_count: int = 0,
        suppress_main_turn_requested: bool = False,
        suppress_main_turn_applied: bool = False,
        error_type: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        data: dict[str, object] = {
            "provider_configured": provider_configured,
            "evaluated": evaluated,
            "failed": failed,
            "skipped_scope": skipped_scope,
            "delegation_tool_available": delegation_tool_available,
            "match_count": match_count,
            "injected_message_count": injected_message_count,
            "input_char_count": input_char_count,
            "rendered_char_count": rendered_char_count,
            "truncated_message_count": truncated_message_count,
            "dropped_message_count": dropped_message_count,
            "suppress_main_turn_requested": suppress_main_turn_requested,
            "suppress_main_turn_applied": suppress_main_turn_applied,
        }
        if scope is not None:
            data["scope"] = scope
        if error_type is not None:
            data["error_type"] = error_type
        if duration_s is not None:
            data["duration_s"] = duration_s
        self._emit_observability(
            "agent_trigger.policy.completed",
            event_context,
            data,
        )

    async def _apply_context_providers(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> QueryConfig:
        """Resolve typed context fragments for this submitted turn."""

        if not self._deps.context_providers:
            self._emit_observability(
                "context.providers.completed",
                event_context,
                {
                    "provider_count": 0,
                    "fragment_count": 0,
                    "system_context_char_count": 0,
                    "user_context_message_count": 0,
                    "failure_count": 0,
                },
            )
            return config

        started_at = self._deps.clock.now()
        self._emit_observability(
            "context.providers.started",
            event_context,
            {"provider_count": len(self._deps.context_providers)},
        )
        fragments: list[ContextFragment] = []
        failure_count = 0
        for provider in self._deps.context_providers:
            try:
                provided = await provider(config, ctx)
            except Exception as exc:
                failure_count += 1
                _LOGGER.debug("context provider failed: %s", exc)
                continue
            fragments.extend(provided)

        scoped = scope_context_fragments(fragments, agent_id=ctx.agent_id)
        system_context = render_system_context(scoped)
        user_context_messages = render_user_context_messages(scoped)
        self._emit_observability(
            "context.providers.completed",
            event_context,
            {
                "provider_count": len(self._deps.context_providers),
                "fragment_count": len(fragments),
                "scoped_fragment_count": len(scoped),
                "system_context_char_count": (
                    len(system_context) if system_context is not None else 0
                ),
                "user_context_message_count": len(user_context_messages),
                "failure_count": failure_count,
                "duration_s": self._deps.clock.now() - started_at,
            },
        )

        if system_context is None and not user_context_messages:
            return config

        return replace(
            config,
            context_system_prompt=(
                _append_system_prompt(config.context_system_prompt, system_context)
                if system_context is not None
                else config.context_system_prompt
            ),
            context_messages=(*config.context_messages, *user_context_messages),
        )

    async def _apply_system_prompt_provider(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> QueryConfig:
        """Append optional non-memory prompt fragments to this turn config."""
        provider = self._deps.system_prompt_provider
        if provider is None:
            self._emit_observability(
                "system_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": False,
                    "added_char_count": 0,
                    "failed": False,
                },
            )
            return config
        started_at = self._deps.clock.now()
        try:
            addition = await provider(config, ctx)
        except Exception as exc:
            _LOGGER.debug("system prompt provider failed: %s", exc)
            self._emit_observability(
                "system_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": True,
                    "added_char_count": 0,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        if addition is None or not addition.strip():
            self._emit_observability(
                "system_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": True,
                    "added_char_count": 0,
                    "failed": False,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        stripped = addition.strip()
        self._emit_observability(
            "system_prompt.provider.completed",
            event_context,
            {
                "provider_configured": True,
                "added_char_count": len(stripped),
                "failed": False,
                "duration_s": self._deps.clock.now() - started_at,
            },
        )
        return replace(
            config,
            system_prompt=_append_system_prompt(config.system_prompt, stripped),
        )

    async def _apply_memory_prompt_provider(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> QueryConfig:
        """Append memory mechanics after other prompt fragments."""
        provider = self._deps.memory_prompt_provider
        if provider is None:
            self._emit_observability(
                "memory_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": False,
                    "added_char_count": 0,
                    "failed": False,
                },
            )
            return config
        started_at = self._deps.clock.now()
        try:
            memory_prompt = await provider(config, ctx)
        except Exception as exc:
            _LOGGER.debug("memory prompt provider failed: %s", exc)
            self._emit_observability(
                "memory_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": True,
                    "added_char_count": 0,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        if memory_prompt is None or not memory_prompt.strip():
            self._emit_observability(
                "memory_prompt.provider.completed",
                event_context,
                {
                    "provider_configured": True,
                    "added_char_count": 0,
                    "failed": False,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        stripped = memory_prompt.strip()
        self._emit_observability(
            "memory_prompt.provider.completed",
            event_context,
            {
                "provider_configured": True,
                "added_char_count": len(stripped),
                "failed": False,
                "duration_s": self._deps.clock.now() - started_at,
            },
        )
        return replace(
            config,
            system_prompt=_append_system_prompt(
                config.system_prompt,
                stripped,
            ),
        )

    async def _load_turn_skills(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> tuple[SkillDefinition, ...]:
        """Load turn-visible skills through the optional provider.

        Fail-soft: malformed skill roots or unavailable plugin/MCP sources
        should not make a normal text turn fail. The provider owns its own
        diagnostics; QueryEngine only logs a debug line and continues with no
        skills.
        """
        provider = self._deps.skill_provider
        if provider is None:
            self._emit_observability(
                "skill.provider.completed",
                event_context,
                {
                    "provider_configured": False,
                    "skill_count": 0,
                    "failed": False,
                },
            )
            return ()
        started_at = self._deps.clock.now()
        self._emit_observability(
            "skill.provider.started",
            event_context,
            {"provider_configured": True},
        )
        try:
            skills = tuple(await provider(config, ctx))
        except Exception as exc:
            _LOGGER.debug("skill provider failed: %s", exc)
            self._emit_observability(
                "skill.provider.completed",
                event_context,
                {
                    "provider_configured": True,
                    "skill_count": 0,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return ()
        self._emit_observability(
            "skill.provider.completed",
            event_context,
            {
                "provider_configured": True,
                "skill_count": len(skills),
                "skill_names": tuple(skill.name for skill in skills),
                "failed": False,
                "duration_s": self._deps.clock.now() - started_at,
            },
        )
        return skills

    async def _build_tool_catalog(
        self,
        config: QueryConfig,
        skills: Sequence[SkillDefinition],
        ctx: ToolUseContext,
        event_context: KernelEventContext,
    ) -> QueryConfig:
        """Apply optional tool-catalog expansion for the turn.

        This is the future Skill/ToolSearch adapter seam. It intentionally only
        rewrites `QueryConfig.tools`; execution/orchestration still remains
        owned by the tool orchestration layer.
        """
        provider = self._deps.tool_catalog_provider
        if provider is None:
            self._emit_observability(
                "tool.catalog.completed",
                event_context,
                {
                    "provider_configured": False,
                    "skill_count": len(skills),
                    "input_tool_count": len(config.tools),
                    "output_tool_count": len(config.tools),
                    "tool_names": tuple(tool.name for tool in config.tools),
                    "failed": False,
                },
            )
            return config
        started_at = self._deps.clock.now()
        self._emit_observability(
            "tool.catalog.started",
            event_context,
            {
                "provider_configured": True,
                "skill_count": len(skills),
                "input_tool_count": len(config.tools),
            },
        )
        try:
            tools = await provider(config, ctx, skills)
        except Exception as exc:
            _LOGGER.debug("tool catalog provider failed: %s", exc)
            self._emit_observability(
                "tool.catalog.completed",
                event_context,
                {
                    "provider_configured": True,
                    "skill_count": len(skills),
                    "input_tool_count": len(config.tools),
                    "output_tool_count": len(config.tools),
                    "tool_names": tuple(tool.name for tool in config.tools),
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        if tools is None:
            self._emit_observability(
                "tool.catalog.completed",
                event_context,
                {
                    "provider_configured": True,
                    "skill_count": len(skills),
                    "input_tool_count": len(config.tools),
                    "output_tool_count": len(config.tools),
                    "tool_names": tuple(tool.name for tool in config.tools),
                    "failed": False,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return config
        tool_tuple = tuple(tools)
        self._emit_observability(
            "tool.catalog.completed",
            event_context,
            {
                "provider_configured": True,
                "skill_count": len(skills),
                "input_tool_count": len(config.tools),
                "output_tool_count": len(tool_tuple),
                "tool_names": tuple(tool.name for tool in tool_tuple),
                "failed": False,
                "duration_s": self._deps.clock.now() - started_at,
            },
        )
        return replace(config, tools=tool_tuple)

    def _schedule_memory_extraction(
        self,
        terminal: Terminal,
        ctx: ToolUseContext,
        turn_config: QueryConfig,
    ) -> None:
        """Schedule optional extraction only after a successful completed turn."""
        extractor = self._deps.memory_extractor
        if extractor is None or terminal.reason != "completed":
            return
        if ctx.agent_id is not None:
            return
        if terminal.final_state is None:
            return

        task = asyncio.create_task(
            self._run_memory_extraction(
                list(terminal.final_state.messages),
                ctx,
                turn_config,
            )
        )
        self._memory_extraction_tasks.add(task)
        task.add_done_callback(self._memory_extraction_tasks.discard)
        self._deps.observability.emit(
            "memory.extraction.scheduled",
            context=(
                ctx.observability_context.with_source("memory")
                if ctx.observability_context is not None
                else self._observability_context(
                    turn_id=f"turn-{self._turn_index}",
                    source="memory",
                )
            ),
            data={
                "message_count": len(terminal.final_state.messages),
                "task_count": len(self._memory_extraction_tasks),
            },
        )

    async def _run_memory_extraction(
        self,
        messages: list[MessageParam],
        ctx: ToolUseContext,
        turn_config: QueryConfig,
    ) -> None:
        extractor = self._deps.memory_extractor
        if extractor is None:
            return
        started_at = self._deps.clock.now()
        event_context = (
            ctx.observability_context.with_source("memory")
            if ctx.observability_context is not None
            else self._observability_context(
                turn_id=f"turn-{self._turn_index}",
                source="memory",
            )
        )
        try:
            outcome: object = await extractor(messages, turn_config, ctx)
        except (Exception, asyncio.CancelledError) as exc:
            # Best effort like the reference: extraction failures should not
            # change the user-visible turn result.
            _LOGGER.debug("memory extraction hook failed: %s", exc)
            self._deps.observability.emit(
                "memory.extraction.completed",
                context=event_context,
                data={
                    "status": "failed",
                    "message_count": len(messages),
                    "error_type": type(exc).__name__,
                    "duration_s": self._deps.clock.now() - started_at,
                },
            )
            return
        self._deps.observability.emit(
            "memory.extraction.completed",
            context=event_context,
            data=_memory_extraction_payload(
                outcome,
                message_count=len(messages),
                duration_s=self._deps.clock.now() - started_at,
            ),
        )

    async def drain_memory_extractions(self, timeout_s: float | None = 60.0) -> bool:
        """Wait for scheduled memory extraction tasks without cancelling them."""
        current = asyncio.current_task()
        pending = {
            task
            for task in self._memory_extraction_tasks
            if task is not current and not task.done()
        }
        if not pending:
            return True

        done, still_pending = await asyncio.wait(pending, timeout=timeout_s)
        for task in done:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
        return not still_pending

    # -----------------------------------------------------------------
    # Event translation — one switch per QueryEvent variant.
    # -----------------------------------------------------------------

    async def _translate_event(self, event: Any) -> SDKMessage | None:
        """Map an internal `QueryEvent` to an `SDKMessage`. None = absorbed.

        Side effects: append to `self._messages`, update usage/denials.
        """
        if isinstance(event, _QueryStreamStart):
            # Not surfaced to SDK by default.
            return None

        if isinstance(event, _QueryAssistantMessage):
            transcript_message = event.api_message or event.message
            self._append_message(transcript_message)
            await self._record_transcript_message(transcript_message)
            return SDKAssistantMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryToolResultMessage):
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryTaskNotificationsMessage):
            # Persist drained task notifications to the cross-turn log.
            # Without this, drains would only mutate per-turn state and
            # the next turn (which initializes from self._messages) would
            # never see them. Surface as SDKUserMessage — semantically
            # the drained content IS a user-role synthetic message.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryCoordinatorRuntimeMessage):
            # Mutable coordinator runtime digests are model-visible synthetic
            # user context. Persist them immediately so transcript replay sees
            # the same work-ledger snapshot ordering as the live model call.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryLocalAgentMessagesMessage):
            # SendMessage-to-local-agent prompts are model-visible queued
            # commands. Persist immediately so sidechain resume observes the
            # same input sequence the live child model saw.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryMemoryRecallMessage):
            # Persist recalled memory context exactly like task notifications:
            # it is model-visible user-side context that must survive into
            # QueryEngine's cross-turn transcript.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryStopHookMessage):
            # Stop-hook blocking feedback is model-visible synthetic user
            # context. Reference yields it before retrying the loop; persist it
            # immediately so replay matches the live model transcript.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryPostCompactMessage):
            # Compaction is a rewrite, but post-compact messages are still
            # API-visible and transcript-visible. Surface them immediately so
            # SDK consumers see the same timeline QueryEngine keeps after
            # terminal reconciliation.
            self._append_message(event.message)
            await self._record_transcript_message(event.message)
            if event.message.get("role") == "assistant":
                return SDKAssistantMessage(
                    session_id=self._config.session_id,
                    message=event.message,
                )
            return SDKUserMessage(
                session_id=self._config.session_id,
                message=event.message,
            )

        if isinstance(event, _QueryContentReplacementRecords):
            await self._record_content_replacements(event)
            return None

        if isinstance(event, _QueryModelStreamEvent):
            await self._record_stream_event(event)
            return None

        if isinstance(event, _QueryCompactBoundary):
            await self._record_compact_boundary(event)
            return SDKCompactBoundary(
                session_id=self._config.session_id,
                kind=event.kind,
                summary=event.summary,
            )

        if isinstance(event, _QueryTombstone):
            await self._record_tombstone(event)
            return None

        return None

    # -----------------------------------------------------------------
    # Terminal construction
    # -----------------------------------------------------------------

    def _build_terminal_result(
        self,
        terminal: Terminal | None,
        *,
        turn_count: int,
    ) -> SDKResult:
        """Decode a Terminal into an SDKResult. Called once per submit_message().

        Even for a `success` subtype, `is_error` may be True when the last
        assistant message carries an API-error marker — mirrors the reference
        regardless of `subtype: 'success'`. A turn can technically complete
        (loop exited cleanly) while the last message is an error surface that
        the SDK consumer needs to see as a failed turn.
        """
        last_is_api_error = _last_message_is_api_error(self._messages)

        if terminal is None:
            return SDKResult(
                subtype="success",
                session_id=self._config.session_id,
                is_error=last_is_api_error,
                num_turns=turn_count,
                result=_last_text_from(self._messages),
                usage=self._total_usage,
                permission_denials=tuple(self._permission_denials),
            )

        subtype = _terminal_reason_to_sdk_subtype(terminal.reason)
        is_error = subtype != "success" or last_is_api_error
        return SDKResult(
            subtype=subtype,
            session_id=self._config.session_id,
            is_error=is_error,
            num_turns=turn_count,
            result="" if is_error else _last_text_from(self._messages),
            usage=self._total_usage,
            permission_denials=tuple(self._permission_denials),
            errors=(terminal.message,) if terminal.message else (),
        )

    def _build_error_result(
        self,
        subtype: SDKResultSubtype,
        errors: list[str],
        *,
        turn_count: int,
    ) -> SDKResult:
        """Build a terminal SDKResult for the exception path."""
        return SDKResult(
            subtype=subtype,
            session_id=self._config.session_id,
            is_error=True,
            num_turns=turn_count,
            result="",
            usage=self._total_usage,
            permission_denials=tuple(self._permission_denials),
            errors=tuple(errors),
        )

    # -----------------------------------------------------------------
    # Cross-turn mutation — narrow choke points for future persistence.
    # -----------------------------------------------------------------

    def _append_message(self, msg: MessageParam) -> None:
        """Single site where the conversation log grows. Future: transcript
        recording, crash-recovery write-ahead log, observability hooks."""
        self._messages.append(msg)
        self._track_discovered_tool_names(msg)

    async def _record_transcript_message(self, msg: MessageParam) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None:
            return
        entry = TranscriptMessageEntry(
            parent_entry_id=self._transcript_parent_entry_id,
            logical_parent_entry_id=self._transcript_parent_entry_id,
            session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
            agent_id=scope.agent_id,
            is_sidechain=scope.is_sidechain,
            created_at=self._deps.clock.now(),
            cwd=self._ctx.cwd,
            version=RAYGENT_VERSION,
            message=msg,
            provider_message_id=_provider_message_id(msg),
        )
        try:
            await store.append(scope, entry)
        except Exception as exc:
            _LOGGER.debug("transcript message append failed: %s", exc)
            self._emit_transcript_observability(
                "transcript.append_failed",
                {
                    "entry_type": "message",
                    "role": _message_role(msg),
                    "error_type": type(exc).__name__,
                    "is_sidechain": scope.is_sidechain,
                    "agent_id_present": scope.agent_id is not None,
                },
            )
            return
        self._transcript_parent_entry_id = entry.entry_id
        self._emit_transcript_observability(
            "transcript.appended",
            {
                "entry_type": "message",
                "entry_id": entry.entry_id,
                "role": _message_role(msg),
                "is_sidechain": scope.is_sidechain,
                "agent_id_present": scope.agent_id is not None,
                "provider_message_id_present": entry.provider_message_id is not None,
            },
        )

    async def _record_compact_boundary(self, event: _QueryCompactBoundary) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None:
            self._transcript_parent_entry_id = None
            return
        entry = CompactBoundaryEntry(
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            created_at=self._deps.clock.now(),
            boundary=CompactBoundary(
                message_index=event.message_index,
                kind=event.kind,
                summary=event.summary,
            ),
            post_compact_message_count=None,
        )
        # A compact boundary starts a new replay epoch. Reset even if the
        # best-effort boundary append fails; otherwise later post-compact
        # messages would parent to pre-compact entries and resurrect stale
        # history during replay.
        self._transcript_parent_entry_id = None
        try:
            await store.append(scope, entry)
        except Exception as exc:
            _LOGGER.debug("transcript compact-boundary append failed: %s", exc)
            self._emit_transcript_observability(
                "transcript.append_failed",
                {
                    "entry_type": "compact_boundary",
                    "kind": event.kind,
                    "message_index": event.message_index,
                    "summary_char_count": len(event.summary),
                    "error_type": type(exc).__name__,
                    "is_sidechain": scope.is_sidechain,
                    "agent_id_present": scope.agent_id is not None,
                },
            )
            return
        self._emit_transcript_observability(
            "transcript.appended",
            {
                "entry_type": "compact_boundary",
                "entry_id": entry.entry_id,
                "kind": event.kind,
                "message_index": event.message_index,
                "summary_char_count": len(event.summary),
                "is_sidechain": scope.is_sidechain,
                "agent_id_present": scope.agent_id is not None,
            },
        )

    async def _record_content_replacements(
        self,
        event: _QueryContentReplacementRecords,
    ) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None or len(event.replacements) == 0:
            return
        entry = ContentReplacementEntry(
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            created_at=self._deps.clock.now(),
            replacements=event.replacements,
        )
        try:
            await store.append(scope, entry)
        except Exception as exc:
            _LOGGER.debug("transcript content-replacement append failed: %s", exc)
            self._emit_transcript_observability(
                "transcript.append_failed",
                {
                    "entry_type": "content_replacement",
                    "replacement_count": len(event.replacements),
                    "error_type": type(exc).__name__,
                    "is_sidechain": scope.is_sidechain,
                    "agent_id_present": scope.agent_id is not None,
                },
            )
            return
        self._emit_transcript_observability(
            "transcript.appended",
            {
                "entry_type": "content_replacement",
                "entry_id": entry.entry_id,
                "replacement_count": len(event.replacements),
                "is_sidechain": scope.is_sidechain,
                "agent_id_present": scope.agent_id is not None,
            },
        )

    async def _record_stream_event(self, event: _QueryModelStreamEvent) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None:
            return
        entry = StreamEventEntry(
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            created_at=self._deps.clock.now(),
            event=event.event,
        )
        try:
            await store.append(scope, entry)
        except Exception as exc:
            _LOGGER.debug("transcript stream-event append failed: %s", exc)
            self._emit_transcript_observability(
                "transcript.append_failed",
                {
                    "entry_type": "stream_event",
                    "stream_event_type": _stream_event_type(event.event),
                    "error_type": type(exc).__name__,
                    "is_sidechain": scope.is_sidechain,
                    "agent_id_present": scope.agent_id is not None,
                },
            )
            return
        self._emit_transcript_observability(
            "transcript.appended",
            {
                "entry_type": "stream_event",
                "entry_id": entry.entry_id,
                "stream_event_type": _stream_event_type(event.event),
                "is_sidechain": scope.is_sidechain,
                "agent_id_present": scope.agent_id is not None,
            },
        )

    async def _record_tombstone(self, event: _QueryTombstone) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None:
            return
        entry = TombstoneEntry(
            session_id=scope.session_id,
            agent_id=scope.agent_id,
            created_at=self._deps.clock.now(),
            target_entry_id=event.target_entry_id,
            target_message_id=event.target_message_id,
            reason=event.reason,
            event=event.event,
        )
        try:
            await store.append(scope, entry)
        except Exception as exc:
            _LOGGER.debug("transcript tombstone append failed: %s", exc)
            self._emit_transcript_observability(
                "transcript.append_failed",
                {
                    "entry_type": "tombstone",
                    "reason_category": _safe_tombstone_reason(event.reason),
                    "reason_char_count": len(event.reason),
                    "error_type": type(exc).__name__,
                    "is_sidechain": scope.is_sidechain,
                    "agent_id_present": scope.agent_id is not None,
                },
            )
            return
        self._emit_transcript_observability(
            "transcript.appended",
            {
                "entry_type": "tombstone",
                "entry_id": entry.entry_id,
                "reason_category": _safe_tombstone_reason(event.reason),
                "reason_char_count": len(event.reason),
                "target_entry_id_present": event.target_entry_id is not None,
                "target_message_id_present": event.target_message_id is not None,
                "is_sidechain": scope.is_sidechain,
                "agent_id_present": scope.agent_id is not None,
            },
        )

    async def _flush_transcript(self) -> None:
        store = self._deps.transcript_store
        scope = self._transcript_scope
        if store is None or scope is None:
            return
        try:
            await store.flush(scope)
        except Exception as exc:
            _LOGGER.debug("transcript flush failed: %s", exc)

    def _reconcile_terminal_state(self, terminal: Terminal) -> None:
        """Replace cross-turn history with the terminal state's messages.

        Event translation is append-only, but compaction is a rewrite:
        `query()` may replace a long pre-call transcript with a compacted
        summary and then build `Terminal.final_state` from that post-compact
        view. If the engine kept only the append-only event log, the next
        `submit_message()` would resurrect pre-compact history and compact
        forever. Reference query continue sites also carry forward the
        """
        if terminal.final_state is None:
            return
        self._messages = list(terminal.final_state.messages)
        self._compact_boundaries = terminal.final_state.compact_boundaries
        self._permission_denials = list(terminal.final_state.permission_denials)
        self._track_usage(terminal.final_state.usage)
        self._track_discovered_tool_names(*terminal.final_state.messages)

    def _track_usage(self, delta: UsageTotals) -> None:
        """Single site for usage accumulation. Future: budget enforcement
        wrappers, per-model breakdown."""
        self._total_usage = UsageTotals(
            input_tokens=self._total_usage.input_tokens + delta.input_tokens,
            output_tokens=self._total_usage.output_tokens + delta.output_tokens,
            cache_creation_input_tokens=(
                self._total_usage.cache_creation_input_tokens
                + delta.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self._total_usage.cache_read_input_tokens
                + delta.cache_read_input_tokens
            ),
            cost_usd=self._total_usage.cost_usd + delta.cost_usd,
        )

    def _track_denial(self, denial: PermissionDenial) -> None:
        """Single site for recording denials. Future: replay into next turn's
        context via the context pipeline."""
        self._permission_denials.append(denial)

    def _track_discovered_tool_names(self, *messages: MessageParam) -> None:
        from raygent_harness.tools.tool_search_tool import (
            selected_tool_names_from_messages,
        )

        self._discovered_tool_names.update(selected_tool_names_from_messages(messages))

    def _observability_context(
        self,
        *,
        turn_id: str,
        source: str,
    ) -> KernelEventContext:
        scope = self._transcript_scope
        return KernelEventContext(
            session_id=(
                scope.session_id
                if scope is not None
                else self._config.session_id or self._ctx.session_id
            ),
            runtime_session_id=scope.runtime_session_id if scope is not None else None,
            agent_id=(
                self._ctx.agent_id
                or self._config.agent_id
                or (scope.agent_id if scope is not None else None)
            ),
            turn_id=turn_id,
            source=source,
        )

    def _emit_observability(
        self,
        event_type: str,
        context: KernelEventContext,
        data: dict[str, object],
    ) -> None:
        self._deps.observability.emit(event_type, context=context, data=data)

    def _emit_transcript_observability(
        self,
        event_type: str,
        data: dict[str, object],
    ) -> None:
        turn_id = f"turn-{self._turn_index}" if self._turn_index else "turn-0"
        self._emit_observability(
            event_type,
            self._observability_context(turn_id=turn_id, source="transcript"),
            data,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_AGENT_TRIGGER_TRUNCATION_MARKER = "\n[agent trigger guidance truncated]"
_AGENT_TRIGGER_FIELD_MAX_CHARS = 160
_AGENT_TRIGGER_METADATA_MATCH_LIMIT = 64


@dataclass(frozen=True)
class _BoundedAgentTriggerField:
    value: str | None
    truncated: bool


@dataclass(frozen=True)
class _AgentTriggerCandidateMessage:
    content: str
    matches: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _AgentTriggerRenderResult:
    messages: tuple[MessageParam, ...]
    input_char_count: int
    rendered_char_count: int
    truncated_message_count: int
    dropped_message_count: int


def _agent_trigger_scope_allows(
    spec: AgentTriggerPolicySpec,
    agent_id: str | None,
) -> bool:
    if spec.agent_scope == "all":
        return True
    if spec.agent_scope == "main":
        return agent_id is None
    return agent_id is not None


def _render_agent_trigger_messages(
    decision: AgentTriggerDecision,
    spec: AgentTriggerPolicySpec,
    *,
    delegation_tool_available: bool,
) -> _AgentTriggerRenderResult:
    max_messages = max(0, spec.max_messages)
    max_message_chars = max(0, spec.max_message_chars)
    total_candidate_count = (
        (1 if decision.matches else 0) + len(decision.model_visible_messages)
    )
    candidates: list[_AgentTriggerCandidateMessage] = []

    if max_messages == 0 or max_message_chars == 0:
        return _AgentTriggerRenderResult(
            messages=(),
            input_char_count=0,
            rendered_char_count=0,
            truncated_message_count=0,
            dropped_message_count=total_candidate_count,
        )

    if decision.matches:
        candidates.append(
            _agent_trigger_guidance_candidate(
                decision.matches,
                delegation_tool_available=delegation_tool_available,
            )
        )
    for message in decision.model_visible_messages:
        if len(candidates) >= max_messages:
            break
        content = _message_content_to_text(message.get("content", ""))
        if content.strip():
            candidates.append(_AgentTriggerCandidateMessage(content=content, matches=()))

    input_char_count = sum(len(candidate.content) for candidate in candidates)

    kept = candidates[:max_messages]
    dropped_message_count = max(0, total_candidate_count - len(kept))
    messages: list[MessageParam] = []
    rendered_char_count = 0
    truncated_message_count = 0
    for candidate in kept:
        rendered_content, truncated = _truncate_agent_trigger_text(
            candidate.content,
            max_message_chars,
        )
        rendered_char_count += len(rendered_content)
        if truncated:
            truncated_message_count += 1
        metadata = {
            "type": "agent_trigger",
            "match_count": len(decision.matches),
            "delegation_tool_available": delegation_tool_available,
            "rendered_char_count": len(rendered_content),
            "input_char_count": len(candidate.content),
            "truncated_message_count": 1 if truncated else 0,
            "dropped_message_count": dropped_message_count,
            "max_messages": max_messages,
            "max_message_chars": max_message_chars,
            "suppress_main_turn_requested": decision.suppress_main_turn,
            "suppress_main_turn_applied": False,
            "matches": list(candidate.matches),
        }
        messages.append(
            cast(
                MessageParam,
                {
                    "role": "user",
                    "content": rendered_content,
                    "raygentMessageKind": "agent_trigger",
                    "raygentAgentTrigger": metadata,
                },
            )
        )

    return _AgentTriggerRenderResult(
        messages=tuple(messages),
        input_char_count=input_char_count,
        rendered_char_count=rendered_char_count,
        truncated_message_count=truncated_message_count,
        dropped_message_count=dropped_message_count,
    )


def _agent_trigger_guidance_candidate(
    matches: Sequence[AgentTriggerMatch],
    *,
    delegation_tool_available: bool,
) -> _AgentTriggerCandidateMessage:
    if delegation_tool_available:
        routing_line = (
            "If appropriate, use the available Agent/Task tool path to delegate "
            "work; do not assume delegation already happened."
        )
    else:
        routing_line = (
            "No agent-delegation tool is available in this turn's tool set; use "
            "these matches as planning/routing context, not as a delegation call."
        )
    lines = [
        "Agent trigger policy matched relevant agents.",
        routing_line,
        "",
        "Matches:",
    ]
    metadata_matches: list[dict[str, object]] = []
    rendered_matches = matches[:_AGENT_TRIGGER_METADATA_MATCH_LIMIT]
    for index, match in enumerate(rendered_matches, start=1):
        bounded_id = _bounded_agent_trigger_field(match.id)
        bounded_agent = _bounded_agent_trigger_field(match.agent_name)
        bounded_reason = _bounded_agent_trigger_field(match.reason)
        bounded_hint = _bounded_agent_trigger_field(match.prompt_hint)
        bounded_source = _bounded_agent_trigger_field(match.source)
        line = f"{index}. agent={bounded_agent.value or 'unknown'}"
        if bounded_id.value:
            line += f" id={bounded_id.value}"
        if match.confidence is not None:
            line += f" confidence={match.confidence:.3g}"
        if bounded_reason.value:
            line += f"\n   reason: {bounded_reason.value}"
        if bounded_hint.value:
            line += f"\n   prompt_hint: {bounded_hint.value}"
        if bounded_source.value:
            line += f"\n   source: {bounded_source.value}"
        lines.append(line)
        if len(metadata_matches) < _AGENT_TRIGGER_METADATA_MATCH_LIMIT:
            metadata_matches.append(
                {
                    "id": bounded_id.value or "",
                    "agent_name": bounded_agent.value or "",
                    "reason": bounded_reason.value,
                    "prompt_hint": bounded_hint.value,
                    "confidence": match.confidence,
                    "source": bounded_source.value,
                    "truncated": (
                        bounded_id.truncated
                        or bounded_agent.truncated
                        or bounded_reason.truncated
                        or bounded_hint.truncated
                        or bounded_source.truncated
                    ),
                }
            )
    if len(matches) > len(rendered_matches):
        lines.append(
            f"... {len(matches) - len(rendered_matches)} "
            "additional matches omitted from metadata."
        )
    return _AgentTriggerCandidateMessage(
        content="\n".join(lines),
        matches=tuple(metadata_matches),
    )


def _bounded_agent_trigger_field(value: object) -> _BoundedAgentTriggerField:
    if value is None:
        return _BoundedAgentTriggerField(value=None, truncated=False)
    text = str(value).strip()
    if len(text) <= _AGENT_TRIGGER_FIELD_MAX_CHARS:
        return _BoundedAgentTriggerField(value=text, truncated=False)
    marker = "..."
    budget = max(0, _AGENT_TRIGGER_FIELD_MAX_CHARS - len(marker))
    return _BoundedAgentTriggerField(value=f"{text[:budget]}{marker}", truncated=True)


def _truncate_agent_trigger_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len(_AGENT_TRIGGER_TRUNCATION_MARKER):
        return _AGENT_TRIGGER_TRUNCATION_MARKER[:max_chars], True
    keep_chars = max_chars - len(_AGENT_TRIGGER_TRUNCATION_MARKER)
    return f"{text[:keep_chars]}{_AGENT_TRIGGER_TRUNCATION_MARKER}", True


def _message_content_char_count(message: MessageParam) -> int:
    return len(_message_content_to_text(message.get("content", "")))


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _agent_delegation_tool_available(config: QueryConfig, ctx: ToolUseContext) -> bool:
    selected_deferred_tool_names = {
        *ctx.discovered_tool_names,
        *selected_tool_names_from_messages(ctx.messages),
    }
    for tool in config.tools:
        is_delegation_tool = tool.name in {"Agent", "Task"} or any(
            alias in {"Agent", "Task"} for alias in tool.aliases
        )
        if not is_delegation_tool:
            continue
        if tool_visible_to_model(tool, selected_deferred_tool_names):
            return True
    return False


def _coerce_user_message(prompt: str | MessageParam) -> MessageParam:
    """String prompts become `{role: 'user', content: str}`. Otherwise pass
    through. Keeps the public API ergonomic without losing the structured path.
    """
    if isinstance(prompt, str):
        return {"role": "user", "content": prompt}
    return prompt


def _default_transcript_scope(
    config: QueryConfig,
    ctx: ToolUseContext,
) -> TranscriptScope | None:
    agent_id = ctx.agent_id or config.agent_id
    if agent_id is not None:
        # Sidechains must be grouped under the parent session id, not the
        # child's runtime session id. Chunk 5 passes that explicit scope.
        return None
    return TranscriptScope(
        session_id=config.session_id or ctx.session_id,
    )


def _provider_message_id(message: MessageParam) -> str | None:
    message_id = message.get("id") or message.get("uuid")
    return message_id if isinstance(message_id, str) else None


def _message_role(message: MessageParam) -> str | None:
    return message.get("role")


def _stream_event_type(event: object) -> str | None:
    event_type = getattr(event, "type", None)
    if isinstance(event_type, str):
        return event_type
    if isinstance(event, dict):
        event_mapping = cast(Mapping[str, object], event)
        raw_type = event_mapping.get("type")
        return raw_type if isinstance(raw_type, str) else None
    return None


def _safe_tombstone_reason(reason: str) -> str:
    if reason == "streaming_transport_fallback_started":
        return "streaming_transport_fallback_started"
    if reason:
        return "provider_supplied"
    return "unknown"


_SAFE_EXTRACTION_STATUSES = frozenset(
    {
        "ran",
        "coalesced",
        "throttled",
        "skipped_disabled",
        "skipped_remote",
        "skipped_subagent",
        "skipped_direct_write",
        "error",
        "completed",
        "failed",
    }
)


def _safe_extraction_status(status: str) -> str:
    return status if status in _SAFE_EXTRACTION_STATUSES else "custom"


def _memory_extraction_payload(
    outcome: object,
    *,
    message_count: int,
    duration_s: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "completed",
        "message_count": message_count,
        "duration_s": duration_s,
    }
    status = getattr(outcome, "status", None)
    if isinstance(status, str):
        payload["status"] = _safe_extraction_status(status)
        if status not in _SAFE_EXTRACTION_STATUSES:
            payload["status_char_count"] = len(status)
    new_message_count = getattr(outcome, "new_message_count", None)
    if isinstance(new_message_count, int):
        payload["new_message_count"] = new_message_count
    written_paths = getattr(outcome, "written_paths", None)
    if isinstance(written_paths, tuple | list):
        payload["written_path_count"] = len(cast(Sequence[object], written_paths))
    memory_paths = getattr(outcome, "memory_paths", None)
    if isinstance(memory_paths, tuple | list):
        payload["memory_path_count"] = len(cast(Sequence[object], memory_paths))
    error = getattr(outcome, "error", None)
    if isinstance(error, str):
        payload["error_char_count"] = len(error)
    return payload


def _append_system_prompt(base: str, addition: str) -> str:
    """Append one system-prompt segment with stable paragraph separation."""
    if not base:
        return addition
    return f"{base}\n\n{addition}"


def _last_text_from(messages: list[MessageParam]) -> str:
    """Extract the last assistant text content. For SDKResult.result.

    Conservative: only returns text when the last assistant message is a
    pure-text string. Structured content blocks return empty; callers that
    need structured output should read `_messages` directly.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        return ""
    return ""


_TERMINAL_TO_SUBTYPE: dict[TerminalReason, SDKResultSubtype] = {
    "completed": "success",
    "max_turns": "error_max_turns",
    "budget_exceeded": "error_max_budget_usd",
    "blocking_limit": "error_during_execution",
    "prompt_too_long": "error_during_execution",
    "image_error": "error_during_execution",
    "model_error": "error_during_execution",
    "aborted_streaming": "error_aborted",
    "aborted_tools": "error_aborted",
    "hook_stopped": "error_during_execution",
    "stop_hook_prevented": "error_during_execution",
    "fallback_exhausted": "error_during_execution",
}
"""Every `TerminalReason` must have an explicit entry. The `_terminal_reason_
to_sdk_subtype` fallback below is a fail-closed guard — a missing key means
a producer added a reason the SDK surface hasn't been taught about yet, and
we want that obvious (ideally caught in tests). Don't rely on the fallback."""


def _terminal_reason_to_sdk_subtype(reason: TerminalReason) -> SDKResultSubtype:
    """Map the internal Terminal reason to the SDK-facing subtype.

    Many internal reasons collapse into `error_during_execution` at the SDK
    boundary — callers can read `SDKResult.errors` for the detail. Keep the
    mapping table explicit so the SDK surface doesn't silently drift when we
    add new terminal reasons.
    """
    return _TERMINAL_TO_SUBTYPE.get(reason, "error_during_execution")


def _last_message_is_api_error(messages: list[MessageParam]) -> bool:
    """Return True when the last assistant message is an API-error surface.

    Mirrors the `isApiError = Boolean(result.isApiErrorMessage)` check at
    `completed`) while the last message is still an error the consumer needs
    to flag as `is_error: true`.

    Provider-normalized API-error handling wires provider-normalized API-error messages into the
    Raygent-owned `MessageParam` shape, so this can now read the flag directly.
    """
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        return message.get("isApiErrorMessage") is True
    return False


def _usage_totals_payload(usage: UsageTotals) -> dict[str, object]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cost_usd": usage.cost_usd,
    }


__all__ = [
    "QueryEngine",
    "SDKAssistantMessage",
    "SDKCompactBoundary",
    "SDKMessage",
    "SDKResult",
    "SDKResultSubtype",
    "SDKSystemInit",
    "SDKUserMessage",
]
