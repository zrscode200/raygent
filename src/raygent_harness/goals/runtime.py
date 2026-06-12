"""Goal runtime lifecycle service over a headless Raygent session."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, cast

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps, ToolCatalogProvider
from raygent_harness.core.model_types import FrozenJson
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.state import PermissionDenial, UsageTotals
from raygent_harness.core.tool import Tool, ToolUseContext
from raygent_harness.goals.context import GoalContextProvider
from raygent_harness.goals.evaluators import (
    GoalCompletionEvaluation,
    GoalCompletionEvaluator,
    GoalEvaluatorFailureStatus,
    GoalProgressEvaluator,
    GoalProgressLedger,
    goal_completion_evaluation_to_dict,
    goal_progress_ledger_to_dict,
)
from raygent_harness.goals.events import GoalEventEmitter, GoalEventType
from raygent_harness.goals.models import (
    GoalPolicy,
    GoalSpec,
    GoalState,
    GoalStatus,
    create_goal_state,
)
from raygent_harness.goals.store import (
    GoalNotFoundError,
    GoalStateConflictError,
    GoalStore,
    InMemoryGoalStore,
    JsonGoalStore,
)
from raygent_harness.goals.tools import (
    GET_GOAL_TOOL_NAME,
    UPDATE_GOAL_TOOL_NAME,
    GoalUpdatePolicyDecision,
    UpdateGoalInput,
    build_get_goal_tool,
    build_update_goal_tool,
)
from raygent_harness.skills.models import SkillDefinition

GoalRuntimeStopReason = Literal[
    "continued",
    "no_active_goal",
    "auto_continue_disabled",
    "max_continuations_reached",
    "budget_limited",
    "usage_limited",
    "completed",
    "blocked",
    "paused",
    "cancelled",
    "failed",
]

_RESUMABLE_STATUSES: frozenset[GoalStatus] = frozenset({"paused", "blocked"})
_CANCELLABLE_STATUSES: frozenset[GoalStatus] = frozenset(
    {"active", "paused", "blocked", "usage_limited"}
)
_GOAL_TOOL_NAMES = frozenset({GET_GOAL_TOOL_NAME, UPDATE_GOAL_TOOL_NAME})
_PermissionDenialKey = tuple[str, str, str]


class GoalRuntimeError(Exception):
    """Base exception for goal runtime lifecycle failures."""


class GoalRuntimeStateError(GoalRuntimeError):
    """Raised when a lifecycle operation is invalid for current goal state."""


class GoalRuntimeSession(Protocol):
    """Narrow RaygentSession shape consumed by the goal runtime."""

    engine: Any
    config: QueryConfig
    deps: QueryDeps
    ctx: ToolUseContext

    @property
    def session_id(self) -> str:
        """Conversation/session id used to scope goal state."""
        ...

    async def run_until_result(self, prompt: str) -> SDKResult:
        """Submit one turn and return the terminal SDK result."""
        ...


@dataclass(frozen=True, slots=True)
class GoalRuntimeConfig:
    """Runtime controls for autonomous goal continuation.

    The runtime is opt-in and session-scoped. It wires goal context/tools into
    one supplied session, but does not parse product `/goal` commands, choose a
    provider, or install product storage.
    """

    continuation_prompt: str = (
        "Continue working toward the active Raygent goal. Use the goal context "
        "and goal tools to preserve scope, audit completion evidence, and report "
        "complete or blocked only when the durable goal policy allows it."
    )
    max_continuations_per_run: int = 10
    install_goal_tools: bool = True
    install_goal_context_provider: bool = True
    event_emitter: GoalEventEmitter | None = None
    completion_evaluator: GoalCompletionEvaluator | None = None
    progress_evaluator: GoalProgressEvaluator | None = None
    evaluator_failure_status: GoalEvaluatorFailureStatus = "blocked"

    def __post_init__(self) -> None:
        if not self.continuation_prompt.strip():
            raise ValueError("GoalRuntimeConfig.continuation_prompt must be non-empty")
        if self.max_continuations_per_run < 1:
            raise ValueError("max_continuations_per_run must be >= 1")
        if self.evaluator_failure_status not in {"blocked", "failed", "ignore"}:
            raise ValueError("evaluator_failure_status must be blocked, failed, or ignore")


@dataclass(frozen=True, slots=True)
class GoalRuntimeResult:
    """Result of one or more runtime-owned continuation turns."""

    state: GoalState | None
    stop_reason: GoalRuntimeStopReason
    continuation_count: int = 0
    sdk_results: tuple[SDKResult, ...] = ()

    @property
    def continued(self) -> bool:
        return self.continuation_count > 0


@dataclass
class GoalRuntime:
    """Session-scoped goal service that owns runtime lifecycle transitions.

    Model-visible tools may report only `complete` or `blocked`. This runtime
    owns system/product statuses such as pause, cancel, budget limits, usage
    limits, and failed turns, and drives idle continuation through the supplied
    `RaygentSession` seam.
    """

    session: GoalRuntimeSession
    store: GoalStore = field(default_factory=InMemoryGoalStore)
    config: GoalRuntimeConfig = field(default_factory=GoalRuntimeConfig)
    _last_session_token_total: int = field(default=0, init=False, repr=False)
    _consecutive_errors: dict[str, int] = field(
        default_factory=dict[str, int],
        init=False,
        repr=False,
    )
    _consecutive_no_progress: dict[str, int] = field(
        default_factory=dict[str, int],
        init=False,
        repr=False,
    )
    _seen_permission_denials: dict[str, frozenset[_PermissionDenialKey]] = field(
        default_factory=dict[str, frozenset[_PermissionDenialKey]],
        init=False,
        repr=False,
    )
    _event_emitter: GoalEventEmitter | None = field(default=None, init=False, repr=False)
    _catalog_provider_installed: bool = field(default=False, init=False, repr=False)
    _original_catalog_provider: ToolCatalogProvider | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        store_emitter = (
            self.store.event_emitter
            if isinstance(self.store, (InMemoryGoalStore, JsonGoalStore))
            else None
        )
        self._event_emitter = self.config.event_emitter or store_emitter
        if (
            isinstance(self.store, (InMemoryGoalStore, JsonGoalStore))
            and self.store.event_emitter is None
        ):
            self.store.event_emitter = self._event_emitter
        self._last_session_token_total = _session_usage_total_tokens(self.session)

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def install(self) -> None:
        """Install goal context/tools into the wrapped session at a turn boundary."""

        _raise_if_session_active(self.session)
        if self.config.install_goal_context_provider:
            self._install_context_provider()
        if self.config.install_goal_tools:
            self._install_goal_tools()

    def start(
        self,
        spec: GoalSpec | str,
        *,
        policy: GoalPolicy | None = None,
        goal_id: str | None = None,
    ) -> GoalState:
        """Create a new active goal for this session and install runtime surfaces."""

        _raise_if_session_active(self.session)
        resolved_spec = spec if isinstance(spec, GoalSpec) else GoalSpec(objective=spec)
        state = create_goal_state(
            goal_id=goal_id or _new_goal_id(),
            session_id=self.session_id,
            spec=resolved_spec,
            policy=policy,
            now=self._now(),
        )
        created = self.store.create(state)
        self._consecutive_errors[created.goal_id] = 0
        self._consecutive_no_progress[created.goal_id] = 0
        self.install()
        self._last_session_token_total = _session_usage_total_tokens(self.session)
        self._baseline_permission_denials(created.goal_id)
        return created

    def resume(self, goal_id: str | None = None, *, reason: str | None = None) -> GoalState:
        """Resume a paused or blocked session goal."""

        _raise_if_session_active(self.session)
        current = self._resolve_goal_for_operation(goal_id, operation="resume")
        if current.status == "active":
            self.install()
            self._last_session_token_total = _session_usage_total_tokens(self.session)
            self._baseline_permission_denials(current.goal_id)
            self._reset_no_progress_count(current.goal_id)
            return current
        if current.status not in _RESUMABLE_STATUSES:
            raise GoalRuntimeStateError(
                f"Cannot resume goal {current.goal_id} from status={current.status}"
            )
        updated = self.store.update(
            current.goal_id,
            lambda state: state.with_status(
                "active",
                reason=reason or "resumed",
                now=self._now(),
            ),
        )
        self.install()
        self._last_session_token_total = _session_usage_total_tokens(self.session)
        self._baseline_permission_denials(updated.goal_id)
        self._reset_no_progress_count(updated.goal_id)
        return updated

    def pause(self, *, reason: str | None = None) -> GoalState:
        """Pause the active goal and stop autonomous continuation."""

        current = self.store.get_active_for_session(self.session_id)
        if current is None:
            raise GoalNotFoundError(f"No active goal for session: {self.session_id}")
        return self.store.update(
            current.goal_id,
            lambda state: state.with_status(
                "paused",
                reason=reason or "paused",
                now=self._now(),
            ),
        )

    def cancel(self, goal_id: str | None = None, *, reason: str | None = None) -> GoalState:
        """Cancel an active, paused, blocked, or usage-limited session goal."""

        current = self._resolve_goal_for_operation(goal_id, operation="cancel")
        if current.status not in _CANCELLABLE_STATUSES:
            raise GoalRuntimeStateError(
                f"Cannot cancel goal {current.goal_id} from status={current.status}"
            )
        return self.store.update(
            current.goal_id,
            lambda state: state.with_status(
                "cancelled",
                reason=reason or "cancelled",
                now=self._now(),
            ),
        )

    async def run_idle_once(self) -> GoalRuntimeResult:
        """Run one automatic continuation turn when the active goal allows it."""

        state = self.store.get_active_for_session(self.session_id)
        if state is None:
            return GoalRuntimeResult(state=None, stop_reason="no_active_goal")
        self.install()
        if not state.policy.continuation.auto_continue_when_idle:
            return GoalRuntimeResult(state=state, stop_reason="auto_continue_disabled")

        limit_reason = _budget_limit_reason(state)
        if limit_reason is not None:
            limited = self._transition(
                state,
                "budget_limited",
                reason=limit_reason,
            )
            return GoalRuntimeResult(state=limited, stop_reason="budget_limited")

        prompt = self.config.continuation_prompt
        blocked_count_before_turn = state.blocked_turn_count
        permission_denials_before_turn = self._permission_denials_before_turn(
            state.goal_id
        )
        self._emit(
            "goal_continuation_started",
            state,
            data={"prompt_char_count": len(prompt)},
        )
        self._emit("goal_turn_started", state)
        started_at = self._now()
        try:
            sdk_result = await self.session.run_until_result(prompt)
        except Exception as exc:
            elapsed = max(0.0, self._now() - started_at)
            failed = self._account_turn_exception(state, exc, elapsed)
            return GoalRuntimeResult(
                state=failed,
                stop_reason="failed",
                continuation_count=1,
            )

        elapsed = max(0.0, self._now() - started_at)
        accounted = self._account_turn_result(state.goal_id, sdk_result, elapsed)
        final = self._apply_terminal_transition(
            accounted,
            sdk_result,
            blocked_count_before_turn=blocked_count_before_turn,
            permission_denials_before_turn=permission_denials_before_turn,
        )
        final = await self._apply_evaluator_policies(final, sdk_result)
        return GoalRuntimeResult(
            state=final,
            stop_reason=_stop_reason_for_state(final),
            continuation_count=1,
            sdk_results=(sdk_result,),
        )

    async def run_until_idle(
        self,
        *,
        max_continuations: int | None = None,
    ) -> GoalRuntimeResult:
        """Continue an active goal until it stops or a configured run cap is hit."""

        limit = self._continuation_limit(max_continuations)
        sdk_results: list[SDKResult] = []
        state: GoalState | None = None
        count = 0
        while count < limit:
            step = await self.run_idle_once()
            state = step.state
            sdk_results.extend(step.sdk_results)
            count += step.continuation_count
            if step.continuation_count == 0 or step.stop_reason != "continued":
                return GoalRuntimeResult(
                    state=state,
                    stop_reason=step.stop_reason,
                    continuation_count=count,
                    sdk_results=tuple(sdk_results),
                )
        return GoalRuntimeResult(
            state=state or self.store.get_active_for_session(self.session_id),
            stop_reason="max_continuations_reached",
            continuation_count=count,
            sdk_results=tuple(sdk_results),
        )

    def _install_context_provider(self) -> None:
        providers = self.session.deps.context_providers
        if any(
            isinstance(provider, GoalContextProvider) and provider.store is self.store
            for provider in providers
        ):
            return
        self.session.deps.context_providers = (
            *providers,
            GoalContextProvider(self.store),
        )

    def _install_goal_tools(self) -> None:
        goal_tools = self._goal_tools()
        tools = _merge_goal_tools(self.session.config.tools, goal_tools)
        new_config = replace(self.session.config, tools=tools)
        self.session.config = new_config
        self.session.ctx = replace(self.session.ctx, tools=tools)
        self.session.engine._config = new_config
        self.session.engine._ctx = self.session.ctx
        if not self._catalog_provider_installed:
            self._original_catalog_provider = self.session.deps.tool_catalog_provider
            self.session.deps.tool_catalog_provider = self._goal_catalog_provider(
                self._original_catalog_provider,
            )
            self._catalog_provider_installed = True

    def _goal_tools(self) -> tuple[Tool, Tool]:
        return (
            build_get_goal_tool(store=self.store),
            build_update_goal_tool(
                store=self.store,
                update_policy=self._authorize_model_update,
            ),
        )

    def _authorize_model_update(
        self,
        state: GoalState,
        update: UpdateGoalInput,
        _ctx: ToolUseContext,
        /,
    ) -> GoalUpdatePolicyDecision:
        if update.status == "complete":
            blockers = self._completion_blockers(state, source="model")
            if blockers:
                content = (
                    "update_goal rejected: completion is blocked by runtime "
                    f"invariants: {'; '.join(blockers)}"
                )
                updated = self.store.update(
                    state.goal_id,
                    lambda current: replace(
                        current,
                        last_reason=content,
                        updated_at=self._now(),
                    ),
                )
                return GoalUpdatePolicyDecision(
                    proceed=False,
                    content=content,
                    is_error=True,
                    updated_state=updated,
                )
            return GoalUpdatePolicyDecision(proceed=True)

        if update.status == "blocked":
            return self._record_blocked_report(state, reason=update.reason)

        return GoalUpdatePolicyDecision(
            proceed=False,
            content="update_goal rejected: unsupported runtime status.",
            is_error=True,
        )

    def _goal_catalog_provider(
        self,
        upstream: ToolCatalogProvider | None,
    ) -> ToolCatalogProvider:
        async def provider(
            config: QueryConfig,
            ctx: ToolUseContext,
            skills: Sequence[SkillDefinition],
            /,
        ) -> Sequence[Tool] | None:
            upstream_tools = None
            if upstream is not None:
                upstream_tools = await upstream(config, ctx, skills)
            base_tools = config.tools if upstream_tools is None else tuple(upstream_tools)
            return _merge_goal_tools(base_tools, self._goal_tools())

        return provider

    def _resolve_goal_for_operation(
        self,
        goal_id: str | None,
        *,
        operation: str,
    ) -> GoalState:
        if goal_id is not None:
            state = self.store.get(goal_id)
            if state is None or state.session_id != self.session_id:
                raise GoalNotFoundError(f"Goal not found for session: {goal_id}")
            return state
        active = self.store.get_active_for_session(self.session_id)
        if active is not None:
            return active
        candidates = sorted(
            self.store.list_for_session(self.session_id),
            key=lambda item: item.updated_at,
            reverse=True,
        )
        for candidate in candidates:
            if operation == "resume" and candidate.status in _RESUMABLE_STATUSES:
                return candidate
            if operation == "cancel" and candidate.status in _CANCELLABLE_STATUSES:
                return candidate
        raise GoalNotFoundError(f"No goal available to {operation} for session: {self.session_id}")

    def _account_turn_result(
        self,
        goal_id: str,
        sdk_result: SDKResult,
        elapsed_s: float,
    ) -> GoalState:
        total = _usage_total_tokens(sdk_result.usage)
        token_delta = max(0, total - self._last_session_token_total)
        self._last_session_token_total = total
        accounted = self.store.update(
            goal_id,
            lambda state: state.with_accounting(
                turn_delta=1,
                token_delta=token_delta,
                time_delta_s=elapsed_s,
                now=self._now(),
            ),
        )
        self._emit(
            "goal_turn_accounted",
            accounted,
            data={
                "turn_delta": 1,
                "token_delta": token_delta,
                "time_delta_s": elapsed_s,
                "sdk_subtype": sdk_result.subtype,
                "sdk_is_error": sdk_result.is_error,
            },
        )
        return accounted

    def _account_turn_exception(
        self,
        state: GoalState,
        exc: Exception,
        elapsed_s: float,
    ) -> GoalState:
        accounted = self.store.update(
            state.goal_id,
            lambda current: current.with_accounting(
                turn_delta=1,
                token_delta=0,
                time_delta_s=elapsed_s,
                now=self._now(),
            ),
        )
        self._emit(
            "goal_turn_accounted",
            accounted,
            data={
                "turn_delta": 1,
                "token_delta": 0,
                "time_delta_s": elapsed_s,
                "error_type": type(exc).__name__,
            },
        )
        error_count = self._increment_error_count(accounted.goal_id)
        return self._transition(
            accounted,
            _error_status_for_policy(
                accounted.policy,
                error_count,
            ),
            reason=f"Goal continuation failed: {type(exc).__name__}",
        )

    def _apply_terminal_transition(
        self,
        state: GoalState,
        sdk_result: SDKResult,
        *,
        blocked_count_before_turn: int,
        permission_denials_before_turn: frozenset[_PermissionDenialKey],
    ) -> GoalState:
        new_permission_denials = self._new_permission_denials(
            state.goal_id,
            sdk_result.permission_denials,
            permission_denials_before_turn,
        )
        current = self.store.get(state.goal_id) or state
        if current.status != "active":
            return current
        if sdk_result.subtype == "error_max_budget_usd":
            return self._transition(
                current,
                "usage_limited",
                reason=_sdk_error_reason(sdk_result) or "provider usage limit reached",
            )
        if (
            new_permission_denials
            and current.policy.approvals.pause_on_pending_approval
        ):
            status = current.policy.approvals.unresolved_approval_status
            count = len(new_permission_denials)
            return self._transition(
                current,
                status,
                reason=(
                    "goal stopped on permission/approval boundary: "
                    f"{count} unresolved permission decision(s)"
                ),
            )
        if sdk_result.is_error:
            error_count = self._increment_error_count(current.goal_id)
            return self._transition(
                current,
                _error_status_for_policy(current.policy, error_count),
                reason=_sdk_error_reason(sdk_result) or f"turn failed: {sdk_result.subtype}",
            )
        self._reset_error_count(current.goal_id)
        limit_reason = _budget_limit_reason(current)
        if limit_reason is not None:
            return self._transition(current, "budget_limited", reason=limit_reason)
        if (
            current.blocked_turn_count > 0
            and current.blocked_turn_count == blocked_count_before_turn
        ):
            return self.store.update(
                current.goal_id,
                lambda latest: replace(
                    latest,
                    blocked_turn_count=0,
                    updated_at=self._now(),
                ),
            )
        return current

    def _transition(
        self,
        state: GoalState,
        status: GoalStatus,
        *,
        reason: str | None,
    ) -> GoalState:
        try:
            return self.store.update(
                state.goal_id,
                lambda current: current.with_status(
                    status,
                    reason=reason,
                    now=self._now(),
                ),
            )
        except GoalStateConflictError:
            latest = self.store.get(state.goal_id)
            if latest is None:
                raise
            return latest

    def _continuation_limit(self, requested: int | None) -> int:
        if requested is not None:
            if requested < 1:
                raise ValueError("max_continuations must be >= 1")
            return requested
        active = self.store.get_active_for_session(self.session_id)
        policy_limit = (
            active.policy.continuation.max_idle_continuations
            if active is not None
            else None
        )
        return policy_limit or self.config.max_continuations_per_run

    def _emit(
        self,
        event_type: GoalEventType,
        state: GoalState,
        *,
        data: dict[str, object] | None = None,
    ) -> None:
        emitter = self._event_emitter
        if emitter is None:
            return
        event = emitter.emit(event_type, state=state, data=data)
        self.store.append_event(event)

    def _now(self) -> float:
        return self.session.deps.clock.now()

    async def _apply_evaluator_policies(
        self,
        state: GoalState,
        sdk_result: SDKResult,
    ) -> GoalState:
        if state.status != "active":
            return state

        current = state
        missing_evaluator_name = self._missing_completion_evaluator_name(current)
        if missing_evaluator_name is not None:
            current = self._handle_evaluator_unavailable(
                current,
                evaluator_name=missing_evaluator_name,
            )
            if current.status != "active":
                return current
        completion_evaluator = self._completion_evaluator_for(current)
        if completion_evaluator is not None:
            try:
                evaluation = await completion_evaluator.evaluate(
                    state=current,
                    sdk_result=sdk_result,
                    session=self.session,
                )
            except Exception as exc:
                current = self._handle_evaluator_failure(
                    current,
                    evaluator_name=completion_evaluator.name,
                    exc=exc,
                )
                if current.status != "active":
                    return current
            else:
                current = self._apply_completion_evaluation(
                    current,
                    evaluation,
                    evaluator_name=completion_evaluator.name,
                )
                if current.status != "active":
                    return current

        progress_evaluator = self.config.progress_evaluator
        if progress_evaluator is not None:
            try:
                ledger = await progress_evaluator.evaluate(
                    state=current,
                    sdk_result=sdk_result,
                    session=self.session,
                )
            except Exception as exc:
                return self._handle_evaluator_failure(
                    current,
                    evaluator_name=progress_evaluator.name,
                    exc=exc,
                )
            current = self._apply_progress_ledger(
                current,
                ledger,
                evaluator_name=progress_evaluator.name,
            )

        return current

    def _completion_evaluator_for(
        self,
        state: GoalState,
    ) -> GoalCompletionEvaluator | None:
        evaluator = self.config.completion_evaluator
        if evaluator is None:
            return None
        requested_name = state.policy.completion.evaluator_name
        if requested_name is not None and evaluator.name != requested_name:
            return None
        return evaluator

    def _missing_completion_evaluator_name(self, state: GoalState) -> str | None:
        requested_name = state.policy.completion.evaluator_name
        if requested_name is None:
            return None
        evaluator = self.config.completion_evaluator
        if evaluator is not None and evaluator.name == requested_name:
            return None
        return requested_name

    def _apply_completion_evaluation(
        self,
        state: GoalState,
        evaluation: GoalCompletionEvaluation,
        *,
        evaluator_name: str,
    ) -> GoalState:
        updated = self.store.update(
            state.goal_id,
            lambda current: replace(
                current,
                metadata=_with_goal_metadata(
                    current,
                    goal_completion_evaluation={
                        "evaluator_name": evaluator_name,
                        **goal_completion_evaluation_to_dict(evaluation),
                    },
                ),
                last_reason=evaluation.reason,
                updated_at=self._now(),
            ),
        )
        if evaluation.is_complete:
            blockers = self._completion_blockers(updated, source="evaluator")
            if blockers:
                reason = (
                    "completion evaluator reported complete, but runtime "
                    f"invariants still block completion: {'; '.join(blockers)}"
                )
                return self.store.update(
                    updated.goal_id,
                    lambda current: replace(
                        current,
                        last_reason=reason,
                        updated_at=self._now(),
                    ),
                )
            return self._transition(updated, "complete", reason=evaluation.reason)
        if evaluation.is_blocked:
            return self._transition(updated, "blocked", reason=evaluation.reason)
        return updated

    def _apply_progress_ledger(
        self,
        state: GoalState,
        ledger: GoalProgressLedger,
        *,
        evaluator_name: str,
    ) -> GoalState:
        no_progress_count = (
            self._increment_no_progress_count(state.goal_id)
            if ledger.indicates_no_progress
            else 0
        )
        if not ledger.indicates_no_progress:
            self._reset_no_progress_count(state.goal_id)
        updated = self.store.update(
            state.goal_id,
            lambda current: replace(
                current,
                metadata=_with_goal_metadata(
                    current,
                    goal_progress_ledger={
                        "evaluator_name": evaluator_name,
                        **goal_progress_ledger_to_dict(ledger),
                    },
                    no_progress_turn_count=no_progress_count,
                ),
                last_reason=ledger.reason,
                updated_at=self._now(),
            ),
        )
        if ledger.request_satisfied:
            blockers = self._completion_blockers(updated, source="evaluator")
            if blockers:
                reason = (
                    "progress evaluator reported request satisfied, but runtime "
                    f"invariants still block completion: {'; '.join(blockers)}"
                )
                return self.store.update(
                    updated.goal_id,
                    lambda current: replace(
                        current,
                        last_reason=reason,
                        updated_at=self._now(),
                    ),
                )
            return self._transition(updated, "complete", reason=ledger.reason)

        threshold = updated.policy.blocking.max_consecutive_no_progress_turns
        if (
            ledger.indicates_no_progress
            and threshold is not None
            and no_progress_count >= threshold
        ):
            return self._transition(
                updated,
                "blocked",
                reason=(
                    "goal no-progress threshold reached: "
                    f"{no_progress_count} consecutive evaluation(s)"
                ),
            )
        return updated

    def _handle_evaluator_failure(
        self,
        state: GoalState,
        *,
        evaluator_name: str,
        exc: Exception,
    ) -> GoalState:
        reason = (
            f"Goal evaluator {evaluator_name} failed: "
            f"{type(exc).__name__}: {exc}"
        )
        updated = self.store.update(
            state.goal_id,
            lambda current: replace(
                current,
                metadata=_with_goal_metadata(
                    current,
                    goal_evaluator_error={
                        "evaluator_name": evaluator_name,
                        "error_type": type(exc).__name__,
                        "status_policy": self.config.evaluator_failure_status,
                    },
                ),
                last_reason=reason,
                updated_at=self._now(),
            ),
        )
        if self.config.evaluator_failure_status == "ignore":
            return updated
        return self._transition(
            updated,
            self.config.evaluator_failure_status,
            reason=reason,
        )

    def _handle_evaluator_unavailable(
        self,
        state: GoalState,
        *,
        evaluator_name: str,
    ) -> GoalState:
        reason = f"Goal evaluator {evaluator_name} is not available"
        updated = self.store.update(
            state.goal_id,
            lambda current: replace(
                current,
                metadata=_with_goal_metadata(
                    current,
                    goal_evaluator_error={
                        "evaluator_name": evaluator_name,
                        "error_type": "EvaluatorUnavailable",
                        "status_policy": self.config.evaluator_failure_status,
                    },
                ),
                last_reason=reason,
                updated_at=self._now(),
            ),
        )
        if self.config.evaluator_failure_status == "ignore":
            return updated
        return self._transition(
            updated,
            self.config.evaluator_failure_status,
            reason=reason,
        )

    def _baseline_permission_denials(self, goal_id: str) -> None:
        self._seen_permission_denials[goal_id] = _permission_denial_keys(
            _session_permission_denials(self.session)
        )

    def _permission_denials_before_turn(
        self,
        goal_id: str,
    ) -> frozenset[_PermissionDenialKey]:
        keys = self._seen_permission_denials.get(goal_id, frozenset())
        live_keys = _permission_denial_keys(_session_permission_denials(self.session))
        combined = keys | live_keys
        self._seen_permission_denials[goal_id] = combined
        return combined

    def _new_permission_denials(
        self,
        goal_id: str,
        denials: Sequence[PermissionDenial],
        before_turn: frozenset[_PermissionDenialKey],
    ) -> tuple[PermissionDenial, ...]:
        all_keys = _permission_denial_keys(denials)
        self._seen_permission_denials[goal_id] = before_turn | all_keys
        return tuple(
            denial
            for denial in denials
            if _permission_denial_key(denial) not in before_turn
        )

    def _completion_blockers(
        self,
        state: GoalState,
        *,
        source: Literal["model", "evaluator"],
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        if source == "model" and not state.policy.completion.allow_model_complete:
            blockers.append("model-reported completion is disabled by goal policy")
        if (
            source == "model"
            and state.policy.completion.require_acceptance_checks
            and state.spec.acceptance_checks
        ):
            blockers.append(
                "acceptance checks require an evaluator or product verification"
            )
        if state.policy.completion.require_pending_tasks_clear:
            for task_id in state.pending_task_ids:
                task = self.session.deps.task_store.tasks.get(task_id)
                if task is None:
                    blockers.append(f"required task {task_id} is missing")
                elif task.status != "completed":
                    blockers.append(
                        f"required task {task_id} is not completed (status={task.status})"
                    )
        return tuple(blockers)

    def _record_blocked_report(
        self,
        state: GoalState,
        *,
        reason: str,
    ) -> GoalUpdatePolicyDecision:
        threshold = state.policy.blocking.blocked_audit_turns

        def update(current: GoalState) -> GoalState:
            count = current.blocked_turn_count + 1
            status: GoalStatus = "blocked" if count >= threshold else "active"
            return replace(
                current,
                status=status,
                blocked_turn_count=count,
                last_reason=reason,
                updated_at=self._now(),
            )

        updated = self.store.update(state.goal_id, update)
        if updated.status == "blocked":
            return GoalUpdatePolicyDecision(
                proceed=False,
                content=(
                    f"Goal {updated.goal_id} updated to status=blocked. "
                    f"Reason: {updated.last_reason or 'not provided'}"
                ),
                updated_state=updated,
            )
        return GoalUpdatePolicyDecision(
            proceed=False,
            content=(
                f"Blocked report recorded for goal {updated.goal_id} "
                f"({updated.blocked_turn_count}/{threshold}). Continue the audit; "
                "do not stop autonomous work until the configured blocked threshold "
                "is reached."
            ),
            updated_state=updated,
        )

    def _increment_error_count(self, goal_id: str) -> int:
        next_count = self._consecutive_errors.get(goal_id, 0) + 1
        self._consecutive_errors[goal_id] = next_count
        return next_count

    def _reset_error_count(self, goal_id: str) -> None:
        self._consecutive_errors[goal_id] = 0

    def _increment_no_progress_count(self, goal_id: str) -> int:
        next_count = self._consecutive_no_progress.get(goal_id, 0) + 1
        self._consecutive_no_progress[goal_id] = next_count
        return next_count

    def _reset_no_progress_count(self, goal_id: str) -> None:
        self._consecutive_no_progress[goal_id] = 0


GoalService = GoalRuntime


def _merge_goal_tools(
    tools: Sequence[Tool],
    goal_tools: Sequence[Tool],
) -> tuple[Tool, ...]:
    retained = tuple(tool for tool in tools if tool.name not in _GOAL_TOOL_NAMES)
    return (*retained, *tuple(goal_tools))


def _usage_total_tokens(usage: UsageTotals) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )


def _session_usage_total_tokens(session: GoalRuntimeSession) -> int:
    usage = session.engine._total_usage if hasattr(session.engine, "_total_usage") else None
    if isinstance(usage, UsageTotals):
        return _usage_total_tokens(usage)
    return 0


def _session_permission_denials(
    session: GoalRuntimeSession,
) -> tuple[PermissionDenial, ...]:
    raw_denials: object = getattr(session.engine, "_permission_denials", ())
    if not isinstance(raw_denials, Sequence):
        return ()
    denials = cast(Sequence[object], raw_denials)
    return tuple(denial for denial in denials if isinstance(denial, PermissionDenial))


def _permission_denial_keys(
    denials: Sequence[PermissionDenial],
) -> frozenset[_PermissionDenialKey]:
    return frozenset(_permission_denial_key(denial) for denial in denials)


def _permission_denial_key(denial: PermissionDenial) -> _PermissionDenialKey:
    return (denial.tool_use_id, denial.tool_name, denial.reason)


def _with_goal_metadata(
    state: GoalState,
    **updates: object,
) -> Mapping[str, FrozenJson]:
    metadata: dict[str, object] = {**dict(state.metadata), **updates}
    return cast(Mapping[str, FrozenJson], metadata)


def _budget_limit_reason(state: GoalState) -> str | None:
    budget = state.policy.budget
    if budget.max_turns is not None and state.turn_count >= budget.max_turns:
        return f"goal max_turns reached: {budget.max_turns}"
    token_budget = state.token_budget
    if token_budget is not None and state.tokens_used >= token_budget:
        return f"goal token_budget reached: {token_budget}"
    if (
        budget.wall_clock_budget_s is not None
        and state.time_used_s >= budget.wall_clock_budget_s
    ):
        return f"goal wall_clock_budget_s reached: {budget.wall_clock_budget_s}"
    return None


def _error_status_for_policy(policy: GoalPolicy, consecutive_errors: int) -> GoalStatus:
    limit = policy.blocking.max_consecutive_errors
    if limit is not None and consecutive_errors >= limit:
        return "failed"
    return "blocked"


def _sdk_error_reason(result: SDKResult) -> str | None:
    if result.errors:
        return "; ".join(result.errors)
    if result.subtype != "success":
        return result.subtype
    return None


def _stop_reason_for_state(state: GoalState) -> GoalRuntimeStopReason:
    if state.status == "active":
        return "continued"
    if state.status == "complete":
        return "completed"
    if state.status == "budget_limited":
        return "budget_limited"
    if state.status == "usage_limited":
        return "usage_limited"
    if state.status == "blocked":
        return "blocked"
    if state.status == "paused":
        return "paused"
    if state.status == "cancelled":
        return "cancelled"
    return "failed"


def _new_goal_id() -> str:
    return f"goal_{uuid.uuid4().hex[:16]}"


def _raise_if_session_active(session: GoalRuntimeSession) -> None:
    if bool(getattr(session, "_active_turn", False)):
        raise GoalRuntimeStateError(
            "GoalRuntime cannot install goal surfaces while a session turn is active"
        )


__all__ = [
    "GoalRuntime",
    "GoalRuntimeConfig",
    "GoalRuntimeError",
    "GoalRuntimeResult",
    "GoalRuntimeSession",
    "GoalRuntimeStateError",
    "GoalRuntimeStopReason",
    "GoalService",
]
