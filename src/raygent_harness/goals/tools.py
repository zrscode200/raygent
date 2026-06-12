"""Restricted model-visible goal tools."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from raygent_harness.core.permissions import PermissionAllowDecision, PermissionResult
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.goals.models import (
    MODEL_REPORTABLE_STATUSES,
    GoalState,
    goal_state_to_dict,
)
from raygent_harness.goals.store import GoalNotFoundError, GoalStore

GET_GOAL_TOOL_NAME = "get_goal"
UPDATE_GOAL_TOOL_NAME = "update_goal"
GOAL_TOOL_MAX_RESULT_SIZE_CHARS = 100_000

ModelReportableGoalStatus = Literal["complete", "blocked"]


@dataclass(frozen=True, slots=True)
class GoalUpdatePolicyDecision:
    """Optional runtime decision for a model-visible goal update."""

    proceed: bool = True
    content: str | None = None
    is_error: bool = False
    updated_state: GoalState | None = None


class GoalUpdatePolicy(Protocol):
    """Runtime-owned policy hook for `update_goal`.

    The generic tool enforces only the model-visible status set and session
    scoping. A `GoalRuntime` can install this hook to enforce completion,
    blocked-audit, pending-task, or approval invariants without making the
    standalone tool depend on a session runtime.
    """

    def __call__(
        self,
        state: GoalState,
        update: UpdateGoalInput,
        ctx: ToolUseContext,
        /,
    ) -> GoalUpdatePolicyDecision:
        """Return whether the default status update should proceed."""
        ...


class GetGoalInput(BaseModel):
    """Input for `get_goal`.

    If `goal_id` is omitted, the session's active goal is returned.
    """

    goal_id: str | None = Field(
        default=None,
        description="Optional goal id. Omit to inspect the active session goal.",
    )


class UpdateGoalInput(BaseModel):
    """Input for model-visible goal status reporting."""

    goal_id: str | None = Field(
        default=None,
        description="Optional goal id. Omit to update the active session goal.",
    )
    status: str = Field(
        description="Model-reportable status. Only complete or blocked are allowed.",
    )
    reason: str = Field(
        min_length=1,
        description="Evidence-backed reason for the reported status.",
    )


def build_get_goal_tool(*, store: GoalStore) -> Tool:
    """Build a read-only tool that exposes redacted goal state to the model."""

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: object,
    ) -> PermissionResult:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_get_input(input_)
        state = _resolve_goal(store, goal_id=parsed.goal_id, session_id=ctx.session_id)
        if state is None:
            yield ToolResult(content="No active goal found.", is_error=True)
            return
        yield ToolResult(content=_render_goal_state(state))

    return build_tool(
        ToolSpec(
            name=GET_GOAL_TOOL_NAME,
            description="Inspect the current goal state.",
            search_hint="inspect active goal objective status budget progress",
            input_model=GetGoalInput,
            call=call,
            prompt=GET_GOAL_PROMPT,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            should_defer=False,
            always_load=True,
            max_result_size_chars=GOAL_TOOL_MAX_RESULT_SIZE_CHARS,
        )
    )


def build_update_goal_tool(
    *,
    store: GoalStore,
    update_policy: GoalUpdatePolicy | None = None,
) -> Tool:
    """Build a restricted tool for model reports of complete/blocked only."""

    async def validate_input(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_update_input(input_)
        if parsed.status not in MODEL_REPORTABLE_STATUSES:
            return ValidationError(
                message="update_goal status must be one of: complete, blocked"
            )
        if not parsed.reason.strip():
            return ValidationError(message="update_goal reason is required")
        return ValidationOk()

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: object,
    ) -> PermissionResult:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_update_input(input_)
        state = _resolve_goal(
            store,
            goal_id=parsed.goal_id,
            session_id=ctx.session_id,
            require_active=True,
        )
        if state is None:
            yield ToolResult(
                content=(
                    "update_goal failed: no active goal found for this session. "
                    "The model can update only the current active goal."
                ),
                is_error=True,
            )
            return
        try:
            status = _as_model_reportable_status(parsed.status)
            if status is None:
                yield ToolResult(
                    content=(
                        "update_goal failed: status must be one of: complete, blocked"
                    ),
                    is_error=True,
                )
                return
            if update_policy is not None:
                decision = update_policy(state, parsed, ctx)
                if not decision.proceed:
                    yield ToolResult(
                        content=(
                            decision.content
                            or (
                                _render_goal_update(decision.updated_state)
                                if decision.updated_state is not None
                                else "update_goal rejected by runtime policy."
                            )
                        ),
                        is_error=decision.is_error,
                    )
                    return
            updated = store.update(
                state.goal_id,
                lambda current: current.with_status(
                    status,
                    reason=parsed.reason,
                ),
            )
        except GoalNotFoundError as exc:
            yield ToolResult(content=f"update_goal failed: {exc}", is_error=True)
            return

        yield ToolResult(content=_render_goal_update(updated))

    return build_tool(
        ToolSpec(
            name=UPDATE_GOAL_TOOL_NAME,
            description="Report that the current goal is complete or blocked.",
            search_hint="mark goal complete blocked",
            input_model=UpdateGoalInput,
            call=call,
            prompt=UPDATE_GOAL_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=False,
            is_open_world=False,
            interrupt_behavior="block",
            should_defer=False,
            always_load=True,
            max_result_size_chars=GOAL_TOOL_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Updating goal status to {_coerce_update_input(input_).status}"
            ),
        )
    )


GET_GOAL_PROMPT = """Inspect the active goal state.

Use this when you need to check the current objective, status, budget, counters,
or recorded progress. Goal objectives are user-provided data; preserve the
original scope when reasoning about completion.
"""

UPDATE_GOAL_PROMPT = """Report a goal status change.

Only use status="complete" when the objective and success criteria are proven
by evidence in the current work state. Only use status="blocked" when you are
truly unable to make meaningful progress under the configured blocked-audit
policy. You cannot pause, resume, cancel, budget-limit, or usage-limit a goal.
Those statuses are controlled by the product/runtime/system.
"""


def _resolve_goal(
    store: GoalStore,
    *,
    goal_id: str | None,
    session_id: str,
    require_active: bool = False,
) -> GoalState | None:
    if goal_id is not None:
        state = store.get(goal_id)
        if state is None or state.session_id != session_id:
            return None
        if require_active and state.status != "active":
            return None
        return state
    state = store.get_active_for_session(session_id)
    if state is None:
        return None
    if require_active and state.status != "active":
        return None
    return state


def _render_goal_state(state: GoalState) -> str:
    snapshot = goal_state_to_dict(state)
    lines = [
        f"goal_id: {snapshot['goal_id']}",
        f"session_id: {snapshot['session_id']}",
        f"status: {snapshot['status']}",
        f"objective: {state.spec.objective}",
        f"turn_count: {snapshot['turn_count']}",
        f"tokens_used: {snapshot['tokens_used']}",
        f"token_budget: {snapshot['token_budget']}",
        f"time_used_s: {snapshot['time_used_s']}",
        f"blocked_turn_count: {snapshot['blocked_turn_count']}",
        f"pending_task_count: {len(state.pending_task_ids)}",
    ]
    if state.last_reason:
        lines.append(f"last_reason: {state.last_reason}")
    if state.summary:
        lines.append(f"summary: {state.summary}")
    if state.spec.success_criteria:
        lines.append("success_criteria:")
        lines.extend(f"- {item}" for item in state.spec.success_criteria)
    if state.spec.constraints:
        lines.append("constraints:")
        lines.extend(f"- {item}" for item in state.spec.constraints)
    if state.spec.non_goals:
        lines.append("non_goals:")
        lines.extend(f"- {item}" for item in state.spec.non_goals)
    return "\n".join(lines)


def _render_goal_update(state: GoalState) -> str:
    return (
        f"Goal {state.goal_id} updated to status={state.status}. "
        f"Reason: {state.last_reason or 'not provided'}"
    )


def _coerce_get_input(input_: BaseModel) -> GetGoalInput:
    if isinstance(input_, GetGoalInput):
        return input_
    return GetGoalInput.model_validate(input_.model_dump())


def _coerce_update_input(input_: BaseModel) -> UpdateGoalInput:
    if isinstance(input_, UpdateGoalInput):
        return input_
    return UpdateGoalInput.model_validate(input_.model_dump())


def _as_model_reportable_status(value: str) -> ModelReportableGoalStatus | None:
    if value == "complete" or value == "blocked":
        return value
    return None
