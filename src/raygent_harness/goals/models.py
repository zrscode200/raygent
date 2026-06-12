"""Goal-runner control-plane models.

The goal state is intentionally separate from transcript messages. Transcript
content is evidence; these dataclasses are the durable control plane that later
runtime waves will use for continuation, budgets, and lifecycle transitions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from raygent_harness.core.model_types import FrozenJson, freeze_json

GoalStatus = Literal[
    "active",
    "paused",
    "blocked",
    "usage_limited",
    "budget_limited",
    "complete",
    "cancelled",
    "failed",
]
"""Persistent goal lifecycle status.

Model-visible tooling may report only ``complete`` or ``blocked``. Product,
runtime, and system paths own pause/resume/cancel/budget/usage/error statuses.
"""

MODEL_REPORTABLE_STATUSES: frozenset[GoalStatus] = frozenset({"complete", "blocked"})
NON_CONTINUING_STATUSES: frozenset[GoalStatus] = frozenset(
    {
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
        "cancelled",
        "failed",
    }
)
_GOAL_STATUSES: frozenset[str] = frozenset(
    {
        "active",
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
        "cancelled",
        "failed",
    }
)
_GOAL_PLAN_STEP_STATUSES: frozenset[str] = frozenset(
    {"pending", "active", "complete", "blocked", "cancelled"}
)
_UNRESOLVED_APPROVAL_STATUSES: frozenset[str] = frozenset({"blocked", "failed"})


def _empty_metadata() -> Mapping[str, FrozenJson]:
    return {}


@dataclass(frozen=True, slots=True)
class GoalOutputSpec:
    """Expected artifact/output description for a goal."""

    name: str
    description: str
    required: bool = True
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "GoalOutputSpec.name")
        _require_non_empty(self.description, "GoalOutputSpec.description")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalAcceptanceCheck:
    """A named acceptance condition that can be verified by policy/runtime."""

    name: str
    instruction: str
    required: bool = True
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "GoalAcceptanceCheck.name")
        _require_non_empty(self.instruction, "GoalAcceptanceCheck.instruction")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """User-provided goal specification.

    ``objective`` is user data. Later prompt builders must quote or label it as
    such rather than treating it as higher-priority system instruction text.
    """

    objective: str
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    expected_outputs: tuple[GoalOutputSpec, ...] = ()
    acceptance_checks: tuple[GoalAcceptanceCheck, ...] = ()
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.objective, "GoalSpec.objective")
        object.__setattr__(self, "success_criteria", tuple(self.success_criteria))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "non_goals", tuple(self.non_goals))
        object.__setattr__(self, "expected_outputs", tuple(self.expected_outputs))
        object.__setattr__(self, "acceptance_checks", tuple(self.acceptance_checks))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalBudgetPolicy:
    """Budget limits enforced by the goal runtime."""

    max_turns: int | None = None
    token_budget: int | None = None
    wall_clock_budget_s: float | None = None

    def __post_init__(self) -> None:
        _require_positive_or_none(self.max_turns, "GoalBudgetPolicy.max_turns")
        _require_positive_or_none(self.token_budget, "GoalBudgetPolicy.token_budget")
        _require_positive_or_none(
            self.wall_clock_budget_s,
            "GoalBudgetPolicy.wall_clock_budget_s",
        )


@dataclass(frozen=True, slots=True)
class GoalBlockingPolicy:
    """Rules for when repeated blocked reports become authoritative."""

    blocked_audit_turns: int = 3
    max_consecutive_no_progress_turns: int | None = None
    max_consecutive_errors: int | None = None

    def __post_init__(self) -> None:
        _require_positive(self.blocked_audit_turns, "blocked_audit_turns")
        _require_positive_or_none(
            self.max_consecutive_no_progress_turns,
            "max_consecutive_no_progress_turns",
        )
        _require_positive_or_none(self.max_consecutive_errors, "max_consecutive_errors")


@dataclass(frozen=True, slots=True)
class GoalCompletionPolicy:
    """Completion authority and runtime invariant policy."""

    allow_model_complete: bool = True
    require_pending_tasks_clear: bool = True
    require_acceptance_checks: bool = False
    evaluator_name: str | None = None


@dataclass(frozen=True, slots=True)
class GoalContinuationPolicy:
    """When the runtime may automatically continue an active goal."""

    auto_continue_when_idle: bool = True
    continue_after_compaction: bool = True
    max_idle_continuations: int | None = None

    def __post_init__(self) -> None:
        _require_positive_or_none(
            self.max_idle_continuations,
            "GoalContinuationPolicy.max_idle_continuations",
        )


@dataclass(frozen=True, slots=True)
class GoalApprovalPolicy:
    """Behavior when tools or runtime need user/HITL approval."""

    pause_on_pending_approval: bool = True
    unresolved_approval_status: Literal["blocked", "failed"] = "blocked"

    def __post_init__(self) -> None:
        _require_literal(
            self.unresolved_approval_status,
            _UNRESOLVED_APPROVAL_STATUSES,
            "GoalApprovalPolicy.unresolved_approval_status",
        )


@dataclass(frozen=True, slots=True)
class GoalCompactionPolicy:
    """Goal-state preservation requirements across compaction."""

    preserve_goal_state: bool = True
    checkpoint_before_compaction: bool = True


@dataclass(frozen=True, slots=True)
class GoalPolicy:
    """Composable goal policies that affect runtime behavior."""

    budget: GoalBudgetPolicy = field(default_factory=GoalBudgetPolicy)
    blocking: GoalBlockingPolicy = field(default_factory=GoalBlockingPolicy)
    completion: GoalCompletionPolicy = field(default_factory=GoalCompletionPolicy)
    continuation: GoalContinuationPolicy = field(default_factory=GoalContinuationPolicy)
    approvals: GoalApprovalPolicy = field(default_factory=GoalApprovalPolicy)
    compaction: GoalCompactionPolicy = field(default_factory=GoalCompactionPolicy)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalCheckpoint:
    """Durable checkpoint marker for resume/compaction waves."""

    checkpoint_id: str
    summary: str
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.checkpoint_id, "GoalCheckpoint.checkpoint_id")
        _require_non_empty(self.summary, "GoalCheckpoint.summary")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalPlanStep:
    """Optional progress-planning slot preserved for later ledger waves."""

    step_id: str
    description: str
    status: Literal["pending", "active", "complete", "blocked", "cancelled"] = "pending"
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.step_id, "GoalPlanStep.step_id")
        _require_non_empty(self.description, "GoalPlanStep.description")
        _require_literal(
            self.status,
            _GOAL_PLAN_STEP_STATUSES,
            "GoalPlanStep.status",
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalArtifact:
    """Metadata-only artifact reference produced while pursuing a goal."""

    artifact_id: str
    kind: str
    uri: str | None = None
    description: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_id, "GoalArtifact.artifact_id")
        _require_non_empty(self.kind, "GoalArtifact.kind")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalState:
    """Durable goal control-plane snapshot."""

    goal_id: str
    session_id: str
    status: GoalStatus
    spec: GoalSpec
    policy: GoalPolicy = field(default_factory=GoalPolicy)
    turn_count: int = 0
    tokens_used: int = 0
    token_budget: int | None = None
    time_used_s: float = 0.0
    last_reason: str | None = None
    blocked_turn_count: int = 0
    summary: str | None = None
    checkpoints: tuple[GoalCheckpoint, ...] = ()
    plan_steps: tuple[GoalPlanStep, ...] = ()
    artifacts: tuple[GoalArtifact, ...] = ()
    pending_task_ids: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Mapping[str, FrozenJson] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _require_non_empty(self.goal_id, "GoalState.goal_id")
        _require_non_empty(self.session_id, "GoalState.session_id")
        _require_literal(self.status, _GOAL_STATUSES, "GoalState.status")
        if self.turn_count < 0:
            raise ValueError("GoalState.turn_count must be >= 0")
        if self.tokens_used < 0:
            raise ValueError("GoalState.tokens_used must be >= 0")
        _require_positive_or_none(self.token_budget, "GoalState.token_budget")
        if self.time_used_s < 0:
            raise ValueError("GoalState.time_used_s must be >= 0")
        if self.blocked_turn_count < 0:
            raise ValueError("GoalState.blocked_turn_count must be >= 0")
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "plan_steps", tuple(self.plan_steps))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "pending_task_ids", tuple(self.pending_task_ids))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def is_continuable(self) -> bool:
        return self.status == "active"

    @property
    def is_terminal(self) -> bool:
        return self.status in {"complete", "cancelled", "failed", "budget_limited"}

    def with_status(
        self,
        status: GoalStatus,
        *,
        reason: str | None = None,
        now: float | None = None,
    ) -> GoalState:
        """Return a new state with lifecycle status updated."""

        blocked_count = (
            self.blocked_turn_count + 1
            if status == "blocked" and self.status == "blocked"
            else 1
            if status == "blocked"
            else 0
        )
        return replace(
            self,
            status=status,
            last_reason=reason,
            blocked_turn_count=blocked_count,
            updated_at=time.time() if now is None else now,
        )

    def with_accounting(
        self,
        *,
        turn_delta: int = 0,
        token_delta: int = 0,
        time_delta_s: float = 0.0,
        now: float | None = None,
    ) -> GoalState:
        """Return a new state with monotonic usage counters advanced."""

        if turn_delta < 0:
            raise ValueError("turn_delta must be >= 0")
        if token_delta < 0:
            raise ValueError("token_delta must be >= 0")
        if time_delta_s < 0:
            raise ValueError("time_delta_s must be >= 0")
        return replace(
            self,
            turn_count=self.turn_count + turn_delta,
            tokens_used=self.tokens_used + token_delta,
            time_used_s=self.time_used_s + time_delta_s,
            updated_at=time.time() if now is None else now,
        )


def create_goal_state(
    *,
    goal_id: str,
    session_id: str,
    spec: GoalSpec,
    policy: GoalPolicy | None = None,
    now: float | None = None,
) -> GoalState:
    """Create a fresh active goal state."""

    created_at = time.time() if now is None else now
    resolved_policy = policy or GoalPolicy()
    return GoalState(
        goal_id=goal_id,
        session_id=session_id,
        status="active",
        spec=spec,
        policy=resolved_policy,
        token_budget=resolved_policy.budget.token_budget,
        created_at=created_at,
        updated_at=created_at,
    )


def goal_state_to_dict(state: GoalState) -> dict[str, FrozenJson]:
    """Return a JSON-serializable immutable snapshot dictionary."""

    frozen = freeze_json(
        {
            "goal_id": state.goal_id,
            "session_id": state.session_id,
            "status": state.status,
            "spec": _goal_spec_to_dict(state.spec),
            "policy": _goal_policy_to_dict(state.policy),
            "turn_count": state.turn_count,
            "tokens_used": state.tokens_used,
            "token_budget": state.token_budget,
            "time_used_s": state.time_used_s,
            "last_reason": state.last_reason,
            "blocked_turn_count": state.blocked_turn_count,
            "summary": state.summary,
            "checkpoints": tuple(
                _goal_checkpoint_to_dict(checkpoint)
                for checkpoint in state.checkpoints
            ),
            "plan_steps": tuple(_goal_plan_step_to_dict(step) for step in state.plan_steps),
            "artifacts": tuple(
                _goal_artifact_to_dict(artifact) for artifact in state.artifacts
            ),
            "pending_task_ids": state.pending_task_ids,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "metadata": dict(state.metadata),
        }
    )
    if not isinstance(frozen, Mapping):
        raise TypeError("GoalState did not serialize to an object")
    return dict(cast(Mapping[str, FrozenJson], frozen))


def goal_state_from_dict(data: Mapping[str, object]) -> GoalState:
    """Rehydrate a goal state from a JSON-like dictionary."""

    spec_data = _expect_mapping(data.get("spec"), "spec")
    policy_data = _expect_mapping(data.get("policy", {}), "policy")
    return GoalState(
        goal_id=str(data["goal_id"]),
        session_id=str(data["session_id"]),
        status=cast(
            GoalStatus,
            _literal_from_object(data["status"], _GOAL_STATUSES, "status"),
        ),
        spec=_goal_spec_from_mapping(spec_data),
        policy=_goal_policy_from_mapping(policy_data),
        turn_count=_int_from_object(data.get("turn_count", 0), "turn_count"),
        tokens_used=_int_from_object(data.get("tokens_used", 0), "tokens_used"),
        token_budget=_optional_int(data.get("token_budget")),
        time_used_s=_float_from_object(data.get("time_used_s", 0.0), "time_used_s"),
        last_reason=_optional_str(data.get("last_reason")),
        blocked_turn_count=_int_from_object(
            data.get("blocked_turn_count", 0),
            "blocked_turn_count",
        ),
        summary=_optional_str(data.get("summary")),
        checkpoints=tuple(
            _goal_checkpoint_from_mapping(item)
            for item in _mapping_sequence(data.get("checkpoints", ()), "checkpoints")
        ),
        plan_steps=tuple(
            _goal_plan_step_from_mapping(item)
            for item in _mapping_sequence(data.get("plan_steps", ()), "plan_steps")
        ),
        artifacts=tuple(
            _goal_artifact_from_mapping(item)
            for item in _mapping_sequence(data.get("artifacts", ()), "artifacts")
        ),
        pending_task_ids=_string_tuple(data.get("pending_task_ids", ()), "pending_task_ids"),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        updated_at=_float_from_object(data.get("updated_at", 0.0), "updated_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_spec_to_dict(spec: GoalSpec) -> dict[str, object]:
    return {
        "objective": spec.objective,
        "success_criteria": spec.success_criteria,
        "constraints": spec.constraints,
        "non_goals": spec.non_goals,
        "expected_outputs": tuple(
            _goal_output_spec_to_dict(output) for output in spec.expected_outputs
        ),
        "acceptance_checks": tuple(
            _goal_acceptance_check_to_dict(check) for check in spec.acceptance_checks
        ),
        "metadata": dict(spec.metadata),
    }


def _goal_spec_from_mapping(data: Mapping[str, object]) -> GoalSpec:
    return GoalSpec(
        objective=str(data["objective"]),
        success_criteria=_string_tuple(data.get("success_criteria", ()), "success_criteria"),
        constraints=_string_tuple(data.get("constraints", ()), "constraints"),
        non_goals=_string_tuple(data.get("non_goals", ()), "non_goals"),
        expected_outputs=tuple(
            _goal_output_spec_from_mapping(item)
            for item in _mapping_sequence(
                data.get("expected_outputs", ()),
                "expected_outputs",
            )
        ),
        acceptance_checks=tuple(
            _goal_acceptance_check_from_mapping(item)
            for item in _mapping_sequence(
                data.get("acceptance_checks", ()),
                "acceptance_checks",
            )
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_output_spec_to_dict(output: GoalOutputSpec) -> dict[str, object]:
    return {
        "name": output.name,
        "description": output.description,
        "required": output.required,
        "metadata": dict(output.metadata),
    }


def _goal_output_spec_from_mapping(data: Mapping[str, object]) -> GoalOutputSpec:
    return GoalOutputSpec(
        name=str(data["name"]),
        description=str(data["description"]),
        required=bool(data.get("required", True)),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_acceptance_check_to_dict(check: GoalAcceptanceCheck) -> dict[str, object]:
    return {
        "name": check.name,
        "instruction": check.instruction,
        "required": check.required,
        "metadata": dict(check.metadata),
    }


def _goal_acceptance_check_from_mapping(
    data: Mapping[str, object],
) -> GoalAcceptanceCheck:
    return GoalAcceptanceCheck(
        name=str(data["name"]),
        instruction=str(data["instruction"]),
        required=bool(data.get("required", True)),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_policy_to_dict(policy: GoalPolicy) -> dict[str, object]:
    return {
        "budget": {
            "max_turns": policy.budget.max_turns,
            "token_budget": policy.budget.token_budget,
            "wall_clock_budget_s": policy.budget.wall_clock_budget_s,
        },
        "blocking": {
            "blocked_audit_turns": policy.blocking.blocked_audit_turns,
            "max_consecutive_no_progress_turns": (
                policy.blocking.max_consecutive_no_progress_turns
            ),
            "max_consecutive_errors": policy.blocking.max_consecutive_errors,
        },
        "completion": {
            "allow_model_complete": policy.completion.allow_model_complete,
            "require_pending_tasks_clear": (
                policy.completion.require_pending_tasks_clear
            ),
            "require_acceptance_checks": policy.completion.require_acceptance_checks,
            "evaluator_name": policy.completion.evaluator_name,
        },
        "continuation": {
            "auto_continue_when_idle": policy.continuation.auto_continue_when_idle,
            "continue_after_compaction": policy.continuation.continue_after_compaction,
            "max_idle_continuations": policy.continuation.max_idle_continuations,
        },
        "approvals": {
            "pause_on_pending_approval": policy.approvals.pause_on_pending_approval,
            "unresolved_approval_status": policy.approvals.unresolved_approval_status,
        },
        "compaction": {
            "preserve_goal_state": policy.compaction.preserve_goal_state,
            "checkpoint_before_compaction": policy.compaction.checkpoint_before_compaction,
        },
        "metadata": dict(policy.metadata),
    }


def _goal_policy_from_mapping(data: Mapping[str, object]) -> GoalPolicy:
    return GoalPolicy(
        budget=_goal_budget_policy_from_mapping(
            _expect_mapping(data.get("budget", {}), "budget")
        ),
        blocking=_goal_blocking_policy_from_mapping(
            _expect_mapping(data.get("blocking", {}), "blocking")
        ),
        completion=_goal_completion_policy_from_mapping(
            _expect_mapping(data.get("completion", {}), "completion")
        ),
        continuation=_goal_continuation_policy_from_mapping(
            _expect_mapping(data.get("continuation", {}), "continuation")
        ),
        approvals=_goal_approval_policy_from_mapping(
            _expect_mapping(data.get("approvals", {}), "approvals")
        ),
        compaction=_goal_compaction_policy_from_mapping(
            _expect_mapping(data.get("compaction", {}), "compaction")
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_budget_policy_from_mapping(data: Mapping[str, object]) -> GoalBudgetPolicy:
    return GoalBudgetPolicy(
        max_turns=_optional_int(data.get("max_turns")),
        token_budget=_optional_int(data.get("token_budget")),
        wall_clock_budget_s=_optional_float(data.get("wall_clock_budget_s")),
    )


def _goal_blocking_policy_from_mapping(
    data: Mapping[str, object],
) -> GoalBlockingPolicy:
    return GoalBlockingPolicy(
        blocked_audit_turns=_int_from_object(
            data.get("blocked_audit_turns", 3),
            "blocked_audit_turns",
        ),
        max_consecutive_no_progress_turns=_optional_int(
            data.get("max_consecutive_no_progress_turns")
        ),
        max_consecutive_errors=_optional_int(data.get("max_consecutive_errors")),
    )


def _goal_completion_policy_from_mapping(
    data: Mapping[str, object],
) -> GoalCompletionPolicy:
    return GoalCompletionPolicy(
        allow_model_complete=bool(data.get("allow_model_complete", True)),
        require_pending_tasks_clear=bool(data.get("require_pending_tasks_clear", True)),
        require_acceptance_checks=bool(data.get("require_acceptance_checks", False)),
        evaluator_name=_optional_str(data.get("evaluator_name")),
    )


def _goal_continuation_policy_from_mapping(
    data: Mapping[str, object],
) -> GoalContinuationPolicy:
    return GoalContinuationPolicy(
        auto_continue_when_idle=bool(data.get("auto_continue_when_idle", True)),
        continue_after_compaction=bool(data.get("continue_after_compaction", True)),
        max_idle_continuations=_optional_int(data.get("max_idle_continuations")),
    )


def _goal_approval_policy_from_mapping(data: Mapping[str, object]) -> GoalApprovalPolicy:
    return GoalApprovalPolicy(
        pause_on_pending_approval=bool(data.get("pause_on_pending_approval", True)),
        unresolved_approval_status=cast(
            Literal["blocked", "failed"],
            _literal_from_object(
                data.get("unresolved_approval_status", "blocked"),
                _UNRESOLVED_APPROVAL_STATUSES,
                "unresolved_approval_status",
            ),
        ),
    )


def _goal_compaction_policy_from_mapping(
    data: Mapping[str, object],
) -> GoalCompactionPolicy:
    return GoalCompactionPolicy(
        preserve_goal_state=bool(data.get("preserve_goal_state", True)),
        checkpoint_before_compaction=bool(data.get("checkpoint_before_compaction", True)),
    )


def _goal_checkpoint_to_dict(checkpoint: GoalCheckpoint) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "summary": checkpoint.summary,
        "created_at": checkpoint.created_at,
        "metadata": dict(checkpoint.metadata),
    }


def _goal_checkpoint_from_mapping(data: Mapping[str, object]) -> GoalCheckpoint:
    return GoalCheckpoint(
        checkpoint_id=str(data["checkpoint_id"]),
        summary=str(data["summary"]),
        created_at=_float_from_object(data.get("created_at", 0.0), "created_at"),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_plan_step_to_dict(step: GoalPlanStep) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "description": step.description,
        "status": step.status,
        "metadata": dict(step.metadata),
    }


def _goal_plan_step_from_mapping(data: Mapping[str, object]) -> GoalPlanStep:
    return GoalPlanStep(
        step_id=str(data["step_id"]),
        description=str(data["description"]),
        status=cast(
            Literal["pending", "active", "complete", "blocked", "cancelled"],
            _literal_from_object(
                data.get("status", "pending"),
                _GOAL_PLAN_STEP_STATUSES,
                "plan_step.status",
            ),
        ),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _goal_artifact_to_dict(artifact: GoalArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind,
        "uri": artifact.uri,
        "description": artifact.description,
        "metadata": dict(artifact.metadata),
    }


def _goal_artifact_from_mapping(data: Mapping[str, object]) -> GoalArtifact:
    return GoalArtifact(
        artifact_id=str(data["artifact_id"]),
        kind=str(data["kind"]),
        uri=_optional_str(data.get("uri")),
        description=_optional_str(data.get("description")),
        metadata=_metadata_from_object(data.get("metadata", {})),
    )


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, FrozenJson]:
    frozen = freeze_json(metadata)
    if not isinstance(frozen, Mapping):
        raise TypeError("metadata must be a JSON object")
    return cast(Mapping[str, FrozenJson], frozen)


def _metadata_from_object(value: object) -> Mapping[str, FrozenJson]:
    return _freeze_metadata(_expect_mapping(value, "metadata"))


def _expect_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    raise TypeError(f"{field_name} must be a mapping")


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{field_name} must be a sequence")
    raw_items = cast(tuple[object, ...] | list[object], value)
    items: tuple[object, ...] = tuple(raw_items)
    return tuple(_expect_mapping(item, field_name) for item in items)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{field_name} must be a sequence")
    raw_items = cast(tuple[object, ...] | list[object], value)
    items: tuple[object, ...] = tuple(raw_items)
    return tuple(str(item) for item in items)


def _optional_int(value: object) -> int | None:
    return None if value is None else _int_from_object(value, "optional_int")


def _optional_float(value: object) -> float | None:
    return None if value is None else _float_from_object(value, "optional_float")


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _int_from_object(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{field_name} must be an integer")


def _float_from_object(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{field_name} must be a number")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _literal_from_object(
    value: object,
    allowed: frozenset[str],
    field_name: str,
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _require_literal(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")


def _require_positive(value: int, field_name: str) -> None:
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")


def _require_positive_or_none(value: int | float | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0")
