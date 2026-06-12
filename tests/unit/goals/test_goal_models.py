from __future__ import annotations

import pytest

from raygent_harness.goals import (
    GoalAcceptanceCheck,
    GoalBlockingPolicy,
    GoalBudgetPolicy,
    GoalOutputSpec,
    GoalPolicy,
    GoalSpec,
    create_goal_state,
    goal_state_from_dict,
    goal_state_to_dict,
)


def test_goal_state_serializes_round_trip_with_rich_spec_and_policy() -> None:
    spec = GoalSpec(
        objective="Ship goal runner service",
        success_criteria=("state model exists", "tool specs exist"),
        constraints=("headless kernel only",),
        non_goals=("no slash command parser",),
        expected_outputs=(
            GoalOutputSpec(
                name="package",
                description="raygent_harness.goals package",
                metadata={"kind": "python"},
            ),
        ),
        acceptance_checks=(
            GoalAcceptanceCheck(
                name="validation",
                instruction="Run focused tests",
            ),
        ),
        metadata={"owner": "kernel"},
    )
    policy = GoalPolicy(
        budget=GoalBudgetPolicy(max_turns=6, token_budget=10_000),
        blocking=GoalBlockingPolicy(blocked_audit_turns=3),
    )
    state = create_goal_state(
        goal_id="g_1",
        session_id="s_1",
        spec=spec,
        policy=policy,
        now=100.0,
    ).with_accounting(turn_delta=1, token_delta=50, time_delta_s=2.5, now=102.5)

    snapshot = goal_state_to_dict(state)
    restored = goal_state_from_dict(snapshot)

    assert restored == state
    assert restored.status == "active"
    assert restored.token_budget == 10_000
    assert restored.spec.objective == "Ship goal runner service"
    assert restored.spec.expected_outputs[0].metadata["kind"] == "python"


def test_goal_state_rejects_invalid_budget_values() -> None:
    with pytest.raises(ValueError, match="token_budget"):
        GoalBudgetPolicy(token_budget=0)
