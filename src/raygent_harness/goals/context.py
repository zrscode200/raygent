"""Goal context provider for model-input steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from raygent_harness.core.context_providers import (
    ContextAgentScope,
    ContextFragment,
    ContextKind,
)
from raygent_harness.goals.steering import (
    GoalSteeringConfig,
    build_goal_continuation_steering,
)
from raygent_harness.goals.store import GoalStore

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.tool import ToolUseContext


@dataclass(frozen=True, slots=True)
class GoalContextProvider:
    """Attach active-goal continuation guidance as non-persistent context."""

    store: GoalStore
    steering_config: GoalSteeringConfig = field(default_factory=GoalSteeringConfig)
    priority: int = 5
    agent_scope: ContextAgentScope = "main"
    context_kind: ContextKind = "goal"

    async def __call__(
        self,
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        state = self.store.get_active_for_session(ctx.session_id)
        if state is None:
            return ()
        return (
            ContextFragment(
                id=f"goal_context:{state.goal_id}",
                content=build_goal_continuation_steering(
                    state,
                    config=self.steering_config,
                ),
                channel="user_context",
                source="goal_runner",
                priority=self.priority,
                agent_scope=self.agent_scope,
                render_mode="context",
                kind=self.context_kind,
            ),
        )


__all__ = ["GoalContextProvider"]
