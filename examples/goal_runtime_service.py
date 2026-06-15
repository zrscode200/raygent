"""Product-owned goal command/API mapped to Raygent's goal runtime.

Run from the project root:

    uv run python examples/goal_runtime_service.py

Raygent does not parse `/goal` or own product UI. A CLI, mobile app, web app,
or API handler can parse whatever trigger it wants, then call the kernel service
shown here.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.core.messages import (
    MessageParam,
    assistant_message,
    model_response_from_message_param,
)
from raygent_harness.core.model_provider import classify_exception_by_name
from raygent_harness.core.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelStreamEvent,
    ProviderError,
    TokenCountRequest,
)
from raygent_harness.goals import (
    UPDATE_GOAL_TOOL_NAME,
    GoalPolicy,
    GoalRuntime,
    GoalSpec,
    JsonGoalStore,
)
from raygent_harness.sdk import RaygentGoalRuntimeOptions, create_raygent


def goal_completed_tool_message(reason: str) -> MessageParam:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "goal_complete",
                "name": UPDATE_GOAL_TOOL_NAME,
                "input": {"status": "complete", "reason": reason},
            }
        ],
    }


@dataclass
class ScriptedGoalProvider:
    """Small local provider used only for this example."""

    responses: tuple[MessageParam, ...]
    requests: list[ModelRequest]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        response = (
            self.responses[index]
            if index < len(self.responses)
            else assistant_message("No more scripted responses.")
        )
        return model_response_from_message_param(response)

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        return self._empty_stream()

    async def _empty_stream(self) -> AsyncIterator[ModelStreamEvent]:
        if False:
            yield  # pragma: no cover

    async def count_tokens(self, request: TokenCountRequest) -> int:
        return sum(
            len(str(message.provider_payload or message.message))
            for message in request.messages
        )

    def resolve_model(self, requested: str, context: ModelResolveContext) -> str:
        _ = context
        return requested

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(model=model, context_window=128_000, max_output_tokens_default=4096)

    def classify_error(self, error: BaseException) -> ProviderError:
        return classify_exception_by_name(error)


@dataclass
class ProductGoalController:
    """Thin product layer over Raygent's headless goal runtime."""

    project_root: Path

    def create_runtime(self, provider: ScriptedGoalProvider) -> GoalRuntime:
        session = create_raygent(
            provider=provider,
            model="demo-model",
            cwd=self.project_root,
            session_id="product-session",
            tools="none",
            context="none",
            goal_runtime_options=RaygentGoalRuntimeOptions(
                store=JsonGoalStore(project_root=self.project_root),
            ),
        )
        runtime = session.handles.goal_runtime
        if runtime is None:
            raise RuntimeError("Goal runtime was not attached to the Raygent session.")
        return runtime

    def start_user_goal(self, runtime: GoalRuntime, objective: str) -> str:
        state = runtime.start(
            GoalSpec(
                objective=objective,
                success_criteria=("The model reports completion through update_goal.",),
            ),
            policy=GoalPolicy(),
            goal_id="product-goal",
        )
        return state.goal_id


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-goal-example-") as tmp:
        controller = ProductGoalController(project_root=Path(tmp))

        first_provider = ScriptedGoalProvider(responses=(), requests=[])
        first_runtime = controller.create_runtime(first_provider)
        goal_id = controller.start_user_goal(
            first_runtime,
            "Prepare a short release checklist.",
        )
        first_runtime.pause(reason="product process restarted")

        resumed_provider = ScriptedGoalProvider(
            responses=(
                goal_completed_tool_message("release checklist drafted"),
                assistant_message("The release checklist is ready."),
            ),
            requests=[],
        )
        resumed_runtime = controller.create_runtime(resumed_provider)
        resumed_runtime.resume(goal_id, reason="resumed from durable store")
        result = await resumed_runtime.run_until_idle(max_continuations=3)

        print(f"goal: {goal_id}")
        print(f"status: {result.state.status if result.state else 'none'}")
        print(f"stop_reason: {result.stop_reason}")
        print(f"model_requests: {len(resumed_provider.requests)}")


if __name__ == "__main__":
    asyncio.run(main())
