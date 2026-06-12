"""Raygent embedding with a custom tool and opt-in context provider.

Run from the project root:

    uv run python examples/with_tools_and_context.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pydantic import BaseModel

from raygent_harness.context_providers.defaults import build_default_context_providers
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import assistant_message, model_response_from_message_param
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
from raygent_harness.core.permissions import PermissionAllowDecision, ToolPermissionContext
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKResult,
    SDKSystemInit,
    SDKUserMessage,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)


class EchoInput(BaseModel):
    text: str


async def echo_call(input_: BaseModel, ctx: ToolUseContext) -> AsyncIterator[ToolCallEvent]:
    _ = ctx
    if not isinstance(input_, EchoInput):
        raise TypeError("Echo tool received unexpected input")
    yield ToolResult(content=f"echo: {input_.text}")


async def allow_echo(
    input_: BaseModel,
    ctx: ToolUseContext,
    permission_context: ToolPermissionContext,
) -> PermissionAllowDecision:
    _ = input_, ctx, permission_context
    return PermissionAllowDecision()


def build_echo_tool():
    return build_tool(
        ToolSpec(
            name="Echo",
            description="Echo text back to the model.",
            input_model=EchoInput,
            call=echo_call,
            check_permissions=allow_echo,
            is_read_only=True,
            is_concurrency_safe=True,
            is_destructive=False,
            is_open_world=False,
        )
    )


@dataclass
class ToolCallingProvider:
    """Fake provider that asks for Echo once, then returns a final answer."""

    calls: int = 0
    request_message_counts: list[int] = field(default_factory=list[int])

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.request_message_counts.append(len(request.messages))
        self.calls += 1
        if self.calls == 1:
            return model_response_from_message_param(
                assistant_message(
                    [
                        {"type": "text", "text": "I will use the Echo tool."},
                        {
                            "type": "tool_use",
                            "id": "toolu_echo_1",
                            "name": "Echo",
                            "input": {"text": "hello from a tool"},
                        },
                    ]
                )
            )
        return model_response_from_message_param(
            assistant_message("The tool returned its result, and the turn is complete.")
        )

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


def build_engine(provider: ToolCallingProvider) -> QueryEngine:
    session_id = "tool-session"
    tool = build_echo_tool()
    config = QueryConfig(
        model="demo-model",
        session_id=session_id,
        system_prompt="You can use tools when they are available.",
        tools=(tool,),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        context_providers=build_default_context_providers(
            cwd=".",
            include_environment=True,
            include_git_status=False,
            include_project_instructions=False,
        ),
    )
    ctx = ToolUseContext(
        session_id=session_id,
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id=session_id, depth=0),
    )
    return QueryEngine(config, deps, ctx)


async def main() -> None:
    provider = ToolCallingProvider()
    engine = build_engine(provider)
    async for event in engine.submit_message("Use Echo, then summarize."):
        if isinstance(event, SDKSystemInit):
            print(f"system: tools={event.tools}")
        elif isinstance(event, SDKAssistantMessage):
            print(f"assistant: {event.message['content']}")
        elif isinstance(event, SDKUserMessage):
            print(f"user-event: {event.message['content']}")
        elif isinstance(event, SDKResult):
            print(f"result[{event.subtype}]: {event.result}")
    print(f"provider message counts: {provider.request_message_counts}")


if __name__ == "__main__":
    asyncio.run(main())
