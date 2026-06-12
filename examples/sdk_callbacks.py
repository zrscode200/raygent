"""Raygent SDK callbacks with a fake model provider.

Run from the project root:

    uv run python examples/sdk_callbacks.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from raygent_harness.core.messages import (
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
from raygent_harness.core.observability import KernelEvent
from raygent_harness.core.query_engine import SDKMessage, SDKResult
from raygent_harness.sdk import RaygentRunCallbacks, create_raygent


@dataclass
class StaticModelProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return model_response_from_message_param(
            assistant_message(f"Callback example saw {len(request.messages)} message(s).")
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


async def main() -> None:
    seen_messages: list[str] = []
    seen_events: list[str] = []

    def on_message(message: SDKMessage) -> None:
        seen_messages.append(type(message).__name__)

    def on_result(result: SDKResult) -> None:
        print(f"terminal: {result.subtype} -> {result.result}")

    def on_kernel_event(event: KernelEvent) -> None:
        if event.type.startswith("query.turn."):
            seen_events.append(event.type)

    session = create_raygent(
        provider=StaticModelProvider(),
        model="demo-model",
        session_id="callback-example-session",
    )
    await session.run_until_result(
        "Show callback handling.",
        callbacks=RaygentRunCallbacks(
            on_message=on_message,
            on_result=on_result,
            on_kernel_event=on_kernel_event,
        ),
    )

    print(f"sdk messages: {', '.join(seen_messages)}")
    print(f"kernel events: {', '.join(seen_events)}")


if __name__ == "__main__":
    asyncio.run(main())
