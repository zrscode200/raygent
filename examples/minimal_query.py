"""Minimal Raygent SDK embedding with a fake model provider.

Run from the project root:

    uv run python examples/minimal_query.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
from raygent_harness.core.query_engine import SDKAssistantMessage, SDKResult
from raygent_harness.sdk import create_raygent


@dataclass
class StaticModelProvider:
    """Small local provider used only for this example."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        prompt_count = len(request.messages)
        return model_response_from_message_param(
            assistant_message(f"Hello from Raygent. I saw {prompt_count} message(s).")
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
    session = create_raygent(
        provider=StaticModelProvider(),
        model="demo-model",
        session_id="minimal-session",
        system_prompt="You are a concise assistant running inside Raygent.",
    )

    async for event in session.run("Say hello."):
        if isinstance(event, SDKAssistantMessage):
            print(f"assistant: {event.message['content']}")
        elif isinstance(event, SDKResult):
            print(f"result[{event.subtype}]: {event.result}")


if __name__ == "__main__":
    asyncio.run(main())
