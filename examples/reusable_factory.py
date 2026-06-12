"""Reusable RaygentFactory and product-wrapper example.

Run from the project root:

    uv run python examples/reusable_factory.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

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
from raygent_harness.sdk import (
    RaygentFactory,
    RaygentFactoryConfig,
    RaygentModelOptions,
    RaygentSession,
    RaygentSessionFactory,
    RaygentSessionOptions,
)


@dataclass
class StaticModelProvider:
    """Small local provider used only for this example."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return model_response_from_message_param(
            assistant_message(
                f"Factory model {request.model} saw {len(request.messages)} message(s)."
            )
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


@dataclass(frozen=True)
class ProductSessionBuilder:
    """Product-layer wrapper around Raygent's narrow factory Protocol.

    The product owns provider choice, session naming, and policy. Raygent's
    factory only assembles kernel objects from the explicit config.
    """

    factory: RaygentSessionFactory
    provider: StaticModelProvider
    cwd: Path

    def create_user_session(self, user_id: str) -> RaygentSession:
        return self.factory.create_session(
            RaygentFactoryConfig(
                model_options=RaygentModelOptions(
                    provider=self.provider,
                    model="demo-model",
                    system_prompt="You are running inside a product-owned session.",
                ),
                session_options=RaygentSessionOptions(
                    cwd=self.cwd,
                    session_id=f"user-{user_id}",
                ),
            )
        )


async def main() -> None:
    builder = ProductSessionBuilder(
        factory=RaygentFactory(),
        provider=StaticModelProvider(),
        cwd=Path.cwd(),
    )
    session = builder.create_user_session("alice")
    result = await session.run_until_result("Show reusable factory wiring.")

    print(f"factory session: {session.session_id}")
    print(f"result[{result.subtype}]: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
