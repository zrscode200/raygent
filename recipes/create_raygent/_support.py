from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from raygent_harness.core.config import QueryConfig
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
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.tool import Tool, ToolUseContext
from raygent_harness.sdk import RaygentSession
from raygent_harness.skills.models import SkillDefinition


@dataclass
class StaticModelProvider:
    """Small no-network provider used only by recipes."""

    label: str
    requests: list[ModelRequest] = field(default_factory=list[ModelRequest])

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        tool_names = ", ".join(tool.name for tool in request.tools) or "none"
        return model_response_from_message_param(
            assistant_message(f"{self.label}: tools={tool_names}")
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


async def run_and_print(session: RaygentSession, prompt: str) -> SDKResult:
    result = await session.run_until_result(prompt)
    print(f"result[{result.subtype}]: {result.result}")
    return result


async def turn_tool_names(session: RaygentSession) -> tuple[str, ...]:
    provider = session.deps.tool_catalog_provider
    if provider is None:
        return tuple(tool.name for tool in session.config.tools)
    tools = await provider(session.config, session.ctx, _NO_SKILLS)
    if tools is None:
        tools = session.config.tools
    return tuple(tool.name for tool in tools)


_NO_SKILLS: Sequence[SkillDefinition] = ()


async def _unused_catalog(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    _skills: Sequence[SkillDefinition],
) -> Sequence[Tool] | None:
    return ()
