"""Opt-in SDK stream events with a fake streaming model provider.

Run from the project root:

    uv run python examples/sdk_stream_events.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic import BaseModel

from raygent_harness.core.messages import assistant_message, model_response_from_message_param
from raygent_harness.core.model_provider import classify_exception_by_name
from raygent_harness.core.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelStreamEvent,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    TokenCountRequest,
    ToolUseContentBlock,
    Usage,
)
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    ToolPermissionContext,
)
from raygent_harness.core.tool import (
    ToolCallEvent,
    ToolProgress,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.sdk import (
    RaygentRunCallbacks,
    SDKMessageDelta,
    SDKReasoningAvailable,
    SDKStreamEvent,
    SDKStreamOptions,
    SDKToolProgress,
    SDKToolStart,
    create_raygent,
)


class LookupInput(BaseModel):
    city: str


def _identity(
    *,
    message_id: str,
    block_index: int | None = None,
) -> StreamIdentity:
    return StreamIdentity(
        message_id=message_id,
        content_block_index=block_index,
        provider_request_id="sdk-stream-example-request",
        attempt_id="sdk-stream-example-attempt",
    )


@dataclass
class StreamingModelProvider:
    stream_calls: int = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        _ = request
        return model_response_from_message_param(
            assistant_message("Complete fallback response.")
        )

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        self.stream_calls += 1
        if self.stream_calls == 1:
            return self._tool_stream()
        return self._answer_stream()

    async def _tool_stream(self) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.message_start(
            _identity(message_id="msg_tool"),
            usage=Usage(input_tokens=4),
        )
        yield ModelStreamEvent.content_block_start(
            _identity(message_id="msg_tool", block_index=0),
            block=ToolUseContentBlock(id="toolu_lookup", name="Lookup", input={}),
        )
        yield ModelStreamEvent.content_block_delta(
            _identity(message_id="msg_tool", block_index=0),
            delta={"type": "input_json_delta", "partial_json": '{"city": "Lisbon"}'},
        )
        yield ModelStreamEvent.content_block_stop(
            _identity(message_id="msg_tool", block_index=0)
        )
        yield ModelStreamEvent.message_stop(
            _identity(message_id="msg_tool"),
            usage=Usage(output_tokens=8),
            stop_reason="tool_use",
        )

    async def _answer_stream(self) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.message_start(
            _identity(message_id="msg_final"),
            usage=Usage(input_tokens=6),
        )
        yield ModelStreamEvent.content_block_start(
            _identity(message_id="msg_final", block_index=0),
            block=TextContentBlock(text=""),
        )
        yield ModelStreamEvent.content_block_delta(
            _identity(message_id="msg_final", block_index=0),
            delta={"type": "text_delta", "text": "Lisbon is clear."},
        )
        yield ModelStreamEvent.content_block_stop(
            _identity(message_id="msg_final", block_index=0)
        )
        yield ModelStreamEvent.message_stop(
            _identity(message_id="msg_final"),
            usage=Usage(output_tokens=5),
            stop_reason="end_turn",
        )

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


def build_lookup_tool():
    async def allow(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionAllowDecision:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        assert isinstance(input_, LookupInput)
        secret = "ghp_" + "a" * 36
        yield ToolProgress(
            message=f"checking /Users/example/private.txt token={secret}",
        )
        yield ToolResult(content=f"{input_.city}: clear")

    return build_tool(
        ToolSpec(
            name="Lookup",
            description="Look up local demo weather",
            input_model=LookupInput,
            call=call,
            check_permissions=allow,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


async def main() -> None:
    def on_stream_event(event: SDKStreamEvent) -> None:
        if isinstance(event, SDKMessageDelta):
            print(f"text delta: {event.text}")
        elif isinstance(event, SDKReasoningAvailable):
            print("reasoning available")
        elif isinstance(event, SDKToolStart):
            print(f"tool start: {event.tool_name}")
        elif isinstance(event, SDKToolProgress):
            print(f"tool progress: {event.message}")
        else:
            print(f"tool complete: {event.tool_name} {event.status}")

    session = create_raygent(
        provider=StreamingModelProvider(),
        model="demo-stream-model",
        session_id="sdk-stream-example-session",
        tools=(build_lookup_tool(),),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        experiments={"streaming_tool_execution": True},
    )
    result = await session.run_until_result(
        "Use Lookup, then answer.",
        callbacks=RaygentRunCallbacks(
            on_stream_event=on_stream_event,
            stream_options=SDKStreamOptions(max_tool_preview_chars=80),
        ),
    )
    print(f"result[{result.subtype}]: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
