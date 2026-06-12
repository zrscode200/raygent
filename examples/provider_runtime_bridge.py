"""No-network provider-runtime bridge example.

Run from the project root:

    uv run python examples/provider_runtime_bridge.py

This example uses an Anthropic Messages-shaped protocol translator plus an
in-memory transport. A live integration would replace only `DemoTransport` with
SDK/HTTP/SSE code; Raygent core still receives a `ProtocolModelProvider`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import cast

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.adapters.model_runtime import (
    ProtocolModelProvider,
    ProviderRetryPolicy,
    ProviderTransportRequest,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    ModelCapabilities,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    TextContentBlock,
    TokenCountRequest,
    TokenCountResult,
)
from raygent_harness.core.query_engine import QueryEngine, SDKAssistantMessage, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext


class DemoTransport:
    """In-memory transport standing in for a live provider SDK/HTTP client."""

    async def complete(self, request: ProviderTransportRequest) -> ProviderResponse:
        body = _body(request)
        message_count = len(cast(list[object], body.get("messages", [])))
        return {
            "id": "msg_demo_complete",
            "request_id": "req_demo_complete",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Protocol bridge complete response. "
                        f"Model={request.prepared_request.model}; "
                        f"messages={message_count}."
                    ),
                }
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }

    async def stream(
        self,
        request: ProviderTransportRequest,
    ) -> AsyncIterator[ProviderEvent]:
        _ = request
        yield {
            "type": "message_start",
            "request_id": "req_demo_stream",
            "message": {"id": "msg_demo_stream"},
        }
        yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "streamed "},
        }
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "through the bridge"},
        }
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_stop", "stop_reason": "end_turn"}

    async def count_tokens(self, request: ProviderTransportRequest) -> TokenCountResult:
        body = _body(request)
        # Exact under this demo transport's intentionally tiny tokenizer.
        token_count = len(str(body).split())
        return TokenCountResult(
            token_count=max(token_count, 1),
            provider_request_id="req_demo_count",
            safe_metadata={"transport": "demo", "tokenizer": "whitespace"},
        )


def _body(request: ProviderTransportRequest) -> Mapping[str, object]:
    body = thaw_json(request.prepared_request.body)
    if not isinstance(body, Mapping):
        raise TypeError("prepared request body was not a mapping")
    return cast(Mapping[str, object], body)


def build_provider() -> ProtocolModelProvider:
    return ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=DemoTransport(),
        models=(
            ModelInfo(
                model="demo-anthropic-messages",
                aliases=("demo-model",),
                context_window=200_000,
                max_output_tokens_default=8192,
                capabilities=ModelCapabilities(supports_streaming=True, supports_tools=True),
            ),
        ),
        retry_policy=ProviderRetryPolicy(max_attempts=2, retry_rate_limit=True),
    )


def build_engine(provider: ProtocolModelProvider) -> QueryEngine:
    session_id = "provider-runtime-session"
    config = QueryConfig(
        model="demo-model",
        session_id=session_id,
        system_prompt="You are running through Raygent's provider-runtime bridge.",
    )
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
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
    provider = build_provider()
    engine = build_engine(provider)

    async for event in engine.submit_message("Use the protocol runtime bridge."):
        if isinstance(event, SDKAssistantMessage):
            print(f"assistant: {event.message['content']}")
        elif isinstance(event, SDKResult):
            print(f"result[{event.subtype}]: {event.result}")

    token_count = await provider.count_tokens(
        TokenCountRequest(
            model="demo-anthropic-messages",
            messages=(
                ApiMessage(
                    message=ModelMessage(
                        role="user",
                        content=(
                            TextContentBlock(text="Count this through the transport."),
                        ),
                    )
                ),
            ),
            system_prompt="Provider-bound system prompt.",
        )
    )
    if isinstance(token_count, TokenCountResult):
        print(f"tokens: {token_count.token_count} request={token_count.provider_request_id}")
    else:
        print(f"tokens: {token_count}")

    stream_response = assemble_model_stream(
        [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="demo-anthropic-messages",
                    messages=(
                        ApiMessage(
                            message=ModelMessage(
                                role="user",
                                content=(TextContentBlock(text="Stream this."),),
                            )
                        ),
                    ),
                )
            )
        ]
    )
    stream_text = cast(TextContentBlock, stream_response.api_message.message.content[0]).text
    print(f"stream: {stream_text}")


if __name__ == "__main__":
    asyncio.run(main())
