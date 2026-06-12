"""Provider-neutral MCP client seam.

Wave 4 intentionally stops at an injected protocol plus in-memory test client.
Real stdio/HTTP transports, OAuth, process supervision, and credential storage
belong behind this seam in a later wave/group.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.services.mcp.models import (
    McpRegistrySnapshot,
    McpToolIdentity,
)


def _empty_json_object() -> FrozenJson:
    return {}


@dataclass(frozen=True, slots=True)
class McpToolCallRequest:
    """One MCP `tools/call` request after Raygent tool-name resolution."""

    identity: McpToolIdentity
    arguments: FrozenJson = field(default_factory=_empty_json_object)
    tool_use_id: str | None = None
    timeout_s: float | None = None
    metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", freeze_json(self.arguments))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @property
    def server_name(self) -> str:
        return self.identity.server_name

    @property
    def tool_name(self) -> str:
        return self.identity.tool_name

    @property
    def raygent_name(self) -> str:
        return self.identity.raygent_name


@dataclass(frozen=True, slots=True)
class McpToolCallResult:
    """Replay-safe result returned by an MCP client adapter."""

    content: FrozenJson = ""
    is_error: bool = False
    structured_content: FrozenJson | None = None
    metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", freeze_json(self.content))
        if self.structured_content is not None:
            object.__setattr__(
                self,
                "structured_content",
                freeze_json(self.structured_content),
            )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


McpProgressEmitter = Callable[[FrozenJson], Awaitable[None]]
McpElicitationHandler = Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class McpToolCallContext:
    """Non-replay call controls for one in-flight MCP tool call."""

    abort_event: asyncio.Event | None = None
    emit_progress: McpProgressEmitter | None = None
    handle_elicitation: McpElicitationHandler | None = None


McpClientErrorKind = Literal[
    "server_not_found",
    "tool_not_found",
    "tool_unavailable",
    "transport_error",
    "timeout",
    "invalid_result",
]


class McpClientError(Exception):
    """Recoverable MCP client failure surfaced as a model-visible tool error."""

    def __init__(self, message: str, *, kind: McpClientErrorKind = "transport_error"):
        super().__init__(message)
        self.kind = kind


class McpClient(Protocol):
    """Injected MCP client/transport seam."""

    async def registry_snapshot(self) -> McpRegistrySnapshot:
        """Return the current replay-safe server/tool snapshot."""
        ...

    async def call_tool(
        self,
        request: McpToolCallRequest,
        context: McpToolCallContext,
        /,
    ) -> McpToolCallResult:
        """Call one MCP tool by original server/tool identity."""
        ...


McpInMemoryResponse = McpToolCallResult | McpClientError | Exception


@dataclass
class InMemoryMcpClient:
    """Small deterministic fake client for tests and embedding examples."""

    snapshot: McpRegistrySnapshot
    responses: Mapping[str, McpInMemoryResponse] = field(
        default_factory=dict[str, McpInMemoryResponse]
    )
    calls: list[McpToolCallRequest] = field(default_factory=list[McpToolCallRequest])

    async def registry_snapshot(self) -> McpRegistrySnapshot:
        return self.snapshot

    async def call_tool(
        self,
        request: McpToolCallRequest,
        context: McpToolCallContext,
        /,
    ) -> McpToolCallResult:
        _ = context
        self.calls.append(request)
        response = self.responses.get(request.raygent_name)
        if response is None:
            raise McpClientError(
                f"No in-memory MCP response registered for {request.raygent_name}",
                kind="tool_not_found",
            )
        if isinstance(response, McpClientError):
            raise response
        if isinstance(response, Exception):
            raise response
        return response


__all__ = [
    "InMemoryMcpClient",
    "McpClient",
    "McpClientError",
    "McpClientErrorKind",
    "McpElicitationHandler",
    "McpInMemoryResponse",
    "McpProgressEmitter",
    "McpToolCallContext",
    "McpToolCallRequest",
    "McpToolCallResult",
]
