"""Injected model transport contracts for protocol-backed providers.

The runtime layer is intentionally transport-shape only. It does not own HTTP,
SSE, SDK clients, credentials, or provider account state; embedders inject those
behind these protocols.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from raygent_harness.adapters.model_protocols import (
    PreparedModelRequest,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.core.model_types import FrozenJson, TokenCountResult, freeze_json


@dataclass(frozen=True, slots=True)
class ProviderTransportRequest:
    """One prepared provider request plus runtime execution controls."""

    prepared_request: PreparedModelRequest
    abort_event: asyncio.Event | None = None
    timeout_s: float | None = None
    metadata: FrozenJson = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


class ProviderTransport(Protocol):
    """Transport injected into `ProtocolModelProvider`.

    Implementations may wrap HTTP clients, SDK clients, in-memory fixtures, or
    test doubles. They should honor `abort_event` promptly where possible.
    """

    async def complete(self, request: ProviderTransportRequest) -> ProviderResponse:
        """Execute a non-streaming provider request and return provider payload."""
        ...

    def stream(
        self,
        request: ProviderTransportRequest,
    ) -> AsyncIterator[ProviderEvent]:
        """Execute a streaming provider request and yield provider events."""
        ...

    async def count_tokens(
        self,
        request: ProviderTransportRequest,
    ) -> int | TokenCountResult:
        """Execute a token-count request and return an exact provider count."""
        ...


class ProviderPayloadError(RuntimeError):
    """Exception wrapper for provider-shaped error payloads.

    Transports can raise this when they have a structured provider error body
    but do not want to normalize it themselves. `ProtocolModelProvider` delegates
    the payload to the protocol adapter's `classify_error(...)` method.
    """

    def __init__(
        self,
        payload: object,
        message: str = "Provider payload error",
    ) -> None:
        super().__init__(message)
        self.payload = payload


def response_mapping(payload: Mapping[str, object]) -> ProviderResponse:
    """Type helper for tests/fakes returning provider response dictionaries."""

    return payload


__all__ = [
    "ProviderPayloadError",
    "ProviderTransport",
    "ProviderTransportRequest",
    "response_mapping",
]
