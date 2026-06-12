"""Provider-protocol adapter interfaces.

This module is intentionally transport-free. It lets tests and future provider
packages prove protocol lowering/parsing fidelity without importing vendor SDKs
or teaching `core` about provider wire formats.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from raygent_harness.core.model_types import (
    FrozenJson,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderError,
    TokenCountRequest,
    freeze_json,
)

ProviderEvent = Mapping[str, object]
ProviderResponse = Mapping[str, object]


@dataclass(frozen=True, slots=True, init=False)
class PreparedModelRequest:
    """Provider-shaped request payload ready for a transport layer.

    `body` and metadata accept provider-shaped JSON dictionaries/lists at the
    adapter boundary, then freeze them so tests can assert API-bound replay
    stability without sharing mutable provider dictionaries.
    """

    protocol_id: str
    model: str
    body: FrozenJson
    headers: FrozenJson
    options: FrozenJson

    def __init__(
        self,
        *,
        protocol_id: str,
        model: str,
        body: object,
        headers: object | None = None,
        options: object | None = None,
    ) -> None:
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "body", freeze_json(body))
        object.__setattr__(self, "headers", freeze_json(headers or {}))
        object.__setattr__(self, "options", freeze_json(options or {}))


class ModelProtocolAdapter(Protocol):
    """Translate between Raygent model types and one provider protocol family."""

    @property
    def protocol_id(self) -> str:
        """Stable protocol identifier for prepared requests and catalogs."""
        ...

    def prepare_request(self, request: ModelRequest) -> PreparedModelRequest:
        """Lower a Raygent model request to the default streaming request body."""
        ...

    def prepare_complete_request(self, request: ModelRequest) -> PreparedModelRequest:
        """Lower a Raygent model request for non-streaming execution."""
        ...

    def prepare_stream_request(self, request: ModelRequest) -> PreparedModelRequest:
        """Lower a Raygent model request for streaming execution."""
        ...

    def parse_response(self, provider_response: ProviderResponse) -> ModelResponse:
        """Raise a provider-shaped complete response into a Raygent response."""
        ...

    def create_stream_parser(self) -> ModelProtocolStreamParser:
        """Create one stateful parser for an incremental provider stream."""
        ...

    def stream_events(
        self,
        provider_events: Iterable[ProviderEvent],
    ) -> Iterable[ModelStreamEvent]:
        """Raise provider-shaped stream events into Raygent stream events."""
        ...

    def classify_error(self, provider_error_payload: object) -> ProviderError:
        """Normalize a provider-shaped error payload into Raygent recovery facts."""
        ...

    def prepare_token_count(self, request: TokenCountRequest) -> PreparedModelRequest:
        """Lower a token-count request when the provider protocol supports it."""
        ...


class ModelProtocolStreamParser(Protocol):
    """Stateful parser for one provider stream attempt.

    `stream_events(...)` remains the finite fixture convenience API, but live
    runtimes should feed async provider events into a single parser instance so
    protocol state is not reset between chunks.
    """

    def feed(self, provider_event: ProviderEvent) -> Iterable[ModelStreamEvent]:
        """Translate one provider event into zero or more Raygent events."""
        ...

    def finish(self) -> Iterable[ModelStreamEvent]:
        """Flush any terminal parser state after the provider stream ends."""
        ...


__all__ = [
    "ModelProtocolAdapter",
    "ModelProtocolStreamParser",
    "PreparedModelRequest",
    "ProviderEvent",
    "ProviderResponse",
]
