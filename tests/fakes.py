from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

from raygent_harness.adapters.model_protocols.base import (
    ModelProtocolAdapter,
    PreparedModelRequest,
    ProviderEvent,
)
from raygent_harness.core.messages import MessageParam, model_response_from_message_param
from raygent_harness.core.model_provider import classify_exception_by_name
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelStreamEvent,
    ProviderError,
    TokenCountRequest,
    TokenCountResult,
)

type FakeModelResponse = MessageParam | ModelResponse | BaseException
type FakeStreamEvent = ModelStreamEvent | BaseException
type FakeTokenCount = int | TokenCountResult | BaseException


def _empty_requests() -> list[ModelRequest]:
    return []


@dataclass
class FakeModelProvider:
    responses: Sequence[FakeModelResponse] = ()
    requests: list[ModelRequest] = field(default_factory=_empty_requests)
    stream_events: Sequence[FakeStreamEvent] = ()
    stream_requests: list[ModelRequest] = field(default_factory=list[ModelRequest])
    token_counts: Sequence[FakeTokenCount] = ()
    token_requests: list[TokenCountRequest] = field(default_factory=list[TokenCountRequest])
    model_infos: dict[str, ModelInfo] = field(default_factory=dict[str, ModelInfo])
    model_info_requests: list[str] = field(default_factory=list[str])
    resolved_models: dict[str, str] = field(default_factory=dict[str, str])
    resolve_requests: list[tuple[str, ModelResolveContext]] = field(
        default_factory=list[tuple[str, ModelResolveContext]]
    )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        response = (
            self.responses[index]
            if index < len(self.responses)
            else cast(MessageParam, {"role": "assistant", "content": ""})
        )
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, ModelResponse):
            return response
        return model_response_from_message_param(response)

    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        return self._stream_events()

    async def _stream_events(self) -> AsyncIterator[ModelStreamEvent]:
        for event in self.stream_events:
            if isinstance(event, BaseException):
                raise event
            yield event

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        self.token_requests.append(request)
        index = len(self.token_requests) - 1
        result = self.token_counts[index] if index < len(self.token_counts) else 0
        if isinstance(result, BaseException):
            raise result
        return result

    def resolve_model(self, requested: str, context: ModelResolveContext) -> str:
        self.resolve_requests.append((requested, context))
        return self.resolved_models.get(requested, requested)

    def model_info(self, model: str) -> ModelInfo:
        self.model_info_requests.append(model)
        return self.model_infos.get(model, ModelInfo(model=model))

    def classify_error(self, error: BaseException) -> ProviderError:
        return classify_exception_by_name(error)


@dataclass(frozen=True)
class AdapterProviderPayloadError(Exception):
    """Test-only exception carrying a provider-shaped error payload."""

    payload: object

    def __str__(self) -> str:
        return str(self.payload)


@dataclass
class AdapterBackedFakeModelProvider:
    """Test-only provider that proves query requests can cross adapter seams.

    The harness accepts Raygent `ModelRequest`s, lowers them through a chosen
    `ModelProtocolAdapter`, then replays provider-shaped stream fixtures through
    the same adapter back into normalized `ModelStreamEvent`s.
    """

    adapter: ModelProtocolAdapter
    provider_event_batches: Sequence[Sequence[ProviderEvent]] = ()
    complete_event_batches: Sequence[Sequence[ProviderEvent]] = ()
    complete_errors: Sequence[object] = ()
    token_counts: Sequence[FakeTokenCount] = ()
    model_infos: dict[str, ModelInfo] = field(default_factory=dict[str, ModelInfo])
    resolved_models: dict[str, str] = field(default_factory=dict[str, str])
    stream_requests: list[ModelRequest] = field(default_factory=list[ModelRequest])
    complete_requests: list[ModelRequest] = field(default_factory=list[ModelRequest])
    prepared_requests: list[PreparedModelRequest] = field(
        default_factory=list[PreparedModelRequest]
    )
    token_requests: list[TokenCountRequest] = field(default_factory=list[TokenCountRequest])
    prepared_token_requests: list[PreparedModelRequest] = field(
        default_factory=list[PreparedModelRequest]
    )
    model_info_requests: list[str] = field(default_factory=list[str])
    resolve_requests: list[tuple[str, ModelResolveContext]] = field(
        default_factory=list[tuple[str, ModelResolveContext]]
    )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_requests.append(request)
        self.prepared_requests.append(self.adapter.prepare_request(request))
        index = len(self.complete_requests) - 1
        if index < len(self.complete_errors):
            raise AdapterProviderPayloadError(self.complete_errors[index])
        batch = (
            self.complete_event_batches[index]
            if index < len(self.complete_event_batches)
            else ()
        )
        return assemble_model_stream(self.adapter.stream_events(batch))

    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        self.prepared_requests.append(self.adapter.prepare_request(request))
        index = len(self.stream_requests) - 1
        batch = (
            self.provider_event_batches[index]
            if index < len(self.provider_event_batches)
            else ()
        )
        return self._stream_events(self.adapter.stream_events(batch))

    async def _stream_events(
        self,
        events: Iterable[ModelStreamEvent],
    ) -> AsyncIterator[ModelStreamEvent]:
        for event in events:
            yield event

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        self.token_requests.append(request)
        self.prepared_token_requests.append(self.adapter.prepare_token_count(request))
        index = len(self.token_requests) - 1
        result = self.token_counts[index] if index < len(self.token_counts) else 0
        if isinstance(result, BaseException):
            raise result
        return result

    def resolve_model(self, requested: str, context: ModelResolveContext) -> str:
        self.resolve_requests.append((requested, context))
        return self.resolved_models.get(requested, requested)

    def model_info(self, model: str) -> ModelInfo:
        self.model_info_requests.append(model)
        return self.model_infos.get(model, ModelInfo(model=model))

    def classify_error(self, error: BaseException) -> ProviderError:
        if isinstance(error, AdapterProviderPayloadError):
            return self.adapter.classify_error(error.payload)
        return classify_exception_by_name(error)
