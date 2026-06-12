"""Provider protocol for Raygent-owned model types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

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


def classify_exception_by_name(error: BaseException) -> ProviderError:
    """Fallback classifier for tests and minimal providers.

    Real providers should implement `classify_error` using their own exception
    surfaces. This helper preserves the pre-Group-7 fake-error behavior without
    teaching the query loop vendor exception classes.
    """
    name = type(error).__name__.lower()
    message = str(error) or type(error).__name__
    if "fallback" in name:
        return ProviderError(kind="model_fallback_triggered", message=message)
    if "contextoverflow" in name or "prompttoolong" in name:
        return ProviderError(kind="context_overflow", message=message)
    if "media" in name and ("overflow" in name or "too" in name or "large" in name):
        return ProviderError(kind="media_overflow", message=message)
    if "maxoutputtokens" in name or "max_tokens" in name:
        return ProviderError(kind="max_output_tokens", message=message)
    if "ratelimit" in name:
        return ProviderError(kind="rate_limit", message=message, retryable=True)
    if "timeout" in name or "transient" in name:
        return ProviderError(kind="transient", message=message, retryable=True)
    if "abort" in name or "cancel" in name:
        return ProviderError(kind="user_abort", message=message)
    if "auth" in name or "config" in name:
        return ProviderError(kind="auth_config", message=message)
    return ProviderError(kind="fatal_unknown", message=message)


class ModelProvider(Protocol):
    """Model/provider boundary consumed by the query loop.

    Concrete providers translate Raygent's normalized model types to their
    vendor SDK payloads. Core runtime code should depend on this protocol, not
    on provider clients or exceptions.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a complete assistant response for non-streaming execution."""
        ...

    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Yield normalized stream events for future streaming execution."""
        ...

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        """Return provider-aware token count when available."""
        ...

    def resolve_model(
        self,
        requested: str,
        context: ModelResolveContext,
    ) -> str:
        """Resolve aliases/window suffixes into a concrete provider model id."""
        ...

    def model_info(self, model: str) -> ModelInfo:
        """Return context window, output limits, and capability metadata."""
        ...

    def classify_error(self, error: BaseException) -> ProviderError:
        """Map provider exceptions into Raygent recovery categories."""
        ...


@dataclass(frozen=True)
class UnavailableModelProvider:
    """Fail-closed default used when callers forget to provide a model backend."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        _ = request
        raise RuntimeError("No model_provider configured")

    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        raise RuntimeError("No model_provider configured")
        yield  # pragma: no cover

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        _ = request
        raise RuntimeError("No model_provider configured")

    def resolve_model(
        self,
        requested: str,
        context: ModelResolveContext,
    ) -> str:
        _ = context
        return requested

    def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(model=model)

    def classify_error(self, error: BaseException) -> ProviderError:
        return classify_exception_by_name(error)


__all__ = [
    "ModelProvider",
    "UnavailableModelProvider",
    "classify_exception_by_name",
]
