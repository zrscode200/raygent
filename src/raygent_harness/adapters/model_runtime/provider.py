"""Runtime bridge from protocol adapters plus transports to `ModelProvider`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from raygent_harness.adapters.model_protocols import ModelProtocolAdapter
from raygent_harness.adapters.model_runtime.catalog import (
    ProviderModelCatalog,
    registry_from_catalogs,
)
from raygent_harness.adapters.model_runtime.retry import (
    ProviderRetryPolicy,
    classify_retry_decision,
    should_fallback_stream_to_complete,
)
from raygent_harness.adapters.model_runtime.transport import (
    ProviderPayloadError,
    ProviderTransport,
    ProviderTransportRequest,
)
from raygent_harness.core.model_provider import classify_exception_by_name
from raygent_harness.core.model_registry import ModelRegistry
from raygent_harness.core.model_types import (
    FrozenJson,
    ModelInfo,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelStreamEvent,
    ProviderError,
    StreamIdentity,
    TokenCountRequest,
    TokenCountResult,
    freeze_json,
)


@dataclass(slots=True)
class ProtocolModelProvider:
    """Concrete `ModelProvider` backed by a protocol adapter and transport."""

    adapter: ModelProtocolAdapter
    transport: ProviderTransport
    models: Iterable[ModelInfo] = ()
    catalogs: Iterable[ProviderModelCatalog] = ()
    replace_model_capabilities: bool = False
    timeout_s: float | None = None
    retry_policy: ProviderRetryPolicy = field(default_factory=ProviderRetryPolicy)
    retry_sleep: Callable[[float], Awaitable[None]] | None = field(
        default=None,
        repr=False,
    )
    _registry: ModelRegistry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._registry = registry_from_catalogs(
            catalogs=self.catalogs,
            model_infos=self.models,
            replace_capabilities=self.replace_model_capabilities,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        attempt = 1
        while True:
            self._raise_if_aborted(request.abort_event)
            try:
                prepared = self.adapter.prepare_complete_request(request)
                provider_response = await self.transport.complete(
                    ProviderTransportRequest(
                        prepared_request=prepared,
                        abort_event=request.abort_event,
                        timeout_s=self.timeout_s,
                        metadata=_model_request_metadata(
                            request,
                            operation="complete",
                            attempt=attempt,
                            max_attempts=self.retry_policy.max_attempts,
                        ),
                    )
                )
                self._raise_if_aborted(request.abort_event)
                return self.adapter.parse_response(provider_response)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                decision = classify_retry_decision(
                    self.classify_error(error),
                    operation="complete",
                    attempt=attempt,
                    policy=self.retry_policy,
                )
                if not decision.should_retry:
                    raise
                await self._sleep_before_retry(decision.delay_s, request.abort_event)
                attempt += 1

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        attempt = 1
        while True:
            self._raise_if_aborted(request.abort_event)
            parser = self.adapter.create_stream_parser()
            stream_events_emitted = False
            try:
                prepared = self.adapter.prepare_stream_request(request)
                transport_request = ProviderTransportRequest(
                    prepared_request=prepared,
                    abort_event=request.abort_event,
                    timeout_s=self.timeout_s,
                    metadata=_model_request_metadata(
                        request,
                        operation="stream",
                        attempt=attempt,
                        max_attempts=self.retry_policy.max_attempts,
                    ),
                )
                async for provider_event in self.transport.stream(transport_request):
                    self._raise_if_aborted(request.abort_event)
                    for event in parser.feed(provider_event):
                        self._raise_if_aborted(request.abort_event)
                        stream_events_emitted = True
                        yield event
                for event in parser.finish():
                    self._raise_if_aborted(request.abort_event)
                    stream_events_emitted = True
                    yield event
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._raise_if_aborted(request.abort_event)
                provider_error = self.classify_error(error)
                decision = classify_retry_decision(
                    provider_error,
                    operation="stream",
                    attempt=attempt,
                    policy=self.retry_policy,
                    stream_events_emitted=stream_events_emitted,
                )
                if decision.should_retry:
                    await self._sleep_before_retry(
                        decision.delay_s,
                        request.abort_event,
                    )
                    attempt += 1
                    continue
                if should_fallback_stream_to_complete(
                    provider_error,
                    policy=self.retry_policy,
                    stream_events_emitted=stream_events_emitted,
                ):
                    reason = _provider_error_reason(provider_error)
                    identity = StreamIdentity(
                        attempt_id=f"stream-fallback-attempt-{attempt}"
                    )
                    yield ModelStreamEvent.streaming_transport_fallback_started(
                        identity,
                        reason=reason,
                    )
                    replacement = await self.complete(request)
                    self._raise_if_aborted(request.abort_event)
                    yield ModelStreamEvent.streaming_transport_fallback_completed(
                        identity,
                        reason="non-streaming completion succeeded",
                        replacement_response=replacement,
                    )
                    return
                raise

    async def count_tokens(self, request: TokenCountRequest) -> int | TokenCountResult:
        attempt = 1
        while True:
            try:
                prepared = self.adapter.prepare_token_count(request)
                result = await self.transport.count_tokens(
                    ProviderTransportRequest(
                        prepared_request=prepared,
                        timeout_s=self.timeout_s,
                        metadata=_token_count_metadata(
                            request,
                            attempt=attempt,
                            max_attempts=self.retry_policy.max_attempts,
                        ),
                    )
                )
                if isinstance(result, TokenCountResult):
                    return result
                count = result
                if count < 0:
                    raise ValueError("Provider token count cannot be negative")
                return count
            except asyncio.CancelledError:
                raise
            except Exception as error:
                decision = classify_retry_decision(
                    self.classify_error(error),
                    operation="count_tokens",
                    attempt=attempt,
                    policy=self.retry_policy,
                )
                if not decision.should_retry:
                    raise
                await self._sleep_before_retry(decision.delay_s, None)
                attempt += 1

    def resolve_model(
        self,
        requested: str,
        context: ModelResolveContext,
    ) -> str:
        return self._registry.resolve_model(requested, context)

    def model_info(self, model: str) -> ModelInfo:
        return self._registry.model_info(model)

    def classify_error(self, error: BaseException) -> ProviderError:
        if isinstance(error, ProviderPayloadError):
            return self.adapter.classify_error(error.payload)
        return classify_exception_by_name(error)

    @staticmethod
    def _raise_if_aborted(abort_event: asyncio.Event | None) -> None:
        if abort_event is not None and abort_event.is_set():
            raise asyncio.CancelledError

    async def _sleep_before_retry(
        self,
        delay_s: float,
        abort_event: asyncio.Event | None,
    ) -> None:
        if delay_s <= 0:
            self._raise_if_aborted(abort_event)
            return
        sleep = self.retry_sleep or asyncio.sleep
        if self.retry_sleep is not None:
            await sleep(delay_s)
            self._raise_if_aborted(abort_event)
            return
        if abort_event is None:
            await sleep(delay_s)
            return
        try:
            await asyncio.wait_for(abort_event.wait(), timeout=delay_s)
        except TimeoutError:
            return
        raise asyncio.CancelledError


def _model_request_metadata(
    request: ModelRequest,
    *,
    operation: str,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> FrozenJson:
    metadata: dict[str, object] = {
        "operation": operation,
        "model": request.model,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
        "has_system_prompt": bool(request.system_prompt),
        "has_abort_event": request.abort_event is not None,
        "has_pending_mcp_servers": request.has_pending_mcp_servers,
        "active_agent_count": len(request.active_agents),
        "allowed_agent_types": list(request.allowed_agent_types),
        "mcp_tool_count": len(request.mcp_tool_names),
    }
    _set_attempt_metadata(metadata, attempt=attempt, max_attempts=max_attempts)
    _set_if_not_none(metadata, "fallback_model", request.fallback_model)
    _set_if_not_none(metadata, "effort", request.effort)
    _set_if_not_none(metadata, "agent_id", request.agent_id)
    _set_if_not_none(metadata, "query_source", request.query_source)
    _set_if_not_none(metadata, "tool_choice", request.tool_choice)
    _set_if_not_none(
        metadata,
        "max_output_tokens_override",
        request.max_output_tokens_override,
    )
    if request.task_budget is not None:
        metadata["task_budget"] = {
            "total": request.task_budget.total,
            "remaining": request.task_budget.remaining,
        }
    metadata["cache_policy"] = {
        "skip_cache_write": request.cache_policy.skip_cache_write,
        "cache_scope": request.cache_policy.cache_scope,
    }
    if request.permission_context is not None:
        metadata["permission_context"] = {
            "mode": request.permission_context.mode,
            "always_allow_rule_source_count": _mapping_len(
                request.permission_context.always_allow_rules
            ),
            "always_deny_rule_source_count": _mapping_len(
                request.permission_context.always_deny_rules
            ),
            "always_ask_rule_source_count": _mapping_len(
                request.permission_context.always_ask_rules
            ),
            "should_avoid_permission_prompts": (
                request.permission_context.should_avoid_permission_prompts
            ),
            "is_bypass_permissions_mode_available": (
                request.permission_context.is_bypass_permissions_mode_available
            ),
            "is_auto_mode_available": (
                request.permission_context.is_auto_mode_available
            ),
        }
    if request.budget is not None:
        metadata["budget"] = {
            "requested_model": request.budget.requested_model,
            "effective_model": request.budget.effective_model,
            "context_window": request.budget.context_window,
            "default_max_output_tokens": request.budget.default_max_output_tokens,
            "upper_max_output_tokens": request.budget.upper_max_output_tokens,
            "requested_max_tokens": request.budget.requested_max_tokens,
            "effective_max_tokens": request.budget.effective_max_tokens,
            "input_token_count": request.budget.input_token_count,
            "provider_input_token_count": request.budget.provider_input_token_count,
            "fallback_input_token_count": request.budget.fallback_input_token_count,
            "token_count_fallback_used": request.budget.token_count_fallback_used,
            "token_count_error_type": request.budget.token_count_error_type,
        }
    if request.media_budget is not None:
        metadata["media_budget"] = {
            "max_media_items": request.media_budget.max_media_items,
            "original_media_items": request.media_budget.original_media_items,
            "retained_media_items": request.media_budget.retained_media_items,
            "stripped_media_items": request.media_budget.stripped_media_items,
            "top_level_media_items": request.media_budget.top_level_media_items,
            "nested_media_items": request.media_budget.nested_media_items,
            "mode": request.media_budget.mode,
        }
    return freeze_json(metadata)


def _token_count_metadata(
    request: TokenCountRequest,
    *,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> FrozenJson:
    metadata: dict[str, object] = {
        "operation": "count_tokens",
        "model": request.model,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
        "system_prompt_char_count": len(request.system_prompt),
        "has_thinking": request.thinking is not None,
        "has_media_context": request.media_context is not None,
    }
    _set_attempt_metadata(metadata, attempt=attempt, max_attempts=max_attempts)
    _set_if_not_none(metadata, "effort", request.effort)
    return freeze_json(metadata)


def _set_attempt_metadata(
    metadata: dict[str, object],
    *,
    attempt: int | None,
    max_attempts: int | None,
) -> None:
    if attempt is not None:
        metadata["attempt"] = attempt
    if max_attempts is not None:
        metadata["max_attempts"] = max_attempts


def _set_if_not_none(
    target: dict[str, object],
    key: str,
    value: object | None,
) -> None:
    if value is not None:
        target[key] = value


def _mapping_len(value: object) -> int:
    if isinstance(value, Mapping):
        return len(cast(Mapping[object, object], value))
    return 0


def _provider_error_reason(error: ProviderError) -> str:
    if error.status_code is None:
        return f"{error.kind}: {error.message}"
    return f"{error.kind} ({error.status_code}): {error.message}"


__all__ = ["ProtocolModelProvider"]
