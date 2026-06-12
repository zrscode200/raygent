"""Anthropic Messages-shaped protocol translator.

This is a no-SDK, no-network protocol proof. It translates Raygent-owned model
types to Anthropic Messages-shaped dictionaries and raises Anthropic-shaped
stream fixtures back into `ModelStreamEvent`s.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from raygent_harness.adapters.model_protocols.base import (
    PreparedModelRequest,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_stream import assemble_model_stream
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelContentBlock,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelToolSpec,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    ThinkingContentBlock,
    TokenCountRequest,
    ToolResultContentBlock,
    ToolUseContentBlock,
    UnknownContentBlock,
    Usage,
    build_model_api_error_message,
)

_PROTOCOL_ID = "anthropic_messages"
_SERVER_TOOL_RESULT_NAMES = {
    "web_search_tool_result": "web_search",
    "code_execution_tool_result": "code_execution",
    "web_fetch_tool_result": "web_fetch",
}
_PROMPT_TOKEN_RE = re.compile(r"(\d[\d,]*)\D+(\d[\d,]*)\D+(?:max|maximum|limit)", re.I)


@dataclass(frozen=True, slots=True)
class AnthropicMessagesAdapter:
    """Transport-free Anthropic Messages protocol adapter."""

    protocol_id: str = _PROTOCOL_ID

    def prepare_request(self, request: ModelRequest) -> PreparedModelRequest:
        return self.prepare_stream_request(request)

    def prepare_complete_request(self, request: ModelRequest) -> PreparedModelRequest:
        body = _request_body(request, include_generation=True, stream=False)
        return PreparedModelRequest(
            protocol_id=self.protocol_id,
            model=request.model,
            body=body,
            headers=_headers_from_options(request.provider_options),
            options={"operation": "messages", "mode": "complete"},
        )

    def prepare_stream_request(self, request: ModelRequest) -> PreparedModelRequest:
        body = _request_body(request, include_generation=True, stream=True)
        return PreparedModelRequest(
            protocol_id=self.protocol_id,
            model=request.model,
            body=body,
            headers=_headers_from_options(request.provider_options),
            options={"operation": "messages", "mode": "stream"},
        )

    def parse_response(self, provider_response: ProviderResponse) -> ModelResponse:
        return assemble_model_stream(self.stream_events(_response_events(provider_response)))

    def create_stream_parser(self) -> AnthropicMessagesStreamParser:
        return AnthropicMessagesStreamParser(self)

    def stream_events(
        self,
        provider_events: Iterable[ProviderEvent],
    ) -> Iterable[ModelStreamEvent]:
        parser = self.create_stream_parser()
        for provider_event in provider_events:
            yield from parser.feed(provider_event)
        yield from parser.finish()

    def classify_error(self, provider_error_payload: object) -> ProviderError:
        payload = _mapping(provider_error_payload) or {}
        status_code = _int(payload.get("status_code") or payload.get("status"))
        retry_after_s = _retry_after_s(payload)
        error_type = _string(payload.get("type") or payload.get("error"))
        message = _string(payload.get("message")) or error_type or "Anthropic error"
        raw_details = _raw_details(provider_error_payload, message)
        lowered = f"{error_type or ''} {message}".lower()

        actual_tokens, limit_tokens = _token_gap(raw_details)
        if "prompt is too long" in lowered or (
            "context" in lowered and "too long" in lowered
        ):
            api_error = build_model_api_error_message(
                kind="context_overflow",
                public_message="Prompt is too long",
                raw_details=raw_details,
                actual_tokens=actual_tokens,
                limit_tokens=limit_tokens,
                status_code=status_code,
            )
            return ProviderError(
                kind="context_overflow",
                message=message,
                raw_details=raw_details,
                actual_tokens=actual_tokens,
                limit_tokens=limit_tokens,
                status_code=status_code,
                api_error=api_error,
            )

        if _is_media_overflow(lowered):
            api_error = build_model_api_error_message(
                kind="media_overflow",
                public_message="Media input is too large",
                raw_details=raw_details,
                status_code=status_code,
            )
            return ProviderError(
                kind="media_overflow",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
                api_error=api_error,
            )

        if "max_tokens" in lowered or "max output" in lowered:
            return ProviderError(
                kind="max_output_tokens",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
            )
        if status_code == 429 or "rate" in lowered:
            return ProviderError(
                kind="rate_limit",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
                retry_after_s=retry_after_s,
                retryable=True,
            )
        if status_code == 529 or "overload" in lowered:
            return ProviderError(
                kind="server_overload",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
                retry_after_s=retry_after_s,
                retryable=True,
                safe_to_fallback=True,
            )
        if "timeout" in lowered or "temporar" in lowered or "transient" in lowered:
            return ProviderError(
                kind="transient",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
                retry_after_s=retry_after_s,
                retryable=True,
            )
        if status_code in (401, 403) or "auth" in lowered or "api key" in lowered:
            return ProviderError(
                kind="auth_config",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
            )
        return ProviderError(
            kind="fatal_unknown",
            message=message,
            raw_details=raw_details,
            status_code=status_code,
        )

    def prepare_token_count(self, request: TokenCountRequest) -> PreparedModelRequest:
        body: dict[str, object] = {
            "model": request.model,
            "messages": _lower_messages(request.messages),
        }
        if request.system_prompt:
            body["system"] = [{"type": "text", "text": request.system_prompt}]
        tools = _lower_tools(request.tools)
        if tools:
            body["tools"] = tools
        thinking = _thinking_payload(request.thinking, request.effort)
        if thinking is None:
            thinking = _thinking_payload(request.provider_options, request.effort)
        if thinking is not None:
            body["thinking"] = thinking
        options: dict[str, object] = {"operation": "count_tokens"}
        provider_options = _mapping(thaw_json(request.provider_options))
        if provider_options:
            options["provider_options"] = dict(provider_options)
        return PreparedModelRequest(
            protocol_id=self.protocol_id,
            model=request.model,
            body=body,
            headers=_headers_from_options(request.provider_options),
            options=options,
        )


@dataclass(slots=True)
class AnthropicMessagesStreamParser:
    """Stateful parser for one Anthropic Messages-shaped stream."""

    adapter: AnthropicMessagesAdapter
    message_id: str | None = None
    provider_request_id: str | None = None

    def feed(self, provider_event: ProviderEvent) -> Iterable[ModelStreamEvent]:
        event_type = _string(provider_event.get("type")) or "unknown"
        self.provider_request_id = (
            _string(provider_event.get("request_id"))
            or _string(provider_event.get("provider_request_id"))
            or self.provider_request_id
        )

        if event_type == "message_start":
            message = _mapping(provider_event.get("message"))
            self.message_id = (
                _string(message.get("id")) if message is not None else self.message_id
            )
            usage = (
                _usage_from_anthropic(message.get("usage")) if message is not None else None
            )
            yield ModelStreamEvent.message_start(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                ),
                usage=usage,
            )
            return

        if event_type == "content_block_start":
            index = _index(provider_event.get("index"))
            block_payload = _mapping(provider_event.get("content_block"))
            if block_payload is None:
                return
            yield ModelStreamEvent.content_block_start(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                    content_block_index=index,
                ),
                block=_content_block_from_anthropic(block_payload),
            )
            return

        if event_type == "content_block_delta":
            index = _index(provider_event.get("index"))
            delta = _mapping(provider_event.get("delta")) or {}
            yield ModelStreamEvent.content_block_delta(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                    content_block_index=index,
                ),
                delta=cast(FrozenJson, dict(delta)),
            )
            return

        if event_type == "content_block_stop":
            index = _index(provider_event.get("index"))
            yield ModelStreamEvent.content_block_stop(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                    content_block_index=index,
                )
            )
            return

        if event_type == "message_delta":
            delta = _mapping(provider_event.get("delta")) or {}
            usage = _usage_from_anthropic(provider_event.get("usage"))
            yield ModelStreamEvent.message_delta(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                ),
                usage=usage,
                stop_reason=_string(delta.get("stop_reason")),
                delta=cast(FrozenJson, dict(delta)),
            )
            return

        if event_type == "message_stop":
            yield ModelStreamEvent.message_stop(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                )
            )
            return

        if event_type == "error":
            yield ModelStreamEvent.provider_error_event(
                _identity(
                    message_id=self.message_id,
                    provider_request_id=self.provider_request_id,
                ),
                provider_error=self.adapter.classify_error(provider_event.get("error")),
            )

    def finish(self) -> Iterable[ModelStreamEvent]:
        return ()


def _request_body(
    request: ModelRequest,
    *,
    include_generation: bool,
    stream: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": request.model,
        "messages": _lower_messages(request.messages),
    }
    if request.system_prompt:
        body["system"] = [{"type": "text", "text": request.system_prompt}]
    tools = _lower_tools(request.tools)
    if tools and request.tool_choice != "none":
        body["tools"] = tools
    tool_choice = _tool_choice(request.tool_choice)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    thinking = _thinking_payload(request.provider_options, request.effort)
    if thinking is not None:
        body["thinking"] = thinking
    if include_generation:
        if stream:
            body["stream"] = True
        body["max_tokens"] = (
            request.max_output_tokens_override or request.sampling.max_tokens
        )
        if request.sampling.temperature is not None:
            body["temperature"] = request.sampling.temperature
        if request.sampling.top_p is not None:
            body["top_p"] = request.sampling.top_p
        if request.sampling.top_k is not None:
            body["top_k"] = request.sampling.top_k
        if request.sampling.stop_sequences:
            body["stop_sequences"] = list(request.sampling.stop_sequences)
    return body


def _lower_messages(messages: Sequence[ApiMessage]) -> list[dict[str, object]]:
    lowered: list[dict[str, object]] = []
    for api_message in messages:
        message = api_message.message
        if message.role == "system":
            continue
        role: Literal["user", "assistant"] = (
            "assistant" if message.role == "assistant" else "user"
        )
        lowered.append(
            {
                "role": role,
                "content": [
                    _lower_content_block(block, role=role) for block in message.content
                ],
            }
        )
    return lowered


def _lower_content_block(
    block: ModelContentBlock,
    *,
    role: Literal["user", "assistant"],
) -> dict[str, object]:
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, MediaContentBlock):
        return _lower_media(block)
    if isinstance(block, ToolUseContentBlock):
        payload: dict[str, object] = {
            "type": "server_tool_use" if block.provider_executed else "tool_use",
            "id": block.id,
            "name": block.name,
            "input": thaw_json(block.input),
        }
        metadata = _mapping(thaw_json(block.provider_metadata))
        if metadata is not None and "caller" in metadata:
            payload["caller"] = metadata["caller"]
        return payload
    if isinstance(block, ToolResultContentBlock):
        payload = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": _lower_tool_result_content(block.content),
        }
        if block.is_error:
            payload["is_error"] = True
        metadata = _mapping(thaw_json(block.provider_metadata))
        if metadata is not None:
            for key in ("cache_control", "cache_reference"):
                if key in metadata:
                    payload[key] = metadata[key]
        return payload
    if isinstance(block, ThinkingContentBlock):
        return _lower_thinking_block(block)
    raw_payload = thaw_json(block.payload)
    payload_mapping = _mapping(raw_payload)
    if payload_mapping is not None:
        return dict(payload_mapping)
    return {"type": block.block_type, "payload": raw_payload}


def _lower_tools(tools: Sequence[ModelToolSpec]) -> list[dict[str, object]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": thaw_json(tool.input_schema),
        }
        for tool in tools
    ]


def _lower_media(block: MediaContentBlock) -> dict[str, object]:
    data = thaw_json(block.data)
    raw = dict(_mapping(data) or {"data": data})
    source = raw.get("source")
    source_mapping = _mapping(source)
    if source_mapping is not None:
        return {"type": block.media_kind, "source": dict(source_mapping)}
    if block.media_kind == "image":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": str(raw.get("data", "")),
            },
        }
    raw.setdefault("type", block.media_kind)
    raw.setdefault("media_type", block.media_type)
    return raw


def _lower_tool_result_content(content: FrozenJson) -> object:
    raw: object = thaw_json(content)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        raw_items = cast(list[object], raw)
        if _is_tool_result_content_array(raw_items):
            return [_lower_tool_result_item(item) for item in raw_items]
        return _stable_json_text(raw_items)
    return _stable_json_text(raw)


def _lower_tool_result_item(item: object) -> object:
    mapping = _mapping(item)
    if mapping is None:
        return item
    item_type = mapping.get("type")
    if item_type == "text":
        return {"type": "text", "text": str(mapping.get("text", ""))}
    if item_type == "tool_reference":
        return dict(mapping)
    if item_type == "document":
        source = mapping.get("source")
        source_mapping = _mapping(source)
        if source_mapping is not None:
            return {"type": "document", "source": dict(source_mapping)}
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": str(mapping.get("media_type", "application/pdf")),
                "data": str(mapping.get("data", "")),
            },
        }
    if item_type in ("image", "media"):
        media_type = str(mapping.get("media_type", mapping.get("mediaType", "")))
        source = mapping.get("source")
        source_mapping = _mapping(source)
        if source_mapping is not None:
            return {"type": "image", "source": dict(source_mapping)}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": str(mapping.get("data", "")),
            },
        }
    return dict(mapping)


def _is_tool_result_content_array(raw: list[object]) -> bool:
    if not raw:
        return True
    for item in raw:
        mapping = _mapping(item)
        if mapping is None:
            return False
        item_type = mapping.get("type")
        if item_type not in ("text", "image", "media", "document", "tool_reference"):
            return False
    return True


def _stable_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lower_thinking_block(block: ThinkingContentBlock) -> dict[str, object]:
    metadata = block.provider_metadata
    raw = thaw_json(metadata) if metadata is not None else None
    raw_mapping = _mapping(raw)
    if block.redacted and raw_mapping is not None and "type" in raw_mapping:
        return dict(raw_mapping)
    payload: dict[str, object] = {"type": "thinking", "thinking": block.text}
    if block.signature is not None:
        payload["signature"] = block.signature
    return payload


def _tool_choice(value: str | None) -> dict[str, object] | None:
    if value is None:
        return None
    if value == "auto":
        return {"type": "auto"}
    if value in ("any", "required"):
        return {"type": "any"}
    if value == "none":
        return None
    if value.startswith("tool:"):
        return {"type": "tool", "name": value.removeprefix("tool:")}
    return {"type": "tool", "name": value}


def _thinking_payload(options: object, effort: str | int | None) -> dict[str, object] | None:
    option_payload = _mapping(thaw_json(cast(FrozenJson, options)))
    thinking: object | None = None
    if option_payload is not None:
        if option_payload.get("type") == "enabled":
            thinking = option_payload
        else:
            anthropic = option_payload.get("anthropic")
            anthropic_mapping = _mapping(anthropic)
            if anthropic_mapping is not None:
                thinking = anthropic_mapping.get("thinking")
            thinking = thinking or option_payload.get("thinking")
    thinking_mapping = _mapping(thinking)
    if thinking_mapping is not None:
        if thinking_mapping.get("type") == "enabled":
            direct_budget = thinking_mapping.get("budget_tokens") or thinking_mapping.get(
                "budgetTokens"
            )
            if isinstance(direct_budget, int | float):
                return {"type": "enabled", "budget_tokens": int(direct_budget)}
        budget = thinking_mapping.get("budget_tokens") or thinking_mapping.get(
            "budgetTokens"
        )
        if isinstance(budget, int | float):
            return {"type": "enabled", "budget_tokens": int(budget)}
    if isinstance(effort, int) and effort > 0:
        return {"type": "enabled", "budget_tokens": effort}
    if isinstance(effort, str):
        budgets = {"low": 1024, "medium": 4096, "high": 8192}
        budget = budgets.get(effort)
        if budget is not None:
            return {"type": "enabled", "budget_tokens": budget}
    return None


def _headers_from_options(options: FrozenJson) -> dict[str, object]:
    raw = _mapping(thaw_json(options))
    if raw is None:
        return {}
    anthropic = raw.get("anthropic")
    anthropic_mapping = _mapping(anthropic)
    if anthropic_mapping is None:
        return {}
    headers = _mapping(anthropic_mapping.get("headers"))
    if headers is None:
        return {}
    return dict(headers)


def _content_block_from_anthropic(block: Mapping[str, object]) -> ModelContentBlock:
    block_type = _string(block.get("type")) or "unknown"
    if block_type == "text":
        return TextContentBlock(text=_string(block.get("text")) or "")
    if block_type == "thinking":
        return ThinkingContentBlock(
            text=_string(block.get("thinking")) or "",
            signature=_string(block.get("signature")),
        )
    if block_type in ("redacted_thinking", "encrypted_thinking"):
        return ThinkingContentBlock(
            text=_string(block.get("thinking") or block.get("text")) or "",
            signature=_string(block.get("signature")),
            redacted=True,
            provider_metadata=cast(FrozenJson, dict(block)),
        )
    if block_type in ("tool_use", "server_tool_use"):
        return ToolUseContentBlock(
            id=_string(block.get("id")) or "",
            name=_string(block.get("name")) or "",
            input=cast(FrozenJson, block.get("input", {})),
            provider_executed=block_type == "server_tool_use",
            provider_metadata=(
                cast(FrozenJson, {"type": "server_tool_use"})
                if block_type == "server_tool_use"
                else None
            ),
        )
    if block_type in _SERVER_TOOL_RESULT_NAMES:
        return ToolResultContentBlock(
            tool_use_id=_string(block.get("tool_use_id")) or "",
            content=cast(FrozenJson, block.get("content", {})),
            is_error=_server_tool_result_is_error(block.get("content")),
            provider_metadata=cast(
                FrozenJson,
                {"type": block_type, "provider_executed": True},
            ),
        )
    return UnknownContentBlock(block_type=block_type, payload=cast(FrozenJson, dict(block)))


def _usage_from_anthropic(value: object) -> Usage | None:
    usage = _mapping(value)
    if usage is None:
        return None
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0
    cache_creation = _int(usage.get("cache_creation_input_tokens")) or 0
    cache_read = _int(usage.get("cache_read_input_tokens")) or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        provider_metadata=cast(FrozenJson, {"anthropic": dict(usage)}),
    )


def _identity(
    *,
    message_id: str | None,
    provider_request_id: str | None,
    content_block_index: int | None = None,
) -> StreamIdentity:
    return StreamIdentity(
        message_id=message_id,
        provider_request_id=provider_request_id or (
            "anthropic_stream" if message_id is None else None
        ),
        content_block_index=content_block_index,
    )


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in cast(Mapping[object, object], value).items()
        }
    return None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _retry_after_s(payload: Mapping[str, object]) -> float | None:
    for key in (
        "retry_after_s",
        "retry_after",
        "retry-after",
        "Retry-After",
        "retry_after_seconds",
    ):
        delay = _float(payload.get(key))
        if delay is not None:
            return delay
    for key in ("headers", "response_headers"):
        headers = _mapping(payload.get(key))
        if headers is not None:
            delay = _retry_after_header_s(headers)
            if delay is not None:
                return delay
    return None


def _retry_after_header_s(headers: Mapping[str, object]) -> float | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    return _float(lowered.get("retry-after") or lowered.get("retry_after"))


def _index(value: object) -> int:
    index = _int(value)
    return index if index is not None else 0


def _raw_details(payload: object, fallback: str) -> str:
    payload_mapping = _mapping(payload)
    if payload_mapping is not None:
        return str(dict(payload_mapping))
    return fallback


def _token_gap(raw: str) -> tuple[int | None, int | None]:
    match = _PROMPT_TOKEN_RE.search(raw.replace(",", ""))
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _is_media_overflow(text: str) -> bool:
    return (
        ("image exceeds" in text and "maximum" in text)
        or "image dimensions exceed" in text
        or "many-image" in text
        or ("pdf pages" in text and "maximum" in text)
    )


def _server_tool_result_is_error(content: object) -> bool:
    content_payload = _mapping(content)
    content_type = _string(content_payload.get("type")) if content_payload else None
    return bool(content_type and content_type.endswith("_tool_result_error"))


def _response_events(response: ProviderResponse) -> Iterable[ProviderEvent]:
    message_id = _string(response.get("id"))
    request_id = _string(response.get("request_id") or response.get("provider_request_id"))
    usage = response.get("usage")
    yield {
        "type": "message_start",
        "request_id": request_id,
        "message": {"id": message_id, "usage": usage},
    }

    content = response.get("content")
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        for index, raw_block in enumerate(cast(Sequence[object], content)):
            block = _mapping(raw_block)
            if block is None:
                continue
            yield {
                "type": "content_block_start",
                "index": index,
                "content_block": dict(block),
            }
            yield from _content_block_deltas(index, block)
            yield {"type": "content_block_stop", "index": index}

    stop_reason = _string(response.get("stop_reason"))
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": usage,
    }
    yield {"type": "message_stop"}


def _content_block_deltas(
    index: int,
    block: Mapping[str, object],
) -> Iterable[ProviderEvent]:
    block_type = _string(block.get("type")) or "unknown"
    if block_type == "text":
        text = _string(block.get("text")) or ""
        if text:
            yield {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
        return
    if block_type == "thinking":
        thinking = _string(block.get("thinking") or block.get("text")) or ""
        if thinking:
            yield {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": thinking},
            }
        signature = _string(block.get("signature"))
        if signature is not None:
            yield {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": signature},
            }
        return
    if block_type in ("redacted_thinking", "encrypted_thinking"):
        text = _string(block.get("thinking") or block.get("text")) or ""
        if text:
            yield {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": text},
            }
        signature = _string(block.get("signature"))
        if signature is not None:
            yield {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": signature},
            }
        return


__all__ = ["AnthropicMessagesAdapter", "AnthropicMessagesStreamParser"]
