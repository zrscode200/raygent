"""OpenAI Responses-shaped protocol translator.

This is a no-SDK, no-network protocol proof. It translates Raygent-owned model
requests to OpenAI Responses-shaped dictionaries and raises Responses-shaped
stream fixtures back into `ModelStreamEvent`s.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

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

_PROTOCOL_ID = "openai_responses"
_CONTEXT_TOKEN_RE = re.compile(
    r"(\d[\d,]*)\D+(\d[\d,]*)\D+(?:tokens|context|maximum|limit)",
    re.I,
)
_REQUESTED_TOKEN_RES = (
    re.compile(r"requested\s+(\d[\d,]*)\s+tokens", re.I),
    re.compile(r"request[^\d]*(\d[\d,]*)\s+tokens", re.I),
    re.compile(r"input[^\d]*(\d[\d,]*)\s+tokens", re.I),
    re.compile(r"prompt[^\d]*(\d[\d,]*)\s+tokens", re.I),
)
_LIMIT_TOKEN_RES = (
    re.compile(r"maximum context length is\s+(\d[\d,]*)\s+tokens", re.I),
    re.compile(
        r"max(?:imum)?(?: context)?(?: length)?[^\d]*(\d[\d,]*)\s+tokens",
        re.I,
    ),
    re.compile(r"limit[^\d]*(\d[\d,]*)\s+tokens", re.I),
)
_PDF_PAGE_LIMIT_RE = re.compile(r"maximum of \d+ pdf pages", re.I)
_HOSTED_TOOLS: Mapping[str, str] = {
    "web_search_call": "web_search",
    "web_search_preview_call": "web_search_preview",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
    "computer_use_call": "computer_use",
    "image_generation_call": "image_generation",
    "mcp_call": "mcp",
    "local_shell_call": "local_shell",
}


@dataclass(frozen=True, slots=True)
class OpenAIResponsesAdapter:
    """Transport-free OpenAI Responses protocol adapter."""

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
            options={"operation": "responses", "mode": "complete"},
        )

    def prepare_stream_request(self, request: ModelRequest) -> PreparedModelRequest:
        body = _request_body(request, include_generation=True, stream=True)
        return PreparedModelRequest(
            protocol_id=self.protocol_id,
            model=request.model,
            body=body,
            headers=_headers_from_options(request.provider_options),
            options={"operation": "responses", "mode": "stream"},
        )

    def parse_response(self, provider_response: ProviderResponse) -> ModelResponse:
        return assemble_model_stream(self.stream_events(_response_events(provider_response)))

    def create_stream_parser(self) -> OpenAIResponsesStreamParser:
        return OpenAIResponsesStreamParser(self)

    def stream_events(
        self,
        provider_events: Iterable[ProviderEvent],
    ) -> Iterable[ModelStreamEvent]:
        parser = self.create_stream_parser()
        for provider_event in provider_events:
            yield from parser.feed(provider_event)
        yield from parser.finish()

    def classify_error(self, provider_error_payload: object) -> ProviderError:
        payload = _error_payload(provider_error_payload)
        status_code = _int(payload.get("status_code") or payload.get("status"))
        retry_after_s = _retry_after_s(payload)
        code = _string(payload.get("code") or payload.get("type") or payload.get("error"))
        message = _string(payload.get("message")) or code or "OpenAI Responses error"
        raw_details = _raw_details(provider_error_payload, message)
        lowered = f"{code or ''} {message} {raw_details}".lower()

        actual_tokens, limit_tokens = _token_gap(message)
        if actual_tokens is None or limit_tokens is None:
            actual_tokens, limit_tokens = _token_gap(raw_details)
        if _is_context_overflow(lowered):
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

        if "max_output_tokens" in lowered or "max output" in lowered:
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
        if status_code in (503, 529) or "overload" in lowered or "capacity" in lowered:
            return ProviderError(
                kind="server_overload",
                message=message,
                raw_details=raw_details,
                status_code=status_code,
                retry_after_s=retry_after_s,
                retryable=True,
                safe_to_fallback=True,
            )
        if status_code in (500, 502, 504) or "timeout" in lowered or "temporar" in lowered:
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
            "input": _lower_messages(
                request.messages,
                system_prompt=request.system_prompt,
                provider_options=request.provider_options,
            ),
        }
        tools = _lower_tools(request.tools)
        if tools:
            body["tools"] = tools
        reasoning = _reasoning_config(request.provider_options, request.effort)
        if reasoning is None:
            reasoning = _reasoning_config(request.thinking, request.effort)
        if reasoning is not None:
            body["reasoning"] = reasoning
        if request.media_context is not None:
            body["media_context"] = thaw_json(request.media_context)
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
class _OpenAIStreamState:
    message_started: bool = False
    message_id: str | None = None
    provider_request_id: str | None = None
    next_block_index: int = 0
    open_blocks: dict[str, int] = field(default_factory=dict[str, int])
    tool_delta_seen: set[str] = field(default_factory=set[str])
    has_function_call: bool = False

    def identity(self, *, block_key: str | None = None) -> StreamIdentity:
        return StreamIdentity(
            message_id=self.message_id,
            provider_request_id=self.provider_request_id or "openai_responses_stream",
            content_block_index=(
                self.open_blocks[block_key]
                if block_key is not None and block_key in self.open_blocks
                else None
            ),
        )

    def ensure_message_started(self) -> list[ModelStreamEvent]:
        if self.message_started:
            return []
        self.message_started = True
        return [ModelStreamEvent.message_start(self.identity())]

    def block_index(self, block_key: str) -> int:
        existing = self.open_blocks.get(block_key)
        if existing is not None:
            return existing
        index = self.next_block_index
        self.next_block_index += 1
        self.open_blocks[block_key] = index
        return index

    def close_block(self, block_key: str) -> int | None:
        return self.open_blocks.pop(block_key, None)


@dataclass(slots=True)
class OpenAIResponsesStreamParser:
    """Stateful parser for one OpenAI Responses-shaped stream."""

    adapter: OpenAIResponsesAdapter
    state: _OpenAIStreamState = field(default_factory=_OpenAIStreamState)

    def feed(self, provider_event: ProviderEvent) -> Iterable[ModelStreamEvent]:
        return _stream_event(provider_event, self.state, self.adapter)

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
        "input": _lower_messages(
            request.messages,
            system_prompt=request.system_prompt,
            provider_options=request.provider_options,
        ),
    }
    tools = _lower_tools(request.tools)
    if tools:
        body["tools"] = tools
    tool_choice = _tool_choice(request.tool_choice)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    body.update(_openai_options(request.provider_options, request.effort))
    if include_generation:
        if stream:
            body["stream"] = True
        body["max_output_tokens"] = (
            request.max_output_tokens_override or request.sampling.max_tokens
        )
        if request.sampling.temperature is not None:
            body["temperature"] = request.sampling.temperature
        if request.sampling.top_p is not None:
            body["top_p"] = request.sampling.top_p
    return body


def _lower_messages(
    messages: Sequence[ApiMessage],
    *,
    system_prompt: str,
    provider_options: object,
) -> list[dict[str, object]]:
    lowered: list[dict[str, object]] = []
    if system_prompt:
        lowered.append({"role": "system", "content": system_prompt})

    store = _openai_store(provider_options)
    for api_message in messages:
        message = api_message.message
        if message.role == "system":
            text = _join_text_blocks(message.content)
            if text:
                lowered.append({"role": "system", "content": text})
            continue
        if message.role == "user":
            lowered.extend(_lower_user_message(message.content))
            continue
        if message.role == "assistant":
            lowered.extend(_lower_assistant_message(message.content, store=store))
            continue
        if message.role == "tool":
            for block in message.content:
                if isinstance(block, ToolResultContentBlock):
                    lowered.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.tool_use_id,
                            "output": _lower_tool_result_output(block),
                        }
                    )
    return lowered


def _lower_user_message(
    content: Sequence[ModelContentBlock],
) -> list[dict[str, object]]:
    """Lower Raygent user-role messages to Responses input items.

    Raygent's model-visible tool results are user-role messages for provider
    neutrality. OpenAI Responses expects those as top-level
    `function_call_output` items, not as ordinary user input content.
    """
    lowered: list[dict[str, object]] = []
    user_items: list[dict[str, object]] = []

    def flush_user_items() -> None:
        if not user_items:
            return
        lowered.append({"role": "user", "content": list(user_items)})
        user_items.clear()

    for block in content:
        if isinstance(block, ToolResultContentBlock):
            flush_user_items()
            lowered.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": _lower_tool_result_output(block),
                }
            )
            continue
        user_items.append(_lower_user_content(block))
    flush_user_items()
    return lowered


def _lower_assistant_message(
    content: Sequence[ModelContentBlock],
    *,
    store: bool | None,
) -> list[dict[str, object]]:
    lowered: list[dict[str, object]] = []
    text_items: list[dict[str, object]] = []

    def flush_text() -> None:
        if not text_items:
            return
        lowered.append({"role": "assistant", "content": list(text_items)})
        text_items.clear()

    for block in content:
        if isinstance(block, TextContentBlock):
            text_items.append({"type": "output_text", "text": block.text})
            continue
        if isinstance(block, ThinkingContentBlock):
            flush_text()
            reasoning = _lower_reasoning_item(block, store=store)
            if reasoning is not None:
                lowered.append(reasoning)
            continue
        if isinstance(block, ToolUseContentBlock):
            flush_text()
            lowered.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": _stable_json_text(thaw_json(block.input)),
                }
            )
            continue
        if isinstance(block, UnknownContentBlock):
            payload = thaw_json(block.payload)
            payload_mapping = _mapping(payload)
            if payload_mapping is not None:
                flush_text()
                lowered.append(dict(payload_mapping))
    flush_text()
    return lowered


def _lower_user_content(block: ModelContentBlock) -> dict[str, object]:
    if isinstance(block, TextContentBlock):
        return {"type": "input_text", "text": block.text}
    if isinstance(block, MediaContentBlock) and block.media_kind == "image":
        return {"type": "input_image", "image_url": _media_url(block)}
    if isinstance(block, MediaContentBlock):
        raise ValueError("OpenAI Responses user media content only supports images")
    if isinstance(block, UnknownContentBlock):
        payload = thaw_json(block.payload)
        payload_mapping = _mapping(payload)
        if payload_mapping is not None:
            return dict(payload_mapping)
    return {"type": "input_text", "text": _stringify_content(block)}


def _lower_reasoning_item(
    block: ThinkingContentBlock,
    *,
    store: bool | None,
) -> dict[str, object] | None:
    metadata = _openai_metadata(block.provider_metadata)
    item_id = _string(metadata.get("itemId") or metadata.get("item_id") or metadata.get("id"))
    encrypted = _string(
        metadata.get("reasoningEncryptedContent")
        or metadata.get("reasoning_encrypted_content")
        or metadata.get("encrypted_content")
    )
    if item_id is None:
        return None
    if store is False and encrypted is None:
        return None
    payload: dict[str, object] = {
        "type": "reasoning",
        "id": item_id,
        "summary": ([{"type": "summary_text", "text": block.text}] if block.text else []),
    }
    if encrypted is not None:
        payload["encrypted_content"] = encrypted
    else:
        payload["encrypted_content"] = None
    return payload


def _lower_tool_result_output(block: ToolResultContentBlock) -> object:
    content = thaw_json(block.content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        raw_items = cast(list[object], content)
        if _is_tool_result_content_array(raw_items):
            return [_lower_tool_result_item(item) for item in raw_items]
        return _stable_json_text(raw_items)
    return _stable_json_text(content)


def _lower_tool_result_item(item: object) -> dict[str, object]:
    mapping = _mapping(item) or {}
    item_type = mapping.get("type")
    if item_type == "text":
        return {"type": "input_text", "text": str(mapping.get("text", ""))}
    if item_type in ("image", "media"):
        media_type = str(mapping.get("media_type", mapping.get("mediaType", "")))
        if media_type and not media_type.startswith("image/"):
            raise ValueError("OpenAI Responses tool-result media content only supports images")
        return {
            "type": "input_image",
            "image_url": _media_url_from_mapping(mapping, media_type=media_type),
        }
    return {"type": "input_text", "text": _stable_json_text(dict(mapping))}


def _lower_tools(tools: Sequence[ModelToolSpec]) -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": thaw_json(tool.input_schema),
        }
        for tool in tools
    ]


def _tool_choice(value: str | None) -> object | None:
    if value is None:
        return None
    if value in ("auto", "none"):
        return value
    if value in ("any", "required"):
        return "required"
    if value.startswith("tool:"):
        return {"type": "function", "name": value.removeprefix("tool:")}
    return {"type": "function", "name": value}


def _openai_options(options: object, effort: str | int | None) -> dict[str, object]:
    openai = _openai_options_mapping(options)
    payload: dict[str, object] = {}
    instructions = _string(openai.get("instructions"))
    if instructions:
        payload["instructions"] = instructions
    store = _openai_store(options)
    if store is not None:
        payload["store"] = store
    prompt_cache_key = _string(openai.get("prompt_cache_key") or openai.get("promptCacheKey"))
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    include = openai.get("include")
    if isinstance(include, Sequence) and not isinstance(include, str | bytes | bytearray):
        payload["include"] = [str(item) for item in cast(Sequence[object], include)]
    elif openai.get("encrypted_reasoning") is True or openai.get("encryptedReasoning") is True:
        payload["include"] = ["reasoning.encrypted_content"]
    reasoning = _reasoning_config(options, effort)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    verbosity = _string(openai.get("text_verbosity") or openai.get("textVerbosity"))
    if verbosity:
        payload["text"] = {"verbosity": verbosity}
    return payload


def _reasoning_config(options: object, effort: str | int | None) -> dict[str, object] | None:
    openai = _openai_options_mapping(options)
    raw_reasoning = _mapping(openai.get("reasoning"))
    configured_effort = None
    if raw_reasoning is not None:
        configured_effort = _string(raw_reasoning.get("effort"))
    configured_effort = configured_effort or _string(
        openai.get("reasoning_effort") or openai.get("reasoningEffort")
    )
    if configured_effort is None and isinstance(effort, str):
        configured_effort = effort
    summary = _string(openai.get("reasoning_summary") or openai.get("reasoningSummary"))
    if raw_reasoning is not None and summary is None:
        summary = _string(raw_reasoning.get("summary"))
    payload: dict[str, object] = {}
    if configured_effort is not None:
        payload["effort"] = configured_effort
    if summary is not None:
        payload["summary"] = summary
    return payload or None


def _headers_from_options(options: FrozenJson) -> dict[str, object]:
    openai = _openai_options_mapping(options)
    headers = _mapping(openai.get("headers"))
    return dict(headers) if headers is not None else {}


def _stream_event(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
    adapter: OpenAIResponsesAdapter,
) -> list[ModelStreamEvent]:
    event_type = _string(provider_event.get("type")) or "unknown"
    _update_stream_identity(provider_event, state)

    if event_type in ("response.created", "response.in_progress"):
        return state.ensure_message_started()

    if event_type == "response.output_text.delta":
        return _open_text_delta(provider_event, state)

    if event_type in (
        "response.output_text.done",
        "response.content_part.done",
    ):
        return _close_text(provider_event, state)

    if event_type in (
        "response.reasoning_text.delta",
        "response.reasoning_summary.delta",
        "response.reasoning_summary_text.delta",
    ):
        return _reasoning_delta(provider_event, state)

    if event_type == "response.output_item.added":
        return _output_item_added(provider_event, state)

    if event_type == "response.function_call_arguments.delta":
        return _function_arguments_delta(provider_event, state)

    if event_type in (
        "response.function_call_arguments.done",
        "response.output_item.done",
    ):
        return _output_item_done(provider_event, state)

    if event_type in ("response.completed", "response.incomplete"):
        return _response_finished(provider_event, state)

    if event_type in ("response.failed", "error"):
        identity = state.identity()
        return [
            *state.ensure_message_started(),
            ModelStreamEvent.provider_error_event(
                identity,
                provider_error=adapter.classify_error(_provider_error_payload(provider_event)),
            ),
        ]

    return []


def _update_stream_identity(provider_event: ProviderEvent, state: _OpenAIStreamState) -> None:
    response = _mapping(provider_event.get("response"))
    response_id = _string(response.get("id")) if response is not None else None
    if response_id is None:
        response_id = _string(provider_event.get("response_id"))
    if response_id is not None:
        state.message_id = state.message_id or response_id
        state.provider_request_id = state.provider_request_id or response_id
    request_id = _string(
        provider_event.get("request_id") or provider_event.get("provider_request_id")
    )
    if request_id is not None:
        state.provider_request_id = request_id


def _open_text_delta(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    block_key = _item_id(provider_event, fallback="text-0")
    events = state.ensure_message_started()
    if block_key not in state.open_blocks:
        state.block_index(block_key)
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=block_key),
                block=TextContentBlock(text=""),
            )
        )
    delta = str(provider_event.get("delta", ""))
    if delta:
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=block_key),
                delta={"type": "text_delta", "text": delta},
            )
        )
    return events


def _close_text(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    block_key = _item_id(provider_event, fallback="text-0")
    index = state.close_block(block_key)
    if index is None:
        return []
    return [
        ModelStreamEvent.content_block_stop(
            StreamIdentity(
                message_id=state.message_id,
                provider_request_id=state.provider_request_id or "openai_responses_stream",
                content_block_index=index,
            )
        )
    ]


def _output_item_added(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item = _mapping(provider_event.get("item"))
    if item is None or item.get("type") != "function_call":
        return state.ensure_message_started()
    item_id = _string(item.get("id")) or _item_id(provider_event, fallback="function-0")
    call_id = _string(item.get("call_id")) or item_id
    name = _string(item.get("name")) or ""
    state.has_function_call = True
    events = state.ensure_message_started()
    if item_id not in state.open_blocks:
        state.block_index(item_id)
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=item_id),
                block=ToolUseContentBlock(
                    id=call_id,
                    name=name,
                    input={},
                    provider_metadata={"openai": {"item_id": item_id}},
                ),
            )
        )
    return events


def _function_arguments_delta(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item_id = _item_id(provider_event, fallback="function-0")
    events = state.ensure_message_started()
    if item_id not in state.open_blocks:
        state.block_index(item_id)
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=item_id),
                block=ToolUseContentBlock(id=item_id, name="", input={}),
            )
        )
    delta = str(provider_event.get("delta", ""))
    if delta:
        state.tool_delta_seen.add(item_id)
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=item_id),
                delta={"type": "input_json_delta", "partial_json": delta},
            )
        )
    return events


def _output_item_done(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item = _mapping(provider_event.get("item"))
    if item is None:
        return []
    item_type = _string(item.get("type")) or "unknown"
    if item_type == "function_call":
        return _finish_function_call(item, provider_event, state)
    if item_type == "reasoning":
        return _finish_reasoning(item, provider_event, state)
    if item_type in _HOSTED_TOOLS:
        return _finish_hosted_tool(item, state)
    return []


def _finish_function_call(
    item: Mapping[str, object],
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item_id = _string(item.get("id")) or _item_id(provider_event, fallback="function-0")
    call_id = _string(item.get("call_id")) or item_id
    name = _string(item.get("name")) or ""
    state.has_function_call = True
    events = state.ensure_message_started()
    if item_id not in state.open_blocks:
        state.block_index(item_id)
        input_value: object = (
            _parse_json(item.get("arguments"))
            if item.get("arguments") is not None
            else cast(dict[str, object], {})
        )
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=item_id),
                block=ToolUseContentBlock(
                    id=call_id,
                    name=name,
                    input=cast(FrozenJson, input_value),
                    provider_metadata={"openai": {"item_id": item_id}},
                ),
            )
        )
    elif item_id not in state.tool_delta_seen and item.get("arguments") is not None:
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=item_id),
                delta={
                    "type": "input_json_delta",
                    "partial_json": str(item.get("arguments", "")),
                },
            )
        )
    index = state.close_block(item_id)
    if index is not None:
        events.append(
            ModelStreamEvent.content_block_stop(
                StreamIdentity(
                    message_id=state.message_id,
                    provider_request_id=state.provider_request_id or "openai_responses_stream",
                    content_block_index=index,
                )
            )
        )
    return events


def _finish_reasoning(
    item: Mapping[str, object],
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item_id = _string(item.get("id")) or _item_id(provider_event, fallback="reasoning-0")
    summary_text = _reasoning_summary_text(item)
    encrypted = item.get("encrypted_content")
    metadata = {
        "openai": {
            "itemId": item_id,
            "reasoningEncryptedContent": encrypted,
        }
    }
    events = state.ensure_message_started()
    started_block = False
    if item_id not in state.open_blocks:
        started_block = True
        state.block_index(item_id)
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=item_id),
                block=ThinkingContentBlock(
                    text="",
                    redacted=encrypted is not None,
                    provider_metadata=cast(FrozenJson, metadata),
                ),
            )
        )
    if encrypted is not None:
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=item_id),
                delta=cast(FrozenJson, {
                    "type": "provider_metadata_delta",
                    "provider_metadata": metadata,
                    "redacted": True,
                }),
            )
        )
    if summary_text and started_block:
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=item_id),
                delta={"type": "thinking_delta", "thinking": summary_text},
            )
        )
    index = state.close_block(item_id)
    if index is not None:
        events.append(
            ModelStreamEvent.content_block_stop(
                StreamIdentity(
                    message_id=state.message_id,
                    provider_request_id=state.provider_request_id or "openai_responses_stream",
                    content_block_index=index,
                )
            )
        )
    return events


def _finish_hosted_tool(
    item: Mapping[str, object],
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item_type = _string(item.get("type")) or "unknown"
    item_id = _string(item.get("id")) or item_type
    name = _HOSTED_TOOLS[item_type]
    metadata = cast(FrozenJson, {"openai": {"item_id": item_id, "item_type": item_type}})
    events = state.ensure_message_started()

    call_key = f"{item_id}:call"
    state.block_index(call_key)
    events.append(
        ModelStreamEvent.content_block_start(
            state.identity(block_key=call_key),
            block=ToolUseContentBlock(
                id=item_id,
                name=name,
                input=cast(FrozenJson, _hosted_tool_input(item)),
                provider_executed=True,
                provider_metadata=metadata,
            ),
        )
    )
    call_index = state.close_block(call_key)
    if call_index is not None:
        events.append(
            ModelStreamEvent.content_block_stop(
                StreamIdentity(
                    message_id=state.message_id,
                    provider_request_id=state.provider_request_id or "openai_responses_stream",
                    content_block_index=call_index,
                )
            )
        )

    result_key = f"{item_id}:result"
    state.block_index(result_key)
    events.append(
        ModelStreamEvent.content_block_start(
            state.identity(block_key=result_key),
            block=ToolResultContentBlock(
                tool_use_id=item_id,
                content=cast(FrozenJson, dict(item)),
                is_error=item.get("error") is not None,
                provider_metadata=metadata,
            ),
        )
    )
    result_index = state.close_block(result_key)
    if result_index is not None:
        events.append(
            ModelStreamEvent.content_block_stop(
                StreamIdentity(
                    message_id=state.message_id,
                    provider_request_id=state.provider_request_id or "openai_responses_stream",
                    content_block_index=result_index,
                )
            )
        )
    return events


def _response_finished(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    events = state.ensure_message_started()
    for block_key, index in list(state.open_blocks.items()):
        state.close_block(block_key)
        events.append(
            ModelStreamEvent.content_block_stop(
                StreamIdentity(
                    message_id=state.message_id,
                    provider_request_id=state.provider_request_id or "openai_responses_stream",
                    content_block_index=index,
                )
            )
        )
    response = _mapping(provider_event.get("response"))
    usage = _usage_from_openai(response.get("usage") if response is not None else None)
    stop_reason = _finish_reason(provider_event, state)
    events.append(
        ModelStreamEvent.message_delta(
            state.identity(),
            usage=usage,
            stop_reason=stop_reason,
            delta={"type": event_type_or_unknown(provider_event), "stop_reason": stop_reason},
        )
    )
    events.append(ModelStreamEvent.message_stop(state.identity()))
    return events


def event_type_or_unknown(provider_event: ProviderEvent) -> str:
    return _string(provider_event.get("type")) or "unknown"


def _provider_error_payload(provider_event: ProviderEvent) -> object:
    if provider_event.get("type") == "response.failed":
        response = _mapping(provider_event.get("response"))
        if response is not None:
            error = response.get("error")
            if error is not None:
                return error
    if provider_event.get("error") is not None:
        return provider_event.get("error")
    return dict(provider_event)


def _usage_from_openai(value: object) -> Usage | None:
    usage = _mapping(value)
    if usage is None:
        return None
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0
    total_tokens = _int(usage.get("total_tokens"))
    input_details = _mapping(usage.get("input_tokens_details")) or {}
    output_details = _mapping(usage.get("output_tokens_details")) or {}
    cached_tokens = _int(input_details.get("cached_tokens")) or 0
    reasoning_tokens = _int(output_details.get("reasoning_tokens")) or 0
    return Usage(
        input_tokens=max(0, input_tokens - cached_tokens),
        output_tokens=output_tokens,
        cache_read_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        provider_metadata=cast(FrozenJson, {"openai": dict(usage)}),
    )


def _reasoning_delta(
    provider_event: ProviderEvent,
    state: _OpenAIStreamState,
) -> list[ModelStreamEvent]:
    item_id = _item_id(provider_event, fallback="reasoning-0")
    events = state.ensure_message_started()
    if item_id not in state.open_blocks:
        state.block_index(item_id)
        events.append(
            ModelStreamEvent.content_block_start(
                state.identity(block_key=item_id),
                block=ThinkingContentBlock(
                    text="",
                    provider_metadata={"openai": {"itemId": item_id}},
                ),
            )
        )
    delta = str(provider_event.get("delta", ""))
    if delta:
        events.append(
            ModelStreamEvent.content_block_delta(
                state.identity(block_key=item_id),
                delta={"type": "thinking_delta", "thinking": delta},
            )
        )
    return events


def _finish_reason(provider_event: ProviderEvent, state: _OpenAIStreamState) -> str:
    if provider_event.get("type") == "response.incomplete":
        response = _mapping(provider_event.get("response"))
        details = _mapping(response.get("incomplete_details")) if response is not None else None
        reason = _string(details.get("reason")) if details is not None else None
        if reason == "max_output_tokens":
            return "max_output_tokens"
        if reason == "content_filter":
            return "content_filter"
        return reason or "incomplete"
    return "tool_use" if state.has_function_call else "end_turn"


def _item_id(provider_event: ProviderEvent, *, fallback: str) -> str:
    value = provider_event.get("item_id")
    if isinstance(value, str) and value:
        return value
    item = _mapping(provider_event.get("item"))
    if item is not None:
        item_id = _string(item.get("id"))
        if item_id:
            return item_id
    return fallback


def _reasoning_summary_text(item: Mapping[str, object]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, Sequence) or isinstance(summary, str | bytes | bytearray):
        return ""
    parts: list[str] = []
    for raw in cast(Sequence[object], summary):
        mapping = _mapping(raw)
        if mapping is not None:
            text = _string(mapping.get("text"))
            if text:
                parts.append(text)
    return "".join(parts)


def _hosted_tool_input(item: Mapping[str, object]) -> object:
    item_type = _string(item.get("type")) or "unknown"
    if item_type in (
        "web_search_call",
        "web_search_preview_call",
        "computer_use_call",
        "local_shell_call",
    ):
        return item.get("action") or {}
    if item_type == "file_search_call":
        return {"queries": item.get("queries") or []}
    if item_type == "code_interpreter_call":
        return {"code": item.get("code"), "container_id": item.get("container_id")}
    if item_type == "mcp_call":
        return {
            "server_label": item.get("server_label"),
            "name": item.get("name"),
            "arguments": item.get("arguments"),
        }
    return {}


def _media_url(block: MediaContentBlock) -> str:
    data = thaw_json(block.data)
    mapping = _mapping(data)
    if mapping is None:
        raw = str(data)
        if raw.startswith(("data:", "http://", "https://")):
            return raw
        return f"data:{block.media_type};base64,{raw}"
    return _media_url_from_mapping(mapping, media_type=block.media_type)


def _media_url_from_mapping(mapping: Mapping[str, object], *, media_type: str) -> str:
    for key in ("image_url", "url"):
        value = _string(mapping.get(key))
        if value:
            return value
    source = _mapping(mapping.get("source"))
    if source is not None:
        for key in ("url", "image_url"):
            value = _string(source.get(key))
            if value:
                return value
        data = _string(source.get("data"))
        source_media_type = _string(source.get("media_type")) or media_type
        if data:
            return data if data.startswith("data:") else f"data:{source_media_type};base64,{data}"
    data = _string(mapping.get("data")) or ""
    if data.startswith(("data:", "http://", "https://")):
        return data
    return f"data:{media_type};base64,{data}"


def _is_tool_result_content_array(raw: list[object]) -> bool:
    if not raw:
        return True
    for item in raw:
        mapping = _mapping(item)
        if mapping is None:
            return False
        if mapping.get("type") not in ("text", "image", "media"):
            return False
    return True


def _join_text_blocks(blocks: Sequence[ModelContentBlock]) -> str:
    return "\n".join(block.text for block in blocks if isinstance(block, TextContentBlock))


def _stringify_content(block: ModelContentBlock) -> str:
    if isinstance(block, ToolResultContentBlock):
        return _stable_json_text(thaw_json(block.content))
    if isinstance(block, ThinkingContentBlock):
        return block.text
    if isinstance(block, ToolUseContentBlock):
        return _stable_json_text(
            {"id": block.id, "name": block.name, "input": thaw_json(block.input)}
        )
    if isinstance(block, MediaContentBlock):
        return _stable_json_text(thaw_json(block.data))
    if isinstance(block, UnknownContentBlock):
        return _stable_json_text(thaw_json(block.payload))
    return ""


def _stable_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_json(value: object) -> object:
    if not isinstance(value, str):
        return value if value is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _openai_store(options: object) -> bool | None:
    openai = _openai_options_mapping(options)
    value = openai.get("store")
    return value if isinstance(value, bool) else None


def _openai_metadata(metadata: FrozenJson | None) -> Mapping[str, object]:
    raw = thaw_json(metadata) if metadata is not None else None
    mapping = _mapping(raw)
    if mapping is None:
        return {}
    openai = _mapping(mapping.get("openai"))
    return openai if openai is not None else mapping


def _openai_options_mapping(options: object) -> Mapping[str, object]:
    raw: object = thaw_json(cast(FrozenJson, options)) if options is not None else {}
    mapping = _mapping(raw)
    if mapping is None:
        return {}
    openai = _mapping(mapping.get("openai"))
    return openai if openai is not None else mapping


def _error_payload(payload: object) -> Mapping[str, object]:
    mapping = _mapping(payload)
    if mapping is None:
        return {"message": str(payload)}
    error = _mapping(mapping.get("error"))
    if error is not None:
        merged = dict(error)
        for key in ("status", "status_code", "headers", "response_headers"):
            if key in mapping and key not in merged:
                merged[key] = mapping[key]
        return merged
    return mapping


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in cast(Mapping[object, object], value).items()}
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


def _raw_details(payload: object, fallback: str) -> str:
    payload_mapping = _mapping(payload)
    if payload_mapping is not None:
        return str(dict(payload_mapping))
    return fallback


def _token_gap(raw: str) -> tuple[int | None, int | None]:
    cleaned = raw.replace(",", "")
    actual = _first_token_match(cleaned, _REQUESTED_TOKEN_RES)
    limit = _first_token_match(cleaned, _LIMIT_TOKEN_RES)
    if actual is not None and limit is not None:
        return actual, limit
    match = _CONTEXT_TOKEN_RE.search(cleaned)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _first_token_match(raw: str, patterns: Sequence[re.Pattern[str]]) -> int | None:
    for pattern in patterns:
        match = pattern.search(raw)
        if match is not None:
            return int(match.group(1))
    return None


def _is_context_overflow(text: str) -> bool:
    return (
        "context_length_exceeded" in text
        or "maximum context" in text
        or "context length" in text
        or "too many tokens" in text
        or "prompt is too long" in text
    )


def _is_media_overflow(text: str) -> bool:
    if _PDF_PAGE_LIMIT_RE.search(text):
        return True
    media_related = any(marker in text for marker in ("image", "media", "pdf", "document"))
    request_size_related = "request too large" in text or "payload too large" in text
    if not media_related and not request_size_related:
        return False
    return (
        request_size_related
        or "too large" in text
        or "too many images" in text
        or "many-image" in text
        or ("dimension" in text and "exceed" in text)
        or ("exceed" in text and ("maximum" in text or "limit" in text))
        or ("maximum" in text and ("size" in text or "limit" in text))
    )


def _response_events(response: ProviderResponse) -> Iterable[ProviderEvent]:
    response_id = _string(response.get("id")) or "openai_response"
    yield {"type": "response.created", "response": {"id": response_id}}

    output = response.get("output")
    if isinstance(output, Sequence) and not isinstance(output, str | bytes | bytearray):
        for index, raw_item in enumerate(cast(Sequence[object], output)):
            item = _mapping(raw_item)
            if item is None:
                continue
            yield from _output_item_events(item, response_id=response_id, index=index)

    yield {"type": "response.completed", "response": dict(response)}


def _output_item_events(
    item: Mapping[str, object],
    *,
    response_id: str,
    index: int,
) -> Iterable[ProviderEvent]:
    item_type = _string(item.get("type")) or "unknown"
    if item_type == "message":
        yield from _message_output_events(item, response_id=response_id, index=index)
        return
    if item_type == "output_text":
        item_id = _string(item.get("id")) or f"msg_{index}"
        text = _string(item.get("text")) or ""
        if text:
            yield {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "delta": text,
            }
        yield {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": item_id,
        }
        return
    yield {
        "type": "response.output_item.done",
        "response_id": response_id,
        "item": dict(item),
    }


def _message_output_events(
    item: Mapping[str, object],
    *,
    response_id: str,
    index: int,
) -> Iterable[ProviderEvent]:
    content = item.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
        return
    for content_index, raw_content in enumerate(cast(Sequence[object], content)):
        content_item = _mapping(raw_content)
        if content_item is None:
            continue
        content_type = _string(content_item.get("type")) or "unknown"
        if content_type not in ("output_text", "text"):
            continue
        item_id = (
            _string(content_item.get("id"))
            or _string(item.get("id"))
            or f"msg_{index}_{content_index}"
        )
        text = _string(content_item.get("text")) or ""
        if text:
            yield {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "delta": text,
            }
        yield {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": item_id,
        }


__all__ = ["OpenAIResponsesAdapter", "OpenAIResponsesStreamParser"]
