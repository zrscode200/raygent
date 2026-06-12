"""Raygent-owned model-visible message shapes.

The current transcript wire shape intentionally remains a small dict-based
format (`{"role": ..., "content": ...}`) because compaction, tool execution,
memory extraction, and tests already operate on that structure. Owning the type
here removes the dependency on any vendor SDK while preserving the existing
runtime behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, cast

from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelApiErrorMessage,
    ModelContentBlock,
    ModelMessage,
    ModelResponse,
    ModelToolUseBlock,
    ObservableMessage,
    TextContentBlock,
    ThinkingContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
    UnknownContentBlock,
)

MessageRole = Literal["system", "user", "assistant", "tool"]
MessageContent = str | list[dict[str, Any]]
ContentBlockParam = dict[str, Any]
RaygentMessageKind = Literal[
    "memory_recall",
    "continuation_context",
    "agent_trigger",
    "coordinator_runtime",
]


class RaygentMemoryRecallItem(TypedDict):
    path: str
    content_bytes: int


class RaygentMemoryRecallMetadata(TypedDict):
    type: Literal["relevant_memories"]
    memories: list[RaygentMemoryRecallItem]


class RaygentContinuationContextFragmentMetadata(TypedDict):
    id: str
    input_chars: int
    rendered_chars: int
    truncated: bool
    source: str | None
    reason: str | None


class RaygentContinuationContextMetadata(TypedDict):
    type: Literal["continuation_context"]
    fragment_count: int
    input_char_count: int
    rendered_char_count: int
    truncated_fragment_count: int
    dropped_empty_fragment_count: int
    dropped_fragment_count: int
    rendered_message_truncated: bool
    total_budget_chars: int
    per_fragment_budget_chars: int
    max_fragment_count: int
    fragments: list[RaygentContinuationContextFragmentMetadata]


class RaygentAgentTriggerMatchMetadata(TypedDict):
    id: str
    agent_name: str
    reason: str | None
    prompt_hint: str | None
    confidence: float | None
    source: str | None
    truncated: bool


class RaygentAgentTriggerMetadata(TypedDict):
    type: Literal["agent_trigger"]
    match_count: int
    delegation_tool_available: bool
    rendered_char_count: int
    input_char_count: int
    truncated_message_count: int
    dropped_message_count: int
    max_messages: int
    max_message_chars: int
    suppress_main_turn_requested: bool
    suppress_main_turn_applied: bool
    matches: list[RaygentAgentTriggerMatchMetadata]


class RaygentCoordinatorRuntimeMetadata(TypedDict):
    type: Literal["coordinator_runtime"]
    work_item_count: int
    blackboard_entry_count: int
    rendered_work_item_count: int
    rendered_blackboard_entry_count: int
    dropped_work_item_count: int
    dropped_blackboard_entry_count: int
    truncated: bool
    rendered_char_count: int
    max_total_chars: int
    max_work_items: int
    max_blackboard_entries: int
    max_entry_chars: int
    work_item_ids: list[str]
    blackboard_entry_ids: list[str]


class _RequiredMessageParam(TypedDict):
    role: MessageRole
    content: MessageContent


class MessageParam(_RequiredMessageParam, total=False):
    id: str
    uuid: str
    isApiErrorMessage: bool
    apiError: str
    error: str
    errorDetails: str
    raygentMessageKind: RaygentMessageKind
    raygentAgentTrigger: RaygentAgentTriggerMetadata
    raygentCoordinatorRuntime: RaygentCoordinatorRuntimeMetadata
    raygentContinuationContext: RaygentContinuationContextMetadata
    raygentMemoryRecall: RaygentMemoryRecallMetadata


def user_message(content: object) -> MessageParam:
    return {"role": "user", "content": _coerce_message_content(content)}


def assistant_message(content: object) -> MessageParam:
    return {"role": "assistant", "content": _coerce_message_content(content)}


def api_message_from_message_param(message: MessageParam) -> ApiMessage:
    """Build an immutable API-bound wrapper from Raygent's transcript shape.

    `provider_payload` stores the exact Raygent-owned message payload so external
    providers can translate it without relying on lossy normalized block
    reconstruction.
    """
    return ApiMessage(
        message=model_message_from_message_param(message),
        provider_payload=cast(FrozenJson, message),
    )


def observable_message_from_message_param(message: MessageParam) -> ObservableMessage:
    return ObservableMessage(
        message=model_message_from_message_param(message),
        provider_payload=cast(FrozenJson, message),
    )


def model_response_from_message_param(
    message: MessageParam,
    *,
    stop_reason: str | None = None,
    provider_request_id: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        api_message=api_message_from_message_param(message),
        observable_message=observable_message_from_message_param(message),
        tool_uses=tuple(_tool_uses_from_message_param(message)),
        stop_reason=stop_reason,
        provider_request_id=provider_request_id,
    )


def message_param_from_api_message(message: ApiMessage | ObservableMessage) -> MessageParam:
    """Recover Raygent's transcript shape from a normalized message wrapper."""
    if message.provider_payload is not None:
        payload = thaw_json(message.provider_payload)
        if isinstance(payload, Mapping):
            mapping = cast(Mapping[object, object], payload)
            raw = cast(MessageParam, {str(key): value for key, value in mapping.items()})
            if "role" in raw and "content" in raw:
                return raw
    return message_param_from_model_message(message.message)


def message_param_from_model_api_error(
    message: ModelApiErrorMessage,
    *,
    observable: bool = False,
) -> MessageParam:
    """Convert a provider-normalized API error into Raygent transcript shape."""
    source = message.observable_message if observable else message.api_message
    raw = message_param_from_api_message(source)
    raw["role"] = "assistant"
    raw["isApiErrorMessage"] = True
    raw["apiError"] = message.kind
    raw["error"] = message.kind
    if message.raw_details is not None:
        raw["errorDetails"] = message.raw_details
    return raw


def model_message_from_message_param(message: MessageParam) -> ModelMessage:
    role = _role_from_message(message)
    content = message.get("content", "")
    blocks = _model_blocks_from_content(content)
    message_id = message.get("id") or message.get("uuid")
    return ModelMessage(
        role=role,
        content=tuple(blocks),
        id=str(message_id) if isinstance(message_id, str) else None,
    )


def message_param_from_model_message(message: ModelMessage) -> MessageParam:
    blocks = [_block_param_from_model_block(block) for block in message.content]
    content: str | list[dict[str, Any]]
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        content = str(blocks[0].get("text", ""))
    else:
        content = blocks
    raw: MessageParam = {"role": message.role, "content": content}
    if message.id is not None:
        raw["id"] = message.id
    return raw


def thaw_json(value: FrozenJson) -> object:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _role_from_message(message: Mapping[str, Any]) -> MessageRole:
    role = message.get("role")
    if role in ("system", "user", "assistant", "tool"):
        return role
    return "user"


def _coerce_message_content(content: object) -> MessageContent:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return cast(list[dict[str, Any]], content)
    return str(content)


def _model_blocks_from_content(content: object) -> list[ModelContentBlock]:
    if isinstance(content, str):
        return [TextContentBlock(text=content)]
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        blocks: list[ModelContentBlock] = []
        for item in cast(Sequence[object], content):
            if isinstance(item, Mapping):
                blocks.append(_model_block_from_mapping(cast(Mapping[str, Any], item)))
            else:
                blocks.append(TextContentBlock(text=str(item)))
        return blocks
    return [TextContentBlock(text=str(content))]


def _tool_uses_from_message_param(message: MessageParam) -> list[ModelToolUseBlock]:
    content = message.get("content", "")
    if isinstance(content, str):
        return []
    tool_uses: list[ModelToolUseBlock] = []
    for index, block in enumerate(content):
        if block.get("type") != "tool_use":
            continue
        tool_uses.append(
            ModelToolUseBlock(
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
                input=block.get("input", {}),
                index=index,
            )
        )
    return tool_uses


def _model_block_from_mapping(block: Mapping[str, Any]) -> ModelContentBlock:
    block_type = block.get("type")
    if block_type == "text":
        return TextContentBlock(text=str(block.get("text", "")))
    if block_type in ("tool_use", "server_tool_use"):
        provider_metadata = block.get("provider_metadata")
        return ToolUseContentBlock(
            id=str(block.get("id", "")),
            name=str(block.get("name", "")),
            input=block.get("input", {}),
            provider_executed=block_type == "server_tool_use"
            or bool(block.get("provider_executed", False)),
            provider_metadata=(
                cast(FrozenJson, provider_metadata)
                if provider_metadata is not None
                else None
            ),
        )
    if block_type == "tool_result":
        provider_metadata = block.get("provider_metadata")
        return ToolResultContentBlock(
            tool_use_id=str(block.get("tool_use_id", "")),
            content=cast(FrozenJson, block.get("content", "")),
            is_error=bool(block.get("is_error", False)),
            provider_metadata=(
                cast(FrozenJson, provider_metadata)
                if provider_metadata is not None
                else None
            ),
        )
    if block_type in ("thinking", "reasoning", "redacted_thinking", "encrypted_thinking"):
        signature = block.get("signature")
        provider_metadata = block.get("provider_metadata")
        is_redacted = bool(block.get("redacted", False)) or block_type in (
            "redacted_thinking",
            "encrypted_thinking",
        )
        metadata: FrozenJson | None
        if provider_metadata is not None:
            metadata = cast(FrozenJson, provider_metadata)
        elif block_type != "thinking":
            metadata = cast(FrozenJson, dict(block))
        else:
            metadata = None
        return ThinkingContentBlock(
            text=str(block.get("text", "")),
            signature=signature if isinstance(signature, str) else None,
            redacted=is_redacted,
            provider_metadata=metadata,
        )
    if block_type in ("image", "document"):
        provider_metadata = block.get("provider_metadata")
        return MediaContentBlock(
            media_kind=block_type,
            media_type=str(block.get("media_type", block.get("type", "unknown"))),
            data=dict(block),
            provider_metadata=(
                cast(FrozenJson, provider_metadata)
                if provider_metadata is not None
                else None
            ),
        )
    return UnknownContentBlock(
        block_type=str(block_type) if block_type is not None else "unknown",
        payload=dict(block),
    )


def _block_param_from_model_block(block: ModelContentBlock) -> dict[str, Any]:
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseContentBlock):
        raw: dict[str, Any] = {
            "type": "server_tool_use" if block.provider_executed else "tool_use",
            "id": block.id,
            "name": block.name,
            "input": thaw_json(block.input),
        }
        if block.provider_executed:
            raw["provider_executed"] = True
        if block.provider_metadata is not None:
            raw["provider_metadata"] = thaw_json(block.provider_metadata)
        return raw
    if isinstance(block, ToolResultContentBlock):
        raw: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": thaw_json(block.content),
        }
        if block.is_error:
            raw["is_error"] = True
        if block.provider_metadata is not None:
            raw["provider_metadata"] = thaw_json(block.provider_metadata)
        return raw
    if isinstance(block, ThinkingContentBlock):
        raw = {"type": "thinking", "text": block.text}
        if block.signature is not None:
            raw["signature"] = block.signature
        if block.redacted:
            raw["redacted"] = True
        if block.provider_metadata is not None:
            raw["provider_metadata"] = thaw_json(block.provider_metadata)
        return raw
    if isinstance(block, MediaContentBlock):
        data = thaw_json(block.data)
        raw = (
            {str(key): value for key, value in cast(Mapping[object, object], data).items()}
            if isinstance(data, Mapping)
            else {"data": data}
        )
        raw.setdefault("type", block.media_kind)
        raw.setdefault("media_type", block.media_type)
        if block.provider_metadata is not None:
            raw["provider_metadata"] = thaw_json(block.provider_metadata)
        return raw
    payload = thaw_json(block.payload)
    raw = (
        {str(key): value for key, value in cast(Mapping[object, object], payload).items()}
        if isinstance(payload, Mapping)
        else {"payload": payload}
    )
    raw.setdefault("type", block.block_type)
    return raw


__all__ = [
    "ContentBlockParam",
    "MessageContent",
    "MessageParam",
    "MessageRole",
    "RaygentContinuationContextFragmentMetadata",
    "RaygentContinuationContextMetadata",
    "RaygentCoordinatorRuntimeMetadata",
    "RaygentMemoryRecallItem",
    "RaygentMemoryRecallMetadata",
    "RaygentMessageKind",
    "api_message_from_message_param",
    "assistant_message",
    "message_param_from_api_message",
    "message_param_from_model_api_error",
    "message_param_from_model_message",
    "model_message_from_message_param",
    "model_response_from_message_param",
    "observable_message_from_message_param",
    "thaw_json",
    "user_message",
]
