"""Provider-neutral media counting and downscoping.

Reference grounding:
  nested media inside tool_result content, then strips oldest media first.
  retry/summary path must remove media without making the removal silent.

Raygent keeps the same kernel behavior without provider SDK types: scan
Raygent `ApiMessage` blocks, preserve the newest media items under a configured
limit, and use explicit text placeholders whenever model-visible media is
removed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from raygent_harness.core.messages import (
    MessageParam,
    api_message_from_message_param,
    message_param_from_api_message,
    message_param_from_model_message,
    thaw_json,
)
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaBudgetSnapshot,
    MediaContentBlock,
    ModelContentBlock,
    ModelInfo,
    ModelMessage,
    TextContentBlock,
    ToolResultContentBlock,
)

API_MAX_MEDIA_PER_REQUEST = 100
EXCESS_MEDIA_REMOVED_PLACEHOLDER = (
    "[Media content removed - request exceeded media item budget]"
)
MEDIA_OVERFLOW_RETRY_REMOVED_PLACEHOLDER = (
    "[Media content removed after provider media-size rejection; retrying without media]"
)

MediaBudgetMode = Literal["request_limit", "media_overflow_retry"]


@dataclass(frozen=True, slots=True)
class MediaBudgetApplication:
    messages: tuple[ApiMessage, ...]
    snapshot: MediaBudgetSnapshot | None
    changed: bool = False


@dataclass(frozen=True, slots=True)
class MessageMediaDownscope:
    messages: tuple[MessageParam, ...]
    snapshot: MediaBudgetSnapshot | None
    changed: bool = False


def media_item_limit_for_model(model_info: ModelInfo) -> int:
    """Return the request media-item cap for a model/provider."""

    limit = model_info.max_media_items_per_request
    if limit is None:
        return API_MAX_MEDIA_PER_REQUEST
    return max(0, limit)


def apply_media_budget_to_api_messages(
    messages: tuple[ApiMessage, ...],
    *,
    max_media_items: int,
    mode: MediaBudgetMode = "request_limit",
    placeholder: str = EXCESS_MEDIA_REMOVED_PLACEHOLDER,
) -> MediaBudgetApplication:
    """Return API messages with oldest excess media replaced by text markers."""

    max_media_items = max(0, max_media_items)
    counts = _count_media(messages)
    original_count = counts.total
    to_remove = max(0, original_count - max_media_items)
    if original_count == 0:
        return MediaBudgetApplication(messages=messages, snapshot=None)
    if to_remove == 0:
        return MediaBudgetApplication(
            messages=messages,
            snapshot=MediaBudgetSnapshot(
                max_media_items=max_media_items,
                original_media_items=original_count,
                retained_media_items=original_count,
                stripped_media_items=0,
                top_level_media_items=counts.top_level,
                nested_media_items=counts.nested,
                mode=mode,
            ),
        )

    remaining = to_remove
    normalized: list[ApiMessage] = []
    changed = False
    for message in messages:
        if remaining <= 0:
            normalized.append(message)
            continue
        normalized_message, removed = _strip_oldest_media_from_message(
            message,
            to_remove=remaining,
            placeholder=placeholder,
        )
        remaining -= removed
        changed = changed or removed > 0
        normalized.append(normalized_message)

    stripped = to_remove - remaining
    return MediaBudgetApplication(
        messages=tuple(normalized),
        changed=changed,
        snapshot=MediaBudgetSnapshot(
            max_media_items=max_media_items,
            original_media_items=original_count,
            retained_media_items=original_count - stripped,
            stripped_media_items=stripped,
            top_level_media_items=counts.top_level,
            nested_media_items=counts.nested,
            mode=mode,
        ),
    )


def downscope_message_params_for_media_retry(
    messages: Sequence[MessageParam],
) -> MessageMediaDownscope:
    """Replace all model-visible media in transcript-shaped messages.

    Used after a provider media-overflow rejection. The provider did not accept
    the last request, so we retry with explicit placeholders rather than routing
    the failure through token compaction.
    """

    api_messages = tuple(api_message_from_message_param(message) for message in messages)
    application = apply_media_budget_to_api_messages(
        api_messages,
        max_media_items=0,
        mode="media_overflow_retry",
        placeholder=MEDIA_OVERFLOW_RETRY_REMOVED_PLACEHOLDER,
    )
    return MessageMediaDownscope(
        messages=tuple(
            message_param_from_api_message(message)
            for message in application.messages
        ),
        snapshot=application.snapshot,
        changed=application.changed,
    )


@dataclass(frozen=True, slots=True)
class _MediaCounts:
    total: int = 0
    top_level: int = 0
    nested: int = 0


def _count_media(messages: tuple[ApiMessage, ...]) -> _MediaCounts:
    top_level = 0
    nested = 0
    for message in messages:
        for block in message.message.content:
            if isinstance(block, MediaContentBlock):
                top_level += 1
            elif isinstance(block, ToolResultContentBlock):
                nested += _count_nested_media(block)
    return _MediaCounts(
        total=top_level + nested,
        top_level=top_level,
        nested=nested,
    )


def _count_nested_media(block: ToolResultContentBlock) -> int:
    raw = thaw_json(block.content)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        return 0
    return sum(
        1
        for item in cast(Sequence[object], raw)
        if _media_kind_from_mapping(item) is not None
    )


def _strip_oldest_media_from_message(
    message: ApiMessage,
    *,
    to_remove: int,
    placeholder: str,
) -> tuple[ApiMessage, int]:
    # Match reference `stripExcessMediaItems`: inside one message, nested
    # `tool_result.content` media is processed before top-level media blocks.
    nested_removed = 0
    first_pass: list[ModelContentBlock] = []
    changed = False

    for block in message.message.content:
        if nested_removed < to_remove and isinstance(block, ToolResultContentBlock):
            rewritten, nested_stripped = _strip_oldest_nested_media(
                block,
                to_remove=to_remove - nested_removed,
                placeholder=placeholder,
            )
            nested_removed += nested_stripped
            changed = changed or nested_stripped > 0
            first_pass.append(rewritten)
            continue
        first_pass.append(block)

    top_level_removed = 0
    remaining = to_remove - nested_removed
    content: list[ModelContentBlock] = []
    for block in first_pass:
        if top_level_removed < remaining and isinstance(block, MediaContentBlock):
            top_level_removed += 1
            changed = True
            content.append(
                TextContentBlock(text=_placeholder_for_media_block(block, placeholder))
            )
            continue
        content.append(block)

    if not changed:
        return message, 0
    return _api_message_with_content(
        message,
        tuple(content),
    ), nested_removed + top_level_removed


def _strip_oldest_nested_media(
    block: ToolResultContentBlock,
    *,
    to_remove: int,
    placeholder: str,
) -> tuple[ToolResultContentBlock, int]:
    raw = thaw_json(block.content)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        return block, 0

    removed = 0
    normalized: list[object] = []
    for item in cast(Sequence[object], raw):
        if removed < to_remove:
            media_kind = _media_kind_from_mapping(item)
            if media_kind is not None:
                removed += 1
                normalized.append(
                    {
                        "type": "text",
                        "text": _placeholder_for_media_kind(media_kind, placeholder),
                    }
                )
                continue
        normalized.append(item)

    if removed == 0:
        return block, 0
    return replace(block, content=cast(FrozenJson, normalized)), removed


def _media_kind_from_mapping(item: object) -> str | None:
    if not isinstance(item, Mapping):
        return None
    mapping = cast(Mapping[object, object], item)
    item_type = mapping.get("type")
    if item_type in ("image", "document"):
        return str(item_type)
    media_type = mapping.get("media_type") or mapping.get("mediaType")
    if item_type == "media" and isinstance(media_type, str):
        if media_type.startswith("image/"):
            return "image"
        if media_type == "application/pdf" or media_type.startswith("application/"):
            return "document"
        return "media"
    return None


def _placeholder_for_media_block(
    block: MediaContentBlock,
    placeholder: str,
) -> str:
    return _placeholder_for_media_kind(block.media_kind, placeholder)


def _placeholder_for_media_kind(media_kind: str, placeholder: str) -> str:
    if media_kind in ("image", "document"):
        return f"{placeholder} ({media_kind})"
    return placeholder


def _api_message_with_content(
    source: ApiMessage,
    content: tuple[ModelContentBlock, ...],
) -> ApiMessage:
    message = ModelMessage(
        role=source.message.role,
        content=content,
        id=source.message.id,
    )
    return ApiMessage(
        message=message,
        provider_payload=cast(
            FrozenJson,
            _provider_payload_with_replaced_content(source, message),
        ),
    )


def _provider_payload_with_replaced_content(
    source: ApiMessage,
    message: ModelMessage,
) -> MessageParam:
    replacement = message_param_from_model_message(message)
    raw = thaw_json(source.provider_payload) if source.provider_payload is not None else None
    if not isinstance(raw, Mapping):
        return replacement
    merged = {str(key): value for key, value in cast(Mapping[object, object], raw).items()}
    merged["role"] = replacement["role"]
    merged["content"] = replacement["content"]
    if "id" in replacement:
        merged["id"] = replacement["id"]
    return cast(MessageParam, merged)


__all__ = [
    "API_MAX_MEDIA_PER_REQUEST",
    "EXCESS_MEDIA_REMOVED_PLACEHOLDER",
    "MEDIA_OVERFLOW_RETRY_REMOVED_PLACEHOLDER",
    "MediaBudgetApplication",
    "MessageMediaDownscope",
    "apply_media_budget_to_api_messages",
    "downscope_message_params_for_media_retry",
    "media_item_limit_for_model",
]
