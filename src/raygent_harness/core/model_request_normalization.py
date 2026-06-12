"""API-bound model request normalization.

The stored transcript is Raygent-owned and may carry provider-native metadata
that is useful for replay, compaction, or deferred-tool discovery. Providers
still need a valid request payload for the selected model. This module performs
the reference-shaped normalization at the model-call boundary: strip unsupported
tool-search fields and repair tool_use/tool_result pairing without mutating
conversation state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from raygent_harness.core.media_budget import (
    apply_media_budget_to_api_messages,
    media_item_limit_for_model,
)
from raygent_harness.core.messages import (
    message_param_from_model_message,
    thaw_json,
)
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelContentBlock,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
    freeze_json,
)

SYNTHETIC_TOOL_RESULT_PLACEHOLDER = "[Tool result missing due to internal error]"
TOOL_REFERENCES_REMOVED_PLACEHOLDER = (
    "[Tool references removed - tool search not enabled]"
)
UNAVAILABLE_TOOL_REFERENCES_REMOVED_PLACEHOLDER = (
    "[Tool references removed - tools no longer available]"
)
ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER = (
    "[Orphaned tool result removed due to conversation resume]"
)
TOOL_USE_INTERRUPTED_PLACEHOLDER = "[Tool use interrupted]"
UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER = (
    "[Image content removed - target model does not support image input]"
)
UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER = (
    "[Document content removed - target model does not support document input]"
)
UNSUPPORTED_MEDIA_REMOVED_PLACEHOLDER = (
    "[Media content removed - target model does not support this media type]"
)


def normalize_model_request_for_provider(
    request: ModelRequest,
    *,
    model_info: ModelInfo,
) -> ModelRequest:
    """Return the provider/API-bound request for the selected model.

    The original request object remains untouched. When the target model does
    not support provider-native ToolSearch references, stale `tool_reference`
    content blocks and assistant `caller` metadata are stripped before pairing
    repair runs.
    """

    capabilities = model_info.capabilities
    supports_tool_references = capabilities.supports_tool_references
    supports_images = capabilities.supports_images or capabilities.supports_media
    supports_documents = capabilities.supports_documents
    available_tool_names = {tool.name for tool in request.tools}
    messages = tuple(
        _normalize_api_bound_fields(
            message,
            supports_tool_references=supports_tool_references,
            available_tool_names=available_tool_names,
            supports_images=supports_images,
            supports_documents=supports_documents,
        )
        for message in request.messages
    )
    messages = _ensure_tool_result_pairing(messages)
    media_budget = apply_media_budget_to_api_messages(
        messages,
        max_media_items=media_item_limit_for_model(model_info),
    )
    messages = media_budget.messages
    if messages == request.messages and media_budget.snapshot == request.media_budget:
        return request
    return replace(request, messages=messages, media_budget=media_budget.snapshot)


def _normalize_api_bound_fields(
    message: ApiMessage,
    *,
    supports_tool_references: bool,
    available_tool_names: set[str],
    supports_images: bool,
    supports_documents: bool,
) -> ApiMessage:
    normalized = message
    if message.message.role == "assistant" and not supports_tool_references:
        normalized = _strip_caller_from_assistant_message(normalized)
    if message.message.role in ("user", "tool"):
        normalized = _strip_tool_references_from_result_message(
            normalized,
            available_tool_names=(
                available_tool_names if supports_tool_references else None
            ),
        )
    return _strip_unsupported_media_from_message(
        normalized,
        supports_images=supports_images,
        supports_documents=supports_documents,
    )


def _strip_caller_from_assistant_message(message: ApiMessage) -> ApiMessage:
    changed = False
    content: list[ModelContentBlock] = []
    for block in message.message.content:
        if isinstance(block, ToolUseContentBlock):
            stripped_metadata = _metadata_without_caller(block.provider_metadata)
            if stripped_metadata != block.provider_metadata:
                changed = True
                content.append(replace(block, provider_metadata=stripped_metadata))
                continue
        content.append(block)
    if not changed:
        return message
    return _api_message_with_content(message, tuple(content))


def _strip_tool_references_from_result_message(
    message: ApiMessage,
    *,
    available_tool_names: set[str] | None,
) -> ApiMessage:
    changed = False
    content: list[ModelContentBlock] = []
    for block in message.message.content:
        if isinstance(block, ToolResultContentBlock):
            stripped = _strip_tool_reference_content(
                block,
                available_tool_names=available_tool_names,
            )
            if stripped != block:
                changed = True
                content.append(stripped)
                continue
        content.append(block)
    if not changed:
        return message
    return _api_message_with_content(message, tuple(content))


def _strip_tool_reference_content(
    block: ToolResultContentBlock,
    *,
    available_tool_names: set[str] | None,
) -> ToolResultContentBlock:
    raw = thaw_json(block.content)
    if not isinstance(raw, list):
        return block
    filtered: list[object] = []
    removed = False
    for item in cast(list[object], raw):
        if _should_strip_tool_reference(
            item,
            available_tool_names=available_tool_names,
        ):
            removed = True
            continue
        filtered.append(item)
    if not removed:
        return block
    if not filtered:
        placeholder = (
            TOOL_REFERENCES_REMOVED_PLACEHOLDER
            if available_tool_names is None
            else UNAVAILABLE_TOOL_REFERENCES_REMOVED_PLACEHOLDER
        )
        filtered = [{"type": "text", "text": placeholder}]
    return replace(block, content=cast(FrozenJson, filtered))


def _strip_unsupported_media_from_message(
    message: ApiMessage,
    *,
    supports_images: bool,
    supports_documents: bool,
) -> ApiMessage:
    changed = False
    content: list[ModelContentBlock] = []
    for block in message.message.content:
        if isinstance(block, MediaContentBlock):
            placeholder = _media_placeholder(
                media_kind=block.media_kind,
                media_type=block.media_type,
                supports_images=supports_images,
                supports_documents=supports_documents,
            )
            if placeholder is not None:
                changed = True
                content.append(TextContentBlock(text=placeholder))
                continue
        if isinstance(block, ToolResultContentBlock):
            stripped = _strip_unsupported_media_from_tool_result(
                block,
                supports_images=supports_images,
                supports_documents=supports_documents,
            )
            if stripped != block:
                changed = True
                content.append(stripped)
                continue
        content.append(block)
    if not changed:
        return message
    return _api_message_with_content(message, tuple(content))


def _strip_unsupported_media_from_tool_result(
    block: ToolResultContentBlock,
    *,
    supports_images: bool,
    supports_documents: bool,
) -> ToolResultContentBlock:
    raw = thaw_json(block.content)
    if not isinstance(raw, list):
        return block
    normalized: list[object] = []
    changed = False
    for item in cast(list[object], raw):
        placeholder = _media_item_placeholder(
            item,
            supports_images=supports_images,
            supports_documents=supports_documents,
        )
        if placeholder is None:
            normalized.append(item)
            continue
        changed = True
        normalized.append({"type": "text", "text": placeholder})
    if not changed:
        return block
    return replace(block, content=cast(FrozenJson, normalized))


def _media_item_placeholder(
    item: object,
    *,
    supports_images: bool,
    supports_documents: bool,
) -> str | None:
    if not isinstance(item, Mapping):
        return None
    mapping = cast(Mapping[object, object], item)
    item_type = mapping.get("type")
    media_type = mapping.get("media_type") or mapping.get("mediaType")
    if item_type == "image":
        media_kind = "image"
    elif item_type == "document":
        media_kind = "document"
    elif item_type == "media" and isinstance(media_type, str):
        media_kind = _media_kind_from_type(media_type)
    else:
        return None
    return _media_placeholder(
        media_kind=media_kind,
        media_type=str(media_type) if isinstance(media_type, str) else "",
        supports_images=supports_images,
        supports_documents=supports_documents,
    )


def _media_placeholder(
    *,
    media_kind: str,
    media_type: str,
    supports_images: bool,
    supports_documents: bool,
) -> str | None:
    kind = media_kind if media_kind != "unknown_media" else _media_kind_from_type(media_type)
    if kind == "image":
        return None if supports_images else UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER
    if kind == "document":
        return (
            None
            if supports_documents
            else UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER
        )
    return (
        None
        if supports_images or supports_documents
        else UNSUPPORTED_MEDIA_REMOVED_PLACEHOLDER
    )


def _media_kind_from_type(media_type: str) -> str:
    if media_type.startswith("image/"):
        return "image"
    if media_type == "application/pdf" or media_type.startswith("application/"):
        return "document"
    return "unknown_media"


def _metadata_without_caller(metadata: FrozenJson | None) -> FrozenJson | None:
    if metadata is None:
        return None
    raw = thaw_json(metadata)
    if not isinstance(raw, Mapping):
        return metadata
    raw_mapping = cast(Mapping[object, object], raw)
    mapping: dict[str, FrozenJson] = {
        str(key): cast(FrozenJson, value) for key, value in raw_mapping.items()
    }
    if "caller" not in mapping:
        return metadata
    mapping.pop("caller", None)
    if not mapping:
        return None
    return freeze_json(mapping)


def _ensure_tool_result_pairing(messages: tuple[ApiMessage, ...]) -> tuple[ApiMessage, ...]:
    result: list[ApiMessage] = []
    seen_tool_use_ids: set[str] = set()
    index = 0

    while index < len(messages):
        message = messages[index]
        if message.message.role != "assistant":
            repaired = _strip_orphaned_tool_results(
                message,
                previous_is_assistant=(
                    bool(result) and result[-1].message.role == "assistant"
                ),
                is_first_message=not result,
            )
            if repaired is not None:
                result.append(repaired)
            index += 1
            continue

        assistant_message, current_tool_use_ids = _dedupe_assistant_tool_uses(
            message,
            seen_tool_use_ids,
        )
        result.append(assistant_message)

        next_message = messages[index + 1] if index + 1 < len(messages) else None
        if next_message is None or next_message.message.role not in ("user", "tool"):
            if current_tool_use_ids:
                result.append(_synthetic_tool_result_message(current_tool_use_ids))
            index += 1
            continue

        patched_next = _repair_following_tool_result_message(
            next_message,
            tool_use_ids=current_tool_use_ids,
        )
        if patched_next is not None:
            result.append(patched_next)
        else:
            result.append(
                _plain_user_message(
                    text=ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER,
                )
            )
        index += 2

    return tuple(result)


def _dedupe_assistant_tool_uses(
    message: ApiMessage,
    seen_tool_use_ids: set[str],
) -> tuple[ApiMessage, list[str]]:
    content: list[ModelContentBlock] = []
    current_tool_use_ids: list[str] = []
    changed = False
    for block in message.message.content:
        if isinstance(block, ToolUseContentBlock):
            if block.id in seen_tool_use_ids:
                changed = True
                continue
            seen_tool_use_ids.add(block.id)
            current_tool_use_ids.append(block.id)
        content.append(block)
    if not content:
        content = [TextContentBlock(text=TOOL_USE_INTERRUPTED_PLACEHOLDER)]
        changed = True
    if not changed:
        return message, current_tool_use_ids
    return _api_message_with_content(message, tuple(content)), current_tool_use_ids


def _repair_following_tool_result_message(
    message: ApiMessage,
    *,
    tool_use_ids: list[str],
) -> ApiMessage | None:
    tool_use_id_set = set(tool_use_ids)
    existing_result_ids = _tool_result_ids(message)
    missing_ids = [
        tool_use_id
        for tool_use_id in tool_use_ids
        if tool_use_id not in existing_result_ids
    ]
    orphaned_ids = existing_result_ids - tool_use_id_set

    if not missing_ids and not orphaned_ids and not _has_duplicate_tool_results(message):
        return message

    seen_result_ids: set[str] = set()
    repaired_content: list[ModelContentBlock] = [
        ToolResultContentBlock(
            tool_use_id=tool_use_id,
            content=SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
            is_error=True,
        )
        for tool_use_id in missing_ids
    ]
    for block in message.message.content:
        if not isinstance(block, ToolResultContentBlock):
            repaired_content.append(block)
            continue
        if block.tool_use_id in orphaned_ids:
            continue
        if block.tool_use_id in seen_result_ids:
            continue
        seen_result_ids.add(block.tool_use_id)
        repaired_content.append(block)

    if not repaired_content:
        return None
    return _api_message_with_content(message, tuple(repaired_content))


def _strip_orphaned_tool_results(
    message: ApiMessage,
    *,
    previous_is_assistant: bool,
    is_first_message: bool,
) -> ApiMessage | None:
    if message.message.role not in ("user", "tool") or not _has_tool_result(message):
        return message
    if previous_is_assistant:
        return message

    stripped = [
        block
        for block in message.message.content
        if not isinstance(block, ToolResultContentBlock)
    ]
    if stripped:
        return _api_message_from_model_message(
            ModelMessage(role="user", content=tuple(stripped))
        )
    if not is_first_message:
        return None
    return _plain_user_message(
        text=ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER,
    )


def _tool_result_ids(message: ApiMessage) -> set[str]:
    return {
        block.tool_use_id
        for block in message.message.content
        if isinstance(block, ToolResultContentBlock)
    }


def _has_duplicate_tool_results(message: ApiMessage) -> bool:
    seen: set[str] = set()
    for block in message.message.content:
        if not isinstance(block, ToolResultContentBlock):
            continue
        if block.tool_use_id in seen:
            return True
        seen.add(block.tool_use_id)
    return False


def _has_tool_result(message: ApiMessage) -> bool:
    return any(
        isinstance(block, ToolResultContentBlock) for block in message.message.content
    )


def _is_tool_reference_item(item: object) -> bool:
    if not isinstance(item, Mapping):
        return False
    mapping = cast(Mapping[object, object], item)
    return mapping.get("type") == "tool_reference"


def _should_strip_tool_reference(
    item: object,
    *,
    available_tool_names: set[str] | None,
) -> bool:
    if not _is_tool_reference_item(item):
        return False
    if available_tool_names is None:
        return True
    mapping = cast(Mapping[object, object], item)
    tool_name = mapping.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return False
    return tool_name not in available_tool_names


def _api_message_with_content(
    source: ApiMessage,
    content: tuple[ModelContentBlock, ...],
) -> ApiMessage:
    message = replace(
        source.message,
        content=content,
    )
    return _api_message_from_model_message(message)


def _plain_user_message(*, text: str) -> ApiMessage:
    return _api_message_from_model_message(
        ModelMessage(role="user", content=(TextContentBlock(text=text),))
    )


def _synthetic_tool_result_message(tool_use_ids: list[str]) -> ApiMessage:
    return _api_message_from_model_message(
        ModelMessage(
            role="user",
            content=tuple(
                ToolResultContentBlock(
                    tool_use_id=tool_use_id,
                    content=SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
                    is_error=True,
                )
                for tool_use_id in tool_use_ids
            ),
        )
    )


def _api_message_from_model_message(message: ModelMessage) -> ApiMessage:
    return ApiMessage(
        message=message,
        provider_payload=cast(FrozenJson, message_param_from_model_message(message)),
    )


__all__ = [
    "ORPHANED_TOOL_RESULT_REMOVED_PLACEHOLDER",
    "SYNTHETIC_TOOL_RESULT_PLACEHOLDER",
    "TOOL_REFERENCES_REMOVED_PLACEHOLDER",
    "UNAVAILABLE_TOOL_REFERENCES_REMOVED_PLACEHOLDER",
    "UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER",
    "UNSUPPORTED_IMAGE_REMOVED_PLACEHOLDER",
    "UNSUPPORTED_MEDIA_REMOVED_PLACEHOLDER",
    "normalize_model_request_for_provider",
]
