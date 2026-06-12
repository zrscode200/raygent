"""Model response normalization for the query loop.

The query loop normalizes model responses into a stable internal shape before
tool execution. Streaming adapters can record `tool_use` blocks as they arrive;
the normalized content-block semantics stay model-facing and provider-neutral.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """Normalized assistant `tool_use` content block.

    `input` intentionally stays `Any`: the model can emit invalid/non-object
    input, and the execution layer must be able to turn that into a
    model-visible validation error instead of losing the original shape.
    """

    id: str
    name: str
    input: Any
    index: int


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """Normalized assistant turn returned by a model call."""

    message: MessageParam
    tool_uses: tuple[ToolUseBlock, ...]


def normalize_assistant_turn(response: Any) -> AssistantTurn:
    """Normalize a provider response object or test dictionary.

    Supported shapes:
    - SDK-like objects with `.content`
    - dictionaries with `content`
    - legacy tests/simple fakes with `text`
    """
    raw_content = _response_field(response, "content")
    if raw_content is None:
        text = _response_field(response, "text")
        content: str | list[dict[str, Any]] = "" if text is None else str(text)
    else:
        content = _normalize_content(raw_content)

    message = cast("MessageParam", {"role": "assistant", "content": content})
    return AssistantTurn(message=message, tool_uses=tuple(_extract_tool_uses(content)))


def _response_field(response: Any, field: str) -> Any:
    if isinstance(response, Mapping):
        mapping = cast(Mapping[str, Any], response)
        return mapping.get(field)
    return getattr(response, field, None)


def _normalize_content(raw_content: Any) -> str | list[dict[str, Any]]:
    if isinstance(raw_content, str):
        return raw_content
    if _is_sequence_content(raw_content):
        return [_normalize_block(block) for block in raw_content]
    return str(raw_content)


def _is_sequence_content(raw_content: Any) -> bool:
    return isinstance(raw_content, Sequence) and not isinstance(
        raw_content,
        str | bytes | bytearray,
    )


def _normalize_block(block: Any) -> dict[str, Any]:
    mapping = _block_mapping(block)
    block_type = mapping.get("type")

    if block_type == "text":
        return {
            "type": "text",
            "text": str(mapping.get("text", "")),
        }

    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": str(mapping.get("id", "")),
            "name": str(mapping.get("name", "")),
            "input": mapping.get("input", {}),
        }

    return mapping


def _block_mapping(block: Any) -> dict[str, Any]:
    if isinstance(block, Mapping):
        return _string_key_dict(cast(Mapping[Any, Any], block))

    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, Mapping):
            return _string_key_dict(cast(Mapping[Any, Any], dumped))

    to_dict = getattr(block, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return _string_key_dict(cast(Mapping[Any, Any], dumped))

    data: dict[str, Any] = {}
    for attr in ("type", "text", "id", "name", "input"):
        if hasattr(block, attr):
            data[attr] = getattr(block, attr)
    if data:
        return data

    return {"type": "text", "text": str(block)}


def _string_key_dict(mapping: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in mapping.items()}


def _extract_tool_uses(content: str | list[dict[str, Any]]) -> list[ToolUseBlock]:
    if isinstance(content, str):
        return []

    tool_uses: list[ToolUseBlock] = []
    for index, block in enumerate(content):
        if block.get("type") != "tool_use":
            continue
        tool_uses.append(
            ToolUseBlock(
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
                input=block.get("input", {}),
                index=index,
            )
        )
    return tool_uses


__all__ = [
    "AssistantTurn",
    "ToolUseBlock",
    "normalize_assistant_turn",
]
