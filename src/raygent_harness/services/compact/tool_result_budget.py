"""Tool-result budget rewrite.

  `tool_use_id`, re-applies stable replacements, and only replaces fresh
  over-budget results.

Raygent v1 keeps the same semantics over Raygent's provider-neutral message shape:
oversized `tool_result` blocks are written to disk and replaced with a stable
marker. Transcript record/reconstruction is deferred until resume/session
storage exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from raygent_harness.core.tool import ContentReplacementState

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


PERSISTED_TOOL_RESULT_TAG = "[tool result persisted]"


@dataclass(frozen=True)
class ToolResultReplacementRecord:
    """One newly persisted replacement made during a budget pass."""

    tool_use_id: str
    replacement: str
    path: str
    original_size_chars: int


@dataclass(frozen=True)
class ToolResultBudgetResult:
    """Result of one tool-result budget pass."""

    messages: list[MessageParam]
    newly_replaced: tuple[ToolResultReplacementRecord, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    message_index: int
    block_index: int
    tool_use_id: str
    content: object
    size: int


async def apply_tool_result_budget(
    messages: list[MessageParam],
    state: ContentReplacementState | None,
) -> ToolResultBudgetResult:
    """Apply per-message aggregate tool-result budget.

    Mutates `state.seen_ids` and `state.replacements`, matching the reference's
    stateful prefix-stability contract. Returns the original message list when
    no replacement is needed.
    """
    if state is None:
        return ToolResultBudgetResult(messages=messages)

    groups = _collect_candidates_by_wire_message(messages)
    replacement_map: dict[str, str] = {}
    selected: list[_Candidate] = []

    for candidates in groups:
        must_reapply: list[_Candidate] = []
        frozen: list[_Candidate] = []
        fresh: list[_Candidate] = []

        for candidate in candidates:
            replacement = state.replacements.get(candidate.tool_use_id)
            if replacement is not None:
                must_reapply.append(candidate)
                replacement_map[candidate.tool_use_id] = replacement
            elif candidate.tool_use_id in state.seen_ids:
                frozen.append(candidate)
            else:
                fresh.append(candidate)

        if not fresh:
            for candidate in candidates:
                state.seen_ids.add(candidate.tool_use_id)
            continue

        frozen_size = sum(candidate.size for candidate in frozen)
        fresh_size = sum(candidate.size for candidate in fresh)
        to_replace = (
            _select_fresh_to_replace(
                fresh,
                frozen_size=frozen_size,
                limit=state.max_result_size_chars,
            )
            if frozen_size + fresh_size > state.max_result_size_chars
            else []
        )
        selected_ids = {candidate.tool_use_id for candidate in to_replace}

        for candidate in candidates:
            if candidate.tool_use_id not in selected_ids:
                state.seen_ids.add(candidate.tool_use_id)

        selected.extend(to_replace)

    newly_replaced: list[ToolResultReplacementRecord] = []
    for candidate in selected:
        state.seen_ids.add(candidate.tool_use_id)
        record = _persist_tool_result(candidate, state.replaced_outputs_dir)
        if record is None:
            continue
        state.replacements[candidate.tool_use_id] = record.replacement
        replacement_map[candidate.tool_use_id] = record.replacement
        newly_replaced.append(record)

    if not replacement_map:
        return ToolResultBudgetResult(messages=messages)

    return ToolResultBudgetResult(
        messages=_replace_tool_result_contents(messages, replacement_map),
        newly_replaced=tuple(newly_replaced),
    )


def _collect_candidates_by_wire_message(
    messages: list[MessageParam],
) -> list[list[_Candidate]]:
    groups: list[list[_Candidate]] = []
    current: list[_Candidate] = []
    seen_assistant_ids: set[str] = set()

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
        current = []

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            current.extend(_collect_candidates_from_message(message_index, message))
        elif role == "assistant":
            # Same-id assistant fragments are merged before the API call in
            # the reference, so a repeated id must not split the following
            # tool_result blocks into separate wire-message budget groups.
            assistant_id = message.get("id")
            if isinstance(assistant_id, str):
                if assistant_id in seen_assistant_ids:
                    continue
                seen_assistant_ids.add(assistant_id)
            flush()
    flush()
    return groups


def _collect_candidates_from_message(
    message_index: int,
    message: MessageParam,
) -> list[_Candidate]:
    content = message.get("content")
    if not isinstance(content, list):
        return []

    candidates: list[_Candidate] = []
    for block_index, block in enumerate(content):
        if block.get("type") != "tool_result":
            continue
        tool_use_id = block.get("tool_use_id")
        block_content = block.get("content")
        if not isinstance(tool_use_id, str) or block_content in (None, ""):
            continue
        if _is_already_persisted(block_content) or _has_image_block(block_content):
            continue
        candidates.append(
            _Candidate(
                message_index=message_index,
                block_index=block_index,
                tool_use_id=tool_use_id,
                content=block_content,
                size=_content_size(block_content),
            )
        )
    return candidates


def _select_fresh_to_replace(
    fresh: list[_Candidate],
    *,
    frozen_size: int,
    limit: int,
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    remaining = frozen_size + sum(candidate.size for candidate in fresh)
    for candidate in sorted(fresh, key=lambda c: c.size, reverse=True):
        if remaining <= limit:
            break
        selected.append(candidate)
        remaining -= candidate.size
    return selected


def _replace_tool_result_contents(
    messages: list[MessageParam],
    replacement_map: dict[str, str],
) -> list[MessageParam]:
    rewritten: list[MessageParam] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            rewritten.append(message)
            continue

        changed = False
        next_content: list[Any] = []
        for block in content:
            block_dict = _as_dict(block)
            tool_use_id = block_dict.get("tool_use_id") if block_dict else None
            if (
                block_dict is not None
                and block_dict.get("type") == "tool_result"
                and isinstance(tool_use_id, str)
                and tool_use_id in replacement_map
            ):
                next_block = {**block_dict, "content": replacement_map[tool_use_id]}
                next_content.append(next_block)
                changed = True
            else:
                next_content.append(block)

        rewritten.append(
            cast(
                "MessageParam",
                {**message, "content": next_content} if changed else message,
            )
        )
    return rewritten


def _persist_tool_result(
    candidate: _Candidate,
    output_dir: str,
) -> ToolResultReplacementRecord | None:
    if _has_non_text_block(candidate.content):
        return None

    path = Path(output_dir).expanduser()
    file_path = path / f"{_safe_filename(candidate.tool_use_id)}.txt"
    serialized = _serialize_content(candidate.content)
    try:
        path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(serialized, encoding="utf-8")
    except OSError:
        return None

    replacement = (
        f"{PERSISTED_TOOL_RESULT_TAG}\n"
        f"tool_use_id: {candidate.tool_use_id}\n"
        f"original_chars: {candidate.size}\n"
        f"path: {file_path}"
    )
    return ToolResultReplacementRecord(
        tool_use_id=candidate.tool_use_id,
        replacement=replacement,
        path=str(file_path),
        original_size_chars=candidate.size,
    )


def _content_size(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in cast("list[object]", content):
            block_dict = _as_dict(block)
            if block_dict is not None and block_dict.get("type") == "text":
                text = block_dict.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return len(str(content))


def _serialize_content(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _has_image_block(content: object) -> bool:
    if not isinstance(content, list):
        return False
    blocks = cast("list[object]", content)
    return any(
        (block_dict := _as_dict(block)) is not None
        and block_dict.get("type") == "image"
        for block in blocks
    )


def _has_non_text_block(content: object) -> bool:
    if not isinstance(content, list):
        return False
    blocks = cast("list[object]", content)
    return any(
        (block_dict := _as_dict(block)) is None or block_dict.get("type") != "text"
        for block in blocks
    )


def _is_already_persisted(content: object) -> bool:
    return isinstance(content, str) and content.startswith(PERSISTED_TOOL_RESULT_TAG)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "tool_result"


def _as_dict(value: object) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


__all__ = [
    "PERSISTED_TOOL_RESULT_TAG",
    "ToolResultBudgetResult",
    "ToolResultReplacementRecord",
    "apply_tool_result_budget",
]
