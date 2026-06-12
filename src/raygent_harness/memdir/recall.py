"""Provider-independent relevant-memory surfacing primitives.

"""

from __future__ import annotations

import asyncio
import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from raygent_harness.core.file_state import (
    FileState,
    ReadFileStateCache,
    normalize_file_state_path,
)
from raygent_harness.core.messages import MessageParam, user_message
from raygent_harness.memdir.memory_age import memory_age, memory_freshness_text
from raygent_harness.memdir.relevance import RelevantMemory

MAX_RELEVANT_MEMORY_FILES = 5
MAX_RELEVANT_MEMORY_LINES = 200
MAX_RELEVANT_MEMORY_BYTES = 4096
MAX_RELEVANT_MEMORY_SESSION_BYTES = 60 * 1024

RELEVANT_MEMORIES_TAG = "relevant_memories"
RELEVANT_MEMORY_TAG = "memory"
MEMORY_RECALL_MESSAGE_KIND: Final = "memory_recall"
RELEVANT_MEMORY_REMINDER = (
    "Relevant memory context was automatically recalled for this turn.\n"
    "Treat memory as context that may be stale; verify file paths or repo facts "
    "against current state before acting on them."
)
MEMORY_TRUNCATION_NOTE = (
    "Use the Read tool to view the complete file at: {path}"
)


@dataclass(frozen=True, slots=True)
class MemoryRecallFile:
    """Selected memory file ready for bounded surfacing."""

    path: Path
    mtime_ms: float


@dataclass(frozen=True, slots=True)
class SurfacedMemoryFile:
    """Memory content that will be injected into model-visible context."""

    path: Path
    content: str
    mtime_ms: float
    header: str
    limit: int | None = None
    truncated: bool = False
    truncated_by_bytes: bool = False
    truncated_by_lines: bool = False

    @property
    def content_bytes(self) -> int:
        return len(self.content.encode())


@dataclass(frozen=True, slots=True)
class SurfacedMemoryScan:
    """Relevant memories already visible in the current transcript."""

    paths: frozenset[str]
    total_bytes: int


@dataclass(frozen=True, slots=True)
class MemoryRecallMarkResult:
    """Result of consume-time duplicate filtering and read-state marking."""

    memories: tuple[SurfacedMemoryFile, ...]
    duplicate_filtered_count: int
    marked_count: int


def memory_header(
    path: str | os.PathLike[str],
    mtime_ms: float,
    *,
    now_ms: float | None = None,
) -> str:
    """Build the reference-style header for one recalled memory."""

    normalized = normalize_file_state_path(path)
    staleness = memory_freshness_text(mtime_ms, now_ms=now_ms)
    if staleness:
        return f"{staleness}\n\nMemory: {normalized}:"
    return f"Memory (saved {memory_age(mtime_ms, now_ms=now_ms)}): {normalized}:"


def message_from_surfaced_memories(
    memories: tuple[SurfacedMemoryFile, ...],
) -> MessageParam | None:
    """Format recalled memories as one provider-neutral user message."""

    if not memories:
        return None
    blocks = "\n\n".join(_format_memory_block(memory) for memory in memories)
    message = user_message(
        "<system-reminder>\n"
        f"{RELEVANT_MEMORY_REMINDER}\n\n"
        f"<{RELEVANT_MEMORIES_TAG}>\n"
        f"{blocks}\n"
        f"</{RELEVANT_MEMORIES_TAG}>\n"
        "</system-reminder>"
    )
    message["raygentMessageKind"] = MEMORY_RECALL_MESSAGE_KIND
    message["raygentMemoryRecall"] = {
        "type": RELEVANT_MEMORIES_TAG,
        "memories": [
            {
                "path": normalize_file_state_path(memory.path),
                "content_bytes": memory.content_bytes,
            }
            for memory in memories
        ],
    }
    return message


def collect_surfaced_memories(messages: tuple[MessageParam, ...]) -> SurfacedMemoryScan:
    """Collect relevant-memory paths and byte counts from current messages."""

    paths: set[str] = set()
    total_bytes = 0
    for message in messages:
        for parsed in _memory_recall_metadata(message):
            paths.add(parsed.path)
            total_bytes += parsed.content_bytes
    return SurfacedMemoryScan(paths=frozenset(paths), total_bytes=total_bytes)


def is_memory_recall_message(message: MessageParam) -> bool:
    """Return whether `message` is a Raygent-generated memory recall message."""

    return message.get("raygentMessageKind") == MEMORY_RECALL_MESSAGE_KIND


async def read_memories_for_surfacing(
    selected: tuple[RelevantMemory | MemoryRecallFile, ...],
    *,
    abort_event: asyncio.Event | None = None,
    max_files: int = MAX_RELEVANT_MEMORY_FILES,
    max_lines: int = MAX_RELEVANT_MEMORY_LINES,
    max_bytes: int = MAX_RELEVANT_MEMORY_BYTES,
) -> tuple[SurfacedMemoryFile, ...]:
    """Read selected memory files with reference-sized line and byte caps."""

    surfaced: list[SurfacedMemoryFile] = []
    for memory in selected[:max_files]:
        _raise_if_aborted(abort_event)
        try:
            surfaced.append(
                await asyncio.to_thread(
                    read_memory_for_surfacing,
                    memory.path,
                    memory.mtime_ms,
                    max_lines=max_lines,
                    max_bytes=max_bytes,
                )
            )
        except (OSError, UnicodeError):
            continue
    return tuple(surfaced)


def read_memory_for_surfacing(
    path: str | os.PathLike[str],
    mtime_ms: float | None = None,
    *,
    max_lines: int = MAX_RELEVANT_MEMORY_LINES,
    max_bytes: int = MAX_RELEVANT_MEMORY_BYTES,
    now_ms: float | None = None,
) -> SurfacedMemoryFile:
    """Read one memory file, truncating selected content instead of failing."""

    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    normalized = Path(normalize_file_state_path(path))
    stats = normalized.stat()
    effective_mtime_ms = mtime_ms if mtime_ms is not None else stats.st_mtime * 1000
    content, line_count, truncated_by_lines, truncated_by_bytes = _read_bounded_text(
        normalized,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
    truncated = truncated_by_lines or truncated_by_bytes
    if truncated:
        reason = f"{max_bytes} byte limit" if truncated_by_bytes else f"first {max_lines} lines"
        content = (
            f"{content}\n\n"
            f"> This memory file was truncated ({reason}). "
            f"{MEMORY_TRUNCATION_NOTE.format(path=normalized)}"
        )

    return SurfacedMemoryFile(
        path=normalized,
        content=content,
        mtime_ms=effective_mtime_ms,
        header=memory_header(normalized, effective_mtime_ms, now_ms=now_ms),
        limit=line_count if truncated else None,
        truncated=truncated,
        truncated_by_bytes=truncated_by_bytes,
        truncated_by_lines=truncated_by_lines,
    )


def filter_and_mark_recalled_memories(
    memories: tuple[SurfacedMemoryFile, ...],
    read_file_state: ReadFileStateCache,
) -> MemoryRecallMarkResult:
    """Filter duplicate recalls against read state, then mark survivors."""

    survivors: list[SurfacedMemoryFile] = []
    duplicate_filtered_count = 0
    for memory in memories:
        if read_file_state.has(memory.path):
            duplicate_filtered_count += 1
            continue
        survivors.append(memory)

    for memory in survivors:
        read_file_state.set(
            memory.path,
            FileState(
                content=memory.content,
                timestamp=int(memory.mtime_ms),
                offset=1,
                limit=memory.limit,
                is_partial_view=memory.truncated,
            ),
        )

    return MemoryRecallMarkResult(
        memories=tuple(survivors),
        duplicate_filtered_count=duplicate_filtered_count,
        marked_count=len(survivors),
    )


def collect_recent_successful_tools(
    messages: tuple[MessageParam, ...],
    *,
    last_user_message: MessageParam | None = None,
) -> tuple[str, ...]:
    """Return tool names that succeeded since the previous real user turn."""

    boundary = last_user_message or _last_human_user_message(messages)
    use_id_to_name: dict[str, str] = {}
    result_by_use_id: dict[str, bool] = {}

    for message in reversed(messages):
        if (
            _is_human_user_message(message)
            and boundary is not None
            and message is not boundary
        ):
            break

        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    tool_name = block.get("name")
                    if isinstance(tool_id, str) and isinstance(tool_name, str):
                        use_id_to_name[tool_id] = tool_name
        elif message.get("role") == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if isinstance(tool_id, str):
                        result_by_use_id[tool_id] = block.get("is_error") is True

    failed: set[str] = set()
    succeeded: list[str] = []
    for tool_id, tool_name in use_id_to_name.items():
        errored = result_by_use_id.get(tool_id)
        if errored is None:
            continue
        if errored:
            failed.add(tool_name)
        elif tool_name not in succeeded:
            succeeded.append(tool_name)

    return tuple(tool for tool in succeeded if tool not in failed)


def _format_memory_block(memory: SurfacedMemoryFile) -> str:
    attrs: dict[str, str] = {
        "path": normalize_file_state_path(memory.path),
        "mtime_ms": str(int(memory.mtime_ms)),
        "content_bytes": str(memory.content_bytes),
    }
    if memory.limit is not None:
        attrs["limit"] = str(memory.limit)
    if memory.truncated:
        attrs["truncated"] = "true"
    attr_text = " ".join(
        f'{key}="{html.escape(value, quote=True)}"' for key, value in attrs.items()
    )
    return (
        f"<{RELEVANT_MEMORY_TAG} {attr_text}>\n"
        f"{memory.header}\n"
        f"{memory.content}\n"
        f"</{RELEVANT_MEMORY_TAG}>"
    )


@dataclass(frozen=True, slots=True)
class _ParsedMemoryBlock:
    path: str
    content_bytes: int


def _memory_recall_metadata(message: MessageParam) -> tuple[_ParsedMemoryBlock, ...]:
    if not is_memory_recall_message(message):
        return ()
    metadata = message.get("raygentMemoryRecall")
    if metadata is None or metadata.get("type") != RELEVANT_MEMORIES_TAG:
        return ()
    parsed: list[_ParsedMemoryBlock] = []
    for raw_memory in metadata["memories"]:
        if raw_memory["content_bytes"] < 0:
            continue
        parsed.append(
            _ParsedMemoryBlock(
                path=normalize_file_state_path(raw_memory["path"]),
                content_bytes=raw_memory["content_bytes"],
            )
        )
    return tuple(parsed)


def _read_bounded_text(
    path: Path,
    *,
    max_lines: int,
    max_bytes: int,
) -> tuple[str, int, bool, bool]:
    parts: list[str] = []
    selected_bytes = 0
    selected_lines = 0
    truncated_by_lines = False
    truncated_by_bytes = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line_index >= max_lines:
                truncated_by_lines = True
                break

            line_bytes = line.encode()
            next_size = selected_bytes + len(line_bytes)
            if next_size <= max_bytes:
                parts.append(line)
                selected_bytes = next_size
                selected_lines += 1
                continue

            remaining = max_bytes - selected_bytes
            if remaining > 0:
                parts.append(_truncate_to_utf8_bytes(line, remaining))
                selected_lines += 1
            truncated_by_bytes = True
            break

    return (
        "".join(parts).removesuffix("\n"),
        selected_lines,
        truncated_by_lines,
        truncated_by_bytes,
    )


def _truncate_to_utf8_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    return text.encode()[:max_bytes].decode("utf-8", errors="ignore")


def _last_human_user_message(messages: tuple[MessageParam, ...]) -> MessageParam | None:
    for message in reversed(messages):
        if _is_human_user_message(message):
            return message
    return None


def _is_human_user_message(message: MessageParam) -> bool:
    if message.get("role") != "user":
        return False
    if _has_tool_result_content(message):
        return False
    return not is_memory_recall_message(message)


def _has_tool_result_content(message: MessageParam) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("type") == "tool_result" for block in content)


def _raise_if_aborted(abort_event: asyncio.Event | None) -> None:
    if abort_event is not None and abort_event.is_set():
        raise asyncio.CancelledError()


def memory_recall_file_from_relevant(memory: RelevantMemory) -> MemoryRecallFile:
    return MemoryRecallFile(path=memory.path, mtime_ms=memory.mtime_ms)


__all__ = [
    "MAX_RELEVANT_MEMORY_BYTES",
    "MAX_RELEVANT_MEMORY_FILES",
    "MAX_RELEVANT_MEMORY_LINES",
    "MAX_RELEVANT_MEMORY_SESSION_BYTES",
    "MEMORY_RECALL_MESSAGE_KIND",
    "MEMORY_TRUNCATION_NOTE",
    "RELEVANT_MEMORIES_TAG",
    "RELEVANT_MEMORY_REMINDER",
    "MemoryRecallFile",
    "MemoryRecallMarkResult",
    "SurfacedMemoryFile",
    "SurfacedMemoryScan",
    "collect_recent_successful_tools",
    "collect_surfaced_memories",
    "filter_and_mark_recalled_memories",
    "is_memory_recall_message",
    "memory_header",
    "memory_recall_file_from_relevant",
    "message_from_surfaced_memories",
    "read_memories_for_surfacing",
    "read_memory_for_surfacing",
]
