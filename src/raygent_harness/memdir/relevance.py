# ruff: noqa: E501
"""Relevant-memory selection with an injected selector.

"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from raygent_harness.memdir.memory_scan import (
    MemoryHeader,
    format_memory_manifest,
    scan_memory_files,
)

_LOGGER = logging.getLogger(__name__)

SELECT_MEMORIES_SYSTEM_PROMPT = """You are selecting memories that will be useful to this agent harness as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of filenames for the memories that will clearly be useful to this agent harness as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools (the current agent is already exercising them). DO still select memories containing warnings, gotchas, or known issues about those tools - active use is exactly when those matter.
"""


@dataclass(frozen=True)
class RelevantMemory:
    """Selected memory file and its mtime."""

    path: Path
    mtime_ms: float


class MemorySelector(Protocol):
    """Injected selector for model-backed or deterministic relevance choices."""

    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        """Return candidate memory filenames selected from `manifest`."""
        ...


class NoOpMemorySelector:
    """Default selector used until Raygent has a side-query model seam."""

    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools, abort_event
        return []


def build_select_memories_user_prompt(
    *,
    query: str,
    manifest: str,
    recent_tools: tuple[str, ...] = (),
) -> str:
    """Build the reference selector user prompt body."""
    tools_section = f"\n\nRecently used tools: {', '.join(recent_tools)}" if recent_tools else ""
    return f"Query: {query}\n\nAvailable memories:\n{manifest}{tools_section}"


async def select_relevant_memories(
    *,
    query: str,
    memories: list[MemoryHeader],
    selector: MemorySelector,
    recent_tools: tuple[str, ...] = (),
    abort_event: asyncio.Event | None = None,
) -> list[str]:
    """Ask the injected selector and validate returned filenames."""
    if abort_event is not None and abort_event.is_set():
        return []

    valid_filenames = {memory.filename for memory in memories}
    manifest = format_memory_manifest(memories)

    try:
        selected = await selector.select(
            query=query,
            manifest=manifest,
            recent_tools=recent_tools,
            abort_event=abort_event,
        )
    except asyncio.CancelledError:
        if abort_event is not None and abort_event.is_set():
            return []
        raise
    except Exception as exc:
        if abort_event is not None and abort_event.is_set():
            return []
        _LOGGER.warning("[memdir] select_relevant_memories failed: %s", exc)
        return []

    if abort_event is not None and abort_event.is_set():
        return []

    return [filename for filename in selected if filename in valid_filenames]


async def find_relevant_memories(
    *,
    query: str,
    memory_dir: Path | str,
    selector: MemorySelector | None = None,
    recent_tools: tuple[str, ...] = (),
    already_surfaced: set[Path | str] | None = None,
    abort_event: asyncio.Event | None = None,
) -> list[RelevantMemory]:
    """Return relevant memory paths + mtimes for `query`.

    Scans memory headers, filters already surfaced files before selection, asks
    the injected selector, validates selected filenames, and maps them back to
    file paths. The default selector returns no memories until a real side-query
    seam exists.
    """
    if abort_event is not None and abort_event.is_set():
        return []

    surfaced = {Path(path) for path in (already_surfaced or set())}
    memories = [
        memory
        for memory in scan_memory_files(memory_dir)
        if memory.file_path not in surfaced
    ]
    if not memories:
        return []

    selected_filenames = await select_relevant_memories(
        query=query,
        memories=memories,
        selector=selector or NoOpMemorySelector(),
        recent_tools=recent_tools,
        abort_event=abort_event,
    )
    by_filename = {memory.filename: memory for memory in memories}
    selected = [by_filename[filename] for filename in selected_filenames if filename in by_filename]

    return [RelevantMemory(path=memory.file_path, mtime_ms=memory.mtime_ms) for memory in selected]


__all__ = [
    "SELECT_MEMORIES_SYSTEM_PROMPT",
    "MemorySelector",
    "NoOpMemorySelector",
    "RelevantMemory",
    "build_select_memories_user_prompt",
    "find_relevant_memories",
    "select_relevant_memories",
]
