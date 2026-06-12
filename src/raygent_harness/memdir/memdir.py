# ruff: noqa: E501
"""Memory prompt builders and MEMORY.md entrypoint loading.

"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raygent_harness.memdir.memory_types import (
    MEMORY_FRONTMATTER_EXAMPLE,
    TRUSTING_RECALL_SECTION,
    TYPES_SECTION_INDIVIDUAL,
    WHAT_NOT_TO_SAVE_SECTION,
    WHEN_TO_ACCESS_SECTION,
)
from raygent_harness.memdir.paths import (
    AUTO_MEM_ENTRYPOINT_NAME,
    MemorySettings,
    get_auto_mem_path,
    is_auto_memory_enabled,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ENTRYPOINT_NAME = AUTO_MEM_ENTRYPOINT_NAME
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000
AUTO_MEM_DISPLAY_NAME = "auto memory"
DIR_EXISTS_GUIDANCE = (
    "This directory already exists - write to it directly with the Write tool "
    "(do not run mkdir or check for its existence)."
)
DIRS_EXIST_GUIDANCE = (
    "Both directories already exist - write to them directly with the Write tool "
    "(do not run mkdir or check for their existence)."
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntrypointTruncation:
    """Result of MEMORY.md line/byte capping."""

    content: str
    line_count: int
    byte_count: int
    was_line_truncated: bool
    was_byte_truncated: bool


def _format_file_size(byte_count: int) -> str:
    """Small human-readable formatter for truncation warnings."""
    kib = byte_count / 1024
    if kib < 1:
        return f"{byte_count} bytes"
    if kib < 1024:
        formatted = f"{kib:.1f}".removesuffix(".0")
        return f"{formatted}KB"
    mib = kib / 1024
    if mib < 1024:
        formatted = f"{mib:.1f}".removesuffix(".0")
        return f"{formatted}MB"
    formatted = f"{mib / 1024:.1f}".removesuffix(".0")
    return f"{formatted}GB"


def _js_len(text: str) -> int:
    """Return JS string length: UTF-16 code units."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def _truncate_to_js_units_at_line_boundary(text: str, max_units: int) -> str:
    if _js_len(text) <= max_units:
        return text

    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line]) if kept else line
        if _js_len(candidate) > max_units:
            break
        kept.append(line)

    if kept:
        return "\n".join(kept)

    encoded = text.encode("utf-16-le", errors="surrogatepass")[: max_units * 2]
    return encoded.decode("utf-16-le", errors="ignore")


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    """Trim and cap MEMORY.md by line count and JS string length.

    The reference names this a byte cap but implements it with JS string
    length. Raygent mirrors that behavior by counting UTF-16 code units.
    """
    trimmed = raw.strip()
    content_lines = trimmed.split("\n")
    line_count = len(content_lines)
    byte_count = _js_len(trimmed)

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(
            content=trimmed,
            line_count=line_count,
            byte_count=byte_count,
            was_line_truncated=False,
            was_byte_truncated=False,
        )

    truncated = "\n".join(content_lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    truncated = _truncate_to_js_units_at_line_boundary(truncated, MAX_ENTRYPOINT_BYTES)

    if was_byte_truncated and not was_line_truncated:
        reason = (
            f"{_format_file_size(byte_count)} (limit: {_format_file_size(MAX_ENTRYPOINT_BYTES)}) "
            "- index entries are too long"
        )
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {_format_file_size(byte_count)}"

    warning = (
        f"> WARNING: {ENTRYPOINT_NAME} is {reason}. Only part of it was loaded. "
        "Keep index entries to one line under ~200 chars; move detail into topic files."
    )
    return EntrypointTruncation(
        content=f"{truncated}\n\n{warning}",
        line_count=line_count,
        byte_count=byte_count,
        was_line_truncated=was_line_truncated,
        was_byte_truncated=was_byte_truncated,
    )


def ensure_memory_dir_exists(memory_dir: Path | str) -> bool:
    """Create `memory_dir` recursively, failing soft like the reference."""
    try:
        Path(memory_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _LOGGER.debug("ensure_memory_dir_exists failed for %s: %s", memory_dir, exc)
        return False
    return True


def build_searching_past_context_section(
    memory_dir: Path | str,
    *,
    enabled: bool = False,
    project_transcript_dir: Path | str | None = None,
    grep_tool_name: str = "Grep",
    embedded_search_tools: bool = False,
) -> list[str]:
    """Build the optional past-context search section.

    Reference gates this behind a feature flag. Raygent keeps it disabled by
    default until transcript/session search exists.
    """
    if not enabled:
        return []

    memory_path = os.fspath(memory_dir)
    transcript_path = os.fspath(project_transcript_dir or "<project-transcripts>")
    if embedded_search_tools:
        mem_search = f'grep -rn "<search term>" {memory_path} --include="*.md"'
        transcript_search = f'grep -rn "<search term>" {transcript_path}/ --include="*.jsonl"'
    else:
        mem_search = f'{grep_tool_name} with pattern="<search term>" path="{memory_path}" glob="*.md"'
        transcript_search = (
            f'{grep_tool_name} with pattern="<search term>" '
            f'path="{transcript_path}/" glob="*.jsonl"'
        )

    return [
        "## Searching past context",
        "",
        "When looking for past context:",
        "1. Search topic files in your memory directory:",
        "```",
        mem_search,
        "```",
        "2. Session transcript logs (last resort - large files, slow):",
        "```",
        transcript_search,
        "```",
        "Use narrow search terms (error messages, file paths, function names) rather than broad keywords.",
        "",
    ]


def build_memory_lines(
    display_name: str,
    memory_dir: Path | str,
    extra_guidelines: Sequence[str] | None = None,
    *,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> list[str]:
    """Build typed-memory behavioral instructions without MEMORY.md content."""
    memory_path = os.fspath(memory_dir)
    if skip_index:
        how_to_save = [
            "## How to save memories",
            "",
            "Write each memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
            "- Keep the name, description, and type fields in memory files up-to-date with the content",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]
    else:
        how_to_save = [
            "## How to save memories",
            "",
            "Saving a memory is a two-step process:",
            "",
            "**Step 1** - write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
            f"**Step 2** - add a pointer to that file in `{ENTRYPOINT_NAME}`. `{ENTRYPOINT_NAME}` is an index, not a memory - each entry should be one line, under ~150 characters: `- [Title](file.md) - one-line hook`. It has no frontmatter. Never write memory content directly into `{ENTRYPOINT_NAME}`.",
            "",
            f"- `{ENTRYPOINT_NAME}` is always loaded into your conversation context - lines after {MAX_ENTRYPOINT_LINES} will be truncated, so keep the index concise",
            "- Keep the name, description, and type fields in memory files up-to-date with the content",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]

    lines: list[str] = [
        f"# {display_name}",
        "",
        f"You have a persistent, file-based memory system at `{memory_path}`. {DIR_EXISTS_GUIDANCE}",
        "",
        "You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they'd like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.",
        "",
        "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.",
        "",
        *TYPES_SECTION_INDIVIDUAL,
        *WHAT_NOT_TO_SAVE_SECTION,
        "",
        *how_to_save,
        "",
        *WHEN_TO_ACCESS_SECTION,
        "",
        *TRUSTING_RECALL_SECTION,
        "",
        "## Memory and other forms of persistence",
        "Memory is one of several persistence mechanisms available to you as you assist the user in a given conversation. The distinction is often that memory can be recalled in future conversations and should not be used for persisting information that is only useful within the scope of the current conversation.",
        "- When to use or update a plan instead of memory: If you are about to start a non-trivial implementation task and would like to reach alignment with the user on your approach you should use a Plan rather than saving this information to memory. Similarly, if you already have a plan within the conversation and you have changed your approach persist that change by updating the plan rather than saving a memory.",
        "- When to use or update tasks instead of memory: When you need to break your work in current conversation into discrete steps or keep track of your progress use tasks instead of saving to memory. Tasks are great for persisting information about the work that needs to be done in the current conversation, but memory should be reserved for information that will be useful in future conversations.",
        "",
        *(extra_guidelines or ()),
        "",
    ]
    lines.extend(
        build_searching_past_context_section(
            memory_path,
            enabled=include_searching_past_context,
            project_transcript_dir=project_transcript_dir,
        )
    )
    return lines


def _entrypoint_path(memory_dir: Path | str) -> Path:
    return Path(memory_dir) / ENTRYPOINT_NAME


def build_memory_prompt(
    *,
    display_name: str,
    memory_dir: Path | str,
    extra_guidelines: Sequence[str] | None = None,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> str:
    """Build typed-memory instructions with MEMORY.md content included."""
    lines = build_memory_lines(
        display_name,
        memory_dir,
        extra_guidelines,
        skip_index=skip_index,
        include_searching_past_context=include_searching_past_context,
        project_transcript_dir=project_transcript_dir,
    )

    entrypoint_content = ""
    with suppress(OSError):
        entrypoint_content = _entrypoint_path(memory_dir).read_text(
            encoding="utf-8",
            errors="replace",
        )

    if entrypoint_content.strip():
        truncation = truncate_entrypoint_content(entrypoint_content)
        lines.extend([f"## {ENTRYPOINT_NAME}", "", truncation.content])
    else:
        lines.extend(
            [
                f"## {ENTRYPOINT_NAME}",
                "",
                f"Your {ENTRYPOINT_NAME} is currently empty. When you save new memories, they will appear here.",
            ]
        )

    return "\n".join(lines)


def load_memory_prompt(
    settings: MemorySettings,
    *,
    extra_guidelines: Sequence[str] | None = None,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> str | None:
    """Load the auto-only memory prompt for system-prompt inclusion.

    Returns `None` when auto memory is disabled. As in the reference, this
    system-prompt path creates the directory and returns behavioral mechanics;
    `build_memory_prompt()` is the entrypoint-content-inclusive variant.
    """
    if not is_auto_memory_enabled(settings):
        return None

    auto_dir = get_auto_mem_path(settings)
    ensure_memory_dir_exists(auto_dir)
    return "\n".join(
        build_memory_lines(
            AUTO_MEM_DISPLAY_NAME,
            auto_dir,
            extra_guidelines,
            skip_index=skip_index,
            include_searching_past_context=include_searching_past_context,
            project_transcript_dir=project_transcript_dir,
        )
    )


__all__ = [
    "AUTO_MEM_DISPLAY_NAME",
    "DIRS_EXIST_GUIDANCE",
    "DIR_EXISTS_GUIDANCE",
    "ENTRYPOINT_NAME",
    "MAX_ENTRYPOINT_BYTES",
    "MAX_ENTRYPOINT_LINES",
    "EntrypointTruncation",
    "build_memory_lines",
    "build_memory_prompt",
    "build_searching_past_context_section",
    "ensure_memory_dir_exists",
    "load_memory_prompt",
    "truncate_entrypoint_content",
]
