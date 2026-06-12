# ruff: noqa: E501
"""Prompt templates for background memory extraction.

"""

from __future__ import annotations

from collections.abc import Sequence

from raygent_harness.memdir.memdir import ENTRYPOINT_NAME
from raygent_harness.memdir.memory_types import (
    MEMORY_FRONTMATTER_EXAMPLE,
    TYPES_SECTION_INDIVIDUAL,
    WHAT_NOT_TO_SAVE_SECTION,
)

FILE_READ_TOOL_NAME = "Read"
FILE_WRITE_TOOL_NAME = "Write"
FILE_EDIT_TOOL_NAME = "Edit"
GLOB_TOOL_NAME = "Glob"
GREP_TOOL_NAME = "Grep"
BASH_TOOL_NAME = "Bash"


def _opener(
    new_message_count: int,
    existing_memories: str,
    *,
    allowed_tool_names: Sequence[str] | None = None,
) -> str:
    manifest = ""
    if existing_memories:
        manifest = (
            "\n\n## Existing memory files\n\n"
            f"{existing_memories}\n\n"
            "Check this list before writing - update an existing file rather than creating a duplicate."
        )

    tool_policy = (
        _default_tool_policy_text()
        if allowed_tool_names is None
        else _tool_policy_text(allowed_tool_names)
    )
    strategy = (
        _default_strategy_text()
        if allowed_tool_names is None
        else _strategy_text(allowed_tool_names)
    )

    return "\n".join(
        [
            "You are now acting as the memory extraction subagent. Analyze the "
            f"most recent ~{new_message_count} messages above and use them to "
            "update your persistent memory systems.",
            "",
            tool_policy,
            "",
            strategy,
            "",
            f"You MUST only use content from the last ~{new_message_count} "
            "messages to update your persistent memories. Do not waste any "
            "turns attempting to investigate or verify that content further - "
            "no grepping source files, no reading code to confirm a pattern "
            "exists, no git commands."
            + manifest,
        ]
    )


def _default_tool_policy_text() -> str:
    return (
        f"Available tools: {FILE_READ_TOOL_NAME}, {GREP_TOOL_NAME}, "
        f"{GLOB_TOOL_NAME}, read-only {BASH_TOOL_NAME} "
        f"(ls/find/cat/stat/wc/head/tail and similar), and "
        f"{FILE_EDIT_TOOL_NAME}/{FILE_WRITE_TOOL_NAME} for paths inside "
        f"the memory directory only. {BASH_TOOL_NAME} rm is not permitted. "
        f"All other tools - MCP, Agent, write-capable {BASH_TOOL_NAME}, "
        "etc - will be denied."
    )


def _default_strategy_text() -> str:
    return (
        f"You have a limited turn budget. {FILE_EDIT_TOOL_NAME} requires a "
        f"prior {FILE_READ_TOOL_NAME} of the same file, so the efficient "
        f"strategy is: turn 1 - issue all {FILE_READ_TOOL_NAME} calls in "
        f"parallel for every file you might update; turn 2 - issue all "
        f"{FILE_WRITE_TOOL_NAME}/{FILE_EDIT_TOOL_NAME} calls in parallel. "
        "Do not interleave reads and writes across multiple turns."
    )


def _tool_policy_text(allowed_tool_names: Sequence[str]) -> str:
    allowed = tuple(dict.fromkeys(allowed_tool_names))
    read_tools = tuple(
        name
        for name in (FILE_READ_TOOL_NAME, GREP_TOOL_NAME, GLOB_TOOL_NAME)
        if name in allowed
    )
    write_tools = tuple(
        name for name in (FILE_WRITE_TOOL_NAME, FILE_EDIT_TOOL_NAME) if name in allowed
    )
    pieces: list[str] = []
    if read_tools:
        pieces.append(_join_tool_names(read_tools))
    if BASH_TOOL_NAME in allowed:
        pieces.append(
            f"read-only {BASH_TOOL_NAME} (ls/find/cat/stat/wc/head/tail and similar)"
        )
    if write_tools:
        pieces.append(
            f"{_join_tool_names(write_tools, separator='/')} for paths inside "
            "the memory directory only"
        )
    if not pieces:
        return "No tools are available. All tools will be denied."

    denied = "All other tools - MCP, Agent, Task, Skill, remote, and team tools - will be denied."
    if BASH_TOOL_NAME in allowed:
        denied = (
            f"{BASH_TOOL_NAME} rm and write-capable {BASH_TOOL_NAME} are not permitted. "
            f"{denied}"
        )
    return f"Available tools: {_join_tool_names(tuple(pieces))}. {denied}"


def _strategy_text(allowed_tool_names: Sequence[str]) -> str:
    allowed = frozenset(allowed_tool_names)
    writable = tuple(
        name for name in (FILE_WRITE_TOOL_NAME, FILE_EDIT_TOOL_NAME) if name in allowed
    )
    if FILE_READ_TOOL_NAME in allowed and FILE_EDIT_TOOL_NAME in writable:
        return _default_strategy_text()
    if FILE_READ_TOOL_NAME in allowed and writable:
        return (
            f"You have a limited turn budget. Use {FILE_READ_TOOL_NAME} first "
            "for any existing file you might update, then issue all "
            f"{_join_tool_names(writable, separator='/')} calls together. Do not "
            "interleave reads and writes across multiple turns."
        )
    if writable:
        return (
            f"You have a limited turn budget. Use {_join_tool_names(writable, separator='/')} "
            "only for clear memory updates from the recent messages. Do not waste "
            "turns trying unavailable tools."
        )
    return "You have a limited turn budget. Do not waste turns trying unavailable tools."


def _join_tool_names(names: Sequence[str], *, separator: str = ", ") -> str:
    if not names:
        return ""
    if separator != ", ":
        return separator.join(names)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def build_extract_auto_only_prompt(
    new_message_count: int,
    existing_memories: str,
    *,
    skip_index: bool = False,
    allowed_tool_names: Sequence[str] | None = None,
) -> str:
    """Build the extraction prompt for auto-only memory."""
    if skip_index:
        how_to_save = [
            "## How to save memories",
            "",
            "Write each memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
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
            f"- `{ENTRYPOINT_NAME}` is always loaded into your system prompt - lines after 200 will be truncated, so keep the index concise",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]

    return "\n".join(
        [
            _opener(
                new_message_count,
                existing_memories,
                allowed_tool_names=allowed_tool_names,
            ),
            "",
            "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.",
            "",
            *TYPES_SECTION_INDIVIDUAL,
            *WHAT_NOT_TO_SAVE_SECTION,
            "",
            *how_to_save,
        ]
    )


def build_extract_combined_prompt(
    new_message_count: int,
    existing_memories: str,
    *,
    skip_index: bool = False,
    allowed_tool_names: Sequence[str] | None = None,
) -> str:
    """Team-memory prompt placeholder.

    Team-memory sync/path routing is handled by the dedicated sync service, so
    the combined prompt degrades to the auto-only prompt when team memory is not
    enabled.
    """
    return build_extract_auto_only_prompt(
        new_message_count,
        existing_memories,
        skip_index=skip_index,
        allowed_tool_names=allowed_tool_names,
    )


__all__ = [
    "BASH_TOOL_NAME",
    "FILE_EDIT_TOOL_NAME",
    "FILE_READ_TOOL_NAME",
    "FILE_WRITE_TOOL_NAME",
    "GLOB_TOOL_NAME",
    "GREP_TOOL_NAME",
    "build_extract_auto_only_prompt",
    "build_extract_combined_prompt",
]
