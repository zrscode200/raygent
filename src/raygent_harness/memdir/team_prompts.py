# ruff: noqa: E501
"""Combined private/team memory prompt builders.

"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from raygent_harness.memdir.memdir import (
    DIRS_EXIST_GUIDANCE,
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_LINES,
    build_searching_past_context_section,
    ensure_memory_dir_exists,
)
from raygent_harness.memdir.memory_types import (
    MEMORY_DRIFT_CAVEAT,
    MEMORY_FRONTMATTER_EXAMPLE,
    TRUSTING_RECALL_SECTION,
    TYPES_SECTION_COMBINED,
    WHAT_NOT_TO_SAVE_SECTION,
)
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path
from raygent_harness.memdir.team_paths import (
    get_team_mem_path,
    is_team_memory_enabled,
)


def build_combined_memory_prompt(
    settings: MemorySettings,
    extra_guidelines: Sequence[str] | None = None,
    *,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> str:
    """Build the mechanics prompt for private and shared team memory."""
    auto_dir = get_auto_mem_path(settings)
    team_dir = get_team_mem_path(settings)

    if skip_index:
        how_to_save = [
            "## How to save memories",
            "",
            "Write each memory to its own file in the chosen directory (private or team, per the type's scope guidance) using this frontmatter format:",
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
            "**Step 1** - write the memory to its own file in the chosen directory (private or team, per the type's scope guidance) using this frontmatter format:",
            "",
            *MEMORY_FRONTMATTER_EXAMPLE,
            "",
            f"**Step 2** - add a pointer to that file in the same directory's `{ENTRYPOINT_NAME}`. Each directory (private and team) has its own `{ENTRYPOINT_NAME}` index - each entry should be one line, under ~150 characters: `- [Title](file.md) - one-line hook`. They have no frontmatter. Never write memory content directly into a `{ENTRYPOINT_NAME}`.",
            "",
            f"- Both `{ENTRYPOINT_NAME}` indexes are loaded into your conversation context - lines after {MAX_ENTRYPOINT_LINES} will be truncated, so keep them concise",
            "- Keep the name, description, and type fields in memory files up-to-date with the content",
            "- Organize memory semantically by topic, not chronologically",
            "- Update or remove memories that turn out to be wrong or outdated",
            "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        ]

    lines = [
        "# Memory",
        "",
        f"You have a persistent, file-based memory system with two directories: a private directory at `{auto_dir}` and a shared team directory at `{team_dir}`. {DIRS_EXIST_GUIDANCE}",
        "",
        "You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they'd like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.",
        "",
        "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.",
        "",
        "## Memory scope",
        "",
        "There are two scope levels:",
        "",
        f"- private: memories that are private between you and the current user. They persist across conversations with only this specific user and are stored at the root `{auto_dir}`.",
        f"- team: memories that are shared with and contributed by all of the users who work within this project directory. Team memories are synced at the beginning of every session and they are stored at `{team_dir}`.",
        "",
        *TYPES_SECTION_COMBINED,
        *WHAT_NOT_TO_SAVE_SECTION,
        "- You MUST avoid saving sensitive data within shared team memories. For example, never save API keys or user credentials.",
        "",
        *how_to_save,
        "",
        "## When to access memories",
        "- When memories (personal or team) seem relevant, or the user references prior work with them or others in their organization.",
        "- You MUST access memory when the user explicitly asks you to check, recall, or remember.",
        "- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.",
        MEMORY_DRIFT_CAVEAT,
        "",
        *TRUSTING_RECALL_SECTION,
        "",
        "## Memory and other forms of persistence",
        "Memory is one of several persistence mechanisms available to you as you assist the user in a given conversation. The distinction is often that memory can be recalled in future conversations and should not be used for persisting information that is only useful within the scope of the current conversation.",
        "- When to use or update a plan instead of memory: If you are about to start a non-trivial implementation task and would like to reach alignment with the user on your approach you should use a Plan rather than saving this information to memory. Similarly, if you already have a plan within the conversation and you have changed your approach persist that change by updating the plan rather than saving a memory.",
        "- When to use or update tasks instead of memory: When you need to break your work in current conversation into discrete steps or keep track of your progress use tasks instead of saving to memory. Tasks are great for persisting information about the work that needs to be done in the current conversation, but memory should be reserved for information that will be useful in future conversations.",
        *(extra_guidelines or ()),
        "",
        *build_searching_past_context_section(
            auto_dir,
            enabled=include_searching_past_context,
            project_transcript_dir=project_transcript_dir,
        ),
    ]
    return "\n".join(lines)


def load_combined_memory_prompt(
    settings: MemorySettings,
    *,
    extra_guidelines: Sequence[str] | None = None,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> str | None:
    """Load the combined private/team mechanics prompt when team memory is enabled."""
    if not is_team_memory_enabled(settings):
        return None

    ensure_memory_dir_exists(get_auto_mem_path(settings))
    ensure_memory_dir_exists(get_team_mem_path(settings))
    return build_combined_memory_prompt(
        settings,
        extra_guidelines,
        skip_index=skip_index,
        include_searching_past_context=include_searching_past_context,
        project_transcript_dir=project_transcript_dir,
    )


__all__ = [
    "build_combined_memory_prompt",
    "load_combined_memory_prompt",
]
