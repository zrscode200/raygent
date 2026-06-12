"""Tool-hook adapters for team-memory safety and sync notification.

Team-memory file-tool behavior:
  that contain secrets before the file is changed.
  successful Edit/Write access to a team-memory file.

Raygent-owned concrete `Write`/`Edit` tools perform direct secret blocking when
memory settings are injected. This module remains the reusable hook seam for
adapter-owned write tools and for the file-tool runtime helper that wires
successful-write sync notifications onto `QueryDeps`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.tool_hooks import (
    PostToolUseContext,
    PostToolUseHook,
    PreToolUseContext,
    PreToolUseHook,
    PreToolUseResult,
)
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import is_team_mem_file
from raygent_harness.services.team_memory_sync.secret_guard import check_team_mem_secrets

DEFAULT_TEAM_MEMORY_WRITE_TOOL_NAMES = frozenset(
    {
        "Write",
        "Edit",
        "FileWrite",
        "FileEdit",
        "file_write",
        "file_edit",
    }
)

TeamMemoryWriteNotifier = Callable[[], Awaitable[object]]


def create_team_memory_pre_tool_use_hook(
    settings: MemorySettings,
    *,
    tool_names: Collection[str] | None = DEFAULT_TEAM_MEMORY_WRITE_TOOL_NAMES,
) -> PreToolUseHook:
    """Return a PreToolUse hook that blocks secrets in team-memory writes."""

    async def hook(context: PreToolUseContext, /) -> PreToolUseResult | None:
        payload = _extract_write_payload(context.input, context.tool_use_context.cwd)
        if payload is None or not _matches_tool(context.tool.name, tool_names):
            return None

        file_path, content = payload
        error = check_team_mem_secrets(file_path, content, settings)
        if error is None:
            return None
        return PreToolUseResult(stop=True, stop_reason=error)

    return hook


def create_team_memory_post_tool_use_hook(
    settings: MemorySettings,
    notify_team_memory_write: TeamMemoryWriteNotifier,
    *,
    tool_names: Collection[str] | None = DEFAULT_TEAM_MEMORY_WRITE_TOOL_NAMES,
) -> PostToolUseHook:
    """Return a PostToolUse hook that notifies sync after successful writes."""

    async def hook(context: PostToolUseContext, /) -> None:
        payload = _extract_write_payload(context.input, context.tool_use_context.cwd)
        if payload is None or not _matches_tool(context.tool.name, tool_names):
            return
        if not _is_successful_tool_result(context.result_message, context.tool_use.id):
            return

        file_path, _content = payload
        if is_team_mem_file(file_path, settings):
            await notify_team_memory_write()

    return hook


def create_team_memory_write_hooks(
    settings: MemorySettings,
    notify_team_memory_write: TeamMemoryWriteNotifier,
    *,
    tool_names: Collection[str] | None = DEFAULT_TEAM_MEMORY_WRITE_TOOL_NAMES,
) -> tuple[PreToolUseHook, PostToolUseHook]:
    """Return the safety+notification hook pair for file-write adapters."""
    return (
        create_team_memory_pre_tool_use_hook(settings, tool_names=tool_names),
        create_team_memory_post_tool_use_hook(
            settings,
            notify_team_memory_write,
            tool_names=tool_names,
        ),
    )


def _matches_tool(tool_name: str, allowed: Collection[str] | None) -> bool:
    return allowed is None or tool_name in allowed


def _extract_write_payload(input_model: BaseModel, cwd: str) -> tuple[Path, str] | None:
    data = input_model.model_dump()
    raw_path = _string_field(data, "file_path")
    if raw_path is None:
        return None

    content = _string_field(data, "content")
    if content is None:
        content = _string_field(data, "new_string")
    if content is None:
        return None

    return (_expand_input_path(raw_path, cwd), content)


def _string_field(data: Mapping[str, Any], name: str) -> str | None:
    value = data.get(name)
    return value if isinstance(value, str) else None


def _expand_input_path(raw_path: str, cwd: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return Path(cwd) / path


def _is_successful_tool_result(message: MessageParam, tool_use_id: str) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for block in content:
        if block.get("type") != "tool_result" or block.get("tool_use_id") != tool_use_id:
            continue
        return block.get("is_error") is not True

    return False


__all__ = [
    "DEFAULT_TEAM_MEMORY_WRITE_TOOL_NAMES",
    "TeamMemoryWriteNotifier",
    "create_team_memory_post_tool_use_hook",
    "create_team_memory_pre_tool_use_hook",
    "create_team_memory_write_hooks",
]
