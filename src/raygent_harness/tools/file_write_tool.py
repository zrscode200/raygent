"""Concrete model-callable full file writer.

"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from raygent_harness.core.file_state import FileState
from raygent_harness.core.permissions import PermissionResult
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.services.team_memory_sync.secret_guard import check_team_mem_secrets
from raygent_harness.tools.file_permissions import (
    FILE_WRITE_TOOL_NAME,
    check_write_permission_for_path,
    expand_file_path,
    matching_file_permission_rule,
)
from raygent_harness.tools.file_text_utils import (
    FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    assert_not_stale,
    has_full_file_state,
    mtime_ms,
    read_text_snapshot,
    write_text,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


FILE_WRITE_TOOL_ALIAS = "FileWrite"
FILE_WRITE_MAX_RESULT_SIZE_CHARS = 100_000

FILE_WRITE_PROMPT = (
    "Writes a file to the local filesystem.\n\n"
    "Usage:\n"
    "- This tool will overwrite the existing file if there is one at the provided path.\n"
    "- If this is an existing file, you MUST use the Read tool first to read the "
    "file's contents. This tool will fail if you did not read the file first.\n"
    "- Prefer the Edit tool for modifying existing files; it only sends the diff. "
    "Only use this tool to create new files or for complete rewrites.\n"
    "- NEVER create documentation files (*.md) or README files unless explicitly "
    "requested by the user.\n"
    "- Only use emojis if the user explicitly requests it. Avoid writing emojis to "
    "files unless asked."
)


class WriteInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to write.")
    content: str = Field(description="The content to write to the file.")


class FileWriteToolError(Exception):
    """Model-visible operational write error."""


def build_file_write_tool(*, memory_settings: MemorySettings | None = None) -> Tool:
    """Build the concrete text `Write` tool."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.file_path.strip():
            return ValidationError(message="file_path is required for Write")

        full_path = expand_file_path(parsed.file_path, cwd=ctx.cwd)
        if memory_settings is not None:
            secret_error = check_team_mem_secrets(full_path, parsed.content, memory_settings)
            if secret_error is not None:
                return ValidationError(message=secret_error)

        if matching_file_permission_rule(
            full_path,
            ctx.permission_context,
            tool_type="edit",
            behavior="deny",
            cwd=ctx.cwd,
        ) is not None:
            return ValidationError(
                message="File is in a directory that is denied by your permission settings."
            )

        if _should_skip_filesystem_validation(full_path):
            return ValidationOk()

        try:
            stats = await asyncio.to_thread(os.stat, full_path)
        except FileNotFoundError:
            return ValidationOk()
        except OSError as exc:
            return ValidationError(message=f"Cannot write '{parsed.file_path}': {exc}")

        if _is_directory_mode(stats.st_mode):
            return ValidationError(
                message=f"Cannot write '{parsed.file_path}': path is a directory."
            )

        last_read = ctx.read_file_state.get(full_path)
        if not has_full_file_state(last_read):
            return ValidationError(
                message="File has not been read yet. Read it first before writing to it."
            )

        last_write_time = int(stats.st_mtime_ns // 1_000_000)
        if last_read is not None and last_write_time > last_read.timestamp:
            return ValidationError(message=FILE_UNEXPECTEDLY_MODIFIED_ERROR)

        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        return check_write_permission_for_path(
            parsed.file_path,
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        full_path = expand_file_path(parsed.file_path, cwd=ctx.cwd)

        if ctx.abort_event.is_set():
            raise asyncio.CancelledError()

        try:
            result = await asyncio.to_thread(_write_file, parsed, full_path, ctx)
        except FileWriteToolError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        except OSError as exc:
            yield ToolResult(content=f"Cannot write '{parsed.file_path}': {exc}", is_error=True)
            return
        except UnicodeError as exc:
            yield ToolResult(content=f"Cannot write '{parsed.file_path}': {exc}", is_error=True)
            return

        yield ToolResult(content=result)

    return build_tool(
        ToolSpec(
            name=FILE_WRITE_TOOL_NAME,
            aliases=(FILE_WRITE_TOOL_ALIAS,),
            description="Write a file to the local filesystem.",
            search_hint="create or overwrite files",
            input_model=WriteInput,
            call=call,
            prompt=FILE_WRITE_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=True,
            is_open_world=False,
            should_defer=False,
            always_load=False,
            max_result_size_chars=FILE_WRITE_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Writing {_coerce_input(input_).file_path}"
            ),
        )
    )


def create_file_write_catalog_provider(
    *,
    enabled: bool = True,
    memory_settings: MemorySettings | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends `Write` when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != FILE_WRITE_TOOL_NAME)
        if not enabled:
            return without_existing
        return (*without_existing, build_file_write_tool(memory_settings=memory_settings))

    return provider


def _write_file(input_: WriteInput, full_path: str, ctx: ToolUseContext) -> str:
    if ctx.abort_event.is_set():
        raise asyncio.CancelledError()

    snapshot = read_text_snapshot(full_path)
    if snapshot.exists:
        last_read = ctx.read_file_state.get(full_path)
        try:
            assert_not_stale(path=full_path, snapshot=snapshot, state=last_read)
        except RuntimeError as exc:
            raise FileWriteToolError(str(exc)) from exc

    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    encoding = snapshot.encoding if snapshot.exists else "utf-8"
    write_text(full_path, input_.content, encoding=encoding)
    ctx.read_file_state.set(
        full_path,
        FileState(
            content=input_.content,
            timestamp=mtime_ms(full_path),
            offset=None,
            limit=None,
        ),
    )

    if snapshot.exists:
        return f"The file {input_.file_path} has been updated successfully."
    return f"File created successfully at: {input_.file_path}"


def _coerce_input(input_: BaseModel) -> WriteInput:
    if isinstance(input_, WriteInput):
        return input_
    return WriteInput.model_validate(input_.model_dump())


def _is_directory_mode(mode: int) -> bool:
    return (mode & 0o170000) == 0o040000


def _should_skip_filesystem_validation(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


__all__ = [
    "FILE_UNEXPECTEDLY_MODIFIED_ERROR",
    "FILE_WRITE_MAX_RESULT_SIZE_CHARS",
    "FILE_WRITE_PROMPT",
    "FILE_WRITE_TOOL_ALIAS",
    "FILE_WRITE_TOOL_NAME",
    "FileWriteToolError",
    "WriteInput",
    "build_file_write_tool",
    "create_file_write_catalog_provider",
]
