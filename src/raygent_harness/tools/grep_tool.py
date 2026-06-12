"""Concrete model-callable Grep tool."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionResult,
    ToolPermissionContext,
)
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
from raygent_harness.tools.file_permissions import (
    check_read_permission_for_path,
    expand_file_path,
)
from raygent_harness.tools.search_backend import (
    DEFAULT_GREP_HEAD_LIMIT,
    GREP_OUTPUT_MODES,
    GrepOutputMode,
    GrepSearchRequest,
    GrepSearchResult,
    SearchBackend,
    SearchBackendError,
    SearchTimeoutError,
    create_default_search_backend,
    expand_glob_patterns,
    supported_type_names,
)

GREP_TOOL_NAME = "Grep"
GREP_MAX_RESULT_SIZE_CHARS = 20_000

GREP_PROMPT = """A powerful local search tool for file contents.

Usage:
- Use Grep for search tasks instead of invoking grep or rg through Bash.
- Supports regular expressions such as "log.*Error" or "function\\s+\\w+".
- Filter files with glob (for example "*.py" or "*.{ts,tsx}") or type.
- Output modes: "files_with_matches" (default), "content", and "count".
- For open-ended searches requiring multiple rounds, use Agent when available."""


class GrepInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pattern: str = Field(description="The regular expression pattern to search for.")
    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to current working directory.",
    )
    glob: str | None = Field(
        default=None,
        description='Glob pattern to filter files, for example "*.py" or "*.{ts,tsx}".',
    )
    output_mode: GrepOutputMode = Field(
        default="files_with_matches",
        description='One of "content", "files_with_matches", or "count".',
    )
    context_before: int | None = Field(default=None, alias="-B", ge=0)
    context_after: int | None = Field(default=None, alias="-A", ge=0)
    context_around: int | None = Field(default=None, alias="-C", ge=0)
    context: int | None = Field(default=None, ge=0)
    show_line_numbers: bool = Field(default=True, alias="-n")
    case_insensitive: bool = Field(default=False, alias="-i")
    type: str | None = Field(default=None, description="Common file type filter.")
    head_limit: int | None = Field(default=DEFAULT_GREP_HEAD_LIMIT, ge=0)
    offset: int = Field(default=0, ge=0)
    multiline: bool = Field(default=False)


def build_grep_tool(
    *,
    backend: SearchBackend | None = None,
) -> Tool:
    """Build the concrete `Grep` tool."""

    search_backend = backend or create_default_search_backend()

    async def validate_input(input_: BaseModel, ctx: ToolUseContext) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.pattern.strip():
            return ValidationError(message="pattern is required for Grep")
        try:
            re.compile(parsed.pattern)
        except re.error as exc:
            return ValidationError(message=f"Invalid regular expression: {exc}")
        if parsed.output_mode not in GREP_OUTPUT_MODES:
            return ValidationError(
                message="output_mode must be content, files_with_matches, or count"
            )
        if _invalid_optional_path_value(parsed.path):
            return ValidationError(message="path must be omitted or a real file/directory path")
        if parsed.type is not None and parsed.type not in supported_type_names():
            return ValidationError(
                message=(
                    f"Unsupported Grep type '{parsed.type}'. Supported types: "
                    + ", ".join(supported_type_names())
                )
            )

        root = _resolve_grep_root(parsed, ctx.cwd)
        if _is_unc_like_path(str(root)):
            return ValidationOk()
        if not root.exists():
            return ValidationError(message=f"Path does not exist: {parsed.path or str(root)}")
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        root = _resolve_grep_root(parsed, ctx.cwd)
        return check_read_permission_for_path(
            str(root),
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(exclude_none=True, by_alias=True),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        root = _resolve_grep_root(parsed, ctx.cwd)
        before, after = _context_window(parsed)

        try:
            result = await search_backend.grep(
                GrepSearchRequest(
                    pattern=parsed.pattern,
                    root=root,
                    glob_patterns=expand_glob_patterns(parsed.glob),
                    output_mode=parsed.output_mode,
                    case_insensitive=parsed.case_insensitive,
                    show_line_numbers=parsed.show_line_numbers,
                    context_before=before,
                    context_after=after,
                    head_limit=parsed.head_limit,
                    offset=parsed.offset,
                    multiline=parsed.multiline,
                    type_name=parsed.type,
                    abort_event=ctx.abort_event,
                    is_path_allowed=lambda path: _read_allowed(path, ctx),
                )
            )
        except asyncio.CancelledError:
            raise
        except SearchTimeoutError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        except SearchBackendError as exc:
            yield ToolResult(content=f"Grep failed: {exc}", is_error=True)
            return
        except OSError as exc:
            yield ToolResult(content=f"Grep failed for '{root}': {exc}", is_error=True)
            return

        yield ToolResult(
            content=_format_grep_result(
                result,
                ctx.cwd,
                parsed.output_mode,
                show_line_numbers=parsed.show_line_numbers,
            )
        )

    return build_tool(
        ToolSpec(
            name=GREP_TOOL_NAME,
            description="Search file contents with regex.",
            search_hint="search file contents with regex",
            input_model=GrepInput,
            call=call,
            prompt=GREP_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            max_result_size_chars=GREP_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Searching for {_coerce_input(input_).pattern}"
            ),
        )
    )


def _format_grep_result(
    result: GrepSearchResult,
    cwd: str,
    requested_mode: GrepOutputMode,
    *,
    show_line_numbers: bool,
) -> str:
    if requested_mode == "content":
        return _cap_content(
            _format_content_result(result, cwd, show_line_numbers=show_line_numbers),
            GREP_MAX_RESULT_SIZE_CHARS,
        )
    if requested_mode == "count":
        return _cap_content(_format_count_result(result, cwd), GREP_MAX_RESULT_SIZE_CHARS)
    return _cap_content(_format_files_result(result, cwd), GREP_MAX_RESULT_SIZE_CHARS)


def _format_files_result(result: GrepSearchResult, cwd: str) -> str:
    if not result.files:
        content = "No files found"
    else:
        limit_info = _format_limit_info(result.applied_limit, result.applied_offset)
        paths = "\n".join(_render_path(item.path, cwd) for item in result.files)
        content = (
            f"Found {len(result.files)} {_plural(len(result.files), 'file')}"
            f"{limit_info}\n{paths}"
        )
    return _append_partial_note(content, result.partial)


def _format_count_result(result: GrepSearchResult, cwd: str) -> str:
    if not result.count_lines:
        content = "No matches found"
    else:
        lines = "\n".join(
            f"{_render_path(path, cwd)}:{count}" for path, count in result.count_lines
        )
        matches = sum(count for _path, count in result.count_lines)
        files = len(result.count_lines)
        limit_info = _format_limit_info(result.applied_limit, result.applied_offset)
        content = (
            f"{lines}\n\nFound {matches} total {_plural(matches, 'occurrence')} "
            f"across {files} {_plural(files, 'file')}.{limit_info}"
        )
    return _append_partial_note(content, result.partial)


def _format_content_result(
    result: GrepSearchResult,
    cwd: str,
    *,
    show_line_numbers: bool,
) -> str:
    if not result.content_lines:
        content = "No matches found"
    else:
        lines: list[str] = []
        for item in result.content_lines:
            rendered_path = _render_path(item.path, cwd)
            if not show_line_numbers:
                lines.append(f"{rendered_path}:{item.text}")
            elif item.line_number and item.is_match:
                lines.append(f"{rendered_path}:{item.line_number}:{item.text}")
            elif item.line_number:
                lines.append(f"{rendered_path}-{item.line_number}-{item.text}")
            else:
                lines.append(f"{rendered_path}:{item.text}")
        content = "\n".join(lines)

    limit_info = _format_limit_info(result.applied_limit, result.applied_offset)
    if limit_info:
        content += f"\n\n[Showing results with pagination ={limit_info}]"
    return _append_partial_note(content, result.partial)


def _context_window(input_: GrepInput) -> tuple[int, int]:
    context = input_.context
    if context is None:
        context = input_.context_around
    if context is not None:
        return context, context
    return input_.context_before or 0, input_.context_after or 0


def _resolve_grep_root(input_: GrepInput, cwd: str) -> Path:
    return Path(expand_file_path(input_.path or ".", cwd=cwd))


def _read_allowed(path: Path, ctx: ToolUseContext) -> bool:
    decision = check_read_permission_for_path(
        str(path),
        ctx.permission_context,
        cwd=ctx.cwd,
        input={"file_path": str(path)},
    )
    return isinstance(decision, PermissionAllowDecision)


def _coerce_input(input_: BaseModel) -> GrepInput:
    if isinstance(input_, GrepInput):
        return input_
    return GrepInput.model_validate(input_.model_dump())


def _render_path(path: Path, cwd: str) -> str:
    try:
        return os.path.relpath(path, cwd)
    except ValueError:
        return str(path)


def _format_limit_info(limit: int | None, offset: int | None) -> str:
    parts: list[str] = []
    if limit is not None:
        parts.append(f"limit: {limit}")
    if offset:
        parts.append(f"offset: {offset}")
    return f" ({', '.join(parts)})" if parts else ""


def _append_partial_note(content: str, partial: bool) -> str:
    if not partial:
        return content
    return content + "\n\n(Some paths were inaccessible, binary, too large, or skipped.)"


def _cap_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Output truncated to fit tool result bounds.]"


def _plural(count: int, singular: str) -> str:
    return singular if count == 1 else singular + "s"


def _invalid_optional_path_value(path: str | None) -> bool:
    return path is not None and path.strip().lower() in {"", "undefined", "null"}


def _is_unc_like_path(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


__all__ = [
    "GREP_MAX_RESULT_SIZE_CHARS",
    "GREP_PROMPT",
    "GREP_TOOL_NAME",
    "GrepInput",
    "build_grep_tool",
]
