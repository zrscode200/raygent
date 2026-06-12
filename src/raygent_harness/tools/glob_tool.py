"""Concrete model-callable Glob tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field

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
    DEFAULT_GLOB_LIMIT,
    GlobSearchRequest,
    SearchBackend,
    create_default_search_backend,
    extract_glob_base_directory,
)

GLOB_TOOL_NAME = "Glob"
GLOB_MAX_RESULT_SIZE_CHARS = 100_000

GLOB_PROMPT = """Fast file pattern matching tool that works with any codebase size.

- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- For open-ended code exploration that requires multiple rounds of globbing and
  grepping, use Agent when available"""


class GlobInput(BaseModel):
    pattern: str = Field(description="The glob pattern to match files against.")
    path: str | None = Field(
        default=None,
        description=(
            "The directory to search in. If omitted, the current working directory "
            "is used. Must be a valid directory path if provided."
        ),
    )


def build_glob_tool(
    *,
    backend: SearchBackend | None = None,
    max_results: int = DEFAULT_GLOB_LIMIT,
) -> Tool:
    """Build the concrete `Glob` tool."""

    search_backend = backend or create_default_search_backend()

    async def validate_input(input_: BaseModel, ctx: ToolUseContext) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.pattern.strip():
            return ValidationError(message="pattern is required for Glob")
        if _invalid_optional_path_value(parsed.path):
            return ValidationError(message="path must be omitted or a real directory path")

        root, _pattern = _resolve_glob_search(parsed, ctx.cwd)
        if _is_unc_like_path(str(root)):
            return ValidationOk()
        if not root.exists():
            return ValidationError(
                message=f"Directory does not exist: {parsed.path or str(root)}"
            )
        if not root.is_dir():
            return ValidationError(message=f"Path is not a directory: {parsed.path}")
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        root, _pattern = _resolve_glob_search(parsed, ctx.cwd)
        return check_read_permission_for_path(
            str(root),
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(exclude_none=True),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        root, pattern = _resolve_glob_search(parsed, ctx.cwd)

        try:
            result = await search_backend.glob(
                GlobSearchRequest(
                    pattern=pattern,
                    root=root,
                    limit=max_results,
                    abort_event=ctx.abort_event,
                    is_path_allowed=lambda path: _read_allowed(path, ctx),
                )
            )
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            yield ToolResult(content=f"Glob failed for '{root}': {exc}", is_error=True)
            return

        paths = tuple(_render_path(item.path, ctx.cwd) for item in result.files)
        if not paths:
            yield ToolResult(content="No files found")
            return

        content = "\n".join(
            (
                *paths,
                *(
                    (
                        "(Results are truncated. Consider using a more specific "
                        "path or pattern.)",
                    )
                    if result.truncated
                    else ()
                ),
            )
        )
        yield ToolResult(content=_cap_content(content, GLOB_MAX_RESULT_SIZE_CHARS))

    return build_tool(
        ToolSpec(
            name=GLOB_TOOL_NAME,
            description="Find files by name pattern or wildcard.",
            search_hint="find files by name pattern or wildcard",
            input_model=GlobInput,
            call=call,
            prompt=GLOB_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            max_result_size_chars=GLOB_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Finding {_coerce_input(input_).pattern}"
            ),
        )
    )


def _resolve_glob_search(input_: GlobInput, cwd: str) -> tuple[Path, str]:
    if input_.path:
        return Path(expand_file_path(input_.path, cwd=cwd)), input_.pattern

    if os.path.isabs(input_.pattern):
        base_dir, relative_pattern = extract_glob_base_directory(input_.pattern)
        if base_dir:
            return Path(expand_file_path(base_dir, cwd=cwd)), relative_pattern
    return Path(expand_file_path(".", cwd=cwd)), input_.pattern


def _read_allowed(path: Path, ctx: ToolUseContext) -> bool:
    decision = check_read_permission_for_path(
        str(path),
        ctx.permission_context,
        cwd=ctx.cwd,
        input={"file_path": str(path)},
    )
    return isinstance(decision, PermissionAllowDecision)


def _coerce_input(input_: BaseModel) -> GlobInput:
    if isinstance(input_, GlobInput):
        return input_
    return GlobInput.model_validate(input_.model_dump())


def _render_path(path: Path, cwd: str) -> str:
    try:
        return os.path.relpath(path, cwd)
    except ValueError:
        return str(path)


def _invalid_optional_path_value(path: str | None) -> bool:
    return path is not None and path.strip().lower() in {"", "undefined", "null"}


def _is_unc_like_path(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


def _cap_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Output truncated to fit tool result bounds.]"


__all__ = [
    "GLOB_MAX_RESULT_SIZE_CHARS",
    "GLOB_PROMPT",
    "GLOB_TOOL_NAME",
    "GlobInput",
    "build_glob_tool",
]
