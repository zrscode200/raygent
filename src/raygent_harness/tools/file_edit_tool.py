"""Concrete model-callable exact string editor.

"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
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
    FILE_EDIT_TOOL_NAME,
    check_write_permission_for_path,
    expand_file_path,
    matching_file_permission_rule,
)
from raygent_harness.tools.file_text_utils import (
    FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    apply_line_ending,
    assert_not_stale,
    content_matches_state,
    has_full_file_state,
    mtime_ms,
    normalize_line_endings,
    read_text_snapshot,
    write_text,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


FILE_EDIT_TOOL_ALIAS = "FileEdit"
FILE_EDIT_MAX_RESULT_SIZE_CHARS = 100_000
MAX_EDIT_FILE_SIZE_BYTES = 1024 * 1024 * 1024

FILE_EDIT_PROMPT = (
    "Performs exact string replacements in files.\n\n"
    "Usage:\n"
    "- You must use your Read tool at least once in the conversation before editing. "
    "This tool will error if you attempt an edit without reading the file.\n"
    "- When editing text from Read tool output, preserve the exact indentation after "
    "the line number prefix. Never include any part of the line number prefix in "
    "old_string or new_string.\n"
    "- ALWAYS prefer editing existing files in the codebase. NEVER write new files "
    "unless explicitly required.\n"
    "- Only use emojis if the user explicitly requests it. Avoid adding emojis to "
    "files unless asked.\n"
    "- The edit will FAIL if old_string is not unique in the file. Either provide "
    "more surrounding context to make it unique or use replace_all to change every "
    "instance of old_string.\n"
    "- Use replace_all for replacing and renaming strings across the file."
)

_NOTEBOOK_EXTENSIONS = frozenset({"ipynb"})
LEFT_SINGLE_CURLY_QUOTE = "\u2018"
RIGHT_SINGLE_CURLY_QUOTE = "\u2019"
LEFT_DOUBLE_CURLY_QUOTE = "\u201c"
RIGHT_DOUBLE_CURLY_QUOTE = "\u201d"


class EditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to modify.")
    old_string: str = Field(description="The text to replace.")
    new_string: str = Field(description="The text to replace it with.")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string. Defaults to false.",
    )


@dataclass(frozen=True, slots=True)
class EditApplication:
    updated_content: str
    actual_old_string: str
    actual_new_string: str
    replacement_count: int


class FileEditToolError(Exception):
    """Model-visible operational edit error."""


def build_file_edit_tool(*, memory_settings: MemorySettings | None = None) -> Tool:
    """Build the concrete text `Edit` tool."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.file_path.strip():
            return ValidationError(message="file_path is required for Edit")
        if parsed.old_string == parsed.new_string:
            return ValidationError(
                message="No changes to make: old_string and new_string are exactly the same."
            )

        full_path = expand_file_path(parsed.file_path, cwd=ctx.cwd)
        if memory_settings is not None:
            secret_error = check_team_mem_secrets(full_path, parsed.new_string, memory_settings)
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

        if _is_notebook_path(full_path):
            return ValidationError(
                message=(
                    "File is a Jupyter Notebook. Use NotebookEdit to edit this file."
                )
            )

        if _should_skip_filesystem_validation(full_path):
            return ValidationOk()

        try:
            size = await asyncio.to_thread(_file_size_or_none, full_path)
        except OSError as exc:
            return ValidationError(message=f"Cannot edit '{parsed.file_path}': {exc}")
        if size is not None and size > MAX_EDIT_FILE_SIZE_BYTES:
            return ValidationError(
                message=(
                    f"File is too large to edit ({size} bytes). Maximum editable "
                    f"file size is {MAX_EDIT_FILE_SIZE_BYTES} bytes."
                )
            )

        try:
            snapshot = await asyncio.to_thread(read_text_snapshot, full_path)
        except IsADirectoryError:
            return ValidationError(
                message=f"Cannot edit '{parsed.file_path}': path is a directory."
            )
        except UnicodeError as exc:
            return ValidationError(message=f"Cannot edit '{parsed.file_path}' as text: {exc}")
        except OSError as exc:
            return ValidationError(message=f"Cannot edit '{parsed.file_path}': {exc}")

        if not snapshot.exists:
            if parsed.old_string == "":
                return ValidationOk()
            return ValidationError(
                message=(
                    "File does not exist. If you want to create a new file, use "
                    "old_string as an empty string."
                )
            )

        if parsed.old_string == "":
            if snapshot.normalized_content.strip() != "":
                return ValidationError(message="Cannot create new file - file already exists.")
            return ValidationOk()

        read_state = ctx.read_file_state.get(full_path)
        if not has_full_file_state(read_state):
            return ValidationError(
                message="File has not been read yet. Read it first before writing to it."
            )
        if (
            read_state is not None
            and snapshot.mtime_ms is not None
            and snapshot.mtime_ms > read_state.timestamp
            and not content_matches_state(snapshot.content, read_state)
        ):
            return ValidationError(message=FILE_UNEXPECTEDLY_MODIFIED_ERROR)

        try:
            _preview_edit(snapshot.normalized_content, parsed)
        except FileEditToolError as exc:
            return ValidationError(message=str(exc))

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
            result = await asyncio.to_thread(_edit_file, parsed, full_path, ctx)
        except FileEditToolError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        except OSError as exc:
            yield ToolResult(content=f"Cannot edit '{parsed.file_path}': {exc}", is_error=True)
            return
        except UnicodeError as exc:
            yield ToolResult(content=f"Cannot edit '{parsed.file_path}': {exc}", is_error=True)
            return

        yield ToolResult(content=result)

    return build_tool(
        ToolSpec(
            name=FILE_EDIT_TOOL_NAME,
            aliases=(FILE_EDIT_TOOL_ALIAS,),
            description="A tool for editing files.",
            search_hint="modify file contents in place",
            input_model=EditInput,
            call=call,
            prompt=FILE_EDIT_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=True,
            is_open_world=False,
            should_defer=False,
            always_load=False,
            max_result_size_chars=FILE_EDIT_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: f"Editing {_coerce_input(input_).file_path}",
        )
    )


def create_file_edit_catalog_provider(
    *,
    enabled: bool = True,
    memory_settings: MemorySettings | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends `Edit` when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != FILE_EDIT_TOOL_NAME)
        if not enabled:
            return without_existing
        return (*without_existing, build_file_edit_tool(memory_settings=memory_settings))

    return provider


def apply_edit_to_content(
    content: str,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditApplication:
    normalized_content = normalize_line_endings(content)
    parsed = EditInput(
        file_path="/dev/null",
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )
    return _preview_edit(normalized_content, parsed)


def find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string

    normalized_search = normalize_quotes(search_string)
    normalized_file = normalize_quotes(file_content)
    index = normalized_file.find(normalized_search)
    if index == -1:
        return None
    return file_content[index : index + len(search_string)]


def normalize_quotes(value: str) -> str:
    return (
        value.replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, '"')
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, '"')
    )


def preserve_quote_style(old_string: str, actual_old_string: str, new_string: str) -> str:
    if old_string == actual_old_string:
        return new_string

    result = new_string
    if (
        LEFT_DOUBLE_CURLY_QUOTE in actual_old_string
        or RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string
    ):
        result = _apply_curly_double_quotes(result)
    if (
        LEFT_SINGLE_CURLY_QUOTE in actual_old_string
        or RIGHT_SINGLE_CURLY_QUOTE in actual_old_string
    ):
        result = _apply_curly_single_quotes(result)
    return result


def _edit_file(input_: EditInput, full_path: str, ctx: ToolUseContext) -> str:
    if ctx.abort_event.is_set():
        raise asyncio.CancelledError()

    snapshot = read_text_snapshot(full_path)
    if snapshot.exists:
        state = ctx.read_file_state.get(full_path)
        try:
            assert_not_stale(path=full_path, snapshot=snapshot, state=state)
        except RuntimeError as exc:
            raise FileEditToolError(str(exc)) from exc

    application = _preview_edit(snapshot.normalized_content, input_)
    output_content = apply_line_ending(application.updated_content, snapshot.line_ending)

    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    write_text(full_path, output_content, encoding=snapshot.encoding)
    ctx.read_file_state.set(
        full_path,
        FileState(
            content=application.updated_content,
            timestamp=mtime_ms(full_path),
            offset=None,
            limit=None,
        ),
    )

    if input_.replace_all:
        return (
            f"The file {input_.file_path} has been updated. "
            "All occurrences were successfully replaced."
        )
    return f"The file {input_.file_path} has been updated successfully."


def _preview_edit(content: str, input_: EditInput) -> EditApplication:
    if input_.old_string == "":
        return EditApplication(
            updated_content=input_.new_string,
            actual_old_string="",
            actual_new_string=input_.new_string,
            replacement_count=1,
        )

    actual_old = find_actual_string(content, input_.old_string)
    if actual_old is None:
        raise FileEditToolError(
            f"String to replace not found in file.\nString: {input_.old_string}"
        )

    matches = content.count(actual_old)
    if matches > 1 and not input_.replace_all:
        raise FileEditToolError(
            f"Found {matches} matches of the string to replace, but replace_all is false. "
            "To replace all occurrences, set replace_all to true. To replace only one "
            "occurrence, please provide more context to uniquely identify the instance.\n"
            f"String: {input_.old_string}"
        )

    actual_new = preserve_quote_style(input_.old_string, actual_old, input_.new_string)
    updated = _replace_text(
        content,
        actual_old,
        actual_new,
        replace_all=input_.replace_all,
    )
    if updated == content:
        raise FileEditToolError("Original and edited file match exactly. Failed to apply edit.")

    return EditApplication(
        updated_content=updated,
        actual_old_string=actual_old,
        actual_new_string=actual_new,
        replacement_count=matches if input_.replace_all else 1,
    )


def _replace_text(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
) -> str:
    if new_string == "" and not old_string.endswith("\n") and old_string + "\n" in content:
        old_string = old_string + "\n"
    count = -1 if replace_all else 1
    return content.replace(old_string, new_string, count)


def _coerce_input(input_: BaseModel) -> EditInput:
    if isinstance(input_, EditInput):
        return input_
    return EditInput.model_validate(input_.model_dump())


def _file_size_or_none(path: str) -> int | None:
    try:
        stats = os.stat(path)
    except FileNotFoundError:
        return None
    return stats.st_size


def _is_notebook_path(path: str) -> bool:
    return Path(path).suffix.lower().lstrip(".") in _NOTEBOOK_EXTENSIONS


def _should_skip_filesystem_validation(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


def _is_opening_context(chars: list[str], index: int) -> bool:
    if index == 0:
        return True
    return chars[index - 1] in {" ", "\t", "\n", "\r", "(", "[", "{", "\u2014", "\u2013"}


def _apply_curly_double_quotes(value: str) -> str:
    chars = list(value)
    output: list[str] = []
    for index, char in enumerate(chars):
        if char == '"':
            output.append(
                LEFT_DOUBLE_CURLY_QUOTE
                if _is_opening_context(chars, index)
                else RIGHT_DOUBLE_CURLY_QUOTE
            )
        else:
            output.append(char)
    return "".join(output)


def _apply_curly_single_quotes(value: str) -> str:
    chars = list(value)
    output: list[str] = []
    for index, char in enumerate(chars):
        if char != "'":
            output.append(char)
            continue
        prev = chars[index - 1] if index > 0 else ""
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if prev.isalpha() and next_char.isalpha():
            output.append(RIGHT_SINGLE_CURLY_QUOTE)
        else:
            output.append(
                LEFT_SINGLE_CURLY_QUOTE
                if _is_opening_context(chars, index)
                else RIGHT_SINGLE_CURLY_QUOTE
            )
    return "".join(output)


__all__ = [
    "FILE_EDIT_MAX_RESULT_SIZE_CHARS",
    "FILE_EDIT_PROMPT",
    "FILE_EDIT_TOOL_ALIAS",
    "FILE_EDIT_TOOL_NAME",
    "FILE_UNEXPECTEDLY_MODIFIED_ERROR",
    "EditApplication",
    "EditInput",
    "FileEditToolError",
    "apply_edit_to_content",
    "build_file_edit_tool",
    "create_file_edit_catalog_provider",
    "find_actual_string",
    "normalize_quotes",
    "preserve_quote_style",
]
