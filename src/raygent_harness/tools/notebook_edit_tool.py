"""Concrete model-callable Jupyter notebook cell editor.

"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import stat
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
from raygent_harness.services.file_media import NOTEBOOK_MAX_RAW_BYTES
from raygent_harness.services.team_memory_sync.secret_guard import check_team_mem_secrets
from raygent_harness.tools.file_permissions import (
    check_write_permission_for_path,
    expand_file_path,
    matching_file_permission_rule,
)
from raygent_harness.tools.file_text_utils import (
    FILE_UNEXPECTEDLY_MODIFIED_ERROR,
    has_full_file_state,
    mtime_ms,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


NOTEBOOK_EDIT_TOOL_NAME = "NotebookEdit"
NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS = 100_000

NOTEBOOK_EDIT_PROMPT = (
    "Completely replaces, inserts, or deletes cells in a Jupyter notebook "
    "(.ipynb file). Jupyter notebooks are interactive documents that combine "
    "code, text, and visualizations. You must use Read on the notebook before "
    "editing it. Use cell IDs shown by Read, such as a real notebook cell id or "
    "the generated zero-based cell-N identifier. Use edit_mode=insert to add a "
    "new cell after cell_id, or at the beginning when cell_id is omitted. "
    "Use edit_mode=delete to delete a cell. cell_type is required for insert "
    "and may be code or markdown."
)

NotebookEditMode = Literal["replace", "insert", "delete"]
NotebookCellType = Literal["code", "markdown"]


class NotebookEditInput(BaseModel):
    notebook_path: str = Field(
        description="The absolute path to the Jupyter notebook file to edit."
    )
    cell_id: str | None = Field(
        default=None,
        description=(
            "The ID of the cell to edit. For insert, the new cell is inserted "
            "after this cell, or at the beginning if omitted."
        ),
    )
    new_source: str = Field(description="The new source for the cell.")
    cell_type: NotebookCellType | None = Field(
        default=None,
        description=(
            "The cell type, code or markdown. Required for edit_mode=insert; "
            "optional for replace."
        ),
    )
    edit_mode: NotebookEditMode = Field(
        default="replace",
        description="The edit mode: replace, insert, or delete. Defaults to replace.",
    )


@dataclass(frozen=True, slots=True)
class NotebookEditResult:
    message: str
    cell_id: str | None
    edit_mode: NotebookEditMode


@dataclass(frozen=True, slots=True)
class _LoadedNotebook:
    raw_content: str
    notebook: dict[str, Any]
    cells: list[dict[str, Any]]
    mtime_ms: int


class NotebookEditToolError(Exception):
    """Model-visible notebook edit error."""


def build_notebook_edit_tool(*, memory_settings: MemorySettings | None = None) -> Tool:
    """Build the concrete `NotebookEdit` tool."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        full_path = expand_file_path(parsed.notebook_path, cwd=ctx.cwd)
        syntax_error = _validate_notebook_edit_syntax(parsed, full_path, ctx)
        if syntax_error is not None:
            return ValidationError(message=syntax_error)

        if memory_settings is not None:
            secret_error = check_team_mem_secrets(
                full_path,
                parsed.new_source,
                memory_settings,
            )
            if secret_error is not None:
                return ValidationError(message=secret_error)

        if _should_skip_filesystem_validation(full_path):
            return ValidationOk()

        try:
            await asyncio.to_thread(_validate_notebook_edit_target, parsed, full_path, ctx)
        except NotebookEditToolError as exc:
            return ValidationError(message=str(exc))
        except UnicodeError as exc:
            return ValidationError(message=f"Cannot edit notebook as UTF-8 JSON: {exc}")
        except OSError as exc:
            return ValidationError(message=f"Cannot edit '{parsed.notebook_path}': {exc}")
        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        return check_write_permission_for_path(
            parsed.notebook_path,
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(exclude_none=True),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        full_path = expand_file_path(parsed.notebook_path, cwd=ctx.cwd)

        if ctx.abort_event.is_set():
            raise asyncio.CancelledError()

        try:
            result = await asyncio.to_thread(_edit_notebook_file, parsed, full_path, ctx)
        except NotebookEditToolError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        except UnicodeError as exc:
            yield ToolResult(content=f"Cannot edit notebook as UTF-8 JSON: {exc}", is_error=True)
            return
        except OSError as exc:
            yield ToolResult(content=f"Cannot edit '{parsed.notebook_path}': {exc}", is_error=True)
            return

        yield ToolResult(content=result.message)

    return build_tool(
        ToolSpec(
            name=NOTEBOOK_EDIT_TOOL_NAME,
            description="Replace, insert, or delete cells in Jupyter notebooks.",
            search_hint="edit Jupyter notebook cells ipynb",
            input_model=NotebookEditInput,
            call=call,
            prompt=NOTEBOOK_EDIT_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=True,
            is_open_world=False,
            should_defer=True,
            always_load=False,
            max_result_size_chars=NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Editing notebook {_coerce_input(input_).notebook_path}"
            ),
        )
    )


def create_notebook_edit_catalog_provider(
    *,
    enabled: bool = True,
    memory_settings: MemorySettings | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends `NotebookEdit` when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != NOTEBOOK_EDIT_TOOL_NAME)
        if not enabled:
            return without_existing
        return (*without_existing, build_notebook_edit_tool(memory_settings=memory_settings))

    return provider


def _validate_notebook_edit_syntax(
    input_: NotebookEditInput,
    full_path: str,
    ctx: ToolUseContext,
) -> str | None:
    if not input_.notebook_path.strip():
        return "notebook_path is required for NotebookEdit"

    if matching_file_permission_rule(
        full_path,
        ctx.permission_context,
        tool_type="edit",
        behavior="deny",
        cwd=ctx.cwd,
    ) is not None:
        return "File is in a directory that is denied by your permission settings."

    is_notebook = Path(full_path).suffix.lower() == ".ipynb"
    if not _should_skip_filesystem_validation(full_path) and not is_notebook:
        return (
            "File must be a Jupyter notebook (.ipynb file). For editing other "
            "file types, use the Edit tool."
        )

    if input_.edit_mode == "insert" and input_.cell_type is None:
        return "Cell type is required when using edit_mode=insert."
    if input_.edit_mode in {"replace", "delete"} and not input_.cell_id:
        return "Cell ID must be specified when not inserting a new cell."
    return None


def _validate_notebook_edit_target(
    input_: NotebookEditInput,
    full_path: str,
    ctx: ToolUseContext,
) -> None:
    loaded = _load_notebook_for_edit(full_path, ctx)
    if input_.cell_id is not None:
        _resolve_cell_index(loaded.cells, input_.cell_id)


def _edit_notebook_file(
    input_: NotebookEditInput,
    full_path: str,
    ctx: ToolUseContext,
) -> NotebookEditResult:
    if ctx.abort_event.is_set():
        raise asyncio.CancelledError()

    syntax_error = _validate_notebook_edit_syntax(input_, full_path, ctx)
    if syntax_error is not None:
        raise NotebookEditToolError(syntax_error)

    loaded = _load_notebook_for_edit(full_path, ctx)
    selected_cell_id: str | None = input_.cell_id
    if input_.edit_mode == "insert":
        insert_index = 0
        if input_.cell_id is not None:
            insert_index = _resolve_cell_index(loaded.cells, input_.cell_id) + 1
        inserted_cell = _new_notebook_cell(
            cell_type=input_.cell_type or "code",
            source=input_.new_source,
            include_id=_notebook_uses_cell_ids(loaded.notebook),
        )
        loaded.cells.insert(insert_index, inserted_cell)
        raw_inserted_id = inserted_cell.get("id")
        selected_cell_id = (
            raw_inserted_id
            if isinstance(raw_inserted_id, str) and raw_inserted_id
            else f"cell-{insert_index}"
        )
        message = f"Inserted cell {selected_cell_id} with {input_.new_source}"
    elif input_.edit_mode == "delete":
        if input_.cell_id is None:
            raise NotebookEditToolError("Cell ID must be specified when not inserting a new cell.")
        delete_index = _resolve_cell_index(loaded.cells, input_.cell_id)
        del loaded.cells[delete_index]
        message = f"Deleted cell {input_.cell_id}"
    else:
        if input_.cell_id is None:
            raise NotebookEditToolError("Cell ID must be specified when not inserting a new cell.")
        replace_index = _resolve_cell_index(loaded.cells, input_.cell_id)
        target_cell = loaded.cells[replace_index]
        target_cell["source"] = input_.new_source
        if input_.cell_type is not None:
            target_cell["cell_type"] = input_.cell_type
        _normalize_cell_for_type(target_cell)
        message = f"Updated cell {input_.cell_id} with {input_.new_source}"

    updated_content = json.dumps(loaded.notebook, ensure_ascii=False, indent=1) + "\n"
    Path(full_path).write_text(updated_content, encoding="utf-8", newline="")
    ctx.read_file_state.set(
        full_path,
        FileState(
            content=updated_content,
            timestamp=mtime_ms(full_path),
            offset=None,
            limit=None,
            is_partial_view=False,
        ),
    )
    return NotebookEditResult(
        message=message,
        cell_id=selected_cell_id,
        edit_mode=input_.edit_mode,
    )


def _load_notebook_for_edit(full_path: str, ctx: ToolUseContext) -> _LoadedNotebook:
    try:
        stats = os.stat(full_path)
    except FileNotFoundError as exc:
        raise NotebookEditToolError("Notebook file does not exist.") from exc
    if stat.S_ISDIR(stats.st_mode):
        raise NotebookEditToolError(f"Cannot edit '{full_path}': path is a directory.")
    if not stat.S_ISREG(stats.st_mode):
        raise NotebookEditToolError(f"Cannot edit '{full_path}': not a regular file.")
    if stats.st_size > NOTEBOOK_MAX_RAW_BYTES:
        raise NotebookEditToolError(
            "Notebook file exceeds maximum editable size "
            f"({_format_file_size(NOTEBOOK_MAX_RAW_BYTES)})."
        )

    raw_content = Path(full_path).read_text(encoding="utf-8")
    current_mtime_ms = int(stats.st_mtime_ns // 1_000_000)
    read_state = ctx.read_file_state.get(full_path)
    _assert_notebook_not_stale(
        path=full_path,
        raw_content=raw_content,
        current_mtime_ms=current_mtime_ms,
        read_state=read_state,
    )
    notebook, cells = _parse_notebook_document(raw_content)
    return _LoadedNotebook(
        raw_content=raw_content,
        notebook=notebook,
        cells=cells,
        mtime_ms=current_mtime_ms,
    )


def _assert_notebook_not_stale(
    *,
    path: str,
    raw_content: str,
    current_mtime_ms: int,
    read_state: FileState | None,
) -> None:
    if not has_full_file_state(read_state):
        raise NotebookEditToolError(
            "File has not been read yet. Read it first before writing to it."
        )
    if read_state is None:
        raise NotebookEditToolError(
            "File has not been read yet. Read it first before writing to it."
        )
    if raw_content == read_state.content:
        return
    if current_mtime_ms >= read_state.timestamp or raw_content != read_state.content:
        raise NotebookEditToolError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)


def _parse_notebook_document(raw_content: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        loaded: object = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise NotebookEditToolError("Notebook is not valid JSON.") from exc
    if not isinstance(loaded, dict):
        raise NotebookEditToolError("Notebook root must be a JSON object.")
    notebook = cast(dict[str, Any], loaded)
    raw_cells = notebook.get("cells")
    if not isinstance(raw_cells, list):
        raise NotebookEditToolError("Notebook is missing a cells list.")
    cell_items = cast(list[object], raw_cells)
    for index, raw_cell in enumerate(cell_items):
        if not isinstance(raw_cell, dict):
            raise NotebookEditToolError(f"Notebook cell {index} must be a JSON object.")
        cell = cast(dict[str, Any], raw_cell)
        raw_cell_type = cell.get("cell_type")
        if not isinstance(raw_cell_type, str):
            raise NotebookEditToolError(
                f"Notebook cell {index} is missing a string cell_type."
            )
        cell_items[index] = cell
    cells = cast(list[dict[str, Any]], cell_items)
    return notebook, cells


def _resolve_cell_index(cells: list[dict[str, Any]], cell_id: str) -> int:
    for index, cell in enumerate(cells):
        raw_id = cell.get("id")
        if isinstance(raw_id, str) and raw_id == cell_id:
            return index

    parsed_index = parse_notebook_cell_id(cell_id)
    if parsed_index is not None:
        if 0 <= parsed_index < len(cells):
            return parsed_index
        raise NotebookEditToolError(
            f"Cell with index {parsed_index} does not exist in notebook."
        )
    raise NotebookEditToolError(f'Cell with ID "{cell_id}" not found in notebook.')


def parse_notebook_cell_id(cell_id: str) -> int | None:
    """Parse generated zero-based `cell-N` notebook view identifiers."""

    prefix = "cell-"
    if not cell_id.startswith(prefix):
        return None
    suffix = cell_id[len(prefix) :]
    if not suffix.isdecimal():
        return None
    return int(suffix)


def _new_notebook_cell(
    *,
    cell_type: NotebookCellType,
    source: str,
    include_id: bool,
) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "source": source,
        "metadata": {},
    }
    if include_id:
        cell["id"] = secrets.token_hex(6)
    _normalize_cell_for_type(cell)
    return cell


def _normalize_cell_for_type(cell: dict[str, Any]) -> None:
    raw_cell_type = cell.get("cell_type")
    if raw_cell_type == "markdown":
        cell.pop("execution_count", None)
        cell.pop("outputs", None)
        return
    if raw_cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def _notebook_uses_cell_ids(notebook: dict[str, Any]) -> bool:
    nbformat = notebook.get("nbformat")
    nbformat_minor = notebook.get("nbformat_minor")
    if isinstance(nbformat, int) and nbformat > 4:
        return True
    return (
        isinstance(nbformat, int)
        and nbformat == 4
        and isinstance(nbformat_minor, int)
        and nbformat_minor >= 5
    )


def _coerce_input(input_: BaseModel) -> NotebookEditInput:
    if isinstance(input_, NotebookEditInput):
        return input_
    return NotebookEditInput.model_validate(input_.model_dump())


def _should_skip_filesystem_validation(path: str) -> bool:
    return path.startswith("\\\\") or path.startswith("//")


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


__all__ = [
    "NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS",
    "NOTEBOOK_EDIT_PROMPT",
    "NOTEBOOK_EDIT_TOOL_NAME",
    "NotebookCellType",
    "NotebookEditInput",
    "NotebookEditMode",
    "NotebookEditResult",
    "NotebookEditToolError",
    "build_notebook_edit_tool",
    "create_notebook_edit_catalog_provider",
    "parse_notebook_cell_id",
]
