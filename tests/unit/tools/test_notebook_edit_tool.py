from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.core.file_state import FileState
from raygent_harness.core.permissions import PermissionAllowDecision, ToolPermissionContext
from raygent_harness.core.tool import Tool, ToolCallEvent, ToolResult, ToolUseContext
from raygent_harness.tools.file_text_utils import FILE_UNEXPECTEDLY_MODIFIED_ERROR
from raygent_harness.tools.file_tools import create_file_tooling_runtime
from raygent_harness.tools.notebook_edit_tool import (
    NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS,
    NOTEBOOK_EDIT_PROMPT,
    NOTEBOOK_EDIT_TOOL_NAME,
    NotebookEditInput,
    build_notebook_edit_tool,
    parse_notebook_cell_id,
)
from raygent_harness.tools.tool_search import run_tool_search


def _ctx(cwd: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
    )


async def _call_tool(tool: Tool, input_: BaseModel, ctx: ToolUseContext) -> ToolResult:
    events: list[ToolCallEvent] = [event async for event in tool.call(input_, ctx)]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


def _mtime_ms(path: Path) -> int:
    return int(os.stat(path).st_mtime_ns // 1_000_000)


def _write_notebook(path: Path, notebook: Mapping[str, Any]) -> str:
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    path.write_text(content, encoding="utf-8")
    return content


def _seed_notebook_read(ctx: ToolUseContext, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    ctx.read_file_state.set(
        path,
        FileState(
            content=content,
            timestamp=_mtime_ms(path),
            offset=None,
            limit=None,
            is_partial_view=False,
        ),
    )


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.asyncio
async def test_notebook_edit_axes_prompt_cell_id_parser_and_permissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(path, {"nbformat": 4, "nbformat_minor": 5, "cells": []})
    ctx = _ctx(tmp_path)
    tool = build_notebook_edit_tool()
    input_ = NotebookEditInput(
        notebook_path=str(path),
        cell_id="cell-0",
        new_source="print('x')",
    )

    permission = await tool.check_permissions(
        input_,
        ctx,
        ToolPermissionContext(mode="acceptEdits"),
    )

    assert tool.name == NOTEBOOK_EDIT_TOOL_NAME
    assert tool.should_defer is True
    assert tool.max_result_size_chars == NOTEBOOK_EDIT_MAX_RESULT_SIZE_CHARS
    assert tool.is_read_only(input_) is False
    assert tool.is_destructive(input_) is True
    assert tool.is_concurrency_safe(input_) is False
    assert tool.is_open_world(input_) is False
    assert await tool.prompt() == NOTEBOOK_EDIT_PROMPT
    assert parse_notebook_cell_id("cell-12") == 12
    assert parse_notebook_cell_id("cell-x") is None
    assert isinstance(permission, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_notebook_edit_validation_requires_notebook_and_prior_full_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    text_path = tmp_path / "notes.txt"
    raw = _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"id": "first", "cell_type": "code", "source": "x = 1"}],
        },
    )
    text_path.write_text("notes", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_notebook_edit_tool()

    non_notebook = await tool.validate_input(
        NotebookEditInput(notebook_path=str(text_path), cell_id="cell-0", new_source="x"),
        ctx,
    )
    missing_read = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), cell_id="first", new_source="x = 2"),
        ctx,
    )
    ctx.read_file_state.set(
        path,
        FileState(content=raw, timestamp=_mtime_ms(path), offset=1, limit=1),
    )
    partial_read = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), cell_id="first", new_source="x = 2"),
        ctx,
    )

    assert non_notebook.result == "error"
    assert ".ipynb" in non_notebook.message
    assert missing_read.result == "error"
    assert "File has not been read yet" in missing_read.message
    assert partial_read.result == "error"
    assert "File has not been read yet" in partial_read.message


@pytest.mark.asyncio
async def test_notebook_edit_validation_modes_cell_resolution_and_denies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"id": "first", "cell_type": "code", "source": "x = 1"}],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    tool = build_notebook_edit_tool()

    insert_without_type = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), new_source="notes", edit_mode="insert"),
        ctx,
    )
    replace_without_cell = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), new_source="x = 2"),
        ctx,
    )
    missing_cell = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), cell_id="missing", new_source="x = 2"),
        ctx,
    )
    missing_index = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), cell_id="cell-3", new_source="x = 2"),
        ctx,
    )
    ctx.permission_context = ToolPermissionContext(
        always_deny_rules={"session": (f"Edit({tmp_path}/**)",)}
    )
    denied = await tool.validate_input(
        NotebookEditInput(notebook_path=str(path), cell_id="first", new_source="x = 2"),
        ctx,
    )

    assert insert_without_type.result == "error"
    assert "Cell type is required" in insert_without_type.message
    assert replace_without_cell.result == "error"
    assert "Cell ID must be specified" in replace_without_cell.message
    assert missing_cell.result == "error"
    assert 'Cell with ID "missing" not found' in missing_cell.message
    assert missing_index.result == "error"
    assert "Cell with index 3 does not exist" in missing_index.message
    assert denied.result == "error"
    assert "denied by your permission settings" in denied.message


@pytest.mark.asyncio
async def test_notebook_edit_replace_by_real_id_clears_code_outputs_and_updates_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"language_info": {"name": "python"}},
            "cells": [
                {
                    "id": "first",
                    "cell_type": "code",
                    "source": "x = 1",
                    "execution_count": 7,
                    "outputs": [{"output_type": "stream", "text": "old"}],
                    "metadata": {"tags": ["keep"]},
                }
            ],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    tool = build_notebook_edit_tool()

    result = await _call_tool(
        tool,
        NotebookEditInput(notebook_path=str(path), cell_id="first", new_source="x = 2"),
        ctx,
    )

    notebook = _load(path)
    cell = cast(dict[str, Any], cast(list[object], notebook["cells"])[0])
    state = ctx.read_file_state.get(path)
    assert result.is_error is False
    assert result.content == "Updated cell first with x = 2"
    assert cell["source"] == "x = 2"
    assert cell["execution_count"] is None
    assert cell["outputs"] == []
    assert cell["metadata"] == {"tags": ["keep"]}
    assert state is not None
    assert state.content == path.read_text(encoding="utf-8")
    assert state.offset is None
    assert state.limit is None
    assert state.is_partial_view is False


@pytest.mark.asyncio
async def test_notebook_edit_replace_by_generated_index_and_cell_type_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"cell_type": "markdown", "source": "old", "metadata": {}}],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    tool = build_notebook_edit_tool()

    result = await _call_tool(
        tool,
        NotebookEditInput(
            notebook_path=str(path),
            cell_id="cell-0",
            new_source="print('converted')",
            cell_type="code",
        ),
        ctx,
    )

    cell = cast(dict[str, Any], cast(list[object], _load(path)["cells"])[0])
    assert result.is_error is False
    assert cell["cell_type"] == "code"
    assert cell["source"] == "print('converted')"
    assert cell["execution_count"] is None
    assert cell["outputs"] == []


@pytest.mark.asyncio
async def test_notebook_edit_insert_beginning_and_after_target(tmp_path: Path) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"id": "first", "cell_type": "markdown", "source": "first"}],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    tool = build_notebook_edit_tool()

    first_result = await _call_tool(
        tool,
        NotebookEditInput(
            notebook_path=str(path),
            new_source="# Intro",
            cell_type="markdown",
            edit_mode="insert",
        ),
        ctx,
    )
    second_result = await _call_tool(
        tool,
        NotebookEditInput(
            notebook_path=str(path),
            cell_id="first",
            new_source="print('after')",
            cell_type="code",
            edit_mode="insert",
        ),
        ctx,
    )

    cells = cast(list[dict[str, Any]], _load(path)["cells"])
    assert first_result.is_error is False
    assert second_result.is_error is False
    assert cells[0]["cell_type"] == "markdown"
    assert cells[0]["source"] == "# Intro"
    assert isinstance(cells[0].get("id"), str)
    assert cells[2]["cell_type"] == "code"
    assert cells[2]["source"] == "print('after')"
    assert cells[2]["execution_count"] is None
    assert cells[2]["outputs"] == []
    assert isinstance(cells[2].get("id"), str)


@pytest.mark.asyncio
async def test_notebook_edit_delete_cell(tmp_path: Path) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 4,
            "cells": [
                {"cell_type": "markdown", "source": "remove"},
                {"cell_type": "code", "source": "keep", "outputs": []},
            ],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    tool = build_notebook_edit_tool()

    result = await _call_tool(
        tool,
        NotebookEditInput(
            notebook_path=str(path),
            cell_id="cell-0",
            new_source="",
            edit_mode="delete",
        ),
        ctx,
    )

    cells = cast(list[dict[str, Any]], _load(path)["cells"])
    assert result.is_error is False
    assert result.content == "Deleted cell cell-0"
    assert len(cells) == 1
    assert cells[0]["source"] == "keep"


@pytest.mark.asyncio
async def test_notebook_edit_rejects_stale_content_even_with_timestamp_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.ipynb"
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"id": "first", "cell_type": "code", "source": "x = 1"}],
        },
    )
    ctx = _ctx(tmp_path)
    _seed_notebook_read(ctx, path)
    state = ctx.read_file_state.get(path)
    assert state is not None
    _write_notebook(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"id": "first", "cell_type": "code", "source": "external"}],
        },
    )
    os.utime(path, ns=(state.timestamp * 1_000_000, state.timestamp * 1_000_000))
    tool = build_notebook_edit_tool()

    result = await _call_tool(
        tool,
        NotebookEditInput(notebook_path=str(path), cell_id="first", new_source="x = 2"),
        ctx,
    )

    assert result.is_error is True
    assert FILE_UNEXPECTEDLY_MODIFIED_ERROR in str(result.content)
    assert "external" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_notebook_edit_rejects_invalid_json_and_schema_without_state_update(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.ipynb"
    missing_cells = tmp_path / "missing.ipynb"
    invalid.write_text("{", encoding="utf-8")
    missing_cells.write_text("{}", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        invalid,
        FileState(content="{", timestamp=_mtime_ms(invalid), offset=None, limit=None),
    )
    ctx.read_file_state.set(
        missing_cells,
        FileState(content="{}", timestamp=_mtime_ms(missing_cells), offset=None, limit=None),
    )
    tool = build_notebook_edit_tool()

    invalid_result = await tool.validate_input(
        NotebookEditInput(notebook_path=str(invalid), cell_id="cell-0", new_source="x"),
        ctx,
    )
    schema_result = await tool.validate_input(
        NotebookEditInput(notebook_path=str(missing_cells), cell_id="cell-0", new_source="x"),
        ctx,
    )

    assert invalid_result.result == "error"
    assert "Notebook is not valid JSON" in invalid_result.message
    assert schema_result.result == "error"
    assert "missing a cells list" in schema_result.message


@pytest.mark.asyncio
async def test_notebook_edit_is_bundled_and_toolsearch_discoverable() -> None:
    runtime = create_file_tooling_runtime()

    result = await run_tool_search(
        query="edit jupyter notebook cells",
        tools=list(runtime.tools),
        max_results=2,
    )

    assert NOTEBOOK_EDIT_TOOL_NAME in tuple(tool.name for tool in runtime.tools)
    assert result.matches[0] == NOTEBOOK_EDIT_TOOL_NAME
