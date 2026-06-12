from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.file_state import FileState
from raygent_harness.core.permissions import PermissionAllowDecision, ToolPermissionContext
from raygent_harness.core.tool import Tool, ToolCallEvent, ToolResult, ToolUseContext
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.tools.file_edit_tool import (
    FILE_EDIT_PROMPT,
    FILE_EDIT_TOOL_ALIAS,
    FILE_EDIT_TOOL_NAME,
    EditInput,
    apply_edit_to_content,
    build_file_edit_tool,
    create_file_edit_catalog_provider,
)
from raygent_harness.tools.file_read_tool import ReadInput, build_file_read_tool
from raygent_harness.tools.file_text_utils import FILE_UNEXPECTEDLY_MODIFIED_ERROR
from raygent_harness.tools.file_write_tool import (
    FILE_WRITE_PROMPT,
    FILE_WRITE_TOOL_ALIAS,
    FILE_WRITE_TOOL_NAME,
    WriteInput,
    build_file_write_tool,
    create_file_write_catalog_provider,
)


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


def _content(result: ToolResult) -> str:
    assert isinstance(result.content, str)
    return result.content


def _mtime_ms(path: Path) -> int:
    return int(os.stat(path).st_mtime_ns // 1_000_000)


def _read_state_for(path: Path, content: str) -> FileState:
    return FileState(content=content, timestamp=_mtime_ms(path), offset=1, limit=None)


def _settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path,
        home_dir=tmp_path,
        memory_base_dir=tmp_path / ".raygent" / "memory",
        team_memory_enabled=True,
    )


@pytest.mark.asyncio
async def test_file_write_tool_axes_prompt_and_create_updates_cache(tmp_path: Path) -> None:
    tool = build_file_write_tool()
    input_ = WriteInput(file_path=str(tmp_path / "new.txt"), content="alpha\n")
    ctx = _ctx(tmp_path)

    assert tool.name == FILE_WRITE_TOOL_NAME
    assert tool.aliases == (FILE_WRITE_TOOL_ALIAS,)
    assert tool.is_concurrency_safe(input_) is False
    assert tool.is_read_only(input_) is False
    assert tool.is_destructive(input_) is True
    assert tool.is_open_world(input_) is False
    assert await tool.prompt() == FILE_WRITE_PROMPT

    result = await _call_tool(tool, input_, ctx)

    assert result.is_error is False
    assert "File created successfully" in _content(result)
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "alpha\n"
    state = ctx.read_file_state.get(tmp_path / "new.txt")
    assert state is not None
    assert state.content == "alpha\n"
    assert state.offset is None
    assert state.limit is None


@pytest.mark.asyncio
async def test_file_write_requires_prior_full_read_for_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_write_tool()

    missing = await tool.validate_input(WriteInput(file_path=str(path), content="new\n"), ctx)
    ctx.read_file_state.set(
        path,
        FileState(content="old\n", timestamp=_mtime_ms(path), offset=2, limit=1),
    )
    partial = await tool.validate_input(WriteInput(file_path=str(path), content="new\n"), ctx)
    ctx.read_file_state.set(path, _read_state_for(path, "old\n"))
    ok = await tool.validate_input(WriteInput(file_path=str(path), content="new\n"), ctx)

    assert missing.result == "error"
    assert "File has not been read yet" in missing.message
    assert partial.result == "error"
    assert ok.result == "ok"


@pytest.mark.asyncio
async def test_file_write_call_allows_timestamp_false_positive_when_content_matches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        path,
        FileState(content="old\n", timestamp=0, offset=1, limit=None),
    )
    tool = build_file_write_tool()

    result = await _call_tool(tool, WriteInput(file_path=str(path), content="new\n"), ctx)

    assert result.is_error is False
    assert path.read_text(encoding="utf-8") == "new\n"


@pytest.mark.asyncio
async def test_file_write_preserves_existing_utf16le_encoding(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_bytes(b"\xff\xfe" + "old\n".encode("utf-16le"))
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        path,
        FileState(content="old\n", timestamp=_mtime_ms(path), offset=1, limit=None),
    )
    tool = build_file_write_tool()

    result = await _call_tool(tool, WriteInput(file_path=str(path), content="new\n"), ctx)

    assert result.is_error is False
    assert path.read_bytes() == "new\n".encode("utf-16le")


@pytest.mark.asyncio
async def test_file_write_call_rejects_stale_content_change(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        path,
        FileState(content="different\n", timestamp=0, offset=1, limit=None),
    )
    tool = build_file_write_tool()

    result = await _call_tool(tool, WriteInput(file_path=str(path), content="new\n"), ctx)

    assert result.is_error is True
    assert FILE_UNEXPECTEDLY_MODIFIED_ERROR in _content(result)
    assert path.read_text(encoding="utf-8") == "old\n"


@pytest.mark.asyncio
async def test_file_write_blocks_team_memory_secrets(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    target = team_dir / "note.md"
    ctx = _ctx(tmp_path)
    tool = build_file_write_tool(memory_settings=cfg)

    result = await tool.validate_input(
        WriteInput(file_path=str(target), content="token=ghp_" + "a" * 36),
        ctx,
    )

    assert result.result == "error"
    assert "potential secrets" in result.message


@pytest.mark.asyncio
async def test_file_write_check_permissions_delegates_to_edit_permission(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    ctx = _ctx(tmp_path)
    tool = build_file_write_tool()

    result = await tool.check_permissions(
        WriteInput(file_path=str(target), content="x"),
        ctx,
        ToolPermissionContext(mode="acceptEdits"),
    )

    assert isinstance(result, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_file_edit_tool_axes_prompt_and_exact_replace_updates_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(path, _read_state_for(path, "alpha\nbeta\n"))
    tool = build_file_edit_tool()
    input_ = EditInput(file_path=str(path), old_string="beta", new_string="gamma")

    assert tool.name == FILE_EDIT_TOOL_NAME
    assert tool.aliases == (FILE_EDIT_TOOL_ALIAS,)
    assert tool.is_concurrency_safe(input_) is False
    assert tool.is_read_only(input_) is False
    assert tool.is_destructive(input_) is True
    assert tool.is_open_world(input_) is False
    assert await tool.prompt() == FILE_EDIT_PROMPT

    result = await _call_tool(tool, input_, ctx)

    assert result.is_error is False
    assert "updated successfully" in _content(result)
    assert path.read_text(encoding="utf-8") == "alpha\ngamma\n"
    state = ctx.read_file_state.get(path)
    assert state is not None
    assert state.content == "alpha\ngamma\n"
    assert state.offset is None
    assert state.limit is None


@pytest.mark.asyncio
async def test_file_edit_validation_rejects_missing_ambiguous_and_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("one\ntwo\ntwo\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(path, _read_state_for(path, "one\ntwo\ntwo\n"))
    tool = build_file_edit_tool()

    missing = await tool.validate_input(
        EditInput(file_path=str(path), old_string="three", new_string="four"),
        ctx,
    )
    ambiguous = await tool.validate_input(
        EditInput(file_path=str(path), old_string="two", new_string="TWO"),
        ctx,
    )
    noop = await tool.validate_input(
        EditInput(file_path=str(path), old_string="two", new_string="two"),
        ctx,
    )
    ok = await tool.validate_input(
        EditInput(file_path=str(path), old_string="two", new_string="TWO", replace_all=True),
        ctx,
    )

    assert missing.result == "error"
    assert "String to replace not found" in missing.message
    assert ambiguous.result == "error"
    assert "Found 2 matches" in ambiguous.message
    assert noop.result == "error"
    assert "No changes to make" in noop.message
    assert ok.result == "ok"


@pytest.mark.asyncio
async def test_file_edit_replace_all_and_missing_file_create(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("foo foo\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(path, _read_state_for(path, "foo foo\n"))
    tool = build_file_edit_tool()

    replaced = await _call_tool(
        tool,
        EditInput(file_path=str(path), old_string="foo", new_string="bar", replace_all=True),
        ctx,
    )
    created_path = tmp_path / "created.txt"
    created = await _call_tool(
        tool,
        EditInput(file_path=str(created_path), old_string="", new_string="new\n"),
        ctx,
    )
    missing_non_empty = await tool.validate_input(
        EditInput(file_path=str(tmp_path / "missing.txt"), old_string="x", new_string="y"),
        ctx,
    )

    assert "All occurrences" in _content(replaced)
    assert path.read_text(encoding="utf-8") == "bar bar\n"
    assert created.is_error is False
    assert created_path.read_text(encoding="utf-8") == "new\n"
    assert missing_non_empty.result == "error"
    assert "File does not exist" in missing_non_empty.message


@pytest.mark.asyncio
async def test_file_edit_preserves_line_endings_and_quote_style(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text('say \u201chello\u201d\r\n', encoding="utf-8", newline="")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        path,
        FileState(
            content='say \u201chello\u201d\n',
            timestamp=_mtime_ms(path),
            offset=1,
            limit=None,
        ),
    )
    tool = build_file_edit_tool()

    result = await _call_tool(
        tool,
        EditInput(file_path=str(path), old_string='"hello"', new_string='"bye"'),
        ctx,
    )

    assert result.is_error is False
    assert path.read_bytes().decode("utf-8") == 'say \u201cbye\u201d\r\n'


@pytest.mark.asyncio
async def test_file_edit_preserves_existing_utf16le_encoding(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_bytes(b"\xff\xfe" + "alpha\nbeta\n".encode("utf-16le"))
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(
        path,
        FileState(content="alpha\nbeta\n", timestamp=_mtime_ms(path), offset=1, limit=None),
    )
    tool = build_file_edit_tool()

    result = await _call_tool(
        tool,
        EditInput(file_path=str(path), old_string="beta", new_string="gamma"),
        ctx,
    )

    assert result.is_error is False
    assert path.read_bytes() == b"\xff\xfe" + "alpha\ngamma\n".encode("utf-16le")


def test_file_edit_quote_preservation_treats_dash_as_opening_context() -> None:
    preview = apply_edit_to_content(
        "old \u201cvalue\u201d",
        old_string='"value"',
        new_string='\u2014"next"',
    )

    assert preview.updated_content == "old \u2014\u201cnext\u201d"


@pytest.mark.asyncio
async def test_file_edit_rejects_partial_read_and_stale_content(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_edit_tool()

    missing_read = await tool.validate_input(
        EditInput(file_path=str(path), old_string="alpha", new_string="beta"),
        ctx,
    )
    ctx.read_file_state.set(
        path,
        FileState(content="alpha\n", timestamp=_mtime_ms(path), offset=1, limit=1),
    )
    partial_read = await tool.validate_input(
        EditInput(file_path=str(path), old_string="alpha", new_string="beta"),
        ctx,
    )
    ctx.read_file_state.set(
        path,
        FileState(content="different\n", timestamp=0, offset=1, limit=None),
    )
    stale_call = await _call_tool(
        tool,
        EditInput(file_path=str(path), old_string="alpha", new_string="beta"),
        ctx,
    )

    assert missing_read.result == "error"
    assert partial_read.result == "error"
    assert stale_call.is_error is True
    assert FILE_UNEXPECTEDLY_MODIFIED_ERROR in _content(stale_call)


@pytest.mark.asyncio
async def test_file_edit_blocks_team_memory_secrets_but_scans_new_string_only(
    tmp_path: Path,
) -> None:
    cfg = _settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    target = team_dir / "note.md"
    target.write_text("remove me\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(target, _read_state_for(target, "remove me\n"))
    tool = build_file_edit_tool(memory_settings=cfg)

    blocked = await tool.validate_input(
        EditInput(file_path=str(target), old_string="remove me", new_string="ghp_" + "a" * 36),
        ctx,
    )
    cleanup = await tool.validate_input(
        EditInput(file_path=str(target), old_string="remove me", new_string="safe"),
        ctx,
    )

    assert blocked.result == "error"
    assert "potential secrets" in blocked.message
    assert cleanup.result == "ok"


@pytest.mark.asyncio
async def test_file_edit_rejects_notebooks_and_trailing_newline_deletion(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    text = tmp_path / "example.txt"
    text.write_text("alpha\nbeta\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.read_file_state.set(notebook, _read_state_for(notebook, "{}"))
    ctx.read_file_state.set(text, _read_state_for(text, "alpha\nbeta\n"))
    tool = build_file_edit_tool()

    notebook_result = await tool.validate_input(
        EditInput(file_path=str(notebook), old_string="{}", new_string="[]"),
        ctx,
    )
    await _call_tool(
        tool,
        EditInput(file_path=str(text), old_string="alpha", new_string=""),
        ctx,
    )

    assert notebook_result.result == "error"
    assert "Use NotebookEdit" in notebook_result.message
    assert text.read_text(encoding="utf-8") == "beta\n"


@pytest.mark.asyncio
async def test_file_edit_validation_preserves_explicit_deny_before_notebook_error(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.permission_context = ToolPermissionContext(
        always_deny_rules={"session": (f"{FILE_EDIT_TOOL_NAME}({tmp_path}/**)",)}
    )
    tool = build_file_edit_tool()

    result = await tool.validate_input(
        EditInput(file_path=str(notebook), old_string="{}", new_string="[]"),
        ctx,
    )

    assert result.result == "error"
    assert "denied by your permission settings" in result.message


def test_apply_edit_to_content_preview_helper() -> None:
    preview = apply_edit_to_content(
        "a\nb\nb\n",
        old_string="b",
        new_string="B",
        replace_all=True,
    )

    assert preview.updated_content == "a\nB\nB\n"
    assert preview.replacement_count == 2


@pytest.mark.asyncio
async def test_file_read_marks_range_cache_entries_partial(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    read_tool = build_file_read_tool()

    await _call_tool(read_tool, ReadInput(file_path=str(path), offset=2, limit=1), ctx)

    state = ctx.read_file_state.get(path)
    assert state is not None
    assert state.is_partial_view is True


@pytest.mark.asyncio
async def test_file_write_edit_catalog_providers_append_and_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    config = QueryConfig(model="model", tools=(build_file_write_tool(), build_file_edit_tool()))

    write_provider = create_file_write_catalog_provider()
    edit_provider = create_file_edit_catalog_provider()

    write_tools = await write_provider(config, ctx, ())
    edit_tools = await edit_provider(config, ctx, ())

    assert write_tools is not None
    assert tuple(tool.name for tool in write_tools).count(FILE_WRITE_TOOL_NAME) == 1
    assert edit_tools is not None
    assert tuple(tool.name for tool in edit_tools).count(FILE_EDIT_TOOL_NAME) == 1
