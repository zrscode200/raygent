from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionAskDecision,
    ToolPermissionContext,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.tools.discovery_tools import (
    create_discovery_tooling_runtime,
    create_discovery_tools_catalog_provider,
)
from raygent_harness.tools.glob_tool import (
    GLOB_MAX_RESULT_SIZE_CHARS,
    GLOB_PROMPT,
    GLOB_TOOL_NAME,
    GlobInput,
    build_glob_tool,
)
from raygent_harness.tools.grep_tool import (
    GREP_MAX_RESULT_SIZE_CHARS,
    GREP_PROMPT,
    GREP_TOOL_NAME,
    GrepInput,
    build_grep_tool,
)
from raygent_harness.tools.search_backend import (
    COMMON_TYPE_GLOBS,
    DEFAULT_GLOB_LIMIT,
    DEFAULT_GREP_HEAD_LIMIT,
    GrepSearchRequest,
    SearchTimeoutError,
    StdlibSearchBackend,
    expand_glob_patterns,
    extract_glob_base_directory,
)


class EmptyInput(BaseModel):
    pass


async def _dummy_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _dummy_tool(name: str, *, aliases: tuple[str, ...] = ()) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            aliases=aliases,
            description=f"{name} tool",
            input_model=EmptyInput,
            call=_dummy_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx(
    cwd: Path,
    *,
    permission_context: ToolPermissionContext | None = None,
) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        permission_context=permission_context or ToolPermissionContext(),
    )


async def _call_tool(
    tool: Tool,
    input_: BaseModel,
    ctx: ToolUseContext,
) -> ToolResult:
    events = [event async for event in tool.call(input_, ctx)]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


def _content(result: ToolResult) -> str:
    assert isinstance(result.content, str)
    return result.content


def _set_mtime_ms(path: Path, mtime_ms: int) -> None:
    os.utime(path, ns=(mtime_ms * 1_000_000, mtime_ms * 1_000_000))


def test_search_backend_helpers_expand_common_patterns() -> None:
    assert expand_glob_patterns("*.{ts,tsx} *.py,*.md") == (
        "*.ts",
        "*.tsx",
        "*.py",
        "*.md",
    )

    base, pattern = extract_glob_base_directory("/repo/src/**/*.py")
    assert base == "/repo/src"
    assert pattern == "**/*.py"
    assert "*.py" in COMMON_TYPE_GLOBS["py"]


def test_glob_tool_axes_and_prompt() -> None:
    tool = build_glob_tool()
    input_ = GlobInput(pattern="**/*.py")

    assert tool.name == GLOB_TOOL_NAME
    assert tool.is_concurrency_safe(input_) is True
    assert tool.is_read_only(input_) is True
    assert tool.is_destructive(input_) is False
    assert tool.is_open_world(input_) is False
    assert tool.max_result_size_chars == GLOB_MAX_RESULT_SIZE_CHARS
    assert asyncio.run(tool.prompt()) == GLOB_PROMPT


@pytest.mark.asyncio
async def test_glob_finds_matching_files_with_mtime_order_and_truncation(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older.py"
    newer = tmp_path / "pkg" / "newer.py"
    ignored = tmp_path / "pkg" / "ignored.txt"
    newer.parent.mkdir()
    older.write_text("old\n", encoding="utf-8")
    newer.write_text("new\n", encoding="utf-8")
    ignored.write_text("no\n", encoding="utf-8")
    _set_mtime_ms(older, 1_000)
    _set_mtime_ms(newer, 2_000)
    tool = build_glob_tool(max_results=1)

    result = await _call_tool(tool, GlobInput(pattern="**/*.py"), _ctx(tmp_path))

    content = _content(result)
    assert content.splitlines()[0] == "pkg/newer.py"
    assert "older.py" not in content
    assert "Results are truncated" in content


@pytest.mark.asyncio
async def test_glob_globstar_matches_direct_and_nested_children(tmp_path: Path) -> None:
    direct = tmp_path / "src" / "direct.py"
    nested = tmp_path / "src" / "pkg" / "nested.py"
    nested.parent.mkdir(parents=True)
    direct.write_text("direct\n", encoding="utf-8")
    nested.write_text("nested\n", encoding="utf-8")
    tool = build_glob_tool()

    result = await _call_tool(tool, GlobInput(pattern="src/**/*.py"), _ctx(tmp_path))

    content = _content(result)
    assert "src/direct.py" in content
    assert "src/pkg/nested.py" in content


@pytest.mark.asyncio
async def test_glob_validates_path_and_checks_read_permission(tmp_path: Path) -> None:
    tool = build_glob_tool()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    validation = await tool.validate_input(
        GlobInput(pattern="*.py", path=str(tmp_path / "missing")),
        _ctx(tmp_path),
    )
    assert isinstance(validation, ValidationError)
    assert "Directory does not exist" in validation.message

    permission = await tool.check_permissions(
        GlobInput(pattern="*.py", path=str(outside)),
        _ctx(tmp_path),
        ToolPermissionContext(),
    )
    assert isinstance(permission, PermissionAskDecision)


@pytest.mark.asyncio
async def test_glob_filters_denied_child_paths(tmp_path: Path) -> None:
    public = tmp_path / "public.py"
    secret = tmp_path / "secret.py"
    public.write_text("public\n", encoding="utf-8")
    secret.write_text("secret\n", encoding="utf-8")
    ctx = _ctx(
        tmp_path,
        permission_context=ToolPermissionContext(
            always_deny_rules={"session": (f"Read({secret})",)}
        ),
    )
    tool = build_glob_tool()

    result = await _call_tool(tool, GlobInput(pattern="*.py"), ctx)

    content = _content(result)
    assert "public.py" in content
    assert "secret.py" not in content


def test_grep_tool_axes_and_prompt() -> None:
    tool = build_grep_tool()
    input_ = GrepInput(pattern="needle")

    assert tool.name == GREP_TOOL_NAME
    assert tool.is_concurrency_safe(input_) is True
    assert tool.is_read_only(input_) is True
    assert tool.is_destructive(input_) is False
    assert tool.is_open_world(input_) is False
    assert tool.max_result_size_chars == GREP_MAX_RESULT_SIZE_CHARS
    assert asyncio.run(tool.prompt()) == GREP_PROMPT


@pytest.mark.asyncio
async def test_grep_files_with_matches_defaults_to_head_limit_and_order(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older.py"
    newer = tmp_path / "newer.py"
    other = tmp_path / "notes.md"
    older.write_text("needle\n", encoding="utf-8")
    newer.write_text("needle\n", encoding="utf-8")
    other.write_text("needle\n", encoding="utf-8")
    _set_mtime_ms(older, 1_000)
    _set_mtime_ms(newer, 2_000)
    tool = build_grep_tool()

    result = await _call_tool(
        tool,
        GrepInput(pattern="needle", glob="*.py"),
        _ctx(tmp_path),
    )

    lines = _content(result).splitlines()
    assert lines[0] == "Found 2 files"
    assert lines[1:] == ["newer.py", "older.py"]


@pytest.mark.asyncio
async def test_grep_globstar_filter_matches_direct_and_nested_children(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "src" / "direct.py"
    nested = tmp_path / "src" / "pkg" / "nested.py"
    nested.parent.mkdir(parents=True)
    direct.write_text("needle\n", encoding="utf-8")
    nested.write_text("needle\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(
        tool,
        GrepInput(pattern="needle", glob="src/**/*.py"),
        _ctx(tmp_path),
    )

    content = _content(result)
    assert "src/direct.py" in content
    assert "src/pkg/nested.py" in content


@pytest.mark.asyncio
async def test_grep_content_mode_supports_context_and_line_numbers(
    tmp_path: Path,
) -> None:
    target = tmp_path / "example.py"
    target.write_text("before\nneedle\nAfter\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(
        tool,
        GrepInput.model_validate(
            {"pattern": "needle", "output_mode": "content", "-B": 1}
        ),
        _ctx(tmp_path),
    )

    content = _content(result)
    assert "example.py-1-before" in content
    assert "example.py:2:needle" in content
    assert "example.py:3:After" not in content


@pytest.mark.asyncio
async def test_grep_content_mode_respects_line_number_flag(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("needle\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(
        tool,
        GrepInput.model_validate(
            {"pattern": "needle", "output_mode": "content", "-n": False}
        ),
        _ctx(tmp_path),
    )

    content = _content(result)
    assert content == "example.py:needle"
    assert "example.py:1:needle" not in content


@pytest.mark.asyncio
async def test_grep_count_mode_type_filter_and_case_insensitive(
    tmp_path: Path,
) -> None:
    py_file = tmp_path / "example.py"
    ts_file = tmp_path / "example.ts"
    py_file.write_text("Needle\nneedle\n", encoding="utf-8")
    ts_file.write_text("needle\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(
        tool,
        GrepInput.model_validate(
            {
                "pattern": "needle",
                "output_mode": "count",
                "-i": True,
                "type": "py",
            }
        ),
        _ctx(tmp_path),
    )

    content = _content(result)
    assert "example.py:2" in content
    assert "example.ts" not in content
    assert "Found 2 total occurrences across 1 file" in content


@pytest.mark.asyncio
async def test_grep_validates_regex_path_and_type(tmp_path: Path) -> None:
    tool = build_grep_tool()

    regex_validation = await tool.validate_input(GrepInput(pattern="["), _ctx(tmp_path))
    assert isinstance(regex_validation, ValidationError)
    assert "Invalid regular expression" in regex_validation.message

    path_validation = await tool.validate_input(
        GrepInput(pattern="x", path=str(tmp_path / "missing")),
        _ctx(tmp_path),
    )
    assert isinstance(path_validation, ValidationError)
    assert "Path does not exist" in path_validation.message

    type_validation = await tool.validate_input(
        GrepInput(pattern="x", type="definitely-not-a-type"),
        _ctx(tmp_path),
    )
    assert isinstance(type_validation, ValidationError)
    assert "Unsupported Grep type" in type_validation.message


@pytest.mark.asyncio
async def test_grep_permission_and_child_filtering(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.py"
    denied = tmp_path / "denied.py"
    allowed.write_text("needle\n", encoding="utf-8")
    denied.write_text("needle\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-grep"
    outside.mkdir()
    tool = build_grep_tool()

    permission = await tool.check_permissions(
        GrepInput(pattern="needle", path=str(outside)),
        _ctx(tmp_path),
        ToolPermissionContext(),
    )
    assert isinstance(permission, PermissionAskDecision)

    ctx = _ctx(
        tmp_path,
        permission_context=ToolPermissionContext(
            always_deny_rules={"session": (f"Read({denied})",)}
        ),
    )
    result = await _call_tool(tool, GrepInput(pattern="needle"), ctx)
    content = _content(result)
    assert "allowed.py" in content
    assert "denied.py" not in content


@pytest.mark.asyncio
async def test_grep_skips_binary_and_excludes_vcs_directories(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    binary = tmp_path / "binary.txt"
    git_file = tmp_path / ".git" / "objects" / "leak.txt"
    git_file.parent.mkdir(parents=True)
    good.write_text("needle\n", encoding="utf-8")
    binary.write_bytes(b"needle\x00secret")
    git_file.write_text("needle\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(tool, GrepInput(pattern="needle"), _ctx(tmp_path))

    content = _content(result)
    assert "good.txt" in content
    assert "binary.txt" not in content
    assert ".git" not in content


@pytest.mark.asyncio
async def test_grep_searches_large_text_files_with_bounded_output(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_text(("x" * 1_100_000) + "\nneedle\n", encoding="utf-8")
    tool = build_grep_tool()

    result = await _call_tool(tool, GrepInput(pattern="needle"), _ctx(tmp_path))

    content = _content(result)
    assert "large.txt" in content
    assert "No files found" not in content


@pytest.mark.asyncio
async def test_grep_backend_times_out_pathological_regex(tmp_path: Path) -> None:
    target = tmp_path / "pathological.txt"
    target.write_text(("a" * 5_000) + "!\n", encoding="utf-8")
    backend = StdlibSearchBackend()

    with pytest.raises(SearchTimeoutError):
        await backend.grep(
            GrepSearchRequest(
                pattern="(a+)+$",
                root=target,
                timeout_s=0.05,
            )
        )


@pytest.mark.asyncio
async def test_grep_backend_terminates_worker_on_abort(tmp_path: Path) -> None:
    target = tmp_path / "pathological_abort.txt"
    target.write_text(("a" * 5_000) + "!\n", encoding="utf-8")
    abort_event = asyncio.Event()
    backend = StdlibSearchBackend()

    async def abort_soon() -> None:
        await asyncio.sleep(0.05)
        abort_event.set()

    abort_task = asyncio.create_task(abort_soon())
    try:
        with pytest.raises(asyncio.CancelledError):
            await backend.grep(
                GrepSearchRequest(
                    pattern="(a+)+$",
                    root=target,
                    timeout_s=5,
                    abort_event=abort_event,
                )
            )
    finally:
        abort_task.cancel()


@pytest.mark.asyncio
async def test_grep_head_limit_offset_and_unlimited_escape_hatch(tmp_path: Path) -> None:
    for index in range(3):
        path = tmp_path / f"{index}.txt"
        path.write_text("needle\n", encoding="utf-8")
        _set_mtime_ms(path, 1_000 + index)
    tool = build_grep_tool()

    limited = await _call_tool(
        tool,
        GrepInput(pattern="needle", head_limit=1, offset=1),
        _ctx(tmp_path),
    )
    assert "Found 1 file (limit: 1, offset: 1)" in _content(limited)
    assert "1.txt" in _content(limited)
    assert "2.txt" not in _content(limited)

    unlimited = await _call_tool(
        tool,
        GrepInput(pattern="needle", head_limit=0),
        _ctx(tmp_path),
    )
    assert "Found 3 files" in _content(unlimited)


@pytest.mark.asyncio
async def test_discovery_runtime_catalog_composes_and_replaces_collisions(
    tmp_path: Path,
) -> None:
    keep = _dummy_tool("Keep")
    glob_collision = _dummy_tool(GLOB_TOOL_NAME)
    grep_collision = _dummy_tool("LegacyGrep", aliases=(GREP_TOOL_NAME,))
    seen_skills: list[int] = []

    async def upstream(
        config: QueryConfig,
        _ctx: ToolUseContext,
        skills: Sequence[object],
        /,
    ) -> Sequence[Tool] | None:
        seen_skills.append(len(skills))
        return (keep, glob_collision, grep_collision, *config.tools)

    provider = create_discovery_tools_catalog_provider(upstream=upstream)
    tools = await provider(QueryConfig(model="m", tools=(_dummy_tool("Base"),)), _ctx(tmp_path), ())

    assert tools is not None
    assert seen_skills == [0]
    assert tuple(tool.name for tool in tools) == (
        "Keep",
        "Base",
        GLOB_TOOL_NAME,
        GREP_TOOL_NAME,
    )

    runtime = create_discovery_tooling_runtime()
    assert tuple(tool.name for tool in runtime.tools) == (GLOB_TOOL_NAME, GREP_TOOL_NAME)


@pytest.mark.asyncio
async def test_discovery_tools_allow_root_with_bypass_permissions(tmp_path: Path) -> None:
    glob_tool = build_glob_tool()
    grep_tool = build_grep_tool()
    ctx = _ctx(tmp_path)
    permission_context = ToolPermissionContext(mode="bypassPermissions")

    glob_decision = await glob_tool.check_permissions(
        GlobInput(pattern="*.py"),
        ctx,
        permission_context,
    )
    grep_decision = await grep_tool.check_permissions(
        GrepInput(pattern="needle"),
        ctx,
        permission_context,
    )

    assert isinstance(glob_decision, PermissionAllowDecision)
    assert isinstance(grep_decision, PermissionAllowDecision)
    assert DEFAULT_GLOB_LIMIT == 100
    assert DEFAULT_GREP_HEAD_LIMIT == 250
