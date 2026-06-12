from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    ToolPermissionContext,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    build_tool,
    find_tool_by_name,
)
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path
from raygent_harness.services.extract_memories import (
    build_default_extraction_tool_catalog,
    build_extraction_tool_policy,
)
from raygent_harness.tools.bash_tool import BashInput
from raygent_harness.tools.file_tools import create_file_tooling_runtime
from raygent_harness.tools.glob_tool import GlobInput
from raygent_harness.tools.grep_tool import GrepInput


class EmptyInput(BaseModel):
    pass


async def _unused_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="unused")


async def _call_tool(
    tool: Tool,
    input_: BaseModel,
    ctx: ToolUseContext,
) -> tuple[ToolResult, ...]:
    events: list[ToolResult] = []
    async for event in tool.call(input_, ctx):
        if isinstance(event, ToolResult):
            events.append(event)
    return tuple(events)


def _content(results: tuple[ToolResult, ...]) -> str:
    parts: list[str] = []
    for result in results:
        content = result.content
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _agent_tool():
    return build_tool(
        ToolSpec(
            name="Agent",
            description="Launch agent",
            input_model=EmptyInput,
            call=_unused_call,
            is_read_only=False,
            is_destructive=True,
        )
    )


def _settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "project",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "memory-base",
    )


def _ctx(tmp_path: Path, permission_context: ToolPermissionContext) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=str(tmp_path / "project"),
        permission_context=permission_context,
    )


async def test_extraction_policy_exposes_only_safe_available_tools(tmp_path: Path) -> None:
    memory_settings = _settings(tmp_path)
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    parent_context = ToolPermissionContext(
        mode="bypassPermissions",
        always_allow_rules={"session": ("Agent", "Edit(/tmp/**)")},
    )

    policy = build_extraction_tool_policy(
        (*runtime.tools, _agent_tool()),
        settings=memory_settings,
        parent_permission_context=parent_context,
    )

    assert policy.tool_names == ("Read", "Write", "Edit")
    assert [tool.name for tool in policy.tools] == ["Read", "Write", "Edit"]
    assert find_tool_by_name(policy.tools, "Agent") is None
    assert policy.permission_context.mode == "default"
    assert "Edit(/tmp/**)" not in str(policy.permission_context.always_allow_rules)
    assert str(get_auto_mem_path(memory_settings)) in str(
        policy.permission_context.always_allow_rules
    )


async def test_extraction_policy_exposes_concrete_default_search_and_bash_tools(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    deps = QueryDeps(task_store=AppStateStore())
    parent_context = ToolPermissionContext(
        always_allow_rules={"session": ("Agent", "mcp__github__issue")},
    )

    catalog = build_default_extraction_tool_catalog(
        parent_tools=(_agent_tool(),),
        parent_deps=deps,
        settings=memory_settings,
    )
    policy = build_extraction_tool_policy(
        catalog,
        settings=memory_settings,
        parent_permission_context=parent_context,
    )

    assert policy.tool_names == ("Read", "Grep", "Glob", "Bash", "Write", "Edit")
    assert [tool.name for tool in policy.tools] == [
        "Read",
        "Grep",
        "Glob",
        "Bash",
        "Write",
        "Edit",
    ]
    assert find_tool_by_name(policy.tools, "Agent") is None
    assert find_tool_by_name(policy.tools, "mcp__github__issue") is None
    assert "Agent" not in str(policy.permission_context.always_allow_rules)
    assert "mcp__github__issue" not in str(policy.permission_context.always_allow_rules)


async def test_extraction_policy_allows_memory_writes_and_denies_outside_even_with_bypass(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    policy = build_extraction_tool_policy(
        runtime.tools,
        settings=memory_settings,
        parent_permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    write = find_tool_by_name(policy.tools, "Write")
    assert write is not None
    ctx = _ctx(tmp_path, policy.permission_context)

    memory_file = get_auto_mem_path(memory_settings) / "topic.md"
    inside = write.input_model.model_validate(
        {"file_path": str(memory_file), "content": "remembered"}
    )
    inside_validation = await write.validate_input(inside, ctx)
    inside_permission = await write.check_permissions(
        inside,
        ctx,
        policy.permission_context,
    )

    assert isinstance(inside_validation, ValidationOk)
    assert isinstance(inside_permission, PermissionAllowDecision)

    outside = write.input_model.model_validate(
        {"file_path": str(tmp_path / "outside.md"), "content": "nope"}
    )
    outside_validation = await write.validate_input(outside, ctx)
    outside_permission = await write.check_permissions(
        outside,
        ctx,
        ToolPermissionContext(
            mode="bypassPermissions",
            always_allow_rules={"session": ("Edit(/tmp/**)",)},
        ),
    )

    assert isinstance(outside_validation, ValidationError)
    assert isinstance(outside_permission, PermissionDenyDecision)
    assert not (tmp_path / "outside.md").exists()


async def test_extraction_policy_keeps_file_safety_checks_for_memory_writes(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    policy = build_extraction_tool_policy(runtime.tools, settings=memory_settings)
    write = find_tool_by_name(policy.tools, "Write")
    assert write is not None
    ctx = _ctx(tmp_path, policy.permission_context)

    sensitive = write.input_model.model_validate(
        {
            "file_path": str(get_auto_mem_path(memory_settings) / ".env"),
            "content": "SECRET=value\n",
        }
    )
    sensitive_permission = await write.check_permissions(
        sensitive,
        ctx,
        policy.permission_context,
    )

    assert isinstance(sensitive_permission, PermissionAskDecision)


async def test_extraction_policy_allows_search_tools_without_prompt(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.md").write_text("needle\n")
    deps = QueryDeps(task_store=AppStateStore())
    policy = build_extraction_tool_policy(
        build_default_extraction_tool_catalog(
            parent_tools=(),
            parent_deps=deps,
            settings=memory_settings,
        ),
        settings=memory_settings,
    )
    ctx = _ctx(tmp_path, policy.permission_context)

    grep = find_tool_by_name(policy.tools, "Grep")
    glob = find_tool_by_name(policy.tools, "Glob")
    assert grep is not None
    assert glob is not None

    grep_input = GrepInput(pattern="needle", path=str(project))
    glob_input = GlobInput(pattern="*.md", path=str(project))
    grep_permission = await grep.check_permissions(
        grep_input,
        ctx,
        policy.permission_context,
    )
    glob_permission = await glob.check_permissions(
        glob_input,
        ctx,
        policy.permission_context,
    )

    assert isinstance(grep_permission, PermissionAllowDecision)
    assert isinstance(glob_permission, PermissionAllowDecision)


async def test_extraction_policy_allows_read_and_search_outside_cwd_without_prompt(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside-readable"
    outside.mkdir()
    outside_file = outside / "external.md"
    outside_file.write_text("outside needle\n", encoding="utf-8")
    deps = QueryDeps(task_store=AppStateStore())
    policy = build_extraction_tool_policy(
        build_default_extraction_tool_catalog(
            parent_tools=(),
            parent_deps=deps,
            settings=memory_settings,
        ),
        settings=memory_settings,
    )
    ctx = _ctx(tmp_path, policy.permission_context)

    read = find_tool_by_name(policy.tools, "Read")
    grep = find_tool_by_name(policy.tools, "Grep")
    glob = find_tool_by_name(policy.tools, "Glob")
    assert read is not None
    assert grep is not None
    assert glob is not None

    read_input = read.input_model.model_validate({"file_path": str(outside_file)})
    read_permission = await read.check_permissions(
        read_input,
        ctx,
        policy.permission_context,
    )
    grep_input = GrepInput(pattern="needle", path=str(outside))
    glob_input = GlobInput(pattern="*.md", path=str(outside))
    grep_permission = await grep.check_permissions(
        grep_input,
        ctx,
        policy.permission_context,
    )
    glob_permission = await glob.check_permissions(
        glob_input,
        ctx,
        policy.permission_context,
    )

    assert isinstance(read_permission, PermissionAllowDecision)
    assert isinstance(grep_permission, PermissionAllowDecision)
    assert isinstance(glob_permission, PermissionAllowDecision)

    grep_content = _content(await _call_tool(grep, grep_input, ctx))
    glob_content = _content(await _call_tool(glob, glob_input, ctx))

    assert "external.md" in grep_content
    assert "external.md" in glob_content


async def test_extraction_policy_preserves_explicit_read_denies(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    allowed = project / "allowed.md"
    denied = project / "denied.md"
    allowed.write_text("needle\n", encoding="utf-8")
    denied.write_text("needle\n", encoding="utf-8")
    deps = QueryDeps(task_store=AppStateStore())
    policy = build_extraction_tool_policy(
        build_default_extraction_tool_catalog(
            parent_tools=(),
            parent_deps=deps,
            settings=memory_settings,
        ),
        settings=memory_settings,
        parent_permission_context=ToolPermissionContext(
            always_deny_rules={"session": (f"Read({denied})",)}
        ),
    )
    ctx = _ctx(tmp_path, policy.permission_context)

    read = find_tool_by_name(policy.tools, "Read")
    grep = find_tool_by_name(policy.tools, "Grep")
    glob = find_tool_by_name(policy.tools, "Glob")
    assert read is not None
    assert grep is not None
    assert glob is not None

    denied_read_input = read.input_model.model_validate({"file_path": str(denied)})
    denied_read_validation = await read.validate_input(denied_read_input, ctx)
    denied_read_permission = await read.check_permissions(
        denied_read_input,
        ctx,
        policy.permission_context,
    )

    assert isinstance(denied_read_validation, ValidationError)
    assert isinstance(denied_read_permission, PermissionDenyDecision)

    grep_content = _content(await _call_tool(grep, GrepInput(pattern="needle"), ctx))
    glob_content = _content(await _call_tool(glob, GlobInput(pattern="*.md"), ctx))

    assert "allowed.md" in grep_content
    assert "denied.md" not in grep_content
    assert "allowed.md" in glob_content
    assert "denied.md" not in glob_content


async def test_extraction_policy_allows_safe_bash_and_denies_unsafe_bash(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    deps = QueryDeps(task_store=AppStateStore())
    policy = build_extraction_tool_policy(
        build_default_extraction_tool_catalog(
            parent_tools=(),
            parent_deps=deps,
            settings=memory_settings,
        ),
        settings=memory_settings,
    )
    bash = find_tool_by_name(policy.tools, "Bash")
    assert bash is not None
    ctx = _ctx(tmp_path, policy.permission_context)

    safe = BashInput(command="ls")
    safe_validation = await bash.validate_input(safe, ctx)
    safe_permission = await bash.check_permissions(safe, ctx, policy.permission_context)

    assert isinstance(safe_validation, ValidationOk)
    assert isinstance(safe_permission, PermissionAllowDecision)
    assert bash.is_read_only(safe)
    assert not bash.is_destructive(safe)

    unsafe = BashInput(command="rm topic.md")
    unsafe_validation = await bash.validate_input(unsafe, ctx)
    unsafe_permission = await bash.check_permissions(
        unsafe,
        ctx,
        policy.permission_context,
    )

    assert isinstance(unsafe_validation, ValidationError)
    assert isinstance(unsafe_permission, PermissionDenyDecision)

    background = BashInput(command="ls", run_in_background=True)
    background_validation = await bash.validate_input(background, ctx)
    background_permission = await bash.check_permissions(
        background,
        ctx,
        policy.permission_context,
    )

    assert isinstance(background_validation, ValidationError)
    assert isinstance(background_permission, PermissionDenyDecision)
