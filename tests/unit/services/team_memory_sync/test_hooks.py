from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.core.tool_hooks import PostToolUseContext, PreToolUseContext
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.team_memory_sync import (
    create_team_memory_post_tool_use_hook,
    create_team_memory_pre_tool_use_hook,
    create_team_memory_write_hooks,
)


class WriteInput(BaseModel):
    file_path: str
    content: str


class EditInput(BaseModel):
    file_path: str
    new_string: str


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        team_memory_enabled=True,
        **kwargs,
    )


def ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def tool_use(name: str = "Write") -> ToolUseBlock:
    return ToolUseBlock(id="toolu_1", name=name, input={}, index=0)


async def call_ok(_input: BaseModel, _ctx: ToolUseContext) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def write_tool(name: str = "Write") -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description="write",
            input_model=WriteInput,
            call=call_ok,
        )
    )


def edit_tool() -> Tool:
    return build_tool(
        ToolSpec(
            name="Edit",
            description="edit",
            input_model=EditInput,
            call=call_ok,
        )
    )


def pre_context(
    *,
    tool: Tool,
    input_model: BaseModel,
    tmp_path: Path,
) -> PreToolUseContext:
    return PreToolUseContext(
        tool=tool,
        tool_use=tool_use(tool.name),
        input=input_model,
        tool_use_context=ctx(tmp_path),
        assistant_message=cast("MessageParam", {"role": "assistant", "content": ""}),
    )


def post_context(
    *,
    tool: Tool,
    input_model: BaseModel,
    tmp_path: Path,
    is_error: bool = False,
) -> PostToolUseContext:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "ok",
    }
    if is_error:
        block["is_error"] = True
    return PostToolUseContext(
        tool=tool,
        tool_use=tool_use(tool.name),
        input=input_model,
        tool_use_context=ctx(tmp_path),
        assistant_message=cast("MessageParam", {"role": "assistant", "content": ""}),
        result_message=cast("MessageParam", {"role": "user", "content": [block]}),
    )


async def test_pre_tool_hook_blocks_secret_team_memory_write(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    target = get_team_mem_path(cfg) / "MEMORY.md"
    hook = create_team_memory_pre_tool_use_hook(cfg)

    result = await hook(
        pre_context(
            tool=write_tool(),
            input_model=WriteInput(
                file_path=str(target),
                content="token=ghp_" + "a" * 36,
            ),
            tmp_path=tmp_path,
        )
    )

    assert result is not None
    assert result.stop is True
    assert result.stop_reason is not None
    assert "GitHub PAT" in result.stop_reason


async def test_pre_tool_hook_supports_edit_new_string_and_relative_paths(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    target_dir = get_team_mem_path(cfg)
    relative_path = target_dir.relative_to(tmp_path)
    hook = create_team_memory_pre_tool_use_hook(cfg)

    result = await hook(
        pre_context(
            tool=edit_tool(),
            input_model=EditInput(
                file_path=str(relative_path / "MEMORY.md"),
                new_string="token=ghp_" + "b" * 36,
            ),
            tmp_path=tmp_path,
        )
    )

    assert result is not None
    assert result.stop is True


async def test_pre_tool_hook_ignores_outside_paths_and_unregistered_tools(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    hook = create_team_memory_pre_tool_use_hook(cfg)

    outside = await hook(
        pre_context(
            tool=write_tool(),
            input_model=WriteInput(
                file_path=str(tmp_path / "outside.md"),
                content="token=ghp_" + "c" * 36,
            ),
            tmp_path=tmp_path,
        )
    )
    unregistered = await hook(
        pre_context(
            tool=write_tool("NotAWriteTool"),
            input_model=WriteInput(
                file_path=str(get_team_mem_path(cfg) / "MEMORY.md"),
                content="token=ghp_" + "d" * 36,
            ),
            tmp_path=tmp_path,
        )
    )

    assert outside is None
    assert unregistered is None


async def test_post_tool_hook_notifies_after_successful_team_memory_write(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    calls = 0

    async def notify() -> bool:
        nonlocal calls
        calls += 1
        return True

    hook = create_team_memory_post_tool_use_hook(cfg, notify)
    await hook(
        post_context(
            tool=write_tool(),
            input_model=WriteInput(
                file_path=str(get_team_mem_path(cfg) / "MEMORY.md"),
                content="project note",
            ),
            tmp_path=tmp_path,
        )
    )

    assert calls == 1


async def test_post_tool_hook_skips_error_and_non_team_results(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    calls = 0

    async def notify() -> bool:
        nonlocal calls
        calls += 1
        return True

    hook = create_team_memory_post_tool_use_hook(cfg, notify)
    await hook(
        post_context(
            tool=write_tool(),
            input_model=WriteInput(
                file_path=str(get_team_mem_path(cfg) / "MEMORY.md"),
                content="project note",
            ),
            tmp_path=tmp_path,
            is_error=True,
        )
    )
    await hook(
        post_context(
            tool=write_tool(),
            input_model=WriteInput(
                file_path=str(tmp_path / "outside.md"),
                content="project note",
            ),
            tmp_path=tmp_path,
        )
    )

    assert calls == 0


def test_create_team_memory_write_hooks_returns_pre_and_post_hooks(tmp_path: Path) -> None:
    cfg = settings(tmp_path)

    pre_hook, post_hook = create_team_memory_write_hooks(cfg, lambda: call_notify())

    assert pre_hook is not None
    assert post_hook is not None


async def call_notify() -> bool:
    return True
