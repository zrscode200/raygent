from __future__ import annotations

import asyncio
import shlex
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.permissions import PermissionAskDecision, ToolPermissionContext
from raygent_harness.core.stall_watchdog import QuiescenceWatchdog
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tasks.local_bash import read_local_bash_output, run_until_done
from raygent_harness.core.tasks.stop_task import stop_task
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.services.task_output import FileTaskOutputStore
from raygent_harness.tools.bash_tool import (
    BASH_MAX_RESULT_SIZE_CHARS,
    BASH_PROMPT,
    BASH_RESTRICTED_PROFILE_NAME,
    BASH_TOOL_ALIAS,
    BASH_TOOL_NAME,
    BashInput,
    build_bash_tool,
    create_bash_catalog_provider,
    validate_restricted_bash_command,
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
    agent_id: str | None = None,
) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        permission_context=permission_context or ToolPermissionContext(),
        tool_use_id="toolu_bash",
    )


def _deps(store: AppStateStore) -> QueryDeps:
    return QueryDeps(task_store=store)


async def _call_tool(
    tool: Tool,
    input_: BashInput,
    ctx: ToolUseContext,
) -> ToolResult:
    events = [event async for event in tool.call(input_, ctx)]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


def _blocks(result: ToolResult) -> list[dict[str, object]]:
    assert isinstance(result.content, list)
    return result.content


def _text(result: ToolResult) -> str:
    blocks = _blocks(result)
    first = blocks[0]
    assert first["type"] == "text"
    text = first["text"]
    assert isinstance(text, str)
    return text


def _quote(path: Path) -> str:
    return shlex.quote(str(path))


def test_bash_tool_axes_prompt_and_restricted_validator() -> None:
    tool = build_bash_tool(deps=_deps(AppStateStore()))
    input_ = BashInput(command="git status --short")

    assert tool.name == BASH_TOOL_NAME
    assert tool.aliases == (BASH_TOOL_ALIAS,)
    assert tool.is_concurrency_safe(input_) is False
    assert tool.is_read_only(input_) is False
    assert tool.is_destructive(input_) is True
    assert tool.is_open_world(input_) is True
    assert tool.interrupt_behavior() == "cancel"
    assert tool.max_result_size_chars == BASH_MAX_RESULT_SIZE_CHARS
    assert asyncio.run(tool.prompt()) == BASH_PROMPT

    for command in (
        "git status --short",
        "cat pyproject.toml | head -n 5",
        "rg TODO .",
        "find . -name '*.py'",
    ):
        assert validate_restricted_bash_command(command).allowed, command

    blocked = {
        "cat pyproject.toml > copy.txt": "redirects",
        "rm -rf build": "rm",
        "echo $(touch owned)": "command substitution",
        "cat $HOME/.ssh/id_rsa": "parameter expansion",
        "cat pyproject.toml | sh": "sh",
        "find . -delete": "-delete",
        "find . -fprint out.txt": "-fprint",
        "find . -fls out.txt": "-fls",
        "find . -fprintf out.txt '%p'": "-fprintf",
        "git add .": "add",
        "git -c diff.external=sh diff": "-c",
        "git --config-env=diff.external=EVIL diff": "--config-env",
        "git --exec-path=/tmp diff": "--exec-path",
        "git diff --output=out.patch": "--output",
        "git grep --open-files-in-pager=sh pattern": "--open-files-in-pager",
        "git remote prune origin": "prune",
        "rg --pre sh pattern .": "--pre",
        "sort -o out.txt input.txt": "-o",
    }
    for command, message_part in blocked.items():
        result = validate_restricted_bash_command(command)
        assert not result.allowed
        assert result.message is not None
        assert message_part in result.message


@pytest.mark.asyncio
async def test_bash_validates_restricted_profile_and_asks_permission(tmp_path: Path) -> None:
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store))
    ctx = _ctx(tmp_path)

    validation = await tool.validate_input(BashInput(command="rm -rf build"), ctx)
    assert isinstance(validation, ValidationError)
    assert BASH_RESTRICTED_PROFILE_NAME in validation.message

    permission = await tool.check_permissions(
        BashInput(command="git status --short"),
        ctx,
        ToolPermissionContext(),
    )
    assert isinstance(permission, PermissionAskDecision)
    assert BASH_RESTRICTED_PROFILE_NAME in permission.message


@pytest.mark.asyncio
async def test_bash_foreground_success_returns_output_and_suppresses_notification(
    tmp_path: Path,
) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)
    source = tmp_path / "hello.txt"
    source.write_text("hello\n", encoding="utf-8")

    result = await _call_tool(
        tool,
        BashInput(command=f"cat {_quote(source)}", output_read_bytes=100),
        _ctx(tmp_path),
    )

    assert not result.is_error
    text = _text(result)
    assert "Command completed (exit 0)." in text
    assert "hello" in text
    metadata = _blocks(result)[1]
    assert metadata["type"] == "bash_result"
    assert metadata["status"] == "completed"
    assert isinstance(metadata["output_file"], str)
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_foreground_nonzero_exit_is_error(tmp_path: Path) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)
    source = tmp_path / "data.txt"
    source.write_text("alpha\n", encoding="utf-8")

    result = await _call_tool(
        tool,
        BashInput(command=f"grep beta {_quote(source)}", output_read_bytes=100),
        _ctx(tmp_path),
    )

    assert result.is_error
    text = _text(result)
    assert "Command failed (exit 1)." in text
    metadata = _blocks(result)[1]
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 1
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_foreground_timeout_kills_process(tmp_path: Path) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)

    result = await _call_tool(
        tool,
        BashInput(command="tail -f /dev/null", timeout_s=0.1, output_read_bytes=100),
        _ctx(tmp_path),
    )

    assert result.is_error
    text = _text(result)
    assert "Command killed after timeout_s=" in text
    metadata = _blocks(result)[1]
    assert metadata["status"] == "killed"
    assert metadata["killed_by_timeout"] is True
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_foreground_quiet_command_does_not_emit_stall_notification(
    tmp_path: Path,
) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(
        deps=_deps(store),
        output_store=output_store,
        watchdog=QuiescenceWatchdog(check_interval_s=0.01, threshold_s=0.02),
    )

    result = await _call_tool(
        tool,
        BashInput(command="tail -f /dev/null", timeout_s=0.08, output_read_bytes=100),
        _ctx(tmp_path),
    )

    assert result.is_error
    metadata = _blocks(result)[1]
    assert metadata["status"] == "killed"
    assert metadata["killed_by_timeout"] is True
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_foreground_output_tail_is_bounded(tmp_path: Path) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)
    source = tmp_path / "large.txt"
    source.write_text("0123456789\n" * 20, encoding="utf-8")

    result = await _call_tool(
        tool,
        BashInput(command=f"cat {_quote(source)}", output_read_bytes=16),
        _ctx(tmp_path),
    )

    assert not result.is_error
    text = _text(result)
    assert "Output truncated: earlier output omitted" in text
    metadata = _blocks(result)[1]
    assert metadata["truncated_before"] is True
    assert metadata["bytes_total"] == source.stat().st_size


@pytest.mark.asyncio
async def test_bash_background_quiet_command_preserves_stall_notification(
    tmp_path: Path,
) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(
        deps=_deps(store),
        output_store=output_store,
        watchdog=QuiescenceWatchdog(check_interval_s=0.01, threshold_s=0.02),
    )

    result = await _call_tool(
        tool,
        BashInput(command="tail -f /dev/null", run_in_background=True, timeout_s=5),
        _ctx(tmp_path),
    )
    task_id = _blocks(result)[1]["task_id"]
    assert isinstance(task_id, str)

    notifications = []
    for _ in range(20):
        notifications = store.drain_notifications(None)
        if notifications:
            break
        await asyncio.sleep(0.01)

    assert len(notifications) == 1
    assert notifications[0].task_id == task_id
    assert notifications[0].kind == "stalled"

    await stop_task(task_id, store)
    final = await run_until_done(task_id, store)
    assert final.status == "killed"
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_background_returns_task_id_output_file_and_notification(
    tmp_path: Path,
) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)
    source = tmp_path / "hello.txt"
    source.write_text("background\n", encoding="utf-8")

    result = await _call_tool(
        tool,
        BashInput(command=f"cat {_quote(source)}", run_in_background=True),
        _ctx(tmp_path),
    )

    assert not result.is_error
    blocks = _blocks(result)
    task_block = blocks[1]
    assert task_block["type"] == "bash_task"
    task_id = task_block["task_id"]
    assert isinstance(task_id, str)
    assert isinstance(task_block["output_file"], str)
    assert "Command running in background" in _text(result)

    final = await run_until_done(task_id, store)
    assert final.status == "completed"
    output = await read_local_bash_output(task_id, store, output_store=output_store)
    assert b"background" in output.content

    notifications = store.drain_notifications(None)
    assert len(notifications) == 1
    assert notifications[0].task_id == task_id
    assert notifications[0].kind == "completed"
    assert str(task_block["output_file"]) in notifications[0].message


@pytest.mark.asyncio
async def test_bash_background_can_be_stopped_through_stop_task(tmp_path: Path) -> None:
    output_store = FileTaskOutputStore(tmp_path / "outputs", session_id="s")
    store = AppStateStore()
    tool = build_bash_tool(deps=_deps(store), output_store=output_store)

    result = await _call_tool(
        tool,
        BashInput(command="tail -f /dev/null", run_in_background=True, timeout_s=5),
        _ctx(tmp_path),
    )
    task_id = _blocks(result)[1]["task_id"]
    assert isinstance(task_id, str)

    stopped = await stop_task(task_id, store)
    assert stopped.task_id == task_id
    final = await run_until_done(task_id, store)
    assert final.status == "killed"
    assert final.notified is True
    assert store.drain_notifications(None) == []


@pytest.mark.asyncio
async def test_bash_catalog_provider_replaces_name_and_alias_collisions(
    tmp_path: Path,
) -> None:
    store = AppStateStore()
    existing = (
        _dummy_tool("Other"),
        _dummy_tool("OldBash", aliases=(BASH_TOOL_NAME,)),
        _dummy_tool(BASH_TOOL_ALIAS),
    )
    provider = create_bash_catalog_provider(parent_deps=_deps(store))

    tools = await provider(
        QueryConfig(model="m", tools=existing),
        _ctx(tmp_path),
        (),
    )

    assert tools is not None
    names = [tool.name for tool in tools]
    assert names == ["Other", BASH_TOOL_NAME]
