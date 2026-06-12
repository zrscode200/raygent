from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from raygent_harness.context_providers import (
    GitCommandResult,
    build_default_context_providers,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import message_param_from_api_message
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from tests.fakes import FakeModelProvider


def _ctx(*, cwd: str | Path = ".", agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


async def _git_runner(
    args: tuple[str, ...],
    _cwd: Path,
    _timeout_s: float,
    _max_output_bytes: int,
) -> GitCommandResult:
    if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
        return GitCommandResult(returncode=0, stdout="true\n")
    if args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
        return GitCommandResult(returncode=0, stdout="main\n")
    if args == (
        "--no-optional-locks",
        "symbolic-ref",
        "--short",
        "refs/remotes/origin/HEAD",
    ):
        return GitCommandResult(returncode=0, stdout="origin/main\n")
    if args == ("config", "user.name"):
        return GitCommandResult(returncode=0, stdout="Rui\n")
    if args == ("--no-optional-locks", "status", "--short"):
        return GitCommandResult(returncode=0, stdout=" M src/app.py\n")
    if args == ("--no-optional-locks", "log", "--oneline", "-n", "5"):
        return GitCommandResult(returncode=0, stdout="abc123 init\n")
    return GitCommandResult(returncode=1, stderr="unexpected")


@pytest.mark.asyncio
async def test_default_context_provider_stack_is_opt_in_and_non_persistent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "Project policy\n@./extra.md")
    _write(root / "extra.md", "Included policy")
    _write(root / ".claude" / "rules" / "general.md", "Rule policy")
    model = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=build_default_context_providers(
            cwd=root,
            workspace_root=root,
            today=lambda: date(2026, 5, 27),
            git_command_runner=_git_runner,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", system_prompt="base"),
        deps,
        _ctx(cwd=root),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    request = model.requests[0]
    assert request.system_prompt is not None
    assert request.system_prompt.startswith("base\n\n")
    assert "<env>" in request.system_prompt
    assert "<git_status>" in request.system_prompt
    request_messages = [
        message_param_from_api_message(message) for message in request.messages
    ]
    assert len(request_messages) == 2
    instruction_context = str(request_messages[0]["content"])
    assert "Project policy" in instruction_context
    assert "Included policy" in instruction_context
    assert "Rule policy" in instruction_context
    assert request_messages[1] == {"role": "user", "content": "hi"}
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_default_context_provider_stack_has_explicit_subagent_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "Project policy")
    model = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    git_calls = 0

    async def git_runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        nonlocal git_calls
        git_calls += 1
        return GitCommandResult(returncode=0, stdout="## main\n M src/app.py\n")

    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=build_default_context_providers(
            cwd=root,
            workspace_root=root,
            today=lambda: date(2026, 5, 27),
            git_command_runner=git_runner,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", agent_id="child"),
        deps,
        _ctx(cwd=root, agent_id="child"),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    system_prompt = model.requests[0].system_prompt or ""
    assert "<env>" in system_prompt
    assert "<git_status>" not in system_prompt
    assert git_calls == 0
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    assert "Project policy" in str(request_messages[0]["content"])


@pytest.mark.asyncio
async def test_default_context_provider_stack_can_omit_project_context_for_subagents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "Project policy")
    model = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=build_default_context_providers(
            cwd=root,
            workspace_root=root,
            today=lambda: date(2026, 5, 27),
            git_command_runner=_git_runner,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            project_instruction_agent_scope="main",
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", agent_id="child"),
        deps,
        _ctx(cwd=root, agent_id="child"),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    assert request_messages == [{"role": "user", "content": "hi"}]
