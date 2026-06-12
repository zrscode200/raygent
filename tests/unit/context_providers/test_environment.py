from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from time import monotonic

import pytest

from raygent_harness.context_providers import (
    EnvironmentContextProvider,
    GitCommandResult,
    GitStatusContextProvider,
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


@pytest.mark.asyncio
async def test_environment_context_provider_formats_model_agnostic_env_block(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    cwd = tmp_path / "src"
    cwd.mkdir()

    provider = EnvironmentContextProvider(
        today=lambda: date(2026, 5, 26),
        platform_name="test-platform",
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx(cwd=cwd))

    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.channel == "system"
    assert fragment.source == "environment"
    assert fragment.agent_scope == "all"
    assert "<env>" in fragment.content
    assert f"Working directory: {cwd.resolve()}" in fragment.content
    assert f"Workspace root folder: {tmp_path.resolve()}" in fragment.content
    assert "Is directory a git repo: yes" in fragment.content
    assert "Platform: test-platform" in fragment.content
    assert "Today's date: 2026-05-26" in fragment.content
    assert "Configured model: model-1" in fragment.content


@pytest.mark.asyncio
async def test_environment_context_provider_allows_explicit_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    provider = EnvironmentContextProvider(
        workspace_root=workspace,
        is_git_repo=False,
        include_model=False,
        today=lambda: date(2026, 5, 26),
        platform_name="test-platform",
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))

    assert f"Workspace root folder: {workspace.resolve()}" in fragments[0].content
    assert "Is directory a git repo: no" in fragments[0].content
    assert "Configured model" not in fragments[0].content

    short_provider = EnvironmentContextProvider(
        today=lambda: date(2026, 5, 26),
        platform_name="test-platform",
        max_chars=10,
    )
    short_fragments = await short_provider(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))
    assert "truncated" in short_fragments[0].content


@pytest.mark.asyncio
async def test_git_status_provider_uses_injected_runner_and_formats_snapshot(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], Path, float, int]] = []

    async def runner(
        args: tuple[str, ...],
        cwd: Path,
        timeout_s: float,
        max_output_bytes: int,
    ) -> GitCommandResult:
        calls.append((args, cwd, timeout_s, max_output_bytes))
        if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        if args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
            return GitCommandResult(returncode=0, stdout="feature\n")
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
            return GitCommandResult(returncode=0, stdout="abc123 initial\n")
        return GitCommandResult(returncode=1, stderr="unexpected")

    provider = GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=runner,
        timeout_s=1.5,
        max_output_bytes=123,
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx(cwd="/elsewhere"))

    assert calls[0] == (
        ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"),
        tmp_path.resolve(),
        1.5,
        123,
    )
    assert {call[0] for call in calls[1:]} == {
        ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"),
        (
            "--no-optional-locks",
            "symbolic-ref",
            "--short",
            "refs/remotes/origin/HEAD",
        ),
        ("config", "user.name"),
        ("--no-optional-locks", "status", "--short"),
        ("--no-optional-locks", "log", "--oneline", "-n", "5"),
    }
    assert all(call[1:] == (tmp_path.resolve(), 1.5, 123) for call in calls[1:])
    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.channel == "system"
    assert fragment.source == "git"
    assert fragment.agent_scope == "main"
    assert "<git_status>" in fragment.content
    assert "Current branch: feature" in fragment.content
    assert "Main branch (you will usually use this for PRs): main" in fragment.content
    assert "Git user: Rui" in fragment.content
    assert "M src/app.py" in fragment.content
    assert "Recent commits:\nabc123 initial" in fragment.content
    assert "snapshot in time" in fragment.content


@pytest.mark.asyncio
async def test_git_status_provider_allows_section_level_configuration(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def runner(
        args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        calls.append(args)
        if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        return GitCommandResult(returncode=0, stdout=" M src/app.py\n")

    fragments = await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=runner,
        include_branch=False,
        include_default_branch=False,
        include_git_user=False,
        include_recent_commits=False,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))

    assert calls == [
        ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"),
        ("--no-optional-locks", "status", "--short"),
    ]
    assert "Status:\nM src/app.py" in fragments[0].content
    assert "Current branch" not in fragments[0].content
    assert "Recent commits" not in fragments[0].content


@pytest.mark.asyncio
async def test_git_status_provider_fails_soft_on_timeout_error_and_non_git(
    tmp_path: Path,
) -> None:
    async def timed_out_runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        return GitCommandResult(returncode=-9, timed_out=True)

    async def failing_runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        raise RuntimeError("git unavailable")

    async def non_git_runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        return GitCommandResult(returncode=128, stderr="not a git repo")

    assert await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=timed_out_runner,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path)) == ()
    assert await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=failing_runner,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path)) == ()
    assert await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=non_git_runner,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path)) == ()


@pytest.mark.asyncio
async def test_git_status_provider_skips_non_git_even_if_global_user_exists(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def runner(
        args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        calls.append(args)
        if args == ("config", "user.name"):
            return GitCommandResult(returncode=0, stdout="Global User\n")
        return GitCommandResult(returncode=128, stderr="not a git repo")

    fragments = await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=runner,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))

    assert fragments == ()
    assert calls == [("--no-optional-locks", "rev-parse", "--is-inside-work-tree")]


@pytest.mark.asyncio
async def test_git_status_provider_enforces_outer_timeout_on_hanging_runner(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def hanging_runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        if _args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        started.set()
        await asyncio.sleep(60)
        return GitCommandResult(returncode=0, stdout="should not return")

    start = monotonic()
    fragments = await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=hanging_runner,
        timeout_s=0.01,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))
    elapsed = monotonic() - start

    assert started.is_set()
    assert fragments == ()
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_git_status_provider_marks_truncated_output(tmp_path: Path) -> None:
    async def runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        if _args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        return GitCommandResult(
            returncode=0,
            stdout="## main\n" + (" M file.py\n" * 5),
            stdout_truncated=True,
        )

    fragments = await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=runner,
        max_output_bytes=12,
        include_branch=False,
        include_default_branch=False,
        include_git_user=False,
        include_recent_commits=False,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))

    assert "truncated because git status exceeded 12 bytes" in fragments[0].content
    assert 'run "git status --short" for more' in fragments[0].content


@pytest.mark.asyncio
async def test_git_status_provider_reports_detached_head(
    tmp_path: Path,
) -> None:
    async def runner(
        args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        if args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
            return GitCommandResult(returncode=0, stdout="HEAD\n")
        return GitCommandResult(returncode=1, stderr="disabled")

    fragments = await GitStatusContextProvider(
        cwd=tmp_path,
        command_runner=runner,
        include_default_branch=False,
        include_git_user=False,
        include_status=False,
        include_recent_commits=False,
    )(QueryConfig(model="model-1"), _ctx(cwd=tmp_path))

    assert "Current branch: HEAD" in fragments[0].content


@pytest.mark.asyncio
async def test_environment_and_git_providers_integrate_with_query_engine(
    tmp_path: Path,
) -> None:
    async def runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        if _args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        if _args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
            return GitCommandResult(returncode=0, stdout="main\n")
        if _args == (
            "--no-optional-locks",
            "symbolic-ref",
            "--short",
            "refs/remotes/origin/HEAD",
        ):
            return GitCommandResult(returncode=0, stdout="origin/main\n")
        if _args == ("config", "user.name"):
            return GitCommandResult(returncode=0, stdout="Rui\n")
        if _args == ("--no-optional-locks", "status", "--short"):
            return GitCommandResult(returncode=0, stdout=" M src/app.py\n")
        if _args == ("--no-optional-locks", "log", "--oneline", "-n", "5"):
            return GitCommandResult(returncode=0, stdout="abc123 init\n")
        return GitCommandResult(returncode=1, stderr="unexpected")

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        context_providers=(
            EnvironmentContextProvider(
                cwd=tmp_path,
                today=lambda: date(2026, 5, 26),
                platform_name="test-platform",
            ),
            GitStatusContextProvider(cwd=tmp_path, command_runner=runner),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", system_prompt="base"),
        deps,
        _ctx(cwd=tmp_path),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    request = provider.requests[0]
    assert request.system_prompt is not None
    assert request.system_prompt.startswith("base\n\n")
    assert "<env>" in request.system_prompt
    assert "<git_status>" in request.system_prompt
    assert "Current branch: main" in request.system_prompt
    assert "Recent commits:\nabc123 init" in request.system_prompt
    request_messages = [
        message_param_from_api_message(message) for message in request.messages
    ]
    assert request_messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_git_context_default_scope_omits_subagent_requests(tmp_path: Path) -> None:
    async def runner(
        _args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        return GitCommandResult(returncode=0, stdout="## main\n M src/app.py\n")

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        context_providers=(
            EnvironmentContextProvider(
                cwd=tmp_path,
                today=lambda: date(2026, 5, 26),
                platform_name="test-platform",
            ),
            GitStatusContextProvider(cwd=tmp_path, command_runner=runner),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", agent_id="child-agent"),
        deps,
        _ctx(cwd=tmp_path, agent_id="child-agent"),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    system_prompt = provider.requests[0].system_prompt or ""
    assert "<env>" in system_prompt
    assert "<git_status>" not in system_prompt


@pytest.mark.asyncio
async def test_git_context_can_be_included_for_subagents(tmp_path: Path) -> None:
    async def runner(
        args: tuple[str, ...],
        _cwd: Path,
        _timeout_s: float,
        _max_output_bytes: int,
    ) -> GitCommandResult:
        if args == ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"):
            return GitCommandResult(returncode=0, stdout="true\n")
        if args == ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"):
            return GitCommandResult(returncode=0, stdout="main\n")
        if args == ("--no-optional-locks", "status", "--short"):
            return GitCommandResult(returncode=0, stdout=" M src/app.py\n")
        return GitCommandResult(returncode=1, stderr="disabled")

    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "ok"},)
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        context_providers=(
            GitStatusContextProvider(
                cwd=tmp_path,
                command_runner=runner,
                agent_scope="all",
            ),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s", agent_id="child-agent"),
        deps,
        _ctx(cwd=tmp_path, agent_id="child-agent"),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    assert "<git_status>" in (provider.requests[0].system_prompt or "")
