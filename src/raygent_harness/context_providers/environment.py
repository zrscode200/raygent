"""Environment and git context providers.

Reference grounding:
  context;

Raygent keeps both as opt-in providers outside `core` so embedders can choose
cwd/workspace/git policy without changing the query loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from asyncio import StreamReader
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import (
    ContextAgentScope,
    ContextFragment,
    ContextKind,
    context_agent_scope_includes,
)
from raygent_harness.core.tool import ToolUseContext

type TodayProvider = Callable[[], date]
type GitCommandRunner = Callable[
    [tuple[str, ...], Path, float, int],
    Awaitable["GitCommandResult"],
]


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Bounded git command result returned by a git command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True, slots=True)
class EnvironmentContextProvider:
    """Emit model-agnostic environment facts as system context."""

    context_kind: ClassVar[ContextKind] = "environment"

    cwd: str | Path | None = None
    workspace_root: str | Path | None = None
    is_git_repo: bool | None = None
    include_model: bool = True
    today: TodayProvider = date.today
    platform_name: str | None = None
    max_chars: int = 4000
    priority: int = 0
    agent_scope: ContextAgentScope = "all"
    fragment_id: str = "environment"

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        if not context_agent_scope_includes(self.agent_scope, agent_id=ctx.agent_id):
            return ()

        cwd = _resolve_path(self.cwd or ctx.cwd)
        git_root = _find_git_root(cwd)
        workspace_root = _resolve_path(self.workspace_root) if self.workspace_root else git_root
        if workspace_root is None:
            workspace_root = cwd

        is_git_repo = self.is_git_repo
        if is_git_repo is None:
            is_git_repo = git_root is not None

        lines = [
            "Here is some useful information about the environment you are running in:",
            "<env>",
            f"  Working directory: {cwd}",
            f"  Workspace root folder: {workspace_root}",
            f"  Is directory a git repo: {'yes' if is_git_repo else 'no'}",
            f"  Platform: {self.platform_name or sys.platform}",
            f"  Today's date: {self.today().isoformat()}",
        ]
        if self.include_model and config.model:
            lines.append(f"  Configured model: {config.model}")
        lines.append("</env>")

        content = _truncate_chars("\n".join(lines), self.max_chars)
        return (
            ContextFragment(
                id=self.fragment_id,
                content=content,
                channel="system",
                source="environment",
                priority=self.priority,
                agent_scope=self.agent_scope,
                kind=self.context_kind,
            ),
        )


@dataclass(frozen=True, slots=True)
class GitStatusContextProvider:
    """Emit a bounded git-context snapshot as system context."""

    context_kind: ClassVar[ContextKind] = "git"

    cwd: str | Path | None = None
    command_runner: GitCommandRunner = field(
        default_factory=lambda: default_git_command_runner
    )
    include_branch: bool = True
    include_default_branch: bool = True
    include_git_user: bool = True
    include_status: bool = True
    include_recent_commits: bool = True
    recent_commit_count: int = 5
    timeout_s: float = 2.0
    max_output_bytes: int = 2000
    priority: int = 10
    agent_scope: ContextAgentScope = "main"
    fragment_id: str = "git-status"

    async def __call__(
        self,
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        if not context_agent_scope_includes(self.agent_scope, agent_id=ctx.agent_id):
            return ()

        cwd = _resolve_path(self.cwd or ctx.cwd)
        if not await _is_inside_git_work_tree(self, cwd):
            return ()

        commands: dict[str, asyncio.Task[GitCommandResult | None]] = {}
        if self.include_branch:
            commands["branch"] = asyncio.create_task(
                _run_git_command(
                    self,
                    ("--no-optional-locks", "rev-parse", "--abbrev-ref", "HEAD"),
                    cwd,
                )
            )
        if self.include_default_branch:
            commands["default_branch"] = asyncio.create_task(
                _run_git_command(
                    self,
                    (
                        "--no-optional-locks",
                        "symbolic-ref",
                        "--short",
                        "refs/remotes/origin/HEAD",
                    ),
                    cwd,
                )
            )
        if self.include_git_user:
            commands["git_user"] = asyncio.create_task(
                _run_git_command(self, ("config", "user.name"), cwd)
            )
        if self.include_status:
            commands["status"] = asyncio.create_task(
                _run_git_command(
                    self,
                    ("--no-optional-locks", "status", "--short"),
                    cwd,
                )
            )
        if self.include_recent_commits and self.recent_commit_count > 0:
            commands["recent_commits"] = asyncio.create_task(
                _run_git_command(
                    self,
                    (
                        "--no-optional-locks",
                        "log",
                        "--oneline",
                        "-n",
                        str(self.recent_commit_count),
                    ),
                    cwd,
                )
            )
        if commands:
            await asyncio.gather(*commands.values())

        sections: list[str] = []

        if "branch" in commands:
            branch = _git_result_text(commands["branch"].result())
            if branch:
                sections.append(f"Current branch: {branch}")

        if "default_branch" in commands:
            default_branch = _git_result_text(commands["default_branch"].result())
            if default_branch:
                sections.append(
                    "Main branch (you will usually use this for PRs): "
                    f"{_normalize_default_branch(default_branch)}"
                )

        if "git_user" in commands:
            git_user = _git_result_text(commands["git_user"].result())
            if git_user:
                sections.append(f"Git user: {git_user}")

        if "status" in commands:
            status_result = commands["status"].result()
            if status_result is not None:
                status = status_result.stdout.strip() or "(clean)"
                if status_result.stdout_truncated:
                    status = (
                        f"{status}\n"
                        "... (truncated because git status exceeded "
                        f"{self.max_output_bytes} bytes; "
                        'run "git status --short" for more.)'
                    )
                sections.append(f"Status:\n{status}")

        if "recent_commits" in commands:
            recent_commits = _git_result_text(commands["recent_commits"].result())
            if recent_commits:
                sections.append(f"Recent commits:\n{recent_commits}")

        if not sections:
            return ()

        content = "\n".join(
            [
                "This is the git context at the start of this turn. "
                "It is a snapshot in time and will not update during the turn.",
                "<git_status>",
                "\n\n".join(sections),
                "</git_status>",
            ]
        )
        return (
            ContextFragment(
                id=self.fragment_id,
                content=content,
                channel="system",
                source="git",
                priority=self.priority,
                agent_scope=self.agent_scope,
                kind=self.context_kind,
            ),
        )


async def _is_inside_git_work_tree(
    provider: GitStatusContextProvider,
    cwd: Path,
) -> bool:
    result = await _run_git_command(
        provider,
        ("--no-optional-locks", "rev-parse", "--is-inside-work-tree"),
        cwd,
    )
    if result is None:
        return False
    return result.stdout.strip().lower() == "true"


def _git_result_text(result: GitCommandResult | None) -> str:
    if result is None:
        return ""
    return result.stdout.strip()


async def _run_git_command(
    provider: GitStatusContextProvider,
    args: tuple[str, ...],
    cwd: Path,
) -> GitCommandResult | None:
    try:
        # Runner receives the timeout for process cleanup; this outer guard
        # prevents a custom runner from wedging context resolution forever.
        result = await asyncio.wait_for(
            provider.command_runner(
                args,
                cwd,
                provider.timeout_s,
                provider.max_output_bytes,
            ),
            timeout=_runner_guard_timeout(provider.timeout_s),
        )
    except TimeoutError:
        return None
    except Exception:
        return None

    if result.timed_out:
        return None
    if result.returncode != 0 and not result.stdout.strip():
        return None
    return result


def _normalize_default_branch(branch: str) -> str:
    normalized = branch.strip()
    if normalized.startswith("origin/"):
        return normalized.removeprefix("origin/")
    return normalized


async def default_git_command_runner(
    args: tuple[str, ...],
    cwd: Path,
    timeout_s: float,
    max_output_bytes: int,
) -> GitCommandResult:
    """Run git with timeout and bounded stdout/stderr buffers."""

    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(_read_limited(process.stdout, max_output_bytes))
    stderr_task = asyncio.create_task(
        _read_limited(process.stderr, min(max_output_bytes, 4096))
    )
    wait_task = asyncio.create_task(process.wait())
    timed_out = False

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while not wait_task.done():
            if _task_truncated(stdout_task) or _task_truncated(stderr_task):
                _kill_process(process)
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                _kill_process(process)
                break
            pending = {
                task
                for task in (wait_task, stdout_task, stderr_task)
                if not task.done()
            }
            if not pending:
                break
            done, _ = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                _kill_process(process)
                break

        with contextlib.suppress(Exception):
            await asyncio.wait_for(wait_task, timeout=0.5)
    finally:
        if not wait_task.done():
            _kill_process(process)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(wait_task, timeout=0.5)

    stdout, stdout_truncated = await stdout_task
    stderr, stderr_truncated = await stderr_task
    return GitCommandResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def _task_truncated(task: asyncio.Task[tuple[bytes, bool]]) -> bool:
    return task.done() and not task.cancelled() and task.exception() is None and task.result()[1]


async def _read_limited(
    stream: StreamReader | None,
    max_bytes: int,
) -> tuple[bytes, bool]:
    if stream is None or max_bytes <= 0:
        return b"", False

    chunks = bytearray()
    while True:
        remaining = max_bytes - len(chunks)
        read_size = min(4096, max(remaining + 1, 1))
        chunk = await stream.read(read_size)
        if not chunk:
            return bytes(chunks), False
        if len(chunk) > remaining:
            chunks.extend(chunk[:remaining])
            return bytes(chunks), True
        chunks.extend(chunk)


def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()


def _runner_guard_timeout(timeout_s: float) -> float:
    cleanup_grace = min(0.5, max(0.05, timeout_s * 0.1))
    return max(timeout_s, 0.0) + cleanup_grace


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _find_git_root(cwd: Path) -> Path | None:
    current = cwd if cwd.is_dir() else cwd.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _truncate_chars(content: str, max_chars: int) -> str:
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}\n... (truncated)"


__all__ = [
    "EnvironmentContextProvider",
    "GitCommandResult",
    "GitCommandRunner",
    "GitStatusContextProvider",
    "default_git_command_runner",
]
