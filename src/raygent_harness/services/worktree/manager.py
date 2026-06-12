"""Worktree manager protocol and small git-backed implementation."""

from __future__ import annotations

import asyncio
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from raygent_harness.core.observability import KernelEventBus, KernelEventContext
from raygent_harness.services.worktree.models import (
    WorktreeCleanupResult,
    WorktreeInfo,
    WorktreeSweepEntry,
    WorktreeSweepResult,
    WorktreeSweepSkipReason,
)

_SAFE_SLUG = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_EPHEMERAL_WORKTREE_PATTERNS = (
    re.compile(r"^agent-a[0-9a-f]{7}$"),
    re.compile(r"^wf_[0-9a-f]{8}-[0-9a-f]{3}-\d+$"),
    re.compile(r"^wf-\d+$"),
    re.compile(r"^bridge-[A-Za-z0-9_]+(-[A-Za-z0-9_]+)*$"),
    re.compile(r"^job-[A-Za-z0-9._-]{1,55}-[0-9a-f]{8}$"),
)


class WorktreeManager(Protocol):
    """Create/check/cleanup isolated agent worktrees."""

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        """Create a worktree for `slug` rooted from `cwd`."""
        ...

    async def has_changes(self, info: WorktreeInfo) -> bool:
        """Return True when cleanup should keep the worktree."""
        ...

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        """Remove unchanged worktrees and keep changed/hook-based worktrees."""
        ...


class StaleWorktreeSweeper(Protocol):
    """Optional extension for managers that can safely sweep stale worktrees."""

    async def cleanup_stale_worktrees(
        self,
        *,
        cwd: str,
        cutoff_time: float,
        current_worktree_path: str | None = None,
        observability: KernelEventBus | None = None,
    ) -> WorktreeSweepResult:
        """Remove old clean Raygent ephemeral worktrees."""
        ...


@dataclass(frozen=True)
class GitWorktreeManager:
    """Minimal stdlib git worktree manager.

    This intentionally avoids reference product hooks, sparse checkout, and
    network fetches. It creates a branch from the current HEAD and fails closed
    by keeping the worktree if change detection or cleanup fails.
    """

    worktrees_dir_name: str = ".raygent/worktrees"

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        _validate_slug(slug)
        git_root = await _canonical_git_root(cwd)
        if git_root is None:
            raise RuntimeError(
                "Cannot create agent worktree: current cwd is not inside a git "
                "repository. Provide a custom WorktreeManager for non-git VCS."
            )
        head_commit = await _git_stdout(git_root, "rev-parse", "HEAD")
        if head_commit is None:
            raise RuntimeError("Cannot create agent worktree: failed to resolve HEAD.")

        worktrees_dir = (Path(git_root) / self.worktrees_dir_name).resolve()
        worktrees_dir.mkdir(parents=True, exist_ok=True)
        path = str(worktrees_dir / slug)
        branch = worktree_branch_name(slug)

        code, _stdout, stderr = await _git(
            git_root,
            "worktree",
            "add",
            "-B",
            branch,
            path,
            "HEAD",
        )
        if code != 0:
            raise RuntimeError(f"Failed to create agent worktree: {stderr.strip()}")
        timestamps = await _path_timestamps(path)
        return WorktreeInfo(
            path=path,
            branch=branch,
            head_commit=head_commit,
            git_root=git_root,
            slug=slug,
            created_at=timestamps[0],
            touched_at=timestamps[1],
            cleanup_policy="remove_if_clean",
        )

    async def has_changes(self, info: WorktreeInfo) -> bool:
        code, stdout, _stderr = await _git(info.path, "status", "--porcelain")
        if code != 0:
            return True
        if stdout.strip():
            return True
        if not info.head_commit:
            return True
        code, stdout, _stderr = await _git(
            info.path,
            "rev-list",
            "--count",
            f"{info.head_commit}..HEAD",
        )
        if code != 0:
            return True
        try:
            return int(stdout.strip() or "0") > 0
        except ValueError:
            return True

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        if info.hook_based:
            return WorktreeCleanupResult(
                kept=True,
                reason="hook_based",
                path=info.path,
                branch=info.branch,
            )
        if keep is True or info.cleanup_policy == "keep":
            return WorktreeCleanupResult(
                kept=True,
                reason="kept",
                path=info.path,
                branch=info.branch,
            )
        if await self.has_changes(info):
            return WorktreeCleanupResult(
                kept=True,
                reason="changed",
                path=info.path,
                branch=info.branch,
            )

        git_root = info.git_root or os.path.dirname(info.path)
        code, _stdout, _stderr = await _git(
            git_root,
            "worktree",
            "remove",
            "--force",
            info.path,
        )
        if code != 0:
            return WorktreeCleanupResult(
                kept=True,
                reason="cleanup_failed",
                path=info.path,
                branch=info.branch,
            )
        if info.branch:
            await _git(git_root, "branch", "-D", info.branch)
        return WorktreeCleanupResult(kept=False, reason="removed")

    async def cleanup_stale_worktrees(
        self,
        *,
        cwd: str,
        cutoff_time: float,
        current_worktree_path: str | None = None,
        observability: KernelEventBus | None = None,
    ) -> WorktreeSweepResult:
        """Remove old clean Raygent ephemeral worktrees.

        This is opt-in and fail-closed by construction. It scans only this
        manager's configured worktree directory, only considers exact ephemeral
        slug patterns, and skips candidates on any uncertainty.
        """
        git_root = await _canonical_git_root(cwd)
        if git_root is None:
            result = WorktreeSweepResult()
            _emit_sweep_completed(observability, result, reason="git_root_missing")
            return result

        worktrees_dir = (Path(git_root) / self.worktrees_dir_name).resolve()
        try:
            entries = sorted(await asyncio.to_thread(_list_dir_names, worktrees_dir))
        except OSError:
            result = WorktreeSweepResult()
            _emit_sweep_completed(observability, result, reason="worktree_dir_missing")
            return result

        current_path = _resolved_optional(current_worktree_path)
        removed_entries: list[WorktreeSweepEntry] = []
        skipped_entries: list[WorktreeSweepEntry] = []
        scanned = 0

        for slug in entries:
            if not is_ephemeral_worktree_slug(slug):
                continue
            scanned += 1
            candidate_path = worktrees_dir / slug
            branch = worktree_branch_name(slug)
            path = _safe_candidate_path(candidate_path, worktrees_dir)
            if path is None:
                skipped_entries.append(
                    _skipped(slug, candidate_path, branch, reason="unsafe_path")
                )
                continue
            if current_path is not None and path == current_path:
                skipped_entries.append(
                    _skipped(slug, path, branch, reason="current_worktree")
                )
                continue

            try:
                stat_result = await asyncio.to_thread(candidate_path.lstat)
            except OSError:
                skipped_entries.append(_skipped(slug, path, branch, reason="stat_failed"))
                continue
            if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISDIR(
                stat_result.st_mode
            ):
                skipped_entries.append(_skipped(slug, path, branch, reason="unsafe_path"))
                continue
            if stat_result.st_mtime >= cutoff_time:
                skipped_entries.append(_skipped(slug, path, branch, reason="not_stale"))
                continue

            skip_reason = await _stale_candidate_skip_reason(path)
            if skip_reason is not None:
                skipped_entries.append(_skipped(slug, path, branch, reason=skip_reason))
                continue

            code, _stdout, _stderr = await _git(
                git_root,
                "worktree",
                "remove",
                "--force",
                str(path),
            )
            if code != 0:
                skipped_entries.append(_skipped(slug, path, branch, reason="remove_failed"))
                continue
            await _git(git_root, "branch", "-D", branch)
            removed_entries.append(
                WorktreeSweepEntry(slug=slug, path=str(path), branch=branch)
            )

        if removed_entries:
            await _git(git_root, "worktree", "prune")

        result = WorktreeSweepResult(
            scanned=scanned,
            removed=len(removed_entries),
            skipped=len(skipped_entries),
            removed_entries=tuple(removed_entries),
            skipped_entries=tuple(skipped_entries),
        )
        _emit_sweep_completed(observability, result, reason=None)
        return result


def worktree_branch_name(slug: str) -> str:
    """Return Raygent's deterministic branch name for a worktree slug."""

    return f"worktree-{slug}"


def is_ephemeral_worktree_slug(slug: str) -> bool:
    """Return True for exact Raygent/reference ephemeral worktree slugs."""

    return any(pattern.fullmatch(slug) is not None for pattern in _EPHEMERAL_WORKTREE_PATTERNS)


def _safe_candidate_path(candidate_path: Path, worktrees_dir: Path) -> Path | None:
    try:
        resolved = candidate_path.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(worktrees_dir)
    except ValueError:
        return None
    return resolved


async def _canonical_git_root(cwd: str) -> str | None:
    common_dir = await _git_stdout(
        cwd,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    )
    if common_dir is not None:
        common_path = Path(common_dir).resolve()
        if common_path.name == ".git":
            return str(common_path.parent)

    return await _git_stdout(cwd, "rev-parse", "--show-toplevel")


def _validate_slug(slug: str) -> None:
    if not _SAFE_SLUG.fullmatch(slug):
        raise ValueError(
            "Worktree slug must contain only letters, numbers, '.', '_', or '-'."
        )


async def _stale_candidate_skip_reason(path: Path) -> WorktreeSweepSkipReason | None:
    status_code, status_stdout, _status_stderr = await _git(
        str(path),
        "--no-optional-locks",
        "status",
        "--porcelain",
        "-uno",
    )
    if status_code != 0:
        return "git_failed"
    if status_stdout.strip():
        return "dirty"

    unpushed_code, unpushed_stdout, _unpushed_stderr = await _git(
        str(path),
        "rev-list",
        "--max-count=1",
        "HEAD",
        "--not",
        "--remotes",
    )
    if unpushed_code != 0:
        return "git_failed"
    if unpushed_stdout.strip():
        return "unpushed"
    return None


def _skipped(
    slug: str,
    path: Path,
    branch: str,
    *,
    reason: WorktreeSweepSkipReason,
) -> WorktreeSweepEntry:
    return WorktreeSweepEntry(
        slug=slug,
        path=str(path),
        branch=branch,
        reason=reason,
    )


def _emit_sweep_completed(
    observability: KernelEventBus | None,
    result: WorktreeSweepResult,
    *,
    reason: str | None,
) -> None:
    if observability is None:
        return
    observability.emit(
        "worktree.stale_sweep.completed",
        context=KernelEventContext(source="worktree"),
        data={
            "scanned": result.scanned,
            "removed": result.removed,
            "skipped": result.skipped,
            "reason": reason,
            "removed_slugs": tuple(entry.slug for entry in result.removed_entries),
            "skip_reasons": tuple(
                entry.reason for entry in result.skipped_entries if entry.reason is not None
            ),
        },
    )


def _resolved_optional(path: str | None) -> Path | None:
    if path is None:
        return None
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _list_dir_names(path: Path) -> list[str]:
    return [entry.name for entry in path.iterdir()]


async def _path_timestamps(path: str) -> tuple[float, float]:
    try:
        stat_result = await asyncio.to_thread(Path(path).stat)
    except OSError:
        now = time.time()
        return now, now
    return stat_result.st_ctime, stat_result.st_mtime


async def _git_stdout(cwd: str, *args: str) -> str | None:
    code, stdout, _stderr = await _git(cwd, *args)
    if code != 0:
        return None
    return stdout.strip()


async def _git(cwd: str, *args: str) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return 1, "", str(exc)
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


__all__ = [
    "GitWorktreeManager",
    "StaleWorktreeSweeper",
    "WorktreeManager",
    "is_ephemeral_worktree_slug",
    "worktree_branch_name",
]
