"""Headless worktree service models.

Worktree isolation creates temporary working trees for child agents, runs
child filesystem work inside that cwd, then removes unchanged worktrees. Raygent
keeps the lifecycle as an injectable service seam so core remains product- and
VCS-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

WorktreeCleanupReason = Literal[
    "removed",
    "kept",
    "changed",
    "hook_based",
    "cleanup_failed",
]

WorktreeCleanupPolicy = Literal["remove_if_clean", "keep", "manual"]

WorktreeSweepSkipReason = Literal[
    "current_worktree",
    "not_stale",
    "stat_failed",
    "unsafe_path",
    "dirty",
    "unpushed",
    "git_failed",
    "remove_failed",
]


@dataclass(frozen=True)
class WorktreeInfo:
    """Worktree created for an isolated agent."""

    path: str
    branch: str | None = None
    head_commit: str | None = None
    git_root: str | None = None
    hook_based: bool = False
    slug: str | None = None
    created_at: float | None = None
    touched_at: float | None = None
    owner_task_id: str | None = None
    cleanup_policy: WorktreeCleanupPolicy = "remove_if_clean"


@dataclass(frozen=True)
class WorktreeCleanupResult:
    """Result of terminal worktree cleanup.

    `path`/`branch` are intentionally None when the worktree was removed. Kept
    or failed cleanup returns the path so model-visible notifications can tell
    the parent where isolated changes live.
    """

    kept: bool
    reason: WorktreeCleanupReason
    path: str | None = None
    branch: str | None = None


@dataclass(frozen=True)
class WorktreeSweepEntry:
    """One stale-sweep candidate outcome."""

    slug: str
    path: str
    branch: str | None = None
    reason: WorktreeSweepSkipReason | None = None


@dataclass(frozen=True)
class WorktreeSweepResult:
    """Metadata-only result from an opt-in stale worktree sweep."""

    scanned: int = 0
    removed: int = 0
    skipped: int = 0
    removed_entries: tuple[WorktreeSweepEntry, ...] = field(default_factory=tuple)
    skipped_entries: tuple[WorktreeSweepEntry, ...] = field(default_factory=tuple)


__all__ = [
    "WorktreeCleanupPolicy",
    "WorktreeCleanupReason",
    "WorktreeCleanupResult",
    "WorktreeInfo",
    "WorktreeSweepEntry",
    "WorktreeSweepResult",
    "WorktreeSweepSkipReason",
]
