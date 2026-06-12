from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.services.worktree import (
    GitWorktreeManager,
    WorktreeInfo,
    is_ephemeral_worktree_slug,
    worktree_branch_name,
)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "init", "--initial-branch", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "HEAD:main")
    return repo


async def _create_old_worktree(
    manager: GitWorktreeManager,
    repo: Path,
    slug: str,
) -> WorktreeInfo:
    info = await manager.create_agent_worktree(slug, cwd=str(repo))
    old = time.time() - 60 * 60 * 24 * 45
    os.utime(info.path, (old, old))
    return info


@pytest.mark.asyncio
async def test_git_worktree_manager_sweeps_old_clean_ephemeral_worktree(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    slug = "agent-aabcdef0"
    info = await _create_old_worktree(manager, repo, slug)
    sink = RecordingKernelEventSink()

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
        observability=KernelEventBus([sink]),
    )

    assert result.scanned == 1
    assert result.removed == 1
    assert result.skipped == 0
    assert result.removed_entries[0].slug == slug
    assert not Path(info.path).exists()
    assert _git(repo, "branch", "--list", worktree_branch_name(slug)).strip() == ""
    assert sink.events[-1].type == "worktree.stale_sweep.completed"
    assert sink.events[-1].data["removed"] == 1


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_dirty_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "agent-aabcdef1")
    Path(info.path, "file.txt").write_text("dirty\n", encoding="utf-8")

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "dirty"
    assert Path(info.path).exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_unpushed_commits(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "agent-aabcdef2")
    worktree = Path(info.path)
    worktree.joinpath("file.txt").write_text("new commit\n", encoding="utf-8")
    _git(worktree, "add", "file.txt")
    _git(worktree, "commit", "-m", "worktree commit")
    old = time.time() - 60 * 60 * 24 * 45
    os.utime(info.path, (old, old))

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "unpushed"
    assert worktree.exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_non_ephemeral_user_slug(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "wf-myfeature")

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.scanned == 0
    assert result.removed == 0
    assert Path(info.path).exists()
    assert is_ephemeral_worktree_slug("wf-myfeature") is False


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_current_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "agent-aabcdef3")

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
        current_worktree_path=info.path,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "current_worktree"
    assert Path(info.path).exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_bad_current_path_does_not_abort_sweep(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "agent-aabcdef9")
    loop = tmp_path / "current-loop"
    try:
        loop.symlink_to(loop)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
        current_worktree_path=str(loop),
    )

    assert result.removed == 1
    assert not Path(info.path).exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_unsafe_non_directory_candidate(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    worktrees_dir = repo / manager.worktrees_dir_name
    worktrees_dir.mkdir(parents=True)
    fake_candidate = worktrees_dir / "agent-aabcdef4"
    fake_candidate.write_text("not a directory\n", encoding="utf-8")
    old = time.time() - 60 * 60 * 24 * 45
    os.utime(fake_candidate, (old, old))

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "unsafe_path"
    assert fake_candidate.exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_symlink_escape(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    outside = tmp_path / "outside-target"
    outside.mkdir()
    worktrees_dir = repo / manager.worktrees_dir_name
    worktrees_dir.mkdir(parents=True)
    symlink_candidate = worktrees_dir / "agent-aabcdef4"
    try:
        symlink_candidate.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "unsafe_path"
    assert symlink_candidate.is_symlink()
    assert outside.exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_skips_git_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    info = await _create_old_worktree(manager, repo, "agent-aabcdef6")

    async def fake_git(cwd: str, *args: str) -> tuple[int, str, str]:
        if args == ("rev-parse", "--path-format=absolute", "--git-common-dir"):
            return 0, f"{repo / '.git'}\n", ""
        if args == ("rev-parse", "--show-toplevel"):
            return 0, f"{repo}\n", ""
        if args == ("--no-optional-locks", "status", "--porcelain", "-uno"):
            return 1, "", "boom"
        return 0, "", ""

    monkeypatch.setattr("raygent_harness.services.worktree.manager._git", fake_git)

    result = await manager.cleanup_stale_worktrees(
        cwd=str(repo),
        cutoff_time=time.time() - 60 * 60,
    )

    assert result.removed == 0
    assert result.skipped_entries[0].reason == "git_failed"
    assert Path(info.path).exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_uses_canonical_root_from_nested_worktree(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()
    parent = await manager.create_agent_worktree("agent-aabcdeaa", cwd=str(repo))
    child = await manager.create_agent_worktree("agent-aabcdeab", cwd=parent.path)
    old = time.time() - 60 * 60 * 24 * 45
    os.utime(child.path, (old, old))

    main_worktrees_dir = (repo / manager.worktrees_dir_name).resolve()

    assert Path(child.path).parent == main_worktrees_dir
    assert child.slug is not None
    assert not (Path(parent.path) / manager.worktrees_dir_name / child.slug).exists()

    result = await manager.cleanup_stale_worktrees(
        cwd=parent.path,
        cutoff_time=time.time() - 60 * 60,
        current_worktree_path=parent.path,
    )

    assert result.removed == 1
    assert result.removed_entries[0].slug == "agent-aabcdeab"
    assert not Path(child.path).exists()


@pytest.mark.asyncio
async def test_git_worktree_manager_create_records_resume_metadata(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    manager = GitWorktreeManager()

    info = await manager.create_agent_worktree("agent-aabcdef5", cwd=str(repo))

    assert info.slug == "agent-aabcdef5"
    assert info.branch == "worktree-agent-aabcdef5"
    assert info.created_at is not None
    assert info.touched_at is not None
    assert info.cleanup_policy == "remove_if_clean"
