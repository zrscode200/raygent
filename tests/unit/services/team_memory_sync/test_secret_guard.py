from __future__ import annotations

from pathlib import Path
from typing import Any

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.team_memory_sync.secret_guard import check_team_mem_secrets


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        **kwargs,
    )


def test_secret_guard_blocks_team_memory_path_even_when_runtime_sync_is_disabled(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    target = get_team_mem_path(cfg) / "MEMORY.md"

    result = check_team_mem_secrets(target, "ghp_" + "a" * 36, cfg)

    assert result is not None
    assert "GitHub PAT" in result


def test_secret_guard_ignores_paths_outside_team_memory(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    assert check_team_mem_secrets(tmp_path / "other.md", "ghp_" + "a" * 36, cfg) is None


def test_secret_guard_blocks_team_memory_writes_with_secret_labels(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    target = get_team_mem_path(cfg) / "MEMORY.md"

    result = check_team_mem_secrets(target, "token=ghp_" + "a" * 36, cfg)

    assert result == (
        "Content contains potential secrets (GitHub PAT) and cannot be written to team memory. "
        "Team memory is shared with all repository collaborators. "
        "Remove the sensitive content and try again."
    )
