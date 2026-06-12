from __future__ import annotations

from pathlib import Path
from typing import Any

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.team_memory_sync import (
    MAX_FILE_SIZE_BYTES,
    read_local_team_memory,
    write_remote_entries_to_local,
)


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        team_memory_enabled=True,
        **kwargs,
    )


def test_read_local_team_memory_collects_flat_entries_and_empty_files(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    (team_dir / "nested").mkdir(parents=True)
    (team_dir / "MEMORY.md").write_text("index", encoding="utf-8")
    (team_dir / "nested" / "empty.md").write_text("", encoding="utf-8")

    result = read_local_team_memory(cfg)

    assert result.entries == {
        "MEMORY.md": "index",
        "nested/empty.md": "",
    }
    assert result.skipped_secrets == ()


def test_read_local_team_memory_skips_oversized_and_secret_files(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "safe.md").write_text("safe", encoding="utf-8")
    (team_dir / "big.md").write_text("x" * (MAX_FILE_SIZE_BYTES + 1), encoding="utf-8")
    (team_dir / "secret.md").write_text("token=ghp_" + "a" * 36, encoding="utf-8")

    result = read_local_team_memory(cfg)

    assert result.entries == {"safe.md": "safe"}
    assert [(item.path, item.rule_id, item.label) for item in result.skipped_secrets] == [
        ("secret.md", "github-pat", "GitHub PAT")
    ]


def test_read_local_team_memory_applies_deterministic_learned_cap(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "c.md").write_text("c", encoding="utf-8")
    (team_dir / "a.md").write_text("a", encoding="utf-8")
    (team_dir / "b.md").write_text("b", encoding="utf-8")

    result = read_local_team_memory(cfg, max_entries=2)

    assert result.entries == {"a.md": "a", "b.md": "b"}


def test_write_remote_entries_to_local_validates_paths_and_skips_oversized(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    entries = {
        "nested/a.md": "one",
        "../evil.md": "evil",
        "big.md": "x" * (MAX_FILE_SIZE_BYTES + 1),
    }

    assert write_remote_entries_to_local(entries, cfg) == 1
    assert (team_dir / "nested" / "a.md").read_text(encoding="utf-8") == "one"
    assert not (team_dir / "big.md").exists()
    assert not (team_dir.parent / "evil.md").exists()


def test_write_remote_entries_to_local_elides_unchanged_files(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    target = team_dir / "MEMORY.md"
    target.parent.mkdir(parents=True)
    target.write_text("same", encoding="utf-8")

    assert write_remote_entries_to_local({"MEMORY.md": "same"}, cfg) == 0
    assert write_remote_entries_to_local({"MEMORY.md": "changed"}, cfg) == 1
    assert target.read_text(encoding="utf-8") == "changed"
