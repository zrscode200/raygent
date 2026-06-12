from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from raygent_harness.memdir.paths import (
    AUTO_MEM_ENTRYPOINT_NAME,
    MemorySettings,
    get_auto_mem_path,
)
from raygent_harness.memdir.team_paths import (
    TEAM_MEM_DIRNAME,
    PathTraversalError,
    get_team_mem_entrypoint,
    get_team_mem_path,
    is_team_mem_file,
    is_team_mem_path,
    is_team_memory_enabled,
    validate_team_mem_key,
    validate_team_mem_write_path,
)


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        **kwargs,
    )


def test_team_memory_enabled_requires_auto_memory_and_team_gate(tmp_path: Path) -> None:
    assert is_team_memory_enabled(settings(tmp_path)) is False
    assert is_team_memory_enabled(settings(tmp_path, team_memory_enabled=True)) is True
    assert (
        is_team_memory_enabled(
            settings(tmp_path, team_memory_enabled=True, disable_auto_memory="1")
        )
        is False
    )


def test_team_memory_path_shape_and_entrypoint(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    assert get_team_mem_path(cfg) == get_auto_mem_path(cfg) / TEAM_MEM_DIRNAME
    assert get_team_mem_entrypoint(cfg) == get_team_mem_path(cfg) / AUTO_MEM_ENTRYPOINT_NAME


def test_is_team_mem_path_uses_directory_membership(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    team_dir = get_team_mem_path(cfg)

    assert is_team_mem_path(team_dir, cfg) is False
    assert is_team_mem_path(team_dir / "nested" / ".." / "topic.md", cfg) is True
    assert is_team_mem_path(str(team_dir) + "-sibling/topic.md", cfg) is False
    assert is_team_mem_path(str(team_dir) + "\0tail", cfg) is False
    assert is_team_mem_file(team_dir / "topic.md", cfg) is True
    assert is_team_mem_file(team_dir / "topic.md", settings(tmp_path)) is False


def test_validate_team_mem_key_accepts_safe_nested_keys(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    assert validate_team_mem_key("MEMORY.md", cfg) == get_team_mem_path(cfg) / "MEMORY.md"
    assert (
        validate_team_mem_key("nested/patterns.md", cfg)
        == get_team_mem_path(cfg) / "nested" / "patterns.md"
    )


@pytest.mark.parametrize(
    "key",
    [
        "../evil.md",
        "",
        ".",
        "/absolute.md",
        "nested\\evil.md",
        "safe%2Fencoded.md",
        "%2e%2e%2fsecret.md",
        "fullwidth\uff0e\uff0e\uff0fevil.md",
        "null\0byte.md",
    ],
)
def test_validate_team_mem_key_rejects_traversal_and_injection(
    tmp_path: Path, key: str
) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    with pytest.raises(PathTraversalError):
        validate_team_mem_key(key, cfg)


def test_validate_team_mem_write_path_accepts_safe_paths(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    team_dir = get_team_mem_path(cfg)
    target = team_dir / "nested" / "topic.md"

    assert validate_team_mem_write_path(target, cfg) == target


def test_validate_team_mem_write_path_rejects_siblings_and_nulls(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    team_dir = get_team_mem_path(cfg)

    with pytest.raises(PathTraversalError):
        validate_team_mem_write_path(str(team_dir) + "-sibling/topic.md", cfg)
    with pytest.raises(PathTraversalError):
        validate_team_mem_write_path(team_dir, cfg)
    with pytest.raises(PathTraversalError):
        validate_team_mem_write_path(str(team_dir / "topic.md") + "\0tail", cfg)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_validate_team_mem_key_rejects_symlink_escape(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    team_dir = get_team_mem_path(cfg)
    outside = tmp_path / "outside"
    team_dir.mkdir(parents=True)
    outside.mkdir()
    os.symlink(outside, team_dir / "link")

    with pytest.raises(PathTraversalError):
        validate_team_mem_key("link/escaped.md", cfg)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_validate_team_mem_key_rejects_dangling_symlink(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    os.symlink(tmp_path / "missing-target", team_dir / "dangling")

    with pytest.raises(PathTraversalError):
        validate_team_mem_key("dangling/escaped.md", cfg)
