from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from raygent_harness.memdir.paths import (
    MemorySettings,
    get_auto_mem_base,
    get_auto_mem_entrypoint,
    get_auto_mem_path,
    get_auto_mem_path_override,
    get_auto_mem_path_setting,
    get_memory_base_dir,
    has_auto_mem_path_override,
    is_auto_mem_path,
    is_auto_memory_enabled,
    is_extract_mode_active,
    sanitize_path,
    validate_memory_path,
)


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "my repo",
        home_dir=tmp_path / "home",
        **kwargs,
    )


def test_is_auto_memory_enabled_matches_reference_priority(tmp_path: Path) -> None:
    assert is_auto_memory_enabled(settings(tmp_path)) is True
    assert is_auto_memory_enabled(settings(tmp_path, disable_auto_memory="1")) is False
    assert is_auto_memory_enabled(settings(tmp_path, disable_auto_memory="true")) is False
    assert is_auto_memory_enabled(settings(tmp_path, disable_auto_memory="0")) is True
    assert is_auto_memory_enabled(settings(tmp_path, simple_mode=True)) is False
    assert is_auto_memory_enabled(settings(tmp_path, remote_mode=True)) is False
    assert (
        is_auto_memory_enabled(
            settings(tmp_path, remote_mode=True, remote_memory_dir=tmp_path / "remote")
        )
        is True
    )
    assert is_auto_memory_enabled(settings(tmp_path, auto_memory_enabled=False)) is False


def test_extract_mode_gate_separates_feature_from_interactive_mode() -> None:
    assert is_extract_mode_active(feature_enabled=False) is False
    assert is_extract_mode_active(feature_enabled=True) is True
    assert is_extract_mode_active(feature_enabled=True, non_interactive=True) is False
    assert (
        is_extract_mode_active(
            feature_enabled=True, non_interactive=True, allow_non_interactive=True
        )
        is True
    )


def test_validate_memory_path_rejects_unsafe_candidates(tmp_path: Path) -> None:
    home = tmp_path / "home"
    rejected = [
        None,
        "",
        "relative/path",
        "/",
        "/a",
        "C:",
        "C:/",
        "//server/share",
        "\\\\server\\share",
        "/tmp/evil\0tail",
        "~",
        "~/",
        "~/.",
        "~/..",
        "~/foo/../..",
    ]
    for raw in rejected:
        assert validate_memory_path(raw, expand_tilde=True, home_dir=home) is None


def test_validate_memory_path_normalizes_and_expands_trusted_tilde(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert validate_memory_path("~/memory/../mem", expand_tilde=True, home_dir=home) == home / "mem"
    assert validate_memory_path("~/mem", expand_tilde=False, home_dir=home) is None
    assert (
        validate_memory_path(str(tmp_path / "a" / ".." / "b"), expand_tilde=False, home_dir=home)
        == tmp_path / "b"
    )


def test_auto_mem_path_resolution_order_and_entrypoint(tmp_path: Path) -> None:
    override = tmp_path / "override" / "memory"
    setting_path = tmp_path / "setting" / "memory"
    base = tmp_path / "base"

    with_override = settings(
        tmp_path,
        memory_base_dir=base,
        auto_memory_path_override=str(override),
        auto_memory_directory=str(setting_path),
    )
    assert has_auto_mem_path_override(with_override) is True
    assert get_auto_mem_path_override(with_override) == override
    assert get_auto_mem_path(with_override) == override

    with_setting = settings(
        tmp_path,
        memory_base_dir=base,
        auto_memory_path_override="relative/bad",
        auto_memory_directory=str(setting_path),
    )
    assert has_auto_mem_path_override(with_setting) is False
    assert get_auto_mem_path_setting(with_setting) == setting_path
    assert get_auto_mem_path(with_setting) == setting_path

    defaulted = settings(tmp_path, memory_base_dir=base)
    expected = base / "projects" / sanitize_path(str(get_auto_mem_base(defaulted))) / "memory"
    assert get_memory_base_dir(defaulted) == base
    assert get_auto_mem_path(defaulted) == expected
    assert get_auto_mem_entrypoint(defaulted) == expected / "MEMORY.md"


def test_auto_mem_path_uses_canonical_project_root_when_supplied(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    canonical = tmp_path / "repo.git"
    cfg = MemorySettings(
        project_root=root,
        canonical_project_root=canonical,
        memory_base_dir=tmp_path / "base",
        home_dir=tmp_path / "home",
    )
    assert get_auto_mem_base(cfg) == canonical
    assert sanitize_path(str(canonical)) in str(get_auto_mem_path(cfg))


def test_is_auto_mem_path_is_normalized_directory_membership(tmp_path: Path) -> None:
    cfg = settings(tmp_path, memory_base_dir=tmp_path / "base")
    memory_dir = get_auto_mem_path(cfg)

    assert is_auto_mem_path(memory_dir, cfg) is True
    assert is_auto_mem_path(memory_dir / "topic.md", cfg) is True
    assert is_auto_mem_path(memory_dir / "nested" / ".." / "topic.md", cfg) is True
    assert is_auto_mem_path(str(memory_dir) + "-sibling/topic.md", cfg) is False
    assert is_auto_mem_path(Path("relative/topic.md"), cfg) is False
    assert is_auto_mem_path(str(memory_dir) + "\0tail", cfg) is False


def test_sanitize_path_matches_reference_shape_and_hashes_long_values() -> None:
    assert sanitize_path("/Users/foo/my-project") == "-Users-foo-my-project"
    assert sanitize_path("plugin:name:server") == "plugin-name-server"
    assert sanitize_path("repo😀x") == "repo--x"

    raw = "/" + ("a" * 220)
    sanitized = sanitize_path(raw)
    assert len(sanitized) > 200
    assert sanitized.startswith("-" + ("a" * 199))
    assert sanitized[200] == "-"

    astral_raw = "/" + ("a" * 198) + "😀"
    astral_sanitized = sanitize_path(astral_raw)
    assert astral_sanitized.startswith("-" + ("a" * 198) + "-")
    assert astral_sanitized[200] == "-"
    assert len(astral_sanitized) > 200


def test_invalid_config_base_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe memory path"):
        get_memory_base_dir(settings(tmp_path, memory_base_dir=Path("relative")))
