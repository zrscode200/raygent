from __future__ import annotations

from pathlib import Path

from raygent_harness.skills.activation import (
    activated_skill_names_for_paths,
    discover_skill_dirs_for_paths,
    parse_skill_paths,
    skill_matches_path,
    split_path_in_frontmatter,
)
from raygent_harness.skills.models import SkillDefinition


def _skill(name: str, paths: tuple[str, ...]) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=name,
        markdown_content="",
        source="projectSettings",
        loaded_from="skills",
        content_length=0,
        paths=paths,
    )


def test_split_and_parse_skill_paths_matches_reference_shape() -> None:
    assert split_path_in_frontmatter("src/*.{py,md}, docs/**") == (
        "src/*.py",
        "src/*.md",
        "docs/**",
    )
    assert parse_skill_paths(["**"]) == ()
    assert parse_skill_paths("docs/**") == ("docs",)


def test_skill_matches_cwd_relative_paths_only(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    (cwd / "src").mkdir(parents=True)
    touched = cwd / "src" / "main.py"
    touched.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    skill = _skill("python", ("src/*.py",))

    assert skill_matches_path(skill, touched, cwd)
    assert skill_matches_path(skill, "src/main.py", cwd)
    assert not skill_matches_path(skill, outside, cwd)


def test_skill_globs_do_not_cross_path_segments_by_default(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    nested = cwd / "src" / "pkg" / "main.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")

    assert not skill_matches_path(_skill("python", ("src/*.py",)), nested, cwd)


def test_activated_skill_names_for_paths_is_pure(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    (cwd / "docs").mkdir(parents=True)
    readme = cwd / "docs" / "README.md"
    readme.write_text("", encoding="utf-8")

    assert activated_skill_names_for_paths(
        [_skill("docs", ("docs",)), _skill("src", ("src",))],
        [readme],
        cwd,
    ) == ("docs",)


def test_discover_skill_dirs_for_paths_returns_nested_dirs_deepest_first(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    shallow = cwd / "pkg" / ".claude" / "skills"
    deep = cwd / "pkg" / "src" / ".claude" / "skills"
    shallow.mkdir(parents=True)
    deep.mkdir(parents=True)
    touched = cwd / "pkg" / "src" / "main.py"
    touched.write_text("", encoding="utf-8")

    assert discover_skill_dirs_for_paths([touched], cwd) == (deep, shallow)


def test_discover_skill_dirs_for_paths_treats_relative_paths_as_cwd_relative(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    skills = cwd / "pkg" / ".claude" / "skills"
    skills.mkdir(parents=True)

    assert discover_skill_dirs_for_paths(["pkg/main.py"], cwd) == (skills,)
