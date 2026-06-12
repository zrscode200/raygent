"""Path activation helpers for conditional skills.

Skill activation uses a segment-aware glob subset here. The core contract is:
patterns are cwd-relative, absolute/outside-cwd files do not match, and `/**`
suffixes collapse to the directory pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

from raygent_harness.skills.models import SkillDefinition


def split_path_in_frontmatter(input_value: object) -> tuple[str, ...]:
    """Split comma-separated or list path frontmatter with brace expansion."""

    if isinstance(input_value, str):
        return tuple(
            expanded
            for part in _split_commas_outside_braces(input_value)
            for expanded in _expand_braces(part)
            if expanded
        )
    if isinstance(input_value, list | tuple):
        values: list[str] = []
        for item in cast(Sequence[object], input_value):
            values.extend(split_path_in_frontmatter(item))
        return tuple(values)
    return ()


def parse_skill_paths(frontmatter_value: object) -> tuple[str, ...]:
    """Parse `paths:` frontmatter.

    Match-all entries (`**`) are treated as no conditional activation.
    """

    patterns = tuple(
        pattern[:-3] if pattern.endswith("/**") else pattern
        for pattern in split_path_in_frontmatter(frontmatter_value)
        if pattern
    )
    if not patterns or all(pattern == "**" for pattern in patterns):
        return ()
    return patterns


def skill_matches_path(skill: SkillDefinition, file_path: str | Path, cwd: str | Path) -> bool:
    """Return whether a conditional skill should activate for a file path."""

    if not skill.paths:
        return False

    relative_path = _relative_to_cwd(Path(file_path), Path(cwd))
    if relative_path is None:
        return False

    return any(_matches_pattern(relative_path, pattern) for pattern in skill.paths)


def activated_skill_names_for_paths(
    skills: Iterable[SkillDefinition],
    file_paths: Iterable[str | Path],
    cwd: str | Path,
) -> tuple[str, ...]:
    """Pure helper: which conditional skills activate for the touched paths."""

    touched = tuple(file_paths)
    activated: list[str] = []
    for skill in skills:
        if any(skill_matches_path(skill, path, cwd) for path in touched):
            activated.append(skill.name)
    return tuple(activated)


def discover_skill_dirs_for_paths(
    file_paths: Iterable[str | Path],
    cwd: str | Path,
) -> tuple[Path, ...]:
    """Discover nested `.claude/skills` dirs from touched files, deepest first."""

    cwd_path = Path(cwd).resolve()
    seen: set[Path] = set()
    discovered: list[Path] = []
    for file_path in file_paths:
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = cwd_path / candidate
        try:
            current = candidate.resolve().parent
        except OSError:
            current = candidate.absolute().parent

        while _is_strict_relative_to(current, cwd_path):
            skills_dir = current / ".claude" / "skills"
            if skills_dir not in seen:
                seen.add(skills_dir)
                if skills_dir.exists() and skills_dir.is_dir():
                    discovered.append(skills_dir)
            parent = current.parent
            if parent == current:
                break
            current = parent

    return tuple(
        sorted(discovered, key=lambda path: len(path.parts), reverse=True)
    )


def _split_commas_outside_braces(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    current: list[str] = []
    brace_depth = 0
    for char in value:
        if char == "{":
            brace_depth += 1
            current.append(char)
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
        elif char == "," and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return tuple(parts)


def _expand_braces(pattern: str) -> tuple[str, ...]:
    start = pattern.find("{")
    if start == -1:
        return (pattern,)
    end = pattern.find("}", start + 1)
    if end == -1:
        return (pattern,)

    prefix = pattern[:start]
    alternatives = pattern[start + 1 : end]
    suffix = pattern[end + 1 :]
    expanded: list[str] = []
    for alternative in alternatives.split(","):
        expanded.extend(_expand_braces(f"{prefix}{alternative.strip()}{suffix}"))
    return tuple(expanded)


def _relative_to_cwd(path: Path, cwd: Path) -> str | None:
    absolute_path = path if path.is_absolute() else cwd / path
    try:
        relative = absolute_path.resolve().relative_to(cwd.resolve())
    except (OSError, ValueError):
        try:
            relative = absolute_path.absolute().relative_to(cwd.absolute())
        except ValueError:
            return None
    if not relative.parts:
        return None
    return relative.as_posix()


def _matches_pattern(relative_path: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    if _segment_glob_matches(relative_path, normalized):
        return True
    if _has_glob_syntax(normalized):
        return False
    return relative_path == normalized or relative_path.startswith(f"{normalized}/")


def _segment_glob_matches(relative_path: str, pattern: str) -> bool:
    return re.fullmatch(_segment_glob_to_regex(pattern), relative_path) is not None


def _segment_glob_to_regex(pattern: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                parts.append(".*")
                index += 2
                continue
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


def _has_glob_syntax(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _is_strict_relative_to(path: Path, cwd: Path) -> bool:
    try:
        path.relative_to(cwd)
    except ValueError:
        return False
    return path != cwd


__all__ = [
    "activated_skill_names_for_paths",
    "discover_skill_dirs_for_paths",
    "parse_skill_paths",
    "skill_matches_path",
    "split_path_in_frontmatter",
]
