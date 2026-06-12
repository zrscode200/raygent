"""Bundled skill registry and safe reference-file extraction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from raygent_harness.skills.models import BundledSkillDefinition, SkillDefinition


class BundledSkillRegistry:
    """In-process registry for programmatic bundled skills."""

    def __init__(self) -> None:
        self._definitions: dict[str, BundledSkillDefinition] = {}
        self._extractions: dict[tuple[str, Path], Path | None] = {}

    def register(self, definition: BundledSkillDefinition) -> None:
        self._definitions[definition.name] = definition

    def get_skills(self, extraction_root: str | Path | None = None) -> tuple[SkillDefinition, ...]:
        root = Path(extraction_root) if extraction_root is not None else None
        return tuple(
            definition.to_skill(
                get_bundled_skill_extract_dir(root, definition.name)
                if root is not None and definition.files
                else None
            )
            for definition in self._definitions.values()
        )

    def clear(self) -> None:
        self._definitions.clear()
        self._extractions.clear()

    def render_prompt(
        self,
        name: str,
        *,
        args: str = "",
        extraction_root: str | Path | None = None,
    ) -> str:
        definition = self._definitions[name]
        prompt = definition.prompt.replace("$ARGUMENTS", args)
        if not definition.files or extraction_root is None:
            return prompt

        root = Path(extraction_root)
        key = (name, root)
        if key not in self._extractions:
            self._extractions[key] = extract_bundled_skill_files(
                name,
                definition.files,
                root,
            )
        extracted_dir = self._extractions[key]
        if extracted_dir is None:
            return prompt
        return f"Base directory for this skill: {extracted_dir}\n\n{prompt}"


DEFAULT_BUNDLED_SKILL_REGISTRY = BundledSkillRegistry()


def register_bundled_skill(definition: BundledSkillDefinition) -> None:
    DEFAULT_BUNDLED_SKILL_REGISTRY.register(definition)


def get_bundled_skills(
    extraction_root: str | Path | None = None,
) -> tuple[SkillDefinition, ...]:
    return DEFAULT_BUNDLED_SKILL_REGISTRY.get_skills(extraction_root)


def clear_bundled_skills() -> None:
    DEFAULT_BUNDLED_SKILL_REGISTRY.clear()


def get_bundled_skill_extract_dir(root: str | Path, skill_name: str) -> Path:
    return Path(root) / skill_name


def extract_bundled_skill_files(
    skill_name: str,
    files: Mapping[str, str],
    extraction_root: str | Path,
) -> Path | None:
    """Safely extract bundled reference files once per caller.

    Invalid traversal paths or write errors return `None`, matching the
    reference's "skill continues without base-dir prefix" behavior.
    """

    target_dir = get_bundled_skill_extract_dir(extraction_root, skill_name)
    try:
        _write_skill_files(target_dir, files)
    except OSError:
        return None
    except ValueError:
        return None
    return target_dir


def resolve_bundled_skill_file_path(base_dir: str | Path, relative_path: str) -> Path:
    """Normalize and validate a bundled-skill relative file path."""

    if "\\" in relative_path:
        raise ValueError(f"bundled skill file path escapes skill dir: {relative_path}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part == ".." for part in pure.parts):
        raise ValueError(f"bundled skill file path escapes skill dir: {relative_path}")
    if any(part in {"", "."} for part in pure.parts):
        raise ValueError(f"invalid bundled skill file path: {relative_path}")
    return Path(base_dir).joinpath(*pure.parts)


def _write_skill_files(target_dir: Path, files: Mapping[str, str]) -> None:
    grouped: dict[Path, list[tuple[Path, str]]] = {}
    for relative_path, content in files.items():
        target = resolve_bundled_skill_file_path(target_dir, relative_path)
        grouped.setdefault(target.parent, []).append((target, content))

    for parent, entries in grouped.items():
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        for target, content in entries:
            _safe_write_file(target, content)


def _safe_write_file(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
    finally:
        if fd != -1:
            os.close(fd)


__all__ = [
    "DEFAULT_BUNDLED_SKILL_REGISTRY",
    "BundledSkillRegistry",
    "clear_bundled_skills",
    "extract_bundled_skill_files",
    "get_bundled_skill_extract_dir",
    "get_bundled_skills",
    "register_bundled_skill",
    "resolve_bundled_skill_file_path",
]
