"""Memory file scan and manifest formatting.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from raygent_harness.memdir.memdir import ENTRYPOINT_NAME
from raygent_harness.memdir.memory_types import MemoryType, parse_memory_type

MAX_MEMORY_FILES = 200
FRONTMATTER_MAX_LINES = 30


@dataclass(frozen=True)
class MemoryHeader:
    """Header metadata for one memory file."""

    filename: str
    file_path: Path
    mtime_ms: float
    description: str | None
    type: MemoryType | None


def _read_first_lines(path: Path, max_lines: int) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= max_lines:
                break
            normalized = line.rstrip("\n").removesuffix("\r")
            if index == 0:
                normalized = normalized.removeprefix("\ufeff")
            lines.append(normalized)
    return "\n".join(lines)


def _strip_yaml_scalar(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_memory_frontmatter(content: str) -> dict[str, str]:
    """Parse simple YAML frontmatter from the scanned header range.

    This intentionally supports the small subset the memory subsystem needs:
    a leading `---` block with scalar `key: value` lines. Invalid or missing
    frontmatter returns an empty dict, matching the reference's fail-soft scan.
    """
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return frontmatter
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        frontmatter[normalized_key] = _strip_yaml_scalar(value)

    return {}


def _scan_one(memory_dir: Path, path: Path) -> MemoryHeader:
    content = _read_first_lines(path, FRONTMATTER_MAX_LINES)
    stat = path.stat()
    frontmatter = parse_memory_frontmatter(content)
    description = frontmatter.get("description") or None
    return MemoryHeader(
        filename=path.relative_to(memory_dir).as_posix(),
        file_path=path,
        mtime_ms=stat.st_mtime * 1000,
        description=description,
        type=parse_memory_type(frontmatter.get("type")),
    )


def scan_memory_files(memory_dir: Path | str) -> list[MemoryHeader]:
    """Scan memory files and return newest-first headers capped at 200.

    Directory-level and per-file failures return/skip quietly. The scan reads
    only the first 30 lines of each candidate file, matching the reference's
    header-only strategy.
    """
    root = Path(memory_dir)
    try:
        candidates = [
            path
            for path in root.rglob("*.md")
            if path.name != ENTRYPOINT_NAME
        ]
    except OSError:
        return []

    headers: list[MemoryHeader] = []
    for path in candidates:
        try:
            headers.append(_scan_one(root, path))
        except Exception:
            continue

    return sorted(headers, key=lambda item: item.mtime_ms, reverse=True)[:MAX_MEMORY_FILES]


def _iso_from_mtime_ms(mtime_ms: float) -> str:
    return datetime.fromtimestamp(mtime_ms / 1000, tz=UTC).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def format_memory_manifest(memories: list[MemoryHeader]) -> str:
    """Format headers as `[type] filename (timestamp): description` lines."""
    lines: list[str] = []
    for memory in memories:
        tag = f"[{memory.type}] " if memory.type else ""
        timestamp = _iso_from_mtime_ms(memory.mtime_ms)
        if memory.description:
            lines.append(f"- {tag}{memory.filename} ({timestamp}): {memory.description}")
        else:
            lines.append(f"- {tag}{memory.filename} ({timestamp})")
    return "\n".join(lines)


__all__ = [
    "FRONTMATTER_MAX_LINES",
    "MAX_MEMORY_FILES",
    "MemoryHeader",
    "format_memory_manifest",
    "parse_memory_frontmatter",
    "scan_memory_files",
]
