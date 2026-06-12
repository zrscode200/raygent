from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from raygent_harness.memdir.memory_scan import (
    FRONTMATTER_MAX_LINES,
    MAX_MEMORY_FILES,
    MemoryHeader,
    format_memory_manifest,
    parse_memory_frontmatter,
    scan_memory_files,
)


def write_memory(
    path: Path,
    *,
    description: str | None,
    type_: str | None,
    body: str = "body",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = ["---"]
    if description is not None:
        frontmatter.append(f"description: {description}")
    if type_ is not None:
        frontmatter.append(f"type: {type_}")
    frontmatter.extend(["---", "", body])
    path.write_text("\n".join(frontmatter), encoding="utf-8")


def set_mtime_ms(path: Path, mtime_ms: float) -> None:
    seconds = mtime_ms / 1000
    os.utime(path, (seconds, seconds))


def test_parse_memory_frontmatter_accepts_simple_scalar_fields() -> None:
    parsed = parse_memory_frontmatter(
        "---\nname: Test\ndescription: 'Quoted value'\ntype: feedback\n---\nbody"
    )

    assert parsed == {
        "name": "Test",
        "description": "Quoted value",
        "type": "feedback",
    }


def test_parse_memory_frontmatter_fails_soft_without_complete_block() -> None:
    assert parse_memory_frontmatter("description: no block") == {}
    assert parse_memory_frontmatter("---\ndescription: no terminator") == {}


def test_scan_memory_files_recurses_skips_entrypoint_and_parses_headers(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    write_memory(memory_dir / "older.md", description="Old desc", type_="project")
    write_memory(memory_dir / "nested" / "newer.md", description="New desc", type_="feedback")
    write_memory(memory_dir / "bad_type.md", description="Bad type", type_="unknown")
    (memory_dir / "MEMORY.md").write_text("- [Index](older.md)", encoding="utf-8")
    (memory_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    set_mtime_ms(memory_dir / "older.md", 1_000)
    set_mtime_ms(memory_dir / "nested" / "newer.md", 3_000)
    set_mtime_ms(memory_dir / "bad_type.md", 2_000)

    headers = scan_memory_files(memory_dir)

    assert [header.filename for header in headers] == ["nested/newer.md", "bad_type.md", "older.md"]
    assert headers[0].file_path == memory_dir / "nested" / "newer.md"
    assert headers[0].mtime_ms == 3_000
    assert headers[0].description == "New desc"
    assert headers[0].type == "feedback"
    assert headers[1].type is None
    assert headers[2].type == "project"


def test_scan_memory_files_strips_bom_and_crlf_like_reference(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    path = memory_dir / "bom.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\ufeff---\r\ndescription: CRLF desc\r\ntype: user\r\n---\r\nbody",
        encoding="utf-8",
    )

    [header] = scan_memory_files(memory_dir)

    assert header.description == "CRLF desc"
    assert header.type == "user"


def test_scan_memory_files_reads_only_first_30_lines_for_frontmatter(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    path = memory_dir / "late.md"
    path.parent.mkdir(parents=True)
    lines = ["---", "description: too late", *[f"line {i}" for i in range(FRONTMATTER_MAX_LINES)]]
    lines.append("---")
    path.write_text("\n".join(lines), encoding="utf-8")

    [header] = scan_memory_files(memory_dir)

    assert header.description is None
    assert header.type is None


def test_scan_memory_files_fails_soft_for_missing_dir_and_bad_candidates(tmp_path: Path) -> None:
    assert scan_memory_files(tmp_path / "missing") == []

    memory_dir = tmp_path / "memory"
    write_memory(memory_dir / "ok.md", description="OK", type_="user")
    (memory_dir / "broken.md").mkdir(parents=True)

    headers = scan_memory_files(memory_dir)

    assert [header.filename for header in headers] == ["ok.md"]


def test_scan_memory_files_caps_newest_200(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    for index in range(MAX_MEMORY_FILES + 5):
        path = memory_dir / f"memory_{index:03}.md"
        write_memory(path, description=f"desc {index}", type_="reference")
        set_mtime_ms(path, float(index * 1000))

    headers = scan_memory_files(memory_dir)

    assert len(headers) == MAX_MEMORY_FILES
    assert headers[0].filename == "memory_204.md"
    assert headers[-1].filename == "memory_005.md"
    assert "memory_004.md" not in {header.filename for header in headers}


def test_format_memory_manifest_matches_reference_lines() -> None:
    first = MemoryHeader(
        filename="nested/newer.md",
        file_path=Path("/memory/nested/newer.md"),
        mtime_ms=datetime(2026, 5, 8, 12, 1, 2, 345000, tzinfo=UTC).timestamp() * 1000,
        description="New desc",
        type="feedback",
    )
    second = MemoryHeader(
        filename="older.md",
        file_path=Path("/memory/older.md"),
        mtime_ms=datetime(2026, 5, 8, 12, 1, 3, tzinfo=UTC).timestamp() * 1000,
        description=None,
        type=None,
    )

    assert format_memory_manifest([first, second]) == (
        "- [feedback] nested/newer.md (2026-05-08T12:01:02.345Z): New desc\n"
        "- older.md (2026-05-08T12:01:03.000Z)"
    )
