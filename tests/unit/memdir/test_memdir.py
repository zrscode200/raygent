from __future__ import annotations

import logging
from pathlib import Path

import pytest

from raygent_harness.memdir.memdir import (
    DIR_EXISTS_GUIDANCE,
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    build_memory_lines,
    build_memory_prompt,
    build_searching_past_context_section,
    ensure_memory_dir_exists,
    load_memory_prompt,
    truncate_entrypoint_content,
)
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path


def test_truncate_entrypoint_content_keeps_small_content_trimmed() -> None:
    result = truncate_entrypoint_content("\n- [One](one.md) - hook\n")

    assert result.content == "- [One](one.md) - hook"
    assert result.line_count == 1
    assert result.byte_count == len(result.content)
    assert result.was_line_truncated is False
    assert result.was_byte_truncated is False


def test_truncate_entrypoint_content_line_cap_adds_warning() -> None:
    raw = "\n".join(f"line {index}" for index in range(MAX_ENTRYPOINT_LINES + 3))

    result = truncate_entrypoint_content(raw)

    assert result.was_line_truncated is True
    assert result.was_byte_truncated is False
    assert result.line_count == MAX_ENTRYPOINT_LINES + 3
    assert "line 199" in result.content
    assert "line 200" not in result.content
    assert (
        f"WARNING: {ENTRYPOINT_NAME} is 203 lines (limit: {MAX_ENTRYPOINT_LINES})"
        in result.content
    )


def test_truncate_entrypoint_content_js_length_cap_adds_reference_style_warning() -> None:
    raw = "a" * (MAX_ENTRYPOINT_BYTES + 10)

    result = truncate_entrypoint_content(raw)

    assert result.was_line_truncated is False
    assert result.was_byte_truncated is True
    assert result.byte_count == MAX_ENTRYPOINT_BYTES + 10
    assert result.content.startswith("a" * MAX_ENTRYPOINT_BYTES)
    assert "24.4KB (limit: 24.4KB) - index entries are too long" in result.content


def test_truncate_entrypoint_content_counts_non_bmp_like_js() -> None:
    result = truncate_entrypoint_content("😀" * ((MAX_ENTRYPOINT_BYTES // 2) + 1))

    assert result.was_byte_truncated is True
    assert result.byte_count == MAX_ENTRYPOINT_BYTES + 2


def test_truncate_entrypoint_content_line_and_length_warning() -> None:
    long_line = "x" * 200
    raw = "\n".join(long_line for _ in range(MAX_ENTRYPOINT_LINES + 1))

    result = truncate_entrypoint_content(raw)

    assert result.was_line_truncated is True
    assert result.was_byte_truncated is True
    assert "201 lines and" in result.content


def test_ensure_memory_dir_exists_creates_directory_and_fails_soft(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "nested" / "memory"
    assert ensure_memory_dir_exists(target) is True
    assert target.is_dir()

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    caplog.set_level(logging.DEBUG, logger="raygent_harness.memdir.memdir")
    assert ensure_memory_dir_exists(blocker / "child") is False
    assert "ensure_memory_dir_exists failed" in caplog.text


def test_ensure_memory_dir_exists_catches_non_os_path_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="raygent_harness.memdir.memdir")

    assert ensure_memory_dir_exists("bad\0path") is False
    assert "ensure_memory_dir_exists failed" in caplog.text


def test_build_memory_lines_matches_reference_mechanics(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    lines = build_memory_lines("auto memory", memory_dir, ["Extra rule."])
    text = "\n".join(lines)

    assert text.startswith("# auto memory")
    assert f"`{memory_dir}`. {DIR_EXISTS_GUIDANCE}" in text
    assert "## Types of memory" in text
    assert "Saving a memory is a two-step process" in text
    assert f"add a pointer to that file in `{ENTRYPOINT_NAME}`" in text
    assert "## Before recommending from memory" in text
    assert "## Memory and other forms of persistence" in text
    assert "Extra rule." in text


def test_build_memory_lines_skip_index_removes_memory_md_step(tmp_path: Path) -> None:
    text = "\n".join(build_memory_lines("auto memory", tmp_path / "memory", skip_index=True))

    assert "Saving a memory is a two-step process" not in text
    assert f"add a pointer to that file in `{ENTRYPOINT_NAME}`" not in text
    assert "Write each memory to its own file" in text


def test_build_memory_prompt_empty_entrypoint_fallback(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    prompt = build_memory_prompt(display_name="auto memory", memory_dir=memory_dir)

    assert f"## {ENTRYPOINT_NAME}" in prompt
    assert f"Your {ENTRYPOINT_NAME} is currently empty" in prompt


def test_build_memory_prompt_loads_non_empty_entrypoint(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ENTRYPOINT_NAME).write_text("\n- [User](user.md) - profile\n", encoding="utf-8")

    prompt = build_memory_prompt(display_name="auto memory", memory_dir=memory_dir)

    assert f"## {ENTRYPOINT_NAME}" in prompt
    assert "- [User](user.md) - profile" in prompt
    assert "currently empty" not in prompt


def test_build_memory_prompt_replaces_malformed_utf8(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ENTRYPOINT_NAME).write_bytes(b"valid\xffinvalid")

    prompt = build_memory_prompt(display_name="auto memory", memory_dir=memory_dir)

    assert "valid\ufffdinvalid" in prompt
    assert "currently empty" not in prompt


def test_build_memory_prompt_truncates_entrypoint(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ENTRYPOINT_NAME).write_text(
        "\n".join(f"line {index}" for index in range(MAX_ENTRYPOINT_LINES + 2)),
        encoding="utf-8",
    )

    prompt = build_memory_prompt(display_name="auto memory", memory_dir=memory_dir)

    assert "line 199" in prompt
    assert "line 200" not in prompt
    assert "Only part of it was loaded" in prompt


def test_build_searching_past_context_section_disabled_by_default(tmp_path: Path) -> None:
    assert build_searching_past_context_section(tmp_path / "memory") == []

    text = "\n".join(
        build_searching_past_context_section(
            tmp_path / "memory",
            enabled=True,
            project_transcript_dir=tmp_path / "transcripts",
        )
    )
    assert "## Searching past context" in text
    assert "Grep with pattern" in text
    assert "*.jsonl" in text


def test_load_memory_prompt_respects_enabled_gate_and_creates_auto_dir(tmp_path: Path) -> None:
    settings = MemorySettings(
        project_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
    )

    prompt = load_memory_prompt(settings)

    assert prompt is not None
    assert get_auto_mem_path(settings).is_dir()
    assert f"`{get_auto_mem_path(settings)}`" in prompt
    assert f"## {ENTRYPOINT_NAME}" not in prompt

    disabled = MemorySettings(
        project_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        disable_auto_memory="1",
    )
    assert load_memory_prompt(disabled) is None
