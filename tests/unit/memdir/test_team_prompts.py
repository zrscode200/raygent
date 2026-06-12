from __future__ import annotations

from pathlib import Path
from typing import Any

from raygent_harness.memdir.memdir import ENTRYPOINT_NAME, MAX_ENTRYPOINT_LINES
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.memdir.team_prompts import (
    build_combined_memory_prompt,
    load_combined_memory_prompt,
)


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        **kwargs,
    )


def test_build_combined_memory_prompt_matches_reference_shape(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    prompt = build_combined_memory_prompt(cfg, ["Extra team guideline."])

    assert prompt.startswith("# Memory")
    assert f"private directory at `{get_auto_mem_path(cfg)}`" in prompt
    assert f"shared team directory at `{get_team_mem_path(cfg)}`" in prompt
    assert "## Memory scope" in prompt
    assert "- private: memories that are private between you and the current user" in prompt
    assert "- team: memories that are shared with and contributed by all of the users" in prompt
    assert "<scope>always private</scope>" in prompt
    assert "<scope>usually team</scope>" in prompt
    assert "MUST avoid saving sensitive data within shared team memories" in prompt
    assert f"Both `{ENTRYPOINT_NAME}` indexes are loaded" in prompt
    assert f"lines after {MAX_ENTRYPOINT_LINES} will be truncated" in prompt
    assert "Extra team guideline." in prompt


def test_build_combined_memory_prompt_skip_index_removes_index_step(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    prompt = build_combined_memory_prompt(cfg, skip_index=True)

    assert "Saving a memory is a two-step process" not in prompt
    assert f"Both `{ENTRYPOINT_NAME}` indexes are loaded" not in prompt
    assert "Write each memory to its own file in the chosen directory" in prompt


def test_build_combined_memory_prompt_can_include_searching_past_context(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    prompt = build_combined_memory_prompt(
        cfg,
        include_searching_past_context=True,
        project_transcript_dir=tmp_path / "transcripts",
    )

    assert "## Searching past context" in prompt
    assert f'Grep with pattern="<search term>" path="{get_auto_mem_path(cfg)}"' in prompt
    assert "*.jsonl" in prompt


def test_load_combined_memory_prompt_respects_gate_and_creates_dirs(tmp_path: Path) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    prompt = load_combined_memory_prompt(cfg)

    assert prompt is not None
    assert get_auto_mem_path(cfg).is_dir()
    assert get_team_mem_path(cfg).is_dir()
    assert "shared team directory" in prompt

    assert load_combined_memory_prompt(settings(tmp_path)) is None
    assert (
        load_combined_memory_prompt(settings(tmp_path, team_memory_enabled=True, simple_mode=True))
        is None
    )
