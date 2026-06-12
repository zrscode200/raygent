from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.memdir import (
    MemorySettings,
    create_memory_prompt_provider,
    get_auto_mem_path,
    get_team_mem_path,
    load_configured_memory_prompt,
)


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        **kwargs,
    )


def ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def test_load_configured_memory_prompt_dispatches_to_auto_only(tmp_path: Path) -> None:
    cfg = settings(tmp_path)

    prompt = load_configured_memory_prompt(cfg, extra_guidelines=["extra guideline"])

    assert prompt is not None
    assert "# auto memory" in prompt
    assert "extra guideline" in prompt
    assert "shared team directory" not in prompt
    assert get_auto_mem_path(cfg).is_dir()
    assert not get_team_mem_path(cfg).exists()


def test_load_configured_memory_prompt_dispatches_to_combined_team_prompt(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)

    prompt = load_configured_memory_prompt(cfg)

    assert prompt is not None
    assert "# Memory" in prompt
    assert "shared team directory" in prompt
    assert str(get_auto_mem_path(cfg)) in prompt
    assert str(get_team_mem_path(cfg)) in prompt
    assert get_auto_mem_path(cfg).is_dir()
    assert get_team_mem_path(cfg).is_dir()


def test_load_configured_memory_prompt_returns_none_when_disabled(tmp_path: Path) -> None:
    cfg = settings(tmp_path, disable_auto_memory=True, team_memory_enabled=True)

    assert load_configured_memory_prompt(cfg) is None


async def test_create_memory_prompt_provider_matches_query_deps_signature(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path, team_memory_enabled=True)
    provider = create_memory_prompt_provider(cfg)

    prompt = await provider(
        QueryConfig(model="claude-opus-4-7", session_id="s"),
        ctx(tmp_path),
    )

    assert prompt is not None
    assert "shared team directory" in prompt
