from __future__ import annotations

import asyncio
from pathlib import Path

from raygent_harness.core.file_state import (
    FileState,
    ReadFileStateCache,
    clone_read_file_state_cache,
    normalize_file_state_path,
)
from raygent_harness.core.tool import ToolUseContext


def _state(content: str, *, timestamp: int = 1) -> FileState:
    return FileState(content=content, timestamp=timestamp, offset=1, limit=None)


def test_read_file_state_cache_normalizes_keys_and_refreshes_lru(tmp_path: Path) -> None:
    cache = ReadFileStateCache(max_entries=2, max_size_bytes=1024)
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    third = tmp_path / "c.txt"

    cache.set(str(first), _state("a"))
    cache.set(str(second), _state("b"))

    assert cache.get(str(tmp_path / "." / "a.txt")) == _state("a")

    cache.set(str(third), _state("c"))

    assert cache.has(first)
    assert not cache.has(second)
    assert cache.has(third)
    assert cache.keys() == (
        normalize_file_state_path(first),
        normalize_file_state_path(third),
    )


def test_read_file_state_cache_enforces_byte_limit(tmp_path: Path) -> None:
    cache = ReadFileStateCache(max_entries=10, max_size_bytes=5)

    cache.set(tmp_path / "a.txt", _state("1234"))
    cache.set(tmp_path / "b.txt", _state("1234"))

    assert not cache.has(tmp_path / "a.txt")
    assert cache.has(tmp_path / "b.txt")
    assert cache.calculated_size_bytes == 4

    cache.set(tmp_path / "huge.txt", _state("123456"))

    assert not cache.has(tmp_path / "huge.txt")
    assert cache.calculated_size_bytes == 4


def test_read_file_state_cache_clone_preserves_config_and_is_independent(
    tmp_path: Path,
) -> None:
    cache = ReadFileStateCache(max_entries=3, max_size_bytes=64)
    cache.set(tmp_path / "a.txt", _state("a"))

    cloned = clone_read_file_state_cache(cache)
    cloned.set(tmp_path / "b.txt", _state("b"))

    assert cloned.max_entries == 3
    assert cloned.max_size_bytes == 64
    assert cache.has(tmp_path / "a.txt")
    assert not cache.has(tmp_path / "b.txt")
    assert cloned.has(tmp_path / "b.txt")


def test_tool_use_context_uses_typed_read_file_state_cache() -> None:
    ctx = ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
    )

    assert isinstance(ctx.read_file_state, ReadFileStateCache)
