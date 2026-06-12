from __future__ import annotations

import asyncio
import os
from pathlib import Path

from raygent_harness.memdir.memory_scan import MemoryHeader
from raygent_harness.memdir.relevance import (
    NoOpMemorySelector,
    build_select_memories_user_prompt,
    find_relevant_memories,
    select_relevant_memories,
)


class CapturingSelector:
    def __init__(self, selected: list[str]) -> None:
        self.selected = selected
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del abort_event
        self.calls.append((query, manifest, recent_tools))
        return self.selected


class RaisingSelector:
    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools, abort_event
        raise RuntimeError("selector failed")


class CancellingSelector:
    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools, abort_event
        raise asyncio.CancelledError


class LocalAbortSelector:
    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools
        assert abort_event is not None
        abort_event.set()
        raise asyncio.CancelledError


class SleepingSelector:
    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        del query, manifest, recent_tools, abort_event
        await asyncio.sleep(60)
        return ["a.md"]


def write_memory(path: Path, *, description: str, type_: str, mtime_ms: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"description: {description}",
                f"type: {type_}",
                "---",
                "body",
            ]
        ),
        encoding="utf-8",
    )
    os.utime(path, (mtime_ms / 1000, mtime_ms / 1000))


def header(filename: str, path: Path, mtime_ms: float) -> MemoryHeader:
    return MemoryHeader(
        filename=filename,
        file_path=path,
        mtime_ms=mtime_ms,
        description=f"desc {filename}",
        type="project",
    )


def test_build_select_memories_user_prompt_includes_recent_tools_section() -> None:
    prompt = build_select_memories_user_prompt(
        query="fix auth",
        manifest="- [project] auth.md (...): Auth context",
        recent_tools=("Read", "Bash"),
    )

    assert prompt == (
        "Query: fix auth\n\n"
        "Available memories:\n"
        "- [project] auth.md (...): Auth context\n\n"
        "Recently used tools: Read, Bash"
    )


def test_build_select_memories_user_prompt_omits_empty_recent_tools() -> None:
    assert build_select_memories_user_prompt(query="q", manifest="m") == (
        "Query: q\n\nAvailable memories:\nm"
    )


async def test_select_relevant_memories_validates_selector_filenames(tmp_path: Path) -> None:
    memories = [
        header("a.md", tmp_path / "a.md", 1_000),
        header("nested/b.md", tmp_path / "nested" / "b.md", 2_000),
    ]
    selector = CapturingSelector(["missing.md", "nested/b.md", "a.md"])

    selected = await select_relevant_memories(
        query="use b",
        memories=memories,
        selector=selector,
        recent_tools=("Grep",),
    )

    assert selected == ["nested/b.md", "a.md"]
    assert selector.calls == [
        (
            "use b",
            "- [project] a.md (1970-01-01T00:00:01.000Z): desc a.md\n"
            "- [project] nested/b.md (1970-01-01T00:00:02.000Z): desc nested/b.md",
            ("Grep",),
        )
    ]


async def test_select_relevant_memories_failures_and_abort_return_empty(tmp_path: Path) -> None:
    memories = [header("a.md", tmp_path / "a.md", 1_000)]

    assert (
        await select_relevant_memories(
            query="q",
            memories=memories,
            selector=RaisingSelector(),
        )
        == []
    )
    abort_event = asyncio.Event()
    abort_event.set()
    selector = CapturingSelector(["a.md"])
    assert (
        await select_relevant_memories(
            query="q",
            memories=memories,
            selector=selector,
            abort_event=abort_event,
        )
        == []
    )
    assert selector.calls == []


async def test_select_relevant_memories_local_abort_returns_empty(tmp_path: Path) -> None:
    abort_event = asyncio.Event()

    assert (
        await select_relevant_memories(
            query="q",
            memories=[header("a.md", tmp_path / "a.md", 1_000)],
            selector=LocalAbortSelector(),
            abort_event=abort_event,
        )
        == []
    )
    assert abort_event.is_set()


async def test_select_relevant_memories_outer_cancellation_propagates(tmp_path: Path) -> None:
    task = asyncio.create_task(
        select_relevant_memories(
            query="q",
            memories=[header("a.md", tmp_path / "a.md", 1_000)],
            selector=SleepingSelector(),
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("outer cancellation should propagate")


async def test_find_relevant_memories_filters_already_surfaced_before_selection(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    write_memory(memory_dir / "old.md", description="Old", type_="user", mtime_ms=1_000)
    write_memory(memory_dir / "new.md", description="New", type_="feedback", mtime_ms=2_000)
    selector = CapturingSelector(["old.md", "new.md"])

    selected = await find_relevant_memories(
        query="what should I know?",
        memory_dir=memory_dir,
        selector=selector,
        recent_tools=("Read",),
        already_surfaced={memory_dir / "old.md"},
    )

    # old.md was filtered before selector, so even if returned it is ignored.
    # new.md remains and is mapped to path + mtime.
    assert len(selected) == 1
    assert selected[0].path == memory_dir / "new.md"
    assert selected[0].mtime_ms == 2_000
    assert "old.md" not in selector.calls[0][1]
    assert "new.md" in selector.calls[0][1]
    assert selector.calls[0][2] == ("Read",)


async def test_find_relevant_memories_default_noop_selector_returns_empty(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    write_memory(memory_dir / "user.md", description="User", type_="user", mtime_ms=1_000)

    assert await find_relevant_memories(query="q", memory_dir=memory_dir) == []
    assert (
        await NoOpMemorySelector().select(
            query="q",
            manifest="m",
            recent_tools=(),
            abort_event=None,
        )
        == []
    )


async def test_find_relevant_memories_returns_empty_without_candidates_or_when_aborted(
    tmp_path: Path,
) -> None:
    selector = CapturingSelector(["a.md"])
    assert (
        await find_relevant_memories(
            query="q",
            memory_dir=tmp_path / "missing",
            selector=selector,
        )
        == []
    )
    assert selector.calls == []

    memory_dir = tmp_path / "memory"
    write_memory(memory_dir / "a.md", description="A", type_="reference", mtime_ms=1_000)
    abort_event = asyncio.Event()
    abort_event.set()
    assert (
        await find_relevant_memories(
            query="q",
            memory_dir=memory_dir,
            selector=selector,
            abort_event=abort_event,
        )
        == []
    )
