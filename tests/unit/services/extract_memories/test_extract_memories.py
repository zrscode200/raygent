from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, cast

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.tool import ToolUseContext
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path
from raygent_harness.services.extract_memories import (
    ExtractionRequest,
    ExtractionResult,
    SavedMemoryNotification,
    create_memory_extraction_scheduler,
    extract_written_paths,
)

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def msg(role: str, content: object, *, id_: str | None = None) -> MessageParam:
    raw: dict[str, object] = {"role": role, "content": content}
    if id_ is not None:
        raw["id"] = id_
    return cast("MessageParam", raw)


def write_block(path: Path | str, *, name: str = "Write") -> dict[str, object]:
    return {
        "type": "tool_use",
        "name": name,
        "input": {"file_path": str(path)},
    }


def settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "project",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "memory-base",
    )


def ctx(tmp_path: Path, *, agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=str(tmp_path),
    )


class CapturingRunner:
    def __init__(self, result: ExtractionResult | None = None) -> None:
        self.requests: list[ExtractionRequest] = []
        self.parent_configs: list[QueryConfig | None] = []
        self.parent_contexts: list[ToolUseContext | None] = []
        self.result = result or ExtractionResult()

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        self.requests.append(request)
        self.parent_configs.append(parent_config)
        self.parent_contexts.append(parent_ctx)
        return self.result


class BlockingRunner:
    def __init__(self) -> None:
        self.requests: list[ExtractionRequest] = []
        self.parent_configs: list[QueryConfig | None] = []
        self.parent_contexts: list[ToolUseContext | None] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        self.requests.append(request)
        self.parent_configs.append(parent_config)
        self.parent_contexts.append(parent_ctx)
        self.started.set()
        await self.release.wait()
        return ExtractionResult()


class FailingOnceRunner:
    def __init__(self) -> None:
        self.requests: list[ExtractionRequest] = []
        self.failed = False

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        del parent_config, parent_ctx
        self.requests.append(request)
        if not self.failed:
            self.failed = True
            raise RuntimeError("boom")
        return ExtractionResult()


async def test_scheduler_counts_since_cursor_and_falls_back_after_compaction(
    tmp_path: Path,
) -> None:
    runner = CapturingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )

    first = [msg("user", "u1", id_="u1"), msg("assistant", "a1", id_="a1")]
    second = [*first, msg("user", "u2", id_="u2")]
    compacted = [msg("user", "summary without old ids", id_="c1")]

    assert (await scheduler.execute(first)).status == "ran"
    assert (await scheduler.execute(second)).status == "ran"
    assert (await scheduler.execute(compacted)).status == "ran"

    assert [request.new_message_count for request in runner.requests] == [2, 1, 1]


async def test_index_cursor_falls_back_when_compacted_history_grows_past_old_index(
    tmp_path: Path,
) -> None:
    runner = CapturingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )

    first = [msg("user", "u1"), msg("assistant", "a1")]
    compacted_then_grown = [
        msg("user", "summary replacing prior history"),
        msg("user", "u2"),
        msg("assistant", "a2"),
    ]

    assert (await scheduler.execute(first)).status == "ran"
    assert (await scheduler.execute(compacted_then_grown)).status == "ran"

    assert [request.new_message_count for request in runner.requests] == [2, 3]


async def test_direct_memory_write_skips_runner_and_advances_cursor(
    tmp_path: Path,
) -> None:
    memory_settings = settings(tmp_path)
    memory_file = get_auto_mem_path(memory_settings) / "feedback.md"
    runner = CapturingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=memory_settings,
        runner=runner,
    )

    direct_write = [
        msg("user", "remember terse review", id_="u1"),
        msg("assistant", [write_block(memory_file)], id_="a1"),
    ]
    result = await scheduler.execute(direct_write)

    assert result.status == "skipped_direct_write"
    assert runner.requests == []

    later = [*direct_write, msg("user", "next", id_="u2")]
    assert (await scheduler.execute(later)).status == "ran"
    assert [request.new_message_count for request in runner.requests] == [1]


async def test_direct_write_scan_does_not_fallback_when_id_cursor_missing(
    tmp_path: Path,
) -> None:
    memory_settings = settings(tmp_path)
    memory_file = get_auto_mem_path(memory_settings) / "feedback.md"
    runner = CapturingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=memory_settings,
        runner=runner,
    )

    first = [msg("user", "u1", id_="u1"), msg("assistant", "a1", id_="a1")]
    compacted_with_write = [
        msg("user", "summary replacing prior history", id_="c1"),
        msg("assistant", [write_block(memory_file)], id_="a2"),
    ]

    assert (await scheduler.execute(first)).status == "ran"
    # The visible-count path falls back to full history, but direct-write
    # detection does not. This matches the reference's separate cursor logic.
    assert (await scheduler.execute(compacted_with_write)).status == "ran"

    assert [request.new_message_count for request in runner.requests] == [2, 2]


async def test_scheduler_coalesces_overlapping_runs_into_one_trailing_run(
    tmp_path: Path,
) -> None:
    runner = BlockingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )
    first = [msg("user", "u1", id_="u1"), msg("assistant", "a1", id_="a1")]
    second = [*first, msg("user", "u2", id_="u2"), msg("assistant", "a2", id_="a2")]
    first_config = QueryConfig(model="first-model", session_id="s")
    second_config = QueryConfig(model="second-model", session_id="s")
    first_ctx = ctx(tmp_path)
    second_ctx = ctx(tmp_path / "second")

    first_task = asyncio.create_task(
        scheduler.execute(first, turn_config=first_config, ctx=first_ctx)
    )
    await runner.started.wait()
    second_result = await scheduler.execute(
        second,
        turn_config=second_config,
        ctx=second_ctx,
    )

    assert second_result.status == "coalesced"

    runner.release.set()
    assert (await first_task).status == "ran"

    assert len(runner.requests) == 2
    assert runner.requests[0].messages == tuple(first)
    assert runner.requests[0].new_message_count == 2
    assert runner.requests[1].messages == tuple(second)
    assert runner.requests[1].new_message_count == 2
    assert runner.parent_configs == [first_config, second_config]
    assert runner.parent_contexts == [first_ctx, second_ctx]


async def test_drain_pending_extraction_times_out_without_cancelling(
    tmp_path: Path,
) -> None:
    runner = BlockingRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )

    task = asyncio.create_task(scheduler.execute([msg("user", "u1", id_="u1")]))
    await runner.started.wait()

    assert not await scheduler.drain_pending_extraction(timeout_s=0.001)
    assert not task.done()

    runner.release.set()
    assert (await task).status == "ran"
    assert await scheduler.drain_pending_extraction(timeout_s=0.001)


async def test_runner_error_does_not_advance_cursor(tmp_path: Path) -> None:
    runner = FailingOnceRunner()
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )
    messages = [msg("user", "u1", id_="u1"), msg("assistant", "a1", id_="a1")]

    first = await scheduler.execute(messages)
    second = await scheduler.execute(messages)

    assert first.status == "error"
    assert first.error == "boom"
    assert second.status == "ran"
    assert [request.new_message_count for request in runner.requests] == [2, 2]


async def test_runner_error_result_does_not_advance_cursor(tmp_path: Path) -> None:
    runner = CapturingRunner(result=ExtractionResult(status="error", error="cap hit"))
    scheduler = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )
    messages = [msg("user", "u1", id_="u1"), msg("assistant", "a1", id_="a1")]

    first = await scheduler.execute(messages)
    runner.result = ExtractionResult()
    second = await scheduler.execute(messages)

    assert first.status == "error"
    assert first.error == "cap hit"
    assert second.status == "ran"
    assert [request.new_message_count for request in runner.requests] == [2, 2]


async def test_written_paths_extracted_and_reported_without_memory_md(
    tmp_path: Path,
) -> None:
    memory_settings = settings(tmp_path)
    memory_dir = get_auto_mem_path(memory_settings)
    topic = memory_dir / "user.md"
    entrypoint = memory_dir / "MEMORY.md"
    runner_messages = (
        msg("assistant", [write_block(topic), write_block(entrypoint)], id_="agent-a1"),
    )
    runner = CapturingRunner(result=ExtractionResult(messages=runner_messages))
    notifications: list[SavedMemoryNotification] = []
    scheduler = create_memory_extraction_scheduler(
        settings=memory_settings,
        runner=runner,
    )

    result = await scheduler.execute(
        [msg("user", "remember this", id_="u1")],
        append_saved_memory=notifications.append,
    )

    assert result.status == "ran"
    assert result.written_paths == (topic, entrypoint)
    assert result.memory_paths == (topic,)
    assert notifications == [SavedMemoryNotification(memory_paths=(topic,))]
    assert extract_written_paths(runner_messages) == (topic, entrypoint)


async def test_runner_written_paths_are_filtered_to_memory_dir(tmp_path: Path) -> None:
    memory_settings = settings(tmp_path)
    memory_dir = get_auto_mem_path(memory_settings)
    topic = memory_dir / "user.md"
    outside = tmp_path / "outside.md"
    runner = CapturingRunner(result=ExtractionResult(written_paths=(topic, outside)))
    notifications: list[SavedMemoryNotification] = []
    scheduler = create_memory_extraction_scheduler(
        settings=memory_settings,
        runner=runner,
    )

    result = await scheduler.execute(
        [msg("user", "remember this", id_="u1")],
        append_saved_memory=notifications.append,
    )

    assert result.status == "ran"
    assert result.written_paths == (topic,)
    assert result.memory_paths == (topic,)
    assert notifications == [SavedMemoryNotification(memory_paths=(topic,))]


async def test_disabled_subagent_and_throttle_paths_do_not_run(
    tmp_path: Path,
) -> None:
    runner = CapturingRunner()
    disabled = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
        feature_enabled=False,
    )
    assert (await disabled.execute([msg("user", "u")])).status == "skipped_disabled"

    subagent = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
    )
    assert (
        await subagent.execute([msg("user", "u")], agent_id="local_agent_1")
    ).status == "skipped_subagent"

    remote = create_memory_extraction_scheduler(
        settings=MemorySettings(
            project_root=tmp_path / "project",
            home_dir=tmp_path / "home",
            remote_mode=True,
            remote_memory_dir=tmp_path / "remote-memory",
        ),
        runner=runner,
    )
    assert (await remote.execute([msg("user", "u")])).status == "skipped_remote"

    throttled = create_memory_extraction_scheduler(
        settings=settings(tmp_path),
        runner=runner,
        throttle_turns=2,
    )
    assert (await throttled.execute([msg("user", "u1", id_="u1")])).status == "throttled"
    assert (await throttled.execute([msg("user", "u1", id_="u1")])).status == "ran"
    assert len(runner.requests) == 1
