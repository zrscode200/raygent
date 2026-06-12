from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import MessageParam, message_param_from_api_message
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import ToolUseContext
from raygent_harness.memdir.paths import MemorySettings, get_auto_mem_path
from raygent_harness.services.extract_memories import (
    ExtractionRequest,
    RestrictedChildExtractionRunner,
    SavedMemoryNotification,
    build_default_extraction_tool_catalog,
    create_child_agent_memory_extractor,
    create_memory_extraction_scheduler,
)
from raygent_harness.tools.file_tools import create_file_tooling_runtime
from tests.fakes import FakeModelProvider


def _settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "project",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "memory-base",
    )


def _ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="system",
        cwd=str(tmp_path / "project"),
    )


def _request(memory_settings: MemorySettings) -> ExtractionRequest:
    memory_dir = get_auto_mem_path(memory_settings)
    return ExtractionRequest(
        messages=({"role": "user", "content": "remember preference"},),
        memory_dir=memory_dir,
        prompt="legacy prompt",
        new_message_count=1,
        existing_memories="",
    )


def _write_response(path: Path, content: str, *, tool_use_id: str = "tu_write") -> MessageParam:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "Write",
                "input": {"file_path": str(path), "content": content},
            }
        ],
    }


async def test_restricted_child_runner_writes_memory_through_concrete_write_tool(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    topic = get_auto_mem_path(memory_settings) / "topic.md"
    provider = FakeModelProvider(
        responses=(
            _write_response(topic, "---\ntype: fact\n---\nlikes precise reviews\n"),
            {"role": "assistant", "content": "saved"},
        )
    )
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    runner = RestrictedChildExtractionRunner(parent_deps=deps, settings=memory_settings)

    result = await runner(
        _request(memory_settings),
        parent_config=config,
        parent_ctx=_ctx(tmp_path),
    )

    assert result.status == "success"
    assert topic.read_text() == "---\ntype: fact\n---\nlikes precise reviews\n"
    assert len(provider.requests) == 2
    first_request = provider.requests[0]
    assert tuple(tool.name for tool in first_request.tools) == (
        "Read",
        "Grep",
        "Glob",
        "Bash",
        "Write",
        "Edit",
    )
    request_messages = [
        message_param_from_api_message(message) for message in first_request.messages
    ]
    assert "legacy prompt" not in str(request_messages[-1]["content"])
    assert "Available tools: Read, Grep, and Glob" in str(
        request_messages[-1]["content"]
    )
    assert "read-only Bash" in str(request_messages[-1]["content"])
    assert "Write/Edit for paths inside the memory directory only" in str(
        request_messages[-1]["content"]
    )
    assert first_request.query_source == "extract_memories"
    assert first_request.agent_id is not None
    assert first_request.agent_id.startswith("extract_memories_")


async def test_restricted_child_runner_denies_outside_write_without_mutation(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    outside = tmp_path / "outside.md"
    provider = FakeModelProvider(
        responses=(
            _write_response(outside, "should not be written"),
            {"role": "assistant", "content": "done"},
        )
    )
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    runner = RestrictedChildExtractionRunner(parent_deps=deps, settings=memory_settings)

    result = await runner(
        _request(memory_settings),
        parent_config=config,
        parent_ctx=_ctx(tmp_path),
    )

    assert result.status == "success"
    assert not outside.exists()
    assert any(
        "restricted to the auto-memory directory" in str(message)
        for message in result.messages
    )


async def test_restricted_child_runner_max_turn_error_does_not_advance_scheduler_cursor(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    memory_dir = get_auto_mem_path(memory_settings)
    provider = FakeModelProvider(
        responses=(
            *(
                _write_response(memory_dir / f"topic_{index}.md", f"attempt {index}")
                for index in range(5)
            ),
            {"role": "assistant", "content": "second attempt succeeds"},
        )
    )
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    runner = RestrictedChildExtractionRunner(parent_deps=deps, settings=memory_settings)
    scheduler = create_memory_extraction_scheduler(settings=memory_settings, runner=runner)
    messages: tuple[MessageParam, ...] = (
        {"role": "user", "content": "remember this"},
    )

    first = await scheduler.execute(messages, turn_config=config, ctx=_ctx(tmp_path))
    second = await scheduler.execute(messages, turn_config=config, ctx=_ctx(tmp_path))

    assert first.status == "error"
    assert first.error == "reached max_turns=5"
    assert second.status == "ran"
    assert second.new_message_count == 1


async def test_restricted_child_runner_missing_writer_does_not_advance_scheduler_cursor(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "second attempt succeeds"},)
    )
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
    runner = RestrictedChildExtractionRunner(
        parent_deps=deps,
        settings=memory_settings,
        include_default_tools=False,
    )
    scheduler = create_memory_extraction_scheduler(settings=memory_settings, runner=runner)
    messages: tuple[MessageParam, ...] = (
        {"role": "user", "content": "remember this"},
    )
    read_only_config = QueryConfig(
        model="parent-model",
        session_id="s",
        tools=(runtime.tools[0],),
    )
    full_config = QueryConfig(
        model="parent-model",
        session_id="s",
        tools=runtime.tools,
    )

    first = await scheduler.execute(messages, turn_config=read_only_config, ctx=_ctx(tmp_path))
    second = await scheduler.execute(messages, turn_config=full_config, ctx=_ctx(tmp_path))

    assert first.status == "error"
    assert first.error is not None
    assert "at least one memory writer tool" in first.error
    assert second.status == "ran"
    assert second.new_message_count == 1
    assert len(provider.requests) == 1


async def test_restricted_child_runner_default_catalog_adds_concrete_extraction_tools(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
    config = QueryConfig(model="parent-model", session_id="s", tools=())
    runner = RestrictedChildExtractionRunner(parent_deps=deps, settings=memory_settings)

    result = await runner(
        _request(memory_settings),
        parent_config=config,
        parent_ctx=_ctx(tmp_path),
    )

    assert result.status == "success"
    assert len(provider.requests) == 1
    assert tuple(tool.name for tool in provider.requests[0].tools) == (
        "Read",
        "Grep",
        "Glob",
        "Bash",
        "Write",
        "Edit",
    )


def test_default_extraction_catalog_replaces_colliding_parent_tools(tmp_path: Path) -> None:
    memory_settings = _settings(tmp_path)
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(task_store=AppStateStore())

    catalog = build_default_extraction_tool_catalog(
        parent_tools=runtime.tools,
        parent_deps=deps,
        settings=memory_settings,
    )

    assert tuple(tool.name for tool in catalog) == (
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Bash",
    )


class RecordingMemoryRecallProvider:
    def __init__(self) -> None:
        self.calls = 0

    def start(
        self,
        _messages: object,
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> None:
        self.calls += 1
        return None


async def _recording_context_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    /,
) -> Sequence[ContextFragment]:
    raise AssertionError("context provider should be disabled for extraction child")


async def test_restricted_child_runner_disables_recursive_memory_and_context_deps(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    recall = RecordingMemoryRecallProvider()
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        memory_recall_provider=recall,
        context_providers=(_recording_context_provider,),
    )
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    runner = RestrictedChildExtractionRunner(parent_deps=deps, settings=memory_settings)

    result = await runner(
        _request(memory_settings),
        parent_config=config,
        parent_ctx=_ctx(tmp_path),
    )

    assert result.status == "success"
    assert recall.calls == 0


async def test_child_agent_memory_extractor_integrates_with_query_engine_and_notifications(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    memory_dir = get_auto_mem_path(memory_settings)
    topic = memory_dir / "topic.md"
    entrypoint = memory_dir / "MEMORY.md"
    provider = FakeModelProvider(
        responses=(
            {"role": "assistant", "content": "main complete"},
            _write_response(topic, "topic memory", tool_use_id="tu_topic"),
            _write_response(entrypoint, "- [Topic](topic.md) - topic", tool_use_id="tu_index"),
            {"role": "assistant", "content": "saved"},
        )
    )
    sink = RecordingKernelEventSink()
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    notifications: list[SavedMemoryNotification] = []
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        observability=KernelEventBus([sink]),
    )
    deps.memory_extractor = create_child_agent_memory_extractor(
        settings=memory_settings,
        parent_deps=deps,
        append_saved_memory=notifications.append,
    )
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    engine = QueryEngine(config, deps, _ctx(tmp_path))

    events = [event async for event in engine.submit_message("remember this")]
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert topic.read_text() == "topic memory"
    assert entrypoint.read_text() == "- [Topic](topic.md) - topic"
    assert notifications == [SavedMemoryNotification(memory_paths=(topic,))]
    assert "memory.extraction.runner.started" in sink.event_types
    assert "memory.extraction.runner.completed" in sink.event_types
    runner_completed = sink.by_type("memory.extraction.runner.completed")[0]
    assert runner_completed.data["written_path_count"] == 2
    assert str(topic) not in str(runner_completed.data)
    assert sink.by_type("memory.extraction.completed")[0].data["memory_path_count"] == 1


async def test_child_agent_memory_extractor_skips_direct_main_memory_write(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    topic = get_auto_mem_path(memory_settings) / "direct.md"
    provider = FakeModelProvider(
        responses=(
            _write_response(topic, "direct memory"),
            {"role": "assistant", "content": "main done"},
        )
    )
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(
            always_allow_rules={"session": (f"Edit({get_auto_mem_path(memory_settings)}/**)",)}
        ),
    )
    deps.memory_extractor = create_child_agent_memory_extractor(
        settings=memory_settings,
        parent_deps=deps,
    )
    config = QueryConfig(model="parent-model", session_id="s", tools=runtime.tools)
    engine = QueryEngine(config, deps, _ctx(tmp_path))

    events = [event async for event in engine.submit_message("remember directly")]
    assert isinstance(events[-1], SDKResult)
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert topic.read_text() == "direct memory"
    assert len(provider.requests) == 2


async def test_child_agent_memory_extractor_does_not_schedule_for_subagents(
    tmp_path: Path,
) -> None:
    memory_settings = _settings(tmp_path)
    provider = FakeModelProvider(responses=({"role": "assistant", "content": "done"},))
    runtime = create_file_tooling_runtime(memory_settings=memory_settings)
    deps = QueryDeps(task_store=AppStateStore(), model_provider=provider)
    deps.memory_extractor = create_child_agent_memory_extractor(
        settings=memory_settings,
        parent_deps=deps,
    )
    config = QueryConfig(
        model="parent-model",
        session_id="s",
        tools=runtime.tools,
        agent_id="local_agent_1",
    )
    ctx = _ctx(tmp_path)
    ctx.agent_id = "local_agent_1"
    engine = QueryEngine(config, deps, ctx)

    events = [event async for event in engine.submit_message("subagent turn")]
    assert isinstance(events[-1], SDKResult)
    assert await engine.drain_memory_extractions(timeout_s=1.0)

    assert len(provider.requests) == 1
