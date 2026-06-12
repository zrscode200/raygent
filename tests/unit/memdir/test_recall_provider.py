from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.file_state import FileState
from raygent_harness.core.messages import (
    api_message_from_message_param,
    assistant_message,
    observable_message_from_message_param,
    user_message,
)
from raygent_harness.core.model_types import ModelResponse
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    RecordingKernelEventSink,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolRuntimeContext, ToolUseContext
from raygent_harness.memdir import (
    MAX_RELEVANT_MEMORY_SESSION_BYTES,
    ConfiguredMemoryRecallProvider,
    MemorySettings,
    ModelProviderMemorySelector,
    RelevantMemoryRecallPrefetch,
    SurfacedMemoryFile,
    create_relevant_memory_recall_provider,
    get_auto_mem_path,
    get_team_mem_path,
    is_memory_recall_message,
    message_from_surfaced_memories,
)
from tests.fakes import FakeModelProvider


@dataclass
class CapturingSelector:
    selected: list[str]
    calls: list[tuple[str, str, tuple[str, ...]]] = field(
        default_factory=list[tuple[str, str, tuple[str, ...]]]
    )

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


def _settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        **kwargs,
    )


def _ctx(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _write_memory(path: Path, *, description: str = "auth memory") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                f"description: {description}",
                "type: project",
                "---",
                "",
                f"body for {path.name}",
            ]
        ),
        encoding="utf-8",
    )
    os.utime(path, (1_700_000_000, 1_700_000_000))


async def _wait_settled(prefetch: RelevantMemoryRecallPrefetch) -> None:
    for _ in range(100):
        if prefetch.settled_at is not None:
            return
        await asyncio.sleep(0)
    raise AssertionError("memory recall prefetch did not settle")


async def _cancel_and_drain(prefetch: RelevantMemoryRecallPrefetch) -> None:
    prefetch.cancel()
    await asyncio.sleep(0)


async def test_configured_memory_recall_provider_surfaces_auto_memory(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory_path = get_auto_mem_path(settings) / "auth.md"
    _write_memory(memory_path)
    selector = CapturingSelector(["auth.md"])
    ctx = _ctx(tmp_path)
    provider = create_relevant_memory_recall_provider(settings, selector=selector)

    prefetch = provider.start(
        (user_message("please fix auth"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    )

    assert isinstance(prefetch, RelevantMemoryRecallPrefetch)
    await _wait_settled(prefetch)
    messages = await prefetch.consume_if_ready(ctx=ctx, iteration=0)

    assert len(messages) == 1
    assert is_memory_recall_message(messages[0])
    assert "body for auth.md" in str(messages[0]["content"])
    assert ctx.read_file_state.has(memory_path)
    assert prefetch.consumed_on_iteration == 0
    assert prefetch.lifecycle.selected_count == 1
    assert prefetch.lifecycle.read_count == 1
    assert prefetch.lifecycle.surfaced_count == 1
    assert prefetch.lifecycle.marked_count == 1
    prefetch.cancel()
    assert prefetch.lifecycle.cancelled is False
    assert selector.calls == [
        (
            "please fix auth",
            "- [project] auth.md (2023-11-14T22:13:20.000Z): auth memory",
            (),
        )
    ]


async def test_memory_recall_emits_metadata_only_lifecycle_events(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory_path = get_auto_mem_path(settings) / "SECRET-auth.md"
    _write_memory(memory_path, description="SECRET description")
    selector = CapturingSelector(["SECRET-auth.md"])
    provider = create_relevant_memory_recall_provider(settings, selector=selector)
    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        observability=KernelEventBus([sink]),
    )
    config = QueryConfig(model="main-model", session_id="s")
    ctx = replace(
        _ctx(tmp_path),
        observability_context=KernelEventContext(
            session_id="s",
            turn_id="turn-1",
            source="query",
        ),
        runtime=ToolRuntimeContext(config=config, deps=deps, effective_model="main-model"),
    )

    prefetch = provider.start((user_message("please fix auth"),), config, ctx)

    assert isinstance(prefetch, RelevantMemoryRecallPrefetch)
    await _wait_settled(prefetch)
    messages = await prefetch.consume_if_ready(ctx=ctx, iteration=2)

    assert len(messages) == 1
    assert sink.event_types == (
        "memory.recall.started",
        "memory.recall.completed",
    )
    started = sink.by_type("memory.recall.started")[0]
    completed = sink.by_type("memory.recall.completed")[0]
    assert started.source == "memory"
    assert started.turn_id == "turn-1"
    assert started.data["memory_dir_count"] == 1
    assert started.data["query_char_count"] == len("please fix auth")
    assert completed.data["status"] == "completed"
    assert completed.data["selected_count"] == 1
    assert completed.data["surfaced_count"] == 1
    assert completed.data["consumed_on_iteration"] == 2
    combined_payload = "\n".join(str(event.data) for event in sink.events)
    assert "body for SECRET-auth.md" not in combined_payload
    assert "SECRET description" not in combined_payload
    assert str(memory_path) not in combined_payload


async def test_configured_memory_recall_provider_searches_team_memory_under_auto_root(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, team_memory_enabled=True)
    team_memory = get_team_mem_path(settings) / "team.md"
    _write_memory(team_memory, description="team context")
    provider = create_relevant_memory_recall_provider(
        settings,
        selector=CapturingSelector(["team/team.md"]),
    )
    ctx = _ctx(tmp_path)

    prefetch = provider.start(
        (user_message("please use team memory"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    )

    assert isinstance(prefetch, RelevantMemoryRecallPrefetch)
    await _wait_settled(prefetch)
    messages = await prefetch.consume_if_ready(ctx=ctx, iteration=0)

    assert len(messages) == 1
    assert "body for team.md" in str(messages[0]["content"])
    assert prefetch.lifecycle.memory_dirs == (get_auto_mem_path(settings),)


async def test_memory_recall_prefetch_marks_settled_empty_consumes(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory_path = get_auto_mem_path(settings) / "dup.md"
    _write_memory(memory_path)
    ctx = _ctx(tmp_path)
    provider = create_relevant_memory_recall_provider(
        settings,
        selector=CapturingSelector(["dup.md"]),
    )

    prefetch = provider.start(
        (user_message("please recall duplicate"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    )

    assert isinstance(prefetch, RelevantMemoryRecallPrefetch)
    assert await prefetch.consume_if_ready(ctx=ctx, iteration=0) == ()
    assert prefetch.consumed_on_iteration is None

    await _wait_settled(prefetch)
    ctx.read_file_state.set(
        memory_path,
        FileState(content="already visible", timestamp=1, offset=1, limit=None),
    )

    assert await prefetch.consume_if_ready(ctx=ctx, iteration=1) == ()
    assert prefetch.consumed_on_iteration == 1
    assert prefetch.lifecycle.selected_count == 1
    assert prefetch.lifecycle.read_count == 1
    assert prefetch.lifecycle.surfaced_count == 0
    assert prefetch.lifecycle.duplicate_filtered_count == 1
    assert prefetch.lifecycle.marked_count == 0


async def test_configured_memory_recall_provider_gates_disabled_single_word_and_session_cap(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    ctx = _ctx(tmp_path)
    provider = create_relevant_memory_recall_provider(
        settings,
        selector=CapturingSelector(["auth.md"]),
    )
    capped_message = message_from_surfaced_memories(
        (
            SurfacedMemoryFile(
                path=tmp_path / "large.md",
                content="x" * MAX_RELEVANT_MEMORY_SESSION_BYTES,
                mtime_ms=1,
                header="Memory large:",
            ),
        )
    )
    assert capped_message is not None

    assert provider.start(
        (user_message("hello"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    ) is None
    active_prefetch = provider.start(
        (user_message("please recall"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    )
    assert isinstance(active_prefetch, RelevantMemoryRecallPrefetch)
    await _cancel_and_drain(active_prefetch)
    assert provider.start(
        (user_message("please recall"), capped_message),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    ) is None

    disabled = create_relevant_memory_recall_provider(
        _settings(tmp_path, disable_auto_memory=True),
        selector=CapturingSelector(["auth.md"]),
    )
    assert disabled.start(
        (user_message("please recall"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    ) is None


async def test_model_provider_memory_selector_parses_json_shapes() -> None:
    fake_provider = FakeModelProvider(
        responses=(
            assistant_message('{"selected_memories": ["a.md", "b.md", 4]}'),
            assistant_message('["c.md", false]'),
        )
    )
    selector = ModelProviderMemorySelector(
        provider=fake_provider,
        model="selector-model",
    )

    first = await selector.select(
        query="fix auth",
        manifest="- a.md",
        recent_tools=("Read",),
        abort_event=None,
    )
    second = await selector.select(
        query="fix cache",
        manifest="- c.md",
        recent_tools=(),
        abort_event=None,
    )

    assert first == ["a.md", "b.md"]
    assert second == ["c.md"]
    first_request = fake_provider.requests[0]
    assert first_request.model == "selector-model"
    assert first_request.sampling.max_tokens == 256
    assert first_request.query_source == "memdir_relevance"
    assert "Recently used tools: Read" in str(first_request.messages[0].provider_payload)


async def test_model_provider_memory_selector_resolves_model_alias() -> None:
    fake_provider = FakeModelProvider(
        responses=(assistant_message('{"selected_memories": ["a.md"]}'),),
        resolved_models={"selector-alias": "selector-resolved"},
    )
    selector = ModelProviderMemorySelector(
        provider=fake_provider,
        model="selector-alias",
    )

    selected = await selector.select(
        query="fix auth",
        manifest="- a.md",
        recent_tools=(),
        abort_event=None,
    )

    assert selected == ["a.md"]
    assert fake_provider.requests[0].model == "selector-resolved"
    requested, context = fake_provider.resolve_requests[0]
    assert requested == "selector-alias"
    assert context.query_source == "memdir_relevance"


async def test_model_provider_memory_selector_uses_api_bound_response() -> None:
    fake_provider = FakeModelProvider(
        responses=(
            ModelResponse(
                api_message=api_message_from_message_param(
                    {"role": "assistant", "content": '{"selected_memories": ["api.md"]}'}
                ),
                observable_message=observable_message_from_message_param(
                    {
                        "role": "assistant",
                        "content": '{"selected_memories": ["observable.md"]}',
                    }
                ),
            ),
        )
    )
    selector = ModelProviderMemorySelector(
        provider=fake_provider,
        model="selector-model",
    )

    selected = await selector.select(
        query="fix auth",
        manifest="- api.md\n- observable.md",
        recent_tools=(),
        abort_event=None,
    )

    assert selected == ["api.md"]


async def test_factory_uses_model_provider_selector_when_no_selector_is_supplied(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    memory_path = get_auto_mem_path(settings) / "model.md"
    _write_memory(memory_path, description="model-selected memory")
    fake_provider = FakeModelProvider(
        responses=(assistant_message('{"selected_memories": ["model.md"]}'),)
    )
    provider = create_relevant_memory_recall_provider(
        settings,
        model_provider=fake_provider,
        selector_model="selector-model",
    )
    ctx = _ctx(tmp_path)

    prefetch = provider.start(
        (user_message("please use model selector"),),
        QueryConfig(model="main-model", session_id="s"),
        ctx,
    )

    assert isinstance(prefetch, RelevantMemoryRecallPrefetch)
    await _wait_settled(prefetch)
    messages = await prefetch.consume_if_ready(ctx=ctx, iteration=0)

    assert len(messages) == 1
    assert "body for model.md" in str(messages[0]["content"])
    assert fake_provider.requests[0].model == "selector-model"


def test_factory_returns_configured_memory_recall_provider(tmp_path: Path) -> None:
    provider = create_relevant_memory_recall_provider(_settings(tmp_path))

    assert isinstance(provider, ConfiguredMemoryRecallProvider)
    assert provider.start(
        (user_message("please recall"),),
        QueryConfig(model="main-model", session_id="s"),
        _ctx(tmp_path),
    ) is None
