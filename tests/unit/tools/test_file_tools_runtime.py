from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.context_providers import (
    ProjectInstructionConfig,
    ProjectInstructionsContextProvider,
    ReadAdjacentProjectInstructionsContextProvider,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import message_param_from_api_message
from raygent_harness.core.model_types import ModelCapabilities, ModelInfo
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolRuntimeContext,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.file_media import (
    PdfPageCountResult,
    PdfPageExtractionRequest,
    PdfPageExtractionResult,
)
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.tools.file_edit_tool import FILE_EDIT_TOOL_ALIAS, FILE_EDIT_TOOL_NAME
from raygent_harness.tools.file_read_tool import FILE_READ_TOOL_ALIAS, FILE_READ_TOOL_NAME
from raygent_harness.tools.file_tools import (
    create_file_tooling_runtime,
    create_file_tools_catalog_provider,
)
from raygent_harness.tools.file_write_tool import FILE_WRITE_TOOL_ALIAS, FILE_WRITE_TOOL_NAME
from raygent_harness.tools.notebook_edit_tool import NOTEBOOK_EDIT_TOOL_NAME
from tests.fakes import FakeModelProvider


class EmptyInput(BaseModel):
    pass


class FakePdfService:
    def get_page_count(self, file_path: str) -> PdfPageCountResult:
        return PdfPageCountResult(page_count=None)

    def extract_pages(
        self,
        request: PdfPageExtractionRequest,
    ) -> PdfPageExtractionResult:
        raise AssertionError("not needed")

    def cleanup_extraction(self, result: PdfPageExtractionResult) -> None:
        raise AssertionError("not needed")


async def _dummy_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _dummy_tool(name: str, *, aliases: tuple[str, ...] = ()) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            aliases=aliases,
            description=f"{name} tool",
            input_model=EmptyInput,
            call=_dummy_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx(cwd: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path,
        home_dir=tmp_path,
        memory_base_dir=tmp_path / "memory-root",
        team_memory_enabled=True,
    )


@pytest.mark.asyncio
async def test_file_tooling_runtime_bundles_tools_and_optional_team_memory_hook(
    tmp_path: Path,
) -> None:
    calls = 0

    async def notify() -> bool:
        nonlocal calls
        calls += 1
        return True

    no_memory = create_file_tooling_runtime()
    memory_without_notifier = create_file_tooling_runtime(memory_settings=_settings(tmp_path))
    memory_with_notifier = create_file_tooling_runtime(
        memory_settings=_settings(tmp_path),
        notify_team_memory_write=notify,
    )

    assert tuple(tool.name for tool in no_memory.tools) == (
        FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        FILE_EDIT_TOOL_NAME,
        NOTEBOOK_EDIT_TOOL_NAME,
    )
    assert no_memory.pre_tool_use_hooks == ()
    assert no_memory.post_tool_use_hooks == ()
    assert memory_without_notifier.post_tool_use_hooks == ()
    assert len(memory_with_notifier.post_tool_use_hooks) == 1
    assert calls == 0


@pytest.mark.asyncio
async def test_file_tooling_runtime_passes_pdf_service_to_read_tool(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    runtime = create_file_tooling_runtime(pdf_document_service=FakePdfService())
    read = next(tool for tool in runtime.tools if tool.name == FILE_READ_TOOL_NAME)
    ctx = _ctx(tmp_path)
    ctx.runtime = ToolRuntimeContext(
        config=QueryConfig(model="image-model"),
        deps=QueryDeps(
            task_store=AppStateStore(),
            model_provider=FakeModelProvider(
                model_infos={
                    "image-model": ModelInfo(
                        model="image-model",
                        capabilities=ModelCapabilities(supports_images=True),
                    )
                }
            ),
        ),
        effective_model="image-model",
    )

    validation = await read.validate_input(
        read.input_model(file_path=str(pdf), pages="1"),
        ctx,
    )

    assert validation.result == "ok"


@pytest.mark.asyncio
async def test_file_tools_catalog_provider_composes_upstream_and_replaces_collisions(
    tmp_path: Path,
) -> None:
    keep = _dummy_tool("Keep")
    read_collision = _dummy_tool(FILE_READ_TOOL_NAME)
    write_alias_collision = _dummy_tool(FILE_WRITE_TOOL_ALIAS)
    edit_alias_collision = _dummy_tool("LegacyEdit", aliases=(FILE_EDIT_TOOL_NAME,))
    seen_skills: list[int] = []

    async def upstream(
        config: QueryConfig,
        _ctx: ToolUseContext,
        skills: Sequence[object],
        /,
    ) -> Sequence[Tool] | None:
        seen_skills.append(len(skills))
        return (
            keep,
            read_collision,
            write_alias_collision,
            edit_alias_collision,
            *config.tools,
        )

    provider = create_file_tools_catalog_provider(upstream=upstream)
    tools = await provider(QueryConfig(model="m", tools=(_dummy_tool("Base"),)), _ctx(tmp_path), ())

    assert tools is not None
    assert seen_skills == [0]
    assert tuple(tool.name for tool in tools) == (
        "Keep",
        "Base",
        FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        FILE_EDIT_TOOL_NAME,
        NOTEBOOK_EDIT_TOOL_NAME,
    )
    aliases_by_name = {tool.name: tool.aliases for tool in tools}
    assert aliases_by_name[FILE_READ_TOOL_NAME] == (FILE_READ_TOOL_ALIAS,)
    assert aliases_by_name[FILE_WRITE_TOOL_NAME] == (FILE_WRITE_TOOL_ALIAS,)
    assert aliases_by_name[FILE_EDIT_TOOL_NAME] == (FILE_EDIT_TOOL_ALIAS,)


@pytest.mark.asyncio
async def test_file_runtime_post_hook_notifies_after_query_write_to_team_memory(
    tmp_path: Path,
) -> None:
    cfg = _settings(tmp_path)
    target = get_team_mem_path(cfg) / "MEMORY.md"
    calls = 0

    async def notify() -> bool:
        nonlocal calls
        calls += 1
        return True

    runtime = create_file_tooling_runtime(
        memory_settings=cfg,
        notify_team_memory_write=notify,
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_write",
                        "name": FILE_WRITE_TOOL_NAME,
                        "input": {
                            "file_path": str(target),
                            "content": "shared project note\n",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        tool_catalog_provider=create_file_tools_catalog_provider(runtime=runtime),
        post_tool_use_hooks=list(runtime.post_tool_use_hooks),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, _ctx(tmp_path))

    events = [event async for event in engine.submit_message("write memory")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert target.read_text(encoding="utf-8") == "shared project note\n"
    assert calls == 1


@pytest.mark.asyncio
async def test_file_runtime_post_hook_notifies_after_query_edit_to_team_memory(
    tmp_path: Path,
) -> None:
    cfg = _settings(tmp_path)
    target = get_team_mem_path(cfg) / "MEMORY.md"
    target.parent.mkdir(parents=True)
    target.write_text("old project note\n", encoding="utf-8")
    calls = 0

    async def notify() -> bool:
        nonlocal calls
        calls += 1
        return True

    runtime = create_file_tooling_runtime(
        memory_settings=cfg,
        notify_team_memory_write=notify,
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(target)},
                    }
                ],
            },
            {"role": "assistant", "content": "read done"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_edit",
                        "name": FILE_EDIT_TOOL_NAME,
                        "input": {
                            "file_path": str(target),
                            "old_string": "old",
                            "new_string": "new",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "edit done"},
        )
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        tool_catalog_provider=create_file_tools_catalog_provider(runtime=runtime),
        post_tool_use_hooks=list(runtime.post_tool_use_hooks),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, _ctx(tmp_path))

    first_turn = [event async for event in engine.submit_message("read memory")]
    second_turn = [event async for event in engine.submit_message("edit memory")]

    assert isinstance(first_turn[-1], SDKResult)
    assert first_turn[-1].subtype == "success"
    assert isinstance(second_turn[-1], SDKResult)
    assert second_turn[-1].subtype == "success"
    assert target.read_text(encoding="utf-8") == "new project note\n"
    assert calls == 1


@pytest.mark.asyncio
async def test_query_engine_file_runtime_preserves_read_state_across_turns(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    runtime = create_file_tooling_runtime()
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(target)},
                    }
                ],
            },
            {"role": "assistant", "content": "read done"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_edit",
                        "name": FILE_EDIT_TOOL_NAME,
                        "input": {
                            "file_path": str(target),
                            "old_string": "hello",
                            "new_string": "bye",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "edit done"},
        )
    )
    ctx = _ctx(tmp_path)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        tool_catalog_provider=create_file_tools_catalog_provider(runtime=runtime),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, ctx)

    first_turn = [event async for event in engine.submit_message("read note")]
    second_turn = [event async for event in engine.submit_message("edit note")]

    assert isinstance(first_turn[-1], SDKResult)
    assert first_turn[-1].subtype == "success"
    assert isinstance(second_turn[-1], SDKResult)
    assert second_turn[-1].subtype == "success"
    assert target.read_text(encoding="utf-8") == "bye\n"
    state = ctx.read_file_state.get(target)
    assert state is not None
    assert state.content == "bye\n"
    assert state.offset is None
    assert state.limit is None


@pytest.mark.asyncio
async def test_query_engine_calls_post_tool_context_providers_without_text_reads(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        read_paths: Sequence[str],
        already_attached_sources: Sequence[str],
        /,
    ) -> tuple[ContextFragment, ...]:
        calls.append((tuple(read_paths), tuple(already_attached_sources)))
        return (
            ContextFragment(
                id="post-tool:probe",
                content="generic post-tool context",
                channel="user_context",
                source="post-tool:probe",
            ),
        )

    model_provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_probe",
                        "name": "Probe",
                        "input": {},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model_provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        post_tool_context_providers=(provider,),
    )
    engine = QueryEngine(
        QueryConfig(model="m", session_id="s", tools=(_dummy_tool("Probe"),)),
        deps,
        _ctx(tmp_path),
    )

    events = [event async for event in engine.submit_message("run probe")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert calls == [((), ())]
    assert len(model_provider.requests) == 2
    second_request = [
        message_param_from_api_message(message)
        for message in model_provider.requests[1].messages
    ]
    assert "generic post-tool context" in str(second_request[0]["content"])


@pytest.mark.asyncio
async def test_query_engine_attaches_read_adjacent_instructions_transiently(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    target = root / "pkg" / "feature" / "src" / "main.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('hi')\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("root project policy\n", encoding="utf-8")
    feature_instructions = root / "pkg" / "feature" / "AGENTS.md"
    feature_instructions.write_text("feature project policy\n", encoding="utf-8")
    feature_rule = root / "pkg" / "feature" / ".claude" / "rules" / "python.md"
    feature_rule.parent.mkdir(parents=True)
    feature_rule.write_text(
        "---\npaths: src/*.py\n---\nfeature python rule\n",
        encoding="utf-8",
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(target)},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    instruction_config = ProjectInstructionConfig(
        cwd=root,
        workspace_root=root,
        project_filenames=("AGENTS.md",),
        local_filenames=(),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        transcript_store=transcript_store,
        tool_catalog_provider=create_file_tools_catalog_provider(
            runtime=create_file_tooling_runtime(),
        ),
        context_providers=(
            ProjectInstructionsContextProvider(instruction_config),
        ),
        post_tool_context_providers=(
            ReadAdjacentProjectInstructionsContextProvider(instruction_config),
        ),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, _ctx(root))

    events = [event async for event in engine.submit_message("read file")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert len(provider.requests) == 2
    first_request = [
        message_param_from_api_message(message)
        for message in provider.requests[0].messages
    ]
    second_request = [
        message_param_from_api_message(message)
        for message in provider.requests[1].messages
    ]
    assert "root project policy" in str(first_request[0]["content"])
    assert "feature project policy" not in str(first_request[0]["content"])
    assert "feature project policy" in str(second_request[1]["content"])
    assert "feature python rule" in str(second_request[1]["content"])
    assert "root project policy" not in str(second_request[1]["content"])
    assert second_request[2] == {"role": "user", "content": "read file"}
    assert all(
        "feature project policy" not in str(message.get("content"))
        and "feature python rule" not in str(message.get("content"))
        for message in engine._messages  # pyright: ignore[reportPrivateUsage]
    )
    transcript_entries = await transcript_store.read_entries(
        TranscriptScope(session_id="s")
    )
    transcript_text = "\n".join(
        str(entry.message.get("content"))
        for entry in transcript_entries
        if isinstance(entry, TranscriptMessageEntry)
    )
    assert "feature project policy" not in transcript_text
    assert "feature python rule" not in transcript_text
