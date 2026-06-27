from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.adapters.model_protocols import OpenAIResponsesAdapter
from raygent_harness.adapters.model_protocols.base import (
    ModelProtocolAdapter,
    PreparedModelRequest,
    ProviderEvent,
)
from raygent_harness.context_providers import (
    ProjectInstructionConfig,
    ReadAdjacentProjectInstructionsContextProvider,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import (
    MessageParam,
    message_param_from_api_message,
    thaw_json,
)
from raygent_harness.core.model_types import ModelCapabilities, ModelInfo
from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.file_media import (
    PdfExtractedPage,
    PdfPageCountResult,
    PdfPageExtractionRequest,
    PdfPageExtractionResult,
)
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptScope,
    load_session_replay,
)
from raygent_harness.tools.file_edit_tool import EditInput, build_file_edit_tool
from raygent_harness.tools.file_permissions import FILE_READ_TOOL_NAME
from raygent_harness.tools.file_read_tool import build_file_read_tool
from raygent_harness.tools.file_tools import (
    create_file_tooling_runtime,
    create_file_tools_catalog_provider,
)
from raygent_harness.tools.notebook_edit_tool import (
    NOTEBOOK_EDIT_TOOL_NAME,
    build_notebook_edit_tool,
)
from raygent_harness.tools.tool_search import TOOL_SEARCH_TOOL_NAME
from raygent_harness.tools.tool_search_tool import create_tool_search_catalog_provider
from tests.fakes import AdapterBackedFakeModelProvider, FakeModelProvider


def _ctx(cwd: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _openai_adapter() -> ModelProtocolAdapter:
    return cast(ModelProtocolAdapter, OpenAIResponsesAdapter())


def _openai_text_events(text: str) -> tuple[ProviderEvent, ...]:
    return (
        {"type": "response.created", "response": {"id": "resp_file_media"}},
        {"type": "response.output_text.delta", "item_id": "text_1", "delta": text},
        {"type": "response.output_text.done", "item_id": "text_1"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_file_media",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )


def _prepared_body(prepared: PreparedModelRequest) -> Mapping[str, object]:
    raw = thaw_json(prepared.body)
    assert isinstance(raw, Mapping)
    return cast(Mapping[str, object], raw)


class FakePdfService:
    def __init__(self, page_dir: Path) -> None:
        self.page_dir = page_dir
        self.cleanup_count = 0

    def get_page_count(self, file_path: str) -> PdfPageCountResult:
        del file_path
        return PdfPageCountResult(page_count=2)

    def extract_pages(
        self,
        request: PdfPageExtractionRequest,
    ) -> PdfPageExtractionResult:
        pages: list[PdfExtractedPage] = []
        for page_number in range(request.first_page, request.last_page + 1):
            page_path = self.page_dir / f"page-{page_number}.jpg"
            page_path.write_bytes(b"\xff\xd8\xffjpeg-page")
            pages.append(
                PdfExtractedPage(
                    file_path=str(page_path),
                    page_number=page_number,
                    size_bytes=page_path.stat().st_size,
                )
            )
        return PdfPageExtractionResult(
            file_path=request.file_path,
            original_size=Path(request.file_path).stat().st_size,
            output_dir=str(self.page_dir),
            pages=tuple(pages),
        )

    def cleanup_extraction(self, result: PdfPageExtractionResult) -> None:
        del result
        self.cleanup_count += 1


def _messages_from_request(provider: FakeModelProvider, index: int) -> list[MessageParam]:
    return [
        message_param_from_api_message(message)
        for message in provider.requests[index].messages
    ]


def _tool_result_payload(messages: list[MessageParam], tool_use_id: str) -> object:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result" and block.get("tool_use_id") == tool_use_id:
                return block.get("content")
    raise AssertionError(f"tool result {tool_use_id} not found")


def _structured_tool_result_content(
    messages: list[MessageParam],
    tool_use_id: str,
) -> list[dict[str, Any]]:
    result_content = _tool_result_payload(messages, tool_use_id)
    assert isinstance(result_content, list)
    return cast(list[dict[str, Any]], result_content)


@pytest.mark.asyncio
async def test_query_transcript_replay_preserves_pdf_page_image_tool_results(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = FakePdfService(tmp_path / "pages")
    service.page_dir.mkdir()
    runtime = create_file_tooling_runtime(pdf_document_service=service)
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_pdf_pages",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(pdf), "pages": "1-2"},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ),
        model_infos={
            "m": ModelInfo(
                model="m",
                capabilities=ModelCapabilities(supports_images=True),
            )
        },
    )
    store = JsonlTranscriptStore(tmp_path / "transcripts")
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        transcript_store=store,
        tool_catalog_provider=create_file_tools_catalog_provider(runtime=runtime),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, _ctx(tmp_path))

    events = [event async for event in engine.submit_message("read pdf pages")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert service.cleanup_count == 1
    second_request_messages = _messages_from_request(provider, 1)
    content = _structured_tool_result_content(second_request_messages, "tu_pdf_pages")
    assert content[0]["type"] == "text"
    assert "PDF page extraction" in content[0]["text"]
    page_blocks = [block for block in content if block.get("type") == "image"]
    assert [block["media_type"] for block in page_blocks] == ["image/jpeg", "image/jpeg"]
    assert [block["metadata"]["page_number"] for block in page_blocks] == [1, 2]
    replay = await load_session_replay(store, TranscriptScope(session_id="s"))
    assert replay.messages == engine._messages  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_openai_adapter_lowering_serializes_rich_file_tool_result_media(
    tmp_path: Path,
) -> None:
    provider = AdapterBackedFakeModelProvider(
        adapter=_openai_adapter(),
        complete_event_batches=(_openai_text_events("adapter ok"),),
        model_infos={
            "gpt-test": ModelInfo(
                model="gpt-test",
                capabilities=ModelCapabilities(supports_images=True),
            )
        },
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )
    engine = QueryEngine(
        QueryConfig(model="gpt-test", session_id="s"),
        deps,
        _ctx(tmp_path),
    )
    await engine.seed_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_rich",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(tmp_path / "rich.pdf")},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_rich",
                        "content": [
                            {
                                "type": "text",
                                "text": "PDF page extraction: rich.pdf pages 1",
                            },
                            {
                                "type": "image",
                                "media_type": "image/jpeg",
                                "source": {"type": "base64", "data": "/9j/page1"},
                                "metadata": {"page_number": 1},
                            },
                            {
                                "type": "text",
                                "text": (
                                    '<cell id="plot-cell">'
                                    "display(plot)"
                                    '</cell id="plot-cell">'
                                ),
                            },
                            {
                                "type": "image",
                                "media_type": "image/png",
                                "source": {
                                    "type": "base64",
                                    "data": "iVBORw0KGgo=",
                                },
                                "metadata": {
                                    "cell_id": "plot-cell",
                                    "output_type": "display_data",
                                },
                            },
                        ],
                    }
                ],
            },
        ]
    )

    events = [event async for event in engine.submit_message("continue")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    body = _prepared_body(provider.prepared_requests[0])
    input_items = cast(list[dict[str, object]], body["input"])
    output_item = next(
        item
        for item in input_items
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "tu_rich"
    )
    lowered_output = cast(list[dict[str, object]], output_item["output"])
    assert {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,/9j/page1",
    } in lowered_output
    assert {
        "type": "input_image",
        "image_url": "data:image/png;base64,iVBORw0KGgo=",
    } in lowered_output
    assert any(
        item.get("type") == "input_text"
        and "PDF page extraction" in str(item.get("text"))
        for item in lowered_output
    )
    assert any(
        item.get("type") == "input_text" and "plot-cell" in str(item.get("text"))
        for item in lowered_output
    )


@pytest.mark.asyncio
async def test_rich_file_tool_prompt_schema_and_edit_redirect_are_current(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "analysis.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    ctx = _ctx(tmp_path)
    read_tool = build_file_read_tool()
    edit_tool = build_file_edit_tool()
    notebook_edit_tool = build_notebook_edit_tool()

    read_prompt = await read_tool.prompt()
    notebook_edit_prompt = await notebook_edit_tool.prompt()
    read_pages_description = cast(
        dict[str, str],
        read_tool.input_model.model_json_schema()["properties"]["pages"],
    )["description"]
    notebook_mode_description = cast(
        dict[str, str],
        notebook_edit_tool.input_model.model_json_schema()["properties"]["edit_mode"],
    )["description"]
    notebook_rejection = await edit_tool.validate_input(
        EditInput(file_path=str(notebook), old_string="{}", new_string="[]"),
        ctx,
    )

    assert "Use pages with PDFs" in read_prompt
    assert "Jupyter notebooks (.ipynb) are returned as bounded structured cells" in read_prompt
    assert "configured PDF page extractor" in read_pages_description
    assert "Use edit_mode=insert" in notebook_edit_prompt
    assert "generated zero-based cell-N" in notebook_edit_prompt
    assert "replace, insert, or delete" in notebook_mode_description
    assert notebook_rejection.result == "error"
    assert "Use NotebookEdit" in notebook_rejection.message


@pytest.mark.asyncio
async def test_query_toolsearch_notebook_edit_then_read_preserves_structured_notebook(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    notebook = root / "pkg" / "analysis.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {"language_info": {"name": "python"}},
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "display(plot)",
                        "execution_count": 3,
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {
                                    "text/plain": "plot",
                                    "image/png": "iVBORw0KGgo=",
                                },
                            }
                        ],
                    }
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    (notebook.parent / "AGENTS.md").write_text(
        "notebook adjacent policy should stay staged\n",
        encoding="utf-8",
    )
    runtime = create_file_tooling_runtime()
    file_provider = create_file_tools_catalog_provider(runtime=runtime)
    catalog_provider = create_tool_search_catalog_provider(upstream=file_provider)
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_search",
                        "name": TOOL_SEARCH_TOOL_NAME,
                        "input": {"query": f"select:{NOTEBOOK_EDIT_TOOL_NAME}"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read_before",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(notebook)},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_edit",
                        "name": NOTEBOOK_EDIT_TOOL_NAME,
                        "input": {
                            "notebook_path": str(notebook),
                            "cell_id": "cell-0",
                            "new_source": "print('after edit')",
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_read_after",
                        "name": FILE_READ_TOOL_NAME,
                        "input": {"file_path": str(notebook)},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ),
        model_infos={
            "m": ModelInfo(
                model="m",
                capabilities=ModelCapabilities(supports_images=True),
            )
        },
    )
    store = JsonlTranscriptStore(tmp_path / "transcripts")
    instruction_config = ProjectInstructionConfig(
        cwd=root,
        workspace_root=root,
        project_filenames=("AGENTS.md",),
        local_filenames=(),
    )
    ctx = _ctx(root)
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        transcript_store=store,
        tool_catalog_provider=catalog_provider,
        post_tool_context_providers=(
            ReadAdjacentProjectInstructionsContextProvider(instruction_config),
        ),
    )
    engine = QueryEngine(QueryConfig(model="m", session_id="s"), deps, ctx)

    events = [event async for event in engine.submit_message("update notebook")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert NOTEBOOK_EDIT_TOOL_NAME not in {tool.name for tool in provider.requests[0].tools}
    assert TOOL_SEARCH_TOOL_NAME in {tool.name for tool in provider.requests[0].tools}
    assert NOTEBOOK_EDIT_TOOL_NAME in {tool.name for tool in provider.requests[1].tools}
    before_content = _structured_tool_result_content(
        _messages_from_request(provider, 2),
        "tu_read_before",
    )
    assert any(block.get("type") == "image" for block in before_content)
    edit_content = _tool_result_payload(_messages_from_request(provider, 3), "tu_edit")
    assert "Updated cell cell-0" in str(edit_content)
    after_content = _structured_tool_result_content(
        _messages_from_request(provider, 4),
        "tu_read_after",
    )
    after_text = "\n".join(str(block.get("text", "")) for block in after_content)
    assert "print('after edit')" in after_text
    assert not any(block.get("type") == "image" for block in after_content)
    written_cell = json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]
    assert written_cell["execution_count"] is None
    assert written_cell["outputs"] == []
    assert ctx.successful_text_read_paths == []
    assert all(
        "notebook adjacent policy should stay staged" not in str(message.get("content"))
        for request in provider.requests
        for message in (message_param_from_api_message(item) for item in request.messages)
    )
    replay = await load_session_replay(store, TranscriptScope(session_id="s"))
    assert replay.messages == engine._messages  # pyright: ignore[reportPrivateUsage]

    resumed_provider = FakeModelProvider(
        responses=({"role": "assistant", "content": "resumed done"},),
        model_infos={
            "m": ModelInfo(
                model="m",
                capabilities=ModelCapabilities(supports_images=True),
            )
        },
    )
    resumed_deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=resumed_provider,
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
        transcript_store=JsonlTranscriptStore(tmp_path / "resume-transcripts"),
        tool_catalog_provider=catalog_provider,
        post_tool_context_providers=(
            ReadAdjacentProjectInstructionsContextProvider(instruction_config),
        ),
    )
    resumed_engine = QueryEngine.from_replay(
        QueryConfig(model="m", session_id="s"),
        resumed_deps,
        _ctx(root),
        replay,
    )

    resumed_events = [
        event async for event in resumed_engine.submit_message("resume check")
    ]

    assert isinstance(resumed_events[-1], SDKResult)
    assert resumed_events[-1].subtype == "success"
    resumed_tool_names = {tool.name for tool in resumed_provider.requests[0].tools}
    assert NOTEBOOK_EDIT_TOOL_NAME not in resumed_tool_names
    assert TOOL_SEARCH_TOOL_NAME in resumed_tool_names
    resumed_messages = _messages_from_request(resumed_provider, 0)
    resumed_before_content = _structured_tool_result_content(
        resumed_messages,
        "tu_read_before",
    )
    assert any(block.get("type") == "image" for block in resumed_before_content)
    resumed_after_content = _structured_tool_result_content(
        resumed_messages,
        "tu_read_after",
    )
    resumed_after_text = "\n".join(
        str(block.get("text", "")) for block in resumed_after_content
    )
    assert "print('after edit')" in resumed_after_text
    assert not any(block.get("type") == "image" for block in resumed_after_content)
    assert all(
        "notebook adjacent policy should stay staged" not in str(message.get("content"))
        for message in resumed_messages
    )
