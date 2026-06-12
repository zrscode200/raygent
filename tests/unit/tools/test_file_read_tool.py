from __future__ import annotations

import asyncio
import builtins
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.file_state import FileState
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.model_types import ModelCapabilities, ModelInfo
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    ToolPermissionContext,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    ContentReplacementState,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolRuntimeContext,
    ToolUseContext,
    ValidationError,
    ValidationOk,
)
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.services.file_media import (
    PdfDocumentService,
    PdfExtractedPage,
    PdfPageCountResult,
    PdfPageExtractionRequest,
    PdfPageExtractionResult,
    PdfServiceError,
)
from raygent_harness.tools.file_permissions import (
    FILE_READ_TOOL_NAME as READ_RULE_NAME,
)
from raygent_harness.tools.file_read_tool import (
    FILE_READ_DEFAULT_MAX_SIZE_BYTES,
    FILE_READ_IMAGE_MAX_BYTES,
    FILE_READ_PDF_MAX_BYTES,
    FILE_READ_PROMPT,
    FILE_READ_TOOL_ALIAS,
    FILE_READ_TOOL_NAME,
    FILE_UNCHANGED_STUB,
    FileReadToolError,
    ReadInput,
    build_file_read_tool,
    create_file_read_catalog_provider,
    read_structured_file_content,
    read_text_file_in_range,
)
from tests.fakes import FakeModelProvider

_PNG_BYTES = b"\x89PNG\r\n\x1a\npayload"
_JPEG_BYTES = b"\xff\xd8\xffjpeg"


class FakePdfDocumentService:
    def __init__(
        self,
        *,
        page_count: int | None = None,
        extraction_result: PdfPageExtractionResult | None = None,
        extraction_error: PdfServiceError | None = None,
    ) -> None:
        self.page_count = page_count
        self.extraction_result = extraction_result
        self.extraction_error = extraction_error
        self.count_requests: list[str] = []
        self.extraction_requests: list[PdfPageExtractionRequest] = []
        self.cleaned_results: list[PdfPageExtractionResult] = []

    def get_page_count(self, file_path: str) -> PdfPageCountResult:
        self.count_requests.append(file_path)
        return PdfPageCountResult(page_count=self.page_count)

    def extract_pages(
        self,
        request: PdfPageExtractionRequest,
    ) -> PdfPageExtractionResult:
        self.extraction_requests.append(request)
        if self.extraction_error is not None:
            raise self.extraction_error
        assert self.extraction_result is not None
        return self.extraction_result

    def cleanup_extraction(self, result: PdfPageExtractionResult) -> None:
        self.cleaned_results.append(result)


def _ctx(cwd: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
    )


def _ctx_with_model_capabilities(
    cwd: Path,
    capabilities: ModelCapabilities,
    *,
    pdf_document_service: PdfDocumentService | None = None,
) -> ToolUseContext:
    ctx = _ctx(cwd)
    config = QueryConfig(model="model-with-media")
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=FakeModelProvider(
            model_infos={
                "model-with-media": ModelInfo(
                    model="model-with-media",
                    capabilities=capabilities,
                )
            }
        ),
        pdf_document_service=pdf_document_service,
    )
    ctx.runtime = ToolRuntimeContext(
        config=config,
        deps=deps,
        effective_model="model-with-media",
    )
    return ctx


async def _call_read(
    tool: Tool,
    input_: ReadInput,
    ctx: ToolUseContext,
) -> ToolResult:
    events: list[ToolCallEvent] = [event async for event in tool.call(input_, ctx)]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


def _content(result: ToolResult) -> str:
    assert isinstance(result.content, str)
    return result.content


def _mtime_ms(path: Path) -> int:
    return int(os.stat(path).st_mtime_ns // 1_000_000)


def test_file_read_tool_axes_and_prompt() -> None:
    tool = build_file_read_tool()
    input_ = ReadInput(file_path="/tmp/example.txt")

    assert tool.name == FILE_READ_TOOL_NAME
    assert tool.aliases == (FILE_READ_TOOL_ALIAS,)
    assert tool.is_concurrency_safe(input_) is True
    assert tool.is_read_only(input_) is True
    assert tool.is_destructive(input_) is False
    assert tool.is_open_world(input_) is False
    assert tool.max_result_size_chars == float("inf")
    assert asyncio.run(tool.prompt()) == FILE_READ_PROMPT


@pytest.mark.asyncio
async def test_file_read_reads_text_with_line_numbers_and_updates_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    content = _content(result)
    assert result.is_error is False
    assert "1\talpha\n2\tbeta" in content
    assert "Whenever you read a file" in content
    state = ctx.read_file_state.get(str(path))
    assert state is not None
    assert state.content == "alpha\nbeta\n"
    assert state.offset == 1
    assert state.limit is None
    assert state.timestamp == _mtime_ms(path)
    assert ctx.successful_text_read_paths == [str(path)]


@pytest.mark.asyncio
async def test_file_read_preserves_trailing_newline_in_content_and_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert "1\talpha\n2\tbeta\n3\t" in _content(result)
    state = ctx.read_file_state.get(str(path))
    assert state is not None
    assert state.content == "alpha\nbeta\n"


@pytest.mark.asyncio
async def test_file_read_range_uses_offset_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(
        tool,
        ReadInput(file_path=str(path), offset=2, limit=2),
        ctx,
    )

    content = _content(result)
    assert "2\tb\n3\tc" in content
    assert "1\ta" not in content
    state = ctx.read_file_state.get(str(path))
    assert state is not None
    assert state.offset == 2
    assert state.limit == 2


@pytest.mark.asyncio
async def test_file_read_large_file_allows_targeted_range(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text(
        "\n".join(["first", *("padding" for _ in range(40_000)), "last"]),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(
        tool,
        ReadInput(file_path=str(path), offset=1, limit=1),
        ctx,
    )

    assert result.is_error is False
    assert _content(result).startswith("1\tfirst")


@pytest.mark.asyncio
async def test_file_read_large_file_without_range_errors(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * (FILE_READ_DEFAULT_MAX_SIZE_BYTES + 1), encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert result.is_error is True
    assert "exceeds maximum allowed size" in _content(result)


@pytest.mark.asyncio
async def test_file_read_unchanged_stub_only_for_prior_read_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\n", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    first = await _call_read(tool, ReadInput(file_path=str(path)), ctx)
    second = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert FILE_UNCHANGED_STUB not in _content(first)
    assert _content(second) == FILE_UNCHANGED_STUB
    assert ctx.successful_text_read_paths == [str(path)]

    ctx.read_file_state.set(
        str(path),
        FileState(
            content="post-write content",
            timestamp=_mtime_ms(path),
            offset=None,
            limit=None,
        ),
    )
    third = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert FILE_UNCHANGED_STUB not in _content(third)
    assert "1\talpha" in _content(third)


@pytest.mark.asyncio
async def test_file_read_empty_and_short_offset_reminders(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    short = tmp_path / "short.txt"
    short.write_text("one\ntwo", encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    empty_result = await _call_read(tool, ReadInput(file_path=str(empty)), ctx)
    short_result = await _call_read(
        tool,
        ReadInput(file_path=str(short), offset=10, limit=2),
        ctx,
    )

    assert "contents are empty" in _content(empty_result)
    assert "shorter than the provided offset (10)" in _content(short_result)
    assert "2 lines" in _content(short_result)


@pytest.mark.asyncio
async def test_file_read_validates_pages_parameter(tmp_path: Path) -> None:
    tool = build_file_read_tool()
    ctx = _ctx(tmp_path)

    invalid = await tool.validate_input(ReadInput(file_path="plain.txt", pages="0"), ctx)
    too_many = await tool.validate_input(
        ReadInput(file_path="plain.txt", pages="1-21"),
        ctx,
    )
    valid = await tool.validate_input(ReadInput(file_path="plain.txt", pages="2-3"), ctx)

    assert isinstance(invalid, ValidationError)
    assert "Invalid pages parameter" in invalid.message
    assert isinstance(too_many, ValidationError)
    assert "exceeds maximum" in too_many.message
    assert isinstance(valid, ValidationOk)


@pytest.mark.asyncio
async def test_file_read_validation_preserves_explicit_deny_before_file_type_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = build_file_read_tool()
    ctx = _ctx(tmp_path)
    ctx.permission_context = ToolPermissionContext(
        always_deny_rules={"session": (f"{READ_RULE_NAME}({tmp_path}/**)",)}
    )

    def fail_if_classified(path: str) -> Any:
        raise AssertionError(f"deny should happen before classification for {path}")

    monkeypatch.setattr(
        "raygent_harness.tools.file_read_tool.classify_file_media",
        fail_if_classified,
    )

    pdf = await tool.validate_input(ReadInput(file_path=str(tmp_path / "report.pdf")), ctx)
    binary = await tool.validate_input(ReadInput(file_path=str(tmp_path / "archive.zip")), ctx)
    notebook = await tool.validate_input(
        ReadInput(file_path=str(tmp_path / "notebook.ipynb")),
        ctx,
    )

    for result in (pdf, binary, notebook):
        assert isinstance(result, ValidationError)
        assert "denied by your permission settings" in result.message


@pytest.mark.asyncio
async def test_file_read_permission_ask_happens_before_sample_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    target = outside / "image.txt"
    target.write_bytes(_PNG_BYTES)
    ctx = _ctx(repo)
    tool = build_file_read_tool()

    def fail_if_sampled(path: str) -> Any:
        raise AssertionError(f"classification sampled before permission: {path}")

    monkeypatch.setattr(
        "raygent_harness.tools.file_read_tool.classify_file_media",
        fail_if_sampled,
    )

    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_read",
                name=FILE_READ_TOOL_NAME,
                input={"file_path": str(target)},
                index=0,
            ),
            assistant_message={"role": "assistant", "content": []},
            tools=(tool,),
            deps=QueryDeps(task_store=AppStateStore()),
            ctx=ctx,
        )
    ]

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolExecutionResult)
    assert result.permission_denials
    assert "has not been granted yet" in str(result.message)


@pytest.mark.asyncio
async def test_file_read_returns_errors_for_missing_directory_and_binary(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"abc\x00def")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    missing_result = await _call_read(
        tool,
        ReadInput(file_path=str(tmp_path / "missing.txt")),
        ctx,
    )
    directory_result = await _call_read(tool, ReadInput(file_path=str(tmp_path)), ctx)
    binary_result = await _call_read(tool, ReadInput(file_path=str(binary)), ctx)

    assert missing_result.is_error is True
    assert "File does not exist" in _content(missing_result)
    assert directory_result.is_error is True
    assert "directory" in _content(directory_result)
    assert binary_result.is_error is True
    assert "binary files" in _content(binary_result)
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_validation_rejects_unsupported_and_device_paths(
    tmp_path: Path,
) -> None:
    tool = build_file_read_tool()
    ctx = _ctx(tmp_path)

    for filename, expected in (
        ("image.png", "does not advertise image input support"),
        ("paper.pdf", "does not advertise PDF/document input support"),
        ("archive.zip", "binary zip"),
    ):
        result = await tool.validate_input(ReadInput(file_path=filename), ctx)
        assert isinstance(result, ValidationError)
        assert expected in result.message

    notebook_result = await tool.validate_input(ReadInput(file_path="notebook.ipynb"), ctx)
    assert isinstance(notebook_result, ValidationOk)

    device_result = await tool.validate_input(ReadInput(file_path="/dev/zero"), ctx)
    assert isinstance(device_result, ValidationError)
    assert "device file would block" in device_result.message

    ok_result = await tool.validate_input(ReadInput(file_path="plain.txt"), ctx)
    assert isinstance(ok_result, ValidationOk)


@pytest.mark.asyncio
async def test_file_read_returns_structured_image_when_model_supports_images(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.png"
    path.write_bytes(_PNG_BYTES)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(path)), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert isinstance(validation, ValidationOk)
    assert result.is_error is False
    assert isinstance(result.content, list)
    image = result.content[0]
    assert image["type"] == "image"
    assert image["media_type"] == "image/png"
    assert image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgpwYXlsb2Fk",
    }
    assert ctx.read_file_state.get(str(path)) is None
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_rejects_unsupported_image_type_even_with_image_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.bmp"
    path.write_bytes(b"bmp-bytes")
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(path)), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert isinstance(validation, ValidationError)
    assert "Supported image types are PNG, JPEG, GIF, and WebP" in validation.message
    assert result.is_error is True
    assert "Supported image types are PNG, JPEG, GIF, and WebP" in _content(result)


@pytest.mark.asyncio
async def test_file_read_rejects_empty_and_corrupt_images(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not-png")
    mislabeled = tmp_path / "mislabeled.jpg"
    mislabeled.write_bytes(_PNG_BYTES)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
    )
    tool = build_file_read_tool()

    empty_result = await _call_read(tool, ReadInput(file_path=str(empty)), ctx)
    corrupt_result = await _call_read(tool, ReadInput(file_path=str(corrupt)), ctx)
    mislabeled_result = await _call_read(tool, ReadInput(file_path=str(mislabeled)), ctx)

    assert empty_result.is_error is True
    assert "Image file is empty" in _content(empty_result)
    assert corrupt_result.is_error is True
    assert "not a valid PNG, JPEG, GIF, or WebP image" in _content(corrupt_result)
    assert mislabeled_result.is_error is False
    assert isinstance(mislabeled_result.content, list)
    assert mislabeled_result.content[0]["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_file_read_uses_magic_classification_for_mislabeled_structured_media(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.txt"
    path.write_bytes(_PNG_BYTES)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(path)), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert isinstance(validation, ValidationOk)
    assert result.is_error is False
    assert isinstance(result.content, list)
    assert result.content[0]["media_type"] == "image/png"
    assert ctx.read_file_state.get(str(path)) is None
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_returns_structured_full_pdf_when_model_supports_documents(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_documents=True),
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(path)), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(path)), ctx)

    assert isinstance(validation, ValidationOk)
    assert result.is_error is False
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "text"
    assert "PDF file read:" in result.content[0]["text"]
    document = result.content[1]
    assert document["type"] == "document"
    assert document["media_type"] == "application/pdf"
    assert document["source"] == {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "JVBERi0xLjQKJSVFT0YK",
    }
    assert ctx.read_file_state.get(str(path)) is None
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_rejects_empty_corrupt_and_large_page_count_pdfs(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not-pdf")
    many_pages = tmp_path / "many-pages.pdf"
    many_pages.write_bytes(
        b"%PDF-1.4\n"
        + b"\n".join(b"<< /Type /Page >>" for _ in range(11))
        + b"\n%%EOF\n"
    )
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_documents=True),
    )
    tool = build_file_read_tool()

    empty_result = await _call_read(tool, ReadInput(file_path=str(empty)), ctx)
    corrupt_result = await _call_read(tool, ReadInput(file_path=str(corrupt)), ctx)
    many_pages_result = await _call_read(tool, ReadInput(file_path=str(many_pages)), ctx)

    assert empty_result.is_error is True
    assert "PDF file is empty" in _content(empty_result)
    assert corrupt_result.is_error is True
    assert "missing %PDF- header" in _content(corrupt_result)
    assert many_pages_result.is_error is True
    assert "too many to read at once" in _content(many_pages_result)
    assert "Maximum 20 pages per request" in _content(many_pages_result)


@pytest.mark.asyncio
async def test_file_read_pdf_page_range_requires_configured_pdf_service(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True, supports_documents=True),
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(path), pages="1-2"), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(path), pages="1-2"), ctx)

    assert isinstance(validation, ValidationError)
    assert "no PDF document service is configured" in validation.message
    assert result.is_error is True
    assert "no PDF document service is configured" in _content(result)


@pytest.mark.asyncio
async def test_file_read_returns_pdf_page_images_with_configured_service(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    page_dir = tmp_path / "pages"
    page_dir.mkdir()
    page_1 = page_dir / "page-01.jpg"
    page_2 = page_dir / "page-02.jpg"
    page_1.write_bytes(_JPEG_BYTES)
    page_2.write_bytes(_JPEG_BYTES)
    extraction = PdfPageExtractionResult(
        file_path=str(pdf),
        original_size=pdf.stat().st_size,
        output_dir=str(page_dir),
        pages=(
            PdfExtractedPage(file_path=str(page_1), page_number=1, size_bytes=len(_JPEG_BYTES)),
            PdfExtractedPage(file_path=str(page_2), page_number=2, size_bytes=len(_JPEG_BYTES)),
        ),
    )
    service = FakePdfDocumentService(extraction_result=extraction)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True, supports_documents=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(pdf), pages="1-2"), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(pdf), pages="1-2"), ctx)

    assert isinstance(validation, ValidationOk)
    assert result.is_error is False
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "text"
    assert "PDF page extraction:" in result.content[0]["text"]
    assert [block["media_type"] for block in result.content[1:]] == [
        "image/jpeg",
        "image/jpeg",
    ]
    assert result.content[1]["source"]["data"] == "/9j/anBlZw=="
    assert result.content[1]["metadata"] == {
        "file_path": str(pdf),
        "page_number": 1,
        "original_size": pdf.stat().st_size,
    }
    assert service.extraction_requests == [
        PdfPageExtractionRequest(
            file_path=str(pdf),
            first_page=1,
            last_page=2,
            max_pages=20,
        )
    ]
    assert service.cleaned_results == [extraction]
    assert ctx.read_file_state.get(str(pdf)) is None
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_pdf_pages_requires_image_capability_even_with_service(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = FakePdfDocumentService()
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_documents=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(pdf), pages="1"), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(pdf), pages="1"), ctx)

    assert isinstance(validation, ValidationError)
    assert "does not advertise image input support" in validation.message
    assert result.is_error is True
    assert "does not advertise image input support" in _content(result)
    assert service.extraction_requests == []


@pytest.mark.asyncio
async def test_file_read_pdf_pages_surfaces_service_errors_and_cleans_success(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = FakePdfDocumentService(
        extraction_error=PdfServiceError(
            "password_protected",
            "PDF is password-protected. Please provide an unprotected version.",
        )
    )
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(pdf), pages="1"), ctx)

    assert result.is_error is True
    assert "password-protected" in _content(result)
    assert service.cleaned_results == []


@pytest.mark.asyncio
async def test_file_read_pdf_pages_rejects_empty_service_output_and_cleans(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    extraction = PdfPageExtractionResult(
        file_path=str(pdf),
        original_size=pdf.stat().st_size,
        output_dir=str(tmp_path / "pages"),
        pages=(),
    )
    service = FakePdfDocumentService(extraction_result=extraction)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_images=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(pdf), pages="1"), ctx)

    assert result.is_error is True
    assert "produced no output pages" in _content(result)
    assert service.cleaned_results == [extraction]


@pytest.mark.asyncio
async def test_file_read_full_pdf_uses_service_page_count_before_lightweight_fallback(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = FakePdfDocumentService(page_count=11)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_documents=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(pdf)), ctx)

    assert result.is_error is True
    assert "This PDF has 11 pages" in _content(result)
    assert service.count_requests == [str(pdf)]


@pytest.mark.asyncio
async def test_file_read_large_full_pdf_uses_service_page_count_before_size_guard(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "large-paper.pdf"
    with pdf.open("wb") as stream:
        stream.write(b"%PDF-1.4\n")
        stream.truncate(FILE_READ_PDF_MAX_BYTES + 1)
    service = FakePdfDocumentService(page_count=11)
    ctx = _ctx_with_model_capabilities(
        tmp_path,
        ModelCapabilities(supports_documents=True),
        pdf_document_service=service,
    )
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(pdf)), ctx)

    assert result.is_error is True
    assert "This PDF has 11 pages" in _content(result)
    assert "exceeds maximum allowed structured-read size" not in _content(result)
    assert service.count_requests == [str(pdf)]


@pytest.mark.asyncio
async def test_file_read_returns_structured_notebook_cells_and_updates_baseline(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "analysis.ipynb"
    raw = json.dumps(
        {
            "metadata": {"language_info": {"name": "python"}},
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Heading\n", "notes"],
                },
                {
                    "id": "plot-cell",
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
                },
            ],
        }
    )
    notebook.write_text(raw, encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    validation = await tool.validate_input(ReadInput(file_path=str(notebook)), ctx)
    result = await _call_read(tool, ReadInput(file_path=str(notebook)), ctx)

    assert isinstance(validation, ValidationOk)
    assert result.is_error is False
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "text"
    text = result.content[0]["text"]
    assert '<cell id="cell-0"><cell_type>markdown</cell_type># Heading' in text
    assert '<cell id="plot-cell">display(plot)</cell id="plot-cell">' in text
    assert "\nplot" in text
    assert result.content[1] == {
        "type": "image",
        "media_type": "image/png",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo=",
        },
        "metadata": {
            "cell_id": "plot-cell",
            "output_type": "display_data",
        },
    }
    state = ctx.read_file_state.get(str(notebook))
    assert state is not None
    assert state.content == raw
    assert state.offset == 1
    assert state.limit is None
    assert state.is_partial_view is False
    assert state.timestamp == _mtime_ms(notebook)
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_repeated_unchanged_notebook_returns_stub(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "analysis.ipynb"
    raw = json.dumps(
        {
            "metadata": {"language_info": {"name": "python"}},
            "cells": [{"cell_type": "code", "source": "display(plot)", "outputs": []}],
        }
    )
    notebook.write_text(raw, encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    first = await _call_read(tool, ReadInput(file_path=str(notebook)), ctx)
    second = await _call_read(tool, ReadInput(file_path=str(notebook)), ctx)

    assert first.is_error is False
    assert isinstance(first.content, list)
    assert FILE_UNCHANGED_STUB not in str(first.content)
    assert second.is_error is False
    assert second.content == FILE_UNCHANGED_STUB
    state = ctx.read_file_state.get(str(notebook))
    assert state is not None
    assert state.offset == 1
    assert state.limit is None
    assert state.is_partial_view is False


@pytest.mark.asyncio
async def test_file_read_notebook_errors_do_not_update_baseline(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.ipynb"
    invalid.write_text("{", encoding="utf-8")
    oversized = tmp_path / "oversized.ipynb"
    oversized.write_text(
        json.dumps(
            {
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": "x" * (FILE_READ_DEFAULT_MAX_SIZE_BYTES + 1),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    invalid_result = await _call_read(tool, ReadInput(file_path=str(invalid)), ctx)
    oversized_result = await _call_read(tool, ReadInput(file_path=str(oversized)), ctx)

    assert invalid_result.is_error is True
    assert "Notebook is not valid JSON" in _content(invalid_result)
    assert oversized_result.is_error is True
    assert "Notebook content" in _content(oversized_result)
    assert "jq '.cells[:20]'" in _content(oversized_result)
    assert ctx.read_file_state.get(str(invalid)) is None
    assert ctx.read_file_state.get(str(oversized)) is None
    assert ctx.successful_text_read_paths == []


@pytest.mark.asyncio
async def test_file_read_notebook_raw_cap_errors_without_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = tmp_path / "raw-large.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": "x" * 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("raygent_harness.tools.file_read_tool.NOTEBOOK_MAX_RAW_BYTES", 32)
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(notebook)), ctx)

    assert result.is_error is True
    assert "structured-read size" in _content(result)
    assert ctx.read_file_state.get(str(notebook)) is None


@pytest.mark.asyncio
async def test_file_read_empty_notebook_returns_warning_and_updates_baseline(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "empty.ipynb"
    raw = '{"metadata": {}, "cells": []}'
    notebook.write_text(raw, encoding="utf-8")
    ctx = _ctx(tmp_path)
    tool = build_file_read_tool()

    result = await _call_read(tool, ReadInput(file_path=str(notebook)), ctx)

    assert result.is_error is False
    assert isinstance(result.content, list)
    assert "notebook exists but contains no cells" in result.content[0]["text"]
    state = ctx.read_file_state.get(str(notebook))
    assert state is not None
    assert state.content == raw
    assert ctx.successful_text_read_paths == []


def test_read_structured_file_content_enforces_media_size_cap(tmp_path: Path) -> None:
    image = tmp_path / "huge.png"
    image.write_bytes(b"x" * (FILE_READ_IMAGE_MAX_BYTES + 1))
    pdf = tmp_path / "huge.pdf"
    pdf.write_bytes(b"x" * (FILE_READ_PDF_MAX_BYTES + 1))

    for path, caps, expected in (
        (image, ModelCapabilities(supports_images=True), "Image file"),
        (pdf, ModelCapabilities(supports_documents=True), "PDF file"),
    ):
        with pytest.raises(FileReadToolError, match=expected):
            read_structured_file_content(str(path), capabilities=caps)


@pytest.mark.asyncio
async def test_file_read_check_permissions_uses_file_permission_helper(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    inside_file = repo / "inside.txt"
    outside_file = outside / "outside.txt"
    inside_file.write_text("inside", encoding="utf-8")
    outside_file.write_text("outside", encoding="utf-8")
    ctx = _ctx(repo)
    tool = build_file_read_tool()

    inside_result = await tool.check_permissions(
        ReadInput(file_path=str(inside_file)),
        ctx,
        ToolPermissionContext(),
    )
    outside_result = await tool.check_permissions(
        ReadInput(file_path=str(outside_file)),
        ctx,
        ToolPermissionContext(),
    )

    assert isinstance(inside_result, PermissionAllowDecision)
    assert isinstance(outside_result, PermissionAskDecision)
    assert outside_result.blocked_path == str(outside_file)


@pytest.mark.asyncio
async def test_file_read_allows_internal_persisted_outputs_after_rule_checks(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    outputs = tmp_path / "tool-results"
    repo.mkdir()
    outputs.mkdir()
    persisted = outputs / "toolu_1.txt"
    persisted.write_text("large output", encoding="utf-8")
    ctx = _ctx(repo)
    ctx.content_replacement = ContentReplacementState(
        max_result_size_chars=10,
        replaced_outputs_dir=str(outputs),
    )
    tool = build_file_read_tool()

    allowed = await tool.check_permissions(
        ReadInput(file_path=str(persisted)),
        ctx,
        ToolPermissionContext(),
    )
    denied = await tool.check_permissions(
        ReadInput(file_path=str(persisted)),
        ctx,
        ToolPermissionContext(
            always_deny_rules={
                "session": (f"{READ_RULE_NAME}({outputs}/**)",),
            },
        ),
    )

    assert isinstance(allowed, PermissionAllowDecision)
    assert isinstance(denied, PermissionDenyDecision)


def test_file_read_range_raises_cancelled_error_between_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "example.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    abort_event = asyncio.Event()

    real_open = builtins.open

    class AbortAfterFirstRead:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self._reads = 0

        def __enter__(self) -> AbortAfterFirstRead:
            self._wrapped.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            return self._wrapped.__exit__(exc_type, exc, traceback)

        def read(self, size: int = -1) -> bytes:
            self._reads += 1
            data = self._wrapped.read(size)
            if self._reads == 1:
                abort_event.set()
            return data

    def fake_open(*args: Any, **kwargs: Any) -> AbortAfterFirstRead:
        return AbortAfterFirstRead(real_open(*args, **kwargs))

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(asyncio.CancelledError):
        read_text_file_in_range(str(path), abort_event=abort_event)


@pytest.mark.asyncio
async def test_file_read_catalog_provider_appends_and_filters_existing(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    existing = build_file_read_tool()
    config = QueryConfig(model="model", tools=(existing,))

    provider = create_file_read_catalog_provider()
    tools = await provider(config, ctx, ())

    assert tools is not None
    assert tuple(tool.name for tool in tools) == (FILE_READ_TOOL_NAME,)
    assert tools[0] is not existing

    disabled = create_file_read_catalog_provider(enabled=False)
    disabled_tools = await disabled(config, ctx, ())
    assert disabled_tools == ()
