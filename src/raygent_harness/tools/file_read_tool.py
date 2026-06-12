"""Concrete model-callable file reader.


Raygent's `Read` implements text-file semantics plus provider-neutral structured
image/PDF-document results when the current model advertises the needed modality.
Notebook reads return bounded provider-neutral cell/output content.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import os
import re
import stat
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from raygent_harness.core.file_state import FileState
from raygent_harness.core.model_registry import model_info_with_fallback
from raygent_harness.core.model_types import ModelCapabilities
from raygent_harness.core.permissions import PermissionResult
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.services.file_media import (
    BINARY_FILE_EXTENSIONS,
    NATIVE_IMAGE_EXTENSIONS,
    NOTEBOOK_FILE_EXTENSIONS,
    NOTEBOOK_MAX_RAW_BYTES,
    PDF_FILE_EXTENSIONS,
    UNSUPPORTED_IMAGE_EXTENSIONS,
    FileMediaClassification,
    NotebookServiceError,
    PdfDocumentService,
    PdfPageExtractionRequest,
    PdfPageExtractionResult,
    PdfServiceError,
    classify_file_extension,
    classify_file_media,
    detect_supported_image_media_type,
    extension_for_path,
    is_blocked_device_path,
    notebook_cells_to_content,
    parse_notebook_content,
)
from raygent_harness.tools.file_permissions import (
    FILE_READ_TOOL_NAME,
    check_read_permission_for_path,
    expand_file_path,
    matching_file_permission_rule,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


FILE_READ_TOOL_ALIAS = "FileRead"
FILE_READ_MAX_RESULT_SIZE_CHARS = float("inf")
FILE_READ_DEFAULT_MAX_SIZE_BYTES = 256 * 1024
FILE_READ_FAST_PATH_MAX_BYTES = 10 * 1024 * 1024
FILE_READ_CHUNK_SIZE_BYTES = 512 * 1024
FILE_READ_DEFAULT_MAX_LINES = 2000
FILE_READ_IMAGE_MAX_BYTES = (5 * 1024 * 1024 * 3) // 4
FILE_READ_PDF_MAX_BYTES = 20 * 1024 * 1024
PDF_MAX_PAGES_PER_READ = 20
PDF_AT_MENTION_INLINE_THRESHOLD = 10

FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read "
    "tool_result in this conversation is still current -- refer to that "
    "instead of re-reading."
)

FILE_READ_SAFETY_REMINDER = (
    "\n\n<system-reminder>\n"
    "Whenever you read a file, you should consider whether it would be "
    "considered malware. You CAN and SHOULD provide analysis of malware, what "
    "it is doing. But you MUST refuse to improve or augment the code. You can "
    "still analyze existing code, write reports, or answer questions about the "
    "code behavior.\n"
    "</system-reminder>\n"
)

_PDF_PAGE_OBJECT_RE = re.compile(rb"/Type\s*/Page\b")


class ReadInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read.")
    offset: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The line number to start reading from. Only provide if the file is "
            "too large to read at once."
        ),
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description=(
            "The number of lines to read. Only provide if the file is too large "
            "to read at once."
        ),
    )
    pages: str | None = Field(
        default=None,
        description=(
            "PDF page range such as '1-5'. Requires an image-capable model and "
            "a configured PDF page extractor."
        ),
    )


@dataclass(frozen=True)
class ReadRangeResult:
    content: str
    line_count: int
    total_lines: int
    total_bytes: int
    read_bytes: int
    mtime_ms: int


@dataclass(frozen=True)
class ReadMediaResult:
    content: list[dict[str, Any]]
    media_kind: str
    media_type: str
    total_bytes: int
    mtime_ms: int
    state_content: str | None = None


class FileReadToolError(Exception):
    """Model-visible operational read error."""


class FileTooLargeError(FileReadToolError):
    def __init__(self, size_in_bytes: int, max_size_bytes: int) -> None:
        super().__init__(
            f"File content ({_format_file_size(size_in_bytes)}) exceeds maximum "
            f"allowed size ({_format_file_size(max_size_bytes)}). Use offset and "
            "limit parameters to read specific portions of the file, or search "
            "for specific content instead of reading the whole file."
        )


def build_file_read_tool(
    *,
    pdf_document_service: PdfDocumentService | None = None,
) -> Tool:
    """Build the concrete `Read` tool."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.file_path.strip():
            return ValidationError(message="file_path is required for Read")

        pages_error = _validate_pages(parsed.pages)
        if pages_error is not None:
            return ValidationError(message=pages_error)

        full_path = expand_file_path(parsed.file_path, cwd=ctx.cwd)
        if matching_file_permission_rule(
            full_path,
            ctx.permission_context,
            tool_type="read",
            behavior="deny",
            cwd=ctx.cwd,
        ) is not None:
            return ValidationError(
                message="File is in a directory that is denied by your permission settings."
            )

        classification = classify_file_extension(full_path)
        unsupported = _unsupported_error_for_classification(
            classification,
            ctx=ctx,
            pages=parsed.pages,
            pdf_document_service=_pdf_document_service_for_context(
                ctx,
                override=pdf_document_service,
            ),
        )
        if unsupported is not None:
            return ValidationError(message=unsupported)

        binary_error = _binary_error_for_classification(classification)
        if binary_error is not None:
            return ValidationError(message=binary_error)

        if is_blocked_device_path(full_path):
            return ValidationError(
                message=(
                    f"Cannot read '{parsed.file_path}': this device file would "
                    "block or produce infinite output."
                )
            )

        return ValidationOk()

    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = _coerce_input(input_)
        return check_read_permission_for_path(
            parsed.file_path,
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(exclude_none=True),
            extra_allowed_read_roots=_internal_read_roots(ctx),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        offset = parsed.offset if parsed.offset is not None else 1
        full_path = expand_file_path(parsed.file_path, cwd=ctx.cwd)
        classification = _classify_for_read(full_path)

        unsupported = _unsupported_error_for_classification(
            classification,
            ctx=ctx,
            pages=parsed.pages,
            pdf_document_service=_pdf_document_service_for_context(
                ctx,
                override=pdf_document_service,
            ),
        )
        if unsupported is not None:
            yield ToolResult(content=unsupported, is_error=True)
            return

        binary_error = _binary_error_for_classification(classification)
        if binary_error is not None:
            yield ToolResult(content=binary_error, is_error=True)
            return

        if ctx.abort_event.is_set():
            raise asyncio.CancelledError()

        if classification.kind == "notebook" and await _is_unchanged_read(
            full_path,
            ctx,
            offset=offset,
            limit=parsed.limit,
        ):
            yield ToolResult(content=FILE_UNCHANGED_STUB)
            return

        if _is_structured_media_classification(classification):
            try:
                media_result = await asyncio.to_thread(
                    read_structured_file_content,
                    full_path,
                    pages=parsed.pages,
                    capabilities=_effective_model_capabilities(ctx),
                    abort_event=ctx.abort_event,
                    classification=classification,
                    pdf_document_service=_pdf_document_service_for_context(
                        ctx,
                        override=pdf_document_service,
                    ),
                )
            except FileReadToolError as exc:
                yield ToolResult(content=str(exc), is_error=True)
                return
            except OSError as exc:
                yield ToolResult(
                    content=_format_os_error(parsed.file_path, full_path, exc, ctx.cwd),
                    is_error=True,
                )
                return
            if media_result.state_content is not None:
                state_offset = offset if media_result.media_kind == "notebook" else None
                state_limit = parsed.limit if media_result.media_kind == "notebook" else None
                ctx.read_file_state.set(
                    full_path,
                    FileState(
                        content=media_result.state_content,
                        timestamp=media_result.mtime_ms,
                        offset=state_offset,
                        limit=state_limit,
                        is_partial_view=False,
                    ),
                )
            yield ToolResult(content=media_result.content)
            return

        if await _is_unchanged_read(
            full_path,
            ctx,
            offset=offset,
            limit=parsed.limit,
        ):
            yield ToolResult(content=FILE_UNCHANGED_STUB)
            return

        try:
            result = await asyncio.to_thread(
                read_text_file_in_range,
                full_path,
                offset=offset,
                limit=parsed.limit,
                max_size_bytes=FILE_READ_DEFAULT_MAX_SIZE_BYTES,
                abort_event=ctx.abort_event,
            )
        except FileReadToolError as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        except OSError as exc:
            yield ToolResult(
                content=_format_os_error(parsed.file_path, full_path, exc, ctx.cwd),
                is_error=True,
            )
            return
        except UnicodeError as exc:
            yield ToolResult(
                content=f"Cannot read '{parsed.file_path}' as UTF-8 text: {exc}",
                is_error=True,
            )
            return

        ctx.read_file_state.set(
            full_path,
            FileState(
                content=result.content,
                timestamp=result.mtime_ms,
                offset=offset,
                limit=parsed.limit,
                is_partial_view=parsed.limit is not None or offset not in {0, 1},
            ),
        )
        ctx.successful_text_read_paths.append(full_path)
        yield ToolResult(content=_format_text_result(result, start_line=offset))

    return build_tool(
        ToolSpec(
            name=FILE_READ_TOOL_NAME,
            aliases=(FILE_READ_TOOL_ALIAS,),
            description="Read a file from the local filesystem.",
            search_hint="read files, images, PDFs, and notebooks from disk",
            input_model=ReadInput,
            call=call,
            prompt=FILE_READ_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            should_defer=False,
            always_load=False,
            max_result_size_chars=FILE_READ_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Reading {_coerce_input(input_).file_path}"
            ),
        )
    )


def create_file_read_catalog_provider(
    *,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
    pdf_document_service: PdfDocumentService | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends `Read` when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != FILE_READ_TOOL_NAME)
        if not enabled:
            return without_existing
        return (
            *without_existing,
            build_file_read_tool(pdf_document_service=pdf_document_service),
        )

    return provider


def read_text_file_in_range(
    path: str,
    *,
    offset: int = 1,
    limit: int | None = None,
    max_size_bytes: int = FILE_READ_DEFAULT_MAX_SIZE_BYTES,
    abort_event: asyncio.Event | None = None,
) -> ReadRangeResult:
    """Read a line range while bounding selected output memory."""

    _raise_if_aborted(abort_event)
    stats = os.stat(path)
    _raise_if_aborted(abort_event)
    if stat.S_ISDIR(stats.st_mode):
        raise FileReadToolError(
            f"Cannot read '{path}': this path is a directory. Use a shell list "
            "command to inspect directories."
        )
    if not stat.S_ISREG(stats.st_mode):
        raise FileReadToolError(
            f"Cannot read '{path}': this path is a device or special file."
        )
    if limit is None and stats.st_size > max_size_bytes:
        raise FileTooLargeError(stats.st_size, max_size_bytes)

    line_offset = 0 if offset == 0 else offset - 1
    end_line = line_offset + limit if limit is not None else None
    reader = _RangeReader(
        line_offset=line_offset,
        end_line=end_line,
        max_output_bytes=max_size_bytes,
    )
    total_bytes = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    _raise_if_aborted(abort_event)
    with open(path, "rb") as file:
        first_chunk = True
        while True:
            _raise_if_aborted(abort_event)
            chunk = file.read(FILE_READ_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            if b"\x00" in chunk:
                raise FileReadToolError(
                    "This tool cannot read binary files. The file appears to "
                    "contain NUL bytes."
                )
            total_bytes += len(chunk)
            if limit is None and total_bytes > max_size_bytes:
                raise FileTooLargeError(total_bytes, max_size_bytes)
            text = decoder.decode(chunk)
            if first_chunk:
                first_chunk = False
                if text.startswith("\ufeff"):
                    text = text[1:]
            reader.feed(text)

        tail = decoder.decode(b"", final=True)
        if tail:
            reader.feed(tail)
        reader.finish(saw_bytes=total_bytes > 0)

    content = "\n".join(reader.selected_lines)
    return ReadRangeResult(
        content=content,
        line_count=len(reader.selected_lines),
        total_lines=reader.total_lines,
        total_bytes=total_bytes,
        read_bytes=len(content.encode()),
        mtime_ms=_mtime_ms(stats),
    )


def read_structured_file_content(
    path: str,
    *,
    pages: str | None = None,
    capabilities: ModelCapabilities | None = None,
    abort_event: asyncio.Event | None = None,
    classification: FileMediaClassification | None = None,
    pdf_document_service: PdfDocumentService | None = None,
) -> ReadMediaResult:
    """Read a structured image/document payload with bounded in-memory bytes."""

    _raise_if_aborted(abort_event)
    stats = os.stat(path)
    _raise_if_aborted(abort_event)
    if stat.S_ISDIR(stats.st_mode):
        raise FileReadToolError(
            f"Cannot read '{path}': this path is a directory. Use a shell list "
            "command to inspect directories."
        )
    if not stat.S_ISREG(stats.st_mode):
        raise FileReadToolError(
            f"Cannot read '{path}': this path is a device or special file."
        )

    caps = capabilities or ModelCapabilities()
    media = classification or _classify_for_read(path)
    if media.kind in {"native_image", "unsupported_image"}:
        return _read_image_file(
            path,
            stats=stats,
            capabilities=caps,
            abort_event=abort_event,
            classification=media,
        )
    if media.kind == "pdf":
        return _read_pdf_file(
            path,
            stats=stats,
            pages=pages,
            capabilities=caps,
            abort_event=abort_event,
            pdf_document_service=pdf_document_service,
        )
    if media.kind == "notebook":
        return _read_notebook_file(
            path,
            stats=stats,
            abort_event=abort_event,
        )
    raise FileReadToolError(f"Cannot read '{path}' as structured media.")


def _read_image_file(
    path: str,
    *,
    stats: os.stat_result,
    capabilities: ModelCapabilities,
    abort_event: asyncio.Event | None,
    classification: FileMediaClassification,
) -> ReadMediaResult:
    if not _supports_images(capabilities):
        raise FileReadToolError(
            "The current model/provider does not advertise image input support. "
            "Use a text extraction or shell command instead."
        )
    if classification.kind != "native_image":
        raise FileReadToolError(
            f"Reading image type '.{classification.extension}' is not supported by "
            "Raygent's provider-neutral Read tool. Supported image types are "
            "PNG, JPEG, GIF, and WebP."
        )
    if stats.st_size == 0:
        raise FileReadToolError(f"Image file is empty: {path}")
    if stats.st_size > FILE_READ_IMAGE_MAX_BYTES:
        raise FileReadToolError(
            f"Image file ({_format_file_size(stats.st_size)}) exceeds maximum "
            f"allowed structured-read size ({_format_file_size(FILE_READ_IMAGE_MAX_BYTES)})."
        )
    raw_data = _read_file_bytes(
        path,
        max_bytes=FILE_READ_IMAGE_MAX_BYTES,
        abort_event=abort_event,
    )
    media_type = detect_supported_image_media_type(raw_data)
    if media_type is None:
        raise FileReadToolError(
            f"Image file is not a valid PNG, JPEG, GIF, or WebP image: {path}"
        )
    data = base64.b64encode(raw_data).decode("ascii")
    block = {
        "type": "image",
        "media_type": media_type,
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
        "metadata": {
            "file_path": path,
            "original_size": stats.st_size,
        },
    }
    return ReadMediaResult(
        content=[block],
        media_kind="image",
        media_type=media_type,
        total_bytes=stats.st_size,
        mtime_ms=_mtime_ms(stats),
    )


def _read_pdf_file(
    path: str,
    *,
    stats: os.stat_result,
    pages: str | None,
    capabilities: ModelCapabilities,
    abort_event: asyncio.Event | None,
    pdf_document_service: PdfDocumentService | None,
) -> ReadMediaResult:
    if pages:
        return _read_pdf_pages(
            path,
            stats=stats,
            pages=pages,
            capabilities=capabilities,
            abort_event=abort_event,
            pdf_document_service=pdf_document_service,
        )
    if not capabilities.supports_documents:
        raise FileReadToolError(
            "The current model/provider does not advertise PDF/document input "
            "support. Use the pages parameter only after a page-image extractor "
            "is configured, or use a shell/text extraction command."
        )
    if stats.st_size == 0:
        raise FileReadToolError(f"PDF file is empty: {path}")
    page_count = _service_pdf_page_count(pdf_document_service, path)
    if page_count is not None and page_count > PDF_AT_MENTION_INLINE_THRESHOLD:
        raise FileReadToolError(
            f"This PDF has {page_count} pages, which is too many to read at once. "
            'Use the pages parameter to read specific page ranges (e.g., pages: "1-5"). '
            f"Maximum {PDF_MAX_PAGES_PER_READ} pages per request."
        )
    if stats.st_size > FILE_READ_PDF_MAX_BYTES:
        raise FileReadToolError(
            f"PDF file ({_format_file_size(stats.st_size)}) exceeds maximum "
            f"allowed structured-read size ({_format_file_size(FILE_READ_PDF_MAX_BYTES)})."
        )
    raw_data = _read_file_bytes(
        path,
        max_bytes=FILE_READ_PDF_MAX_BYTES,
        abort_event=abort_event,
    )
    if not raw_data.startswith(b"%PDF-"):
        raise FileReadToolError(f"File is not a valid PDF (missing %PDF- header): {path}")
    if page_count is None:
        page_count = _detect_pdf_page_count(raw_data)
    if page_count is not None and page_count > PDF_AT_MENTION_INLINE_THRESHOLD:
        raise FileReadToolError(
            f"This PDF has {page_count} pages, which is too many to read at once. "
            'Use the pages parameter to read specific page ranges (e.g., pages: "1-5"). '
            f"Maximum {PDF_MAX_PAGES_PER_READ} pages per request."
        )
    data = base64.b64encode(raw_data).decode("ascii")
    metadata_text = f"PDF file read: {path} ({_format_file_size(stats.st_size)})"
    return ReadMediaResult(
        content=[
            {"type": "text", "text": metadata_text},
            {
                "type": "document",
                "media_type": "application/pdf",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
                "metadata": {
                    "file_path": path,
                    "original_size": stats.st_size,
                },
            },
        ],
        media_kind="document",
        media_type="application/pdf",
        total_bytes=stats.st_size,
        mtime_ms=_mtime_ms(stats),
    )


def _read_pdf_pages(
    path: str,
    *,
    stats: os.stat_result,
    pages: str,
    capabilities: ModelCapabilities,
    abort_event: asyncio.Event | None,
    pdf_document_service: PdfDocumentService | None,
) -> ReadMediaResult:
    if not _supports_images(capabilities):
        raise FileReadToolError(
            "The current model/provider does not advertise image input support "
            "required for PDF page-range extraction."
        )
    if pdf_document_service is None:
        raise FileReadToolError(_pdf_extraction_unavailable_message())
    if stats.st_size == 0:
        raise FileReadToolError(f"PDF file is empty: {path}")
    first, last = _page_range_for_extraction(pages)
    _raise_if_aborted(abort_event)
    try:
        result = pdf_document_service.extract_pages(
            PdfPageExtractionRequest(
                file_path=path,
                first_page=first,
                last_page=last,
                max_pages=PDF_MAX_PAGES_PER_READ,
            )
        )
    except PdfServiceError as exc:
        raise FileReadToolError(exc.message) from exc
    if not result.pages:
        pdf_document_service.cleanup_extraction(result)
        raise FileReadToolError(
            "PDF page extraction produced no output pages. The PDF may be invalid."
        )
    try:
        content = _pdf_page_result_content(
            result,
            requested_pages=pages,
            abort_event=abort_event,
        )
    finally:
        pdf_document_service.cleanup_extraction(result)
    return ReadMediaResult(
        content=content,
        media_kind="pdf_pages",
        media_type="image/jpeg",
        total_bytes=stats.st_size,
        mtime_ms=_mtime_ms(stats),
    )


def _pdf_page_result_content(
    result: PdfPageExtractionResult,
    *,
    requested_pages: str,
    abort_event: asyncio.Event | None,
) -> list[dict[str, Any]]:
    extracted = "page" if result.page_count == 1 else "pages"
    metadata_text = (
        f"PDF page extraction: {result.file_path} pages {requested_pages} "
        f"({result.page_count} {extracted}, original size "
        f"{_format_file_size(result.original_size)})"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": metadata_text}]
    for page in result.pages:
        if page.size_bytes is not None and page.size_bytes > FILE_READ_IMAGE_MAX_BYTES:
            raise FileReadToolError(
                f"Extracted PDF page {page.page_number} "
                f"({_format_file_size(page.size_bytes)}) exceeds maximum "
                f"allowed structured-read size ({_format_file_size(FILE_READ_IMAGE_MAX_BYTES)})."
            )
        raw_data = _read_file_bytes(
            page.file_path,
            max_bytes=FILE_READ_IMAGE_MAX_BYTES,
            abort_event=abort_event,
        )
        media_type = detect_supported_image_media_type(raw_data)
        if media_type is None:
            raise FileReadToolError(
                f"Extracted PDF page {page.page_number} is not a valid image."
            )
        data = base64.b64encode(raw_data).decode("ascii")
        content.append(
            {
                "type": "image",
                "media_type": media_type,
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
                "metadata": {
                    "file_path": result.file_path,
                    "page_number": page.page_number,
                    "original_size": result.original_size,
                },
            }
        )
    return content


def _read_notebook_file(
    path: str,
    *,
    stats: os.stat_result,
    abort_event: asyncio.Event | None,
) -> ReadMediaResult:
    if stats.st_size == 0:
        raise FileReadToolError(f"Notebook file is empty: {path}")
    raw_data = _read_file_bytes(
        path,
        max_bytes=NOTEBOOK_MAX_RAW_BYTES,
        abort_event=abort_event,
    )
    try:
        raw_content = raw_data.decode("utf-8")
    except UnicodeError as exc:
        raise FileReadToolError(f"Cannot read notebook as UTF-8 JSON: {exc}") from exc
    try:
        notebook = parse_notebook_content(raw_content, file_path=path)
    except NotebookServiceError as exc:
        raise FileReadToolError(exc.message) from exc

    content = notebook_cells_to_content(notebook.cells)
    if not content:
        content = [
            {
                "type": "text",
                "text": (
                    "<system-reminder>Warning: the notebook exists but contains "
                    "no cells.</system-reminder>"
                ),
            }
        ]
    return ReadMediaResult(
        content=content,
        media_kind="notebook",
        media_type="application/x-ipynb+json",
        total_bytes=stats.st_size,
        mtime_ms=_mtime_ms(stats),
        state_content=notebook.raw_content,
    )


def _read_file_bytes(
    path: str,
    *,
    max_bytes: int,
    abort_event: asyncio.Event | None,
) -> bytes:
    _raise_if_aborted(abort_event)
    with open(path, "rb") as handle:
        data = handle.read(max_bytes + 1)
    _raise_if_aborted(abort_event)
    if len(data) > max_bytes:
        raise FileReadToolError(
            f"File content exceeds maximum allowed structured-read size "
            f"({_format_file_size(max_bytes)})."
        )
    return data


def _detect_pdf_page_count(data: bytes) -> int | None:
    matches = _PDF_PAGE_OBJECT_RE.findall(data)
    return len(matches) if matches else None


@dataclass
class _RangeReader:
    line_offset: int
    end_line: int | None
    max_output_bytes: int
    selected_lines: list[str] = field(default_factory=list[str])
    total_lines: int = 0
    selected_bytes: int = 0
    _partial: str = ""
    _pending_line_has_data: bool = False
    _last_char_was_newline: bool = False

    def feed(self, text: str) -> None:
        data = self._partial + text if self._partial else text
        self._last_char_was_newline = data.endswith("\n")
        self._partial = ""
        start = 0
        while True:
            newline = data.find("\n", start)
            if newline == -1:
                break
            line = data[start:newline]
            if line.endswith("\r"):
                line = line[:-1]
            self._complete_line(line)
            start = newline + 1

        if start < len(data):
            fragment = data[start:]
            self._pending_line_has_data = True
            if self._line_is_selected(self.total_lines):
                self._partial = fragment
                self._ensure_current_partial_fits()

    def finish(self, *, saw_bytes: bool) -> None:
        if not saw_bytes:
            return
        if self._pending_line_has_data:
            self._complete_line(self._partial)
            self._partial = ""
            self._pending_line_has_data = False
        elif self._last_char_was_newline:
            self._complete_line("")

    def _complete_line(self, line: str) -> None:
        if self._line_is_selected(self.total_lines):
            self._append_selected_line(line)
        self.total_lines += 1
        self._pending_line_has_data = False

    def _append_selected_line(self, line: str) -> None:
        next_size = self._size_after_adding(line)
        if next_size > self.max_output_bytes:
            raise FileTooLargeError(next_size, self.max_output_bytes)
        self.selected_bytes = next_size
        self.selected_lines.append(line)

    def _ensure_current_partial_fits(self) -> None:
        if self._size_after_adding(self._partial) > self.max_output_bytes:
            raise FileTooLargeError(
                self._size_after_adding(self._partial),
                self.max_output_bytes,
            )

    def _size_after_adding(self, line: str) -> int:
        separator_bytes = 1 if self.selected_lines else 0
        return self.selected_bytes + separator_bytes + len(line.encode())

    def _line_is_selected(self, line_index: int) -> bool:
        return line_index >= self.line_offset and (
            self.end_line is None or line_index < self.end_line
        )


async def _is_unchanged_read(
    full_path: str,
    ctx: ToolUseContext,
    *,
    offset: int,
    limit: int | None,
) -> bool:
    existing = ctx.read_file_state.get(full_path)
    if existing is None or existing.is_partial_view or existing.offset is None:
        return False
    if existing.offset != offset or existing.limit != limit:
        return False
    try:
        current_mtime = await asyncio.to_thread(_path_mtime_ms, full_path)
    except OSError:
        return False
    return current_mtime == existing.timestamp


def _format_text_result(result: ReadRangeResult, *, start_line: int) -> str:
    if result.line_count > 0:
        return _add_line_numbers(
            result.content,
            start_line=start_line,
        ) + FILE_READ_SAFETY_REMINDER
    if result.total_lines == 0:
        return (
            "<system-reminder>Warning: the file exists but the contents are "
            "empty.</system-reminder>"
        )
    return (
        "<system-reminder>Warning: the file exists but is shorter than the "
        f"provided offset ({start_line}). The file has {result.total_lines} "
        "lines.</system-reminder>"
    )


def _add_line_numbers(content: str, *, start_line: int) -> str:
    return "\n".join(
        f"{index + start_line}\t{line}"
        for index, line in enumerate(content.split("\n"))
    )


def _coerce_input(input_: BaseModel) -> ReadInput:
    if isinstance(input_, ReadInput):
        return input_
    return ReadInput.model_validate(input_.model_dump())


def _internal_read_roots(ctx: ToolUseContext) -> tuple[str, ...]:
    if ctx.content_replacement is None:
        return ()
    return (ctx.content_replacement.replaced_outputs_dir,)


def _raise_if_aborted(abort_event: asyncio.Event | None) -> None:
    if abort_event is not None and abort_event.is_set():
        raise asyncio.CancelledError()


def _validate_pages(pages: str | None) -> str | None:
    if pages is None:
        return None
    parsed = _parse_page_range(pages)
    if parsed is None:
        return (
            f'Invalid pages parameter: "{pages}". Use formats like "1-5", "3", '
            'or "10-20". Pages are 1-indexed.'
        )
    first, last = parsed
    range_size = PDF_MAX_PAGES_PER_READ + 1 if last is None else last - first + 1
    if range_size > PDF_MAX_PAGES_PER_READ:
        return (
            f'Page range "{pages}" exceeds maximum of {PDF_MAX_PAGES_PER_READ} '
            "pages per request. Please use a smaller range."
        )
    return None


def _parse_page_range(pages: str) -> tuple[int, int | None] | None:
    trimmed = pages.strip()
    if not trimmed:
        return None
    if trimmed.endswith("-"):
        try:
            first = int(trimmed[:-1])
        except ValueError:
            return None
        return (first, None) if first >= 1 else None
    if "-" not in trimmed:
        try:
            page = int(trimmed)
        except ValueError:
            return None
        return (page, page) if page >= 1 else None
    first_text, last_text = trimmed.split("-", 1)
    try:
        first = int(first_text)
        last = int(last_text)
    except ValueError:
        return None
    if first < 1 or last < 1 or last < first:
        return None
    return first, last


def _page_range_for_extraction(pages: str) -> tuple[int, int]:
    parsed = _parse_page_range(pages)
    if parsed is None:
        raise FileReadToolError(
            f'Invalid pages parameter: "{pages}". Use formats like "1-5", "3", '
            'or "10-20". Pages are 1-indexed.'
        )
    first, last = parsed
    if last is None:
        raise FileReadToolError(
            f'Page range "{pages}" exceeds maximum of {PDF_MAX_PAGES_PER_READ} '
            "pages per request. Please use a smaller range."
        )
    range_size = last - first + 1
    if range_size > PDF_MAX_PAGES_PER_READ:
        raise FileReadToolError(
            f'Page range "{pages}" exceeds maximum of {PDF_MAX_PAGES_PER_READ} '
            "pages per request. Please use a smaller range."
        )
    return first, last


def _classify_for_read(path: str) -> FileMediaClassification:
    try:
        return classify_file_media(path)
    except OSError:
        return FileMediaClassification(
            path=path,
            extension=extension_for_path(path),
            kind="unknown",
            source="fallback",
            exists=False,
        )


def _unsupported_error_for_classification(
    classification: FileMediaClassification,
    *,
    ctx: ToolUseContext,
    pages: str | None,
    pdf_document_service: PdfDocumentService | None,
) -> str | None:
    ext = classification.extension
    if classification.kind == "notebook" or ext in NOTEBOOK_FILE_EXTENSIONS:
        return None
    if classification.kind == "pdf" or ext in PDF_FILE_EXTENSIONS:
        caps = _effective_model_capabilities(ctx)
        if pages:
            if not _supports_images(caps):
                return (
                    "The current model/provider does not advertise image input "
                    "support required for PDF page-range extraction."
                )
            return (
                None
                if pdf_document_service is not None
                else _pdf_extraction_unavailable_message()
            )
        return (
            None
            if caps.supports_documents
            else "The current model/provider does not advertise PDF/document input support."
        )
    if classification.kind == "unsupported_image" or ext in UNSUPPORTED_IMAGE_EXTENSIONS:
        return (
            f"Reading image type '.{ext}' is not supported by Raygent's "
            "provider-neutral Read tool. Supported image types are PNG, JPEG, "
            "GIF, and WebP."
        )
    if classification.kind == "native_image" or ext in NATIVE_IMAGE_EXTENSIONS:
        return (
            None
            if _supports_images(_effective_model_capabilities(ctx))
            else "The current model/provider does not advertise image input support."
        )
    return None


def _is_structured_media_classification(
    classification: FileMediaClassification,
) -> bool:
    return classification.kind in {"native_image", "unsupported_image", "pdf", "notebook"}


def _supports_images(capabilities: ModelCapabilities) -> bool:
    return capabilities.supports_images or capabilities.supports_media


def _effective_model_capabilities(ctx: ToolUseContext) -> ModelCapabilities:
    runtime = ctx.runtime
    if runtime is None:
        return ModelCapabilities()
    model = runtime.effective_model or runtime.config.model
    try:
        return model_info_with_fallback(
            model,
            provider=runtime.deps.model_provider,
        ).capabilities
    except Exception:
        return ModelCapabilities()


def _pdf_document_service_for_context(
    ctx: ToolUseContext,
    *,
    override: PdfDocumentService | None,
) -> PdfDocumentService | None:
    if override is not None:
        return override
    runtime = ctx.runtime
    if runtime is None:
        return None
    if runtime.pdf_document_service is not None:
        return runtime.pdf_document_service
    return runtime.deps.pdf_document_service


def _service_pdf_page_count(
    pdf_document_service: PdfDocumentService | None,
    path: str,
) -> int | None:
    if pdf_document_service is None:
        return None
    try:
        return pdf_document_service.get_page_count(path).page_count
    except Exception:
        return None


def _pdf_extraction_unavailable_message() -> str:
    return (
        "PDF page-range extraction is not available because no PDF document "
        "service is configured. Configure a provider-neutral PDF extractor, or "
        "read the full PDF with a document-capable model."
    )


def _binary_error_for_classification(
    classification: FileMediaClassification,
) -> str | None:
    ext = classification.extension
    if classification.kind == "binary" or ext in BINARY_FILE_EXTENSIONS:
        suffix = f" {ext}" if classification.binary_reason == "extension" and ext else ""
        return (
            f"This tool cannot read binary files. The file appears to be a "
            f"binary{suffix} file. Please use appropriate tools for binary "
            "file analysis."
        )
    return None


def _format_os_error(original_path: str, full_path: str, exc: OSError, cwd: str) -> str:
    if exc.errno == 2:
        return (
            "File does not exist. Current working directory is "
            f"{cwd}. Path requested: {original_path}."
        )
    reason = exc.strerror or str(exc)
    return f"Cannot read '{original_path}' ({full_path}): {reason}"


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _mtime_ms(stats: os.stat_result) -> int:
    return int(stats.st_mtime_ns // 1_000_000)


def _path_mtime_ms(path: str) -> int:
    return _mtime_ms(os.stat(path))


FILE_READ_PROMPT = (
    "Reads a file from the local filesystem.\n\n"
    "Usage:\n"
    "- The file_path parameter should be an absolute path.\n"
    "- Text results are returned using cat -n format, with line numbers starting "
    "at 1. Use offset and limit to read a specific text line range.\n"
    "- Images and full PDFs can be returned as structured model-visible content "
    "only when the current model/provider advertises image or document support.\n"
    "- Use pages with PDFs to extract specific page ranges as image blocks when "
    'a PDF extractor is configured, for example pages: "1-5".\n'
    "- Jupyter notebooks (.ipynb) are returned as bounded structured cells and "
    "outputs, not raw JSON.\n"
    "- This tool can only read files, not directories.\n"
    "- If you read a file that exists but has empty contents you will receive a "
    "system reminder warning in place of file contents.\n"
    f"- Files larger than {_format_file_size(FILE_READ_DEFAULT_MAX_SIZE_BYTES)} "
    "require offset and limit or a more targeted search tool for text reads.\n"
)


__all__ = [
    "FILE_READ_DEFAULT_MAX_LINES",
    "FILE_READ_DEFAULT_MAX_SIZE_BYTES",
    "FILE_READ_MAX_RESULT_SIZE_CHARS",
    "FILE_READ_PROMPT",
    "FILE_READ_TOOL_ALIAS",
    "FILE_READ_TOOL_NAME",
    "FILE_UNCHANGED_STUB",
    "FileReadToolError",
    "FileTooLargeError",
    "ReadInput",
    "ReadMediaResult",
    "ReadRangeResult",
    "build_file_read_tool",
    "create_file_read_catalog_provider",
    "read_structured_file_content",
    "read_text_file_in_range",
]
