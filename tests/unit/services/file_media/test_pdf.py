from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from raygent_harness.services.file_media import (
    PDF_MAX_EXTRACT_SIZE_BYTES,
    PDF_PAGE_RENDER_DPI,
    CommandBackedPdfDocumentService,
    CommandResult,
    PdfPageExtractionRequest,
    PdfServiceError,
    SubprocessCommandRunner,
)

_JPEG_BYTES = b"\xff\xd8\xffjpeg"


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self.pdfinfo = CommandResult(code=0, stdout="Title: x\nPages: 12\n")
        self.available = CommandResult(code=0, stderr="pdftoppm version")
        self.extract = CommandResult(code=0)
        self.write_pages = True

    def run(self, args: Sequence[str], *, timeout_s: float) -> CommandResult:
        command = tuple(args)
        self.calls.append((command, timeout_s))
        if command[:1] == ("pdfinfo",):
            return self.pdfinfo
        if command == ("pdftoppm", "-v"):
            return self.available
        if command[:1] == ("pdftoppm",):
            if self.write_pages:
                prefix = command[-1]
                Path(f"{prefix}-02.jpg").write_bytes(_JPEG_BYTES)
                Path(f"{prefix}-01.jpg").write_bytes(_JPEG_BYTES)
            return self.extract
        raise AssertionError(f"unexpected command: {command}")


class FailingExtractRunner(RecordingRunner):
    def run(self, args: Sequence[str], *, timeout_s: float) -> CommandResult:
        command = tuple(args)
        if command[:1] == ("pdftoppm",) and command != ("pdftoppm", "-v"):
            raise OSError("binary disappeared")
        return super().run(args, timeout_s=timeout_s)


class ExplodingExtractRunner(RecordingRunner):
    def run(self, args: Sequence[str], *, timeout_s: float) -> CommandResult:
        command = tuple(args)
        if command[:1] == ("pdftoppm",) and command != ("pdftoppm", "-v"):
            raise RuntimeError("runner exploded")
        return super().run(args, timeout_s=timeout_s)


def test_subprocess_runner_uses_argument_vector_without_project_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        captured["args"] = args
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = SubprocessCommandRunner().run(("pdfinfo", "/tmp/file.pdf"), timeout_s=3.0)

    assert result == CommandResult(code=0, stdout="ok", stderr="")
    assert captured["args"] == ["pdfinfo", "/tmp/file.pdf"]
    assert captured["shell"] is False
    assert captured["cwd"] == os.path.abspath(os.sep)
    assert captured["timeout"] == 3.0
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_command_service_parses_pdfinfo_page_count() -> None:
    runner = RecordingRunner()
    service = CommandBackedPdfDocumentService(command_runner=runner)

    page_count = service.get_page_count("/tmp/report.pdf")
    runner.pdfinfo = CommandResult(code=1, stderr="missing")
    unavailable = service.get_page_count("/tmp/report.pdf")

    assert page_count.page_count == 12
    assert unavailable.page_count is None
    assert runner.calls[0] == (("pdfinfo", "/tmp/report.pdf"), 10.0)


def test_command_service_extracts_sorted_pages_and_cleans_up(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    runner = RecordingRunner()
    service = CommandBackedPdfDocumentService(
        command_runner=runner,
        output_root=str(tmp_path),
    )

    result = service.extract_pages(
        PdfPageExtractionRequest(
            file_path=str(pdf),
            first_page=2,
            last_page=3,
            max_pages=20,
        )
    )

    assert [page.page_number for page in result.pages] == [2, 3]
    assert [Path(page.file_path).name for page in result.pages] == [
        "page-01.jpg",
        "page-02.jpg",
    ]
    assert result.original_size == pdf.stat().st_size
    assert Path(result.output_dir).exists()
    assert runner.calls[1][0][:8] == (
        "pdftoppm",
        "-jpeg",
        "-r",
        str(PDF_PAGE_RENDER_DPI),
        "-f",
        "2",
        "-l",
        "3",
    )

    service.cleanup_extraction(result)

    assert not Path(result.output_dir).exists()


def test_command_service_caches_pdftoppm_availability(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    runner = RecordingRunner()
    service = CommandBackedPdfDocumentService(command_runner=runner, output_root=str(tmp_path))
    request = PdfPageExtractionRequest(
        file_path=str(pdf),
        first_page=1,
        last_page=1,
        max_pages=20,
    )

    first = service.extract_pages(request)
    service.cleanup_extraction(first)
    second = service.extract_pages(request)
    service.cleanup_extraction(second)

    availability_calls = [
        call for call in runner.calls if call[0] == ("pdftoppm", "-v")
    ]
    assert len(availability_calls) == 1


def test_command_service_fails_softly_and_cleans_temp_output(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    runner = RecordingRunner()
    runner.extract = CommandResult(code=1, stderr="password required")
    runner.write_pages = False
    service = CommandBackedPdfDocumentService(command_runner=runner, output_root=str(tmp_path))

    with pytest.raises(PdfServiceError) as exc_info:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(pdf),
                first_page=1,
                last_page=1,
                max_pages=20,
            )
        )

    assert exc_info.value.reason == "password_protected"
    assert not list(tmp_path.glob("raygent-pdf-*"))


def test_command_service_converts_extract_oserror_and_cleans_output(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = CommandBackedPdfDocumentService(
        command_runner=FailingExtractRunner(),
        output_root=str(tmp_path),
    )

    with pytest.raises(PdfServiceError) as exc_info:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(pdf),
                first_page=1,
                last_page=1,
                max_pages=20,
            )
        )

    assert exc_info.value.reason == "unavailable"
    assert not list(tmp_path.glob("raygent-pdf-*"))


def test_command_service_converts_unexpected_extract_error_and_cleans_output(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = CommandBackedPdfDocumentService(
        command_runner=ExplodingExtractRunner(),
        output_root=str(tmp_path),
    )

    with pytest.raises(PdfServiceError) as exc_info:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(pdf),
                first_page=1,
                last_page=1,
                max_pages=20,
            )
        )

    assert exc_info.value.reason == "unknown"
    assert "runner exploded" in exc_info.value.message
    assert not list(tmp_path.glob("raygent-pdf-*"))


def test_command_service_bounds_input_size_and_empty_output(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    too_large = tmp_path / "large.pdf"
    too_large.write_bytes(b"%PDF-1.4\n" + b"x" * 20)
    no_output = tmp_path / "no-output.pdf"
    no_output.write_bytes(b"%PDF-1.4\n%%EOF\n")
    runner = RecordingRunner()
    runner.write_pages = False
    service = CommandBackedPdfDocumentService(command_runner=runner, output_root=str(tmp_path))

    with pytest.raises(PdfServiceError, match="empty") as empty_exc:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(empty),
                first_page=1,
                last_page=1,
                max_pages=20,
            )
        )
    with pytest.raises(PdfServiceError, match="maximum") as large_exc:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(too_large),
                first_page=1,
                last_page=1,
                max_pages=20,
                max_input_bytes=10,
            )
        )
    with pytest.raises(PdfServiceError, match="no output") as output_exc:
        service.extract_pages(
            PdfPageExtractionRequest(
                file_path=str(no_output),
                first_page=1,
                last_page=1,
                max_pages=20,
            )
        )

    assert empty_exc.value.reason == "empty"
    assert large_exc.value.reason == "too_large"
    assert output_exc.value.reason == "corrupted"
    assert PDF_MAX_EXTRACT_SIZE_BYTES == 100 * 1024 * 1024
