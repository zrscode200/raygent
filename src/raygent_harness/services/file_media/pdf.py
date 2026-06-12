"""PDF metadata and page extraction services.

The command-backed implementation is optional and dependency-light: it shells
out through an injected runner and fails softly when poppler commands are not
available. Core tools consume the protocol, so tests and embedders can provide
fake or custom implementations without installing PDF tooling.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

PDF_MAX_EXTRACT_SIZE_BYTES = 100 * 1024 * 1024
PDF_PAGE_RENDER_DPI = 100
PDFINFO_TIMEOUT_S = 10.0
PDFTOPPM_AVAILABILITY_TIMEOUT_S = 5.0
PDFTOPPM_EXTRACT_TIMEOUT_S = 120.0

PdfServiceErrorReason = Literal[
    "empty",
    "too_large",
    "unavailable",
    "password_protected",
    "corrupted",
    "timeout",
    "unknown",
]


class PdfServiceError(Exception):
    """Model-visible PDF service failure."""

    def __init__(self, reason: PdfServiceErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class PdfPageCountResult:
    """Best-effort PDF page-count result."""

    page_count: int | None
    error: PdfServiceError | None = None


@dataclass(frozen=True)
class PdfPageExtractionRequest:
    """Bounded request for PDF page-image extraction."""

    file_path: str
    first_page: int
    last_page: int
    max_pages: int
    max_input_bytes: int = PDF_MAX_EXTRACT_SIZE_BYTES

    def __post_init__(self) -> None:
        if self.first_page < 1:
            raise ValueError("first_page must be 1 or greater")
        if self.last_page < self.first_page:
            raise ValueError("last_page must be greater than or equal to first_page")
        if self.max_pages < 1:
            raise ValueError("max_pages must be 1 or greater")
        if self.page_count > self.max_pages:
            raise ValueError("page range exceeds max_pages")

    @property
    def page_count(self) -> int:
        return self.last_page - self.first_page + 1


@dataclass(frozen=True)
class PdfExtractedPage:
    """One extracted PDF page image."""

    file_path: str
    page_number: int
    media_type: str = "image/jpeg"
    size_bytes: int | None = None


@dataclass(frozen=True)
class PdfPageExtractionResult:
    """Successful PDF page extraction result."""

    file_path: str
    original_size: int
    output_dir: str
    pages: tuple[PdfExtractedPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PdfDocumentService(Protocol):
    """PDF metadata/extraction service consumed by tools."""

    def get_page_count(self, file_path: str) -> PdfPageCountResult:
        """Return best-effort page count, or None when unavailable."""
        ...

    def extract_pages(
        self,
        request: PdfPageExtractionRequest,
    ) -> PdfPageExtractionResult:
        """Extract pages as image files or raise `PdfServiceError`."""
        ...

    def cleanup_extraction(self, result: PdfPageExtractionResult) -> None:
        """Release temporary extraction files after callers read them."""
        ...


@dataclass(frozen=True)
class CommandResult:
    """Result from an argument-vector command invocation."""

    code: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Argument-vector command runner for optional PDF tooling."""

    def run(self, args: Sequence[str], *, timeout_s: float) -> CommandResult:
        """Run a command without shell interpolation."""
        ...


class SubprocessCommandRunner:
    """Default safe command runner: no shell and no project cwd dependence."""

    def run(self, args: Sequence[str], *, timeout_s: float) -> CommandResult:
        try:
            completed = subprocess.run(
                list(args),
                check=False,
                capture_output=True,
                cwd=os.path.abspath(os.sep),
                shell=False,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfServiceError(
                "timeout",
                f"PDF command timed out after {timeout_s:.0f}s: {args[0]}",
            ) from exc
        return CommandResult(
            code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


class CommandBackedPdfDocumentService:
    """Optional poppler-backed PDF service using `pdfinfo` and `pdftoppm`."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        output_root: str | None = None,
    ) -> None:
        self._command_runner = command_runner or SubprocessCommandRunner()
        self._output_root = output_root
        self._pdftoppm_available: bool | None = None

    def get_page_count(self, file_path: str) -> PdfPageCountResult:
        try:
            result = self._command_runner.run(
                ("pdfinfo", file_path),
                timeout_s=PDFINFO_TIMEOUT_S,
            )
        except PdfServiceError as exc:
            return PdfPageCountResult(page_count=None, error=exc)
        except OSError as exc:
            return PdfPageCountResult(
                page_count=None,
                error=PdfServiceError("unavailable", str(exc)),
            )
        if result.code != 0:
            return PdfPageCountResult(page_count=None)
        match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
        if match is None:
            return PdfPageCountResult(page_count=None)
        try:
            return PdfPageCountResult(page_count=int(match.group(1)))
        except ValueError:
            return PdfPageCountResult(page_count=None)

    def extract_pages(
        self,
        request: PdfPageExtractionRequest,
    ) -> PdfPageExtractionResult:
        try:
            stats = os.stat(request.file_path)
        except OSError as exc:
            raise PdfServiceError("unknown", str(exc)) from exc
        original_size = stats.st_size
        if original_size == 0:
            raise PdfServiceError("empty", f"PDF file is empty: {request.file_path}")
        if original_size > request.max_input_bytes:
            raise PdfServiceError(
                "too_large",
                "PDF file exceeds maximum allowed size for page extraction "
                f"({_format_file_size(request.max_input_bytes)}).",
            )
        if not self._is_pdftoppm_available():
            raise PdfServiceError(
                "unavailable",
                "pdftoppm is not installed. Install poppler-utils "
                "(e.g. `brew install poppler` or `apt-get install poppler-utils`) "
                "to enable PDF page rendering.",
            )

        output_dir: str | None = None
        try:
            output_dir = tempfile.mkdtemp(prefix="raygent-pdf-", dir=self._output_root)
            prefix = os.path.join(output_dir, "page")
            args = [
                "pdftoppm",
                "-jpeg",
                "-r",
                str(PDF_PAGE_RENDER_DPI),
                "-f",
                str(request.first_page),
                "-l",
                str(request.last_page),
                request.file_path,
                prefix,
            ]
            try:
                result = self._command_runner.run(
                    args,
                    timeout_s=PDFTOPPM_EXTRACT_TIMEOUT_S,
                )
            except OSError as exc:
                raise PdfServiceError("unavailable", str(exc)) from exc
            if result.code != 0:
                raise _pdf_error_from_stderr(result.stderr)
            page_files = sorted(
                filename
                for filename in os.listdir(output_dir)
                if filename.lower().endswith(".jpg")
            )
            if not page_files:
                raise PdfServiceError(
                    "corrupted",
                    "pdftoppm produced no output pages. The PDF may be invalid.",
                )
            pages = tuple(
                PdfExtractedPage(
                    file_path=os.path.join(output_dir, filename),
                    page_number=request.first_page + index,
                    size_bytes=os.stat(os.path.join(output_dir, filename)).st_size,
                )
                for index, filename in enumerate(page_files)
            )
            return PdfPageExtractionResult(
                file_path=request.file_path,
                original_size=original_size,
                output_dir=output_dir,
                pages=pages,
            )
        except PdfServiceError:
            if output_dir is not None:
                shutil.rmtree(output_dir, ignore_errors=True)
            raise
        except Exception as exc:
            if output_dir is not None:
                shutil.rmtree(output_dir, ignore_errors=True)
            raise PdfServiceError("unknown", str(exc)) from exc

    def cleanup_extraction(self, result: PdfPageExtractionResult) -> None:
        shutil.rmtree(result.output_dir, ignore_errors=True)

    def _is_pdftoppm_available(self) -> bool:
        if self._pdftoppm_available is not None:
            return self._pdftoppm_available
        try:
            result = self._command_runner.run(
                ("pdftoppm", "-v"),
                timeout_s=PDFTOPPM_AVAILABILITY_TIMEOUT_S,
            )
        except (OSError, PdfServiceError):
            self._pdftoppm_available = False
            return False
        self._pdftoppm_available = result.code == 0 or bool(result.stderr)
        return self._pdftoppm_available


def _pdf_error_from_stderr(stderr: str) -> PdfServiceError:
    if re.search(r"password", stderr, re.IGNORECASE):
        return PdfServiceError(
            "password_protected",
            "PDF is password-protected. Please provide an unprotected version.",
        )
    if re.search(r"damaged|corrupt|invalid", stderr, re.IGNORECASE):
        return PdfServiceError("corrupted", "PDF file is corrupted or invalid.")
    return PdfServiceError("unknown", f"pdftoppm failed: {stderr}")


def _format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"
