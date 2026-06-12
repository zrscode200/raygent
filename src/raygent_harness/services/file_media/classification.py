"""Bounded file/media classification primitives.

The classifier deliberately returns metadata only: path, extension, media type,
size, and the signal source. It never includes raw file bytes or text.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FileMediaKind = Literal[
    "text",
    "native_image",
    "unsupported_image",
    "pdf",
    "notebook",
    "binary",
    "directory",
    "special",
    "missing",
    "unknown",
]

FileMediaSource = Literal["extension", "magic", "stat", "fallback"]

FILE_MEDIA_SAMPLE_BYTES = 4096

NATIVE_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp"})
UNSUPPORTED_IMAGE_EXTENSIONS = frozenset({"bmp", "tif", "tiff", "ico", "svg"})
IMAGE_FILE_EXTENSIONS = NATIVE_IMAGE_EXTENSIONS | UNSUPPORTED_IMAGE_EXTENSIONS
PDF_FILE_EXTENSIONS = frozenset({"pdf"})
NOTEBOOK_FILE_EXTENSIONS = frozenset({"ipynb"})
BINARY_FILE_EXTENSIONS = frozenset(
    {
        "7z",
        "a",
        "bin",
        "class",
        "dmg",
        "dll",
        "dylib",
        "exe",
        "gz",
        "jar",
        "o",
        "parquet",
        "pyc",
        "so",
        "sqlite",
        "tar",
        "wasm",
        "xz",
        "zip",
    }
)

BLOCKED_DEVICE_PATHS = frozenset(
    {
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/full",
        "/dev/stdin",
        "/dev/tty",
        "/dev/console",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/fd/0",
        "/dev/fd/1",
        "/dev/fd/2",
    }
)


@dataclass(frozen=True)
class FileMediaClassification:
    """Metadata-only classification for a filesystem path."""

    path: str
    extension: str
    kind: FileMediaKind
    source: FileMediaSource
    media_type: str | None = None
    size_bytes: int | None = None
    exists: bool = True
    is_regular_file: bool = False
    sample_size_bytes: int = 0
    binary_reason: str | None = None


def classify_file_media(
    path: str,
    *,
    sample_bytes: int = FILE_MEDIA_SAMPLE_BYTES,
) -> FileMediaClassification:
    """Classify a path using stat, extension, and a bounded byte sample."""

    extension = extension_for_path(path)
    try:
        stats = os.stat(path)
    except FileNotFoundError:
        return _extension_classification(
            path,
            extension=extension,
            size_bytes=None,
            exists=False,
        )

    size_bytes = stats.st_size
    if stat.S_ISDIR(stats.st_mode):
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="directory",
            source="stat",
            size_bytes=size_bytes,
            is_regular_file=False,
        )
    if not stat.S_ISREG(stats.st_mode) or is_blocked_device_path(path):
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="special",
            source="stat",
            size_bytes=size_bytes,
            is_regular_file=False,
        )

    sample = _read_sample(path, max_bytes=sample_bytes)
    magic = classify_magic_bytes(
        sample,
        path=path,
        extension=extension,
        size_bytes=size_bytes,
    )
    if magic is not None:
        return magic

    return _extension_classification(
        path,
        extension=extension,
        size_bytes=size_bytes,
        exists=True,
        is_regular_file=True,
        sample_size_bytes=len(sample),
    )


def classify_file_extension(path: str) -> FileMediaClassification:
    """Classify by extension only, without statting or sampling the path."""

    extension = extension_for_path(path)
    classification = _extension_classification(
        path,
        extension=extension,
        size_bytes=None,
        exists=False,
    )
    if classification.kind == "missing":
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="unknown",
            source="fallback",
            exists=False,
        )
    return classification


def classify_magic_bytes(
    data: bytes,
    *,
    path: str = "",
    extension: str | None = None,
    size_bytes: int | None = None,
) -> FileMediaClassification | None:
    """Classify a bounded byte sample by magic bytes or binary heuristics."""

    ext = extension_for_path(path) if extension is None else extension
    media_type = detect_supported_image_media_type(data)
    if media_type is not None:
        return FileMediaClassification(
            path=path,
            extension=ext,
            kind="native_image",
            source="magic",
            media_type=media_type,
            size_bytes=size_bytes,
            is_regular_file=True,
            sample_size_bytes=len(data),
        )
    if data.startswith(b"BM"):
        return FileMediaClassification(
            path=path,
            extension=ext,
            kind="unsupported_image",
            source="magic",
            media_type="image/bmp",
            size_bytes=size_bytes,
            is_regular_file=True,
            sample_size_bytes=len(data),
        )
    if data.startswith(b"%PDF-"):
        return FileMediaClassification(
            path=path,
            extension=ext,
            kind="pdf",
            source="magic",
            media_type="application/pdf",
            size_bytes=size_bytes,
            is_regular_file=True,
            sample_size_bytes=len(data),
        )
    if _sample_looks_binary(data):
        return FileMediaClassification(
            path=path,
            extension=ext,
            kind="binary",
            source="magic",
            size_bytes=size_bytes,
            is_regular_file=True,
            sample_size_bytes=len(data),
            binary_reason="sample",
        )
    return None


def detect_supported_image_media_type(data: bytes) -> str | None:
    """Return the media type for natively supported image bytes."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def extension_for_path(path: str) -> str:
    """Return the lowercase extension without a leading dot."""

    return Path(path).suffix.lower().lstrip(".")


def is_blocked_device_path(path: str) -> bool:
    """Return true for device paths that can block or stream indefinitely."""

    if path in BLOCKED_DEVICE_PATHS:
        return True
    return path.startswith("/proc/") and (
        path.endswith("/fd/0") or path.endswith("/fd/1") or path.endswith("/fd/2")
    )


def _extension_classification(
    path: str,
    *,
    extension: str,
    size_bytes: int | None,
    exists: bool,
    is_regular_file: bool = False,
    sample_size_bytes: int = 0,
) -> FileMediaClassification:
    if extension in NATIVE_IMAGE_EXTENSIONS:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="native_image",
            source="extension",
            media_type=_image_media_type_for_extension(extension),
            size_bytes=size_bytes,
            exists=exists,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
        )
    if extension in UNSUPPORTED_IMAGE_EXTENSIONS:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="unsupported_image",
            source="extension",
            media_type=_unsupported_image_media_type_for_extension(extension),
            size_bytes=size_bytes,
            exists=exists,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
        )
    if extension in PDF_FILE_EXTENSIONS:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="pdf",
            source="extension",
            media_type="application/pdf",
            size_bytes=size_bytes,
            exists=exists,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
        )
    if extension in NOTEBOOK_FILE_EXTENSIONS:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="notebook",
            source="extension",
            media_type="application/x-ipynb+json",
            size_bytes=size_bytes,
            exists=exists,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
        )
    if extension in BINARY_FILE_EXTENSIONS:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="binary",
            source="extension",
            size_bytes=size_bytes,
            exists=exists,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
            binary_reason="extension",
        )
    if exists:
        return FileMediaClassification(
            path=path,
            extension=extension,
            kind="text",
            source="fallback",
            size_bytes=size_bytes,
            exists=True,
            is_regular_file=is_regular_file,
            sample_size_bytes=sample_size_bytes,
        )
    return FileMediaClassification(
        path=path,
        extension=extension,
        kind="missing",
        source="fallback",
        size_bytes=size_bytes,
        exists=False,
        is_regular_file=False,
        sample_size_bytes=sample_size_bytes,
    )


def _read_sample(path: str, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return b""
    with open(path, "rb") as handle:
        return handle.read(max_bytes)


def _sample_looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    non_printable = 0
    for byte in data:
        if byte < 9 or (byte > 13 and byte < 32):
            non_printable += 1
    return non_printable / len(data) > 0.3


def _image_media_type_for_extension(extension: str) -> str:
    if extension == "png":
        return "image/png"
    if extension in {"jpg", "jpeg"}:
        return "image/jpeg"
    if extension == "gif":
        return "image/gif"
    if extension == "webp":
        return "image/webp"
    return "application/octet-stream"


def _unsupported_image_media_type_for_extension(extension: str) -> str | None:
    if extension == "bmp":
        return "image/bmp"
    if extension in {"tif", "tiff"}:
        return "image/tiff"
    if extension == "ico":
        return "image/x-icon"
    if extension == "svg":
        return "image/svg+xml"
    return None
