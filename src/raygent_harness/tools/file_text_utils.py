"""Shared text-file helpers for concrete file mutation tools."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from raygent_harness.core.file_state import FileState

FILE_UNEXPECTEDLY_MODIFIED_ERROR = (
    "File has been modified since read, either by the user or by a linter. "
    "Read it again before attempting to write it."
)

LineEnding = Literal["LF", "CRLF", "CR"]
FileEncoding = Literal["utf-8", "utf-16le", "utf-16be"]


@dataclass(frozen=True, slots=True)
class TextFileSnapshot:
    content: str
    normalized_content: str
    exists: bool
    mtime_ms: int | None
    line_ending: LineEnding
    encoding: FileEncoding


def mtime_ms(path: str | os.PathLike[str]) -> int:
    return int(os.stat(path).st_mtime_ns // 1_000_000)


def read_text_snapshot(path: str | os.PathLike[str]) -> TextFileSnapshot:
    """Read a text file and normalize content for stale/edit comparisons."""

    try:
        stats = os.stat(path)
    except FileNotFoundError:
        return TextFileSnapshot(
            content="",
            normalized_content="",
            exists=False,
            mtime_ms=None,
            line_ending="LF",
            encoding="utf-8",
        )

    if stat.S_ISDIR(stats.st_mode):
        raise IsADirectoryError(os.fspath(path))
    if not stat.S_ISREG(stats.st_mode):
        raise OSError(f"not a regular file: {os.fspath(path)}")

    data = Path(path).read_bytes()
    if data.startswith(b"\xff\xfe"):
        encoding: FileEncoding = "utf-16le"
    elif data.startswith(b"\xfe\xff"):
        encoding = "utf-16be"
    else:
        encoding = "utf-8"

    if encoding == "utf-8" and b"\x00" in data:
        raise UnicodeError("file appears to be binary")

    content = data.decode(encoding)

    return TextFileSnapshot(
        content=content,
        normalized_content=normalize_line_endings(content),
        exists=True,
        mtime_ms=int(stats.st_mtime_ns // 1_000_000),
        line_ending=detect_line_ending(content),
        encoding=encoding,
    )


def normalize_line_endings(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def detect_line_ending(content: str) -> LineEnding:
    crlf = content.count("\r\n")
    without_crlf = content.replace("\r\n", "")
    cr = without_crlf.count("\r")
    lf = without_crlf.count("\n")
    if crlf >= cr and crlf >= lf and crlf > 0:
        return "CRLF"
    if cr > lf and cr > 0:
        return "CR"
    return "LF"


def apply_line_ending(content: str, line_ending: LineEnding) -> str:
    normalized = normalize_line_endings(content)
    if line_ending == "CRLF":
        return normalized.replace("\n", "\r\n")
    if line_ending == "CR":
        return normalized.replace("\n", "\r")
    return normalized


def has_full_file_state(state: FileState | None) -> bool:
    if state is None or state.is_partial_view:
        return False
    if state.limit is not None:
        return False
    return state.offset is None or state.offset in {0, 1}


def content_matches_state(content: str, state: FileState) -> bool:
    return content == state.content or normalize_line_endings(content) == state.content


def assert_not_stale(
    *,
    path: str,
    snapshot: TextFileSnapshot,
    state: FileState | None,
) -> None:
    if not snapshot.exists:
        return
    if not has_full_file_state(state):
        raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)
    if (
        state is not None
        and snapshot.mtime_ms is not None
        and snapshot.mtime_ms > state.timestamp
        and not content_matches_state(snapshot.content, state)
    ):
        raise RuntimeError(FILE_UNEXPECTEDLY_MODIFIED_ERROR)


def write_text(
    path: str | os.PathLike[str],
    content: str,
    *,
    encoding: FileEncoding = "utf-8",
) -> None:
    Path(path).write_text(content, encoding=encoding, newline="")


__all__ = [
    "FILE_UNEXPECTEDLY_MODIFIED_ERROR",
    "FileEncoding",
    "LineEnding",
    "TextFileSnapshot",
    "apply_line_ending",
    "assert_not_stale",
    "content_matches_state",
    "detect_line_ending",
    "has_full_file_state",
    "mtime_ms",
    "normalize_line_endings",
    "read_text_snapshot",
    "write_text",
]
