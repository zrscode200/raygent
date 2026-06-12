"""File-backed task output storage.

Background task output stays outside task state and is read back through
bounded tail/range helpers. Raygent keeps that kernel property with a small
protocol-backed service instead of product UI surfaces.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

DEFAULT_TASK_OUTPUT_DIR = ".raygent/tasks"
DEFAULT_MAX_READ_BYTES = 8 * 1024 * 1024

_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_PROCESS_SESSION_ID = f"session-{secrets.token_hex(8)}"


@dataclass(frozen=True)
class TaskOutputReference:
    """Opaque-ish reference to a task output stream."""

    task_id: str
    path: str | None = None
    store_kind: str = "filesystem"


@dataclass(frozen=True)
class TaskOutputReadResult:
    """Bounded task-output read result."""

    task_id: str
    content: bytes
    start_offset: int
    bytes_read: int
    bytes_total: int
    next_offset: int
    truncated_before: bool = False
    truncated_after: bool = False


class TaskOutputStore(Protocol):
    """Protocol for append-only task output storage."""

    async def init_task_output(self, task_id: str) -> TaskOutputReference:
        """Create a new output stream for `task_id`.

        Must fail rather than overwrite unexpected pre-existing output.
        """
        ...

    async def append_task_output(self, task_id: str, chunk: bytes) -> None:
        """Append one output chunk."""
        ...

    async def flush_task_output(self, task_id: str) -> None:
        """Wait for pending writes for `task_id` to become readable."""
        ...

    async def evict_task_output(self, task_id: str) -> None:
        """Drop in-memory writer state while preserving readable output."""
        ...

    async def cleanup_task_output(self, task_id: str) -> None:
        """Remove output for `task_id`."""
        ...

    async def read_tail(
        self,
        task_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> TaskOutputReadResult:
        """Read at most `max_bytes` from the end of the output."""
        ...

    async def read_range(
        self,
        task_id: str,
        *,
        offset: int,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> TaskOutputReadResult:
        """Read at most `max_bytes` starting at `offset`."""
        ...

    async def size(self, task_id: str) -> int:
        """Return current output byte length, or zero when missing."""
        ...


class FileTaskOutputStore:
    """Safe filesystem-backed task output store.

    Paths are rooted under `<base>/<session_id>/tasks/<safe_task_id>.output`.
    Creation uses exclusive/no-follow flags where available so attacker-created
    symlinks or pre-existing files are not overwritten.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        base = resolve_task_output_base_dir(base_dir, project_root=project_root)
        session = safe_task_output_component(session_id or _PROCESS_SESSION_ID)
        self._task_dir = (base / session / "tasks").resolve()

    @property
    def task_dir(self) -> Path:
        """Concrete output directory for this store."""

        return self._task_dir

    def path_for_task(self, task_id: str) -> Path:
        """Return the concrete output path for `task_id`."""

        safe_task_id = safe_task_output_component(task_id)
        path = self._task_dir / f"{safe_task_id}.output"
        if path.parent != self._task_dir:
            raise ValueError(f"task output path escapes output root: {path}")
        return path

    async def init_task_output(self, task_id: str) -> TaskOutputReference:
        path = self.path_for_task(task_id)
        await asyncio.to_thread(self._init_sync, path)
        return TaskOutputReference(task_id=task_id, path=str(path), store_kind="filesystem")

    async def append_task_output(self, task_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        path = self.path_for_task(task_id)
        await asyncio.to_thread(self._append_sync, path, chunk)

    async def flush_task_output(self, task_id: str) -> None:
        _ = task_id
        # Appends are awaited synchronously, so there is no background queue to
        # drain. The method remains part of the protocol for compatibility
        # and future queued implementations.

    async def evict_task_output(self, task_id: str) -> None:
        _ = task_id
        # No writer cache in this implementation.

    async def cleanup_task_output(self, task_id: str) -> None:
        path = self.path_for_task(task_id)
        await asyncio.to_thread(self._cleanup_sync, path)

    async def read_tail(
        self,
        task_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> TaskOutputReadResult:
        path = self.path_for_task(task_id)
        return await read_task_output_file_tail(
            task_id,
            path,
            max_bytes=max_bytes,
        )

    async def read_range(
        self,
        task_id: str,
        *,
        offset: int,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> TaskOutputReadResult:
        path = self.path_for_task(task_id)
        return await read_task_output_file_range(
            task_id,
            path,
            offset,
            max_bytes=max_bytes,
        )

    async def size(self, task_id: str) -> int:
        path = self.path_for_task(task_id)
        return await asyncio.to_thread(self._size_sync, path)

    def _init_sync(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        os.close(fd)

    def _append_sync(self, path: Path, chunk: bytes) -> None:
        flags = os.O_WRONLY | os.O_APPEND | _O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)

    def _cleanup_sync(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _size_sync(self, path: Path) -> int:
        fd = _open_read_fd(path)
        if fd is None:
            return 0
        try:
            return os.fstat(fd).st_size
        finally:
            os.close(fd)


def resolve_task_output_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the root directory for task output files."""

    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / DEFAULT_TASK_OUTPUT_DIR).resolve()


def safe_task_output_component(value: str) -> str:
    """Make a task/session id safe for one filesystem component."""

    if value == "":
        raise ValueError("task output path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    if normalized == value:
        return normalized
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    return f"{normalized}-{digest}"


async def read_task_output_file_tail(
    task_id: str,
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> TaskOutputReadResult:
    """Read a bounded tail directly from a recorded task-output path."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    return await asyncio.to_thread(
        _read_tail_path_sync,
        task_id,
        Path(path).expanduser(),
        max_bytes,
    )


async def read_task_output_file_range(
    task_id: str,
    path: str | Path,
    offset: int,
    *,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> TaskOutputReadResult:
    """Read a bounded range directly from a recorded task-output path."""

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    return await asyncio.to_thread(
        _read_range_path_sync,
        task_id,
        Path(path).expanduser(),
        offset,
        max_bytes,
    )


def _read_tail_path_sync(
    task_id: str,
    path: Path,
    max_bytes: int,
) -> TaskOutputReadResult:
    fd = _open_read_fd(path)
    if fd is None:
        return _empty_read(task_id)
    try:
        bytes_total = os.fstat(fd).st_size
        start = max(0, bytes_total - max_bytes)
        os.lseek(fd, start, os.SEEK_SET)
        content = _read_fd(fd, min(max_bytes, bytes_total - start))
        bytes_read = len(content)
        return TaskOutputReadResult(
            task_id=task_id,
            content=content,
            start_offset=start,
            bytes_read=bytes_read,
            bytes_total=bytes_total,
            next_offset=start + bytes_read,
            truncated_before=start > 0,
            truncated_after=False,
        )
    finally:
        os.close(fd)


def _read_range_path_sync(
    task_id: str,
    path: Path,
    offset: int,
    max_bytes: int,
) -> TaskOutputReadResult:
    fd = _open_read_fd(path)
    if fd is None:
        return _empty_read(task_id)
    try:
        bytes_total = os.fstat(fd).st_size
        if offset >= bytes_total or max_bytes == 0:
            return TaskOutputReadResult(
                task_id=task_id,
                content=b"",
                start_offset=offset,
                bytes_read=0,
                bytes_total=bytes_total,
                next_offset=offset,
                truncated_before=offset > 0,
                truncated_after=offset < bytes_total,
            )
        os.lseek(fd, offset, os.SEEK_SET)
        content = _read_fd(fd, min(max_bytes, bytes_total - offset))
        bytes_read = len(content)
        next_offset = offset + bytes_read
        return TaskOutputReadResult(
            task_id=task_id,
            content=content,
            start_offset=offset,
            bytes_read=bytes_read,
            bytes_total=bytes_total,
            next_offset=next_offset,
            truncated_before=offset > 0,
            truncated_after=next_offset < bytes_total,
        )
    finally:
        os.close(fd)


def _open_read_fd(path: Path) -> int | None:
    try:
        return os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    except FileNotFoundError:
        return None


def _read_fd(fd: int, max_bytes: int) -> bytes:
    remaining = max_bytes
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _empty_read(task_id: str) -> TaskOutputReadResult:
    return TaskOutputReadResult(
        task_id=task_id,
        content=b"",
        start_offset=0,
        bytes_read=0,
        bytes_total=0,
        next_offset=0,
    )


__all__ = [
    "DEFAULT_MAX_READ_BYTES",
    "DEFAULT_TASK_OUTPUT_DIR",
    "FileTaskOutputStore",
    "TaskOutputReadResult",
    "TaskOutputReference",
    "TaskOutputStore",
    "read_task_output_file_range",
    "read_task_output_file_tail",
    "resolve_task_output_base_dir",
    "safe_task_output_component",
]
