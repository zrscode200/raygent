"""Session-scoped file read state.


The concrete Read/Write/Edit tools use this cache to detect unchanged reads and
stale writes. The cache is intentionally mutable and session-scoped, matching
the existing `ToolUseContext` lifecycle.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Self

READ_FILE_STATE_CACHE_SIZE = 100
DEFAULT_MAX_READ_FILE_STATE_CACHE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileState:
    content: str
    timestamp: int
    offset: int | None
    limit: int | None
    is_partial_view: bool = False


type FileStateCacheDump = tuple[tuple[str, FileState], ...]


def normalize_file_state_path(path: str | os.PathLike[str]) -> str:
    """Normalize cache keys to absolute filesystem paths."""

    return os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(path))))


class ReadFileStateCache:
    """Size-bounded LRU cache for file state.

    Python's stdlib has no size-aware LRU map, so this mirrors the reference
    behavior with an `OrderedDict`: reads refresh recency, writes evict from the
    least-recent end, and oversized entries are not retained.
    """

    def __init__(
        self,
        *,
        max_entries: int = READ_FILE_STATE_CACHE_SIZE,
        max_size_bytes: int = DEFAULT_MAX_READ_FILE_STATE_CACHE_BYTES,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._entries: OrderedDict[str, FileState] = OrderedDict()
        self._calculated_size_bytes = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_size_bytes(self) -> int:
        return self._max_size_bytes

    @property
    def calculated_size_bytes(self) -> int:
        return self._calculated_size_bytes

    def get(self, path: str | os.PathLike[str]) -> FileState | None:
        key = normalize_file_state_path(path)
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, path: str | os.PathLike[str], value: FileState) -> Self:
        key = normalize_file_state_path(path)
        if key in self._entries:
            self._calculated_size_bytes -= _file_state_size_bytes(self._entries[key])
            del self._entries[key]

        value_size = _file_state_size_bytes(value)
        if value_size > self._max_size_bytes:
            return self

        self._entries[key] = value
        self._calculated_size_bytes += value_size
        self._evict_to_limits()
        return self

    def has(self, path: str | os.PathLike[str]) -> bool:
        return normalize_file_state_path(path) in self._entries

    def delete(self, path: str | os.PathLike[str]) -> bool:
        key = normalize_file_state_path(path)
        value = self._entries.pop(key, None)
        if value is None:
            return False
        self._calculated_size_bytes -= _file_state_size_bytes(value)
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._calculated_size_bytes = 0

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries.keys())

    def entries(self) -> FileStateCacheDump:
        return tuple(self._entries.items())

    def dump(self) -> FileStateCacheDump:
        return self.entries()

    def load(self, entries: FileStateCacheDump) -> None:
        self.clear()
        for key, value in entries:
            self.set(key, value)

    def clone(self) -> ReadFileStateCache:
        cloned = ReadFileStateCache(
            max_entries=self._max_entries,
            max_size_bytes=self._max_size_bytes,
        )
        cloned.load(self.dump())
        return cloned

    def _evict_to_limits(self) -> None:
        while (
            len(self._entries) > self._max_entries
            or self._calculated_size_bytes > self._max_size_bytes
        ):
            _key, value = self._entries.popitem(last=False)
            self._calculated_size_bytes -= _file_state_size_bytes(value)


def create_read_file_state_cache(
    *,
    max_entries: int = READ_FILE_STATE_CACHE_SIZE,
    max_size_bytes: int = DEFAULT_MAX_READ_FILE_STATE_CACHE_BYTES,
) -> ReadFileStateCache:
    return ReadFileStateCache(
        max_entries=max_entries,
        max_size_bytes=max_size_bytes,
    )


def clone_read_file_state_cache(cache: ReadFileStateCache) -> ReadFileStateCache:
    return cache.clone()


def _file_state_size_bytes(value: FileState) -> int:
    return max(1, len(value.content.encode()))


__all__ = [
    "DEFAULT_MAX_READ_FILE_STATE_CACHE_BYTES",
    "READ_FILE_STATE_CACHE_SIZE",
    "FileState",
    "FileStateCacheDump",
    "ReadFileStateCache",
    "clone_read_file_state_cache",
    "create_read_file_state_cache",
    "normalize_file_state_path",
]
