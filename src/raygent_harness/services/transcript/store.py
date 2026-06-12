"""Transcript store protocol and JSONL implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from raygent_harness.services.transcript.models import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    StreamEventEntry,
    TombstoneEntry,
    TranscriptDecodeError,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    transcript_entry_from_json,
    transcript_entry_to_json,
)
from raygent_harness.services.transcript.paths import (
    resolve_transcript_base_dir,
    safe_transcript_component,
    transcript_path_for_scope,
)

_COMPACT_BOUNDARY_MARKERS = (
    b'"type": "compact_boundary"',
    b'"type":"compact_boundary"',
)


@dataclass(frozen=True)
class TranscriptReadStats:
    total_lines_scanned: int = 0
    decoded_entries: int = 0
    entries_retained: int = 0
    entries_skipped_before_active_compact_boundary: int = 0
    used_precompact_skip: bool = False
    used_full_read_fallback: bool = False


@dataclass(frozen=True)
class _BoundaryScanResult:
    total_lines_scanned: int = 0
    start_offset: int = 0
    start_line_number: int = 1
    skipped_non_empty_lines: int = 0
    found_boundary: bool = False


@dataclass(frozen=True)
class _DecodeResult:
    entries: tuple[TranscriptEntry, ...]
    warnings: tuple[str, ...]
    decoded_entries: int
    total_lines_scanned: int


@dataclass(frozen=True)
class TranscriptReadResult:
    entries: tuple[TranscriptEntry, ...]
    warnings: tuple[str, ...] = ()
    stats: TranscriptReadStats = field(default_factory=TranscriptReadStats)


class TranscriptStore(Protocol):
    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None: ...

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None: ...

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]: ...

    async def flush(self, scope: TranscriptScope | None = None) -> None: ...

    def path_for(self, scope: TranscriptScope) -> str | None: ...


class JsonlTranscriptStore:
    """Append-only JSONL transcript store.

    Writes are serialized per physical file. The implementation opens and
    closes the file for every append batch, so `flush()` is a compatibility
    barrier for QueryEngine integration rather than an active file-handle flush.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = resolve_transcript_base_dir(base_dir)
        self._locks: dict[Path, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
        await self.append_many(scope, [entry])

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        if len(entries) == 0:
            return
        path = self._path(scope)
        lock = await self._lock_for(path)
        lines = [transcript_entry_to_json(entry) for entry in entries]
        async with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line)
                    handle.write("\n")
                handle.flush()

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        result = await self._read_result(scope, skip_precompact=False)
        return list(result.entries)

    async def read_result(self, scope: TranscriptScope) -> TranscriptReadResult:
        return await self._read_result(scope, skip_precompact=True)

    async def _read_result(
        self,
        scope: TranscriptScope,
        *,
        skip_precompact: bool,
    ) -> TranscriptReadResult:
        path = self._path(scope)
        if not path.exists():
            return TranscriptReadResult(entries=())
        lock = await self._lock_for(path)
        async with lock:
            scan = (
                _scan_latest_compact_boundary(path, scope)
                if skip_precompact
                else _BoundaryScanResult()
            )
            decoded = _decode_entries_from_offset(
                path,
                start_offset=scan.start_offset,
                start_line_number=scan.start_line_number,
            )
            used_precompact_skip = scan.found_boundary and any(
                _is_message_for_scope(entry, scope) for entry in decoded.entries
            )
            if scan.found_boundary and not used_precompact_skip:
                decoded = _decode_entries_from_offset(
                    path,
                    start_offset=0,
                    start_line_number=1,
                )
            total_lines_scanned = (
                scan.total_lines_scanned
                if skip_precompact
                else decoded.total_lines_scanned
            )
        skipped_before_retained = (
            scan.skipped_non_empty_lines if used_precompact_skip else 0
        )
        used_full_read_fallback = not used_precompact_skip
        stats = TranscriptReadStats(
            total_lines_scanned=total_lines_scanned,
            decoded_entries=decoded.decoded_entries,
            entries_retained=len(decoded.entries),
            entries_skipped_before_active_compact_boundary=skipped_before_retained,
            used_precompact_skip=used_precompact_skip,
            used_full_read_fallback=used_full_read_fallback,
        )
        return TranscriptReadResult(
            entries=decoded.entries,
            warnings=decoded.warnings,
            stats=stats,
        )

    async def flush(self, scope: TranscriptScope | None = None) -> None:
        if scope is None:
            return
        # Writes close their file handles immediately. Resolve the path to catch
        # invalid sidechain scopes at the same barrier QueryEngine will call.
        self._path(scope)

    def path_for(self, scope: TranscriptScope) -> str:
        return str(self._path(scope))

    async def list_sidechain_agent_ids(self, session_id: str) -> tuple[str, ...]:
        """Return agent ids with sidechain files for a parent session.

        This is intentionally a concrete-store capability, not part of the
        base `TranscriptStore` protocol. Selected-agent retrieval works on any
        store; all-agent discovery needs filesystem/index support.
        """
        session_part = safe_transcript_component(session_id)
        subagents_dir = self.base_dir / session_part / "subagents"
        if not subagents_dir.exists():
            return ()

        agent_ids: list[str] = []
        seen: set[str] = set()
        for path in sorted(subagents_dir.glob("agent-*.jsonl")):
            if not path.is_file():
                continue
            agent_id = await self._agent_id_from_sidechain_file(path)
            if agent_id is None:
                agent_id = _agent_id_from_filename(path.name)
            if agent_id is None or agent_id in seen:
                continue
            seen.add(agent_id)
            agent_ids.append(agent_id)
        return tuple(agent_ids)

    def _path(self, scope: TranscriptScope) -> Path:
        return transcript_path_for_scope(self.base_dir, scope)

    async def _lock_for(self, path: Path) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(path)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[path] = lock
            return lock

    async def _agent_id_from_sidechain_file(self, path: Path) -> str | None:
        lock = await self._lock_for(path)
        async with lock:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped == "":
                            continue
                        try:
                            entry = transcript_entry_from_json(stripped)
                        except TranscriptDecodeError:
                            continue
                        agent_id = _entry_agent_id(entry)
                        if agent_id is not None:
                            return agent_id
            except OSError:
                return None
        return None


def _scan_latest_compact_boundary(
    path: Path,
    scope: TranscriptScope,
) -> _BoundaryScanResult:
    latest = _BoundaryScanResult()
    non_empty_lines_seen = 0
    line_number = 0
    with path.open("rb") as handle:
        while True:
            line_start_offset = handle.tell()
            raw_line = handle.readline()
            if raw_line == b"":
                break
            line_number += 1
            stripped = raw_line.strip()
            if stripped == b"":
                continue
            if _has_compact_boundary_marker(stripped) and _raw_boundary_matches_scope(
                stripped,
                scope,
            ):
                latest = _BoundaryScanResult(
                    total_lines_scanned=line_number,
                    start_offset=line_start_offset,
                    start_line_number=line_number,
                    skipped_non_empty_lines=non_empty_lines_seen,
                    found_boundary=True,
                )
            non_empty_lines_seen += 1
    if latest.found_boundary:
        return _BoundaryScanResult(
            total_lines_scanned=line_number,
            start_offset=latest.start_offset,
            start_line_number=latest.start_line_number,
            skipped_non_empty_lines=latest.skipped_non_empty_lines,
            found_boundary=True,
        )
    return _BoundaryScanResult(total_lines_scanned=line_number)


def _has_compact_boundary_marker(raw_line: bytes) -> bool:
    return any(marker in raw_line for marker in _COMPACT_BOUNDARY_MARKERS)


def _raw_boundary_matches_scope(raw_line: bytes, scope: TranscriptScope) -> bool:
    try:
        line = raw_line.decode("utf-8")
        entry = transcript_entry_from_json(line)
    except (TranscriptDecodeError, UnicodeDecodeError):
        return False
    return _is_compact_boundary_for_scope(entry, scope)


def _decode_entries_from_offset(
    path: Path,
    *,
    start_offset: int,
    start_line_number: int,
) -> _DecodeResult:
    entries: list[TranscriptEntry] = []
    warnings: list[str] = []
    decoded_entries = 0
    last_line_number = start_line_number - 1
    with path.open("rb") as handle:
        handle.seek(start_offset)
        for line_number, raw_line in enumerate(handle, start=start_line_number):
            last_line_number = line_number
            stripped = raw_line.strip()
            if stripped == b"":
                continue
            try:
                line = stripped.decode("utf-8")
            except UnicodeDecodeError as exc:
                warnings.append(
                    f"{path}:{line_number}: malformed UTF-8: {exc.reason}"
                )
                continue
            try:
                entries.append(transcript_entry_from_json(line))
            except TranscriptDecodeError as exc:
                warnings.append(f"{path}:{line_number}: {exc}")
                continue
            decoded_entries += 1
    return _DecodeResult(
        entries=tuple(entries),
        warnings=tuple(warnings),
        decoded_entries=decoded_entries,
        total_lines_scanned=last_line_number,
    )


def _entry_agent_id(entry: TranscriptEntry) -> str | None:
    if isinstance(
        entry,
        (
            TranscriptMessageEntry,
            CompactBoundaryEntry,
            ContentReplacementEntry,
            StreamEventEntry,
            TombstoneEntry,
        ),
    ):
        return entry.agent_id
    return None


def _is_compact_boundary_for_scope(
    entry: TranscriptEntry,
    scope: TranscriptScope,
) -> bool:
    return (
        isinstance(entry, CompactBoundaryEntry)
        and entry.session_id == scope.session_id
        and entry.agent_id == scope.agent_id
    )


def _is_message_for_scope(
    entry: TranscriptEntry,
    scope: TranscriptScope,
) -> bool:
    if not isinstance(entry, TranscriptMessageEntry):
        return False
    if entry.session_id != scope.session_id or entry.is_sidechain != scope.is_sidechain:
        return False
    if scope.agent_id is None:
        return entry.agent_id is None
    return entry.agent_id == scope.agent_id


def _agent_id_from_filename(name: str) -> str | None:
    prefix = "agent-"
    suffix = ".jsonl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    value = name[len(prefix) : -len(suffix)]
    return value or None


__all__ = [
    "JsonlTranscriptStore",
    "TranscriptReadResult",
    "TranscriptReadStats",
    "TranscriptStore",
]
