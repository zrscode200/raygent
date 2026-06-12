"""Local file IO helpers for team-memory sync.

"""

from __future__ import annotations

import errno
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import (
    PathTraversalError,
    get_team_mem_path,
    validate_team_mem_key,
)
from raygent_harness.services.team_memory_sync.models import (
    MAX_CONFLICT_RETRIES,
    MAX_FILE_SIZE_BYTES,
    MAX_PUT_BODY_BYTES,
    MAX_RETRIES,
    LocalTeamMemoryReadResult,
    SkippedSecretFile,
    SyncState,
    TeamMemoryFetchResult,
    TeamMemoryHashesResult,
    TeamMemoryPullResult,
    TeamMemoryPushResult,
    TeamMemorySyncResult,
    TeamMemoryUploadResult,
    hash_content,
)
from raygent_harness.services.team_memory_sync.secret_scanner import scan_for_secrets

Sleeper = Callable[[float], Awaitable[None]]


class TeamMemoryTransport(Protocol):
    """Injected backend boundary for team-memory sync."""

    async def fetch_team_memory(self, etag: str | None) -> TeamMemoryFetchResult:
        """Fetch full team memory, using `etag` for conditional requests."""
        ...

    async def fetch_team_memory_hashes(self) -> TeamMemoryHashesResult:
        """Fetch metadata-only server entry checksums."""
        ...

    async def upload_team_memory(
        self,
        entries: dict[str, str],
        if_match_checksum: str | None,
    ) -> TeamMemoryUploadResult:
        """Upload a delta with optimistic locking."""
        ...


def _size_bytes(content: str) -> int:
    return len(content.encode("utf-8"))


def _should_ignore_walk_error(exc: OSError) -> bool:
    return exc.errno in {errno.ENOENT, errno.EACCES, errno.EPERM}


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            if _should_ignore_walk_error(exc):
                return
            raise

        for child in children:
            if child.is_symlink():
                continue
            try:
                if child.is_dir():
                    visit(child)
                elif child.is_file():
                    files.append(child)
            except OSError as exc:
                if not _should_ignore_walk_error(exc):
                    raise

    visit(root)
    return files


def read_local_team_memory(
    settings: MemorySettings,
    *,
    max_entries: int | None = None,
) -> LocalTeamMemoryReadResult:
    """Read local team-memory files into a flat upload map.

    Files containing secrets are skipped before they can enter the upload
    payload. Oversized and unreadable files are also skipped.
    """
    team_dir = get_team_mem_path(settings)
    entries: dict[str, str] = {}
    skipped_secrets: list[SkippedSecretFile] = []

    for file_path in _walk_files(team_dir):
        try:
            stats = file_path.stat()
        except OSError:
            continue
        if stats.st_size > MAX_FILE_SIZE_BYTES:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        rel_path = file_path.relative_to(team_dir).as_posix()
        secret_matches = scan_for_secrets(content)
        if len(secret_matches) > 0:
            first_match = secret_matches[0]
            skipped_secrets.append(
                SkippedSecretFile(
                    path=rel_path,
                    rule_id=first_match.rule_id,
                    label=first_match.label,
                )
            )
            continue

        entries[rel_path] = content

    keys = sorted(entries)
    if max_entries is not None and len(keys) > max_entries:
        entries = {key: entries[key] for key in keys[:max_entries]}

    return LocalTeamMemoryReadResult(
        entries=entries,
        skipped_secrets=tuple(skipped_secrets),
    )


def write_remote_entries_to_local(
    entries: dict[str, str],
    settings: MemorySettings,
) -> int:
    """Write validated remote entries into the local team-memory directory.

    Returns the number of files whose on-disk content changed.
    """
    written = 0

    for rel_path, content in entries.items():
        try:
            target = validate_team_mem_key(rel_path, settings)
        except PathTraversalError:
            continue

        if _size_bytes(content) > MAX_FILE_SIZE_BYTES:
            continue

        try:
            if target.read_text(encoding="utf-8", errors="replace") == content:
                continue
        except OSError:
            pass

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError:
            continue
        written += 1

    return written


def _json_size(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def batch_delta_by_bytes(delta: dict[str, str]) -> list[dict[str, str]]:
    """Split a delta into deterministic JSON-body-sized batches."""
    keys = sorted(delta)
    if len(keys) == 0:
        return []

    empty_body_bytes = len(b'{"entries":{}}')

    def entry_bytes(key: str, value: str) -> int:
        return _json_size(key) + _json_size(value) + 2

    batches: list[dict[str, str]] = []
    current: dict[str, str] = {}
    current_bytes = empty_body_bytes

    for key in keys:
        added = entry_bytes(key, delta[key])
        if current_bytes + added > MAX_PUT_BODY_BYTES and len(current) > 0:
            batches.append(current)
            current = {}
            current_bytes = empty_body_bytes
        current[key] = delta[key]
        current_bytes += added

    batches.append(current)
    return batches


def get_retry_delay(attempt: int) -> float:
    """Return deterministic reference-shaped exponential fetch retry delay."""
    return min(0.5 * (2 ** (attempt - 1)), 32.0)


async def _sleep_noop(_delay_s: float) -> None:
    return None


async def _fetch_team_memory_with_retries(
    transport: TeamMemoryTransport,
    etag: str | None,
    *,
    sleeper: Sleeper = _sleep_noop,
) -> TeamMemoryFetchResult:
    last_result: TeamMemoryFetchResult | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        last_result = await transport.fetch_team_memory(etag)
        if last_result.success or last_result.skip_retry:
            return last_result
        if attempt > MAX_RETRIES:
            return last_result
        await sleeper(get_retry_delay(attempt))

    assert last_result is not None
    return last_result


async def pull_team_memory(
    state: SyncState,
    settings: MemorySettings,
    transport: TeamMemoryTransport,
    *,
    skip_etag_cache: bool = False,
    sleeper: Sleeper = _sleep_noop,
) -> TeamMemoryPullResult:
    """Pull team memory through the injected transport and write local files."""
    etag = None if skip_etag_cache else state.last_known_checksum
    result = await _fetch_team_memory_with_retries(transport, etag, sleeper=sleeper)

    if not result.success:
        return TeamMemoryPullResult(
            success=False,
            files_written=0,
            entry_count=0,
            error=result.error,
        )

    if result.checksum is not None:
        state.last_known_checksum = result.checksum

    if result.not_modified:
        return TeamMemoryPullResult(
            success=True,
            files_written=0,
            entry_count=0,
            not_modified=True,
        )

    if result.is_empty or result.data is None:
        state.server_checksums.clear()
        if result.is_empty:
            state.last_known_checksum = None
        return TeamMemoryPullResult(success=True, files_written=0, entry_count=0)

    entries = result.data.content.entries
    response_checksums = result.data.content.entry_checksums
    state.server_checksums.clear()
    if response_checksums is not None:
        state.server_checksums.update(response_checksums)

    files_written = write_remote_entries_to_local(entries, settings)
    return TeamMemoryPullResult(
        success=True,
        files_written=files_written,
        entry_count=len(entries),
    )


async def push_team_memory(
    state: SyncState,
    settings: MemorySettings,
    transport: TeamMemoryTransport,
) -> TeamMemoryPushResult:
    """Push local team memory using delta upload and conflict hash probes."""
    local_read = read_local_team_memory(settings, max_entries=state.server_max_entries)
    entries = local_read.entries
    skipped_secrets = local_read.skipped_secrets
    local_hashes = {key: hash_content(content) for key, content in entries.items()}
    saw_conflict = False

    for conflict_attempt in range(MAX_CONFLICT_RETRIES + 1):
        delta = {
            key: entries[key]
            for key, local_hash in local_hashes.items()
            if state.server_checksums.get(key) != local_hash
        }

        if len(delta) == 0:
            return TeamMemoryPushResult(
                success=True,
                files_uploaded=0,
                skipped_secrets=skipped_secrets,
            )

        batches = batch_delta_by_bytes(delta)
        files_uploaded = 0
        result: TeamMemoryUploadResult | None = None

        for batch in batches:
            result = await transport.upload_team_memory(batch, state.last_known_checksum)
            if not result.success:
                break

            if result.checksum is not None:
                state.last_known_checksum = result.checksum
            for key in batch:
                state.server_checksums[key] = local_hashes[key]
            files_uploaded += len(batch)

        if result is None:
            return TeamMemoryPushResult(
                success=True,
                files_uploaded=0,
                skipped_secrets=skipped_secrets,
            )

        if result.success:
            return TeamMemoryPushResult(
                success=True,
                files_uploaded=files_uploaded,
                checksum=result.checksum,
                skipped_secrets=skipped_secrets,
            )

        if not result.conflict:
            if result.server_max_entries is not None:
                state.server_max_entries = result.server_max_entries
            return TeamMemoryPushResult(
                success=False,
                files_uploaded=files_uploaded,
                error=result.error,
                error_type=result.error_type,
                http_status=result.http_status,
            )

        saw_conflict = True
        if conflict_attempt >= MAX_CONFLICT_RETRIES:
            return TeamMemoryPushResult(
                success=False,
                files_uploaded=0,
                conflict=True,
                error="Conflict resolution failed after retries",
            )

        probe = await transport.fetch_team_memory_hashes()
        if not probe.success or probe.entry_checksums is None:
            return TeamMemoryPushResult(
                success=False,
                files_uploaded=0,
                conflict=True,
                error=f"Conflict resolution hashes probe failed: {probe.error}",
            )

        if probe.checksum is not None:
            state.last_known_checksum = probe.checksum
        state.server_checksums.clear()
        state.server_checksums.update(probe.entry_checksums)

    return TeamMemoryPushResult(
        success=False,
        files_uploaded=0,
        conflict=saw_conflict,
        error="Unexpected end of conflict resolution loop",
    )


async def sync_team_memory(
    state: SyncState,
    settings: MemorySettings,
    transport: TeamMemoryTransport,
) -> TeamMemorySyncResult:
    """Run reference-shaped full sync: pull first, then push."""
    pull_result = await pull_team_memory(
        state,
        settings,
        transport,
        skip_etag_cache=True,
    )
    if not pull_result.success:
        return TeamMemorySyncResult(
            success=False,
            files_pulled=0,
            files_pushed=0,
            error=pull_result.error,
        )

    push_result = await push_team_memory(state, settings, transport)
    if not push_result.success:
        return TeamMemorySyncResult(
            success=False,
            files_pulled=pull_result.files_written,
            files_pushed=0,
            error=push_result.error,
        )

    return TeamMemorySyncResult(
        success=True,
        files_pulled=pull_result.files_written,
        files_pushed=push_result.files_uploaded,
    )


__all__ = [
    "MAX_CONFLICT_RETRIES",
    "MAX_FILE_SIZE_BYTES",
    "MAX_PUT_BODY_BYTES",
    "MAX_RETRIES",
    "TeamMemoryTransport",
    "batch_delta_by_bytes",
    "get_retry_delay",
    "pull_team_memory",
    "push_team_memory",
    "read_local_team_memory",
    "sync_team_memory",
    "write_remote_entries_to_local",
]
