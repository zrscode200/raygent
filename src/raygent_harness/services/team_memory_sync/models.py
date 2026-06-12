"""Shared data shapes for team-memory sync.

"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

MAX_FILE_SIZE_BYTES = 250_000
MAX_PUT_BODY_BYTES = 200_000
MAX_RETRIES = 3
MAX_CONFLICT_RETRIES = 2

TeamMemoryErrorType = Literal[
    "auth",
    "timeout",
    "network",
    "parse",
    "conflict",
    "unknown",
    "no_oauth",
    "no_repo",
]
TeamMemoryServerErrorCode = Literal["team_memory_too_many_entries"]


def _empty_entries() -> dict[str, str]:
    return {}


def _empty_checksums() -> dict[str, str]:
    return {}


def hash_content(content: str) -> str:
    """Return the reference `sha256:<hex>` hash over UTF-8 content bytes."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SkippedSecretFile:
    """A local team-memory file skipped because it contains a detected secret."""

    path: str
    rule_id: str
    label: str


@dataclass(frozen=True)
class LocalTeamMemoryReadResult:
    """Result of reading local team-memory files for upload."""

    entries: dict[str, str] = field(default_factory=_empty_entries)
    skipped_secrets: tuple[SkippedSecretFile, ...] = ()


@dataclass(frozen=True)
class TeamMemoryContent:
    """Flat key/content team-memory payload plus optional per-entry checksums."""

    entries: dict[str, str] = field(default_factory=_empty_entries)
    entry_checksums: dict[str, str] | None = None


@dataclass(frozen=True)
class TeamMemoryData:
    """Full fetched team-memory payload."""

    organization_id: str
    repo: str
    version: int
    last_modified: str
    checksum: str
    content: TeamMemoryContent


@dataclass(frozen=True)
class TeamMemoryFetchResult:
    """Result returned by a transport fetch."""

    success: bool
    data: TeamMemoryData | None = None
    is_empty: bool = False
    not_modified: bool = False
    checksum: str | None = None
    error: str | None = None
    skip_retry: bool = False
    error_type: TeamMemoryErrorType | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class TeamMemoryHashesResult:
    """Metadata-only server checksum probe result."""

    success: bool
    version: int | None = None
    checksum: str | None = None
    entry_checksums: dict[str, str] | None = None
    error: str | None = None
    error_type: TeamMemoryErrorType | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class TeamMemoryUploadResult:
    """Result returned by a transport upload."""

    success: bool
    checksum: str | None = None
    last_modified: str | None = None
    conflict: bool = False
    error: str | None = None
    error_type: TeamMemoryErrorType | None = None
    http_status: int | None = None
    server_error_code: TeamMemoryServerErrorCode | None = None
    server_max_entries: int | None = None
    server_received_entries: int | None = None


@dataclass(frozen=True)
class TeamMemoryPullResult:
    """Public result from pulling remote team memory into local files."""

    success: bool
    files_written: int
    entry_count: int
    not_modified: bool = False
    error: str | None = None


@dataclass(frozen=True)
class TeamMemoryPushResult:
    """Public result from pushing local team memory to the transport."""

    success: bool
    files_uploaded: int
    checksum: str | None = None
    conflict: bool = False
    error: str | None = None
    error_type: TeamMemoryErrorType | None = None
    http_status: int | None = None
    skipped_secrets: tuple[SkippedSecretFile, ...] = ()


@dataclass(frozen=True)
class TeamMemorySyncResult:
    """Public result from full pull-then-push sync."""

    success: bool
    files_pulled: int
    files_pushed: int
    error: str | None = None


@dataclass
class SyncState:
    """Mutable sync state carried across team-memory sync operations."""

    last_known_checksum: str | None = None
    server_checksums: dict[str, str] = field(default_factory=_empty_checksums)
    server_max_entries: int | None = None


def create_sync_state() -> SyncState:
    """Return a fresh reference-shaped sync state."""
    return SyncState()


__all__ = [
    "MAX_CONFLICT_RETRIES",
    "MAX_FILE_SIZE_BYTES",
    "MAX_PUT_BODY_BYTES",
    "MAX_RETRIES",
    "LocalTeamMemoryReadResult",
    "SkippedSecretFile",
    "SyncState",
    "TeamMemoryContent",
    "TeamMemoryData",
    "TeamMemoryErrorType",
    "TeamMemoryFetchResult",
    "TeamMemoryHashesResult",
    "TeamMemoryPullResult",
    "TeamMemoryPushResult",
    "TeamMemoryServerErrorCode",
    "TeamMemorySyncResult",
    "TeamMemoryUploadResult",
    "create_sync_state",
    "hash_content",
]
