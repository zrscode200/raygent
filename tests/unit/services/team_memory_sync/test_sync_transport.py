from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.team_memory_sync import (
    MAX_PUT_BODY_BYTES,
    SyncState,
    TeamMemoryContent,
    TeamMemoryData,
    TeamMemoryFetchResult,
    TeamMemoryHashesResult,
    TeamMemoryUploadResult,
    batch_delta_by_bytes,
    create_sync_state,
    hash_content,
    pull_team_memory,
    push_team_memory,
    sync_team_memory,
)


def _empty_fetch_results() -> list[TeamMemoryFetchResult]:
    return []


def _empty_hash_results() -> list[TeamMemoryHashesResult]:
    return []


def _empty_upload_results() -> list[TeamMemoryUploadResult]:
    return []


def _empty_fetch_etags() -> list[str | None]:
    return []


def _empty_uploads() -> list[tuple[dict[str, str], str | None]]:
    return []


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        team_memory_enabled=True,
        **kwargs,
    )


def data(
    entries: dict[str, str],
    *,
    checksum: str = "remote-1",
    entry_checksums: dict[str, str] | None = None,
) -> TeamMemoryData:
    return TeamMemoryData(
        organization_id="org",
        repo="owner/repo",
        version=1,
        last_modified="2026-05-14T00:00:00Z",
        checksum=checksum,
        content=TeamMemoryContent(
            entries=entries,
            entry_checksums=entry_checksums,
        ),
    )


@dataclass
class FakeTransport:
    fetch_results: list[TeamMemoryFetchResult] = field(default_factory=_empty_fetch_results)
    hash_results: list[TeamMemoryHashesResult] = field(default_factory=_empty_hash_results)
    upload_results: list[TeamMemoryUploadResult] = field(default_factory=_empty_upload_results)
    fetch_etags: list[str | None] = field(default_factory=_empty_fetch_etags)
    uploads: list[tuple[dict[str, str], str | None]] = field(default_factory=_empty_uploads)
    hash_probe_count: int = 0

    async def fetch_team_memory(self, etag: str | None) -> TeamMemoryFetchResult:
        self.fetch_etags.append(etag)
        if len(self.fetch_results) == 0:
            return TeamMemoryFetchResult(success=True, is_empty=True)
        return self.fetch_results.pop(0)

    async def fetch_team_memory_hashes(self) -> TeamMemoryHashesResult:
        self.hash_probe_count += 1
        if len(self.hash_results) == 0:
            return TeamMemoryHashesResult(success=True, entry_checksums={})
        return self.hash_results.pop(0)

    async def upload_team_memory(
        self,
        entries: dict[str, str],
        if_match_checksum: str | None,
    ) -> TeamMemoryUploadResult:
        self.uploads.append((dict(entries), if_match_checksum))
        if len(self.upload_results) == 0:
            return TeamMemoryUploadResult(success=True, checksum=f"upload-{len(self.uploads)}")
        return self.upload_results.pop(0)


def test_hash_content_matches_reference_sha256_format() -> None:
    expected = hashlib.sha256(b"hello").hexdigest()

    assert hash_content("hello") == f"sha256:{expected}"


def test_batch_delta_by_bytes_is_sorted_and_byte_bounded() -> None:
    delta = {
        "c.md": "c",
        "a.md": "x" * (MAX_PUT_BODY_BYTES - 20),
        "b.md": "b",
    }

    batches = batch_delta_by_bytes(delta)

    assert batches == [
        {"a.md": delta["a.md"]},
        {"b.md": "b", "c.md": "c"},
    ]


def test_batch_delta_by_bytes_keeps_single_oversized_entry_in_solo_batch() -> None:
    delta = {"huge.md": "x" * (MAX_PUT_BODY_BYTES + 1)}

    assert batch_delta_by_bytes(delta) == [delta]


async def test_pull_uses_etag_cache_and_writes_remote_entries(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    state = create_sync_state()
    state.last_known_checksum = "cached"
    remote_entries = {"MEMORY.md": "remote", "nested/topic.md": "topic"}
    transport = FakeTransport(
        fetch_results=[
            TeamMemoryFetchResult(
                success=True,
                data=data(
                    remote_entries,
                    checksum="remote-2",
                    entry_checksums={
                        "MEMORY.md": hash_content("remote"),
                        "nested/topic.md": hash_content("topic"),
                    },
                ),
                checksum="remote-2",
            )
        ]
    )

    result = await pull_team_memory(state, cfg, transport)

    assert result.success is True
    assert result.files_written == 2
    assert result.entry_count == 2
    assert transport.fetch_etags == ["cached"]
    assert state.last_known_checksum == "remote-2"
    assert state.server_checksums == {
        "MEMORY.md": hash_content("remote"),
        "nested/topic.md": hash_content("topic"),
    }
    assert (get_team_mem_path(cfg) / "nested" / "topic.md").read_text() == "topic"


async def test_pull_not_modified_and_empty_paths(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    state = SyncState(last_known_checksum="cached", server_checksums={"old.md": "hash"})
    not_modified_transport = FakeTransport(
        fetch_results=[
            TeamMemoryFetchResult(success=True, not_modified=True, checksum="cached")
        ]
    )

    not_modified = await pull_team_memory(state, cfg, not_modified_transport)

    assert not_modified.not_modified is True
    assert state.server_checksums == {"old.md": "hash"}

    empty = await pull_team_memory(
        state,
        cfg,
        FakeTransport(fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)]),
    )

    assert empty.success is True
    assert state.server_checksums == {}
    assert state.last_known_checksum is None


async def test_pull_retries_non_terminal_fetch_failures(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    state = create_sync_state()
    sleep_delays: list[float] = []

    async def sleeper(delay_s: float) -> None:
        sleep_delays.append(delay_s)

    result = await pull_team_memory(
        state,
        cfg,
        FakeTransport(
            fetch_results=[
                TeamMemoryFetchResult(success=False, error="network", error_type="network"),
                TeamMemoryFetchResult(success=True, is_empty=True),
            ]
        ),
        sleeper=sleeper,
    )

    assert result.success is True
    assert sleep_delays == [0.5]


async def test_push_skips_unchanged_hashes_and_secret_files(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "safe.md").write_text("same", encoding="utf-8")
    (team_dir / "secret.md").write_text("ghp_" + "a" * 36, encoding="utf-8")
    state = SyncState(server_checksums={"safe.md": hash_content("same")})
    transport = FakeTransport()

    result = await push_team_memory(state, cfg, transport)

    assert result.success is True
    assert result.files_uploaded == 0
    assert [(item.path, item.rule_id) for item in result.skipped_secrets] == [
        ("secret.md", "github-pat")
    ]
    assert transport.uploads == []


async def test_push_uploads_delta_and_updates_state(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "same.md").write_text("same", encoding="utf-8")
    (team_dir / "changed.md").write_text("new", encoding="utf-8")
    state = SyncState(
        last_known_checksum="etag-1",
        server_checksums={
            "same.md": hash_content("same"),
            "changed.md": hash_content("old"),
        },
    )
    transport = FakeTransport(
        upload_results=[TeamMemoryUploadResult(success=True, checksum="etag-2")]
    )

    result = await push_team_memory(state, cfg, transport)

    assert result.success is True
    assert result.files_uploaded == 1
    assert transport.uploads == [({"changed.md": "new"}, "etag-1")]
    assert state.last_known_checksum == "etag-2"
    assert state.server_checksums["changed.md"] == hash_content("new")


async def test_push_conflict_probe_recomputes_delta_without_rereading_disk(
    tmp_path: Path,
) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "a.md").write_text("ours", encoding="utf-8")
    (team_dir / "b.md").write_text("ours-too", encoding="utf-8")
    state = SyncState(last_known_checksum="etag-1")
    transport = FakeTransport(
        upload_results=[
            TeamMemoryUploadResult(success=False, conflict=True, error="ETag mismatch"),
            TeamMemoryUploadResult(success=True, checksum="etag-3"),
        ],
        hash_results=[
            TeamMemoryHashesResult(
                success=True,
                checksum="etag-2",
                entry_checksums={"a.md": hash_content("ours")},
            )
        ],
    )

    result = await push_team_memory(state, cfg, transport)

    assert result.success is True
    assert result.files_uploaded == 1
    assert transport.hash_probe_count == 1
    assert transport.uploads == [
        ({"a.md": "ours", "b.md": "ours-too"}, "etag-1"),
        ({"b.md": "ours-too"}, "etag-2"),
    ]


async def test_push_conflict_probe_failure_surfaces_conflict(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "a.md").write_text("ours", encoding="utf-8")
    state = SyncState(last_known_checksum="etag-1")
    transport = FakeTransport(
        upload_results=[TeamMemoryUploadResult(success=False, conflict=True)],
        hash_results=[TeamMemoryHashesResult(success=False, error="probe down")],
    )

    result = await push_team_memory(state, cfg, transport)

    assert result.success is False
    assert result.conflict is True
    assert result.error == "Conflict resolution hashes probe failed: probe down"


async def test_push_conflict_retry_exhaustion(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "a.md").write_text("ours", encoding="utf-8")
    state = SyncState(last_known_checksum="etag-1")
    transport = FakeTransport(
        upload_results=[
            TeamMemoryUploadResult(success=False, conflict=True),
            TeamMemoryUploadResult(success=False, conflict=True),
            TeamMemoryUploadResult(success=False, conflict=True),
        ],
        hash_results=[
            TeamMemoryHashesResult(success=True, checksum="etag-2", entry_checksums={}),
            TeamMemoryHashesResult(success=True, checksum="etag-3", entry_checksums={}),
        ],
    )

    result = await push_team_memory(state, cfg, transport)

    assert result.success is False
    assert result.conflict is True
    assert result.error == "Conflict resolution failed after retries"
    assert transport.hash_probe_count == 2
    assert len(transport.uploads) == 3


async def test_push_learns_structured_413_max_entries(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "a.md").write_text("a", encoding="utf-8")
    state = create_sync_state()
    transport = FakeTransport(
        upload_results=[
            TeamMemoryUploadResult(
                success=False,
                error="too many entries",
                error_type="unknown",
                http_status=413,
                server_error_code="team_memory_too_many_entries",
                server_max_entries=10,
                server_received_entries=11,
            )
        ]
    )

    result = await push_team_memory(state, cfg, transport)

    assert result.success is False
    assert result.http_status == 413
    assert state.server_max_entries == 10


async def test_sync_team_memory_pulls_first_then_pushes(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    transport = FakeTransport(
        fetch_results=[
            TeamMemoryFetchResult(
                success=True,
                data=data(
                    {"remote.md": "server"},
                    entry_checksums={"remote.md": hash_content("server")},
                ),
                checksum="etag-1",
            )
        ],
        upload_results=[TeamMemoryUploadResult(success=True, checksum="etag-2")],
    )
    state = create_sync_state()

    result = await sync_team_memory(state, cfg, transport)

    assert result.success is True
    assert result.files_pulled == 1
    assert result.files_pushed == 0
    assert transport.fetch_etags == [None]
    assert transport.uploads == []


async def test_sync_team_memory_propagates_push_failure_after_pull(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "local.md").write_text("local", encoding="utf-8")
    transport = FakeTransport(
        fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)],
        upload_results=[
            TeamMemoryUploadResult(success=False, error="push failed", error_type="network")
        ],
    )

    result = await sync_team_memory(create_sync_state(), cfg, transport)

    assert result.success is False
    assert result.files_pulled == 0
    assert result.files_pushed == 0
    assert result.error == "push failed"
