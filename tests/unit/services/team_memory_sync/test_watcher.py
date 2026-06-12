from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path
from raygent_harness.services.team_memory_sync import (
    TeamMemoryContent,
    TeamMemoryData,
    TeamMemoryFetchResult,
    TeamMemoryHashesResult,
    TeamMemoryPushResult,
    TeamMemorySyncService,
    TeamMemoryUploadResult,
    is_permanent_failure,
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


def _empty_sleep_futures() -> list[asyncio.Future[None]]:
    return []


def settings(tmp_path: Path, **kwargs: Any) -> MemorySettings:
    team_memory_enabled = kwargs.pop("team_memory_enabled", True)
    return MemorySettings(
        project_root=tmp_path / "workspace" / "repo",
        home_dir=tmp_path / "home",
        memory_base_dir=tmp_path / "base",
        team_memory_enabled=team_memory_enabled,
        **kwargs,
    )


def data(entries: dict[str, str], *, checksum: str = "remote-1") -> TeamMemoryData:
    return TeamMemoryData(
        organization_id="org",
        repo="owner/repo",
        version=1,
        last_modified="2026-05-14T00:00:00Z",
        checksum=checksum,
        content=TeamMemoryContent(entries=entries),
    )


@dataclass
class FakeTransport:
    fetch_results: list[TeamMemoryFetchResult] = field(default_factory=_empty_fetch_results)
    hash_results: list[TeamMemoryHashesResult] = field(default_factory=_empty_hash_results)
    upload_results: list[TeamMemoryUploadResult] = field(default_factory=_empty_upload_results)
    fetch_etags: list[str | None] = field(default_factory=_empty_fetch_etags)
    uploads: list[tuple[dict[str, str], str | None]] = field(default_factory=_empty_uploads)

    async def fetch_team_memory(self, etag: str | None) -> TeamMemoryFetchResult:
        self.fetch_etags.append(etag)
        if len(self.fetch_results) == 0:
            return TeamMemoryFetchResult(success=True, is_empty=True)
        return self.fetch_results.pop(0)

    async def fetch_team_memory_hashes(self) -> TeamMemoryHashesResult:
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


class BlockingFirstUploadTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__(fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)])
        self.first_upload_started = asyncio.Event()
        self.release_first_upload = asyncio.Event()

    async def upload_team_memory(
        self,
        entries: dict[str, str],
        if_match_checksum: str | None,
    ) -> TeamMemoryUploadResult:
        self.uploads.append((dict(entries), if_match_checksum))
        if len(self.uploads) == 1:
            self.first_upload_started.set()
            await self.release_first_upload.wait()
            return TeamMemoryUploadResult(success=True, checksum="upload-1")
        return TeamMemoryUploadResult(success=True, checksum=f"upload-{len(self.uploads)}")


@dataclass
class ManualSleeper:
    futures: list[asyncio.Future[None]] = field(default_factory=_empty_sleep_futures)

    async def __call__(self, _delay_s: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.futures.append(future)
        await future

    def release_latest(self) -> None:
        for future in reversed(self.futures):
            if not future.done():
                future.set_result(None)
                return
        raise AssertionError("no pending sleep future")


async def spin_until(predicate: Any, *, attempts: int = 20) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not met")


def test_is_permanent_failure_matches_reference_rules() -> None:
    assert is_permanent_failure(TeamMemoryPushResult(False, 0, error_type="no_oauth"))
    assert is_permanent_failure(TeamMemoryPushResult(False, 0, error_type="no_repo"))
    assert is_permanent_failure(TeamMemoryPushResult(False, 0, http_status=413))
    assert not is_permanent_failure(TeamMemoryPushResult(False, 0, http_status=409))
    assert not is_permanent_failure(TeamMemoryPushResult(False, 0, http_status=429))
    assert not is_permanent_failure(TeamMemoryPushResult(False, 0, error_type="network"))


@pytest.mark.parametrize(
    ("overrides", "service_kwargs", "reason"),
    [
        ({}, {"feature_enabled": False}, "feature_disabled"),
        ({"team_memory_enabled": False}, {}, "team_memory_disabled"),
        ({}, {"sync_available": False}, "sync_unavailable"),
        ({}, {"repo_available": False}, "no_repo"),
    ],
)
async def test_start_gates_before_creating_state_or_fetching(
    tmp_path: Path,
    overrides: dict[str, Any],
    service_kwargs: dict[str, bool],
    reason: str,
) -> None:
    transport = FakeTransport()
    service = TeamMemorySyncService(
        settings=settings(tmp_path, **overrides),
        transport=transport,
        **cast(Any, service_kwargs),
    )

    result = await service.start()

    assert result.started is False
    assert result.skip_reason == reason
    assert service.state is None
    assert transport.fetch_etags == []


async def test_start_performs_initial_pull_and_creates_team_directory(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    transport = FakeTransport(
        fetch_results=[
            TeamMemoryFetchResult(
                success=True,
                data=data({"MEMORY.md": "remote"}),
                checksum="remote-1",
            )
        ]
    )
    service = TeamMemorySyncService(settings=cfg, transport=transport)

    result = await service.start()

    assert result.started is True
    assert result.initial_pull_success is True
    assert result.initial_files_pulled == 1
    assert result.server_has_content is True
    assert transport.fetch_etags == [None]
    assert get_team_mem_path(cfg).is_dir()
    assert (get_team_mem_path(cfg) / "MEMORY.md").read_text(encoding="utf-8") == "remote"


async def test_notify_before_start_is_noop(tmp_path: Path) -> None:
    service = TeamMemorySyncService(settings=settings(tmp_path), transport=FakeTransport())

    assert await service.notify_team_memory_write() is False


async def test_notify_debounces_multiple_writes_into_one_push(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "MEMORY.md").write_text("local", encoding="utf-8")
    sleeper = ManualSleeper()
    transport = FakeTransport(fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)])
    service = TeamMemorySyncService(settings=cfg, transport=transport, sleeper=sleeper)
    await service.start()

    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 1)
    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 2)

    sleeper.release_latest()
    await spin_until(lambda: len(transport.uploads) == 1)

    assert transport.uploads == [({"MEMORY.md": "local"}, None)]
    assert service.has_pending_changes is False


async def test_stop_flushes_pending_debounced_changes(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "MEMORY.md").write_text("local", encoding="utf-8")
    service = TeamMemorySyncService(
        settings=cfg,
        transport=FakeTransport(fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)]),
        sleeper=ManualSleeper(),
    )
    await service.start()
    assert await service.notify_team_memory_write() is True

    result = await service.stop()

    assert result.flushed_pending is True
    assert result.flush_success is True
    assert service.started is False


async def test_notify_during_in_flight_push_rearms_next_push(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    target = team_dir / "MEMORY.md"
    target.write_text("first", encoding="utf-8")
    sleeper = ManualSleeper()
    transport = BlockingFirstUploadTransport()
    service = TeamMemorySyncService(settings=cfg, transport=transport, sleeper=sleeper)
    await service.start()

    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 1)
    sleeper.release_latest()
    await transport.first_upload_started.wait()
    assert service.push_in_progress is True

    target.write_text("second", encoding="utf-8")
    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 2)
    sleeper.release_latest()
    await spin_until(lambda: len(sleeper.futures) == 3)

    transport.release_first_upload.set()
    await spin_until(lambda: service.push_in_progress is False)
    sleeper.release_latest()
    await spin_until(lambda: len(transport.uploads) == 2)

    assert transport.uploads == [
        ({"MEMORY.md": "first"}, None),
        ({"MEMORY.md": "second"}, "upload-1"),
    ]
    assert service.has_pending_changes is False


async def test_permanent_failure_suppresses_until_unlink_recovery(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "MEMORY.md").write_text("local", encoding="utf-8")
    sleeper = ManualSleeper()
    transport = FakeTransport(
        fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)],
        upload_results=[
            TeamMemoryUploadResult(success=False, http_status=413, error="too many"),
            TeamMemoryUploadResult(success=True, checksum="recovered"),
        ],
    )
    service = TeamMemorySyncService(settings=cfg, transport=transport, sleeper=sleeper)
    await service.start()

    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 1)
    sleeper.release_latest()
    await spin_until(lambda: service.push_suppressed_reason == "http_413")

    assert await service.notify_team_memory_write() is False
    assert len(transport.uploads) == 1

    assert await service.notify_team_memory_unlink() is True
    await spin_until(lambda: len(sleeper.futures) == 2)
    sleeper.release_latest()
    await spin_until(lambda: len(transport.uploads) == 2)

    assert service.push_suppressed_reason is None
    assert service.has_pending_changes is False


async def test_stop_awaits_in_flight_push(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    team_dir = get_team_mem_path(cfg)
    team_dir.mkdir(parents=True)
    (team_dir / "MEMORY.md").write_text("local", encoding="utf-8")
    sleeper = ManualSleeper()
    transport = FakeTransport(fetch_results=[TeamMemoryFetchResult(success=True, is_empty=True)])
    service = TeamMemorySyncService(settings=cfg, transport=transport, sleeper=sleeper)
    await service.start()
    assert await service.notify_team_memory_write() is True
    await spin_until(lambda: len(sleeper.futures) == 1)
    sleeper.release_latest()
    await spin_until(lambda: service.push_in_progress or len(transport.uploads) == 1)

    result = await service.stop()

    assert result.in_flight_awaited in {True, False}
    assert len(transport.uploads) == 1
