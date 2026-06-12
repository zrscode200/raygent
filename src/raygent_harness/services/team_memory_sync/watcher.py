"""Headless team-memory sync service lifecycle.

"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import get_team_mem_path, is_team_memory_enabled
from raygent_harness.services.team_memory_sync.models import (
    SyncState,
    TeamMemoryPushResult,
    create_sync_state,
)
from raygent_harness.services.team_memory_sync.sync import (
    TeamMemoryTransport,
    pull_team_memory,
    push_team_memory,
)

DEBOUNCE_S = 2.0

Sleeper = Callable[[float], Awaitable[None]]


async def _asyncio_sleep(delay_s: float) -> None:
    await asyncio.sleep(delay_s)


@dataclass(frozen=True)
class TeamMemoryServiceStartResult:
    """Result from starting the headless team-memory sync service."""

    started: bool
    initial_pull_success: bool = False
    initial_files_pulled: int = 0
    server_has_content: bool = False
    skip_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TeamMemoryServiceStopResult:
    """Result from stopping and flushing the headless sync service."""

    in_flight_awaited: bool = False
    flushed_pending: bool = False
    flush_success: bool | None = None
    error: str | None = None


def is_permanent_failure(result: TeamMemoryPushResult) -> bool:
    """Return whether retrying the push without user action is futile."""
    if result.error_type in {"no_oauth", "no_repo"}:
        return True
    return (
        result.http_status is not None
        and 400 <= result.http_status < 500
        and result.http_status not in {409, 429}
    )


class TeamMemorySyncService:
    """Headless equivalent of the reference team-memory watcher lifecycle."""

    def __init__(
        self,
        *,
        settings: MemorySettings,
        transport: TeamMemoryTransport,
        state: SyncState | None = None,
        debounce_s: float = DEBOUNCE_S,
        sleeper: Sleeper = _asyncio_sleep,
        feature_enabled: bool = True,
        sync_available: bool = True,
        repo_available: bool = True,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.state = state
        self.debounce_s = debounce_s
        self.sleeper = sleeper
        self.feature_enabled = feature_enabled
        self.sync_available = sync_available
        self.repo_available = repo_available

        self.started = False
        self.push_in_progress = False
        self.has_pending_changes = False
        self.push_suppressed_reason: str | None = None

        self._debounce_task: asyncio.Task[None] | None = None
        self._current_push_task: asyncio.Task[None] | None = None

    async def start(self) -> TeamMemoryServiceStartResult:
        """Create sync state, perform initial pull, and enable explicit notify."""
        if self.started:
            return TeamMemoryServiceStartResult(started=True)
        if not self.feature_enabled:
            return TeamMemoryServiceStartResult(started=False, skip_reason="feature_disabled")
        if not is_team_memory_enabled(self.settings):
            return TeamMemoryServiceStartResult(started=False, skip_reason="team_memory_disabled")
        if not self.sync_available:
            return TeamMemoryServiceStartResult(started=False, skip_reason="sync_unavailable")
        if not self.repo_available:
            return TeamMemoryServiceStartResult(started=False, skip_reason="no_repo")

        self.state = self.state or create_sync_state()

        initial_pull_success = False
        initial_files_pulled = 0
        server_has_content = False
        error: str | None = None

        try:
            pull_result = await pull_team_memory(self.state, self.settings, self.transport)
            initial_pull_success = pull_result.success
            initial_files_pulled = pull_result.files_written if pull_result.success else 0
            server_has_content = pull_result.entry_count > 0
            if not pull_result.success:
                error = pull_result.error
        except Exception as exc:
            error = str(exc)

        get_team_mem_path(self.settings).mkdir(parents=True, exist_ok=True)
        self.started = True
        return TeamMemoryServiceStartResult(
            started=True,
            initial_pull_success=initial_pull_success,
            initial_files_pulled=initial_files_pulled,
            server_has_content=server_has_content,
            error=error,
        )

    async def notify_team_memory_write(self) -> bool:
        """Schedule a debounced push after an explicit team-memory write."""
        if not self.started or self.state is None:
            return False
        return self._schedule_push()

    async def notify_team_memory_unlink(self) -> bool:
        """Clear permanent-failure suppression and schedule a recovery push."""
        if self.push_suppressed_reason is not None:
            self.push_suppressed_reason = None
        return await self.notify_team_memory_write()

    def _schedule_push(self) -> bool:
        if self.push_suppressed_reason is not None:
            return False

        self.has_pending_changes = True
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_then_push())
        return True

    async def _debounce_then_push(self) -> None:
        try:
            await self.sleeper(self.debounce_s)
        except asyncio.CancelledError:
            return

        if self.push_in_progress:
            self._schedule_push()
            return

        self._current_push_task = asyncio.create_task(self._execute_push())

    async def _execute_push(self) -> None:
        if self.state is None:
            return

        self.push_in_progress = True
        try:
            result = await push_team_memory(self.state, self.settings, self.transport)
            if result.success:
                self.has_pending_changes = False
            elif is_permanent_failure(result) and self.push_suppressed_reason is None:
                self.push_suppressed_reason = (
                    f"http_{result.http_status}"
                    if result.http_status is not None
                    else result.error_type or "unknown"
                )
        except Exception:
            pass
        finally:
            self.push_in_progress = False
            self._current_push_task = None

    async def flush(self) -> bool | None:
        """Flush pending changes immediately, respecting permanent suppression."""
        if self.state is None or not self.has_pending_changes:
            return None
        if self.push_suppressed_reason is not None:
            return False

        result = await push_team_memory(self.state, self.settings, self.transport)
        if result.success:
            self.has_pending_changes = False
        elif is_permanent_failure(result) and self.push_suppressed_reason is None:
            self.push_suppressed_reason = (
                f"http_{result.http_status}"
                if result.http_status is not None
                else result.error_type or "unknown"
            )
        return result.success

    async def stop(self) -> TeamMemoryServiceStopResult:
        """Cancel debounce, await in-flight push, and best-effort flush pending changes."""
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = None

        in_flight_awaited = False
        if self._current_push_task is not None:
            in_flight_awaited = True
            with suppress(Exception):
                await self._current_push_task

        flushed_pending = self.has_pending_changes and self.state is not None
        try:
            flush_success = await self.flush()
            error = None
        except Exception as exc:
            flush_success = False
            error = str(exc)

        self.started = False
        return TeamMemoryServiceStopResult(
            in_flight_awaited=in_flight_awaited,
            flushed_pending=flushed_pending,
            flush_success=flush_success,
            error=error,
        )


__all__ = [
    "DEBOUNCE_S",
    "TeamMemoryServiceStartResult",
    "TeamMemoryServiceStopResult",
    "TeamMemorySyncService",
    "is_permanent_failure",
]
