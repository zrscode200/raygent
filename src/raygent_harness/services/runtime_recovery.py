"""Headless runtime recovery orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from raygent_harness.coordinator import (
    CoordinatorRuntime,
    CoordinatorRuntimeSnapshotLoadResult,
    CoordinatorRuntimeSnapshotStore,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.query_engine import QueryEngine
from raygent_harness.core.task import AgentRouteRecord, TaskStateBase
from raygent_harness.core.tasks.remote_agent import restore_remote_agents
from raygent_harness.core.tool import ToolUseContext
from raygent_harness.services.agent_routes import (
    AgentRouteRecordStore,
    normalize_agent_route_record_for_resume,
)
from raygent_harness.services.improvement_runtime import (
    ImprovementRecordStore,
    ImprovementRuntimeRecordQuery,
    ImprovementRuntimeRecoveryRequest,
    ImprovementRuntimeRecoveryResult,
    recover_improvement_runtime_chain,
)
from raygent_harness.services.task_notification_replay import (
    TaskNotificationReplayRecord,
    TaskNotificationReplayResult,
    replay_task_notifications,
)
from raygent_harness.services.task_output import TaskOutputStore
from raygent_harness.services.transcript import (
    SessionReplay,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptReadResult,
    TranscriptReadStats,
    TranscriptScope,
    TranscriptStore,
    load_session_replay,
    transcript_entry_from_json,
    transcript_entry_to_json,
)
from raygent_harness.services.worktree import WorktreeCleanupPolicy, WorktreeInfo

RuntimeRecoveryWarningSource = Literal[
    "transcript",
    "coordinator",
    "route_record",
    "remote_agent",
    "task_notification",
    "worktree",
    "task_output",
    "improvement_runtime",
]


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryWarning:
    source: RuntimeRecoveryWarningSource
    reason: str
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryWorktreeStatus:
    task_id: str
    agent_id: str
    path: str | None
    exists: bool
    has_changes: bool | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryTaskOutputStatus:
    task_id: str
    output_file: str | None
    available: bool
    byte_count: int = 0
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryRequest:
    config: QueryConfig
    deps: QueryDeps
    ctx: ToolUseContext
    transcript_scope: TranscriptScope | None = None
    transcript_path: str | None = None
    runtime_session_id: str | None = None
    coordinator_snapshot_store: CoordinatorRuntimeSnapshotStore | None = None
    agent_route_record_store: AgentRouteRecordStore | None = None
    task_output_store: TaskOutputStore | None = None
    improvement_record_store: ImprovementRecordStore | None = None
    improvement_record_query: ImprovementRuntimeRecordQuery | None = None
    remote_poll_interval_s: float = 0.05
    restore_remote_agents_enabled: bool = True
    offline_task_notifications: Sequence[TaskNotificationReplayRecord] = ()


@dataclass(frozen=True, slots=True)
class RuntimeRecoveryResult:
    replay: SessionReplay
    transcript_scope: TranscriptScope
    transcript_path: str | None
    last_message_entry_id: str | None
    coordinator_runtime_restored: bool = False
    restored_route_records: tuple[AgentRouteRecord, ...] = ()
    restored_agent_names: tuple[str, ...] = ()
    restored_remote_task_ids: tuple[str, ...] = ()
    worktree_statuses: tuple[RuntimeRecoveryWorktreeStatus, ...] = ()
    task_output_statuses: tuple[RuntimeRecoveryTaskOutputStatus, ...] = ()
    task_notification_replay: TaskNotificationReplayResult = field(
        default_factory=TaskNotificationReplayResult
    )
    improvement_chain_recovery: ImprovementRuntimeRecoveryResult | None = None
    warnings: tuple[RuntimeRecoveryWarning, ...] = ()
    transcript_store: TranscriptStore | None = None

    def build_query_engine(
        self,
        config: QueryConfig,
        deps: QueryDeps,
        ctx: ToolUseContext,
    ) -> QueryEngine:
        """Build a QueryEngine that appends future turns to the resumed chain."""

        resumed_config = replace(
            config,
            session_id=self.transcript_scope.session_id,
            agent_id=self.transcript_scope.agent_id,
        )
        resumed_ctx = replace(
            ctx,
            session_id=self.transcript_scope.session_id,
            agent_id=self.transcript_scope.agent_id,
        )
        resumed_deps = (
            replace(deps, transcript_store=self.transcript_store)
            if self.transcript_store is not None
            else deps
        )
        return QueryEngine.from_replay(
            resumed_config,
            resumed_deps,
            resumed_ctx,
            self.replay,
            transcript_scope=self.transcript_scope,
        )


async def resume_runtime_session(
    request: RuntimeRecoveryRequest,
) -> RuntimeRecoveryResult:
    """Restore transcript and available runtime state for a resumed session."""

    service = RuntimeRecoveryService()
    return await service.resume(request)


class RuntimeRecoveryService:
    """Coordinates restart/resume state across injected stores and backends."""

    async def resume(
        self,
        request: RuntimeRecoveryRequest,
    ) -> RuntimeRecoveryResult:
        deps = request.deps
        transcript_store = _transcript_store_for_replay(request)

        warnings: list[RuntimeRecoveryWarning] = []
        initial_scope = await _initial_transcript_scope(request, transcript_store)
        replay = await load_session_replay(
            transcript_store,
            initial_scope,
            observability=deps.observability,
            observability_context=_recovery_context(request, initial_scope),
        )
        warnings.extend(
            RuntimeRecoveryWarning(source="transcript", reason=warning)
            for warning in replay.warnings
        )
        transcript_scope = _adopted_transcript_scope(
            initial_scope,
            replay,
            request.runtime_session_id,
        )
        transcript_path = _resolve_transcript_path(request, transcript_scope, warnings)

        coordinator_runtime_restored = await _restore_coordinator_runtime(
            request,
            transcript_scope,
            warnings,
        )
        restored_routes = await _restore_agent_routes(
            request,
            transcript_scope,
            warnings,
        )
        worktree_statuses = await _check_worktrees(request, restored_routes, warnings)
        replay_results: list[TaskNotificationReplayResult] = []
        offline_task_notification_count = [len(request.offline_task_notifications)]
        if request.offline_task_notifications:
            replay_results.append(
                _replay_offline_task_notifications(
                    request,
                    transcript_scope,
                    tuple(request.offline_task_notifications),
                    warnings,
                )
            )
        restored_remote_ids = await _restore_remote_agents(
            request,
            transcript_scope,
            warnings,
            replay_results,
            offline_task_notification_count,
        )
        task_output_statuses = await _check_task_outputs(request, restored_routes, warnings)
        improvement_chain_recovery = await _recover_improvement_runtime_chain(
            request,
            transcript_scope,
            warnings,
        )
        replay_result = _combine_task_notification_replay_results(replay_results)

        deps.observability.emit(
            "runtime_recovery.completed",
            context=_recovery_context(request, transcript_scope),
            data={
                "message_count": len(replay.messages),
                "compact_boundary_count": len(replay.compact_boundaries),
                "content_replacement_count": len(replay.content_replacements),
                "warning_count": len(warnings),
                "coordinator_runtime_restored": coordinator_runtime_restored,
                "restored_route_count": len(restored_routes),
                "restored_agent_name_count": len(
                    tuple(record.name for record in restored_routes if record.name)
                ),
                "restored_remote_task_count": len(restored_remote_ids),
                "worktree_status_count": len(worktree_statuses),
                "task_output_status_count": len(task_output_statuses),
                "improvement_chain_recovery_present": (
                    improvement_chain_recovery is not None
                ),
                "improvement_chain_status": None
                if improvement_chain_recovery is None
                else improvement_chain_recovery.status,
                "improvement_chain_record_count": 0
                if improvement_chain_recovery is None
                else len(improvement_chain_recovery.records),
                "improvement_chain_warning_count": 0
                if improvement_chain_recovery is None
                else len(improvement_chain_recovery.warnings),
                "improvement_chain_next_record_kind_present": False
                if improvement_chain_recovery is None
                else improvement_chain_recovery.next_record_kind is not None,
                "improvement_chain_permission_summary_present": False
                if improvement_chain_recovery is None
                else improvement_chain_recovery.permission_summary is not None,
                "offline_task_notification_count": offline_task_notification_count[0],
                "replayed_task_notification_count": replay_result.enqueued_count,
                "skipped_replay_duplicate_count": (
                    replay_result.skipped_duplicate_count
                ),
                "skipped_replay_explicit_stop_count": (
                    replay_result.skipped_explicit_stop_count
                ),
                "skipped_replay_coordinator_processed_count": (
                    replay_result.skipped_coordinator_processed_count
                ),
                "transcript_path_present": transcript_path is not None,
                "last_message_entry_id_present": replay.last_message_entry_id is not None,
            },
        )
        return RuntimeRecoveryResult(
            replay=replay,
            transcript_scope=transcript_scope,
            transcript_path=transcript_path,
            last_message_entry_id=replay.last_message_entry_id,
            coordinator_runtime_restored=coordinator_runtime_restored,
            restored_route_records=restored_routes,
            restored_agent_names=tuple(
                record.name for record in restored_routes if record.name is not None
            ),
            restored_remote_task_ids=restored_remote_ids,
            worktree_statuses=worktree_statuses,
            task_output_statuses=task_output_statuses,
            task_notification_replay=replay_result,
            improvement_chain_recovery=improvement_chain_recovery,
            warnings=tuple(warnings),
            transcript_store=transcript_store,
        )


async def _recover_improvement_runtime_chain(
    request: RuntimeRecoveryRequest,
    transcript_scope: TranscriptScope,
    warnings: list[RuntimeRecoveryWarning],
) -> ImprovementRuntimeRecoveryResult | None:
    store = request.improvement_record_store
    query = request.improvement_record_query
    if store is None and query is None:
        return None
    if store is None or query is None:
        warnings.append(
            RuntimeRecoveryWarning(
                source="improvement_runtime",
                reason=(
                    "improvement_record_store and improvement_record_query are both "
                    "required for improvement runtime recovery"
                ),
            )
        )
        return None

    result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id=f"runtime_recovery:{transcript_scope.session_id}",
            record_store=store,
            query=query,
            expected_session_id=transcript_scope.session_id,
            metadata={"source": "runtime_recovery"},
        )
    )
    if result.status == "blocked":
        warnings.append(
            RuntimeRecoveryWarning(
                source="improvement_runtime",
                reason="improvement runtime recovery blocked",
            )
        )
    return result


async def _initial_transcript_scope(
    request: RuntimeRecoveryRequest,
    transcript_store: TranscriptStore,
) -> TranscriptScope:
    if request.transcript_scope is not None:
        return request.transcript_scope
    session_id = request.config.session_id or request.ctx.session_id
    if not session_id:
        raise ValueError("runtime recovery requires a session id")
    fallback = TranscriptScope(
        session_id=session_id,
        agent_id=request.ctx.agent_id,
        is_sidechain=request.ctx.agent_id is not None,
        runtime_session_id=request.runtime_session_id,
    )
    if request.transcript_path is not None and isinstance(
        transcript_store,
        _ExplicitTranscriptPathStore,
    ):
        return await transcript_store.infer_scope(fallback)
    return fallback


def _adopted_transcript_scope(
    scope: TranscriptScope,
    replay: SessionReplay,
    requested_runtime_session_id: str | None,
) -> TranscriptScope:
    return replace(
        scope,
        session_id=replay.session_id,
        agent_id=replay.agent_id if replay.agent_id is not None else scope.agent_id,
        is_sidechain=replay.is_sidechain or scope.is_sidechain,
        runtime_session_id=(
            requested_runtime_session_id
            or replay.runtime_session_id
            or scope.runtime_session_id
        ),
    )


def _transcript_store_for_replay(request: RuntimeRecoveryRequest) -> TranscriptStore:
    if request.transcript_path is not None:
        return _ExplicitTranscriptPathStore(request.transcript_path)
    if request.deps.transcript_store is None:
        raise ValueError("runtime recovery requires QueryDeps.transcript_store")
    return request.deps.transcript_store


def _resolve_transcript_path(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
    warnings: list[RuntimeRecoveryWarning],
) -> str | None:
    if request.transcript_path is not None:
        return request.transcript_path
    store = request.deps.transcript_store
    if store is None:
        return None
    try:
        return store.path_for(scope)
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="transcript",
                reason="transcript_path_unavailable",
                error_type=type(exc).__name__,
            )
        )
        return None


async def _restore_coordinator_runtime(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
    warnings: list[RuntimeRecoveryWarning],
) -> bool:
    snapshot_store = request.coordinator_snapshot_store
    if snapshot_store is None:
        return False

    async def load(
        runtime_session_id: str | None,
    ) -> CoordinatorRuntimeSnapshotLoadResult:
        return await snapshot_store.load(
            scope.session_id,
            runtime_session_id=runtime_session_id,
        )

    try:
        load_result = await load(scope.runtime_session_id)
        if load_result.snapshot is None and scope.runtime_session_id is not None:
            fallback = await load(None)
            if fallback.snapshot is not None:
                load_result = fallback
            elif fallback.warnings:
                warnings.extend(
                    RuntimeRecoveryWarning(source="coordinator", reason=warning)
                    for warning in fallback.warnings
                )
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="coordinator",
                reason="snapshot_load_failed",
                error_type=type(exc).__name__,
            )
        )
        return False

    warnings.extend(
        RuntimeRecoveryWarning(source="coordinator", reason=warning)
        for warning in load_result.warnings
    )
    if load_result.snapshot is None:
        return False
    try:
        request.deps.coordinator_runtime = CoordinatorRuntime.from_snapshot(
            load_result.snapshot,
            observability=request.deps.observability,
            event_context=_recovery_context(request, scope),
            clock=request.deps.clock.now,
        )
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="coordinator",
                reason="snapshot_restore_failed",
                error_type=type(exc).__name__,
            )
        )
        return False
    return True


async def _restore_agent_routes(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
    warnings: list[RuntimeRecoveryWarning],
) -> tuple[AgentRouteRecord, ...]:
    route_store = request.agent_route_record_store or request.deps.agent_route_record_store
    if route_store is None:
        return ()
    try:
        load_result = await route_store.list_records(scope.session_id)
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="route_record",
                reason="route_records_load_failed",
                error_type=type(exc).__name__,
            )
        )
        return ()

    warnings.extend(
        RuntimeRecoveryWarning(source="route_record", reason=warning)
        for warning in load_result.warnings
    )
    restored: list[AgentRouteRecord] = []
    for record in sorted(
        load_result.records,
        key=lambda item: (item.route_registered_at, item.task_id),
    ):
        if record.task_type != "local_agent":
            warnings.append(
                RuntimeRecoveryWarning(
                    source="route_record",
                    reason="skipped_non_local_agent_route",
                )
            )
            continue
        normalized = normalize_agent_route_record_for_resume(
            record,
            parent_session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
        )
        request.deps.task_store.agent_route_records[normalized.task_id] = normalized
        if normalized.name is not None:
            request.deps.task_store.agent_name_registry[normalized.name] = (
                normalized.task_id
            )
        restored.append(normalized)
    return tuple(restored)


async def _check_worktrees(
    request: RuntimeRecoveryRequest,
    records: tuple[AgentRouteRecord, ...],
    warnings: list[RuntimeRecoveryWarning],
) -> tuple[RuntimeRecoveryWorktreeStatus, ...]:
    statuses: list[RuntimeRecoveryWorktreeStatus] = []
    manager = request.deps.worktree_manager
    for record in records:
        if record.worktree_path is None:
            continue
        exists = Path(record.worktree_path).exists()
        has_changes: bool | None = None
        error_type: str | None = None
        if exists and manager is not None:
            try:
                has_changes = await manager.has_changes(
                    WorktreeInfo(
                        path=record.worktree_path,
                        branch=record.worktree_branch,
                        slug=record.worktree_slug,
                        created_at=record.worktree_created_at,
                        touched_at=record.worktree_touched_at,
                        owner_task_id=record.task_id,
                        cleanup_policy=_worktree_cleanup_policy(record),
                    )
                )
            except Exception as exc:
                error_type = type(exc).__name__
                warnings.append(
                    RuntimeRecoveryWarning(
                        source="worktree",
                        reason="worktree_check_failed",
                        error_type=error_type,
                    )
                )
        statuses.append(
            RuntimeRecoveryWorktreeStatus(
                task_id=record.task_id,
                agent_id=record.agent_id,
                path=record.worktree_path,
                exists=exists,
                has_changes=has_changes,
                error_type=error_type,
            )
        )
    return tuple(statuses)


async def _restore_remote_agents(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
    warnings: list[RuntimeRecoveryWarning],
    replay_results: list[TaskNotificationReplayResult],
    offline_task_notification_count: list[int],
) -> tuple[str, ...]:
    if not request.restore_remote_agents_enabled:
        return ()

    async def replay_terminal_record(record: TaskNotificationReplayRecord) -> bool:
        offline_task_notification_count[0] += 1
        result = _replay_offline_task_notifications(
            request,
            scope,
            (record,),
            warnings,
        )
        replay_results.append(result)
        return _task_notification_replay_allows_remote_delete(result)

    try:
        return await restore_remote_agents(
            deps=request.deps,
            poll_interval_s=request.remote_poll_interval_s,
            parent_observability_context=_recovery_context(request, scope),
            terminal_replay_sink=replay_terminal_record,
        )
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="remote_agent",
                reason="remote_restore_failed",
                error_type=type(exc).__name__,
            )
        )
        return ()


def _replay_offline_task_notifications(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
    records: tuple[TaskNotificationReplayRecord, ...],
    warnings: list[RuntimeRecoveryWarning],
) -> TaskNotificationReplayResult:
    if not records:
        return TaskNotificationReplayResult()
    try:
        result = replay_task_notifications(
            store=request.deps.task_store,
            records=records,
            coordinator_runtime=request.deps.coordinator_runtime,
            observability=request.deps.observability,
            event_context=_recovery_context(request, scope),
        )
    except Exception as exc:
        warnings.append(
            RuntimeRecoveryWarning(
                source="task_notification",
                reason="offline_notification_replay_failed",
                error_type=type(exc).__name__,
            )
        )
        return TaskNotificationReplayResult(
            warning_count=1,
            warnings=(f"offline_notification_replay_failed:{type(exc).__name__}",),
        )
    warnings.extend(
        RuntimeRecoveryWarning(source="task_notification", reason=warning)
        for warning in result.warnings
    )
    return result


def _task_notification_replay_allows_remote_delete(
    result: TaskNotificationReplayResult,
) -> bool:
    accepted_or_intentionally_suppressed = (
        result.enqueued_count
        + result.skipped_duplicate_count
        + result.skipped_explicit_stop_count
        + result.skipped_coordinator_processed_count
    )
    return accepted_or_intentionally_suppressed > 0 and result.warning_count == 0


def _combine_task_notification_replay_results(
    results: Sequence[TaskNotificationReplayResult],
) -> TaskNotificationReplayResult:
    if not results:
        return TaskNotificationReplayResult()
    return TaskNotificationReplayResult(
        enqueued_notifications=tuple(
            notification
            for result in results
            for notification in result.enqueued_notifications
        ),
        skipped_duplicate_count=sum(
            result.skipped_duplicate_count for result in results
        ),
        skipped_explicit_stop_count=sum(
            result.skipped_explicit_stop_count for result in results
        ),
        skipped_coordinator_processed_count=sum(
            result.skipped_coordinator_processed_count for result in results
        ),
        warning_count=sum(result.warning_count for result in results),
        warnings=tuple(warning for result in results for warning in result.warnings),
    )


async def _check_task_outputs(
    request: RuntimeRecoveryRequest,
    records: tuple[AgentRouteRecord, ...],
    warnings: list[RuntimeRecoveryWarning],
) -> tuple[RuntimeRecoveryTaskOutputStatus, ...]:
    task_ids = set(request.deps.task_store.tasks)
    task_ids.update(record.task_id for record in records)
    statuses: list[RuntimeRecoveryTaskOutputStatus] = []
    output_store = request.task_output_store
    for task_id in sorted(task_ids):
        task = request.deps.task_store.tasks.get(task_id)
        if output_store is None:
            status = _task_output_status_from_state(task_id, task)
            if status is not None:
                statuses.append(status)
            continue
        try:
            byte_count = await output_store.size(task_id)
        except Exception as exc:
            warnings.append(
                RuntimeRecoveryWarning(
                    source="task_output",
                    reason="task_output_check_failed",
                    error_type=type(exc).__name__,
                )
            )
            statuses.append(
                RuntimeRecoveryTaskOutputStatus(
                    task_id=task_id,
                    output_file=task.output_file if task is not None else None,
                    available=False,
                    error_type=type(exc).__name__,
                )
            )
            continue
        statuses.append(
            RuntimeRecoveryTaskOutputStatus(
                task_id=task_id,
                output_file=task.output_file if task is not None else None,
                available=byte_count > 0,
                byte_count=byte_count,
            )
        )
    return tuple(statuses)


def _task_output_status_from_state(
    task_id: str,
    task: TaskStateBase | None,
) -> RuntimeRecoveryTaskOutputStatus | None:
    if task is None or task.output_file is None:
        return None
    path = Path(task.output_file)
    try:
        stat_result = path.stat()
    except OSError as exc:
        return RuntimeRecoveryTaskOutputStatus(
            task_id=task_id,
            output_file=task.output_file,
            available=False,
            error_type=type(exc).__name__,
        )
    return RuntimeRecoveryTaskOutputStatus(
        task_id=task_id,
        output_file=task.output_file,
        available=path.is_file(),
        byte_count=stat_result.st_size if path.is_file() else 0,
    )


def _worktree_cleanup_policy(record: AgentRouteRecord) -> WorktreeCleanupPolicy:
    if record.worktree_cleanup_policy in {"remove_if_clean", "keep", "manual"}:
        return cast(WorktreeCleanupPolicy, record.worktree_cleanup_policy)
    return "manual"


def _recovery_context(
    request: RuntimeRecoveryRequest,
    scope: TranscriptScope,
) -> KernelEventContext:
    base = request.ctx.observability_context or KernelEventContext(
        session_id=request.ctx.session_id,
        agent_id=request.ctx.agent_id,
        source="runtime_recovery",
    )
    return replace(
        base,
        session_id=scope.session_id,
        runtime_session_id=scope.runtime_session_id,
        agent_id=scope.agent_id,
        source="runtime_recovery",
    )


class _ExplicitTranscriptPathStore:
    """Transcript store bound to one explicit JSONL path."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()

    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
        await self.append_many(scope, (entry,))

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        _ = scope
        if len(entries) == 0:
            return
        await asyncio.to_thread(_append_explicit_transcript_path, self._path, entries)

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        result = await self.read_result(scope)
        return list(result.entries)

    async def read_result(self, scope: TranscriptScope) -> TranscriptReadResult:
        _ = scope
        return await asyncio.to_thread(_read_explicit_transcript_path, self._path)

    async def flush(self, scope: TranscriptScope | None = None) -> None:
        _ = scope

    def path_for(self, scope: TranscriptScope) -> str:
        _ = scope
        return str(self._path)

    async def infer_scope(self, fallback: TranscriptScope) -> TranscriptScope:
        result = await self.read_result(fallback)
        return _infer_explicit_path_scope(result.entries, fallback)


def _read_explicit_transcript_path(path: Path) -> TranscriptReadResult:
    if not path.exists():
        return TranscriptReadResult(entries=())
    entries: list[TranscriptEntry] = []
    warnings: list[str] = []
    total_lines = 0
    decoded_entries = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines = line_number
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(transcript_entry_from_json(line))
                decoded_entries += 1
            except Exception as exc:
                warnings.append(
                    f"skipped transcript line {line_number}: {type(exc).__name__}"
                )
    return TranscriptReadResult(
        entries=tuple(entries),
        warnings=tuple(warnings),
        stats=TranscriptReadStats(
            total_lines_scanned=total_lines,
            decoded_entries=decoded_entries,
            entries_retained=len(entries),
            used_full_read_fallback=True,
        ),
    )


def _append_explicit_transcript_path(
    path: Path,
    entries: Sequence[TranscriptEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(transcript_entry_to_json(entry))
            handle.write("\n")
        handle.flush()


def _infer_explicit_path_scope(
    entries: Sequence[TranscriptEntry],
    fallback: TranscriptScope,
) -> TranscriptScope:
    messages = [entry for entry in entries if isinstance(entry, TranscriptMessageEntry)]
    if not messages:
        return fallback
    if fallback.agent_id is not None:
        candidates = [entry for entry in messages if entry.agent_id == fallback.agent_id]
        if not candidates:
            candidates = [entry for entry in messages if entry.is_sidechain]
    else:
        candidates = [entry for entry in messages if not entry.is_sidechain]
    if not candidates:
        candidates = messages
    latest = max(candidates, key=lambda entry: entry.created_at)
    return replace(
        fallback,
        session_id=latest.session_id,
        runtime_session_id=(
            fallback.runtime_session_id or latest.runtime_session_id
        ),
        agent_id=latest.agent_id,
        is_sidechain=latest.is_sidechain,
    )


__all__ = [
    "RuntimeRecoveryRequest",
    "RuntimeRecoveryResult",
    "RuntimeRecoveryService",
    "RuntimeRecoveryTaskOutputStatus",
    "RuntimeRecoveryWarning",
    "RuntimeRecoveryWarningSource",
    "RuntimeRecoveryWorktreeStatus",
    "resume_runtime_session",
]
