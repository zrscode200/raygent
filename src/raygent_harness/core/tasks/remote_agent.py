"""RemoteAgentTask — protocol-backed remote background agents.

Reference remote agents delegate execution to a product backend, register a
`remote_agent` task, and use a detached poller to turn remote completion into a
model-facing task notification. Raygent keeps the kernel lifecycle and delegates
backend details to `QueryDeps.remote_agent_backend`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventBus, KernelEventContext
from raygent_harness.core.task import (
    TERMINAL_STATUSES,
    AppStateStore,
    TaskNotification,
    TaskStateBase,
    TaskType,
    generate_task_id,
    mark_notified_if_unset,
    register_task_impl,
)
from raygent_harness.services.handoff import (
    HandoffClassificationRequest,
    classify_handoff_warning,
)
from raygent_harness.services.remote_agent import (
    RemoteAgentBackend,
    RemoteAgentLaunchRequest,
    RemoteAgentPersistenceRecord,
    RemoteAgentPersistenceStore,
    RemoteAgentPollRequest,
    RemoteAgentRestoreRequest,
    RemoteAgentRestoreResult,
    RemoteAgentStopRequest,
)
from raygent_harness.services.task_notification_replay import (
    TaskNotificationReplayRecord,
    remote_agent_restore_replay_record,
    remote_agent_terminal_dedupe_key,
)


@dataclass
class RemoteAgentState(TaskStateBase):
    """State for a remote_agent task.

    `id` is Raygent's task id. `remote_id` is the backend-specific session/job
    id returned by the installed backend.
    """

    parent_agent_id: str | None = None
    prompt: str = ""
    agent_type: str | None = None
    model: str | None = None
    cwd: str | None = None
    remote_id: str = ""
    session_url: str | None = None
    title: str = ""
    final_message: str | None = None
    error: str | None = None
    is_error: bool = False
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass
class RemoteAgentTask:
    """Task vtable for protocol-backed remote agents."""

    name: str = "remote_agent"
    type: TaskType = "remote_agent"

    async def kill(self, task_id: str, store: AppStateStore) -> None:
        """Stop/archive remote work and mark the task killed.

        Reference `RemoteAgentTask.kill` flips to killed, marks notified, and
        archives the remote session without enqueueing a model-facing
        completion notification. Raygent mirrors that: explicit stop is already
        user/model initiated, so the poller should not later emit a duplicate.
        """
        task = store.tasks.get(task_id)
        if not isinstance(task, RemoteAgentState):
            return
        if task.status in TERMINAL_STATUSES:
            return

        killed = False

        def update(t: TaskStateBase) -> TaskStateBase:
            nonlocal killed
            if not isinstance(t, RemoteAgentState) or t.status != "running":
                return t
            killed = True
            return replace(
                t,
                status="killed",
                end_time=time.time(),
                notified=True,
                is_error=True,
                error="remote agent stopped",
            )

        store.update_task(task_id, update)
        if not killed:
            return

        poller = _POLL_TASKS.get(task_id)
        if poller is not None and not poller.done():
            poller.cancel()

        backend = _BACKENDS.get(task_id)
        if backend is not None:
            _schedule_backend_stop(
                backend,
                RemoteAgentStopRequest(
                    task_id=task_id,
                    remote_id=task.remote_id,
                    metadata=dict(task.metadata),
                ),
            )
        persistence_store = _PERSISTENCE_STORES.get(task_id)
        if persistence_store is not None:
            _schedule_persistence_delete(
                persistence_store,
                task_id,
                event_context=KernelEventContext(
                    agent_id=task_id,
                    parent_agent_id=task.parent_agent_id,
                    source="agent",
                ),
                observability=store.observability,
                operation="delete",
            )


_POLL_TASKS: dict[str, asyncio.Task[None]] = {}
_STOP_TASKS: dict[str, asyncio.Task[None]] = {}
_BACKENDS: dict[str, RemoteAgentBackend] = {}
_PERSISTENCE_STORES: dict[str, RemoteAgentPersistenceStore] = {}
_PERSISTENCE_TASKS: dict[str, asyncio.Task[None]] = {}

_RestoreCallable = Callable[[RemoteAgentRestoreRequest], Awaitable[RemoteAgentRestoreResult]]
_TerminalReplaySink = Callable[[TaskNotificationReplayRecord], Awaitable[bool]]


async def spawn_remote_agent(
    *,
    prompt: str,
    description: str,
    agent_type: str,
    parent_agent_id: str | None,
    parent_deps: QueryDeps,
    tool_use_id: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    poll_interval_s: float = 0.05,
    parent_observability_context: KernelEventContext | None = None,
) -> str:
    """Launch a remote task through the installed backend and start polling."""
    backend = parent_deps.remote_agent_backend
    if backend is None:
        raise RuntimeError("remote AgentTool backend is not configured")

    task_id = generate_task_id("remote_agent")
    result = await backend.launch(
        RemoteAgentLaunchRequest(
            task_id=task_id,
            prompt=prompt,
            description=description,
            agent_type=agent_type,
            parent_agent_id=parent_agent_id,
            tool_use_id=tool_use_id,
            model=model,
            cwd=cwd,
        )
    )

    start_time = time.time()
    state = RemoteAgentState(
        id=task_id,
        type="remote_agent",
        description=description,
        status="running",
        start_time=start_time,
        tool_use_id=tool_use_id,
        parent_agent_id=parent_agent_id,
        prompt=prompt,
        agent_type=agent_type,
        model=model,
        cwd=cwd,
        remote_id=result.remote_id,
        session_url=result.session_url,
        title=result.title,
        metadata=dict(result.metadata),
    )
    parent_deps.task_store.register_task(state)
    _BACKENDS[task_id] = backend
    persistence_store = parent_deps.remote_agent_persistence_store
    if persistence_store is not None:
        _PERSISTENCE_STORES[task_id] = persistence_store
    event_context = (
        parent_observability_context.for_child_agent(task_id).with_source("agent")
        if parent_observability_context is not None
        else KernelEventContext(
            agent_id=task_id,
            parent_agent_id=parent_agent_id,
            source="agent",
        )
    )
    if persistence_store is not None:
        _schedule_persistence_save(
            persistence_store,
            _record_from_state(state),
            event_context=event_context,
            observability=parent_deps.observability,
            operation="save",
        )
    parent_deps.observability.emit(
        "agent.child.started",
        context=event_context,
        data={
            "task_id": task_id,
            "task_type": "remote_agent",
            "remote_id_present": bool(result.remote_id),
            "remote_id_char_count": len(result.remote_id),
            "parent_agent_id": parent_agent_id,
            "agent_type": agent_type,
            "tool_use_id": tool_use_id,
            "prompt_char_count": len(prompt),
            "model": model,
            "session_url_present": result.session_url is not None,
            "title_char_count": len(result.title),
            "metadata_present": bool(result.metadata),
            "metadata_key_count": len(result.metadata),
        },
    )

    poller = asyncio.create_task(
        _poll_remote_agent(
            task_id=task_id,
            deps=parent_deps,
            backend=backend,
            poll_interval_s=poll_interval_s,
            event_context=event_context,
        ),
        name=f"remote-agent-poller:{task_id}",
    )
    _POLL_TASKS[task_id] = poller
    return task_id


async def _poll_remote_agent(
    *,
    task_id: str,
    deps: QueryDeps,
    backend: RemoteAgentBackend,
    poll_interval_s: float,
    event_context: KernelEventContext,
) -> None:
    store = deps.task_store
    try:
        while True:
            task = store.tasks.get(task_id)
            if not isinstance(task, RemoteAgentState) or task.status != "running":
                return

            try:
                deps.observability.emit(
                    "remote_agent.poll.started",
                    context=_remote_event_context(task, event_context),
                    data={
                        "task_id": task_id,
                        "remote_id_present": bool(task.remote_id),
                        "remote_id_char_count": len(task.remote_id),
                        "metadata_present": bool(task.metadata),
                        "metadata_key_count": len(task.metadata),
                    },
                )
                result = await backend.poll(
                    RemoteAgentPollRequest(
                        task_id=task_id,
                        remote_id=task.remote_id,
                        metadata=dict(task.metadata),
                    )
                )
            except Exception as exc:
                deps.observability.emit(
                    "remote_agent.poll.failed",
                    context=_remote_event_context(task, event_context),
                    data={
                        "task_id": task_id,
                        "remote_id_present": bool(task.remote_id),
                        "remote_id_char_count": len(task.remote_id),
                        "error_type": type(exc).__name__,
                        "message_char_count": len(str(exc)),
                    },
                )
                await asyncio.sleep(poll_interval_s)
                continue
            deps.observability.emit(
                "remote_agent.poll.completed",
                context=_remote_event_context(task, event_context),
                data={
                    "task_id": task_id,
                    "remote_id_present": bool(task.remote_id),
                    "remote_id_char_count": len(task.remote_id),
                    "status": result.status,
                    "message_char_count": len(result.message),
                    "error_char_count": len(result.error) if result.error else 0,
                    "metadata_present": bool(result.metadata),
                    "metadata_key_count": len(result.metadata),
                },
            )

            if result.status == "running":
                if result.metadata:
                    store.update_task(task_id, _metadata_updater(result.metadata))
                    updated_task = store.tasks.get(task_id)
                    persistence_store = deps.remote_agent_persistence_store
                    if (
                        persistence_store is not None
                        and isinstance(updated_task, RemoteAgentState)
                    ):
                        _schedule_persistence_save(
                            persistence_store,
                            _record_from_state(updated_task),
                            event_context=_remote_event_context(
                                updated_task,
                                event_context,
                            ),
                            observability=deps.observability,
                            operation="save",
                        )
                await asyncio.sleep(poll_interval_s)
                continue

            final_status: Literal["completed", "failed"] = (
                "completed" if result.status == "completed" else "failed"
            )
            final_message = result.message
            error = result.error

            # Status first: notification embellishments must not block awaiters.
            store.update_task(
                task_id,
                _terminal_updater(
                    final_status=final_status,
                    final_message=final_message,
                    error=error,
                    metadata=result.metadata,
                ),
            )
            persistence_store = deps.remote_agent_persistence_store
            if persistence_store is not None:
                _schedule_persistence_delete(
                    persistence_store,
                    task_id,
                    event_context=_remote_event_context(task, event_context),
                    observability=deps.observability,
                    operation="delete",
                )

            warning = await _handoff_warning(
                deps=deps,
                task_id=task_id,
                final_status=final_status,
                final_message=final_message,
                error=error,
            )
            if warning:
                final_message = f"{warning}\n\n{final_message}".strip()
            deps.observability.emit(
                "agent.handoff.classified",
                context=_remote_event_context(task, event_context),
                data={
                    "task_id": task_id,
                    "task_type": "remote_agent",
                    "final_status": final_status,
                    "warning_emitted": bool(warning),
                    "warning_char_count": len(warning) if warning else 0,
                },
            )
            deps.observability.emit(
                "agent.child.completed"
                if final_status == "completed"
                else "agent.child.failed",
                context=_remote_event_context(task, event_context),
                data={
                    "task_id": task_id,
                    "task_type": "remote_agent",
                    "remote_id_present": bool(task.remote_id),
                    "remote_id_char_count": len(task.remote_id),
                    "parent_agent_id": task.parent_agent_id,
                    "final_status": final_status,
                    "final_message_char_count": len(final_message),
                    "error_char_count": len(error) if error else 0,
                },
            )

            if mark_notified_if_unset(store, task_id):
                dedupe_key = remote_agent_terminal_dedupe_key(
                    task_id=task_id,
                    remote_id=task.remote_id,
                    final_status=final_status,
                )
                store.enqueue_notification(
                    TaskNotification(
                        task_id=task_id,
                        message=_build_notification_message(
                            task_id=task_id,
                            description=_get_description(store, task_id),
                            final_status=final_status,
                            final_message=final_message,
                            error=error,
                        ),
                        kind="completed" if final_status == "completed" else "error",
                        tool_use_id=_get_tool_use_id(store, task_id),
                        priority="later",
                        agent_id=_get_parent_agent_id(store, task_id),
                        dedupe_key=dedupe_key,
                    )
                )
            return
    finally:
        _POLL_TASKS.pop(task_id, None)
        _BACKENDS.pop(task_id, None)
        _PERSISTENCE_STORES.pop(task_id, None)


async def restore_remote_agents(
    *,
    deps: QueryDeps,
    poll_interval_s: float = 0.05,
    parent_observability_context: KernelEventContext | None = None,
    terminal_replay_sink: _TerminalReplaySink | None = None,
) -> tuple[str, ...]:
    """Restore running remote-agent tasks from persisted sidecars.

    The persistence record is an identity hint only. Restore asks the installed
    backend for live state before recreating task state and restarting a poller.
    Recoverable backend/persistence failures are metadata-only diagnostics and
    leave the sidecar record in place for a future resume attempt.
    """
    backend = deps.remote_agent_backend
    persistence_store = deps.remote_agent_persistence_store
    if backend is None or persistence_store is None:
        return ()

    restore = _backend_restore_callable(backend)
    if restore is None:
        deps.observability.emit(
            "remote_agent.restore.skipped",
            context=(parent_observability_context or KernelEventContext(source="agent")),
            data={"reason": "backend_restore_unsupported"},
        )
        return ()

    try:
        records = await persistence_store.list_records()
    except Exception as exc:
        _emit_persistence_error(
            deps.observability,
            event_context=parent_observability_context
            or KernelEventContext(source="agent"),
            task_id=None,
            operation="list",
            exc=exc,
        )
        return ()

    restored: list[str] = []
    for record in records:
        existing = deps.task_store.tasks.get(record.task_id)
        if isinstance(existing, RemoteAgentState) and existing.status == "running":
            restored.append(record.task_id)
            continue
        if existing is not None and existing.status in TERMINAL_STATUSES:
            _schedule_persistence_delete(
                persistence_store,
                record.task_id,
                event_context=_record_event_context(record, parent_observability_context),
                observability=deps.observability,
                operation="delete",
            )
            continue

        try:
            result = await restore(
                RemoteAgentRestoreRequest(
                    task_id=record.task_id,
                    remote_id=record.remote_id,
                    metadata=dict(record.metadata),
                )
            )
        except Exception as exc:
            deps.observability.emit(
                "remote_agent.restore.skipped",
                context=_record_event_context(record, parent_observability_context),
                data={
                    "task_id": record.task_id,
                    "reason": "backend_error",
                    "error_type": type(exc).__name__,
                    "message_char_count": len(str(exc)),
                },
            )
            continue

        deps.observability.emit(
            "remote_agent.restore.checked",
            context=_record_event_context(record, parent_observability_context),
            data={
                "task_id": record.task_id,
                "status": result.status,
                "remote_id_present": bool(record.remote_id),
                "remote_id_char_count": len(record.remote_id),
                "metadata_present": bool(result.metadata),
                "metadata_key_count": len(result.metadata),
            },
        )

        if result.status in {"completed", "failed"}:
            replay_record = remote_agent_restore_replay_record(record, result)
            should_delete = terminal_replay_sink is None or replay_record is None
            if terminal_replay_sink is not None and replay_record is not None:
                try:
                    should_delete = await terminal_replay_sink(replay_record)
                except Exception as exc:
                    deps.observability.emit(
                        "remote_agent.restore.notification_replay_failed",
                        context=_record_event_context(
                            record,
                            parent_observability_context,
                        ),
                        data={
                            "task_id": record.task_id,
                            "status": result.status,
                            "error_type": type(exc).__name__,
                            "message_char_count": len(str(exc)),
                        },
                    )
            if should_delete:
                _schedule_persistence_delete(
                    persistence_store,
                    record.task_id,
                    event_context=_record_event_context(
                        record,
                        parent_observability_context,
                    ),
                    observability=deps.observability,
                    operation="delete",
                )
            continue

        if result.status in {"archived", "gone"}:
            _schedule_persistence_delete(
                persistence_store,
                record.task_id,
                event_context=_record_event_context(record, parent_observability_context),
                observability=deps.observability,
                operation="delete",
            )
            continue

        state = _state_from_record(record, result=result)
        deps.task_store.register_task(state)
        _BACKENDS[record.task_id] = backend
        _PERSISTENCE_STORES[record.task_id] = persistence_store
        event_context = _record_event_context(record, parent_observability_context)
        poller = asyncio.create_task(
            _poll_remote_agent(
                task_id=record.task_id,
                deps=deps,
                backend=backend,
                poll_interval_s=poll_interval_s,
                event_context=event_context,
            ),
            name=f"remote-agent-poller:{record.task_id}",
        )
        _POLL_TASKS[record.task_id] = poller
        restored.append(record.task_id)

    return tuple(restored)


def _backend_restore_callable(
    backend: RemoteAgentBackend,
) -> _RestoreCallable | None:
    restore = getattr(backend, "restore", None)
    if not callable(restore):
        return None
    return cast(_RestoreCallable, restore)


def _record_from_state(task: RemoteAgentState) -> RemoteAgentPersistenceRecord:
    return RemoteAgentPersistenceRecord(
        task_id=task.id,
        remote_id=task.remote_id,
        description=task.description,
        parent_agent_id=task.parent_agent_id,
        tool_use_id=task.tool_use_id,
        agent_type=task.agent_type,
        model=task.model,
        cwd=task.cwd,
        session_url=task.session_url,
        title=task.title,
        metadata=dict(task.metadata),
        start_time=task.start_time,
        updated_at=time.time(),
    )


def _state_from_record(
    record: RemoteAgentPersistenceRecord,
    *,
    result: RemoteAgentRestoreResult,
) -> RemoteAgentState:
    return RemoteAgentState(
        id=record.task_id,
        type="remote_agent",
        description=record.description,
        status="running",
        start_time=record.start_time or time.time(),
        tool_use_id=record.tool_use_id,
        parent_agent_id=record.parent_agent_id,
        prompt="",
        agent_type=record.agent_type,
        model=record.model,
        cwd=record.cwd,
        remote_id=record.remote_id,
        session_url=result.session_url if result.session_url is not None else record.session_url,
        title=result.title if result.title is not None else record.title,
        metadata={**record.metadata, **result.metadata},
    )


def _schedule_persistence_save(
    persistence_store: RemoteAgentPersistenceStore,
    record: RemoteAgentPersistenceRecord,
    *,
    event_context: KernelEventContext,
    observability: KernelEventBus,
    operation: str,
) -> None:
    async def save_record() -> None:
        await _save_persistence_record(
            persistence_store,
            record,
            event_context=event_context,
            observability=observability,
            operation=operation,
        )

    _schedule_persistence_operation(
        record.task_id,
        save_record,
        name=f"remote-agent-persistence-save:{record.task_id}",
    )


def _schedule_persistence_delete(
    persistence_store: RemoteAgentPersistenceStore,
    task_id: str,
    *,
    event_context: KernelEventContext,
    observability: KernelEventBus,
    operation: str,
) -> None:
    async def delete_record() -> None:
        try:
            await persistence_store.delete(task_id)
        except Exception as exc:
            _emit_persistence_error(
                observability,
                event_context=event_context,
                task_id=task_id,
                operation=operation,
                exc=exc,
            )
            return
        finally:
            _PERSISTENCE_STORES.pop(task_id, None)

        observability.emit(
            "remote_agent.persistence.deleted",
            context=event_context,
            data={"task_id": task_id},
        )

    _schedule_persistence_operation(
        task_id,
        delete_record,
        name=f"remote-agent-persistence-delete:{task_id}",
    )


def _schedule_persistence_operation(
    task_id: str,
    operation: Callable[[], Awaitable[None]],
    *,
    name: str,
) -> None:
    previous = _PERSISTENCE_TASKS.get(task_id)

    async def run_serialized() -> None:
        if previous is not None and not previous.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await previous
        try:
            await operation()
        finally:
            if _PERSISTENCE_TASKS.get(task_id) is task:
                _PERSISTENCE_TASKS.pop(task_id, None)

    task = asyncio.create_task(run_serialized(), name=name)
    _PERSISTENCE_TASKS[task_id] = task


async def _save_persistence_record(
    persistence_store: RemoteAgentPersistenceStore,
    record: RemoteAgentPersistenceRecord,
    *,
    event_context: KernelEventContext,
    observability: KernelEventBus,
    operation: str,
) -> None:
    try:
        await persistence_store.save(record)
    except Exception as exc:
        _emit_persistence_error(
            observability,
            event_context=event_context,
            task_id=record.task_id,
            operation=operation,
            exc=exc,
        )
        return
    observability.emit(
        "remote_agent.persistence.saved",
        context=event_context,
        data={
            "task_id": record.task_id,
            "remote_id_present": bool(record.remote_id),
            "remote_id_char_count": len(record.remote_id),
            "metadata_present": bool(record.metadata),
            "metadata_key_count": len(record.metadata),
        },
    )


def _emit_persistence_error(
    observability: KernelEventBus,
    *,
    event_context: KernelEventContext,
    task_id: str | None,
    operation: str,
    exc: Exception,
) -> None:
    observability.emit(
        "remote_agent.persistence.failed",
        context=event_context,
        data={
            "task_id": task_id,
            "operation": operation,
            "error_type": type(exc).__name__,
            "message_char_count": len(str(exc)),
        },
    )


def _record_event_context(
    record: RemoteAgentPersistenceRecord,
    base: KernelEventContext | None,
) -> KernelEventContext:
    if base is not None:
        return KernelEventContext(
            session_id=base.session_id,
            runtime_session_id=base.runtime_session_id,
            agent_id=record.task_id,
            parent_agent_id=record.parent_agent_id,
            turn_id=base.turn_id,
            span_id=base.span_id,
            parent_span_id=base.parent_span_id,
            source="agent",
        )
    return KernelEventContext(
        agent_id=record.task_id,
        parent_agent_id=record.parent_agent_id,
        source="agent",
    )


def _metadata_updater(metadata: dict[str, str]) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, RemoteAgentState):
            return t
        return replace(t, metadata={**t.metadata, **metadata})

    return update


def _schedule_backend_stop(
    backend: RemoteAgentBackend,
    request: RemoteAgentStopRequest,
) -> None:
    async def stop_remote() -> None:
        try:
            await backend.stop(request)
        except Exception:
            return
        finally:
            _STOP_TASKS.pop(request.task_id, None)

    task = asyncio.create_task(
        stop_remote(),
        name=f"remote-agent-stop:{request.task_id}",
    )
    _STOP_TASKS[request.task_id] = task


def _terminal_updater(
    *,
    final_status: Literal["completed", "failed"],
    final_message: str,
    error: str | None,
    metadata: dict[str, str],
) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, RemoteAgentState):
            return t
        if t.status in TERMINAL_STATUSES:
            return t
        return replace(
            t,
            status=final_status,
            end_time=time.time(),
            final_message=final_message or None,
            error=error,
            is_error=final_status != "completed",
            metadata={**t.metadata, **metadata},
        )

    return update


async def _handoff_warning(
    *,
    deps: QueryDeps,
    task_id: str,
    final_status: Literal["completed", "failed"],
    final_message: str,
    error: str | None,
) -> str | None:
    if final_status != "completed":
        return None
    task = deps.task_store.tasks.get(task_id)
    if not isinstance(task, RemoteAgentState):
        return None
    return await classify_handoff_warning(
        deps.handoff_classifier,
        HandoffClassificationRequest(
            task_id=task_id,
            task_type="remote_agent",
            agent_type=task.agent_type,
            description=task.description,
            final_status=final_status,
            final_message=final_message,
            error=error,
        ),
        timeout_s=deps.handoff_classifier_timeout_s,
    )


def _get_description(store: AppStateStore, task_id: str) -> str:
    task = store.tasks.get(task_id)
    if isinstance(task, RemoteAgentState):
        return task.description
    return task_id


def _get_tool_use_id(store: AppStateStore, task_id: str) -> str | None:
    task = store.tasks.get(task_id)
    return task.tool_use_id if isinstance(task, RemoteAgentState) else None


def _get_parent_agent_id(store: AppStateStore, task_id: str) -> str | None:
    task = store.tasks.get(task_id)
    return task.parent_agent_id if isinstance(task, RemoteAgentState) else None


def _remote_event_context(
    task: RemoteAgentState,
    base: KernelEventContext,
) -> KernelEventContext:
    return KernelEventContext(
        session_id=base.session_id,
        runtime_session_id=base.runtime_session_id,
        agent_id=task.id,
        parent_agent_id=task.parent_agent_id,
        turn_id=base.turn_id,
        span_id=base.span_id,
        parent_span_id=base.parent_span_id,
        source="agent",
    )


def _build_notification_message(
    *,
    task_id: str,
    description: str,
    final_status: Literal["completed", "failed"],
    final_message: str,
    error: str | None,
) -> str:
    parts = [
        "<task_notification>",
        f"<task_id>{task_id}</task_id>",
        "<task_type>remote_agent</task_type>",
        f"<status>{final_status}</status>",
    ]
    summary = (
        f'Remote agent "{description}" completed'
        if final_status == "completed"
        else f'Remote agent "{description}" failed' + (f": {error}" if error else "")
    )
    parts.append(f"<summary>{summary}</summary>")
    if final_message:
        tag = "result" if final_status == "completed" else "partial_result"
        parts.append(f"<{tag}>{final_message}</{tag}>")
    parts.append("</task_notification>")
    return "\n".join(parts)


register_task_impl(RemoteAgentTask())


__all__ = [
    "RemoteAgentState",
    "RemoteAgentTask",
    "restore_remote_agents",
    "spawn_remote_agent",
]
