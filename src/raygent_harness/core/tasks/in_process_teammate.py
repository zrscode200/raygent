"""InProcessTeammateTask - addressable, long-lived teammate runtime.


This differs from `local_agent`: a local agent is a one-shot background task,
while an in-process teammate remains `status="running"` after a turn completes,
marks itself idle, and waits for another routed message.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.query_engine import QueryEngine, SDKResult
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
from raygent_harness.core.tasks.local_bash import cleanup_shell_tasks_for_agent
from raygent_harness.core.tool import ContentReplacementState, QueryTracking, ToolUseContext
from raygent_harness.services.transcript import TranscriptScope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from raygent_harness.coordinator.team import TeamStateStore
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.core.tool import Tool


@dataclass(frozen=True)
class TeammateIdentity:
    """Stable teammate identity stored in task state."""

    agent_id: str
    name: str
    team_name: str
    parent_session_id: str
    agent_type: str
    model: str


@dataclass(frozen=True)
class TeammateMessage:
    """Structured routed message consumed by an in-process teammate."""

    sender: str
    content: str
    summary: str | None = None


@dataclass
class InProcessTeammateState(TaskStateBase):
    """State for an addressable teammate session."""

    identity: TeammateIdentity | None = None
    parent_agent_id: str | None = None
    prompt: str = ""
    pending_messages: tuple[TeammateMessage, ...] = ()
    is_idle: bool = False
    shutdown_requested: bool = False
    final_message: str | None = None
    error: str | None = None
    is_error: bool = False
    transcript_path: str | None = None


@dataclass
class InProcessTeammateTask:
    """Task-vtable implementation for `type == "in_process_teammate"`."""

    name: str = "InProcessTeammateTask"
    type: TaskType = "in_process_teammate"

    async def kill(self, task_id: str, store: AppStateStore) -> None:
        task = store.tasks.get(task_id)
        if not isinstance(task, InProcessTeammateState):
            return
        if task.status in TERMINAL_STATUSES:
            return

        store.update_task(task_id, _kill_status_updater)
        abort_event = _ABORT_EVENTS.get(task_id)
        if abort_event is not None:
            abort_event.set()
        wake_event = _WAKE_EVENTS.get(task_id)
        if wake_event is not None:
            wake_event.set()
        driver = _DRIVER_TASKS.get(task_id)
        if driver is not None and not driver.done():
            driver.cancel()
        _cleanup_teammate_membership(task_id, store)


class TeammateNotFoundError(RuntimeError):
    """Raised when SendMessage cannot resolve a teammate name."""


class TeammateNotRunningError(RuntimeError):
    """Raised when SendMessage resolves a terminal teammate."""


_DRIVER_TASKS: dict[str, asyncio.Task[None]] = {}
_ABORT_EVENTS: dict[str, asyncio.Event] = {}
_WAKE_EVENTS: dict[str, asyncio.Event] = {}
_TEAM_STORES: dict[str, TeamStateStore] = {}


def _silent_notify(_message: str) -> None:
    return


async def spawn_in_process_teammate(
    *,
    name: str,
    team_name: str,
    prompt: str,
    parent_agent_id: str | None,
    parent_config: QueryConfig,
    parent_deps: QueryDeps,
    parent_ctx: ToolUseContext,
    description: str = "",
    agent_type: str | None = None,
    team_store: TeamStateStore | None = None,
    tool_use_id: str | None = None,
    child_system_prompt: str | None = None,
    child_model: str | None = None,
    child_tools: Sequence[Tool] | None = None,
    child_permission_context: ToolPermissionContext | None = None,
    child_cwd: str | None = None,
    task_id: str | None = None,
) -> str:
    """Spawn a persistent teammate and return its task id."""

    teammate_name = _sanitize_name(name)
    normalized_team = _sanitize_name(team_name)
    teammate_agent_id = _format_agent_id(teammate_name, normalized_team)
    task_id = task_id or generate_task_id("in_process_teammate")
    store = parent_deps.task_store

    child_config = replace(
        parent_config,
        agent_id=teammate_agent_id,
        session_id=f"teammate-{task_id}-{uuid.uuid4().hex[:8]}",
        system_prompt=(
            child_system_prompt
            if child_system_prompt is not None
            else parent_config.system_prompt
        ),
        model=child_model if child_model is not None else parent_config.model,
        tools=tuple(child_tools) if child_tools is not None else parent_config.tools,
        context_messages=(),
        context_system_prompt="",
    )

    transcript_scope: TranscriptScope | None = None
    transcript_path: str | None = None
    if parent_deps.transcript_store is not None:
        transcript_scope = TranscriptScope(
            session_id=parent_config.session_id or parent_ctx.session_id,
            agent_id=teammate_agent_id,
            is_sidechain=True,
            runtime_session_id=child_config.session_id,
        )
        with contextlib.suppress(Exception):
            transcript_path = parent_deps.transcript_store.path_for(transcript_scope)

    child_deps = replace(
        parent_deps,
        notify=_silent_notify,
        permission_context=(
            child_permission_context
            if child_permission_context is not None
            else parent_deps.permission_context
        ),
    )
    child_query_tracking = (
        QueryTracking(
            chain_id=parent_ctx.query_tracking.chain_id,
            depth=parent_ctx.query_tracking.depth + 1,
        )
        if parent_ctx.query_tracking is not None
        else QueryTracking(chain_id=task_id, depth=1)
    )
    child_observability_context = (
        parent_ctx.observability_context.for_child_agent(teammate_agent_id).with_source(
            "agent"
        )
        if parent_ctx.observability_context is not None
        else KernelEventContext(
            session_id=parent_config.session_id or parent_ctx.session_id,
            agent_id=teammate_agent_id,
            parent_agent_id=parent_agent_id,
            source="agent",
        )
    )
    child_ctx = ToolUseContext(
        session_id=child_config.session_id,
        agent_id=teammate_agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt=child_config.system_prompt,
        cwd=child_cwd if child_cwd is not None else parent_ctx.cwd,
        tools=tuple(child_tools) if child_tools is not None else parent_ctx.tools,
        permission_context=(
            child_permission_context
            if child_permission_context is not None
            else parent_ctx.permission_context
        ),
        content_replacement=_clone_content_replacement(parent_ctx.content_replacement),
        query_tracking=child_query_tracking,
        observability_context=child_observability_context,
        add_notification=None,
        handle_elicitation=None,
    )

    identity = TeammateIdentity(
        agent_id=teammate_agent_id,
        name=teammate_name,
        team_name=normalized_team,
        parent_session_id=parent_config.session_id or parent_ctx.session_id,
        agent_type=agent_type or description or teammate_name,
        model=child_config.model,
    )
    state = InProcessTeammateState(
        id=task_id,
        type="in_process_teammate",
        description=description or f"{teammate_name}: {_truncate_for_description(prompt)}",
        status="running",
        start_time=time.time(),
        tool_use_id=tool_use_id,
        identity=identity,
        parent_agent_id=parent_agent_id,
        prompt=prompt,
        transcript_path=transcript_path,
    )
    store.register_task(state)
    store.agent_name_registry[teammate_name] = task_id
    parent_deps.observability.emit(
        "agent.child.started",
        context=child_observability_context,
        data={
            "task_id": task_id,
            "agent_id": teammate_agent_id,
            "parent_agent_id": parent_agent_id,
            "agent_type": identity.agent_type,
            "team_name": normalized_team,
            "agent_name": teammate_name,
            "tool_use_id": tool_use_id,
            "prompt_char_count": len(prompt),
            "model": child_config.model,
            "persistent": True,
        },
    )

    _ABORT_EVENTS[task_id] = child_ctx.abort_event
    _WAKE_EVENTS[task_id] = asyncio.Event()
    if team_store is not None:
        _TEAM_STORES[task_id] = team_store
    driver = asyncio.create_task(
        _drive(
            task_id=task_id,
            initial_prompt=prompt,
            parent_agent_id=parent_agent_id,
            child_config=child_config,
            child_deps=child_deps,
            child_ctx=child_ctx,
            store=store,
            tool_use_id=tool_use_id,
            transcript_scope=transcript_scope,
        ),
        name=f"in-process-teammate-driver:{task_id}",
    )
    _DRIVER_TASKS[task_id] = driver
    return task_id


def find_teammate_task_by_name(
    store: AppStateStore,
    name: str,
    *,
    team_name: str | None = None,
) -> InProcessTeammateState | None:
    normalized_name = _sanitize_name(name)
    registered = store.agent_name_registry.get(normalized_name)
    if registered is not None:
        task = store.tasks.get(registered)
        if isinstance(task, InProcessTeammateState) and _matches_teammate(
            task, normalized_name, team_name
        ):
            return task

    fallback: InProcessTeammateState | None = None
    for task in store.tasks.values():
        if not _matches_teammate(task, normalized_name, team_name):
            continue
        assert isinstance(task, InProcessTeammateState)
        if task.status == "running":
            return task
        if fallback is None:
            fallback = task
    return fallback


def running_teammates_for_team(
    store: AppStateStore,
    team_name: str,
) -> tuple[InProcessTeammateState, ...]:
    normalized_team = _sanitize_name(team_name)
    return tuple(
        task
        for task in store.tasks.values()
        if isinstance(task, InProcessTeammateState)
        and task.identity is not None
        and task.identity.team_name == normalized_team
        and task.status == "running"
    )


def queue_message_to_teammate(
    store: AppStateStore,
    *,
    name: str,
    message: str,
    sender: str = "team-lead",
    summary: str | None = None,
    team_name: str | None = None,
) -> str:
    task = find_teammate_task_by_name(store, name, team_name=team_name)
    if task is None:
        raise TeammateNotFoundError(f'Teammate "{_sanitize_name(name)}" was not found.')
    if task.status in TERMINAL_STATUSES:
        raise TeammateNotRunningError(
            f'Teammate "{task.identity.name if task.identity else name}" is {task.status}.'
        )
    queue_pending_message(
        task.id,
        TeammateMessage(sender=sender, content=message, summary=summary),
        store,
    )
    return task.id


def queue_pending_message(
    task_id: str,
    message: str | TeammateMessage,
    store: AppStateStore,
    *,
    sender: str = "user",
    summary: str | None = None,
) -> bool:
    task = store.tasks.get(task_id)
    if not isinstance(task, InProcessTeammateState):
        return False
    if task.status in TERMINAL_STATUSES:
        return False
    pending = (
        message
        if isinstance(message, TeammateMessage)
        else TeammateMessage(sender=sender, content=message, summary=summary)
    )

    store.update_task(
        task_id,
        lambda t: replace(t, pending_messages=(*t.pending_messages, pending))
        if isinstance(t, InProcessTeammateState) and t.status not in TERMINAL_STATUSES
        else t,
    )
    wake_event = _WAKE_EVENTS.get(task_id)
    if wake_event is not None:
        wake_event.set()
    return True


async def _drive(
    *,
    task_id: str,
    initial_prompt: str,
    parent_agent_id: str | None,
    child_config: QueryConfig,
    child_deps: QueryDeps,
    child_ctx: ToolUseContext,
    store: AppStateStore,
    tool_use_id: str | None,
    transcript_scope: TranscriptScope | None,
) -> None:
    final_status: Literal["completed", "failed", "killed"] = "completed"
    final_message = ""
    error: str | None = None
    is_error = False
    current_prompt = initial_prompt

    try:
        engine = (
            QueryEngine(child_config, child_deps, child_ctx)
            if transcript_scope is None
            else QueryEngine(
                child_config,
                child_deps,
                child_ctx,
                transcript_scope=transcript_scope,
            )
        )
        while True:
            store.update_task(task_id, _activity_updater(is_idle=False))
            last_result: SDKResult | None = None
            async for sdk_msg in engine.submit_message(current_prompt):
                if isinstance(sdk_msg, SDKResult):
                    last_result = sdk_msg

            if last_result is None:
                final_status = "failed"
                is_error = True
                error = "teammate engine produced no terminal result"
                break
            final_message = last_result.result
            if last_result.is_error:
                final_status = "failed"
                is_error = True
                error = (
                    "; ".join(last_result.errors)
                    if last_result.errors
                    else f"subtype={last_result.subtype}"
                )
                break

            store.update_task(
                task_id,
                _turn_success_updater(final_message=final_message),
            )
            store.emit_task_progress(
                task_id,
                {
                    "progress_type": "teammate_idle",
                    "final_message_char_count": len(final_message),
                },
            )
            _enqueue_idle_notification(
                store,
                task_id=task_id,
                parent_agent_id=parent_agent_id,
                tool_use_id=tool_use_id,
                final_message=final_message,
            )
            next_message = await _wait_for_next_message(task_id, store)
            if next_message is None:
                final_status = "killed"
                is_error = True
                error = "teammate killed while idle"
                break
            current_prompt = _format_routed_message(next_message)

    except asyncio.CancelledError:
        final_status = "killed"
        is_error = True
        error = "teammate killed via Task.kill"
    except Exception as exc:
        final_status = "failed"
        is_error = True
        error = f"{type(exc).__name__}: {exc}"

    store.update_task(
        task_id,
        _terminal_updater(
            final_status=final_status,
            final_message=final_message,
            error=error,
            is_error=is_error,
        ),
    )
    with contextlib.suppress(Exception):
        await cleanup_shell_tasks_for_agent(child_ctx.agent_id or task_id, store)
    _cleanup_teammate_membership(task_id, store)

    _DRIVER_TASKS.pop(task_id, None)
    _ABORT_EVENTS.pop(task_id, None)
    _WAKE_EVENTS.pop(task_id, None)
    _TEAM_STORES.pop(task_id, None)
    child_deps.observability.emit(
        "agent.child.failed" if is_error else "agent.child.completed",
        context=child_ctx.observability_context,
        data={
            "task_id": task_id,
            "agent_id": child_ctx.agent_id,
            "parent_agent_id": parent_agent_id,
            "final_status": final_status,
            "is_error": is_error,
            "final_message_char_count": len(final_message),
            "error_char_count": len(error) if error else 0,
            "tool_use_id": tool_use_id,
            "persistent": True,
        },
    )
    if mark_notified_if_unset(store, task_id):
        store.enqueue_notification(
            TaskNotification(
                task_id=task_id,
                message=_build_notification_message(
                    store,
                    task_id=task_id,
                    status=final_status,
                    final_message=final_message,
                    error=error,
                ),
                kind="error",
                tool_use_id=tool_use_id,
                priority="later",
                agent_id=parent_agent_id,
            )
        )


async def _wait_for_next_message(
    task_id: str,
    store: AppStateStore,
) -> TeammateMessage | None:
    abort_event = _ABORT_EVENTS[task_id]
    wake_event = _WAKE_EVENTS[task_id]
    while not abort_event.is_set():
        wake_event.clear()
        message = _pop_next_pending_message(task_id, store)
        if message is not None:
            return message
        abort_wait = asyncio.create_task(abort_event.wait())
        wake_wait = asyncio.create_task(wake_event.wait())
        done, pending = await asyncio.wait(
            {abort_wait, wake_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    return None


def _pop_next_pending_message(
    task_id: str,
    store: AppStateStore,
) -> TeammateMessage | None:
    holder: list[TeammateMessage] = []

    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, InProcessTeammateState) or not t.pending_messages:
            return t
        holder.append(t.pending_messages[0])
        return replace(
            t,
            pending_messages=t.pending_messages[1:],
            is_idle=False,
        )

    store.update_task(task_id, update)
    return holder[0] if holder else None


def _kill_status_updater(t: TaskStateBase) -> TaskStateBase:
    if not isinstance(t, InProcessTeammateState) or t.status != "running":
        return t
    return replace(
        t,
        status="killed",
        end_time=time.time(),
        is_idle=True,
        is_error=True,
        error="teammate killed via Task.kill",
    )


def _cleanup_teammate_membership(task_id: str, store: AppStateStore) -> None:
    task = store.tasks.get(task_id)
    if not isinstance(task, InProcessTeammateState) or task.identity is None:
        return
    identity = task.identity
    if store.agent_name_registry.get(identity.name) == task_id:
        store.agent_name_registry.pop(identity.name, None)
    team_store = _TEAM_STORES.get(task_id)
    if team_store is not None:
        with contextlib.suppress(Exception):
            team_store.remove_member_by_agent_id(identity.agent_id)


def _activity_updater(*, is_idle: bool) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, InProcessTeammateState) or t.status != "running":
            return t
        return replace(t, is_idle=is_idle)

    return update


def _turn_success_updater(*, final_message: str) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, InProcessTeammateState) or t.status != "running":
            return t
        return replace(
            t,
            is_idle=True,
            final_message=final_message or None,
            error=None,
            is_error=False,
        )

    return update


def _terminal_updater(
    *,
    final_status: Literal["completed", "failed", "killed"],
    final_message: str,
    error: str | None,
    is_error: bool,
) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, InProcessTeammateState):
            return t
        if t.status in TERMINAL_STATUSES:
            return replace(
                t,
                end_time=t.end_time if t.end_time is not None else time.time(),
                final_message=t.final_message
                if t.final_message is not None
                else (final_message or None),
                error=t.error if t.error is not None else error,
                is_idle=True,
                is_error=t.is_error or is_error,
                pending_messages=(),
            )
        return replace(
            t,
            status=final_status,
            end_time=time.time(),
            final_message=final_message or None,
            error=error,
            is_error=is_error,
            is_idle=True,
            pending_messages=(),
        )

    return update


def _enqueue_idle_notification(
    store: AppStateStore,
    *,
    task_id: str,
    parent_agent_id: str | None,
    tool_use_id: str | None,
    final_message: str,
) -> None:
    task = store.tasks.get(task_id)
    if not isinstance(task, InProcessTeammateState) or task.identity is None:
        return
    store.enqueue_notification(
        TaskNotification(
            task_id=task_id,
            message=_build_notification_message(
                store,
                task_id=task_id,
                status="idle",
                final_message=final_message,
                error=None,
            ),
            kind="completed",
            tool_use_id=tool_use_id,
            priority="later",
            agent_id=parent_agent_id,
        )
    )


def _build_notification_message(
    store: AppStateStore,
    *,
    task_id: str,
    status: Literal["idle", "completed", "failed", "killed"],
    final_message: str,
    error: str | None,
) -> str:
    task = store.tasks.get(task_id)
    identity = task.identity if isinstance(task, InProcessTeammateState) else None
    parts = [
        "<teammate_notification>",
        f"<task_id>{task_id}</task_id>",
        f"<status>{status}</status>",
    ]
    if identity is not None:
        parts.extend(
            [
                f"<agent_id>{identity.agent_id}</agent_id>",
                f"<name>{identity.name}</name>",
                f"<team_name>{identity.team_name}</team_name>",
            ]
        )
    summary_name = identity.name if identity is not None else task_id
    if status == "idle":
        parts.append(f"<summary>Teammate {summary_name} is idle and available.</summary>")
    else:
        parts.append(f"<summary>Teammate {summary_name} {status}.</summary>")
    if final_message:
        parts.append(f"<result>{final_message}</result>")
    if error:
        parts.append(f"<error>{error}</error>")
    parts.append("</teammate_notification>")
    return "\n".join(parts)


def _format_routed_message(message: TeammateMessage) -> str:
    if message.sender == "user":
        return message.content
    summary_attr = f' summary="{message.summary}"' if message.summary else ""
    return (
        f'<teammate-message teammate_id="{message.sender}"{summary_attr}>\n'
        f"{message.content}\n"
        "</teammate-message>"
    )


def _matches_teammate(
    task: TaskStateBase | None,
    name: str,
    team_name: str | None,
) -> bool:
    if not isinstance(task, InProcessTeammateState) or task.identity is None:
        return False
    if task.identity.name != _sanitize_name(name):
        return False
    return not (team_name is not None and task.identity.team_name != _sanitize_name(team_name))


def _sanitize_name(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "team"


def _format_agent_id(agent_name: str, team_name: str) -> str:
    return f"{_sanitize_name(agent_name)}@{_sanitize_name(team_name)}"


def _truncate_for_description(prompt: str, limit: int = 50) -> str:
    one_line = prompt.replace("\n", " ").strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def _clone_content_replacement(
    src: ContentReplacementState | None,
) -> ContentReplacementState | None:
    if src is None:
        return None
    return ContentReplacementState(
        max_result_size_chars=src.max_result_size_chars,
        replaced_outputs_dir=src.replaced_outputs_dir,
        replacements=dict(src.replacements),
        seen_ids=set(src.seen_ids),
    )


register_task_impl(InProcessTeammateTask())


__all__ = [
    "InProcessTeammateState",
    "InProcessTeammateTask",
    "TeammateIdentity",
    "TeammateMessage",
    "TeammateNotFoundError",
    "TeammateNotRunningError",
    "find_teammate_task_by_name",
    "queue_message_to_teammate",
    "queue_pending_message",
    "running_teammates_for_team",
    "spawn_in_process_teammate",
]
