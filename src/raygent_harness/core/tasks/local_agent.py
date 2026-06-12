"""Local agent task implementation.

A local agent task runs a child QueryEngine with its own agent id, state, tool
context, and optional sidechain transcript. Terminal status is written before
cleanup, and the parent is notified through the task notification queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKResult,
    SDKUserMessage,
)
from raygent_harness.core.task import (
    TERMINAL_STATUSES,
    AgentRouteRecord,
    AppStateStore,
    TaskNotification,
    TaskStateBase,
    TaskType,
    generate_task_id,
    mark_notified_if_unset,
    register_task_impl,
)
from raygent_harness.core.tasks.local_bash import cleanup_shell_tasks_for_agent
from raygent_harness.core.tool import (
    ContentReplacementState,
    QueryTracking,
    ToolUseContext,
)
from raygent_harness.services.handoff import (
    HandoffClassificationRequest,
    classify_handoff_warning,
)
from raygent_harness.services.transcript import (
    TranscriptScope,
    get_agent_transcript,
)
from raygent_harness.services.worktree import WorktreeCleanupResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.core.tool import Tool
    from raygent_harness.services.transcript import SessionReplay
    from raygent_harness.services.worktree import WorktreeInfo, WorktreeManager


# ---------------------------------------------------------------------------
# LocalAgentState — per-task state record.
# ---------------------------------------------------------------------------


_FORK_AGENT_TYPE = "fork"
_FORK_QUERY_SOURCE = "agent:builtin:fork"


@dataclass
class LocalAgentPendingMessage:
    """Plain-text message queued for a running local_agent by SendMessage."""

    sender: str
    content: str
    summary: str | None = None


class LocalAgentRouteNotFoundError(RuntimeError):
    """Raised when SendMessage cannot resolve a local-agent route."""


class LocalAgentNotRunningError(RuntimeError):
    """Raised when a route resolves to a terminal active task."""


class LocalAgentResumeError(RuntimeError):
    """Raised when a stopped/evicted local agent cannot be resumed."""


@dataclass
class LocalAgentState(TaskStateBase):
    """State for a local_agent task. Extends TaskStateBase.

    Identity invariant: `id` IS the child agent_id. We intentionally do
    NOT carry a separate `agent_id` field; reusing `id` matches the

    `messages` is deliberately absent. Holding the child transcript in
    AppStateStore would re-create the memory pressure the reference
    (`getAgentTranscriptPath` + sidechain JSONL) avoids. When transcript
    persistence is enabled, durable child history is loaded through
    `services.transcript`, not from task state.
    """

    parent_agent_id: str | None = None
    """Which agent spawned this subagent. None = main thread. Used as
    the `agent_id` when enqueueing terminal notifications, so the
    parent's drain (filtered by its own `agent_id`) sees them."""

    prompt: str = ""
    """The initial user prompt sent to the child engine via
    `submit_message`."""

    agent_type: str | None = None
    """Selected AgentDefinition type when spawned through AgentTool."""

    name: str | None = None
    """Optional SendMessage route name. Ordinary named local agents keep
    reference-style latest-wins name mapping outside TeamCreate."""

    model: str | None = None
    """Effective child model used at spawn/resume time."""

    system_prompt: str | None = None
    """Effective child system prompt used for resume reconstruction."""

    tool_names: tuple[str, ...] = ()
    """Resolved child tool names. Resume uses this to select the same tools
    from the current parent catalog when possible."""

    permission_mode: str | None = None
    """Effective child permission mode at spawn/resume time."""

    cwd: str | None = None
    """Effective child working directory."""

    runtime_session_id: str | None = None
    """Runtime session id for the child QueryEngine sidechain."""

    pending_messages: tuple[LocalAgentPendingMessage, ...] = ()
    """Messages queued while the child is running. Drained at model-call
    boundaries and injected as plain user messages."""

    final_message: str | None = None
    """Final assistant text, populated on success. Empty / None on
    error or kill paths."""

    error: str | None = None
    """Error description on failure or kill paths."""

    is_error: bool = False
    """Mirrors `SDKResult.is_error`. Distinct from `status` because
    `failed`/`killed` both imply `is_error=True` but a successful
    completion can also surface a non-fatal warning later."""

    transcript_path: str | None = None
    """Optional durable sidechain transcript path. The task state stores
    only a pointer, never the full child message list."""

    worktree_path: str | None = None
    """Path of the isolated worktree while running or when cleanup kept it."""

    worktree_branch: str | None = None
    """Branch associated with `worktree_path`, when available."""

    worktree_slug: str | None = None
    """Safe worktree slug for resume/diagnostic metadata, when available."""

    worktree_created_at: float | None = None
    """Worktree creation timestamp if reported by the manager."""

    worktree_touched_at: float | None = None
    """Last observed worktree touch timestamp if reported by the manager."""

    worktree_cleanup_policy: str | None = None
    """Manager cleanup policy captured for resume/diagnostic metadata."""


# ---------------------------------------------------------------------------
# LocalAgentTask — the vtable impl.
# ---------------------------------------------------------------------------


@dataclass
class LocalAgentTask:
    """Task-vtable implementation for `type == "local_agent"`.

    Registered at module import. `kill(task_id, store)` is the ONLY
    cancellation path for v1 — no parent-abort linkage, no bulk
    cancel, no idle-timeout kill.

    """

    name: str = "local_agent"
    type: TaskType = "local_agent"

    async def kill(self, task_id: str, store: AppStateStore) -> None:
        """Idempotent cancel. Atomically flips status to `killed`,
        signals the child's abort event (cooperative shutdown for code
        watching `ctx.abort_event`), then cancels the driver task.

        does the status flip, abort, and cleanup-handle release inside
        `killAsyncAgent` itself — synchronously, not via the
        driver's CancelledError handler. Doing it here means that a
        caller observing `task.status == "killed"` immediately after
        `await impl.kill(...)` returns sees the right state, even if
        the driver is stuck on a slow / non-cooperative await. The
        driver's terminal path preserves already-terminal status so a
        race between kill and a clean completion still resolves to
        `killed`.

        Notification enqueue stays in the driver — the partial-output
        capture (final_message / error) lives there, and reference
        lifecycle catch block, not from `killAsyncAgent`.

        Idempotent via:
        1. The `if status in TERMINAL_STATUSES` guard at entry.
        2. `mark_notified_if_unset` in the driver's terminal step.
        """
        task = store.tasks.get(task_id)
        if not isinstance(task, LocalAgentState):
            return
        if task.status in TERMINAL_STATUSES:
            return

        # Atomic flip. Updater also defends against a concurrent
        # transition (e.g., driver completing simultaneously) by
        # re-checking status inside the updater.
        store.update_task(task_id, _kill_status_updater)

        # Signal cooperative abort. Code paths inside the child's
        # query loop that poll `ctx.abort_event.is_set()` will see
        # this and unwind without waiting for asyncio cancel
        # delivery. The driver's `except CancelledError` branch
        # still runs the cleanup + notification.
        abort_event = _ABORT_EVENTS.get(task_id)
        if abort_event is not None:
            abort_event.set()

        driver = _DRIVER_TASKS.get(task_id)
        if driver is not None and not driver.done():
            driver.cancel()


def _kill_status_updater(t: TaskStateBase) -> TaskStateBase:
    """Flip running tasks to killed while preserving terminal states."""
    if not isinstance(t, LocalAgentState):
        return t
    if t.status != "running":
        return t
    return replace(
        t,
        status="killed",
        end_time=time.time(),
        is_error=True,
        error="subagent killed via Task.kill",
    )


# ---------------------------------------------------------------------------
# Module-level registries.
# Driver objects and asyncio.Event instances are not serializable, so they live
# here instead of on LocalAgentState. Strong references also prevent asyncio
# from collecting live driver tasks mid-run.
# ---------------------------------------------------------------------------


_DRIVER_TASKS: dict[str, asyncio.Task[None]] = {}
_ABORT_EVENTS: dict[str, asyncio.Event] = {}
_ROUTE_PERSISTENCE_TASKS: dict[str, asyncio.Task[None]] = {}
# `_ABORT_EVENTS` maps task_id to the child context's abort event. It lets
# `kill()` signal cooperative abort without retaining the full ToolUseContext.


# ---------------------------------------------------------------------------
# spawn_local_agent — the public entry point.
# ---------------------------------------------------------------------------


def _silent_notify(_message: str) -> None:
    """Default subagent notify sink: suppress user-facing child text."""
    return


async def spawn_local_agent(
    *,
    prompt: str | MessageParam,
    parent_agent_id: str | None,
    parent_config: QueryConfig,
    parent_deps: QueryDeps,
    parent_ctx: ToolUseContext,
    description: str = "",
    tool_use_id: str | None = None,
    child_system_prompt: str | None = None,
    child_model: str | None = None,
    child_tools: Sequence[Tool] | None = None,
    child_permission_context: ToolPermissionContext | None = None,
    child_cwd: str | None = None,
    child_agent_type: str | None = None,
    name: str | None = None,
    task_id: str | None = None,
    parent_session_id: str | None = None,
    resume_replay: SessionReplay | None = None,
    initial_messages: Sequence[MessageParam] = (),
    display_prompt: str | None = None,
    child_query_source: str | None = None,
    worktree_info: WorktreeInfo | None = None,
    worktree_manager: WorktreeManager | None = None,
) -> str:
    """Spawn a subagent task. Returns the new task_id (== child agent_id).

    Wholesale-inherit parent's tools + system_prompt + model. Narrow:
    new agent_id, new session_id, fresh abort event, child-tagged
    QueryTracking (depth + 1), suppressed notify sink.

    The driver runs the child `QueryEngine.submit_message(prompt)` to
    completion, then transitions task status, runs cleanup
    (`kill_shell_tasks_for_agent`), and finally enqueues a
    `TaskNotification` addressed to `parent_agent_id`.

    Caller can either await the notification via
    `parent_deps.task_store.drain_notifications(parent_agent_id)` at
    the top of the next iteration (the agent loop's normal path) or
    `await run_until_done(...)` for synchronous flows.
    """
    task_id = task_id or generate_task_id("local_agent")
    store = parent_deps.task_store
    parent_session = parent_session_id or parent_config.session_id or parent_ctx.session_id
    runtime_session_id = f"sub-{task_id}-{uuid.uuid4().hex[:8]}"
    prompt_text = display_prompt or _prompt_display_text(prompt)

    # Build child config: inherit model/system_prompt/tools/sampling/etc.
    # Override identity (agent_id, session_id) and any per-conversation
    # state. Frozen dataclass — `replace()` produces a new instance.
    child_config = replace(
        parent_config,
        agent_id=task_id,
        session_id=runtime_session_id,
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
            session_id=parent_session,
            agent_id=task_id,
            is_sidechain=True,
            runtime_session_id=child_config.session_id,
        )
        with contextlib.suppress(Exception):
            transcript_path = parent_deps.transcript_store.path_for(transcript_scope)

    # Build child deps: share infrastructure, narrow notify.
    # `task_store` is shared so kill_shell_tasks_for_agent can find
    # child-spawned shells; `model_provider` is shared so embedding apps can
    # manage provider connection pools; `clock` is shared. `notify` is
    # suppressed so subagent
    # user-facing text doesn't leak.
    #
    # `stop_hooks` is shared from parent for v1 — hooks may be
    # subagent-relevant (e.g., budget enforcement) and we have no
    # filtering shape yet. Revisit if/when stop_hooks gain agent-scope
    # awareness.
    child_deps = replace(
        parent_deps,
        notify=_silent_notify,
        permission_context=(
            child_permission_context
            if child_permission_context is not None
            else parent_deps.permission_context
        ),
    )

    # Build child ctx: fresh abort event (no parent linkage), child
    # agent_id, child query tracking (depth + 1).
    child_query_tracking = (
        QueryTracking(
            chain_id=parent_ctx.query_tracking.chain_id,
            depth=parent_ctx.query_tracking.depth + 1,
        )
        if parent_ctx.query_tracking is not None
        else QueryTracking(chain_id=task_id, depth=1)
    )
    child_observability_context = (
        parent_ctx.observability_context.for_child_agent(task_id).with_source("agent")
        if parent_ctx.observability_context is not None
        else KernelEventContext(
            session_id=parent_config.session_id or parent_ctx.session_id,
            agent_id=task_id,
            parent_agent_id=parent_agent_id,
            source="agent",
        )
    )
    child_ctx = ToolUseContext(
        session_id=child_config.session_id,
        agent_id=task_id,
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
        query_source=child_query_source,
        observability_context=child_observability_context,
        add_notification=None,
        handle_elicitation=None,
    )

    # Register state BEFORE spawning the driver. With a pure-Python
    # driver there's no fallible spawn step — but if we created the
    # driver first and it raced to completion before the store had the
    # task, status/notification updates inside the driver would be
    # silent no-ops (`store.update_task` skips unknown ids). See
    # invariant 2 in module docstring.
    state = LocalAgentState(
        id=task_id,
        type="local_agent",
        description=description or _truncate_for_description(prompt_text),
        status="running",
        start_time=time.time(),
        tool_use_id=tool_use_id,
        parent_agent_id=parent_agent_id,
        prompt=prompt_text,
        agent_type=child_agent_type,
        name=_normalize_route_name(name),
        model=child_config.model,
        system_prompt=child_config.system_prompt,
        tool_names=tuple(tool.name for tool in child_config.tools),
        permission_mode=child_ctx.permission_context.mode,
        cwd=child_ctx.cwd,
        runtime_session_id=child_config.session_id,
        transcript_path=transcript_path,
        worktree_path=worktree_info.path if worktree_info is not None else None,
        worktree_branch=worktree_info.branch if worktree_info is not None else None,
        worktree_slug=worktree_info.slug if worktree_info is not None else None,
        worktree_created_at=(
            worktree_info.created_at if worktree_info is not None else None
        ),
        worktree_touched_at=(
            worktree_info.touched_at if worktree_info is not None else None
        ),
        worktree_cleanup_policy=(
            worktree_info.cleanup_policy if worktree_info is not None else None
        ),
    )
    store.register_task(state)
    _upsert_local_agent_route_record(
        store,
        state=state,
        parent_session_id=parent_session,
        deps=parent_deps,
        event_context=child_observability_context,
    )
    parent_deps.observability.emit(
        "agent.child.started",
        context=child_observability_context,
        data={
            "task_id": task_id,
            "parent_agent_id": parent_agent_id,
            "agent_type": child_agent_type,
            "tool_use_id": tool_use_id,
            "prompt_char_count": len(prompt_text),
            "model": child_config.model,
            "tool_count": len(child_config.tools),
            "worktree_attached": worktree_info is not None,
            "initial_message_count": len(initial_messages),
        },
    )
    if worktree_info is not None:
        parent_deps.observability.emit(
            "worktree.created",
            context=child_observability_context,
            data={
                "task_id": task_id,
                "path_char_count": len(worktree_info.path),
                "branch": worktree_info.branch,
                "head_commit": worktree_info.head_commit,
                "slug": worktree_info.slug,
                "git_root_char_count": (
                    len(worktree_info.git_root) if worktree_info.git_root else 0
                ),
                "hook_based": worktree_info.hook_based,
                "owner_task_id": worktree_info.owner_task_id,
                "cleanup_policy": worktree_info.cleanup_policy,
                "created_at_present": worktree_info.created_at is not None,
                "touched_at_present": worktree_info.touched_at is not None,
            },
        )

    # Register the abort event BEFORE creating the driver so kill()
    # has a stable handle even if the driver is cancelled before its
    # first scheduled step.
    _ABORT_EVENTS[task_id] = child_ctx.abort_event

    driver = asyncio.create_task(
        _drive(
            task_id=task_id,
            prompt=prompt,
            parent_agent_id=parent_agent_id,
            child_config=child_config,
            child_deps=child_deps,
            child_ctx=child_ctx,
            store=store,
            tool_use_id=tool_use_id,
            transcript_scope=transcript_scope,
            resume_replay=resume_replay,
            initial_messages=tuple(initial_messages),
            worktree_info=worktree_info,
            worktree_manager=worktree_manager,
            parent_session_id=parent_session,
        ),
        name=f"local-agent-driver:{task_id}",
    )
    _DRIVER_TASKS[task_id] = driver

    return task_id


async def run_until_done(
    task_id: str,
    store: AppStateStore,
) -> LocalAgentState:
    """Await until the subagent reaches a terminal status. Useful for
    synchronous tool-call patterns that don't go through the
    notification queue."""
    while True:
        task = store.tasks.get(task_id)
        if isinstance(task, LocalAgentState) and task.status in TERMINAL_STATUSES:
            return task
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# _drive — owns the subagent lifecycle.
# ---------------------------------------------------------------------------


async def _drive(
    *,
    task_id: str,
    prompt: str | MessageParam,
    parent_agent_id: str | None,
    child_config: QueryConfig,
    child_deps: QueryDeps,
    child_ctx: ToolUseContext,
    store: AppStateStore,
    tool_use_id: str | None,
    transcript_scope: TranscriptScope | None,
    resume_replay: SessionReplay | None,
    initial_messages: tuple[MessageParam, ...],
    worktree_info: WorktreeInfo | None,
    worktree_manager: WorktreeManager | None,
    parent_session_id: str,
) -> None:
    """Run the child engine to completion, then transition status,
    cleanup, and notify — in that order. See invariant 5 in module
    docstring for ordering rationale.
    """
    final_message: str = ""
    is_error: bool = False
    error: str | None = None
    final_status: Literal["completed", "failed", "killed"] = "failed"
    child_messages: list[MessageParam] = []

    try:
        if resume_replay is not None:
            engine = QueryEngine.from_replay(
                child_config,
                child_deps,
                child_ctx,
                resume_replay,
                transcript_scope=transcript_scope,
            )
        elif transcript_scope is None:
            engine = QueryEngine(child_config, child_deps, child_ctx)
        else:
            engine = QueryEngine(
                child_config,
                child_deps,
                child_ctx,
                transcript_scope=transcript_scope,
            )
        if initial_messages:
            await engine.seed_messages(initial_messages)
        last_result: SDKResult | None = None
        async for sdk_msg in engine.submit_message(prompt):
            if isinstance(sdk_msg, SDKAssistantMessage | SDKUserMessage):
                child_messages.append(sdk_msg.message)
            if isinstance(sdk_msg, SDKResult):
                last_result = sdk_msg

        if last_result is None:
            # Engine ended without a terminal result — shouldn't happen
            # per the QueryEngine contract, but handle defensively.
            final_status = "failed"
            is_error = True
            error = "child engine produced no terminal result"
        elif last_result.is_error:
            final_status = "failed"
            is_error = True
            error = (
                "; ".join(last_result.errors)
                if last_result.errors
                else f"subtype={last_result.subtype}"
            )
            final_message = last_result.result
        else:
            final_status = "completed"
            is_error = False
            final_message = last_result.result

    except asyncio.CancelledError:
        final_status = "killed"
        is_error = True
        error = "subagent killed via Task.kill"
        # Re-raising would let CancelledError propagate up to the
        # asyncio runtime and skip our cleanup + notification. We
        # deliberately swallow it so the kill path still produces a
        # terminal notification — matches reference at
        # branch unconditionally enqueues the notification.

    except Exception as exc:
        final_status = "failed"
        is_error = True
        error = f"{type(exc).__name__}: {exc}"

    # 1. Status FIRST. Awaiters of run_until_done unblock immediately;
    # other code that sees `status in TERMINAL_STATUSES` can proceed
    # without waiting on the slow cleanup below. Updater PRESERVES an
    # already-terminal status (e.g., kill() flipped to "killed" while
    # the engine was about to surface a clean result) — mirrors
    # reference's `if (task.status !== 'running') return task` at
    store.update_task(
        task_id,
        _build_terminal_updater(
            final_status=final_status,
            final_message=final_message,
            error=error,
            is_error=is_error,
        ),
    )

    # 2. Cleanup. Kill any local_bash tasks the subagent spawned.
    # leaking child-addressed notifications into an undrained queue:
    #   (a) Pre-mark child shells `notified=True` so their `_drive`
    #       terminal step skips `enqueue_notification` (gated by
    #       `mark_notified_if_unset`). Reference uses the same
    #       (`markAgentsNotified`).
    #   (b) After the cleanup kills, drain any notifications that
    #       were already enqueued before we marked — those would
    #       otherwise sit in the queue forever (the child's drain
    #       loop is gone). Drop the drained list; the parent doesn't
    #       want shell-completion noise from a dying child.
    # Tolerate failures so we always reach the notification step.
    with contextlib.suppress(Exception):
        await cleanup_shell_tasks_for_agent(task_id, store)

    worktree_result = await _cleanup_worktree_if_needed(
        info=worktree_info,
        manager=worktree_manager,
    )
    if worktree_result is not None:
        store.update_task(task_id, _build_worktree_cleanup_updater(worktree_result))
        _refresh_route_record_for_task(
            child_deps,
            store,
            task_id=task_id,
            parent_session_id=parent_session_id,
        )
        child_deps.observability.emit(
            "worktree.cleanup.completed",
            context=child_ctx.observability_context,
            data={
                "task_id": task_id,
                "kept": worktree_result.kept,
                "reason": worktree_result.reason,
                "path_char_count": (
                    len(worktree_result.path) if worktree_result.path else 0
                ),
                "branch": worktree_result.branch,
            },
        )

    handoff_warning = await _handoff_warning(
        child_deps=child_deps,
        store=store,
        task_id=task_id,
        final_status=final_status,
        final_message=final_message,
        error=error,
        messages=tuple(child_messages),
        tool_names=tuple(tool.name for tool in child_config.tools),
        permission_mode=child_ctx.permission_context.mode,
        total_tool_use_count=_count_tool_uses(child_messages),
    )
    child_deps.observability.emit(
        "agent.handoff.classified",
        context=child_ctx.observability_context,
        data={
            "task_id": task_id,
            "task_type": "local_agent",
            "final_status": final_status,
            "warning_emitted": bool(handoff_warning),
            "warning_char_count": len(handoff_warning) if handoff_warning else 0,
            "message_count": len(child_messages),
            "tool_use_count": _count_tool_uses(child_messages),
        },
    )
    if handoff_warning:
        final_message = f"{handoff_warning}\n\n{final_message}".strip()

    child_deps.observability.emit(
        "agent.child.completed" if final_status == "completed" else "agent.child.failed",
        context=child_ctx.observability_context,
        data={
            "task_id": task_id,
            "parent_agent_id": parent_agent_id,
            "final_status": final_status,
            "is_error": is_error,
            "final_message_char_count": len(final_message),
            "error_char_count": len(error) if error else 0,
            "tool_use_id": tool_use_id,
        },
    )

    # 3. Notification LAST. Atomic check-and-set on `notified` prevents
    # duplicate enqueues (e.g., if a future bulk-cancel path ever
    # marks notified before we get here).
    _DRIVER_TASKS.pop(task_id, None)
    _ABORT_EVENTS.pop(task_id, None)
    if mark_notified_if_unset(store, task_id):
        store.enqueue_notification(
            TaskNotification(
                task_id=task_id,
                message=_build_notification_message(
                    task_id=task_id,
                    description=_get_description(store, task_id),
                    final_status=final_status,
                    final_message=final_message,
                    error=error,
                    worktree_result=worktree_result,
                ),
                kind="completed" if final_status == "completed" else "error",
                tool_use_id=tool_use_id,
                # Match reference default; see LocalBashTask's terminal
                # notification site for rationale.
                priority="later",
                agent_id=parent_agent_id,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_terminal_updater(
    *,
    final_status: Literal["completed", "failed", "killed"],
    final_message: str,
    error: str | None,
    is_error: bool,
) -> Callable[[TaskStateBase], TaskStateBase]:
    """Driver-side terminal status updater.

    Preserves an already-terminal status: if `kill()` raced ahead and
    flipped status to "killed" while the engine was about to surface a
    clean result, we keep "killed". Final-message / error fields still
    get filled in so the notification body has partial-output context.

    'running') return task` in the completion path.
    """

    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, LocalAgentState):
            return t
        # Preserve already-terminal status; only fill in companion
        # fields (final_message / error / is_error / end_time) if
        # they aren't already set.
        if t.status in TERMINAL_STATUSES:
            return replace(
                t,
                end_time=t.end_time if t.end_time is not None else time.time(),
                final_message=t.final_message
                if t.final_message is not None
                else (final_message or None),
                error=t.error if t.error is not None else error,
                is_error=t.is_error or is_error,
            )
        return replace(
            t,
            status=final_status,
            end_time=time.time(),
            final_message=final_message or None,
            error=error,
            is_error=is_error,
        )

    return update


def _get_description(store: AppStateStore, task_id: str) -> str:
    task = store.tasks.get(task_id)
    if isinstance(task, LocalAgentState):
        return task.description
    return task_id


def _build_notification_message(
    *,
    task_id: str,
    description: str,
    final_status: Literal["completed", "failed", "killed"],
    final_message: str,
    error: str | None,
    worktree_result: WorktreeCleanupResult | None = None,
) -> str:
    """Build the model-facing terminal notification body."""
    parts: list[str] = [
        "<task_notification>",
        f"<task_id>{task_id}</task_id>",
        f"<status>{final_status}</status>",
    ]
    if final_status == "completed":
        summary = f'Subagent "{description}" completed'
        parts.append(f"<summary>{summary}</summary>")
        if final_message:
            parts.append(f"<result>{final_message}</result>")
    else:
        summary = (
            f'Subagent "{description}" {final_status}'
            + (f": {error}" if error else "")
        )
        parts.append(f"<summary>{summary}</summary>")
        if final_message:
            parts.append(f"<partial_result>{final_message}</partial_result>")
    if worktree_result is not None and worktree_result.path:
        parts.append("<worktree>")
        parts.append(f"<worktreePath>{worktree_result.path}</worktreePath>")
        if worktree_result.branch:
            parts.append(f"<worktreeBranch>{worktree_result.branch}</worktreeBranch>")
        parts.append("</worktree>")
    parts.append("</task_notification>")
    return "\n".join(parts)


async def _cleanup_worktree_if_needed(
    *,
    info: WorktreeInfo | None,
    manager: WorktreeManager | None,
) -> WorktreeCleanupResult | None:
    if info is None:
        return None
    if manager is None:
        return WorktreeCleanupResult(
            kept=True,
            reason="cleanup_failed",
            path=info.path,
            branch=info.branch,
        )
    try:
        return await manager.cleanup(info)
    except Exception:
        return WorktreeCleanupResult(
            kept=True,
            reason="cleanup_failed",
            path=info.path,
            branch=info.branch,
        )


def _build_worktree_cleanup_updater(
    result: WorktreeCleanupResult,
) -> Callable[[TaskStateBase], TaskStateBase]:
    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, LocalAgentState):
            return t
        return replace(
            t,
            cwd=result.path if result.path is not None else None,
            worktree_path=result.path,
            worktree_branch=result.branch,
            worktree_slug=t.worktree_slug if result.path is not None else None,
            worktree_created_at=t.worktree_created_at if result.path is not None else None,
            worktree_touched_at=t.worktree_touched_at if result.path is not None else None,
            worktree_cleanup_policy=(
                t.worktree_cleanup_policy if result.path is not None else None
            ),
        )

    return update


async def _handoff_warning(
    *,
    child_deps: QueryDeps,
    store: AppStateStore,
    task_id: str,
    final_status: Literal["completed", "failed", "killed"],
    final_message: str,
    error: str | None,
    messages: tuple[MessageParam, ...],
    tool_names: tuple[str, ...],
    permission_mode: str | None,
    total_tool_use_count: int,
) -> str | None:
    if final_status != "completed":
        return None
    task = store.tasks.get(task_id)
    if not isinstance(task, LocalAgentState):
        return None
    return await classify_handoff_warning(
        child_deps.handoff_classifier,
        HandoffClassificationRequest(
            task_id=task_id,
            task_type="local_agent",
            agent_type=task.agent_type,
            description=task.description,
            final_status=final_status,
            final_message=final_message,
            error=error,
            messages=messages,
            tool_names=tool_names,
            permission_mode=permission_mode,
            total_tool_use_count=total_tool_use_count,
        ),
        timeout_s=child_deps.handoff_classifier_timeout_s,
    )


def _count_tool_uses(messages: list[MessageParam]) -> int:
    count = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(
            1
            for block in content
            if block.get("type") == "tool_use"
        )
    return count


def _truncate_for_description(prompt: str, limit: int = 80) -> str:
    one_line = prompt.replace("\n", " ").strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1] + "…"


def _prompt_display_text(prompt: str | MessageParam) -> str:
    if isinstance(prompt, str):
        return prompt
    content = prompt.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
        result = block.get("content")
        if isinstance(result, str):
            parts.append(result)
    return "\n".join(parts)


def _clone_content_replacement(
    src: ContentReplacementState | None,
) -> ContentReplacementState | None:
    """Subagent gets its own ContentReplacementState reference so future
    per-agent oversize policy can diverge. For v1 the values are copied
    verbatim from the parent."""
    if src is None:
        return None
    return ContentReplacementState(
        max_result_size_chars=src.max_result_size_chars,
        replaced_outputs_dir=src.replaced_outputs_dir,
        replacements=dict(src.replacements),
        seen_ids=set(src.seen_ids),
    )


def find_local_agent_route(
    store: AppStateStore,
    target: str,
) -> tuple[LocalAgentState | None, AgentRouteRecord | None]:
    """Resolve a SendMessage target to local-agent task/route state.

    Resolution order matches the reference SendMessage path: name registry
    first, then raw local-agent id, then route metadata for evicted agents.
    TeamCreate teammates are intentionally ignored here; callers fall back to
    the teammate route if this returns nothing.
    """

    target_key = target.strip()
    if not target_key:
        return None, None
    task_id = store.agent_name_registry.get(target_key, target_key)
    task = store.tasks.get(task_id)
    if isinstance(task, LocalAgentState):
        return task, store.agent_route_records.get(task.id)
    record = store.agent_route_records.get(task_id)
    if record is None:
        return None, None
    if record.task_type != "local_agent":
        return None, None
    return None, record


def queue_message_to_local_agent(
    store: AppStateStore,
    *,
    target: str,
    message: str,
    sender: str = "team-lead",
    summary: str | None = None,
) -> str:
    task, record = find_local_agent_route(store, target)
    if task is None:
        name = target.strip()
        raise LocalAgentRouteNotFoundError(f'Local agent "{name}" was not found.')
    if task.status in TERMINAL_STATUSES:
        raise LocalAgentNotRunningError(
            f'Local agent "{task.name or record.name if record else target}" is {task.status}.'
        )
    pending = LocalAgentPendingMessage(
        sender=sender,
        content=message,
        summary=summary,
    )
    store.update_task(
        task.id,
        lambda t: replace(t, pending_messages=(*t.pending_messages, pending))
        if isinstance(t, LocalAgentState) and t.status not in TERMINAL_STATUSES
        else t,
    )
    return task.id


def drain_pending_messages_for_agent(
    store: AppStateStore,
    agent_id: str | None,
) -> tuple[LocalAgentPendingMessage, ...]:
    """Atomically drain queued SendMessage prompts for one running local agent."""

    if agent_id is None:
        return ()
    drained: list[LocalAgentPendingMessage] = []

    def update(t: TaskStateBase) -> TaskStateBase:
        if (
            not isinstance(t, LocalAgentState)
            or t.id != agent_id
            or not t.pending_messages
        ):
            return t
        drained.extend(t.pending_messages)
        return replace(t, pending_messages=())

    store.update_task(agent_id, update)
    return tuple(drained)


async def resume_local_agent_background(
    *,
    target: str,
    prompt: str,
    parent_agent_id: str | None,
    parent_config: QueryConfig,
    parent_deps: QueryDeps,
    parent_ctx: ToolUseContext,
    tool_use_id: str | None = None,
) -> str:
    """Resume a stopped/evicted local_agent from sidechain transcript.

    Reference `resumeAgentBackground(...)` reconstructs API-visible messages
    from the sidechain JSONL, re-registers the same agent id, and submits the
    routed prompt as the next user turn. Raygent mirrors that kernel behavior
    without product metadata, UI panes, or worktree freshness side effects.
    """

    task, record = find_local_agent_route(parent_deps.task_store, target)
    if task is not None and task.status not in TERMINAL_STATUSES:
        raise LocalAgentResumeError(
            f'Local agent "{target.strip()}" is still running; queue a message instead.'
        )
    if record is None and task is not None:
        record = _route_record_from_state(
            task,
            parent_session_id=parent_config.session_id or parent_ctx.session_id,
            route_registered_at=task.start_time,
        )
    if record is None:
        raise LocalAgentResumeError(f'Local agent "{target.strip()}" has no resume record.')
    if parent_deps.transcript_store is None:
        raise LocalAgentResumeError(
            f'Local agent "{target.strip()}" has no transcript store for resume.'
        )
    parent_session_id = (
        record.parent_session_id or parent_config.session_id or parent_ctx.session_id
    )
    replay = await get_agent_transcript(
        parent_deps.transcript_store,
        parent_session_id,
        record.agent_id,
    )
    if replay is None:
        raise LocalAgentResumeError(
            f'Local agent "{target.strip()}" has no sidechain transcript to resume.'
        )

    child_tools = _resolve_record_tools(record, parent_ctx.tools or parent_config.tools)
    child_permission_context = parent_ctx.permission_context
    if record.permission_mode:
        child_permission_context = replace(
            child_permission_context,
            mode=record.permission_mode,  # pyright: ignore[reportArgumentType]
        )

    resumed_id = await spawn_local_agent(
        prompt=prompt,
        parent_agent_id=parent_agent_id,
        parent_config=parent_config,
        parent_deps=parent_deps,
        parent_ctx=parent_ctx,
        description=record.description,
        tool_use_id=tool_use_id,
        child_system_prompt=record.system_prompt,
        child_model=record.model,
        child_tools=child_tools,
        child_permission_context=child_permission_context,
        child_cwd=_resume_cwd_from_record(record, parent_ctx.cwd),
        child_agent_type=record.agent_type,
        name=record.name,
        task_id=record.agent_id,
        parent_session_id=parent_session_id,
        resume_replay=replay,
        child_query_source=_query_source_for_route_record(record),
    )
    _preserve_resumed_worktree_route_metadata(
        parent_deps,
        parent_deps.task_store,
        task_id=resumed_id,
        record=record,
        parent_session_id=parent_session_id,
    )
    return resumed_id


def _upsert_local_agent_route_record(
    store: AppStateStore,
    *,
    state: LocalAgentState,
    parent_session_id: str,
    deps: QueryDeps,
    event_context: KernelEventContext | None = None,
) -> None:
    existing = store.agent_route_records.get(state.id)
    record = _route_record_from_state(
        state,
        parent_session_id=parent_session_id,
        route_registered_at=(
            existing.route_registered_at if existing is not None else state.start_time
        ),
    )
    store.agent_route_records[state.id] = record
    if state.name is not None:
        store.agent_name_registry[state.name] = state.id
    _schedule_route_record_persist(
        deps,
        record,
        event_context=event_context,
    )


def _route_record_from_state(
    state: LocalAgentState,
    *,
    parent_session_id: str,
    route_registered_at: float,
) -> AgentRouteRecord:
    return AgentRouteRecord(
        agent_id=state.id,
        task_id=state.id,
        task_type="local_agent",
        name=state.name,
        parent_agent_id=state.parent_agent_id,
        parent_session_id=parent_session_id,
        runtime_session_id=state.runtime_session_id,
        agent_type=state.agent_type,
        description=state.description,
        model=state.model,
        system_prompt=state.system_prompt,
        tool_names=state.tool_names,
        permission_mode=state.permission_mode,
        cwd=state.cwd,
        worktree_path=state.worktree_path,
        worktree_branch=state.worktree_branch,
        worktree_slug=state.worktree_slug,
        worktree_created_at=state.worktree_created_at,
        worktree_touched_at=state.worktree_touched_at,
        worktree_cleanup_policy=state.worktree_cleanup_policy,
        transcript_path=state.transcript_path,
        is_sidechain=True,
        content_replacement_replay=True,
        route_registered_at=route_registered_at,
    )


def _preserve_resumed_worktree_route_metadata(
    deps: QueryDeps,
    store: AppStateStore,
    *,
    task_id: str,
    record: AgentRouteRecord,
    parent_session_id: str,
) -> None:
    if record.worktree_path is None or not Path(record.worktree_path).exists():
        return

    def update(t: TaskStateBase) -> TaskStateBase:
        if not isinstance(t, LocalAgentState):
            return t
        return replace(
            t,
            worktree_path=t.worktree_path or record.worktree_path,
            worktree_branch=t.worktree_branch or record.worktree_branch,
            worktree_slug=t.worktree_slug or record.worktree_slug,
            worktree_created_at=t.worktree_created_at or record.worktree_created_at,
            worktree_touched_at=t.worktree_touched_at or record.worktree_touched_at,
            worktree_cleanup_policy=(
                t.worktree_cleanup_policy or record.worktree_cleanup_policy
            ),
        )

    store.update_task(task_id, update)
    task = store.tasks.get(task_id)
    if isinstance(task, LocalAgentState):
        _upsert_local_agent_route_record(
            store,
            state=task,
            parent_session_id=parent_session_id,
            deps=deps,
            event_context=_task_observability_context(
                task,
                parent_session_id=parent_session_id,
            ),
        )


def _refresh_route_record_for_task(
    deps: QueryDeps,
    store: AppStateStore,
    *,
    task_id: str,
    parent_session_id: str,
) -> None:
    task = store.tasks.get(task_id)
    if isinstance(task, LocalAgentState):
        _upsert_local_agent_route_record(
            store,
            state=task,
            parent_session_id=parent_session_id,
            deps=deps,
            event_context=_task_observability_context(
                task,
                parent_session_id=parent_session_id,
            ),
        )


def _schedule_route_record_persist(
    deps: QueryDeps,
    record: AgentRouteRecord,
    *,
    event_context: KernelEventContext | None,
) -> None:
    route_store = deps.agent_route_record_store
    if route_store is None:
        return
    context = event_context or KernelEventContext(
        session_id=record.parent_session_id,
        runtime_session_id=record.runtime_session_id,
        agent_id=record.agent_id,
        parent_agent_id=record.parent_agent_id,
        source="agent",
    )
    previous = _ROUTE_PERSISTENCE_TASKS.get(record.task_id)

    async def persist_after_previous() -> None:
        if previous is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await previous
        await _persist_route_record(deps, record, context=context)

    task = asyncio.create_task(
        persist_after_previous(),
        name=f"local-agent-route-record-save:{record.task_id}",
    )
    _ROUTE_PERSISTENCE_TASKS[record.task_id] = task

    def clear_if_current(done: asyncio.Task[None]) -> None:
        if _ROUTE_PERSISTENCE_TASKS.get(record.task_id) is done:
            _ROUTE_PERSISTENCE_TASKS.pop(record.task_id, None)

    task.add_done_callback(clear_if_current)


async def _persist_route_record(
    deps: QueryDeps,
    record: AgentRouteRecord,
    *,
    context: KernelEventContext,
) -> None:
    route_store = deps.agent_route_record_store
    if route_store is None:
        return
    try:
        await route_store.save(record)
    except Exception as exc:
        deps.observability.emit(
            "agent.route_record.persistence_failed",
            context=context,
            data={
                "task_id": record.task_id,
                "agent_id": record.agent_id,
                "error_type": type(exc).__name__,
                "name_present": record.name is not None,
            },
        )
        return
    deps.observability.emit(
        "agent.route_record.saved",
        context=context,
        data={
            "task_id": record.task_id,
            "agent_id": record.agent_id,
            "name_present": record.name is not None,
            "tool_count": len(record.tool_names),
            "worktree_path_present": record.worktree_path is not None,
            "transcript_path_present": record.transcript_path is not None,
        },
    )


def _task_observability_context(
    task: LocalAgentState,
    *,
    parent_session_id: str,
) -> KernelEventContext:
    return KernelEventContext(
        session_id=parent_session_id,
        runtime_session_id=task.runtime_session_id,
        agent_id=task.id,
        parent_agent_id=task.parent_agent_id,
        source="agent",
    )


def _resume_cwd_from_record(record: AgentRouteRecord, parent_cwd: str) -> str | None:
    if record.worktree_path is not None:
        return record.worktree_path if Path(record.worktree_path).exists() else parent_cwd
    if record.worktree_slug is not None or record.worktree_branch is not None:
        if record.cwd is not None and Path(record.cwd).exists():
            return record.cwd
        return parent_cwd
    return record.cwd


def _resolve_record_tools(
    record: AgentRouteRecord,
    tools: Sequence[Tool],
) -> tuple[Tool, ...]:
    if not record.tool_names:
        return tuple(tools)
    by_name = {tool.name: tool for tool in tools}
    resolved = tuple(
        tool
        for name in record.tool_names
        if (tool := by_name.get(name)) is not None
    )
    return resolved or tuple(tools)


def _query_source_for_route_record(record: AgentRouteRecord) -> str | None:
    # Keep fork resumes cache/guard equivalent to the initial fork launch.
    # This string intentionally matches `tools.agent_tool.FORK_QUERY_SOURCE`
    # without importing from `tools` and creating a core/tools cycle.
    if record.agent_type == _FORK_AGENT_TYPE:
        return _FORK_QUERY_SOURCE
    return None


def _normalize_route_name(name: str | None) -> str | None:
    if name is None:
        return None
    stripped = name.strip()
    return stripped or None


# ---------------------------------------------------------------------------
# Module-init: register with the Task registry.
# ---------------------------------------------------------------------------


register_task_impl(LocalAgentTask())


__all__ = [
    "LocalAgentNotRunningError",
    "LocalAgentPendingMessage",
    "LocalAgentResumeError",
    "LocalAgentRouteNotFoundError",
    "LocalAgentState",
    "LocalAgentTask",
    "drain_pending_messages_for_agent",
    "find_local_agent_route",
    "queue_message_to_local_agent",
    "resume_local_agent_background",
    "run_until_done",
    "spawn_local_agent",
]
