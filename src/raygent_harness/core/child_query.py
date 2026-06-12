"""Synchronous child QueryEngine runner.

Child-agent loops need that are not background tasks: forked skills
and foreground/forked AgentTool calls must return a same-turn tool result, not
enqueue a task notification. This module provides that shared headless runner
while reusing QueryEngine for the actual loop semantics.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.file_state import clone_read_file_state_cache
from raygent_harness.core.messages import MessageParam
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKResult,
    SDKUserMessage,
)
from raygent_harness.core.tasks.local_bash import cleanup_shell_tasks_for_agent
from raygent_harness.core.tool import (
    ContentReplacementState,
    QueryTracking,
    ToolUseContext,
)
from raygent_harness.services.transcript import TranscriptScope

if TYPE_CHECKING:
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.core.tool import Tool


@dataclass(frozen=True)
class ChildQueryRequest:
    """Inputs for a synchronous child query loop."""

    prompt_messages: tuple[MessageParam, ...]
    parent_config: QueryConfig
    parent_deps: QueryDeps
    parent_ctx: ToolUseContext
    agent_id: str
    agent_type: str | None = None
    system_prompt: str | None = None
    model: str | None = None
    model_override: str | None = None
    effort: str | int | None = None
    tools: tuple[Tool, ...] | None = None
    permission_context: ToolPermissionContext | None = None
    cwd: str | None = None
    query_source: str | None = None
    transcript_label: str | None = None
    link_abort_to_parent: bool = True
    max_turns: int | None = None
    transcript_enabled: bool = True


@dataclass(frozen=True)
class ChildQueryResult:
    """Terminal result of a synchronous child query loop."""

    agent_id: str
    final_message: str
    is_error: bool
    errors: tuple[str, ...] = ()
    subtype: str = "success"
    transcript_path: str | None = None
    messages_seen: int = 0
    messages: tuple[MessageParam, ...] = ()


async def run_child_query(request: ChildQueryRequest) -> ChildQueryResult:
    """Run a child QueryEngine synchronously and return its terminal result.

    Unlike `spawn_local_agent`, this does not register a task and does not
    enqueue notifications. It does record an optional sidechain transcript and
    cleans up child-owned bash tasks on every terminal path.
    """

    if not request.prompt_messages:
        raise ValueError("ChildQueryRequest.prompt_messages cannot be empty")

    child_config = _build_child_config(request)
    transcript_scope, transcript_path = _build_transcript_scope(request, child_config)
    child_permission_context = (
        request.permission_context
        if request.permission_context is not None
        else request.parent_deps.permission_context_for(request.parent_ctx)
    )
    child_deps = replace(
        request.parent_deps,
        notify=_silent_notify,
        permission_context=child_permission_context,
    )
    child_ctx = _build_child_context(request, child_config)
    engine = QueryEngine(
        child_config,
        child_deps,
        child_ctx,
        transcript_scope=transcript_scope,
    )

    last_result: SDKResult | None = None
    child_messages: list[MessageParam] = []
    try:
        initial_messages = request.prompt_messages[:-1]
        prompt = request.prompt_messages[-1]
        if initial_messages:
            await engine.seed_messages(initial_messages)
        async for sdk_msg in engine.submit_message(prompt):
            if isinstance(sdk_msg, SDKAssistantMessage | SDKUserMessage):
                child_messages.append(sdk_msg.message)
            if isinstance(sdk_msg, SDKResult):
                last_result = sdk_msg
    finally:
        with contextlib.suppress(Exception):
            await cleanup_shell_tasks_for_agent(request.agent_id, request.parent_deps.task_store)

    if last_result is None:
        return ChildQueryResult(
            agent_id=request.agent_id,
            final_message="",
            is_error=True,
            errors=("child query produced no terminal result",),
            subtype="error_during_execution",
            transcript_path=transcript_path,
            messages_seen=len(request.prompt_messages),
            messages=tuple(child_messages),
        )

    return ChildQueryResult(
        agent_id=request.agent_id,
        final_message=last_result.result,
        is_error=last_result.is_error,
        errors=last_result.errors,
        subtype=last_result.subtype,
        transcript_path=transcript_path,
        messages_seen=len(request.prompt_messages),
        messages=tuple(child_messages),
    )


def _build_child_config(request: ChildQueryRequest) -> QueryConfig:
    budget = (
        request.parent_config.budget
        if request.max_turns is None
        else replace(request.parent_config.budget, max_turns=request.max_turns)
    )
    return replace(
        request.parent_config,
        agent_id=request.agent_id,
        session_id=f"child-{request.agent_id}-{uuid.uuid4().hex[:8]}",
        system_prompt=(
            request.system_prompt
            if request.system_prompt is not None
            else request.parent_config.system_prompt
        ),
        model=request.model if request.model is not None else request.parent_config.model,
        tools=request.tools if request.tools is not None else request.parent_config.tools,
        budget=budget,
        context_messages=(),
        context_system_prompt="",
        experiments=dict(request.parent_config.experiments),
    )


def _build_transcript_scope(
    request: ChildQueryRequest,
    child_config: QueryConfig,
) -> tuple[TranscriptScope | None, str | None]:
    if not request.transcript_enabled:
        return None, None

    store = request.parent_deps.transcript_store
    if store is None:
        return None, None

    scope = TranscriptScope(
        session_id=request.parent_config.session_id or request.parent_ctx.session_id,
        agent_id=request.agent_id,
        is_sidechain=True,
        runtime_session_id=child_config.session_id,
    )
    transcript_path: str | None = None
    with contextlib.suppress(Exception):
        transcript_path = store.path_for(scope)
    return scope, transcript_path


def _build_child_context(
    request: ChildQueryRequest,
    child_config: QueryConfig,
) -> ToolUseContext:
    parent_ctx = request.parent_ctx
    abort_event = parent_ctx.abort_event if request.link_abort_to_parent else asyncio.Event()
    child_query_tracking = (
        QueryTracking(
            chain_id=parent_ctx.query_tracking.chain_id,
            depth=parent_ctx.query_tracking.depth + 1,
        )
        if parent_ctx.query_tracking is not None
        else QueryTracking(chain_id=request.agent_id, depth=1)
    )
    tools = request.tools if request.tools is not None else parent_ctx.tools
    permission_context = (
        request.permission_context
        if request.permission_context is not None
        else request.parent_deps.permission_context_for(parent_ctx)
    )
    return ToolUseContext(
        session_id=child_config.session_id,
        agent_id=request.agent_id,
        abort_event=abort_event,
        rendered_system_prompt=child_config.system_prompt,
        cwd=request.cwd if request.cwd is not None else parent_ctx.cwd,
        tools=tools,
        permission_context=permission_context,
        discovered_tool_names=frozenset(parent_ctx.discovered_tool_names),
        content_replacement=_clone_content_replacement(parent_ctx.content_replacement),
        model_override=request.model_override,
        reasoning_effort_override=request.effort,
        query_tracking=child_query_tracking,
        query_source=request.query_source,
        tool_use_id=parent_ctx.tool_use_id,
        add_notification=None,
        handle_elicitation=None,
        read_file_state=clone_read_file_state_cache(parent_ctx.read_file_state),
    )


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


def _silent_notify(_message: str) -> None:
    return


__all__ = [
    "ChildQueryRequest",
    "ChildQueryResult",
    "run_child_query",
]
