"""Headless SendMessage tool for addressable teammate sessions.


Raygent implements the kernel path only: direct and team broadcast delivery to
in-process teammate task queues. Cross-session bridge/UDS transport, structured
shutdown/plan messages, and UI rendering remain adapter/follow-up work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from raygent_harness.coordinator.team import TEAM_LEAD_NAME, TeamStateStore
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.permissions import PermissionAllowDecision, PermissionResult
from raygent_harness.core.tasks.in_process_teammate import (
    InProcessTeammateState,
    TeammateNotFoundError,
    TeammateNotRunningError,
    queue_message_to_teammate,
    running_teammates_for_team,
)
from raygent_harness.core.tasks.local_agent import (
    LocalAgentNotRunningError,
    LocalAgentResumeError,
    LocalAgentRouteNotFoundError,
    LocalAgentState,
    find_local_agent_route,
    queue_message_to_local_agent,
    resume_local_agent_background,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    ValidationOk,
    ValidationResult,
    build_tool,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.core.task import AppStateStore
    from raygent_harness.skills.models import SkillDefinition


SEND_MESSAGE_TOOL_NAME = "SendMessage"
SEND_MESSAGE_MAX_RESULT_SIZE_CHARS = 100_000


@dataclass(frozen=True, slots=True)
class _CoordinatorSendRecordIds:
    work_item_ids: tuple[str, ...] = ()
    blackboard_entry_id: str | None = None


class SendMessageInput(BaseModel):
    to: str = Field(description='Recipient teammate name, or "*" for broadcast.')
    message: str = Field(description="Plain text message content.")
    summary: str | None = Field(
        default=None,
        description="A 5-10 word summary shown as a routing preview.",
    )


def build_send_message_tool(
    *,
    team_store: TeamStateStore,
    task_store: AppStateStore,
) -> Tool:
    """Build a concrete SendMessage tool over the shared team/task stores."""

    async def validate_input(
        input_: BaseModel,
        _ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        if not parsed.to.strip():
            return ValidationError(message="to must not be empty")
        if "@" in parsed.to:
            return ValidationError(
                message='to must be a bare teammate name or "*" - one team per session'
            )
        if not parsed.message.strip():
            return ValidationError(message="message must not be empty")
        if not parsed.summary or not parsed.summary.strip():
            return ValidationError(message="summary is required when message is a string")
        if parsed.to != "*" and _has_local_agent_route(task_store, parsed.to):
            return ValidationOk()
        if team_store.current_team is None:
            return ValidationError(
                message="SendMessage requires an active TeamCreate context or local-agent target"
            )
        return ValidationOk()

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = _coerce_input(input_)
        team = team_store.current_team
        sender = _sender_name(ctx, task_store)
        if parsed.to != "*":
            local_agent_result = await _route_local_agent_message(
                parsed,
                ctx=ctx,
                sender=sender,
                task_store=task_store,
            )
            if local_agent_result is not None:
                yield local_agent_result
                return
        if team is None:
            yield ToolResult(
                content="SendMessage requires an active TeamCreate context or local-agent target",
                is_error=True,
            )
            return

        if parsed.to == "*":
            recipients: list[str] = []
            recipient_task_ids: list[str] = []
            recipient_agent_ids: list[str] = []
            for teammate in running_teammates_for_team(task_store, team.team_name):
                if teammate.identity is None or teammate.identity.name == sender:
                    continue
                queue_message_to_teammate(
                    task_store,
                    name=teammate.identity.name,
                    message=parsed.message,
                    sender=sender,
                    summary=parsed.summary,
                    team_name=team.team_name,
                )
                recipients.append(teammate.identity.name)
                recipient_task_ids.append(teammate.id)
                recipient_agent_ids.append(teammate.identity.agent_id)
            coordinator_ids = (
                _record_coordinator_send_message(
                    ctx=ctx,
                    sender=sender,
                    target="@team",
                    summary=parsed.summary,
                    message=parsed.message,
                    recipient_task_ids=recipient_task_ids,
                    recipient_agent_ids=recipient_agent_ids,
                    team_name=team.team_name,
                )
                if recipient_task_ids
                else _CoordinatorSendRecordIds()
            )
            text = (
                "No teammates to broadcast to."
                if not recipients
                else f"Message broadcast to {len(recipients)} teammate(s): {', '.join(recipients)}"
            )
            yield ToolResult(
                content=_message_result_content(
                    success=True,
                    message=text,
                    sender=sender,
                    target="@team",
                    summary=parsed.summary,
                    content=parsed.message,
                    recipients=recipients,
                    coordinator_work_item_ids=coordinator_ids.work_item_ids,
                    coordinator_blackboard_entry_id=coordinator_ids.blackboard_entry_id,
                )
            )
            return

        try:
            task_id = queue_message_to_teammate(
                task_store,
                name=parsed.to,
                message=parsed.message,
                sender=sender,
                summary=parsed.summary,
                team_name=team.team_name,
            )
        except (TeammateNotFoundError, TeammateNotRunningError) as exc:
            yield ToolResult(content=str(exc), is_error=True)
            return
        task = task_store.tasks.get(task_id)
        recipient_agent_id = (
            task.identity.agent_id
            if isinstance(task, InProcessTeammateState) and task.identity is not None
            else None
        )
        coordinator_ids = _record_coordinator_send_message(
            ctx=ctx,
            sender=sender,
            target=f"@{parsed.to}",
            summary=parsed.summary,
            message=parsed.message,
            recipient_task_ids=(task_id,),
            recipient_agent_ids=(recipient_agent_id,) if recipient_agent_id else (),
            team_name=team.team_name,
        )

        yield ToolResult(
            content=_message_result_content(
                success=True,
                message=(
                    f"Message queued for delivery to {parsed.to} at its next idle turn."
                ),
                sender=sender,
                target=f"@{parsed.to}",
                summary=parsed.summary,
                content=parsed.message,
                task_id=task_id,
                coordinator_work_item_ids=coordinator_ids.work_item_ids,
                coordinator_blackboard_entry_id=coordinator_ids.blackboard_entry_id,
            )
        )

    return build_tool(
        ToolSpec(
            name=SEND_MESSAGE_TOOL_NAME,
            description="Send a message to an addressable teammate.",
            search_hint="send messages to agent teammates",
            input_model=SendMessageInput,
            call=call,
            prompt=SEND_MESSAGE_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=False,
            is_read_only=False,
            is_destructive=False,
            is_open_world=False,
            should_defer=True,
            always_load=False,
            max_result_size_chars=SEND_MESSAGE_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Sending message to {_coerce_input(input_).to}"
            ),
        )
    )


def create_send_message_catalog_provider(
    *,
    team_store: TeamStateStore,
    task_store: AppStateStore,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Append SendMessage when team routing is enabled for the session."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != SEND_MESSAGE_TOOL_NAME)
        if (
            not enabled
            or (team_store.current_team is None and not _has_local_agent_route(task_store))
        ):
            return without_existing
        return (
            *without_existing,
            build_send_message_tool(team_store=team_store, task_store=task_store),
        )

    return provider


SEND_MESSAGE_PROMPT = """# SendMessage

Send a plain text message to another addressable agent.

Use this after launching a named Agent when you need to route follow-up work
without starting a new worker. Direct recipients can be active-TeamCreate
teammates, ordinary non-team named local agents, or raw local-agent task IDs.

`to="*"` broadcasts only to every running teammate in the active team; it does
not broadcast to ordinary non-team local agents.
"""


def _coerce_input(input_: BaseModel) -> SendMessageInput:
    if isinstance(input_, SendMessageInput):
        return input_
    return SendMessageInput.model_validate(input_.model_dump())


def _message_result_content(
    *,
    success: bool,
    message: str,
    sender: str,
    target: str,
    summary: str | None,
    content: str,
    task_id: str | None = None,
    recipients: Sequence[str] | None = None,
    coordinator_work_item_ids: Sequence[str] = (),
    coordinator_blackboard_entry_id: str | None = None,
) -> list[dict[str, Any]]:
    data: dict[str, Any] = {
        "type": "send_message",
        "success": success,
        "message": message,
        "routing": {
            "sender": sender,
            "target": target,
            "summary": summary,
            "content": content,
        },
    }
    if task_id is not None:
        data["task_id"] = task_id
    if recipients is not None:
        data["recipients"] = list(recipients)
    if coordinator_work_item_ids:
        data["coordinator_work_item_ids"] = list(coordinator_work_item_ids)
    if coordinator_blackboard_entry_id is not None:
        data["coordinator_blackboard_entry_id"] = coordinator_blackboard_entry_id
    return [{"type": "text", "text": message}, data]


def _record_coordinator_send_message(
    *,
    ctx: ToolUseContext,
    sender: str,
    target: str,
    summary: str | None,
    message: str,
    recipient_task_ids: Sequence[str],
    recipient_agent_ids: Sequence[str],
    team_name: str | None,
) -> _CoordinatorSendRecordIds:
    if ctx.runtime is None or ctx.runtime.deps.coordinator_runtime is None:
        return _CoordinatorSendRecordIds()
    try:
        result = ctx.runtime.deps.coordinator_runtime.record_send_message(
            sender=sender,
            target=target,
            summary=summary,
            message_chars=len(message),
            recipient_task_ids=tuple(recipient_task_ids),
            recipient_agent_ids=tuple(recipient_agent_ids),
            team_name=team_name,
        )
    except Exception as exc:
        _emit_coordinator_runtime_failure(
            ctx=ctx,
            operation="record_send_message",
            exc=exc,
        )
        return _CoordinatorSendRecordIds()
    return _send_record_ids(result)


async def _route_local_agent_message(
    parsed: SendMessageInput,
    *,
    ctx: ToolUseContext,
    sender: str,
    task_store: AppStateStore,
) -> ToolResult | None:
    task, record = find_local_agent_route(task_store, parsed.to)
    if task is None and record is None:
        return None

    target_label = f"@{parsed.to}"
    if task is not None and task.status == "running":
        try:
            task_id = queue_message_to_local_agent(
                task_store,
                target=parsed.to,
                message=parsed.message,
                sender=sender,
                summary=parsed.summary,
            )
        except (LocalAgentRouteNotFoundError, LocalAgentNotRunningError) as exc:
            return ToolResult(content=str(exc), is_error=True)
        coordinator_ids = _record_coordinator_send_message(
            ctx=ctx,
            sender=sender,
            target=target_label,
            summary=parsed.summary,
            message=parsed.message,
            recipient_task_ids=(task_id,),
            recipient_agent_ids=(task_id,),
            team_name=None,
        )
        return ToolResult(
            content=_message_result_content(
                success=True,
                message=(
                    f"Message queued for delivery to {parsed.to} at its next tool round."
                ),
                sender=sender,
                target=target_label,
                summary=parsed.summary,
                content=parsed.message,
                task_id=task_id,
                coordinator_work_item_ids=coordinator_ids.work_item_ids,
                coordinator_blackboard_entry_id=coordinator_ids.blackboard_entry_id,
            )
        )

    if ctx.runtime is None:
        return ToolResult(
            content=(
                f'Local agent "{parsed.to}" is stopped or evicted and cannot be '
                "resumed without tool runtime context."
            ),
            is_error=True,
        )
    try:
        task_id = await resume_local_agent_background(
            target=parsed.to,
            prompt=parsed.message,
            parent_agent_id=ctx.agent_id,
            parent_config=ctx.runtime.config,
            parent_deps=ctx.runtime.deps,
            parent_ctx=ctx,
            tool_use_id=ctx.tool_use_id,
        )
    except LocalAgentResumeError as exc:
        return ToolResult(content=str(exc), is_error=True)

    coordinator_ids = _record_coordinator_send_message(
        ctx=ctx,
        sender=sender,
        target=target_label,
        summary=parsed.summary,
        message=parsed.message,
        recipient_task_ids=(task_id,),
        recipient_agent_ids=(task_id,),
        team_name=None,
    )
    return ToolResult(
        content=_message_result_content(
            success=True,
            message=(
                f'Agent "{parsed.to}" was stopped or evicted; resumed it in '
                "the background with your message."
            ),
            sender=sender,
            target=target_label,
            summary=parsed.summary,
            content=parsed.message,
            task_id=task_id,
            coordinator_work_item_ids=coordinator_ids.work_item_ids,
            coordinator_blackboard_entry_id=coordinator_ids.blackboard_entry_id,
        )
    )


def _has_local_agent_route(task_store: AppStateStore, target: str | None = None) -> bool:
    if target is not None:
        task, record = find_local_agent_route(task_store, target)
        return task is not None or record is not None
    if any(
        isinstance(task, LocalAgentState)
        for task in task_store.tasks.values()
    ):
        return True
    return any(
        record.task_type == "local_agent"
        for record in task_store.agent_route_records.values()
    )


def _send_record_ids(result: object) -> _CoordinatorSendRecordIds:
    work_items = getattr(result, "work_items", ())
    work_item_ids = tuple(
        work_item_id
        for item in work_items
        if isinstance((work_item_id := getattr(item, "id", None)), str)
    )
    blackboard_entry = getattr(result, "blackboard_entry", None)
    blackboard_entry_id = getattr(blackboard_entry, "id", None)
    return _CoordinatorSendRecordIds(
        work_item_ids=work_item_ids,
        blackboard_entry_id=(
            blackboard_entry_id if isinstance(blackboard_entry_id, str) else None
        ),
    )


def _emit_coordinator_runtime_failure(
    *,
    ctx: ToolUseContext,
    operation: str,
    exc: Exception,
) -> None:
    if ctx.runtime is None:
        return
    event_context = (
        ctx.observability_context.with_source("coordinator")
        if ctx.observability_context is not None
        else KernelEventContext(
            session_id=ctx.session_id,
            agent_id=ctx.agent_id,
            source="coordinator",
        )
    )
    ctx.runtime.deps.observability.emit(
        "coordinator.runtime.integration_failed",
        context=event_context,
        data={
            "operation": operation,
            "error_type": type(exc).__name__,
        },
    )


def _sender_name(ctx: ToolUseContext, task_store: AppStateStore | None = None) -> str:
    if ctx.agent_id is None:
        return TEAM_LEAD_NAME
    store = task_store
    if store is None:
        if ctx.runtime is None:
            return "teammate"
        store = ctx.runtime.deps.task_store
    for task in store.tasks.values():
        if (
            isinstance(task, InProcessTeammateState)
            and task.identity is not None
            and task.identity.agent_id == ctx.agent_id
        ):
            return task.identity.name
    return "teammate"


__all__ = [
    "SEND_MESSAGE_MAX_RESULT_SIZE_CHARS",
    "SEND_MESSAGE_PROMPT",
    "SEND_MESSAGE_TOOL_NAME",
    "SendMessageInput",
    "build_send_message_tool",
    "create_send_message_catalog_provider",
]
