"""Model-callable task stop wrapper.


Raygent already had the core stop wrapper for programmatic callers. This tool
exposes the same policy to the model so coordinator prompts do not advertise a
nonexistent stop surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from raygent_harness.core.observability import KernelEventContext
from raygent_harness.core.permissions import PermissionAllowDecision, PermissionResult
from raygent_harness.core.task import StopTaskError
from raygent_harness.core.tasks.stop_task import stop_task
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
    from raygent_harness.core.deps import QueryDeps, ToolCatalogProvider
    from raygent_harness.core.permissions import ToolPermissionContext
    from raygent_harness.skills.models import SkillDefinition


TASK_STOP_TOOL_NAME = "TaskStop"
TASK_STOP_MAX_RESULT_SIZE_CHARS = 100_000


class TaskStopInput(BaseModel):
    task_id: str | None = Field(
        default=None,
        description="ID of the running background task to stop.",
    )
    shell_id: str | None = Field(
        default=None,
        description="Deprecated: use task_id instead. Kept for KillShell compatibility.",
    )


def build_task_stop_tool(*, deps: QueryDeps) -> Tool:
    """Build a concrete TaskStop tool over `deps.task_store`."""

    async def validate_input(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        parsed = _coerce_input(input_)
        task_id = _task_id_from_input(parsed)
        if task_id is None:
            return ValidationError(message="task_id is required for TaskStop")
        if ctx.agent_id is not None:
            return ValidationError(
                message="TaskStop is only available to the main coordinator."
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
        task_id = _task_id_from_input(parsed)
        if task_id is None:
            yield ToolResult(
                content="TaskStop failed (missing_input): task_id is required",
                is_error=True,
            )
            return
        if ctx.agent_id is not None:
            yield ToolResult(
                content=(
                    "TaskStop failed (not_allowed): TaskStop is only available "
                    "to the main coordinator."
                ),
                is_error=True,
            )
            return
        try:
            result = await stop_task(task_id, deps.task_store)
        except StopTaskError as exc:
            yield ToolResult(
                content=f"TaskStop failed ({exc.code}): {exc}",
                is_error=True,
            )
            return
        coordinator_ids = _record_coordinator_task_stop(
            deps=deps,
            ctx=ctx,
            task_id=result.task_id,
            task_type=result.task_type,
            description=result.description,
        )

        yield ToolResult(
            content=[
                {
                    "type": "text",
                    "text": f"Stopped task {result.task_id} ({result.task_type}).",
                },
                {
                    "type": "task_stopped",
                    "task_id": result.task_id,
                    "task_type": result.task_type,
                    "description": result.description,
                    **coordinator_ids,
                },
            ]
        )

    return build_tool(
        ToolSpec(
            name=TASK_STOP_TOOL_NAME,
            aliases=("KillShell",),
            description="Stop a running background task.",
            search_hint="stop a background worker or task",
            input_model=TaskStopInput,
            call=call,
            prompt=TASK_STOP_PROMPT,
            validate_input=validate_input,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=False,
            is_destructive=True,
            is_open_world=False,
            should_defer=True,
            always_load=False,
            max_result_size_chars=TASK_STOP_MAX_RESULT_SIZE_CHARS,
            get_activity_description=lambda input_: (
                f"Stopping task {_task_id_from_input(_coerce_input(input_)) or 'unknown'}"
            ),
        )
    )


def create_task_stop_catalog_provider(
    *,
    parent_deps: QueryDeps,
    enabled: bool = True,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a catalog provider that appends TaskStop when enabled."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        without_existing = tuple(tool for tool in tools if tool.name != TASK_STOP_TOOL_NAME)
        if not enabled or ctx.agent_id is not None:
            return without_existing
        return (*without_existing, build_task_stop_tool(deps=parent_deps))

    return provider


TASK_STOP_PROMPT = """Stop a running background task by task ID.

Use TaskStop when a worker or shell task was launched with the wrong objective,
is no longer needed, or should be cancelled after the user changes direction.
"""


def _record_coordinator_task_stop(
    *,
    deps: QueryDeps,
    ctx: ToolUseContext,
    task_id: str,
    task_type: str,
    description: str | None,
) -> dict[str, str]:
    runtime_deps = ctx.runtime.deps if ctx.runtime is not None else deps
    runtime = runtime_deps.coordinator_runtime
    if runtime is None:
        return {}
    try:
        result = runtime.record_task_stop(
            task_id=task_id,
            task_type=task_type,
            description=description,
        )
    except Exception as exc:
        _emit_coordinator_runtime_failure(
            deps=runtime_deps,
            ctx=ctx,
            operation="record_task_stop",
            exc=exc,
        )
        return {}

    metadata: dict[str, str] = {}
    work_item = getattr(result, "work_item", None)
    work_item_id = getattr(work_item, "id", None)
    if isinstance(work_item_id, str):
        metadata["coordinator_work_item_id"] = work_item_id
    blackboard_entry = getattr(result, "blackboard_entry", None)
    blackboard_entry_id = getattr(blackboard_entry, "id", None)
    if isinstance(blackboard_entry_id, str):
        metadata["coordinator_blackboard_entry_id"] = blackboard_entry_id
    return metadata


def _emit_coordinator_runtime_failure(
    *,
    deps: QueryDeps,
    ctx: ToolUseContext,
    operation: str,
    exc: Exception,
) -> None:
    event_context = (
        ctx.observability_context.with_source("coordinator")
        if ctx.observability_context is not None
        else KernelEventContext(
            session_id=ctx.session_id,
            agent_id=ctx.agent_id,
            source="coordinator",
        )
    )
    deps.observability.emit(
        "coordinator.runtime.integration_failed",
        context=event_context,
        data={
            "operation": operation,
            "error_type": type(exc).__name__,
        },
    )


def _coerce_input(input_: BaseModel) -> TaskStopInput:
    if isinstance(input_, TaskStopInput):
        return input_
    return TaskStopInput.model_validate(input_.model_dump())


def _task_id_from_input(input_: TaskStopInput) -> str | None:
    task_id = input_.task_id if input_.task_id is not None else input_.shell_id
    if task_id is None:
        return None
    stripped = task_id.strip()
    return stripped or None


__all__ = [
    "TASK_STOP_MAX_RESULT_SIZE_CHARS",
    "TASK_STOP_PROMPT",
    "TASK_STOP_TOOL_NAME",
    "TaskStopInput",
    "build_task_stop_tool",
    "create_task_stop_catalog_provider",
]
