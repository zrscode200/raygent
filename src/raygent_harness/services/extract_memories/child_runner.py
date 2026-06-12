"""Concrete restricted child-agent runner for background memory extraction."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from raygent_harness.core.child_query import ChildQueryRequest, run_child_query
from raygent_harness.core.deps import MemoryExtractor, QueryDeps
from raygent_harness.core.observability import KernelEventContext
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.services.extract_memories.extract_memories import (
    AppendSavedMemory,
    ExtractionRequest,
    ExtractionResult,
    ExtractionRunResult,
    create_memory_extraction_scheduler,
    extract_written_paths,
)
from raygent_harness.services.extract_memories.permissions import (
    ExtractionToolPolicy,
    build_extraction_tool_policy,
)
from raygent_harness.services.extract_memories.prompts import build_extract_auto_only_prompt
from raygent_harness.tools.bash_tool import build_bash_tool
from raygent_harness.tools.discovery_tools import create_discovery_tooling_runtime
from raygent_harness.tools.file_permissions import (
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
)
from raygent_harness.tools.file_tools import create_file_tooling_runtime

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.tool import Tool, ToolUseContext


@dataclass(frozen=True)
class RestrictedChildExtractionRunner:
    """Run extraction through a restricted synchronous child QueryEngine.

    This is the concrete counterpart to the reference forked extraction agent.
    It deliberately stays behind the existing `ExtractionRunner` protocol so
    embedders can keep custom runners where needed.
    """

    parent_deps: QueryDeps
    settings: MemorySettings
    tools: tuple[Tool, ...] | None = None
    cwd: str | None = None
    transcript_enabled: bool = False
    include_default_tools: bool = True

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        if parent_config is None:
            return ExtractionResult(
                status="error",
                error="memory extraction child runner requires parent_config",
            )
        if parent_ctx is None:
            return ExtractionResult(
                status="error",
                error="memory extraction child runner requires parent_ctx",
            )

        catalog = _build_extraction_catalog(
            explicit_tools=self.tools,
            parent_tools=parent_config.tools or parent_ctx.tools,
            parent_deps=self.parent_deps,
            settings=self.settings,
            include_default_tools=self.include_default_tools,
        )
        policy = build_extraction_tool_policy(
            catalog,
            settings=self.settings,
            parent_permission_context=parent_ctx.permission_context,
        )
        child_agent_id = f"extract_memories_{uuid.uuid4().hex[:8]}"
        started_at = self.parent_deps.clock.now()
        event_context = _runner_event_context(parent_ctx, child_agent_id)
        _emit_runner_event(
            self.parent_deps,
            event_context,
            "memory.extraction.runner.started",
            {
                "message_count": len(request.messages),
                "new_message_count": request.new_message_count,
                "tool_count": len(policy.tools),
                "tool_names": policy.tool_names,
                "missing_tool_count": len(policy.missing_tool_names),
                "missing_required_tool_count": len(policy.missing_required_tool_names),
                "filtered_tool_count": policy.filtered_tool_count,
                "max_turns": request.max_turns,
                "transcript_enabled": self.transcript_enabled,
            },
        )
        if not policy.is_usable:
            missing = ", ".join(policy.missing_required_tool_names)
            error = (
                "memory extraction requires Read and at least one memory "
                f"writer tool; missing: {missing}"
            )
            _emit_runner_event(
                self.parent_deps,
                event_context,
                "memory.extraction.runner.completed",
                {
                    "status": "error",
                    "subtype": "missing_required_tools",
                    "message_count": 0,
                    "written_path_count": 0,
                    "cap_exhausted": False,
                    "tool_count": len(policy.tools),
                    "missing_tool_count": len(policy.missing_tool_names),
                    "missing_required_tool_count": len(
                        policy.missing_required_tool_names
                    ),
                    "filtered_tool_count": policy.filtered_tool_count,
                    "duration_s": self.parent_deps.clock.now() - started_at,
                    "error_char_count": len(error),
                },
            )
            return ExtractionResult(status="error", error=error)

        child_deps = _build_extraction_child_deps(self.parent_deps, policy)
        prompt = build_extract_auto_only_prompt(
            request.new_message_count,
            request.existing_memories,
            skip_index=request.skip_index,
            allowed_tool_names=policy.tool_names,
        )
        child_request = ChildQueryRequest(
            prompt_messages=(*request.messages, {"role": "user", "content": prompt}),
            parent_config=parent_config,
            parent_deps=child_deps,
            parent_ctx=parent_ctx,
            agent_id=child_agent_id,
            agent_type="extract_memories",
            tools=policy.tools,
            permission_context=policy.permission_context,
            cwd=self.cwd or parent_ctx.cwd,
            query_source=request.query_source,
            transcript_label="extract_memories",
            link_abort_to_parent=True,
            max_turns=request.max_turns,
            transcript_enabled=self.transcript_enabled,
        )
        try:
            child_result = await run_child_query(child_request)
        except Exception as exc:
            _emit_runner_event(
                self.parent_deps,
                event_context,
                "memory.extraction.runner.completed",
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "duration_s": self.parent_deps.clock.now() - started_at,
                    "tool_count": len(policy.tools),
                    "missing_tool_count": len(policy.missing_tool_names),
                    "missing_required_tool_count": len(
                        policy.missing_required_tool_names
                    ),
                    "filtered_tool_count": policy.filtered_tool_count,
                },
            )
            return ExtractionResult(status="error", error=str(exc))

        status = "error" if child_result.is_error else "success"
        error = _child_error_text(child_result.errors, child_result.final_message)
        written_path_count = len(extract_written_paths(child_result.messages))
        _emit_runner_event(
            self.parent_deps,
            event_context,
            "memory.extraction.runner.completed",
            {
                "status": status,
                "subtype": child_result.subtype,
                "message_count": len(child_result.messages),
                "written_path_count": written_path_count,
                "cap_exhausted": child_result.subtype == "error_max_turns",
                "tool_count": len(policy.tools),
                "missing_tool_count": len(policy.missing_tool_names),
                "missing_required_tool_count": len(policy.missing_required_tool_names),
                "filtered_tool_count": policy.filtered_tool_count,
                "duration_s": self.parent_deps.clock.now() - started_at,
                "error_char_count": len(error) if child_result.is_error else 0,
            },
        )
        if child_result.is_error:
            return ExtractionResult(
                status="error",
                messages=child_result.messages,
                error=error,
            )
        return ExtractionResult(messages=child_result.messages)


def create_child_agent_memory_extractor(
    *,
    settings: MemorySettings,
    parent_deps: QueryDeps,
    tools: Sequence[Tool] | None = None,
    cwd: str | None = None,
    transcript_enabled: bool = False,
    feature_enabled: bool = True,
    non_interactive: bool = False,
    allow_non_interactive: bool = False,
    throttle_turns: int = 1,
    skip_index: bool = False,
    append_saved_memory: AppendSavedMemory | None = None,
    include_default_tools: bool = True,
) -> MemoryExtractor:
    """Create a `QueryDeps.memory_extractor` backed by the restricted runner."""

    runner = RestrictedChildExtractionRunner(
        parent_deps=parent_deps,
        settings=settings,
        tools=tuple(tools) if tools is not None else None,
        cwd=cwd,
        transcript_enabled=transcript_enabled,
        include_default_tools=include_default_tools,
    )
    scheduler = create_memory_extraction_scheduler(
        settings=settings,
        runner=runner,
        feature_enabled=feature_enabled,
        non_interactive=non_interactive,
        allow_non_interactive=allow_non_interactive,
        throttle_turns=throttle_turns,
        skip_index=skip_index,
    )

    async def extractor(
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> ExtractionRunResult:
        return await scheduler.execute(
            messages,
            turn_config=config,
            ctx=ctx,
            append_saved_memory=append_saved_memory,
        )

    return extractor


def build_default_extraction_tool_catalog(
    *,
    parent_tools: Sequence[Tool],
    parent_deps: QueryDeps,
    settings: MemorySettings,
) -> tuple[Tool, ...]:
    """Return parent tools plus Raygent-owned concrete extraction defaults.

    Parent tools are kept when their names/aliases do not collide. Raygent-owned
    concrete `Read`/`Write`/`Edit`/`Glob`/`Grep`/`Bash` implementations replace
    collisions so the extraction policy can rely on known validation and
    resource-protection behavior. Callers that need a fully custom extraction
    catalog can pass `tools=` to `RestrictedChildExtractionRunner` or
    `create_child_agent_memory_extractor(...)`.
    """

    runtime_file_tools = create_file_tooling_runtime(memory_settings=settings).tools
    file_tools = tuple(
        tool
        for tool in runtime_file_tools
        if tool.name in {FILE_READ_TOOL_NAME, FILE_WRITE_TOOL_NAME, FILE_EDIT_TOOL_NAME}
    )
    discovery_tools = create_discovery_tooling_runtime().tools
    bash_tool = build_bash_tool(deps=parent_deps)
    default_tools = (
        *file_tools,
        *discovery_tools,
        bash_tool,
    )
    collision_tools = (
        *runtime_file_tools,
        *discovery_tools,
        bash_tool,
    )
    return (*_without_colliding_tools(tuple(parent_tools), collision_tools), *default_tools)


def _build_extraction_catalog(
    *,
    explicit_tools: tuple[Tool, ...] | None,
    parent_tools: Sequence[Tool],
    parent_deps: QueryDeps,
    settings: MemorySettings,
    include_default_tools: bool,
) -> tuple[Tool, ...]:
    if explicit_tools is not None:
        return explicit_tools
    if not include_default_tools:
        return tuple(parent_tools)
    return build_default_extraction_tool_catalog(
        parent_tools=parent_tools,
        parent_deps=parent_deps,
        settings=settings,
    )


def _without_colliding_tools(
    tools: tuple[Tool, ...],
    default_tools: tuple[Tool, ...],
) -> tuple[Tool, ...]:
    reserved_names: set[str] = set()
    for tool in default_tools:
        reserved_names.update(_tool_identity_names(tool))
    return tuple(
        tool for tool in tools if _tool_identity_names(tool).isdisjoint(reserved_names)
    )


def _tool_identity_names(tool: Tool) -> set[str]:
    return {tool.name, *tool.aliases}


def _build_extraction_child_deps(
    parent_deps: QueryDeps,
    policy: ExtractionToolPolicy,
) -> QueryDeps:
    return replace(
        parent_deps,
        notify=_silent_notify,
        permission_context=policy.permission_context,
        permission_handler=None,
        memory_extractor=None,
        memory_recall_provider=None,
        agent_trigger_policy=None,
        coordinator_runtime=None,
        context_providers=(),
        system_prompt_provider=None,
        memory_prompt_provider=None,
        skill_provider=None,
        tool_catalog_provider=None,
        stop_hooks=[],
        pre_tool_use_hooks=[],
        post_tool_use_hooks=[],
        post_tool_use_failure_hooks=[],
        handoff_classifier=None,
    )


def _runner_event_context(
    parent_ctx: ToolUseContext,
    child_agent_id: str,
) -> KernelEventContext:
    base = parent_ctx.observability_context or KernelEventContext(
        session_id=parent_ctx.session_id,
        agent_id=parent_ctx.agent_id,
        source="memory",
    )
    return base.with_source("memory").for_child_agent(child_agent_id)


def _emit_runner_event(
    deps: QueryDeps,
    context: KernelEventContext,
    event_type: str,
    data: dict[str, object],
) -> None:
    deps.observability.emit(event_type, context=context, data=data)


def _child_error_text(errors: Sequence[str], final_message: str) -> str:
    if errors:
        return "; ".join(errors)
    return final_message or "memory extraction child returned an error"


def _silent_notify(_message: str) -> None:
    return


__all__ = [
    "RestrictedChildExtractionRunner",
    "create_child_agent_memory_extractor",
]
