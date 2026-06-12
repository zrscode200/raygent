"""QueryDeps memory provider helpers.

Memory prompt loading dispatches one source-specific loader at a time across
team+private memory, auto-only memory, and disabled memory. Raygent keeps that
dispatch outside `core` so QueryEngine stays headless and receives memory
mechanics through `QueryDeps.memory_prompt_provider`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from raygent_harness.core.messages import (
    MessageParam,
    api_message_from_message_param,
    message_param_from_api_message,
    user_message,
)
from raygent_harness.core.model_types import (
    ApiMessage,
    ModelRequest,
    ModelResolveContext,
    ModelSampling,
    ObservableMessage,
)
from raygent_harness.core.observability import KernelEventBus, KernelEventContext
from raygent_harness.memdir.memdir import load_memory_prompt
from raygent_harness.memdir.paths import (
    MemorySettings,
    get_auto_mem_path,
    is_auto_memory_enabled,
)
from raygent_harness.memdir.recall import (
    MAX_RELEVANT_MEMORY_BYTES,
    MAX_RELEVANT_MEMORY_FILES,
    MAX_RELEVANT_MEMORY_LINES,
    MAX_RELEVANT_MEMORY_SESSION_BYTES,
    MemoryRecallFile,
    SurfacedMemoryFile,
    collect_recent_successful_tools,
    collect_surfaced_memories,
    filter_and_mark_recalled_memories,
    is_memory_recall_message,
    memory_recall_file_from_relevant,
    message_from_surfaced_memories,
    read_memories_for_surfacing,
)
from raygent_harness.memdir.relevance import (
    SELECT_MEMORIES_SYSTEM_PROMPT,
    MemorySelector,
    NoOpMemorySelector,
    RelevantMemory,
    build_select_memories_user_prompt,
    find_relevant_memories,
)
from raygent_harness.memdir.team_paths import is_team_memory_enabled
from raygent_harness.memdir.team_prompts import load_combined_memory_prompt

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import MemoryPromptProvider, MemoryRecallProvider
    from raygent_harness.core.model_provider import ModelProvider
    from raygent_harness.core.tool import ToolUseContext

_LOGGER = logging.getLogger(__name__)
_MULTI_WORD_RE = re.compile(r"\s")
type MemoryDirsProvider = Callable[
    [str, MemorySettings, object, object],
    Sequence[Path | str],
]


@dataclass(slots=True)
class MemoryRecallLifecycle:
    """Provider-neutral lifecycle facts for future observability sinks."""

    started_at: float
    query: str
    memory_dirs: tuple[Path, ...]
    recent_tools: tuple[str, ...] = ()
    already_surfaced_count: int = 0
    settled_at: float | None = None
    consumed_on_iteration: int | None = None
    selected_count: int = 0
    read_count: int = 0
    surfaced_count: int = 0
    duplicate_filtered_count: int = 0
    marked_count: int = 0
    truncated_count: int = 0
    cancelled: bool = False
    failed: bool = False


@dataclass(frozen=True, slots=True)
class _PrefetchResult:
    memories: tuple[SurfacedMemoryFile, ...]
    selected_count: int


@dataclass(slots=True)
class RelevantMemoryRecallPrefetch:
    """Concrete turn-scoped relevant-memory prefetch handle."""

    _task: asyncio.Task[_PrefetchResult]
    lifecycle: MemoryRecallLifecycle
    _observability: KernelEventBus | None = None
    _observability_context: KernelEventContext | None = None
    _completion_emitted: bool = False

    def __post_init__(self) -> None:
        self._task.add_done_callback(self._mark_settled)

    @property
    def settled_at(self) -> float | None:
        return self.lifecycle.settled_at

    @property
    def consumed_on_iteration(self) -> int | None:
        return self.lifecycle.consumed_on_iteration

    async def consume_if_ready(
        self,
        *,
        ctx: ToolUseContext,
        iteration: int,
    ) -> tuple[MessageParam, ...]:
        """Consume a settled prefetch without blocking or duplicate marking."""
        if self.lifecycle.consumed_on_iteration is not None:
            return ()
        if self.lifecycle.settled_at is None:
            return ()

        try:
            result = await self._task
        except asyncio.CancelledError:
            self.lifecycle.cancelled = True
            self.lifecycle.consumed_on_iteration = iteration
            self._emit_completed("cancelled")
            if ctx.abort_event.is_set():
                return ()
            raise
        except Exception as exc:
            self.lifecycle.failed = True
            self.lifecycle.consumed_on_iteration = iteration
            _LOGGER.warning("[memdir] memory recall prefetch failed: %s", exc)
            self._emit_completed("failed", error_type=type(exc).__name__)
            return ()

        mark_result = filter_and_mark_recalled_memories(
            result.memories,
            ctx.read_file_state,
        )
        self.lifecycle.selected_count = result.selected_count
        self.lifecycle.read_count = len(result.memories)
        self.lifecycle.surfaced_count = len(mark_result.memories)
        self.lifecycle.duplicate_filtered_count = mark_result.duplicate_filtered_count
        self.lifecycle.marked_count = mark_result.marked_count
        self.lifecycle.truncated_count = sum(
            1 for memory in mark_result.memories if memory.truncated
        )
        self.lifecycle.consumed_on_iteration = iteration

        message = message_from_surfaced_memories(mark_result.memories)
        self._emit_completed("completed", has_message=message is not None)
        return (message,) if message is not None else ()

    def cancel(self) -> None:
        if not self._task.done():
            self.lifecycle.cancelled = True
            self._task.cancel()
            self._emit_completed("cancelled")
            return
        if self.lifecycle.consumed_on_iteration is None:
            status = "failed" if self.lifecycle.failed else "not_consumed"
            self._emit_completed(status)

    def _mark_settled(self, task: asyncio.Task[_PrefetchResult]) -> None:
        self.lifecycle.settled_at = time.time()
        if task.cancelled():
            self.lifecycle.cancelled = True
            return
        try:
            if task.exception() is not None:
                self.lifecycle.failed = True
        except asyncio.CancelledError:
            self.lifecycle.cancelled = True

    def _emit_completed(
        self,
        status: str,
        *,
        has_message: bool = False,
        error_type: str | None = None,
    ) -> None:
        if self._completion_emitted or self._observability is None:
            return
        self._completion_emitted = True
        data: dict[str, object] = {
            "status": status,
            "memory_dir_count": len(self.lifecycle.memory_dirs),
            "recent_tool_count": len(self.lifecycle.recent_tools),
            "already_surfaced_count": self.lifecycle.already_surfaced_count,
            "selected_count": self.lifecycle.selected_count,
            "read_count": self.lifecycle.read_count,
            "surfaced_count": self.lifecycle.surfaced_count,
            "duplicate_filtered_count": self.lifecycle.duplicate_filtered_count,
            "marked_count": self.lifecycle.marked_count,
            "truncated_count": self.lifecycle.truncated_count,
            "cancelled": self.lifecycle.cancelled,
            "failed": self.lifecycle.failed,
            "has_message": has_message,
        }
        if self.lifecycle.consumed_on_iteration is not None:
            data["consumed_on_iteration"] = self.lifecycle.consumed_on_iteration
        if self.lifecycle.settled_at is not None:
            data["duration_s"] = self.lifecycle.settled_at - self.lifecycle.started_at
        if error_type is not None:
            data["error_type"] = error_type
        self._observability.emit(
            "memory.recall.completed",
            context=self._observability_context,
            data=data,
        )


@dataclass(frozen=True, slots=True)
class ConfiguredMemoryRecallProvider:
    """Configured memdir-backed `QueryDeps.memory_recall_provider`."""

    settings: MemorySettings
    selector: MemorySelector | None = None
    model_provider: ModelProvider | None = None
    selector_model: str | None = None
    memory_dirs_provider: MemoryDirsProvider | None = None
    max_files: int = MAX_RELEVANT_MEMORY_FILES
    max_lines: int = MAX_RELEVANT_MEMORY_LINES
    max_bytes: int = MAX_RELEVANT_MEMORY_BYTES
    max_session_bytes: int = MAX_RELEVANT_MEMORY_SESSION_BYTES

    def start(
        self,
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> RelevantMemoryRecallPrefetch | None:
        if not is_auto_memory_enabled(self.settings):
            return None

        messages_tuple = tuple(messages)
        last_user_message = _last_human_user_message(messages_tuple)
        if last_user_message is None:
            return None

        query = _message_text(last_user_message).strip()
        if not query or _MULTI_WORD_RE.search(query) is None:
            return None

        surfaced = collect_surfaced_memories(messages_tuple)
        if surfaced.total_bytes >= self.max_session_bytes:
            return None

        if self.selector is None and self.model_provider is None:
            return None

        memory_dirs = _unique_paths(
            _resolve_memory_dirs(
                query=query,
                settings=self.settings,
                config=config,
                ctx=ctx,
                memory_dirs_provider=self.memory_dirs_provider,
            )
        )
        if not memory_dirs:
            return None

        selector = self._selector_for(config)
        recent_tools = collect_recent_successful_tools(
            messages_tuple,
            last_user_message=last_user_message,
        )
        lifecycle = MemoryRecallLifecycle(
            started_at=time.time(),
            query=query,
            memory_dirs=memory_dirs,
            recent_tools=recent_tools,
            already_surfaced_count=len(surfaced.paths),
        )
        observability = _observability_bus(ctx)
        event_context = _observability_context(ctx)
        if observability is not None:
            observability.emit(
                "memory.recall.started",
                context=event_context,
                data={
                    "memory_dir_count": len(memory_dirs),
                    "recent_tool_count": len(recent_tools),
                    "already_surfaced_count": len(surfaced.paths),
                    "query_char_count": len(query),
                    "max_files": self.max_files,
                    "max_lines": self.max_lines,
                    "max_bytes": self.max_bytes,
                    "max_session_bytes": self.max_session_bytes,
                    "selector_type": type(selector).__name__,
                },
            )
        task = asyncio.create_task(
            _prefetch_relevant_memories(
                query=query,
                memory_dirs=memory_dirs,
                selector=selector,
                recent_tools=recent_tools,
                already_surfaced=surfaced.paths,
                ctx=ctx,
                max_files=self.max_files,
                max_lines=self.max_lines,
                max_bytes=self.max_bytes,
            )
        )
        return RelevantMemoryRecallPrefetch(
            task,
            lifecycle,
            _observability=observability,
            _observability_context=event_context,
        )

    def _selector_for(self, config: QueryConfig) -> MemorySelector:
        if self.selector is not None:
            return self.selector
        if self.model_provider is not None:
            return ModelProviderMemorySelector(
                provider=self.model_provider,
                model=self.selector_model or config.model,
            )
        return NoOpMemorySelector()


@dataclass(frozen=True, slots=True)
class ModelProviderMemorySelector:
    """ModelProvider-backed implementation of the memory selector seam."""

    provider: ModelProvider
    model: str
    max_tokens: int = 256

    async def select(
        self,
        *,
        query: str,
        manifest: str,
        recent_tools: tuple[str, ...],
        abort_event: asyncio.Event | None,
    ) -> list[str]:
        if abort_event is not None and abort_event.is_set():
            return []

        resolved_model = self.provider.resolve_model(
            self.model,
            ModelResolveContext(query_source="memdir_relevance"),
        )
        response = await self.provider.complete(
            ModelRequest(
                model=resolved_model,
                messages=(
                    api_message_from_message_param(
                        user_message(
                            build_select_memories_user_prompt(
                                query=query,
                                manifest=manifest,
                                recent_tools=recent_tools,
                            )
                        )
                    ),
                ),
                system_prompt=SELECT_MEMORIES_SYSTEM_PROMPT,
                sampling=ModelSampling(max_tokens=self.max_tokens),
                abort_event=abort_event,
                query_source="memdir_relevance",
            )
        )
        return _parse_selected_memory_names(
            _message_text_from_response(response.api_message)
        )


async def _prefetch_relevant_memories(
    *,
    query: str,
    memory_dirs: tuple[Path, ...],
    selector: MemorySelector,
    recent_tools: tuple[str, ...],
    already_surfaced: frozenset[str],
    ctx: ToolUseContext,
    max_files: int,
    max_lines: int,
    max_bytes: int,
) -> _PrefetchResult:
    selected = await _find_relevant_across_dirs(
        query=query,
        memory_dirs=memory_dirs,
        selector=selector,
        recent_tools=recent_tools,
        already_surfaced=already_surfaced,
        ctx=ctx,
    )
    fresh_selected: list[MemoryRecallFile] = []
    seen: set[str] = set()
    for memory in selected:
        normalized = str(memory.path.absolute())
        if (
            normalized in seen
            or normalized in already_surfaced
            or ctx.read_file_state.has(memory.path)
        ):
            continue
        seen.add(normalized)
        fresh_selected.append(memory_recall_file_from_relevant(memory))
        if len(fresh_selected) >= max_files:
            break

    memories = await read_memories_for_surfacing(
        tuple(fresh_selected),
        abort_event=ctx.abort_event,
        max_files=max_files,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
    return _PrefetchResult(memories=memories, selected_count=len(fresh_selected))


async def _find_relevant_across_dirs(
    *,
    query: str,
    memory_dirs: tuple[Path, ...],
    selector: MemorySelector,
    recent_tools: tuple[str, ...],
    already_surfaced: frozenset[str],
    ctx: ToolUseContext,
) -> list[RelevantMemory]:
    async def find_one(memory_dir: Path) -> list[RelevantMemory]:
        try:
            return await find_relevant_memories(
                query=query,
                memory_dir=memory_dir,
                selector=selector,
                recent_tools=recent_tools,
                already_surfaced=set(already_surfaced),
                abort_event=ctx.abort_event,
            )
        except asyncio.CancelledError:
            if ctx.abort_event.is_set():
                return []
            raise
        except Exception as exc:
            _LOGGER.warning("[memdir] find_relevant_memories failed for %s: %s", memory_dir, exc)
            return []

    results = await asyncio.gather(*(find_one(memory_dir) for memory_dir in memory_dirs))
    return [memory for memories in results for memory in memories]


def create_relevant_memory_recall_provider(
    settings: MemorySettings,
    *,
    selector: MemorySelector | None = None,
    model_provider: ModelProvider | None = None,
    selector_model: str | None = None,
    memory_dirs_provider: MemoryDirsProvider | None = None,
    max_files: int = MAX_RELEVANT_MEMORY_FILES,
    max_lines: int = MAX_RELEVANT_MEMORY_LINES,
    max_bytes: int = MAX_RELEVANT_MEMORY_BYTES,
    max_session_bytes: int = MAX_RELEVANT_MEMORY_SESSION_BYTES,
) -> MemoryRecallProvider:
    """Create a QueryDeps-compatible provider for query-time memory recall."""
    return ConfiguredMemoryRecallProvider(
        settings=settings,
        selector=selector,
        model_provider=model_provider,
        selector_model=selector_model,
        memory_dirs_provider=memory_dirs_provider,
        max_files=max_files,
        max_lines=max_lines,
        max_bytes=max_bytes,
        max_session_bytes=max_session_bytes,
    )


def configured_memory_recall_dirs(
    _query: str,
    settings: MemorySettings,
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> tuple[Path, ...]:
    """Return the default memory search roots for relevant-memory recall."""
    if not is_auto_memory_enabled(settings):
        return ()
    return (get_auto_mem_path(settings),)


def load_configured_memory_prompt(
    settings: MemorySettings,
    *,
    extra_guidelines: Sequence[str] | None = None,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> str | None:
    """Load the correct memory mechanics prompt for these settings.

    Team memory requires auto memory, so the branches match the reference:
    team+auto gets the combined prompt, auto-only gets the auto-memory prompt,
    and disabled memory returns `None`.
    """
    if is_team_memory_enabled(settings):
        return load_combined_memory_prompt(
            settings,
            extra_guidelines=extra_guidelines,
            skip_index=skip_index,
            include_searching_past_context=include_searching_past_context,
            project_transcript_dir=project_transcript_dir,
        )
    return load_memory_prompt(
        settings,
        extra_guidelines=extra_guidelines,
        skip_index=skip_index,
        include_searching_past_context=include_searching_past_context,
        project_transcript_dir=project_transcript_dir,
    )


def create_memory_prompt_provider(
    settings: MemorySettings,
    *,
    extra_guidelines: Sequence[str] | None = None,
    skip_index: bool = False,
    include_searching_past_context: bool = False,
    project_transcript_dir: Path | str | None = None,
) -> MemoryPromptProvider:
    """Create a QueryDeps-compatible provider for configured memory mechanics."""

    async def provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> str | None:
        return load_configured_memory_prompt(
            settings,
            extra_guidelines=extra_guidelines,
            skip_index=skip_index,
            include_searching_past_context=include_searching_past_context,
            project_transcript_dir=project_transcript_dir,
        )

    return provider


def _resolve_memory_dirs(
    *,
    query: str,
    settings: MemorySettings,
    config: QueryConfig,
    ctx: ToolUseContext,
    memory_dirs_provider: MemoryDirsProvider | None,
) -> Sequence[Path | str]:
    if memory_dirs_provider is None:
        return configured_memory_recall_dirs(query, settings, config, ctx)
    return memory_dirs_provider(query, settings, config, ctx)


def _unique_paths(paths: Sequence[Path | str]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _observability_bus(ctx: ToolUseContext) -> KernelEventBus | None:
    if ctx.runtime is None:
        return None
    return ctx.runtime.deps.observability


def _observability_context(ctx: ToolUseContext) -> KernelEventContext:
    if ctx.observability_context is not None:
        return ctx.observability_context.with_source("memory")
    return KernelEventContext(
        session_id=ctx.session_id,
        agent_id=ctx.agent_id,
        source="memory",
    )


def _last_human_user_message(messages: tuple[MessageParam, ...]) -> MessageParam | None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        if is_memory_recall_message(message) or _has_tool_result_content(message):
            continue
        return message
    return None


def _has_tool_result_content(message: MessageParam) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(block.get("type") == "tool_result" for block in content)


def _message_text(message: MessageParam) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    parts = [
        str(block.get("text"))
        for block in content
        if block.get("type") == "text" and block.get("text") is not None
    ]
    return "\n".join(parts)


def _message_text_from_response(message: ApiMessage | ObservableMessage) -> str:
    return _message_text(message_param_from_api_message(message))


def _parse_selected_memory_names(text: str) -> list[str]:
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return []
    raw_items = (
        cast(dict[str, object], parsed).get("selected_memories")
        if isinstance(parsed, dict)
        else parsed
    )
    if not isinstance(raw_items, list):
        return []
    return [item for item in cast(list[object], raw_items) if isinstance(item, str)]


__all__ = [
    "ConfiguredMemoryRecallProvider",
    "MemoryRecallLifecycle",
    "ModelProviderMemorySelector",
    "RelevantMemoryRecallPrefetch",
    "configured_memory_recall_dirs",
    "create_memory_prompt_provider",
    "create_relevant_memory_recall_provider",
    "load_configured_memory_prompt",
]
