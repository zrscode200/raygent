"""Background memory extraction scheduler.

"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from raygent_harness.core.state import UsageTotals
from raygent_harness.memdir.memdir import ENTRYPOINT_NAME, ensure_memory_dir_exists
from raygent_harness.memdir.memory_scan import format_memory_manifest, scan_memory_files
from raygent_harness.memdir.paths import (
    MemorySettings,
    get_auto_mem_path,
    is_auto_mem_path,
    is_auto_memory_enabled,
    is_extract_mode_active,
)
from raygent_harness.services.extract_memories.prompts import (
    FILE_EDIT_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    build_extract_auto_only_prompt,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.tool import ToolUseContext

_LOGGER = logging.getLogger(__name__)

WRITE_TOOL_NAMES = frozenset(
    {
        FILE_EDIT_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        "FileEdit",
        "FileWrite",
    }
)

ExtractionStatus = Literal[
    "ran",
    "coalesced",
    "throttled",
    "skipped_disabled",
    "skipped_remote",
    "skipped_subagent",
    "skipped_direct_write",
    "error",
]
ExtractionRunStatus = Literal["success", "error"]


@dataclass(frozen=True)
class ExtractionRequest:
    """Inputs handed to the injected extraction runner."""

    messages: tuple[MessageParam, ...]
    memory_dir: Path
    prompt: str
    new_message_count: int
    existing_memories: str
    skip_index: bool = False
    query_source: str = "extract_memories"
    max_turns: int = 5


@dataclass(frozen=True)
class ExtractionResult:
    """Result returned by the injected extraction runner."""

    status: ExtractionRunStatus = "success"
    messages: tuple[MessageParam, ...] = ()
    written_paths: tuple[Path, ...] = ()
    usage: UsageTotals = field(default_factory=UsageTotals)
    error: str | None = None


class ExtractionRunner(Protocol):
    """Injected runner for the memory-extraction agent.

    Kept as a protocol so embedders can choose whether memory extraction runs
    through a restricted child agent, a local service, or a custom backend.
    """

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        ...


@dataclass(frozen=True)
class SavedMemoryNotification:
    """System-facing record that memory files were saved."""

    memory_paths: tuple[Path, ...]


AppendSavedMemory = Callable[[SavedMemoryNotification], None]


@dataclass(frozen=True)
class ExtractionRunResult:
    """Observable summary for tests/integration callers."""

    status: ExtractionStatus
    new_message_count: int = 0
    written_paths: tuple[Path, ...] = ()
    memory_paths: tuple[Path, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class _Cursor:
    message_id: str | None
    index: int | None
    fingerprint: str | None


@dataclass(frozen=True)
class _PendingContext:
    messages: tuple[MessageParam, ...]
    agent_id: str | None
    turn_config: QueryConfig | None
    ctx: ToolUseContext | None
    append_saved_memory: AppendSavedMemory | None


def _message_id(message: MessageParam) -> str | None:
    for key in ("uuid", "id"):
        value = cast("dict[str, Any]", message).get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _message_fingerprint(message: MessageParam) -> str:
    return repr(message)


def _message_role(message: MessageParam) -> str | None:
    raw = cast("dict[str, Any]", message)
    role = raw.get("role")
    if isinstance(role, str):
        return role
    msg_type = raw.get("type")
    return msg_type if isinstance(msg_type, str) else None


def is_model_visible_message(message: MessageParam) -> bool:
    """Return true for messages that are included in model calls."""
    return _message_role(message) in {"user", "assistant"}


def _find_cursor_index(messages: Sequence[MessageParam], cursor: _Cursor | None) -> int | None:
    if cursor is None:
        return None

    if cursor.message_id is not None:
        for index, message in enumerate(messages):
            if _message_id(message) == cursor.message_id:
                return index
        return None

    if cursor.index is None:
        return None
    if (
        cursor.index < len(messages)
        and cursor.fingerprint == _message_fingerprint(messages[cursor.index])
    ):
        return cursor.index
    return None


def messages_since_cursor(
    messages: Sequence[MessageParam],
    cursor: _Cursor | None,
) -> list[MessageParam]:
    """Return messages after cursor; fall back to full history if missing.

    The full-history fallback matches the reference compaction behavior:
    when the UUID cursor was compacted away, count/process all visible messages
    rather than permanently disabling extraction.
    """
    index = _find_cursor_index(messages, cursor)
    if index is None:
        return list(messages)
    return list(messages[index + 1 :])


def _messages_after_cursor_no_fallback(
    messages: Sequence[MessageParam],
    cursor: _Cursor | None,
) -> list[MessageParam]:
    """Return messages after cursor, but do not fall back on missing cursor.

    Reference only applies full-history fallback to the visible-message count.
    Direct-write detection returns false when the cursor UUID was compacted
    away, so a stale memory write in compacted history cannot suppress a fresh
    extraction run.
    """
    if cursor is None:
        return list(messages)
    index = _find_cursor_index(messages, cursor)
    if index is None:
        return []
    return list(messages[index + 1 :])


def count_model_visible_messages_since(
    messages: Sequence[MessageParam],
    cursor: _Cursor | None = None,
) -> int:
    """Count model-visible messages after cursor, with compaction fallback."""
    return sum(
        1
        for message in messages_since_cursor(messages, cursor)
        if is_model_visible_message(message)
    )


def _last_cursor(messages: Sequence[MessageParam]) -> _Cursor | None:
    if not messages:
        return None
    index = len(messages) - 1
    message = messages[index]
    return _Cursor(
        message_id=_message_id(message),
        index=index,
        fingerprint=_message_fingerprint(message),
    )


def _assistant_content(message: MessageParam) -> object:
    raw = cast("dict[str, Any]", message)
    if raw.get("role") == "assistant":
        return raw.get("content")
    if raw.get("type") == "assistant":
        nested = raw.get("message")
        if isinstance(nested, dict):
            nested_message = cast("dict[str, object]", nested)
            return nested_message.get("content")
    return None


def _written_file_path_from_block(block: object) -> Path | None:
    if not isinstance(block, dict):
        return None
    raw_block = cast("dict[str, object]", block)
    if raw_block.get("type") != "tool_use":
        return None
    if raw_block.get("name") not in WRITE_TOOL_NAMES:
        return None

    raw_input = raw_block.get("input")
    if not isinstance(raw_input, dict):
        return None
    input_dict = cast("dict[str, object]", raw_input)

    raw_path = input_dict.get("file_path")
    if raw_path is None:
        raw_path = input_dict.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return Path(raw_path)


def extract_written_paths(messages: Sequence[MessageParam]) -> tuple[Path, ...]:
    """Extract unique file paths from Write/Edit tool_use blocks."""
    paths: dict[Path, None] = {}
    for message in messages:
        if _message_role(message) != "assistant":
            continue
        content = _assistant_content(message)
        if not isinstance(content, list):
            continue
        for block in cast("list[object]", content):
            path = _written_file_path_from_block(block)
            if path is not None:
                paths[path] = None
    return tuple(paths)


def has_memory_writes_since(
    messages: Sequence[MessageParam],
    cursor: _Cursor | None,
    settings: MemorySettings,
) -> bool:
    """Return whether the main assistant already wrote an auto-memory file."""
    for path in extract_written_paths(_messages_after_cursor_no_fallback(messages, cursor)):
        if is_auto_mem_path(path, settings):
            return True
    return False


class NoOpExtractionRunner:
    """Default extraction runner: no saved memories."""

    async def __call__(
        self,
        request: ExtractionRequest,
        *,
        parent_config: QueryConfig | None = None,
        parent_ctx: ToolUseContext | None = None,
    ) -> ExtractionResult:
        del request, parent_config, parent_ctx
        return ExtractionResult()


class MemoryExtractionScheduler:
    """Closure-scoped extraction scheduler.

    Holds the same mutable state as the reference `initExtractMemories()`
    closure: cursor, in-progress guard, pending trailing context, and throttle
    counter. The execution substrate is intentionally injected.
    """

    def __init__(
        self,
        *,
        settings: MemorySettings,
        runner: ExtractionRunner | None = None,
        feature_enabled: bool = True,
        non_interactive: bool = False,
        allow_non_interactive: bool = False,
        throttle_turns: int = 1,
        skip_index: bool = False,
    ) -> None:
        self._settings = settings
        self._runner = runner or NoOpExtractionRunner()
        self._feature_enabled = feature_enabled
        self._non_interactive = non_interactive
        self._allow_non_interactive = allow_non_interactive
        self._throttle_turns = max(1, throttle_turns)
        self._skip_index = skip_index

        self._cursor: _Cursor | None = None
        self._turns_since_last_extraction = 0
        self._in_progress = False
        self._pending: _PendingContext | None = None
        self._in_flight: set[asyncio.Task[Any]] = set()

    async def execute(
        self,
        messages: Sequence[MessageParam],
        *,
        agent_id: str | None = None,
        turn_config: QueryConfig | None = None,
        ctx: ToolUseContext | None = None,
        append_saved_memory: AppendSavedMemory | None = None,
    ) -> ExtractionRunResult:
        """Run or schedule memory extraction for a completed main-agent turn."""
        task = asyncio.current_task()
        if task is not None:
            self._in_flight.add(task)
        try:
            return await self._execute_impl(
                tuple(messages),
                agent_id=agent_id if agent_id is not None else (ctx.agent_id if ctx else None),
                turn_config=turn_config,
                ctx=ctx,
                append_saved_memory=append_saved_memory,
            )
        finally:
            if task is not None:
                self._in_flight.discard(task)

    async def _execute_impl(
        self,
        messages: tuple[MessageParam, ...],
        *,
        agent_id: str | None,
        turn_config: QueryConfig | None,
        ctx: ToolUseContext | None,
        append_saved_memory: AppendSavedMemory | None,
    ) -> ExtractionRunResult:
        if agent_id is not None:
            return ExtractionRunResult(status="skipped_subagent")

        if self._settings.remote_mode:
            return ExtractionRunResult(status="skipped_remote")

        if not self._enabled():
            return ExtractionRunResult(status="skipped_disabled")

        if self._in_progress:
            self._pending = _PendingContext(
                messages=messages,
                agent_id=agent_id,
                turn_config=turn_config,
                ctx=ctx,
                append_saved_memory=append_saved_memory,
            )
            return ExtractionRunResult(status="coalesced")

        return await self._run_extraction(
            messages=messages,
            turn_config=turn_config,
            ctx=ctx,
            append_saved_memory=append_saved_memory,
            is_trailing_run=False,
        )

    def _enabled(self) -> bool:
        return is_extract_mode_active(
            feature_enabled=self._feature_enabled,
            non_interactive=self._non_interactive,
            allow_non_interactive=self._allow_non_interactive,
        ) and is_auto_memory_enabled(self._settings)

    async def _run_extraction(
        self,
        *,
        messages: tuple[MessageParam, ...],
        turn_config: QueryConfig | None,
        ctx: ToolUseContext | None,
        append_saved_memory: AppendSavedMemory | None,
        is_trailing_run: bool,
    ) -> ExtractionRunResult:
        memory_dir = get_auto_mem_path(self._settings)
        new_message_count = count_model_visible_messages_since(messages, self._cursor)

        if has_memory_writes_since(messages, self._cursor, self._settings):
            self._cursor = _last_cursor(messages)
            return ExtractionRunResult(
                status="skipped_direct_write",
                new_message_count=new_message_count,
            )

        if not is_trailing_run:
            self._turns_since_last_extraction += 1
            if self._turns_since_last_extraction < self._throttle_turns:
                return ExtractionRunResult(
                    status="throttled",
                    new_message_count=new_message_count,
                )
        self._turns_since_last_extraction = 0

        ensure_memory_dir_exists(memory_dir)
        self._in_progress = True
        try:
            existing_memories = format_memory_manifest(scan_memory_files(memory_dir))
            prompt = build_extract_auto_only_prompt(
                new_message_count,
                existing_memories,
                skip_index=self._skip_index,
            )
            request = ExtractionRequest(
                messages=messages,
                memory_dir=memory_dir,
                prompt=prompt,
                new_message_count=new_message_count,
                existing_memories=existing_memories,
                skip_index=self._skip_index,
            )
            result = await self._runner(
                request,
                parent_config=turn_config,
                parent_ctx=ctx,
            )
        except Exception as exc:
            _LOGGER.debug("memory extraction failed: %s", exc)
            return ExtractionRunResult(
                status="error",
                new_message_count=new_message_count,
                error=str(exc),
            )
        else:
            if result.status == "error":
                _LOGGER.debug("memory extraction runner returned error: %s", result.error)
                return ExtractionRunResult(
                    status="error",
                    new_message_count=new_message_count,
                    error=result.error,
                )
            self._cursor = _last_cursor(messages)
            raw_written_paths = result.written_paths or extract_written_paths(result.messages)
            written_paths = tuple(
                path for path in raw_written_paths if is_auto_mem_path(path, self._settings)
            )
            memory_paths = tuple(path for path in written_paths if path.name != ENTRYPOINT_NAME)
            if memory_paths and append_saved_memory is not None:
                append_saved_memory(SavedMemoryNotification(memory_paths))
            return ExtractionRunResult(
                status="ran",
                new_message_count=new_message_count,
                written_paths=written_paths,
                memory_paths=memory_paths,
            )
        finally:
            self._in_progress = False
            pending = self._pending
            self._pending = None
            if pending is not None:
                await self._run_extraction(
                    messages=pending.messages,
                    turn_config=pending.turn_config,
                    ctx=pending.ctx,
                    append_saved_memory=pending.append_saved_memory,
                    is_trailing_run=True,
                )

    async def drain_pending_extraction(self, timeout_s: float | None = 60.0) -> bool:
        """Wait for in-flight extraction without cancelling on timeout."""
        current = asyncio.current_task()
        pending = {
            task
            for task in self._in_flight
            if task is not current and not task.done()
        }
        if not pending:
            return True

        done, still_pending = await asyncio.wait(pending, timeout=timeout_s)
        for task in done:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
        return not still_pending


def create_memory_extraction_scheduler(
    *,
    settings: MemorySettings,
    runner: ExtractionRunner | None = None,
    feature_enabled: bool = True,
    non_interactive: bool = False,
    allow_non_interactive: bool = False,
    throttle_turns: int = 1,
    skip_index: bool = False,
) -> MemoryExtractionScheduler:
    """Factory mirroring the reference's `initExtractMemories()` closure."""
    return MemoryExtractionScheduler(
        settings=settings,
        runner=runner,
        feature_enabled=feature_enabled,
        non_interactive=non_interactive,
        allow_non_interactive=allow_non_interactive,
        throttle_turns=throttle_turns,
        skip_index=skip_index,
    )


__all__ = [
    "AppendSavedMemory",
    "ExtractionRequest",
    "ExtractionResult",
    "ExtractionRunResult",
    "ExtractionRunStatus",
    "ExtractionRunner",
    "ExtractionStatus",
    "MemoryExtractionScheduler",
    "NoOpExtractionRunner",
    "SavedMemoryNotification",
    "count_model_visible_messages_since",
    "create_memory_extraction_scheduler",
    "extract_written_paths",
    "has_memory_writes_since",
    "is_model_visible_message",
    "messages_since_cursor",
]
