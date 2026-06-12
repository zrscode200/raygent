"""Runtime wiring for Raygent-owned concrete file tools.

The individual `Read`, `Write`, and `Edit` builders stay standalone so tests and
adapters can opt into them narrowly. This facade is the headless equivalent of
session-level file-tool wiring: it enables the concrete
tools as a coherent set and attaches the team-memory post-write notification
hook when team-memory sync is configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raygent_harness.core.tool import Tool
from raygent_harness.core.tool_hooks import PostToolUseHook, PreToolUseHook
from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.services.team_memory_sync.hooks import (
    TeamMemoryWriteNotifier,
    create_team_memory_post_tool_use_hook,
)
from raygent_harness.tools.file_edit_tool import build_file_edit_tool
from raygent_harness.tools.file_read_tool import build_file_read_tool
from raygent_harness.tools.file_write_tool import build_file_write_tool
from raygent_harness.tools.notebook_edit_tool import build_notebook_edit_tool

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.tool import ToolUseContext
    from raygent_harness.services.file_media import PdfDocumentService
    from raygent_harness.skills.models import SkillDefinition


@dataclass(frozen=True)
class FileToolingRuntime:
    """Concrete file-tool bundle plus hooks that must travel with it."""

    tools: tuple[Tool, ...]
    pre_tool_use_hooks: tuple[PreToolUseHook, ...] = ()
    post_tool_use_hooks: tuple[PostToolUseHook, ...] = ()


def create_file_tooling_runtime(
    *,
    memory_settings: MemorySettings | None = None,
    notify_team_memory_write: TeamMemoryWriteNotifier | None = None,
    pdf_document_service: PdfDocumentService | None = None,
) -> FileToolingRuntime:
    """Build the concrete text file-tool runtime.

    `Write`/`Edit` receive `memory_settings` directly so their validation can
    block team-memory secrets before filesystem mutation. The post-write sync
    hook is included only when a notifier is supplied, avoiding global watcher
    state in the headless core.
    """

    post_hooks: tuple[PostToolUseHook, ...] = ()
    if memory_settings is not None and notify_team_memory_write is not None:
        post_hooks = (
            create_team_memory_post_tool_use_hook(
                memory_settings,
                notify_team_memory_write,
            ),
        )

    return FileToolingRuntime(
        tools=(
            build_file_read_tool(pdf_document_service=pdf_document_service),
            build_file_write_tool(memory_settings=memory_settings),
            build_file_edit_tool(memory_settings=memory_settings),
            build_notebook_edit_tool(memory_settings=memory_settings),
        ),
        post_tool_use_hooks=post_hooks,
    )


def create_file_tools_catalog_provider(
    *,
    runtime: FileToolingRuntime | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Return a catalog provider that appends the concrete file-tool runtime.

    Existing tools whose primary names or aliases collide with the runtime
    tools are replaced. This prevents an older `FileWrite` primary tool from
    shadowing the new `Write` tool's `FileWrite` alias in `find_tool_by_name`.
    """

    active_runtime = runtime or create_file_tooling_runtime()

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        existing_tools = _without_colliding_tools(tuple(tools), active_runtime.tools)
        return (*existing_tools, *active_runtime.tools)

    return provider


def _without_colliding_tools(
    tools: tuple[Tool, ...],
    runtime_tools: tuple[Tool, ...],
) -> tuple[Tool, ...]:
    reserved_names: set[str] = set()
    for tool in runtime_tools:
        reserved_names.update(_tool_identity_names(tool))
    return tuple(
        tool for tool in tools if _tool_identity_names(tool).isdisjoint(reserved_names)
    )


def _tool_identity_names(tool: Tool) -> set[str]:
    return {tool.name, *tool.aliases}


__all__ = [
    "FileToolingRuntime",
    "create_file_tooling_runtime",
    "create_file_tools_catalog_provider",
]
