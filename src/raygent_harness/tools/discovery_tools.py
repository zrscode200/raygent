"""Runtime wiring for concrete local discovery tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raygent_harness.core.tool import Tool
from raygent_harness.tools.glob_tool import build_glob_tool
from raygent_harness.tools.grep_tool import build_grep_tool
from raygent_harness.tools.search_backend import SearchBackend

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.tool import ToolUseContext
    from raygent_harness.skills.models import SkillDefinition


@dataclass(frozen=True)
class DiscoveryToolingRuntime:
    """Concrete search/discovery tool bundle."""

    tools: tuple[Tool, ...]


def create_discovery_tooling_runtime(
    *,
    backend: SearchBackend | None = None,
) -> DiscoveryToolingRuntime:
    return DiscoveryToolingRuntime(
        tools=(
            build_glob_tool(backend=backend),
            build_grep_tool(backend=backend),
        )
    )


def create_discovery_tools_catalog_provider(
    *,
    runtime: DiscoveryToolingRuntime | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Return a catalog provider that appends local discovery tools."""

    active_runtime = runtime or create_discovery_tooling_runtime()

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
    "DiscoveryToolingRuntime",
    "create_discovery_tooling_runtime",
    "create_discovery_tools_catalog_provider",
]
