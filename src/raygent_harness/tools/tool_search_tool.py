"""Concrete model-callable ToolSearch wrapper.


The pure search/select primitives live separately. This module adapts
those primitives into Raygent's `Tool` contract and provides the catalog helper
that keeps ToolSearch visible while deferred tools stay hidden until selected.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from raygent_harness.core.model_types import FrozenJson
from raygent_harness.core.permissions import PermissionAllowDecision, PermissionResult
from raygent_harness.core.tool import (
    InterruptBehavior,
    Tool,
    ToolCallEvent,
    ToolPromptContext,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationResult,
    build_tool,
)
from raygent_harness.tools.tool_search import (
    TOOL_SEARCH_MAX_RESULT_SIZE_CHARS,
    TOOL_SEARCH_TOOL_NAME,
    ToolSearchIndex,
    ToolSearchInput,
    is_deferred_tool,
    map_tool_search_result_content,
    run_tool_search,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.skills.models import SkillDefinition


PendingMcpServersProvider = Callable[
    [ToolUseContext],
    Sequence[str] | Awaitable[Sequence[str]],
]


TOOL_SEARCH_PROMPT = """Fetches full schema definitions for deferred tools so they can be called.

Deferred tools appear by name in prior system or user messages. Until fetched,
only the name is known; there is no parameter schema, so the tool cannot be
invoked. This tool takes a query, matches it against the deferred tool list,
and returns the matched tools' schema references. Once a tool's schema appears
in the result, it is callable exactly like any tool defined at the top of the
prompt.

Query forms:
- "select:Read,Edit,Grep" - fetch these exact tools by name
- "notebook jupyter" - keyword search, up to max_results best matches
- "+slack send" - require "slack" in the name, rank by remaining terms"""


def build_tool_search_tool(
    *,
    index: ToolSearchIndex | None = None,
    pending_mcp_servers_provider: PendingMcpServersProvider | None = None,
) -> Tool:
    """Build the concrete ToolSearch tool.

    Reference properties: read-only, concurrency-safe, hidden UI message, and
    permission auto-allow. Raygent keeps the UI pieces out of the headless tool
    contract and encodes the rest in `ToolSpec`.
    """

    search_index = index or ToolSearchIndex()

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: object,
    ) -> PermissionAllowDecision:
        return PermissionAllowDecision()

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        parsed = (
            input_
            if isinstance(input_, ToolSearchInput)
            else ToolSearchInput.model_validate(input_.model_dump())
        )
        pending_mcp_servers = await _resolve_pending_mcp_servers(
            pending_mcp_servers_provider,
            ctx,
        )
        result = await run_tool_search(
            query=parsed.query,
            tools=list(ctx.tools),
            max_results=parsed.max_results,
            index=search_index,
            pending_mcp_servers=tuple(pending_mcp_servers),
            prompt_context=ToolPromptContext(
                permission_context=ctx.permission_context,
                tools=ctx.tools,
            ),
        )
        yield ToolResult(
            content=map_tool_search_result_content(result),
            discovered_tool_names=result.matches,
        )

    return build_tool(
        ToolSpec(
            name=TOOL_SEARCH_TOOL_NAME,
            description="Search deferred tools and return selected schemas.",
            input_model=ToolSearchInput,
            call=call,
            prompt=TOOL_SEARCH_PROMPT,
            check_permissions=check_permissions,
            is_concurrency_safe=True,
            is_read_only=True,
            is_destructive=False,
            is_open_world=False,
            should_defer=False,
            always_load=True,
            max_result_size_chars=TOOL_SEARCH_MAX_RESULT_SIZE_CHARS,
        )
    )


def create_tool_search_catalog_provider(
    *,
    index: ToolSearchIndex | None = None,
    pending_mcp_servers_provider: PendingMcpServersProvider | None = None,
    upstream: ToolCatalogProvider | None = None,
) -> ToolCatalogProvider:
    """Create a provider that adds ToolSearch and exposes selected tools.

    `upstream` lets skill/plugin providers expand the base catalog before
    ToolSearch applies its deferred-tool visibility rules.
    """

    tool_search_tool = build_tool_search_tool(
        index=index,
        pending_mcp_servers_provider=pending_mcp_servers_provider,
    )

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        pending_mcp_servers = await _resolve_pending_mcp_servers(
            pending_mcp_servers_provider,
            ctx,
        )
        return apply_tool_search_catalog(
            tuple(tools),
            ctx.discovered_tool_names,
            tool_search_tool=tool_search_tool,
            force_tool_search=bool(pending_mcp_servers),
        )

    return provider


def apply_tool_search_catalog(
    tools: Sequence[Tool],
    selected_tool_names: Iterable[object] | None = (),
    *,
    tool_search_tool: Tool | None = None,
    force_tool_search: bool = False,
) -> tuple[Tool, ...]:
    """Add ToolSearch and mark selected deferred tools as model-visible.

    The runtime catalog keeps all tools for execution lookup. Selected names
    must come from trusted runtime discovery state, not raw transcript parsing.
    The model-visible request is filtered later by `query._build_model_tool_specs`,
    but wrapping selected deferred tools here also makes provider output faithful
    for future adapters that inspect the turn catalog directly.
    """

    search_tool = tool_search_tool or build_tool_search_tool()
    selected = _trusted_selected_name_set(selected_tool_names)
    without_existing_search = tuple(
        tool for tool in tools if tool.name != TOOL_SEARCH_TOOL_NAME
    )
    has_deferred_tools = any(is_deferred_tool(tool) for tool in without_existing_search)
    exposed = tuple(
        _VisibleDeferredTool(tool)
        if _is_selected_deferred_tool(tool, selected)
        else tool
        for tool in without_existing_search
    )
    if not has_deferred_tools and not force_tool_search:
        return exposed
    return (*exposed, search_tool)


def selected_tool_names_from_messages(messages: Sequence[MessageParam]) -> set[str]:
    """Return tool names discovered by prior `tool_reference` blocks.

    This parser is only for the engine-owned assistant/tool-result slice from a
    live ToolSearch execution. It is not an authorization source for arbitrary
    transcript history, seed messages, or replay.
    """

    selected: set[str] = set()
    tool_search_tool_use_ids: set[str] = set()

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            tool_search_tool_use_ids.update(_tool_search_tool_use_ids(content))
            continue
        if role != "user":
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if (
                not isinstance(tool_use_id, str)
                or tool_use_id not in tool_search_tool_use_ids
            ):
                continue
            selected.update(_tool_reference_names(block.get("content")))

    return selected


@dataclass
class _VisibleDeferredTool:
    """Proxy a selected deferred tool as prompt-visible without mutating it."""

    wrapped: Tool
    name: str = field(init=False)
    aliases: tuple[str, ...] = field(init=False)
    description: str = field(init=False)
    search_hint: str | None = field(init=False)
    input_model: type[BaseModel] = field(init=False)
    input_schema: FrozenJson | None = field(init=False)
    should_defer: bool = field(init=False, default=False)
    always_load: bool = field(init=False, default=True)
    max_result_size_chars: int | float = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.wrapped.name
        self.aliases = self.wrapped.aliases
        self.description = self.wrapped.description
        self.search_hint = self.wrapped.search_hint
        self.input_model = self.wrapped.input_model
        self.input_schema = self.wrapped.input_schema
        self.max_result_size_chars = self.wrapped.max_result_size_chars

    def is_enabled(self) -> bool:
        return self.wrapped.is_enabled()

    def is_concurrency_safe(self, input: BaseModel) -> bool:
        return self.wrapped.is_concurrency_safe(input)

    def is_read_only(self, input: BaseModel) -> bool:
        return self.wrapped.is_read_only(input)

    def is_destructive(self, input: BaseModel) -> bool:
        return self.wrapped.is_destructive(input)

    def is_open_world(self, input: BaseModel) -> bool:
        return self.wrapped.is_open_world(input)

    def requires_user_interaction(self) -> bool:
        return self.wrapped.requires_user_interaction()

    def interrupt_behavior(self) -> InterruptBehavior:
        return self.wrapped.interrupt_behavior()

    async def validate_input(
        self,
        input: BaseModel,
        ctx: ToolUseContext,
    ) -> ValidationResult:
        return await self.wrapped.validate_input(input, ctx)

    async def check_permissions(
        self,
        input: BaseModel,
        ctx: ToolUseContext,
        permission_context: Any,
    ) -> PermissionResult:
        return await self.wrapped.check_permissions(input, ctx, permission_context)

    def call(
        self,
        input: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        return self.wrapped.call(input, ctx)

    async def describe(self, input: BaseModel, ctx: Any) -> str:
        return await self.wrapped.describe(input, ctx)

    async def prompt(
        self,
        ctx: ToolPromptContext | ToolUseContext | None = None,
    ) -> str:
        return await self.wrapped.prompt(ctx)

    def get_activity_description(self, input: BaseModel) -> str | None:
        return self.wrapped.get_activity_description(input)


async def _resolve_pending_mcp_servers(
    provider: PendingMcpServersProvider | None,
    ctx: ToolUseContext,
) -> tuple[str, ...]:
    if provider is None:
        return ()
    result = provider(ctx)
    if inspect.isawaitable(result):
        result = await result
    return tuple(str(item) for item in result)


def _is_selected_deferred_tool(tool: Tool, selected: set[str]) -> bool:
    if not selected or not is_deferred_tool(tool):
        return False
    return tool.name in selected or any(alias in selected for alias in tool.aliases)


def _trusted_selected_name_set(selected_tool_names: Iterable[object] | None) -> set[str]:
    if selected_tool_names is None:
        return set()
    if isinstance(selected_tool_names, str):
        return {selected_tool_names}
    selected: set[str] = set()
    for name in selected_tool_names:
        if not isinstance(name, str):
            raise TypeError(
                "selected_tool_names must be strings from trusted runtime "
                "discovery state; pass ctx.discovered_tool_names, not "
                "transcript messages"
            )
        if name:
            selected.add(name)
    return selected


def _tool_search_tool_use_ids(content: object) -> set[str]:
    ids: set[str] = set()
    if not isinstance(content, list):
        return ids
    for block in cast(list[object], content):
        if not isinstance(block, Mapping):
            continue
        item = cast(Mapping[str, object], block)
        if item.get("type") != "tool_use":
            continue
        if item.get("name") != TOOL_SEARCH_TOOL_NAME:
            continue
        tool_use_id = item.get("id")
        if isinstance(tool_use_id, str) and tool_use_id:
            ids.add(tool_use_id)
    return ids


def _tool_reference_names(content: object) -> set[str]:
    names: set[str] = set()
    content_blocks: list[Mapping[str, object]]
    if isinstance(content, Mapping):
        content_blocks = [cast(Mapping[str, object], content)]
    elif isinstance(content, list):
        content_blocks = []
        for item in cast(list[object], content):
            if isinstance(item, Mapping):
                content_blocks.append(cast(Mapping[str, object], item))
    else:
        return names
    for item in content_blocks:
        if item.get("type") != "tool_reference":
            continue
        tool_name = item.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            names.add(tool_name)
    return names


__all__ = [
    "TOOL_SEARCH_PROMPT",
    "PendingMcpServersProvider",
    "apply_tool_search_catalog",
    "build_tool_search_tool",
    "create_tool_search_catalog_provider",
    "selected_tool_names_from_messages",
]
