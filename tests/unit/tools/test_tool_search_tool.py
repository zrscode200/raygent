from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    ToolPermissionContext,
    empty_tool_permission_context,
)
from raygent_harness.core.query import CompactBoundaryEvent, LayerResult
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.state import State
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.skills.models import SkillDefinition
from raygent_harness.tools.tool_search import TOOL_SEARCH_TOOL_NAME, ToolSearchInput
from raygent_harness.tools.tool_search_tool import (
    build_tool_search_tool,
    create_tool_search_catalog_provider,
    selected_tool_names_from_messages,
)
from tests.fakes import FakeModelProvider

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


class EmptyInput(BaseModel):
    pass


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _tool(
    name: str,
    *,
    should_defer: bool = False,
    always_load: bool = False,
) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} description",
            input_model=EmptyInput,
            call=_call,
            prompt=f"{name} prompt",
            is_read_only=True,
            is_concurrency_safe=True,
            should_defer=should_defer,
            always_load=always_load,
        )
    )


def _ctx(
    *,
    messages: list[MessageParam] | None = None,
    tools: Sequence[Tool] = (),
    permission_context: ToolPermissionContext | None = None,
) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        messages=list(messages or []),
        tools=tuple(tools),
        permission_context=permission_context or empty_tool_permission_context(),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _tool_search_messages(tool_name: str = "DeferredTool") -> list[MessageParam]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": TOOL_SEARCH_TOOL_NAME,
                    "input": {"query": f"select:{tool_name}"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_search",
                    "content": [
                        {"type": "tool_reference", "tool_name": tool_name},
                    ],
                }
            ],
        },
    ]


def _tool_reference_result_message(tool_name: str = "DeferredTool") -> MessageParam:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_search",
                "content": [
                    {"type": "tool_reference", "tool_name": tool_name},
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_tool_search_tool_is_read_only_safe_and_permission_auto_allowed() -> None:
    tool = build_tool_search_tool()
    parsed = ToolSearchInput(query="select:DeferredTool")

    assert tool.name == TOOL_SEARCH_TOOL_NAME
    assert tool.is_read_only(parsed)
    assert tool.is_concurrency_safe(parsed)
    assert not tool.is_destructive(parsed)
    assert not tool.is_open_world(parsed)
    decision = await tool.check_permissions(
        parsed,
        _ctx(),
        empty_tool_permission_context(),
    )
    assert isinstance(decision, PermissionAllowDecision)


@pytest.mark.asyncio
async def test_tool_search_tool_selects_deferred_and_already_loaded_tools() -> None:
    deferred = _tool("DeferredTool", should_defer=True)
    loaded = _tool("AlreadyLoaded")
    tool_search = build_tool_search_tool()
    parsed = ToolSearchInput(query="select:DeferredTool,AlreadyLoaded")

    events = [
        event
        async for event in tool_search.call(
            parsed,
            _ctx(tools=(deferred, loaded, tool_search)),
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    assert events[0].content == [
        {"type": "tool_reference", "tool_name": "DeferredTool"},
        {"type": "tool_reference", "tool_name": "AlreadyLoaded"},
    ]


@pytest.mark.asyncio
async def test_tool_search_tool_no_match_includes_pending_mcp_servers() -> None:
    async def pending(_ctx: ToolUseContext) -> Sequence[str]:
        return ("github",)

    tool_search = build_tool_search_tool(pending_mcp_servers_provider=pending)
    events = [
        event
        async for event in tool_search.call(
            ToolSearchInput(query="select:Missing"),
            _ctx(tools=(_tool("DeferredTool", should_defer=True), tool_search)),
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    assert events[0].content == (
        "No matching deferred tools found. Some MCP servers are still connecting: "
        "github. Their tools will become available shortly - try searching again."
    )


@pytest.mark.asyncio
async def test_catalog_provider_keeps_tool_search_visible_and_deferred_hidden() -> None:
    loaded = _tool("Base")
    deferred = _tool("DeferredTool", should_defer=True)
    provider = create_tool_search_catalog_provider()
    config = QueryConfig(model="claude-opus-4-7", tools=(loaded, deferred))

    tools = await provider(config, _ctx(), ())
    assert tools is not None
    assert tuple(tool.name for tool in tools) == (
        "Base",
        "DeferredTool",
        TOOL_SEARCH_TOOL_NAME,
    )

    hidden_deferred = next(tool for tool in tools if tool.name == "DeferredTool")
    assert hidden_deferred.should_defer
    assert not hidden_deferred.always_load


@pytest.mark.asyncio
async def test_catalog_provider_exposes_selected_deferred_tools_from_history() -> None:
    loaded = _tool("Base")
    deferred = _tool("DeferredTool", should_defer=True)
    provider = create_tool_search_catalog_provider()
    config = QueryConfig(model="claude-opus-4-7", tools=(loaded, deferred))
    messages = _tool_search_messages()

    tools = await provider(config, _ctx(messages=messages), ())
    assert tools is not None
    assert tuple(tool.name for tool in tools) == (
        "Base",
        "DeferredTool",
        TOOL_SEARCH_TOOL_NAME,
    )
    assert selected_tool_names_from_messages(messages) == {"DeferredTool"}

    visible_deferred = next(tool for tool in tools if tool.name == "DeferredTool")
    assert not visible_deferred.should_defer
    assert visible_deferred.always_load


@pytest.mark.asyncio
async def test_catalog_provider_keeps_tool_search_visible_for_pending_mcp_servers() -> None:
    async def pending(_ctx: ToolUseContext) -> Sequence[str]:
        return ("github",)

    loaded = _tool("Base")
    provider = create_tool_search_catalog_provider(pending_mcp_servers_provider=pending)

    tools = await provider(
        QueryConfig(model="claude-opus-4-7", tools=(loaded,)),
        _ctx(),
        (),
    )

    assert tools is not None
    assert tuple(tool.name for tool in tools) == ("Base", TOOL_SEARCH_TOOL_NAME)


def test_selected_tool_names_scan_user_tool_reference_blocks_without_join() -> None:
    messages = [_tool_reference_result_message("DeferredTool")]

    assert selected_tool_names_from_messages(messages) == {"DeferredTool"}


@pytest.mark.asyncio
async def test_catalog_provider_composes_with_upstream_provider() -> None:
    base = _tool("Base")
    deferred = _tool("DeferredTool", should_defer=True)

    async def upstream(
        config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        return (*config.tools, deferred)

    provider = create_tool_search_catalog_provider(upstream=upstream)

    tools = await provider(
        QueryConfig(model="claude-opus-4-7", tools=(base,)),
        _ctx(),
        (),
    )

    assert tools is not None
    assert tuple(tool.name for tool in tools) == (
        "Base",
        "DeferredTool",
        TOOL_SEARCH_TOOL_NAME,
    )


@pytest.mark.asyncio
async def test_query_loop_tool_search_select_exposes_schema_on_next_model_call() -> None:
    loaded = _tool("Base")
    deferred = _tool("DeferredTool", should_defer=True)
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "fetching schema"},
                    {
                        "type": "tool_use",
                        "id": "toolu_search",
                        "name": TOOL_SEARCH_TOOL_NAME,
                        "input": {"query": "select:DeferredTool"},
                    },
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    deps = QueryDeps(
        model_provider=provider,
        task_store=AppStateStore(),
        tool_catalog_provider=create_tool_search_catalog_provider(),
    )
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(loaded, deferred),
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("find the deferred tool")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert [tuple(spec.name for spec in request.tools) for request in provider.requests] == [
        ("Base", TOOL_SEARCH_TOOL_NAME),
        ("Base", "DeferredTool", TOOL_SEARCH_TOOL_NAME),
    ]


@pytest.mark.asyncio
async def test_query_loop_tool_search_discovery_survives_compaction_before_next_call() -> None:
    async def compact_after_tool_search(
        messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        if selected_tool_names_from_messages(messages):
            summary: MessageParam = {
                "role": "user",
                "content": "summary without tool references",
            }
            return LayerResult(
                messages=[summary],
                boundary=CompactBoundaryEvent(
                    kind="microcompact",
                    message_index=len(messages) - 1,
                    summary="summary",
                ),
            )
        return LayerResult(messages=messages)

    loaded = _tool("Base")
    deferred = _tool("DeferredTool", should_defer=True)
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "fetching schema"},
                    {
                        "type": "tool_use",
                        "id": "toolu_search",
                        "name": TOOL_SEARCH_TOOL_NAME,
                        "input": {"query": "select:DeferredTool"},
                    },
                ],
            },
            {"role": "assistant", "content": "done"},
        )
    )
    deps = QueryDeps(
        model_provider=provider,
        task_store=AppStateStore(),
        microcompact=compact_after_tool_search,
        tool_catalog_provider=create_tool_search_catalog_provider(),
    )
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(loaded, deferred),
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("find the deferred tool")]

    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert [tuple(spec.name for spec in request.tools) for request in provider.requests] == [
        ("Base", TOOL_SEARCH_TOOL_NAME),
        ("Base", "DeferredTool", TOOL_SEARCH_TOOL_NAME),
    ]
