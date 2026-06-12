from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.permissions import (
    AddPermissionRules,
    PermissionPassthrough,
    empty_tool_permission_context,
)
from raygent_harness.core.query import (
    _build_model_tool_specs,  # pyright: ignore[reportPrivateUsage]
)
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolProgress,
    ToolResult,
    ToolUseContext,
)
from raygent_harness.services.mcp import (
    InMemoryMcpClient,
    McpClientError,
    McpRegistrySnapshot,
    McpServerIdentity,
    McpServerState,
    McpToolCallContext,
    McpToolCallRequest,
    McpToolCallResult,
    allocate_mcp_tool_schemas,
)
from raygent_harness.tools.mcp_tool import (
    MCP_TRUNCATION_MESSAGE,
    McpToolInput,
    build_mcp_tool,
    create_mcp_tools_catalog_provider,
    create_pending_mcp_servers_provider,
    map_mcp_tool_result_content,
    normalize_mcp_input_schema,
    pending_mcp_servers_from_client,
)
from raygent_harness.tools.tool_search import TOOL_SEARCH_TOOL_NAME, ToolSearchInput
from raygent_harness.tools.tool_search_tool import create_tool_search_catalog_provider


def _ctx(
    *,
    tools: tuple[Tool, ...] = (),
    discovered: frozenset[str] = frozenset(),
    tool_use_id: str | None = "toolu_1",
    handle_elicitation: Callable[[str], Awaitable[str]] | None = None,
) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        tools=tools,
        permission_context=empty_tool_permission_context(),
        query_tracking=QueryTracking(chain_id="c", depth=0),
        discovered_tool_names=discovered,
        tool_use_id=tool_use_id,
        handle_elicitation=handle_elicitation,
    )


def _snapshot() -> tuple[McpRegistrySnapshot, str]:
    server = McpServerIdentity(name="GitHub", normalized_name="github")
    schema = allocate_mcp_tool_schemas(
        server,
        (
            {
                "name": "create issue",
                "description": "Create an issue",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "openWorldHint": True,
                },
                "search_hint": "issue tracker",
            },
        ),
    )[0]
    snapshot = McpRegistrySnapshot(
        (
            McpServerState(
                identity=server,
                status="connected",
                capabilities=("tools",),
                tools=(schema,),
            ),
        )
    )
    return snapshot, schema.raygent_name


@pytest.mark.asyncio
async def test_mcp_tool_preserves_schema_axes_and_permission_passthrough() -> None:
    snapshot, raygent_name = _snapshot()
    schema = snapshot.available_tools()[0]
    client = InMemoryMcpClient(
        snapshot=snapshot,
        responses={raygent_name: McpToolCallResult(content="created")},
    )
    tool = build_mcp_tool(schema, client=client)
    parsed = McpToolInput.model_validate({"title": "Bug"})

    assert tool.name == "mcp__github__create_issue"
    assert tool.search_hint == "issue tracker"
    assert tool.should_defer is True
    assert tool.always_load is False
    assert tool.is_read_only(parsed)
    assert tool.is_concurrency_safe(parsed)
    assert not tool.is_destructive(parsed)
    assert tool.is_open_world(parsed)
    assert tool.input_schema is not None
    assert thaw_json(tool.input_schema) == {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    decision = await tool.check_permissions(
        parsed,
        _ctx(),
        empty_tool_permission_context(),
    )
    assert isinstance(decision, PermissionPassthrough)
    suggestion = decision.suggestions[0]
    assert isinstance(suggestion, AddPermissionRules)
    assert suggestion.rules[0].tool_name == raygent_name


@pytest.mark.asyncio
async def test_mcp_tool_call_preserves_original_identity_and_arguments() -> None:
    snapshot, raygent_name = _snapshot()
    schema = snapshot.available_tools()[0]
    client = InMemoryMcpClient(
        snapshot=snapshot,
        responses={
            raygent_name: McpToolCallResult(
                content=({"type": "text", "text": "created"},),
            )
        },
    )
    tool = build_mcp_tool(schema, client=client)
    parsed = McpToolInput.model_validate({"title": "Bug"})

    events = [event async for event in tool.call(parsed, _ctx())]

    assert isinstance(events[0], ToolProgress)
    assert isinstance(events[1], ToolProgress)
    assert isinstance(events[2], ToolResult)
    assert events[2].content == [{"type": "text", "text": "created"}]
    assert client.calls[0].server_name == "GitHub"
    assert client.calls[0].tool_name == "create issue"
    assert client.calls[0].tool_use_id == "toolu_1"
    assert thaw_json(client.calls[0].arguments) == {"title": "Bug"}


@pytest.mark.asyncio
async def test_mcp_tool_call_passes_context_and_streams_client_progress() -> None:
    snapshot, _raygent_name = _snapshot()
    schema = snapshot.available_tools()[0]

    async def handle_elicitation(prompt: str) -> str:
        return f"handled:{prompt}"

    class RecordingClient:
        context: McpToolCallContext | None = None

        async def registry_snapshot(self) -> McpRegistrySnapshot:
            return snapshot

        async def call_tool(
            self,
            _request: McpToolCallRequest,
            context: McpToolCallContext,
        ) -> McpToolCallResult:
            self.context = context
            assert context.abort_event is not None
            assert context.handle_elicitation is handle_elicitation
            assert context.emit_progress is not None
            await context.emit_progress({"status": "server_progress", "step": 1})
            return McpToolCallResult(content="created")

    client = RecordingClient()
    tool = build_mcp_tool(schema, client=client)

    events = [
        event
        async for event in tool.call(
            McpToolInput.model_validate({"title": "Bug"}),
            _ctx(handle_elicitation=handle_elicitation),
        )
    ]

    assert client.context is not None
    progress_events = [event for event in events if isinstance(event, ToolProgress)]
    assert [event.data["status"] for event in progress_events if event.data] == [
        "started",
        "server_progress",
        "completed",
    ]


@pytest.mark.asyncio
async def test_mcp_tool_maps_client_error_to_model_visible_error() -> None:
    snapshot, raygent_name = _snapshot()
    schema = snapshot.available_tools()[0]
    client = InMemoryMcpClient(
        snapshot=snapshot,
        responses={
            raygent_name: McpClientError("server disconnected", kind="transport_error")
        },
    )
    tool = build_mcp_tool(schema, client=client)

    events = [
        event
        async for event in tool.call(
            McpToolInput.model_validate({"title": "Bug"}),
            _ctx(),
        )
    ]

    result = events[-1]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "server disconnected" in result.content
    failed_progress = events[-2]
    assert isinstance(failed_progress, ToolProgress)
    assert failed_progress.data is not None
    assert failed_progress.data["status"] == "failed"


def test_mcp_result_mapping_truncates_text_and_blocks() -> None:
    text = map_mcp_tool_result_content(
        McpToolCallResult(content="abcdef"),
        max_result_size_chars=5,
    )
    assert isinstance(text, str)
    assert MCP_TRUNCATION_MESSAGE in text

    blocks = map_mcp_tool_result_content(
        McpToolCallResult(content=({"type": "text", "text": "abcdef"},)),
        max_result_size_chars=5,
    )
    assert isinstance(blocks, list)
    assert MCP_TRUNCATION_MESSAGE in blocks[0]["text"]


def test_mcp_input_schema_normalizes_to_object_shape() -> None:
    assert thaw_json(normalize_mcp_input_schema({"type": "string"})) == {
        "type": "object",
        "properties": {},
    }
    assert thaw_json(normalize_mcp_input_schema(())) == {
        "type": "object",
        "properties": {},
    }


@pytest.mark.asyncio
async def test_mcp_catalog_provider_appends_tools_and_deferred_schema() -> None:
    snapshot, raygent_name = _snapshot()
    client = InMemoryMcpClient(snapshot=snapshot)
    provider = create_mcp_tools_catalog_provider(client=client)
    config = QueryConfig(model="test-model", tools=())

    tools = await provider(config, _ctx(), ())

    assert tools is not None
    assert tuple(tool.name for tool in tools) == (raygent_name,)
    assert tools[0].should_defer is True
    hidden_specs = await _build_model_tool_specs(tools, _ctx(tools=tuple(tools)))
    assert hidden_specs == []

    visible_specs = await _build_model_tool_specs(
        tools,
        _ctx(tools=tuple(tools), discovered=frozenset({raygent_name})),
    )
    assert len(visible_specs) == 1
    assert visible_specs[0].name == raygent_name
    visible_schema = cast(Mapping[str, object], thaw_json(visible_specs[0].input_schema))
    assert visible_schema["properties"] == {"title": {"type": "string"}}


@pytest.mark.asyncio
async def test_mcp_provider_composes_with_tool_search_for_discovery() -> None:
    snapshot, raygent_name = _snapshot()
    client = InMemoryMcpClient(snapshot=snapshot)
    provider = create_tool_search_catalog_provider(
        upstream=create_mcp_tools_catalog_provider(client=client),
        pending_mcp_servers_provider=create_pending_mcp_servers_provider(client),
    )
    config = QueryConfig(model="test-model", tools=())

    tools = await provider(config, _ctx(), ())

    assert tools is not None
    assert tuple(tool.name for tool in tools) == (raygent_name, TOOL_SEARCH_TOOL_NAME)
    model_specs = await _build_model_tool_specs(tools, _ctx(tools=tuple(tools)))
    assert tuple(spec.name for spec in model_specs) == (TOOL_SEARCH_TOOL_NAME,)
    tool_search = tools[-1]
    events = [
        event
        async for event in tool_search.call(
            ToolSearchInput(query=f"select:{raygent_name}"),
            _ctx(tools=tuple(tools)),
        )
    ]

    result = events[-1]
    assert isinstance(result, ToolResult)
    assert result.content == [{"type": "tool_reference", "tool_name": raygent_name}]


@pytest.mark.asyncio
async def test_pending_mcp_servers_provider_reports_only_connecting_servers() -> None:
    pending = McpServerState(
        identity=McpServerIdentity(name="Linear", normalized_name="linear"),
        status="pending",
    )
    needs_auth = McpServerState(
        identity=McpServerIdentity(name="Slack", normalized_name="slack"),
        status="needs_auth",
    )
    client = InMemoryMcpClient(
        snapshot=McpRegistrySnapshot((pending, needs_auth)),
    )

    assert await pending_mcp_servers_from_client(client, _ctx()) == ("linear",)
