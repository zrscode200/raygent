from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from raygent_harness.core.permissions import empty_tool_permission_context
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolPromptContext,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.tools.tool_search import (
    TOOL_SEARCH_TOOL_NAME,
    ToolSearchIndex,
    deferred_tools_cache_key,
    is_deferred_tool,
    map_tool_search_result_content,
    parse_tool_name,
    run_tool_search,
    search_tool_names,
    tool_search_query_type,
)


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
    aliases: tuple[str, ...] = (),
    prompt: str = "",
    search_hint: str | None = None,
    should_defer: bool = True,
    always_load: bool = False,
) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} description",
            input_model=EmptyInput,
            call=_call,
            aliases=aliases,
            prompt=prompt or f"Prompt for {name}",
            search_hint=search_hint,
            should_defer=should_defer,
            always_load=always_load,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def test_is_deferred_tool_respects_always_load_and_tool_search_name() -> None:
    assert is_deferred_tool(_tool("Deferred"))
    assert not is_deferred_tool(_tool("Loaded", always_load=True))
    assert not is_deferred_tool(_tool("NotDeferred", should_defer=False))
    assert not is_deferred_tool(_tool(TOOL_SEARCH_TOOL_NAME))


def test_parse_tool_name_handles_regular_and_mcp_names() -> None:
    assert parse_tool_name("NotebookRead") == (
        ("notebook", "read"),
        "notebook read",
        False,
    )
    assert parse_tool_name("mcp__github__create_issue") == (
        ("github", "create", "issue"),
        "github create issue",
        True,
    )


@pytest.mark.asyncio
async def test_select_query_returns_alias_aware_unique_matches() -> None:
    tools = [
        _tool("ReadFile", aliases=("read",)),
        _tool("EditFile", aliases=("edit",)),
        _tool("AlreadyLoaded", should_defer=False),
    ]

    result = await run_tool_search(
        query="select:read, EditFile, AlreadyLoaded, read",
        tools=tools,
    )

    assert result.matches == ("ReadFile", "EditFile", "AlreadyLoaded")
    assert result.total_deferred_tools == 2
    assert tool_search_query_type("select:ReadFile") == "select"


@pytest.mark.asyncio
async def test_select_query_includes_pending_servers_only_on_empty_result() -> None:
    result = await run_tool_search(
        query="select:Missing",
        tools=[_tool("ReadFile")],
        pending_mcp_servers=("github",),
    )

    assert result.matches == ()
    assert result.pending_mcp_servers == ("github",)
    assert (
        map_tool_search_result_content(result)
        == "No matching deferred tools found. Some MCP servers are still connecting: "
        "github. Their tools will become available shortly - try searching again."
    )


@pytest.mark.asyncio
async def test_keyword_search_scores_name_hint_and_prompt_text() -> None:
    tools = [
        _tool("NotebookRead", prompt="Read notebook cells", search_hint="jupyter notebook"),
        _tool("SlackSend", prompt="Send a team message", search_hint="slack message"),
        _tool("FileWrite", prompt="Write local files"),
    ]

    matches = await search_tool_names(
        query="notebook jupyter",
        deferred_tools=tools,
        tools=tools,
        max_results=2,
    )

    assert matches[0] == "NotebookRead"
    assert "SlackSend" not in matches


@pytest.mark.asyncio
async def test_keyword_search_required_terms_filter_candidates() -> None:
    tools = [
        _tool("SlackSend", prompt="Send a message", search_hint="slack chat"),
        _tool("EmailSend", prompt="Send mail", search_hint="email message"),
    ]

    assert await search_tool_names(
        query="+slack send",
        deferred_tools=tools,
        tools=tools,
    ) == ("SlackSend",)


@pytest.mark.asyncio
async def test_keyword_search_exact_name_and_mcp_prefix_fast_paths() -> None:
    tools = [
        _tool("mcp__github__create_issue"),
        _tool("mcp__github__list_repos"),
        _tool("ReadFile"),
    ]

    assert await search_tool_names(
        query="readfile",
        deferred_tools=tools,
        tools=tools,
    ) == ("ReadFile",)
    assert await search_tool_names(
        query="mcp__github",
        deferred_tools=tools,
        tools=tools,
    ) == ("mcp__github__create_issue", "mcp__github__list_repos")


@pytest.mark.asyncio
async def test_tool_search_index_invalidates_when_deferred_tool_set_changes() -> None:
    calls: list[str] = []

    def counted_tool(name: str) -> Tool:
        async def prompt(_ctx: ToolPromptContext | ToolUseContext | None = None) -> str:
            calls.append(name)
            return f"{name} prompt"

        return build_tool(
            ToolSpec(
                name=name,
                description=name,
                input_model=EmptyInput,
                call=_call,
                prompt=prompt,
                should_defer=True,
            )
        )

    first = [counted_tool("AlphaRead")]
    second = [*first, counted_tool("BetaWrite")]
    index = ToolSearchIndex()

    await run_tool_search(query="prompt", tools=first, index=index)
    await run_tool_search(query="prompt", tools=first, index=index)
    await run_tool_search(query="prompt", tools=second, index=index)

    assert calls == ["AlphaRead", "AlphaRead", "BetaWrite"]
    assert deferred_tools_cache_key(second) == "AlphaRead,BetaWrite"


@pytest.mark.asyncio
async def test_prompt_context_is_passed_to_dynamic_tool_prompts() -> None:
    observed: list[ToolPromptContext] = []

    async def prompt(ctx: ToolPromptContext | ToolUseContext | None = None) -> str:
        assert isinstance(ctx, ToolPromptContext)
        observed.append(ctx)
        return "contextual prompt"

    tool = build_tool(
        ToolSpec(
            name="ContextTool",
            description="context",
            input_model=EmptyInput,
            call=_call,
            prompt=prompt,
            should_defer=True,
        )
    )
    prompt_context = ToolPromptContext(
        permission_context=empty_tool_permission_context(),
        tools=(tool,),
        agents=("reviewer",),
        allowed_agent_types=("worker",),
    )

    await run_tool_search(
        query="contextual",
        tools=[tool],
        prompt_context=prompt_context,
    )

    assert observed == [prompt_context]


@pytest.mark.asyncio
async def test_tool_search_index_invalidates_when_prompt_context_changes() -> None:
    calls: list[tuple[str, ...]] = []

    async def prompt(ctx: ToolPromptContext | ToolUseContext | None = None) -> str:
        assert isinstance(ctx, ToolPromptContext)
        calls.append(tuple(ctx.agents))
        return "alpha docs" if ctx.agents == ("alpha",) else "beta docs"

    tool = build_tool(
        ToolSpec(
            name="ContextTool",
            description="context",
            input_model=EmptyInput,
            call=_call,
            prompt=prompt,
            should_defer=True,
        )
    )
    index = ToolSearchIndex()
    alpha_context = ToolPromptContext(
        permission_context=empty_tool_permission_context(),
        tools=(tool,),
        agents=("alpha",),
    )
    beta_context = ToolPromptContext(
        permission_context=empty_tool_permission_context(),
        tools=(tool,),
        agents=("beta",),
    )

    alpha = await run_tool_search(
        query="alpha",
        tools=[tool],
        index=index,
        prompt_context=alpha_context,
    )
    beta = await run_tool_search(
        query="beta",
        tools=[tool],
        index=index,
        prompt_context=beta_context,
    )

    assert alpha.matches == ("ContextTool",)
    assert beta.matches == ("ContextTool",)
    assert calls == [("alpha",), ("beta",)]


def test_map_tool_search_result_content_returns_tool_reference_dicts() -> None:
    result = asyncio.run(
        run_tool_search(
            query="select:ReadFile",
            tools=[_tool("ReadFile")],
        )
    )

    assert map_tool_search_result_content(result) == [
        {"type": "tool_reference", "tool_name": "ReadFile"}
    ]
