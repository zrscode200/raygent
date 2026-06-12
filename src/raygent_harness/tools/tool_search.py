"""Deferred tool search/select primitives.

This module is pure and headless: it computes ToolSearch matches and result
payloads, but it does not mutate the prompt-visible tool catalog. Tool
orchestration can wrap it as a concrete tool call.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from raygent_harness.core.permissions import (
    ToolPermissionContext,
    empty_tool_permission_context,
)
from raygent_harness.core.tool import (
    Tool,
    ToolPromptContext,
    find_tool_by_name,
)
from raygent_harness.services.mcp import parse_mcp_tool_name

TOOL_SEARCH_TOOL_NAME = "ToolSearch"
DEFAULT_MAX_RESULTS = 5
TOOL_SEARCH_MAX_RESULT_SIZE_CHARS = 100_000


class ToolSearchInput(BaseModel):
    query: str = Field(
        description=(
            'Query to find deferred tools. Use "select:<tool_name>" for direct '
            "selection, or keywords to search."
        )
    )
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1)


@dataclass(frozen=True)
class ToolSearchResult:
    matches: tuple[str, ...]
    query: str
    total_deferred_tools: int
    pending_mcp_servers: tuple[str, ...] = ()

    def to_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "matches": list(self.matches),
            "query": self.query,
            "total_deferred_tools": self.total_deferred_tools,
        }
        if self.pending_mcp_servers:
            data["pending_mcp_servers"] = list(self.pending_mcp_servers)
        return data


@dataclass(frozen=True)
class ToolSearchMatch:
    name: str
    score: int


@dataclass
class ToolSearchIndex:
    """Memoized prompt cache keyed by deferred tool set and prompt context."""

    _cache_key: str | None = None
    _descriptions: dict[str, str] = field(default_factory=dict[str, str])

    def maybe_invalidate(
        self,
        deferred_tools: list[Tool],
        prompt_context: ToolPromptContext | None = None,
    ) -> None:
        cache_key = tool_search_cache_key(deferred_tools, prompt_context)
        if cache_key != self._cache_key:
            self._descriptions.clear()
            self._cache_key = cache_key

    def clear(self) -> None:
        self._cache_key = None
        self._descriptions.clear()

    async def get_tool_prompt(
        self,
        tool_name: str,
        tools: list[Tool],
        prompt_context: ToolPromptContext | None = None,
    ) -> str:
        if tool_name not in self._descriptions:
            tool = find_tool_by_name(tools, tool_name)
            if tool is None:
                self._descriptions[tool_name] = ""
            else:
                self._descriptions[tool_name] = await tool.prompt(
                    prompt_context
                    or ToolPromptContext(
                        permission_context=empty_tool_permission_context(),
                        tools=tuple(tools),
                    )
                )
        return self._descriptions[tool_name]


def is_deferred_tool(tool: Tool) -> bool:
    """Whether a tool should be hidden until ToolSearch selects it."""

    if tool.always_load:
        return False
    if tool.name == TOOL_SEARCH_TOOL_NAME:
        return False
    return tool.should_defer is True


def deferred_tools_cache_key(deferred_tools: list[Tool]) -> str:
    return ",".join(sorted(tool.name for tool in deferred_tools))


def tool_search_cache_key(
    deferred_tools: list[Tool],
    prompt_context: ToolPromptContext | None = None,
) -> str:
    return "\x1f".join(
        (
            deferred_tools_cache_key(deferred_tools),
            prompt_context_cache_key(prompt_context),
        )
    )


def prompt_context_cache_key(prompt_context: ToolPromptContext | None) -> str:
    """Stable key for prompt text that may vary by ToolPromptContext."""

    if prompt_context is None:
        return ""
    return "\x1e".join(
        (
            _permission_context_cache_key(prompt_context.permission_context),
            _names_key(tool.name for tool in prompt_context.tools),
            _names_key(prompt_context.agents),
            _names_key(prompt_context.allowed_agent_types),
            _mapping_key(prompt_context.extra),
        )
    )


def parse_tool_name(name: str) -> tuple[tuple[str, ...], str, bool]:
    """Split regular and MCP-style tool names into searchable parts."""

    mcp_info = parse_mcp_tool_name(name)
    if mcp_info is not None:
        mcp_parts = (mcp_info.server_name, mcp_info.tool_name or "")
        without_prefix = "__".join(part for part in mcp_parts if part).lower()
        parts = tuple(
            part
            for chunk in without_prefix.split("__")
            for part in chunk.split("_")
            if part
        )
        return parts, without_prefix.replace("__", " ").replace("_", " "), True

    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
    parts = tuple(part for part in spaced.lower().split() if part)
    return parts, " ".join(parts), False


async def search_tool_names(
    *,
    query: str,
    deferred_tools: list[Tool],
    tools: list[Tool],
    max_results: int = DEFAULT_MAX_RESULTS,
    index: ToolSearchIndex | None = None,
    prompt_context: ToolPromptContext | None = None,
) -> tuple[str, ...]:
    """Keyword search over deferred tool names, search hints, and prompt text."""

    query_lower = query.lower().strip()
    exact_match = _find_tool_by_lower_name(deferred_tools, query_lower) or _find_tool_by_lower_name(
        tools,
        query_lower,
    )
    if exact_match is not None:
        return (exact_match.name,)

    if query_lower.startswith("mcp__") and len(query_lower) > len("mcp__"):
        prefix_matches = tuple(
            tool.name
            for tool in deferred_tools
            if tool.name.lower().startswith(query_lower)
        )
        if prefix_matches:
            return prefix_matches[:max_results]

    query_terms = tuple(term for term in query_lower.split() if term)
    required_terms = tuple(
        term[1:] for term in query_terms if term.startswith("+") and len(term) > 1
    )
    optional_terms = tuple(
        term for term in query_terms if not (term.startswith("+") and len(term) > 1)
    )
    scoring_terms = (
        (*required_terms, *optional_terms) if required_terms else query_terms
    )
    if not scoring_terms:
        return ()

    search_index = index or ToolSearchIndex()
    search_index.maybe_invalidate(deferred_tools, prompt_context)
    candidate_tools = deferred_tools
    if required_terms:
        filtered: list[Tool] = []
        for tool in deferred_tools:
            if await _matches_required_terms(
                tool,
                required_terms,
                tools,
                search_index,
                prompt_context,
            ):
                filtered.append(tool)
        candidate_tools = filtered

    scored = [
        await _score_tool(tool, scoring_terms, tools, search_index, prompt_context)
        for tool in candidate_tools
    ]
    return tuple(
        item.name
        for item in sorted(
            (item for item in scored if item.score > 0),
            key=lambda item: item.score,
            reverse=True,
        )[:max_results]
    )


async def run_tool_search(
    *,
    query: str,
    tools: list[Tool],
    max_results: int = DEFAULT_MAX_RESULTS,
    index: ToolSearchIndex | None = None,
    pending_mcp_servers: tuple[str, ...] = (),
    prompt_context: ToolPromptContext | None = None,
) -> ToolSearchResult:
    """Run select or keyword ToolSearch and return the structured result."""

    deferred_tools = [tool for tool in tools if is_deferred_tool(tool)]
    search_index = index or ToolSearchIndex()
    search_index.maybe_invalidate(deferred_tools, prompt_context)

    select_query = _parse_select_query(query)
    if select_query is not None:
        found = _select_tools(select_query, deferred_tools, tools)
        return ToolSearchResult(
            matches=found,
            query=query,
            total_deferred_tools=len(deferred_tools),
            pending_mcp_servers=pending_mcp_servers if not found else (),
        )

    matches = await search_tool_names(
        query=query,
        deferred_tools=deferred_tools,
        tools=tools,
        max_results=max_results,
        index=search_index,
        prompt_context=prompt_context,
    )
    return ToolSearchResult(
        matches=matches,
        query=query,
        total_deferred_tools=len(deferred_tools),
        pending_mcp_servers=pending_mcp_servers if not matches else (),
    )


def map_tool_search_result_content(result: ToolSearchResult) -> str | list[dict[str, str]]:
    """Map result data to future tool-result content.

    Raygent has no SDK tool-reference block type yet, so matches are represented
    as dictionaries that can be converted by orchestration later.
    """

    if not result.matches:
        text = "No matching deferred tools found"
        if result.pending_mcp_servers:
            text += (
                ". Some MCP servers are still connecting: "
                + ", ".join(result.pending_mcp_servers)
                + ". Their tools will become available shortly - try searching again."
            )
        return text
    return [{"type": "tool_reference", "tool_name": name} for name in result.matches]


def tool_search_query_type(query: str) -> Literal["select", "keyword"]:
    return "select" if _parse_select_query(query) is not None else "keyword"


def _parse_select_query(query: str) -> tuple[str, ...] | None:
    match = re.match(r"^select:(.+)$", query, flags=re.IGNORECASE)
    if match is None:
        return None
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def _select_tools(
    requested_names: tuple[str, ...],
    deferred_tools: list[Tool],
    tools: list[Tool],
) -> tuple[str, ...]:
    found: list[str] = []
    for tool_name in requested_names:
        tool = find_tool_by_name(deferred_tools, tool_name) or find_tool_by_name(
            tools,
            tool_name,
        )
        if tool is not None and tool.name not in found:
            found.append(tool.name)
    return tuple(found)


def _find_tool_by_lower_name(tools: list[Tool], query_lower: str) -> Tool | None:
    for tool in tools:
        if tool.name.lower() == query_lower:
            return tool
    return None


async def _matches_required_terms(
    tool: Tool,
    required_terms: tuple[str, ...],
    tools: list[Tool],
    index: ToolSearchIndex,
    prompt_context: ToolPromptContext | None,
) -> bool:
    parts, _full, _is_mcp = parse_tool_name(tool.name)
    description = (
        await index.get_tool_prompt(tool.name, tools, prompt_context)
    ).lower()
    hint = (tool.search_hint or "").lower()
    return all(
        term in parts
        or any(term in part for part in parts)
        or _word_match(term, description)
        or (bool(hint) and _word_match(term, hint))
        for term in required_terms
    )


async def _score_tool(
    tool: Tool,
    terms: tuple[str, ...],
    tools: list[Tool],
    index: ToolSearchIndex,
    prompt_context: ToolPromptContext | None,
) -> ToolSearchMatch:
    parts, full, is_mcp = parse_tool_name(tool.name)
    description = (
        await index.get_tool_prompt(tool.name, tools, prompt_context)
    ).lower()
    hint = (tool.search_hint or "").lower()
    score = 0
    for term in terms:
        term_score = 0
        if term in parts:
            term_score += 12 if is_mcp else 10
        elif any(term in part for part in parts):
            term_score += 6 if is_mcp else 5

        if full.find(term) != -1 and term_score == 0:
            term_score += 3
        if hint and _word_match(term, hint):
            term_score += 4
        if _word_match(term, description):
            term_score += 2
        score += term_score
    return ToolSearchMatch(name=tool.name, score=score)


def _word_match(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _permission_context_cache_key(context: ToolPermissionContext) -> str:
    return "\x1d".join(
        (
            context.mode,
            _mapping_key(context.always_allow_rules),
            _mapping_key(context.always_deny_rules),
            _mapping_key(context.always_ask_rules),
            _mapping_key(
                {
                    path: f"{directory.path}:{directory.source}"
                    for path, directory in context.additional_working_directories.items()
                }
            ),
            str(context.is_bypass_permissions_mode_available),
            str(context.is_auto_mode_available),
            str(context.should_avoid_permission_prompts),
            str(context.await_automated_checks_before_dialog),
            context.pre_plan_mode or "",
        )
    )


def _names_key(names: Iterable[object]) -> str:
    return ",".join(str(name) for name in names)


def _mapping_key(mapping: Mapping[Any, Any]) -> str:
    return ",".join(
        f"{key}={value!r}"
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    )


__all__ = [
    "DEFAULT_MAX_RESULTS",
    "TOOL_SEARCH_MAX_RESULT_SIZE_CHARS",
    "TOOL_SEARCH_TOOL_NAME",
    "ToolSearchIndex",
    "ToolSearchInput",
    "ToolSearchMatch",
    "ToolSearchResult",
    "deferred_tools_cache_key",
    "is_deferred_tool",
    "map_tool_search_result_content",
    "parse_tool_name",
    "prompt_context_cache_key",
    "run_tool_search",
    "search_tool_names",
    "tool_search_cache_key",
    "tool_search_query_type",
]
