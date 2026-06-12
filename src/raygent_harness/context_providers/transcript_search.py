"""Context provider for bounded transcript search results."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from raygent_harness.core.context_providers import (
    ContextAgentScope,
    ContextFragment,
    ContextKind,
)
from raygent_harness.services.transcript import (
    TranscriptSearchRequest,
    TranscriptSearchResult,
    TranscriptSearchScope,
    TranscriptSearchService,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.tool import ToolUseContext


class TranscriptSearchQueryResolver(Protocol):
    def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str | None:
        ...


@dataclass(frozen=True, slots=True)
class TranscriptSearchContextProvider:
    """Attach bounded transcript search hits as non-persistent user context.

    QueryEngine context providers run before the submitted user prompt is added
    to history, so this provider intentionally uses an explicit query or query
    resolver supplied by the embedder. Prompt-derived retrieval can be layered
    later without hiding policy in the kernel.
    """

    search_service: TranscriptSearchService
    scope: TranscriptSearchScope
    query: str | None = None
    query_resolver: TranscriptSearchQueryResolver | None = None
    max_results: int = 5
    max_snippet_chars: int = 240
    max_total_snippet_chars: int = 2_000
    agent_scope: ContextAgentScope = "main"
    priority: int = 40
    context_kind: ContextKind = "memory"

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> Sequence[ContextFragment]:
        query = self.query
        if self.query_resolver is not None:
            query = self.query_resolver(config, ctx)
        if query is None or not query.strip():
            return ()

        result = await self.search_service.search(
            TranscriptSearchRequest(
                query=query,
                scope=self.scope,
                max_results=self.max_results,
                max_snippet_chars=self.max_snippet_chars,
                max_total_snippet_chars=self.max_total_snippet_chars,
            )
        )
        if not result.matches:
            return ()
        return (
            ContextFragment(
                id="transcript_search",
                source="transcript_search",
                channel="user_context",
                content=_render_transcript_search_result(query, result),
                priority=self.priority,
                agent_scope=self.agent_scope,
                kind=self.context_kind,
            ),
        )


def _render_transcript_search_result(
    query: str,
    result: TranscriptSearchResult,
) -> str:
    lines = [
        '<transcript_search_results '
        f'query_chars="{len(query)}" '
        f'match_count="{len(result.matches)}" '
        f'warning_count="{len(result.warnings)}" '
        f'truncated="{str(result.truncated).lower()}">'
    ]
    for match in result.matches:
        agent = (
            f' agent_id="{html.escape(match.agent_id)}"'
            if match.agent_id is not None
            else ""
        )
        runtime = (
            f' runtime_session_id="{html.escape(match.runtime_session_id)}"'
            if match.runtime_session_id is not None
            else ""
        )
        source_path = (
            f' source_path="{html.escape(match.source_path)}"'
            if match.source_path is not None
            else ""
        )
        lines.append(
            "  "
            f'<match session_id="{html.escape(match.session_id)}"'
            f"{runtime}{agent}"
            f' sidechain="{str(match.is_sidechain).lower()}"'
            f' role="{html.escape(match.role)}"'
            f' entry_id="{html.escape(match.entry_id)}"'
            f' score="{match.score}"'
            f' snippet_truncated="{str(match.snippet_truncated).lower()}"'
            f"{source_path}>"
            f"{html.escape(match.snippet)}"
            "</match>"
        )
    lines.append("</transcript_search_results>")
    return "\n".join(lines)


__all__ = [
    "TranscriptSearchContextProvider",
    "TranscriptSearchQueryResolver",
]
