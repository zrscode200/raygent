"""Bounded transcript/session search.

This module is a headless retrieval primitive over Raygent transcript stores.
It is intentionally provider-neutral and UI-free: callers choose the query and
scope, and the service returns bounded snippets plus metadata rather than raw
JSONL entries or full transcripts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

from raygent_harness.core.messages import MessageParam
from raygent_harness.services.transcript.models import TranscriptMessageEntry, TranscriptScope
from raygent_harness.services.transcript.replay import TranscriptStoreWithSidechainListing
from raygent_harness.services.transcript.store import (
    TranscriptReadResult,
    TranscriptReadStats,
    TranscriptStore,
)

TranscriptSearchCompactMode = Literal["active", "full"]
TranscriptSearchOrder = Literal["newest_first", "oldest_first"]


@runtime_checkable
class TranscriptStoreWithReadResult(Protocol):
    async def read_result(self, scope: TranscriptScope) -> TranscriptReadResult: ...


@dataclass(frozen=True, slots=True)
class TranscriptSearchScope:
    """Search scope rooted at one Raygent parent session."""

    session_id: str
    runtime_session_id: str | None = None
    include_main: bool = True
    sidechain_agent_ids: tuple[str, ...] = ()
    include_all_sidechains: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptSearchRequest:
    """Bounded transcript search request."""

    query: str
    scope: TranscriptSearchScope
    max_results: int = 5
    max_snippet_chars: int = 240
    max_total_snippet_chars: int = 2_000
    case_sensitive: bool = False
    compact_mode: TranscriptSearchCompactMode = "active"
    order: TranscriptSearchOrder = "newest_first"
    roles: tuple[str, ...] = ("user", "assistant")

    def __post_init__(self) -> None:
        if self.max_results < 1:
            raise ValueError("TranscriptSearchRequest.max_results must be >= 1")
        if self.max_snippet_chars < 16:
            raise ValueError("TranscriptSearchRequest.max_snippet_chars must be >= 16")
        if self.max_total_snippet_chars < self.max_snippet_chars:
            raise ValueError(
                "TranscriptSearchRequest.max_total_snippet_chars must be >= "
                "max_snippet_chars"
            )


@dataclass(frozen=True, slots=True)
class TranscriptSearchMatch:
    """One bounded transcript match."""

    session_id: str
    entry_id: str
    role: str
    snippet: str
    score: int
    order: int
    created_at: float
    runtime_session_id: str | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
    source_path: str | None = None
    snippet_truncated: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptSearchResult:
    """Search result plus metadata/warnings."""

    matches: tuple[TranscriptSearchMatch, ...] = ()
    warnings: tuple[str, ...] = ()
    scopes_searched: tuple[TranscriptScope, ...] = ()
    scanned_entry_count: int = 0
    matched_entry_count: int = 0
    dropped_match_count: int = 0
    truncated: bool = False
    read_stats: tuple[TranscriptReadStats, ...] = ()


class TranscriptSearchService:
    """Search Raygent transcript stores without exposing raw JSONL."""

    def __init__(self, store: TranscriptStore) -> None:
        self._store = store

    async def search(self, request: TranscriptSearchRequest) -> TranscriptSearchResult:
        query = request.query.strip()
        if not query:
            return TranscriptSearchResult()

        scopes, scope_warnings = await _resolve_search_scopes(self._store, request.scope)
        warnings = list(scope_warnings)
        candidates: list[TranscriptSearchMatch] = []
        read_stats: list[TranscriptReadStats] = []
        scanned_entry_count = 0

        for scope_index, scope in enumerate(scopes):
            read = await _read_scope(self._store, scope, compact_mode=request.compact_mode)
            warnings.extend(read.warnings)
            read_stats.append(read.stats)
            path = _safe_path_for(self._store, scope)
            for entry in read.entries:
                if not isinstance(entry, TranscriptMessageEntry):
                    continue
                if not _entry_matches_search_scope(entry, scope):
                    continue
                if request.roles and entry.message.get("role") not in request.roles:
                    continue
                scanned_entry_count += 1
                text = _message_text(entry.message)
                match_index = _find_match(text, query, case_sensitive=request.case_sensitive)
                if match_index < 0:
                    continue
                candidates.append(
                    _build_match(
                        entry,
                        text=text,
                        match_index=match_index,
                        request=request,
                        order=len(candidates),
                        scope_index=scope_index,
                        source_path=path,
                    )
                )

        ordered = _order_matches(candidates, request.order)
        bounded = _bound_matches(ordered, request)
        return TranscriptSearchResult(
            matches=tuple(bounded.matches),
            warnings=tuple(warnings),
            scopes_searched=tuple(scopes),
            scanned_entry_count=scanned_entry_count,
            matched_entry_count=len(candidates),
            dropped_match_count=max(0, len(candidates) - len(bounded.matches)),
            truncated=bounded.truncated or len(candidates) > len(bounded.matches),
            read_stats=tuple(read_stats),
        )


@dataclass(frozen=True, slots=True)
class _BoundedMatches:
    matches: tuple[TranscriptSearchMatch, ...]
    truncated: bool = False


async def _resolve_search_scopes(
    store: TranscriptStore,
    scope: TranscriptSearchScope,
) -> tuple[tuple[TranscriptScope, ...], tuple[str, ...]]:
    scopes: list[TranscriptScope] = []
    warnings: list[str] = []
    if scope.include_main:
        scopes.append(
            TranscriptScope(
                session_id=scope.session_id,
                runtime_session_id=scope.runtime_session_id,
            )
        )

    sidechain_ids = list(scope.sidechain_agent_ids)
    if scope.include_all_sidechains:
        if isinstance(store, TranscriptStoreWithSidechainListing):
            try:
                discovered = await store.list_sidechain_agent_ids(scope.session_id)
            except Exception as exc:
                warnings.append(f"sidechain listing failed: {type(exc).__name__}")
                discovered = ()
            for agent_id in discovered:
                if agent_id not in sidechain_ids:
                    sidechain_ids.append(agent_id)
        else:
            warnings.append("sidechain listing unsupported")

    for agent_id in sidechain_ids:
        scopes.append(
            TranscriptScope(
                session_id=scope.session_id,
                agent_id=agent_id,
                is_sidechain=True,
                runtime_session_id=scope.runtime_session_id,
            )
        )
    return tuple(scopes), tuple(warnings)


async def _read_scope(
    store: TranscriptStore,
    scope: TranscriptScope,
    *,
    compact_mode: TranscriptSearchCompactMode,
) -> TranscriptReadResult:
    if compact_mode == "active" and isinstance(store, TranscriptStoreWithReadResult):
        return await store.read_result(scope)
    entries = tuple(await store.read_entries(scope))
    return TranscriptReadResult(
        entries=entries,
        stats=TranscriptReadStats(
            decoded_entries=len(entries),
            entries_retained=len(entries),
            used_full_read_fallback=True,
        ),
    )


def _entry_matches_search_scope(
    entry: TranscriptMessageEntry,
    scope: TranscriptScope,
) -> bool:
    if entry.session_id != scope.session_id or entry.is_sidechain != scope.is_sidechain:
        return False
    if (
        scope.runtime_session_id is not None
        and entry.runtime_session_id != scope.runtime_session_id
    ):
        return False
    if scope.agent_id is None:
        return entry.agent_id is None
    return entry.agent_id == scope.agent_id


def _find_match(text: str, query: str, *, case_sensitive: bool) -> int:
    if case_sensitive:
        return text.find(query)
    return text.lower().find(query.lower())


def _build_match(
    entry: TranscriptMessageEntry,
    *,
    text: str,
    match_index: int,
    request: TranscriptSearchRequest,
    order: int,
    scope_index: int,
    source_path: str | None,
) -> TranscriptSearchMatch:
    snippet, truncated = _snippet_around(
        text,
        query=request.query,
        match_index=match_index,
        max_chars=request.max_snippet_chars,
        case_sensitive=request.case_sensitive,
    )
    score = _score_match(entry, scope_index=scope_index, match_index=match_index)
    return TranscriptSearchMatch(
        session_id=entry.session_id,
        runtime_session_id=entry.runtime_session_id,
        agent_id=entry.agent_id,
        is_sidechain=entry.is_sidechain,
        entry_id=entry.entry_id,
        role=str(entry.message.get("role", "")),
        snippet=snippet,
        snippet_truncated=truncated,
        score=score,
        order=order,
        created_at=entry.created_at,
        source_path=source_path,
    )


def _score_match(
    entry: TranscriptMessageEntry,
    *,
    scope_index: int,
    match_index: int,
) -> int:
    recency_bonus = min(max(int(entry.created_at), 0), 1_000_000)
    role_bonus = 50 if entry.message.get("role") == "user" else 0
    return recency_bonus + role_bonus - scope_index - min(match_index, 10_000)


def _order_matches(
    matches: Sequence[TranscriptSearchMatch],
    order: TranscriptSearchOrder,
) -> tuple[TranscriptSearchMatch, ...]:
    if order == "newest_first":
        return tuple(sorted(matches, key=lambda match: (-match.created_at, match.order)))
    return tuple(sorted(matches, key=lambda match: (match.created_at, match.order)))


def _bound_matches(
    matches: Sequence[TranscriptSearchMatch],
    request: TranscriptSearchRequest,
) -> _BoundedMatches:
    kept: list[TranscriptSearchMatch] = []
    total_chars = 0
    truncated = False
    for match in matches:
        if len(kept) >= request.max_results:
            truncated = True
            break
        next_total = total_chars + len(match.snippet)
        if next_total > request.max_total_snippet_chars:
            truncated = True
            break
        kept.append(match)
        total_chars = next_total
    return _BoundedMatches(matches=tuple(kept), truncated=truncated)


def _snippet_around(
    text: str,
    *,
    query: str,
    match_index: int,
    max_chars: int,
    case_sensitive: bool,
) -> tuple[str, bool]:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized, False
    normalized_query = _normalize_whitespace(query)
    normalized_match_index = (
        _find_match(normalized, normalized_query, case_sensitive=case_sensitive)
        if normalized_query
        else -1
    )
    center = (
        normalized_match_index
        if normalized_match_index >= 0
        else min(match_index, len(normalized) - 1)
    )
    start = max(0, center - max_chars // 3)
    end = min(len(normalized), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(normalized) else ""
    body_budget = max_chars - len(prefix) - len(suffix)
    if body_budget < 1:
        return normalized[:max_chars], True
    snippet = normalized[start : start + body_budget].strip()
    snippet = f"{prefix}{snippet}{suffix}"
    return snippet, True


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _message_text(message: MessageParam) -> str:
    message_mapping = cast(Mapping[str, object], message)
    content = message_mapping.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    blocks = cast(list[object], content)
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        block_mapping = cast(Mapping[str, object], block)
        block_type = block_mapping.get("type")
        if block_type == "text":
            text = block_mapping.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "tool_result":
            result = block_mapping.get("content")
            if isinstance(result, str):
                parts.append(result)
        elif block_type == "tool_use":
            name = block_mapping.get("name")
            if isinstance(name, str):
                parts.append(f"tool_use:{name}")
    return "\n".join(parts)


def _safe_path_for(store: TranscriptStore, scope: TranscriptScope) -> str | None:
    try:
        return store.path_for(scope)
    except Exception:
        return None


__all__ = [
    "TranscriptSearchCompactMode",
    "TranscriptSearchMatch",
    "TranscriptSearchOrder",
    "TranscriptSearchRequest",
    "TranscriptSearchResult",
    "TranscriptSearchScope",
    "TranscriptSearchService",
]
