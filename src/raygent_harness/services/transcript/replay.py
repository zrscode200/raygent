"""Replay helpers for transcript JSONL entries."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.observability import KernelEventBus, KernelEventContext
from raygent_harness.core.state import CompactBoundary
from raygent_harness.core.tool import ContentReplacementState
from raygent_harness.services.compact.tool_result_budget import (
    ToolResultReplacementRecord,
)
from raygent_harness.services.transcript.models import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
)
from raygent_harness.services.transcript.store import (
    TranscriptReadResult,
    TranscriptReadStats,
    TranscriptStore,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionReplay:
    session_id: str
    messages: list[MessageParam]
    compact_boundaries: tuple[CompactBoundary, ...] = ()
    content_replacements: tuple[ToolResultReplacementRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    agent_id: str | None = None
    is_sidechain: bool = False
    runtime_session_id: str | None = None
    last_message_entry_id: str | None = None


@runtime_checkable
class TranscriptStoreWithReadResult(Protocol):
    async def read_result(self, scope: TranscriptScope) -> TranscriptReadResult: ...


@runtime_checkable
class TranscriptStoreWithSidechainListing(Protocol):
    async def list_sidechain_agent_ids(self, session_id: str) -> tuple[str, ...]: ...


async def load_session_replay(
    store: TranscriptStore,
    scope: TranscriptScope,
    *,
    observability: KernelEventBus | None = None,
    observability_context: KernelEventContext | None = None,
) -> SessionReplay:
    if isinstance(store, TranscriptStoreWithReadResult):
        result = await store.read_result(scope)
        entries = result.entries
        warnings = result.warnings
        read_stats = result.stats
    else:
        entries = tuple(await store.read_entries(scope))
        warnings = ()
        read_stats = TranscriptReadStats(
            decoded_entries=len(entries),
            entries_retained=len(entries),
            used_full_read_fallback=True,
        )
    replay = replay_entries(entries, scope=scope, warnings=warnings)
    if observability is not None:
        observability.emit(
            "transcript.replay.completed",
            context=_replay_observability_context(scope, observability_context),
            data={
                "message_count": len(replay.messages),
                "compact_boundary_count": len(replay.compact_boundaries),
                "content_replacement_count": len(replay.content_replacements),
                "warning_count": len(replay.warnings),
                "is_sidechain": replay.is_sidechain,
                "agent_id_present": replay.agent_id is not None,
                "transcript_total_lines_scanned": read_stats.total_lines_scanned,
                "transcript_entries_decoded": read_stats.decoded_entries,
                "transcript_entries_retained": read_stats.entries_retained,
                "transcript_entries_skipped_precompact": (
                    read_stats.entries_skipped_before_active_compact_boundary
                ),
                "transcript_used_precompact_skip": read_stats.used_precompact_skip,
                "transcript_used_full_read_fallback": (
                    read_stats.used_full_read_fallback
                ),
            },
        )
    return replay


def _replay_observability_context(
    scope: TranscriptScope,
    context: KernelEventContext | None,
) -> KernelEventContext:
    if context is None:
        return KernelEventContext(
            session_id=scope.session_id,
            runtime_session_id=scope.runtime_session_id,
            agent_id=scope.agent_id,
            source="transcript",
        )
    if (
        context.session_id is not None
        and context.runtime_session_id is not None
        and context.agent_id is not None
    ):
        return context
    return replace(
        context,
        session_id=context.session_id or scope.session_id,
        runtime_session_id=context.runtime_session_id or scope.runtime_session_id,
        agent_id=context.agent_id or scope.agent_id,
    )


async def get_agent_transcript(
    store: TranscriptStore,
    parent_session_id: str,
    agent_id: str,
) -> SessionReplay | None:
    """Load one subagent sidechain transcript for a parent session.

    Returns None when no messages exist for that agent, matching the reference
    `getAgentTranscript(...)` behavior. Content-replacement records are
    preserved on the returned `SessionReplay`.
    """

    try:
        replay = await load_session_replay(
            store,
            TranscriptScope(
                session_id=parent_session_id,
                agent_id=agent_id,
                is_sidechain=True,
            ),
        )
    except Exception as exc:
        _LOGGER.debug("failed to load subagent transcript %s: %s", agent_id, exc)
        return None
    if len(replay.messages) == 0:
        return None
    return replay


async def load_subagent_transcripts(
    store: TranscriptStore,
    parent_session_id: str,
    agent_ids: Sequence[str],
) -> dict[str, SessionReplay]:
    """Load selected subagent transcripts, skipping missing/unreadable agents."""

    transcripts: dict[str, SessionReplay] = {}
    for agent_id in agent_ids:
        try:
            replay = await get_agent_transcript(store, parent_session_id, agent_id)
        except Exception as exc:
            _LOGGER.debug("failed to load subagent transcript %s: %s", agent_id, exc)
            continue
        if replay is not None:
            transcripts[agent_id] = replay
    return transcripts


async def load_all_subagent_transcripts(
    store: TranscriptStore,
    parent_session_id: str,
) -> dict[str, SessionReplay]:
    """Load every sidechain transcript discoverable for a parent session.

    Store-wide discovery is an optional capability. Stores without an index or
    filesystem listing support return an empty mapping; callers that already
    know agent ids should use `load_subagent_transcripts(...)`.
    """

    if not isinstance(store, TranscriptStoreWithSidechainListing):
        return {}
    try:
        agent_ids = await store.list_sidechain_agent_ids(parent_session_id)
    except Exception as exc:
        _LOGGER.debug(
            "failed to list subagent transcripts for %s: %s",
            parent_session_id,
            exc,
        )
        return {}
    return await load_subagent_transcripts(store, parent_session_id, agent_ids)


def replay_entries(
    entries: Sequence[TranscriptEntry],
    *,
    scope: TranscriptScope,
    warnings: Sequence[str] = (),
) -> SessionReplay:
    replay_warnings = list(warnings)
    message_epoch: list[TranscriptMessageEntry] = []
    compact_boundaries: list[CompactBoundary] = []
    replacements: list[ToolResultReplacementRecord] = []
    pending_boundary: CompactBoundary | None = None
    pending_message_epoch: list[TranscriptMessageEntry] | None = None

    for entry in entries:
        if not _entry_matches_scope(entry, scope):
            continue
        if isinstance(entry, CompactBoundaryEntry):
            pending_boundary = entry.boundary
            pending_message_epoch = []
        elif isinstance(entry, TranscriptMessageEntry):
            if pending_message_epoch is not None:
                pending_message_epoch.append(entry)
                assert pending_boundary is not None
                compact_boundaries.append(pending_boundary)
                message_epoch = pending_message_epoch
                pending_boundary = None
                pending_message_epoch = None
            else:
                message_epoch.append(entry)
        elif isinstance(entry, ContentReplacementEntry):
            replacements.extend(entry.replacements)

    if pending_boundary is not None:
        replay_warnings.append("ignored dangling compact boundary with no messages")

    chain_entries, chain_warnings = reconstruct_message_chain(message_epoch)
    replay_warnings.extend(chain_warnings)
    runtime_session_id = scope.runtime_session_id
    if runtime_session_id is None:
        runtime_session_id = _runtime_session_id_from_chain(chain_entries)
    return SessionReplay(
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        is_sidechain=scope.is_sidechain,
        runtime_session_id=runtime_session_id,
        messages=[_clone_message(entry.message) for entry in chain_entries],
        compact_boundaries=tuple(compact_boundaries),
        content_replacements=tuple(replacements),
        warnings=tuple(replay_warnings),
        last_message_entry_id=chain_entries[-1].entry_id if chain_entries else None,
    )


def _entry_matches_scope(entry: TranscriptEntry, scope: TranscriptScope) -> bool:
    if isinstance(entry, TranscriptMessageEntry):
        if entry.session_id != scope.session_id or entry.is_sidechain != scope.is_sidechain:
            return False
        if scope.agent_id is None:
            return entry.agent_id is None
        return entry.agent_id == scope.agent_id
    if isinstance(entry, CompactBoundaryEntry | ContentReplacementEntry):
        return entry.session_id == scope.session_id and entry.agent_id == scope.agent_id
    return True


def reconstruct_message_chain(
    entries: Sequence[TranscriptMessageEntry],
) -> tuple[list[TranscriptMessageEntry], tuple[str, ...]]:
    """Reconstruct the latest message chain from envelope parent ids."""

    if len(entries) == 0:
        return [], ()
    if all(
        entry.parent_entry_id is None and entry.logical_parent_entry_id is None
        for entry in entries
    ):
        return list(entries), ()

    by_id: dict[str, TranscriptMessageEntry] = {entry.entry_id: entry for entry in entries}
    leaf = entries[-1]
    current = leaf
    reversed_chain: list[TranscriptMessageEntry] = []
    seen: set[str] = set()
    warnings: list[str] = []

    while True:
        if current.entry_id in seen:
            warnings.append(f"message chain cycle detected at {current.entry_id}")
            break
        seen.add(current.entry_id)
        reversed_chain.append(current)
        parent_id = current.parent_entry_id or current.logical_parent_entry_id
        if parent_id is None:
            break
        parent = by_id.get(parent_id)
        if parent is None:
            warnings.append(
                f"message chain for {leaf.entry_id} stops at missing parent {parent_id}"
            )
            break
        current = parent

    chain = list(reversed(reversed_chain))
    if len(chain) < len(entries):
        warnings.append("message chain excluded off-chain transcript messages")
    return chain, tuple(warnings)


def _runtime_session_id_from_chain(
    entries: Sequence[TranscriptMessageEntry],
) -> str | None:
    for entry in reversed(entries):
        if entry.runtime_session_id is not None:
            return entry.runtime_session_id
    return None


def content_replacement_state_from_replay(
    replay: SessionReplay,
    *,
    max_result_size_chars: int,
    replaced_outputs_dir: str,
) -> ContentReplacementState:
    seen_ids = _tool_result_ids_from_messages(replay.messages)
    replacements = {
        record.tool_use_id: record.replacement
        for record in replay.content_replacements
        if record.tool_use_id in seen_ids
    }
    return ContentReplacementState(
        max_result_size_chars=max_result_size_chars,
        replaced_outputs_dir=replaced_outputs_dir,
        replacements=replacements,
        seen_ids=seen_ids,
    )


def _tool_result_ids_from_messages(messages: Sequence[MessageParam]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            block_content = block.get("content")
            if isinstance(tool_use_id, str) and block_content not in (None, ""):
                ids.add(tool_use_id)
    return ids


def _clone_message(message: MessageParam) -> MessageParam:
    return deepcopy(message)


__all__ = [
    "SessionReplay",
    "TranscriptStoreWithReadResult",
    "TranscriptStoreWithSidechainListing",
    "content_replacement_state_from_replay",
    "get_agent_transcript",
    "load_all_subagent_transcripts",
    "load_session_replay",
    "load_subagent_transcripts",
    "reconstruct_message_chain",
    "replay_entries",
]
