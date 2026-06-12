from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.state import CompactBoundary
from raygent_harness.services.compact import PERSISTED_TOOL_RESULT_TAG
from raygent_harness.services.compact.tool_result_budget import (
    ToolResultReplacementRecord,
)
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    JsonlTranscriptStore,
    StreamEventEntry,
    TombstoneEntry,
    TranscriptEntry,
    TranscriptMessageEntry,
    TranscriptScope,
    content_replacement_state_from_replay,
    get_agent_transcript,
    load_all_subagent_transcripts,
    load_session_replay,
    load_subagent_transcripts,
    replay_entries,
    transcript_entry_to_json,
)


def _msg(content: str) -> MessageParam:
    return cast(MessageParam, {"role": "user", "content": content})


def _tool_result(tool_use_id: str, content: object) -> MessageParam:
    return cast(
        MessageParam,
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        },
    )


def _entry(
    entry_id: str,
    content: str,
    *,
    parent_entry_id: str | None = None,
    session_id: str = "s",
) -> TranscriptMessageEntry:
    return TranscriptMessageEntry(
        entry_id=entry_id,
        parent_entry_id=parent_entry_id,
        session_id=session_id,
        created_at=1.0,
        message=_msg(content),
    )


class _RaisingReadStore:
    async def append(self, scope: TranscriptScope, entry: TranscriptEntry) -> None:
        _ = (scope, entry)
        raise AssertionError("not used")

    async def append_many(
        self,
        scope: TranscriptScope,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        _ = (scope, entries)
        raise AssertionError("not used")

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        _ = scope
        raise OSError("cannot read")

    async def flush(self, scope: TranscriptScope | None = None) -> None:
        _ = scope
        return

    def path_for(self, scope: TranscriptScope) -> str | None:
        _ = scope
        return None


class _SelectiveReadStore(_RaisingReadStore):
    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        if scope.agent_id == "bad-agent":
            raise OSError("bad agent")
        return [
            TranscriptMessageEntry(
                entry_id=f"m-{scope.agent_id}",
                session_id=scope.session_id,
                agent_id=scope.agent_id,
                is_sidechain=scope.is_sidechain,
                message=_msg(str(scope.agent_id)),
            )
        ]


class _ListReadStore(_RaisingReadStore):
    def __init__(self, entries: Sequence[TranscriptEntry]) -> None:
        self._entries = list(entries)

    async def read_entries(self, scope: TranscriptScope) -> list[TranscriptEntry]:
        _ = scope
        return list(self._entries)


class _RaisingListingStore(_SelectiveReadStore):
    async def list_sidechain_agent_ids(self, session_id: str) -> tuple[str, ...]:
        _ = session_id
        raise OSError("cannot list")


def test_replay_reconstructs_latest_parent_chain_and_drops_off_chain_siblings() -> None:
    scope = TranscriptScope(session_id="s")
    entries = [
        _entry("a", "a"),
        _entry("b", "b", parent_entry_id="a"),
        _entry("orphan", "orphan", parent_entry_id="a"),
        _entry("c", "c", parent_entry_id="b"),
    ]

    replay = replay_entries(entries, scope=scope)

    assert [message["content"] for message in replay.messages] == ["a", "b", "c"]
    assert replay.warnings == ("message chain excluded off-chain transcript messages",)


def test_replay_uses_latest_compact_boundary_as_message_epoch() -> None:
    scope = TranscriptScope(session_id="s")
    boundary = CompactBoundary(
        message_index=10,
        kind="autocompact",
        summary="summary",
    )

    replay = replay_entries(
        [
            _entry("old", "old"),
            CompactBoundaryEntry(
                entry_id="boundary",
                session_id="s",
                created_at=2.0,
                boundary=boundary,
                post_compact_message_count=2,
            ),
            _entry("summary", "summary"),
            _entry("tail", "tail", parent_entry_id="summary"),
        ],
        scope=scope,
    )

    assert [message["content"] for message in replay.messages] == ["summary", "tail"]
    assert replay.compact_boundaries == (boundary,)


def test_replay_ignores_dangling_compact_boundary_without_messages() -> None:
    scope = TranscriptScope(session_id="s")
    boundary = CompactBoundary(
        message_index=10,
        kind="autocompact",
        summary="summary",
    )

    replay = replay_entries(
        [
            _entry("old", "old"),
            CompactBoundaryEntry(
                entry_id="boundary",
                session_id="s",
                created_at=2.0,
                boundary=boundary,
                post_compact_message_count=2,
            ),
        ],
        scope=scope,
    )

    assert [message["content"] for message in replay.messages] == ["old"]
    assert replay.compact_boundaries == ()
    assert replay.warnings == ("ignored dangling compact boundary with no messages",)


def test_replay_reconstructs_content_replacement_state() -> None:
    scope = TranscriptScope(session_id="s")
    record = ToolResultReplacementRecord(
        tool_use_id="toolu_1",
        replacement="[tool result persisted]",
        path="/tmp/toolu_1.txt",
        original_size_chars=500,
    )

    replay = replay_entries(
        [
            ContentReplacementEntry(
                entry_id="r1",
                session_id="s",
                created_at=1.0,
                replacements=(record,),
            ),
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="s",
                created_at=2.0,
                message=_tool_result(
                    "toolu_1",
                    f"{PERSISTED_TOOL_RESULT_TAG}\npath: /tmp/toolu_1.txt",
                ),
            ),
            TranscriptMessageEntry(
                entry_id="m2",
                session_id="s",
                created_at=3.0,
                message=_tool_result("toolu_unreplaced", "small output"),
            ),
        ],
        scope=scope,
    )
    state = content_replacement_state_from_replay(
        replay,
        max_result_size_chars=50_000,
        replaced_outputs_dir="/tmp/outputs",
    )

    assert replay.content_replacements == (record,)
    assert state.replacements == {"toolu_1": "[tool result persisted]"}
    assert state.seen_ids == {"toolu_1", "toolu_unreplaced"}
    assert state.replaced_outputs_dir == "/tmp/outputs"


def test_replay_skips_replacement_records_for_ids_absent_after_compaction() -> None:
    scope = TranscriptScope(session_id="s")
    stale_record = ToolResultReplacementRecord(
        tool_use_id="toolu_before_compact",
        replacement="stale",
        path="/tmp/stale.txt",
        original_size_chars=100,
    )

    replay = replay_entries(
        [
            ContentReplacementEntry(
                entry_id="r1",
                session_id="s",
                replacements=(stale_record,),
            ),
            _entry("m1", "summary without tool result"),
        ],
        scope=scope,
    )
    state = content_replacement_state_from_replay(
        replay,
        max_result_size_chars=50_000,
        replaced_outputs_dir="/tmp/outputs",
    )

    assert state.replacements == {}
    assert state.seen_ids == set()


def test_replay_filters_entries_by_sidechain_scope() -> None:
    scope = TranscriptScope(session_id="s", agent_id="agent-1", is_sidechain=True)
    wanted_record = ToolResultReplacementRecord(
        tool_use_id="toolu_wanted",
        replacement="wanted",
        path="/tmp/wanted.txt",
        original_size_chars=10,
    )
    other_record = ToolResultReplacementRecord(
        tool_use_id="toolu_other",
        replacement="other",
        path="/tmp/other.txt",
        original_size_chars=20,
    )

    replay = replay_entries(
        [
            TranscriptMessageEntry(
                entry_id="main",
                session_id="s",
                is_sidechain=False,
                message=_msg("main"),
            ),
            TranscriptMessageEntry(
                entry_id="wanted",
                session_id="s",
                agent_id="agent-1",
                is_sidechain=True,
                message=_msg("wanted"),
            ),
            TranscriptMessageEntry(
                entry_id="other-agent",
                session_id="s",
                agent_id="agent-2",
                is_sidechain=True,
                message=_msg("other"),
            ),
            ContentReplacementEntry(
                entry_id="r1",
                session_id="s",
                agent_id="agent-1",
                replacements=(wanted_record,),
            ),
            ContentReplacementEntry(
                entry_id="r2",
                session_id="s",
                agent_id="agent-2",
                replacements=(other_record,),
            ),
        ],
        scope=scope,
    )

    assert [message["content"] for message in replay.messages] == ["wanted"]
    assert replay.content_replacements == (wanted_record,)


@pytest.mark.asyncio
async def test_load_session_replay_preserves_store_read_warnings(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="s")
    path = Path(store.path_for(scope))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        transcript_entry_to_json(_entry("m1", "ok")) + "\n{not-json\n",
        encoding="utf-8",
    )

    replay = await load_session_replay(store, scope)

    assert [message["content"] for message in replay.messages] == ["ok"]
    assert len(replay.warnings) == 1
    assert "malformed JSON" in replay.warnings[0]


@pytest.mark.asyncio
async def test_load_session_replay_reports_full_read_fallback_for_plain_store() -> None:
    sink = RecordingKernelEventSink()
    replay = await load_session_replay(
        _ListReadStore([_entry("m1", "ok")]),
        TranscriptScope(session_id="s"),
        observability=KernelEventBus([sink]),
    )

    event = sink.by_type("transcript.replay.completed")[0]
    assert replay.messages == [_msg("ok")]
    assert event.data["transcript_entries_decoded"] == 1
    assert event.data["transcript_entries_retained"] == 1
    assert event.data["transcript_used_precompact_skip"] is False
    assert event.data["transcript_used_full_read_fallback"] is True


@pytest.mark.asyncio
async def test_load_session_replay_uses_compact_aware_read_result(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="s")
    old_content = "old" * 100_000
    boundary = CompactBoundary(
        message_index=10,
        kind="autocompact",
        summary="summary",
    )
    replacement = ToolResultReplacementRecord(
        tool_use_id="toolu_after",
        replacement="[persisted after compact]",
        path="/tmp/toolu_after.txt",
        original_size_chars=100,
    )
    stream_entry = StreamEventEntry(
        entry_id="stream-after",
        session_id="s",
        event={"type": "message_delta", "shape": "text"},
    )
    tombstone_entry = TombstoneEntry(
        entry_id="tombstone-after",
        session_id="s",
        target_entry_id="stale-stream",
        reason="fallback",
    )
    await store.append_many(
        scope,
        [
            _entry("old", old_content),
            CompactBoundaryEntry(
                entry_id="boundary",
                session_id="s",
                created_at=2.0,
                boundary=boundary,
                post_compact_message_count=2,
            ),
            ContentReplacementEntry(
                entry_id="replacement-after",
                session_id="s",
                replacements=(replacement,),
            ),
            stream_entry,
            tombstone_entry,
            _entry("summary", "summary"),
            _entry("tail", "tail", parent_entry_id="summary"),
        ],
    )

    result = await store.read_result(scope)
    sink = RecordingKernelEventSink()
    replay = await load_session_replay(
        store,
        scope,
        observability=KernelEventBus([sink]),
    )
    retained_message_contents = [
        entry.message["content"]
        for entry in result.entries
        if isinstance(entry, TranscriptMessageEntry)
    ]

    assert old_content not in retained_message_contents
    assert [message["content"] for message in replay.messages] == ["summary", "tail"]
    assert replay.compact_boundaries == (boundary,)
    assert replay.content_replacements == (replacement,)
    assert stream_entry in result.entries
    assert tombstone_entry in result.entries
    assert result.stats.total_lines_scanned == 7
    assert result.stats.decoded_entries == 6
    assert result.stats.entries_retained == 6
    assert result.stats.entries_skipped_before_active_compact_boundary == 1
    assert result.stats.used_precompact_skip is True
    assert result.stats.used_full_read_fallback is False
    replay_event = sink.by_type("transcript.replay.completed")[0]
    assert replay_event.data["transcript_total_lines_scanned"] == 7
    assert replay_event.data["transcript_entries_decoded"] == 6
    assert replay_event.data["transcript_entries_retained"] == 6
    assert replay_event.data["transcript_entries_skipped_precompact"] == 1
    assert replay_event.data["transcript_used_precompact_skip"] is True
    assert replay_event.data["transcript_used_full_read_fallback"] is False

    raw_entries = await store.read_entries(scope)
    assert raw_entries[0] == _entry("old", old_content)


@pytest.mark.asyncio
async def test_compact_aware_read_does_not_decode_stale_preboundary_lines(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="s")
    boundary = CompactBoundaryEntry(
        entry_id="boundary",
        session_id="s",
        boundary=CompactBoundary(
            message_index=10,
            kind="autocompact",
            summary="summary",
        ),
    )
    summary = _entry("summary", "summary")
    path = Path(store.path_for(scope))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '{"type": "message", "message": "' + ("stale" * 100_000),
                transcript_entry_to_json(boundary),
                transcript_entry_to_json(summary),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = await store.read_result(scope)
    replay = await load_session_replay(store, scope)

    assert result.entries == (boundary, summary)
    assert result.warnings == ()
    assert replay.messages == [_msg("summary")]
    assert replay.warnings == ()
    assert result.stats.total_lines_scanned == 3
    assert result.stats.decoded_entries == 2
    assert result.stats.entries_skipped_before_active_compact_boundary == 1
    assert result.stats.used_precompact_skip is True


@pytest.mark.asyncio
async def test_load_session_replay_preserves_dangling_compact_boundary_warning(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="s")
    boundary = CompactBoundary(
        message_index=10,
        kind="autocompact",
        summary="summary",
    )
    old_entry = _entry("old", "old")
    boundary_entry = CompactBoundaryEntry(
        entry_id="boundary",
        session_id="s",
        boundary=boundary,
    )
    await store.append_many(
        scope,
        [old_entry, boundary_entry],
    )

    result = await store.read_result(scope)
    replay = await load_session_replay(store, scope)

    assert result.entries == (old_entry, boundary_entry)
    assert result.stats.used_precompact_skip is False
    assert result.stats.entries_skipped_before_active_compact_boundary == 0
    assert [message["content"] for message in replay.messages] == ["old"]
    assert replay.warnings == ("ignored dangling compact boundary with no messages",)


@pytest.mark.asyncio
async def test_sidechain_replay_uses_compact_aware_read_result(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="parent", agent_id="agent-1", is_sidechain=True)
    boundary = CompactBoundary(
        message_index=4,
        kind="microcompact",
        summary="child summary",
    )
    await store.append_many(
        scope,
        [
            TranscriptMessageEntry(
                entry_id="child-old",
                session_id="parent",
                agent_id="agent-1",
                is_sidechain=True,
                message=_msg("child old"),
            ),
            CompactBoundaryEntry(
                entry_id="child-boundary",
                session_id="parent",
                agent_id="agent-1",
                boundary=boundary,
            ),
            TranscriptMessageEntry(
                entry_id="child-summary",
                session_id="parent",
                runtime_session_id="runtime-child",
                agent_id="agent-1",
                is_sidechain=True,
                message=_msg("child summary"),
            ),
        ],
    )

    result = await store.read_result(scope)
    replay = await get_agent_transcript(store, "parent", "agent-1")

    assert replay is not None
    assert replay.messages == [_msg("child summary")]
    assert replay.compact_boundaries == (boundary,)
    assert replay.runtime_session_id == "runtime-child"
    assert result.stats.entries_skipped_before_active_compact_boundary == 1


@pytest.mark.asyncio
async def test_get_agent_transcript_loads_sidechain_with_replacements(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="parent", agent_id="agent-1", is_sidechain=True)
    record = ToolResultReplacementRecord(
        tool_use_id="toolu_1",
        replacement="persisted",
        path="/tmp/toolu_1.txt",
        original_size_chars=500,
    )
    await store.append_many(
        scope,
        [
            TranscriptMessageEntry(
                entry_id="m1",
                session_id="parent",
                runtime_session_id="child-runtime",
                agent_id="agent-1",
                is_sidechain=True,
                message=_msg("child prompt"),
            ),
            TranscriptMessageEntry(
                entry_id="m2",
                parent_entry_id="m1",
                session_id="parent",
                runtime_session_id="child-runtime",
                agent_id="agent-1",
                is_sidechain=True,
                message=_msg("child answer"),
            ),
            ContentReplacementEntry(
                entry_id="r1",
                session_id="parent",
                agent_id="agent-1",
                replacements=(record,),
            ),
        ],
    )

    replay = await get_agent_transcript(store, "parent", "agent-1")

    assert replay is not None
    assert replay.is_sidechain is True
    assert replay.agent_id == "agent-1"
    assert replay.runtime_session_id == "child-runtime"
    assert [message["content"] for message in replay.messages] == [
        "child prompt",
        "child answer",
    ]
    assert replay.content_replacements == (record,)


@pytest.mark.asyncio
async def test_get_agent_transcript_returns_none_for_missing_agent(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)

    replay = await get_agent_transcript(store, "parent", "missing-agent")

    assert replay is None


@pytest.mark.asyncio
async def test_get_agent_transcript_returns_none_for_read_failure() -> None:
    replay = await get_agent_transcript(_RaisingReadStore(), "parent", "agent-1")

    assert replay is None


@pytest.mark.asyncio
async def test_load_subagent_transcripts_filters_selected_agents(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    for agent_id in ("agent-a", "agent-b"):
        scope = TranscriptScope(session_id="parent", agent_id=agent_id, is_sidechain=True)
        await store.append(
            scope,
            TranscriptMessageEntry(
                entry_id=f"m-{agent_id}",
                session_id="parent",
                agent_id=agent_id,
                is_sidechain=True,
                message=_msg(agent_id),
            ),
        )

    transcripts = await load_subagent_transcripts(
        store,
        "parent",
        ["agent-b", "missing"],
    )

    assert set(transcripts) == {"agent-b"}
    assert transcripts["agent-b"].messages == [_msg("agent-b")]


@pytest.mark.asyncio
async def test_load_subagent_transcripts_skips_per_agent_failures() -> None:
    transcripts = await load_subagent_transcripts(
        _SelectiveReadStore(),
        "parent",
        ["good-agent", "bad-agent"],
    )

    assert set(transcripts) == {"good-agent"}
    assert transcripts["good-agent"].messages == [_msg("good-agent")]


@pytest.mark.asyncio
async def test_load_all_subagent_transcripts_uses_store_sidechain_listing(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    for agent_id in ("agent-a", "agent-b"):
        scope = TranscriptScope(session_id="parent", agent_id=agent_id, is_sidechain=True)
        await store.append(
            scope,
            TranscriptMessageEntry(
                entry_id=f"m-{agent_id}",
                session_id="parent",
                agent_id=agent_id,
                is_sidechain=True,
                message=_msg(agent_id),
            ),
        )

    transcripts = await load_all_subagent_transcripts(store, "parent")

    assert set(transcripts) == {"agent-a", "agent-b"}
    assert transcripts["agent-a"].messages == [_msg("agent-a")]
    assert transcripts["agent-b"].messages == [_msg("agent-b")]


@pytest.mark.asyncio
async def test_load_all_subagent_transcripts_returns_empty_on_listing_failure() -> None:
    transcripts = await load_all_subagent_transcripts(_RaisingListingStore(), "parent")

    assert transcripts == {}
