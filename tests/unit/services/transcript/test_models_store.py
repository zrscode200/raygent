from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.state import CompactBoundary
from raygent_harness.services.compact.tool_result_budget import (
    ToolResultReplacementRecord,
)
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    ContentReplacementEntry,
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
    transcript_entry_from_dict,
    transcript_entry_to_dict,
    transcript_entry_to_json,
    transcript_path_for_scope,
)


def _msg(role: str, content: object) -> MessageParam:
    return cast(MessageParam, {"role": role, "content": content})


def test_message_entry_json_round_trip_preserves_payload_and_discriminator() -> None:
    entry = TranscriptMessageEntry(
        entry_id="e1",
        parent_entry_id="e0",
        session_id="session-1",
        runtime_session_id="runtime-1",
        agent_id="agent-1",
        is_sidechain=True,
        created_at=1.5,
        cwd="/repo",
        version="0.1.0",
        message=_msg(
            "assistant",
            [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"p": "a"}}],
        ),
        provider_message_id="provider-msg-1",
    )

    raw = transcript_entry_to_dict(entry)
    decoded = transcript_entry_from_dict(raw)

    assert raw["type"] == "message"
    assert decoded == entry


def test_compact_boundary_and_replacement_entries_round_trip() -> None:
    boundary = CompactBoundaryEntry(
        entry_id="b1",
        session_id="s",
        created_at=2.0,
        boundary=CompactBoundary(
            message_index=4,
            kind="autocompact",
            summary="summarized",
        ),
        post_compact_message_count=1,
    )
    replacements = ContentReplacementEntry(
        entry_id="r1",
        session_id="s",
        created_at=3.0,
        replacements=(
            ToolResultReplacementRecord(
                tool_use_id="toolu_1",
                replacement="[persisted]",
                path="/tmp/toolu_1.txt",
                original_size_chars=100,
            ),
        ),
    )

    assert transcript_entry_from_dict(transcript_entry_to_dict(boundary)) == boundary
    assert transcript_entry_from_dict(transcript_entry_to_dict(replacements)) == replacements


def test_sidechain_path_lives_under_parent_session_not_runtime_session(tmp_path: Path) -> None:
    base = tmp_path / "transcripts"
    scope = TranscriptScope(
        session_id="parent/session",
        agent_id="../child agent",
        is_sidechain=True,
        runtime_session_id="child-runtime",
    )

    path = transcript_path_for_scope(base, scope)

    assert path.parent == base / path.parent.parent.name / "subagents"
    assert path.parent.parent.name.startswith("parent_session-")
    assert path.name.startswith("agent-child_agent-")
    assert path.name.endswith(".jsonl")
    assert "child-runtime" not in str(path)


def test_transcript_path_rejects_empty_session_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        transcript_path_for_scope(tmp_path, TranscriptScope(session_id=""))


def test_sidechain_path_requires_agent_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires agent_id"):
        transcript_path_for_scope(
            tmp_path,
            TranscriptScope(session_id="s", is_sidechain=True),
        )


def test_transcript_path_components_avoid_unsafe_id_collisions(tmp_path: Path) -> None:
    slash_path = transcript_path_for_scope(tmp_path, TranscriptScope(session_id="a/b"))
    underscore_path = transcript_path_for_scope(tmp_path, TranscriptScope(session_id="a_b"))

    assert slash_path != underscore_path
    assert slash_path.name.startswith("a_b-")
    assert underscore_path.name == "a_b.jsonl"


@pytest.mark.asyncio
async def test_jsonl_store_append_read_round_trip_and_ordering(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    first = TranscriptMessageEntry(
        entry_id="e1",
        session_id="session-1",
        created_at=1.0,
        message=_msg("user", "hello"),
    )
    second = TranscriptMessageEntry(
        entry_id="e2",
        parent_entry_id="e1",
        session_id="session-1",
        created_at=2.0,
        message=_msg("assistant", "world"),
    )

    await store.append_many(scope, [first, second])
    await store.flush(scope)

    assert await store.read_entries(scope) == [first, second]
    path = Path(store.path_for(scope))
    assert path.read_text(encoding="utf-8").count("\n") == 2


@pytest.mark.asyncio
async def test_jsonl_store_skips_malformed_and_unknown_entries_with_warnings(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    valid = TranscriptMessageEntry(
        entry_id="e1",
        session_id="session-1",
        created_at=1.0,
        message=_msg("user", "hello"),
    )
    path = Path(store.path_for(scope))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                transcript_entry_to_json(valid),
                "{not-json",
                json.dumps({"type": "future_entry", "entry_id": "x"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = await store.read_result(scope)

    assert result.entries == (valid,)
    assert len(result.warnings) == 2
    assert "malformed JSON" in result.warnings[0]
    assert "unknown transcript entry type" in result.warnings[1]
