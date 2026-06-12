from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.state import CompactBoundary
from raygent_harness.services.transcript import (
    CompactBoundaryEntry,
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
    TranscriptSearchRequest,
    TranscriptSearchScope,
    TranscriptSearchService,
    transcript_entry_to_json,
)


@pytest.mark.asyncio
async def test_transcript_search_returns_bounded_snippets_without_full_transcript(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    long_secret_tail = " SECRET_TAIL_SHOULD_NOT_LEAK" * 20
    await store.append(
        scope,
        TranscriptMessageEntry(
            entry_id="m1",
            session_id="session-1",
            created_at=1.0,
            message={
                "role": "assistant",
                "content": f"before words needle useful context{long_secret_tail}",
            },
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
            max_results=1,
            max_snippet_chars=48,
            max_total_snippet_chars=48,
        )
    )

    assert len(result.matches) == 1
    assert "needle" in result.matches[0].snippet
    assert len(result.matches[0].snippet) <= 54  # includes ellipsis
    assert "SECRET_TAIL_SHOULD_NOT_LEAK" not in result.matches[0].snippet
    assert result.matches[0].snippet_truncated is True


@pytest.mark.asyncio
async def test_transcript_search_snippet_includes_query_after_whitespace_normalization(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append(
        scope,
        TranscriptMessageEntry(
            entry_id="m1",
            session_id="session-1",
            message={
                "role": "assistant",
                "content": (
                    "prefix"
                    + (" \n\t" * 200)
                    + "needle"
                    + (" trailing words" * 40)
                ),
            },
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
            max_results=1,
            max_snippet_chars=40,
            max_total_snippet_chars=40,
        )
    )

    assert len(result.matches) == 1
    assert "needle" in result.matches[0].snippet
    assert len(result.matches[0].snippet) <= 40


@pytest.mark.asyncio
async def test_transcript_search_preserves_sidechain_and_runtime_metadata(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    main_scope = TranscriptScope(session_id="session-1")
    child_scope = TranscriptScope(
        session_id="session-1",
        agent_id="agent-1",
        is_sidechain=True,
        runtime_session_id="runtime-child",
    )
    await store.append(
        main_scope,
        TranscriptMessageEntry(
            entry_id="main",
            session_id="session-1",
            message={"role": "user", "content": "main needle"},
        ),
    )
    await store.append(
        child_scope,
        TranscriptMessageEntry(
            entry_id="child",
            session_id="session-1",
            runtime_session_id="runtime-child",
            agent_id="agent-1",
            is_sidechain=True,
            message={"role": "assistant", "content": "child needle"},
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(
                session_id="session-1",
                include_main=False,
                sidechain_agent_ids=("agent-1",),
            ),
        )
    )

    assert [match.entry_id for match in result.matches] == ["child"]
    assert result.matches[0].agent_id == "agent-1"
    assert result.matches[0].runtime_session_id == "runtime-child"
    assert result.matches[0].is_sidechain is True


@pytest.mark.asyncio
async def test_transcript_search_filters_by_runtime_session_id(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append_many(
        scope,
        (
            TranscriptMessageEntry(
                entry_id="stale",
                session_id="session-1",
                runtime_session_id="runtime-old",
                message={"role": "user", "content": "old needle"},
            ),
            TranscriptMessageEntry(
                entry_id="current",
                session_id="session-1",
                runtime_session_id="runtime-current",
                message={"role": "user", "content": "current needle"},
            ),
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(
                session_id="session-1",
                runtime_session_id="runtime-current",
            ),
        )
    )

    assert [match.entry_id for match in result.matches] == ["current"]


@pytest.mark.asyncio
async def test_transcript_search_can_discover_all_sidechains(tmp_path: Path) -> None:
    store = JsonlTranscriptStore(tmp_path)
    child_scope = TranscriptScope(
        session_id="session-1",
        agent_id="agent-1",
        is_sidechain=True,
    )
    await store.append(
        child_scope,
        TranscriptMessageEntry(
            entry_id="child",
            session_id="session-1",
            agent_id="agent-1",
            is_sidechain=True,
            message={"role": "assistant", "content": "sidechain needle"},
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(
                session_id="session-1",
                include_main=False,
                include_all_sidechains=True,
            ),
        )
    )

    assert [scope.agent_id for scope in result.scopes_searched] == ["agent-1"]
    assert [match.entry_id for match in result.matches] == ["child"]


@pytest.mark.asyncio
async def test_transcript_search_is_compact_boundary_aware_by_default(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append_many(
        scope,
        (
            TranscriptMessageEntry(
                entry_id="pre",
                session_id="session-1",
                message={"role": "user", "content": "old needle"},
            ),
            CompactBoundaryEntry(
                entry_id="c1",
                session_id="session-1",
                boundary=CompactBoundary(
                    message_index=1,
                    kind="microcompact",
                    summary="summary",
                ),
            ),
            TranscriptMessageEntry(
                entry_id="post",
                session_id="session-1",
                message={"role": "assistant", "content": "post compact only"},
            ),
        ),
    )
    service = TranscriptSearchService(store)
    request = TranscriptSearchRequest(
        query="needle",
        scope=TranscriptSearchScope(session_id="session-1"),
    )

    active = await service.search(request)
    full = await service.search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
            compact_mode="full",
        )
    )

    assert active.matches == ()
    assert [match.entry_id for match in full.matches] == ["pre"]


@pytest.mark.asyncio
async def test_transcript_search_surfaces_malformed_entry_warnings(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    path = Path(store.path_for(scope))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "{bad",
                transcript_entry_to_json(
                    TranscriptMessageEntry(
                        entry_id="m1",
                        session_id="session-1",
                        message={"role": "user", "content": "needle after warning"},
                    )
                ),
            )
        ),
        encoding="utf-8",
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
        )
    )

    assert len(result.matches) == 1
    assert result.warnings
    assert "TranscriptDecodeError" not in result.matches[0].snippet


@pytest.mark.asyncio
async def test_transcript_search_surfaces_malformed_message_content_warnings(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    path = Path(store.path_for(scope))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "message",
                        "entry_id": "bad-null",
                        "session_id": "session-1",
                        "created_at": 1.0,
                        "message": {"role": "user", "content": None},
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "entry_id": "bad-list",
                        "session_id": "session-1",
                        "created_at": 2.0,
                        "message": {"role": "user", "content": ["bad"]},
                    }
                ),
                transcript_entry_to_json(
                    TranscriptMessageEntry(
                        entry_id="good",
                        session_id="session-1",
                        message={"role": "user", "content": "needle after warning"},
                    )
                ),
            )
        ),
        encoding="utf-8",
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
        )
    )

    assert [match.entry_id for match in result.matches] == ["good"]
    assert any("message.content" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_transcript_search_skips_direct_entries_with_malformed_blocks(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append_many(
        scope,
        (
            TranscriptMessageEntry(
                entry_id="bad-block",
                session_id="session-1",
                message=cast(
                    MessageParam,
                    {"role": "user", "content": ["bad", {"type": "text"}]},
                ),
            ),
            TranscriptMessageEntry(
                entry_id="good-block",
                session_id="session-1",
                message={
                    "role": "user",
                    "content": [{"type": "text", "text": "needle in block"}],
                },
            ),
        ),
    )

    result = await TranscriptSearchService(store).search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
        )
    )

    assert [match.entry_id for match in result.matches] == ["good-block"]


@pytest.mark.asyncio
async def test_transcript_search_preserves_fifo_within_same_timestamp(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append_many(
        scope,
        (
            TranscriptMessageEntry(
                entry_id="first",
                session_id="session-1",
                created_at=1.0,
                message={"role": "user", "content": "first needle"},
            ),
            TranscriptMessageEntry(
                entry_id="second",
                session_id="session-1",
                created_at=1.0,
                message={"role": "assistant", "content": "second needle"},
            ),
        ),
    )
    service = TranscriptSearchService(store)

    newest = await service.search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
            order="newest_first",
        )
    )
    oldest = await service.search(
        TranscriptSearchRequest(
            query="needle",
            scope=TranscriptSearchScope(session_id="session-1"),
            order="oldest_first",
        )
    )

    assert [match.entry_id for match in newest.matches] == ["first", "second"]
    assert [match.entry_id for match in oldest.matches] == ["first", "second"]
