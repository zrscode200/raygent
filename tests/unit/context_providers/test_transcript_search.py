from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from raygent_harness.context_providers import TranscriptSearchContextProvider
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import render_user_context_messages
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import message_param_from_api_message
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from raygent_harness.services.transcript import (
    JsonlTranscriptStore,
    TranscriptMessageEntry,
    TranscriptScope,
    TranscriptSearchScope,
    TranscriptSearchService,
)
from tests.fakes import FakeModelProvider


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="session-1",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="chain-1", depth=0),
    )


@pytest.mark.asyncio
async def test_transcript_search_context_provider_renders_bounded_user_context(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append(
        scope,
        TranscriptMessageEntry(
            entry_id="m1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            message={"role": "assistant", "content": "needle context"},
        ),
    )
    provider = TranscriptSearchContextProvider(
        search_service=TranscriptSearchService(store),
        scope=TranscriptSearchScope(session_id="session-1"),
        query="needle",
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx())
    messages = render_user_context_messages(fragments)

    assert len(fragments) == 1
    assert fragments[0].channel == "user_context"
    assert fragments[0].source == "transcript_search"
    assert fragments[0].kind == "memory"
    assert len(messages) == 1
    content = str(messages[0]["content"])
    assert "<transcript_search_results" in content
    assert "query_chars=" in content
    assert "needle context" in content
    assert "runtime-1" in content


@pytest.mark.asyncio
async def test_transcript_search_context_provider_noops_without_query_or_matches(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    service = TranscriptSearchService(store)
    provider_without_query = TranscriptSearchContextProvider(
        search_service=service,
        scope=TranscriptSearchScope(session_id="session-1"),
    )
    provider_without_matches = TranscriptSearchContextProvider(
        search_service=service,
        scope=TranscriptSearchScope(session_id="session-1"),
        query_resolver=lambda _config, _ctx: "needle",
    )

    assert await provider_without_query(QueryConfig(model="model-1"), _ctx()) == ()
    assert await provider_without_matches(QueryConfig(model="model-1"), _ctx()) == ()


@pytest.mark.asyncio
async def test_transcript_search_context_provider_honors_context_kind(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    await store.append(
        TranscriptScope(session_id="session-1"),
        TranscriptMessageEntry(
            entry_id="m1",
            session_id="session-1",
            message={"role": "assistant", "content": "needle context"},
        ),
    )
    provider = TranscriptSearchContextProvider(
        search_service=TranscriptSearchService(store),
        scope=TranscriptSearchScope(session_id="session-1"),
        query="needle",
        context_kind="custom",
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx())

    assert len(fragments) == 1
    assert fragments[0].kind == "custom"


@pytest.mark.asyncio
async def test_transcript_search_context_is_model_visible_but_not_persisted(
    tmp_path: Path,
) -> None:
    store = JsonlTranscriptStore(tmp_path)
    scope = TranscriptScope(session_id="session-1")
    await store.append(
        scope,
        TranscriptMessageEntry(
            entry_id="old",
            session_id="session-1",
            message={"role": "assistant", "content": "needle SEARCH_CONTEXT_ONLY"},
        ),
    )
    model = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        transcript_store=store,
        context_providers=(
            TranscriptSearchContextProvider(
                search_service=TranscriptSearchService(store),
                scope=TranscriptSearchScope(session_id="session-1"),
                query="needle",
            ),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="session-1"),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("current prompt")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    request_text = "\n".join(str(message.get("content", "")) for message in request_messages)
    assert "SEARCH_CONTEXT_ONLY" in request_text
    assert request_messages[-1] == {"role": "user", "content": "current prompt"}

    engine_text = "\n".join(
        str(message.get("content", ""))
        for message in engine._messages  # pyright: ignore[reportPrivateUsage]
    )
    assert "SEARCH_CONTEXT_ONLY" not in engine_text

    transcript_entries = await store.read_entries(scope)
    transcript_text = "\n".join(
        str(entry.message.get("content", ""))
        for entry in transcript_entries
        if isinstance(entry, TranscriptMessageEntry)
    )
    assert transcript_text.count("SEARCH_CONTEXT_ONLY") == 1
