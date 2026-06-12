"""Tests for tool-result budget rewrite.

"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.query import LayerResult, TerminalEvent, query
from raygent_harness.core.state import State
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    ContentReplacementState,
    QueryTracking,
    ToolUseContext,
)
from raygent_harness.services.compact import (
    PERSISTED_TOOL_RESULT_TAG,
    apply_tool_result_budget,
)

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def _ctx(*, replacement: ContentReplacementState | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        content_replacement=replacement,
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _tool_result(tool_use_id: str, content: object) -> MessageParam:
    return cast(
        "MessageParam",
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


def _assistant(content: object = "ok", *, message_id: str | None = None) -> MessageParam:
    msg: dict[str, object] = {"role": "assistant", "content": content}
    if message_id is not None:
        msg["id"] = message_id
    return cast("MessageParam", msg)


def _blocks(message: MessageParam) -> list[dict[str, Any]]:
    content = message.get("content")
    assert isinstance(content, list)
    return content


@pytest.mark.asyncio
async def test_tool_result_budget_noops_without_state() -> None:
    messages = [_tool_result("t1", "x" * 100)]

    result = await apply_tool_result_budget(messages, None)

    assert result.messages is messages
    assert result.newly_replaced == ()


@pytest.mark.asyncio
async def test_tool_result_budget_persists_largest_fresh_result(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "tool-results"
    state = ContentReplacementState(
        max_result_size_chars=50,
        replaced_outputs_dir=str(out_dir),
    )
    messages = [
        cast(
            "MessageParam",
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "small",
                        "content": "s" * 20,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "large",
                        "content": "l" * 80,
                    },
                ],
            },
        )
    ]

    result = await apply_tool_result_budget(messages, state)

    assert len(result.newly_replaced) == 1
    record = result.newly_replaced[0]
    assert record.tool_use_id == "large"
    assert record.original_size_chars == 80
    assert (out_dir / "large.txt").read_text() == "l" * 80

    content = _blocks(result.messages[0])
    assert content[0]["content"] == "s" * 20
    assert str(content[1]["content"]).startswith(PERSISTED_TOOL_RESULT_TAG)
    assert state.seen_ids == {"small", "large"}
    assert state.replacements["large"] == content[1]["content"]


@pytest.mark.asyncio
async def test_tool_result_budget_reapplies_existing_replacement_without_file_io(
    tmp_path: Path,
) -> None:
    state = ContentReplacementState(
        max_result_size_chars=10,
        replaced_outputs_dir=str(tmp_path),
        replacements={"t1": "cached replacement"},
        seen_ids={"t1"},
    )

    result = await apply_tool_result_budget([_tool_result("t1", "new full text")], state)

    content = _blocks(result.messages[0])
    assert content[0]["content"] == "cached replacement"
    assert result.newly_replaced == ()


@pytest.mark.asyncio
async def test_tool_result_budget_freezes_seen_unreplaced_ids(
    tmp_path: Path,
) -> None:
    state = ContentReplacementState(
        max_result_size_chars=1,
        replaced_outputs_dir=str(tmp_path),
        seen_ids={"t1"},
    )

    result = await apply_tool_result_budget([_tool_result("t1", "x" * 100)], state)

    assert result.messages == [_tool_result("t1", "x" * 100)]
    assert result.newly_replaced == ()
    assert state.replacements == {}


@pytest.mark.asyncio
async def test_tool_result_budget_groups_repeated_assistant_ids(
    tmp_path: Path,
) -> None:
    """Same-id assistant fragments normalize into one wire assistant message.

    Their following tool_result blocks therefore share one wire user-message
    size budget. Without assistant-id grouping, both results below sit under
    the per-group cap and neither is replaced.
    """
    state = ContentReplacementState(
        max_result_size_chars=50,
        replaced_outputs_dir=str(tmp_path),
    )
    messages = [
        _assistant(message_id="asst-1"),
        _tool_result("t1", "x" * 40),
        _assistant(message_id="asst-1"),
        _tool_result("t2", "y" * 40),
    ]

    result = await apply_tool_result_budget(messages, state)

    assert len(result.newly_replaced) == 1
    assert {record.tool_use_id for record in result.newly_replaced} in (
        {"t1"},
        {"t2"},
    )


@pytest.mark.asyncio
async def test_tool_result_budget_skips_image_blocks(
    tmp_path: Path,
) -> None:
    state = ContentReplacementState(
        max_result_size_chars=1,
        replaced_outputs_dir=str(tmp_path),
    )
    image_content = [{"type": "image", "source": {"type": "base64", "data": "x"}}]

    result = await apply_tool_result_budget([_tool_result("img", image_content)], state)

    assert result.messages == [_tool_result("img", image_content)]
    assert result.newly_replaced == ()
    assert state.seen_ids == set()


@pytest.mark.asyncio
async def test_tool_result_budget_fails_soft_when_output_dir_cannot_be_created(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "not-a-dir"
    occupied.write_text("occupied")
    state = ContentReplacementState(
        max_result_size_chars=1,
        replaced_outputs_dir=str(occupied / "tool-results"),
    )
    messages = [_tool_result("t1", "x" * 100)]

    result = await apply_tool_result_budget(messages, state)

    assert result.messages == messages
    assert result.newly_replaced == ()
    assert state.seen_ids == {"t1"}
    assert state.replacements == {}


@pytest.mark.asyncio
async def test_tool_result_budget_fails_soft_for_non_text_structured_content(
    tmp_path: Path,
) -> None:
    state = ContentReplacementState(
        max_result_size_chars=1,
        replaced_outputs_dir=str(tmp_path),
    )
    structured_content = [
        {"type": "text", "text": "x" * 100},
        {"type": "document", "source": "payload"},
    ]
    messages = [_tool_result("structured", structured_content)]

    result = await apply_tool_result_budget(messages, state)

    assert result.messages == messages
    assert result.newly_replaced == ()
    assert state.seen_ids == {"structured"}
    assert state.replacements == {}
    assert not (tmp_path / "structured.txt").exists()


@pytest.mark.asyncio
async def test_query_pipeline_applies_tool_result_budget_before_microcompact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_by_microcompact: list[list[MessageParam]] = []
    seen_by_model: list[list[MessageParam]] = []

    async def fake_microcompact(
        messages: list[MessageParam],
        _state: State,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> LayerResult:
        seen_by_microcompact.append(list(messages))
        return LayerResult(messages=messages)

    async def fake_call(
        messages: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_by_model.append(list(messages))
        return {"text": "ok"}

    def fake_assistant(_response: Any) -> MessageParam:
        return _assistant("ok")

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)

    replacement = ContentReplacementState(
        max_result_size_chars=10,
        replaced_outputs_dir=str(tmp_path),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        microcompact=fake_microcompact,
    )

    events: list[Any] = []
    async for ev in query(
        State(messages=[_tool_result("big", "x" * 100)]),
        QueryConfig(model="claude-opus-4-7"),
        deps,
        _ctx(replacement=replacement),
    ):
        events.append(ev)

    assert len(seen_by_microcompact) == 1
    micro_content = _blocks(seen_by_microcompact[0][0])
    assert str(micro_content[0]["content"]).startswith(PERSISTED_TOOL_RESULT_TAG)
    assert seen_by_model == seen_by_microcompact

    terminal = next(ev for ev in events if isinstance(ev, TerminalEvent)).terminal
    assert terminal.final_state is not None
    assert terminal.final_state.messages[-1] == _assistant("ok")
