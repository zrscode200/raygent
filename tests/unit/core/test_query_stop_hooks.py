"""Stop-hook integration with the loop body — `_evaluate_stop_hooks`.

Per item-11 review (group-1 Highs):
- Stop hooks must see the post-assistant transcript, not pre-assistant.
- `prevent_continuation` must yield `Terminal(reason="stop_hook_prevented")`,
  never silently fall through to `completed`.
- `block`-without-prevent must continue with assistant + blocking messages
  carried into the next iteration.
- The clean-completion `Terminal.final_state.messages` must include the
  assistant message.

Tests call `_evaluate_stop_hooks` directly with a constructed State + a
canned assistant message and a tiny in-process StopHook. Direct unit
tests, not full `query()` integration — `_call_model` is still a stub
(full query integration is covered by separate loop tests).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.observability import KernelEventBus, RecordingKernelEventSink
from raygent_harness.core.query import (
    StopHookOutcome,
    _evaluate_stop_hooks,  # pyright: ignore[reportPrivateUsage]
)
from raygent_harness.core.state import State
from raygent_harness.core.stop_hooks import (
    DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS,
    DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS,
    ContinuationContextFragment,
    HookBlock,
    HookContext,
    HookContinue,
    HookContinueWithContext,
    HookPreventContinuation,
    HookResult,
)
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


def _ctx(*, agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _deps(*hooks: object) -> QueryDeps:
    # `hooks` are callables; QueryDeps.stop_hooks is typed list[StopHook]
    # (Callable[[HookContext], Awaitable[HookResult]]).
    return QueryDeps(
        task_store=AppStateStore(),
        stop_hooks=list(hooks),  # pyright: ignore[reportArgumentType]
    )


def _config() -> QueryConfig:
    return QueryConfig(model="claude-opus-4-7")


def _state(messages: list[MessageParam] | None = None) -> State:
    return State(messages=messages or [])


def _assistant(text: str = "answer") -> MessageParam:
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Outcome 1: clean completion. No hooks, or hooks all said "continue".
# Final state must include the assistant message.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_hooks_clean_completion_carries_assistant_message() -> None:
    initial: list[MessageParam] = [{"role": "user", "content": "hi"}]
    outcome = await _evaluate_stop_hooks(
        _state(initial), list(initial), _assistant("hello"), _config(), _deps(), _ctx()
    )

    assert outcome.terminal is None
    assert outcome.should_continue is False
    # Outcome state must include the assistant — otherwise the loop's
    # `Terminal(completed, ..., final_state=outcome.state)` would lose it.
    assert len(outcome.state.messages) == 2
    assert outcome.state.messages[-1] == {"role": "assistant", "content": "hello"}


@pytest.mark.asyncio
async def test_hook_continue_clean_completion_includes_assistant() -> None:
    async def ok_hook(_hc: HookContext) -> HookResult:
        return HookContinue()

    outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant("done"), _config(), _deps(ok_hook), _ctx()
    )
    assert outcome.terminal is None
    assert outcome.should_continue is False
    assert outcome.state.messages[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_hook_sees_post_assistant_transcript() -> None:
    # Hooks must see the post-assistant transcript, not the pre-turn state.
    seen_messages: list[list[MessageParam]] = []

    async def inspect_hook(hc: HookContext) -> HookResult:
        seen_messages.append(list(hc.messages))
        return HookContinue()

    initial: list[MessageParam] = [{"role": "user", "content": "q"}]
    await _evaluate_stop_hooks(
        _state(initial),
        list(initial),
        _assistant("a"),
        _config(),
        _deps(inspect_hook),
        _ctx(),
    )

    assert len(seen_messages) == 1
    # Hook must see BOTH the prior user msg AND the new assistant msg.
    assert seen_messages[0] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


# ---------------------------------------------------------------------------
# Outcome 2: prevent_continuation. Reference returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prevent_continuation_returns_stop_hook_prevented_terminal() -> None:
    async def prevent_hook(_hc: HookContext) -> HookResult:
        return HookPreventContinuation(reason="not done yet")

    outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), _deps(prevent_hook), _ctx()
    )

    assert outcome.terminal is not None
    assert outcome.terminal.reason == "stop_hook_prevented"
    assert outcome.terminal.message == "not done yet"
    # The veto wins — should_continue must be False.
    assert outcome.should_continue is False


@pytest.mark.asyncio
async def test_prevent_wins_even_when_blocking_message_also_present() -> None:
    # Prevent-continuation takes precedence over block/retry semantics.
    async def block_hook(_hc: HookContext) -> HookResult:
        return HookBlock(message="block reason")

    async def prevent_hook(_hc: HookContext) -> HookResult:
        return HookPreventContinuation(reason="prevent reason")

    outcome = await _evaluate_stop_hooks(
        _state(),
        [],
        _assistant(),
        _config(),
        _deps(block_hook, prevent_hook),
        _ctx(),
    )
    assert outcome.terminal is not None
    assert outcome.terminal.reason == "stop_hook_prevented"
    assert outcome.should_continue is False


@pytest.mark.asyncio
async def test_prevent_wins_even_when_continuation_context_present() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            fragments=(
                ContinuationContextFragment(
                    id="ctx-1",
                    content="additional facts",
                    source="policy",
                    reason="needs more context",
                ),
            )
        )

    async def prevent_hook(_hc: HookContext) -> HookResult:
        return HookPreventContinuation(reason="hard veto")

    outcome = await _evaluate_stop_hooks(
        _state(),
        [],
        _assistant(),
        _config(),
        _deps(context_hook, prevent_hook),
        _ctx(),
    )

    assert outcome.terminal is not None
    assert outcome.terminal.reason == "stop_hook_prevented"
    assert outcome.should_continue is False
    assert len(outcome.continuation_messages) == 1
    final_state = outcome.terminal.final_state
    assert final_state is not None
    assert final_state.messages[-1].get("raygentMessageKind") == "continuation_context"


@pytest.mark.asyncio
async def test_prevent_terminal_final_state_includes_assistant_and_blocking() -> None:
    # Terminal final_state preserves assistant and blocking messages for audit.
    async def block_then_prevent_hook(_hc: HookContext) -> HookResult:
        return HookBlock(message="block-msg")

    async def prevent_hook(_hc: HookContext) -> HookResult:
        return HookPreventContinuation(reason="veto")

    initial: list[MessageParam] = [{"role": "user", "content": "q"}]
    outcome = await _evaluate_stop_hooks(
        _state(initial),
        list(initial),
        _assistant("a"),
        _config(),
        _deps(block_then_prevent_hook, prevent_hook),
        _ctx(),
    )

    assert outcome.terminal is not None
    final = outcome.terminal.final_state
    assert final is not None
    # user + assistant + blocking
    assert len(final.messages) == 3
    assert final.messages[1] == {"role": "assistant", "content": "a"}
    assert final.messages[2]["role"] == "user"


# ---------------------------------------------------------------------------
# Outcome 3: block-only retry. Continue with assistant + blocking msgs in
# state.messages (so the next iteration's model call sees both).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_only_continues_and_carries_assistant_plus_blocking() -> None:
    async def block_hook(_hc: HookContext) -> HookResult:
        return HookBlock(message="please continue working")

    initial: list[MessageParam] = [{"role": "user", "content": "q"}]
    outcome = await _evaluate_stop_hooks(
        _state(initial),
        list(initial),
        _assistant("a"),
        _config(),
        _deps(block_hook),
        _ctx(),
    )

    assert outcome.terminal is None
    assert outcome.should_continue is True
    # Carry-state must include both the assistant message AND the blocking
    # message — the loop's continue-site does NOT re-append the assistant.
    msgs = outcome.state.messages
    assert len(msgs) == 3
    assert msgs[1] == {"role": "assistant", "content": "a"}
    assert msgs[2]["role"] == "user"
    assert "please continue working" in str(msgs[2]["content"])


@pytest.mark.asyncio
async def test_continue_with_context_continues_with_typed_bounded_message() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            message="Use this extra context before deciding.",
            fragments=(
                ContinuationContextFragment(
                    id="later",
                    content="later priority",
                    source="policy-b",
                    reason="secondary",
                    priority=10,
                ),
                ContinuationContextFragment(
                    id="first",
                    content="first priority",
                    source="policy-a",
                    reason="primary",
                    priority=-1,
                ),
            ),
        )

    outcome = await _evaluate_stop_hooks(
        _state([{"role": "user", "content": "q"}]),
        [{"role": "user", "content": "q"}],
        _assistant("a"),
        _config(),
        _deps(context_hook),
        _ctx(),
    )

    assert outcome.terminal is None
    assert outcome.should_continue is True
    assert outcome.blocking_messages == ()
    assert len(outcome.continuation_messages) == 1
    context_message = outcome.continuation_messages[0]
    assert context_message["role"] == "user"
    assert context_message.get("raygentMessageKind") == "continuation_context"
    content = str(context_message["content"])
    assert "[continuation context]" in content
    assert content.index("id=first") < content.index("id=later")

    metadata = context_message.get("raygentContinuationContext")
    assert metadata is not None
    assert metadata["type"] == "continuation_context"
    assert metadata["fragment_count"] == 2
    assert metadata["truncated_fragment_count"] == 0
    assert metadata["dropped_empty_fragment_count"] == 0
    assert metadata["fragments"][0]["id"] == "first"

    assert outcome.state.messages == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        context_message,
    ]


@pytest.mark.asyncio
async def test_continue_with_context_truncates_fragment_content() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            fragments=(
                ContinuationContextFragment(
                    id="large-" + ("i" * 150),
                    content="x" * 50,
                    source="source-" + ("s" * 150),
                    reason="reason-" + ("r" * 150),
                    max_chars=12,
                ),
                ContinuationContextFragment(id="empty", content="  "),
            )
        )

    outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), _deps(context_hook), _ctx()
    )

    assert len(outcome.continuation_messages) == 1
    context_message = outcome.continuation_messages[0]
    content = str(context_message["content"])
    assert "x" * 12 in content
    assert "x" * 13 not in content
    assert "[truncated to 12 chars]" in content
    assert "i" * 130 not in content
    assert "s" * 130 not in content
    assert "r" * 130 not in content
    metadata = context_message.get("raygentContinuationContext")
    assert metadata is not None
    assert metadata["fragment_count"] == 1
    assert len(metadata["fragments"][0]["id"]) <= 123
    assert metadata["fragments"][0]["source"] is not None
    assert len(metadata["fragments"][0]["source"]) <= 123
    assert metadata["fragments"][0]["reason"] is not None
    assert len(metadata["fragments"][0]["reason"]) <= 123
    assert metadata["input_char_count"] == 50
    assert metadata["truncated_fragment_count"] == 1
    assert metadata["dropped_empty_fragment_count"] == 1
    assert metadata["rendered_message_truncated"] is False
    assert metadata["dropped_fragment_count"] == 0


@pytest.mark.asyncio
async def test_continue_with_context_truncates_oversized_lead_message() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            message="LEAD-" + ("m" * (DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS * 2)),
            fragments=(
                ContinuationContextFragment(id="after-lead", content="tail context"),
            ),
        )

    outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), _deps(context_hook), _ctx()
    )

    assert len(outcome.continuation_messages) == 1
    context_message = outcome.continuation_messages[0]
    content = str(context_message["content"])
    assert len(content) <= DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS
    assert "[lead message truncated]" in content
    assert "tail context" not in content
    metadata = context_message.get("raygentContinuationContext")
    assert metadata is not None
    assert metadata["rendered_message_truncated"] is True
    assert metadata["dropped_fragment_count"] == 1


@pytest.mark.asyncio
async def test_continue_with_context_caps_many_tiny_fragments() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            fragments=tuple(
                ContinuationContextFragment(id=f"ctx-{index}", content="x")
                for index in range(DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS + 10)
            )
        )

    outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), _deps(context_hook), _ctx()
    )

    assert len(outcome.continuation_messages) == 1
    context_message = outcome.continuation_messages[0]
    content = str(context_message["content"])
    assert len(content) <= DEFAULT_CONTINUATION_CONTEXT_TOTAL_CHARS
    assert "ctx-0" in content
    assert f"ctx-{DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS + 9}" not in content
    metadata = context_message.get("raygentContinuationContext")
    assert metadata is not None
    assert metadata["fragment_count"] == DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS
    assert metadata["max_fragment_count"] == DEFAULT_CONTINUATION_CONTEXT_MAX_FRAGMENTS
    assert metadata["dropped_fragment_count"] == 10


@pytest.mark.asyncio
async def test_continue_with_context_observability_is_metadata_only() -> None:
    async def context_hook(_hc: HookContext) -> HookResult:
        return HookContinueWithContext(
            fragments=(
                ContinuationContextFragment(
                    id="secret-fragment",
                    content="SECRET_CONTEXT_VALUE",
                    source="policy",
                    reason="safe reason",
                ),
            )
        )

    sink = RecordingKernelEventSink()
    deps = QueryDeps(
        task_store=AppStateStore(),
        stop_hooks=[context_hook],
        observability=KernelEventBus([sink]),
    )
    await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), deps, _ctx()
    )

    completed = sink.by_type("hook.stop.completed")[0]
    assert completed.data["continuation_message_count"] == 1
    assert completed.data["continuation_fragment_count"] == 1
    assert completed.data["continuation_input_char_count"] == len("SECRET_CONTEXT_VALUE")
    assert "SECRET_CONTEXT_VALUE" not in str(completed.data)


@pytest.mark.asyncio
async def test_continue_with_context_scope_can_gate_on_agent_id() -> None:
    async def main_only_hook(hc: HookContext) -> HookResult:
        if hc.tool_use_context.agent_id is not None:
            return HookContinue()
        return HookContinueWithContext(
            fragments=(ContinuationContextFragment(id="main", content="main context"),)
        )

    main_outcome = await _evaluate_stop_hooks(
        _state(), [], _assistant(), _config(), _deps(main_only_hook), _ctx()
    )
    child_outcome = await _evaluate_stop_hooks(
        _state(),
        [],
        _assistant(),
        _config(),
        _deps(main_only_hook),
        _ctx(agent_id="child-agent"),
    )

    assert main_outcome.should_continue is True
    assert len(main_outcome.continuation_messages) == 1
    assert child_outcome.should_continue is False
    assert child_outcome.continuation_messages == ()


# ---------------------------------------------------------------------------
# StopHookOutcome shape regression — discriminator can't lie.
# ---------------------------------------------------------------------------


def test_outcome_dataclass_defaults_are_clean_completion() -> None:
    # A bare StopHookOutcome means clean completion.
    outcome = StopHookOutcome(state=_state())
    assert outcome.terminal is None
    assert outcome.should_continue is False


# ---------------------------------------------------------------------------
# Regression: hook context + outcome state must be built from
# `messages_for_model`, not pre-pipeline `state.messages`. Pre-fix the
# compacted view was dropped at the no-tool boundary — compaction
# would have been silently lost.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_sees_compacted_messages_not_pre_pipeline_state() -> None:
    # Stop hooks must inspect the compacted model-input transcript, not the
    # pre-pipeline state message list.
    seen: list[list[MessageParam]] = []

    async def inspect_hook(hc: HookContext) -> HookResult:
        seen.append(list(hc.messages))
        return HookContinue()

    pre_pipeline: list[MessageParam] = [
        {"role": "user", "content": "msg-1"},
        {"role": "user", "content": "msg-2"},
        {"role": "user", "content": "msg-3"},
    ]
    compacted: list[MessageParam] = [{"role": "user", "content": "summary-of-three"}]

    outcome = await _evaluate_stop_hooks(
        _state(pre_pipeline),
        compacted,
        _assistant("a"),
        _config(),
        _deps(inspect_hook),
        _ctx(),
    )

    # Hook saw the COMPACTED view + assistant — not the pre-pipeline 3
    # messages.
    assert len(seen) == 1
    assert seen[0] == [
        {"role": "user", "content": "summary-of-three"},
        {"role": "assistant", "content": "a"},
    ]
    # Outcome state (used for Terminal.final_state on completed) also
    # carries the compacted view.
    assert outcome.terminal is None
    assert outcome.state.messages == [
        {"role": "user", "content": "summary-of-three"},
        {"role": "assistant", "content": "a"},
    ]


@pytest.mark.asyncio
async def test_block_only_carry_state_uses_compacted_view() -> None:
    """Block-only retry must build carry-state from `messages_for_model`,
    not pre-pipeline state — otherwise the next iteration re-feeds
    pre-compact history and compaction is lost across iterations."""

    async def block_hook(_hc: HookContext) -> HookResult:
        return HookBlock(message="more please")

    pre_pipeline: list[MessageParam] = [
        {"role": "user", "content": "msg-1"},
        {"role": "user", "content": "msg-2"},
    ]
    compacted: list[MessageParam] = [{"role": "user", "content": "summary"}]

    outcome = await _evaluate_stop_hooks(
        _state(pre_pipeline),
        compacted,
        _assistant("a"),
        _config(),
        _deps(block_hook),
        _ctx(),
    )

    assert outcome.should_continue is True
    # summary + assistant + blocking — pre-pipeline msgs must be gone.
    assert outcome.state.messages == [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "more please"},
    ]


@pytest.mark.asyncio
async def test_prevent_terminal_final_state_uses_compacted_view() -> None:
    """Same invariant for the prevent path: Terminal.final_state.messages
    is built off the compacted view, not pre-pipeline state."""

    async def prevent_hook(_hc: HookContext) -> HookResult:
        return HookPreventContinuation(reason="not yet")

    pre_pipeline: list[MessageParam] = [
        {"role": "user", "content": "msg-1"},
        {"role": "user", "content": "msg-2"},
    ]
    compacted: list[MessageParam] = [{"role": "user", "content": "summary"}]

    outcome = await _evaluate_stop_hooks(
        _state(pre_pipeline),
        compacted,
        _assistant("a"),
        _config(),
        _deps(prevent_hook),
        _ctx(),
    )

    assert outcome.terminal is not None
    final = outcome.terminal.final_state
    assert final is not None
    assert final.messages == [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "a"},
    ]
