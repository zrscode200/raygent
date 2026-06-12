"""stop_task wrapper tests — typed-error propagation, bash suppression,
agent-path payload preservation, public-API regression.

Per item-10 review: the wrapper at `core/tasks/stop_task.py` is the
ONLY supported public stop entry. The bypass surface
(`core.task.stop_task` exported in `__all__`) was removed; the
dispatcher is now `core.task.dispatch_stop_task` and excluded from
`__all__`. The regression test below codifies that.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from raygent_harness.core import task as core_task
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.query_engine import SDKResult
from raygent_harness.core.task import (
    AppStateStore,
    TaskNotFoundError,
    TaskNotRunningError,
)
from raygent_harness.core.tasks import local_agent as agent_mod
from raygent_harness.core.tasks.local_agent import spawn_local_agent
from raygent_harness.core.tasks.local_bash import (
    run_until_done as run_bash_until_done,
)
from raygent_harness.core.tasks.local_bash import spawn_local_bash
from raygent_harness.core.tasks.stop_task import stop_task
from raygent_harness.core.tool import QueryTracking, ToolUseContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.asyncio
async def test_not_found_raises_task_not_found_error() -> None:
    store = AppStateStore()
    with pytest.raises(TaskNotFoundError):
        await stop_task("does-not-exist", store)


@pytest.mark.asyncio
async def test_already_terminal_raises_task_not_running_error() -> None:
    store = AppStateStore()
    s = await spawn_local_bash("echo done", store, agent_id="a1")
    await run_bash_until_done(s.id, store)
    assert store.tasks[s.id].status == "completed"

    with pytest.raises(TaskNotRunningError):
        await stop_task(s.id, store)


@pytest.mark.asyncio
async def test_bash_kill_via_wrapper_suppresses_notification() -> None:
    store = AppStateStore()
    s = await spawn_local_bash("sleep 5", store, agent_id="a1")
    await asyncio.sleep(0.05)
    store.drain_notifications("a1")  # baseline

    result = await stop_task(s.id, store)
    assert result.task_type == "local_bash"

    # Let the driver finish its terminal path.
    await asyncio.sleep(0.3)

    final = store.tasks[s.id]
    assert final.status == "killed"
    assert final.notified is True
    leaked = store.drain_notifications("a1")
    assert leaked == []


@pytest.mark.asyncio
async def test_agent_kill_via_wrapper_does_not_suppress_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper's bash-only suppression must not gate the agent's payload
    notification, which carries the partial-result XML the parent needs."""

    class _BlockingEngine:
        def __init__(self, _config: Any, _deps: Any, _ctx: Any) -> None:
            pass

        async def submit_message(self, _prompt: str) -> AsyncIterator[Any]:
            await asyncio.Event().wait()
            yield SDKResult()  # pragma: no cover

    monkeypatch.setattr(agent_mod, "QueryEngine", _BlockingEngine)

    store = AppStateStore()
    config = QueryConfig(
        model="claude-opus-4-7",
        agent_id="parent",
        session_id="parent-session",
    )
    deps = QueryDeps(task_store=store)
    ctx = ToolUseContext(
        session_id="parent-session",
        agent_id="parent",
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="parent", depth=0),
    )

    task_id = await spawn_local_agent(
        prompt="x",
        parent_agent_id="parent",
        parent_config=config,
        parent_deps=deps,
        parent_ctx=ctx,
    )
    await asyncio.sleep(0.05)

    result = await stop_task(task_id, store)
    assert result.task_type == "local_agent"

    # Wait for driver terminal path.
    for _ in range(50):
        if task_id not in agent_mod._DRIVER_TASKS:  # pyright: ignore[reportPrivateUsage]
            break
        await asyncio.sleep(0.05)

    # Agent's payload notification SHOULD have been delivered to parent.
    parent_notifs = store.drain_notifications("parent")
    assert len(parent_notifs) == 1
    assert parent_notifs[0].kind == "error"
    assert "<status>killed" in parent_notifs[0].message


def test_dispatcher_is_not_in_public_api_regression() -> None:
    """Regression for the item-10 review finding: callers must use
    `core.tasks.stop_task.stop_task`. The internal dispatcher is named
    `dispatch_stop_task` and excluded from `core.task.__all__`. A
    bare `stop_task` symbol on `core.task` would re-open the bypass
    surface."""
    assert "stop_task" not in core_task.__all__
    assert "dispatch_stop_task" not in core_task.__all__
    # Dispatcher remains importable for the wrapper's own use, but only
    # explicitly — star-imports won't pick it up.
    assert hasattr(core_task, "dispatch_stop_task")


@pytest.mark.asyncio
async def test_unsupported_task_type_raises_unsupported_error() -> None:
    """If a task with a registered impl is in the store but the impl is
    later removed (e.g., dynamic plugin teardown), the dispatcher must
    surface `UnsupportedTaskTypeError` rather than silently no-op."""
    from raygent_harness.core.task import UnsupportedTaskTypeError

    store = AppStateStore()
    s = await spawn_local_bash("sleep 5", store, agent_id="a1")
    await asyncio.sleep(0.05)

    # Temporarily yank the local_bash impl from the registry. Restored
    # in `finally` so the rest of the suite isn't affected.
    saved = core_task._REGISTRY.pop("local_bash")  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(UnsupportedTaskTypeError):
            await stop_task(s.id, store)
    finally:
        core_task._REGISTRY["local_bash"] = saved  # pyright: ignore[reportPrivateUsage]

    # Clean up the still-running bash task so conftest doesn't have to
    # fight a stuck driver.
    await stop_task(s.id, store)
