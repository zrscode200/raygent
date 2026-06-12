"""Shared test fixtures.

Auto-cleanup of process registries and driver tasks between tests so a
hung subprocess or driver from one test cannot leak into the next.
Module-level registries (`_PROCESSES`, `_DRIVER_TASKS`, `_ABORT_EVENTS`)
hold non-serializable refs that AppStateStore deliberately doesn't
own — clearing them via the impl modules' own helpers would mean
adding a prod-only "test reset" surface, so we reach into them
directly here under explicit ignores.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from raygent_harness.core.tasks import in_process_teammate as _teammate_mod
from raygent_harness.core.tasks import local_agent as _agent_mod
from raygent_harness.core.tasks import local_bash as _bash_mod
from raygent_harness.core.tasks import remote_agent as _remote_mod
from raygent_harness.services.compact import clear_post_compact_cleanup_hooks
from raygent_harness.services.task_output import FileTaskOutputStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(autouse=True)
async def reset_task_registries(tmp_path: Path) -> AsyncIterator[None]:
    """Cancel any drivers + reap any subprocesses left in the module
    registries after each test. Best-effort; suppresses errors so a
    failing test never wedges the suite.
    """
    previous_output_store = _bash_mod._default_output_store_instance  # pyright: ignore[reportPrivateUsage]
    _bash_mod._default_output_store_instance = FileTaskOutputStore(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "task-output",
        session_id="test",
    )
    yield

    # 1. Cancel any local_bash drivers; SIGKILL any procs still alive.
    #    Use the production process-tree kill so children spawned by the
    #    shell command are reaped too — a direct `proc.send_signal` only
    #    targets the parent PID, leaving grandchildren as PPID=1
    #    zombies. Expected behavior: process-tree cleanup, not direct-child-only cleanup.
    bash_drivers = list(_bash_mod._DRIVER_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for driver in bash_drivers:
        if not driver.done():
            driver.cancel()
    for driver in bash_drivers:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(driver, timeout=1.0)

    bash_procs = list(_bash_mod._PROCESSES.values())  # pyright: ignore[reportPrivateUsage]
    for proc in bash_procs:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                _bash_mod._kill_process_tree(proc, signal.SIGKILL)  # pyright: ignore[reportPrivateUsage]
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=1.0)

    _bash_mod._DRIVER_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    _bash_mod._PROCESSES.clear()  # pyright: ignore[reportPrivateUsage]
    _bash_mod._default_output_store_instance = previous_output_store  # pyright: ignore[reportPrivateUsage]

    # 2. Cancel any local_agent drivers/sidecar saves and clear abort-event registry.
    agent_drivers = list(_agent_mod._DRIVER_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for driver in agent_drivers:
        if not driver.done():
            driver.cancel()
    for driver in agent_drivers:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(driver, timeout=1.0)

    route_saves = list(_agent_mod._ROUTE_PERSISTENCE_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for task in route_saves:
        if not task.done():
            task.cancel()
    for task in route_saves:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    _agent_mod._DRIVER_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    _agent_mod._ABORT_EVENTS.clear()  # pyright: ignore[reportPrivateUsage]
    _agent_mod._ROUTE_PERSISTENCE_TASKS.clear()  # pyright: ignore[reportPrivateUsage]

    # 3. Cancel remote pollers.
    remote_pollers = list(_remote_mod._POLL_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for poller in remote_pollers:
        if not poller.done():
            poller.cancel()
    for poller in remote_pollers:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(poller, timeout=1.0)

    _remote_mod._POLL_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    remote_stoppers = list(_remote_mod._STOP_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for stopper in remote_stoppers:
        if not stopper.done():
            stopper.cancel()
    for stopper in remote_stoppers:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(stopper, timeout=1.0)

    _remote_mod._STOP_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    _remote_mod._BACKENDS.clear()  # pyright: ignore[reportPrivateUsage]
    remote_persistence_tasks = list(_remote_mod._PERSISTENCE_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for task in remote_persistence_tasks:
        if not task.done():
            task.cancel()
    for task in remote_persistence_tasks:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    _remote_mod._PERSISTENCE_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    _remote_mod._PERSISTENCE_STORES.clear()  # pyright: ignore[reportPrivateUsage]

    # 4. Cancel any persistent teammate drivers.
    teammate_drivers = list(_teammate_mod._DRIVER_TASKS.values())  # pyright: ignore[reportPrivateUsage]
    for driver in teammate_drivers:
        if not driver.done():
            driver.cancel()
    for driver in teammate_drivers:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(driver, timeout=1.0)

    _teammate_mod._DRIVER_TASKS.clear()  # pyright: ignore[reportPrivateUsage]
    _teammate_mod._ABORT_EVENTS.clear()  # pyright: ignore[reportPrivateUsage]
    _teammate_mod._WAKE_EVENTS.clear()  # pyright: ignore[reportPrivateUsage]
    _teammate_mod._TEAM_STORES.clear()  # pyright: ignore[reportPrivateUsage]

    clear_post_compact_cleanup_hooks()
