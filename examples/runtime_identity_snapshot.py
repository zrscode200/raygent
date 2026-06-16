"""Runtime identity snapshots over supplied Raygent handles.

Run from the project root:

    uv run python examples/runtime_identity_snapshot.py

The runtime identity layer exposes kernel descriptors. Product applications can
index or render these descriptors, but Raygent does not provide a catalog,
search ranking layer, dashboard, or product UI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.core.observability import NoopKernelEventBus
from raygent_harness.core.task import AppStateStore, TaskStateBase
from raygent_harness.sdk import RaygentRuntimeHandles
from raygent_harness.services.runtime_identity import (
    RuntimeIdentitySnapshotOptions,
    describe_runtime_session,
)
from raygent_harness.services.task_output import FileTaskOutputStore
from raygent_harness.services.transcript import TranscriptScope


def main() -> None:
    with TemporaryDirectory(prefix="raygent-runtime-identity-") as tmp:
        root = Path(tmp)
        task_store = AppStateStore()
        task_store.register_task(
            TaskStateBase(
                id="task-1",
                type="local_bash",
                description="run local checks",
                status="running",
                start_time=1.0,
                tool_use_id="toolu-1",
                output_file=str(root / "private-task-output.txt"),
            )
        )

        handles = RaygentRuntimeHandles(
            session_id="runtime-identity-example",
            cwd=str(root),
            task_store=task_store,
            output_dir=root / ".raygent" / "task-output",
            task_output_store=FileTaskOutputStore(
                base_dir=root / ".raygent" / "task-output",
                session_id="runtime-identity-example",
            ),
            transcript_store=None,
            transcript_scope=TranscriptScope(
                session_id="runtime-identity-example",
                runtime_session_id="runtime-1",
                agent_id="agent-1",
            ),
            observability=NoopKernelEventBus(),
            abort_event=asyncio.Event(),
        )

        snapshot = describe_runtime_session(
            handles,
            options=RuntimeIdentitySnapshotOptions(
                include_goal_runtime=False,
                max_tasks=2,
            ),
        )
        kinds = ", ".join(
            sorted({descriptor.ref.kind for descriptor in snapshot.descriptors})
        )

        print(f"runtime identity snapshot: {snapshot.session_id}")
        print(f"kinds: {kinds}")
        print(f"warnings: {', '.join(snapshot.warnings) or 'none'}")


if __name__ == "__main__":
    main()
