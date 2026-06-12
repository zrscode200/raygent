from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.sdk import create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-long-task-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("long_running_task"),
            model="demo-model",
            cwd=Path(tmp),
            session_id="long-running-task-recipe",
            preset="long_running_task",
        )
        await run_and_print(session, "Run the long-running task preset.")
        print(f"long_running_task transcript: {session.transcript_path}")
        print(f"long_running_task output_dir: {session.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
