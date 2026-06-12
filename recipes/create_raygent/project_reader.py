from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.sdk import create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print, turn_tool_names


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-reader-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("project_reader"),
            model="demo-model",
            cwd=Path(tmp),
            preset="project_reader",
        )
        await run_and_print(session, "Run the project reader preset.")
        print(f"project_reader tools: {', '.join(await turn_tool_names(session))}")


if __name__ == "__main__":
    asyncio.run(main())
