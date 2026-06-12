from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.sdk import RaygentPresetOptions, create_raygent
from raygent_harness.tools import BASH_TOOL_NAME
from recipes.create_raygent._support import StaticModelProvider, run_and_print, turn_tool_names


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-full-dev-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("full_developer"),
            model="demo-model",
            cwd=Path(tmp),
            preset="full_developer",
            preset_options=RaygentPresetOptions(
                allow_full_developer=True,
                allow_filesystem_mutation=True,
                allow_shell=True,
                allow_agents=True,
                allow_mcp=True,
                allow_worktree=True,
            ),
            permission_context=ToolPermissionContext(),
        )
        await run_and_print(session, "Run the full developer preset.")
        tools = await turn_tool_names(session)
        print(f"full_developer bash_enabled: {BASH_TOOL_NAME in tools}")


if __name__ == "__main__":
    asyncio.run(main())
