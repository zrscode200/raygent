from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.core.permissions import ToolPermissionContext
from raygent_harness.sdk import RaygentPresetOptions, create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print, turn_tool_names


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-maintainer-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("repo_maintainer"),
            model="demo-model",
            cwd=Path(tmp),
            session_id="repo-maintainer-recipe",
            preset="repo_maintainer",
            preset_options=RaygentPresetOptions(
                allow_filesystem_mutation=True,
            ),
            permission_context=ToolPermissionContext(),
        )
        await run_and_print(session, "Run the repo maintainer preset.")
        print(f"repo_maintainer tools: {', '.join(await turn_tool_names(session))}")
        print(f"repo_maintainer output_dir: {session.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
