from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.tool import ToolUseContext
from raygent_harness.sdk import RaygentMemoryOptions, create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print


async def memory_prompt_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> str:
    return "Use caller-provided memory services when relevant."


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-memory-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("memory_agent"),
            model="demo-model",
            cwd=Path(tmp),
            preset="memory_agent",
            memory_options=RaygentMemoryOptions(prompt_provider=memory_prompt_provider),
        )
        await run_and_print(session, "Run the memory preset.")
        print(f"memory_agent prompt_provider: {session.deps.memory_prompt_provider is not None}")


if __name__ == "__main__":
    asyncio.run(main())
