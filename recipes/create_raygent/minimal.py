from __future__ import annotations

import asyncio

from raygent_harness.sdk import create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print


async def main() -> None:
    session = create_raygent(
        provider=StaticModelProvider("minimal"),
        model="demo-model",
        preset="minimal",
    )
    await run_and_print(session, "Run the minimal preset.")
    print("minimal preset ready")


if __name__ == "__main__":
    asyncio.run(main())
