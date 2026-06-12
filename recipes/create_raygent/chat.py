from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from raygent_harness.sdk import create_raygent
from recipes.create_raygent._support import StaticModelProvider, run_and_print


async def main() -> None:
    with TemporaryDirectory(prefix="raygent-chat-recipe-") as tmp:
        session = create_raygent(
            provider=StaticModelProvider("chat"),
            model="demo-model",
            cwd=Path(tmp),
            session_id="chat-recipe",
            preset="chat",
        )
        await run_and_print(session, "Run the chat preset.")
        print(f"chat transcript: {session.transcript_path}")


if __name__ == "__main__":
    asyncio.run(main())
