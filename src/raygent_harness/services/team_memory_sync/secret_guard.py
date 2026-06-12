"""Team-memory write guard for detected secrets.

"""

from __future__ import annotations

from pathlib import Path

from raygent_harness.memdir.paths import MemorySettings
from raygent_harness.memdir.team_paths import is_team_mem_path
from raygent_harness.services.team_memory_sync.secret_scanner import scan_for_secrets


def check_team_mem_secrets(
    file_path: Path | str,
    content: str,
    settings: MemorySettings,
) -> str | None:
    """Return a model-visible write error when team-memory content has secrets."""
    if not is_team_mem_path(file_path, settings):
        return None

    matches = scan_for_secrets(content)
    if len(matches) == 0:
        return None

    labels = ", ".join(match.label for match in matches)
    return (
        f"Content contains potential secrets ({labels}) and cannot be written to team memory. "
        "Team memory is shared with all repository collaborators. "
        "Remove the sensitive content and try again."
    )


__all__ = ["check_team_mem_secrets"]
