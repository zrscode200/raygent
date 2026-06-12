"""Headless transcript path resolution."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from raygent_harness.services.transcript.models import TranscriptScope

DEFAULT_TRANSCRIPT_DIR = ".raygent/transcripts"
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def resolve_transcript_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the root directory for Raygent transcript JSONL files."""

    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / DEFAULT_TRANSCRIPT_DIR).resolve()


def transcript_path_for_scope(base_dir: str | Path, scope: TranscriptScope) -> Path:
    """Return the JSONL path for a main-session or sidechain scope."""

    base = Path(base_dir).expanduser().resolve()
    session_part = safe_transcript_component(scope.session_id)
    if not scope.is_sidechain:
        return _ensure_under_base(base / f"{session_part}.jsonl", base)

    if scope.agent_id is None or scope.agent_id == "":
        raise ValueError("sidechain transcript scope requires agent_id")
    agent_part = safe_transcript_component(scope.agent_id)
    return _ensure_under_base(
        base / session_part / "subagents" / f"agent-{agent_part}.jsonl",
        base,
    )


def safe_transcript_component(value: str) -> str:
    """Make a session or agent id safe for a path component.

    This is intentionally conservative. The unsanitized id stays in the JSONL
    entry; the filesystem path only needs a stable, traversal-safe component.
    Unsafe ids receive a short hash suffix so distinct ids such as `a/b` and
    `a_b` do not silently collide.
    """

    if value == "":
        raise ValueError("transcript path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    if normalized == value:
        return normalized
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    return f"{normalized}-{digest}"


def _ensure_under_base(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"transcript path escapes base directory: {resolved}") from exc
    return resolved


__all__ = [
    "DEFAULT_TRANSCRIPT_DIR",
    "resolve_transcript_base_dir",
    "safe_transcript_component",
    "transcript_path_for_scope",
]
