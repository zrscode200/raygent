"""Team-memory path gates and traversal-safe validation.

"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from raygent_harness.memdir.paths import (
    AUTO_MEM_ENTRYPOINT_NAME,
    MemorySettings,
    get_auto_mem_path,
    is_auto_memory_enabled,
)

TEAM_MEM_DIRNAME = "team"


class PathTraversalError(ValueError):
    """Raised when a team-memory path would escape the team directory."""


def is_team_memory_enabled(settings: MemorySettings) -> bool:
    """Return whether team memory is enabled for this session."""
    return is_auto_memory_enabled(settings) and settings.team_memory_enabled


def _normalize_absolute_no_symlink(path: Path | str) -> Path:
    raw = os.fspath(path)
    normalized = os.path.abspath(os.path.normpath(raw))
    return Path(unicodedata.normalize("NFC", normalized))


def get_team_mem_path(settings: MemorySettings) -> Path:
    """Return `<auto-memory>/team` for the current project."""
    return _normalize_absolute_no_symlink(get_auto_mem_path(settings) / TEAM_MEM_DIRNAME)


def get_team_mem_entrypoint(settings: MemorySettings) -> Path:
    """Return `MEMORY.md` inside the team-memory directory."""
    return get_team_mem_path(settings) / AUTO_MEM_ENTRYPOINT_NAME


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([os.fspath(candidate), os.fspath(parent)]) == os.fspath(parent)
    except ValueError:
        return False


def _is_descendant(candidate: Path, parent: Path) -> bool:
    return candidate != parent and _is_within(candidate, parent)


def _sanitize_path_key(key: str) -> str:
    if key == "":
        raise PathTraversalError("empty path key")
    if "\0" in key:
        raise PathTraversalError(f'null byte in path key: "{key}"')

    decoded = key
    try:
        decoded = unquote(key)
    except Exception:
        decoded = key
    if decoded != key and (".." in decoded or "/" in decoded):
        raise PathTraversalError(f'URL-encoded traversal in path key: "{key}"')

    normalized = unicodedata.normalize("NFKC", key)
    if normalized != key and (
        ".." in normalized or "/" in normalized or "\\" in normalized or "\0" in normalized
    ):
        raise PathTraversalError(f'unicode-normalized traversal in path key: "{key}"')

    if "\\" in key:
        raise PathTraversalError(f'backslash in path key: "{key}"')
    if key.startswith("/"):
        raise PathTraversalError(f'absolute path key: "{key}"')
    return key


def _realpath_deepest_existing(absolute_path: Path) -> Path:
    tail: list[str] = []
    current = absolute_path

    while True:
        try:
            resolved = current.resolve(strict=True)
            return resolved.joinpath(*reversed(tail)) if tail else resolved
        except FileNotFoundError:
            try:
                current.lstat()
                if current.is_symlink():
                    raise PathTraversalError(
                        f'dangling symlink detected: "{os.fspath(current)}"'
                    )
            except FileNotFoundError:
                pass
        except NotADirectoryError:
            pass
        except RuntimeError as exc:
            raise PathTraversalError(f'symlink loop detected: "{os.fspath(current)}"') from exc
        except OSError as exc:
            if exc.errno not in {20, 36}:  # ENOTDIR, ENAMETOOLONG
                raise PathTraversalError(
                    f'cannot verify path containment: "{os.fspath(current)}"'
                ) from exc

        parent = current.parent
        if parent == current:
            return absolute_path
        tail.append(current.name)
        current = parent


def _is_real_path_within_team_dir(real_candidate: Path, settings: MemorySettings) -> bool:
    team_dir = get_team_mem_path(settings)
    try:
        real_team_dir = team_dir.resolve(strict=True)
    except FileNotFoundError:
        return True
    except NotADirectoryError:
        return True
    except OSError:
        return False

    return _is_within(real_candidate, real_team_dir)


def is_team_mem_path(file_path: Path | str, settings: MemorySettings) -> bool:
    """Return whether `file_path` is inside the team-memory directory."""
    raw = os.fspath(file_path)
    if "\0" in raw:
        return False
    resolved = _normalize_absolute_no_symlink(raw)
    return _is_descendant(resolved, get_team_mem_path(settings))


def validate_team_mem_write_path(file_path: Path | str, settings: MemorySettings) -> Path:
    """Validate an absolute write path against the team-memory boundary."""
    raw = os.fspath(file_path)
    if "\0" in raw:
        raise PathTraversalError(f'null byte in path: "{raw}"')

    resolved = _normalize_absolute_no_symlink(raw)
    team_dir = get_team_mem_path(settings)
    if not _is_descendant(resolved, team_dir):
        raise PathTraversalError(f'path escapes team memory directory: "{raw}"')

    real_path = _realpath_deepest_existing(resolved)
    if not _is_real_path_within_team_dir(real_path, settings):
        raise PathTraversalError(f'path escapes team memory directory via symlink: "{raw}"')
    return resolved


def validate_team_mem_key(relative_key: str, settings: MemorySettings) -> Path:
    """Validate a server-provided relative key and return its local path."""
    sanitized = _sanitize_path_key(relative_key)
    full_path = get_team_mem_path(settings) / sanitized
    resolved = _normalize_absolute_no_symlink(full_path)
    team_dir = get_team_mem_path(settings)
    if not _is_descendant(resolved, team_dir):
        raise PathTraversalError(f'key escapes team memory directory: "{relative_key}"')

    real_path = _realpath_deepest_existing(resolved)
    if not _is_real_path_within_team_dir(real_path, settings):
        raise PathTraversalError(
            f'key escapes team memory directory via symlink: "{relative_key}"'
        )
    return resolved


def is_team_mem_file(file_path: Path | str, settings: MemorySettings) -> bool:
    """Return whether `file_path` is an enabled team-memory file."""
    return is_team_memory_enabled(settings) and is_team_mem_path(file_path, settings)


__all__ = [
    "TEAM_MEM_DIRNAME",
    "PathTraversalError",
    "get_team_mem_entrypoint",
    "get_team_mem_path",
    "is_team_mem_file",
    "is_team_mem_path",
    "is_team_memory_enabled",
    "validate_team_mem_key",
    "validate_team_mem_write_path",
]
