"""Auto-memory path gates and validation.

"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

AUTO_MEM_DIRNAME = "memory"
AUTO_MEM_ENTRYPOINT_NAME = "MEMORY.md"
DEFAULT_MEMORY_HOME_DIRNAME = ".raygent"
DEFAULT_MEMORY_ROOT_DIRNAME = "memory"
MAX_SANITIZED_LENGTH = 200

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})
_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:[/\\]?$|^[A-Za-z]:$")
_UNSAFE_ABSOLUTE_PREFIXES = ("//", "\\\\")


@dataclass(frozen=True)
class MemorySettings:
    """Injectable memory path inputs.

    The reference reads process env, settings files, homedir, and project root
    globals directly. Raygent keeps those as explicit inputs so tests and SDK
    callers can resolve memory paths without mutating global process state.
    """

    project_root: Path
    home_dir: Path = field(default_factory=Path.home)
    memory_base_dir: Path | None = None
    remote_memory_dir: Path | None = None
    disable_auto_memory: str | bool | None = None
    simple_mode: bool = False
    remote_mode: bool = False
    auto_memory_enabled: bool | None = None
    auto_memory_path_override: str | None = None
    auto_memory_directory: str | None = None
    canonical_project_root: Path | None = None
    team_memory_enabled: bool = False


def _is_env_truthy(raw: str | bool | None) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _is_env_defined_falsy(raw: str | bool | None) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return not raw
    if raw == "":
        return False
    return raw.strip().lower() in _FALSY_ENV_VALUES


def is_auto_memory_enabled(settings: MemorySettings) -> bool:
    """Return whether auto-memory mechanics are enabled.

    Mirrors `isAutoMemoryEnabled()` priority: explicit disable env, explicit
    falsy env re-enable, simple/bare-mode disable, remote-without-storage
    disable, settings override, then default enabled.
    """
    if _is_env_truthy(settings.disable_auto_memory):
        return False
    if _is_env_defined_falsy(settings.disable_auto_memory):
        return True
    if settings.simple_mode:
        return False
    if settings.remote_mode and settings.remote_memory_dir is None:
        return False
    if settings.auto_memory_enabled is not None:
        return settings.auto_memory_enabled
    return True


def is_extract_mode_active(
    *, feature_enabled: bool, non_interactive: bool = False, allow_non_interactive: bool = False
) -> bool:
    """Return whether background extraction should run this session.

    Reference keeps the feature-flag check outside `isExtractModeActive()` for
    tree-shaking. Python has no equivalent build-time concern, so the flag is
    explicit here.
    """
    if not feature_enabled:
        return False
    return (not non_interactive) or allow_non_interactive


def validate_memory_path(raw: str | None, *, expand_tilde: bool, home_dir: Path) -> Path | None:
    """Normalize and validate a candidate auto-memory directory path.

    Returns a normalized absolute `Path`, or `None` when the path is unset or
    unsafe. This ports the reference reject list: relative paths, root/near-root
    paths, Windows drive roots, UNC/network paths, null bytes, and dangerous
    tilde expansions.
    """
    if not raw:
        return None
    if "\0" in raw:
        return None

    candidate = raw
    if expand_tilde and (candidate.startswith("~/") or candidate.startswith("~\\")):
        rest = candidate[2:]
        rest_norm = os.path.normpath(rest or ".")
        if rest_norm in {".", ".."} or rest_norm.startswith("../") or rest_norm.startswith("..\\"):
            return None
        candidate = os.path.join(os.fspath(home_dir), rest)

    normalized = os.path.normpath(candidate).rstrip("/\\")
    normalized = unicodedata.normalize("NFC", normalized)

    if (
        not os.path.isabs(normalized)
        or len(normalized) < 3
        or _DRIVE_ROOT_RE.match(normalized) is not None
        or normalized.startswith(_UNSAFE_ABSOLUTE_PREFIXES)
        or "\0" in normalized
    ):
        return None

    return Path(normalized)


def _normalize_absolute_path(path: Path) -> Path:
    normalized = os.path.normpath(os.fspath(path)).rstrip("/\\")
    return Path(unicodedata.normalize("NFC", normalized))


def _validated_config_path(path: Path) -> Path:
    validated = validate_memory_path(os.fspath(path), expand_tilde=False, home_dir=Path.home())
    if validated is None:
        raise ValueError(f"unsafe memory path: {path}")
    return validated


def get_memory_base_dir(settings: MemorySettings) -> Path:
    """Return the base directory under which project memory dirs live."""
    if settings.remote_memory_dir is not None:
        return _validated_config_path(settings.remote_memory_dir)
    if settings.memory_base_dir is not None:
        return _validated_config_path(settings.memory_base_dir)
    return _normalize_absolute_path(
        settings.home_dir / DEFAULT_MEMORY_HOME_DIRNAME / DEFAULT_MEMORY_ROOT_DIRNAME
    )


def has_auto_mem_path_override(settings: MemorySettings) -> bool:
    """True only for a valid direct full-path override."""
    return (
        validate_memory_path(
            settings.auto_memory_path_override,
            expand_tilde=False,
            home_dir=settings.home_dir,
        )
        is not None
    )


def get_auto_mem_path_override(settings: MemorySettings) -> Path | None:
    """Return the direct auto-memory path override, if valid."""
    return validate_memory_path(
        settings.auto_memory_path_override,
        expand_tilde=False,
        home_dir=settings.home_dir,
    )


def get_auto_mem_path_setting(settings: MemorySettings) -> Path | None:
    """Return the trusted settings auto-memory directory, if valid."""
    return validate_memory_path(
        settings.auto_memory_directory,
        expand_tilde=True,
        home_dir=settings.home_dir,
    )


def _djb2_hash(raw: str) -> int:
    value = 0
    # JS `charCodeAt` iterates UTF-16 code units, not Unicode code points.
    # Match that so long non-ASCII path keys hash to the same suffix as reference.
    encoded = raw.encode("utf-16-le", errors="surrogatepass")
    for index in range(0, len(encoded), 2):
        code_unit = encoded[index] | (encoded[index + 1] << 8)
        value = ((value << 5) - value + code_unit) & 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def _base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    out = ""
    while value:
        value, remainder = divmod(value, 36)
        out = digits[remainder] + out
    return out


def sanitize_path(raw: str) -> str:
    """Make `raw` safe as one path component.

    Mirrors reference `sanitizePath`: replace non-alphanumerics with hyphens,
    and for components longer than 200 chars append a deterministic djb2 hash.
    """
    # JS regex replacement without `/u` works over UTF-16 code units. Python
    # regex works over Unicode code points, which would turn an emoji into one
    # hyphen instead of the reference's two surrogate-code-unit hyphens.
    encoded = raw.encode("utf-16-le", errors="surrogatepass")
    parts: list[str] = []
    for index in range(0, len(encoded), 2):
        code_unit = encoded[index] | (encoded[index + 1] << 8)
        if (48 <= code_unit <= 57) or (65 <= code_unit <= 90) or (97 <= code_unit <= 122):
            parts.append(chr(code_unit))
        else:
            parts.append("-")
    sanitized = "".join(parts)
    if len(sanitized) <= MAX_SANITIZED_LENGTH:
        return sanitized
    suffix = _base36(abs(_djb2_hash(raw)))
    return f"{sanitized[:MAX_SANITIZED_LENGTH]}-{suffix}"


def get_auto_mem_base(settings: MemorySettings) -> Path:
    """Return canonical project root when supplied, otherwise project root."""
    return _normalize_absolute_path(settings.canonical_project_root or settings.project_root)


def get_auto_mem_path(settings: MemorySettings) -> Path:
    """Resolve the auto-memory directory path.

    Resolution order: direct full-path override, trusted settings full-path
    override, then `<memory-base>/projects/<sanitized-project-root>/memory`.
    """
    override = get_auto_mem_path_override(settings) or get_auto_mem_path_setting(settings)
    if override is not None:
        return override
    return _normalize_absolute_path(
        get_memory_base_dir(settings)
        / "projects"
        / sanitize_path(os.fspath(get_auto_mem_base(settings)))
        / AUTO_MEM_DIRNAME
    )


def get_auto_mem_entrypoint(settings: MemorySettings) -> Path:
    """Return `MEMORY.md` inside the auto-memory directory."""
    return get_auto_mem_path(settings) / AUTO_MEM_ENTRYPOINT_NAME


def is_auto_mem_path(absolute_path: Path | str, settings: MemorySettings) -> bool:
    """Return whether `absolute_path` is inside the auto-memory directory."""
    raw_path = os.fspath(absolute_path)
    if "\0" in raw_path:
        return False
    normalized = _normalize_absolute_path(Path(raw_path))
    if not os.path.isabs(os.fspath(normalized)):
        return False
    memory_dir = get_auto_mem_path(settings)
    try:
        return os.path.commonpath([os.fspath(normalized), os.fspath(memory_dir)]) == os.fspath(
            memory_dir
        )
    except ValueError:
        return False


__all__ = [
    "AUTO_MEM_DIRNAME",
    "AUTO_MEM_ENTRYPOINT_NAME",
    "DEFAULT_MEMORY_HOME_DIRNAME",
    "DEFAULT_MEMORY_ROOT_DIRNAME",
    "MAX_SANITIZED_LENGTH",
    "MemorySettings",
    "get_auto_mem_base",
    "get_auto_mem_entrypoint",
    "get_auto_mem_path",
    "get_auto_mem_path_override",
    "get_auto_mem_path_setting",
    "get_memory_base_dir",
    "has_auto_mem_path_override",
    "is_auto_mem_path",
    "is_auto_memory_enabled",
    "is_extract_mode_active",
    "sanitize_path",
    "validate_memory_path",
]
