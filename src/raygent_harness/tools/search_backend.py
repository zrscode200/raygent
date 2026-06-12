"""Backend-neutral local search primitives for discovery tools.

Wave 1 keeps the default implementation dependency-light, but the public
contract is intentionally backend-shaped so a ripgrep-backed implementation can
replace it without changing the model-callable `Glob`/`Grep` tools.
"""

from __future__ import annotations

import asyncio
import fnmatch
import multiprocessing as mp
import os
import queue
import re
import stat
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

GREP_OUTPUT_MODES = ("content", "files_with_matches", "count")
type GrepOutputMode = Literal["content", "files_with_matches", "count"]

DEFAULT_GLOB_LIMIT = 100
DEFAULT_GREP_HEAD_LIMIT = 250
DEFAULT_MAX_GREP_LINE_LENGTH = 500
DEFAULT_GREP_TIMEOUT_S = 10.0

VCS_DIRECTORIES_TO_EXCLUDE = frozenset({".git", ".svn", ".hg", ".bzr", ".jj", ".sl"})

COMMON_TYPE_GLOBS: dict[str, tuple[str, ...]] = {
    "c": ("*.c", "*.h"),
    "cpp": ("*.cc", "*.cpp", "*.cxx", "*.hpp", "*.hh", "*.hxx"),
    "cs": ("*.cs",),
    "css": ("*.css",),
    "go": ("*.go",),
    "html": ("*.html", "*.htm"),
    "java": ("*.java",),
    "js": ("*.js", "*.mjs", "*.cjs", "*.jsx"),
    "json": ("*.json",),
    "jsx": ("*.jsx",),
    "md": ("*.md", "*.markdown"),
    "php": ("*.php",),
    "py": ("*.py", "*.pyi"),
    "rb": ("*.rb",),
    "rs": ("*.rs",),
    "rust": ("*.rs",),
    "sh": ("*.sh", "*.bash", "*.zsh"),
    "toml": ("*.toml",),
    "ts": ("*.ts", "*.tsx"),
    "tsx": ("*.tsx",),
    "txt": ("*.txt",),
    "yaml": ("*.yaml", "*.yml"),
    "yml": ("*.yml", "*.yaml"),
}


@dataclass(frozen=True)
class SearchPath:
    path: Path
    mtime_ms: int


@dataclass(frozen=True)
class GlobSearchRequest:
    pattern: str
    root: Path
    limit: int = DEFAULT_GLOB_LIMIT
    offset: int = 0
    include_hidden: bool = True
    exclude_vcs: bool = True
    abort_event: asyncio.Event | None = None
    is_path_allowed: Callable[[Path], bool] = field(default=lambda _path: True)


@dataclass(frozen=True)
class GlobSearchResult:
    files: tuple[SearchPath, ...]
    truncated: bool
    total_matches: int


@dataclass(frozen=True)
class GrepSearchRequest:
    pattern: str
    root: Path
    glob_patterns: tuple[str, ...] = ()
    output_mode: GrepOutputMode = "files_with_matches"
    case_insensitive: bool = False
    show_line_numbers: bool = True
    context_before: int = 0
    context_after: int = 0
    head_limit: int | None = DEFAULT_GREP_HEAD_LIMIT
    offset: int = 0
    multiline: bool = False
    type_name: str | None = None
    exclude_vcs: bool = True
    max_line_length: int = DEFAULT_MAX_GREP_LINE_LENGTH
    timeout_s: float = DEFAULT_GREP_TIMEOUT_S
    abort_event: asyncio.Event | None = None
    is_path_allowed: Callable[[Path], bool] = field(default=lambda _path: True)


@dataclass(frozen=True)
class GrepContentLine:
    path: Path
    line_number: int
    text: str
    is_match: bool = True


@dataclass(frozen=True)
class GrepFileMatch:
    path: Path
    mtime_ms: int
    match_count: int


@dataclass(frozen=True)
class GrepSearchResult:
    mode: GrepOutputMode
    files: tuple[GrepFileMatch, ...] = ()
    content_lines: tuple[GrepContentLine, ...] = ()
    count_lines: tuple[tuple[Path, int], ...] = ()
    total_matches: int = 0
    partial: bool = False
    applied_limit: int | None = None
    applied_offset: int | None = None


class SearchBackendError(Exception):
    """Base class for backend-level search failures."""


class SearchTimeoutError(SearchBackendError):
    """Raised when a backend search exceeds its hard execution timeout."""


class SearchBackend(Protocol):
    """Backend seam for local discovery tools."""

    async def glob(self, request: GlobSearchRequest) -> GlobSearchResult:
        ...

    async def grep(self, request: GrepSearchRequest) -> GrepSearchResult:
        ...


class _TerminableProcess(Protocol):
    def terminate(self) -> None:
        ...

    def join(self, timeout: float | None = None) -> None:
        ...

    def is_alive(self) -> bool:
        ...

    def kill(self) -> None:
        ...


class StdlibSearchBackend:
    """Dependency-light search backend.

    This is not intended to be a byte-for-byte ripgrep clone. It implements the
    Wave 1 semantic subset: deterministic ordering, hidden-file inclusion,
    VCS-directory exclusion, glob/type filtering, per-file permission filtering,
    and bounded model-visible result sets.
    """

    async def glob(self, request: GlobSearchRequest) -> GlobSearchResult:
        return await asyncio.to_thread(self._glob_sync, request)

    async def grep(self, request: GrepSearchRequest) -> GrepSearchResult:
        return await asyncio.to_thread(self._grep_sync, request)

    def _glob_sync(self, request: GlobSearchRequest) -> GlobSearchResult:
        root = request.root
        matches: list[SearchPath] = []
        for path in _iter_files(
            root,
            include_hidden=request.include_hidden,
            exclude_vcs=request.exclude_vcs,
            abort_event=request.abort_event,
        ):
            _raise_if_aborted(request.abort_event)
            if not request.is_path_allowed(path):
                continue
            rel = _relative_posix(path, root)
            if not _matches_any_glob(rel, path.name, (request.pattern,)):
                continue
            matches.append(SearchPath(path=path, mtime_ms=_mtime_ms(path)))

        matches.sort(key=lambda item: (-item.mtime_ms, str(item.path)))
        limited, applied_limit = _apply_limit(matches, request.limit, request.offset)
        return GlobSearchResult(
            files=tuple(limited),
            truncated=applied_limit is not None,
            total_matches=len(matches),
        )

    def _grep_sync(self, request: GrepSearchRequest) -> GrepSearchResult:
        candidate_files = tuple(_candidate_grep_files(request))
        return _run_grep_worker(
            _GrepWorkerRequest.from_request(request, candidate_files),
            timeout_s=request.timeout_s,
            abort_event=request.abort_event,
        )


def create_default_search_backend() -> SearchBackend:
    return StdlibSearchBackend()


def supported_type_names() -> tuple[str, ...]:
    return tuple(sorted(COMMON_TYPE_GLOBS))


def expand_glob_patterns(pattern_text: str | None) -> tuple[str, ...]:
    if pattern_text is None:
        return ()

    patterns: list[str] = []
    for raw_part in pattern_text.split():
        if "{" in raw_part and "}" in raw_part:
            patterns.append(raw_part)
        else:
            patterns.extend(part for part in raw_part.split(",") if part)

    expanded: list[str] = []
    for pattern in patterns:
        expanded.extend(_expand_braces(pattern))
    return tuple(dict.fromkeys(expanded))


def extract_glob_base_directory(pattern: str) -> tuple[str, str]:
    """Split an absolute glob pattern into static root and relative pattern."""

    special_index = _first_glob_special_index(pattern)
    if special_index is None:
        return str(Path(pattern).parent), Path(pattern).name

    static_prefix = pattern[:special_index]
    slash_index = max(static_prefix.rfind("/"), static_prefix.rfind(os.sep))
    if slash_index == -1:
        return "", pattern
    base_dir = static_prefix[:slash_index] or os.sep
    return base_dir, pattern[slash_index + 1 :]


def _candidate_grep_files(request: GrepSearchRequest) -> tuple[SearchPath, ...]:
    root = request.root
    if root.is_file():
        paths = (root,)
        base_root = root.parent
    else:
        paths = tuple(
            _iter_files(
                root,
                include_hidden=True,
                exclude_vcs=request.exclude_vcs,
                abort_event=request.abort_event,
            )
        )
        base_root = root

    glob_patterns = request.glob_patterns
    if request.type_name:
        glob_patterns = (*glob_patterns, *COMMON_TYPE_GLOBS[request.type_name])

    candidates: list[SearchPath] = []
    for path in paths:
        _raise_if_aborted(request.abort_event)
        if not request.is_path_allowed(path):
            continue
        rel = _relative_posix(path, base_root)
        if glob_patterns and not _matches_any_glob(rel, path.name, glob_patterns):
            continue
        if _looks_binary_or_special(path):
            continue
        candidates.append(SearchPath(path=path, mtime_ms=_mtime_ms(path)))

    candidates.sort(key=lambda item: str(item.path))
    return tuple(candidates)


@dataclass(frozen=True)
class _GrepWorkerRequest:
    pattern: str
    candidates: tuple[tuple[str, int], ...]
    output_mode: GrepOutputMode
    case_insensitive: bool
    show_line_numbers: bool
    context_before: int
    context_after: int
    head_limit: int | None
    offset: int
    multiline: bool
    max_line_length: int

    @classmethod
    def from_request(
        cls,
        request: GrepSearchRequest,
        candidates: Sequence[SearchPath],
    ) -> _GrepWorkerRequest:
        return cls(
            pattern=request.pattern,
            candidates=tuple((str(item.path), item.mtime_ms) for item in candidates),
            output_mode=request.output_mode,
            case_insensitive=request.case_insensitive,
            show_line_numbers=request.show_line_numbers,
            context_before=request.context_before,
            context_after=request.context_after,
            head_limit=request.head_limit,
            offset=request.offset,
            multiline=request.multiline,
            max_line_length=request.max_line_length,
        )

    def to_search_request(self) -> GrepSearchRequest:
        return GrepSearchRequest(
            pattern=self.pattern,
            root=Path("."),
            output_mode=self.output_mode,
            case_insensitive=self.case_insensitive,
            show_line_numbers=self.show_line_numbers,
            context_before=self.context_before,
            context_after=self.context_after,
            head_limit=self.head_limit,
            offset=self.offset,
            multiline=self.multiline,
            max_line_length=self.max_line_length,
        )

    def to_candidates(self) -> tuple[SearchPath, ...]:
        return tuple(
            SearchPath(path=Path(path), mtime_ms=mtime_ms)
            for path, mtime_ms in self.candidates
        )


def _run_grep_worker(
    request: _GrepWorkerRequest,
    *,
    timeout_s: float,
    abort_event: asyncio.Event | None = None,
) -> GrepSearchResult:
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_grep_worker_entry, args=(request, output_queue))
    process.start()
    deadline = time.monotonic() + timeout_s
    timed_out = False
    try:
        while process.is_alive():
            _raise_if_aborted(abort_event)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            process.join(min(remaining, 0.05))
    except asyncio.CancelledError:
        _terminate_grep_process(process)
        raise

    if not timed_out:
        process.join()

    if timed_out and process.is_alive():
        _terminate_grep_process(process)
        raise SearchTimeoutError(
            f"Grep search exceeded {timeout_s:.1f}s timeout. "
            "Use a more specific pattern/path or a safer regex."
        )

    try:
        kind, payload = output_queue.get_nowait()
    except queue.Empty as exc:
        raise SearchBackendError(
            f"Grep worker exited without a result (exit code {process.exitcode})."
        ) from exc
    finally:
        output_queue.close()
        output_queue.join_thread()

    if kind == "result":
        return cast("GrepSearchResult", payload)
    message = cast("str", payload)
    raise SearchBackendError(message)


def _terminate_grep_process(process: _TerminableProcess) -> None:
    process.terminate()
    process.join(1)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(1)


def _grep_worker_entry(request: _GrepWorkerRequest, output_queue: Any) -> None:
    try:
        output_queue.put(("result", _grep_worker(request)))
    except BaseException:  # pragma: no cover - exercised through parent error path
        output_queue.put(("error", traceback.format_exc(limit=5)))


def _grep_worker(request: _GrepWorkerRequest) -> GrepSearchResult:
    flags = re.MULTILINE
    if request.case_insensitive:
        flags |= re.IGNORECASE
    if request.multiline:
        flags |= re.DOTALL
    regex = re.compile(request.pattern, flags)

    search_request = request.to_search_request()
    candidates = request.to_candidates()
    if request.output_mode == "content":
        return _grep_content(regex, candidates, search_request)
    if request.output_mode == "count":
        return _grep_count(regex, candidates, search_request)
    return _grep_files_with_matches(regex, candidates, search_request)


def _grep_files_with_matches(
    regex: re.Pattern[str],
    candidates: tuple[SearchPath, ...],
    request: GrepSearchRequest,
) -> GrepSearchResult:
    matches: list[GrepFileMatch] = []
    partial = False
    for candidate in candidates:
        _raise_if_aborted(request.abort_event)
        try:
            text = candidate.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            partial = True
            continue
        count = _match_count(regex, text, multiline=request.multiline)
        if count:
            matches.append(
                GrepFileMatch(
                    path=candidate.path,
                    mtime_ms=candidate.mtime_ms,
                    match_count=count,
                )
            )

    matches.sort(key=lambda item: (-item.mtime_ms, str(item.path)))
    limited, applied_limit = _apply_limit(matches, request.head_limit, request.offset)
    return GrepSearchResult(
        mode="files_with_matches",
        files=tuple(limited),
        total_matches=sum(item.match_count for item in matches),
        partial=partial,
        applied_limit=applied_limit,
        applied_offset=request.offset or None,
    )


def _grep_count(
    regex: re.Pattern[str],
    candidates: tuple[SearchPath, ...],
    request: GrepSearchRequest,
) -> GrepSearchResult:
    count_lines: list[tuple[Path, int]] = []
    total = 0
    partial = False
    for candidate in candidates:
        _raise_if_aborted(request.abort_event)
        try:
            text = candidate.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            partial = True
            continue
        count = _match_count(regex, text, multiline=request.multiline)
        if count:
            total += count
            count_lines.append((candidate.path, count))

    count_lines.sort(key=lambda item: str(item[0]))
    limited, applied_limit = _apply_limit(count_lines, request.head_limit, request.offset)
    return GrepSearchResult(
        mode="count",
        count_lines=tuple(limited),
        total_matches=total,
        partial=partial,
        applied_limit=applied_limit,
        applied_offset=request.offset or None,
    )


def _grep_content(
    regex: re.Pattern[str],
    candidates: tuple[SearchPath, ...],
    request: GrepSearchRequest,
) -> GrepSearchResult:
    lines: list[GrepContentLine] = []
    partial = False
    for candidate in candidates:
        _raise_if_aborted(request.abort_event)
        try:
            text = candidate.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            partial = True
            continue
        lines.extend(_matching_content_lines(regex, candidate.path, text, request))

    limited, applied_limit = _apply_limit(lines, request.head_limit, request.offset)
    return GrepSearchResult(
        mode="content",
        content_lines=tuple(limited),
        total_matches=sum(1 for item in lines if item.is_match),
        partial=partial,
        applied_limit=applied_limit,
        applied_offset=request.offset or None,
    )


def _matching_content_lines(
    regex: re.Pattern[str],
    path: Path,
    text: str,
    request: GrepSearchRequest,
) -> tuple[GrepContentLine, ...]:
    raw_lines = text.splitlines()
    if not raw_lines:
        return ()

    match_line_indexes: set[int] = set()
    if request.multiline:
        for match in regex.finditer(text):
            start = text.count("\n", 0, match.start())
            end = text.count("\n", 0, match.end())
            match_line_indexes.update(range(start, min(end + 1, len(raw_lines))))
    else:
        for index, line in enumerate(raw_lines):
            if regex.search(line):
                match_line_indexes.add(index)

    selected: dict[int, bool] = {}
    for index in sorted(match_line_indexes):
        start = max(0, index - request.context_before)
        end = min(len(raw_lines), index + request.context_after + 1)
        for line_index in range(start, end):
            selected[line_index] = selected.get(line_index, False) or line_index == index

    return tuple(
        GrepContentLine(
            path=path,
            line_number=index + 1,
            text=_truncate_line(raw_lines[index], request.max_line_length),
            is_match=is_match,
        )
        for index, is_match in sorted(selected.items())
    )


def _iter_files(
    root: Path,
    *,
    include_hidden: bool,
    exclude_vcs: bool,
    abort_event: asyncio.Event | None,
) -> tuple[Path, ...]:
    root = root.resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        _raise_if_aborted(abort_event)
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if _should_descend(dirname, include_hidden=include_hidden, exclude_vcs=exclude_vcs)
        )
        for filename in sorted(filenames):
            if not include_hidden and filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            files.append(path.resolve())
    return tuple(files)


def _should_descend(dirname: str, *, include_hidden: bool, exclude_vcs: bool) -> bool:
    if exclude_vcs and dirname in VCS_DIRECTORIES_TO_EXCLUDE:
        return False
    return include_hidden or not dirname.startswith(".")


def _matches_any_glob(
    rel_posix: str,
    basename: str,
    patterns: tuple[str, ...],
) -> bool:
    include_patterns = tuple(pattern for pattern in patterns if not pattern.startswith("!"))
    exclude_patterns = tuple(pattern[1:] for pattern in patterns if pattern.startswith("!"))
    if exclude_patterns and any(
        _matches_one_glob(rel_posix, basename, pattern) for pattern in exclude_patterns
    ):
        return False
    if not include_patterns:
        return True
    return any(_matches_one_glob(rel_posix, basename, pattern) for pattern in include_patterns)


def _matches_one_glob(rel_posix: str, basename: str, pattern: str) -> bool:
    normalized = pattern.replace(os.sep, "/")
    candidates = (rel_posix, basename) if "/" not in normalized else (rel_posix,)
    patterns = _globstar_zero_directory_variants(normalized)
    return any(
        fnmatch.fnmatchcase(candidate, candidate_pattern)
        for candidate_pattern in patterns
        for candidate in candidates
    )


def _globstar_zero_directory_variants(pattern: str) -> tuple[str, ...]:
    variants = {pattern}
    queue_: list[str] = [pattern]
    while queue_:
        current = queue_.pop()
        index = current.find("**/")
        while index != -1:
            variant = current[:index] + current[index + 3 :]
            if variant not in variants:
                variants.add(variant)
                queue_.append(variant)
            index = current.find("**/", index + 3)
    return tuple(variants)


def _apply_limit[T](
    items: list[T],
    limit: int | None,
    offset: int,
) -> tuple[list[T], int | None]:
    if limit == 0:
        return items[offset:], None
    effective_limit = DEFAULT_GREP_HEAD_LIMIT if limit is None else limit
    sliced = items[offset : offset + effective_limit]
    applied_limit = effective_limit if len(items) - offset > effective_limit else None
    return sliced, applied_limit


def _match_count(regex: re.Pattern[str], text: str, *, multiline: bool) -> int:
    if multiline:
        return sum(1 for _match in regex.finditer(text))
    return sum(1 for line in text.splitlines() if regex.search(line))


def _looks_binary_or_special(path: Path) -> bool:
    try:
        stats = path.stat()
    except OSError:
        return True
    if not stat.S_ISREG(stats.st_mode):
        return True
    try:
        with path.open("rb") as file:
            return b"\x00" in file.read(8192)
    except OSError:
        return True


def _truncate_line(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _mtime_ms(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns // 1_000_000)
    except OSError:
        return 0


def _first_glob_special_index(pattern: str) -> int | None:
    indexes = [index for index in (pattern.find(ch) for ch in "*?[{") if index >= 0]
    return min(indexes) if indexes else None


def _expand_braces(pattern: str) -> tuple[str, ...]:
    start = pattern.find("{")
    end = pattern.find("}", start + 1)
    if start == -1 or end == -1 or end <= start:
        return (pattern,)
    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    options = tuple(part for part in pattern[start + 1 : end].split(",") if part)
    if not options:
        return (pattern,)
    return tuple(
        expanded
        for option in options
        for expanded in _expand_braces(prefix + option + suffix)
    )


def _raise_if_aborted(abort_event: asyncio.Event | None) -> None:
    if abort_event is not None and abort_event.is_set():
        raise asyncio.CancelledError()


__all__ = [
    "COMMON_TYPE_GLOBS",
    "DEFAULT_GLOB_LIMIT",
    "DEFAULT_GREP_HEAD_LIMIT",
    "DEFAULT_GREP_TIMEOUT_S",
    "DEFAULT_MAX_GREP_LINE_LENGTH",
    "GREP_OUTPUT_MODES",
    "VCS_DIRECTORIES_TO_EXCLUDE",
    "GlobSearchRequest",
    "GlobSearchResult",
    "GrepContentLine",
    "GrepFileMatch",
    "GrepOutputMode",
    "GrepSearchRequest",
    "GrepSearchResult",
    "SearchBackend",
    "SearchBackendError",
    "SearchPath",
    "SearchTimeoutError",
    "StdlibSearchBackend",
    "create_default_search_backend",
    "expand_glob_patterns",
    "extract_glob_base_directory",
    "supported_type_names",
]
