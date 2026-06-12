"""Project instruction context provider.

This implements turn-entry instruction discovery without teaching `core` about
particular filenames. It also provides the read-adjacent provider used by the
query loop's transient post-tool context seam after successful text-file reads.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import ClassVar, Literal

from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import (
    ContextAgentScope,
    ContextFragment,
    ContextKind,
    context_agent_scope_includes,
)
from raygent_harness.core.tool import ToolUseContext

DiscoveryMode = Literal["layered_ancestors", "nearest_first_match"]
InstructionKind = Literal["user", "project", "local", "additional", "rule"]

DEFAULT_ALLOWED_INCLUDE_EXTENSIONS = (
    ".txt",
    ".md",
    ".markdown",
    ".mdx",
    ".rst",
    ".text",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".css",
    ".scss",
    ".html",
    ".xml",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".csv",
)
_FRONTMATTER_PATTERN = re.compile(r"^---[ \t]*\n([\s\S]*?)\n---[ \t]*(?:\n|$)")


@dataclass(frozen=True, slots=True)
class _InstructionRead:
    """Instruction file content plus rule frontmatter classification."""

    content: str
    truncated: bool
    path_patterns: tuple[str, ...] = ()

    @property
    def has_scoped_paths(self) -> bool:
        return bool(self.path_patterns)


@dataclass(frozen=True, slots=True)
class ProjectInstructionConfig:
    """Policy inputs for project-instruction discovery."""

    cwd: str | Path | None = None
    workspace_root: str | Path | None = None
    user_instruction_paths: tuple[str | Path, ...] = ()
    project_filenames: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
    project_rule_dirs: tuple[str | Path, ...] = (".claude/rules",)
    local_filenames: tuple[str, ...] = ("AGENTS.local.md", "CLAUDE.local.md")
    additional_dirs: tuple[str | Path, ...] = ()
    discovery_mode: DiscoveryMode = "layered_ancestors"
    max_file_chars: int = 40000
    max_total_chars: int = 120000
    allow_includes: bool = False
    max_include_depth: int = 5
    allow_external_includes: bool = False
    allowed_include_extensions: tuple[str, ...] = DEFAULT_ALLOWED_INCLUDE_EXTENSIONS
    priority: int = 100
    agent_scope: ContextAgentScope = "all"
    fragment_id_prefix: str = "project-instructions"


@dataclass(frozen=True, slots=True)
class ProjectInstructionFile:
    """A discovered instruction file after safe read and cap enforcement."""

    path: Path
    kind: InstructionKind
    content: str
    truncated: bool = False
    parent: Path | None = None


@dataclass(frozen=True, slots=True)
class ConditionalInstructionRule:
    """A scoped rule file that should attach only for matching target paths."""

    path: Path
    base_dir: Path
    patterns: tuple[str, ...]
    content: str
    truncated: bool = False
    parent: Path | None = None


@dataclass(frozen=True, slots=True)
class ProjectInstructionsContextProvider:
    """Discover instruction files and render them as user-context fragments."""

    context_kind: ClassVar[ContextKind] = "project_instructions"
    config: ProjectInstructionConfig = ProjectInstructionConfig()

    async def __call__(
        self,
        _config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> tuple[ContextFragment, ...]:
        if not context_agent_scope_includes(
            self.config.agent_scope,
            agent_id=ctx.agent_id,
        ):
            return ()

        cwd = _resolve_path(self.config.cwd or ctx.cwd)
        files = discover_project_instruction_files(self.config, cwd=cwd)
        fragments: list[ContextFragment] = []
        for index, file in enumerate(files):
            fragments.append(
                ContextFragment(
                    id=f"{self.config.fragment_id_prefix}:{index}",
                    content=_render_instruction_file(file),
                    channel="user_context",
                    source=str(file.path),
                    priority=self.config.priority + index,
                    agent_scope=self.config.agent_scope,
                    render_mode="instructions",
                    kind=self.context_kind,
                )
            )
        return tuple(fragments)


@dataclass(frozen=True, slots=True)
class ReadAdjacentProjectInstructionsContextProvider:
    """Attach project instructions after successful text-file reads."""

    context_kind: ClassVar[ContextKind] = "project_instructions"
    config: ProjectInstructionConfig = ProjectInstructionConfig()
    fragment_id_prefix: str = "read-adjacent-project-instructions"

    async def __call__(
        self,
        _config: QueryConfig,
        ctx: ToolUseContext,
        read_paths: Sequence[str],
        already_attached_sources: Sequence[str],
        /,
    ) -> tuple[ContextFragment, ...]:
        if not read_paths:
            return ()
        if not context_agent_scope_includes(
            self.config.agent_scope,
            agent_id=ctx.agent_id,
        ):
            return ()

        resolved_cwd = _resolve_path(self.config.cwd or ctx.cwd)
        seen = _initial_read_adjacent_seen_paths(
            self.config,
            cwd=resolved_cwd,
            already_attached_sources=already_attached_sources,
        )
        remaining = self.config.max_total_chars
        fragments: list[ContextFragment] = []

        for read_path in read_paths:
            if remaining <= 0:
                break
            files = resolve_project_instructions_for_target_path(
                read_path,
                self.config,
                cwd=resolved_cwd,
                already_loaded_paths=seen,
            )
            for file in files:
                if remaining <= 0:
                    break
                if file.path in seen:
                    continue
                seen.add(file.path)
                bounded = file
                if len(file.content) > remaining:
                    bounded = replace(
                        file,
                        content=file.content[:remaining],
                        truncated=True,
                    )
                remaining -= len(bounded.content)
                index = len(fragments)
                fragments.append(
                    ContextFragment(
                        id=f"{self.fragment_id_prefix}:{index}",
                        content=_render_instruction_file(bounded),
                        channel="user_context",
                        source=str(bounded.path),
                        priority=self.config.priority + index,
                        agent_scope=self.config.agent_scope,
                        render_mode="instructions",
                        kind=self.context_kind,
                    )
                )

        return tuple(fragments)


def discover_project_instruction_files(
    config: ProjectInstructionConfig,
    *,
    cwd: str | Path | None = None,
) -> tuple[ProjectInstructionFile, ...]:
    """Discover and read bounded project instruction files.

    Missing/unreadable files fail soft. Returned files are de-duplicated by
    normalized absolute path.
    """

    resolved_cwd = _resolve_path(cwd or config.cwd or ".")
    workspace_root = _resolve_workspace_root(resolved_cwd, config.workspace_root)
    seen: set[Path] = set()
    remaining = config.max_total_chars
    files: list[ProjectInstructionFile] = []

    def add_file(
        path: Path,
        kind: InstructionKind,
        *,
        parent: Path | None = None,
        depth: int = 0,
        require_non_conditional_rule: bool = False,
    ) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        if depth >= config.max_include_depth:
            return
        normalized = _normalize_existing_or_candidate(path)
        if normalized in seen:
            return
        seen.add(normalized)
        read = _read_instruction_file(
            normalized,
            max_file_chars=config.max_file_chars,
            remaining_total_chars=remaining,
        )
        if read is None:
            return
        if not require_non_conditional_rule or not read.has_scoped_paths:
            remaining -= len(read.content)
            files.append(
                ProjectInstructionFile(
                    path=normalized,
                    kind=kind,
                    content=read.content,
                    truncated=read.truncated,
                    parent=parent,
                )
            )

        if not config.allow_includes:
            return
        for include_path in _extract_include_paths(
            read.content,
            base_path=normalized,
            workspace_root=workspace_root,
            allow_external=config.allow_external_includes,
            allowed_extensions=config.allowed_include_extensions,
        ):
            add_file(
                include_path,
                kind,
                parent=normalized,
                depth=depth + 1,
                require_non_conditional_rule=require_non_conditional_rule,
            )

    def add_rule_files(directory: Path) -> None:
        for rule_file in _iter_rule_files(directory):
            add_file(rule_file, "rule", require_non_conditional_rule=True)

    for path in config.user_instruction_paths:
        add_file(_resolve_path(path), "user")

    if config.discovery_mode == "layered_ancestors":
        for directory in _ancestor_dirs(resolved_cwd, workspace_root):
            for filename in config.project_filenames:
                add_file(directory / filename, "project")
            for rule_dir in config.project_rule_dirs:
                add_rule_files(_resolve_rule_dir(directory, rule_dir))
            for filename in config.local_filenames:
                add_file(directory / filename, "local")
    else:
        for path in _nearest_project_matches(
            resolved_cwd,
            workspace_root,
            config.project_filenames,
        ):
            add_file(path, "project")
        for path in _nearest_project_matches(
            resolved_cwd,
            workspace_root,
            config.local_filenames,
        ):
            add_file(path, "local")
        for directory in _ancestor_dirs(resolved_cwd, workspace_root):
            for rule_dir in config.project_rule_dirs:
                add_rule_files(_resolve_rule_dir(directory, rule_dir))

    for directory in config.additional_dirs:
        additional_dir = _resolve_path(directory)
        for filename in config.project_filenames:
            add_file(additional_dir / filename, "additional")
        for rule_dir in config.project_rule_dirs:
            add_rule_files(_resolve_rule_dir(additional_dir, rule_dir))

    return tuple(files)


def discover_conditional_instruction_rules(
    config: ProjectInstructionConfig,
    *,
    cwd: str | Path | None = None,
) -> tuple[ConditionalInstructionRule, ...]:
    """Discover scoped project rule files without matching a target path.

    This is Wave 1's parser/discovery seam. Read-adjacent attachment uses
    `resolve_project_instructions_for_target_path(...)` to match these rules
    against files read later in a turn.
    """

    resolved_cwd = _resolve_path(cwd or config.cwd or ".")
    workspace_root = _resolve_workspace_root(resolved_cwd, config.workspace_root)
    seen: set[Path] = set()
    remaining = config.max_total_chars
    rules: list[ConditionalInstructionRule] = []

    def add_rule(
        path: Path,
        base_dir: Path,
        *,
        parent: Path | None = None,
        depth: int = 0,
    ) -> None:
        nonlocal remaining
        if remaining <= 0 or depth >= config.max_include_depth:
            return
        normalized = _normalize_existing_or_candidate(path)
        if normalized in seen:
            return
        seen.add(normalized)
        read = _read_instruction_file(
            normalized,
            max_file_chars=config.max_file_chars,
            remaining_total_chars=remaining,
        )
        if read is None:
            return

        if read.path_patterns:
            remaining -= len(read.content)
            rules.append(
                ConditionalInstructionRule(
                    path=normalized,
                    base_dir=base_dir,
                    patterns=read.path_patterns,
                    content=read.content,
                    truncated=read.truncated,
                    parent=parent,
                )
            )

        if not config.allow_includes:
            return
        for include_path in _extract_include_paths(
            read.content,
            base_path=normalized,
            workspace_root=workspace_root,
            allow_external=config.allow_external_includes,
            allowed_extensions=config.allowed_include_extensions,
        ):
            add_rule(
                include_path,
                base_dir,
                parent=normalized,
                depth=depth + 1,
            )

    def add_rule_dir(base_dir: Path, rule_dir: str | Path) -> None:
        for rule_file in _iter_rule_files(_resolve_rule_dir(base_dir, rule_dir)):
            add_rule(rule_file, base_dir)

    for directory in _ancestor_dirs(resolved_cwd, workspace_root):
        for rule_dir in config.project_rule_dirs:
            add_rule_dir(directory, rule_dir)

    for directory in config.additional_dirs:
        additional_dir = _resolve_path(directory)
        for rule_dir in config.project_rule_dirs:
            add_rule_dir(additional_dir, rule_dir)

    return tuple(rules)


def resolve_project_instructions_for_target_path(
    target_path: str | Path,
    config: ProjectInstructionConfig,
    *,
    cwd: str | Path | None = None,
    already_loaded_paths: Iterable[str | Path] = (),
) -> tuple[ProjectInstructionFile, ...]:
    """Resolve nearby and conditional instructions for a target file path.

    The output is not injected by this function. Wave 2 will wire it into a
    post-tool transient-context seam after successful file reads.
    """

    resolved_cwd = _resolve_path(cwd or config.cwd or ".")
    workspace_root = _resolve_workspace_root(resolved_cwd, config.workspace_root)
    resolved_target = _resolve_target_path(target_path, cwd=resolved_cwd)
    if resolved_target is None or not _path_is_within(resolved_target, workspace_root):
        return ()

    seen: set[Path] = set()
    for path in already_loaded_paths:
        loaded_path = _resolve_target_path(path, cwd=resolved_cwd)
        if loaded_path is not None:
            seen.add(loaded_path)
    remaining = config.max_total_chars
    files: list[ProjectInstructionFile] = []

    def add_file(
        path: Path,
        kind: InstructionKind,
        *,
        parent: Path | None = None,
        depth: int = 0,
        rule_base_dir: Path | None = None,
        include_unconditional_rule: bool = True,
        include_matching_rule: bool = False,
    ) -> None:
        nonlocal remaining
        if remaining <= 0 or depth >= config.max_include_depth:
            return
        normalized = _normalize_existing_or_candidate(path)
        if normalized in seen:
            return
        read = _read_instruction_file(
            normalized,
            max_file_chars=config.max_file_chars,
            remaining_total_chars=remaining,
        )
        if read is None:
            return

        should_add = True
        if kind == "rule":
            if read.path_patterns:
                base_dir = rule_base_dir or normalized.parent
                should_add = include_matching_rule and instruction_rule_matches_path(
                    ConditionalInstructionRule(
                        path=normalized,
                        base_dir=base_dir,
                        patterns=read.path_patterns,
                        content=read.content,
                        truncated=read.truncated,
                        parent=parent,
                    ),
                    resolved_target,
                )
            else:
                should_add = include_unconditional_rule

        if should_add:
            seen.add(normalized)
            remaining -= len(read.content)
            files.append(
                ProjectInstructionFile(
                    path=normalized,
                    kind=kind,
                    content=read.content,
                    truncated=read.truncated,
                    parent=parent,
                )
            )

        if not config.allow_includes:
            return
        for include_path in _extract_include_paths(
            read.content,
            base_path=normalized,
            workspace_root=workspace_root,
            allow_external=config.allow_external_includes,
            allowed_extensions=config.allowed_include_extensions,
        ):
            add_file(
                include_path,
                kind,
                parent=normalized,
                depth=depth + 1,
                rule_base_dir=rule_base_dir,
                include_unconditional_rule=include_unconditional_rule,
                include_matching_rule=include_matching_rule,
            )

    def add_rule_files(
        base_dir: Path,
        *,
        include_unconditional_rule: bool,
        include_matching_rule: bool,
    ) -> None:
        for rule_dir in config.project_rule_dirs:
            for rule_file in _iter_rule_files(_resolve_rule_dir(base_dir, rule_dir)):
                add_file(
                    rule_file,
                    "rule",
                    rule_base_dir=base_dir,
                    include_unconditional_rule=include_unconditional_rule,
                    include_matching_rule=include_matching_rule,
                )

    # Directories below the current CWD are loaded only when a file under them is
    # read. Preserve the reference intra-directory order: configured project
    # files, local files, unconditional rules, then matching conditional rules.
    for directory in _nested_dirs_from_cwd_to_target(
        resolved_cwd,
        resolved_target,
    ):
        for filename in config.project_filenames:
            add_file(directory / filename, "project")
        for filename in config.local_filenames:
            add_file(directory / filename, "local")
        add_rule_files(
            directory,
            include_unconditional_rule=True,
            include_matching_rule=False,
        )
        add_rule_files(
            directory,
            include_unconditional_rule=False,
            include_matching_rule=True,
        )

    # CWD-level instructions are already loaded at turn entry, but scoped rules
    # from those directories still need target-path matching after nested
    # target-directory instructions.
    for directory in _ancestor_dirs(resolved_cwd, workspace_root):
        add_rule_files(
            directory,
            include_unconditional_rule=False,
            include_matching_rule=True,
        )

    for directory in config.additional_dirs:
        additional_dir = _resolve_path(directory)
        add_rule_files(
            additional_dir,
            include_unconditional_rule=False,
            include_matching_rule=True,
        )

    return tuple(files)


def instruction_rule_matches_path(
    rule: ConditionalInstructionRule,
    target_path: str | Path,
) -> bool:
    """Return whether a scoped instruction rule applies to `target_path`."""

    relative_path = _relative_to_base(Path(target_path), rule.base_dir)
    if relative_path is None:
        return False
    return any(
        _instruction_pattern_matches(relative_path, pattern)
        for pattern in rule.patterns
    )


def _render_instruction_file(file: ProjectInstructionFile) -> str:
    description = {
        "user": " (user's private global instructions)",
        "project": " (project instructions, checked into the codebase)",
        "local": " (user's private project instructions, not checked in)",
        "additional": " (additional project instructions)",
        "rule": " (project instructions, checked into the codebase)",
    }[file.kind]
    content = file.content.strip()
    if file.truncated:
        content = f"{content}\n\n... (truncated)"
    return f"Contents of {file.path}{description}:\n\n{content}"


def _initial_read_adjacent_seen_paths(
    config: ProjectInstructionConfig,
    *,
    cwd: Path,
    already_attached_sources: Sequence[str],
) -> set[Path]:
    seen = {
        file.path
        for file in discover_project_instruction_files(
            config,
            cwd=cwd,
        )
    }
    for source in already_attached_sources:
        resolved = _resolve_target_path(source, cwd=cwd)
        if resolved is not None:
            seen.add(resolved)
    return seen


def _read_instruction_file(
    path: Path,
    *,
    max_file_chars: int,
    remaining_total_chars: int,
) -> _InstructionRead | None:
    if max_file_chars <= 0 or remaining_total_chars <= 0:
        return None
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    content = _strip_frontmatter(raw).strip()
    if not content:
        return None
    limit = min(max_file_chars, remaining_total_chars)
    path_patterns = _frontmatter_path_patterns(raw)
    if len(content) > limit:
        return _InstructionRead(content[:limit], True, path_patterns)
    return _InstructionRead(content, False, path_patterns)


def _extract_include_paths(
    content: str,
    *,
    base_path: Path,
    workspace_root: Path,
    allow_external: bool,
    allowed_extensions: tuple[str, ...],
) -> tuple[Path, ...]:
    includes: list[Path] = []
    seen: set[Path] = set()
    in_fence = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        raw_include = _parse_include_directive(stripped)
        if raw_include is None:
            continue
        resolved = _resolve_include_path(raw_include, base_path=base_path)
        if resolved is None:
            continue
        if not _include_extension_allowed(resolved, allowed_extensions):
            continue
        if not allow_external and not _path_is_within(resolved, workspace_root):
            continue
        normalized = _normalize_existing_or_candidate(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        includes.append(normalized)
    return tuple(includes)


def _parse_include_directive(stripped_line: str) -> str | None:
    if not stripped_line.startswith("@") or stripped_line == "@":
        return None

    value = stripped_line[1:]
    chars: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value) and value[index + 1] == " ":
            chars.append(" ")
            index += 2
            continue
        if char.isspace():
            if value[index:].strip():
                return None
            break
        chars.append(char)
        index += 1

    include_path = "".join(chars).strip()
    if not include_path:
        return None
    hash_index = include_path.find("#")
    if hash_index != -1:
        include_path = include_path[:hash_index]
    if not include_path or "://" in include_path:
        return None
    if include_path == "/" or include_path.startswith("@"):
        return None
    if include_path.startswith(("./", "~/", "/")):
        return include_path
    if re.match(r"^[a-zA-Z0-9._-]", include_path):
        return include_path
    return None


def _resolve_include_path(include_path: str, *, base_path: Path) -> Path | None:
    try:
        candidate = Path(include_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (base_path.parent / candidate).resolve()
    except OSError:
        return None


def _include_extension_allowed(path: Path, allowed_extensions: tuple[str, ...]) -> bool:
    suffix = path.suffix.lower()
    if not suffix:
        return True
    return suffix in {extension.lower() for extension in allowed_extensions}


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_rule_dir(base_dir: Path, rule_dir: str | Path) -> Path:
    candidate = Path(rule_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _iter_rule_files(rule_dir: Path) -> tuple[Path, ...]:
    try:
        root = rule_dir.resolve()
        if not root.is_dir():
            return ()
    except OSError:
        return ()

    files: list[Path] = []
    visited: set[Path] = set()

    def visit(directory: Path) -> None:
        try:
            resolved_dir = directory.resolve()
            if resolved_dir in visited or not resolved_dir.is_dir():
                return
            visited.add(resolved_dir)
            entries = sorted(
                resolved_dir.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower(), path.name),
            )
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_dir():
                    visit(entry)
                elif entry.is_file() and entry.suffix.lower() == ".md":
                    files.append(entry.resolve())
            except OSError:
                continue

    visit(root)
    return tuple(files)


def _frontmatter_path_patterns(raw: str) -> tuple[str, ...]:
    match = _FRONTMATTER_PATTERN.match(raw)
    if match is None:
        return ()
    patterns: list[str] = []
    collecting_paths = False
    path_key_indent = 0
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        match_line = re.match(r"^(paths|path)\s*:\s*(.*)$", stripped)
        if match_line is not None:
            if indent != 0:
                continue
            collecting_paths = True
            path_key_indent = indent
            patterns.extend(_split_frontmatter_paths(match_line.group(2)))
            continue

        if not collecting_paths:
            continue
        if indent <= path_key_indent and re.match(r"^[A-Za-z_-][\w-]*\s*:", stripped):
            collecting_paths = False
            continue
        if stripped.startswith("-"):
            patterns.extend(_split_frontmatter_paths(stripped[1:].strip()))

    normalized = [
        _normalize_instruction_pattern(pattern)
        for pattern in patterns
        if pattern
    ]
    if not normalized or all(pattern == "**" for pattern in normalized):
        return ()
    return tuple(pattern for pattern in normalized if pattern)


def _split_frontmatter_paths(value: str) -> tuple[str, ...]:
    cleaned = value.strip()
    if not cleaned:
        return ()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    return tuple(
        expanded
        for part in _split_commas_outside_braces(cleaned)
        for expanded in _expand_braces(part.strip().strip("'\""))
        if expanded.strip().strip("'\"")
    )


def _split_commas_outside_braces(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    current: list[str] = []
    brace_depth = 0
    for char in value:
        if char == "{":
            brace_depth += 1
            current.append(char)
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
            current.append(char)
        elif char == "," and brace_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return tuple(parts)


def _expand_braces(pattern: str) -> tuple[str, ...]:
    start = pattern.find("{")
    if start == -1:
        return (pattern,)
    end = pattern.find("}", start + 1)
    if end == -1:
        return (pattern,)

    prefix = pattern[:start]
    alternatives = pattern[start + 1 : end]
    suffix = pattern[end + 1 :]
    expanded: list[str] = []
    for alternative in alternatives.split(","):
        expanded.extend(_expand_braces(f"{prefix}{alternative.strip()}{suffix}"))
    return tuple(expanded)


def _normalize_instruction_pattern(pattern: str) -> str:
    normalized = pattern.strip().strip("'\"").replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.endswith("/**"):
        normalized = normalized[:-3]
    return normalized


def _relative_to_base(path: Path, base_dir: Path) -> str | None:
    absolute_path = path if path.is_absolute() else base_dir / path
    try:
        relative = absolute_path.resolve().relative_to(base_dir.resolve())
    except (OSError, ValueError):
        try:
            relative = absolute_path.absolute().relative_to(base_dir.absolute())
        except ValueError:
            return None
    if not relative.parts:
        return None
    return relative.as_posix()


def _instruction_pattern_matches(relative_path: str, pattern: str) -> bool:
    normalized = _normalize_instruction_pattern(pattern)
    if not normalized or normalized == "**":
        return True
    if normalized == ".." or normalized.startswith("../"):
        return False
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")

    path_parts = tuple(part for part in relative_path.split("/") if part)
    if not path_parts:
        return False

    has_slash = "/" in normalized
    has_glob = _has_glob_syntax(normalized)
    if not has_slash and not has_glob:
        return any(part == normalized for part in path_parts)
    if not has_slash:
        return any(fnmatchcase(part, normalized) for part in path_parts)

    pattern_parts = tuple(part for part in normalized.split("/") if part)
    if not has_glob:
        normalized_path = "/".join(path_parts)
        return normalized_path == normalized or normalized_path.startswith(f"{normalized}/")
    return _glob_parts_match_prefix(path_parts, pattern_parts)


def _glob_parts_match_prefix(
    path_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
) -> bool:
    for size in range(1, len(path_parts) + 1):
        if _glob_parts_match_exact(path_parts[:size], pattern_parts):
            return True
    return False


def _glob_parts_match_exact(
    path_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
) -> bool:
    if not pattern_parts:
        return not path_parts
    head, *tail_list = pattern_parts
    tail = tuple(tail_list)
    if head == "**":
        return any(
            _glob_parts_match_exact(path_parts[index:], tail)
            for index in range(len(path_parts) + 1)
        )
    if not path_parts:
        return False
    if not fnmatchcase(path_parts[0], head):
        return False
    return _glob_parts_match_exact(path_parts[1:], tail)


def _has_glob_syntax(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _strip_frontmatter(raw: str) -> str:
    match = _FRONTMATTER_PATTERN.match(raw)
    if match is None:
        return raw
    return raw[match.end() :]


def _nearest_project_matches(
    cwd: Path,
    workspace_root: Path,
    filenames: tuple[str, ...],
) -> tuple[Path, ...]:
    """Return all matches for the first filename family, nearest first."""

    for filename in filenames:
        matches = [directory / filename for directory in _ancestor_dirs(cwd, workspace_root)]
        existing = tuple(path for path in reversed(matches) if path.is_file())
        if existing:
            return existing
    return ()


def _ancestor_dirs(cwd: Path, workspace_root: Path) -> tuple[Path, ...]:
    current = cwd if cwd.is_dir() else cwd.parent
    try:
        relative = current.relative_to(workspace_root)
    except ValueError:
        return (current,)

    parts = relative.parts
    dirs = [workspace_root]
    cursor = workspace_root
    for part in parts:
        cursor = cursor / part
        dirs.append(cursor)
    return tuple(dirs)


def _nested_dirs_from_cwd_to_target(cwd: Path, target_path: Path) -> tuple[Path, ...]:
    cwd_dir = cwd if cwd.is_dir() else cwd.parent
    target_dir = target_path if target_path.is_dir() else target_path.parent
    try:
        relative = target_dir.relative_to(cwd_dir)
    except ValueError:
        return ()

    dirs: list[Path] = []
    cursor = cwd_dir
    for part in relative.parts:
        cursor = cursor / part
        dirs.append(cursor)
    return tuple(dirs)


def _resolve_workspace_root(cwd: Path, workspace_root: str | Path | None) -> Path:
    if workspace_root is not None:
        return _resolve_path(workspace_root)
    return _find_git_root(cwd) or cwd


def _find_git_root(cwd: Path) -> Path | None:
    current = cwd if cwd.is_dir() else cwd.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_target_path(path: str | Path, *, cwd: Path) -> Path | None:
    try:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (cwd / candidate).resolve()
    except OSError:
        return None


def _normalize_existing_or_candidate(path: Path) -> Path:
    return path.expanduser().resolve()


__all__ = [
    "ConditionalInstructionRule",
    "DiscoveryMode",
    "InstructionKind",
    "ProjectInstructionConfig",
    "ProjectInstructionFile",
    "ProjectInstructionsContextProvider",
    "ReadAdjacentProjectInstructionsContextProvider",
    "discover_conditional_instruction_rules",
    "discover_project_instruction_files",
    "instruction_rule_matches_path",
    "resolve_project_instructions_for_target_path",
]
