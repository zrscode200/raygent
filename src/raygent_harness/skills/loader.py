"""Filesystem skill loader.


This module is intentionally fail-soft: malformed or unreadable skills are
skipped rather than aborting the session.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from raygent_harness.skills.activation import parse_skill_paths, split_path_in_frontmatter
from raygent_harness.skills.models import (
    HookSettings,
    LoadedSkill,
    SkillContext,
    SkillDefinition,
    SkillLoadedFrom,
    SkillShell,
    SkillSource,
)

FRONTMATTER_PATTERN = re.compile(r"^---[ \t]*\n([\s\S]*?)\n---[ \t]*(?:\n|$)")
MAX_EXTRACTED_DESCRIPTION_CHARS = 100
SKILL_FILENAME = "SKILL.md"
HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Notification",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "StopFailure",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "PermissionRequest",
    "PermissionDenied",
    "Setup",
    "TeammateIdle",
    "TaskCreated",
    "TaskCompleted",
    "Elicitation",
    "ElicitationResult",
    "ConfigChange",
    "WorktreeCreate",
    "WorktreeRemove",
    "InstructionsLoaded",
    "CwdChanged",
    "FileChanged",
)


@dataclass(frozen=True)
class ParsedMarkdown:
    frontmatter: Mapping[str, Any]
    content: str


@dataclass(frozen=True)
class ParsedSkillFields:
    display_name: str | None
    description: str
    has_user_specified_description: bool
    allowed_tools: tuple[str, ...]
    argument_hint: str | None
    argument_names: tuple[str, ...]
    when_to_use: str | None
    version: str | None
    model: str | None
    disable_model_invocation: bool
    user_invocable: bool
    hooks: HookSettings | None
    context: SkillContext | None
    agent: str | None
    effort: str | None
    shell: SkillShell | None


def parse_markdown_frontmatter(markdown: str) -> ParsedMarkdown:
    """Parse YAML-ish frontmatter using the subset skill metadata needs."""

    match = FRONTMATTER_PATTERN.match(markdown)
    if match is None:
        return ParsedMarkdown(frontmatter=MappingProxyType({}), content=markdown)

    frontmatter_text = match.group(1)
    content = markdown[match.end() :]
    try:
        parsed = _parse_yaml_subset(frontmatter_text)
    except ValueError:
        parsed = {}
    return ParsedMarkdown(frontmatter=MappingProxyType(parsed), content=content)


def parse_skill_frontmatter_fields(
    frontmatter: Mapping[str, Any],
    markdown_content: str,
    resolved_name: str,
    *,
    description_fallback_label: str = "Skill",
) -> ParsedSkillFields:
    """Parse fields shared by filesystem and programmatic skill sources."""

    description_value = _coerce_description_to_string(
        frontmatter.get("description")
    )
    description = description_value or extract_description_from_markdown(
        markdown_content,
        description_fallback_label,
    )

    model_value = _as_non_empty_string(frontmatter.get("model"))
    model = None if model_value == "inherit" else model_value

    context = _parse_context(frontmatter.get("context"))
    shell = _parse_shell(frontmatter.get("shell"))

    return ParsedSkillFields(
        display_name=_as_non_empty_string(frontmatter.get("name")),
        description=description,
        has_user_specified_description=description_value is not None,
        allowed_tools=parse_allowed_tools(frontmatter.get("allowed-tools")),
        argument_hint=_as_non_empty_string(frontmatter.get("argument-hint")),
        argument_names=parse_argument_names(frontmatter.get("arguments")),
        when_to_use=_as_non_empty_string(frontmatter.get("when_to_use")),
        version=_as_non_empty_string(frontmatter.get("version")),
        model=model,
        disable_model_invocation=parse_boolean_frontmatter(
            frontmatter.get("disable-model-invocation")
        ),
        user_invocable=True
        if "user-invocable" not in frontmatter
        else parse_boolean_frontmatter(frontmatter.get("user-invocable")),
        hooks=_parse_hooks(frontmatter.get("hooks")),
        context=context,
        agent=_as_non_empty_string(frontmatter.get("agent")),
        effort=_as_non_empty_string(frontmatter.get("effort")),
        shell=shell,
    )


def create_skill_definition(
    *,
    skill_name: str,
    markdown_content: str,
    source: SkillSource,
    loaded_from: SkillLoadedFrom,
    fields: ParsedSkillFields,
    skill_root: Path | None,
    file_path: Path | None,
    paths: tuple[str, ...] = (),
) -> SkillDefinition:
    return SkillDefinition(
        name=skill_name,
        description=fields.description,
        markdown_content=markdown_content,
        source=source,
        loaded_from=loaded_from,
        content_length=len(markdown_content),
        display_name=fields.display_name,
        has_user_specified_description=fields.has_user_specified_description,
        allowed_tools=fields.allowed_tools,
        argument_hint=fields.argument_hint,
        argument_names=fields.argument_names,
        when_to_use=fields.when_to_use,
        version=fields.version,
        model=fields.model,
        disable_model_invocation=fields.disable_model_invocation,
        user_invocable=fields.user_invocable,
        hooks=fields.hooks,
        context=fields.context,
        agent=fields.agent,
        effort=fields.effort,
        shell=fields.shell,
        paths=paths,
        skill_root=skill_root,
        file_path=file_path,
    )


def load_skill_file(
    skill_file_path: str | Path,
    *,
    skill_name: str | None = None,
    source: SkillSource = "projectSettings",
    loaded_from: SkillLoadedFrom = "skills",
    skill_root: str | Path | None = None,
) -> LoadedSkill | None:
    """Load one `SKILL.md` file."""

    path = Path(skill_file_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    parsed = parse_markdown_frontmatter(raw)
    resolved_name = skill_name or path.parent.name
    root = Path(skill_root) if skill_root is not None else path.parent
    fields = parse_skill_frontmatter_fields(
        parsed.frontmatter,
        parsed.content,
        resolved_name,
    )
    skill = create_skill_definition(
        skill_name=resolved_name,
        markdown_content=parsed.content,
        source=source,
        loaded_from=loaded_from,
        fields=fields,
        skill_root=root,
        file_path=path,
        paths=parse_skill_paths(parsed.frontmatter.get("paths")),
    )
    return LoadedSkill(skill=skill, file_path=path, file_identity=get_file_identity(path))


def load_skills_from_dir(
    base_path: str | Path,
    *,
    source: SkillSource = "projectSettings",
) -> tuple[LoadedSkill, ...]:
    """Load directory-format skills: `<base>/<skill-name>/SKILL.md` only."""

    base = Path(base_path)
    try:
        entries = sorted(base.iterdir(), key=lambda item: item.name)
    except OSError:
        return ()

    loaded: list[LoadedSkill] = []
    for entry in entries:
        if not (entry.is_dir() or entry.is_symlink()):
            continue
        skill = load_skill_file(
            entry / SKILL_FILENAME,
            skill_name=entry.name,
            source=source,
            loaded_from="skills",
            skill_root=entry,
        )
        if skill is not None:
            loaded.append(skill)
    return tuple(loaded)


def deduplicate_loaded_skills(skills: Sequence[LoadedSkill]) -> tuple[LoadedSkill, ...]:
    """Deduplicate by canonical file identity; first loaded skill wins."""

    seen: set[str] = set()
    deduplicated: list[LoadedSkill] = []
    for loaded in skills:
        file_id = loaded.file_identity
        if file_id is not None:
            if file_id in seen:
                continue
            seen.add(file_id)
        deduplicated.append(loaded)
    return tuple(deduplicated)


def load_skill_directories(
    directories: Sequence[str | Path],
    *,
    source: SkillSource = "projectSettings",
) -> tuple[LoadedSkill, ...]:
    """Load and file-deduplicate multiple skill directories in order."""

    all_skills: list[LoadedSkill] = []
    for directory in directories:
        all_skills.extend(load_skills_from_dir(directory, source=source))
    return deduplicate_loaded_skills(tuple(all_skills))


def merge_skills_prefer_deepest(
    directories_deepest_first: Sequence[str | Path],
    *,
    source: SkillSource = "projectSettings",
) -> tuple[SkillDefinition, ...]:
    """Merge dynamic skill dirs so deeper paths override same-name skills."""

    loaded_by_dir = [
        load_skills_from_dir(directory, source=source)
        for directory in directories_deepest_first
    ]
    by_name: dict[str, SkillDefinition] = {}

    # Reference loads dirs deepest-first, then applies shallower first so deeper
    # entries replace same-name skills.
    for loaded_group in reversed(loaded_by_dir):
        for loaded in deduplicate_loaded_skills(loaded_group):
            by_name[loaded.skill.name] = loaded.skill
    return tuple(by_name.values())


def get_file_identity(path: str | Path) -> str | None:
    """Canonical identity used for symlink/overlap deduplication."""

    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return None


def extract_description_from_markdown(
    content: str,
    default_description: str = "Skill",
) -> str:
    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if trimmed.startswith("#"):
            trimmed = trimmed.lstrip("#").strip() or trimmed
        if len(trimmed) > MAX_EXTRACTED_DESCRIPTION_CHARS:
            return trimmed[: MAX_EXTRACTED_DESCRIPTION_CHARS - 3] + "..."
        return trimmed
    return default_description


def parse_allowed_tools(value: object) -> tuple[str, ...]:
    tools = _parse_string_list(value)
    if "*" in tools:
        return ("*",)
    return tools


def parse_argument_names(value: object) -> tuple[str, ...]:
    return _parse_string_list(value)


def parse_boolean_frontmatter(value: object) -> bool:
    return value is True or value == "true"


def _parse_string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = split_path_in_frontmatter(value)
        return tuple(value for value in values if value)
    if isinstance(value, list | tuple):
        parsed: list[str] = []
        for item in cast(Sequence[object], value):
            if isinstance(item, str) and item:
                parsed.append(item)
        return tuple(parsed)
    return ()


def _as_non_empty_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_description_to_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool | int | float):
        return str(value)
    return None


def _parse_hooks(value: object) -> HookSettings | None:
    if not isinstance(value, Mapping):
        return None
    raw_mapping = cast(Mapping[object, object], value)
    normalized: dict[str, object] = {}
    for event_name, matchers in raw_mapping.items():
        if not isinstance(event_name, str) or event_name not in HOOK_EVENTS:
            return None
        parsed_matchers = _parse_hook_matchers(matchers)
        if parsed_matchers is None:
            return None
        normalized[event_name] = parsed_matchers
    return cast(HookSettings, MappingProxyType(normalized))


def _parse_context(value: object) -> SkillContext | None:
    if value == "fork":
        return "fork"
    return None


def _parse_hook_matchers(value: object) -> tuple[Mapping[str, object], ...] | None:
    if not isinstance(value, list | tuple):
        return None
    matchers: list[Mapping[str, object]] = []
    for item in cast(Sequence[object], value):
        if not isinstance(item, Mapping):
            return None
        matcher = cast(Mapping[object, object], item)
        matcher_name = matcher.get("matcher")
        if matcher_name is not None and not isinstance(matcher_name, str):
            return None
        hooks = _parse_hook_commands(matcher.get("hooks"))
        if hooks is None:
            return None
        normalized: dict[str, object] = {"hooks": hooks}
        if matcher_name is not None:
            normalized["matcher"] = matcher_name
        matchers.append(MappingProxyType(normalized))
    return tuple(matchers)


def _parse_hook_commands(value: object) -> tuple[Mapping[str, object], ...] | None:
    if not isinstance(value, list | tuple):
        return None
    commands: list[Mapping[str, object]] = []
    for item in cast(Sequence[object], value):
        if not isinstance(item, Mapping):
            return None
        command = _parse_hook_command(cast(Mapping[object, object], item))
        if command is None:
            return None
        commands.append(command)
    return tuple(commands)


def _parse_hook_command(value: Mapping[object, object]) -> Mapping[str, object] | None:
    hook_type = value.get("type")
    if hook_type == "command":
        required_field = "command"
    elif hook_type in {"prompt", "agent"}:
        required_field = "prompt"
    elif hook_type == "http":
        required_field = "url"
    else:
        return None

    required_value = value.get(required_field)
    if not isinstance(required_value, str):
        return None

    normalized: dict[str, object] = {"type": hook_type, required_field: required_value}
    for key, item in value.items():
        if key in {"type", required_field}:
            continue
        if not isinstance(key, str) or not _is_valid_hook_optional_field(key, item):
            return None
        normalized[key] = item
    return MappingProxyType(normalized)


def _is_valid_hook_optional_field(key: str, value: object) -> bool:
    if key in {"if", "shell", "statusMessage", "model"}:
        return value is None or isinstance(value, str)
    if key == "timeout":
        return isinstance(value, int | float) and value > 0
    if key in {"once", "async", "asyncRewake"}:
        return isinstance(value, bool)
    if key == "headers":
        return isinstance(value, Mapping) and all(
            isinstance(header_key, str) and isinstance(header_value, str)
            for header_key, header_value in cast(Mapping[object, object], value).items()
        )
    if key == "allowedEnvVars":
        return isinstance(value, list | tuple) and all(
            isinstance(item, str) for item in cast(Sequence[object], value)
        )
    return False


def _parse_shell(value: object) -> SkillShell | None:
    if value == "bash":
        return "bash"
    if value == "powershell":
        return "powershell"
    return None


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    lines = _normalize_yaml_lines(text)
    parsed, index = _parse_mapping(lines, 0, 0)
    if index < len(lines):
        return parsed
    return parsed


def _normalize_yaml_lines(text: str) -> list[tuple[int, str]]:
    normalized: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        normalized.append((indent, raw.strip()))
    return normalized


def _parse_mapping(
    lines: Sequence[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            break
        if text.startswith("- "):
            break

        key, raw_value = _split_yaml_key_value(text)
        index += 1
        if raw_value in {"|", ">"}:
            value, index = _parse_block_scalar(
                lines,
                index,
                line_indent,
                folded=raw_value == ">",
            )
        elif raw_value == "":
            if index < len(lines) and lines[index][0] > line_indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = None
        else:
            value = _parse_scalar(raw_value)
        result[key] = value
    return result, index


def _parse_yaml_block(
    lines: Sequence[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index < len(lines) and lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_list(
    lines: Sequence[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent != indent or not text.startswith("- "):
            break
        item_text = text[2:].strip()
        index += 1
        if item_text == "":
            if index < len(lines) and lines[index][0] > line_indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = None
            result.append(value)
            continue

        if _is_yaml_key_value(item_text):
            key, raw_value = _split_yaml_key_value(item_text)
            item_map: dict[str, Any] = {key: _parse_scalar(raw_value)}
            if index < len(lines) and lines[index][0] > line_indent:
                child, index = _parse_mapping(lines, index, lines[index][0])
                item_map.update(child)
            result.append(item_map)
            continue

        result.append(_parse_scalar(item_text))
    return result, index


def _parse_block_scalar(
    lines: Sequence[tuple[int, str]],
    index: int,
    parent_indent: int,
    *,
    folded: bool,
) -> tuple[str, int]:
    collected: list[str] = []
    while index < len(lines) and lines[index][0] > parent_indent:
        collected.append(lines[index][1])
        index += 1
    return ((" ".join(collected) if folded else "\n".join(collected)), index)


def _split_yaml_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"invalid frontmatter line: {text}")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"invalid frontmatter key: {text}")
    return key, value.strip()


def _is_yaml_key_value(text: str) -> bool:
    return ":" in text and not text.startswith(("'", '"'))


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in split_path_in_frontmatter(inner)]
    return value


__all__ = [
    "FRONTMATTER_PATTERN",
    "MAX_EXTRACTED_DESCRIPTION_CHARS",
    "SKILL_FILENAME",
    "ParsedMarkdown",
    "ParsedSkillFields",
    "create_skill_definition",
    "deduplicate_loaded_skills",
    "extract_description_from_markdown",
    "get_file_identity",
    "load_skill_directories",
    "load_skill_file",
    "load_skills_from_dir",
    "merge_skills_prefer_deepest",
    "parse_allowed_tools",
    "parse_argument_names",
    "parse_boolean_frontmatter",
    "parse_markdown_frontmatter",
    "parse_skill_frontmatter_fields",
]
