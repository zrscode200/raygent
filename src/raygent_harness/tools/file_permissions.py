"""Headless file path permission helpers.


This module deliberately does not define concrete tools. It is the shared
foundation that Read/Write/Edit builders will call from their
`check_permissions` implementations.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from raygent_harness.core.permission_engine import permission_rule_value_from_string
from raygent_harness.core.permissions import (
    AddPermissionRules,
    AddWorkingDirectories,
    ModePermissionDecisionReason,
    OtherPermissionDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionBehavior,
    PermissionDenyDecision,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
    PermissionRuleValue,
    RulePermissionDecisionReason,
    SafetyCheckPermissionDecisionReason,
    SetPermissionMode,
    ToolPermissionContext,
    WorkingDirPermissionDecisionReason,
)

FILE_READ_TOOL_NAME = "Read"
FILE_EDIT_TOOL_NAME = "Edit"
FILE_WRITE_TOOL_NAME = "Write"

type FilePermissionToolType = Literal["read", "edit"]
type FilePermissionOperation = Literal["read", "write", "create"]

_PERMISSION_RULE_SOURCES: tuple[PermissionRuleSource, ...] = (
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
    "cliArg",
    "command",
    "session",
)

_DANGEROUS_COMPONENTS = frozenset(
    {
        ".codex",
        ".git",
        ".hg",
        ".raygent",
        ".raygent_harness",
        ".ssh",
        ".svn",
    }
)
_DANGEROUS_BASENAMES = frozenset(
    {
        ".bash_profile",
        ".bashrc",
        ".env",
        ".gitconfig",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".zprofile",
        ".zshrc",
        "authorized_keys",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)


@dataclass(frozen=True)
class PathSafetyResult:
    safe: bool
    message: str | None = None
    classifier_approvable: bool = True


def expand_file_path(path: str, *, cwd: str | None = None) -> str:
    """Expand user/relative paths without touching the filesystem."""

    normalized = unicodedata.normalize("NFC", path)
    if _is_unc_like_path(normalized):
        return normalized
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/"):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    elif not os.path.isabs(normalized):
        normalized = os.path.join(cwd or os.getcwd(), normalized)
    return os.path.normpath(normalized)


def get_paths_for_permission_check(path: str, *, cwd: str | None = None) -> tuple[str, ...]:
    """Return all path forms that permission checks must evaluate.

    The returned tuple includes the expanded original path, intermediate
    symlink targets, dangling/new-path deepest-existing-ancestor resolutions,
    and final realpath when available. UNC-like paths are returned immediately
    so permission checks never trigger network filesystem access.
    """

    expanded = expand_file_path(path, cwd=cwd)
    paths: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate is None:
            return
        normalized = os.path.normpath(candidate)
        if normalized not in paths:
            paths.append(normalized)

    add(expanded)

    if _is_unc_like_path(expanded):
        return tuple(paths)

    try:
        current_path = expanded
        visited: set[str] = set()
        for _depth in range(40):
            if current_path in visited:
                break
            visited.add(current_path)

            if not os.path.exists(current_path):
                if current_path == expanded:
                    add(resolve_deepest_existing_ancestor(expanded))
                break

            stats = os.lstat(current_path)
            if _is_special_file(stats.st_mode):
                break
            if not stat.S_ISLNK(stats.st_mode):
                break

            target = os.readlink(current_path)
            absolute_target = (
                target
                if os.path.isabs(target)
                else os.path.abspath(os.path.join(os.path.dirname(current_path), target))
            )
            add(absolute_target)
            current_path = absolute_target
    except OSError:
        pass

    final_realpath = os.path.realpath(expanded)
    if os.path.normpath(final_realpath) != expanded:
        add(final_realpath)

    return tuple(paths)


def resolve_deepest_existing_ancestor(path: str) -> str | None:
    """Resolve the deepest existing ancestor for new/dangling paths."""

    expanded = expand_file_path(path)
    current = expanded
    segments: list[str] = []

    while current != os.path.dirname(current):
        try:
            current_stat = os.lstat(current)
        except OSError:
            segments.insert(0, os.path.basename(current))
            current = os.path.dirname(current)
            continue

        if stat.S_ISLNK(current_stat.st_mode):
            try:
                target = os.readlink(current)
            except OSError:
                return None
            resolved = (
                target
                if os.path.isabs(target)
                else os.path.abspath(os.path.join(os.path.dirname(current), target))
            )
            return _join_tail(resolved, segments)

        try:
            resolved = os.path.realpath(current)
        except OSError:
            return None
        if os.path.normpath(resolved) != os.path.normpath(current):
            return _join_tail(resolved, segments)
        return None

    return None


def path_in_working_path(path: str, working_path: str) -> bool:
    expanded_path = _normalize_for_comparison(expand_file_path(path))
    expanded_working_path = _normalize_for_comparison(expand_file_path(working_path))
    try:
        return os.path.commonpath((expanded_path, expanded_working_path)) == expanded_working_path
    except ValueError:
        return False


def path_in_allowed_working_path(
    path: str,
    permission_context: ToolPermissionContext,
    *,
    cwd: str,
    paths_to_check: Sequence[str] | None = None,
) -> bool:
    checked_paths = tuple(paths_to_check or get_paths_for_permission_check(path, cwd=cwd))
    working_paths = tuple(
        resolved
        for working_dir in _all_working_directories(permission_context, cwd=cwd)
        for resolved in get_paths_for_permission_check(working_dir, cwd=cwd)
    )
    return all(
        any(path_in_working_path(path_to_check, working_path) for working_path in working_paths)
        for path_to_check in checked_paths
    )


def matching_file_permission_rule(
    path: str,
    permission_context: ToolPermissionContext,
    *,
    tool_type: FilePermissionToolType,
    behavior: PermissionBehavior,
    cwd: str,
) -> PermissionRule | None:
    tool_name = FILE_READ_TOOL_NAME if tool_type == "read" else FILE_EDIT_TOOL_NAME
    for rule in _rules_for_tool(permission_context, tool_name, behavior):
        pattern = rule.rule_value.rule_content
        if pattern is not None and _pattern_matches_path(pattern, path, cwd=cwd):
            return rule
    return None


def check_read_permission_for_path(
    path: str,
    permission_context: ToolPermissionContext,
    *,
    cwd: str,
    input: Mapping[str, object] | None = None,
    extra_allowed_read_roots: Sequence[str] = (),
) -> PermissionResult:
    expanded = expand_file_path(path, cwd=cwd)
    paths_to_check = get_paths_for_permission_check(path, cwd=cwd)
    updated_input = dict(input or {"file_path": path})

    unc_result = _check_unc_or_suspicious_read(path, paths_to_check)
    if unc_result is not None:
        return unc_result

    for path_to_check in paths_to_check:
        deny_rule = matching_file_permission_rule(
            path_to_check,
            permission_context,
            tool_type="read",
            behavior="deny",
            cwd=cwd,
        )
        if deny_rule is not None:
            return PermissionDenyDecision(
                message=f"Permission to read {path} has been denied.",
                decision_reason=RulePermissionDecisionReason(rule=deny_rule),
            )

    for path_to_check in paths_to_check:
        ask_rule = matching_file_permission_rule(
            path_to_check,
            permission_context,
            tool_type="read",
            behavior="ask",
            cwd=cwd,
        )
        if ask_rule is not None:
            return PermissionAskDecision(
                message=(
                    f"Raygent requested permission to read from {path}, "
                    "but it has not been granted yet."
                ),
                decision_reason=RulePermissionDecisionReason(rule=ask_rule),
                blocked_path=expanded,
            )

    edit_result = check_write_permission_for_path(
        path,
        permission_context,
        cwd=cwd,
        input=updated_input,
        paths_to_check=paths_to_check,
    )
    if isinstance(edit_result, PermissionAllowDecision):
        return edit_result

    if path_in_allowed_working_path(
        path,
        permission_context,
        cwd=cwd,
        paths_to_check=paths_to_check,
    ):
        return PermissionAllowDecision(
            updated_input=updated_input,
            decision_reason=ModePermissionDecisionReason(mode="default"),
        )

    if _path_in_any_allowed_root(paths_to_check, extra_allowed_read_roots, cwd=cwd):
        return PermissionAllowDecision(
            updated_input=updated_input,
            decision_reason=ModePermissionDecisionReason(mode="default"),
        )

    allow_rule = matching_file_permission_rule(
        path,
        permission_context,
        tool_type="read",
        behavior="allow",
        cwd=cwd,
    )
    if allow_rule is not None:
        return PermissionAllowDecision(
            updated_input=updated_input,
            decision_reason=RulePermissionDecisionReason(rule=allow_rule),
        )

    return PermissionAskDecision(
        message=(
            f"Raygent requested permission to read from {path}, "
            "but it has not been granted yet."
        ),
        suggestions=generate_file_permission_suggestions(
            path,
            "read",
            permission_context,
            cwd=cwd,
            paths_to_check=paths_to_check,
        ),
        blocked_path=expanded,
        decision_reason=WorkingDirPermissionDecisionReason(
            reason="Path is outside allowed working directories"
        ),
    )


def check_write_permission_for_path(
    path: str,
    permission_context: ToolPermissionContext,
    *,
    cwd: str,
    input: Mapping[str, object] | None = None,
    paths_to_check: Sequence[str] | None = None,
) -> PermissionResult:
    expanded = expand_file_path(path, cwd=cwd)
    checked_paths = tuple(paths_to_check or get_paths_for_permission_check(path, cwd=cwd))
    updated_input = dict(input or {"file_path": path})

    for path_to_check in checked_paths:
        deny_rule = matching_file_permission_rule(
            path_to_check,
            permission_context,
            tool_type="edit",
            behavior="deny",
            cwd=cwd,
        )
        if deny_rule is not None:
            return PermissionDenyDecision(
                message=f"Permission to edit {path} has been denied.",
                decision_reason=RulePermissionDecisionReason(rule=deny_rule),
            )

    safety = check_path_safety_for_auto_edit(path, paths_to_check=checked_paths)
    if not safety.safe:
        return PermissionAskDecision(
            message=safety.message
            or f"Raygent requested permission to write to {path}, but it is sensitive.",
            suggestions=generate_file_permission_suggestions(
                path,
                "write",
                permission_context,
                cwd=cwd,
                paths_to_check=checked_paths,
            ),
            blocked_path=expanded,
            decision_reason=SafetyCheckPermissionDecisionReason(
                reason=safety.message or "Sensitive path requires manual approval",
                classifier_approvable=safety.classifier_approvable,
            ),
        )

    for path_to_check in checked_paths:
        ask_rule = matching_file_permission_rule(
            path_to_check,
            permission_context,
            tool_type="edit",
            behavior="ask",
            cwd=cwd,
        )
        if ask_rule is not None:
            return PermissionAskDecision(
                message=(
                    f"Raygent requested permission to write to {path}, "
                    "but it has not been granted yet."
                ),
                decision_reason=RulePermissionDecisionReason(rule=ask_rule),
                blocked_path=expanded,
            )

    is_in_working_dir = path_in_allowed_working_path(
        path,
        permission_context,
        cwd=cwd,
        paths_to_check=checked_paths,
    )
    if permission_context.mode == "acceptEdits" and is_in_working_dir:
        return PermissionAllowDecision(
            updated_input=updated_input,
            decision_reason=ModePermissionDecisionReason(mode=permission_context.mode),
        )

    allow_rule = matching_file_permission_rule(
        path,
        permission_context,
        tool_type="edit",
        behavior="allow",
        cwd=cwd,
    )
    if allow_rule is not None:
        return PermissionAllowDecision(
            updated_input=updated_input,
            decision_reason=RulePermissionDecisionReason(rule=allow_rule),
        )

    return PermissionAskDecision(
        message=(
            f"Raygent requested permission to write to {path}, "
            "but it has not been granted yet."
        ),
        suggestions=generate_file_permission_suggestions(
            path,
            "write",
            permission_context,
            cwd=cwd,
            paths_to_check=checked_paths,
        ),
        blocked_path=expanded,
        decision_reason=(
            WorkingDirPermissionDecisionReason(
                reason="Path is outside allowed working directories"
            )
            if not is_in_working_dir
            else None
        ),
    )


def check_path_safety_for_auto_edit(
    path: str,
    *,
    paths_to_check: Sequence[str] | None = None,
) -> PathSafetyResult:
    checked_paths = tuple(paths_to_check or get_paths_for_permission_check(path))
    for path_to_check in checked_paths:
        if _is_unc_like_path(path_to_check):
            return PathSafetyResult(
                safe=False,
                message=(
                    f"Raygent requested permission to write to {path}, which "
                    "appears to be a UNC path that could access network resources."
                ),
                classifier_approvable=False,
            )

    for path_to_check in checked_paths:
        if has_suspicious_windows_path_pattern(path_to_check):
            return PathSafetyResult(
                safe=False,
                message=(
                    f"Raygent requested permission to write to {path}, which "
                    "contains a suspicious Windows path pattern."
                ),
                classifier_approvable=False,
            )

    for path_to_check in checked_paths:
        if is_dangerous_file_path_to_auto_edit(path_to_check):
            return PathSafetyResult(
                safe=False,
                message=f"Raygent requested permission to edit {path}, which is sensitive.",
                classifier_approvable=True,
            )

    return PathSafetyResult(safe=True)


def generate_file_permission_suggestions(
    path: str,
    operation: FilePermissionOperation,
    permission_context: ToolPermissionContext,
    *,
    cwd: str,
    paths_to_check: Sequence[str] | None = None,
) -> tuple[AddPermissionRules | AddWorkingDirectories | SetPermissionMode, ...]:
    checked_paths = tuple(paths_to_check or get_paths_for_permission_check(path, cwd=cwd))
    is_outside_working_dir = not path_in_allowed_working_path(
        path,
        permission_context,
        cwd=cwd,
        paths_to_check=checked_paths,
    )
    if operation == "read" and is_outside_working_dir:
        directory_paths = get_paths_for_permission_check(
            os.path.dirname(expand_file_path(path, cwd=cwd)),
            cwd=cwd,
        )
        rules = tuple(
            PermissionRuleValue(
                tool_name=FILE_READ_TOOL_NAME,
                rule_content=_directory_rule_content(path_to_check),
            )
            for path_to_check in directory_paths
        )
        return (
            AddPermissionRules(
                destination="session",
                rules=rules,
                behavior="allow",
            ),
        )

    suggestions: list[AddWorkingDirectories | SetPermissionMode] = []
    if permission_context.mode in {"default", "plan"}:
        suggestions.append(SetPermissionMode(destination="session", mode="acceptEdits"))
    if operation in {"write", "create"} and is_outside_working_dir:
        directory_paths = get_paths_for_permission_check(
            os.path.dirname(expand_file_path(path, cwd=cwd)),
            cwd=cwd,
        )
        suggestions.append(
            AddWorkingDirectories(
                destination="session",
                directories=tuple(directory_paths),
            )
        )
    return tuple(suggestions)


def has_suspicious_windows_path_pattern(path: str) -> bool:
    if (
        path.startswith("\\\\?\\")
        or path.startswith("\\\\.\\")
        or path.startswith("//?/")
        or path.startswith("//./")
    ):
        return True
    if re.search(r"(^|[\\/])[^\\/]*~\d($|[\\/])", path):
        return True
    if re.search(r"[.\s]+$", path):
        return True
    if re.search(r"\.(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", path, re.IGNORECASE):
        return True
    if re.search(r"(^|/|\\)\.{3,}(/|\\|$)", path):
        return True
    colon_indexes = [index for index, char in enumerate(path) if char == ":"]
    return bool(colon_indexes and colon_indexes != [1])


def is_dangerous_file_path_to_auto_edit(path: str) -> bool:
    lowered_parts = tuple(part.lower() for part in _path_parts(path))
    basename = lowered_parts[-1] if lowered_parts else ""
    if any(part in _DANGEROUS_COMPONENTS for part in lowered_parts):
        return True
    if basename in _DANGEROUS_BASENAMES:
        return True
    return bool(basename.startswith(".env."))


def _check_unc_or_suspicious_read(
    path: str,
    paths_to_check: Sequence[str],
) -> PermissionAskDecision | None:
    for path_to_check in paths_to_check:
        if _is_unc_like_path(path_to_check):
            return PermissionAskDecision(
                message=(
                    f"Raygent requested permission to read from {path}, which "
                    "appears to be a UNC path that could access network resources."
                ),
                decision_reason=OtherPermissionDecisionReason(reason="UNC path detected"),
                blocked_path=path_to_check,
            )
    for path_to_check in paths_to_check:
        if has_suspicious_windows_path_pattern(path_to_check):
            return PermissionAskDecision(
                message=(
                    f"Raygent requested permission to read from {path}, which "
                    "contains a suspicious Windows path pattern."
                ),
                decision_reason=OtherPermissionDecisionReason(
                    reason="Suspicious Windows path pattern detected"
                ),
                blocked_path=path_to_check,
            )
    return None


def _rules_for_tool(
    permission_context: ToolPermissionContext,
    tool_name: str,
    behavior: PermissionBehavior,
) -> tuple[PermissionRule, ...]:
    """Return file path rules for the requested tool/behavior.

    Chunk 1 treats rule contents as already-canonical headless path patterns.
    The reference re-roots non-session settings rules relative to product
    settings directories; Raygent has no settings-root model yet, so adapters
    should feed canonical absolute or cwd-relative patterns until that seam
    exists.
    """

    rules_by_source = {
        "allow": permission_context.always_allow_rules,
        "deny": permission_context.always_deny_rules,
        "ask": permission_context.always_ask_rules,
    }[behavior]
    rules: list[PermissionRule] = []
    for source in _PERMISSION_RULE_SOURCES:
        for rule_string in rules_by_source.get(source, ()):
            rule_value = permission_rule_value_from_string(rule_string)
            if rule_value.tool_name == tool_name and rule_value.rule_content is not None:
                rules.append(
                    PermissionRule(
                        source=source,
                        rule_behavior=behavior,
                        rule_value=rule_value,
                    )
                )
    return tuple(rules)


def _pattern_matches_path(pattern: str, path: str, *, cwd: str) -> bool:
    normalized_path = _normalize_for_comparison(expand_file_path(path, cwd=cwd))
    if pattern.endswith("/**"):
        base = _normalize_for_comparison(expand_file_path(pattern[:-3], cwd=cwd))
        return path_in_working_path(normalized_path, base)

    expanded_pattern = _normalize_for_comparison(expand_file_path(pattern, cwd=cwd))
    if _has_glob_magic(pattern):
        return fnmatch.fnmatchcase(normalized_path, expanded_pattern)

    return normalized_path == expanded_pattern or path_in_working_path(
        normalized_path,
        expanded_pattern,
    )


def _all_working_directories(
    permission_context: ToolPermissionContext,
    *,
    cwd: str,
) -> tuple[str, ...]:
    return (cwd, *permission_context.additional_working_directories.keys())


def _path_in_any_allowed_root(
    paths_to_check: Sequence[str],
    roots: Sequence[str],
    *,
    cwd: str,
) -> bool:
    if not roots:
        return False
    checked_roots = tuple(
        root_path
        for root in roots
        for root_path in get_paths_for_permission_check(root, cwd=cwd)
    )
    return all(
        any(path_in_working_path(path_to_check, root_path) for root_path in checked_roots)
        for path_to_check in paths_to_check
    )


def _directory_rule_content(path: str) -> str:
    return f"{os.path.normpath(path)}/**"


def _join_tail(root: str, segments: Sequence[str]) -> str:
    if not segments:
        return os.path.normpath(root)
    return os.path.normpath(os.path.join(root, *segments))


def _is_unc_like_path(path: str) -> bool:
    return path.startswith("//") or path.startswith("\\\\")


def _is_special_file(mode: int) -> bool:
    return (
        stat.S_ISFIFO(mode)
        or stat.S_ISSOCK(mode)
        or stat.S_ISCHR(mode)
        or stat.S_ISBLK(mode)
    )


def _path_parts(path: str) -> tuple[str, ...]:
    normalized = os.path.normpath(path)
    parts: list[str] = []
    while True:
        head, tail = os.path.split(normalized)
        if tail:
            parts.insert(0, tail)
            normalized = head
            continue
        if head and head != os.path.dirname(head):
            parts.insert(0, head)
        return tuple(parts)


def _normalize_for_comparison(path: str) -> str:
    normalized = os.path.normpath(path)
    normalized = normalized.replace("/private/var/", "/var/")
    normalized = re.sub(r"^/private/tmp(/|$)", r"/tmp\1", normalized)
    if sys.platform in {"darwin", "win32"}:
        return normalized.lower()
    return os.path.normcase(normalized)


def _has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


__all__ = [
    "FILE_EDIT_TOOL_NAME",
    "FILE_READ_TOOL_NAME",
    "FILE_WRITE_TOOL_NAME",
    "FilePermissionOperation",
    "FilePermissionToolType",
    "PathSafetyResult",
    "check_path_safety_for_auto_edit",
    "check_read_permission_for_path",
    "check_write_permission_for_path",
    "expand_file_path",
    "generate_file_permission_suggestions",
    "get_paths_for_permission_check",
    "has_suspicious_windows_path_pattern",
    "is_dangerous_file_path_to_auto_edit",
    "matching_file_permission_rule",
    "path_in_allowed_working_path",
    "path_in_working_path",
    "resolve_deepest_existing_ancestor",
]
