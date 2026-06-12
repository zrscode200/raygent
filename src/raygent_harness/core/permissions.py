"""Pure permission data model.

This module is intentionally behavior-free: no rule matching, prompting, hook
resolution, or tool execution. Those live in the permission engine and tool
execution layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Permission modes
# ---------------------------------------------------------------------------

ExternalPermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
]

InternalPermissionMode = ExternalPermissionMode | Literal["auto", "bubble"]
PermissionMode = InternalPermissionMode

EXTERNAL_PERMISSION_MODES: tuple[ExternalPermissionMode, ...] = (
    "acceptEdits",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
)
"""Modes accepted from public config/API surfaces."""

INTERNAL_PERMISSION_MODES: tuple[PermissionMode, ...] = (
    *EXTERNAL_PERMISSION_MODES,
    "auto",
    "bubble",
)
"""Exhaustive mode set for internal typing.

The reference exposes `auto` behind a feature flag and keeps `bubble` as an
internal mode. Raygent has no feature-flag subsystem yet, so this is a data
shape only; runtime availability belongs in the future permission engine.
"""

PERMISSION_MODES = INTERNAL_PERMISSION_MODES


# ---------------------------------------------------------------------------
# Permission rules and updates
# ---------------------------------------------------------------------------

PermissionBehavior = Literal["allow", "deny", "ask"]

PermissionRuleSource = Literal[
    "userSettings",
    "projectSettings",
    "localSettings",
    "flagSettings",
    "policySettings",
    "cliArg",
    "command",
    "session",
]

PermissionUpdateDestination = Literal[
    "userSettings",
    "projectSettings",
    "localSettings",
    "session",
    "cliArg",
]

WorkingDirectorySource = PermissionRuleSource
type ToolPermissionRulesBySource = Mapping[PermissionRuleSource, tuple[str, ...]]


def _empty_rule_map() -> ToolPermissionRulesBySource:
    return MappingProxyType({})


@dataclass(frozen=True)
class PermissionRuleValue:
    """The tool plus optional rule content matched by a permission rule."""

    tool_name: str
    rule_content: str | None = None


@dataclass(frozen=True)
class PermissionRule:
    """A permission rule with its provenance and behavior."""

    source: PermissionRuleSource
    rule_behavior: PermissionBehavior
    rule_value: PermissionRuleValue


@dataclass(frozen=True)
class AddPermissionRules:
    destination: PermissionUpdateDestination
    rules: tuple[PermissionRuleValue, ...]
    behavior: PermissionBehavior
    type: Literal["addRules"] = "addRules"


@dataclass(frozen=True)
class ReplacePermissionRules:
    destination: PermissionUpdateDestination
    rules: tuple[PermissionRuleValue, ...]
    behavior: PermissionBehavior
    type: Literal["replaceRules"] = "replaceRules"


@dataclass(frozen=True)
class RemovePermissionRules:
    destination: PermissionUpdateDestination
    rules: tuple[PermissionRuleValue, ...]
    behavior: PermissionBehavior
    type: Literal["removeRules"] = "removeRules"


@dataclass(frozen=True)
class SetPermissionMode:
    destination: PermissionUpdateDestination
    mode: ExternalPermissionMode
    type: Literal["setMode"] = "setMode"


@dataclass(frozen=True)
class AddWorkingDirectories:
    destination: PermissionUpdateDestination
    directories: tuple[str, ...]
    type: Literal["addDirectories"] = "addDirectories"


@dataclass(frozen=True)
class RemoveWorkingDirectories:
    destination: PermissionUpdateDestination
    directories: tuple[str, ...]
    type: Literal["removeDirectories"] = "removeDirectories"


type PermissionUpdate = (
    AddPermissionRules
    | ReplacePermissionRules
    | RemovePermissionRules
    | SetPermissionMode
    | AddWorkingDirectories
    | RemoveWorkingDirectories
)


@dataclass(frozen=True)
class AdditionalWorkingDirectory:
    path: str
    source: WorkingDirectorySource


@dataclass(frozen=True)
class ToolPermissionContext:
    """Permission state available while deciding whether a tool may run."""

    mode: PermissionMode = "default"
    additional_working_directories: Mapping[str, AdditionalWorkingDirectory] = field(
        default_factory=lambda: MappingProxyType({})
    )
    always_allow_rules: ToolPermissionRulesBySource = field(
        default_factory=_empty_rule_map
    )
    always_deny_rules: ToolPermissionRulesBySource = field(default_factory=_empty_rule_map)
    always_ask_rules: ToolPermissionRulesBySource = field(default_factory=_empty_rule_map)
    is_bypass_permissions_mode_available: bool = False
    is_auto_mode_available: bool = False
    stripped_dangerous_rules: ToolPermissionRulesBySource | None = None
    should_avoid_permission_prompts: bool = False
    await_automated_checks_before_dialog: bool = False
    pre_plan_mode: PermissionMode | None = None


def empty_tool_permission_context() -> ToolPermissionContext:
    return ToolPermissionContext()


# ---------------------------------------------------------------------------
# Permission decisions and reasons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermissionCommandMetadata:
    name: str
    description: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class PermissionMetadata:
    command: PermissionCommandMetadata


@dataclass(frozen=True)
class PendingClassifierCheck:
    command: str
    cwd: str
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class RulePermissionDecisionReason:
    rule: PermissionRule
    type: Literal["rule"] = "rule"


@dataclass(frozen=True)
class ModePermissionDecisionReason:
    mode: PermissionMode
    type: Literal["mode"] = "mode"


@dataclass(frozen=True)
class SubcommandResultsPermissionDecisionReason:
    reasons: Mapping[str, PermissionResult]
    type: Literal["subcommandResults"] = "subcommandResults"


@dataclass(frozen=True)
class PermissionPromptToolDecisionReason:
    permission_prompt_tool_name: str
    tool_result: Any
    type: Literal["permissionPromptTool"] = "permissionPromptTool"


@dataclass(frozen=True)
class HookPermissionDecisionReason:
    hook_name: str
    hook_source: str | None = None
    reason: str | None = None
    type: Literal["hook"] = "hook"


@dataclass(frozen=True)
class AsyncAgentPermissionDecisionReason:
    reason: str
    type: Literal["asyncAgent"] = "asyncAgent"


@dataclass(frozen=True)
class SandboxOverridePermissionDecisionReason:
    reason: Literal["excludedCommand", "dangerouslyDisableSandbox"]
    type: Literal["sandboxOverride"] = "sandboxOverride"


@dataclass(frozen=True)
class ClassifierPermissionDecisionReason:
    classifier: str
    reason: str
    type: Literal["classifier"] = "classifier"


@dataclass(frozen=True)
class WorkingDirPermissionDecisionReason:
    reason: str
    type: Literal["workingDir"] = "workingDir"


@dataclass(frozen=True)
class SafetyCheckPermissionDecisionReason:
    reason: str
    classifier_approvable: bool
    type: Literal["safetyCheck"] = "safetyCheck"


@dataclass(frozen=True)
class OtherPermissionDecisionReason:
    reason: str
    type: Literal["other"] = "other"


type PermissionDecisionReason = (
    RulePermissionDecisionReason
    | ModePermissionDecisionReason
    | SubcommandResultsPermissionDecisionReason
    | PermissionPromptToolDecisionReason
    | HookPermissionDecisionReason
    | AsyncAgentPermissionDecisionReason
    | SandboxOverridePermissionDecisionReason
    | ClassifierPermissionDecisionReason
    | WorkingDirPermissionDecisionReason
    | SafetyCheckPermissionDecisionReason
    | OtherPermissionDecisionReason
)


@dataclass(frozen=True)
class PermissionAllowDecision:
    updated_input: Mapping[str, Any] | None = None
    user_modified: bool = False
    decision_reason: PermissionDecisionReason | None = None
    tool_use_id: str | None = None
    accept_feedback: str | None = None
    content_blocks: tuple[Any, ...] = ()
    behavior: Literal["allow"] = "allow"


@dataclass(frozen=True)
class PermissionAskDecision:
    message: str
    updated_input: Mapping[str, Any] | None = None
    decision_reason: PermissionDecisionReason | None = None
    suggestions: tuple[PermissionUpdate, ...] = ()
    blocked_path: str | None = None
    metadata: PermissionMetadata | None = None
    is_bash_security_check_for_misparsing: bool = False
    pending_classifier_check: PendingClassifierCheck | None = None
    content_blocks: tuple[Any, ...] = ()
    behavior: Literal["ask"] = "ask"


@dataclass(frozen=True)
class PermissionDenyDecision:
    message: str
    decision_reason: PermissionDecisionReason
    tool_use_id: str | None = None
    behavior: Literal["deny"] = "deny"


type PermissionDecision = (
    PermissionAllowDecision | PermissionAskDecision | PermissionDenyDecision
)


@dataclass(frozen=True)
class PermissionPassthrough:
    message: str
    decision_reason: PermissionDecisionReason | None = None
    suggestions: tuple[PermissionUpdate, ...] = ()
    blocked_path: str | None = None
    pending_classifier_check: PendingClassifierCheck | None = None
    behavior: Literal["passthrough"] = "passthrough"


type PermissionResult = PermissionDecision | PermissionPassthrough


__all__ = [
    "EXTERNAL_PERMISSION_MODES",
    "INTERNAL_PERMISSION_MODES",
    "PERMISSION_MODES",
    "AddPermissionRules",
    "AddWorkingDirectories",
    "AdditionalWorkingDirectory",
    "AsyncAgentPermissionDecisionReason",
    "ClassifierPermissionDecisionReason",
    "ExternalPermissionMode",
    "HookPermissionDecisionReason",
    "InternalPermissionMode",
    "ModePermissionDecisionReason",
    "OtherPermissionDecisionReason",
    "PendingClassifierCheck",
    "PermissionAllowDecision",
    "PermissionAskDecision",
    "PermissionBehavior",
    "PermissionCommandMetadata",
    "PermissionDecision",
    "PermissionDecisionReason",
    "PermissionDenyDecision",
    "PermissionMetadata",
    "PermissionMode",
    "PermissionPassthrough",
    "PermissionPromptToolDecisionReason",
    "PermissionResult",
    "PermissionRule",
    "PermissionRuleSource",
    "PermissionRuleValue",
    "PermissionUpdate",
    "PermissionUpdateDestination",
    "RemovePermissionRules",
    "RemoveWorkingDirectories",
    "ReplacePermissionRules",
    "RulePermissionDecisionReason",
    "SafetyCheckPermissionDecisionReason",
    "SandboxOverridePermissionDecisionReason",
    "SetPermissionMode",
    "SubcommandResultsPermissionDecisionReason",
    "ToolPermissionContext",
    "ToolPermissionRulesBySource",
    "WorkingDirPermissionDecisionReason",
    "WorkingDirectorySource",
    "empty_tool_permission_context",
]
