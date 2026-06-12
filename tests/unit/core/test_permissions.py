from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from raygent_harness.core.permissions import (
    EXTERNAL_PERMISSION_MODES,
    INTERNAL_PERMISSION_MODES,
    AddPermissionRules,
    AddWorkingDirectories,
    AsyncAgentPermissionDecisionReason,
    ClassifierPermissionDecisionReason,
    HookPermissionDecisionReason,
    ModePermissionDecisionReason,
    OtherPermissionDecisionReason,
    PendingClassifierCheck,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    PermissionPassthrough,
    PermissionPromptToolDecisionReason,
    PermissionRule,
    PermissionRuleValue,
    RemovePermissionRules,
    RemoveWorkingDirectories,
    ReplacePermissionRules,
    RulePermissionDecisionReason,
    SafetyCheckPermissionDecisionReason,
    SandboxOverridePermissionDecisionReason,
    SetPermissionMode,
    SubcommandResultsPermissionDecisionReason,
    WorkingDirPermissionDecisionReason,
)


def test_permission_modes_match_reference_external_order_and_internal_extensions() -> None:
    assert EXTERNAL_PERMISSION_MODES == (
        "acceptEdits",
        "bypassPermissions",
        "default",
        "dontAsk",
        "plan",
    )
    assert INTERNAL_PERMISSION_MODES == (
        "acceptEdits",
        "bypassPermissions",
        "default",
        "dontAsk",
        "plan",
        "auto",
        "bubble",
    )


def test_permission_rule_and_update_shapes_preserve_reference_discriminants() -> None:
    rule_value = PermissionRuleValue(tool_name="Bash", rule_content="git status")
    rule = PermissionRule(
        source="projectSettings",
        rule_behavior="allow",
        rule_value=rule_value,
    )

    assert rule.rule_value.tool_name == "Bash"
    assert rule.rule_value.rule_content == "git status"

    updates = (
        AddPermissionRules(
            destination="localSettings",
            rules=(rule_value,),
            behavior="allow",
        ),
        ReplacePermissionRules(
            destination="session",
            rules=(rule_value,),
            behavior="deny",
        ),
        RemovePermissionRules(
            destination="userSettings",
            rules=(rule_value,),
            behavior="ask",
        ),
        SetPermissionMode(destination="cliArg", mode="plan"),
        AddWorkingDirectories(destination="session", directories=("/tmp/a",)),
        RemoveWorkingDirectories(destination="session", directories=("/tmp/a",)),
    )

    assert [update.type for update in updates] == [
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]


def test_permission_decisions_carry_behavior_updated_input_and_reasons() -> None:
    rule = PermissionRule(
        source="session",
        rule_behavior="allow",
        rule_value=PermissionRuleValue("Read"),
    )
    reason = RulePermissionDecisionReason(rule=rule)

    allow = PermissionAllowDecision(
        updated_input={"file_path": "README.md"},
        decision_reason=reason,
        tool_use_id="tu_1",
    )
    ask = PermissionAskDecision(
        message="Allow?",
        suggestions=(
            AddPermissionRules(
                destination="session",
                rules=(PermissionRuleValue("Read"),),
                behavior="allow",
            ),
        ),
        decision_reason=ModePermissionDecisionReason(mode="default"),
    )
    deny = PermissionDenyDecision(
        message="Denied",
        decision_reason=SafetyCheckPermissionDecisionReason(
            reason="sensitive path",
            classifier_approvable=True,
        ),
    )
    passthrough = PermissionPassthrough(
        message="Ask host",
        decision_reason=HookPermissionDecisionReason(hook_name="pre-tool"),
    )

    assert allow.behavior == "allow"
    assert allow.updated_input == {"file_path": "README.md"}
    assert ask.behavior == "ask"
    assert ask.suggestions[0].type == "addRules"
    assert deny.behavior == "deny"
    assert passthrough.behavior == "passthrough"


def test_permission_reason_variants_preserve_reference_type_tags() -> None:
    nested = PermissionPassthrough(message="child")
    reasons = (
        RulePermissionDecisionReason(
            rule=PermissionRule(
                source="session",
                rule_behavior="allow",
                rule_value=PermissionRuleValue("Read"),
            )
        ),
        ModePermissionDecisionReason(mode="bypassPermissions"),
        SubcommandResultsPermissionDecisionReason(reasons={"git": nested}),
        PermissionPromptToolDecisionReason(
            permission_prompt_tool_name="ask_user",
            tool_result={"ok": True},
        ),
        HookPermissionDecisionReason(hook_name="pre-tool"),
        AsyncAgentPermissionDecisionReason(reason="delegated"),
        SandboxOverridePermissionDecisionReason(reason="excludedCommand"),
        ClassifierPermissionDecisionReason(classifier="bash", reason="safe"),
        WorkingDirPermissionDecisionReason(reason="inside cwd"),
        SafetyCheckPermissionDecisionReason(
            reason="sensitive file",
            classifier_approvable=True,
        ),
        OtherPermissionDecisionReason(reason="manual"),
    )

    assert [reason.type for reason in reasons] == [
        "rule",
        "mode",
        "subcommandResults",
        "permissionPromptTool",
        "hook",
        "asyncAgent",
        "sandboxOverride",
        "classifier",
        "workingDir",
        "safetyCheck",
        "other",
    ]
    assert reasons[2].reasons["git"] is nested


def test_pending_classifier_check_is_carried_by_ask_and_passthrough() -> None:
    pending = PendingClassifierCheck(
        command="git status",
        cwd="/repo",
        descriptions=("read git status",),
    )

    ask = PermissionAskDecision(message="Allow?", pending_classifier_check=pending)
    passthrough = PermissionPassthrough(
        message="Ask host",
        pending_classifier_check=pending,
    )

    assert ask.pending_classifier_check is pending
    assert passthrough.pending_classifier_check is pending


def test_permission_models_are_frozen() -> None:
    rule_value = PermissionRuleValue(tool_name="Bash")

    with pytest.raises(FrozenInstanceError):
        rule_value.tool_name = "Read"  # type: ignore[misc]
