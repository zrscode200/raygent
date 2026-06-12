from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from raygent_harness.core.permission_engine import can_use_tool
from raygent_harness.core.permissions import (
    AddPermissionRules,
    ModePermissionDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    PermissionMode,
    PermissionResult,
    RulePermissionDecisionReason,
    SafetyCheckPermissionDecisionReason,
    ToolPermissionContext,
    WorkingDirPermissionDecisionReason,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.tools.file_permissions import (
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    check_path_safety_for_auto_edit,
    check_read_permission_for_path,
    check_write_permission_for_path,
    get_paths_for_permission_check,
    has_suspicious_windows_path_pattern,
    path_in_allowed_working_path,
)


class FileInput(BaseModel):
    file_path: str


type CheckPermissionsFn = Callable[
    [BaseModel, ToolUseContext, ToolPermissionContext], Awaitable[PermissionResult]
]


def _permission_context(
    *,
    mode: PermissionMode = "default",
    allow: tuple[str, ...] = (),
    deny: tuple[str, ...] = (),
    ask: tuple[str, ...] = (),
) -> ToolPermissionContext:
    return ToolPermissionContext(
        mode=mode,
        always_allow_rules={"session": allow},
        always_deny_rules={"session": deny},
        always_ask_rules={"session": ask},
    )


def _tool_ctx(cwd: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
    )


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _file_tool(check_permissions: CheckPermissionsFn) -> Tool:
    return build_tool(
        ToolSpec(
            name="Write",
            description="write file",
            input_model=FileInput,
            call=_call,
            check_permissions=check_permissions,
            is_read_only=False,
            is_concurrency_safe=False,
        )
    )


def _symlink(target: Path, link: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this filesystem: {exc}")


def test_paths_for_permission_check_include_intermediate_and_final_symlink_targets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("secret")
    link_two = tmp_path / "link-two.txt"
    link_one = tmp_path / "link-one.txt"
    _symlink(target, link_two)
    _symlink(link_two, link_one)

    paths = get_paths_for_permission_check(str(link_one), cwd=str(tmp_path))

    assert str(link_one) in paths
    assert str(link_two) in paths
    assert str(target) in paths


def test_paths_for_permission_check_resolves_dangling_symlink_targets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside" / "missing.txt"
    link = tmp_path / "dangling.txt"
    _symlink(target, link)

    paths = get_paths_for_permission_check(str(link), cwd=str(tmp_path))

    assert str(link) in paths
    assert str(target) in paths


def test_paths_for_permission_check_does_not_touch_filesystem_for_unc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_exists(_path: str) -> bool:
        raise AssertionError("UNC path should not touch filesystem")

    monkeypatch.setattr("os.path.exists", fail_exists)

    assert get_paths_for_permission_check("//server/share/file.txt") == (
        "//server/share/file.txt",
    )


def test_allowed_working_path_requires_all_resolved_forms_inside_cwd(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    _symlink(outside, cwd / "link", target_is_directory=True)

    assert not path_in_allowed_working_path(
        str(cwd / "link" / "secret.txt"),
        ToolPermissionContext(),
        cwd=str(cwd),
    )


def test_read_permission_order_preserves_read_deny_and_ask_before_edit_allow(
    tmp_path: Path,
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    edit_allow = f"{FILE_EDIT_TOOL_NAME}({tmp_path}/**)"
    read_deny = f"{FILE_READ_TOOL_NAME}({tmp_path}/**)"
    read_ask = f"{FILE_READ_TOOL_NAME}({tmp_path}/**)"

    denied = check_read_permission_for_path(
        str(target),
        _permission_context(mode="acceptEdits", allow=(edit_allow,), deny=(read_deny,)),
        cwd=str(tmp_path),
    )
    assert isinstance(denied, PermissionDenyDecision)
    assert isinstance(denied.decision_reason, RulePermissionDecisionReason)

    asked = check_read_permission_for_path(
        str(target),
        _permission_context(mode="acceptEdits", allow=(edit_allow,), ask=(read_ask,)),
        cwd=str(tmp_path),
    )
    assert isinstance(asked, PermissionAskDecision)
    assert isinstance(asked.decision_reason, RulePermissionDecisionReason)


def test_edit_allow_rule_implies_read_when_no_read_rule_blocks(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    target = outside / "file.txt"
    target.write_text("x")

    result = check_read_permission_for_path(
        str(target),
        _permission_context(allow=(f"{FILE_EDIT_TOOL_NAME}({outside}/**)",)),
        cwd=str(cwd),
    )

    assert isinstance(result, PermissionAllowDecision)
    assert isinstance(result.decision_reason, RulePermissionDecisionReason)


def test_read_outside_working_directory_asks_with_read_rule_suggestion(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    target = outside / "file.txt"
    target.write_text("x")

    result = check_read_permission_for_path(
        str(target),
        ToolPermissionContext(),
        cwd=str(cwd),
    )

    assert isinstance(result, PermissionAskDecision)
    assert isinstance(result.decision_reason, WorkingDirPermissionDecisionReason)
    assert isinstance(result.suggestions[0], AddPermissionRules)
    assert result.suggestions[0].rules[0].tool_name == FILE_READ_TOOL_NAME


def test_write_permission_accept_edits_allows_inside_cwd_but_default_asks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "file.txt"

    default_result = check_write_permission_for_path(
        str(target),
        ToolPermissionContext(),
        cwd=str(tmp_path),
    )
    assert isinstance(default_result, PermissionAskDecision)

    accepted = check_write_permission_for_path(
        str(target),
        ToolPermissionContext(mode="acceptEdits"),
        cwd=str(tmp_path),
    )
    assert isinstance(accepted, PermissionAllowDecision)
    assert isinstance(accepted.decision_reason, ModePermissionDecisionReason)


@pytest.mark.asyncio
async def test_dont_ask_mode_turns_file_permission_ask_into_deny(tmp_path: Path) -> None:
    async def check_permissions(
        input_: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        parsed = input_
        assert isinstance(parsed, FileInput)
        return check_write_permission_for_path(
            parsed.file_path,
            permission_context,
            cwd=ctx.cwd,
            input=parsed.model_dump(),
        )

    target = tmp_path / "file.txt"
    result = await can_use_tool(
        tool=_file_tool(check_permissions),
        input=FileInput(file_path=str(target)),
        tool_use_context=_tool_ctx(tmp_path),
        permission_context=ToolPermissionContext(mode="dontAsk"),
    )

    assert isinstance(result.decision, PermissionDenyDecision)


def test_write_safety_check_runs_before_accept_edits_allow(tmp_path: Path) -> None:
    git_config = tmp_path / ".git" / "config"

    result = check_write_permission_for_path(
        str(git_config),
        ToolPermissionContext(mode="acceptEdits"),
        cwd=str(tmp_path),
    )

    assert isinstance(result, PermissionAskDecision)
    assert isinstance(result.decision_reason, SafetyCheckPermissionDecisionReason)


def test_path_safety_checks_all_symlink_resolved_forms(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    sensitive = tmp_path / ".codex"
    cwd.mkdir()
    sensitive.mkdir()
    target = sensitive / "memory.md"
    target.write_text("secret")
    _symlink(target, cwd / "memory.md")

    paths = get_paths_for_permission_check(str(cwd / "memory.md"), cwd=str(cwd))
    safety = check_path_safety_for_auto_edit(
        str(cwd / "memory.md"),
        paths_to_check=paths,
    )

    assert not safety.safe


def test_write_permission_asks_for_unc_even_with_explicit_allow_rule() -> None:
    result = check_write_permission_for_path(
        "//server/share/file.txt",
        _permission_context(allow=(f"{FILE_EDIT_TOOL_NAME}(//server/share/**)",)),
        cwd="/repo",
    )

    assert isinstance(result, PermissionAskDecision)
    assert isinstance(result.decision_reason, SafetyCheckPermissionDecisionReason)


def test_suspicious_windows_patterns_match_reference_subset() -> None:
    assert has_suspicious_windows_path_pattern("//?/C:/repo/file.txt")
    assert has_suspicious_windows_path_pattern("//./C:/repo/file.txt")
    assert has_suspicious_windows_path_pattern("/repo/.git.")
    assert has_suspicious_windows_path_pattern("/repo/settings.json.PRN")
    assert has_suspicious_windows_path_pattern("/repo/.../file.txt")
    assert not has_suspicious_windows_path_pattern("/repo/[...]slug/page.tsx")


def test_explicit_allow_rules_match_original_path_not_resolved_target(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    target = outside / "allowed.txt"
    target.write_text("x")
    link = cwd / "link.txt"
    _symlink(target, link)

    result = check_read_permission_for_path(
        str(link),
        _permission_context(allow=(f"{FILE_READ_TOOL_NAME}({outside}/**)",)),
        cwd=str(cwd),
    )

    assert isinstance(result, PermissionAskDecision)
    assert isinstance(result.decision_reason, WorkingDirPermissionDecisionReason)


def test_permission_suggestions_resolve_original_directory_not_file_target(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "repo"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("x")
    link = cwd / "link.txt"
    _symlink(target, link)

    result = check_read_permission_for_path(
        str(link),
        ToolPermissionContext(),
        cwd=str(cwd),
    )

    assert isinstance(result, PermissionAskDecision)
    suggestion = result.suggestions[0]
    assert isinstance(suggestion, AddPermissionRules)
    rule_contents = tuple(rule.rule_content for rule in suggestion.rules)
    assert f"{outside}/**" not in rule_contents
    assert f"{cwd}/**" in rule_contents
