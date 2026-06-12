from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from raygent_harness.context_providers import (
    ConditionalInstructionRule,
    ProjectInstructionConfig,
    ProjectInstructionFile,
    ProjectInstructionsContextProvider,
    ReadAdjacentProjectInstructionsContextProvider,
    discover_conditional_instruction_rules,
    discover_project_instruction_files,
    instruction_rule_matches_path,
    resolve_project_instructions_for_target_path,
)
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import message_param_from_api_message
from raygent_harness.core.query_engine import QueryEngine, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, ToolUseContext
from tests.fakes import FakeModelProvider


def _ctx(*, cwd: str | Path = ".", agent_id: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=agent_id,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(cwd),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _file_names(files: tuple[ProjectInstructionFile, ...]) -> list[str]:
    return [file.path.name for file in files]


def _file_contents(files: tuple[ProjectInstructionFile, ...]) -> list[str]:
    return [file.content for file in files]


def test_layered_ancestors_discovers_root_to_cwd_project_then_local(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg" / "app"
    _write(root / "AGENTS.md", "root agents")
    _write(root / "CLAUDE.md", "root claude")
    _write(root / "AGENTS.local.md", "root local")
    _write(root / "pkg" / "AGENTS.md", "pkg agents")
    _write(cwd / "CLAUDE.md", "app claude")
    _write(cwd / "CLAUDE.local.md", "app local")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(cwd=cwd, workspace_root=root),
    )

    assert [file.path for file in files] == [
        root / "AGENTS.md",
        root / "CLAUDE.md",
        root / "AGENTS.local.md",
        root / "pkg" / "AGENTS.md",
        cwd / "CLAUDE.md",
        cwd / "CLAUDE.local.md",
    ]
    assert [file.kind for file in files] == [
        "project",
        "project",
        "local",
        "project",
        "project",
        "local",
    ]


def test_nearest_first_match_uses_nearest_file_from_first_matching_family(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg" / "app"
    _write(root / "AGENTS.md", "root agents")
    _write(root / "pkg" / "AGENTS.md", "pkg agents")
    _write(cwd / "CLAUDE.md", "app claude")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=cwd,
            workspace_root=root,
            discovery_mode="nearest_first_match",
        ),
    )

    assert [file.path for file in files] == [
        root / "pkg" / "AGENTS.md",
        root / "AGENTS.md",
    ]
    assert _file_contents(files) == ["pkg agents", "root agents"]


def test_nearest_first_match_falls_back_to_next_filename_family(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg" / "app"
    _write(root / "CLAUDE.md", "root claude")
    _write(cwd / "CLAUDE.md", "app claude")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=cwd,
            workspace_root=root,
            discovery_mode="nearest_first_match",
        ),
    )

    assert [file.path for file in files] == [
        cwd / "CLAUDE.md",
        root / "CLAUDE.md",
    ]
    assert _file_contents(files) == ["app claude", "root claude"]


def test_configurable_filenames_can_disable_compatibility_claude_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "agents")
    _write(root / "CLAUDE.md", "claude")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
    )

    assert _file_names(files) == ["AGENTS.md"]


def test_user_paths_load_first_and_additional_dirs_load_after_project(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    user_file = _write(tmp_path / "user" / "AGENTS.md", "user")
    _write(root / "AGENTS.md", "project")
    additional = tmp_path / "extra"
    _write(additional / "AGENTS.md", "additional")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            user_instruction_paths=(user_file,),
            additional_dirs=(additional,),
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
    )

    assert [file.kind for file in files] == ["user", "project", "additional"]
    assert _file_contents(files) == ["user", "project", "additional"]


def test_discovery_deduplicates_missing_empty_and_duplicate_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    agents = _write(root / "AGENTS.md", "agents")
    _write(root / "CLAUDE.md", "   ")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            user_instruction_paths=(agents, root / "missing.md"),
            project_filenames=("AGENTS.md", "CLAUDE.md"),
            local_filenames=(),
        ),
    )

    assert [file.path for file in files] == [agents]
    assert _file_contents(files) == ["agents"]


def test_file_and_total_caps_truncate_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "abcdef")
    _write(root / "CLAUDE.md", "ghijkl")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            max_file_chars=4,
            max_total_chars=6,
            local_filenames=(),
        ),
    )

    assert _file_contents(files) == ["abcd", "gh"]
    assert [file.truncated for file in files] == [True, True]


def test_includes_are_opt_in_and_parent_before_child(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "root\n@./docs/extra.md")
    extra = _write(root / "docs" / "extra.md", "extra\n@./nested.txt")
    nested = _write(root / "docs" / "nested.txt", "nested")

    without_includes = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            project_rule_dirs=(),
        )
    )
    with_includes = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            project_rule_dirs=(),
            allow_includes=True,
        )
    )

    assert [file.path for file in without_includes] == [root / "AGENTS.md"]
    assert [file.path for file in with_includes] == [
        root / "AGENTS.md",
        extra,
        nested,
    ]
    assert [file.parent for file in with_includes] == [
        None,
        root / "AGENTS.md",
        extra,
    ]


def test_includes_are_bounded_and_restricted_to_safe_local_text_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside.md"
    _write(outside, "outside")
    _write(root / "AGENTS.md", "\n".join([
        "@./inside.md",
        "@./binary.png",
        f"@{outside}",
        "```",
        "@./ignored.md",
        "```",
        "not standalone @./ignored_inline.md",
    ]))
    inside = _write(root / "inside.md", "inside")
    _write(root / "binary.png", "binary")
    _write(root / "ignored.md", "ignored")
    _write(root / "ignored_inline.md", "ignored inline")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            project_rule_dirs=(),
            allow_includes=True,
        )
    )

    assert [file.path for file in files] == [root / "AGENTS.md", inside]


def test_include_depth_and_cycle_protection_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root_file = _write(root / "AGENTS.md", "@./a.md")
    a_file = _write(root / "a.md", "@./b.md")
    b_file = _write(root / "b.md", "@./AGENTS.md")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            project_rule_dirs=(),
            allow_includes=True,
            max_include_depth=2,
        )
    )

    assert [file.path for file in files] == [root_file, a_file]
    assert b_file not in [file.path for file in files]


def test_rule_dirs_load_non_conditional_markdown_recursively(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "project")
    first_rule = _write(root / ".claude" / "rules" / "first.md", "first rule")
    nested_rule = _write(root / ".claude" / "rules" / "nested" / "second.md", "second")
    _write(
        root / ".claude" / "rules" / "conditional.md",
        "---\npaths: src/**\n---\nconditional",
    )
    _write(root / ".claude" / "rules" / "skip.txt", "skip")
    _write(root / "AGENTS.local.md", "local")

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=("AGENTS.local.md",),
        )
    )

    assert [file.path for file in files] == [
        root / "AGENTS.md",
        nested_rule,
        first_rule,
        root / "AGENTS.local.md",
    ]
    assert [file.kind for file in files] == ["project", "rule", "rule", "local"]


def test_rule_files_strip_non_conditional_frontmatter(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / ".claude" / "rules" / "general.md",
        "---\ndescription: general\n---\nRule body",
    )

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert len(files) == 1
    assert files[0].kind == "rule"
    assert files[0].content == "Rule body"


def test_rule_include_graph_filters_conditional_files_per_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    shared = _write(root / ".claude" / "rules" / "b_shared.md", "shared")
    general = _write(
        root / ".claude" / "rules" / "c_general.md",
        "general\n@./d_conditional.md",
    )
    _write(
        root / ".claude" / "rules" / "a_conditional.md",
        "---\npaths: src/**\n---\n@./b_shared.md",
    )
    _write(
        root / ".claude" / "rules" / "d_conditional.md",
        "---\npaths: src/**\n---\nconditional",
    )

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
            allow_includes=True,
        )
    )

    assert [file.path for file in files] == [shared, general]
    assert _file_contents(files) == ["shared", "general\n@./d_conditional.md"]


def test_rule_paths_match_all_frontmatter_loads_as_non_conditional(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    rule = _write(
        root / ".claude" / "rules" / "general.md",
        "---\npaths: **\n---\nApplies everywhere",
    )

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert [file.path for file in files] == [rule]
    assert files[0].content == "Applies everywhere"


def test_rule_yaml_list_paths_are_treated_as_conditional(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    general_rule = _write(root / ".claude" / "rules" / "general.md", "General")
    _write(
        root / ".claude" / "rules" / "conditional.md",
        "---\npaths:\n  - src/**\n---\nConditional",
    )
    match_all_rule = _write(
        root / ".claude" / "rules" / "all.md",
        "---\npaths:\n  - '**'\n---\nApplies everywhere",
    )

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert [file.path for file in files] == [match_all_rule, general_rule]
    assert _file_contents(files) == ["Applies everywhere", "General"]


def test_conditional_rule_discovery_parses_brace_and_yaml_path_patterns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    rule = _write(
        root / ".claude" / "rules" / "typed.md",
        "---\npaths: src/*.{py,md}, docs/**\n---\nTyped rule",
    )
    yaml_rule = _write(
        root / ".claude" / "rules" / "yaml.md",
        "---\npaths:\n  - tests/**\n  - \"lib/*.py\"\n---\nYaml rule",
    )
    _write(
        root / ".claude" / "rules" / "all.md",
        "---\npaths: **\n---\nAlways",
    )

    rules = discover_conditional_instruction_rules(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert [rule_.path for rule_ in rules] == [rule, yaml_rule]
    assert rules[0].patterns == ("src/*.py", "src/*.md", "docs")
    assert rules[1].patterns == ("tests", "lib/*.py")


def test_instruction_rule_matches_target_relative_to_rule_base(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    rule = ConditionalInstructionRule(
        path=root / ".claude" / "rules" / "python.md",
        base_dir=root,
        patterns=("src/*.py", "docs"),
        content="rule",
    )

    assert instruction_rule_matches_path(rule, root / "src" / "main.py")
    assert not instruction_rule_matches_path(rule, root / "src" / "pkg" / "main.py")
    assert instruction_rule_matches_path(rule, root / "docs" / "guide.md")
    assert instruction_rule_matches_path(rule, root / "src" / "docs" / "guide.md")
    assert not instruction_rule_matches_path(rule, tmp_path / "outside.py")


def test_instruction_rule_matches_gitignore_style_common_globs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    any_python = ConditionalInstructionRule(
        path=root / ".claude" / "rules" / "any-python.md",
        base_dir=root,
        patterns=("*.py",),
        content="rule",
    )
    globstar = ConditionalInstructionRule(
        path=root / ".claude" / "rules" / "globstar.md",
        base_dir=root,
        patterns=("**/*.md",),
        content="rule",
    )
    bracket = ConditionalInstructionRule(
        path=root / ".claude" / "rules" / "bracket.md",
        base_dir=root,
        patterns=("src/[ab].txt",),
        content="rule",
    )

    assert instruction_rule_matches_path(any_python, root / "main.py")
    assert instruction_rule_matches_path(any_python, root / "src" / "pkg" / "main.py")
    assert instruction_rule_matches_path(globstar, root / "README.md")
    assert instruction_rule_matches_path(globstar, root / "docs" / "guide.md")
    assert instruction_rule_matches_path(bracket, root / "src" / "a.txt")
    assert not instruction_rule_matches_path(bracket, root / "src" / "c.txt")


def test_target_path_resolver_loads_nested_nearby_and_matching_rules(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg"
    target = cwd / "feature" / "src" / "main.py"
    target.parent.mkdir(parents=True)
    _write(root / "AGENTS.md", "root project")
    root_rule = _write(
        root / ".claude" / "rules" / "root-python.md",
        "---\npaths: pkg/feature/src/*.py\n---\nroot conditional",
    )
    _write(
        root / ".claude" / "rules" / "root-js.md",
        "---\npaths: pkg/feature/src/*.js\n---\nroot nonmatch",
    )
    feature_agents = _write(cwd / "feature" / "AGENTS.md", "feature project")
    feature_local = _write(cwd / "feature" / "AGENTS.local.md", "feature local")
    feature_rule = _write(
        cwd / "feature" / ".claude" / "rules" / "feature-src.md",
        "---\npaths: src/**\n---\nfeature conditional",
    )
    feature_general = _write(
        cwd / "feature" / ".claude" / "rules" / "general.md",
        "feature general",
    )
    src_local = _write(target.parent / "AGENTS.local.md", "src local")

    files = resolve_project_instructions_for_target_path(
        target,
        ProjectInstructionConfig(
            cwd=cwd,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=("AGENTS.local.md",),
        ),
        already_loaded_paths=(root / "AGENTS.md",),
    )

    assert [file.path for file in files] == [
        feature_agents,
        feature_local,
        feature_general,
        feature_rule,
        src_local,
        root_rule,
    ]
    assert _file_contents(files) == [
        "feature project",
        "feature local",
        "feature general",
        "feature conditional",
        "src local",
        "root conditional",
    ]


def test_target_path_resolver_rejects_outside_workspace_and_skips_loaded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg"
    target = cwd / "src" / "main.py"
    target.parent.mkdir(parents=True)
    loaded_rule = _write(
        root / ".claude" / "rules" / "loaded.md",
        "---\npaths: pkg/src/*.py\n---\nloaded",
    )
    fresh_rule = _write(
        root / ".claude" / "rules" / "fresh.md",
        "---\npaths: pkg/src/*.py\n---\nfresh",
    )

    files = resolve_project_instructions_for_target_path(
        target,
        ProjectInstructionConfig(
            cwd=cwd,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        ),
        already_loaded_paths=(loaded_rule,),
    )
    outside = resolve_project_instructions_for_target_path(
        tmp_path / "outside.py",
        ProjectInstructionConfig(
            cwd=cwd,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        ),
    )

    assert [file.path for file in files] == [fresh_rule]
    assert outside == ()


def test_target_path_resolver_uses_cwd_override_for_relative_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    cwd = root / "pkg"
    target = cwd / "feature" / "main.py"
    target.parent.mkdir(parents=True)
    feature_agents = _write(cwd / "feature" / "AGENTS.md", "feature project")

    files = resolve_project_instructions_for_target_path(
        "feature/main.py",
        ProjectInstructionConfig(
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
        cwd=cwd,
    )
    skipped = resolve_project_instructions_for_target_path(
        "feature/main.py",
        ProjectInstructionConfig(
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        ),
        cwd=cwd,
        already_loaded_paths=("feature/AGENTS.md",),
    )

    assert [file.path for file in files] == [feature_agents]
    assert skipped == ()


@pytest.mark.asyncio
async def test_read_adjacent_provider_skips_turn_entry_and_attached_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    target = root / "pkg" / "feature" / "main.py"
    target.parent.mkdir(parents=True)
    _write(root / "AGENTS.md", "root project")
    feature_agents = _write(root / "pkg" / "feature" / "AGENTS.md", "feature project")
    feature_rule = _write(
        root / "pkg" / "feature" / ".claude" / "rules" / "feature.md",
        "---\npaths: \"*.py\"\n---\nfeature conditional",
    )
    root_rule = _write(
        root / ".claude" / "rules" / "root.md",
        "---\npaths: pkg/feature/*.py\n---\nroot conditional",
    )

    provider = ReadAdjacentProjectInstructionsContextProvider(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
        )
    )

    fragments = await provider(
        QueryConfig(model="model-1"),
        _ctx(cwd=root),
        (str(target),),
        (str(feature_rule),),
    )

    assert [fragment.source for fragment in fragments] == [
        str(feature_agents),
        str(root_rule),
    ]
    assert "feature project" in fragments[0].content
    assert all("root project" not in fragment.content for fragment in fragments)
    assert all("feature conditional" not in fragment.content for fragment in fragments)


@pytest.mark.asyncio
async def test_read_adjacent_provider_respects_agent_scope(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "pkg" / "main.py"
    target.parent.mkdir(parents=True)
    _write(root / "pkg" / "AGENTS.md", "pkg project")
    provider = ReadAdjacentProjectInstructionsContextProvider(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            agent_scope="main",
        )
    )

    fragments = await provider(
        QueryConfig(model="model-1"),
        _ctx(cwd=root, agent_id="child"),
        (str(target),),
        (),
    )

    assert fragments == ()


def test_nested_frontmatter_paths_key_is_not_conditional(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    nested = _write(
        root / ".claude" / "rules" / "nested.md",
        "---\nmetadata:\n  paths: src/**\n---\nGeneral rule",
    )

    files = discover_project_instruction_files(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )
    rules = discover_conditional_instruction_rules(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert [file.path for file in files] == [nested]
    assert rules == ()


def test_singular_path_frontmatter_remains_compatibility_extension(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    singular_rule = _write(
        root / ".claude" / "rules" / "singular.md",
        "---\npath: src/*.py\n---\ncompat",
    )

    rules = discover_conditional_instruction_rules(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=(),
            local_filenames=(),
        )
    )

    assert [rule.path for rule in rules] == [singular_rule]
    assert rules[0].patterns == ("src/*.py",)


def test_rule_rendering_matches_project_instruction_description(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / ".claude" / "rules" / "general.md", "Rule body")
    provider = ProjectInstructionsContextProvider(
        ProjectInstructionConfig(cwd=root, workspace_root=root)
    )

    fragments = asyncio.run(provider(QueryConfig(model="model-1"), _ctx(cwd=root)))

    assert len(fragments) == 1
    assert "project instructions, checked into the codebase" in fragments[0].content
    assert "project rule instructions" not in fragments[0].content
    assert "Included from:" not in fragments[0].content


@pytest.mark.asyncio
async def test_project_instruction_provider_renders_user_context_fragments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "Prefer deterministic tests.")
    provider = ProjectInstructionsContextProvider(
        ProjectInstructionConfig(
            cwd=root,
            workspace_root=root,
            project_filenames=("AGENTS.md",),
            local_filenames=(),
            priority=42,
        )
    )

    fragments = await provider(QueryConfig(model="model-1"), _ctx(cwd=root))

    assert len(fragments) == 1
    assert fragments[0].channel == "user_context"
    assert fragments[0].source == str(root / "AGENTS.md")
    assert fragments[0].priority == 42
    assert fragments[0].render_mode == "instructions"
    assert "Contents of" in fragments[0].content
    assert "project instructions" in fragments[0].content
    assert "Prefer deterministic tests." in fragments[0].content


@pytest.mark.asyncio
async def test_project_instruction_context_reaches_model_without_persistence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "Use the local harness rules.")
    model = FakeModelProvider(responses=({"role": "assistant", "content": "ok"},))
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=model,
        context_providers=(
            ProjectInstructionsContextProvider(
                ProjectInstructionConfig(
                    cwd=root,
                    workspace_root=root,
                    project_filenames=("AGENTS.md",),
                    local_filenames=(),
                )
            ),
        ),
    )
    engine = QueryEngine(
        QueryConfig(model="model-1", session_id="s"),
        deps,
        _ctx(cwd=root),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    request_messages = [
        message_param_from_api_message(message) for message in model.requests[0].messages
    ]
    assert len(request_messages) == 2
    assert "Codebase and user instructions are shown below" in str(
        request_messages[0]["content"]
    )
    assert "MUST follow them exactly as written" in str(request_messages[0]["content"])
    assert "may or may not be relevant" not in str(request_messages[0]["content"])
    assert "Use the local harness rules." in str(request_messages[0]["content"])
    assert request_messages[1] == {"role": "user", "content": "hi"}
    assert engine._messages == [  # pyright: ignore[reportPrivateUsage]
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
