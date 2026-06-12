from __future__ import annotations

from pathlib import Path

from raygent_harness.skills.loader import (
    deduplicate_loaded_skills,
    load_skill_file,
    load_skills_from_dir,
    merge_skills_prefer_deepest,
    parse_markdown_frontmatter,
)


def _write_skill(base: Path, name: str, content: str) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def test_parse_skill_frontmatter_keeps_reference_metadata(tmp_path: Path) -> None:
    skill_file = _write_skill(
        tmp_path,
        "reviewer",
        """---
name: Review Bot
description: Review changed files
allowed-tools: Read, Grep, Bash(git diff)
argument-hint: FILES
arguments: files, focus
when_to_use: when reviewing code
version: 1.2.3
model: inherit
disable-model-invocation: true
user-invocable: false
context: fork
agent: general-purpose
effort: high
shell: powershell
paths: src/*.{py,md}, docs/**
hooks:
  PreToolUse:
    - matcher: Bash
      hooks:
        - type: command
          command: echo checking
---
# Reviewer
Use $ARGUMENTS with ${CLAUDE_SKILL_DIR}.
""",
    )

    loaded = load_skill_file(skill_file)

    assert loaded is not None
    skill = loaded.skill
    assert skill.name == "reviewer"
    assert skill.user_facing_name() == "Review Bot"
    assert skill.description == "Review changed files"
    assert skill.allowed_tools == ("Read", "Grep", "Bash(git diff)")
    assert skill.argument_hint == "FILES"
    assert skill.argument_names == ("files", "focus")
    assert skill.when_to_use == "when reviewing code"
    assert skill.version == "1.2.3"
    assert skill.model is None
    assert skill.disable_model_invocation is True
    assert skill.user_invocable is False
    assert skill.is_hidden is True
    assert skill.context == "fork"
    assert skill.agent == "general-purpose"
    assert skill.effort == "high"
    assert skill.shell == "powershell"
    assert skill.paths == ("src/*.py", "src/*.md", "docs")
    assert skill.hooks is not None
    assert "PreToolUse" in skill.hooks
    assert skill.skill_root is not None
    assert (
        skill.render_prompt(args="a.py", session_id="session-1")
        == f"Base directory for this skill: {skill.skill_root}\n\n"
        "# Reviewer\n"
        f"Use a.py with {skill.skill_root.as_posix()}.\n"
    )


def test_parse_markdown_frontmatter_fails_soft_on_invalid_block() -> None:
    parsed = parse_markdown_frontmatter("---\n: broken\n---\nbody")

    assert parsed.frontmatter == {}
    assert parsed.content == "body"


def test_invalid_hooks_are_dropped_fail_soft(tmp_path: Path) -> None:
    skill_file = _write_skill(
        tmp_path,
        "bad-hooks",
        """---
hooks:
  PreToolUse:
    - matcher: Bash
      hooks:
        - type: command
---
Body
""",
    )

    loaded = load_skill_file(skill_file)

    assert loaded is not None
    assert loaded.skill.hooks is None


def test_filesystem_skill_context_inline_is_ignored_like_reference(tmp_path: Path) -> None:
    skill_file = _write_skill(
        tmp_path,
        "inline-context",
        "---\ncontext: inline\n---\nBody\n",
    )

    loaded = load_skill_file(skill_file)

    assert loaded is not None
    assert loaded.skill.context is None


def test_load_skills_from_dir_uses_directory_skill_format_only(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", "# Alpha")
    (tmp_path / "loose.md").write_text("# Loose", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    loaded = load_skills_from_dir(tmp_path)

    assert [entry.skill.name for entry in loaded] == ["alpha"]
    assert loaded[0].skill.description == "Alpha"


def test_deduplicate_loaded_skills_uses_canonical_file_identity(tmp_path: Path) -> None:
    source = _write_skill(tmp_path / "source", "alpha", "# Alpha")
    symlink_dir = tmp_path / "link"
    symlink_dir.mkdir()
    (symlink_dir / "alpha").symlink_to(source.parent, target_is_directory=True)

    loaded = [
        *load_skills_from_dir(tmp_path / "source"),
        *load_skills_from_dir(symlink_dir),
    ]

    deduped = deduplicate_loaded_skills(loaded)

    assert len(loaded) == 2
    assert len(deduped) == 1
    assert deduped[0].skill.name == "alpha"


def test_merge_skills_prefer_deepest_overrides_same_name(tmp_path: Path) -> None:
    shallow = tmp_path / "project" / ".claude" / "skills"
    deep = tmp_path / "project" / "pkg" / ".claude" / "skills"
    _write_skill(shallow, "lint", "---\ndescription: shallow\n---\nshallow")
    _write_skill(deep, "lint", "---\ndescription: deep\n---\ndeep")
    _write_skill(shallow, "format", "---\ndescription: format\n---\nfmt")

    merged = merge_skills_prefer_deepest((deep, shallow))

    by_name = {skill.name: skill for skill in merged}
    assert by_name["lint"].description == "deep"
    assert by_name["format"].description == "format"
