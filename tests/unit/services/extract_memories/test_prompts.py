from __future__ import annotations

from raygent_harness.services.extract_memories import build_extract_auto_only_prompt


def test_build_extract_auto_only_prompt_includes_manifest_and_tool_limits() -> None:
    prompt = build_extract_auto_only_prompt(
        4,
        "- [project] project.md (2026-05-08T00:00:00.000Z): Project context",
    )

    assert "memory extraction subagent" in prompt
    assert "most recent ~4 messages" in prompt
    assert "## Existing memory files" in prompt
    assert "project.md" in prompt
    assert "read-only Bash" in prompt
    assert "MCP, Agent, write-capable Bash" in prompt
    assert "Saving a memory is a two-step process" in prompt
    assert "MEMORY.md" in prompt


def test_build_extract_auto_only_prompt_skip_index_removes_memory_md_step() -> None:
    prompt = build_extract_auto_only_prompt(1, "", skip_index=True)

    assert "Write each memory to its own file" in prompt
    assert "Saving a memory is a two-step process" not in prompt
    assert "add a pointer" not in prompt


def test_build_extract_auto_only_prompt_renders_actual_restricted_tool_set() -> None:
    prompt = build_extract_auto_only_prompt(
        2,
        "",
        allowed_tool_names=("Read", "Write", "Edit"),
    )

    assert (
        "Available tools: Read and Write/Edit for paths inside the memory directory only"
        in prompt
    )
    assert "Grep" not in prompt
    assert "Glob" not in prompt
    assert "Bash" not in prompt
    assert "MCP, Agent, Task, Skill, remote, and team tools" in prompt
