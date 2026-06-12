from __future__ import annotations

from raygent_harness.memdir.memory_types import (
    MEMORY_FRONTMATTER_EXAMPLE,
    MEMORY_TYPES,
    TRUSTING_RECALL_SECTION,
    TYPES_SECTION_INDIVIDUAL,
    WHAT_NOT_TO_SAVE_SECTION,
    WHEN_TO_ACCESS_SECTION,
    parse_memory_type,
)


def test_parse_memory_type_accepts_only_closed_taxonomy() -> None:
    assert MEMORY_TYPES == ("user", "feedback", "project", "reference")
    assert parse_memory_type("user") == "user"
    assert parse_memory_type("feedback") == "feedback"
    assert parse_memory_type("project") == "project"
    assert parse_memory_type("reference") == "reference"
    assert parse_memory_type("other") is None
    assert parse_memory_type(None) is None


def test_prompt_sections_include_reference_behavioral_cues() -> None:
    type_text = "\n".join(TYPES_SECTION_INDIVIDUAL)
    assert "<name>user</name>" in type_text
    assert "<name>feedback</name>" in type_text
    assert "<name>project</name>" in type_text
    assert "<name>reference</name>" in type_text
    assert "Always convert relative dates" in type_text

    not_save_text = "\n".join(WHAT_NOT_TO_SAVE_SECTION)
    assert "What NOT to save" in not_save_text
    assert "Code patterns" in not_save_text
    assert "git log" in not_save_text

    when_text = "\n".join(WHEN_TO_ACCESS_SECTION)
    assert "MUST access memory" in when_text
    assert "ignore" in when_text
    assert "stale" in when_text

    trust_text = "\n".join(TRUSTING_RECALL_SECTION)
    assert "Before recommending from memory" in trust_text
    assert "check the file exists" in trust_text
    assert "grep for it" in trust_text


def test_frontmatter_example_lists_all_memory_types() -> None:
    text = "\n".join(MEMORY_FRONTMATTER_EXAMPLE)
    assert "description:" in text
    assert "type: {{user, feedback, project, reference}}" in text
