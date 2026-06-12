from __future__ import annotations

from pathlib import Path

from raygent_harness.skills.bundled import (
    BundledSkillRegistry,
    extract_bundled_skill_files,
    resolve_bundled_skill_file_path,
)
from raygent_harness.skills.models import BundledSkillDefinition


def test_bundled_registry_returns_skill_metadata_and_prompt_prefix(tmp_path: Path) -> None:
    registry = BundledSkillRegistry()
    registry.register(
        BundledSkillDefinition(
            name="review",
            description="review things",
            prompt="Review $ARGUMENTS",
            aliases=("rev",),
            allowed_tools=("Read", "Grep"),
            files={"refs/guide.md": "# Guide"},
        )
    )

    skills = registry.get_skills(tmp_path)
    prompt = registry.render_prompt("review", args="src", extraction_root=tmp_path)

    assert len(skills) == 1
    assert skills[0].name == "review"
    assert skills[0].aliases == ("rev",)
    assert skills[0].allowed_tools == ("Read", "Grep")
    assert skills[0].skill_root == tmp_path / "review"
    assert (tmp_path / "review" / "refs" / "guide.md").read_text(
        encoding="utf-8"
    ) == "# Guide"
    assert prompt == f"Base directory for this skill: {tmp_path / 'review'}\n\nReview src"

    # Registry memoization avoids rewriting files with O_EXCL on later renders.
    assert (
        registry.render_prompt("review", args="again", extraction_root=tmp_path)
        == f"Base directory for this skill: {tmp_path / 'review'}\n\nReview again"
    )


def test_extract_bundled_skill_files_rejects_traversal(tmp_path: Path) -> None:
    assert extract_bundled_skill_files(
        "bad",
        {"../escape.md": "nope"},
        tmp_path,
    ) is None
    assert extract_bundled_skill_files(
        "bad",
        {"..\\escape.md": "nope"},
        tmp_path,
    ) is None
    assert not (tmp_path / "escape.md").exists()


def test_resolve_bundled_skill_file_path_normalizes_forward_slash_paths(
    tmp_path: Path,
) -> None:
    assert (
        resolve_bundled_skill_file_path(tmp_path / "skill", "refs/guide.md")
        == tmp_path / "skill" / "refs" / "guide.md"
    )
