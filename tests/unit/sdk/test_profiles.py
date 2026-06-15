from __future__ import annotations

from pathlib import Path

import pytest

from raygent_harness.sdk import (
    RaygentPresetCompatibilityError,
    RaygentPresetOptions,
    describe_raygent_preset,
    list_raygent_presets,
    resolve_raygent_preset,
)
from raygent_harness.services.transcript import JsonlTranscriptStore

REQUIRED_PRESETS = (
    "minimal",
    "chat",
    "embedded_app",
    "project_reader",
    "code_review",
    "repo_maintainer",
    "research_agent",
    "memory_agent",
    "long_running_task",
    "full_developer",
)


def test_lists_required_presets_in_documentation_order() -> None:
    assert list_raygent_presets() == REQUIRED_PRESETS


@pytest.mark.parametrize("preset", REQUIRED_PRESETS)
def test_required_presets_are_describable(preset: str) -> None:
    description = describe_raygent_preset(preset)

    assert description.name == preset
    assert description.summary
    assert description.tool_policy
    assert description.context_policy


def test_minimal_resolves_to_no_tools_context_or_memory() -> None:
    resolved = resolve_raygent_preset("minimal")

    assert resolved.factory_kwargs == {"tools": "none", "context": "none"}
    assert resolved.tool_names == ()
    assert resolved.context_profile == "none"
    assert "memory" not in resolved.enabled_capabilities


@pytest.mark.parametrize("preset", ("project_reader", "code_review", "research_agent"))
def test_readonly_presets_expose_only_read_search_tools(preset: str) -> None:
    resolved = resolve_raygent_preset(preset)

    assert resolved.tool_names == ("Read", "Glob", "Grep", "ToolSearch")
    assert "Write" not in resolved.tool_names
    assert "Edit" not in resolved.tool_names
    assert "NotebookEdit" not in resolved.tool_names
    assert "Bash" not in resolved.tool_names
    assert resolved.factory_kwargs["context"] == "project"


def test_transcript_presets_create_jsonl_transcript_store(tmp_path: Path) -> None:
    resolved = resolve_raygent_preset(
        "chat",
        options=RaygentPresetOptions(project_root=tmp_path),
    )

    transcript_store = resolved.factory_kwargs["transcript_store"]
    assert isinstance(transcript_store, JsonlTranscriptStore)
    assert transcript_store.base_dir == (tmp_path / ".raygent" / "transcripts").resolve()


def test_embedded_app_resolves_to_transcripts_and_observability(
    tmp_path: Path,
) -> None:
    resolved = resolve_raygent_preset(
        "embedded_app",
        options=RaygentPresetOptions(project_root=tmp_path),
    )

    assert resolved.factory_kwargs["tools"] == "none"
    assert resolved.factory_kwargs["context"] == "none"
    assert isinstance(resolved.factory_kwargs["transcript_store"], JsonlTranscriptStore)
    assert "observability" in resolved.enabled_capabilities


def test_repo_maintainer_requires_mutation_acknowledgement() -> None:
    with pytest.raises(RaygentPresetCompatibilityError) as exc_info:
        resolve_raygent_preset("repo_maintainer")

    assert exc_info.value.issues == (
        "repo_maintainer requires allow_filesystem_mutation=True",
    )


def test_repo_maintainer_resolves_to_project_tools_after_acknowledgement() -> None:
    resolved = resolve_raygent_preset(
        "repo_maintainer",
        options=RaygentPresetOptions(allow_filesystem_mutation=True),
    )

    assert resolved.factory_kwargs["tools"] == "project"
    assert resolved.factory_kwargs["context"] == "project"
    assert "Write" in resolved.tool_names
    assert "Edit" in resolved.tool_names
    assert "Bash" not in resolved.tool_names
    assert resolved.requires_explicit_permission_options is True


def test_long_running_task_resolves_to_durable_persistence(tmp_path: Path) -> None:
    resolved = resolve_raygent_preset(
        "long_running_task",
        options=RaygentPresetOptions(
            project_root=tmp_path,
            task_output_dir=tmp_path / "tasks",
        ),
    )

    assert resolved.factory_kwargs["tools"] == "none"
    assert resolved.factory_kwargs["context"] == "project"
    assert isinstance(resolved.factory_kwargs["transcript_store"], JsonlTranscriptStore)
    assert resolved.factory_kwargs["task_output_dir"] == tmp_path / "tasks"
    assert "task_output" in resolved.enabled_capabilities
    assert "compaction" in resolved.enabled_capabilities
    assert "recovery" in resolved.enabled_capabilities


def test_memory_agent_requires_caller_memory_options() -> None:
    resolved = resolve_raygent_preset("memory_agent")

    assert "memory" in resolved.enabled_capabilities
    assert "memory_options" in resolved.required_options
    assert resolved.factory_kwargs["context"] == "environment"


def test_full_developer_fails_without_explicit_safety_acknowledgement() -> None:
    with pytest.raises(RaygentPresetCompatibilityError) as exc_info:
        resolve_raygent_preset("full_developer")

    assert "allow_full_developer=True" in str(exc_info.value)
    assert "allow_shell=True" in str(exc_info.value)


def test_full_developer_requires_explicit_permission_options_after_acknowledgement() -> None:
    resolved = resolve_raygent_preset(
        "full_developer",
        options=RaygentPresetOptions(
            allow_full_developer=True,
            allow_filesystem_mutation=True,
            allow_shell=True,
            allow_agents=True,
            allow_mcp=True,
            allow_worktree=True,
        ),
    )

    assert resolved.factory_kwargs["tools"] == "project"
    assert resolved.enable_bash is True
    assert resolved.requires_explicit_permission_options is True
    assert "permission_options or explicit permission handler/context" in (
        resolved.required_options
    )


def test_incompatible_overlay_combinations_raise_structured_error() -> None:
    with pytest.raises(RaygentPresetCompatibilityError) as exc_info:
        resolve_raygent_preset(
            "minimal",
            overlays=("readonly_tools", "file_tools"),
            options=RaygentPresetOptions(allow_filesystem_mutation=True),
        )

    assert exc_info.value.issues == (
        "readonly_tools and file_tools overlays are mutually exclusive",
    )


def test_mutating_overlays_require_explicit_acknowledgement() -> None:
    with pytest.raises(RaygentPresetCompatibilityError) as exc_info:
        resolve_raygent_preset("minimal", overlays=("file_tools",))

    assert exc_info.value.issues == ("file_tools requires allow_filesystem_mutation=True",)


def test_file_tools_overlay_requires_explicit_permission_options_after_ack() -> None:
    resolved = resolve_raygent_preset(
        "minimal",
        overlays=("file_tools",),
        options=RaygentPresetOptions(allow_filesystem_mutation=True),
    )

    assert resolved.factory_kwargs["tools"] == "file"
    assert resolved.requires_explicit_permission_options is True


def test_goals_overlay_is_readiness_metadata_only() -> None:
    resolved = resolve_raygent_preset("long_running_task", overlays=("goals",))

    assert "goals_ready" in resolved.enabled_capabilities
    assert "goal runtime installer" in resolved.required_options
    assert "Goal runtime is not a product /goal command parser." in resolved.safety_notes
    assert "goal_runtime_options" not in resolved.factory_kwargs
