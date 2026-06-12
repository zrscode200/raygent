"""Preset resolver for product-oriented Raygent SDK construction.

Profiles are documented compositions over the existing SDK factory. They do not
create provider clients, install hidden hosted services, or bypass permission
policy. The resolver returns ordinary factory keyword arguments plus metadata so
embedders can inspect what a preset enables before constructing a session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from raygent_harness.core.tool import Tool
from raygent_harness.services.file_media import PdfDocumentService
from raygent_harness.services.transcript import (
    DEFAULT_TRANSCRIPT_DIR,
    JsonlTranscriptStore,
)
from raygent_harness.tools.discovery_tools import create_discovery_tooling_runtime
from raygent_harness.tools.file_read_tool import build_file_read_tool
from raygent_harness.tools.search_backend import SearchBackend
from raygent_harness.tools.tool_search_tool import build_tool_search_tool

RaygentPreset = Literal[
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
]

RaygentOverlay = Literal[
    "transcripts",
    "observability",
    "memory",
    "goals",
    "compaction",
    "recovery",
    "task_output",
    "readonly_tools",
    "file_tools",
    "bash",
    "agents",
    "coordinator",
    "mcp",
    "worktree",
]

_PRESETS: tuple[RaygentPreset, ...] = (
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

_OVERLAYS: tuple[RaygentOverlay, ...] = (
    "transcripts",
    "observability",
    "memory",
    "goals",
    "compaction",
    "recovery",
    "task_output",
    "readonly_tools",
    "file_tools",
    "bash",
    "agents",
    "coordinator",
    "mcp",
    "worktree",
)


@dataclass(frozen=True)
class RaygentPresetOptions:
    """Optional local services and safety acknowledgements for preset resolution."""

    project_root: str | Path | None = None
    transcript_dir: str | Path | None = None
    task_output_dir: str | Path | None = None
    pdf_document_service: PdfDocumentService | None = None
    search_backend: SearchBackend | None = None
    allow_filesystem_mutation: bool = False
    allow_shell: bool = False
    allow_agents: bool = False
    allow_mcp: bool = False
    allow_worktree: bool = False
    allow_full_developer: bool = False


@dataclass(frozen=True)
class RaygentPresetDescription:
    """Inspectable description of one preset's intended construction policy."""

    name: RaygentPreset
    summary: str
    tool_policy: str
    context_policy: str
    capabilities: tuple[str, ...] = ()
    required_options: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RaygentPresetResolution:
    """Resolved factory inputs plus metadata for one preset and overlay set."""

    preset: RaygentPreset
    overlays: tuple[RaygentOverlay, ...] = ()
    factory_kwargs: Mapping[str, object] = field(default_factory=dict[str, object])
    enabled_capabilities: tuple[str, ...] = ()
    required_options: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    context_profile: str = "none"
    enable_bash: bool = False
    requires_explicit_permission_options: bool = False


class RaygentPresetCompatibilityError(ValueError):
    """Raised when a preset/overlay combination is unsafe or unsupported."""

    def __init__(
        self,
        message: str,
        *,
        preset: str,
        overlays: tuple[str, ...],
        issues: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.preset = preset
        self.overlays = overlays
        self.issues = issues


_DESCRIPTIONS: Mapping[RaygentPreset, RaygentPresetDescription] = MappingProxyType(
    {
        "minimal": RaygentPresetDescription(
            name="minimal",
            summary="Model/provider only; no SDK-owned tools, context, memory, or persistence.",
            tool_policy='tools="none"',
            context_policy='context="none"',
            capabilities=("single_turn",),
        ),
        "chat": RaygentPresetDescription(
            name="chat",
            summary="Conversation session with local JSONL transcript support.",
            tool_policy='tools="none"',
            context_policy='context="none"',
            capabilities=("transcripts",),
        ),
        "embedded_app": RaygentPresetDescription(
            name="embedded_app",
            summary=(
                "Application embedding profile with transcripts and observability, "
                "but no filesystem tools."
            ),
            tool_policy='tools="none"',
            context_policy='context="none"',
            capabilities=("transcripts", "observability", "callbacks"),
        ),
        "project_reader": RaygentPresetDescription(
            name="project_reader",
            summary="Project context with read/search tools only.",
            tool_policy="Read, Glob, Grep, ToolSearch",
            context_policy='context="project"',
            capabilities=("readonly_tools", "project_context"),
            safety_notes=("No Write, Edit, NotebookEdit, Bash, or agent tools are installed.",),
        ),
        "code_review": RaygentPresetDescription(
            name="code_review",
            summary="Read/search-oriented repository review profile with mutation disabled.",
            tool_policy="Read, Glob, Grep, ToolSearch",
            context_policy='context="project"',
            capabilities=("readonly_tools", "project_context", "review"),
            safety_notes=("Test execution and shell remain explicit opt-ins.",),
        ),
        "repo_maintainer": RaygentPresetDescription(
            name="repo_maintainer",
            summary="Project workflow profile with persistence and recovery-oriented state.",
            tool_policy='tools="project", Bash still opt-in',
            context_policy='context="project"',
            capabilities=("project_tools", "transcripts", "task_output", "recovery"),
            required_options=(
                "permission_options or explicit permission handler/context",
                "RaygentPresetOptions.allow_filesystem_mutation",
            ),
            safety_notes=(
                "Write/Edit tools require explicit mutation acknowledgement.",
                "Filesystem mutation follows normal Raygent permission policy.",
            ),
        ),
        "research_agent": RaygentPresetDescription(
            name="research_agent",
            summary="Search/MCP-oriented profile with low filesystem authority.",
            tool_policy="Read, Glob, Grep, ToolSearch plus caller-provided catalog/MCP services",
            context_policy='context="project"',
            capabilities=("readonly_tools", "mcp_ready", "research"),
            required_options=("tool_catalog_provider for MCP/web/search services",),
        ),
        "memory_agent": RaygentPresetDescription(
            name="memory_agent",
            summary="Memory-ready profile that requires caller-provided memory services.",
            tool_policy='tools="none"',
            context_policy='context="environment"',
            capabilities=("memory", "transcripts"),
            required_options=("memory_options",),
            safety_notes=("Raygent does not invent memory storage credentials or policy.",),
        ),
        "long_running_task": RaygentPresetDescription(
            name="long_running_task",
            summary="Durable long-session profile with transcripts, task output, and recovery.",
            tool_policy='tools="none"',
            context_policy='context="project"',
            capabilities=("transcripts", "task_output", "compaction", "recovery"),
        ),
        "full_developer": RaygentPresetDescription(
            name="full_developer",
            summary="Broad developer capability bundle with explicit safety acknowledgement.",
            tool_policy='tools="project" with optional Bash enablement',
            context_policy='context="project"',
            capabilities=(
                "project_tools",
                "transcripts",
                "task_output",
                "agents_ready",
                "mcp_ready",
                "worktree_ready",
            ),
            required_options=(
                "permission_options or explicit permission handler/context",
                "RaygentPresetOptions.allow_full_developer",
                "RaygentPresetOptions.allow_filesystem_mutation",
                "RaygentPresetOptions.allow_shell",
                "RaygentPresetOptions.allow_agents",
                "RaygentPresetOptions.allow_mcp",
                "RaygentPresetOptions.allow_worktree",
            ),
            safety_notes=("This preset must not be enabled silently.",),
        ),
    }
)


def list_raygent_presets() -> tuple[RaygentPreset, ...]:
    """Return supported preset names in documentation order."""

    return _PRESETS


def describe_raygent_preset(preset: RaygentPreset | str) -> RaygentPresetDescription:
    """Return an inspectable description for one preset."""

    return _DESCRIPTIONS[_normalize_preset(preset)]


def resolve_raygent_preset(
    preset: RaygentPreset | str,
    *,
    overlays: tuple[RaygentOverlay | str, ...] = (),
    options: RaygentPresetOptions | None = None,
) -> RaygentPresetResolution:
    """Resolve one preset and optional overlays into factory kwargs and metadata."""

    resolved_preset = _normalize_preset(preset)
    resolved_overlays = _normalize_overlays(overlays)
    opts = options or RaygentPresetOptions()
    _raise_for_incompatible_inputs(resolved_preset, resolved_overlays, opts)

    description = describe_raygent_preset(resolved_preset)
    factory_kwargs: dict[str, object] = {}
    capabilities = [*description.capabilities]
    required_options = [*description.required_options]
    safety_notes = [*description.safety_notes]
    tool_names: tuple[str, ...] = ()
    context_profile = _context_profile_for_preset(resolved_preset)
    enable_bash = False
    requires_explicit_permission_options = resolved_preset in {
        "repo_maintainer",
        "full_developer",
    }

    if resolved_preset in {"minimal"}:
        factory_kwargs.update({"tools": "none", "context": "none"})
    elif resolved_preset in {"chat", "embedded_app"}:
        factory_kwargs.update(
            {
                "tools": "none",
                "context": "none",
                "transcript_store": _build_transcript_store(opts),
            }
        )
    elif resolved_preset in {"project_reader", "code_review", "research_agent"}:
        tools = _build_readonly_project_tools(opts)
        factory_kwargs.update({"tools": tools, "context": "project"})
        tool_names = _tool_names(tools)
    elif resolved_preset == "memory_agent":
        factory_kwargs.update(
            {
                "tools": "none",
                "context": "environment",
                "transcript_store": _build_transcript_store(opts),
            }
        )
    elif resolved_preset in {"repo_maintainer", "long_running_task"}:
        factory_kwargs.update(
            {
                "tools": "project" if resolved_preset == "repo_maintainer" else "none",
                "context": "project",
                "transcript_store": _build_transcript_store(opts),
            }
        )
    elif resolved_preset == "full_developer":
        factory_kwargs.update(
            {
                "tools": "project",
                "context": "project",
                "transcript_store": _build_transcript_store(opts),
            }
        )
        enable_bash = True
    else:  # pragma: no cover - guarded by _normalize_preset
        raise AssertionError(f"Unhandled Raygent preset: {resolved_preset}")

    if opts.task_output_dir is not None:
        factory_kwargs["task_output_dir"] = opts.task_output_dir

    for overlay in resolved_overlays:
        overlay_result = _apply_overlay(
            overlay,
            opts=opts,
            factory_kwargs=factory_kwargs,
            capabilities=capabilities,
            required_options=required_options,
            safety_notes=safety_notes,
        )
        if overlay_result.tools is not None:
            tool_names = _tool_names(overlay_result.tools)
        if overlay_result.context_profile is not None:
            context_profile = overlay_result.context_profile
        if overlay_result.enable_bash:
            enable_bash = True
        if overlay_result.requires_explicit_permission_options:
            requires_explicit_permission_options = True

    if not tool_names:
        tool_names = _tool_names_from_policy(factory_kwargs.get("tools"))

    return RaygentPresetResolution(
        preset=resolved_preset,
        overlays=resolved_overlays,
        factory_kwargs=MappingProxyType(dict(factory_kwargs)),
        enabled_capabilities=_dedupe(capabilities),
        required_options=_dedupe(required_options),
        safety_notes=_dedupe(safety_notes),
        tool_names=tool_names,
        context_profile=context_profile,
        enable_bash=enable_bash,
        requires_explicit_permission_options=requires_explicit_permission_options,
    )


@dataclass(frozen=True)
class _OverlayResult:
    tools: tuple[Tool, ...] | None = None
    context_profile: str | None = None
    enable_bash: bool = False
    requires_explicit_permission_options: bool = False


def _apply_overlay(
    overlay: RaygentOverlay,
    *,
    opts: RaygentPresetOptions,
    factory_kwargs: dict[str, object],
    capabilities: list[str],
    required_options: list[str],
    safety_notes: list[str],
) -> _OverlayResult:
    if overlay == "transcripts":
        factory_kwargs["transcript_store"] = _build_transcript_store(opts)
        capabilities.append("transcripts")
        return _OverlayResult()
    if overlay == "observability":
        capabilities.append("observability")
        return _OverlayResult()
    if overlay == "memory":
        capabilities.append("memory")
        required_options.append("memory_options")
        return _OverlayResult()
    if overlay == "goals":
        capabilities.append("goals_ready")
        required_options.append("goal runtime installer")
        safety_notes.append("Goal runtime is not a product /goal command parser.")
        return _OverlayResult()
    if overlay == "compaction":
        capabilities.append("compaction")
        factory_kwargs["transcript_store"] = _build_transcript_store(opts)
        return _OverlayResult()
    if overlay == "recovery":
        capabilities.append("recovery")
        factory_kwargs["transcript_store"] = _build_transcript_store(opts)
        return _OverlayResult()
    if overlay == "task_output":
        capabilities.append("task_output")
        if opts.task_output_dir is not None:
            factory_kwargs["task_output_dir"] = opts.task_output_dir
        return _OverlayResult()
    if overlay == "readonly_tools":
        tools = _build_readonly_project_tools(opts)
        factory_kwargs["tools"] = tools
        capabilities.append("readonly_tools")
        return _OverlayResult(tools=tools)
    if overlay == "file_tools":
        factory_kwargs["tools"] = "file"
        capabilities.append("file_tools")
        return _OverlayResult(requires_explicit_permission_options=True)
    if overlay == "bash":
        factory_kwargs["tools"] = "project"
        capabilities.append("bash")
        return _OverlayResult(enable_bash=True, requires_explicit_permission_options=True)
    if overlay == "agents":
        capabilities.append("agents_ready")
        required_options.append("agent_options")
        return _OverlayResult(requires_explicit_permission_options=True)
    if overlay == "coordinator":
        capabilities.append("coordinator_ready")
        required_options.append("agent_options.coordinator_runtime")
        return _OverlayResult()
    if overlay == "mcp":
        capabilities.append("mcp_ready")
        required_options.append("tool_catalog_provider or MCP catalog provider")
        return _OverlayResult(requires_explicit_permission_options=True)
    if overlay == "worktree":
        capabilities.append("worktree_ready")
        required_options.append("persistence_options.worktree_manager")
        return _OverlayResult(requires_explicit_permission_options=True)
    raise AssertionError(f"Unhandled Raygent overlay: {overlay}")  # pragma: no cover


def _raise_for_incompatible_inputs(
    preset: RaygentPreset,
    overlays: tuple[RaygentOverlay, ...],
    options: RaygentPresetOptions,
) -> None:
    issues: list[str] = []
    if "readonly_tools" in overlays and "file_tools" in overlays:
        issues.append("readonly_tools and file_tools overlays are mutually exclusive")
    if "file_tools" in overlays and not options.allow_filesystem_mutation:
        issues.append("file_tools requires allow_filesystem_mutation=True")
    if "bash" in overlays and not options.allow_shell:
        issues.append("bash requires allow_shell=True")
    if "agents" in overlays and not options.allow_agents:
        issues.append("agents requires allow_agents=True")
    if "mcp" in overlays and not options.allow_mcp:
        issues.append("mcp requires allow_mcp=True")
    if "worktree" in overlays and not options.allow_worktree:
        issues.append("worktree requires allow_worktree=True")
    if preset == "repo_maintainer" and not options.allow_filesystem_mutation:
        issues.append("repo_maintainer requires allow_filesystem_mutation=True")
    if preset == "full_developer":
        if not options.allow_full_developer:
            issues.append("full_developer requires allow_full_developer=True")
        if not options.allow_filesystem_mutation:
            issues.append("full_developer requires allow_filesystem_mutation=True")
        if not options.allow_shell:
            issues.append("full_developer requires allow_shell=True")
        if not options.allow_agents:
            issues.append("full_developer requires allow_agents=True")
        if not options.allow_mcp:
            issues.append("full_developer requires allow_mcp=True")
        if not options.allow_worktree:
            issues.append("full_developer requires allow_worktree=True")
    if issues:
        raise RaygentPresetCompatibilityError(
            "Raygent preset is not compatible with the supplied options: "
            + "; ".join(issues),
            preset=preset,
            overlays=tuple(overlays),
            issues=tuple(issues),
        )


def _normalize_preset(value: RaygentPreset | str) -> RaygentPreset:
    if value not in _PRESETS:
        raise RaygentPresetCompatibilityError(
            f"Unknown Raygent preset: {value!r}",
            preset=str(value),
            overlays=(),
            issues=(f"unknown preset: {value!r}",),
        )
    return value


def _normalize_overlays(values: tuple[RaygentOverlay | str, ...]) -> tuple[RaygentOverlay, ...]:
    overlays: list[RaygentOverlay] = []
    issues: list[str] = []
    for value in values:
        if value not in _OVERLAYS:
            issues.append(f"unknown overlay: {value!r}")
            continue
        normalized = value
        if normalized not in overlays:
            overlays.append(normalized)
    if issues:
        raise RaygentPresetCompatibilityError(
            "Raygent preset overlays are invalid: " + "; ".join(issues),
            preset="",
            overlays=tuple(str(value) for value in values),
            issues=tuple(issues),
        )
    return tuple(overlays)


def _context_profile_for_preset(preset: RaygentPreset) -> str:
    if preset in {"minimal", "chat", "embedded_app"}:
        return "none"
    if preset == "memory_agent":
        return "environment"
    return "project"


def _build_transcript_store(options: RaygentPresetOptions) -> JsonlTranscriptStore:
    return JsonlTranscriptStore(_transcript_base_dir(options))


def _transcript_base_dir(options: RaygentPresetOptions) -> str | Path | None:
    if options.transcript_dir is not None:
        return options.transcript_dir
    if options.project_root is not None:
        return Path(options.project_root).expanduser() / DEFAULT_TRANSCRIPT_DIR
    return None


def _build_readonly_project_tools(options: RaygentPresetOptions) -> tuple[Tool, ...]:
    discovery = create_discovery_tooling_runtime(backend=options.search_backend)
    return (
        build_file_read_tool(pdf_document_service=options.pdf_document_service),
        *discovery.tools,
        build_tool_search_tool(),
    )


def _tool_names(tools: tuple[Tool, ...]) -> tuple[str, ...]:
    return tuple(tool.name for tool in tools)


def _tool_names_from_policy(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        tools: list[Tool] = []
        for item in items:
            if not isinstance(item, Tool):
                return ()
            tools.append(item)
        return _tool_names(tuple(tools))
    if value == "none":
        return ()
    if value == "file":
        return ("Read", "Write", "Edit", "NotebookEdit")
    if value == "project":
        return (
            "Read",
            "Write",
            "Edit",
            "NotebookEdit",
            "Glob",
            "Grep",
            "TaskStop",
            "ToolSearch",
        )
    return ()


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


__all__ = [
    "RaygentOverlay",
    "RaygentPreset",
    "RaygentPresetCompatibilityError",
    "RaygentPresetDescription",
    "RaygentPresetOptions",
    "RaygentPresetResolution",
    "describe_raygent_preset",
    "list_raygent_presets",
    "resolve_raygent_preset",
]
