"""Convenience builders for common context-provider stacks.

The core query loop deliberately has no built-in filesystem/git/product
defaults. Embedders opt into these providers explicitly through QueryDeps.
"""

from __future__ import annotations

from pathlib import Path

from raygent_harness.context_providers.environment import (
    EnvironmentContextProvider,
    GitCommandRunner,
    GitStatusContextProvider,
    TodayProvider,
)
from raygent_harness.context_providers.project_instructions import (
    DiscoveryMode,
    ProjectInstructionConfig,
    ProjectInstructionsContextProvider,
)
from raygent_harness.core.context_providers import ContextAgentScope, ContextProvider


def build_default_context_providers(
    *,
    cwd: str | Path | None = None,
    workspace_root: str | Path | None = None,
    include_environment: bool = True,
    include_git_status: bool = True,
    include_project_instructions: bool = True,
    environment_agent_scope: ContextAgentScope = "all",
    git_status_agent_scope: ContextAgentScope = "main",
    project_instruction_agent_scope: ContextAgentScope = "all",
    today: TodayProvider | None = None,
    git_command_runner: GitCommandRunner | None = None,
    user_instruction_paths: tuple[str | Path, ...] = (),
    project_filenames: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md"),
    project_rule_dirs: tuple[str | Path, ...] = (".claude/rules",),
    local_filenames: tuple[str, ...] = ("AGENTS.local.md", "CLAUDE.local.md"),
    additional_dirs: tuple[str | Path, ...] = (),
    discovery_mode: DiscoveryMode = "layered_ancestors",
    allow_instruction_includes: bool = True,
    allow_external_instruction_includes: bool = False,
) -> tuple[ContextProvider, ...]:
    """Build the standard opt-in Raygent context provider stack.

    Defaults are chosen for headless harness fidelity:
    - environment context applies to main and child loops;
    - git status is main-agent-only by default to avoid stale inherited child
      snapshots;
    - project instructions apply to all loops by default because they are
      codebase policy, but embedders can set `project_instruction_agent_scope`.
    """

    providers: list[ContextProvider] = []

    if include_environment:
        if today is not None:
            providers.append(
                EnvironmentContextProvider(
                    cwd=cwd,
                    workspace_root=workspace_root,
                    agent_scope=environment_agent_scope,
                    today=today,
                )
            )
        else:
            providers.append(
                EnvironmentContextProvider(
                    cwd=cwd,
                    workspace_root=workspace_root,
                    agent_scope=environment_agent_scope,
                )
            )

    if include_git_status:
        if git_command_runner is not None:
            providers.append(
                GitStatusContextProvider(
                    cwd=cwd,
                    agent_scope=git_status_agent_scope,
                    command_runner=git_command_runner,
                )
            )
        else:
            providers.append(
                GitStatusContextProvider(
                    cwd=cwd,
                    agent_scope=git_status_agent_scope,
                )
            )

    if include_project_instructions:
        providers.append(
            ProjectInstructionsContextProvider(
                ProjectInstructionConfig(
                    cwd=cwd,
                    workspace_root=workspace_root,
                    user_instruction_paths=user_instruction_paths,
                    project_filenames=project_filenames,
                    project_rule_dirs=project_rule_dirs,
                    local_filenames=local_filenames,
                    additional_dirs=additional_dirs,
                    discovery_mode=discovery_mode,
                    allow_includes=allow_instruction_includes,
                    allow_external_includes=allow_external_instruction_includes,
                    agent_scope=project_instruction_agent_scope,
                )
            )
        )

    return tuple(providers)


__all__ = ["build_default_context_providers"]
