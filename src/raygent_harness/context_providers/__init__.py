"""Built-in context providers.

Core owns the provider protocol; this package owns concrete, policy-driven
providers that embedders can opt into.
"""

from raygent_harness.context_providers.defaults import build_default_context_providers
from raygent_harness.context_providers.environment import (
    EnvironmentContextProvider,
    GitCommandResult,
    GitCommandRunner,
    GitStatusContextProvider,
    default_git_command_runner,
)
from raygent_harness.context_providers.project_instructions import (
    ConditionalInstructionRule,
    DiscoveryMode,
    InstructionKind,
    ProjectInstructionConfig,
    ProjectInstructionFile,
    ProjectInstructionsContextProvider,
    ReadAdjacentProjectInstructionsContextProvider,
    discover_conditional_instruction_rules,
    discover_project_instruction_files,
    instruction_rule_matches_path,
    resolve_project_instructions_for_target_path,
)
from raygent_harness.context_providers.transcript_search import (
    TranscriptSearchContextProvider,
    TranscriptSearchQueryResolver,
)

__all__ = [
    "ConditionalInstructionRule",
    "DiscoveryMode",
    "EnvironmentContextProvider",
    "GitCommandResult",
    "GitCommandRunner",
    "GitStatusContextProvider",
    "InstructionKind",
    "ProjectInstructionConfig",
    "ProjectInstructionFile",
    "ProjectInstructionsContextProvider",
    "ReadAdjacentProjectInstructionsContextProvider",
    "TranscriptSearchContextProvider",
    "TranscriptSearchQueryResolver",
    "build_default_context_providers",
    "default_git_command_runner",
    "discover_conditional_instruction_rules",
    "discover_project_instruction_files",
    "instruction_rule_matches_path",
    "resolve_project_instructions_for_target_path",
]
