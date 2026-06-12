"""Skill data model.

Skills are prompt commands loaded from `SKILL.md`, bundled registries,
plugins, or MCP. Raygent keeps the metadata shape headless and deliberately
avoids UI command execution in this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

SkillLoadedFrom = Literal[
    "commands_DEPRECATED",
    "skills",
    "plugin",
    "managed",
    "bundled",
    "mcp",
]

SkillSource = Literal[
    "policySettings",
    "userSettings",
    "projectSettings",
    "plugin",
    "managed",
    "bundled",
    "mcp",
]

SkillContext = Literal["inline", "fork"]
SkillShell = Literal["bash", "powershell"]

type HookSettings = Mapping[str, Any]


def _empty_hooks() -> HookSettings:
    return MappingProxyType({})


@dataclass(frozen=True)
class SkillDefinition:
    """Prompt-skill metadata visible to later orchestration/search layers."""

    name: str
    description: str
    markdown_content: str
    source: SkillSource
    loaded_from: SkillLoadedFrom
    content_length: int

    display_name: str | None = None
    has_user_specified_description: bool = False
    aliases: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None
    argument_names: tuple[str, ...] = ()
    when_to_use: str | None = None
    version: str | None = None
    model: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    hooks: HookSettings | None = None
    context: SkillContext | None = None
    agent: str | None = None
    effort: str | None = None
    shell: SkillShell | None = None
    paths: tuple[str, ...] = ()
    skill_root: Path | None = None
    file_path: Path | None = None

    @property
    def is_hidden(self) -> bool:
        return not self.user_invocable

    def user_facing_name(self) -> str:
        return self.display_name or self.name

    def render_prompt(self, args: str = "", session_id: str | None = None) -> str:
        """Render the static prompt body for this skill.

        This is intentionally a small primitive, not the full reference
        command runner. It preserves the base-dir prefix and common
        placeholders so future orchestration can use it without importing UI.
        """

        content = self.markdown_content
        if self.skill_root is not None:
            content = f"Base directory for this skill: {self.skill_root}\n\n{content}"
            content = content.replace("${CLAUDE_SKILL_DIR}", self.skill_root.as_posix())
        content = content.replace("$ARGUMENTS", args)
        if session_id is not None:
            content = content.replace("${CLAUDE_SESSION_ID}", session_id)
        return content


@dataclass(frozen=True)
class LoadedSkill:
    """A skill plus loader provenance needed for deduplication."""

    skill: SkillDefinition
    file_path: Path
    file_identity: str | None = None


@dataclass(frozen=True)
class BundledSkillDefinition:
    """Programmatic bundled skill registration input."""

    name: str
    description: str
    prompt: str
    aliases: tuple[str, ...] = ()
    when_to_use: str | None = None
    argument_hint: str | None = None
    allowed_tools: tuple[str, ...] = ()
    model: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    hooks: HookSettings | None = None
    context: SkillContext | None = None
    agent: str | None = None
    files: Mapping[str, str] = field(default_factory=dict[str, str])

    def to_skill(self, skill_root: Path | None = None) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description=self.description,
            markdown_content=self.prompt,
            source="bundled",
            loaded_from="bundled",
            content_length=0,
            has_user_specified_description=True,
            aliases=self.aliases,
            allowed_tools=self.allowed_tools,
            argument_hint=self.argument_hint,
            when_to_use=self.when_to_use,
            model=self.model,
            disable_model_invocation=self.disable_model_invocation,
            user_invocable=self.user_invocable,
            hooks=self.hooks if self.hooks is not None else _empty_hooks(),
            context=self.context,
            agent=self.agent,
            skill_root=skill_root,
        )


__all__ = [
    "BundledSkillDefinition",
    "HookSettings",
    "LoadedSkill",
    "SkillContext",
    "SkillDefinition",
    "SkillLoadedFrom",
    "SkillShell",
    "SkillSource",
]
