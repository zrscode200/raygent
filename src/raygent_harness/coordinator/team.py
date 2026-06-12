"""Headless team metadata primitives.


Raygent v1 keeps teams as structured metadata and project-local JSON files.
There are no panes, mailboxes, task-list backends, or process supervisors in
core yet; those remain staged coordinator/backend work.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

TEAM_LEAD_NAME = "team-lead"
DEFAULT_TEAMS_DIR = Path(".raygent/teams")


@dataclass(frozen=True)
class TeamMember:
    agent_id: str
    name: str
    agent_type: str
    model: str
    joined_at: float
    cwd: str


@dataclass(frozen=True)
class TeamContext:
    team_name: str
    description: str | None
    team_file_path: str
    lead_agent_id: str
    created_at: float
    members: tuple[TeamMember, ...] = ()


@dataclass
class TeamStateStore:
    """Session-local team context plus project-local metadata files."""

    base_dir: Path = field(default_factory=lambda: DEFAULT_TEAMS_DIR)
    current_team: TeamContext | None = None

    def create_team(
        self,
        *,
        team_name: str,
        description: str | None,
        agent_type: str | None,
        model: str,
        cwd: str,
        now: float | None = None,
    ) -> TeamContext:
        """Create a team metadata file and set current session team context."""

        if self.current_team is not None:
            msg = (
                f'Already leading team "{self.current_team.team_name}". '
                "A coordinator can only manage one team at a time."
            )
            raise TeamAlreadyExistsError(msg)

        timestamp = time.time() if now is None else now
        final_name = self.unique_team_name(team_name)
        lead_agent_id = format_agent_id(TEAM_LEAD_NAME, final_name)
        lead_agent_type = agent_type or TEAM_LEAD_NAME
        member = TeamMember(
            agent_id=lead_agent_id,
            name=TEAM_LEAD_NAME,
            agent_type=lead_agent_type,
            model=model,
            joined_at=timestamp,
            cwd=cwd,
        )
        path = team_config_path(self.base_dir, final_name)
        context = TeamContext(
            team_name=final_name,
            description=description,
            team_file_path=str(path),
            lead_agent_id=lead_agent_id,
            created_at=timestamp,
            members=(member,),
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(context), indent=2, sort_keys=True), encoding="utf-8")
        self.current_team = context
        return context

    def unique_team_name(self, requested_name: str) -> str:
        """Return requested slug, or a deterministic numeric suffix if used."""

        base = sanitize_team_name(requested_name)
        candidate = base
        index = 2
        while team_config_path(self.base_dir, candidate).exists():
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def add_member(
        self,
        *,
        agent_id: str,
        name: str,
        agent_type: str,
        model: str,
        cwd: str,
        team_name: str | None = None,
        now: float | None = None,
    ) -> TeamMember:
        """Add an addressable teammate to the current team metadata."""

        context = self.current_team
        if context is None:
            raise TeamNotFoundError("No active team context. Create a team first.")
        if team_name is not None and sanitize_team_name(team_name) != context.team_name:
            raise TeamNotFoundError(
                f'Team "{sanitize_team_name(team_name)}" is not the current team.'
            )

        normalized_name = sanitize_team_name(name)
        if any(member.name == normalized_name for member in context.members):
            raise TeamMemberAlreadyExistsError(
                f'Team member "{normalized_name}" already exists.'
            )

        member = TeamMember(
            agent_id=agent_id,
            name=normalized_name,
            agent_type=agent_type,
            model=model,
            joined_at=time.time() if now is None else now,
            cwd=cwd,
        )
        updated = replace(context, members=(*context.members, member))
        path = Path(updated.team_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(updated), indent=2, sort_keys=True), encoding="utf-8")
        self.current_team = updated
        return member

    def unique_member_name(self, requested_name: str) -> str:
        """Return a deterministic unique teammate name within the current team."""

        context = self.current_team
        base = sanitize_team_name(requested_name)
        if context is None:
            return base
        existing = {member.name for member in context.members}
        candidate = base
        index = 2
        while candidate in existing:
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def remove_member_by_agent_id(self, agent_id: str) -> bool:
        """Remove a teammate from current team metadata by agent id."""

        context = self.current_team
        if context is None:
            return False
        remaining = tuple(member for member in context.members if member.agent_id != agent_id)
        if len(remaining) == len(context.members):
            return False
        updated = replace(context, members=remaining)
        path = Path(updated.team_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(updated), indent=2, sort_keys=True), encoding="utf-8")
        self.current_team = updated
        return True


class TeamAlreadyExistsError(RuntimeError):
    """Raised when the current session already leads a team."""


class TeamNotFoundError(RuntimeError):
    """Raised when teammate routing requires an active team context."""


class TeamMemberAlreadyExistsError(RuntimeError):
    """Raised when a team member name would collide within a team."""


def sanitize_team_name(value: str) -> str:
    """Normalize a model-provided team name for project-local paths."""

    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "team"


def format_agent_id(agent_name: str, team_name: str) -> str:
    """Reference-compatible team lead id shape."""

    return f"{sanitize_team_name(agent_name)}@{sanitize_team_name(team_name)}"


def team_config_path(base_dir: Path, team_name: str) -> Path:
    return base_dir / sanitize_team_name(team_name) / "config.json"


__all__ = [
    "DEFAULT_TEAMS_DIR",
    "TEAM_LEAD_NAME",
    "TeamAlreadyExistsError",
    "TeamContext",
    "TeamMember",
    "TeamMemberAlreadyExistsError",
    "TeamNotFoundError",
    "TeamStateStore",
    "format_agent_id",
    "sanitize_team_name",
    "team_config_path",
]
