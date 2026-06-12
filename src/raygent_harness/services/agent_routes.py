"""Durable route metadata for resumable local agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

from raygent_harness.core.task import AgentRouteRecord

DEFAULT_AGENT_ROUTE_DIR = ".raygent/agent-routes"
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class AgentRouteRecordLoadResult:
    records: tuple[AgentRouteRecord, ...] = ()
    warnings: tuple[str, ...] = ()


class AgentRouteRecordStore(Protocol):
    """Persistence seam for named/raw-id local-agent route metadata."""

    async def save(self, record: AgentRouteRecord) -> None:
        """Create or replace one route record."""
        ...

    async def list_records(
        self,
        parent_session_id: str,
    ) -> AgentRouteRecordLoadResult:
        """Return recoverable route records for a parent session."""
        ...

    async def delete(self, parent_session_id: str, task_id: str) -> None:
        """Remove a route record if present."""
        ...


class JsonAgentRouteRecordStore:
    """Filesystem-backed route record store.

    Records live under `<base>/<safe-parent-session>/routes/<safe-task-id>.json`.
    Reads are fail-soft: malformed records are skipped with warnings so one bad
    sidecar cannot block the whole runtime recovery pass.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.base_dir = resolve_agent_route_base_dir(
            base_dir,
            project_root=project_root,
        )

    def record_dir(self, parent_session_id: str) -> Path:
        safe_session = safe_agent_route_component(parent_session_id)
        path = (self.base_dir / safe_session / "routes").resolve()
        if not _is_relative_to(path, self.base_dir):
            raise ValueError(f"agent route directory escapes base directory: {path}")
        return path

    def path_for_record(self, parent_session_id: str, task_id: str) -> Path:
        safe_task_id = safe_agent_route_component(task_id)
        record_dir = self.record_dir(parent_session_id)
        path = record_dir / f"{safe_task_id}.json"
        if path.parent != record_dir:
            raise ValueError(f"agent route record path escapes root: {path}")
        return path

    async def save(self, record: AgentRouteRecord) -> None:
        if not record.parent_session_id:
            raise ValueError("agent route record save requires parent_session_id")
        path = self.path_for_record(record.parent_session_id, record.task_id)
        await asyncio.to_thread(_write_record_sync, path, record)

    async def list_records(
        self,
        parent_session_id: str,
    ) -> AgentRouteRecordLoadResult:
        record_dir = self.record_dir(parent_session_id)
        return await asyncio.to_thread(_list_records_sync, record_dir, parent_session_id)

    async def delete(self, parent_session_id: str, task_id: str) -> None:
        path = self.path_for_record(parent_session_id, task_id)
        await asyncio.to_thread(_delete_record_sync, path)


def resolve_agent_route_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / DEFAULT_AGENT_ROUTE_DIR).resolve()


def safe_agent_route_component(value: str) -> str:
    if value == "":
        raise ValueError("agent route path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    if normalized == value:
        return normalized
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    return f"{normalized}-{digest}"


def agent_route_record_to_dict(record: AgentRouteRecord) -> dict[str, object]:
    return asdict(record)


def agent_route_record_from_dict(raw: object) -> AgentRouteRecord:
    if not isinstance(raw, dict):
        raise ValueError("agent route record must be an object")
    data = cast(dict[str, object], raw)
    task_type = _required_str(data, "task_type")
    if task_type != "local_agent":
        raise ValueError("agent route record task_type must be local_agent")
    return AgentRouteRecord(
        agent_id=_required_str(data, "agent_id"),
        task_id=_required_str(data, "task_id"),
        task_type="local_agent",
        name=_optional_str(data, "name"),
        parent_agent_id=_optional_str(data, "parent_agent_id"),
        parent_session_id=_required_str(data, "parent_session_id"),
        runtime_session_id=_optional_str(data, "runtime_session_id"),
        agent_type=_optional_str(data, "agent_type"),
        description=_optional_str(data, "description") or "",
        model=_optional_str(data, "model"),
        system_prompt=_optional_str(data, "system_prompt"),
        tool_names=tuple(
            item
            for item in _optional_str_list(data, "tool_names")
            if item.strip()
        ),
        permission_mode=_optional_str(data, "permission_mode"),
        cwd=_optional_str(data, "cwd"),
        worktree_path=_optional_str(data, "worktree_path"),
        worktree_branch=_optional_str(data, "worktree_branch"),
        worktree_slug=_optional_str(data, "worktree_slug"),
        worktree_created_at=_optional_float(data, "worktree_created_at"),
        worktree_touched_at=_optional_float(data, "worktree_touched_at"),
        worktree_cleanup_policy=_optional_str(data, "worktree_cleanup_policy"),
        transcript_path=_optional_str(data, "transcript_path"),
        is_sidechain=_optional_bool(data, "is_sidechain", default=True),
        content_replacement_replay=_optional_bool(
            data,
            "content_replacement_replay",
            default=True,
        ),
        route_registered_at=_optional_float(data, "route_registered_at") or 0.0,
    )


def normalize_agent_route_record_for_resume(
    record: AgentRouteRecord,
    *,
    parent_session_id: str,
    runtime_session_id: str | None,
) -> AgentRouteRecord:
    return replace(
        record,
        parent_session_id=parent_session_id,
        runtime_session_id=runtime_session_id,
    )


def _write_record_sync(path: Path, record: AgentRouteRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        agent_route_record_to_dict(record),
        sort_keys=True,
        separators=(",", ":"),
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _list_records_sync(
    record_dir: Path,
    parent_session_id: str,
) -> AgentRouteRecordLoadResult:
    if not record_dir.exists():
        return AgentRouteRecordLoadResult()
    records: list[AgentRouteRecord] = []
    warnings: list[str] = []
    for path in sorted(record_dir.glob("*.json")):
        try:
            stat_result = path.lstat()
            if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISREG(
                stat_result.st_mode
            ):
                warnings.append(f"skipped non-regular route record: {path.name}")
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = agent_route_record_from_dict(raw)
            if record.parent_session_id != parent_session_id:
                raise ValueError("route record parent_session_id mismatch")
        except Exception as exc:
            warnings.append(
                f"skipped route record {path.name}: {type(exc).__name__}"
            )
            continue
        records.append(record)
    return AgentRouteRecordLoadResult(records=tuple(records), warnings=tuple(warnings))


def _delete_record_sync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _required_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"agent route record field {key!r} must be non-empty string")
    return value


def _optional_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _optional_str_list(raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list | tuple):
        return ()
    items = cast(Sequence[object], value)
    return tuple(item for item in items if isinstance(item, str))


def _optional_float(raw: dict[str, object], key: str) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_bool(raw: dict[str, object], key: str, *, default: bool) -> bool:
    value = raw.get(key)
    return value if isinstance(value, bool) else default


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


__all__ = [
    "DEFAULT_AGENT_ROUTE_DIR",
    "AgentRouteRecordLoadResult",
    "AgentRouteRecordStore",
    "JsonAgentRouteRecordStore",
    "agent_route_record_from_dict",
    "agent_route_record_to_dict",
    "normalize_agent_route_record_for_resume",
    "resolve_agent_route_base_dir",
    "safe_agent_route_component",
]
