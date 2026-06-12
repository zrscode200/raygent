"""Provider-neutral remote-agent backend and persistence seams.

Reference remote agents are product/backend specific: AgentTool teleports a
session to a hosted backend, registers a `remote_agent` task, and a detached
poller turns remote completion into the same model-facing task notification
queue used by local background agents.

Raygent keeps only the kernel seams here. Embedders provide a backend that knows
how to launch, poll, stop, and optionally restore its own remote execution
environment. Persistence stores identity sidecars only; raw prompt/output stay
out of the restore record by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, cast

RemoteAgentPollStatus = Literal["running", "completed", "failed"]
RemoteAgentRestoreStatus = Literal[
    "running",
    "completed",
    "failed",
    "archived",
    "gone",
]

_DEFAULT_REMOTE_AGENT_PERSISTENCE_DIR = ".raygent/remote-agents"
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_PROCESS_SESSION_ID = f"session-{os.getpid()}-{int(time.time())}"


@dataclass(frozen=True)
class RemoteAgentLaunchRequest:
    """Inputs passed from AgentTool to the installed remote backend."""

    task_id: str
    prompt: str
    description: str
    agent_type: str
    parent_agent_id: str | None = None
    tool_use_id: str | None = None
    model: str | None = None
    cwd: str | None = None


@dataclass(frozen=True)
class RemoteAgentLaunchResult:
    """Backend launch result persisted on the Raygent task state."""

    remote_id: str
    title: str
    session_url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentPollResult:
    """One poll observation from a remote backend."""

    status: RemoteAgentPollStatus
    message: str = ""
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentPollRequest:
    """Poll request with both Raygent and backend identities."""

    task_id: str
    remote_id: str
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentStopRequest:
    """Stop request with enough state for stateless backend adapters."""

    task_id: str
    remote_id: str
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentRestoreRequest:
    """Restore-time live-status request for persisted remote-agent identity."""

    task_id: str
    remote_id: str
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentRestoreResult:
    """Live remote status observed during session restore."""

    status: RemoteAgentRestoreStatus
    title: str | None = None
    session_url: str | None = None
    message: str = ""
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class RemoteAgentPersistenceRecord:
    """Identity sidecar for reconnecting a remote agent after process restart.

    The record intentionally omits raw prompt, final output, and error text.
    Backend metadata is retained because it may be needed for stateless poll and
    stop calls; observability for this record only emits key counts.
    """

    task_id: str
    remote_id: str
    description: str
    parent_agent_id: str | None = None
    tool_use_id: str | None = None
    agent_type: str | None = None
    model: str | None = None
    cwd: str | None = None
    session_url: str | None = None
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict[str, str])
    start_time: float = 0.0
    updated_at: float = 0.0


class RemoteAgentBackend(Protocol):
    """Launch/poll/stop protocol for remote agent execution."""

    async def launch(
        self,
        request: RemoteAgentLaunchRequest,
    ) -> RemoteAgentLaunchResult:
        """Create the remote session for a Raygent task id."""
        ...

    async def poll(self, request: RemoteAgentPollRequest) -> RemoteAgentPollResult:
        """Return the current remote task state."""
        ...

    async def stop(self, request: RemoteAgentStopRequest) -> None:
        """Best-effort remote stop/archive for explicit task kills."""
        ...


class RemoteAgentRestoreBackend(RemoteAgentBackend, Protocol):
    """Optional backend extension for reconnecting persisted remote sessions."""

    async def restore(
        self,
        request: RemoteAgentRestoreRequest,
    ) -> RemoteAgentRestoreResult:
        """Return live state for a persisted remote task identity."""
        ...


class RemoteAgentPersistenceStore(Protocol):
    """Persistence seam for remote-agent restore sidecars."""

    async def save(self, record: RemoteAgentPersistenceRecord) -> None:
        """Create or replace a persisted remote-agent record."""
        ...

    async def list_records(self) -> tuple[RemoteAgentPersistenceRecord, ...]:
        """Return all records scoped to this store's session."""
        ...

    async def delete(self, task_id: str) -> None:
        """Remove the persisted record for `task_id` if present."""
        ...


class JsonRemoteAgentPersistenceStore:
    """Filesystem-backed remote-agent sidecar store.

    Records live under `<base>/<session_id>/remote-agents/<safe_task_id>.json`.
    Writes are atomic replace operations and reads skip malformed records rather
    than failing the whole restore pass.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        base = resolve_remote_agent_persistence_base_dir(
            base_dir,
            project_root=project_root,
        )
        session = safe_remote_agent_component(session_id or _PROCESS_SESSION_ID)
        self._base_dir = base
        self._record_dir = (base / session / "remote-agents").resolve()
        if not _is_relative_to(self._record_dir, base):
            raise ValueError(
                f"remote-agent record directory escapes base directory: {self._record_dir}"
            )

    @property
    def record_dir(self) -> Path:
        """Concrete directory containing this session's records."""

        return self._record_dir

    def path_for_task(self, task_id: str) -> Path:
        safe_task_id = safe_remote_agent_component(task_id)
        path = self._record_dir / f"{safe_task_id}.json"
        if path.parent != self._record_dir:
            raise ValueError(f"remote-agent record path escapes root: {path}")
        return path

    async def save(self, record: RemoteAgentPersistenceRecord) -> None:
        path = self.path_for_task(record.task_id)
        await asyncio.to_thread(_write_record_sync, path, record)

    async def list_records(self) -> tuple[RemoteAgentPersistenceRecord, ...]:
        return await asyncio.to_thread(_list_records_sync, self._record_dir)

    async def delete(self, task_id: str) -> None:
        path = self.path_for_task(task_id)
        await asyncio.to_thread(_delete_record_sync, path)


def resolve_remote_agent_persistence_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the root directory for remote-agent sidecars."""

    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / _DEFAULT_REMOTE_AGENT_PERSISTENCE_DIR).resolve()


def safe_remote_agent_component(value: str) -> str:
    """Make a task/session id safe for one filesystem component."""

    if value == "":
        raise ValueError("remote-agent path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    if normalized == value:
        return normalized
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    return f"{normalized}-{digest}"


def _write_record_sync(path: Path, record: RemoteAgentPersistenceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
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


def _list_records_sync(record_dir: Path) -> tuple[RemoteAgentPersistenceRecord, ...]:
    if not record_dir.exists():
        return ()
    records: list[RemoteAgentPersistenceRecord] = []
    for path in sorted(record_dir.glob("*.json")):
        try:
            stat_result = path.lstat()
            if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISREG(
                stat_result.st_mode
            ):
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = _record_from_raw(raw)
        except Exception:
            continue
        records.append(record)
    return tuple(records)


def _record_from_raw(raw: object) -> RemoteAgentPersistenceRecord:
    if not isinstance(raw, dict):
        raise ValueError("remote-agent record must be an object")
    data = cast(dict[str, object], raw)
    task_id = _required_str(data, "task_id")
    remote_id = _required_str(data, "remote_id")
    description = _required_str(data, "description")
    return RemoteAgentPersistenceRecord(
        task_id=task_id,
        remote_id=remote_id,
        description=description,
        parent_agent_id=_optional_str(data, "parent_agent_id"),
        tool_use_id=_optional_str(data, "tool_use_id"),
        agent_type=_optional_str(data, "agent_type"),
        model=_optional_str(data, "model"),
        cwd=_optional_str(data, "cwd"),
        session_url=_optional_str(data, "session_url"),
        title=_optional_str(data, "title") or "",
        metadata=_str_dict(data.get("metadata")),
        start_time=_float_or(data.get("start_time"), 0.0),
        updated_at=_float_or(data.get("updated_at"), 0.0),
    )


def _required_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"remote-agent record field {key!r} must be non-empty string")
    return value


def _optional_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    data = cast(dict[object, object], value)
    return {str(k): str(v) for k, v in data.items() if isinstance(k, str)}


def _float_or(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _delete_record_sync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


__all__ = [
    "JsonRemoteAgentPersistenceStore",
    "RemoteAgentBackend",
    "RemoteAgentLaunchRequest",
    "RemoteAgentLaunchResult",
    "RemoteAgentPersistenceRecord",
    "RemoteAgentPersistenceStore",
    "RemoteAgentPollRequest",
    "RemoteAgentPollResult",
    "RemoteAgentPollStatus",
    "RemoteAgentRestoreBackend",
    "RemoteAgentRestoreRequest",
    "RemoteAgentRestoreResult",
    "RemoteAgentRestoreStatus",
    "RemoteAgentStopRequest",
    "resolve_remote_agent_persistence_base_dir",
    "safe_remote_agent_component",
]
