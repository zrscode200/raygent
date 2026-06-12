"""Durable coordinator runtime snapshot persistence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from raygent_harness.coordinator.runtime import (
    CoordinatorRuntimeSnapshot,
    CoordinatorRuntimeSnapshotDecodeError,
    coordinator_runtime_snapshot_from_dict,
    coordinator_runtime_snapshot_to_dict,
)
from raygent_harness.core.observability import (
    KernelEventBus,
    KernelEventContext,
    NoopKernelEventBus,
)

DEFAULT_COORDINATOR_SNAPSHOT_DIR = ".raygent/coordinator-runtime"
_UNSAFE_COMPONENT_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class CoordinatorRuntimeSnapshotLoadResult:
    snapshot: CoordinatorRuntimeSnapshot | None = None
    warnings: tuple[str, ...] = ()


class CoordinatorRuntimeSnapshotStore(Protocol):
    """Persistence seam for coordinator runtime snapshots."""

    async def save(self, snapshot: CoordinatorRuntimeSnapshot) -> None:
        """Create or replace a snapshot for its session/runtime identity."""
        ...

    async def load(
        self,
        session_id: str,
        *,
        runtime_session_id: str | None = None,
    ) -> CoordinatorRuntimeSnapshotLoadResult:
        """Return the snapshot for a session/runtime identity if recoverable."""
        ...


class JsonCoordinatorRuntimeSnapshotStore:
    """Filesystem-backed coordinator snapshot store.

    Records live under `<base>/<safe_session_id>/snapshot*.json`. Writes are
    atomic replace operations. Reads are fail-soft: malformed or unsupported
    records produce metadata-only diagnostics and a missing snapshot result.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        project_root: str | Path | None = None,
        observability: KernelEventBus | None = None,
        event_context: KernelEventContext | None = None,
    ) -> None:
        self.base_dir = resolve_coordinator_snapshot_base_dir(
            base_dir,
            project_root=project_root,
        )
        self._observability = observability or NoopKernelEventBus()
        self._event_context = event_context or KernelEventContext(source="coordinator")

    def path_for(
        self,
        session_id: str,
        *,
        runtime_session_id: str | None = None,
    ) -> Path:
        safe_session = safe_coordinator_snapshot_component(session_id)
        filename = (
            "snapshot.json"
            if runtime_session_id is None
            else f"snapshot-{safe_coordinator_snapshot_component(runtime_session_id)}.json"
        )
        path = (self.base_dir / safe_session / filename).resolve()
        if not _is_relative_to(path, self.base_dir):
            raise ValueError(f"coordinator snapshot path escapes base directory: {path}")
        return path

    async def save(self, snapshot: CoordinatorRuntimeSnapshot) -> None:
        session_id = _require_snapshot_session_id(snapshot)
        path = self.path_for(
            session_id,
            runtime_session_id=snapshot.runtime_session_id,
        )
        byte_count = await asyncio.to_thread(_write_snapshot_sync, path, snapshot)
        self._observability.emit(
            "coordinator.snapshot.saved",
            context=self._context(snapshot),
            data={
                "schema_version": snapshot.schema_version,
                "work_item_count": len(snapshot.work_items),
                "blackboard_entry_count": len(snapshot.blackboard_entries),
                "processed_notification_key_count": len(
                    snapshot.processed_notification_keys
                ),
                "processed_notification_count": snapshot.processed_notification_count,
                "byte_count": byte_count,
                "session_id_present": snapshot.session_id is not None,
                "runtime_session_id_present": snapshot.runtime_session_id is not None,
            },
        )

    async def load(
        self,
        session_id: str,
        *,
        runtime_session_id: str | None = None,
    ) -> CoordinatorRuntimeSnapshotLoadResult:
        path = self.path_for(session_id, runtime_session_id=runtime_session_id)
        context = replace(
            self._event_context,
            session_id=session_id,
            runtime_session_id=runtime_session_id,
            source="coordinator",
        )
        try:
            snapshot = await asyncio.to_thread(_read_snapshot_sync, path)
            _validate_snapshot_identity(
                snapshot,
                session_id=session_id,
                runtime_session_id=runtime_session_id,
            )
        except FileNotFoundError:
            self._observability.emit(
                "coordinator.snapshot.load_missing",
                context=context,
                data={
                    "session_id_present": True,
                    "runtime_session_id_present": runtime_session_id is not None,
                },
            )
            return CoordinatorRuntimeSnapshotLoadResult()
        except Exception as exc:
            warning = f"coordinator snapshot load failed: {type(exc).__name__}"
            self._observability.emit(
                "coordinator.snapshot.load_failed",
                context=context,
                data={
                    "error_type": type(exc).__name__,
                    "session_id_present": True,
                    "runtime_session_id_present": runtime_session_id is not None,
                },
            )
            return CoordinatorRuntimeSnapshotLoadResult(warnings=(warning,))

        self._observability.emit(
            "coordinator.snapshot.loaded",
            context=self._context(snapshot),
            data={
                "schema_version": snapshot.schema_version,
                "work_item_count": len(snapshot.work_items),
                "blackboard_entry_count": len(snapshot.blackboard_entries),
                "processed_notification_key_count": len(
                    snapshot.processed_notification_keys
                ),
                "processed_notification_count": snapshot.processed_notification_count,
                "session_id_present": snapshot.session_id is not None,
                "runtime_session_id_present": snapshot.runtime_session_id is not None,
            },
        )
        return CoordinatorRuntimeSnapshotLoadResult(snapshot=snapshot)

    def _context(self, snapshot: CoordinatorRuntimeSnapshot) -> KernelEventContext:
        return replace(
            self._event_context,
            session_id=snapshot.session_id or self._event_context.session_id,
            runtime_session_id=(
                snapshot.runtime_session_id or self._event_context.runtime_session_id
            ),
            source="coordinator",
        )


def resolve_coordinator_snapshot_base_dir(
    base_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    return (root / DEFAULT_COORDINATOR_SNAPSHOT_DIR).resolve()


def safe_coordinator_snapshot_component(value: str) -> str:
    if value == "":
        raise ValueError("coordinator snapshot path component cannot be empty")
    normalized = _UNSAFE_COMPONENT_CHARS.sub("_", value).strip("._")
    if normalized == value:
        return normalized
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    if normalized == "":
        return f"id-{digest}"
    return f"{normalized}-{digest}"


def _require_snapshot_session_id(snapshot: CoordinatorRuntimeSnapshot) -> str:
    if snapshot.session_id is None or snapshot.session_id == "":
        raise ValueError("coordinator snapshot save requires snapshot.session_id")
    return snapshot.session_id


def _write_snapshot_sync(path: Path, snapshot: CoordinatorRuntimeSnapshot) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        coordinator_runtime_snapshot_to_dict(snapshot),
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
    return len(payload.encode("utf-8")) + 1


def _read_snapshot_sync(path: Path) -> CoordinatorRuntimeSnapshot:
    if not path.exists():
        raise FileNotFoundError(path)
    stat_result = path.lstat()
    if stat.S_ISLNK(stat_result.st_mode) or not stat.S_ISREG(stat_result.st_mode):
        raise CoordinatorRuntimeSnapshotDecodeError(
            "Coordinator snapshot path must be a regular file"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CoordinatorRuntimeSnapshotDecodeError("Invalid snapshot JSON") from exc
    return coordinator_runtime_snapshot_from_dict(raw)


def _validate_snapshot_identity(
    snapshot: CoordinatorRuntimeSnapshot,
    *,
    session_id: str,
    runtime_session_id: str | None,
) -> None:
    if snapshot.session_id != session_id:
        raise CoordinatorRuntimeSnapshotDecodeError(
            "Coordinator snapshot session id does not match requested session"
        )
    if snapshot.runtime_session_id != runtime_session_id:
        raise CoordinatorRuntimeSnapshotDecodeError(
            "Coordinator snapshot runtime session id does not match requested runtime"
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "DEFAULT_COORDINATOR_SNAPSHOT_DIR",
    "CoordinatorRuntimeSnapshotLoadResult",
    "CoordinatorRuntimeSnapshotStore",
    "JsonCoordinatorRuntimeSnapshotStore",
    "resolve_coordinator_snapshot_base_dir",
    "safe_coordinator_snapshot_component",
]
