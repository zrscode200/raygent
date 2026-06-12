"""Transcript event-log data shapes.

Raygent persists sessions as JSONL records with explicit parent chaining and
separate content-replacement records. Chain identity lives in the transcript
envelope rather than mutating provider-bound `MessageParam` payloads with UUIDs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from uuid import uuid4

from raygent_harness.core.messages import MessageParam
from raygent_harness.core.state import CompactBoundary
from raygent_harness.services.compact.tool_result_budget import (
    ToolResultReplacementRecord,
)

TranscriptEntryType = Literal[
    "message",
    "compact_boundary",
    "content_replacement",
    "tombstone",
    "session_metadata",
    "stream_event",
]


def new_transcript_entry_id() -> str:
    return f"tr_{uuid4().hex}"


def current_transcript_time() -> float:
    return time.time()


@dataclass(frozen=True)
class TranscriptScope:
    """Physical transcript scope.

    `session_id` is the root session for filesystem grouping. Sidechains carry
    a child `agent_id` and optional child runtime session id while still living
    under the parent session directory.
    """

    session_id: str
    agent_id: str | None = None
    is_sidechain: bool = False
    runtime_session_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class TranscriptMessageEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    parent_entry_id: str | None = None
    logical_parent_entry_id: str | None = None
    session_id: str
    runtime_session_id: str | None = None
    agent_id: str | None = None
    is_sidechain: bool = False
    created_at: float = field(default_factory=current_transcript_time)
    cwd: str | None = None
    version: str | None = None
    message: MessageParam
    provider_message_id: str | None = None
    type: Literal["message"] = "message"


@dataclass(frozen=True, kw_only=True)
class CompactBoundaryEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    session_id: str
    agent_id: str | None = None
    created_at: float = field(default_factory=current_transcript_time)
    boundary: CompactBoundary
    post_compact_message_count: int | None = None
    type: Literal["compact_boundary"] = "compact_boundary"


@dataclass(frozen=True, kw_only=True)
class ContentReplacementEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    session_id: str
    agent_id: str | None = None
    created_at: float = field(default_factory=current_transcript_time)
    replacements: tuple[ToolResultReplacementRecord, ...]
    type: Literal["content_replacement"] = "content_replacement"


@dataclass(frozen=True, kw_only=True)
class TombstoneEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    session_id: str
    agent_id: str | None = None
    created_at: float = field(default_factory=current_transcript_time)
    target_entry_id: str | None = None
    target_message_id: str | None = None
    reason: str = ""
    event: dict[str, Any] | None = None
    type: Literal["tombstone"] = "tombstone"


@dataclass(frozen=True, kw_only=True)
class SessionMetadataEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    session_id: str
    created_at: float = field(default_factory=current_transcript_time)
    cwd: str | None = None
    version: str | None = None
    label: str | None = None
    mode: str | None = None
    type: Literal["session_metadata"] = "session_metadata"


@dataclass(frozen=True, kw_only=True)
class StreamEventEntry:
    entry_id: str = field(default_factory=new_transcript_entry_id)
    session_id: str
    agent_id: str | None = None
    created_at: float = field(default_factory=current_transcript_time)
    event: dict[str, Any]
    type: Literal["stream_event"] = "stream_event"


type TranscriptEntry = (
    TranscriptMessageEntry
    | CompactBoundaryEntry
    | ContentReplacementEntry
    | TombstoneEntry
    | SessionMetadataEntry
    | StreamEventEntry
)


class TranscriptDecodeError(ValueError):
    """Raised when a JSON object cannot be decoded into a transcript entry."""


def transcript_entry_to_dict(entry: TranscriptEntry) -> dict[str, Any]:
    if isinstance(entry, TranscriptMessageEntry):
        raw: dict[str, Any] = {
            "type": entry.type,
            "entry_id": entry.entry_id,
            "parent_entry_id": entry.parent_entry_id,
            "logical_parent_entry_id": entry.logical_parent_entry_id,
            "session_id": entry.session_id,
            "runtime_session_id": entry.runtime_session_id,
            "agent_id": entry.agent_id,
            "is_sidechain": entry.is_sidechain,
            "created_at": entry.created_at,
            "cwd": entry.cwd,
            "version": entry.version,
            "message": _json_clone(entry.message),
        }
        if entry.provider_message_id is not None:
            raw["provider_message_id"] = entry.provider_message_id
        return raw
    if isinstance(entry, CompactBoundaryEntry):
        return {
            "type": entry.type,
            "entry_id": entry.entry_id,
            "session_id": entry.session_id,
            "agent_id": entry.agent_id,
            "created_at": entry.created_at,
            "boundary": _compact_boundary_to_dict(entry.boundary),
            "post_compact_message_count": entry.post_compact_message_count,
        }
    if isinstance(entry, ContentReplacementEntry):
        return {
            "type": entry.type,
            "entry_id": entry.entry_id,
            "session_id": entry.session_id,
            "agent_id": entry.agent_id,
            "created_at": entry.created_at,
            "replacements": [
                _replacement_record_to_dict(record) for record in entry.replacements
            ],
        }
    if isinstance(entry, TombstoneEntry):
        return {
            "type": entry.type,
            "entry_id": entry.entry_id,
            "session_id": entry.session_id,
            "agent_id": entry.agent_id,
            "created_at": entry.created_at,
            "target_entry_id": entry.target_entry_id,
            "target_message_id": entry.target_message_id,
            "reason": entry.reason,
            "event": _json_clone(entry.event) if entry.event is not None else None,
        }
    if isinstance(entry, SessionMetadataEntry):
        return {
            "type": entry.type,
            "entry_id": entry.entry_id,
            "session_id": entry.session_id,
            "created_at": entry.created_at,
            "cwd": entry.cwd,
            "version": entry.version,
            "label": entry.label,
            "mode": entry.mode,
        }
    return {
        "type": entry.type,
        "entry_id": entry.entry_id,
        "session_id": entry.session_id,
        "agent_id": entry.agent_id,
        "created_at": entry.created_at,
        "event": _json_clone(entry.event),
    }


def transcript_entry_to_json(entry: TranscriptEntry) -> str:
    return json.dumps(transcript_entry_to_dict(entry), ensure_ascii=False, sort_keys=True)


def transcript_entry_from_json(line: str) -> TranscriptEntry:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise TranscriptDecodeError(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise TranscriptDecodeError("transcript line is not a JSON object")
    return transcript_entry_from_dict(cast(Mapping[str, object], raw))


def transcript_entry_from_dict(raw: Mapping[str, object]) -> TranscriptEntry:
    entry_type = raw.get("type")
    if entry_type == "message":
        return TranscriptMessageEntry(
            entry_id=_required_str(raw, "entry_id"),
            parent_entry_id=_optional_str(raw, "parent_entry_id"),
            logical_parent_entry_id=_optional_str(raw, "logical_parent_entry_id"),
            session_id=_required_str(raw, "session_id"),
            runtime_session_id=_optional_str(raw, "runtime_session_id"),
            agent_id=_optional_str(raw, "agent_id"),
            is_sidechain=_bool_field(raw, "is_sidechain", default=False),
            created_at=_float_field(raw, "created_at"),
            cwd=_optional_str(raw, "cwd"),
            version=_optional_str(raw, "version"),
            message=_message_field(raw, "message"),
            provider_message_id=_optional_str(raw, "provider_message_id"),
        )
    if entry_type == "compact_boundary":
        return CompactBoundaryEntry(
            entry_id=_required_str(raw, "entry_id"),
            session_id=_required_str(raw, "session_id"),
            agent_id=_optional_str(raw, "agent_id"),
            created_at=_float_field(raw, "created_at"),
            boundary=_compact_boundary_field(raw, "boundary"),
            post_compact_message_count=_optional_int(raw, "post_compact_message_count"),
        )
    if entry_type == "content_replacement":
        return ContentReplacementEntry(
            entry_id=_required_str(raw, "entry_id"),
            session_id=_required_str(raw, "session_id"),
            agent_id=_optional_str(raw, "agent_id"),
            created_at=_float_field(raw, "created_at"),
            replacements=_replacement_records_field(raw, "replacements"),
        )
    if entry_type == "tombstone":
        return TombstoneEntry(
            entry_id=_required_str(raw, "entry_id"),
            session_id=_required_str(raw, "session_id"),
            agent_id=_optional_str(raw, "agent_id"),
            created_at=_float_field(raw, "created_at"),
            target_entry_id=_optional_str(raw, "target_entry_id"),
            target_message_id=_optional_str(raw, "target_message_id"),
            reason=_str_field(raw, "reason", default=""),
            event=_optional_dict_field(raw, "event"),
        )
    if entry_type == "session_metadata":
        return SessionMetadataEntry(
            entry_id=_required_str(raw, "entry_id"),
            session_id=_required_str(raw, "session_id"),
            created_at=_float_field(raw, "created_at"),
            cwd=_optional_str(raw, "cwd"),
            version=_optional_str(raw, "version"),
            label=_optional_str(raw, "label"),
            mode=_optional_str(raw, "mode"),
        )
    if entry_type == "stream_event":
        return StreamEventEntry(
            entry_id=_required_str(raw, "entry_id"),
            session_id=_required_str(raw, "session_id"),
            agent_id=_optional_str(raw, "agent_id"),
            created_at=_float_field(raw, "created_at"),
            event=_dict_field(raw, "event"),
        )
    raise TranscriptDecodeError(f"unknown transcript entry type: {entry_type!r}")


def _compact_boundary_to_dict(boundary: CompactBoundary) -> dict[str, Any]:
    return {
        "message_index": boundary.message_index,
        "kind": boundary.kind,
        "summary": boundary.summary,
    }


def _replacement_record_to_dict(record: ToolResultReplacementRecord) -> dict[str, Any]:
    return {
        "tool_use_id": record.tool_use_id,
        "replacement": record.replacement,
        "path": record.path,
        "original_size_chars": record.original_size_chars,
    }


def _compact_boundary_field(
    raw: Mapping[str, object],
    key: str,
) -> CompactBoundary:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise TranscriptDecodeError(f"{key} must be an object")
    mapping = cast(Mapping[str, object], value)
    kind = _required_str(mapping, "kind")
    if kind not in ("microcompact", "autocompact", "context_collapse", "snip"):
        raise TranscriptDecodeError(f"{key}.kind is invalid: {kind!r}")
    return CompactBoundary(
        message_index=_int_field(mapping, "message_index"),
        kind=kind,
        summary=_required_str(mapping, "summary"),
    )


def _replacement_records_field(
    raw: Mapping[str, object],
    key: str,
) -> tuple[ToolResultReplacementRecord, ...]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TranscriptDecodeError(f"{key} must be a list")
    items = cast(list[object], value)
    records: list[ToolResultReplacementRecord] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TranscriptDecodeError(f"{key}[{index}] must be an object")
        mapping = cast(Mapping[str, object], item)
        records.append(
            ToolResultReplacementRecord(
                tool_use_id=_required_str(mapping, "tool_use_id"),
                replacement=_required_str(mapping, "replacement"),
                path=_required_str(mapping, "path"),
                original_size_chars=_int_field(mapping, "original_size_chars"),
            )
        )
    return tuple(records)


def _message_field(raw: Mapping[str, object], key: str) -> MessageParam:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise TranscriptDecodeError(f"{key} must be an object")
    mapping = cast(Mapping[object, object], value)
    message = {
        str(item_key): _json_clone(item_value)
        for item_key, item_value in mapping.items()
    }
    role = message.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        raise TranscriptDecodeError(f"{key}.role is invalid: {role!r}")
    if "content" not in message:
        raise TranscriptDecodeError(f"{key}.content is required")
    _validate_message_content(message["content"], f"{key}.content")
    return cast(MessageParam, message)


def _validate_message_content(value: object, key: str) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, list):
        items = cast(list[object], value)
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise TranscriptDecodeError(f"{key}[{index}] must be an object")
        return
    raise TranscriptDecodeError(f"{key} must be a string or list of objects")


def _dict_field(raw: Mapping[str, object], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise TranscriptDecodeError(f"{key} must be an object")
    mapping = cast(Mapping[object, object], value)
    return {
        str(item_key): _json_clone(item_value)
        for item_key, item_value in mapping.items()
    }


def _optional_dict_field(raw: Mapping[str, object], key: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TranscriptDecodeError(f"{key} must be an object or null")
    mapping = cast(Mapping[object, object], value)
    return {
        str(item_key): _json_clone(item_value)
        for item_key, item_value in mapping.items()
    }


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value == "":
        raise TranscriptDecodeError(f"{key} must be a non-empty string")
    return value


def _optional_str(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise TranscriptDecodeError(f"{key} must be a string or null")


def _str_field(raw: Mapping[str, object], key: str, *, default: str) -> str:
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    raise TranscriptDecodeError(f"{key} must be a string")


def _int_field(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TranscriptDecodeError(f"{key} must be an integer")
    return value


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TranscriptDecodeError(f"{key} must be an integer or null")
    return value


def _float_field(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TranscriptDecodeError(f"{key} must be numeric")
    return float(value)


def _bool_field(raw: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TranscriptDecodeError(f"{key} must be a boolean")


def _json_clone(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = [
    "CompactBoundaryEntry",
    "ContentReplacementEntry",
    "SessionMetadataEntry",
    "StreamEventEntry",
    "TombstoneEntry",
    "TranscriptDecodeError",
    "TranscriptEntry",
    "TranscriptEntryType",
    "TranscriptMessageEntry",
    "TranscriptScope",
    "current_transcript_time",
    "new_transcript_entry_id",
    "transcript_entry_from_dict",
    "transcript_entry_from_json",
    "transcript_entry_to_dict",
    "transcript_entry_to_json",
]
