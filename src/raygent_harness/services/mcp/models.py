"""Provider-neutral MCP identity and server-state models.


This module intentionally contains no MCP transport, credentials, process
management, OAuth, or live client objects. It is the replay-safe identity/state
substrate that later MCP tool adapters can consume.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from raygent_harness.core.model_types import FrozenJson, freeze_json

MCP_TOOL_PREFIX = "mcp"
MCP_NAME_MAX_LENGTH = 64

McpServerStatus = Literal[
    "connected",
    "pending",
    "failed",
    "needs_auth",
    "needs_client_registration",
    "disabled",
]
McpServerSource = Literal[
    "user",
    "project",
    "workspace",
    "plugin",
    "sdk",
    "runtime",
    "unknown",
]
McpCapability = Literal["tools", "resources", "prompts", "logging"]

_INVALID_MCP_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_REPEATED_UNDERSCORES = re.compile(r"_+")


def _empty_json_object() -> FrozenJson:
    return {}


def normalize_mcp_name(name: str) -> str:
    """Normalize MCP server/tool names for `mcp__server__tool` names.

    Raygent replaces invalid API characters with underscores.
    Raygent also collapses repeated underscores and strips edge underscores for
    every name, not only product-specific server names, so normalized names never
    contain the `__` delimiter used for parsing.
    """

    normalized = _INVALID_MCP_NAME_CHARS.sub("_", name.strip())
    normalized = _REPEATED_UNDERSCORES.sub("_", normalized).strip("_")
    if not normalized:
        normalized = "unnamed"
    truncated = normalized[:MCP_NAME_MAX_LENGTH].strip("_")
    return truncated or "unnamed"


def mcp_prefix(server_name: str) -> str:
    return f"{MCP_TOOL_PREFIX}__{normalize_mcp_name(server_name)}__"


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return f"{mcp_prefix(server_name)}{normalize_mcp_name(tool_name)}"


@dataclass(frozen=True, slots=True)
class McpNameInfo:
    server_name: str
    tool_name: str | None = None


def parse_mcp_tool_name(tool_name: str) -> McpNameInfo | None:
    """Parse `mcp__server` or `mcp__server__tool` names.

    Legacy names with additional `__` in the tool portion are accepted by
    joining the remainder, matching Raygent's tolerant parser contract.
    Newly generated Raygent names never contain `__` inside normalized server or
    tool components.
    """

    parts = tool_name.split("__")
    if len(parts) < 2 or parts[0] != MCP_TOOL_PREFIX or not parts[1]:
        return None
    if len(parts) == 2:
        return McpNameInfo(server_name=parts[1])
    joined_tool = "__".join(parts[2:])
    return McpNameInfo(server_name=parts[1], tool_name=joined_tool or None)


def is_mcp_tool_name(tool_name: str) -> bool:
    return parse_mcp_tool_name(tool_name) is not None


def mcp_server_name_for_tool(tool_name: str) -> str | None:
    info = parse_mcp_tool_name(tool_name)
    if info is None or info.tool_name is None:
        return None
    return info.server_name


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """Replay-safe MCP server configuration facts.

    Transport commands, URLs, headers, OAuth credentials, and live client handles
    belong in adapter-specific config/transport objects, not this kernel model.
    """

    name: str
    enabled: bool = True
    source: McpServerSource = "unknown"
    request_timeout_s: float | None = None
    metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @property
    def normalized_name(self) -> str:
        return normalize_mcp_name(self.name)


@dataclass(frozen=True, slots=True)
class McpServerIdentity:
    """Stable server identity after normalized-name collision resolution."""

    name: str
    normalized_name: str
    display_name: str | None = None
    source: McpServerSource = "unknown"

    @classmethod
    def from_config(
        cls,
        config: McpServerConfig,
        *,
        normalized_name: str | None = None,
    ) -> McpServerIdentity:
        return cls(
            name=config.name,
            normalized_name=normalized_name or config.normalized_name,
            display_name=config.name,
            source=config.source,
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class McpToolIdentity:
    """Identity for one MCP-exposed tool."""

    server_name: str
    tool_name: str
    server_normalized_name: str
    tool_normalized_name: str
    raygent_name: str
    display_name: str | None = None

    @classmethod
    def from_names(
        cls,
        server_name: str,
        tool_name: str,
        *,
        server_normalized_name: str | None = None,
        tool_normalized_name: str | None = None,
        raygent_name: str | None = None,
        display_name: str | None = None,
    ) -> McpToolIdentity:
        normalized_server = server_normalized_name or normalize_mcp_name(server_name)
        normalized_tool = tool_normalized_name or normalize_mcp_name(tool_name)
        return cls(
            server_name=server_name,
            tool_name=tool_name,
            server_normalized_name=normalized_server,
            tool_normalized_name=normalized_tool,
            raygent_name=raygent_name
            or f"{MCP_TOOL_PREFIX}__{normalized_server}__{normalized_tool}",
            display_name=display_name or tool_name,
        )

    @classmethod
    def from_raygent_name(
        cls,
        raygent_name: str,
        *,
        server_name: str | None = None,
        tool_name: str | None = None,
    ) -> McpToolIdentity | None:
        info = parse_mcp_tool_name(raygent_name)
        if info is None or info.tool_name is None:
            return None
        return cls.from_names(
            server_name or info.server_name,
            tool_name or info.tool_name,
            server_normalized_name=info.server_name,
            tool_normalized_name=info.tool_name,
            raygent_name=raygent_name,
        )

    def permission_name(self) -> str:
        """Fully qualified name for permission matching."""

        return self.raygent_name

    def to_metadata(self) -> dict[str, str]:
        return {
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "server_normalized_name": self.server_normalized_name,
            "tool_normalized_name": self.tool_normalized_name,
            "raygent_name": self.raygent_name,
        }


@dataclass(frozen=True, slots=True)
class McpToolSchema:
    """Replay-safe MCP tool descriptor before Raygent `Tool` adaptation."""

    identity: McpToolIdentity
    description: str = ""
    input_schema: FrozenJson = field(default_factory=_empty_json_object)
    annotations: FrozenJson = field(default_factory=_empty_json_object)
    search_hint: str | None = None
    always_load: bool = False
    available: bool = True
    error: str | None = None
    metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))
        object.__setattr__(self, "annotations", freeze_json(self.annotations))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @property
    def raygent_name(self) -> str:
        return self.identity.raygent_name

    def to_metadata(self) -> dict[str, object]:
        return {
            **self.identity.to_metadata(),
            "available": self.available,
            "always_load": self.always_load,
            "has_error": self.error is not None,
            "has_annotations": bool(self.annotations),
        }


@dataclass(frozen=True, slots=True)
class McpServerState:
    """Replay-safe lifecycle state for one MCP server."""

    identity: McpServerIdentity
    status: McpServerStatus
    config: McpServerConfig | None = None
    capabilities: tuple[McpCapability, ...] = ()
    tools: tuple[McpToolSchema, ...] = ()
    instructions: str | None = None
    error: str | None = None
    reconnect_attempt: int | None = None
    max_reconnect_attempts: int | None = None
    updated_at: float | None = None
    metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def normalized_name(self) -> str:
        return self.identity.normalized_name

    def is_available(self) -> bool:
        return self.status == "connected"

    def pending_for_model(self) -> bool:
        return self.status == "pending"

    def to_metadata(self) -> dict[str, object]:
        return {
            **self.identity.to_metadata(),
            "status": self.status,
            "capabilities": list(self.capabilities),
            "tool_count": len(self.tools),
            "has_error": self.error is not None,
            "reconnect_attempt": self.reconnect_attempt,
            "max_reconnect_attempts": self.max_reconnect_attempts,
        }


@dataclass(frozen=True, slots=True)
class McpRegistrySnapshot:
    """Immutable MCP server/tool state snapshot for one query turn."""

    servers: tuple[McpServerState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "servers", tuple(self.servers))

    def connected_servers(self) -> tuple[McpServerState, ...]:
        return tuple(server for server in self.servers if server.status == "connected")

    def pending_server_names(self) -> tuple[str, ...]:
        return tuple(
            server.normalized_name for server in self.servers if server.pending_for_model()
        )

    def auth_required_server_names(self) -> tuple[str, ...]:
        return tuple(
            server.normalized_name
            for server in self.servers
            if server.status == "needs_auth"
        )

    def client_registration_required_server_names(self) -> tuple[str, ...]:
        return tuple(
            server.normalized_name
            for server in self.servers
            if server.status == "needs_client_registration"
        )

    def available_tools(self) -> tuple[McpToolSchema, ...]:
        tools: list[McpToolSchema] = []
        for server in self.connected_servers():
            if "tools" not in server.capabilities:
                continue
            tools.extend(tool for tool in server.tools if tool.available)
        return tuple(tools)

    def tool_by_raygent_name(self, raygent_name: str) -> McpToolSchema | None:
        for tool in self.available_tools():
            if tool.raygent_name == raygent_name:
                return tool
        return None

    def to_metadata(self) -> dict[str, object]:
        return {
            "server_count": len(self.servers),
            "connected_server_count": len(self.connected_servers()),
            "available_tool_count": len(self.available_tools()),
            "pending_servers": list(self.pending_server_names()),
            "auth_required_servers": list(self.auth_required_server_names()),
            "client_registration_required_servers": list(
                self.client_registration_required_server_names()
            ),
        }


def allocate_mcp_server_identities(
    configs: Sequence[McpServerConfig],
) -> tuple[McpServerIdentity, ...]:
    """Return deterministic unique normalized server identities."""

    used: set[str] = set()
    identities: list[McpServerIdentity] = []
    for config in configs:
        unique = _dedupe_name(config.normalized_name, used)
        identities.append(McpServerIdentity.from_config(config, normalized_name=unique))
    return tuple(identities)


def allocate_mcp_tool_schemas(
    server_identity: McpServerIdentity,
    tool_descriptors: Iterable[Mapping[str, object]],
) -> tuple[McpToolSchema, ...]:
    """Build collision-free tool schemas for one server.

    Expected descriptor keys are `name`, optional `description`, `input_schema`,
    `search_hint`, `always_load`, `available`, `error`, and `metadata`. Unknown
    keys are ignored here; transport adapters can keep their raw payloads out of
    replay state or place sanitized facts in `metadata`.
    """

    used: set[str] = set()
    schemas: list[McpToolSchema] = []
    for descriptor in tool_descriptors:
        raw_name = str(descriptor.get("name") or "unnamed")
        normalized_tool = _dedupe_name(normalize_mcp_name(raw_name), used)
        identity = McpToolIdentity.from_names(
            server_identity.name,
            raw_name,
            server_normalized_name=server_identity.normalized_name,
            tool_normalized_name=normalized_tool,
        )
        schemas.append(
            McpToolSchema(
                identity=identity,
                description=str(descriptor.get("description") or ""),
                input_schema=freeze_json(descriptor.get("input_schema") or {}),
                annotations=freeze_json(descriptor.get("annotations") or {}),
                search_hint=_optional_str(descriptor.get("search_hint")),
                always_load=bool(descriptor.get("always_load", False)),
                available=bool(descriptor.get("available", True)),
                error=_optional_str(descriptor.get("error")),
                metadata=freeze_json(descriptor.get("metadata") or {}),
            )
        )
    return tuple(schemas)


def _dedupe_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    counter = 2
    while True:
        suffix = f"_{counter}"
        stem = base[: MCP_NAME_MAX_LENGTH - len(suffix)].rstrip("_")
        candidate = f"{stem or 'unnamed'}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "MCP_NAME_MAX_LENGTH",
    "MCP_TOOL_PREFIX",
    "McpCapability",
    "McpNameInfo",
    "McpRegistrySnapshot",
    "McpServerConfig",
    "McpServerIdentity",
    "McpServerSource",
    "McpServerState",
    "McpServerStatus",
    "McpToolIdentity",
    "McpToolSchema",
    "allocate_mcp_server_identities",
    "allocate_mcp_tool_schemas",
    "build_mcp_tool_name",
    "is_mcp_tool_name",
    "mcp_prefix",
    "mcp_server_name_for_tool",
    "normalize_mcp_name",
    "parse_mcp_tool_name",
]
