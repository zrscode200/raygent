from __future__ import annotations

from raygent_harness.services.mcp import (
    McpRegistrySnapshot,
    McpServerConfig,
    McpServerIdentity,
    McpServerState,
    McpToolIdentity,
    allocate_mcp_server_identities,
    allocate_mcp_tool_schemas,
    build_mcp_tool_name,
    is_mcp_tool_name,
    mcp_prefix,
    mcp_server_name_for_tool,
    normalize_mcp_name,
    parse_mcp_tool_name,
)
from raygent_harness.tools.tool_search import parse_tool_name


def test_mcp_name_normalization_and_roundtrip_parsing() -> None:
    assert normalize_mcp_name("Git Hub.Server") == "Git_Hub_Server"
    assert normalize_mcp_name(" claude.ai  Browser  ") == "claude_ai_Browser"
    assert normalize_mcp_name("***") == "unnamed"
    assert "__" not in normalize_mcp_name("a..b")

    assert mcp_prefix("Git Hub.Server") == "mcp__Git_Hub_Server__"
    name = build_mcp_tool_name("Git Hub.Server", "create issue")
    assert name == "mcp__Git_Hub_Server__create_issue"

    parsed = parse_mcp_tool_name(name)
    assert parsed is not None
    assert parsed.server_name == "Git_Hub_Server"
    assert parsed.tool_name == "create_issue"
    assert is_mcp_tool_name(name)
    assert mcp_server_name_for_tool(name) == "Git_Hub_Server"
    assert mcp_server_name_for_tool("mcp__Git_Hub_Server") is None
    assert parse_mcp_tool_name("Read") is None

    long_server = "a" * 63 + ".tail"
    long_name = build_mcp_tool_name(long_server, "search")
    assert long_name.count("__") == 2
    long_parsed = parse_mcp_tool_name(long_name)
    assert long_parsed is not None
    assert long_parsed.server_name == normalize_mcp_name(long_server)
    assert long_parsed.tool_name == "search"


def test_parse_mcp_tool_name_accepts_server_level_and_legacy_tool_delimiters() -> None:
    server_level = parse_mcp_tool_name("mcp__github")
    assert server_level is not None
    assert server_level.server_name == "github"
    assert server_level.tool_name is None

    legacy = parse_mcp_tool_name("mcp__github__issue__create")
    assert legacy is not None
    assert legacy.server_name == "github"
    assert legacy.tool_name == "issue__create"

    legacy_empty_component = parse_mcp_tool_name("mcp__github__issue____create")
    assert legacy_empty_component is not None
    assert legacy_empty_component.server_name == "github"
    assert legacy_empty_component.tool_name == "issue____create"


def test_allocate_server_identities_dedupes_normalized_collisions() -> None:
    configs = (
        McpServerConfig(name="Git Hub", source="user"),
        McpServerConfig(name="Git.Hub", source="project"),
        McpServerConfig(name="Git__Hub", source="workspace"),
    )

    identities = allocate_mcp_server_identities(configs)

    assert [identity.normalized_name for identity in identities] == [
        "Git_Hub",
        "Git_Hub_2",
        "Git_Hub_3",
    ]
    assert identities[0].to_metadata() == {
        "name": "Git Hub",
        "normalized_name": "Git_Hub",
        "source": "user",
    }


def test_allocate_server_identities_dedupes_without_delimiter_collisions() -> None:
    colliding_name = "a" * 61 + ".bc"
    configs = (
        McpServerConfig(name=colliding_name),
        McpServerConfig(name=colliding_name),
    )

    identities = allocate_mcp_server_identities(configs)

    assert identities[0].normalized_name == "a" * 61 + "_bc"
    assert identities[1].normalized_name == "a" * 61 + "_2"
    for identity in identities:
        raygent_name = build_mcp_tool_name(identity.normalized_name, "search")
        assert raygent_name.count("__") == 2
        assert mcp_server_name_for_tool(raygent_name) == identity.normalized_name


def test_allocate_tool_schemas_dedupes_tools_and_freezes_metadata() -> None:
    server = McpServerIdentity(name="Git Hub", normalized_name="Git_Hub")

    tools = allocate_mcp_tool_schemas(
        server,
        (
            {
                "name": "create issue",
                "description": "Create an issue",
                "input_schema": {"type": "object", "properties": {"title": {}}},
                "annotations": {"readOnlyHint": True},
                "search_hint": "issue tracker",
                "always_load": True,
                "metadata": {"nested": ["value"]},
            },
            {"name": "create.issue"},
        ),
    )

    assert [tool.raygent_name for tool in tools] == [
        "mcp__Git_Hub__create_issue",
        "mcp__Git_Hub__create_issue_2",
    ]
    assert tools[0].description == "Create an issue"
    assert tools[0].search_hint == "issue tracker"
    assert tools[0].always_load is True
    assert tools[0].input_schema == {
        "type": "object",
        "properties": {"title": {}},
    }
    assert tools[0].annotations == {"readOnlyHint": True}
    assert tools[0].metadata == {"nested": ("value",)}
    assert tools[0].to_metadata()["raygent_name"] == "mcp__Git_Hub__create_issue"


def test_mcp_registry_snapshot_filters_by_status_and_availability() -> None:
    connected_identity = McpServerIdentity(name="GitHub", normalized_name="github")
    available = allocate_mcp_tool_schemas(connected_identity, ({"name": "search"},))[0]
    unavailable = allocate_mcp_tool_schemas(
        connected_identity,
        ({"name": "mutate", "available": False, "error": "disabled"},),
    )[0]
    connected = McpServerState(
        identity=connected_identity,
        status="connected",
        capabilities=("tools",),
        tools=(available, unavailable),
    )
    pending = McpServerState(
        identity=McpServerIdentity(name="Linear", normalized_name="linear"),
        status="pending",
        reconnect_attempt=1,
        max_reconnect_attempts=3,
    )
    needs_auth = McpServerState(
        identity=McpServerIdentity(name="Slack", normalized_name="slack"),
        status="needs_auth",
    )
    needs_client_registration = McpServerState(
        identity=McpServerIdentity(name="Jira", normalized_name="jira"),
        status="needs_client_registration",
    )
    failed = McpServerState(
        identity=McpServerIdentity(name="Broken", normalized_name="broken"),
        status="failed",
        error="boom",
    )
    no_tools_capability = McpServerState(
        identity=McpServerIdentity(name="No Tools", normalized_name="no_tools"),
        status="connected",
        capabilities=(),
        tools=allocate_mcp_tool_schemas(
            McpServerIdentity(name="No Tools", normalized_name="no_tools"),
            ({"name": "stale"},),
        ),
    )
    disabled = McpServerState(
        identity=McpServerIdentity(name="Disabled", normalized_name="disabled"),
        status="disabled",
    )

    snapshot = McpRegistrySnapshot(
        (
            connected,
            pending,
            needs_auth,
            needs_client_registration,
            failed,
            no_tools_capability,
            disabled,
        )
    )

    assert snapshot.connected_servers() == (connected, no_tools_capability)
    assert snapshot.pending_server_names() == ("linear",)
    assert snapshot.auth_required_server_names() == ("slack",)
    assert snapshot.client_registration_required_server_names() == ("jira",)
    assert snapshot.available_tools() == (available,)
    assert snapshot.tool_by_raygent_name("mcp__github__search") == available
    assert snapshot.tool_by_raygent_name("mcp__github__mutate") is None
    assert snapshot.tool_by_raygent_name("mcp__no_tools__stale") is None
    assert snapshot.to_metadata() == {
        "server_count": 7,
        "connected_server_count": 2,
        "available_tool_count": 1,
        "pending_servers": ["linear"],
        "auth_required_servers": ["slack"],
        "client_registration_required_servers": ["jira"],
    }


def test_tool_identity_from_raygent_name_and_tool_search_reuse_shared_parser() -> None:
    identity = McpToolIdentity.from_raygent_name("mcp__github__create_issue")

    assert identity is not None
    assert identity.permission_name() == "mcp__github__create_issue"
    assert identity.server_normalized_name == "github"
    assert identity.tool_normalized_name == "create_issue"

    parts, full, is_mcp = parse_tool_name("mcp__github__create_issue")
    assert is_mcp is True
    assert parts == ("github", "create", "issue")
    assert full == "github create issue"
