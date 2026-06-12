"""MCP tool adaptation into Raygent's headless `Tool` contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.core.permissions import (
    AddPermissionRules,
    PermissionPassthrough,
    PermissionResult,
    PermissionRuleValue,
)
from raygent_harness.core.tool import (
    Tool,
    ToolCallEvent,
    ToolProgress,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationOk,
    ValidationResult,
    build_tool,
)
from raygent_harness.services.mcp import (
    McpClient,
    McpClientError,
    McpRegistrySnapshot,
    McpToolCallContext,
    McpToolCallRequest,
    McpToolCallResult,
    McpToolSchema,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import ToolCatalogProvider
    from raygent_harness.skills.models import SkillDefinition


MCP_TOOL_MAX_RESULT_SIZE_CHARS = 100_000
MCP_MAX_DESCRIPTION_LENGTH = 2048
MCP_TRUNCATION_MESSAGE = (
    "\n\n[OUTPUT TRUNCATED - MCP tool output exceeded Raygent's "
    "model-visible size limit. Use a more specific MCP tool query if the "
    "server supports filtering or pagination.]"
)


class McpToolInput(BaseModel):
    """Permissive local parser for externally defined MCP JSON Schema."""

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class McpToolingRuntime:
    """MCP client plus the snapshot used to build the current tool catalog."""

    client: McpClient
    snapshot: McpRegistrySnapshot
    tools: tuple[Tool, ...]


def create_mcp_tooling_runtime(
    *,
    client: McpClient,
    snapshot: McpRegistrySnapshot,
    max_result_size_chars: int = MCP_TOOL_MAX_RESULT_SIZE_CHARS,
) -> McpToolingRuntime:
    tools = tuple(
        build_mcp_tool(
            schema,
            client=client,
            max_result_size_chars=max_result_size_chars,
        )
        for schema in snapshot.available_tools()
    )
    return McpToolingRuntime(client=client, snapshot=snapshot, tools=tools)


def build_mcp_tool(
    schema: McpToolSchema,
    *,
    client: McpClient,
    max_result_size_chars: int = MCP_TOOL_MAX_RESULT_SIZE_CHARS,
) -> Tool:
    """Build one Raygent `Tool` for an MCP-exposed tool schema."""

    async def validate_input(
        _input: BaseModel,
        _ctx: ToolUseContext,
    ) -> ValidationResult:
        if not schema.available:
            return ValidationOk()
        return ValidationOk()

    async def check_permissions(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: object,
    ) -> PermissionResult:
        return PermissionPassthrough(
            message="MCPTool requires permission.",
            suggestions=(
                AddPermissionRules(
                    destination="localSettings",
                    rules=(PermissionRuleValue(tool_name=schema.raygent_name),),
                    behavior="allow",
                ),
            ),
        )

    async def call(
        input_: BaseModel,
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolCallEvent]:
        _raise_if_aborted(ctx)
        yield ToolProgress(
            message=(
                "Calling MCP tool "
                f"{schema.identity.server_normalized_name}/{schema.identity.tool_name}"
            ),
            data={
                "type": "mcp_progress",
                "status": "started",
                "server_name": schema.identity.server_name,
                "tool_name": schema.identity.tool_name,
            },
        )
        request = McpToolCallRequest(
            identity=schema.identity,
            arguments=_mcp_arguments_from_input(input_),
            tool_use_id=ctx.tool_use_id,
        )
        started = asyncio.get_running_loop().time()
        progress_queue: asyncio.Queue[ToolProgress] = asyncio.Queue()

        async def emit_progress(progress: FrozenJson) -> None:
            progress_queue.put_nowait(_mcp_progress_event(schema, progress))

        call_context = McpToolCallContext(
            abort_event=ctx.abort_event,
            emit_progress=emit_progress,
            handle_elicitation=ctx.handle_elicitation,
        )
        call_task = asyncio.create_task(client.call_tool(request, call_context))
        abort_task = asyncio.create_task(ctx.abort_event.wait())
        try:
            while not call_task.done():
                progress_task = asyncio.create_task(progress_queue.get())
                done, pending = await asyncio.wait(
                    {call_task, abort_task, progress_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if abort_task in done:
                    call_task.cancel()
                    raise asyncio.CancelledError()
                if progress_task in done:
                    yield progress_task.result()
                for task in pending:
                    if task is progress_task:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                if call_task in done:
                    break
            while not progress_queue.empty():
                yield progress_queue.get_nowait()
            result = await call_task
        except asyncio.CancelledError:
            call_task.cancel()
            raise
        except McpClientError as exc:
            yield _mcp_status_progress_event(schema, "failed", started)
            yield ToolResult(
                content=_mcp_error_content(schema, str(exc)),
                is_error=True,
            )
            return
        except Exception as exc:
            yield _mcp_status_progress_event(schema, "failed", started)
            yield ToolResult(
                content=_mcp_error_content(schema, f"{type(exc).__name__}: {exc}"),
                is_error=True,
            )
            return
        finally:
            abort_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task
        _raise_if_aborted(ctx)
        yield _mcp_status_progress_event(schema, "completed", started)
        yield ToolResult(
            content=map_mcp_tool_result_content(
                result,
                max_result_size_chars=max_result_size_chars,
            ),
            is_error=result.is_error,
        )

    return build_tool(
        ToolSpec(
            name=schema.raygent_name,
            description=schema.description,
            input_model=McpToolInput,
            input_schema=normalize_mcp_input_schema(schema.input_schema),
            call=call,
            search_hint=schema.search_hint,
            prompt=_truncated_description(schema.description),
            check_permissions=check_permissions,
            validate_input=validate_input,
            is_concurrency_safe=_annotation_bool(schema, "readOnlyHint"),
            is_read_only=_annotation_bool(schema, "readOnlyHint"),
            is_destructive=_annotation_bool(schema, "destructiveHint"),
            is_open_world=_annotation_bool(schema, "openWorldHint"),
            should_defer=not schema.always_load,
            always_load=schema.always_load,
            max_result_size_chars=max_result_size_chars,
        )
    )


def create_mcp_tools_catalog_provider(
    *,
    client: McpClient,
    upstream: ToolCatalogProvider | None = None,
    max_result_size_chars: int = MCP_TOOL_MAX_RESULT_SIZE_CHARS,
) -> ToolCatalogProvider:
    """Catalog provider that appends tools from the current MCP snapshot."""

    async def provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        tools = await upstream(config, ctx, skills) if upstream is not None else config.tools
        if tools is None:
            tools = config.tools
        snapshot = await client.registry_snapshot()
        runtime = create_mcp_tooling_runtime(
            client=client,
            snapshot=snapshot,
            max_result_size_chars=max_result_size_chars,
        )
        existing_tools = _without_colliding_tools(tuple(tools), runtime.tools)
        return (*existing_tools, *runtime.tools)

    return provider


def create_pending_mcp_servers_provider(
    client: McpClient,
) -> Callable[[ToolUseContext], Awaitable[tuple[str, ...]]]:
    async def provider(ctx: ToolUseContext) -> tuple[str, ...]:
        return await pending_mcp_servers_from_client(client, ctx)

    return provider


async def pending_mcp_servers_from_client(
    client: McpClient,
    _ctx: ToolUseContext,
) -> tuple[str, ...]:
    snapshot = await client.registry_snapshot()
    return snapshot.pending_server_names()


def normalize_mcp_input_schema(input_schema: FrozenJson) -> FrozenJson:
    """Return an object-shaped JSON Schema for provider tool calls."""

    raw = thaw_json(input_schema)
    if not isinstance(raw, Mapping):
        return freeze_json({"type": "object", "properties": {}})

    schema = dict(cast(Mapping[str, object], raw))
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), Mapping):
        schema["properties"] = {}
    return freeze_json(schema)


def map_mcp_tool_result_content(
    result: McpToolCallResult,
    *,
    max_result_size_chars: int = MCP_TOOL_MAX_RESULT_SIZE_CHARS,
) -> str | list[dict[str, Any]]:
    raw = thaw_json(result.content)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return _truncate_text(raw, max_result_size_chars)
    raw_items = cast(list[object], raw) if isinstance(raw, list) else None
    if raw_items is not None and all(isinstance(item, Mapping) for item in raw_items):
        return _truncate_blocks(
            [dict(cast(Mapping[str, object], item)) for item in raw_items],
            max_result_size_chars,
        )
    return _truncate_text(
        json.dumps(raw, sort_keys=True, ensure_ascii=False),
        max_result_size_chars,
    )


def _mcp_arguments_from_input(input_: BaseModel) -> FrozenJson:
    return freeze_json(input_.model_dump(mode="python"))


def _annotation_bool(schema: McpToolSchema, name: str) -> bool:
    annotations = thaw_json(schema.annotations)
    if not isinstance(annotations, Mapping):
        return False
    annotation_map = cast(Mapping[str, object], annotations)
    return annotation_map.get(name) is True


def _truncated_description(description: str) -> str:
    if len(description) <= MCP_MAX_DESCRIPTION_LENGTH:
        return description
    return description[:MCP_MAX_DESCRIPTION_LENGTH] + "... [truncated]"


def _mcp_error_content(schema: McpToolSchema, message: str) -> str:
    return (
        "<tool_use_error>"
        f"Error calling MCP tool {schema.raygent_name}: {message}"
        "</tool_use_error>"
    )


def _mcp_progress_event(schema: McpToolSchema, progress: FrozenJson) -> ToolProgress:
    data = thaw_json(progress)
    payload = (
        dict(cast(Mapping[str, Any], data))
        if isinstance(data, Mapping)
        else {"detail": data}
    )
    payload.setdefault("type", "mcp_progress")
    payload.setdefault("status", "progress")
    payload.setdefault("server_name", schema.identity.server_name)
    payload.setdefault("tool_name", schema.identity.tool_name)
    return ToolProgress(
        message=(
            "MCP tool "
            f"{schema.identity.server_normalized_name}/{schema.identity.tool_name}: "
            f"{payload['status']}"
        ),
        data=payload,
    )


def _mcp_status_progress_event(
    schema: McpToolSchema,
    status: str,
    started: float,
) -> ToolProgress:
    return ToolProgress(
        message=(
            f"{status.capitalize()} MCP tool "
            f"{schema.identity.server_normalized_name}/{schema.identity.tool_name}"
        ),
        data={
            "type": "mcp_progress",
            "status": status,
            "server_name": schema.identity.server_name,
            "tool_name": schema.identity.tool_name,
            "elapsed_time_ms": int((asyncio.get_running_loop().time() - started) * 1000),
        },
    )


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    suffix = MCP_TRUNCATION_MESSAGE
    return value[: max(0, max_chars - len(suffix))] + suffix


def _truncate_blocks(
    blocks: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    remaining = max_chars
    for block in blocks:
        if remaining <= 0:
            break
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            truncated = _truncate_text(text, remaining)
            result.append({**block, "text": truncated})
            remaining -= len(truncated)
            if truncated != text:
                break
            continue
        estimated_size = len(json.dumps(block, sort_keys=True, ensure_ascii=False))
        if estimated_size > remaining:
            result.append({"type": "text", "text": MCP_TRUNCATION_MESSAGE.strip()})
            break
        result.append(block)
        remaining -= estimated_size
    if not result and blocks:
        return [{"type": "text", "text": MCP_TRUNCATION_MESSAGE.strip()}]
    return result


def _without_colliding_tools(
    tools: tuple[Tool, ...],
    runtime_tools: tuple[Tool, ...],
) -> tuple[Tool, ...]:
    reserved_names: set[str] = set()
    for tool in runtime_tools:
        reserved_names.update(_tool_identity_names(tool))
    return tuple(
        tool for tool in tools if _tool_identity_names(tool).isdisjoint(reserved_names)
    )


def _tool_identity_names(tool: Tool) -> set[str]:
    return {tool.name, *tool.aliases}


def _raise_if_aborted(ctx: ToolUseContext) -> None:
    if ctx.abort_event.is_set():
        raise asyncio.CancelledError()


__all__ = [
    "MCP_MAX_DESCRIPTION_LENGTH",
    "MCP_TOOL_MAX_RESULT_SIZE_CHARS",
    "MCP_TRUNCATION_MESSAGE",
    "McpToolInput",
    "McpToolingRuntime",
    "build_mcp_tool",
    "create_mcp_tooling_runtime",
    "create_mcp_tools_catalog_provider",
    "create_pending_mcp_servers_provider",
    "map_mcp_tool_result_content",
    "normalize_mcp_input_schema",
    "pending_mcp_servers_from_client",
]
