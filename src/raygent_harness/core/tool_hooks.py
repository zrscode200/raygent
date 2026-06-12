"""Headless tool hook protocols.

PreToolUse/PostToolUse hooks run inside the single-tool
execution lifecycle. Raygent keeps the same kernel seam but leaves hook loading
to adapters/skills; an empty hook list is the default.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from raygent_harness.core.permissions import PermissionResult

if TYPE_CHECKING:
    from pydantic import BaseModel

    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.model_adapter import ToolUseBlock
    from raygent_harness.core.tool import Tool, ToolUseContext


@dataclass(frozen=True)
class PreToolUseContext:
    tool: Tool
    tool_use: ToolUseBlock
    input: BaseModel
    tool_use_context: ToolUseContext
    assistant_message: MessageParam


@dataclass(frozen=True)
class PreToolUseResult:
    """Result from a PreToolUse hook.

    `updated_input` without a permission result flows into normal permission
    resolution. `permission_result` flows through
    `QueryDeps.resolve_hook_tool_permission(...)`, which still enforces deny
    rules so a hook allow cannot bypass policy.
    """

    updated_input: Mapping[str, object] | None = None
    permission_result: PermissionResult | None = None
    additional_messages: tuple[MessageParam, ...] = ()
    should_prevent_continuation: bool = False
    stop_reason: str | None = None
    stop: bool = False


class PreToolUseHook(Protocol):
    def __call__(
        self,
        context: PreToolUseContext,
        /,
    ) -> Awaitable[PreToolUseResult | None]:
        ...


@dataclass(frozen=True)
class PostToolUseContext:
    tool: Tool
    tool_use: ToolUseBlock
    input: BaseModel
    tool_use_context: ToolUseContext
    assistant_message: MessageParam
    result_message: MessageParam


class PostToolUseHook(Protocol):
    def __call__(self, context: PostToolUseContext, /) -> Awaitable[None]:
        ...


@dataclass(frozen=True)
class PostToolUseFailureContext:
    tool: Tool | None
    tool_use: ToolUseBlock
    tool_use_context: ToolUseContext
    assistant_message: MessageParam
    error_message: str
    input: BaseModel | None = None


class PostToolUseFailureHook(Protocol):
    def __call__(self, context: PostToolUseFailureContext, /) -> Awaitable[None]:
        ...


@dataclass(frozen=True)
class PreToolUseHookOutcome:
    input: BaseModel
    permission_result: PermissionResult | None = None
    additional_messages: tuple[MessageParam, ...] = ()
    should_prevent_continuation: bool = False
    stop_reason: str | None = None
    stop: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple[str, ...])


__all__ = [
    "PostToolUseContext",
    "PostToolUseFailureContext",
    "PostToolUseFailureHook",
    "PostToolUseHook",
    "PreToolUseContext",
    "PreToolUseHook",
    "PreToolUseHookOutcome",
    "PreToolUseResult",
]
