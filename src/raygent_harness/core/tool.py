"""Tool contract — multi-axis Tool protocol + ToolUseContext session bag.

Raygent keeps the tool contract headless: execution, capability predicates,
schema duality, two-stage gating (validate -> check_permissions), deferred-load
hooks, and fail-closed defaults via `build_tool`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from raygent_harness.core.file_state import (
    ReadFileStateCache,
    create_read_file_state_cache,
)
from raygent_harness.core.model_types import FrozenJson, freeze_json
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    PermissionResult,
    ToolPermissionContext,
    empty_tool_permission_context,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.deps import QueryDeps
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.observability import KernelEventContext
    from raygent_harness.services.file_media import PdfDocumentService

# ---------------------------------------------------------------------------
# Capability axes — scheduling & UX hints the harness evaluates on a Tool.
# ---------------------------------------------------------------------------

InterruptBehavior = Literal["cancel", "block"]
"""How the harness should treat this tool on user interrupt.

- cancel: cancel the in-flight call and discard its result.
- block: let the current call finish; the new user message waits.
"""

type InputCapability = Callable[[BaseModel], bool]
"""Input-sensitive capability predicate such as read-only or concurrency-safe."""

type InputCapabilitySpec = bool | InputCapability
"""ToolSpec ergonomics: accept a constant bool or an input-sensitive predicate."""

type NullaryCapability = Callable[[], bool]
type NullaryCapabilitySpec = bool | NullaryCapability

type InterruptBehaviorFn = Callable[[], InterruptBehavior]
type InterruptBehaviorSpec = InterruptBehavior | InterruptBehaviorFn

type ValidateInputFn = Callable[[BaseModel, "ToolUseContext"], Awaitable["ValidationResult"]]
type CheckPermissionsFn = Callable[
    [BaseModel, "ToolUseContext", "ToolPermissionContext"],
    Awaitable["PermissionResult"],
]
type PromptTextProvider = Callable[["ToolPromptContext | ToolUseContext | None"], Awaitable[str]]
type PromptTextProviderSpec = str | PromptTextProvider
type DescriptionProvider = Callable[
    [BaseModel, "ToolDescriptionContext"], Awaitable[str]
]

PermissionAllow = PermissionAllowDecision
PermissionAsk = PermissionAskDecision
PermissionDeny = PermissionDenyDecision


# ---------------------------------------------------------------------------
# Validation protocol — model-facing gate. Cheap; no user involvement.
# ---------------------------------------------------------------------------


class ValidationOk(BaseModel):
    result: Literal["ok"] = "ok"


class ValidationError(BaseModel):
    result: Literal["error"] = "error"
    message: str
    """Surfaced back to the model as the tool result (model-fixable)."""


ValidationResult = ValidationOk | ValidationError


# ---------------------------------------------------------------------------
# ToolUseContext — the session-scoped bag every tool call receives.
# ---------------------------------------------------------------------------


@dataclass
class QueryTracking:
    """Chain+depth for subagent-tree observability. `{chainId, depth}` in TS."""

    chain_id: str
    depth: int


@dataclass
class ContentReplacementState:
    """Oversized tool outputs persisted to disk; replacement markers in messages.

    Preserves prompt-cache coherence: the marker is stable across turns, so the
    API-visible message history doesn't drift and invalidate the cache.
    """

    max_result_size_chars: int
    """Per-call output size cap. Above this, outputs are replaced with a marker."""

    replaced_outputs_dir: str
    """Directory where replaced outputs land as files."""

    replacements: dict[str, str] = field(default_factory=dict[str, str])
    """tool_use_id -> stable replacement text. Re-applied every turn."""

    seen_ids: set[str] = field(default_factory=set[str])
    """Tool result ids already evaluated. Unreplaced seen ids are frozen so
    future budget passes don't change cached prefix content."""


@dataclass
class ToolUseContext:
    """Session-scoped bag handed to every tool call.

    Tool-use context carries only the axes Raygent needs. Extend as we add
    subsystems such as memory, compaction, and human-in-the-loop controls.
    """

    # Identity / lifecycle
    session_id: str
    agent_id: str | None
    """None = main thread. Set = running under a subagent/teammate."""

    # Cancellation — cooperative, parent→child cascade via linked events.
    abort_event: asyncio.Event

    # System prompt (frozen at turn start; mutating would bust prompt cache)
    rendered_system_prompt: str
    """DO NOT mutate mid-turn. Kept read-only by convention."""

    # Working directory
    cwd: str

    # Current API-visible transcript for this stage of the turn.
    messages: list[MessageParam] = field(default_factory=list["MessageParam"])
    """Updated by QueryEngine/query() at provider/model/tool boundaries.

    Reference updates `toolUseContext.messages` after the context pipeline so
    tools can reason over the exact compacted transcript the model just saw.
    """

    # Current turn tool catalog and permission context.
    tools: tuple[Tool, ...] = ()
    """Full runtime catalog for this stage of the turn.

    Deferred tools may be hidden from the model request, but ToolSearch needs
    the full catalog to return selected schemas. Reference passes `tools` in
    the tool-call options bag; Raygent carries it on the headless context.
    """

    permission_context: ToolPermissionContext = field(
        default_factory=empty_tool_permission_context
    )
    """Permission state snapshot for dynamic prompts and search scoring."""

    discovered_tool_names: frozenset[str] = field(
        default_factory=lambda: frozenset[str]()
    )
    """Deferred tool schemas discovered through prior ToolSearch results.

    Reference stores this on compact-boundary metadata when compaction removes
    tool_reference blocks. Raygent keeps compact boundaries out-of-band, so the
    headless loop carries the discovered set directly on context.
    """

    # Content replacement for oversized outputs
    content_replacement: ContentReplacementState | None = None

    # Temporary per-loop model override from tools such as Skill.
    model_override: str | None = None
    """Model override applied to later model calls in the same query loop.

    Reference SkillTool carries this through `contextModifier(...).options`.
    Raygent keeps config frozen, so modifiers update the per-loop context
    instead.
    """

    reasoning_effort_override: str | int | None = None
    """Reasoning-effort override applied to later model calls in the same loop.

    Reference SkillTool can set `effort` alongside a model override. Raygent
    keeps this provider-neutral and sends it through `ModelRequest.effort`.
    """

    # Query chain for subagent tree observability
    query_tracking: QueryTracking | None = None

    # Kernel observability correlation for the current submitted turn.
    observability_context: KernelEventContext | None = None

    # Query source for child-loop policy decisions.
    query_source: str | None = None
    """Stable source label for this query loop.

    Fork AgentTool uses this as the compaction-resistant recursive-fork guard,
    matching the reference `querySource` behavior.
    """

    # Current tool-use id while a tool is executing.
    tool_use_id: str | None = None
    """Set by `run_tool_use` for the duration of a concrete tool call.

    Reference ToolUseContext carries `toolUseId`; background AgentTool uses it
    to associate task notifications with the spawning tool call.
    """

    current_assistant_message: MessageParam | None = None
    """Assistant message that contains the currently executing tool use.

    Set by `run_tool_use`. Forked AgentTool needs the exact message so it can
    pair all sibling tool_use blocks with placeholder tool_results, matching the
    reference fork-subagent cache-prefix shape.
    """

    # Notification callbacks (user-facing, not model-facing)
    add_notification: Callable[[str], None] | None = None

    # Elicitation — when a tool needs the user to resolve something mid-call
    # (e.g., OAuth redirect). Deferred to later group.
    handle_elicitation: Callable[[str], Awaitable[str]] | None = None

    # Read-file cache (for tools that care about staleness)
    read_file_state: ReadFileStateCache = field(
        default_factory=create_read_file_state_cache
    )

    successful_text_read_paths: list[str] = field(default_factory=list[str])
    """Absolute paths successfully loaded by the concrete text Read tool.

    The query loop drains this append-only per-turn log after tool execution to
    build read-adjacent transient context for the next model request. Structured
    media reads and failed/unchanged reads do not append here.
    """

    # Current query runtime. Set by query orchestration before tool execution.
    runtime: ToolRuntimeContext | None = None


@dataclass(frozen=True)
class ToolDescriptionContext:
    """Headless context for dynamic permission descriptions.

    Reference tools receive a UI-heavy options object. Raygent keeps only the
    kernel inputs needed for permission prompts and ToolSearch scoring.
    """

    is_non_interactive_session: bool
    permission_context: ToolPermissionContext
    tools: Sequence[Tool] = ()


@dataclass(frozen=True)
class ToolPromptContext:
    """Headless context for dynamic model/search prompt text.

    Reference tools receive permission context access, the tool catalog, agent
    metadata, and allowed-agent-type data. Raygent keeps those as simple data
    fields so ToolSearch can score prompt text without importing UI/session
    option objects.
    """

    permission_context: ToolPermissionContext
    tools: Sequence[Tool] = ()
    agents: Sequence[str] = ()
    allowed_agent_types: Sequence[str] = ()
    extra: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True)
class ToolRuntimeContext:
    """Current query runtime handles for tools that launch child loops.

    Most tools should not use this. Forked SkillTool and upcoming synchronous
    AgentTool need it so they can reuse the current QueryConfig/QueryDeps
    without importing globals or rebuilding provider state.
    """

    config: QueryConfig
    deps: QueryDeps
    effective_model: str | None = None
    """Current parent-loop model before provider alias resolution.

    This may differ from `config.model` after fallback or a prior tool context
    modifier. Child-loop tools use it for reference-style inheritance.
    """

    pdf_document_service: PdfDocumentService | None = None
    """Optional per-runtime PDF service override for concrete file tools."""


type ToolContextModifier = Callable[[ToolUseContext], ToolUseContext]
"""Reference-style context modifier returned by a tool result."""


# ---------------------------------------------------------------------------
# The Tool protocol itself — multi-axis.
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """The headless tool contract. Kept lean around execution-critical axes."""

    # --- identity ---
    name: str
    aliases: tuple[str, ...]
    """Alternative names accepted for lookup/ToolSearch selection."""

    description: str
    """Short human-readable description. `prompt()` can return the full model text."""

    search_hint: str | None
    """Curated one-line phrase used by ToolSearch keyword scoring."""

    # --- schema duality ---
    # input_schema is JSON Schema for the model; input_model is Pydantic for us.
    input_model: type[BaseModel]
    """Pydantic model for parsing the model's tool-call arguments."""

    input_schema: FrozenJson | None
    """Optional provider-facing JSON Schema override.

    Most built-in tools derive their model schema from `input_model`. External
    tools such as MCP already arrive with JSON Schema, so they can preserve that
    provider-facing contract while using a permissive local Pydantic model for
    execution parsing.
    """

    # --- capability predicates (scheduling hints) ---

    def is_enabled(self) -> bool:
        """Whether this tool is currently available."""
        ...

    def is_concurrency_safe(self, input: BaseModel) -> bool:
        """Safe to run concurrent with other tool calls for this input?"""
        ...

    def is_read_only(self, input: BaseModel) -> bool:
        """Observes state only, no mutation for this input."""
        ...

    def is_destructive(self, input: BaseModel) -> bool:
        """Cannot be undone for this input. Forces stricter permission prompts."""
        ...

    def is_open_world(self, input: BaseModel) -> bool:
        """Interacts with the open world (network, OS) for this input."""
        ...

    def requires_user_interaction(self) -> bool:
        """Needs the user live — skip when running headless/backgrounded."""
        ...

    def interrupt_behavior(self) -> InterruptBehavior:
        """How to treat an interrupt while this tool is running."""
        ...

    # --- deferred loading (ToolSearch) ---
    should_defer: bool
    """If True, schema is not surfaced until ToolSearch fetches it."""

    always_load: bool
    """If True, schema is always surfaced regardless of should_defer."""

    max_result_size_chars: int | float
    """Per-tool result persistence threshold. `float("inf")` disables persistence."""

    # --- lifecycle hooks ---

    async def validate_input(
        self, input: BaseModel, ctx: ToolUseContext
    ) -> ValidationResult:
        """Cheap, model-facing gate. Return ValidationError to let model self-correct."""
        ...

    async def check_permissions(
        self,
        input: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        """User-facing gate. May prompt via handle_elicitation."""
        ...

    def call(
        self, input: BaseModel, ctx: ToolUseContext
    ) -> AsyncIterator[ToolCallEvent]:
        """Execute. Async iterator so tools can yield progress + final result."""
        ...

    # --- observability (optional) ---

    async def describe(
        self, input: BaseModel, ctx: ToolDescriptionContext
    ) -> str:
        """Dynamic permission/search description for this input."""
        ...

    async def prompt(
        self,
        ctx: ToolPromptContext | ToolUseContext | None = None,
    ) -> str:
        """Full model/search prompt text for this tool."""
        ...

    def get_activity_description(self, input: BaseModel) -> str | None:
        """Short human-readable line for UI/progress display. None = no display."""
        ...


# ---------------------------------------------------------------------------
# Tool call events — what a tool yields during .call()
# ---------------------------------------------------------------------------


class ToolProgress(BaseModel):
    """Intermediate progress. Not fed back to the model; for UI/observability."""

    type: Literal["progress"] = "progress"
    message: str
    data: dict[str, Any] | None = None


class ToolResult(BaseModel):
    """Final result fed back to the model as the tool_result content block."""

    type: Literal["result"] = "result"
    content: str | list[dict[str, Any]]
    """String or structured content blocks (text, image)."""

    is_error: bool = False
    additional_messages: tuple[dict[str, Any], ...] = ()
    """Reference-style `newMessages` emitted after the tool_result block."""

    discovered_tool_names: tuple[str, ...] = ()
    """Trusted runtime metadata emitted by ToolSearch only."""

    context_modifier: ToolContextModifier | None = None
    """Reference-style context modifier applied after this tool result."""


class ToolCallError(BaseModel):
    """Tool raised an exception. Surfaced to the model as an error tool_result."""

    type: Literal["error"] = "error"
    message: str
    recoverable: bool = True
    """If False, the harness treats this as a terminal error for the turn."""


ToolCallEvent = ToolProgress | ToolResult | ToolCallError


# ---------------------------------------------------------------------------
# build_tool — fail-closed defaults. Every tool goes through this.
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000
"""Reference default global cap for persisted tool results."""


TOOL_DEFAULTS = {
    "is_enabled": True,
    "is_concurrency_safe": False,
    "is_read_only": False,
    "is_destructive": True,
    "is_open_world": True,
    "requires_user_interaction": False,
    "interrupt_behavior": "block",
    "should_defer": False,
    "always_load": False,
    "max_result_size_chars": DEFAULT_MAX_RESULT_SIZE_CHARS,
}
"""Defaults chosen to fail closed: a tool that doesn't declare its axes is
treated as destructive, open-world, unsafe to run concurrently. The tool author
must opt INTO safety — never assume a missing axis means safe."""


@dataclass
class ToolSpec:
    """Inputs to build_tool. Anything not set uses TOOL_DEFAULTS (fail-closed)."""

    name: str
    description: str
    input_model: type[BaseModel]
    call: Callable[
        [BaseModel, ToolUseContext], AsyncIterator[ToolCallEvent]
    ]
    input_schema: object | None = None
    aliases: tuple[str, ...] = ()
    search_hint: str | None = None
    prompt: PromptTextProviderSpec | None = None

    # Overrides. Defaults applied in build_tool.
    validate_input: ValidateInputFn | None = None
    check_permissions: CheckPermissionsFn | None = None
    describe: DescriptionProvider | None = None
    get_activity_description: Callable[[BaseModel], str | None] | None = None

    # Capability overrides (defaults fail-closed). Input-sensitive axes accept
    # either constants or predicates; build_tool normalizes both to methods.
    is_concurrency_safe: InputCapabilitySpec | None = None
    is_read_only: InputCapabilitySpec | None = None
    is_destructive: InputCapabilitySpec | None = None
    is_open_world: InputCapabilitySpec | None = None
    is_enabled: NullaryCapabilitySpec | None = None
    requires_user_interaction: NullaryCapabilitySpec | None = None
    interrupt_behavior: InterruptBehaviorSpec | None = None
    should_defer: bool | None = None
    always_load: bool | None = None
    max_result_size_chars: int | float | None = None


async def _default_validate(_input: BaseModel, _ctx: ToolUseContext) -> ValidationResult:
    return ValidationOk()


async def _default_check_permissions(
    _input: BaseModel,
    _ctx: ToolUseContext,
    _permission_context: ToolPermissionContext,
) -> PermissionResult:
    """Default: ask. Fail-closed — a tool without an explicit policy prompts."""
    return PermissionAsk(message="Permission required (no policy set).")


def _default_activity_description(_input: BaseModel) -> str | None:
    return None


def _constant_input_capability(value: bool) -> InputCapability:
    def predicate(_input: BaseModel) -> bool:
        return value

    return predicate


def _coerce_input_capability(
    value: InputCapabilitySpec | None,
    default: bool,
) -> InputCapability:
    if value is None:
        return _constant_input_capability(default)
    if isinstance(value, bool):
        return _constant_input_capability(value)
    return value


def _constant_nullary_capability(value: bool) -> NullaryCapability:
    def predicate() -> bool:
        return value

    return predicate


def _coerce_nullary_capability(
    value: NullaryCapabilitySpec | None,
    default: bool,
) -> NullaryCapability:
    if value is None:
        return _constant_nullary_capability(default)
    if isinstance(value, bool):
        return _constant_nullary_capability(value)
    return value


def _constant_interrupt_behavior(value: InterruptBehavior) -> InterruptBehaviorFn:
    def get_behavior() -> InterruptBehavior:
        return value

    return get_behavior


def _coerce_interrupt_behavior(
    value: InterruptBehaviorSpec | None,
    default: InterruptBehavior,
) -> InterruptBehaviorFn:
    if value is None:
        return _constant_interrupt_behavior(default)
    if callable(value):
        return value
    return _constant_interrupt_behavior(value)


def _constant_prompt_text(value: str) -> PromptTextProvider:
    async def get_prompt(
        _ctx: ToolPromptContext | ToolUseContext | None = None,
    ) -> str:
        return value

    return get_prompt


def _coerce_prompt_text(
    value: PromptTextProviderSpec | None,
    default: str,
) -> PromptTextProvider:
    if value is None:
        return _constant_prompt_text(default)
    if isinstance(value, str):
        return _constant_prompt_text(value)
    return value


def _constant_description(value: str) -> DescriptionProvider:
    async def describe(_input: BaseModel, _ctx: ToolDescriptionContext) -> str:
        return value

    return describe


@dataclass
class _BuiltTool:
    """Concrete Tool implementation produced by build_tool. Satisfies Tool protocol."""

    name: str
    aliases: tuple[str, ...]
    description: str
    search_hint: str | None
    input_model: type[BaseModel]
    input_schema: FrozenJson | None
    should_defer: bool
    always_load: bool
    max_result_size_chars: int | float
    _is_enabled: NullaryCapability
    _is_concurrency_safe: InputCapability
    _is_read_only: InputCapability
    _is_destructive: InputCapability
    _is_open_world: InputCapability
    _requires_user_interaction: NullaryCapability
    _interrupt_behavior: InterruptBehaviorFn
    _validate: ValidateInputFn
    _check_permissions: CheckPermissionsFn
    _call: Callable[[BaseModel, ToolUseContext], AsyncIterator[ToolCallEvent]]
    _describe: DescriptionProvider
    _prompt: PromptTextProvider
    _activity_description: Callable[[BaseModel], str | None]

    def is_enabled(self) -> bool:
        return self._is_enabled()

    def is_concurrency_safe(self, input: BaseModel) -> bool:
        return self._is_concurrency_safe(input)

    def is_read_only(self, input: BaseModel) -> bool:
        return self._is_read_only(input)

    def is_destructive(self, input: BaseModel) -> bool:
        return self._is_destructive(input)

    def is_open_world(self, input: BaseModel) -> bool:
        return self._is_open_world(input)

    def requires_user_interaction(self) -> bool:
        return self._requires_user_interaction()

    def interrupt_behavior(self) -> InterruptBehavior:
        return self._interrupt_behavior()

    async def validate_input(
        self, input: BaseModel, ctx: ToolUseContext
    ) -> ValidationResult:
        return await self._validate(input, ctx)

    async def check_permissions(
        self,
        input: BaseModel,
        ctx: ToolUseContext,
        permission_context: ToolPermissionContext,
    ) -> PermissionResult:
        return await self._check_permissions(input, ctx, permission_context)

    def call(
        self, input: BaseModel, ctx: ToolUseContext
    ) -> AsyncIterator[ToolCallEvent]:
        return self._call(input, ctx)

    async def describe(
        self, input: BaseModel, ctx: ToolDescriptionContext
    ) -> str:
        return await self._describe(input, ctx)

    async def prompt(
        self,
        ctx: ToolPromptContext | ToolUseContext | None = None,
    ) -> str:
        return await self._prompt(ctx)

    def get_activity_description(self, input: BaseModel) -> str | None:
        return self._activity_description(input)


def build_tool(spec: ToolSpec) -> Tool:
    """Apply fail-closed defaults; produce a Tool-protocol-satisfying object.

    Every tool the harness uses flows through this so the defaults are enforced
    at one choke point. Skipping it is how tools accidentally end up marked
    safe-by-omission.
    """
    return _BuiltTool(
        name=spec.name,
        aliases=spec.aliases,
        description=spec.description,
        search_hint=spec.search_hint,
        input_model=spec.input_model,
        input_schema=freeze_json(spec.input_schema) if spec.input_schema is not None else None,
        should_defer=spec.should_defer
        if spec.should_defer is not None
        else TOOL_DEFAULTS["should_defer"],  # type: ignore[arg-type]
        always_load=spec.always_load
        if spec.always_load is not None
        else TOOL_DEFAULTS["always_load"],  # type: ignore[arg-type]
        max_result_size_chars=spec.max_result_size_chars
        if spec.max_result_size_chars is not None
        else TOOL_DEFAULTS["max_result_size_chars"],  # type: ignore[arg-type]
        _is_enabled=_coerce_nullary_capability(
            spec.is_enabled,
            TOOL_DEFAULTS["is_enabled"],  # type: ignore[arg-type]
        ),
        _is_concurrency_safe=_coerce_input_capability(
            spec.is_concurrency_safe,
            TOOL_DEFAULTS["is_concurrency_safe"],  # type: ignore[arg-type]
        ),
        _is_read_only=_coerce_input_capability(
            spec.is_read_only,
            TOOL_DEFAULTS["is_read_only"],  # type: ignore[arg-type]
        ),
        _is_destructive=_coerce_input_capability(
            spec.is_destructive,
            TOOL_DEFAULTS["is_destructive"],  # type: ignore[arg-type]
        ),
        _is_open_world=_coerce_input_capability(
            spec.is_open_world,
            TOOL_DEFAULTS["is_open_world"],  # type: ignore[arg-type]
        ),
        _requires_user_interaction=_coerce_nullary_capability(
            spec.requires_user_interaction,
            TOOL_DEFAULTS["requires_user_interaction"],  # type: ignore[arg-type]
        ),
        _interrupt_behavior=_coerce_interrupt_behavior(
            spec.interrupt_behavior,
            TOOL_DEFAULTS["interrupt_behavior"],  # type: ignore[arg-type]
        ),
        _validate=spec.validate_input or _default_validate,
        _check_permissions=spec.check_permissions or _default_check_permissions,
        _call=spec.call,
        _describe=spec.describe or _constant_description(spec.description),
        _prompt=_coerce_prompt_text(spec.prompt, spec.description),
        _activity_description=spec.get_activity_description or _default_activity_description,
    )


# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------


def tool_matches_name(tool: Tool, name: str) -> bool:
    """Match a tool by primary name or alias."""
    return tool.name == name or name in tool.aliases


def find_tool_by_name(tools: Sequence[Tool], name: str) -> Tool | None:
    """Linear lookup by primary name or alias; tool lists are small."""
    for t in tools:
        if tool_matches_name(t, name):
            return t
    return None


def tool_selected_by_name(
    tool: Tool,
    selected_tool_names: Collection[str] | None = None,
) -> bool:
    """Whether a prior ToolSearch result selected a tool by primary or alias."""

    if not selected_tool_names:
        return False
    return tool.name in selected_tool_names or any(
        alias in selected_tool_names for alias in tool.aliases
    )


def tool_visible_to_model(
    tool: Tool,
    selected_tool_names: Collection[str] | None = None,
) -> bool:
    """Return whether a tool should be visible/callable for model output."""

    try:
        if not tool.is_enabled():
            return False
    except Exception:
        return False
    if tool_selected_by_name(tool, selected_tool_names):
        return True
    return tool.always_load or not tool.should_defer
