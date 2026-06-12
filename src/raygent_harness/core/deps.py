"""QueryDeps — dependency injection container for the agent loop.

QueryDeps is distinct from QueryConfig:
- Config is *what* the turn should do (model, tools, budgets).
- Deps is *how* the harness interacts with the outside world (model provider,
  disk I/O, clock, notification sink).

Separating them lets tests swap deps without rebuilding config, and makes the
seam for future out-of-process backends (remote agent executor, workflow
runtime) explicit rather than scattered through helpers.

Mutable (not frozen) because some deps — notably the notification sink — are
intentionally stateful across a turn. But treat it as "passed once, read often";
don't reassign fields after the loop starts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from raygent_harness.core.context_providers import (
    ContextProvider,
    PostToolContextProvider,
)
from raygent_harness.core.model_provider import UnavailableModelProvider
from raygent_harness.core.observability import KernelEventBus, NoopKernelEventBus
from raygent_harness.core.permissions import (
    PermissionRuleSource,
    ToolPermissionContext,
    ToolPermissionRulesBySource,
    empty_tool_permission_context,
)
from raygent_harness.core.query import Layer, LayerResult
from raygent_harness.core.stop_hooks import StopHook
from raygent_harness.core.tool_hooks import (
    PostToolUseFailureHook,
    PostToolUseHook,
    PreToolUseHook,
)

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.model_provider import ModelProvider
    from raygent_harness.core.permission_engine import (
        PermissionHandler,
        ResolvedPermission,
    )
    from raygent_harness.core.permissions import PermissionResult
    from raygent_harness.core.state import State
    from raygent_harness.core.task import AppStateStore, TaskNotification
    from raygent_harness.core.tool import Tool, ToolUseContext
    from raygent_harness.services.agent_routes import AgentRouteRecordStore
    from raygent_harness.services.compact.models import CompactionResult
    from raygent_harness.services.file_media import PdfDocumentService
    from raygent_harness.services.handoff import AgentHandoffClassifier
    from raygent_harness.services.remote_agent import (
        RemoteAgentBackend,
        RemoteAgentPersistenceStore,
    )
    from raygent_harness.services.transcript import TranscriptStore
    from raygent_harness.services.worktree import WorktreeManager
    from raygent_harness.skills.models import SkillDefinition


# ---------------------------------------------------------------------------
# Clock — injectable so tests can freeze / advance time.
# ---------------------------------------------------------------------------


class Clock(Protocol):
    """Time source. Production uses time.time(); tests use a fake."""

    def now(self) -> float:
        """Monotonic-ish wall-clock seconds. For task start_time / end_time."""
        ...


@dataclass
class SystemClock:
    """Default Clock impl. Thin wrapper so Protocol conformance is explicit."""

    def now(self) -> float:
        return time.time()


# ---------------------------------------------------------------------------
# Notification sink — how user-facing notifications reach the user.
# ---------------------------------------------------------------------------


NotificationSink = Callable[[str], None]
"""Called for every user-facing notification (task completed, bg signal, etc.).

Separate from ToolUseContext.add_notification because that's per-tool-call and
may be None; this one is turn-scoped and always present (stdout if nothing else).
"""


def _stderr_notification_sink(message: str) -> None:
    """Default sink — writes to stderr. Replace with Push/Slack/etc. in prod."""
    import sys

    print(f"[notification] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Context-pipeline layer default — no-op.
# ---------------------------------------------------------------------------


async def _noop_layer(
    messages: list[MessageParam],
    _state: State,
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> LayerResult:
    """No-op pipeline layer. Default for `microcompact` / `autocompact` until
    the compaction service runs. Returns messages unchanged, no boundary."""
    return LayerResult(messages=messages)


# ---------------------------------------------------------------------------
# Reactive compaction default — no-op.
# ---------------------------------------------------------------------------


class ReactiveCompactor(Protocol):
    """Context-overflow recovery compactor.

    after a real prompt-too-long/media error. The loop owns the error-policy
    guard; this dependency owns the actual summary attempt and returns None
    when no recovery is available.
    """

    async def __call__(
        self,
        messages: list[MessageParam],
        state: State,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> CompactionResult | None:
        """Return a compaction result, or None to surface prompt_too_long."""
        ...


async def _noop_reactive_compact(
    _messages: list[MessageParam],
    _state: State,
    _config: QueryConfig,
    _ctx: ToolUseContext,
    /,
) -> CompactionResult | None:
    """Fail-closed default: no summarizer means no reactive recovery."""
    return None


# ---------------------------------------------------------------------------
# Memory integration seams — optional, no-op by default.
# ---------------------------------------------------------------------------


class SystemPromptProvider(Protocol):
    """Return extra system-prompt text for the current turn.

    Reference QueryEngine composes several prompt/context fragments before each
    model call. Raygent keeps those producers injectable so feature packages
    such as coordinator mode can stay out of `core` while still participating
    in the frozen per-turn `QueryConfig.system_prompt`.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str | None:
        ...


class MemoryPromptProvider(Protocol):
    """Return memory mechanics text to append to the turn system prompt.

    assembling the system prompt. Raygent injects the provider so SDK callers
    can choose memory settings without `core` importing `memdir`.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> str | None:
        ...


class MemoryExtractor(Protocol):
    """Optional post-turn memory extraction hook.

    Reference fires `executeExtractMemories(...)` from stop hooks. Raygent keeps
    this as an injected hook until the concrete extraction runner is wired.
    """

    async def __call__(
        self,
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> object:
        """Run extraction and optionally return a structured run summary."""
        ...


class MemoryRecallPrefetch(Protocol):
    """Turn-scoped relevant-memory prefetch handle.

    Core owns the timing: start once before the loop, check settlement after
    tool results, consume without waiting, and cancel on every generator exit.
    Concrete memory providers own selection, file reads, duplicate filtering,
    read-state marking, and lifecycle facts.
    """

    @property
    def settled_at(self) -> float | None:
        """Wall-clock seconds when the prefetch settled, or None if pending."""
        ...

    @property
    def consumed_on_iteration(self) -> int | None:
        """Loop iteration that consumed the prefetch, or None if unconsumed."""
        ...

    async def consume_if_ready(
        self,
        *,
        ctx: ToolUseContext,
        iteration: int,
    ) -> tuple[MessageParam, ...]:
        """Return model-visible recall messages only if already settled."""
        ...

    def cancel(self) -> None:
        """Cancel background recall work. Must be idempotent."""
        ...


class MemoryRecallProvider(Protocol):
    """Optional query-time relevant-memory recall provider."""

    def start(
        self,
        messages: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> MemoryRecallPrefetch | None:
        """Start a non-blocking prefetch for this user turn."""
        ...


class CoordinatorRuntimeProtocol(Protocol):
    """Optional coordinator runtime seam.

    Concrete coordinator policy lives under `raygent_harness.coordinator`.
    Core only needs to hand task notifications to the runtime and ask for one
    bounded model-visible digest when available.
    """

    def has_processed_task_notification(
        self,
        notification: TaskNotification,
        /,
    ) -> bool:
        """Return True when this task-notification fact was already ingested."""
        ...

    def record_task_notifications(
        self,
        notifications: Sequence[TaskNotification],
        /,
    ) -> object:
        """Record drained task notifications into coordinator state."""
        ...

    def record_agent_launch(
        self,
        *,
        agent_id: str,
        agent_type: str,
        description: str,
        prompt_chars: int,
        task_id: str | None = None,
        agent_name: str | None = None,
        team_name: str | None = None,
        mode: str = "background",
        status: str = "running",
        result_summary: str | None = None,
        error_summary: str | None = None,
    ) -> object:
        """Record a successful AgentTool launch/result into coordinator state."""
        ...

    def record_send_message(
        self,
        *,
        sender: str,
        target: str,
        summary: str | None,
        message_chars: int,
        recipient_task_ids: Sequence[str] = (),
        recipient_agent_ids: Sequence[str] = (),
        team_name: str | None = None,
    ) -> object:
        """Record a successful SendMessage route into coordinator state."""
        ...

    def record_task_stop(
        self,
        *,
        task_id: str,
        task_type: str,
        description: str | None = None,
    ) -> object:
        """Record a successful TaskStop action into coordinator state."""
        ...

    def render_context(self) -> MessageParam | None:
        """Return a bounded synthetic user message, or None when empty."""
        ...


# ---------------------------------------------------------------------------
# Behavioral expansion seams — optional, no-op by default.
# ---------------------------------------------------------------------------


AgentTriggerScope = Literal["main", "subagent", "all"]


@dataclass(frozen=True)
class AgentTriggerMatch:
    """One policy-selected agent-routing hint for the submitted turn."""

    id: str
    agent_name: str
    reason: str | None = None
    prompt_hint: str | None = None
    confidence: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class AgentTriggerDecision:
    """Policy output for optional agent-trigger behavior.

    `matches` are rendered by core as explicit model-visible guidance.
    `model_visible_messages` lets an embedding policy provide additional
    transcript-visible guidance. Hidden launch is intentionally not performed
    by core; `suppress_main_turn` is recorded for audit only.
    """

    matches: tuple[AgentTriggerMatch, ...] = ()
    model_visible_messages: tuple[MessageParam, ...] = ()
    suppress_main_turn: bool = False


class AgentTriggerPolicy(Protocol):
    """Optional policy that suggests agent routing for a submitted turn."""

    async def __call__(
        self,
        prompt: MessageParam,
        history: Sequence[MessageParam],
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> AgentTriggerDecision:
        ...


@dataclass(frozen=True)
class AgentTriggerPolicySpec:
    """Agent-trigger policy plus explicit scope and bounding controls."""

    policy: AgentTriggerPolicy
    agent_scope: AgentTriggerScope = "main"
    max_message_chars: int = 4_000
    max_messages: int = 3


# ---------------------------------------------------------------------------
# Tooling integration seams — optional, no-op by default.
# ---------------------------------------------------------------------------


class ToolPermissionEngine(Protocol):
    """Permission-resolution seam future tool orchestration calls.

    letting each executor invent its own path. Raygent exposes the same
    chokepoint as an injectable object so tests/adapters can swap behavior
    without changing `_orchestrate_tools` when it lands.
    """

    async def can_use_tool(
        self,
        *,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        permission_context: ToolPermissionContext,
        handler: PermissionHandler | None = None,
        tool_use_id: str | None = None,
        force_decision: PermissionResult | None = None,
        is_non_interactive_session: bool = False,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        """Resolve normal validate→permission→handler decisions."""
        ...

    async def resolve_hook_permission_decision(
        self,
        *,
        hook_permission_result: PermissionResult | None,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        permission_context: ToolPermissionContext,
        handler: PermissionHandler | None = None,
        tool_use_id: str | None = None,
        require_can_use_tool: bool = False,
        is_non_interactive_session: bool = False,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        """Resolve PreToolUse hook output without bypassing rule checks."""
        ...


@dataclass(frozen=True)
class DefaultToolPermissionEngine:
    """Adapter over `core.permission_engine`.

    Kept as a small wrapper, not direct function fields, so future orchestration
    can depend on one object with both normal and hook-resolution paths.
    """

    async def can_use_tool(
        self,
        *,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        permission_context: ToolPermissionContext,
        handler: PermissionHandler | None = None,
        tool_use_id: str | None = None,
        force_decision: PermissionResult | None = None,
        is_non_interactive_session: bool = False,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        from raygent_harness.core.permission_engine import can_use_tool

        return await can_use_tool(
            tool=tool,
            input=input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
            handler=handler,
            tool_use_id=tool_use_id,
            force_decision=force_decision,
            is_non_interactive_session=is_non_interactive_session,
            tools=tools,
        )

    async def resolve_hook_permission_decision(
        self,
        *,
        hook_permission_result: PermissionResult | None,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        permission_context: ToolPermissionContext,
        handler: PermissionHandler | None = None,
        tool_use_id: str | None = None,
        require_can_use_tool: bool = False,
        is_non_interactive_session: bool = False,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        from raygent_harness.core.permission_engine import (
            resolve_hook_permission_decision,
        )

        return await resolve_hook_permission_decision(
            hook_permission_result=hook_permission_result,
            tool=tool,
            input=input,
            tool_use_context=tool_use_context,
            permission_context=permission_context,
            handler=handler,
            tool_use_id=tool_use_id,
            require_can_use_tool=require_can_use_tool,
            is_non_interactive_session=is_non_interactive_session,
            tools=tools,
        )


class SkillProvider(Protocol):
    """Return turn-visible skills.

    Reference loads skills while assembling turn/tool context. Raygent keeps
    loading out of `core` and accepts an injected provider that can source
    filesystem, bundled, plugin, or MCP skills.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> Sequence[SkillDefinition]:
        ...


class ToolCatalogProvider(Protocol):
    """Return the turn tool catalog after skill/provider expansion.

    Returning None preserves `config.tools`. This lets future Skill/ToolSearch
    adapters add concrete tools without teaching QueryEngine about those
    packages.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        ...


# ---------------------------------------------------------------------------
# QueryDeps — the DI container.
# ---------------------------------------------------------------------------


@dataclass
class QueryDeps:
    """Turn-scoped dependencies. Passed alongside QueryConfig into query().

    Unlike Config, Deps may contain stateful objects (HTTP client connection
    pool, notification sink with buffering). Don't deep-copy this; pass by
    reference and trust callees not to mutate its shape.

    the pieces test suites commonly want to swap (model call, compaction
    layers). Other seams (runTools, stop hooks, logging) stay as module
    imports until that test friction shows up.
    """

    # --- Task store (for spawning subagents / shells from within the loop) ---
    task_store: AppStateStore
    """Where subagent/shell task state lives. See core.task.AppStateStore."""

    # --- Model provider ---
    model_provider: ModelProvider = field(default_factory=UnavailableModelProvider)
    """Provider-neutral model backend. Embedding apps supply the concrete
    provider; core never imports vendor SDKs."""

    # --- Context-pipeline layers (no-op defaults). ---
    microcompact: Layer = field(default=_noop_layer)
    autocompact: Layer = field(default=_noop_layer)
    reactive_compact: ReactiveCompactor = field(default=_noop_reactive_compact)
    """One-shot prompt-too-long recovery. Default None-result surfaces the
    original error as `prompt_too_long`; callers install
    `services.compact.create_reactive_compact(...)` when a summarizer exists."""

    # --- Stop hooks — in-process callables run at turn-end. Reference
    #     contract narrow (see `core/stop_hooks.py`). Empty list =
    #     no hooks = clean completion without veto. ---
    stop_hooks: list[StopHook] = field(default_factory=list[StopHook])

    # --- Context and memory integration seams ---
    context_providers: tuple[ContextProvider, ...] = ()
    """Optional typed context providers.

    Providers can add system-prompt fragments or non-persistent user-context
    messages for a submitted turn. Default empty preserves explicit-only
    caller prompts.
    """

    post_tool_context_providers: tuple[PostToolContextProvider, ...] = ()
    """Optional transient context providers after tool execution.

    Providers receive successful text read paths and return context fragments
    for subsequent model requests in the same submitted turn. The rendered
    messages are not persisted into `State.messages` or transcripts.
    """

    system_prompt_provider: SystemPromptProvider | None = None
    """Optional provider for non-memory prompt fragments such as coordinator
    mode. Applied before `memory_prompt_provider` so memory mechanics remain
    the final prompt segment, as the final prompt segment.
    """

    memory_prompt_provider: MemoryPromptProvider | None = None
    """Optional provider that appends memory mechanics to the per-turn system
    prompt. Default None preserves the caller-provided `QueryConfig`.
    """

    memory_extractor: MemoryExtractor | None = None
    """Optional post-success extraction hook. Default None means no background
    memory work. Implementations should be best-effort and idempotent.
    """

    memory_recall_provider: MemoryRecallProvider | None = None
    """Optional query-time relevant-memory recall provider.

    Core starts it once per user turn and consumes settled results only after
    tool results. Default None preserves current behavior.
    """

    agent_trigger_policy: AgentTriggerPolicySpec | None = None
    """Optional prompt/history policy for explicit agent-routing guidance.

    Default None preserves current behavior. When configured, QueryEngine
    evaluates the scoped policy before constructing the query state and records
    bounded model-visible guidance as synthetic user messages. Core does not
    launch hidden agents from this seam.
    """

    coordinator_runtime: CoordinatorRuntimeProtocol | None = None
    """Optional coordinator work ledger/blackboard runtime.

    Query drains task notifications into this runtime and appends its bounded
    digest as a transcript-visible synthetic user message. Default None keeps
    non-coordinator sessions unchanged.
    """

    pdf_document_service: PdfDocumentService | None = None
    """Optional PDF metadata/page extraction service for concrete file tools.

    None keeps PDF page extraction unavailable while preserving full-PDF
    document reads on models that support document blocks. Embedders can install
    a command-backed or custom service without adding global PDF dependencies.
    """

    # --- Tooling integration seams ---
    permission_context: ToolPermissionContext = field(
        default_factory=empty_tool_permission_context
    )
    """Turn/session permission state. Future orchestration passes this to
    `permission_engine`; empty context preserves current no-tool behavior."""

    permission_handler: PermissionHandler | None = None
    """Optional ask/interactive adapter for permission decisions. Default None
    leaves `ask` decisions structured but unresolved by UI."""

    permission_engine: ToolPermissionEngine = field(
        default_factory=DefaultToolPermissionEngine
    )
    """Central permission resolver. Default wraps `core.permission_engine`."""

    is_non_interactive_session: bool = False
    """Passed into dynamic permission descriptions and handler requests."""

    skill_provider: SkillProvider | None = None
    """Optional source of turn-visible skills. Default None means no skills are
    loaded by core."""

    tool_catalog_provider: ToolCatalogProvider | None = None
    """Optional tool catalog expansion/rewrite hook. Default None keeps
    `QueryConfig.tools` untouched."""

    pre_tool_use_hooks: list[PreToolUseHook] = field(default_factory=list[PreToolUseHook])
    """Headless PreToolUse hook registry. Empty means no hooks."""

    post_tool_use_hooks: list[PostToolUseHook] = field(default_factory=list[PostToolUseHook])
    """Headless PostToolUse hook registry. Empty means no hooks."""

    post_tool_use_failure_hooks: list[PostToolUseFailureHook] = field(
        default_factory=list[PostToolUseFailureHook]
    )
    """Headless PostToolUseFailure hook registry. Empty means no hooks."""

    max_tool_use_concurrency: int = 10
    """Bound for adjacent concurrency-safe tool calls. Matches the reference
    default and stays injectable for tests/adapters instead of reading an env
    var from core orchestration."""

    # --- Clock ---
    clock: Clock = field(default_factory=SystemClock)

    # --- Notification sink ---
    notify: NotificationSink = field(default=_stderr_notification_sink)

    # --- Output directory for per-task transcripts ---
    output_dir: str = ".raygent/tasks"
    """Base path for per-task JSONL transcripts. Task framework subpath:
    {output_dir}/{task_id}.jsonl. Relative to cwd of the host process."""

    # --- Optional session transcript persistence ---
    transcript_store: TranscriptStore | None = None
    """Optional event-log transcript store.

    None preserves in-memory-only behavior. QueryEngine owns when transcript
    writes occur; stores own where/how they are persisted.
    """

    worktree_manager: WorktreeManager | None = None
    """Optional isolated-worktree manager used by AgentTool.

    None is fail-closed: `isolation="worktree"` validates as unsupported unless
    an embedding app explicitly provides a manager. This keeps core from
    mutating git state implicitly.
    """

    remote_agent_backend: RemoteAgentBackend | None = None
    """Optional remote AgentTool backend.

    None is fail-closed: `isolation="remote"` validates as unsupported unless an
    embedding app installs a launch/poll/stop implementation. Core owns task
    lifecycle semantics; adapters own auth, transport, and remote session
    details.
    """

    remote_agent_persistence_store: RemoteAgentPersistenceStore | None = None
    """Optional identity sidecar store for remote-agent resume.

    None keeps remote agents process-local. When installed, remote-agent launch,
    poll, terminal cleanup, and restore use it fail-softly: persistence
    failures degrade restart recovery but never change already-launched remote
    work or model-visible notifications.
    """

    agent_route_record_store: AgentRouteRecordStore | None = None
    """Optional sidecar store for named/raw-id local-agent route recovery.

    None keeps local-agent routes process-local. When installed, local-agent
    launch/resume/terminal worktree-refresh paths persist route metadata
    fail-softly so a later runtime recovery pass can repopulate
    `AppStateStore.agent_route_records` and `agent_name_registry`.
    """

    handoff_classifier: AgentHandoffClassifier | None = None
    """Optional background-agent handoff classifier.

    Classification is a notification embellishment only. Task status transitions
    happen before this hook runs, and failures/timeouts are fail-soft.
    """

    handoff_classifier_timeout_s: float = 2.0
    """Maximum time a terminal task notification may wait for handoff warning."""

    # --- Kernel observability ---
    observability: KernelEventBus = field(default_factory=NoopKernelEventBus)
    """Observation-only event bus for debug/eval/tracing adapters.

    The no-op default preserves current behavior. Core producers must treat
    event publication as fail-soft and must not make model-visible decisions
    from sink state.
    """

    def __post_init__(self) -> None:
        # Task producers only receive AppStateStore in several low-level APIs.
        # Keep the store bus aligned with QueryDeps so task lifecycle events
        # correlate with the active turn without adding global state.
        self.task_store.observability = self.observability

    async def resolve_tool_permission(
        self,
        *,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        tool_use_id: str | None = None,
        force_decision: PermissionResult | None = None,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        """Resolve a normal tool permission using this deps object's state."""

        return await self.permission_engine.can_use_tool(
            tool=tool,
            input=input,
            tool_use_context=tool_use_context,
            permission_context=self._permission_context_for(tool_use_context),
            handler=self.permission_handler,
            tool_use_id=tool_use_id,
            force_decision=force_decision,
            is_non_interactive_session=self.is_non_interactive_session,
            tools=tools,
        )

    async def resolve_hook_tool_permission(
        self,
        *,
        hook_permission_result: PermissionResult | None,
        tool: Tool,
        input: BaseModel,
        tool_use_context: ToolUseContext,
        tool_use_id: str | None = None,
        require_can_use_tool: bool = False,
        tools: Sequence[Tool] = (),
    ) -> ResolvedPermission:
        """Resolve PreToolUse hook permission output using this deps state."""

        return await self.permission_engine.resolve_hook_permission_decision(
            hook_permission_result=hook_permission_result,
            tool=tool,
            input=input,
            tool_use_context=tool_use_context,
            permission_context=self._permission_context_for(tool_use_context),
            handler=self.permission_handler,
            tool_use_id=tool_use_id,
            require_can_use_tool=require_can_use_tool,
            is_non_interactive_session=self.is_non_interactive_session,
            tools=tools,
        )

    def _permission_context_for(
        self,
        tool_use_context: ToolUseContext,
    ) -> ToolPermissionContext:
        """Return the active permission context for this tool call.

        Tool context modifiers (for example Skill `allowed_tools`) update
        `ToolUseContext.permission_context`. Older tests/calls historically set
        only `QueryDeps.permission_context`, so fall back to deps when the
        context still carries the empty default.
        """

        empty = empty_tool_permission_context()
        if (
            tool_use_context.permission_context == empty
            and self.permission_context != empty
        ):
            return self.permission_context
        if self.permission_context == empty:
            return tool_use_context.permission_context
        return _merge_permission_contexts(
            self.permission_context,
            tool_use_context.permission_context,
        )

    def permission_context_for(
        self,
        tool_use_context: ToolUseContext,
    ) -> ToolPermissionContext:
        """Public wrapper for the active permission context merge policy."""

        return self._permission_context_for(tool_use_context)


def _merge_permission_contexts(
    base: ToolPermissionContext,
    overlay: ToolPermissionContext,
) -> ToolPermissionContext:
    """Layer context-modifier permissions on top of deps-level permissions."""

    empty = empty_tool_permission_context()
    return replace(
        base,
        mode=overlay.mode if overlay.mode != empty.mode else base.mode,
        additional_working_directories={
            **base.additional_working_directories,
            **overlay.additional_working_directories,
        },
        always_allow_rules=_merge_rule_maps(
            base.always_allow_rules,
            overlay.always_allow_rules,
        ),
        always_deny_rules=_merge_rule_maps(
            base.always_deny_rules,
            overlay.always_deny_rules,
        ),
        always_ask_rules=_merge_rule_maps(
            base.always_ask_rules,
            overlay.always_ask_rules,
        ),
        is_bypass_permissions_mode_available=(
            base.is_bypass_permissions_mode_available
            or overlay.is_bypass_permissions_mode_available
        ),
        is_auto_mode_available=(
            base.is_auto_mode_available or overlay.is_auto_mode_available
        ),
        stripped_dangerous_rules=(
            _merge_rule_maps(base.stripped_dangerous_rules, overlay.stripped_dangerous_rules)
            if base.stripped_dangerous_rules is not None
            and overlay.stripped_dangerous_rules is not None
            else overlay.stripped_dangerous_rules or base.stripped_dangerous_rules
        ),
        should_avoid_permission_prompts=(
            base.should_avoid_permission_prompts
            or overlay.should_avoid_permission_prompts
        ),
        await_automated_checks_before_dialog=(
            base.await_automated_checks_before_dialog
            or overlay.await_automated_checks_before_dialog
        ),
        pre_plan_mode=overlay.pre_plan_mode or base.pre_plan_mode,
    )


def _merge_rule_maps(
    base: ToolPermissionRulesBySource | None,
    overlay: ToolPermissionRulesBySource | None,
) -> ToolPermissionRulesBySource:
    merged: dict[PermissionRuleSource, tuple[str, ...]] = {}
    source_order = (*((base or {}).keys()), *((overlay or {}).keys()))
    for source in source_order:
        if source in merged:
            continue
        merged[source] = _dedupe_preserve_order(
            (*((base or {}).get(source, ())), *((overlay or {}).get(source, ())))
        )
    return merged


def _dedupe_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)


__all__ = [
    "AgentTriggerDecision",
    "AgentTriggerMatch",
    "AgentTriggerPolicy",
    "AgentTriggerPolicySpec",
    "AgentTriggerScope",
    "Clock",
    "ContextProvider",
    "CoordinatorRuntimeProtocol",
    "DefaultToolPermissionEngine",
    "KernelEventBus",
    "MemoryExtractor",
    "MemoryPromptProvider",
    "MemoryRecallPrefetch",
    "MemoryRecallProvider",
    "NoopKernelEventBus",
    "NotificationSink",
    "QueryDeps",
    "ReactiveCompactor",
    "SkillProvider",
    "SystemClock",
    "SystemPromptProvider",
    "ToolCatalogProvider",
    "ToolPermissionEngine",
]
