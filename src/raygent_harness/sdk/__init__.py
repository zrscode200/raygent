"""Headless embedding factory for Raygent sessions.

This module is the ergonomic adapter surface over the kernel. It assembles the
same explicit primitives advanced users can still construct directly:
`QueryConfig`, `QueryDeps`, `ToolUseContext`, `AppStateStore`, and
`QueryEngine`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from raygent_harness.core.config import QueryConfig, SamplingParams, TurnBudget
from raygent_harness.core.context_providers import ContextProvider
from raygent_harness.core.deps import NotificationSink, QueryDeps, ToolCatalogProvider
from raygent_harness.core.model_provider import ModelProvider
from raygent_harness.core.observability import (
    KernelEvent,
    KernelEventBus,
    KernelEventSink,
    NoopKernelEventBus,
)
from raygent_harness.core.permissions import (
    ToolPermissionContext,
    empty_tool_permission_context,
)
from raygent_harness.core.query_engine import QueryEngine, SDKMessage, SDKResult
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import QueryTracking, Tool, ToolUseContext

if TYPE_CHECKING:
    from raygent_harness.context_providers.environment import (
        GitCommandRunner,
        TodayProvider,
    )
    from raygent_harness.context_providers.project_instructions import DiscoveryMode
    from raygent_harness.core.context_providers import PostToolContextProvider
    from raygent_harness.core.deps import (
        AgentTriggerPolicySpec,
        Clock,
        CoordinatorRuntimeProtocol,
        MemoryExtractor,
        MemoryPromptProvider,
        MemoryRecallProvider,
        SkillProvider,
        SystemPromptProvider,
        ToolPermissionEngine,
    )
    from raygent_harness.core.permission_engine import PermissionHandler
    from raygent_harness.core.tool_hooks import (
        PostToolUseFailureHook,
        PostToolUseHook,
        PreToolUseHook,
    )
    from raygent_harness.services.agent_routes import AgentRouteRecordStore
    from raygent_harness.services.file_media import PdfDocumentService
    from raygent_harness.services.handoff import AgentHandoffClassifier
    from raygent_harness.services.remote_agent import (
        RemoteAgentBackend,
        RemoteAgentPersistenceStore,
    )
    from raygent_harness.services.task_output import TaskOutputStore
    from raygent_harness.services.transcript import TranscriptScope, TranscriptStore
    from raygent_harness.services.worktree import WorktreeManager
    from raygent_harness.tools.discovery_tools import DiscoveryToolingRuntime
    from raygent_harness.tools.file_tools import FileToolingRuntime
    from raygent_harness.tools.search_backend import SearchBackend

RaygentToolProfile = Literal["none", "file", "project"]
RaygentContextProfile = Literal["none", "environment", "project"]
RaygentToolSelection = RaygentToolProfile | Sequence[Tool]
RaygentContextSelection = RaygentContextProfile | Sequence[ContextProvider]
RaygentSDKMessageCallback = Callable[[SDKMessage], object]
RaygentSDKResultCallback = Callable[[SDKResult], object]
RaygentKernelEventCallback = Callable[[KernelEvent], None]

class RaygentSDKError(Exception):
    """Base exception for the embedding factory/session layer."""


class RaygentSessionBusyError(RaygentSDKError):
    """Raised when a `RaygentSession` already has an active submitted turn."""


class RaygentSessionClosedError(RaygentSDKError):
    """Raised when a closed `RaygentSession` is used for another turn."""


class RaygentSDKProtocolError(RaygentSDKError):
    """Raised when the wrapped kernel stream violates the SDK result contract."""


@dataclass(frozen=True)
class RaygentRunCallbacks:
    """Per-run adapter callbacks over existing SDK/kernel event streams.

    `on_message` receives every yielded `SDKMessage`, including the terminal
    `SDKResult`. `on_result` receives only terminal `SDKResult` messages.
    `on_kernel_event` is attached to the session's existing `KernelEventBus`
    for the duration of the run and is removed when the run generator closes.
    """

    on_message: RaygentSDKMessageCallback | None = None
    on_result: RaygentSDKResultCallback | None = None
    on_kernel_event: RaygentKernelEventCallback | None = None


@dataclass(frozen=True)
class RaygentKernelEventCallbackSink:
    """Kernel event sink that forwards events to a callback."""

    callback: RaygentKernelEventCallback

    def emit(self, event: KernelEvent) -> None:
        self.callback(event)


@dataclass
class RaygentCallbackHandle:
    """Detachable callback registration returned by SDK helper methods."""

    _close: Callable[[], None]
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close()

    def __enter__(self) -> RaygentCallbackHandle:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass(frozen=True)
class RaygentRuntimeHandles:
    """Inspectable runtime handles owned by one SDK session.

    The handles are stable references to existing kernel/runtime objects, not a
    second lifecycle manager. Advanced embedders can inspect them or pass them to
    lower-level services when they need task output, replay, or observability.
    """

    session_id: str
    cwd: str
    task_store: AppStateStore
    output_dir: Path
    task_output_store: TaskOutputStore
    transcript_store: TranscriptStore | None
    transcript_scope: TranscriptScope | None
    observability: KernelEventBus
    abort_event: asyncio.Event

    @property
    def transcript_path(self) -> str | None:
        """Physical transcript path when the configured store exposes one."""

        if self.transcript_store is None or self.transcript_scope is None:
            return None
        return self.transcript_store.path_for(self.transcript_scope)


@dataclass(frozen=True)
class RaygentToolProfileOptions:
    """Options for SDK-owned tool profiles.

    Shell remains opt-in even for the `"project"` profile because Bash has open
    world/process side effects despite Raygent's restricted command validator.
    """

    enable_bash: bool = False
    search_backend: SearchBackend | None = None


@dataclass(frozen=True)
class RaygentContextProfileOptions:
    """Options for SDK-owned context profiles."""

    workspace_root: str | Path | None = None
    today: TodayProvider | None = None
    git_command_runner: GitCommandRunner | None = None
    user_instruction_paths: tuple[str | Path, ...] = ()
    project_filenames: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
    project_rule_dirs: tuple[str | Path, ...] = (".claude/rules",)
    local_filenames: tuple[str, ...] = ("AGENTS.local.md", "CLAUDE.local.md")
    additional_dirs: tuple[str | Path, ...] = ()
    discovery_mode: DiscoveryMode = "layered_ancestors"
    allow_instruction_includes: bool = True
    allow_external_instruction_includes: bool = False


@dataclass(frozen=True)
class RaygentModelOptions:
    """Model-turn options that map to `QueryConfig` and `QueryDeps`.

    This group is a discoverability layer over existing provider-neutral model
    seams. It does not create vendor clients, load credentials, or choose model
    aliases for the caller.
    """

    provider: ModelProvider | None = None
    model: str | None = None
    fallback_model: str | None = None
    sampling: SamplingParams | None = None
    budget: TurnBudget | None = None
    system_prompt: str | None = None
    experiments: dict[str, bool] | None = None


@dataclass(frozen=True)
class RaygentSessionOptions:
    """Session identity and runtime-root options.

    `cwd` controls relative file/task-output resolution. `session_id` controls
    transcript/task-output scoping. Both remain caller-owned product policy.
    """

    cwd: str | Path | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class RaygentToolOptions:
    """Tooling options that map to catalog, hook, and media-service seams.

    This group does not enable dangerous tools automatically. Supplying tools,
    a catalog provider, hooks, or a PDF service only wires already-constructed
    kernel dependencies into the session.
    """

    tools: RaygentToolSelection | None = None
    profile_options: RaygentToolProfileOptions | None = None
    catalog_provider: ToolCatalogProvider | None = None
    pre_tool_use_hooks: tuple[PreToolUseHook, ...] = ()
    post_tool_use_hooks: tuple[PostToolUseHook, ...] = ()
    post_tool_use_failure_hooks: tuple[PostToolUseFailureHook, ...] = ()
    max_tool_use_concurrency: int | None = None
    pdf_document_service: PdfDocumentService | None = None


@dataclass(frozen=True)
class RaygentContextOptions:
    """Context options for turn-time context and prompt-provider seams.

    `context` may be an SDK-owned profile (`"none"`, `"environment"`,
    `"project"`) or explicit context providers. Post-tool and system-prompt
    providers are injected as-is; Raygent does not infer product context policy.
    """

    context: RaygentContextSelection | None = None
    profile_options: RaygentContextProfileOptions | None = None
    post_tool_context_providers: tuple[PostToolContextProvider, ...] = ()
    system_prompt_provider: SystemPromptProvider | None = None


@dataclass(frozen=True)
class RaygentPermissionOptions:
    """Permission options for headless HITL/product policy adapters.

    The kernel exposes the permission chokepoints; embedders decide whether to
    run non-interactively, ask a UI, or install a custom engine.
    """

    context: ToolPermissionContext | None = None
    handler: PermissionHandler | None = None
    engine: ToolPermissionEngine | None = None
    is_non_interactive_session: bool | None = None


@dataclass(frozen=True)
class RaygentMemoryOptions:
    """Memory options for explicit prompt, recall, and extraction providers.

    Passing this group wires caller-provided memory services into `QueryDeps`.
    It does not create memdir settings, write files, or start extraction by
    default.
    """

    prompt_provider: MemoryPromptProvider | None = None
    recall_provider: MemoryRecallProvider | None = None
    extractor: MemoryExtractor | None = None


@dataclass(frozen=True)
class RaygentPersistenceOptions:
    """Persistence options for stores and recovery sidecars.

    These options expose existing kernel stores without selecting a storage
    backend. `task_output_dir` is still session-scoped by `RaygentFactory`.
    """

    task_store: AppStateStore | None = None
    transcript_store: TranscriptStore | None = None
    task_output_dir: str | Path | None = None
    worktree_manager: WorktreeManager | None = None
    remote_agent_persistence_store: RemoteAgentPersistenceStore | None = None
    agent_route_record_store: AgentRouteRecordStore | None = None


@dataclass(frozen=True)
class RaygentAgentOptions:
    """Agent, skill, coordinator, and handoff integration options.

    This group wires existing agent/coordinator seams only. It does not launch
    hidden agents, pick team policy, allocate worktrees, or choose remotes.
    """

    skill_provider: SkillProvider | None = None
    agent_trigger_policy: AgentTriggerPolicySpec | None = None
    coordinator_runtime: CoordinatorRuntimeProtocol | None = None
    remote_agent_backend: RemoteAgentBackend | None = None
    handoff_classifier: AgentHandoffClassifier | None = None
    handoff_classifier_timeout_s: float | None = None


@dataclass(frozen=True)
class RaygentObservabilityOptions:
    """Observability and notification options for embedding adapters.

    Event buses, sinks, clocks, and notification callbacks are observation or
    adapter seams. They must not become model-visible policy by themselves.
    """

    observability: KernelEventBus | Sequence[KernelEventSink] | None = None
    notify: NotificationSink | None = None
    clock: Clock | None = None


@dataclass(frozen=True)
class RaygentFactoryConfig:
    """Reusable input object for `create_raygent(...)`.

    The defaults are intentionally safe and headless: no tools, no automatic
    context, no memory writes, no transcript store, no remote backend, and no
    product telemetry.
    """

    provider: ModelProvider | None = None
    model: str | None = None
    cwd: str | Path = "."
    session_id: str | None = None
    system_prompt: str = ""
    fallback_model: str | None = None
    sampling: SamplingParams = field(default_factory=SamplingParams)
    budget: TurnBudget = field(default_factory=TurnBudget)
    tools: RaygentToolSelection = "none"
    tool_profile_options: RaygentToolProfileOptions = field(
        default_factory=RaygentToolProfileOptions
    )
    tool_catalog_provider: ToolCatalogProvider | None = None
    context: RaygentContextSelection = "none"
    context_profile_options: RaygentContextProfileOptions = field(
        default_factory=RaygentContextProfileOptions
    )
    permission_context: ToolPermissionContext | None = None
    permission_handler: PermissionHandler | None = None
    is_non_interactive_session: bool = False
    observability: KernelEventBus | Sequence[KernelEventSink] | None = None
    transcript_store: TranscriptStore | None = None
    task_store: AppStateStore | None = None
    task_output_dir: str | Path | None = None
    pdf_document_service: PdfDocumentService | None = None
    notify: NotificationSink | None = None
    experiments: dict[str, bool] = field(default_factory=dict[str, bool])
    model_options: RaygentModelOptions | None = None
    session_options: RaygentSessionOptions | None = None
    tool_options: RaygentToolOptions | None = None
    context_options: RaygentContextOptions | None = None
    permission_options: RaygentPermissionOptions | None = None
    memory_options: RaygentMemoryOptions | None = None
    persistence_options: RaygentPersistenceOptions | None = None
    agent_options: RaygentAgentOptions | None = None
    observability_options: RaygentObservabilityOptions | None = None


@dataclass
class RaygentSession:
    """Conversation-scoped wrapper over `QueryEngine` for embedders.

    `engine`, `config`, `deps`, and `ctx` are intentionally exposed as advanced
    mutable escape hatches. Inspecting them is fine; mutating them while a turn
    is active can break the same cross-turn invariants that `QueryEngine` owns.
    """

    engine: QueryEngine
    config: QueryConfig
    deps: QueryDeps
    ctx: ToolUseContext
    _handles: RaygentRuntimeHandles | None = field(default=None, repr=False)
    _active_turn: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _memory_extractions_drained: bool = field(default=True, init=False, repr=False)

    async def __aenter__(self) -> RaygentSession:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    @property
    def session_id(self) -> str:
        return self.config.session_id or self.ctx.session_id

    @property
    def cwd(self) -> str:
        return self.ctx.cwd

    @property
    def abort_event(self) -> asyncio.Event:
        return self.ctx.abort_event

    @property
    def task_store(self) -> AppStateStore:
        return self.deps.task_store

    @property
    def observability(self) -> KernelEventBus:
        return self.deps.observability

    @property
    def handles(self) -> RaygentRuntimeHandles:
        if self._handles is None:
            self._handles = _default_runtime_handles(
                self.engine,
                self.config,
                self.deps,
                self.ctx,
            )
        return self._handles

    @property
    def output_dir(self) -> Path:
        return self.handles.output_dir

    @property
    def task_output_store(self) -> TaskOutputStore:
        return self.handles.task_output_store

    @property
    def transcript_store(self) -> TranscriptStore | None:
        return self.handles.transcript_store

    @property
    def transcript_scope(self) -> TranscriptScope | None:
        return self.handles.transcript_scope

    @property
    def transcript_path(self) -> str | None:
        return self.handles.transcript_path

    @property
    def is_closed(self) -> bool:
        return self._closed

    def abort(self) -> None:
        """Signal cooperative cancellation for the current/future turn."""

        self.ctx.abort_event.set()

    def add_kernel_event_callback(
        self,
        callback: RaygentKernelEventCallback,
    ) -> RaygentCallbackHandle:
        """Attach a callback sink to this session's kernel event bus."""

        sink = RaygentKernelEventCallbackSink(callback)
        self.observability.add_sink(sink)
        return RaygentCallbackHandle(lambda: self.observability.remove_sink(sink))

    async def close(
        self,
        *,
        drain_memory_extractions_timeout_s: float | None = 60.0,
    ) -> bool:
        """Close the session lifecycle boundary.

        Close is cooperative and idempotent: it signals abort, drains scheduled
        memory extraction tasks when possible, flushes the optional transcript
        store, and rejects future turns. It intentionally does not kill tasks in
        the shared `AppStateStore`; task ownership remains with the task APIs.

        Returns True when background memory extraction tasks fully drained, or
        False when the timeout expired.
        """

        if self._active_turn:
            self.abort()
            raise RaygentSessionBusyError(
                "RaygentSession has an active turn; abort was signaled, but close "
                "requires the run stream to finish or be aclosed first."
            )
        if not self._closed:
            self._closed = True
            self.abort()
        self._memory_extractions_drained = await self.engine.drain_memory_extractions(
            timeout_s=drain_memory_extractions_timeout_s,
        )
        await _flush_transcript_store(
            self.handles.transcript_store,
            self.handles.transcript_scope,
        )
        return self._memory_extractions_drained

    def run(
        self,
        prompt: str,
        *,
        callbacks: RaygentRunCallbacks | None = None,
    ) -> AsyncIterator[SDKMessage]:
        """Alias for `submit_message(...)` with embedding-friendly naming."""

        return self.submit_message(prompt, callbacks=callbacks)

    async def submit_message(
        self,
        prompt: str,
        *,
        callbacks: RaygentRunCallbacks | None = None,
    ) -> AsyncIterator[SDKMessage]:
        """Run one turn and yield the existing `QueryEngine` SDK message stream."""

        if self._closed:
            raise RaygentSessionClosedError(
                "RaygentSession is closed; create a new session or resume from a "
                "transcript/recovery service before submitting another turn."
            )
        if self._active_turn:
            raise RaygentSessionBusyError(
                "RaygentSession already has an active turn; create another session "
                "or wait for the current run to finish."
            )
        self._active_turn = True
        callback_handle: RaygentCallbackHandle | None = None
        if callbacks is not None and callbacks.on_kernel_event is not None:
            callback_handle = self.add_kernel_event_callback(callbacks.on_kernel_event)
        try:
            async with contextlib.aclosing(
                self.engine.submit_message(prompt)
            ) as engine_stream:
                async for event in engine_stream:
                    await _invoke_run_callbacks(callbacks, event)
                    yield event
        finally:
            if callback_handle is not None:
                callback_handle.close()
            self._active_turn = False

    async def run_until_result(
        self,
        prompt: str,
        *,
        callbacks: RaygentRunCallbacks | None = None,
    ) -> SDKResult:
        """Consume one turn and return the terminal `SDKResult` message."""

        result: SDKResult | None = None
        async for event in self.submit_message(prompt, callbacks=callbacks):
            if isinstance(event, SDKResult):
                result = event
        if result is None:
            raise RaygentSDKProtocolError(
                "Raygent QueryEngine stream ended without an SDKResult terminal event."
            )
        return result


class RaygentSessionFactory(Protocol):
    """Narrow factory interface for product/application integration layers.

    Product code can accept this Protocol instead of depending on the concrete
    `RaygentFactory` class or the wide `create_raygent(...)` keyword surface.
    """

    def create_session(
        self,
        config: RaygentFactoryConfig,
        /,
    ) -> RaygentSession:
        """Create one fresh `RaygentSession` from an explicit config object."""
        ...


@dataclass(frozen=True)
class _ResolvedFactorySettings:
    provider: ModelProvider
    model: str
    cwd: str | Path
    session_id: str | None
    system_prompt: str
    fallback_model: str | None
    sampling: SamplingParams
    budget: TurnBudget
    tools: RaygentToolSelection
    tool_profile_options: RaygentToolProfileOptions
    tool_catalog_provider: ToolCatalogProvider | None
    pre_tool_use_hooks: tuple[PreToolUseHook, ...]
    post_tool_use_hooks: tuple[PostToolUseHook, ...]
    post_tool_use_failure_hooks: tuple[PostToolUseFailureHook, ...]
    max_tool_use_concurrency: int
    pdf_document_service: PdfDocumentService | None
    context: RaygentContextSelection
    context_profile_options: RaygentContextProfileOptions
    post_tool_context_providers: tuple[PostToolContextProvider, ...]
    system_prompt_provider: SystemPromptProvider | None
    permission_context: ToolPermissionContext | None
    permission_handler: PermissionHandler | None
    permission_engine: ToolPermissionEngine | None
    is_non_interactive_session: bool
    memory_prompt_provider: MemoryPromptProvider | None
    memory_recall_provider: MemoryRecallProvider | None
    memory_extractor: MemoryExtractor | None
    skill_provider: SkillProvider | None
    agent_trigger_policy: AgentTriggerPolicySpec | None
    coordinator_runtime: CoordinatorRuntimeProtocol | None
    remote_agent_backend: RemoteAgentBackend | None
    handoff_classifier: AgentHandoffClassifier | None
    handoff_classifier_timeout_s: float
    observability: KernelEventBus | Sequence[KernelEventSink] | None
    notify: NotificationSink | None
    clock: Clock | None
    transcript_store: TranscriptStore | None
    task_store: AppStateStore | None
    task_output_dir: str | Path | None
    worktree_manager: WorktreeManager | None
    remote_agent_persistence_store: RemoteAgentPersistenceStore | None
    agent_route_record_store: AgentRouteRecordStore | None
    experiments: dict[str, bool]


@dataclass(frozen=True)
class RaygentFactory:
    """Reusable kernel assembly object for headless Raygent sessions.

    The factory owns no mutable session lifecycle state. Each
    `create_session(...)` call allocates a fresh session id when needed, abort
    event, task-output store, transcript scope, `QueryEngine`, and
    `RaygentRuntimeHandles`.
    """

    def create_session(
        self,
        config: RaygentFactoryConfig,
        /,
    ) -> RaygentSession:
        """Create a headless Raygent session from provider-neutral config."""

        settings = _resolve_factory_settings(config)
        runtime_root = Path(settings.cwd).expanduser().resolve()
        resolved_session_id = settings.session_id or f"raygent-{uuid.uuid4().hex}"
        permission_ctx = settings.permission_context or empty_tool_permission_context()
        resolved_tools = _resolve_initial_tools(settings.tools)
        context_providers = _resolve_context_providers(
            settings.context,
            runtime_root=runtime_root,
            options=settings.context_profile_options,
        )
        observability_bus = _resolve_observability(settings.observability)
        task_store_obj = settings.task_store or AppStateStore()
        output_dir = _resolve_output_dir(settings.task_output_dir, runtime_root)
        from raygent_harness.services.task_output import FileTaskOutputStore
        from raygent_harness.services.transcript import TranscriptScope

        transcript_scope = TranscriptScope(session_id=resolved_session_id)
        task_output_store = FileTaskOutputStore(
            base_dir=output_dir,
            session_id=resolved_session_id,
        )
        tool_profile_runtime = _build_tool_profile_runtime(
            settings.tools,
            pdf_document_service=settings.pdf_document_service,
            options=settings.tool_profile_options,
            task_output_store=task_output_store,
        )

        query_config = QueryConfig(
            model=settings.model,
            fallback_model=settings.fallback_model,
            sampling=settings.sampling,
            system_prompt=settings.system_prompt,
            tools=resolved_tools,
            budget=settings.budget,
            session_id=resolved_session_id,
            experiments=dict(settings.experiments),
        )
        ctx = ToolUseContext(
            session_id=resolved_session_id,
            agent_id=None,
            abort_event=asyncio.Event(),
            rendered_system_prompt="",
            cwd=str(runtime_root),
            tools=resolved_tools,
            permission_context=permission_ctx,
            query_tracking=QueryTracking(chain_id=resolved_session_id, depth=0),
            add_notification=settings.notify,
        )
        deps = QueryDeps(
            task_store=task_store_obj,
            model_provider=settings.provider,
            context_providers=context_providers,
            post_tool_context_providers=settings.post_tool_context_providers,
            system_prompt_provider=settings.system_prompt_provider,
            memory_prompt_provider=settings.memory_prompt_provider,
            memory_extractor=settings.memory_extractor,
            memory_recall_provider=settings.memory_recall_provider,
            agent_trigger_policy=settings.agent_trigger_policy,
            coordinator_runtime=settings.coordinator_runtime,
            permission_context=permission_ctx,
            permission_handler=settings.permission_handler,
            is_non_interactive_session=settings.is_non_interactive_session,
            skill_provider=settings.skill_provider,
            notify=settings.notify or _noop_notification_sink,
            output_dir=str(output_dir),
            transcript_store=settings.transcript_store,
            pdf_document_service=settings.pdf_document_service,
            observability=observability_bus,
            pre_tool_use_hooks=[
                *tool_profile_runtime.pre_tool_use_hooks,
                *settings.pre_tool_use_hooks,
            ],
            post_tool_use_hooks=[
                *tool_profile_runtime.post_tool_use_hooks,
                *settings.post_tool_use_hooks,
            ],
            post_tool_use_failure_hooks=list(settings.post_tool_use_failure_hooks),
            max_tool_use_concurrency=settings.max_tool_use_concurrency,
            worktree_manager=settings.worktree_manager,
            remote_agent_backend=settings.remote_agent_backend,
            remote_agent_persistence_store=settings.remote_agent_persistence_store,
            agent_route_record_store=settings.agent_route_record_store,
            handoff_classifier=settings.handoff_classifier,
            handoff_classifier_timeout_s=settings.handoff_classifier_timeout_s,
        )
        if settings.permission_engine is not None:
            deps.permission_engine = settings.permission_engine
        if settings.clock is not None:
            deps.clock = settings.clock
        deps.tool_catalog_provider = _build_tool_catalog_provider(
            tool_profile_runtime,
            deps=deps,
            upstream=settings.tool_catalog_provider,
        )
        engine = QueryEngine(query_config, deps, ctx, transcript_scope=transcript_scope)
        handles = RaygentRuntimeHandles(
            session_id=resolved_session_id,
            cwd=str(runtime_root),
            task_store=task_store_obj,
            output_dir=output_dir,
            task_output_store=task_output_store,
            transcript_store=settings.transcript_store,
            transcript_scope=transcript_scope,
            observability=observability_bus,
            abort_event=ctx.abort_event,
        )
        return RaygentSession(
            engine=engine,
            config=query_config,
            deps=deps,
            ctx=ctx,
            _handles=handles,
        )


def create_raygent(
    config: RaygentFactoryConfig | None = None,
    *,
    provider: ModelProvider | None = None,
    model: str | None = None,
    model_options: RaygentModelOptions | None = None,
    session_options: RaygentSessionOptions | None = None,
    tool_options: RaygentToolOptions | None = None,
    context_options: RaygentContextOptions | None = None,
    permission_options: RaygentPermissionOptions | None = None,
    memory_options: RaygentMemoryOptions | None = None,
    persistence_options: RaygentPersistenceOptions | None = None,
    agent_options: RaygentAgentOptions | None = None,
    observability_options: RaygentObservabilityOptions | None = None,
    cwd: str | Path = ".",
    session_id: str | None = None,
    system_prompt: str = "",
    fallback_model: str | None = None,
    sampling: SamplingParams | None = None,
    budget: TurnBudget | None = None,
    tools: RaygentToolSelection = "none",
    tool_profile_options: RaygentToolProfileOptions | None = None,
    tool_catalog_provider: ToolCatalogProvider | None = None,
    context: RaygentContextSelection = "none",
    context_profile_options: RaygentContextProfileOptions | None = None,
    permission_context: ToolPermissionContext | None = None,
    permission_handler: PermissionHandler | None = None,
    is_non_interactive_session: bool = False,
    observability: KernelEventBus | Sequence[KernelEventSink] | None = None,
    transcript_store: TranscriptStore | None = None,
    task_store: AppStateStore | None = None,
    task_output_dir: str | Path | None = None,
    pdf_document_service: PdfDocumentService | None = None,
    notify: NotificationSink | None = None,
    experiments: dict[str, bool] | None = None,
) -> RaygentSession:
    """Create a headless Raygent session from provider-neutral dependencies."""

    factory_config = _coerce_factory_config(
        config,
        provider=provider,
        model=model,
        model_options=model_options,
        session_options=session_options,
        tool_options=tool_options,
        context_options=context_options,
        permission_options=permission_options,
        memory_options=memory_options,
        persistence_options=persistence_options,
        agent_options=agent_options,
        observability_options=observability_options,
        cwd=cwd,
        session_id=session_id,
        system_prompt=system_prompt,
        fallback_model=fallback_model,
        sampling=sampling,
        budget=budget,
        tools=tools,
        tool_profile_options=tool_profile_options,
        tool_catalog_provider=tool_catalog_provider,
        context=context,
        context_profile_options=context_profile_options,
        permission_context=permission_context,
        permission_handler=permission_handler,
        is_non_interactive_session=is_non_interactive_session,
        observability=observability,
        transcript_store=transcript_store,
        task_store=task_store,
        task_output_dir=task_output_dir,
        pdf_document_service=pdf_document_service,
        notify=notify,
        experiments=experiments,
    )
    return RaygentFactory().create_session(factory_config)


def _coerce_factory_config(
    config: RaygentFactoryConfig | None,
    *,
    provider: ModelProvider | None,
    model: str | None,
    model_options: RaygentModelOptions | None,
    session_options: RaygentSessionOptions | None,
    tool_options: RaygentToolOptions | None,
    context_options: RaygentContextOptions | None,
    permission_options: RaygentPermissionOptions | None,
    memory_options: RaygentMemoryOptions | None,
    persistence_options: RaygentPersistenceOptions | None,
    agent_options: RaygentAgentOptions | None,
    observability_options: RaygentObservabilityOptions | None,
    cwd: str | Path,
    session_id: str | None,
    system_prompt: str,
    fallback_model: str | None,
    sampling: SamplingParams | None,
    budget: TurnBudget | None,
    tools: RaygentToolSelection,
    tool_profile_options: RaygentToolProfileOptions | None,
    tool_catalog_provider: ToolCatalogProvider | None,
    context: RaygentContextSelection,
    context_profile_options: RaygentContextProfileOptions | None,
    permission_context: ToolPermissionContext | None,
    permission_handler: PermissionHandler | None,
    is_non_interactive_session: bool,
    observability: KernelEventBus | Sequence[KernelEventSink] | None,
    transcript_store: TranscriptStore | None,
    task_store: AppStateStore | None,
    task_output_dir: str | Path | None,
    pdf_document_service: PdfDocumentService | None,
    notify: NotificationSink | None,
    experiments: dict[str, bool] | None,
) -> RaygentFactoryConfig:
    if config is not None:
        overrides = _config_override_names(
            provider=provider,
            model=model,
            model_options=model_options,
            session_options=session_options,
            tool_options=tool_options,
            context_options=context_options,
            permission_options=permission_options,
            memory_options=memory_options,
            persistence_options=persistence_options,
            agent_options=agent_options,
            observability_options=observability_options,
            cwd=cwd,
            session_id=session_id,
            system_prompt=system_prompt,
            fallback_model=fallback_model,
            sampling=sampling,
            budget=budget,
            tools=tools,
            tool_profile_options=tool_profile_options,
            tool_catalog_provider=tool_catalog_provider,
            context=context,
            context_profile_options=context_profile_options,
            permission_context=permission_context,
            permission_handler=permission_handler,
            is_non_interactive_session=is_non_interactive_session,
            observability=observability,
            transcript_store=transcript_store,
            task_store=task_store,
            task_output_dir=task_output_dir,
            pdf_document_service=pdf_document_service,
            notify=notify,
            experiments=experiments,
        )
        if overrides:
            joined = ", ".join(overrides)
            raise ValueError(
                "Pass either RaygentFactoryConfig or keyword overrides, not both "
                f"(got: {joined})."
            )
        return config
    option_provider = model_options.provider if model_options is not None else None
    option_model = model_options.model if model_options is not None else None
    if (provider is None and option_provider is None) or (
        model is None and option_model is None
    ):
        raise ValueError("create_raygent requires provider and model when config is not supplied.")
    return RaygentFactoryConfig(
        provider=provider,
        model=model,
        model_options=model_options,
        session_options=session_options,
        tool_options=tool_options,
        context_options=context_options,
        permission_options=permission_options,
        memory_options=memory_options,
        persistence_options=persistence_options,
        agent_options=agent_options,
        observability_options=observability_options,
        cwd=cwd,
        session_id=session_id,
        system_prompt=system_prompt,
        fallback_model=fallback_model,
        sampling=sampling or SamplingParams(),
        budget=budget or TurnBudget(),
        tools=tools,
        tool_profile_options=tool_profile_options or RaygentToolProfileOptions(),
        tool_catalog_provider=tool_catalog_provider,
        context=context,
        context_profile_options=context_profile_options or RaygentContextProfileOptions(),
        permission_context=permission_context,
        permission_handler=permission_handler,
        is_non_interactive_session=is_non_interactive_session,
        observability=observability,
        transcript_store=transcript_store,
        task_store=task_store,
        task_output_dir=task_output_dir,
        pdf_document_service=pdf_document_service,
        notify=notify,
        experiments=experiments or {},
    )


def _resolve_factory_settings(config: RaygentFactoryConfig) -> _ResolvedFactorySettings:
    model_options = config.model_options
    session_options = config.session_options
    tool_options = config.tool_options
    context_options = config.context_options
    permission_options = config.permission_options
    memory_options = config.memory_options
    persistence_options = config.persistence_options
    agent_options = config.agent_options
    observability_options = config.observability_options

    provider = config.provider
    if provider is None and model_options is not None:
        provider = model_options.provider
    model = config.model
    if model is None and model_options is not None:
        model = model_options.model
    if provider is None or model is None:
        raise ValueError("RaygentFactoryConfig requires provider and model.")

    sampling = config.sampling
    if (
        sampling == SamplingParams()
        and model_options is not None
        and model_options.sampling is not None
    ):
        sampling = model_options.sampling
    budget = config.budget
    if (
        budget == TurnBudget()
        and model_options is not None
        and model_options.budget is not None
    ):
        budget = model_options.budget

    return _ResolvedFactorySettings(
        provider=provider,
        model=model,
        cwd=_select_defaulted(
            config.cwd,
            default=".",
            option=session_options.cwd if session_options is not None else None,
        ),
        session_id=_select_optional(
            config.session_id,
            session_options.session_id if session_options is not None else None,
        ),
        system_prompt=_select_defaulted(
            config.system_prompt,
            default="",
            option=(
                model_options.system_prompt if model_options is not None else None
            ),
        ),
        fallback_model=_select_optional(
            config.fallback_model,
            model_options.fallback_model if model_options is not None else None,
        ),
        sampling=sampling,
        budget=budget,
        tools=_select_defaulted(
            config.tools,
            default="none",
            option=tool_options.tools if tool_options is not None else None,
        ),
        tool_profile_options=_select_defaulted(
            config.tool_profile_options,
            default=RaygentToolProfileOptions(),
            option=(
                tool_options.profile_options if tool_options is not None else None
            ),
        ),
        tool_catalog_provider=_select_optional(
            config.tool_catalog_provider,
            tool_options.catalog_provider if tool_options is not None else None,
        ),
        pre_tool_use_hooks=(
            tool_options.pre_tool_use_hooks if tool_options is not None else ()
        ),
        post_tool_use_hooks=(
            tool_options.post_tool_use_hooks if tool_options is not None else ()
        ),
        post_tool_use_failure_hooks=(
            tool_options.post_tool_use_failure_hooks
            if tool_options is not None
            else ()
        ),
        max_tool_use_concurrency=(
            tool_options.max_tool_use_concurrency
            if tool_options is not None
            and tool_options.max_tool_use_concurrency is not None
            else 10
        ),
        pdf_document_service=_select_optional(
            config.pdf_document_service,
            tool_options.pdf_document_service if tool_options is not None else None,
        ),
        context=_select_defaulted(
            config.context,
            default="none",
            option=context_options.context if context_options is not None else None,
        ),
        context_profile_options=_select_defaulted(
            config.context_profile_options,
            default=RaygentContextProfileOptions(),
            option=(
                context_options.profile_options
                if context_options is not None
                else None
            ),
        ),
        post_tool_context_providers=(
            context_options.post_tool_context_providers
            if context_options is not None
            else ()
        ),
        system_prompt_provider=(
            context_options.system_prompt_provider
            if context_options is not None
            else None
        ),
        permission_context=_select_optional(
            config.permission_context,
            permission_options.context if permission_options is not None else None,
        ),
        permission_handler=_select_optional(
            config.permission_handler,
            permission_options.handler if permission_options is not None else None,
        ),
        permission_engine=(
            permission_options.engine if permission_options is not None else None
        ),
        is_non_interactive_session=(
            config.is_non_interactive_session
            if config.is_non_interactive_session
            else (
                permission_options.is_non_interactive_session
                if permission_options is not None
                and permission_options.is_non_interactive_session is not None
                else False
            )
        ),
        memory_prompt_provider=(
            memory_options.prompt_provider if memory_options is not None else None
        ),
        memory_recall_provider=(
            memory_options.recall_provider if memory_options is not None else None
        ),
        memory_extractor=(
            memory_options.extractor if memory_options is not None else None
        ),
        skill_provider=agent_options.skill_provider if agent_options is not None else None,
        agent_trigger_policy=(
            agent_options.agent_trigger_policy if agent_options is not None else None
        ),
        coordinator_runtime=(
            agent_options.coordinator_runtime if agent_options is not None else None
        ),
        remote_agent_backend=(
            agent_options.remote_agent_backend if agent_options is not None else None
        ),
        handoff_classifier=(
            agent_options.handoff_classifier if agent_options is not None else None
        ),
        handoff_classifier_timeout_s=(
            agent_options.handoff_classifier_timeout_s
            if agent_options is not None
            and agent_options.handoff_classifier_timeout_s is not None
            else 2.0
        ),
        observability=_select_optional(
            config.observability,
            (
                observability_options.observability
                if observability_options is not None
                else None
            ),
        ),
        notify=_select_optional(
            config.notify,
            observability_options.notify if observability_options is not None else None,
        ),
        clock=observability_options.clock if observability_options is not None else None,
        transcript_store=_select_optional(
            config.transcript_store,
            (
                persistence_options.transcript_store
                if persistence_options is not None
                else None
            ),
        ),
        task_store=_select_optional(
            config.task_store,
            persistence_options.task_store if persistence_options is not None else None,
        ),
        task_output_dir=_select_optional(
            config.task_output_dir,
            (
                persistence_options.task_output_dir
                if persistence_options is not None
                else None
            ),
        ),
        worktree_manager=(
            persistence_options.worktree_manager
            if persistence_options is not None
            else None
        ),
        remote_agent_persistence_store=(
            persistence_options.remote_agent_persistence_store
            if persistence_options is not None
            else None
        ),
        agent_route_record_store=(
            persistence_options.agent_route_record_store
            if persistence_options is not None
            else None
        ),
        experiments=(
            dict(config.experiments)
            if config.experiments
            else dict(model_options.experiments)
            if model_options is not None and model_options.experiments is not None
            else {}
        ),
    )


def _select_optional[ValueT](
    flat: ValueT | None,
    option: ValueT | None,
) -> ValueT | None:
    return flat if flat is not None else option


def _select_defaulted[ValueT](
    flat: ValueT,
    *,
    default: ValueT,
    option: ValueT | None,
) -> ValueT:
    if flat != default:
        return flat
    if option is not None:
        return option
    return flat


def _config_override_names(
    *,
    provider: ModelProvider | None,
    model: str | None,
    model_options: RaygentModelOptions | None,
    session_options: RaygentSessionOptions | None,
    tool_options: RaygentToolOptions | None,
    context_options: RaygentContextOptions | None,
    permission_options: RaygentPermissionOptions | None,
    memory_options: RaygentMemoryOptions | None,
    persistence_options: RaygentPersistenceOptions | None,
    agent_options: RaygentAgentOptions | None,
    observability_options: RaygentObservabilityOptions | None,
    cwd: str | Path,
    session_id: str | None,
    system_prompt: str,
    fallback_model: str | None,
    sampling: SamplingParams | None,
    budget: TurnBudget | None,
    tools: RaygentToolSelection,
    tool_profile_options: RaygentToolProfileOptions | None,
    tool_catalog_provider: ToolCatalogProvider | None,
    context: RaygentContextSelection,
    context_profile_options: RaygentContextProfileOptions | None,
    permission_context: ToolPermissionContext | None,
    permission_handler: PermissionHandler | None,
    is_non_interactive_session: bool,
    observability: KernelEventBus | Sequence[KernelEventSink] | None,
    transcript_store: TranscriptStore | None,
    task_store: AppStateStore | None,
    task_output_dir: str | Path | None,
    pdf_document_service: PdfDocumentService | None,
    notify: NotificationSink | None,
    experiments: dict[str, bool] | None,
) -> tuple[str, ...]:
    names: list[str] = []
    if provider is not None:
        names.append("provider")
    if model is not None:
        names.append("model")
    if model_options is not None:
        names.append("model_options")
    if session_options is not None:
        names.append("session_options")
    if tool_options is not None:
        names.append("tool_options")
    if context_options is not None:
        names.append("context_options")
    if permission_options is not None:
        names.append("permission_options")
    if memory_options is not None:
        names.append("memory_options")
    if persistence_options is not None:
        names.append("persistence_options")
    if agent_options is not None:
        names.append("agent_options")
    if observability_options is not None:
        names.append("observability_options")
    if cwd != ".":
        names.append("cwd")
    if session_id is not None:
        names.append("session_id")
    if system_prompt != "":
        names.append("system_prompt")
    if fallback_model is not None:
        names.append("fallback_model")
    if sampling is not None:
        names.append("sampling")
    if budget is not None:
        names.append("budget")
    if tools != "none":
        names.append("tools")
    if tool_profile_options is not None:
        names.append("tool_profile_options")
    if tool_catalog_provider is not None:
        names.append("tool_catalog_provider")
    if context != "none":
        names.append("context")
    if context_profile_options is not None:
        names.append("context_profile_options")
    if permission_context is not None:
        names.append("permission_context")
    if permission_handler is not None:
        names.append("permission_handler")
    if is_non_interactive_session:
        names.append("is_non_interactive_session")
    if observability is not None:
        names.append("observability")
    if transcript_store is not None:
        names.append("transcript_store")
    if task_store is not None:
        names.append("task_store")
    if task_output_dir is not None:
        names.append("task_output_dir")
    if pdf_document_service is not None:
        names.append("pdf_document_service")
    if notify is not None:
        names.append("notify")
    if experiments is not None:
        names.append("experiments")
    return tuple(names)


def _resolve_output_dir(
    task_output_dir: str | Path | None,
    runtime_root: Path,
) -> Path:
    if task_output_dir is None:
        return runtime_root / ".raygent" / "tasks"
    path = Path(task_output_dir).expanduser()
    if not path.is_absolute():
        path = runtime_root / path
    return path.resolve()


def _default_runtime_handles(
    engine: QueryEngine,
    config: QueryConfig,
    deps: QueryDeps,
    ctx: ToolUseContext,
) -> RaygentRuntimeHandles:
    """Build handles for manually constructed `RaygentSession` objects."""

    from raygent_harness.services.task_output import FileTaskOutputStore
    from raygent_harness.services.transcript import TranscriptScope

    session_id = config.session_id or ctx.session_id
    output_dir = Path(deps.output_dir).expanduser().resolve()
    missing_scope = object()
    engine_scope = getattr(engine, "transcript_scope", missing_scope)
    if engine_scope is missing_scope:
        transcript_scope: TranscriptScope | None = TranscriptScope(session_id=session_id)
    else:
        transcript_scope = cast("TranscriptScope | None", engine_scope)
    return RaygentRuntimeHandles(
        session_id=session_id,
        cwd=ctx.cwd,
        task_store=deps.task_store,
        output_dir=output_dir,
        task_output_store=FileTaskOutputStore(
            base_dir=output_dir,
            session_id=session_id,
        ),
        transcript_store=deps.transcript_store,
        transcript_scope=transcript_scope,
        observability=deps.observability,
        abort_event=ctx.abort_event,
    )


async def _flush_transcript_store(
    store: TranscriptStore | None,
    scope: TranscriptScope | None,
) -> None:
    if store is None or scope is None:
        return
    # QueryEngine treats transcript persistence as best-effort. Keep close()
    # aligned with that lifecycle contract rather than making close failures
    # depend on optional persistence backends.
    with contextlib.suppress(Exception):
        await store.flush(scope)


async def _invoke_run_callbacks(
    callbacks: RaygentRunCallbacks | None,
    event: SDKMessage,
) -> None:
    if callbacks is None:
        return
    if callbacks.on_message is not None:
        await _invoke_callback(callbacks.on_message, event)
    if isinstance(event, SDKResult) and callbacks.on_result is not None:
        await _invoke_callback(callbacks.on_result, event)


async def _invoke_callback[CallbackValue](
    callback: Callable[[CallbackValue], object],
    value: CallbackValue,
) -> None:
    result = callback(value)
    if inspect.isawaitable(result):
        await cast(Awaitable[object], result)


@dataclass(frozen=True)
class _ToolProfileRuntime:
    profile: RaygentToolProfile
    pre_tool_use_hooks: tuple[PreToolUseHook, ...] = ()
    post_tool_use_hooks: tuple[PostToolUseHook, ...] = ()
    file_runtime: FileToolingRuntime | None = None
    discovery_runtime: DiscoveryToolingRuntime | None = None
    task_output_store: TaskOutputStore | None = None
    enable_bash: bool = False


def _resolve_initial_tools(tools: RaygentToolSelection) -> tuple[Tool, ...]:
    if isinstance(tools, str):
        if tools not in {"none", "file", "project"}:
            raise ValueError(f"Unknown Raygent tool profile: {tools!r}")
        return ()
    return tuple(tools)


def _resolve_context_providers(
    context: RaygentContextSelection,
    *,
    runtime_root: Path,
    options: RaygentContextProfileOptions,
) -> tuple[ContextProvider, ...]:
    if isinstance(context, str):
        if context == "none":
            return ()
        from raygent_harness.context_providers.defaults import (
            build_default_context_providers,
        )

        if context == "environment":
            return build_default_context_providers(
                cwd=runtime_root,
                workspace_root=options.workspace_root,
                include_environment=True,
                include_git_status=False,
                include_project_instructions=False,
                today=options.today,
                git_command_runner=options.git_command_runner,
            )
        if context == "project":
            return build_default_context_providers(
                cwd=runtime_root,
                workspace_root=options.workspace_root,
                include_environment=True,
                include_git_status=True,
                include_project_instructions=True,
                today=options.today,
                git_command_runner=options.git_command_runner,
                user_instruction_paths=options.user_instruction_paths,
                project_filenames=options.project_filenames,
                project_rule_dirs=options.project_rule_dirs,
                local_filenames=options.local_filenames,
                additional_dirs=options.additional_dirs,
                discovery_mode=options.discovery_mode,
                allow_instruction_includes=options.allow_instruction_includes,
                allow_external_instruction_includes=(
                    options.allow_external_instruction_includes
                ),
            )
        raise ValueError(f"Unknown Raygent context profile: {context!r}")
    return tuple(context)


def _build_tool_profile_runtime(
    tools: RaygentToolSelection,
    *,
    pdf_document_service: PdfDocumentService | None,
    options: RaygentToolProfileOptions,
    task_output_store: TaskOutputStore,
) -> _ToolProfileRuntime:
    if not isinstance(tools, str) or tools == "none":
        return _ToolProfileRuntime(profile="none")

    from raygent_harness.tools.discovery_tools import create_discovery_tooling_runtime
    from raygent_harness.tools.file_tools import create_file_tooling_runtime

    if tools == "file":
        file_runtime = create_file_tooling_runtime(
            pdf_document_service=pdf_document_service,
        )
        return _ToolProfileRuntime(
            profile="file",
            pre_tool_use_hooks=tuple(file_runtime.pre_tool_use_hooks),
            post_tool_use_hooks=tuple(file_runtime.post_tool_use_hooks),
            file_runtime=file_runtime,
            task_output_store=task_output_store,
        )
    if tools == "project":
        file_runtime = create_file_tooling_runtime(
            pdf_document_service=pdf_document_service,
        )
        discovery_runtime = create_discovery_tooling_runtime(
            backend=options.search_backend,
        )
        return _ToolProfileRuntime(
            profile="project",
            pre_tool_use_hooks=tuple(file_runtime.pre_tool_use_hooks),
            post_tool_use_hooks=tuple(file_runtime.post_tool_use_hooks),
            file_runtime=file_runtime,
            discovery_runtime=discovery_runtime,
            task_output_store=task_output_store,
            enable_bash=options.enable_bash,
        )
    raise ValueError(f"Unknown Raygent tool profile: {tools!r}")


def _build_tool_catalog_provider(
    runtime: _ToolProfileRuntime,
    *,
    deps: QueryDeps,
    upstream: ToolCatalogProvider | None,
) -> ToolCatalogProvider | None:
    provider = upstream
    if runtime.file_runtime is not None:
        from raygent_harness.tools.file_tools import create_file_tools_catalog_provider

        provider = create_file_tools_catalog_provider(
            runtime=runtime.file_runtime,
            upstream=provider,
        )
    if runtime.profile != "project":
        return provider

    from raygent_harness.tools.discovery_tools import (
        create_discovery_tools_catalog_provider,
    )
    from raygent_harness.tools.task_stop_tool import create_task_stop_catalog_provider
    from raygent_harness.tools.tool_search_tool import create_tool_search_catalog_provider

    provider = create_discovery_tools_catalog_provider(
        runtime=runtime.discovery_runtime,
        upstream=provider,
    )
    provider = create_task_stop_catalog_provider(
        parent_deps=deps,
        upstream=provider,
    )
    if runtime.enable_bash:
        from raygent_harness.tools.bash_tool import create_bash_catalog_provider

        provider = create_bash_catalog_provider(
            parent_deps=deps,
            output_store=runtime.task_output_store,
            upstream=provider,
        )
    return create_tool_search_catalog_provider(upstream=provider)


def _resolve_observability(
    observability: KernelEventBus | Sequence[KernelEventSink] | None,
) -> KernelEventBus:
    if observability is None:
        return NoopKernelEventBus()
    if isinstance(observability, KernelEventBus):
        return observability
    return KernelEventBus(observability)


def _noop_notification_sink(_message: str) -> None:
    return None


__all__ = [
    "RaygentAgentOptions",
    "RaygentCallbackHandle",
    "RaygentContextOptions",
    "RaygentContextProfile",
    "RaygentContextProfileOptions",
    "RaygentContextSelection",
    "RaygentFactory",
    "RaygentFactoryConfig",
    "RaygentKernelEventCallback",
    "RaygentKernelEventCallbackSink",
    "RaygentMemoryOptions",
    "RaygentModelOptions",
    "RaygentObservabilityOptions",
    "RaygentPermissionOptions",
    "RaygentPersistenceOptions",
    "RaygentRunCallbacks",
    "RaygentRuntimeHandles",
    "RaygentSDKError",
    "RaygentSDKMessageCallback",
    "RaygentSDKProtocolError",
    "RaygentSDKResultCallback",
    "RaygentSession",
    "RaygentSessionBusyError",
    "RaygentSessionClosedError",
    "RaygentSessionFactory",
    "RaygentSessionOptions",
    "RaygentToolOptions",
    "RaygentToolProfile",
    "RaygentToolProfileOptions",
    "RaygentToolSelection",
    "create_raygent",
]
