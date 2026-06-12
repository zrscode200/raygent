from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from raygent_harness.context_providers.project_instructions import (
    ProjectInstructionsContextProvider,
)
from raygent_harness.core.config import QueryConfig, SamplingParams, TurnBudget
from raygent_harness.core.context_providers import (
    ContextFragment,
    context_provider_kind,
)
from raygent_harness.core.deps import (
    AgentTriggerPolicySpec,
    CoordinatorRuntimeProtocol,
    MemoryExtractor,
    MemoryRecallProvider,
    QueryDeps,
    SkillProvider,
    ToolPermissionEngine,
)
from raygent_harness.core.messages import assistant_message
from raygent_harness.core.observability import KernelEvent, KernelEventContext
from raygent_harness.core.permission_engine import PermissionRequest
from raygent_harness.core.permissions import (
    PermissionAllowDecision,
    PermissionDecision,
    ToolPermissionContext,
)
from raygent_harness.core.query_engine import (
    QueryEngine,
    SDKAssistantMessage,
    SDKMessage,
    SDKResult,
    SDKSystemInit,
)
from raygent_harness.core.task import AppStateStore, TaskStateBase
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.core.tool_hooks import (
    PostToolUseFailureHook,
    PostToolUseHook,
    PreToolUseHook,
)
from raygent_harness.sdk import (
    RaygentAgentOptions,
    RaygentCallbackHandle,
    RaygentContextOptions,
    RaygentContextProfileOptions,
    RaygentContextSelection,
    RaygentFactory,
    RaygentFactoryConfig,
    RaygentKernelEventCallbackSink,
    RaygentMemoryOptions,
    RaygentModelOptions,
    RaygentObservabilityOptions,
    RaygentPermissionOptions,
    RaygentPersistenceOptions,
    RaygentRunCallbacks,
    RaygentRuntimeHandles,
    RaygentSDKProtocolError,
    RaygentSession,
    RaygentSessionBusyError,
    RaygentSessionClosedError,
    RaygentSessionFactory,
    RaygentSessionOptions,
    RaygentToolOptions,
    RaygentToolProfileOptions,
    RaygentToolSelection,
    create_raygent,
)
from raygent_harness.services.agent_routes import AgentRouteRecordStore
from raygent_harness.services.file_media import PdfDocumentService
from raygent_harness.services.handoff import AgentHandoffClassifier
from raygent_harness.services.remote_agent import (
    RemoteAgentBackend,
    RemoteAgentPersistenceStore,
)
from raygent_harness.services.task_output import FileTaskOutputStore
from raygent_harness.services.transcript import JsonlTranscriptStore, TranscriptScope
from raygent_harness.services.worktree import WorktreeManager
from raygent_harness.skills.models import SkillDefinition
from raygent_harness.tools import (
    BASH_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
    NOTEBOOK_EDIT_TOOL_NAME,
    TASK_STOP_TOOL_NAME,
    TOOL_SEARCH_TOOL_NAME,
)
from raygent_harness.tools.bash_tool import BashInput
from tests.fakes import FakeModelProvider


def _provider(text: str = "ok") -> FakeModelProvider:
    return FakeModelProvider(responses=(assistant_message(text),))


class _EmptyInput(BaseModel):
    pass


async def _unused_call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    if False:
        yield ToolResult(content="unused")


async def _context_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> tuple[ContextFragment, ...]:
    return (
        ContextFragment(
            id="test-fragment",
            content="test context",
            channel="system",
            source="test",
        ),
    )


async def _post_tool_context_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
    _read_paths: Sequence[str],
    _already_attached_sources: Sequence[str],
) -> tuple[ContextFragment, ...]:
    return (
        ContextFragment(
            id="post-tool-fragment",
            content="post-tool context",
            channel="system",
            source="test",
        ),
    )


async def _system_prompt_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> str:
    return "system prompt provider"


async def _memory_prompt_provider(
    _config: QueryConfig,
    _ctx: ToolUseContext,
) -> str:
    return "memory prompt provider"


@dataclass(frozen=True)
class _StaticClock:
    value: float = 123.0

    def now(self) -> float:
        return self.value


async def _turn_tools(session: RaygentSession) -> tuple[str, ...]:
    provider = session.deps.tool_catalog_provider
    if provider is None:
        return tuple(tool.name for tool in session.config.tools)
    tools = await provider(session.config, session.ctx, ())
    if tools is None:
        tools = session.config.tools
    return tuple(tool.name for tool in tools)


def _manual_session_with_engine(
    engine: object,
    tmp_path: Path,
    *,
    session_id: str,
) -> RaygentSession:
    config = QueryConfig(model="demo-model", session_id=session_id)
    deps = QueryDeps(task_store=AppStateStore(), model_provider=_provider())
    ctx = ToolUseContext(
        session_id=session_id,
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path.resolve()),
        query_tracking=QueryTracking(chain_id=session_id, depth=0),
    )
    return RaygentSession(
        engine=cast(QueryEngine, engine),
        config=config,
        deps=deps,
        ctx=ctx,
    )


class _FinalizingEngine:
    def __init__(self, *, session_id: str = "finalizer-session") -> None:
        self.session_id = session_id
        self.finalized_prompts: list[str] = []

    async def submit_message(self, prompt: str) -> AsyncIterator[SDKMessage]:
        try:
            yield SDKSystemInit(session_id=self.session_id, model="demo-model")
            yield SDKResult(session_id=self.session_id, result=f"{prompt}: result")
        finally:
            self.finalized_prompts.append(prompt)

    async def drain_memory_extractions(
        self,
        timeout_s: float | None = 60.0,
    ) -> bool:
        _ = timeout_s
        return True


async def _turn_tool_by_name(session: RaygentSession, name: str) -> Tool:
    provider = session.deps.tool_catalog_provider
    assert provider is not None
    tools = await provider(session.config, session.ctx, ())
    assert tools is not None
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not found")


async def _call_tool_once(tool: Tool, input_: BaseModel, ctx: ToolUseContext) -> ToolResult:
    events = [event async for event in tool.call(input_, ctx)]
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolResult)
    return result


@dataclass
class _RecordingSink:
    events: list[KernelEvent] = field(default_factory=list[KernelEvent])

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)


@dataclass
class _AllowHandler:
    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        _ = request
        return PermissionAllowDecision()


@dataclass(frozen=True)
class _ProductSessionFactory:
    inner: RaygentFactory

    def create_session(
        self,
        config: RaygentFactoryConfig,
        /,
    ) -> RaygentSession:
        return self.inner.create_session(config)


def _create_with_product_factory(
    factory: RaygentSessionFactory,
    config: RaygentFactoryConfig,
) -> RaygentSession:
    return factory.create_session(config)


def test_raygent_factory_creates_same_session_shape_as_create_raygent(
    tmp_path: Path,
) -> None:
    direct = RaygentFactory().create_session(
        RaygentFactoryConfig(
            provider=_provider(),
            model="demo-model",
            cwd=tmp_path,
            session_id="direct-factory-session",
            system_prompt="Be direct.",
            experiments={"factory": True},
        )
    )
    wrapper = create_raygent(
        RaygentFactoryConfig(
            provider=_provider(),
            model="demo-model",
            cwd=tmp_path,
            session_id="wrapper-factory-session",
            system_prompt="Be direct.",
            experiments={"factory": True},
        )
    )

    assert type(direct) is type(wrapper)
    assert direct.config.model == wrapper.config.model == "demo-model"
    assert direct.config.system_prompt == wrapper.config.system_prompt == "Be direct."
    assert direct.config.experiments == wrapper.config.experiments == {"factory": True}
    assert direct.cwd == wrapper.cwd == str(tmp_path.resolve())
    assert direct.ctx.agent_id is wrapper.ctx.agent_id is None
    assert direct.ctx.query_tracking is not None
    assert wrapper.ctx.query_tracking is not None
    assert direct.ctx.query_tracking.depth == wrapper.ctx.query_tracking.depth == 0
    assert direct.deps.context_providers == wrapper.deps.context_providers == ()
    assert direct.config.tools == wrapper.config.tools == ()


def test_reusable_raygent_factory_allocates_fresh_session_state(tmp_path: Path) -> None:
    factory = RaygentFactory()
    config = RaygentFactoryConfig(provider=_provider(), model="demo-model", cwd=tmp_path)

    first = factory.create_session(config)
    second = factory.create_session(config)

    assert first.session_id.startswith("raygent-")
    assert second.session_id.startswith("raygent-")
    assert first.session_id != second.session_id
    assert first.engine is not second.engine
    assert first.ctx is not second.ctx
    assert first.abort_event is not second.abort_event
    assert first.handles is not second.handles
    assert first.task_store is not second.task_store
    assert isinstance(first.task_output_store, FileTaskOutputStore)
    assert isinstance(second.task_output_store, FileTaskOutputStore)
    assert first.task_output_store is not second.task_output_store
    assert first.transcript_scope != second.transcript_scope
    assert first.task_output_store.task_dir != second.task_output_store.task_dir


def test_product_defined_factory_satisfies_session_factory_protocol(
    tmp_path: Path,
) -> None:
    product_factory = _ProductSessionFactory(RaygentFactory())
    session = _create_with_product_factory(
        product_factory,
        RaygentFactoryConfig(
            provider=_provider(),
            model="demo-model",
            cwd=tmp_path,
            session_id="product-owned-session",
        ),
    )

    assert session.session_id == "product-owned-session"
    assert session.cwd == str(tmp_path.resolve())


def test_factory_config_options_map_to_kernel_objects(tmp_path: Path) -> None:
    provider = _provider()
    tool = build_tool(
        ToolSpec(
            name="OptionInspect",
            description="Tool supplied through RaygentToolOptions.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )
    permission_context = ToolPermissionContext(mode="dontAsk")
    permission_handler = _AllowHandler()
    permission_engine = cast(ToolPermissionEngine, object())
    pdf_service = cast(PdfDocumentService, object())
    pre_hook = cast(PreToolUseHook, object())
    post_hook = cast(PostToolUseHook, object())
    failure_hook = cast(PostToolUseFailureHook, object())
    memory_recall = cast(MemoryRecallProvider, object())
    memory_extractor = cast(MemoryExtractor, object())
    skill_provider = cast(SkillProvider, object())
    trigger_policy = cast(AgentTriggerPolicySpec, object())
    coordinator_runtime = cast(CoordinatorRuntimeProtocol, object())
    remote_backend = cast(RemoteAgentBackend, object())
    handoff_classifier = cast(AgentHandoffClassifier, object())
    worktree_manager = cast(WorktreeManager, object())
    remote_persistence = cast(RemoteAgentPersistenceStore, object())
    route_store = cast(AgentRouteRecordStore, object())
    task_store = AppStateStore()
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    sink = _RecordingSink()
    notifications: list[str] = []
    notify = notifications.append
    clock = _StaticClock()
    sampling = SamplingParams(max_tokens=128, temperature=0.2)
    budget = TurnBudget(max_turns=7, max_budget_usd=1.5)

    async def catalog_provider(
        config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool]:
        return config.tools

    session = RaygentFactory().create_session(
        RaygentFactoryConfig(
            model_options=RaygentModelOptions(
                provider=provider,
                model="option-model",
                fallback_model="option-fallback",
                sampling=sampling,
                budget=budget,
                system_prompt="Option system prompt.",
                experiments={"option": True},
            ),
            session_options=RaygentSessionOptions(
                cwd=tmp_path,
                session_id="option-session",
            ),
            tool_options=RaygentToolOptions(
                tools=(tool,),
                catalog_provider=catalog_provider,
                pre_tool_use_hooks=(pre_hook,),
                post_tool_use_hooks=(post_hook,),
                post_tool_use_failure_hooks=(failure_hook,),
                max_tool_use_concurrency=3,
                pdf_document_service=pdf_service,
            ),
            context_options=RaygentContextOptions(
                context=(_context_provider,),
                post_tool_context_providers=(_post_tool_context_provider,),
                system_prompt_provider=_system_prompt_provider,
            ),
            permission_options=RaygentPermissionOptions(
                context=permission_context,
                handler=permission_handler,
                engine=permission_engine,
                is_non_interactive_session=True,
            ),
            memory_options=RaygentMemoryOptions(
                prompt_provider=_memory_prompt_provider,
                recall_provider=memory_recall,
                extractor=memory_extractor,
            ),
            persistence_options=RaygentPersistenceOptions(
                task_store=task_store,
                transcript_store=transcript_store,
                task_output_dir="option-tasks",
                worktree_manager=worktree_manager,
                remote_agent_persistence_store=remote_persistence,
                agent_route_record_store=route_store,
            ),
            agent_options=RaygentAgentOptions(
                skill_provider=skill_provider,
                agent_trigger_policy=trigger_policy,
                coordinator_runtime=coordinator_runtime,
                remote_agent_backend=remote_backend,
                handoff_classifier=handoff_classifier,
                handoff_classifier_timeout_s=4.5,
            ),
            observability_options=RaygentObservabilityOptions(
                observability=(sink,),
                notify=notify,
                clock=clock,
            ),
        )
    )

    assert session.config.model == "option-model"
    assert session.config.fallback_model == "option-fallback"
    assert session.config.sampling is sampling
    assert session.config.budget is budget
    assert session.config.system_prompt == "Option system prompt."
    assert session.config.experiments == {"option": True}
    assert session.config.tools == (tool,)
    assert session.session_id == "option-session"
    assert session.cwd == str(tmp_path.resolve())
    assert session.deps.model_provider is provider
    assert session.deps.tool_catalog_provider is catalog_provider
    assert session.deps.pre_tool_use_hooks == [pre_hook]
    assert session.deps.post_tool_use_hooks == [post_hook]
    assert session.deps.post_tool_use_failure_hooks == [failure_hook]
    assert session.deps.max_tool_use_concurrency == 3
    assert session.deps.pdf_document_service is pdf_service
    assert session.deps.context_providers == (_context_provider,)
    assert session.deps.post_tool_context_providers == (_post_tool_context_provider,)
    assert session.deps.system_prompt_provider is _system_prompt_provider
    assert session.deps.permission_context is permission_context
    assert session.ctx.permission_context is permission_context
    assert session.deps.permission_handler is permission_handler
    assert session.deps.permission_engine is permission_engine
    assert session.deps.is_non_interactive_session is True
    assert session.deps.memory_prompt_provider is _memory_prompt_provider
    assert session.deps.memory_recall_provider is memory_recall
    assert session.deps.memory_extractor is memory_extractor
    assert session.deps.skill_provider is skill_provider
    assert session.deps.agent_trigger_policy is trigger_policy
    assert session.deps.coordinator_runtime is coordinator_runtime
    assert session.deps.remote_agent_backend is remote_backend
    assert session.deps.handoff_classifier is handoff_classifier
    assert session.deps.handoff_classifier_timeout_s == 4.5
    assert session.deps.task_store is task_store
    assert session.task_store is task_store
    assert session.transcript_store is transcript_store
    assert session.output_dir == (tmp_path / "option-tasks").resolve()
    assert session.deps.worktree_manager is worktree_manager
    assert session.deps.remote_agent_persistence_store is remote_persistence
    assert session.deps.agent_route_record_store is route_store
    assert session.deps.notify is notify
    assert session.ctx.add_notification is notify
    assert session.deps.clock is clock
    assert session.handles.observability is session.observability

    event = session.observability.emit(
        "sdk.options",
        context=KernelEventContext(session_id=session.session_id),
    )
    assert sink.events == [event]


def test_create_raygent_accepts_option_only_model_and_session(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        model_options=RaygentModelOptions(
            provider=_provider(),
            model="options-only-model",
        ),
        session_options=RaygentSessionOptions(
            cwd=tmp_path,
            session_id="options-only-session",
        ),
    )

    assert session.config.model == "options-only-model"
    assert session.session_id == "options-only-session"
    assert session.cwd == str(tmp_path.resolve())


def test_flat_factory_fields_take_precedence_over_option_groups(
    tmp_path: Path,
) -> None:
    flat_provider = _provider()
    option_provider = _provider("option")
    flat_task_store = AppStateStore()
    option_task_store = AppStateStore()
    flat_permission_context = ToolPermissionContext(mode="dontAsk")
    option_permission_context = ToolPermissionContext(mode="acceptEdits")
    flat_sink = _RecordingSink()
    option_sink = _RecordingSink()

    session = create_raygent(
        provider=flat_provider,
        model="flat-model",
        model_options=RaygentModelOptions(
            provider=option_provider,
            model="option-model",
            system_prompt="option prompt",
            experiments={"option": True},
        ),
        session_options=RaygentSessionOptions(
            cwd=tmp_path / "option",
            session_id="option-session",
        ),
        cwd=tmp_path,
        session_id="flat-session",
        system_prompt="flat prompt",
        experiments={"flat": True},
        permission_context=flat_permission_context,
        permission_options=RaygentPermissionOptions(context=option_permission_context),
        observability=(flat_sink,),
        observability_options=RaygentObservabilityOptions(observability=(option_sink,)),
        task_store=flat_task_store,
        persistence_options=RaygentPersistenceOptions(task_store=option_task_store),
    )

    assert session.deps.model_provider is flat_provider
    assert session.config.model == "flat-model"
    assert session.config.system_prompt == "flat prompt"
    assert session.config.experiments == {"flat": True}
    assert session.session_id == "flat-session"
    assert session.cwd == str(tmp_path.resolve())
    assert session.deps.permission_context is flat_permission_context
    assert session.deps.task_store is flat_task_store

    event = session.observability.emit(
        "sdk.flat_precedence",
        context=KernelEventContext(session_id=session.session_id),
    )
    assert flat_sink.events == [event]
    assert option_sink.events == []


def test_create_raygent_wires_identity_cwd_permissions_and_defaults(tmp_path: Path) -> None:
    task_store = AppStateStore()
    notifications: list[str] = []
    notify = notifications.append

    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        task_store=task_store,
        notify=notify,
        is_non_interactive_session=True,
    )

    assert session.session_id.startswith("raygent-")
    assert session.config.session_id == session.session_id
    assert session.ctx.session_id == session.session_id
    assert session.ctx.query_tracking == QueryTracking(chain_id=session.session_id, depth=0)
    assert session.cwd == str(tmp_path.resolve())
    assert session.ctx.cwd == str(tmp_path.resolve())
    assert session.deps.output_dir == str(tmp_path.resolve() / ".raygent" / "tasks")
    assert session.deps.task_store is task_store
    assert task_store.observability is session.observability
    assert session.deps.notify is notify
    assert session.ctx.add_notification is notify
    assert session.deps.is_non_interactive_session is True
    assert session.config.tools == ()
    assert session.deps.context_providers == ()


def test_runtime_handles_expose_session_owned_services(tmp_path: Path) -> None:
    task_store = AppStateStore()
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    sink = _RecordingSink()

    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="handle-session",
        task_store=task_store,
        task_output_dir=Path("outputs"),
        transcript_store=transcript_store,
        observability=(sink,),
    )

    handles = session.handles

    assert isinstance(handles, RaygentRuntimeHandles)
    assert handles.session_id == "handle-session"
    assert handles.cwd == str(tmp_path.resolve())
    assert handles.task_store is task_store
    assert handles.output_dir == (tmp_path / "outputs").resolve()
    assert isinstance(handles.task_output_store, FileTaskOutputStore)
    assert handles.task_output_store.task_dir == (
        tmp_path / "outputs" / "handle-session" / "tasks"
    ).resolve()
    assert handles.transcript_store is transcript_store
    assert handles.transcript_scope == TranscriptScope(session_id="handle-session")
    transcript_scope = handles.transcript_scope
    assert transcript_scope is not None
    assert handles.transcript_path == transcript_store.path_for(transcript_scope)
    assert handles.observability is session.observability
    assert handles.abort_event is session.abort_event
    assert session.output_dir == handles.output_dir
    assert session.task_output_store is handles.task_output_store
    assert session.transcript_store is transcript_store
    assert session.transcript_scope == handles.transcript_scope
    assert session.transcript_path == handles.transcript_path


@pytest.mark.asyncio
async def test_transcript_store_persists_turn_and_close_rejects_future_runs(
    tmp_path: Path,
) -> None:
    transcript_store = JsonlTranscriptStore(tmp_path / "transcripts")
    session = create_raygent(
        provider=_provider("persisted"),
        model="demo-model",
        cwd=tmp_path,
        session_id="persisted-session",
        transcript_store=transcript_store,
    )

    result = await session.run_until_result("Persist this turn.")
    drained = await session.close()

    transcript_scope = session.transcript_scope
    assert transcript_scope is not None
    entries = await transcript_store.read_entries(transcript_scope)
    messages = [entry.message for entry in entries if entry.type == "message"]

    assert result.result == "persisted"
    assert drained is True
    assert session.is_closed is True
    assert session.abort_event.is_set()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Persist this turn."
    assert messages[1]["content"] == "persisted"

    with pytest.raises(RaygentSessionClosedError):
        await session.run_until_result("Should not run.")


@pytest.mark.asyncio
async def test_close_is_idempotent_and_does_not_kill_unrelated_tasks(
    tmp_path: Path,
) -> None:
    task_store = AppStateStore()
    task_store.register_task(
        TaskStateBase(
            id="unrelated-task",
            type="local_bash",
            description="unrelated",
            status="running",
            start_time=1.0,
        )
    )
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        task_store=task_store,
    )

    first_close = await session.close()
    second_close = await session.close()

    assert first_close is True
    assert second_close is True
    assert session.is_closed is True
    assert session.abort_event.is_set()
    assert task_store.tasks["unrelated-task"].status == "running"


@pytest.mark.asyncio
async def test_close_during_active_turn_signals_abort_and_requires_generator_close(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider("after-close"),
        model="demo-model",
        cwd=tmp_path,
        session_id="active-close-session",
    )

    first_run = cast(AsyncGenerator[SDKMessage, None], session.run("Start."))
    first_event = await anext(first_run)
    assert isinstance(first_event, SDKSystemInit)

    with pytest.raises(RaygentSessionBusyError, match="active turn"):
        await session.close()

    assert session.abort_event.is_set()
    assert session.is_closed is False

    await first_run.aclose()
    assert await session.close() is True
    assert session.is_closed is True


@pytest.mark.asyncio
async def test_close_retries_unfinished_memory_extraction_drain(tmp_path: Path) -> None:
    class DrainEngine:
        def __init__(self) -> None:
            self.calls = 0

        async def drain_memory_extractions(
            self,
            timeout_s: float | None = 60.0,
        ) -> bool:
            _ = timeout_s
            self.calls += 1
            return self.calls >= 2

        async def submit_message(self, _prompt: str) -> AsyncIterator[SDKMessage]:
            if False:
                yield SDKSystemInit(session_id="drain-session", model="demo-model")

    engine = DrainEngine()
    config = QueryConfig(model="demo-model", session_id="drain-session")
    deps = QueryDeps(task_store=AppStateStore(), model_provider=_provider())
    ctx = ToolUseContext(
        session_id="drain-session",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path.resolve()),
        query_tracking=QueryTracking(chain_id="drain-session", depth=0),
    )
    session = RaygentSession(
        engine=cast(QueryEngine, engine),
        config=config,
        deps=deps,
        ctx=ctx,
    )

    assert await session.close(drain_memory_extractions_timeout_s=0.0) is False
    assert await session.close(drain_memory_extractions_timeout_s=0.0) is True
    assert engine.calls == 2


def test_manual_session_handles_use_engine_transcript_scope(tmp_path: Path) -> None:
    config = QueryConfig(model="demo-model", session_id="configured-session")
    deps = QueryDeps(task_store=AppStateStore(), model_provider=_provider())
    ctx = ToolUseContext(
        session_id="configured-session",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path.resolve()),
        query_tracking=QueryTracking(chain_id="configured-session", depth=0),
    )
    transcript_scope = TranscriptScope(
        session_id="stored-session",
        runtime_session_id="runtime-1",
    )
    engine = QueryEngine(config, deps, ctx, transcript_scope=transcript_scope)
    session = RaygentSession(
        engine=engine,
        config=config,
        deps=deps,
        ctx=ctx,
    )

    assert session.transcript_scope == transcript_scope


def test_manual_session_preserves_engine_transcript_scope_none(tmp_path: Path) -> None:
    config = QueryConfig(
        model="demo-model",
        session_id="parent-session",
        agent_id="agent-1",
    )
    deps = QueryDeps(task_store=AppStateStore(), model_provider=_provider())
    ctx = ToolUseContext(
        session_id="parent-session",
        agent_id="agent-1",
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path.resolve()),
        query_tracking=QueryTracking(chain_id="parent-session", depth=1),
    )
    engine = QueryEngine(config, deps, ctx)
    assert engine.transcript_scope is None

    session = RaygentSession(
        engine=engine,
        config=config,
        deps=deps,
        ctx=ctx,
    )

    assert session.transcript_scope is None
    assert session.transcript_path is None


@pytest.mark.asyncio
async def test_session_async_context_manager_closes_session(tmp_path: Path) -> None:
    async with create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="context-manager-session",
    ) as session:
        assert session.is_closed is False

    assert session.is_closed is True
    assert session.abort_event.is_set()


@pytest.mark.asyncio
async def test_run_callbacks_receive_messages_and_terminal_result(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider("callback-result"),
        model="demo-model",
        cwd=tmp_path,
        session_id="callback-session",
    )
    callback_message_types: list[str] = []
    callback_results: list[str] = []

    async def on_message(event: SDKMessage) -> None:
        await asyncio.sleep(0)
        callback_message_types.append(type(event).__name__)

    def on_result(result: SDKResult) -> None:
        callback_results.append(result.result)

    events = [
        event
        async for event in session.run(
            "Use callbacks.",
            callbacks=RaygentRunCallbacks(
                on_message=on_message,
                on_result=on_result,
            ),
        )
    ]

    assert callback_message_types == [type(event).__name__ for event in events]
    assert callback_message_types == [
        "SDKSystemInit",
        "SDKAssistantMessage",
        "SDKResult",
    ]
    assert callback_results == ["callback-result"]


@pytest.mark.asyncio
async def test_run_callback_exception_closes_inner_engine_stream(
    tmp_path: Path,
) -> None:
    engine = _FinalizingEngine(session_id="callback-exception-session")
    session = _manual_session_with_engine(
        engine,
        tmp_path,
        session_id="callback-exception-session",
    )

    def broken_callback(_event: SDKMessage) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await session.run_until_result(
            "First",
            callbacks=RaygentRunCallbacks(on_message=broken_callback),
        )

    assert engine.finalized_prompts == ["First"]

    result = await session.run_until_result("Second")

    assert result.result == "Second: result"
    assert engine.finalized_prompts == ["First", "Second"]


@pytest.mark.asyncio
async def test_run_aclose_closes_inner_engine_stream(tmp_path: Path) -> None:
    engine = _FinalizingEngine(session_id="aclose-finalizer-session")
    session = _manual_session_with_engine(
        engine,
        tmp_path,
        session_id="aclose-finalizer-session",
    )
    run_stream = cast(
        AsyncGenerator[SDKMessage, None],
        session.run("Early close."),
    )

    first_event = await anext(run_stream)
    assert isinstance(first_event, SDKSystemInit)
    assert engine.finalized_prompts == []

    await run_stream.aclose()

    assert engine.finalized_prompts == ["Early close."]


@pytest.mark.asyncio
async def test_run_kernel_event_callback_is_scoped_to_run(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="kernel-callback-session",
    )
    kernel_events: list[KernelEvent] = []

    await session.run_until_result(
        "Emit kernel events.",
        callbacks=RaygentRunCallbacks(on_kernel_event=kernel_events.append),
    )
    seen_during_run = len(kernel_events)

    session.observability.emit(
        "sdk.after_run",
        context=KernelEventContext(session_id=session.session_id),
    )

    assert any(event.type == "query.turn.started" for event in kernel_events)
    assert seen_during_run > 0
    assert len(kernel_events) == seen_during_run


@pytest.mark.asyncio
async def test_run_kernel_event_callback_detaches_on_early_aclose(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="kernel-callback-aclose-session",
    )
    kernel_events: list[KernelEvent] = []
    run_stream = cast(
        AsyncGenerator[SDKMessage, None],
        session.run(
            "Start then close.",
            callbacks=RaygentRunCallbacks(on_kernel_event=kernel_events.append),
        ),
    )

    first_event = await anext(run_stream)
    assert isinstance(first_event, SDKSystemInit)
    assert any(event.type == "query.turn.started" for event in kernel_events)
    seen_during_run = len(kernel_events)

    await run_stream.aclose()
    session.observability.emit(
        "sdk.after_aclose",
        context=KernelEventContext(session_id=session.session_id),
    )

    assert len(kernel_events) == seen_during_run


def test_add_kernel_event_callback_returns_detachable_handle(tmp_path: Path) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="callback-handle-session",
    )
    kernel_events: list[KernelEvent] = []

    handle = session.add_kernel_event_callback(kernel_events.append)
    emitted = session.observability.emit(
        "sdk.manual",
        context=KernelEventContext(session_id=session.session_id),
    )
    handle.close()
    session.observability.emit(
        "sdk.after_close",
        context=KernelEventContext(session_id=session.session_id),
    )

    assert isinstance(handle, RaygentCallbackHandle)
    assert handle.is_closed is True
    assert kernel_events == [emitted]


def test_kernel_event_callback_sink_uses_bus_fail_soft_errors(tmp_path: Path) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="callback-error-session",
    )

    def broken_callback(_event: KernelEvent) -> None:
        raise RuntimeError("callback exploded")

    sink = RaygentKernelEventCallbackSink(broken_callback)
    session.observability.add_sink(sink)

    session.observability.emit(
        "sdk.callback_error",
        context=KernelEventContext(session_id=session.session_id),
    )

    assert session.observability.sink_errors[-1].sink == "RaygentKernelEventCallbackSink"
    assert session.observability.sink_errors[-1].error_type == "RuntimeError"


def test_create_raygent_accepts_explicit_sequences_and_adapter_dependencies(
    tmp_path: Path,
) -> None:
    tool = build_tool(
        ToolSpec(
            name="Inspect",
            description="Inspect without side effects.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )
    permission_context = ToolPermissionContext(mode="dontAsk")
    permission_handler = _AllowHandler()
    sink = _RecordingSink()

    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="explicit-session",
        tools=(tool,),
        context=(_context_provider,),
        permission_context=permission_context,
        permission_handler=permission_handler,
        observability=(sink,),
    )

    assert session.config.tools == (tool,)
    assert session.ctx.tools == (tool,)
    assert session.deps.context_providers == (_context_provider,)
    assert session.deps.permission_context is permission_context
    assert session.ctx.permission_context is permission_context
    assert session.deps.permission_handler is permission_handler

    event = session.observability.emit(
        "sdk.test",
        context=KernelEventContext(session_id=session.session_id),
    )
    assert sink.events == [event]


@pytest.mark.asyncio
async def test_file_profile_installs_file_tools_with_pdf_service(tmp_path: Path) -> None:
    pdf_service = cast(PdfDocumentService, object())
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tools="file",
        pdf_document_service=pdf_service,
    )

    names = await _turn_tools(session)

    assert names == (
        FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        FILE_EDIT_TOOL_NAME,
        NOTEBOOK_EDIT_TOOL_NAME,
    )
    assert session.config.tools == ()
    assert session.deps.tool_catalog_provider is not None
    assert session.deps.pdf_document_service is pdf_service
    assert session.deps.pre_tool_use_hooks == []
    assert session.deps.post_tool_use_hooks == []


@pytest.mark.asyncio
async def test_project_profile_installs_conservative_project_tools(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tools="project",
    )

    names = await _turn_tools(session)

    assert names == (
        FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        FILE_EDIT_TOOL_NAME,
        NOTEBOOK_EDIT_TOOL_NAME,
        GLOB_TOOL_NAME,
        GREP_TOOL_NAME,
        TASK_STOP_TOOL_NAME,
        TOOL_SEARCH_TOOL_NAME,
    )
    assert BASH_TOOL_NAME not in names
    assert "Agent" not in names
    assert "Skill" not in names
    assert "TeamCreate" not in names
    assert "SendMessage" not in names


@pytest.mark.asyncio
async def test_project_profile_bash_is_explicit_opt_in(tmp_path: Path) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tools="project",
        tool_profile_options=RaygentToolProfileOptions(enable_bash=True),
    )

    names = await _turn_tools(session)

    assert BASH_TOOL_NAME in names
    assert names[-1] == TOOL_SEARCH_TOOL_NAME


@pytest.mark.asyncio
async def test_project_profile_bash_uses_session_output_store(
    tmp_path: Path,
) -> None:
    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="bash-output-session",
        tools="project",
        tool_profile_options=RaygentToolProfileOptions(enable_bash=True),
        task_output_dir="sdk-task-output",
    )
    bash_tool = await _turn_tool_by_name(session, BASH_TOOL_NAME)

    result = await _call_tool_once(
        bash_tool,
        BashInput(command="echo sdk-output", output_read_bytes=100),
        session.ctx,
    )

    task = next(iter(session.task_store.tasks.values()))
    assert result.is_error is False
    assert task.output_file is not None
    output_store = session.task_output_store
    assert isinstance(output_store, FileTaskOutputStore)
    assert Path(task.output_file).parent == output_store.task_dir
    assert output_store.task_dir == (
        tmp_path / "sdk-task-output" / "bash-output-session" / "tasks"
    ).resolve()


@pytest.mark.asyncio
async def test_profile_catalog_provider_composes_upstream_without_collisions(
    tmp_path: Path,
) -> None:
    fake_read = build_tool(
        ToolSpec(
            name=FILE_READ_TOOL_NAME,
            description="Caller-supplied Read replacement.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )
    custom_tool = build_tool(
        ToolSpec(
            name="Inspect",
            description="Caller-supplied non-conflicting tool.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )

    async def upstream(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool]:
        return (fake_read, custom_tool)

    session = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tools="file",
        tool_catalog_provider=upstream,
    )

    provider = session.deps.tool_catalog_provider
    assert provider is not None
    tools = await provider(session.config, session.ctx, ())
    assert tools is not None
    names = tuple(tool.name for tool in tools)

    assert names == (
        "Inspect",
        FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
        FILE_EDIT_TOOL_NAME,
        NOTEBOOK_EDIT_TOOL_NAME,
    )
    assert tools[1] is not fake_read


@pytest.mark.asyncio
async def test_catalog_provider_works_with_none_and_explicit_tools(
    tmp_path: Path,
) -> None:
    explicit_tool = build_tool(
        ToolSpec(
            name="Explicit",
            description="Static explicit tool.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )
    dynamic_tool = build_tool(
        ToolSpec(
            name="Dynamic",
            description="Dynamic catalog tool.",
            input_model=_EmptyInput,
            call=_unused_call,
            is_read_only=True,
            is_destructive=False,
        )
    )

    async def upstream(
        config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool]:
        return (*config.tools, dynamic_tool)

    none_profile = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tool_catalog_provider=upstream,
    )
    explicit = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        tools=(explicit_tool,),
        tool_catalog_provider=upstream,
    )

    assert await _turn_tools(none_profile) == ("Dynamic",)
    assert await _turn_tools(explicit) == ("Explicit", "Dynamic")


def test_unknown_profiles_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Raygent tool profile"):
        create_raygent(
            provider=_provider(),
            model="demo-model",
            cwd=tmp_path,
            tools=cast(RaygentToolSelection, "projects"),
        )

    with pytest.raises(ValueError, match="Unknown Raygent context profile"):
        create_raygent(
            provider=_provider(),
            model="demo-model",
            cwd=tmp_path,
            context=cast(RaygentContextSelection, "projects"),
        )


def test_context_profiles_install_expected_provider_sets(tmp_path: Path) -> None:
    environment = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        context="environment",
    )
    project = create_raygent(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        context="project",
        context_profile_options=RaygentContextProfileOptions(
            project_filenames=("RAYGENT.md",),
            local_filenames=("RAYGENT.local.md",),
        ),
    )

    assert tuple(
        context_provider_kind(provider) for provider in environment.deps.context_providers
    ) == ("environment",)
    assert tuple(
        context_provider_kind(provider) for provider in project.deps.context_providers
    ) == ("environment", "git", "project_instructions")

    project_provider = cast(
        ProjectInstructionsContextProvider,
        project.deps.context_providers[2],
    )
    assert project_provider.config.project_filenames == ("RAYGENT.md",)
    assert project_provider.config.local_filenames == ("RAYGENT.local.md",)
    assert project_provider.config.cwd == tmp_path.resolve()


@pytest.mark.asyncio
async def test_run_yields_existing_sdk_stream_and_run_until_result_returns_terminal(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(assistant_message("hello"), assistant_message("bye")))
    session = create_raygent(
        provider=provider,
        model="demo-model",
        cwd=tmp_path,
        session_id="session-1",
        system_prompt="You are concise.",
    )

    events = [event async for event in session.run("Say hello.")]

    assert isinstance(events[0], SDKSystemInit)
    assert isinstance(events[1], SDKAssistantMessage)
    assert isinstance(events[-1], SDKResult)
    assert events[0].session_id == "session-1"
    assert events[0].model == "demo-model"
    assert events[1].message["content"] == "hello"
    assert events[-1].result == "hello"

    result = await session.run_until_result("Say bye.")

    assert result.subtype == "success"
    assert result.result == "bye"
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_session_rejects_concurrent_turns_and_recovers_after_close(
    tmp_path: Path,
) -> None:
    provider = FakeModelProvider(responses=(assistant_message("completed"),))
    session = create_raygent(
        provider=provider,
        model="demo-model",
        cwd=tmp_path,
        session_id="busy-session",
    )

    first_run = cast(AsyncGenerator[SDKMessage, None], session.run("First."))
    first_event = await anext(first_run)
    assert isinstance(first_event, SDKSystemInit)

    with pytest.raises(RaygentSessionBusyError):
        await anext(session.run("Second."))

    await first_run.aclose()

    result = await session.run_until_result("Second.")
    assert result.result == "completed"
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_run_until_result_returns_error_result_as_data(tmp_path: Path) -> None:
    session = create_raygent(
        provider=FakeModelProvider(responses=(RuntimeError("boom"),)),
        model="demo-model",
        cwd=tmp_path,
        session_id="error-session",
    )

    result = await session.run_until_result("Explode.")

    assert result.subtype == "error_during_execution"
    assert result.is_error is True
    assert result.errors == ("boom",)


@pytest.mark.asyncio
async def test_run_until_result_raises_only_when_terminal_result_missing(
    tmp_path: Path,
) -> None:
    class NoResultEngine:
        async def submit_message(self, _prompt: str) -> AsyncIterator[SDKMessage]:
            yield SDKSystemInit(session_id="broken", model="demo-model")

    config = QueryConfig(model="demo-model", session_id="broken")
    deps = QueryDeps(task_store=AppStateStore(), model_provider=_provider())
    ctx = ToolUseContext(
        session_id="broken",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=str(tmp_path.resolve()),
        query_tracking=QueryTracking(chain_id="broken", depth=0),
    )
    session = RaygentSession(
        engine=cast(QueryEngine, NoResultEngine()),
        config=config,
        deps=deps,
        ctx=ctx,
    )

    with pytest.raises(RaygentSDKProtocolError):
        await session.run_until_result("No terminal.")


def test_config_object_path_and_output_dir_resolution(tmp_path: Path) -> None:
    config = RaygentFactoryConfig(
        provider=_provider(),
        model="demo-model",
        cwd=tmp_path,
        session_id="configured-session",
        task_output_dir=Path("tasks"),
    )

    session = create_raygent(config)

    assert session.session_id == "configured-session"
    assert session.deps.output_dir == str((tmp_path / "tasks").resolve())


def test_factory_config_preserves_positional_flat_field_order(tmp_path: Path) -> None:
    config = RaygentFactoryConfig(
        _provider(),
        "demo-model",
        tmp_path,
        "positional-session",
    )

    session = create_raygent(config)

    assert session.config.model == "demo-model"
    assert session.session_id == "positional-session"
    assert session.cwd == str(tmp_path.resolve())


def test_create_raygent_rejects_ambiguous_config_and_missing_required_args(
    tmp_path: Path,
) -> None:
    config = RaygentFactoryConfig(provider=_provider(), model="demo-model", cwd=tmp_path)

    with pytest.raises(ValueError, match="either RaygentFactoryConfig"):
        create_raygent(config, provider=_provider(), model="other-model")

    with pytest.raises(ValueError, match="session_id"):
        create_raygent(config, session_id="ignored")

    with pytest.raises(ValueError, match="requires provider and model"):
        create_raygent()
