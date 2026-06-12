from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from raygent_harness.agents.models import AgentContextPolicy, AgentDefinition
from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.context_providers import ContextFragment, ContextKind
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.messages import model_response_from_message_param
from raygent_harness.core.model_adapter import ToolUseBlock
from raygent_harness.core.model_types import ModelInfo, ModelRequest
from raygent_harness.core.permissions import (
    AddPermissionRules,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    RulePermissionDecisionReason,
    ToolPermissionContext,
    empty_tool_permission_context,
)
from raygent_harness.core.state import State
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    ValidationError,
    build_tool,
)
from raygent_harness.core.tool_execution import ToolExecutionResult, run_tool_use
from raygent_harness.core.tool_hooks import PreToolUseContext, PreToolUseResult
from raygent_harness.core.tool_orchestration import (
    TOOL_CANCEL_MESSAGE,
    ToolOrchestrationOutcome,
    run_tools,
)
from raygent_harness.services.transcript import JsonlTranscriptStore, get_agent_transcript
from raygent_harness.skills.models import SkillDefinition
from raygent_harness.tools.skill_tool import (
    SKILL_TOOL_NAME,
    SkillToolInput,
    build_skill_tool,
    create_skill_catalog_provider,
    is_model_invocable_skill,
    normalize_skill_name,
    skill_has_only_safe_properties,
)
from tests.fakes import FakeModelProvider


class EmptyInput(BaseModel):
    pass


class RecordingContextProvider:
    context_kind: ContextKind

    def __init__(self, kind: ContextKind, marker: str) -> None:
        self.context_kind = kind
        self.marker = marker

    async def __call__(
        self,
        _config: QueryConfig,
        _ctx: ToolUseContext,
    ) -> tuple[ContextFragment, ...]:
        return (
            ContextFragment(
                id=self.marker,
                content=self.marker,
                channel="user_context",
                kind=self.context_kind,
            ),
        )


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _base_tool(name: str = "Base") -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} description",
            input_model=EmptyInput,
            call=_call,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _skill(
    name: str = "review",
    *,
    markdown_content: str = "Review this: $ARGUMENTS",
    aliases: tuple[str, ...] = (),
    allowed_tools: tuple[str, ...] = (),
    hooks: dict[str, Any] | None = None,
    skill_root: Path | None = None,
    disable_model_invocation: bool = False,
    user_invocable: bool = True,
    context: str | None = None,
    agent: str | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=f"{name} skill",
        markdown_content=markdown_content,
        source="projectSettings",
        loaded_from="skills",
        content_length=len(markdown_content),
        aliases=aliases,
        allowed_tools=allowed_tools,
        hooks=hooks,
        skill_root=skill_root,
        disable_model_invocation=disable_model_invocation,
        user_invocable=user_invocable,
        context=cast(Any, context),
        agent=agent,
        model=model,
        effort=effort,
    )


def _ctx(
    *,
    permission_context: ToolPermissionContext | None = None,
    tools: Sequence[Tool] = (),
) -> ToolUseContext:
    return ToolUseContext(
        session_id="session-123",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        tools=tuple(tools),
        permission_context=permission_context or empty_tool_permission_context(),
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


@pytest.mark.asyncio
async def test_skill_tool_validates_slash_name_and_renders_prompt(tmp_path: Path) -> None:
    skill = _skill(
        markdown_content=(
            "Base=${CLAUDE_SKILL_DIR}\nSession=${CLAUDE_SESSION_ID}\nArgs=$ARGUMENTS"
        ),
        skill_root=tmp_path,
    )
    tool = build_skill_tool((skill,))
    parsed = SkillToolInput(skill="/review", args="diff.patch")

    validation = await tool.validate_input(parsed, _ctx(tools=(tool,)))
    events = [event async for event in tool.call(parsed, _ctx(tools=(tool,)))]

    assert validation.result == "ok"
    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    assert events[0].content == "Launching skill: review"
    assert len(events[0].additional_messages) == 1
    prompt_message = events[0].additional_messages[0]
    assert prompt_message["role"] == "user"
    prompt = prompt_message["content"]
    assert isinstance(prompt, str)
    assert f"Base directory for this skill: {tmp_path}" in prompt
    assert f"Base={tmp_path.as_posix()}" in prompt
    assert "Session=session-123" in prompt
    assert "Args=diff.patch" in prompt


@pytest.mark.asyncio
async def test_unknown_skill_surfaces_tool_result_error() -> None:
    tool = build_skill_tool((_skill(),))
    events = [
        event
        async for event in run_tool_use(
            tool_use=ToolUseBlock(
                id="toolu_skill",
                name=SKILL_TOOL_NAME,
                input={"skill": "missing"},
                index=0,
            ),
            assistant_message={
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "missing"},
                    }
                ],
            },
            tools=(tool,),
            deps=QueryDeps(task_store=AppStateStore()),
            ctx=_ctx(tools=(tool,)),
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], ToolExecutionResult)
    content = events[0].message["content"]
    assert isinstance(content, list)
    block = cast(Mapping[str, Any], content[0])
    assert block.get("is_error") is True
    assert "Unknown skill: missing" in str(block.get("content", ""))


@pytest.mark.asyncio
async def test_skill_tool_rejects_model_disabled_but_allows_model_only_skills() -> None:
    tool = build_skill_tool(
        (
            _skill("disabled", disable_model_invocation=True),
            _skill("model-only", user_invocable=False),
        )
    )

    disabled = await tool.validate_input(SkillToolInput(skill="disabled"), _ctx())
    model_only = await tool.validate_input(SkillToolInput(skill="model-only"), _ctx())

    assert isinstance(disabled, ValidationError)
    assert "disable-model-invocation" in disabled.message
    assert model_only.result == "ok"


@pytest.mark.asyncio
async def test_skill_permission_rules_deny_and_allow_by_skill_name() -> None:
    tool = build_skill_tool((_skill("review"),))
    parsed = SkillToolInput(skill="/review")

    deny = await tool.check_permissions(
        parsed,
        _ctx(),
        ToolPermissionContext(always_deny_rules={"localSettings": ("Skill(review)",)}),
    )
    allow = await tool.check_permissions(
        parsed,
        _ctx(),
        ToolPermissionContext(always_allow_rules={"localSettings": ("Skill(/review)",)}),
    )

    assert isinstance(deny, PermissionDenyDecision)
    assert isinstance(deny.decision_reason, RulePermissionDecisionReason)
    assert deny.decision_reason.rule.rule_value.rule_content == "review"
    assert isinstance(allow, PermissionAllowDecision)
    assert allow.updated_input == {"skill": "review", "args": None}


@pytest.mark.asyncio
async def test_skill_permission_prefix_rule_and_unsafe_metadata_ask() -> None:
    safe_tool = build_skill_tool((_skill("review-pr"),))
    unsafe_tool = build_skill_tool((_skill("deploy", allowed_tools=("Bash(npm test)",)),))

    prefix_allow = await safe_tool.check_permissions(
        SkillToolInput(skill="review-pr"),
        _ctx(),
        ToolPermissionContext(always_allow_rules={"localSettings": ("Skill(review:*)",)}),
    )
    unsafe = await unsafe_tool.check_permissions(
        SkillToolInput(skill="deploy"),
        _ctx(),
        empty_tool_permission_context(),
    )

    assert isinstance(prefix_allow, PermissionAllowDecision)
    assert isinstance(unsafe, PermissionAskDecision)
    assert unsafe.message == "Execute skill: deploy"
    suggestions = [rule for rule in unsafe.suggestions if isinstance(rule, AddPermissionRules)]
    assert [rule.rules[0].rule_content for rule in suggestions] == [
        "deploy",
        "deploy:*",
    ]


@pytest.mark.asyncio
async def test_safe_metadata_auto_allows_but_hooks_and_shell_are_unsafe() -> None:
    safe = _skill("safe")
    with_hook = _skill("with-hook", hooks={"PreToolUse": []})
    with_shell = _skill("with-shell")
    object.__setattr__(with_shell, "shell", "bash")

    tool = build_skill_tool((safe, with_hook, with_shell))
    safe_decision = await tool.check_permissions(
        SkillToolInput(skill="safe"),
        _ctx(),
        empty_tool_permission_context(),
    )
    hook_decision = await tool.check_permissions(
        SkillToolInput(skill="with-hook"),
        _ctx(),
        empty_tool_permission_context(),
    )
    shell_decision = await tool.check_permissions(
        SkillToolInput(skill="with-shell"),
        _ctx(),
        empty_tool_permission_context(),
    )

    assert skill_has_only_safe_properties(safe)
    assert isinstance(safe_decision, PermissionAllowDecision)
    assert isinstance(hook_decision, PermissionAskDecision)
    assert isinstance(shell_decision, PermissionAskDecision)


@pytest.mark.asyncio
async def test_skill_catalog_provider_appends_skill_and_composes_upstream() -> None:
    upstream_tool = _base_tool("Upstream")
    base_tool = _base_tool("Base")

    async def upstream(
        config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        return (*config.tools, upstream_tool)

    provider = create_skill_catalog_provider(upstream=upstream)
    tools = await provider(
        QueryConfig(model="claude-opus-4-7", tools=(base_tool,)),
        _ctx(),
        (_skill("review"),),
    )

    assert tools is not None
    assert tuple(tool.name for tool in tools) == ("Base", "Upstream", SKILL_TOOL_NAME)
    prompt = await tools[-1].prompt(None)
    assert "- review: review skill" in prompt


@pytest.mark.asyncio
async def test_skill_catalog_provider_lists_model_only_skills_but_hides_disabled() -> None:
    provider = create_skill_catalog_provider()
    model_only = _skill("model-only", user_invocable=False)
    disabled = _skill("disabled", disable_model_invocation=True)
    visible = _skill("visible")

    no_visible_tools = await provider(
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
        (disabled,),
    )
    tools = await provider(
        QueryConfig(model="claude-opus-4-7"),
        _ctx(),
        (model_only, disabled, visible),
    )

    assert no_visible_tools == ()
    assert tools is not None
    assert tuple(tool.name for tool in tools) == (SKILL_TOOL_NAME,)
    prompt = await tools[0].prompt(None)
    assert "- model-only: model-only skill" in prompt
    assert "- visible: visible skill" in prompt
    assert "disabled" not in prompt
    assert is_model_invocable_skill(visible)
    assert is_model_invocable_skill(model_only)
    assert not is_model_invocable_skill(disabled)


@pytest.mark.asyncio
async def test_skill_context_modifier_allows_followup_tool_and_sets_model() -> None:
    skill = _skill(
        "deploy",
        allowed_tools=("Example", "Example"),
        model="claude-sonnet-skill",
        effort="high",
    )
    skill_tool = build_skill_tool((skill,))
    example_tool = _base_tool("Example")
    permission_context = ToolPermissionContext(
        always_allow_rules={"localSettings": ("Skill(deploy)",)}
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_skill",
                "name": SKILL_TOOL_NAME,
                "input": {"skill": "deploy"},
            },
            {
                "type": "tool_use",
                "id": "toolu_example",
                "name": "Example",
                "input": {},
            },
        ],
    }

    events = [
        event
        async for event in run_tools(
            tool_uses=(
                ToolUseBlock(
                    id="toolu_skill",
                    name=SKILL_TOOL_NAME,
                    input={"skill": "deploy"},
                    index=0,
                ),
                ToolUseBlock(
                    id="toolu_example",
                    name="Example",
                    input={},
                    index=1,
                ),
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(skill_tool, example_tool),
            deps=QueryDeps(
                task_store=AppStateStore(),
                permission_context=permission_context,
            ),
            ctx=_ctx(
                permission_context=permission_context,
                tools=(skill_tool, example_tool),
            ),
        )
    ]

    results = [event for event in events if isinstance(event, ToolExecutionResult)]
    outcome = next(event for event in events if isinstance(event, ToolOrchestrationOutcome))

    assert len(results) == 2
    assert _message_content(results[0].message) == "Launching skill: deploy"
    assert _message_content(results[1].message) == "ok"
    assert outcome.updated_context is not None
    assert outcome.updated_context.model_override == "claude-sonnet-skill"
    assert outcome.updated_context.reasoning_effort_override == "high"
    assert outcome.updated_context.permission_context.always_allow_rules["command"] == (
        "Example",
    )
    assert [*_message_contents(outcome.tool_result_messages)] == [
        "Launching skill: deploy",
        "Review this: ",
        "ok",
    ]


@pytest.mark.asyncio
async def test_skill_context_modifier_preserves_deps_permission_baseline() -> None:
    skill = _skill("deploy", allowed_tools=("Example",))
    skill_tool = build_skill_tool((skill,))
    example_tool = _base_tool("Example")
    blocked_tool = _base_tool("Blocked")
    permission_context = ToolPermissionContext(
        always_allow_rules={"localSettings": ("Skill(deploy)",)},
        always_deny_rules={"session": ("Blocked",)},
    )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_skill",
                "name": SKILL_TOOL_NAME,
                "input": {"skill": "deploy"},
            },
            {
                "type": "tool_use",
                "id": "toolu_example",
                "name": "Example",
                "input": {},
            },
            {
                "type": "tool_use",
                "id": "toolu_blocked",
                "name": "Blocked",
                "input": {},
            },
        ],
    }

    events = [
        event
        async for event in run_tools(
            tool_uses=(
                ToolUseBlock(
                    id="toolu_skill",
                    name=SKILL_TOOL_NAME,
                    input={"skill": "deploy"},
                    index=0,
                ),
                ToolUseBlock(
                    id="toolu_example",
                    name="Example",
                    input={},
                    index=1,
                ),
                ToolUseBlock(
                    id="toolu_blocked",
                    name="Blocked",
                    input={},
                    index=2,
                ),
            ),
            assistant_message=cast(Any, assistant_message),
            tools=(skill_tool, example_tool, blocked_tool),
            deps=QueryDeps(
                task_store=AppStateStore(),
                permission_context=permission_context,
            ),
            ctx=_ctx(tools=(skill_tool, example_tool, blocked_tool)),
        )
    ]

    results = [event for event in events if isinstance(event, ToolExecutionResult)]
    outcome = next(event for event in events if isinstance(event, ToolOrchestrationOutcome))

    assert [_message_content(result.message) for result in results] == [
        "Launching skill: deploy",
        "ok",
        "Permission to use Blocked has been denied.",
    ]
    assert outcome.updated_context is not None
    assert outcome.updated_context.permission_context.always_allow_rules["command"] == (
        "Example",
    )
    assert [denial.tool_name for denial in outcome.permission_denials] == ["Blocked"]


@pytest.mark.asyncio
async def test_pre_tool_hook_messages_precede_skill_result_and_skill_messages() -> None:
    skill_tool = build_skill_tool((_skill("review"),))

    async def pre_hook(_context: PreToolUseContext) -> PreToolUseResult:
        return PreToolUseResult(
            additional_messages=(
                cast(Any, {"role": "user", "content": "hook before tool result"}),
            )
        )

    events = [
        event
        async for event in run_tools(
            tool_uses=(
                ToolUseBlock(
                    id="toolu_skill",
                    name=SKILL_TOOL_NAME,
                    input={"skill": "review"},
                    index=0,
                ),
            ),
            assistant_message={
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            tools=(skill_tool,),
            deps=QueryDeps(
                task_store=AppStateStore(),
                pre_tool_use_hooks=[pre_hook],
            ),
            ctx=_ctx(tools=(skill_tool,)),
        )
    ]

    outcome = next(event for event in events if isinstance(event, ToolOrchestrationOutcome))

    assert [*_message_contents(outcome.tool_result_messages)] == [
        "hook before tool result",
        "Launching skill: review",
        "Review this: ",
    ]


@pytest.mark.asyncio
async def test_skill_model_override_affects_next_query_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = _skill("review")
    object.__setattr__(skill, "model", "claude-sonnet-skill")
    skill_tool = build_skill_tool((skill,))
    seen_models: list[str] = []
    seen_messages: list[list[Mapping[str, Any]]] = []

    async def fake_call(
        messages: list[Any],
        model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_models.append(model)
        seen_messages.append([cast(Mapping[str, Any], message) for message in messages])
        if len(seen_models) == 1:
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ]
            }
        return {"text": "review done"}

    monkeypatch.setattr(query_mod, "_call_model", fake_call)

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(
                model="claude-opus-base",
                tools=(skill_tool,),
            ),
            QueryDeps(
                task_store=AppStateStore(),
            ),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    tool_result_events = [
        event for event in events if isinstance(event, query_mod.ToolResultMessage)
    ]

    assert terminal.terminal.reason == "completed"
    assert seen_models == ["claude-opus-base", "claude-sonnet-skill"]
    assert len(tool_result_events) == 2
    assert _message_content(tool_result_events[0].message) == "Launching skill: review"
    assert tool_result_events[1].message["content"] == "Review this: "
    assert seen_messages[1][-1]["content"] == "Review this: "


@pytest.mark.asyncio
async def test_skill_model_and_effort_override_resolve_at_model_boundary() -> None:
    skill = _skill("review", model="skill-model", effort="high")
    skill_tool = build_skill_tool((skill,))
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ),
        resolved_models={
            "parent-model[1m]": "provider-parent[1m]",
            "skill-model": "provider-skill",
            "provider-skill[1m]": "provider-skill[1m]",
        },
        model_infos={
            "provider-skill": ModelInfo(
                model="provider-skill",
                context_window=1_000_000,
            )
        },
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(
                model="parent-model[1m]",
                tools=(skill_tool,),
            ),
            QueryDeps(
                task_store=AppStateStore(),
                model_provider=provider,
            ),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))

    assert terminal.terminal.reason == "completed"
    assert [requested for requested, _context in provider.resolve_requests] == [
        "parent-model[1m]",
        "skill-model",
        "provider-skill[1m]",
    ]
    assert provider.resolve_requests[1][1].effort == "high"
    assert provider.resolve_requests[2][1].effort == "high"
    assert [request.model for request in provider.requests] == [
        "provider-parent[1m]",
        "provider-skill[1m]",
    ]
    assert [request.effort for request in provider.requests] == [None, "high"]


@pytest.mark.asyncio
async def test_skill_effort_override_reaches_model_request_without_model_override() -> None:
    skill = _skill("review", effort="medium")
    skill_tool = build_skill_tool((skill,))
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(task_store=AppStateStore(), model_provider=provider),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    requests: list[ModelRequest] = provider.requests

    assert terminal.terminal.reason == "completed"
    assert [request.model for request in requests] == ["parent-model", "parent-model"]
    assert [request.effort for request in requests] == [None, "medium"]


@pytest.mark.asyncio
async def test_forked_skill_runs_child_query_and_returns_same_turn_result(
    tmp_path: Path,
) -> None:
    skill = _skill(
        "review",
        markdown_content="Fork this: $ARGUMENTS",
        allowed_tools=("Example",),
        context="fork",
        model="skill-model",
        effort="high",
    )
    example_tool = _base_tool("Example")
    skill_tool = build_skill_tool((skill,))
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review", "args": "diff.patch"},
                    }
                ],
            },
            {"role": "assistant", "content": "child reviewed diff"},
            {"role": "assistant", "content": "parent done"},
        ),
        resolved_models={
            "parent-model[1m]": "provider-parent[1m]",
            "skill-model": "provider-skill",
            "provider-skill[1m]": "provider-skill[1m]",
        },
        model_infos={
            "provider-skill": ModelInfo(
                model="provider-skill",
                context_window=1_000_000,
            )
        },
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        transcript_store=JsonlTranscriptStore(tmp_path / "transcripts"),
        permission_context=ToolPermissionContext(
            always_allow_rules={"localSettings": ("Skill(review)",)}
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(
                model="parent-model[1m]",
                tools=(skill_tool, example_tool),
                session_id="parent-session",
            ),
            deps,
            _ctx(tools=(skill_tool, example_tool)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    tool_result = next(
        event for event in events if isinstance(event, query_mod.ToolResultMessage)
    )

    assert terminal.terminal.reason == "completed"
    assert _message_content(tool_result.message) == (
        'Skill "review" completed (forked execution).\n\n'
        "Result:\nchild reviewed diff"
    )
    assert len(provider.requests) == 3
    child_request = provider.requests[1]
    assert child_request.agent_id is not None
    assert child_request.agent_id.startswith("a")
    assert child_request.model == "provider-skill[1m]"
    assert child_request.effort == "high"
    assert child_request.permission_context is not None
    child_allow_rules = cast(
        Mapping[str, object],
        child_request.permission_context.always_allow_rules,
    )
    assert child_allow_rules["command"] == (
        "Example",
    )
    assert provider.requests[2].permission_context is not None
    parent_allow_rules = cast(
        Mapping[str, object],
        provider.requests[2].permission_context.always_allow_rules,
    )
    assert "command" not in parent_allow_rules
    assert child_request.messages[-1].provider_payload == {
        "role": "user",
        "content": "Fork this: diff.patch",
    }

    assert deps.transcript_store is not None
    replay = await get_agent_transcript(
        deps.transcript_store,
        parent_session_id="parent-session",
        agent_id=child_request.agent_id,
    )
    assert replay is not None
    assert replay.messages == [
        {"role": "user", "content": "Fork this: diff.patch"},
        {"role": "assistant", "content": "child reviewed diff"},
    ]


@pytest.mark.asyncio
async def test_forked_skill_inherits_current_effective_model_from_context_modifier() -> None:
    set_model_skill = _skill("set-model", model="context-model")
    forked_skill = _skill("review", context="fork", agent="planner")
    skill_tool = build_skill_tool(
        (set_model_skill, forked_skill),
        agent_definitions=(
            AgentDefinition(
                agent_type="planner",
                description="Planner",
                system_prompt="You are the planner.",
                model="inherit",
            ),
        ),
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_set_model",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "set-model"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_review",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    },
                ],
            },
            {"role": "assistant", "content": "child used inherited context model"},
            {"role": "assistant", "content": "parent done"},
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "set model then fork"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(task_store=AppStateStore(), model_provider=provider),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))

    assert terminal.terminal.reason == "completed"
    assert [request.model for request in provider.requests] == [
        "parent-model",
        "context-model",
        "context-model",
    ]


@pytest.mark.asyncio
async def test_forked_skill_applies_selected_agent_permission_mode_to_child() -> None:
    skill_tool = build_skill_tool(
        (_skill("review", context="fork", agent="planner"),),
        agent_definitions=(
            AgentDefinition(
                agent_type="planner",
                description="Planner",
                system_prompt="You are the planner.",
                permission_mode="plan",
            ),
        ),
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_review",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            {"role": "assistant", "content": "child done"},
            {"role": "assistant", "content": "parent done"},
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "fork"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(task_store=AppStateStore(), model_provider=provider),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    child_request = provider.requests[1]

    assert terminal.terminal.reason == "completed"
    assert child_request.permission_context is not None
    assert child_request.permission_context.mode == "plan"


@pytest.mark.asyncio
async def test_forked_skill_filters_selected_agent_context_policy() -> None:
    skill_tool = build_skill_tool(
        (_skill("review", context="fork", agent="planner"),),
        agent_definitions=(
            AgentDefinition(
                agent_type="planner",
                description="Planner",
                system_prompt="You are the planner.",
                context_policy=AgentContextPolicy.minimal(),
            ),
        ),
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_review",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            {"role": "assistant", "content": "child done"},
            {"role": "assistant", "content": "parent done"},
        ),
    )
    deps = QueryDeps(
        task_store=AppStateStore(),
        model_provider=provider,
        context_providers=(
            RecordingContextProvider("project_instructions", "PROJECT-CONTEXT"),
            RecordingContextProvider("git", "GIT-CONTEXT"),
            RecordingContextProvider("custom", "CUSTOM-CONTEXT"),
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "fork"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            deps,
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    assert terminal.terminal.reason == "completed"
    child_payload = "\n".join(
        str(message.provider_payload) for message in provider.requests[1].messages
    )
    assert "CUSTOM-CONTEXT" in child_payload
    assert "PROJECT-CONTEXT" not in child_payload
    assert "GIT-CONTEXT" not in child_payload


@pytest.mark.asyncio
async def test_forked_skill_parent_permission_mode_takes_precedence() -> None:
    skill_tool = build_skill_tool(
        (_skill("review", context="fork", agent="planner"),),
        agent_definitions=(
            AgentDefinition(
                agent_type="planner",
                description="Planner",
                system_prompt="You are the planner.",
                permission_mode="plan",
            ),
        ),
    )
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_review",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            {"role": "assistant", "content": "child done"},
            {"role": "assistant", "content": "parent done"},
        ),
    )
    permission_context = ToolPermissionContext(mode="bypassPermissions")

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "fork"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(
                task_store=AppStateStore(),
                model_provider=provider,
                permission_context=permission_context,
            ),
            _ctx(permission_context=permission_context, tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    child_request = provider.requests[1]

    assert terminal.terminal.reason == "completed"
    assert child_request.permission_context is not None
    assert child_request.permission_context.mode == "bypassPermissions"


@pytest.mark.asyncio
async def test_forked_skill_child_failure_maps_to_error_tool_result() -> None:
    skill_tool = build_skill_tool((_skill("review", context="fork"),))
    provider = FakeModelProvider(
        responses=(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_skill",
                        "name": SKILL_TOOL_NAME,
                        "input": {"skill": "review"},
                    }
                ],
            },
            RuntimeError("child model failed"),
            {"role": "assistant", "content": "parent handled child failure"},
        ),
    )

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(task_store=AppStateStore(), model_provider=provider),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    tool_result = next(
        event for event in events if isinstance(event, query_mod.ToolResultMessage)
    )

    assert terminal.terminal.reason == "completed"
    assert _message_content(tool_result.message) == (
        'Skill "review" completed (forked execution).\n\n'
        "Result:\nchild model failed"
    )
    assert _tool_result_is_error(tool_result.message)


@pytest.mark.asyncio
async def test_forked_skill_parent_abort_during_child_returns_aborted_tools() -> None:
    skill_tool = build_skill_tool((_skill("review", context="fork"),))

    class AbortDuringChildProvider(FakeModelProvider):
        async def complete(self, request: ModelRequest) -> Any:
            self.requests.append(request)
            if len(self.requests) == 1:
                return model_response_from_message_param(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_skill",
                                "name": SKILL_TOOL_NAME,
                                "input": {"skill": "review"},
                            }
                        ],
                    }
                )
            assert request.abort_event is not None
            request.abort_event.set()
            raise asyncio.CancelledError()

    provider = AbortDuringChildProvider()

    events = [
        event
        async for event in query_mod.query(
            State(messages=[{"role": "user", "content": "use review"}]),
            QueryConfig(model="parent-model", tools=(skill_tool,)),
            QueryDeps(task_store=AppStateStore(), model_provider=provider),
            _ctx(tools=(skill_tool,)),
        )
    ]

    terminal = next(event for event in events if isinstance(event, query_mod.TerminalEvent))
    tool_result = next(
        event for event in events if isinstance(event, query_mod.ToolResultMessage)
    )

    assert terminal.terminal.reason == "aborted_tools"
    assert TOOL_CANCEL_MESSAGE in str(tool_result.message["content"])
    assert len(provider.requests) == 2
    assert provider.requests[1].agent_id is not None


def test_normalize_skill_name_strips_one_leading_slash() -> None:
    assert normalize_skill_name(" /review ") == "review"
    assert normalize_skill_name("review") == "review"


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    assert isinstance(content, list)
    block = cast(Mapping[str, Any], content[0])
    return str(block.get("content", ""))


def _message_contents(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
        else:
            values.append(_message_content(message))
    return values


def _tool_result_is_error(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    assert isinstance(content, list)
    block = cast(Mapping[str, Any], content[0])
    return block.get("is_error") is True
