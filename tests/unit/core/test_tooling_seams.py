from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from raygent_harness.core import query as query_mod
from raygent_harness.core.config import QueryConfig
from raygent_harness.core.deps import QueryDeps
from raygent_harness.core.permissions import (
    OtherPermissionDecisionReason,
    PermissionDenyDecision,
    ToolPermissionContext,
)
from raygent_harness.core.query_engine import QueryEngine, SDKResult, SDKSystemInit
from raygent_harness.core.task import AppStateStore
from raygent_harness.core.tool import (
    QueryTracking,
    Tool,
    ToolCallEvent,
    ToolResult,
    ToolSpec,
    ToolUseContext,
    build_tool,
)
from raygent_harness.skills.models import SkillDefinition

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam


class EmptyInput(BaseModel):
    pass


async def _call(
    _input: BaseModel,
    _ctx: ToolUseContext,
) -> AsyncIterator[ToolCallEvent]:
    yield ToolResult(content="ok")


def _tool(name: str, *, check_permissions: Any | None = None) -> Tool:
    return build_tool(
        ToolSpec(
            name=name,
            description=f"{name} description",
            input_model=EmptyInput,
            call=_call,
            check_permissions=check_permissions,
            is_read_only=True,
            is_concurrency_safe=True,
        )
    )


def _ctx() -> ToolUseContext:
    return ToolUseContext(
        session_id="s",
        agent_id=None,
        abort_event=asyncio.Event(),
        rendered_system_prompt="",
        cwd=".",
        query_tracking=QueryTracking(chain_id="c", depth=0),
    )


def _skill(name: str = "review") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=f"{name} skill",
        markdown_content="Use the skill.",
        source="projectSettings",
        loaded_from="skills",
        content_length=len("Use the skill."),
    )


def _patch_clean_response(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seen_tools: list[tuple[str, ...]],
    text: str = "ok",
) -> None:
    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        seen_tools.append(tuple(tool.name for tool in config.tools))
        return {"text": text}

    def fake_assistant(response: Any) -> MessageParam:
        return {"role": "assistant", "content": response["text"]}

    def fake_tool_uses(_response: Any) -> list[Any]:
        return []

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    monkeypatch.setattr(query_mod, "_extract_assistant_message", fake_assistant)
    monkeypatch.setattr(query_mod, "_extract_tool_uses", fake_tool_uses)


@pytest.mark.asyncio
async def test_default_tooling_seams_preserve_turn_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[tuple[str, ...]] = []
    _patch_clean_response(monkeypatch, seen_tools=seen_tools)
    base_tool = _tool("Base")
    deps = QueryDeps(task_store=AppStateStore())
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(base_tool,),
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[0], SDKSystemInit)
    assert events[0].tools == ("Base",)
    assert isinstance(events[-1], SDKResult)
    assert events[-1].subtype == "success"
    assert seen_tools == [("Base",)]
    assert deps.skill_provider is None
    assert deps.tool_catalog_provider is None
    assert deps.permission_handler is None


@pytest.mark.asyncio
async def test_system_init_reports_permission_context_mode_not_config_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[tuple[str, ...]] = []
    _patch_clean_response(monkeypatch, seen_tools=seen_tools)
    deps = QueryDeps(
        task_store=AppStateStore(),
        permission_context=ToolPermissionContext(mode="dontAsk"),
    )
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            permission_mode="bypass",
            tools=(_tool("Base"),),
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[0], SDKSystemInit)
    assert events[0].permission_mode == "dontAsk"
    assert seen_tools == [("Base",)]


@pytest.mark.asyncio
async def test_skill_and_tool_catalog_providers_can_expand_turn_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[tuple[str, ...]] = []
    provider_calls: list[tuple[str, tuple[str, ...]]] = []
    _patch_clean_response(monkeypatch, seen_tools=seen_tools)
    base_tool = _tool("Base")
    skill = _skill("review")
    skill_tool = _tool("SkillReview")

    async def skill_provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> Sequence[SkillDefinition]:
        provider_calls.append(("skill", (ctx.session_id, config.model)))
        return (skill,)

    async def tool_catalog_provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        provider_calls.append(("catalog", tuple(s.name for s in skills)))
        assert ctx.session_id == "s"
        return (*config.tools, skill_tool)

    deps = QueryDeps(
        task_store=AppStateStore(),
        skill_provider=skill_provider,
        tool_catalog_provider=tool_catalog_provider,
    )
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(base_tool,),
        ),
        deps,
        _ctx(),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[0], SDKSystemInit)
    assert events[0].tools == ("Base", "SkillReview")
    assert isinstance(events[-1], SDKResult)
    assert seen_tools == [("Base", "SkillReview")]
    assert provider_calls == [
        ("skill", ("s", "claude-opus-4-7")),
        ("catalog", ("review",)),
    ]


@pytest.mark.asyncio
async def test_tool_catalog_provider_sees_prior_conversation_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[tuple[str, ...]] = []
    seen_provider_messages: list[list[MessageParam]] = []
    _patch_clean_response(monkeypatch, seen_tools=seen_tools)
    base_tool = _tool("Base")

    async def tool_catalog_provider(
        config: QueryConfig,
        ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        seen_provider_messages.append(list(ctx.messages))
        return config.tools

    deps = QueryDeps(
        task_store=AppStateStore(),
        tool_catalog_provider=tool_catalog_provider,
    )
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(base_tool,),
        ),
        deps,
        _ctx(),
    )

    first = [event async for event in engine.submit_message("first")]
    second = [event async for event in engine.submit_message("second")]

    assert isinstance(first[-1], SDKResult)
    assert isinstance(second[-1], SDKResult)
    assert seen_provider_messages == [
        [],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
        ],
    ]


@pytest.mark.asyncio
async def test_provider_failures_are_noop_for_turn_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[tuple[str, ...]] = []
    _patch_clean_response(monkeypatch, seen_tools=seen_tools)
    base_tool = _tool("Base")

    async def failing_skill_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        /,
    ) -> Sequence[SkillDefinition]:
        raise RuntimeError("skill source unavailable")

    async def failing_catalog_provider(
        _config: QueryConfig,
        _ctx: ToolUseContext,
        _skills: Sequence[SkillDefinition],
        /,
    ) -> Sequence[Tool] | None:
        raise RuntimeError("catalog unavailable")

    skill_failure_engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(base_tool,),
        ),
        QueryDeps(
            task_store=AppStateStore(),
            skill_provider=failing_skill_provider,
        ),
        _ctx(),
    )
    catalog_failure_engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(base_tool,),
        ),
        QueryDeps(
            task_store=AppStateStore(),
            tool_catalog_provider=failing_catalog_provider,
        ),
        _ctx(),
    )

    skill_failure_events = [
        event async for event in skill_failure_engine.submit_message("hi")
    ]
    catalog_failure_events = [
        event async for event in catalog_failure_engine.submit_message("hi")
    ]

    assert isinstance(skill_failure_events[-1], SDKResult)
    assert skill_failure_events[-1].subtype == "success"
    assert isinstance(catalog_failure_events[-1], SDKResult)
    assert catalog_failure_events[-1].subtype == "success"
    assert seen_tools == [("Base",), ("Base",)]


@pytest.mark.asyncio
async def test_query_deps_permission_seam_uses_shared_context() -> None:
    tool = _tool("Danger")
    deps = QueryDeps(
        task_store=AppStateStore(),
        permission_context=ToolPermissionContext(
            always_deny_rules={"session": ("Danger",)}
        ),
    )

    result = await deps.resolve_tool_permission(
        tool=tool,
        input=EmptyInput(),
        tool_use_context=_ctx(),
        tools=(tool,),
    )

    assert isinstance(result.decision, PermissionDenyDecision)


@pytest.mark.asyncio
async def test_query_engine_reconciles_tool_permission_denials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[Any] = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_denied",
                    "name": "Denied",
                    "input": {},
                }
            ]
        },
        {"text": "handled"},
    ]

    async def fake_call(
        _msgs: list[MessageParam],
        _model: str,
        _config: QueryConfig,
        _deps: QueryDeps,
        _ctx: ToolUseContext,
    ) -> Any:
        return responses.pop(0)

    async def deny(
        _input: BaseModel,
        _ctx: ToolUseContext,
        _permission_context: ToolPermissionContext,
    ) -> PermissionDenyDecision:
        return PermissionDenyDecision(
            message="denied by test",
            decision_reason=OtherPermissionDecisionReason(reason="test"),
        )

    monkeypatch.setattr(query_mod, "_call_model", fake_call)
    engine = QueryEngine(
        QueryConfig(
            model="claude-opus-4-7",
            session_id="s",
            tools=(_tool("Denied", check_permissions=deny),),
        ),
        QueryDeps(
            task_store=AppStateStore(),
        ),
        _ctx(),
    )

    events = [event async for event in engine.submit_message("hi")]

    assert isinstance(events[-1], SDKResult)
    result = events[-1]
    assert result.subtype == "success"
    assert len(result.permission_denials) == 1
    denial = result.permission_denials[0]
    assert denial.tool_use_id == "tu_denied"
    assert denial.tool_name == "Denied"
    assert denial.reason == "denied by test"
