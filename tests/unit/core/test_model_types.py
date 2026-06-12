from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from raygent_harness.core import model_provider, model_types
from raygent_harness.core.messages import message_param_from_model_api_error
from raygent_harness.core.model_types import (
    AgentDescriptor,
    ApiMessage,
    ModelFallbackControl,
    ModelRequest,
    ModelResolveContext,
    ModelStreamEvent,
    ModelToolUseBlock,
    ObservableMessage,
    PermissionContextSnapshot,
    ProviderError,
    StreamIdentity,
    TextContentBlock,
    TokenCountRequest,
    ToolUseContentBlock,
    build_model_api_error_message,
)


def test_api_and_observable_messages_diverge_without_mutating_api_payload() -> None:
    api_tool_use = ToolUseContentBlock(
        id="toolu_1",
        name="FileRead",
        input={"path": "README.md"},
    )
    api_message = ApiMessage(
        message=model_types.ModelMessage(role="assistant", content=(api_tool_use,)),
    )

    observable_tool_use = ToolUseContentBlock(
        id="toolu_1",
        name="FileRead",
        input={"path": "README.md", "display_path": "/repo/README.md"},
    )
    observable_message = ObservableMessage(
        message=model_types.ModelMessage(role="assistant", content=(observable_tool_use,)),
    )

    api_content = cast(ToolUseContentBlock, api_message.message.content[0])
    observable_content = cast(ToolUseContentBlock, observable_message.message.content[0])
    api_input = cast(Mapping[str, object], api_content.input)
    observable_input = cast(Mapping[str, object], observable_content.input)

    assert api_content.input != observable_content.input
    assert observable_input["display_path"] == "/repo/README.md"
    assert "display_path" not in api_input

    with pytest.raises(TypeError):
        cast(Any, api_content.input)["display_path"] = "/repo/README.md"

    with pytest.raises(FrozenInstanceError):
        cast(Any, api_message).message = observable_message.message


def test_model_api_error_message_preserves_recovery_details() -> None:
    api_error = build_model_api_error_message(
        kind="context_overflow",
        public_message="Prompt is too long",
        raw_details="prompt is too long: 137500 tokens > 135000 maximum",
        actual_tokens=137_500,
        limit_tokens=135_000,
        status_code=400,
    )
    provider_error = ProviderError(
        kind="context_overflow",
        message="Prompt is too long",
        raw_details=api_error.raw_details,
        actual_tokens=api_error.actual_tokens,
        limit_tokens=api_error.limit_tokens,
        status_code=api_error.status_code,
        api_error=api_error,
    )

    assert api_error.token_gap == 2_500
    provider_api_error = provider_error.api_error
    assert provider_api_error is not None
    assert provider_api_error is api_error
    assert provider_api_error.raw_details == (
        "prompt is too long: 137500 tokens > 135000 maximum"
    )

    api_text = cast(TextContentBlock, api_error.api_message.message.content[0])
    observable_text = cast(TextContentBlock, api_error.observable_message.message.content[0])
    assert api_text.text == "Prompt is too long"
    assert observable_text.text == "Prompt is too long"

    message = message_param_from_model_api_error(api_error)
    assert message.get("isApiErrorMessage") is True
    assert message.get("apiError") == "context_overflow"
    assert message.get("error") == "context_overflow"
    assert message.get("errorDetails") == (
        "prompt is too long: 137500 tokens > 135000 maximum"
    )


def test_stream_events_separate_transport_fallback_from_model_fallback() -> None:
    identity = StreamIdentity(
        message_id="msg_1",
        content_block_index=0,
        provider_request_id="req_1",
        attempt_id="attempt_1",
    )

    transport = ModelStreamEvent.streaming_transport_fallback_started(
        identity,
        reason="stream watchdog",
    )
    model = ModelStreamEvent.model_fallback_triggered(
        identity,
        original_model="primary",
        fallback_model="fallback",
        reason="server overload",
    )

    assert transport.type == "streaming_transport_fallback_started"
    assert transport.model_fallback is None
    assert cast(Mapping[str, object], transport.delta)["reason"] == "stream watchdog"

    assert model.type == "model_fallback_triggered"
    assert model.model_fallback is not None
    assert model.model_fallback.original_model == "primary"
    assert model.model_fallback.fallback_model == "fallback"
    assert cast(Mapping[str, object], model.delta)["reason"] == "server overload"

    with pytest.raises(TypeError):
        cast(Any, model.delta)["reason"] = "mutated"


def test_provider_error_carries_model_fallback_control_metadata() -> None:
    fallback = ModelFallbackControl(
        original_model="primary",
        fallback_model="fallback",
        reason="server overload",
    )
    error = ProviderError(
        kind="model_fallback_triggered",
        message="Model fallback triggered",
        retryable=True,
        safe_to_fallback=True,
        model_fallback=fallback,
    )

    assert error.model_fallback is fallback
    error_fallback = error.model_fallback
    assert error_fallback is not None
    assert error_fallback.original_model == "primary"
    assert error_fallback.fallback_model == "fallback"
    assert error_fallback.reason == "server overload"


def test_stream_identity_supports_attempt_level_fallback_before_message_start() -> None:
    identity = StreamIdentity(
        provider_request_id="req_1",
        attempt_id="attempt_1",
    )
    event = ModelStreamEvent.streaming_transport_fallback_started(
        identity,
        reason="stream creation failed before message_start",
    )

    assert event.identity.message_id is None
    assert event.identity.provider_request_id == "req_1"
    assert event.identity.attempt_id == "attempt_1"

    with pytest.raises(ValueError):
        StreamIdentity()


def test_token_count_request_carries_kernel_metadata() -> None:
    message = ApiMessage(
        message=model_types.ModelMessage(
            role="user",
            content=(TextContentBlock(text="hello"),),
        )
    )
    request = ModelRequest(
        model="model-1",
        fallback_model="model-2",
        messages=[message],  # pyright: ignore[reportArgumentType]
        effort="high",
        agent_id="agent_1",
        task_budget=model_types.TaskBudgetInfo(total=10, remaining=7),
        query_source="sdk",
        tool_choice="auto",
        max_output_tokens_override=16_000,
        permission_context=PermissionContextSnapshot(
            mode="plan",
            always_allow_rules={"Bash": ("git status",)},
        ),
        active_agents=[
            AgentDescriptor(
                agent_type="worker",
                description="general worker",
                tools=("Read", "Write"),
                model="model-2",
                permission_mode="default",
                background=True,
                source="built-in",
            )
        ],  # pyright: ignore[reportArgumentType]
        allowed_agent_types=["worker", "reviewer"],  # pyright: ignore[reportArgumentType]
        mcp_tool_names=["mcp__server__tool"],  # pyright: ignore[reportArgumentType]
        has_pending_mcp_servers=True,
        provider_options={"beta_headers": ("context-1m",)},
    )
    token_request = TokenCountRequest(
        model="model-1",
        messages=[message],  # pyright: ignore[reportArgumentType]
        system_prompt="system prompt",
        thinking={"type": "enabled", "budget_tokens": 1024},
        effort="high",
        media_context={"images": 1},
        provider_options={"beta_headers": ("context-1m",)},
    )

    assert request.messages == (message,)
    assert request.fallback_model == "model-2"
    assert request.permission_context is not None
    assert request.permission_context.mode == "plan"
    assert cast(Mapping[str, object], request.permission_context.always_allow_rules)[
        "Bash"
    ] == ("git status",)
    assert request.active_agents[0].agent_type == "worker"
    assert request.active_agents[0].tools == ("Read", "Write")
    assert request.allowed_agent_types == ("worker", "reviewer")
    assert request.mcp_tool_names == ("mcp__server__tool",)
    assert request.has_pending_mcp_servers is True
    assert cast(Mapping[str, object], request.provider_options)["beta_headers"] == (
        "context-1m",
    )

    assert token_request.messages == (message,)
    assert token_request.system_prompt == "system prompt"
    assert cast(Mapping[str, object], token_request.thinking)["budget_tokens"] == 1024
    assert cast(Mapping[str, object], token_request.media_context)["images"] == 1


def test_model_tool_use_block_freezes_invalid_model_input_shape() -> None:
    tool_use = ModelToolUseBlock(
        id="toolu_1",
        name="Example",
        input={"items": ({"name": "a"},)},
        index=3,
    )

    items = cast(tuple[object, ...], cast(Mapping[str, object], tool_use.input)["items"])
    assert cast(Mapping[str, object], items[0])["name"] == "a"
    with pytest.raises(TypeError):
        cast(Any, items[0])["name"] = "b"


def test_model_resolve_context_is_provider_neutral() -> None:
    context = ModelResolveContext(
        permission_mode="plan",
        query_source="sdk",
        agent_id=None,
        exceeds_200k_tokens=True,
        requested_context_window=1_000_000,
        effort="medium",
    )

    assert context.permission_mode == "plan"
    assert context.exceeds_200k_tokens is True
    assert context.requested_context_window == 1_000_000


def test_new_model_modules_do_not_import_provider_sdks() -> None:
    sources = (
        inspect.getsource(model_types),
        inspect.getsource(model_provider),
    )

    assert "anthropic" not in "\n".join(sources).lower()
