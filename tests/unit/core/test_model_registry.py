from __future__ import annotations

from typing import cast

import pytest

from raygent_harness.core.model_registry import (
    CAPPED_DEFAULT_MAX_TOKENS,
    CONTEXT_1M_WINDOW_TOKENS,
    MAX_OUTPUT_TOKENS_DEFAULT,
    ModelRegistry,
    carry_context_window_suffix,
    count_message_tokens,
    count_message_tokens_report,
    get_context_window_for_model,
    get_model_output_limits,
    resolve_skill_model_override,
    split_context_window_suffix,
)
from raygent_harness.core.model_types import (
    ModelCapabilities,
    ModelInfo,
    ModelResolveContext,
    ModelToolSpec,
    TokenCountResult,
)
from tests.fakes import FakeModelProvider


def test_registry_resolves_aliases_and_preserves_window_suffix() -> None:
    registry = ModelRegistry(
        [
            ModelInfo(
                model="provider-sonnet",
                canonical_name="provider-sonnet-20260515",
                context_window=200_000,
                aliases=("sonnet",),
                capabilities=ModelCapabilities(supports_streaming=True),
            )
        ]
    )

    assert registry.resolve_model("sonnet") == "provider-sonnet"
    assert registry.resolve_model("sonnet[1m]") == "provider-sonnet[1m]"
    assert registry.resolve_model("custom-model[1m]") == "custom-model[1m]"

    info = registry.model_info("sonnet[1m]")
    assert info.model == "provider-sonnet[1m]"
    assert info.context_window == CONTEXT_1M_WINDOW_TOKENS
    assert info.capabilities.supports_streaming is True


def test_context_window_and_output_limits_use_metadata_then_fallbacks() -> None:
    info = ModelInfo(
        model="model-1",
        context_window=512_000,
        max_output_tokens_default=12_000,
        max_output_tokens_upper_limit=16_000,
    )

    assert get_context_window_for_model("model-1", model_info=info) == 512_000
    assert get_context_window_for_model("model-1[1m]", model_info=info) == 1_000_000
    assert get_context_window_for_model("unknown") == 200_000

    limits = get_model_output_limits("model-1", model_info=info)
    assert limits.default == 12_000
    assert limits.upper_limit == 16_000

    capped = get_model_output_limits("model-1", model_info=info, cap_default=True)
    assert capped.default == min(12_000, CAPPED_DEFAULT_MAX_TOKENS)

    overridden = get_model_output_limits(
        "model-1",
        model_info=info,
        env={"RAYGENT_MAX_OUTPUT_TOKENS": "99_999"},
    )
    assert overridden.default == 16_000

    fallback = get_model_output_limits("unknown")
    assert fallback.default == MAX_OUTPUT_TOKENS_DEFAULT


def test_split_context_window_suffix_is_provider_neutral() -> None:
    parts = split_context_window_suffix("  Any-Provider-Model[1M] ")

    assert parts.original == "Any-Provider-Model[1M]"
    assert parts.base == "Any-Provider-Model"
    assert parts.requested_context_window == CONTEXT_1M_WINDOW_TOKENS
    assert parts.suffix == "1m"


def test_skill_model_override_carries_parent_context_suffix() -> None:
    assert carry_context_window_suffix("skill-model", "parent-model[1m]") == (
        "skill-model[1m]"
    )
    assert carry_context_window_suffix("skill-model[1m]", "parent-model") == (
        "skill-model[1m]"
    )
    assert carry_context_window_suffix("skill-model", "parent-model") == "skill-model"


def test_resolve_skill_model_override_uses_provider_then_registry() -> None:
    provider = FakeModelProvider(
        resolved_models={
            "skill-alias": "provider-skill",
            "provider-skill[1m]": "provider-skill[1m]",
            "parent[1m]": "provider-parent[1m]",
        },
        model_infos={
            "provider-skill": ModelInfo(
                model="provider-skill",
                context_window=CONTEXT_1M_WINDOW_TOKENS,
            )
        },
    )
    context = ModelResolveContext(agent_id="agent-1", effort="high")

    resolved = resolve_skill_model_override(
        "skill-alias",
        "parent-model[1m]",
        provider=provider,
        context=context,
    )

    assert resolved == "provider-skill[1m]"
    assert provider.resolve_requests[:2] == [
        ("skill-alias", context),
        ("provider-skill[1m]", context),
    ]
    assert resolve_skill_model_override(
        "inherit",
        "parent[1m]",
        provider=provider,
        context=context,
    ) == "provider-parent[1m]"

    registry = ModelRegistry(
        [
            ModelInfo(
                model="registered-model",
                aliases=("skill",),
                context_window=CONTEXT_1M_WINDOW_TOKENS,
            )
        ]
    )
    assert (
        resolve_skill_model_override("skill", "parent-model[1m]", registry=registry)
        == "registered-model[1m]"
    )


def test_resolve_skill_model_override_does_not_carry_suffix_to_unsupported_model() -> None:
    provider = FakeModelProvider(
        resolved_models={"haiku": "provider-haiku"},
        model_infos={"provider-haiku": ModelInfo(model="provider-haiku", context_window=200_000)},
    )

    assert (
        resolve_skill_model_override(
            "haiku",
            "parent-model[1m]",
            provider=provider,
        )
        == "provider-haiku"
    )


@pytest.mark.asyncio
async def test_count_message_tokens_builds_provider_request() -> None:
    provider = FakeModelProvider(token_counts=(123,))
    tool = ModelToolSpec(
        name="Search",
        description="Search files",
        input_schema={"type": "object"},
    )

    count = await count_message_tokens(
        provider=provider,
        model="model-1",
        messages=[{"role": "user", "content": "hello"}],
        tools=(tool,),
        thinking={"type": "enabled", "budget_tokens": 1024},
        effort="high",
        media_context={"images": 1},
        provider_options={"beta_headers": ("context-1m",)},
        fallback_estimator=lambda _messages: 999,
    )

    assert count == 123
    assert len(provider.token_requests) == 1
    request = provider.token_requests[0]
    assert request.model == "model-1"
    assert request.tools == (tool,)
    assert request.messages[0].provider_payload is not None
    assert cast(dict[str, object], request.messages[0].provider_payload)["role"] == "user"
    assert request.effort == "high"


@pytest.mark.asyncio
async def test_count_message_tokens_report_preserves_provider_metadata() -> None:
    provider = FakeModelProvider(
        token_counts=(
            TokenCountResult(
                token_count=123,
                provider_request_id="count_req_1",
                safe_metadata={"cache": "hit", "tokenizer": "provider"},
            ),
        )
    )

    report = await count_message_tokens_report(
        provider=provider,
        model="model-1",
        messages=[{"role": "user", "content": "hello"}],
        system_prompt="system prompt",
        fallback_estimator=lambda _messages: 999,
    )

    assert report.token_count == 123
    assert report.provider_token_count == 123
    assert report.provider_request_id == "count_req_1"
    assert cast(dict[str, object], report.provider_metadata) == {
        "cache": "hit",
        "tokenizer": "provider",
    }
    assert provider.token_requests[0].system_prompt == "system prompt"


@pytest.mark.asyncio
async def test_count_message_tokens_falls_back_when_provider_cannot_count() -> None:
    provider = FakeModelProvider(token_counts=(RuntimeError("count unavailable"),))

    count = await count_message_tokens(
        provider=provider,
        model="model-1",
        messages=[{"role": "user", "content": "hello"}],
        fallback_estimator=lambda messages: len(str(messages[0]["content"])),
    )

    assert count == 5
    assert len(provider.token_requests) == 1
