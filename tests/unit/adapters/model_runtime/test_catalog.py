from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from raygent_harness.adapters.model_protocols import AnthropicMessagesAdapter
from raygent_harness.adapters.model_runtime import (
    ProtocolModelProvider,
    ProviderModelCatalog,
    ProviderModelEntry,
    capabilities_from_modalities,
    merge_model_info,
    registry_from_catalogs,
)
from raygent_harness.core.media_budget import EXCESS_MEDIA_REMOVED_PLACEHOLDER
from raygent_harness.core.messages import thaw_json
from raygent_harness.core.model_registry import (
    CONTEXT_1M_WINDOW_TOKENS,
    get_context_window_for_model,
    get_model_output_limits,
)
from raygent_harness.core.model_request_normalization import (
    TOOL_REFERENCES_REMOVED_PLACEHOLDER,
    UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER,
    normalize_model_request_for_provider,
)
from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    MediaContentBlock,
    ModelCapabilities,
    ModelInfo,
    ModelMessage,
    ModelRequest,
    ModelResolveContext,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from tests.unit.adapters.model_runtime.test_protocol_model_provider import FakeTransport


def test_provider_model_entry_converts_to_model_info_with_capabilities() -> None:
    entry = ProviderModelEntry(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        model_id="sonnet",
        api_model_id="claude-sonnet-4-20260601",
        display_name="Sonnet 4",
        aliases=("default",),
        status="beta",
        context_window=CONTEXT_1M_WINDOW_TOKENS,
        input_token_limit=900_000,
        max_output_tokens_default=12_000,
        max_output_tokens_upper_limit=64_000,
        max_media_items_per_request=5,
        input_modalities=("text", "image/*", "application/pdf"),
        output_modalities=("text", "thinking"),
        safe_metadata={"tier": "context"},
    )

    info = entry.to_model_info()

    assert info.model == "claude-sonnet-4-20260601"
    assert info.canonical_name == "claude-sonnet-4-20260601"
    assert info.provider_id == "anthropic"
    assert info.protocol_id == "anthropic_messages"
    assert info.display_name == "Sonnet 4"
    assert info.status == "beta"
    assert info.context_window == CONTEXT_1M_WINDOW_TOKENS
    assert info.input_token_limit == 900_000
    assert info.max_output_tokens_default == 12_000
    assert info.max_output_tokens_upper_limit == 64_000
    assert info.max_media_items_per_request == 5
    assert info.aliases == ("sonnet", "default")
    assert info.input_modalities == ("text", "image/*", "application/pdf")
    assert info.output_modalities == ("text", "thinking")
    assert info.capabilities == ModelCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_thinking=True,
        supports_effort=False,
        supports_media=True,
        supports_images=True,
        supports_documents=True,
        supports_tool_references=False,
    )
    metadata = thaw_json(info.safe_metadata)
    assert isinstance(metadata, Mapping)
    assert metadata["provider_id"] == "anthropic"
    assert metadata["protocol_id"] == "anthropic_messages"
    assert metadata["model_id"] == "sonnet"
    assert metadata["api_model_id"] == "claude-sonnet-4-20260601"
    assert metadata["tier"] == "context"


def test_catalog_registry_preserves_aliases_and_window_suffixes() -> None:
    catalog = ProviderModelCatalog(
        provider_id="openai",
        protocol_id="openai_responses",
        models=(
            ProviderModelEntry(
                provider_id="openai",
                protocol_id="openai_responses",
                model_id="reasoning",
                api_model_id="gpt-reasoning-2026",
                aliases=("default",),
                context_window=CONTEXT_1M_WINDOW_TOKENS,
            ),
        ),
    )

    registry = catalog.registry()

    assert registry.resolve_model("reasoning") == "gpt-reasoning-2026"
    assert registry.resolve_model("default[1m]") == "gpt-reasoning-2026[1m]"
    assert registry.model_info("default[1m]").context_window == CONTEXT_1M_WINDOW_TOKENS


def test_merge_model_info_applies_application_overrides() -> None:
    base = ProviderModelEntry(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        model_id="sonnet",
        api_model_id="claude-sonnet-base",
        aliases=("default",),
        context_window=200_000,
        max_output_tokens_default=8_000,
        safe_metadata={"source": "sample"},
    ).to_model_info()
    override = ModelInfo(
        model="claude-sonnet-base",
        context_window=CONTEXT_1M_WINDOW_TOKENS,
        max_output_tokens_default=16_000,
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_images=True,
            supports_media=True,
        ),
        aliases=("project-default",),
        safe_metadata={"source": "app", "region": "us"},
    )

    merged = merge_model_info(base, override)

    assert merged.context_window == CONTEXT_1M_WINDOW_TOKENS
    assert merged.max_output_tokens_default == 16_000
    assert merged.aliases == ("sonnet", "default", "project-default")
    assert merged.capabilities.supports_images is True
    assert thaw_json(merged.safe_metadata) == {
        "provider_id": "anthropic",
        "protocol_id": "anthropic_messages",
        "model_id": "sonnet",
        "api_model_id": "claude-sonnet-base",
        "enabled": True,
        "source": "app",
        "region": "us",
    }


def test_catalog_model_info_feeds_limits_and_request_normalization() -> None:
    capabilities = capabilities_from_modalities(
        input_modalities=("text", "image/*"),
        output_modalities=("text",),
        supports_tool_references=False,
    )
    info = ProviderModelEntry(
        provider_id="openai",
        protocol_id="openai_responses",
        model_id="vision-lite",
        context_window=300_000,
        max_output_tokens_default=10_000,
        max_output_tokens_upper_limit=20_000,
        max_media_items_per_request=1,
        input_modalities=("text", "image/*"),
        output_modalities=("text",),
        capabilities=capabilities,
    ).to_model_info()
    request = ModelRequest(
        model="vision-lite",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="assistant",
                    content=(
                        ToolUseContentBlock(
                            id="toolu_search",
                            name="ToolSearch",
                            input={"query": "Read"},
                            provider_metadata={"caller": {"type": "tool_search"}},
                        ),
                    ),
                )
            ),
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(
                        ToolResultContentBlock(
                            tool_use_id="toolu_search",
                            content=cast(
                                FrozenJson,
                                [{"type": "tool_reference", "tool_name": "Read"}],
                            ),
                        ),
                        MediaContentBlock(
                            media_kind="image",
                            media_type="image/png",
                            data={"id": "old"},
                        ),
                        MediaContentBlock(
                            media_kind="image",
                            media_type="image/png",
                            data={"id": "recent"},
                        ),
                        MediaContentBlock(
                            media_kind="document",
                            media_type="application/pdf",
                            data={"id": "paper"},
                        ),
                    ),
                )
            ),
        ),
    )

    limits = get_model_output_limits("vision-lite", model_info=info)
    normalized = normalize_model_request_for_provider(request, model_info=info)

    assert get_context_window_for_model("vision-lite", model_info=info) == 300_000
    assert limits.default == 10_000
    assert limits.upper_limit == 20_000
    result = cast(ToolResultContentBlock, normalized.messages[1].message.content[0])
    assert thaw_json(result.content) == [
        {"type": "text", "text": TOOL_REFERENCES_REMOVED_PLACEHOLDER}
    ]
    image_replacement = cast(TextContentBlock, normalized.messages[1].message.content[1])
    retained_image = cast(MediaContentBlock, normalized.messages[1].message.content[2])
    document_replacement = cast(TextContentBlock, normalized.messages[1].message.content[3])
    assert EXCESS_MEDIA_REMOVED_PLACEHOLDER in image_replacement.text
    assert thaw_json(retained_image.data) == {"id": "recent"}
    assert document_replacement.text == UNSUPPORTED_DOCUMENT_REMOVED_PLACEHOLDER


def test_protocol_model_provider_uses_catalog_plus_model_overrides() -> None:
    catalog = ProviderModelCatalog(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        models=(
            ProviderModelEntry(
                provider_id="anthropic",
                protocol_id="anthropic_messages",
                model_id="sonnet",
                api_model_id="claude-sonnet-base",
                aliases=("default",),
                context_window=200_000,
            ),
        ),
    )
    provider = ProtocolModelProvider(
        adapter=AnthropicMessagesAdapter(),
        transport=FakeTransport(),
        catalogs=(catalog,),
        models=(
            ModelInfo(
                model="claude-sonnet-base",
                context_window=CONTEXT_1M_WINDOW_TOKENS,
                aliases=("project-default",),
            ),
        ),
    )

    assert provider.resolve_model("sonnet", ModelResolveContext()) == "claude-sonnet-base"
    assert provider.resolve_model("project-default[1m]", ModelResolveContext()) == (
        "claude-sonnet-base[1m]"
    )
    assert provider.model_info("default[1m]").context_window == CONTEXT_1M_WINDOW_TOKENS


def test_registry_from_catalogs_rejects_cross_provider_model_collisions() -> None:
    anthropic_catalog = ProviderModelCatalog(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        models=(
            ProviderModelEntry(
                provider_id="anthropic",
                protocol_id="anthropic_messages",
                model_id="shared",
            ),
        ),
    )
    openai_catalog = ProviderModelCatalog(
        provider_id="openai",
        protocol_id="openai_responses",
        models=(
            ProviderModelEntry(
                provider_id="openai",
                protocol_id="openai_responses",
                model_id="shared",
            ),
        ),
    )

    try:
        registry_from_catalogs(catalogs=(anthropic_catalog, openai_catalog))
    except ValueError as exc:
        assert "provider_id" in str(exc)
    else:
        raise AssertionError("Expected cross-provider catalog collision to fail")


def test_catalog_filters_disabled_entries_from_registry() -> None:
    catalog = ProviderModelCatalog(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        models=(
            ProviderModelEntry(
                provider_id="anthropic",
                protocol_id="anthropic_messages",
                model_id="disabled",
                api_model_id="disabled-api",
                aliases=("default",),
                enabled=False,
            ),
        ),
    )

    registry = catalog.registry()

    assert catalog.to_model_infos() == ()
    assert registry.resolve_model("disabled") == "disabled"
    assert registry.resolve_model("default") == "default"


def test_merge_model_info_can_explicitly_clear_capabilities() -> None:
    base = ProviderModelEntry(
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        model_id="vision",
        input_modalities=("text", "image/*"),
        output_modalities=("text",),
        max_media_items_per_request=3,
    ).to_model_info()
    override = ModelInfo(
        model="vision",
        provider_id="anthropic",
        protocol_id="anthropic_messages",
        capabilities=ModelCapabilities(),
        input_modalities=("text",),
        output_modalities=("text",),
        max_media_items_per_request=1,
    )

    default_merge = merge_model_info(base, override)
    replacing_merge = merge_model_info(base, override, replace_capabilities=True)

    assert default_merge.capabilities.supports_images is True
    assert replacing_merge.capabilities.supports_images is False
    assert replacing_merge.capabilities.supports_tool_references is False
