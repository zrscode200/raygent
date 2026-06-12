"""Provider/model catalog helpers for runtime model providers.

The catalog is adapter-runtime data, not a live provider registry. It converts
provider/deployment facts into Raygent-owned `ModelInfo` objects so the existing
`ModelRegistry` remains the single resolver used by `ProtocolModelProvider`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from raygent_harness.core.model_registry import ModelRegistry
from raygent_harness.core.model_types import (
    FrozenJson,
    ModelCapabilities,
    ModelInfo,
    ModelStatus,
    freeze_json,
)


@dataclass(frozen=True, slots=True)
class ProviderModelEntry:
    """One provider model row before conversion to Raygent `ModelInfo`."""

    provider_id: str
    protocol_id: str
    model_id: str
    api_model_id: str | None = None
    display_name: str | None = None
    canonical_name: str | None = None
    aliases: tuple[str, ...] = ()
    status: ModelStatus = "active"
    enabled: bool = True
    context_window: int | None = None
    input_token_limit: int | None = None
    max_output_tokens_default: int | None = None
    max_output_tokens_upper_limit: int | None = None
    max_media_items_per_request: int | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    capabilities: ModelCapabilities | None = None
    safe_metadata: FrozenJson = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "input_modalities", tuple(self.input_modalities))
        object.__setattr__(self, "output_modalities", tuple(self.output_modalities))
        object.__setattr__(self, "safe_metadata", freeze_json(self.safe_metadata))

    @property
    def provider_model_id(self) -> str:
        """Concrete id sent to provider APIs."""

        return self.api_model_id or self.model_id

    def to_model_info(self) -> ModelInfo:
        aliases = _aliases_for_entry(self)
        return ModelInfo(
            model=self.provider_model_id,
            canonical_name=self.canonical_name or self.provider_model_id,
            provider_id=self.provider_id,
            protocol_id=self.protocol_id,
            display_name=self.display_name,
            status=self.status if self.enabled else "deprecated",
            context_window=self.context_window,
            max_output_tokens_default=self.max_output_tokens_default,
            max_output_tokens_upper_limit=self.max_output_tokens_upper_limit,
            input_token_limit=self.input_token_limit,
            max_media_items_per_request=self.max_media_items_per_request,
            input_modalities=self.input_modalities,
            output_modalities=self.output_modalities,
            capabilities=self.capabilities or capabilities_from_modalities(
                input_modalities=self.input_modalities,
                output_modalities=self.output_modalities,
            ),
            aliases=aliases,
            safe_metadata=_entry_metadata(self),
        )


@dataclass(frozen=True, slots=True)
class ProviderModelCatalog:
    """A replay-safe provider/model catalog for one or more deployments."""

    provider_id: str
    protocol_id: str
    models: tuple[ProviderModelEntry, ...]
    safe_metadata: FrozenJson = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "safe_metadata", freeze_json(self.safe_metadata))
        for entry in self.models:
            if entry.provider_id != self.provider_id:
                raise ValueError("Catalog entry provider_id must match catalog provider_id")
            if entry.protocol_id != self.protocol_id:
                raise ValueError("Catalog entry protocol_id must match catalog protocol_id")

    def to_model_infos(self) -> tuple[ModelInfo, ...]:
        return tuple(entry.to_model_info() for entry in self.models if entry.enabled)

    def registry(self) -> ModelRegistry:
        return ModelRegistry(self.to_model_infos())


def capabilities_from_modalities(
    *,
    input_modalities: Sequence[str] = (),
    output_modalities: Sequence[str] = (),
    supports_streaming: bool = True,
    supports_tools: bool = True,
    supports_thinking: bool = False,
    supports_effort: bool = False,
    supports_tool_references: bool = False,
) -> ModelCapabilities:
    """Derive Raygent capability booleans from generic modality strings."""

    input_set = {_normalize_modality(item) for item in input_modalities}
    output_set = {_normalize_modality(item) for item in output_modalities}
    supports_images = _matches_any(input_set, ("image", "image/*"))
    supports_documents = _matches_any(
        input_set,
        ("document", "application/pdf", "text/*", "file"),
    )
    supports_media = bool(input_set - {"text", "text/*"})
    return ModelCapabilities(
        supports_streaming=supports_streaming,
        supports_tools=supports_tools,
        supports_thinking=supports_thinking or _matches_any(output_set, ("thinking",)),
        supports_effort=supports_effort,
        supports_media=supports_media,
        supports_images=supports_images,
        supports_documents=supports_documents,
        supports_tool_references=supports_tool_references,
    )


def merge_model_info(
    base: ModelInfo,
    override: ModelInfo,
    *,
    replace_capabilities: bool = False,
) -> ModelInfo:
    """Return `base` with explicit facts from `override` applied.

    This is intended for combining sample catalog metadata with application
    overrides. The override model id wins; aliases and safe metadata are merged.
    By default a default-valued `ModelCapabilities()` override is treated as
    unset for backwards-compatible lightweight overrides. Set
    `replace_capabilities=True` when the override intentionally clears catalog
    capabilities.
    """

    return replace(
        base,
        model=override.model or base.model,
        canonical_name=override.canonical_name or base.canonical_name,
        provider_id=override.provider_id or base.provider_id,
        protocol_id=override.protocol_id or base.protocol_id,
        display_name=override.display_name or base.display_name,
        status=override.status or base.status,
        context_window=_coalesce_positive(override.context_window, base.context_window),
        max_output_tokens_default=_coalesce_positive(
            override.max_output_tokens_default,
            base.max_output_tokens_default,
        ),
        max_output_tokens_upper_limit=_coalesce_positive(
            override.max_output_tokens_upper_limit,
            base.max_output_tokens_upper_limit,
        ),
        input_token_limit=_coalesce_positive(
            override.input_token_limit,
            base.input_token_limit,
        ),
        max_media_items_per_request=_coalesce_positive(
            override.max_media_items_per_request,
            base.max_media_items_per_request,
        ),
        input_modalities=override.input_modalities or base.input_modalities,
        output_modalities=override.output_modalities or base.output_modalities,
        capabilities=(
            override.capabilities
            if replace_capabilities or override.capabilities != ModelCapabilities()
            else base.capabilities
        ),
        aliases=_merge_strings(base.aliases, override.aliases),
        safe_metadata=_merge_metadata(base.safe_metadata, override.safe_metadata),
    )


def merge_model_infos(
    models: Iterable[ModelInfo],
    *,
    replace_capabilities: bool = False,
) -> tuple[ModelInfo, ...]:
    """Merge model infos by case-insensitive model id in input order."""

    merged: dict[str, ModelInfo] = {}
    order: list[str] = []
    for model in models:
        key = model.model.strip().lower()
        if key not in merged:
            merged[key] = model
            order.append(key)
            continue
        _ensure_merge_compatible(merged[key], model)
        merged[key] = merge_model_info(
            merged[key],
            model,
            replace_capabilities=replace_capabilities,
        )
    return tuple(merged[key] for key in order)


def registry_from_catalogs(
    catalogs: Iterable[ProviderModelCatalog] = (),
    model_infos: Iterable[ModelInfo] = (),
    *,
    replace_capabilities: bool = False,
) -> ModelRegistry:
    """Build a `ModelRegistry` from catalogs plus direct model overrides."""

    infos: list[ModelInfo] = []
    for catalog in catalogs:
        infos.extend(catalog.to_model_infos())
    infos.extend(model_infos)
    return ModelRegistry(
        merge_model_infos(infos, replace_capabilities=replace_capabilities)
    )


def _aliases_for_entry(entry: ProviderModelEntry) -> tuple[str, ...]:
    aliases = list(entry.aliases)
    if entry.api_model_id is not None and entry.model_id != entry.api_model_id:
        aliases.insert(0, entry.model_id)
    return _merge_strings((), aliases)


def _entry_metadata(entry: ProviderModelEntry) -> FrozenJson:
    raw = _metadata_mapping(entry.safe_metadata)
    raw.update(
        {
            "provider_id": entry.provider_id,
            "protocol_id": entry.protocol_id,
            "model_id": entry.model_id,
            "api_model_id": entry.provider_model_id,
            "enabled": entry.enabled,
        }
    )
    return freeze_json(raw)


def _merge_strings(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in (*left, *right):
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return tuple(merged)


def _merge_metadata(left: FrozenJson, right: FrozenJson) -> FrozenJson:
    merged = _metadata_mapping(left)
    merged.update(_metadata_mapping(right))
    return freeze_json(merged)


def _ensure_merge_compatible(left: ModelInfo, right: ModelInfo) -> None:
    if not _same_optional_id(left.provider_id, right.provider_id):
        raise ValueError(
            "Cannot merge model infos with different provider_id values "
            f"for model {left.model!r}"
        )
    if not _same_optional_id(left.protocol_id, right.protocol_id):
        raise ValueError(
            "Cannot merge model infos with different protocol_id values "
            f"for model {left.model!r}"
        )


def _same_optional_id(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right


def _metadata_mapping(value: FrozenJson) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in cast(Mapping[object, object], value).items()}
    return {}


def _normalize_modality(value: str) -> str:
    return value.strip().lower()


def _matches_any(values: set[str], needles: Sequence[str]) -> bool:
    return any(needle in values for needle in needles)


def _coalesce_positive(primary: int | None, fallback: int | None) -> int | None:
    if primary is not None and primary > 0:
        return primary
    return fallback


__all__ = [
    "ProviderModelCatalog",
    "ProviderModelEntry",
    "capabilities_from_modalities",
    "merge_model_info",
    "merge_model_infos",
    "registry_from_catalogs",
]
