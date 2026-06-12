"""Provider-neutral model metadata, alias, and token-count helpers.

Model aliases, context windows, max-output limits, and token counting stay
behind provider seams. Providers own exact metadata/counting, while this module
supplies deterministic fallbacks and data-driven alias/window helpers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from raygent_harness.core.messages import MessageParam, api_message_from_message_param
from raygent_harness.core.model_types import (
    FrozenJson,
    ModelBudgetSnapshot,
    ModelInfo,
    ModelRequest,
    ModelResolveContext,
    ModelToolSpec,
    TokenCountRequest,
    TokenCountResult,
    freeze_json,
)

if TYPE_CHECKING:
    from raygent_harness.core.model_provider import ModelProvider
    from raygent_harness.core.observability import KernelEventBus, KernelEventContext


MODEL_CONTEXT_WINDOW_DEFAULT = 200_000
CONTEXT_1M_WINDOW_TOKENS = 1_000_000

MAX_OUTPUT_TOKENS_DEFAULT = 32_000
MAX_OUTPUT_TOKENS_UPPER_LIMIT = 64_000
CAPPED_DEFAULT_MAX_TOKENS = 8_000

_CONTEXT_WINDOW_SUFFIX_RE = re.compile(r"\[(?P<label>1m)\]\s*$", re.IGNORECASE)

TokenFallbackEstimator = Callable[[list[MessageParam]], int]


@dataclass(frozen=True, slots=True)
class ModelOutputLimits:
    """Default and upper max-output token limits for a model."""

    default: int = MAX_OUTPUT_TOKENS_DEFAULT
    upper_limit: int = MAX_OUTPUT_TOKENS_UPPER_LIMIT


@dataclass(frozen=True, slots=True)
class ModelNameParts:
    """A model name split into base name plus optional context-window suffix."""

    original: str
    base: str
    requested_context_window: int | None = None
    suffix: str | None = None

    @property
    def has_context_suffix(self) -> bool:
        return self.requested_context_window is not None


@dataclass(frozen=True, slots=True)
class TokenCountReport:
    """Provider token-count result plus deterministic fallback provenance."""

    token_count: int
    provider_token_count: int | None = None
    deterministic_token_count: int | None = None
    fallback_used: bool = False
    error_type: str | None = None
    provider_request_id: str | None = None
    provider_metadata: FrozenJson = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_metadata", freeze_json(self.provider_metadata))


class ModelRegistry:
    """Small data-driven registry for aliases and static model metadata.

    Concrete providers may ignore this and implement `resolve_model` /
    `model_info` directly. Embedders that need simple alias behavior without a
    custom provider can register `ModelInfo(aliases=(...))` entries here.
    """

    def __init__(self, models: Iterable[ModelInfo] = ()) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._aliases: dict[str, str] = {}
        for model in models:
            self.register(model)

    def register(self, model: ModelInfo) -> None:
        self._models[_key(model.model)] = model
        if model.canonical_name:
            self._models[_key(model.canonical_name)] = model
        for alias in model.aliases:
            self._aliases[_key(alias)] = model.model

    def resolve_model(
        self,
        requested: str,
        context: ModelResolveContext | None = None,
    ) -> str:
        _ = context
        parts = split_context_window_suffix(requested)
        resolved_base = self._aliases.get(_key(parts.base), parts.base)
        if parts.requested_context_window is not None:
            return with_context_window_suffix(resolved_base, parts.requested_context_window)
        return resolved_base

    def model_info(self, model: str) -> ModelInfo:
        parts = split_context_window_suffix(model)
        info_key = self._aliases.get(_key(parts.base), parts.base)
        registered = self._models.get(_key(info_key))
        info = registered if registered is not None else ModelInfo(model=parts.base)
        return apply_context_window_suffix(info, model)


def split_context_window_suffix(model: str) -> ModelNameParts:
    """Split a model string such as `sonnet[1m]`.

    Reference supports `[1m]` as an explicit context-window opt-in. Raygent keeps
    this generic and provider-neutral: suffix parsing is data-free, while
    providers decide whether a concrete backend can honor the model name.
    """

    original = model.strip()
    match = _CONTEXT_WINDOW_SUFFIX_RE.search(original)
    if match is None:
        return ModelNameParts(original=original, base=original)
    base = original[: match.start()].strip()
    return ModelNameParts(
        original=original,
        base=base,
        requested_context_window=CONTEXT_1M_WINDOW_TOKENS,
        suffix=match.group("label").lower(),
    )


def has_context_window_suffix(model: str) -> bool:
    return split_context_window_suffix(model).has_context_suffix


def with_context_window_suffix(model: str, context_window: int) -> str:
    """Append the normalized context-window suffix if it is recognized."""

    if context_window == CONTEXT_1M_WINDOW_TOKENS:
        parts = split_context_window_suffix(model)
        return f"{parts.base}[1m]"
    return model


def apply_context_window_suffix(info: ModelInfo, model: str) -> ModelInfo:
    """Return `info` with explicit model suffix metadata applied."""

    parts = split_context_window_suffix(model)
    if parts.requested_context_window is None:
        return info
    current = info.context_window or 0
    return replace(
        info,
        model=with_context_window_suffix(info.model, parts.requested_context_window),
        context_window=max(current, parts.requested_context_window),
    )


def model_info_with_fallback(
    model: str,
    *,
    provider: ModelProvider | None = None,
    registry: ModelRegistry | None = None,
) -> ModelInfo:
    """Return provider/registry metadata, falling back to empty `ModelInfo`.

    `model_info` is sync in the provider protocol so metadata can be consulted
    from threshold calculations without making every policy helper async.
    """

    if provider is not None:
        try:
            return apply_context_window_suffix(provider.model_info(model), model)
        except Exception:
            pass
    if registry is not None:
        return registry.model_info(model)
    fallback = ModelInfo(model=split_context_window_suffix(model).base)
    return apply_context_window_suffix(fallback, model)


def resolve_model_name(
    requested: str,
    *,
    context: ModelResolveContext | None = None,
    provider: ModelProvider | None = None,
    registry: ModelRegistry | None = None,
) -> str:
    """Resolve a requested model through provider, registry, or suffix fallback."""

    resolve_context = context or ModelResolveContext()
    if provider is not None:
        try:
            return provider.resolve_model(requested, resolve_context)
        except Exception:
            pass
    if registry is not None:
        return registry.resolve_model(requested, resolve_context)
    parts = split_context_window_suffix(requested)
    if parts.requested_context_window is not None:
        return with_context_window_suffix(parts.base, parts.requested_context_window)
    return parts.base


def carry_context_window_suffix(
    requested: str,
    current_model: str,
) -> str:
    """Carry the current recognized context-window suffix onto an override.

    Reference Skill model overrides preserve a parent `[1m]` opt-in unless the
    skill model explicitly requested its own window. Keep this as a pure string
    helper so call sites can apply the cache-sensitive suffix before provider
    alias resolution.
    """

    requested_parts = split_context_window_suffix(requested)
    if requested_parts.requested_context_window is not None:
        return requested_parts.original

    current_parts = split_context_window_suffix(current_model)
    if current_parts.requested_context_window is None:
        return requested_parts.base
    return with_context_window_suffix(
        requested_parts.base,
        current_parts.requested_context_window,
    )


def resolve_skill_model_override(
    skill_model: str,
    current_model: str,
    *,
    context: ModelResolveContext | None = None,
    provider: ModelProvider | None = None,
    registry: ModelRegistry | None = None,
) -> str:
    """Resolve a Skill model override with suffix carry and alias handling.

    Policy:
    - `inherit` means keep the current model.
    - an explicit skill suffix wins.
    - otherwise a recognized current-model suffix is carried onto the skill
      model before provider/registry alias resolution.
    - provider resolution wins, then registry, then suffix fallback.
    """

    requested = skill_model.strip()
    if requested == "" or requested == "inherit":
        return resolve_model_name(
            current_model,
            context=context,
            provider=provider,
            registry=registry,
        )

    requested_parts = split_context_window_suffix(requested)
    current_parts = split_context_window_suffix(current_model)
    if (
        requested_parts.requested_context_window is not None
        or current_parts.requested_context_window is None
    ):
        return resolve_model_name(
            requested,
            context=context,
            provider=provider,
            registry=registry,
        )

    resolved_base = resolve_model_name(
        requested_parts.base,
        context=context,
        provider=provider,
        registry=registry,
    )
    if not _model_supports_context_window(
        resolved_base,
        current_parts.requested_context_window,
        provider=provider,
        registry=registry,
    ):
        return resolved_base
    return resolve_model_name(
        with_context_window_suffix(resolved_base, current_parts.requested_context_window),
        context=context,
        provider=provider,
        registry=registry,
    )


def _model_supports_context_window(
    model: str,
    context_window: int | None,
    *,
    provider: ModelProvider | None,
    registry: ModelRegistry | None,
) -> bool:
    if context_window is None:
        return False
    info = model_info_with_fallback(
        split_context_window_suffix(model).base,
        provider=provider,
        registry=registry,
    )
    return (info.context_window or 0) >= context_window


def get_context_window_for_model(
    model: str,
    *,
    model_info: ModelInfo | None = None,
    default: int = MODEL_CONTEXT_WINDOW_DEFAULT,
) -> int:
    """Return model context window using metadata, suffixes, then fallback."""

    info = apply_context_window_suffix(model_info, model) if model_info is not None else None
    if info is not None and info.context_window is not None and info.context_window > 0:
        return info.context_window
    parts = split_context_window_suffix(model)
    if parts.requested_context_window is not None:
        return parts.requested_context_window
    return default


def get_model_output_limits(
    model: str,
    *,
    model_info: ModelInfo | None = None,
    cap_default: bool = False,
    env: Mapping[str, str] | None = None,
) -> ModelOutputLimits:
    """Return model max-output defaults and upper limits.

    The optional env override mirrors the reference's bounded max-output env
    override, but stays generic: callers pass an explicit env mapping rather
    than core reading product globals unconditionally.
    """

    _ = model
    default = (
        model_info.max_output_tokens_default
        if model_info is not None and model_info.max_output_tokens_default is not None
        else MAX_OUTPUT_TOKENS_DEFAULT
    )
    upper = (
        model_info.max_output_tokens_upper_limit
        if model_info is not None and model_info.max_output_tokens_upper_limit is not None
        else MAX_OUTPUT_TOKENS_UPPER_LIMIT
    )
    if upper <= 0:
        upper = MAX_OUTPUT_TOKENS_UPPER_LIMIT
    default = max(1, min(default, upper))
    if cap_default:
        default = min(default, CAPPED_DEFAULT_MAX_TOKENS)

    raw_override = env.get("RAYGENT_MAX_OUTPUT_TOKENS") if env is not None else None
    if raw_override:
        try:
            parsed = int(raw_override)
        except ValueError:
            parsed = default
        if parsed > 0:
            default = min(parsed, upper)

    return ModelOutputLimits(default=default, upper_limit=upper)


async def count_message_tokens(
    *,
    provider: ModelProvider,
    model: str,
    messages: list[MessageParam],
    tools: tuple[ModelToolSpec, ...] = (),
    thinking: FrozenJson | None = None,
    effort: str | int | None = None,
    system_prompt: str = "",
    media_context: FrozenJson | None = None,
    provider_options: FrozenJson | None = None,
    fallback_estimator: TokenFallbackEstimator,
    observability: KernelEventBus | None = None,
    observability_context: KernelEventContext | None = None,
) -> int:
    """Ask the provider for token count, falling back to deterministic estimate."""

    report = await count_message_tokens_report(
        provider=provider,
        model=model,
        messages=messages,
        tools=tools,
        thinking=thinking,
        effort=effort,
        system_prompt=system_prompt,
        media_context=media_context,
        provider_options=provider_options,
        fallback_estimator=fallback_estimator,
        observability=observability,
        observability_context=observability_context,
    )
    return report.token_count


async def count_message_tokens_report(
    *,
    provider: ModelProvider,
    model: str,
    messages: list[MessageParam],
    tools: tuple[ModelToolSpec, ...] = (),
    thinking: FrozenJson | None = None,
    effort: str | int | None = None,
    system_prompt: str = "",
    media_context: FrozenJson | None = None,
    provider_options: FrozenJson | None = None,
    fallback_estimator: TokenFallbackEstimator,
    observability: KernelEventBus | None = None,
    observability_context: KernelEventContext | None = None,
) -> TokenCountReport:
    """Ask the provider for token count and keep fallback provenance."""

    request = TokenCountRequest(
        model=model,
        messages=tuple(api_message_from_message_param(message) for message in messages),
        system_prompt=system_prompt,
        tools=tools,
        thinking=thinking,
        effort=effort,
        media_context=media_context,
        provider_options={} if provider_options is None else provider_options,
    )
    if observability is not None:
        observability.emit(
            "model.token_count.started",
            context=observability_context,
            data={
                "model": model,
                "message_count": len(messages),
                "tool_count": len(tools),
                "system_prompt_char_count": len(system_prompt),
                "has_thinking": thinking is not None,
                "has_media_context": media_context is not None,
                "effort": effort,
            },
        )
    try:
        provider_result = _normalize_token_count_result(await provider.count_tokens(request))
        count = provider_result.token_count
        if observability is not None:
            data: dict[str, object] = {
                "model": model,
                "token_count": count,
                "fallback_used": False,
            }
            _set_if_not_none(data, "provider_request_id", provider_result.provider_request_id)
            metadata = _metadata_dict(provider_result.safe_metadata)
            if metadata:
                data["provider_metadata"] = metadata
            observability.emit(
                "model.token_count.completed",
                context=observability_context,
                data=data,
            )
        return TokenCountReport(
            token_count=count,
            provider_token_count=count,
            fallback_used=False,
            provider_request_id=provider_result.provider_request_id,
            provider_metadata=provider_result.safe_metadata,
        )
    except Exception as exc:
        count = max(0, fallback_estimator(messages))
        if observability is not None:
            observability.emit(
                "model.token_count.completed",
                context=observability_context,
                data={
                    "model": model,
                    "token_count": count,
                    "fallback_used": True,
                    "error_type": type(exc).__name__,
                },
            )
        return TokenCountReport(
            token_count=count,
            deterministic_token_count=count,
            fallback_used=True,
            error_type=type(exc).__name__,
        )


async def build_model_budget_snapshot(
    *,
    provider: ModelProvider,
    requested_model: str,
    request: ModelRequest,
    model_info: ModelInfo,
    requested_max_tokens: int,
    messages: list[MessageParam],
    fallback_estimator: TokenFallbackEstimator,
    observability: KernelEventBus | None = None,
    observability_context: KernelEventContext | None = None,
) -> ModelBudgetSnapshot:
    """Build metadata-only model budget facts for one API-bound request."""

    limits = get_model_output_limits(request.model, model_info=model_info)
    token_report = await count_message_tokens_report(
        provider=provider,
        model=request.model,
        messages=messages,
        tools=request.tools,
        system_prompt=request.system_prompt,
        effort=request.effort,
        provider_options=request.provider_options,
        fallback_estimator=fallback_estimator,
        observability=observability,
        observability_context=observability_context,
    )
    return ModelBudgetSnapshot(
        requested_model=requested_model,
        effective_model=request.model,
        context_window=get_context_window_for_model(
            request.model,
            model_info=model_info,
        ),
        default_max_output_tokens=limits.default,
        upper_max_output_tokens=limits.upper_limit,
        requested_max_tokens=requested_max_tokens,
        effective_max_tokens=request.sampling.max_tokens,
        input_token_count=token_report.token_count,
        provider_input_token_count=token_report.provider_token_count,
        fallback_input_token_count=token_report.deterministic_token_count,
        token_count_fallback_used=token_report.fallback_used,
        token_count_error_type=token_report.error_type,
    )


def _normalize_token_count_result(result: int | TokenCountResult) -> TokenCountResult:
    if isinstance(result, TokenCountResult):
        return result
    return TokenCountResult(token_count=result)


def _metadata_dict(value: FrozenJson) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value.items())
    return {}


def _set_if_not_none(target: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        target[key] = value


def _key(value: str) -> str:
    return value.strip().lower()


__all__ = [
    "CAPPED_DEFAULT_MAX_TOKENS",
    "CONTEXT_1M_WINDOW_TOKENS",
    "MAX_OUTPUT_TOKENS_DEFAULT",
    "MAX_OUTPUT_TOKENS_UPPER_LIMIT",
    "MODEL_CONTEXT_WINDOW_DEFAULT",
    "ModelNameParts",
    "ModelOutputLimits",
    "ModelRegistry",
    "TokenCountReport",
    "TokenFallbackEstimator",
    "apply_context_window_suffix",
    "build_model_budget_snapshot",
    "carry_context_window_suffix",
    "count_message_tokens",
    "count_message_tokens_report",
    "get_context_window_for_model",
    "get_model_output_limits",
    "has_context_window_suffix",
    "model_info_with_fallback",
    "resolve_model_name",
    "resolve_skill_model_override",
    "split_context_window_suffix",
    "with_context_window_suffix",
]
