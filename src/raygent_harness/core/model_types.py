"""Provider-neutral model request, response, stream, and error types.

These Raygent-owned shapes: core code can depend on them without importing a provider
SDK, while provider adapters translate to and from concrete API payloads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

type JsonScalar = str | int | float | bool | None
type FrozenJson = JsonScalar | tuple[FrozenJson, ...] | Mapping[str, FrozenJson]

ModelRole = Literal["system", "user", "assistant", "tool"]

ProviderErrorKind = Literal[
    "context_overflow",
    "media_overflow",
    "max_output_tokens",
    "rate_limit",
    "server_overload",
    "transient",
    "auth_config",
    "model_fallback_triggered",
    "user_abort",
    "fatal_unknown",
]

ModelStatus = Literal["alpha", "beta", "deprecated", "active", "unknown"]

ModelStreamEventType = Literal[
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "provider_error",
    "streaming_transport_fallback_started",
    "streaming_transport_fallback_completed",
    "model_fallback_triggered",
]


def freeze_json(value: object) -> FrozenJson:
    """Recursively freeze JSON-like provider metadata.

    This protects the API-bound/observable split: callers can derive observable
    content without accidentally mutating the payload that will be replayed to a
    provider for prompt-cache stability.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Mapping):
        frozen = {
            str(key): freeze_json(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
        return MappingProxyType(frozen)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(freeze_json(item) for item in cast(Sequence[object], value))

    raise TypeError(f"Expected JSON-like provider value, got {type(value).__name__}")


def _empty_json_object() -> FrozenJson:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class TextContentBlock:
    text: str
    type: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True, slots=True)
class ToolUseContentBlock:
    id: str
    name: str
    input: FrozenJson = field(default_factory=_empty_json_object)
    provider_executed: bool = False
    provider_metadata: FrozenJson | None = None
    type: Literal["tool_use"] = field(default="tool_use", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", freeze_json(self.input))
        if self.provider_metadata is not None:
            object.__setattr__(
                self,
                "provider_metadata",
                freeze_json(self.provider_metadata),
            )


@dataclass(frozen=True, slots=True)
class ToolResultContentBlock:
    tool_use_id: str
    content: FrozenJson
    is_error: bool = False
    provider_metadata: FrozenJson | None = None
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", freeze_json(self.content))
        if self.provider_metadata is not None:
            object.__setattr__(
                self,
                "provider_metadata",
                freeze_json(self.provider_metadata),
            )


@dataclass(frozen=True, slots=True)
class ThinkingContentBlock:
    text: str
    signature: str | None = None
    redacted: bool = False
    provider_metadata: FrozenJson | None = None
    type: Literal["thinking"] = field(default="thinking", init=False)

    def __post_init__(self) -> None:
        if self.provider_metadata is not None:
            object.__setattr__(
                self,
                "provider_metadata",
                freeze_json(self.provider_metadata),
            )


@dataclass(frozen=True, slots=True)
class MediaContentBlock:
    media_kind: Literal["image", "document", "unknown_media"]
    media_type: str
    data: FrozenJson
    provider_metadata: FrozenJson | None = None
    type: Literal["media"] = field(default="media", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", freeze_json(self.data))
        if self.provider_metadata is not None:
            object.__setattr__(
                self,
                "provider_metadata",
                freeze_json(self.provider_metadata),
            )


@dataclass(frozen=True, slots=True)
class UnknownContentBlock:
    block_type: str
    payload: FrozenJson = field(default_factory=_empty_json_object)
    type: Literal["unknown"] = field(default="unknown", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_json(self.payload))


type ModelContentBlock = (
    TextContentBlock
    | ToolUseContentBlock
    | ToolResultContentBlock
    | ThinkingContentBlock
    | MediaContentBlock
    | UnknownContentBlock
)


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: ModelRole
    content: tuple[ModelContentBlock, ...]
    id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", tuple(self.content))


@dataclass(frozen=True, slots=True)
class ApiMessage:
    """Provider/API-bound message payload.

    Do not mutate or backfill this payload after request construction.
    Observable/transcript variants should be represented as `ObservableMessage`.
    """

    message: ModelMessage
    provider_payload: FrozenJson | None = None

    def __post_init__(self) -> None:
        if self.provider_payload is not None:
            object.__setattr__(self, "provider_payload", freeze_json(self.provider_payload))


@dataclass(frozen=True, slots=True)
class ObservableMessage:
    """Transcript/progress-visible message payload.

    This can include derived fields that should not flow back into the provider
    request payload.
    """

    message: ModelMessage
    provider_payload: FrozenJson | None = None

    def __post_init__(self) -> None:
        if self.provider_payload is not None:
            object.__setattr__(self, "provider_payload", freeze_json(self.provider_payload))


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int | None = None
    provider_metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_metadata", freeze_json(self.provider_metadata))

    @property
    def total_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def effective_total_tokens(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        return self.total_input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelToolUseBlock:
    id: str
    name: str
    input: FrozenJson
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", freeze_json(self.input))


@dataclass(frozen=True, slots=True)
class ModelToolSpec:
    """Provider-facing tool schema.

    The query loop decides which tools are model-visible and renders dynamic
    prompts before creating a `ModelRequest`. Providers translate this neutral
    schema to their own request format; they should not need executable `Tool`
    objects or permission-engine context to build the API payload.
    """

    name: str
    description: str
    input_schema: FrozenJson

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", freeze_json(self.input_schema))


@dataclass(frozen=True, slots=True)
class ModelSampling:
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))


@dataclass(frozen=True, slots=True)
class TaskBudgetInfo:
    total: int | None = None
    remaining: int | None = None


@dataclass(frozen=True, slots=True)
class CachePolicy:
    skip_cache_write: bool = False
    cache_scope: str | None = None


@dataclass(frozen=True, slots=True)
class PermissionContextSnapshot:
    """Provider-visible permission state snapshot.

    Reference passes a lazy `getToolPermissionContext()` into the model-call
    path. Raygent keeps this provider-neutral and immutable; chunk 3 will decide
    whether to snapshot once or expose a narrow accessor.
    """

    mode: str = "default"
    always_allow_rules: FrozenJson = field(default_factory=_empty_json_object)
    always_deny_rules: FrozenJson = field(default_factory=_empty_json_object)
    always_ask_rules: FrozenJson = field(default_factory=_empty_json_object)
    should_avoid_permission_prompts: bool = False
    is_bypass_permissions_mode_available: bool = False
    is_auto_mode_available: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "always_allow_rules", freeze_json(self.always_allow_rules))
        object.__setattr__(self, "always_deny_rules", freeze_json(self.always_deny_rules))
        object.__setattr__(self, "always_ask_rules", freeze_json(self.always_ask_rules))


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """Provider-visible active-agent metadata.

    This deliberately mirrors only the kernel-relevant AgentDefinition fields.
    Product/UI provenance can stay in adapter metadata.
    """

    agent_type: str
    description: str
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    permission_mode: str | None = None
    background: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "disallowed_tools", tuple(self.disallowed_tools))


@dataclass(frozen=True, slots=True)
class ModelBudgetSnapshot:
    """Metadata-only budget facts that governed one model request.

    This is not provider wire payload. It lets observability, tests, and future
    eval harnesses inspect the same effective model limits and token-count
    outcome used at the model-call boundary without exposing prompt content.
    """

    requested_model: str
    effective_model: str
    context_window: int
    default_max_output_tokens: int
    upper_max_output_tokens: int
    requested_max_tokens: int
    effective_max_tokens: int
    input_token_count: int | None = None
    provider_input_token_count: int | None = None
    fallback_input_token_count: int | None = None
    token_count_fallback_used: bool = False
    token_count_error_type: str | None = None


@dataclass(frozen=True, slots=True)
class MediaBudgetSnapshot:
    """Metadata-only media facts for one provider-bound request.

    This is intentionally separate from token accounting: media count/size
    failures need different recovery behavior from prompt-too-long failures.
    """

    max_media_items: int
    original_media_items: int
    retained_media_items: int
    stripped_media_items: int
    top_level_media_items: int = 0
    nested_media_items: int = 0
    mode: Literal["request_limit", "media_overflow_retry"] = "request_limit"


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    messages: tuple[ApiMessage, ...]
    system_prompt: str = ""
    tools: tuple[ModelToolSpec, ...] = ()
    sampling: ModelSampling = field(default_factory=lambda: ModelSampling(max_tokens=8192))
    fallback_model: str | None = None
    effort: str | int | None = None
    agent_id: str | None = None
    task_budget: TaskBudgetInfo | None = None
    abort_event: asyncio.Event | None = None
    query_source: str | None = None
    tool_choice: str | None = None
    max_output_tokens_override: int | None = None
    permission_context: PermissionContextSnapshot | None = None
    active_agents: tuple[AgentDescriptor, ...] = ()
    allowed_agent_types: tuple[str, ...] = ()
    mcp_tool_names: tuple[str, ...] = ()
    has_pending_mcp_servers: bool = False
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    budget: ModelBudgetSnapshot | None = None
    media_budget: MediaBudgetSnapshot | None = None
    provider_options: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "active_agents", tuple(self.active_agents))
        object.__setattr__(self, "allowed_agent_types", tuple(self.allowed_agent_types))
        object.__setattr__(self, "mcp_tool_names", tuple(self.mcp_tool_names))
        object.__setattr__(self, "provider_options", freeze_json(self.provider_options))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    api_message: ApiMessage
    observable_message: ObservableMessage
    tool_uses: tuple[ModelToolUseBlock, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    provider_request_id: str | None = None
    raw_metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_uses", tuple(self.tool_uses))
        object.__setattr__(self, "raw_metadata", freeze_json(self.raw_metadata))


@dataclass(frozen=True, slots=True)
class ModelApiErrorMessage:
    kind: ProviderErrorKind
    public_message: str
    api_message: ApiMessage
    observable_message: ObservableMessage
    raw_details: str | None = None
    actual_tokens: int | None = None
    limit_tokens: int | None = None
    retry_after_s: float | None = None
    status_code: int | None = None

    @property
    def token_gap(self) -> int | None:
        if self.actual_tokens is None or self.limit_tokens is None:
            return None
        gap = self.actual_tokens - self.limit_tokens
        return gap if gap > 0 else None


@dataclass(frozen=True, slots=True)
class ModelFallbackControl:
    """Query-loop model fallback signal.

    Distinct from streaming transport fallback, which is an intra-provider
    replacement path and may still produce a normal response.
    """

    original_model: str
    fallback_model: str
    reason: str


@dataclass(frozen=True, slots=True)
class StreamingTransportFallbackControl:
    """Intra-provider stream-to-non-stream fallback signal.

    This is intentionally not a model fallback. The provider may replace a
    failed streaming attempt with a non-streaming response for the same model,
    while model fallback still bubbles to the query loop for retry cleanup.
    """

    reason: str
    replacement_response: ModelResponse | None = None


@dataclass(frozen=True, slots=True)
class ProviderError:
    kind: ProviderErrorKind
    message: str
    raw_details: str | None = None
    actual_tokens: int | None = None
    limit_tokens: int | None = None
    retry_after_s: float | None = None
    status_code: int | None = None
    retryable: bool = False
    safe_to_fallback: bool = False
    api_error: ModelApiErrorMessage | None = None
    model_fallback: ModelFallbackControl | None = None


@dataclass(frozen=True, slots=True)
class StreamIdentity:
    message_id: str | None = None
    content_block_index: int | None = None
    provider_request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        if (
            self.message_id is None
            and self.provider_request_id is None
            and self.attempt_id is None
        ):
            raise ValueError(
                "StreamIdentity requires message_id, provider_request_id, or attempt_id"
            )


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    type: ModelStreamEventType
    identity: StreamIdentity
    block: ModelContentBlock | None = None
    delta: FrozenJson | None = None
    usage: Usage | None = None
    stop_reason: str | None = None
    provider_error: ProviderError | None = None
    streaming_transport_fallback: StreamingTransportFallbackControl | None = None
    model_fallback: ModelFallbackControl | None = None

    def __post_init__(self) -> None:
        if self.delta is not None:
            object.__setattr__(self, "delta", freeze_json(self.delta))

    @classmethod
    def message_start(
        cls,
        identity: StreamIdentity,
        *,
        usage: Usage | None = None,
    ) -> ModelStreamEvent:
        return cls(type="message_start", identity=identity, usage=usage)

    @classmethod
    def content_block_start(
        cls,
        identity: StreamIdentity,
        *,
        block: ModelContentBlock,
    ) -> ModelStreamEvent:
        return cls(type="content_block_start", identity=identity, block=block)

    @classmethod
    def content_block_delta(
        cls,
        identity: StreamIdentity,
        *,
        delta: FrozenJson,
    ) -> ModelStreamEvent:
        return cls(type="content_block_delta", identity=identity, delta=delta)

    @classmethod
    def content_block_stop(
        cls,
        identity: StreamIdentity,
    ) -> ModelStreamEvent:
        return cls(type="content_block_stop", identity=identity)

    @classmethod
    def message_delta(
        cls,
        identity: StreamIdentity,
        *,
        usage: Usage | None = None,
        stop_reason: str | None = None,
        delta: FrozenJson | None = None,
    ) -> ModelStreamEvent:
        payload: FrozenJson | None = delta
        if payload is None and stop_reason is not None:
            payload = {"stop_reason": stop_reason}
        return cls(
            type="message_delta",
            identity=identity,
            delta=payload,
            usage=usage,
            stop_reason=stop_reason,
        )

    @classmethod
    def message_stop(
        cls,
        identity: StreamIdentity,
        *,
        usage: Usage | None = None,
        stop_reason: str | None = None,
    ) -> ModelStreamEvent:
        return cls(
            type="message_stop",
            identity=identity,
            usage=usage,
            stop_reason=stop_reason,
        )

    @classmethod
    def provider_error_event(
        cls,
        identity: StreamIdentity,
        *,
        provider_error: ProviderError,
    ) -> ModelStreamEvent:
        return cls(
            type="provider_error",
            identity=identity,
            provider_error=provider_error,
        )

    @classmethod
    def streaming_transport_fallback_started(
        cls,
        identity: StreamIdentity,
        *,
        reason: str,
    ) -> ModelStreamEvent:
        return cls(
            type="streaming_transport_fallback_started",
            identity=identity,
            delta={"reason": reason},
            streaming_transport_fallback=StreamingTransportFallbackControl(
                reason=reason,
            ),
        )

    @classmethod
    def streaming_transport_fallback_completed(
        cls,
        identity: StreamIdentity,
        *,
        reason: str,
        replacement_response: ModelResponse | None = None,
    ) -> ModelStreamEvent:
        return cls(
            type="streaming_transport_fallback_completed",
            identity=identity,
            delta={"reason": reason},
            streaming_transport_fallback=StreamingTransportFallbackControl(
                reason=reason,
                replacement_response=replacement_response,
            ),
        )

    @classmethod
    def model_fallback_triggered(
        cls,
        identity: StreamIdentity,
        *,
        original_model: str,
        fallback_model: str,
        reason: str,
    ) -> ModelStreamEvent:
        return cls(
            type="model_fallback_triggered",
            identity=identity,
            delta={"reason": reason},
            model_fallback=ModelFallbackControl(
                original_model=original_model,
                fallback_model=fallback_model,
                reason=reason,
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_media: bool = False
    supports_images: bool = False
    supports_documents: bool = False
    supports_tool_references: bool = False


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model: str
    canonical_name: str | None = None
    provider_id: str | None = None
    protocol_id: str | None = None
    display_name: str | None = None
    status: ModelStatus | None = None
    context_window: int | None = None
    max_output_tokens_default: int | None = None
    max_output_tokens_upper_limit: int | None = None
    input_token_limit: int | None = None
    max_media_items_per_request: int | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    aliases: tuple[str, ...] = ()
    safe_metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_modalities", tuple(self.input_modalities))
        object.__setattr__(self, "output_modalities", tuple(self.output_modalities))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "safe_metadata", freeze_json(self.safe_metadata))


@dataclass(frozen=True, slots=True)
class ModelResolveContext:
    permission_mode: str | None = None
    query_source: str | None = None
    agent_id: str | None = None
    exceeds_200k_tokens: bool = False
    requested_context_window: int | None = None
    effort: str | int | None = None


@dataclass(frozen=True, slots=True)
class TokenCountRequest:
    model: str
    messages: tuple[ApiMessage, ...]
    system_prompt: str = ""
    tools: tuple[ModelToolSpec, ...] = ()
    thinking: FrozenJson | None = None
    effort: str | int | None = None
    media_context: FrozenJson | None = None
    provider_options: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if self.thinking is not None:
            object.__setattr__(self, "thinking", freeze_json(self.thinking))
        if self.media_context is not None:
            object.__setattr__(self, "media_context", freeze_json(self.media_context))
        object.__setattr__(self, "provider_options", freeze_json(self.provider_options))


@dataclass(frozen=True, slots=True)
class TokenCountResult:
    """Exact provider token-count result plus replay-safe metadata."""

    token_count: int
    provider_request_id: str | None = None
    safe_metadata: FrozenJson = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("Provider token count cannot be negative")
        object.__setattr__(self, "safe_metadata", freeze_json(self.safe_metadata))


def build_model_api_error_message(
    *,
    kind: ProviderErrorKind,
    public_message: str,
    raw_details: str | None = None,
    actual_tokens: int | None = None,
    limit_tokens: int | None = None,
    retry_after_s: float | None = None,
    status_code: int | None = None,
) -> ModelApiErrorMessage:
    content_block: dict[str, FrozenJson] = {"type": "text", "text": public_message}
    provider_payload: dict[str, FrozenJson] = {
        "role": "assistant",
        "content": (content_block,),
        "isApiErrorMessage": True,
        "apiError": kind,
        "error": kind,
    }
    if raw_details is not None:
        provider_payload["errorDetails"] = raw_details
    message = ModelMessage(
        role="assistant",
        content=(TextContentBlock(text=public_message),),
    )
    api_message = ApiMessage(message=message, provider_payload=provider_payload)
    observable_message = ObservableMessage(message=message, provider_payload=provider_payload)
    return ModelApiErrorMessage(
        kind=kind,
        public_message=public_message,
        api_message=api_message,
        observable_message=observable_message,
        raw_details=raw_details,
        actual_tokens=actual_tokens,
        limit_tokens=limit_tokens,
        retry_after_s=retry_after_s,
        status_code=status_code,
    )


__all__ = [
    "AgentDescriptor",
    "ApiMessage",
    "CachePolicy",
    "FrozenJson",
    "MediaContentBlock",
    "ModelApiErrorMessage",
    "ModelBudgetSnapshot",
    "ModelCapabilities",
    "ModelContentBlock",
    "ModelFallbackControl",
    "ModelInfo",
    "ModelMessage",
    "ModelRequest",
    "ModelResolveContext",
    "ModelResponse",
    "ModelRole",
    "ModelSampling",
    "ModelStatus",
    "ModelStreamEvent",
    "ModelStreamEventType",
    "ModelToolSpec",
    "ModelToolUseBlock",
    "ObservableMessage",
    "PermissionContextSnapshot",
    "ProviderError",
    "ProviderErrorKind",
    "StreamIdentity",
    "StreamingTransportFallbackControl",
    "TaskBudgetInfo",
    "TextContentBlock",
    "ThinkingContentBlock",
    "TokenCountRequest",
    "TokenCountResult",
    "ToolResultContentBlock",
    "ToolUseContentBlock",
    "UnknownContentBlock",
    "Usage",
    "build_model_api_error_message",
    "freeze_json",
]
