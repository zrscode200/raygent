"""Provider-neutral stream event assembly helpers.

The query loop consumes normalized `ModelStreamEvent` sequences when streaming
tool execution is enabled, and adapter tests assemble finite event fixtures
through the same path. This module validates and assembles Raygent-owned stream
events without depending on a vendor SDK stream shape.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

from raygent_harness.core.model_types import (
    ApiMessage,
    FrozenJson,
    ModelContentBlock,
    ModelFallbackControl,
    ModelMessage,
    ModelResponse,
    ModelStreamEvent,
    ModelToolUseBlock,
    ObservableMessage,
    ProviderError,
    StreamIdentity,
    StreamingTransportFallbackControl,
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
    Usage,
    freeze_json,
)


class ModelStreamAssemblyError(RuntimeError):
    """Raised when a provider emits an invalid normalized stream sequence."""


@dataclass(frozen=True, slots=True)
class ModelStreamUpdate:
    """Incremental result returned after applying one stream event."""

    event: ModelStreamEvent
    completed_block: ModelContentBlock | None = None
    response: ModelResponse | None = None
    provider_error: ProviderError | None = None
    streaming_transport_fallback: StreamingTransportFallbackControl | None = None
    model_fallback: ModelFallbackControl | None = None


@dataclass(slots=True)
class _BlockBuilder:
    index: int
    start_block: ModelContentBlock
    text: str = ""
    thinking: str = ""
    signature: str | None = None
    redacted: bool = False
    provider_metadata: FrozenJson | None = None
    input_json_parts: list[str] = field(default_factory=list[str])

    @classmethod
    def from_start(cls, index: int, block: ModelContentBlock) -> _BlockBuilder:
        signature = block.signature if isinstance(block, ThinkingContentBlock) else None
        redacted = block.redacted if isinstance(block, ThinkingContentBlock) else False
        provider_metadata = (
            block.provider_metadata
            if isinstance(block, ThinkingContentBlock | ToolUseContentBlock)
            else None
        )
        # The reference ignores text/thinking values present on start events and
        # accumulates content from deltas, avoiding duplicate provider chunks.
        return cls(
            index=index,
            start_block=block,
            signature=signature,
            redacted=redacted,
            provider_metadata=provider_metadata,
        )

    def apply_delta(self, delta: Mapping[str, object]) -> None:
        provider_metadata = delta.get("provider_metadata")
        if provider_metadata is not None:
            self.provider_metadata = _merge_metadata(
                self.provider_metadata or {},
                freeze_json(provider_metadata),
            )
        redacted = delta.get("redacted")
        if isinstance(self.start_block, ThinkingContentBlock) and isinstance(redacted, bool):
            self.redacted = redacted
        delta_type = delta.get("type")
        if delta_type == "text_delta" or "text" in delta:
            if not isinstance(self.start_block, TextContentBlock):
                raise ModelStreamAssemblyError(
                    f"text delta cannot be applied to {self.start_block.type} block"
                )
            self.text += str(delta.get("text", ""))
            return
        if delta_type == "thinking_delta" or "thinking" in delta:
            if not isinstance(self.start_block, ThinkingContentBlock):
                raise ModelStreamAssemblyError(
                    f"thinking delta cannot be applied to {self.start_block.type} block"
                )
            self.thinking += str(delta.get("thinking", ""))
            return
        if delta_type == "signature_delta" or "signature" in delta:
            if not isinstance(self.start_block, ThinkingContentBlock):
                raise ModelStreamAssemblyError(
                    f"signature delta cannot be applied to {self.start_block.type} block"
                )
            signature = delta.get("signature")
            self.signature = str(signature) if signature is not None else ""
            return
        if delta_type == "input_json_delta" or "partial_json" in delta:
            if not isinstance(self.start_block, ToolUseContentBlock):
                raise ModelStreamAssemblyError(
                    f"input JSON delta cannot be applied to {self.start_block.type} block"
                )
            self.input_json_parts.append(str(delta.get("partial_json", "")))
            return

    def build(self) -> ModelContentBlock:
        if isinstance(self.start_block, TextContentBlock):
            return TextContentBlock(text=self.text)
        if isinstance(self.start_block, ThinkingContentBlock):
            return ThinkingContentBlock(
                text=self.thinking,
                signature=self.signature,
                redacted=self.redacted,
                provider_metadata=self.provider_metadata,
            )
        if isinstance(self.start_block, ToolUseContentBlock):
            return ToolUseContentBlock(
                id=self.start_block.id,
                name=self.start_block.name,
                input=self._tool_input(),
                provider_executed=self.start_block.provider_executed,
                provider_metadata=self.provider_metadata,
            )
        return self.start_block

    def _tool_input(self) -> FrozenJson:
        if not self.input_json_parts:
            if isinstance(self.start_block, ToolUseContentBlock):
                return self.start_block.input
            return {}
        raw = "".join(self.input_json_parts)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return freeze_json(parsed)


class ModelStreamAssembler:
    """Accumulates normalized model stream events into a final response.

    The shape mirrors the reference provider path: message_start initializes
    per-message usage, content_block_delta mutates an internal block builder,
    content_block_stop finalizes a block, and message_delta/message_stop supply
    final usage and stop reason. Transport fallback can replace the streaming
    attempt with a non-streaming response; model fallback remains a separate
    control signal for the query loop.
    """

    def __init__(self) -> None:
        self._message_started = False
        self._message_stopped = False
        self._message_identity: StreamIdentity | None = None
        self._provider_request_id: str | None = None
        self._attempt_id: str | None = None
        self._builders: dict[int, _BlockBuilder] = {}
        self._completed_blocks: dict[int, ModelContentBlock] = {}
        self._usage = Usage()
        self._stop_reason: str | None = None
        self._provider_error: ProviderError | None = None
        self._streaming_transport_fallback: StreamingTransportFallbackControl | None = None
        self._model_fallback: ModelFallbackControl | None = None
        self._replacement_response: ModelResponse | None = None

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def provider_error(self) -> ProviderError | None:
        return self._provider_error

    @property
    def streaming_transport_fallback(self) -> StreamingTransportFallbackControl | None:
        return self._streaming_transport_fallback

    @property
    def model_fallback(self) -> ModelFallbackControl | None:
        return self._model_fallback

    def apply(self, event: ModelStreamEvent) -> ModelStreamUpdate:
        if event.type == "message_start":
            self._message_started = True
            self._message_identity = event.identity
            self._provider_request_id = event.identity.provider_request_id
            self._attempt_id = event.identity.attempt_id
            self._merge_usage(event.usage)
            return ModelStreamUpdate(event=event)

        if event.type == "content_block_start":
            self._require_message_started(event.type)
            self._validate_message_identity(event.type, event.identity)
            index = _content_block_index(event.identity)
            if event.block is None:
                raise ModelStreamAssemblyError("content_block_start requires a block")
            self._builders[index] = _BlockBuilder.from_start(index, event.block)
            return ModelStreamUpdate(event=event)

        if event.type == "content_block_delta":
            self._require_message_started(event.type)
            self._validate_message_identity(event.type, event.identity)
            index = _content_block_index(event.identity)
            builder = self._builders.get(index)
            if builder is None:
                raise ModelStreamAssemblyError(
                    f"content_block_delta for unopened block index {index}"
                )
            builder.apply_delta(_delta_mapping(event))
            return ModelStreamUpdate(event=event)

        if event.type == "content_block_stop":
            self._require_message_started(event.type)
            self._validate_message_identity(event.type, event.identity)
            index = _content_block_index(event.identity)
            builder = self._builders.get(index)
            if builder is None:
                raise ModelStreamAssemblyError(
                    f"content_block_stop for unopened block index {index}"
                )
            block = builder.build()
            self._completed_blocks[index] = block
            return ModelStreamUpdate(event=event, completed_block=block)

        if event.type == "message_delta":
            self._require_message_started(event.type)
            self._validate_message_identity(event.type, event.identity)
            self._merge_usage(event.usage)
            self._update_stop_reason(event)
            return ModelStreamUpdate(event=event)

        if event.type == "message_stop":
            self._require_message_started(event.type)
            self._validate_message_identity(event.type, event.identity)
            self._merge_usage(event.usage)
            self._update_stop_reason(event)
            self._message_stopped = True
            response = self.response()
            return ModelStreamUpdate(event=event, response=response)

        if event.type == "provider_error":
            if event.provider_error is None:
                raise ModelStreamAssemblyError("provider_error event requires provider_error")
            self._provider_error = event.provider_error
            return ModelStreamUpdate(event=event, provider_error=event.provider_error)

        if event.type == "streaming_transport_fallback_started":
            self._streaming_transport_fallback = event.streaming_transport_fallback
            self._discard_partial_message()
            return ModelStreamUpdate(
                event=event,
                streaming_transport_fallback=event.streaming_transport_fallback,
            )

        if event.type == "streaming_transport_fallback_completed":
            self._streaming_transport_fallback = event.streaming_transport_fallback
            if event.streaming_transport_fallback is not None:
                self._replacement_response = (
                    event.streaming_transport_fallback.replacement_response
                )
            return ModelStreamUpdate(
                event=event,
                response=self._replacement_response,
                streaming_transport_fallback=event.streaming_transport_fallback,
            )

        if event.type == "model_fallback_triggered":
            if event.model_fallback is None:
                raise ModelStreamAssemblyError(
                    "model_fallback_triggered event requires model_fallback"
                )
            self._model_fallback = event.model_fallback
            return ModelStreamUpdate(event=event, model_fallback=event.model_fallback)

        raise ModelStreamAssemblyError(f"Unsupported stream event type: {event.type}")

    def response(self) -> ModelResponse:
        if self._replacement_response is not None:
            return self._replacement_response
        if self._provider_error is not None:
            raise ModelStreamAssemblyError("Cannot build response after provider_error")
        if self._model_fallback is not None:
            raise ModelStreamAssemblyError("Cannot build response after model fallback")
        if not self._message_started or not self._message_stopped:
            raise ModelStreamAssemblyError("Cannot build response before message_stop")

        content = tuple(
            self._completed_blocks[index] for index in sorted(self._completed_blocks)
        )
        message = ModelMessage(
            role="assistant",
            content=content,
            id=self._message_identity.message_id if self._message_identity else None,
        )
        api_message = ApiMessage(message=message)
        observable_message = ObservableMessage(message=message)
        return ModelResponse(
            api_message=api_message,
            observable_message=observable_message,
            tool_uses=tuple(_tool_uses_from_blocks(content)),
            usage=self._usage,
            stop_reason=self._stop_reason,
            provider_request_id=self._provider_request_id,
            raw_metadata={"attempt_id": self._attempt_id} if self._attempt_id else {},
        )

    def _merge_usage(self, update: Usage | None) -> None:
        if update is None:
            return
        self._usage = Usage(
            input_tokens=update.input_tokens or self._usage.input_tokens,
            output_tokens=update.output_tokens or self._usage.output_tokens,
            cache_creation_input_tokens=(
                update.cache_creation_input_tokens
                or self._usage.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                update.cache_read_input_tokens or self._usage.cache_read_input_tokens
            ),
            reasoning_tokens=update.reasoning_tokens or self._usage.reasoning_tokens,
            total_tokens=(
                update.total_tokens
                if update.total_tokens is not None
                else self._usage.total_tokens
            ),
            provider_metadata=_merge_metadata(
                self._usage.provider_metadata,
                update.provider_metadata,
            ),
        )

    def _update_stop_reason(self, event: ModelStreamEvent) -> None:
        if event.stop_reason is not None:
            self._stop_reason = event.stop_reason
            return
        delta = event.delta
        if isinstance(delta, Mapping):
            reason = delta.get("stop_reason")
            if isinstance(reason, str):
                self._stop_reason = reason

    def _require_message_started(self, event_type: str) -> None:
        if not self._message_started:
            raise ModelStreamAssemblyError(
                f"{event_type} cannot be applied before message_start"
            )

    def _validate_message_identity(
        self,
        event_type: str,
        identity: StreamIdentity,
    ) -> None:
        current = self._message_identity
        if current is None:
            self._message_identity = identity
            return

        _require_matching_identity_field(
            event_type,
            field_name="message_id",
            expected=current.message_id,
            actual=identity.message_id,
        )
        _require_matching_identity_field(
            event_type,
            field_name="provider_request_id",
            expected=current.provider_request_id,
            actual=identity.provider_request_id,
        )
        _require_matching_identity_field(
            event_type,
            field_name="attempt_id",
            expected=current.attempt_id,
            actual=identity.attempt_id,
        )
        self._message_identity = StreamIdentity(
            message_id=current.message_id or identity.message_id,
            provider_request_id=current.provider_request_id
            or identity.provider_request_id,
            attempt_id=current.attempt_id or identity.attempt_id,
        )
        self._provider_request_id = self._provider_request_id or identity.provider_request_id
        self._attempt_id = self._attempt_id or identity.attempt_id

    def _discard_partial_message(self) -> None:
        self._message_started = False
        self._message_stopped = False
        self._message_identity = None
        self._provider_request_id = None
        self._attempt_id = None
        self._builders.clear()
        self._completed_blocks.clear()
        self._usage = Usage()
        self._stop_reason = None


def assemble_model_stream(events: Iterable[ModelStreamEvent]) -> ModelResponse:
    """Assemble a complete response from a finite normalized event list."""
    assembler = ModelStreamAssembler()
    for event in events:
        assembler.apply(event)
    return assembler.response()


def _content_block_index(identity: StreamIdentity) -> int:
    if identity.content_block_index is None:
        raise ModelStreamAssemblyError("content block event requires content_block_index")
    return identity.content_block_index


def _require_matching_identity_field(
    event_type: str,
    *,
    field_name: str,
    expected: str | None,
    actual: str | None,
) -> None:
    if expected is not None and actual is not None and expected != actual:
        raise ModelStreamAssemblyError(
            f"{event_type} identity {field_name} mismatch: "
            f"expected {expected}, got {actual}"
        )


def _delta_mapping(event: ModelStreamEvent) -> Mapping[str, object]:
    if not isinstance(event.delta, Mapping):
        raise ModelStreamAssemblyError("content_block_delta requires a mapping delta")
    return cast(Mapping[str, object], event.delta)


def _tool_uses_from_blocks(
    blocks: tuple[ModelContentBlock, ...],
) -> list[ModelToolUseBlock]:
    tool_uses: list[ModelToolUseBlock] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, ToolUseContentBlock):
            continue
        if block.provider_executed:
            continue
        tool_uses.append(
            ModelToolUseBlock(
                id=block.id,
                name=block.name,
                input=block.input,
                index=index,
            )
        )
    return tool_uses


def _merge_metadata(left: FrozenJson, right: FrozenJson) -> FrozenJson:
    if _is_empty_mapping(right):
        return left
    if _is_empty_mapping(left):
        return right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged: dict[str, FrozenJson] = dict(left)
        for key, value in right.items():
            previous = merged.get(str(key))
            if previous is not None and isinstance(previous, Mapping) and isinstance(
                value,
                Mapping,
            ):
                merged[str(key)] = _merge_metadata_value(previous, value)
            else:
                merged[str(key)] = value
        return freeze_json(merged)
    return right


def _merge_metadata_value(left: FrozenJson, right: FrozenJson) -> FrozenJson:
    if _is_empty_mapping(right):
        return right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        merged: dict[str, FrozenJson] = dict(left)
        for key, value in right.items():
            previous = merged.get(str(key))
            merged[str(key)] = (
                _merge_metadata_value(previous, value)
                if previous is not None
                and isinstance(previous, Mapping)
                and isinstance(value, Mapping)
                else value
            )
        return freeze_json(merged)
    return right


def _is_empty_mapping(value: FrozenJson) -> bool:
    return isinstance(value, Mapping) and len(value) == 0


__all__ = [
    "ModelStreamAssembler",
    "ModelStreamAssemblyError",
    "ModelStreamUpdate",
    "assemble_model_stream",
]
