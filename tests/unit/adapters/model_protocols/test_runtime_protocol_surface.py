from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from raygent_harness.adapters.model_protocols import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
)
from raygent_harness.core.messages import message_param_from_api_message, thaw_json
from raygent_harness.core.model_types import (
    ApiMessage,
    ModelMessage,
    ModelRequest,
    ModelSampling,
    TextContentBlock,
)


def _request() -> ModelRequest:
    return ModelRequest(
        model="model-test",
        messages=(
            ApiMessage(
                message=ModelMessage(
                    role="user",
                    content=(TextContentBlock(text="hello"),),
                )
            ),
        ),
        sampling=ModelSampling(max_tokens=99),
    )


def _body(prepared: object) -> Mapping[str, object]:
    raw = thaw_json(cast(Any, prepared).body)
    assert isinstance(raw, Mapping)
    return cast(Mapping[str, object], raw)


def test_anthropic_complete_request_omits_stream_and_response_parser_is_protocol_owned() -> None:
    adapter = AnthropicMessagesAdapter()

    complete_body = _body(adapter.prepare_complete_request(_request()))
    stream_body = _body(adapter.prepare_stream_request(_request()))
    response = adapter.parse_response(
        {
            "id": "msg_1",
            "request_id": "req_1",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
        }
    )

    assert "stream" not in complete_body
    assert complete_body["max_tokens"] == 99
    assert stream_body["stream"] is True
    assert message_param_from_api_message(response.api_message) == {
        "role": "assistant",
        "id": "msg_1",
        "content": "done",
    }
    assert response.provider_request_id == "req_1"


def test_openai_complete_request_omits_stream_and_response_parser_is_protocol_owned() -> None:
    adapter = OpenAIResponsesAdapter()

    complete_body = _body(adapter.prepare_complete_request(_request()))
    stream_body = _body(adapter.prepare_stream_request(_request()))
    response = adapter.parse_response(
        {
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        }
    )

    assert "stream" not in complete_body
    assert complete_body["max_output_tokens"] == 99
    assert stream_body["stream"] is True
    assert message_param_from_api_message(response.api_message) == {
        "role": "assistant",
        "id": "resp_1",
        "content": "done",
    }
    assert response.provider_request_id == "resp_1"
    assert response.usage.effective_total_tokens == 6
