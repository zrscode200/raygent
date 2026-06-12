"""Model protocol translators.

Protocol adapters lower Raygent model requests to provider-shaped payloads and
raise provider streams/errors back into Raygent-owned model types. They do not
own transport, credentials, or live SDK clients.
"""

from raygent_harness.adapters.model_protocols.anthropic_messages import (
    AnthropicMessagesAdapter,
)
from raygent_harness.adapters.model_protocols.base import (
    ModelProtocolAdapter,
    ModelProtocolStreamParser,
    PreparedModelRequest,
    ProviderEvent,
    ProviderResponse,
)
from raygent_harness.adapters.model_protocols.openai_responses import (
    OpenAIResponsesAdapter,
)

__all__ = [
    "AnthropicMessagesAdapter",
    "ModelProtocolAdapter",
    "ModelProtocolStreamParser",
    "OpenAIResponsesAdapter",
    "PreparedModelRequest",
    "ProviderEvent",
    "ProviderResponse",
]
