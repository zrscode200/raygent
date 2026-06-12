"""Provider-neutral retry policy for protocol-backed model runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from raygent_harness.core.model_types import ProviderError

RetryOperation = Literal["complete", "stream", "count_tokens"]

_RETRYABLE_KINDS = frozenset({"transient", "server_overload"})


@dataclass(frozen=True, slots=True)
class ProviderRetryPolicy:
    """Bounded retry/fallback policy for injected provider transports.

    `max_attempts=1` preserves the historical no-retry behavior. Stream retries
    are opt-in because replaying a stream after model-visible output can corrupt
    transcript continuity.
    """

    max_attempts: int = 1
    base_delay_s: float = 0.0
    max_delay_s: float | None = None
    retry_rate_limit: bool = False
    retry_stream_transport_errors: bool = False
    fallback_stream_to_complete: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s cannot be negative")
        if self.max_delay_s is not None and self.max_delay_s < 0:
            raise ValueError("max_delay_s cannot be negative")


@dataclass(frozen=True, slots=True)
class ProviderRetryDecision:
    """Pure retry classification result."""

    should_retry: bool
    delay_s: float = 0.0
    reason: str = "not_retryable"


def classify_retry_decision(
    error: ProviderError,
    *,
    operation: RetryOperation,
    attempt: int,
    policy: ProviderRetryPolicy,
    stream_events_emitted: bool = False,
) -> ProviderRetryDecision:
    """Return whether a provider operation should be retried.

    Attempts are one-based: `attempt=1` is the first failed attempt. A policy
    with `max_attempts=2` can retry that first failure once.
    """

    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    if attempt >= policy.max_attempts:
        return ProviderRetryDecision(False, reason="max_attempts_exhausted")
    if not error.retryable:
        return ProviderRetryDecision(False, reason="provider_error_not_retryable")
    if operation == "stream":
        if stream_events_emitted:
            return ProviderRetryDecision(False, reason="stream_events_already_emitted")
        if not policy.retry_stream_transport_errors:
            return ProviderRetryDecision(False, reason="stream_retry_disabled")
    if error.kind == "rate_limit":
        if not policy.retry_rate_limit:
            return ProviderRetryDecision(False, reason="rate_limit_retry_disabled")
    elif error.kind not in _RETRYABLE_KINDS:
        return ProviderRetryDecision(False, reason=f"{error.kind}_not_retriable_by_policy")

    return ProviderRetryDecision(
        True,
        delay_s=_retry_delay_s(error, attempt=attempt, policy=policy),
        reason=f"{operation}_{error.kind}_retry",
    )


def should_fallback_stream_to_complete(
    error: ProviderError,
    *,
    policy: ProviderRetryPolicy,
    stream_events_emitted: bool = False,
) -> bool:
    """Return whether a failed stream may be replaced by non-streaming complete."""

    return (
        policy.fallback_stream_to_complete
        and not stream_events_emitted
        and error.retryable
        and error.safe_to_fallback
    )


def _retry_delay_s(
    error: ProviderError,
    *,
    attempt: int,
    policy: ProviderRetryPolicy,
) -> float:
    if error.retry_after_s is not None:
        delay = max(0.0, error.retry_after_s)
    else:
        delay = policy.base_delay_s * (2 ** (attempt - 1))
    if policy.max_delay_s is not None:
        delay = min(delay, policy.max_delay_s)
    return delay


__all__ = [
    "ProviderRetryDecision",
    "ProviderRetryPolicy",
    "RetryOperation",
    "classify_retry_decision",
    "should_fallback_stream_to_complete",
]
