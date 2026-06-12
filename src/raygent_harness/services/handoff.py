"""Optional handoff classifier seam for background agent notifications."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

from raygent_harness.core.messages import MessageParam

AgentTaskKind = Literal["local_agent", "remote_agent", "in_process_teammate"]
AgentTerminalStatus = Literal["completed", "failed", "killed"]


@dataclass(frozen=True)
class HandoffClassificationRequest:
    """Provider-neutral request for reviewing a completed agent handoff."""

    task_id: str
    task_type: AgentTaskKind
    agent_type: str | None
    description: str
    final_status: AgentTerminalStatus
    final_message: str
    error: str | None = None
    messages: tuple[MessageParam, ...] = ()
    """Child transcript messages available to the classifier, when local."""

    tool_names: tuple[str, ...] = ()
    """Runtime tool names visible to the child agent."""

    permission_mode: str | None = None
    """Permission-mode snapshot for auto-mode or policy-aware classifiers."""

    total_tool_use_count: int | None = None
    """Count of child assistant tool_use blocks when available."""


@dataclass(frozen=True)
class HandoffClassificationResult:
    """Classifier output. Empty warning means no model-visible change."""

    warning: str | None = None
    decision: str | None = None


class AgentHandoffClassifier(Protocol):
    """Optional classifier for subagent output before parent handoff."""

    async def classify(
        self,
        request: HandoffClassificationRequest,
    ) -> HandoffClassificationResult:
        """Return a warning to prepend, or no warning."""
        ...


async def classify_handoff_warning(
    classifier: AgentHandoffClassifier | None,
    request: HandoffClassificationRequest,
    *,
    timeout_s: float,
) -> str | None:
    """Run the optional classifier with bounded, fail-soft behavior.

    Reference handoff classification is an embellishment: status transitions
    happen before classification so task awaiters do not block on a model/API
    call. Raygent mirrors that by returning no warning on timeout/failure.
    """
    if classifier is None or timeout_s <= 0:
        return None
    try:
        result = await asyncio.wait_for(
            classifier.classify(request),
            timeout=timeout_s,
        )
    except Exception:
        return None
    warning = result.warning
    if warning is None or not warning.strip():
        return None
    return warning.strip()


__all__ = [
    "AgentHandoffClassifier",
    "AgentTaskKind",
    "AgentTerminalStatus",
    "HandoffClassificationRequest",
    "HandoffClassificationResult",
    "classify_handoff_warning",
]
