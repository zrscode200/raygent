"""QueryConfig — immutable per-turn configuration snapshot.

Config is frozen at turn start. Mutating mid-turn would cause incoherent decisions (e.g., tools
list changing between permission-check and call, or model swapping mid-recovery-
ladder). State is what changes iteration-to-iteration; Config is the scaffold.

Frozen dataclass (eq=True, frozen=True) — attempting to mutate raises at
runtime. This is the Python analogue of TS's `readonly` + `Object.freeze`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from raygent_harness.core.messages import MessageParam
    from raygent_harness.core.tool import Tool, ToolUseContext

# ---------------------------------------------------------------------------
# Model + sampling parameters
# ---------------------------------------------------------------------------

ModelName = str
"""Provider-specific model identifier or alias.

Core treats this as opaque; `ModelProvider.resolve_model(...)` owns any alias,
window-suffix, or provider-specific resolution.
"""


@dataclass(frozen=True)
class SamplingParams:
    """Model sampling knobs. Frozen — a turn runs with one set of params."""

    max_tokens: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Turn budgets — hard stops that produce terminal results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnBudget:
    """Hard per-turn budgets. Exceeding any produces a typed terminal result.

    `None` = unlimited for that dimension. In practice you almost always want
    at least max_turns set — an unbounded loop can burn a lot of tokens before
    you notice something is off.
    """

    max_turns: int | None = 100
    """Max agent-loop iterations (assistant-tool round trips)."""

    max_budget_usd: float | None = None
    """Hard cap on usage dollars. Checked after each model response."""

    max_structured_output_retries: int | None = 3
    """Max retries when structured-output validation fails."""


# ---------------------------------------------------------------------------
# Deprecated permission-mode snapshot. Runtime permission decisions and SDK
# system-init reporting use `QueryDeps.permission_context.mode`, matching the
# runtime state model. This field remains only for older callsites that
# still construct `QueryConfig(permission_mode=...)`.
# ---------------------------------------------------------------------------

PermissionMode = Literal[
    "default",
    "plan",
    "accept_edits",
    "bypass",
]
"""
Legacy values from the pre-Group-4 QueryConfig scaffold. New permission code
uses `raygent_harness.core.permissions.PermissionMode` values on
`QueryDeps.permission_context`.
"""


# ---------------------------------------------------------------------------
# Can-use-tool callback — external override point for permission decisions.
# ---------------------------------------------------------------------------

CanUseToolCallback = Callable[
    [str, dict[str, Any], "ToolUseContext"],
    Awaitable["CanUseToolResult"],
]
"""(tool_name, tool_input, ctx) -> CanUseToolResult. Wraps check_permissions.

Set by the SDK consumer to intercept every tool call before it runs. Returning
`allow` with `updated_input` lets the consumer rewrite the call (e.g., sandbox
a path); returning `deny` with a message surfaces back to the model."""


@dataclass(frozen=True)
class CanUseToolAllow:
    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None


@dataclass(frozen=True)
class CanUseToolDeny:
    message: str = ""
    behavior: Literal["deny"] = "deny"


CanUseToolResult = CanUseToolAllow | CanUseToolDeny


# ---------------------------------------------------------------------------
# QueryConfig — the frozen per-turn snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryConfig:
    """Immutable per-turn configuration. Captured at query() entry, not mutated.

    Why frozen: the agent loop reads config at many points (pre-call gating,
    budget checks, recovery ladder, compaction triggers). If config could shift
    mid-turn, two reads could disagree and drive the loop into a wedged state.
    Freezing forces every config change to happen at turn boundaries.
    """

    # --- model ---
    model: ModelName
    fallback_model: ModelName | None = None
    """If the primary model errors in a recoverable way, retry with this one."""

    sampling: SamplingParams = field(default_factory=SamplingParams)

    # --- system prompt (rendered once, frozen) ---
    system_prompt: str = ""
    """Rendered system prompt. DO NOT template mid-turn — cache coherence."""

    context_messages: tuple[MessageParam, ...] = ()
    """Non-persistent user-context messages prepended to provider requests.

    These mirror the user-context lane: model-visible for the request,
    but not normal conversation history and not replayed from transcripts.
    """

    context_system_prompt: str = ""
    """Non-persistent system-context suffix joined only at model-call time.

    Keeping this separate prevents parent turn env/git context from becoming
    ordinary inherited system prompt text for child loops.
    """

    # --- tools ---
    tools: tuple[Tool, ...] = ()
    """Tools available this turn. Tuple (not list) to reinforce immutability."""

    # --- budgets ---
    budget: TurnBudget = field(default_factory=TurnBudget)

    # --- permissions ---
    permission_mode: PermissionMode = "default"
    can_use_tool: CanUseToolCallback | None = None

    # --- identity / tracking ---
    session_id: str = ""
    agent_id: str | None = None
    """None = main thread. Set when running as subagent; propagated into ctx."""

    # --- experimental toggles ---
    experiments: Mapping[str, bool] = field(default_factory=dict[str, bool])
    """Feature flags for gradual rollouts. Keys are experiment IDs; value True
    means enabled for this turn. Normalized to a read-only mapping."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_messages", tuple(self.context_messages))
        object.__setattr__(
            self,
            "experiments",
            MappingProxyType(dict(self.experiments)),
        )


__all__ = [
    "CanUseToolAllow",
    "CanUseToolCallback",
    "CanUseToolDeny",
    "CanUseToolResult",
    "ModelName",
    "PermissionMode",
    "QueryConfig",
    "SamplingParams",
    "TurnBudget",
]
