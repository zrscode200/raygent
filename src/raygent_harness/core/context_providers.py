"""Provider-neutral turn context fragments.

Reference separates system context (system prompt suffixes such as git status)
from user context (meta user messages such as instruction files/current date).
Raygent models that split without importing product globals or hardcoding
instruction filenames into the query loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from raygent_harness.core.messages import MessageParam

if TYPE_CHECKING:
    from raygent_harness.core.config import QueryConfig
    from raygent_harness.core.tool import ToolUseContext


ContextChannel = Literal["system", "user_context"]
ContextAgentScope = Literal["all", "main", "subagent"]
ContextRenderMode = Literal["context", "instructions"]
ContextKind = Literal[
    "environment",
    "git",
    "project_instructions",
    "memory",
    "goal",
    "custom",
]


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """A bounded piece of context for one submitted turn."""

    id: str
    content: str
    channel: ContextChannel
    source: str | None = None
    priority: int = 0
    agent_scope: ContextAgentScope = "all"
    render_mode: ContextRenderMode = "context"
    kind: ContextKind = "custom"


class ContextProvider(Protocol):
    """Return turn-scoped context fragments.

    Providers should be fail-soft and bounded. QueryEngine catches provider
    exceptions so a broken filesystem/git/instruction provider does not crash a
    normal model turn.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        /,
    ) -> Sequence[ContextFragment]:
        ...


class PostToolContextProvider(Protocol):
    """Return transient context after tools complete in a loop iteration.

    These fragments are model-visible for later requests in the current
    submitted turn, but they must not be persisted into `State.messages` or the
    QueryEngine transcript.
    """

    async def __call__(
        self,
        config: QueryConfig,
        ctx: ToolUseContext,
        read_paths: Sequence[str],
        already_attached_sources: Sequence[str],
        /,
    ) -> Sequence[ContextFragment]:
        ...


def scope_context_fragments(
    fragments: Sequence[ContextFragment],
    *,
    agent_id: str | None,
) -> tuple[ContextFragment, ...]:
    """Filter fragments by main/subagent scope."""

    scoped: list[ContextFragment] = []
    for fragment in fragments:
        if context_agent_scope_includes(fragment.agent_scope, agent_id=agent_id):
            scoped.append(fragment)
    return tuple(scoped)


def context_agent_scope_includes(
    agent_scope: ContextAgentScope,
    *,
    agent_id: str | None,
) -> bool:
    """Return whether a main/subagent scope includes this agent id."""

    return (
        agent_scope == "all"
        or (agent_scope == "main" and agent_id is None)
        or (agent_scope == "subagent" and agent_id is not None)
    )


def context_provider_kind(provider: object) -> ContextKind:
    """Return a provider's filterable kind, preserving custom providers."""

    kind = getattr(provider, "context_kind", "custom")
    if kind in {
        "environment",
        "git",
        "project_instructions",
        "memory",
        "goal",
        "custom",
    }:
        return cast(ContextKind, kind)
    return "custom"


def filter_context_providers_by_kind[ProviderT](
    providers: Sequence[ProviderT],
    *,
    omitted_kinds: Iterable[ContextKind],
) -> tuple[ProviderT, ...]:
    """Filter provider sequences by explicit context kind.

    Unknown/custom providers are retained by default. Embedders that want a
    custom provider to participate in agent context policy can expose a
    `context_kind` attribute with one of `ContextKind`.
    """

    omitted = frozenset(omitted_kinds)
    if not omitted:
        return tuple(providers)
    return tuple(
        provider
        for provider in providers
        if context_provider_kind(provider) not in omitted
    )


def order_context_fragments(
    fragments: Sequence[ContextFragment],
) -> tuple[ContextFragment, ...]:
    """Stable priority ordering across providers."""

    indexed = enumerate(fragments)
    return tuple(
        fragment
        for _index, fragment in sorted(
            indexed,
            key=lambda item: (item[1].priority, item[0]),
        )
    )


def render_system_context(fragments: Sequence[ContextFragment]) -> str | None:
    """Render system-channel fragments as a prompt suffix."""

    parts = [
        fragment.content.strip()
        for fragment in order_context_fragments(fragments)
        if fragment.channel == "system" and fragment.content.strip()
    ]
    if not parts:
        return None
    return "\n\n".join(parts)


def render_user_context_messages(
    fragments: Sequence[ContextFragment],
) -> tuple[MessageParam, ...]:
    """Render user-context fragments as a non-persistent meta-style message."""

    context_parts: list[str] = []
    instruction_parts: list[str] = []
    for fragment in order_context_fragments(fragments):
        content = fragment.content.strip()
        if fragment.channel != "user_context" or not content:
            continue
        label = fragment.source or fragment.id
        part = f"# {label}\n{content}"
        if fragment.render_mode == "instructions":
            instruction_parts.append(part)
        else:
            context_parts.append(part)

    messages: list[MessageParam] = []

    if instruction_parts:
        body = "\n\n".join(instruction_parts)
        messages.append(
            cast(
                MessageParam,
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>\n"
                        "Codebase and user instructions are shown below. Be sure to "
                        "adhere to these instructions. IMPORTANT: These instructions "
                        "OVERRIDE any default behavior and you MUST follow them exactly "
                        "as written.\n"
                        f"{body}\n"
                        "</system-reminder>"
                    ),
                },
            )
        )

    if context_parts:
        body = "\n\n".join(context_parts)
        messages.append(
            cast(
                MessageParam,
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>\n"
                        "As you answer the user's questions, you can use the following context:\n"
                        f"{body}\n\n"
                        "IMPORTANT: this context may or may not be relevant to your task. "
                        "Do not respond to this context unless it is relevant.\n"
                        "</system-reminder>"
                    ),
                },
            )
        )

    return tuple(messages)


__all__ = [
    "ContextAgentScope",
    "ContextChannel",
    "ContextFragment",
    "ContextKind",
    "ContextProvider",
    "ContextRenderMode",
    "PostToolContextProvider",
    "context_agent_scope_includes",
    "context_provider_kind",
    "filter_context_providers_by_kind",
    "order_context_fragments",
    "render_system_context",
    "render_user_context_messages",
    "scope_context_fragments",
]
