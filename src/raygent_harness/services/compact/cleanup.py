"""Post-compact cleanup hook registry.

Post-compact cleanup resets module-level cache-like services after main-thread
compaction. The stable seam is explicit: compaction success paths call
`run_post_compact_cleanup(...)`, and memory/skills/tool caches can register
hooks here without importing product globals.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PostCompactCleanupContext:
    """Context passed to cleanup hooks after a successful compaction."""

    query_source: str | None = None
    """Reference-style query source. `None`, `sdk`, and `repl_main_thread*`
    are main-thread compacts unless an `agent_id` is present."""

    agent_id: str | None = None
    """None = main thread. Set for subagent compacts sharing process state."""

    is_main_thread: bool = field(init=False)
    """Whether main-thread-only cleanup hooks are allowed to run."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "is_main_thread",
            is_main_thread_compact(
                query_source=self.query_source,
                agent_id=self.agent_id,
            ),
        )


PostCompactCleanupHook = Callable[
    [PostCompactCleanupContext],
    None | Awaitable[None],
]


@dataclass(frozen=True)
class PostCompactCleanupResult:
    """Observable result of one cleanup run."""

    context: PostCompactCleanupContext
    called: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RegisteredCleanupHook:
    name: str
    hook: PostCompactCleanupHook
    main_thread_only: bool


_HOOKS: list[_RegisteredCleanupHook] = []


def register_post_compact_cleanup_hook(
    hook: PostCompactCleanupHook,
    *,
    name: str | None = None,
    main_thread_only: bool = False,
) -> Callable[[], None]:
    """Register a hook and return an unregister callback.

    Hook failures are captured by `run_post_compact_cleanup` and do not fail the
    compaction that already succeeded. This keeps cleanup best-effort while
    preserving observability through `PostCompactCleanupResult.errors`.
    """
    registration = _RegisteredCleanupHook(
        name=name or getattr(hook, "__name__", hook.__class__.__name__),
        hook=hook,
        main_thread_only=main_thread_only,
    )
    _HOOKS.append(registration)

    def unregister() -> None:
        with _suppress_value_error():
            _HOOKS.remove(registration)

    return unregister


async def run_post_compact_cleanup(
    *,
    query_source: str | None = None,
    agent_id: str | None = None,
    notify_error: Callable[[str], None] | None = None,
) -> PostCompactCleanupResult:
    """Run registered cleanup hooks after compaction success.

    Main-thread-only hooks run only for main-thread compacts, matching the
    reference's guard for shared module-level state. A snapshot is used so hooks
    can safely register/unregister other hooks during cleanup without mutating
    the active iteration.
    """
    context = PostCompactCleanupContext(
        query_source=query_source,
        agent_id=agent_id,
    )
    called: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for registration in tuple(_HOOKS):
        if registration.main_thread_only and not context.is_main_thread:
            skipped.append(registration.name)
            continue
        try:
            maybe_awaitable = registration.hook(context)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
            called.append(registration.name)
        except Exception as exc:
            error = f"{registration.name}: {exc}"
            errors.append(error)
            if notify_error is not None:
                notify_error(f"post-compact cleanup hook failed: {error}")

    return PostCompactCleanupResult(
        context=context,
        called=tuple(called),
        skipped=tuple(skipped),
        errors=tuple(errors),
    )


def is_main_thread_compact(
    *,
    query_source: str | None = None,
    agent_id: str | None = None,
) -> bool:
    """Return whether main-thread-only cache cleanup is safe.

    Reference treats undefined, `repl_main_thread*`, and `sdk` query sources as
    main-thread compacts. Raygent also checks `agent_id`: a subagent with no
    explicit query_source still shares the process and must not clear parent
    main-thread state.
    """
    if agent_id is not None:
        return False
    return (
        query_source is None
        or query_source == "sdk"
        or query_source.startswith("repl_main_thread")
    )


def clear_post_compact_cleanup_hooks() -> None:
    """Clear registry state. Intended for tests and controlled shutdown."""
    _HOOKS.clear()


class _suppress_value_error:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: Any,
    ) -> bool:
        return exc_type is ValueError


__all__ = [
    "PostCompactCleanupContext",
    "PostCompactCleanupHook",
    "PostCompactCleanupResult",
    "clear_post_compact_cleanup_hooks",
    "is_main_thread_compact",
    "register_post_compact_cleanup_hook",
    "run_post_compact_cleanup",
]
