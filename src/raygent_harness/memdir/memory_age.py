"""Memory freshness helpers.

"""

from __future__ import annotations

SECONDS_PER_DAY = 86_400


def memory_age_days(mtime_ms: float, *, now_ms: float | None = None) -> int:
    """Return floor-rounded days elapsed since `mtime_ms`, clamped at 0."""
    if now_ms is None:
        import time

        now_ms = time.time() * 1000
    return max(0, int((now_ms - mtime_ms) // (SECONDS_PER_DAY * 1000)))


def memory_age(mtime_ms: float, *, now_ms: float | None = None) -> str:
    """Human-readable age string: today, yesterday, or N days ago."""
    days = memory_age_days(mtime_ms, now_ms=now_ms)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_text(mtime_ms: float, *, now_ms: float | None = None) -> str:
    """Plain staleness warning for memories older than one day."""
    days = memory_age_days(mtime_ms, now_ms=now_ms)
    if days <= 1:
        return ""
    return (
        f"This memory is {days} days old. "
        "Memories are point-in-time observations, not live state - "
        "claims about code behavior or file:line citations may be outdated. "
        "Verify against current code before asserting as fact."
    )


def memory_freshness_note(mtime_ms: float, *, now_ms: float | None = None) -> str:
    """Staleness warning wrapped for model context."""
    text = memory_freshness_text(mtime_ms, now_ms=now_ms)
    if not text:
        return ""
    return f"<system-reminder>{text}</system-reminder>\n"


__all__ = [
    "memory_age",
    "memory_age_days",
    "memory_freshness_note",
    "memory_freshness_text",
]
