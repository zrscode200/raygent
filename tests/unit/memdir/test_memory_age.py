from __future__ import annotations

from raygent_harness.memdir.memory_age import (
    memory_age,
    memory_age_days,
    memory_freshness_note,
    memory_freshness_text,
)

DAY_MS = 86_400_000
NOW_MS = 2_000_000_000_000.0


def test_memory_age_days_floor_rounds_and_clamps_future() -> None:
    assert memory_age_days(NOW_MS, now_ms=NOW_MS) == 0
    assert memory_age_days(NOW_MS - DAY_MS + 1, now_ms=NOW_MS) == 0
    assert memory_age_days(NOW_MS - DAY_MS, now_ms=NOW_MS) == 1
    assert memory_age_days(NOW_MS - (2 * DAY_MS), now_ms=NOW_MS) == 2
    assert memory_age_days(NOW_MS + DAY_MS, now_ms=NOW_MS) == 0


def test_memory_age_human_text_matches_reference_thresholds() -> None:
    assert memory_age(NOW_MS, now_ms=NOW_MS) == "today"
    assert memory_age(NOW_MS - DAY_MS, now_ms=NOW_MS) == "yesterday"
    assert memory_age(NOW_MS - (9 * DAY_MS), now_ms=NOW_MS) == "9 days ago"


def test_memory_freshness_warns_only_after_yesterday() -> None:
    assert memory_freshness_text(NOW_MS, now_ms=NOW_MS) == ""
    assert memory_freshness_text(NOW_MS - DAY_MS, now_ms=NOW_MS) == ""

    warning = memory_freshness_text(NOW_MS - (2 * DAY_MS), now_ms=NOW_MS)
    assert "This memory is 2 days old." in warning
    assert "point-in-time observations" in warning
    assert "Verify against current code" in warning

    assert memory_freshness_note(NOW_MS - DAY_MS, now_ms=NOW_MS) == ""
    assert memory_freshness_note(NOW_MS - (2 * DAY_MS), now_ms=NOW_MS).startswith(
        "<system-reminder>"
    )
