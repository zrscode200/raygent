"""Tests for compact summary formatting."""

from __future__ import annotations

from raygent_harness.services.compact import (
    format_compact_summary,
    get_compact_user_summary_message,
)


def test_strips_analysis_block() -> None:
    raw = "<analysis>scratchpad notes</analysis><summary>real summary</summary>"
    out = format_compact_summary(raw)
    assert "<analysis>" not in out
    assert "scratchpad notes" not in out
    assert "real summary" in out


def test_replaces_summary_tags_with_header() -> None:
    raw = "<summary>line 1\nline 2</summary>"
    out = format_compact_summary(raw)
    assert out.startswith("Summary:")
    assert "<summary>" not in out
    assert "</summary>" not in out
    assert "line 1\nline 2" in out


def test_summary_content_is_trimmed_inside_header() -> None:
    # Leading/trailing whitespace inside summary tags should not survive.
    raw = "<summary>\n   indented body   \n</summary>"
    out = format_compact_summary(raw)
    assert out == "Summary:\nindented body"


def test_collapses_runs_of_blank_lines() -> None:
    raw = "<analysis>x</analysis>\n\n\n\n<summary>body</summary>"
    out = format_compact_summary(raw)
    # No more than one blank line between sections after collapse.
    assert "\n\n\n" not in out


def test_missing_analysis_tag_is_noop() -> None:
    raw = "<summary>only summary</summary>"
    out = format_compact_summary(raw)
    assert out == "Summary:\nonly summary"


def test_missing_summary_tag_returns_input_verbatim_modulo_whitespace() -> None:
    # Degenerate path: model produced an analysis but no summary. We strip the
    # analysis and return whatever's left; it's the caller's job to treat
    # that as a failed summary.
    raw = "<analysis>only analysis</analysis>tail text"
    out = format_compact_summary(raw)
    assert out == "tail text"


def test_strips_only_first_analysis_block() -> None:
    # Only the first analysis block is stripped. Later occurrences are treated
    # as literal summary body content.
    raw = (
        "<analysis>first</analysis>"
        "<summary>body mentioning <analysis>second</analysis> tags</summary>"
    )
    out = format_compact_summary(raw)
    assert "first" not in out
    assert "<analysis>second</analysis>" in out


def test_idempotent_on_already_formatted_output() -> None:
    # Running the formatter twice should produce the same string. Guards
    # against accidental re-application stripping content.
    raw = "<analysis>x</analysis><summary>body</summary>"
    once = format_compact_summary(raw)
    twice = format_compact_summary(once)
    assert once == twice


def test_summary_content_with_backslashes_is_preserved_verbatim() -> None:
    # Backslashes in summary content must pass through verbatim; naive regex
    # replacement can interpret them as escape sequences.
    raw = r"<summary>path: C:\Users\name and tex\test</summary>"
    out = format_compact_summary(raw)
    assert r"C:\Users\name" in out
    assert r"tex\test" in out
    assert out.startswith("Summary:")


def test_summary_content_with_backreference_lookalikes() -> None:
    # Backslash-digit sequences (\1, \12) and \g<name> would be interpreted
    # as backreferences by re.sub on a string repl; they must pass through
    # verbatim.
    raw = r"<summary>regex notes: use \1 and \g<name></summary>"
    out = format_compact_summary(raw)
    assert r"\1" in out
    assert r"\g<name>" in out


def test_compact_user_summary_message_wraps_formatted_summary() -> None:
    out = get_compact_user_summary_message("<summary>body</summary>")

    assert out.startswith(
        "This session is being continued from a previous conversation"
    )
    assert "Summary:\nbody" in out
    assert "Continue the conversation" not in out


def test_compact_user_summary_message_can_suppress_follow_up_questions() -> None:
    out = get_compact_user_summary_message(
        "<summary>body</summary>",
        suppress_follow_up_questions=True,
        transcript_path="/tmp/transcript.jsonl",
        recent_messages_preserved=True,
    )

    assert "Summary:\nbody" in out
    assert "/tmp/transcript.jsonl" in out
    assert "Recent messages are preserved verbatim." in out
    assert "Continue the conversation from where it left off" in out
    assert "do not acknowledge the summary" in out
