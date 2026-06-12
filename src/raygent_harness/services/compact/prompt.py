"""Compaction prompt + summary-text formatting.

Chunk 1 only ports the post-summary text formatter. The full compaction
prompt (BASE / PARTIAL variants, no-tools preamble, trailer) lands in
chunk 3 alongside the summarizer protocol — there's no consumer for it
until then.

(`getCompactUserSummaryMessage`).
"""

from __future__ import annotations

import re

# TS regex literals without the /g flag replace the FIRST match only;
# Python's `re.sub` defaults to all matches, so the analysis-strip and
# Summary extraction pins `count=1`; whitespace collapse uses `count=0`
# to replace all matches.
# there.
_ANALYSIS_RE = re.compile(r"<analysis>[\s\S]*?</analysis>")
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>")
_BLANK_LINES_RE = re.compile(r"\n\n+")


def format_compact_summary(summary: str) -> str:
    """Strip the `<analysis>` scratchpad and rewrite `<summary>...</summary>`
    as `Summary:\\n<content>`, then collapse runs of blank lines.

    transform — no model call, no I/O. Used by both the proactive autocompact
    layer (chunk 3) and the reactive recovery path (chunk 4) before the
    summary text is wrapped into a `MessageParam`.

    Edge cases preserved from the reference:
    - Missing `<analysis>` tags → first regex is a no-op; the rest still runs.
    - Missing `<summary>` tags → return the input verbatim (after analysis
      strip and whitespace collapse). Caller is responsible for treating that
      as a degenerate summary.
    - Untrimmed `<summary>` content → trimmed before being concatenated under
      the `Summary:` header (reference does `content.trim()`).
    - Backslashes in the summary content: passed through literally. Python's
      `re.sub` interprets `\\<X>` sequences in a *string* replacement
      (raising `re.error: bad escape \\U` on payloads like `C:\\Users`); JS
      `String.prototype.replace` with a string second arg uses `$`-based
      special patterns and does NOT touch backslashes. We use a function
      replacement here so the captured `content` is concatenated verbatim,
      matching expected behavior.
    """
    formatted = _ANALYSIS_RE.sub("", summary, count=1)

    match = _SUMMARY_RE.search(formatted)
    if match is not None:
        content = (match.group(1) or "").strip()
        replacement = f"Summary:\n{content}"
        formatted = _SUMMARY_RE.sub(lambda _m: replacement, formatted, count=1)

    formatted = _BLANK_LINES_RE.sub("\n\n", formatted)
    return formatted.strip()


def get_compact_user_summary_message(
    summary: str,
    *,
    suppress_follow_up_questions: bool = False,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
) -> str:
    """Wrap a formatted compaction summary for the next model turn.

    reference's proactive feature-gated trailer, which depends on unported
    runtime feature flags.
    """
    formatted_summary = format_compact_summary(summary)
    base_summary = (
        "This session is being continued from a previous conversation that "
        "ran out of context. The summary below covers the earlier portion of "
        "the conversation.\n\n"
        f"{formatted_summary}"
    )

    if transcript_path:
        base_summary += (
            "\n\nIf you need specific details from before compaction (like "
            "exact code snippets, error messages, or content you generated), "
            f"read the full transcript at: {transcript_path}"
        )

    if recent_messages_preserved:
        base_summary += "\n\nRecent messages are preserved verbatim."

    if suppress_follow_up_questions:
        return (
            f"{base_summary}\n"
            "Continue the conversation from where it left off without asking "
            "the user any further questions. Resume directly — do not "
            "acknowledge the summary, do not recap what was happening, do not "
            'preface with "I\'ll continue" or similar. Pick up the last task '
            "as if the break never happened."
        )

    return base_summary


__all__ = ["format_compact_summary", "get_compact_user_summary_message"]
