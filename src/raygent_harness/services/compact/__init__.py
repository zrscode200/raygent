"""services.compact — tiered compaction policy + data structures.

Tiered compaction plugs into Raygent's context-pipeline shape. It covers
proactive autocompact (threshold + circuit breaker) and reactive
context-overflow recovery; microcompact cache edits, session-memory compaction,
and UI warning surfaces remain adapter/service extensions.
"""

from raygent_harness.services.compact.auto_compact import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_DISABLED_QUERY_SOURCES,
    MODEL_CONTEXT_WINDOWS,
    CompactSummarizer,
    CompactSummaryResult,
    TokenEstimator,
    compact_conversation,
    create_autocompact_layer,
    estimate_message_tokens,
    get_auto_compact_threshold,
    get_context_window_for_model,
    get_effective_context_window_size,
    should_auto_compact,
)
from raygent_harness.services.compact.cleanup import (
    PostCompactCleanupContext,
    PostCompactCleanupHook,
    PostCompactCleanupResult,
    clear_post_compact_cleanup_hooks,
    is_main_thread_compact,
    register_post_compact_cleanup_hook,
    run_post_compact_cleanup,
)
from raygent_harness.services.compact.models import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    CompactionResult,
    build_post_compact_messages,
)
from raygent_harness.services.compact.prompt import (
    format_compact_summary,
    get_compact_user_summary_message,
)
from raygent_harness.services.compact.reactive import (
    create_reactive_compact,
    try_reactive_compact,
)
from raygent_harness.services.compact.tool_result_budget import (
    PERSISTED_TOOL_RESULT_TAG,
    ToolResultBudgetResult,
    ToolResultReplacementRecord,
    apply_tool_result_budget,
)

__all__ = [
    "AUTOCOMPACT_BUFFER_TOKENS",
    "DEFAULT_CONTEXT_WINDOW_TOKENS",
    "DEFAULT_DISABLED_QUERY_SOURCES",
    "MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES",
    "MAX_OUTPUT_TOKENS_FOR_SUMMARY",
    "MODEL_CONTEXT_WINDOWS",
    "PERSISTED_TOOL_RESULT_TAG",
    "CompactSummarizer",
    "CompactSummaryResult",
    "CompactionResult",
    "PostCompactCleanupContext",
    "PostCompactCleanupHook",
    "PostCompactCleanupResult",
    "TokenEstimator",
    "ToolResultBudgetResult",
    "ToolResultReplacementRecord",
    "apply_tool_result_budget",
    "build_post_compact_messages",
    "clear_post_compact_cleanup_hooks",
    "compact_conversation",
    "create_autocompact_layer",
    "create_reactive_compact",
    "estimate_message_tokens",
    "format_compact_summary",
    "get_auto_compact_threshold",
    "get_compact_user_summary_message",
    "get_context_window_for_model",
    "get_effective_context_window_size",
    "is_main_thread_compact",
    "register_post_compact_cleanup_hook",
    "run_post_compact_cleanup",
    "should_auto_compact",
    "try_reactive_compact",
]
