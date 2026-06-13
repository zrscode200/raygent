from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import cast

import pytest

import raygent_harness.services.improvement_runtime as runtime_module
from raygent_harness.core.model_types import FrozenJson
from raygent_harness.core.observability import KernelEvent, KernelEventBus
from raygent_harness.improvement import (
    ImprovementEvaluationCheck,
    ImprovementEvaluationPlan,
    ImprovementEvidence,
    ImprovementEvidenceBounds,
    ImprovementPatchCandidatePlan,
    ImprovementPatchCandidateWorktreeAllocator,
    ImprovementPatchCandidateWorktreeApproval,
    ImprovementPatchCandidateWorktreeValidationError,
    ImprovementTarget,
)
from raygent_harness.services.improvement_runtime import (
    DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS,
    DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS,
    DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS,
    DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
    DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS,
    DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS,
    IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION,
    ImprovementEvidenceCollectionBounds,
    ImprovementEvidenceCollectionRequest,
    ImprovementEvidenceCollectionResult,
    ImprovementObservabilitySnapshot,
    ImprovementRuntimeBridge,
    ImprovementRuntimeBridgeConfig,
    ImprovementRuntimeChainSummary,
    ImprovementRuntimeObservabilityEvent,
    ImprovementRuntimeObservabilitySink,
    ImprovementRuntimePermissionPolicy,
    ImprovementRuntimePermissionReport,
    ImprovementRuntimePermissionRequirement,
    ImprovementRuntimeRecord,
    ImprovementRuntimeRecordQuery,
    ImprovementRuntimeRecoveryRequest,
    ImprovementRuntimeRecoveryService,
    ImprovementRuntimeRequest,
    ImprovementRuntimeTransitionResult,
    ImprovementRuntimeValidationError,
    ImprovementTaskOutputEvidenceTarget,
    KernelEventImprovementRuntimeObserver,
    ObservabilitySnapshotImprovementEvidenceAdapter,
    TaskOutputImprovementEvidenceAdapter,
    TranscriptSearchImprovementEvidenceAdapter,
    improvement_evidence_collection_result_from_dict,
    improvement_evidence_collection_result_to_dict,
    improvement_runtime_chain_summary_from_dict,
    improvement_runtime_chain_summary_to_dict,
    improvement_runtime_observability_event_from_dict,
    improvement_runtime_observability_event_to_dict,
    improvement_runtime_permission_report_from_dict,
    improvement_runtime_permission_report_to_dict,
    improvement_runtime_permission_summary_to_dict,
    improvement_runtime_record_from_dict,
    improvement_runtime_record_to_dict,
    improvement_runtime_recovery_result_from_dict,
    improvement_runtime_recovery_result_to_dict,
    recover_improvement_runtime_chain,
    validate_improvement_evidence_collection,
)
from raygent_harness.services.task_output import (
    TaskOutputReadResult,
    TaskOutputReference,
    TaskOutputStore,
)
from raygent_harness.services.transcript import (
    TranscriptSearchMatch,
    TranscriptSearchRequest,
    TranscriptSearchResult,
    TranscriptSearchService,
)
from raygent_harness.services.worktree.manager import WorktreeManager
from raygent_harness.services.worktree.models import (
    WorktreeCleanupResult,
    WorktreeInfo,
)


def _evidence(
    *,
    evidence_id: str = "ev_1",
    source: str = "transcript",
    summary: str = "A bounded transcript summary.",
    excerpt: str | None = "bounded transcript excerpt",
    source_uri: str | None = "transcript://session-1/m1",
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementEvidence:
    resolved_metadata: Mapping[str, FrozenJson] = (
        {"rank": 1} if metadata is None else metadata
    )
    return ImprovementEvidence(
        evidence_id=evidence_id,
        source=source,  # type: ignore[arg-type]
        summary=summary,
        excerpt=excerpt,
        source_uri=source_uri,
        created_at=100.0,
        metadata=resolved_metadata,
    )


def _collection_request() -> ImprovementEvidenceCollectionRequest:
    return ImprovementEvidenceCollectionRequest(
        request_id="ier_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        source_kinds=("transcript", "task_output"),
        query="find regression symptoms",
        metadata={"purpose": "rsi-006a"},
    )


def _empty_collection_requests() -> list[ImprovementEvidenceCollectionRequest]:
    return []


def _empty_runtime_records() -> list[ImprovementRuntimeRecord]:
    return []


def _empty_record_queries() -> list[ImprovementRuntimeRecordQuery]:
    return []


@dataclass
class FakeEvidenceSource:
    evidence: tuple[ImprovementEvidence, ...] = field(
        default_factory=lambda: (_evidence(),)
    )
    warnings: tuple[str, ...] = ("source truncated",)
    requests: list[ImprovementEvidenceCollectionRequest] = field(
        default_factory=_empty_collection_requests
    )

    async def collect(
        self,
        request: ImprovementEvidenceCollectionRequest,
    ) -> ImprovementEvidenceCollectionResult:
        self.requests.append(request)
        return ImprovementEvidenceCollectionResult(
            request_id=request.request_id,
            session_id=request.session_id,
            runtime_session_id=request.runtime_session_id,
            source_id="fake-transcript",
            evidence=self.evidence,
            warnings=self.warnings,
            truncated=True,
            metadata={"adapter": "fake"},
        )


@dataclass
class FakeRecordStore:
    records: list[ImprovementRuntimeRecord] = field(
        default_factory=_empty_runtime_records
    )
    load_requests: list[ImprovementRuntimeRecordQuery] = field(
        default_factory=_empty_record_queries
    )
    append_requests: list[ImprovementRuntimeRecord] = field(
        default_factory=_empty_runtime_records
    )

    async def append_record(
        self,
        record: ImprovementRuntimeRecord,
    ) -> ImprovementRuntimeRecord:
        self.append_requests.append(record)
        self.records.append(record)
        return record

    async def load_records(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> tuple[ImprovementRuntimeRecord, ...]:
        self.load_requests.append(query)
        return tuple(
            record
            for record in self.records
            if _record_matches_query(record, query)
        )

    async def summarize_chain(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> ImprovementRuntimeChainSummary | None:
        records = await self.load_records(query)
        if not records:
            return None
        last = records[-1]
        return ImprovementRuntimeChainSummary(
            session_id=last.session_id,
            runtime_session_id=last.runtime_session_id,
            record_count=len(records),
            status="completed",
            last_record_id=last.record_id,
            last_sequence=last.sequence,
            last_record_kind=last.record_kind,
        )


def _record_matches_query(
    record: ImprovementRuntimeRecord,
    query: ImprovementRuntimeRecordQuery,
) -> bool:
    for field_name, expected in (
        ("session_id", query.session_id),
        ("run_id", query.run_id),
        ("proposal_id", query.proposal_id),
        ("candidate_id", query.candidate_id),
    ):
        if expected is not None and getattr(record, field_name) != expected:
            return False
    return True


@dataclass
class ReplacingRecordStore(FakeRecordStore):
    async def append_record(
        self,
        record: ImprovementRuntimeRecord,
    ) -> ImprovementRuntimeRecord:
        self.records.append(record)
        return replace(record, session_id="other-session")


class NoSummaryRecordStore(FakeRecordStore):
    async def summarize_chain(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> ImprovementRuntimeChainSummary | None:
        _ = query
        return None


@dataclass
class SummaryOnlyRecordStore(FakeRecordStore):
    summary: ImprovementRuntimeChainSummary | None = None

    async def load_records(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> tuple[ImprovementRuntimeRecord, ...]:
        self.load_requests.append(query)
        return ()

    async def summarize_chain(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> ImprovementRuntimeChainSummary | None:
        _ = query
        return self.summary


def _empty_observability_events() -> list[ImprovementRuntimeObservabilityEvent]:
    return []


@dataclass
class FakeImprovementRuntimeObserver:
    events: list[ImprovementRuntimeObservabilityEvent] = field(
        default_factory=_empty_observability_events
    )
    fail: bool = False

    def emit_transition(self, event: ImprovementRuntimeObservabilityEvent) -> None:
        if self.fail:
            raise RuntimeError("observer unavailable")
        self.events.append(event)


def _empty_kernel_events() -> list[KernelEvent]:
    return []


@dataclass
class CapturingKernelEventSink:
    events: list[KernelEvent] = field(default_factory=_empty_kernel_events)

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)


def _empty_worktree_calls() -> list[tuple[str, str]]:
    return []


@dataclass
class FakeWorktreeManager:
    create_calls: list[tuple[str, str]] = field(default_factory=_empty_worktree_calls)

    async def create_agent_worktree(self, slug: str, *, cwd: str) -> WorktreeInfo:
        self.create_calls.append((slug, cwd))
        return WorktreeInfo(
            path=f"/tmp/raygent/{slug}",
            branch=f"worktree-{slug}",
            head_commit="base-1",
            git_root="/repo",
            slug=slug,
            created_at=111.0,
            touched_at=112.0,
            cleanup_policy="remove_if_clean",
        )

    async def has_changes(self, info: WorktreeInfo) -> bool:
        _ = info
        return False

    async def cleanup(
        self,
        info: WorktreeInfo,
        *,
        keep: bool | None = None,
    ) -> WorktreeCleanupResult:
        _ = (info, keep)
        return WorktreeCleanupResult(kept=False, reason="removed")


def _empty_search_requests() -> list[TranscriptSearchRequest]:
    return []


@dataclass
class FakeTranscriptSearchService:
    result: TranscriptSearchResult
    requests: list[TranscriptSearchRequest] = field(default_factory=_empty_search_requests)

    async def search(self, request: TranscriptSearchRequest) -> TranscriptSearchResult:
        self.requests.append(request)
        return self.result


def _empty_task_reads() -> list[tuple[str, str, int, int | None]]:
    return []


def _empty_tail_results() -> dict[str, TaskOutputReadResult]:
    return {}


def _empty_range_results() -> dict[tuple[str, int], TaskOutputReadResult]:
    return {}


@dataclass
class FakeTaskOutputStore:
    tail_results: dict[str, TaskOutputReadResult] = field(
        default_factory=_empty_tail_results
    )
    range_results: dict[tuple[str, int], TaskOutputReadResult] = field(
        default_factory=_empty_range_results
    )
    reads: list[tuple[str, str, int, int | None]] = field(default_factory=_empty_task_reads)

    async def init_task_output(self, task_id: str) -> TaskOutputReference:
        return TaskOutputReference(task_id=task_id, store_kind="fake")

    async def append_task_output(self, task_id: str, chunk: bytes) -> None:
        _ = (task_id, chunk)

    async def flush_task_output(self, task_id: str) -> None:
        _ = task_id

    async def evict_task_output(self, task_id: str) -> None:
        _ = task_id

    async def cleanup_task_output(self, task_id: str) -> None:
        _ = task_id

    async def read_tail(
        self,
        task_id: str,
        *,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> TaskOutputReadResult:
        self.reads.append(("tail", task_id, max_bytes, None))
        return self.tail_results.get(task_id, _empty_task_output(task_id))

    async def read_range(
        self,
        task_id: str,
        *,
        offset: int,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> TaskOutputReadResult:
        self.reads.append(("range", task_id, max_bytes, offset))
        return self.range_results.get((task_id, offset), _empty_task_output(task_id))

    async def size(self, task_id: str) -> int:
        read = self.tail_results.get(task_id)
        return 0 if read is None else read.bytes_total


def _empty_task_output(task_id: str) -> TaskOutputReadResult:
    return TaskOutputReadResult(
        task_id=task_id,
        content=b"",
        start_offset=0,
        bytes_read=0,
        bytes_total=0,
        next_offset=0,
    )


def _runtime_improvement_target() -> ImprovementTarget:
    return ImprovementTarget(
        target_id="src/raygent_harness/services/improvement_runtime.py",
        kind="source_code",
        description="Improvement runtime permission bridge",
    )


def _runtime_evaluation_plan() -> ImprovementEvaluationPlan:
    return ImprovementEvaluationPlan(
        checks=(
            ImprovementEvaluationCheck(
                name="improvement-runtime",
                instruction="Verify runtime bridge behavior.",
            ),
        ),
        success_criteria=("Runtime bridge behavior is bounded.",),
    )


def _runtime_candidate_plan() -> ImprovementPatchCandidatePlan:
    return ImprovementPatchCandidatePlan(
        candidate_id="ipc_runtime",
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        target=_runtime_improvement_target(),
        base_revision="base-1",
        summary="Plan runtime bridge work.",
        planned_changes=("Add permission bridge behavior.",),
        expected_files=("src/raygent_harness/services/improvement_runtime.py",),
        required_permissions=("filesystem_mutation", "worktree"),
        evaluation_plan=_runtime_evaluation_plan(),
        rollback_plan="Discard the candidate worktree.",
    )


def _permission_summary_metadata(
    report: ImprovementRuntimePermissionReport,
) -> Mapping[str, FrozenJson]:
    summary = report.to_summary()
    return cast(
        Mapping[str, FrozenJson],
        {"permission_summary": improvement_runtime_permission_summary_to_dict(summary)},
    )


def test_collection_bounds_compose_with_proposal_bounds() -> None:
    bounds = ImprovementEvidenceCollectionBounds(
        max_items=2,
        max_total_chars=500,
        proposal_evidence_bounds=ImprovementEvidenceBounds(
            max_items=2,
            max_item_text_chars=400,
            max_total_text_chars=500,
        ),
    )

    assert bounds.max_items == 2
    assert bounds.max_total_chars == 500

    with pytest.raises(ImprovementRuntimeValidationError, match="max_items"):
        ImprovementEvidenceCollectionBounds(
            max_items=3,
            proposal_evidence_bounds=ImprovementEvidenceBounds(
                max_items=2,
                max_item_text_chars=400,
                max_total_text_chars=1_000,
            ),
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="max_total_chars"):
        ImprovementEvidenceCollectionBounds(
            max_total_chars=1_001,
            proposal_evidence_bounds=ImprovementEvidenceBounds(
                max_items=12,
                max_item_text_chars=400,
                max_total_text_chars=1_000,
            ),
        )


def test_collection_validation_enforces_named_source_bounds() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="max_excerpt_chars"):
        validate_improvement_evidence_collection(
            (
                _evidence(
                    excerpt="x" * (DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS + 1)
                ),
            )
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="max_item_metadata_chars"):
        validate_improvement_evidence_collection(
            (
                _evidence(
                    metadata={
                        "large": "x"
                        * (DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS + 1)
                    }
                ),
            )
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="warnings"):
        validate_improvement_evidence_collection(
            (_evidence(),),
            warnings=tuple(
                f"warning {index}"
                for index in range(DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS + 1)
            ),
        )


def test_collection_request_and_result_are_serializable_contracts() -> None:
    request = _collection_request()
    result = ImprovementEvidenceCollectionResult(
        request_id=request.request_id,
        session_id=request.session_id,
        runtime_session_id=request.runtime_session_id,
        source_id="fake-source",
        evidence=(_evidence(),),
        warnings=("bounded",),
        metadata={"safe": True},
    )
    snapshot = improvement_evidence_collection_result_to_dict(result)

    json.dumps(snapshot)

    assert request.source_kinds == ("transcript", "task_output")
    assert improvement_evidence_collection_result_from_dict(snapshot) == result


@pytest.mark.asyncio
async def test_transcript_adapter_maps_search_matches_to_bounded_evidence() -> None:
    search = FakeTranscriptSearchService(
        TranscriptSearchResult(
            matches=(
                TranscriptSearchMatch(
                    session_id="session-1",
                    runtime_session_id="runtime-1",
                    entry_id="tr_1",
                    role="assistant",
                    snippet="bounded regression symptom",
                    score=42,
                    order=0,
                    created_at=123.0,
                    agent_id="agent-1",
                    is_sidechain=True,
                    source_path="/not/exported/transcript.jsonl",
                    snippet_truncated=True,
                ),
            ),
            scanned_entry_count=5,
            matched_entry_count=2,
            dropped_match_count=1,
            truncated=True,
        )
    )
    bounds = ImprovementEvidenceCollectionBounds(
        max_items=2,
        max_excerpt_chars=80,
        max_total_chars=800,
        proposal_evidence_bounds=ImprovementEvidenceBounds(
            max_items=2,
            max_item_text_chars=800,
            max_total_text_chars=800,
        ),
    )
    request = ImprovementEvidenceCollectionRequest(
        request_id="ier_transcript",
        session_id="session-1",
        runtime_session_id="runtime-1",
        source_kinds=("transcript",),
        query="regression",
        bounds=bounds,
    )

    result = await TranscriptSearchImprovementEvidenceAdapter(
        cast(TranscriptSearchService, search),
        roles=("assistant",),
        sidechain_agent_ids=("agent-1",),
        include_main=False,
    ).collect(request)

    assert result.request_id == request.request_id
    assert result.runtime_session_id == "runtime-1"
    assert len(search.requests) == 1
    assert search.requests[0].scope.session_id == "session-1"
    assert search.requests[0].scope.runtime_session_id == "runtime-1"
    assert search.requests[0].scope.include_main is False
    assert search.requests[0].scope.sidechain_agent_ids == ("agent-1",)
    assert search.requests[0].max_results == 2
    assert search.requests[0].max_snippet_chars == 80
    assert search.requests[0].max_total_snippet_chars == 400
    assert result.evidence[0].source == "transcript"
    assert result.evidence[0].evidence_id == "iev_transcript_tr_1"
    assert result.evidence[0].source_uri == "transcript://session-1/tr_1"
    assert result.evidence[0].metadata["entry_id"] == "tr_1"
    assert "source_path" not in result.evidence[0].metadata
    assert result.truncated is True
    assert result.warnings == (
        "transcript search dropped 1 matches",
        "transcript search results were truncated",
    )


@pytest.mark.asyncio
async def test_transcript_adapter_fails_closed_for_filters_query_and_tight_bounds() -> None:
    search = FakeTranscriptSearchService(TranscriptSearchResult())
    adapter = TranscriptSearchImprovementEvidenceAdapter(cast(TranscriptSearchService, search))

    skipped = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_skip",
            session_id="session-1",
            source_kinds=("task_output",),
            query="ignored",
        )
    )
    missing_query = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_missing_query",
            session_id="session-1",
            source_kinds=("transcript",),
        )
    )
    tight_bounds = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_tight",
            session_id="session-1",
            source_kinds=("transcript",),
            query="regression",
            bounds=ImprovementEvidenceCollectionBounds(
                max_items=1,
                max_excerpt_chars=8,
                max_total_chars=100,
                proposal_evidence_bounds=ImprovementEvidenceBounds(
                    max_items=1,
                    max_item_text_chars=100,
                    max_total_text_chars=100,
                ),
            ),
        )
    )

    assert skipped.evidence == ()
    assert skipped.warnings == ()
    assert missing_query.evidence == ()
    assert missing_query.warnings == ("transcript evidence query is required",)
    assert tight_bounds.evidence == ()
    assert "max_snippet_chars >= 16" in tight_bounds.warnings[0]
    assert search.requests == []


@pytest.mark.asyncio
async def test_task_output_adapter_requires_targets_and_reads_only_explicit_bounds() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="explicit targets"):
        TaskOutputImprovementEvidenceAdapter(cast(TaskOutputStore, FakeTaskOutputStore()), ())

    store = FakeTaskOutputStore(
        tail_results={
            "task-a": TaskOutputReadResult(
                task_id="task-a",
                content=b"older line\ncurrent failure",
                start_offset=5,
                bytes_read=25,
                bytes_total=30,
                next_offset=30,
                truncated_before=True,
            )
        },
        range_results={
            ("task-b", 4): TaskOutputReadResult(
                task_id="task-b",
                content=b"range bytes",
                start_offset=4,
                bytes_read=11,
                bytes_total=40,
                next_offset=15,
                truncated_before=True,
                truncated_after=True,
            )
        },
    )
    adapter = TaskOutputImprovementEvidenceAdapter(
        cast(TaskOutputStore, store),
        (
            ImprovementTaskOutputEvidenceTarget("task-a", max_bytes=12),
            ImprovementTaskOutputEvidenceTarget(
                "task-b",
                mode="range",
                offset=4,
                metadata={"kind": "verification-log"},
            ),
        ),
    )

    result = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_task_output",
            session_id="session-1",
            runtime_session_id="runtime-1",
            source_kinds=("task_output",),
            bounds=ImprovementEvidenceCollectionBounds(max_excerpt_chars=20),
        )
    )

    assert store.reads == [("tail", "task-a", 12, None), ("range", "task-b", 20, 4)]
    assert result.runtime_session_id == "runtime-1"
    assert [item.source for item in result.evidence] == ["task_output", "task_output"]
    assert result.evidence[0].excerpt == "older line\ncurrent f"
    assert result.evidence[0].metadata["truncated_before"] is True
    assert result.evidence[0].metadata["truncated_after"] is True
    assert result.evidence[0].source_uri == "task-output://task-a?start=5&next=30"
    assert result.evidence[1].metadata["target_metadata"] == {
        "kind": "verification-log"
    }
    assert result.truncated is True


@pytest.mark.asyncio
async def test_task_output_adapter_honors_source_filters_and_missing_output() -> None:
    store = FakeTaskOutputStore()
    adapter = TaskOutputImprovementEvidenceAdapter(
        cast(TaskOutputStore, store),
        (ImprovementTaskOutputEvidenceTarget("missing-task"),),
    )

    skipped = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_skip_task",
            session_id="session-1",
            source_kinds=("transcript",),
            query="ignored",
        )
    )
    missing = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_missing_task",
            session_id="session-1",
            source_kinds=("task_output",),
        )
    )

    assert skipped.evidence == ()
    assert store.reads == [("tail", "missing-task", 1000, None)]
    assert missing.evidence == ()
    assert missing.warnings == ("task output empty or unavailable: missing-task",)


@pytest.mark.asyncio
async def test_observability_snapshot_adapter_accepts_metadata_only_snapshots() -> None:
    adapter = ObservabilitySnapshotImprovementEvidenceAdapter(
        (
            ImprovementObservabilitySnapshot(
                event_id="event-1",
                event_type="model_usage",
                summary="Model usage crossed the configured budget.",
                created_at=456.0,
                metadata={
                    "tokens": 1_200,
                    "tool_result": {
                        "redacted": True,
                        "summary": "raw tool result intentionally omitted",
                    },
                },
            ),
        )
    )

    result = await adapter.collect(
        ImprovementEvidenceCollectionRequest(
            request_id="ier_observability",
            session_id="session-1",
            runtime_session_id="runtime-1",
            source_kinds=("observability",),
        )
    )

    assert result.runtime_session_id == "runtime-1"
    assert result.evidence[0].source == "observability"
    assert result.evidence[0].source_uri == "observability://model_usage/event-1"
    assert result.evidence[0].created_at == 456.0
    assert result.evidence[0].metadata["event_type"] == "model_usage"


def test_observability_snapshot_rejects_raw_looking_content_fields() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="content"):
        ImprovementObservabilitySnapshot(
            event_id="event-1",
            event_type="message",
            summary="Raw content should not enter improvement evidence.",
            metadata={"content": "full prompt or response"},
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="transcript"):
        ImprovementObservabilitySnapshot(
            event_id="event-2",
            event_type="message",
            summary="Nested raw transcript should not enter improvement evidence.",
            metadata={"nested": {"transcript": "full transcript"}},
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="tool_result"):
        ImprovementObservabilitySnapshot(
            event_id="event-3",
            event_type="tool",
            summary="Redaction markers must not smuggle raw payload fields.",
            metadata={
                "tool_result": {
                    "redacted": True,
                    "summary": "raw payload omitted",
                    "content": "raw tool output",
                }
            },
        )


def test_runtime_request_adopts_collection_runtime_session_id() -> None:
    request = ImprovementRuntimeRequest(
        request_id="irt_1",
        session_id="session-1",
        collection_request=_collection_request(),
    )

    assert request.runtime_session_id == "runtime-1"

    with pytest.raises(ImprovementRuntimeValidationError, match="runtime_session_id"):
        ImprovementRuntimeRequest(
            request_id="irt_2",
            session_id="session-1",
            runtime_session_id="other-runtime",
            collection_request=_collection_request(),
        )


def test_runtime_record_is_versioned_immutable_and_recovery_friendly() -> None:
    record = ImprovementRuntimeRecord(
        record_id="irtr_1",
        record_kind="evidence_collected",
        session_id="session-1",
        runtime_session_id="runtime-1",
        run_id="ir_1",
        proposal_id="ip_1",
        candidate_id="ipc_1",
        stage_id="evidence",
        sequence=1,
        payload={"evidence_count": 1},
        warnings=("truncated",),
        stop_reason="caller requested one transition only",
        created_at=200.0,
        metadata={"phase": "rsi-006a"},
    )

    assert record.schema_version == IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION
    assert record.payload == {"evidence_count": 1}
    with pytest.raises(FrozenInstanceError):
        record.sequence = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.payload["evidence_count"] = 2  # type: ignore[index]

    snapshot = improvement_runtime_record_to_dict(record)
    json.dumps(snapshot)

    assert improvement_runtime_record_from_dict(snapshot) == record


def test_runtime_record_requires_payload_or_payload_ref() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="payload"):
        ImprovementRuntimeRecord(
            record_id="irtr_1",
            record_kind="runtime_blocked",
            session_id="session-1",
            sequence=1,
        )

    record = ImprovementRuntimeRecord(
        record_id="irtr_2",
        record_kind="runtime_blocked",
        session_id="session-1",
        sequence=2,
        payload_ref="task-output://session-1/improvement-records/irtr_2.json",
        stop_reason="record payload is stored externally",
    )

    assert record.payload_ref == "task-output://session-1/improvement-records/irtr_2.json"


def test_runtime_record_and_summary_validate_schema_and_bounds() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="schema_version"):
        ImprovementRuntimeRecord(
            record_id="irtr_1",
            record_kind="evidence_collected",
            session_id="session-1",
            sequence=1,
            payload={"ok": True},
            schema_version=99,
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="metadata"):
        ImprovementRuntimeRecord(
            record_id="irtr_1",
            record_kind="evidence_collected",
            session_id="session-1",
            sequence=1,
            payload={"ok": True},
            metadata={
                "large": "x" * (DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS + 1)
            },
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="max_records"):
        ImprovementRuntimeRecordQuery(
            session_id="session-1",
            max_records=DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS + 1,
        )


def test_runtime_chain_summary_round_trips() -> None:
    summary = ImprovementRuntimeChainSummary(
        session_id="session-1",
        runtime_session_id="runtime-1",
        run_id="ir_1",
        proposal_id="ip_1",
        candidate_id="ipc_1",
        record_count=3,
        status="blocked",
        last_record_id="irtr_3",
        last_sequence=3,
        last_record_kind="verification_recorded",
        next_record_kind="outcome_derived",
        blocked_reason="outcome derivation input is missing",
        warnings=("operator review required",),
        metadata={"source": "store-summary"},
    )
    snapshot = improvement_runtime_chain_summary_to_dict(summary)

    json.dumps(snapshot)

    assert improvement_runtime_chain_summary_from_dict(snapshot) == summary


@pytest.mark.asyncio
async def test_recover_runtime_chain_orders_records_and_reports_last_completed() -> None:
    permission_report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="verification",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="verification",
                record_kind="verification_recorded",
                required_permissions=("filesystem_mutation", "shell"),
            ),
        ),
        approved_permissions=("filesystem_mutation",),
    )
    permission_summary_metadata = cast(
        Mapping[str, FrozenJson],
        {
            "permission_summary": improvement_runtime_permission_summary_to_dict(
                permission_report.to_summary()
            )
        },
    )
    store = NoSummaryRecordStore(
        records=[
            ImprovementRuntimeRecord(
                record_id="irtr_2",
                record_kind="verification_recorded",
                session_id="session-1",
                runtime_session_id="runtime-1",
                run_id="ir_1",
                sequence=2,
                payload={"stage": "verification"},
                created_at=20.0,
            ),
            ImprovementRuntimeRecord(
                record_id="irtr_1",
                record_kind="evidence_collected",
                session_id="session-1",
                runtime_session_id="runtime-1",
                run_id="ir_1",
                sequence=1,
                payload={"stage": "evidence"},
                warnings=("evidence truncated",),
                created_at=10.0,
            ),
            ImprovementRuntimeRecord(
                record_id="irtr_3",
                record_kind="runtime_blocked",
                session_id="session-1",
                runtime_session_id="runtime-1",
                run_id="ir_1",
                sequence=3,
                payload={"stage": "promotion"},
                stop_reason="verification approval is incomplete",
                metadata=permission_summary_metadata,
                created_at=30.0,
            ),
        ]
    )

    result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_1",
            record_store=store,
            query=ImprovementRuntimeRecordQuery(run_id="ir_1"),
            expected_session_id="session-1",
        )
    )

    assert result.status == "recovered"
    assert [record.record_id for record in result.records] == [
        "irtr_1",
        "irtr_2",
        "irtr_3",
    ]
    assert result.last_record_id == "irtr_3"
    assert result.last_record_kind == "runtime_blocked"
    assert result.last_completed_record_id == "irtr_2"
    assert result.last_completed_record_kind == "verification_recorded"
    assert result.summary is not None
    assert result.summary.status == "blocked"
    assert result.permission_summary is not None
    assert result.permission_summary.status == "blocked"
    assert store.append_requests == []
    assert "evidence truncated" in result.warnings

    snapshot = improvement_runtime_recovery_result_to_dict(result)
    json.dumps(snapshot)

    assert improvement_runtime_recovery_result_from_dict(snapshot) == result


@pytest.mark.asyncio
async def test_recover_runtime_chain_fallback_summary_does_not_duplicate_record_warnings() -> None:
    store = NoSummaryRecordStore(
        records=[
            ImprovementRuntimeRecord(
                record_id="irtr_1",
                record_kind="evidence_collected",
                session_id="session-1",
                run_id="ir_1",
                sequence=1,
                payload={"stage": "evidence"},
                warnings=tuple(
                    f"warning-{index}"
                    for index in range(DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS)
                ),
            )
        ]
    )

    result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_1",
            record_store=store,
            query=ImprovementRuntimeRecordQuery(run_id="ir_1"),
            expected_session_id="session-1",
        )
    )

    assert result.status == "recovered"
    assert result.summary is not None
    assert result.summary.warnings == ()
    assert len(result.warnings) == DEFAULT_MAX_IMPROVEMENT_RUNTIME_WARNINGS
    assert result.warnings == store.records[0].warnings


@pytest.mark.asyncio
async def test_recover_runtime_chain_blocks_expected_session_mismatch_before_store() -> None:
    store = FakeRecordStore()

    result = await ImprovementRuntimeRecoveryService().recover(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_1",
            record_store=store,
            query=ImprovementRuntimeRecordQuery(session_id="other-session"),
            expected_session_id="session-1",
        )
    )

    assert result.status == "blocked"
    assert result.records == ()
    assert result.session_id == "session-1"
    assert store.load_requests == []
    assert "expected_session_id" in result.warnings[0]


@pytest.mark.asyncio
async def test_recover_runtime_chain_reports_not_found_without_records_or_summary() -> None:
    store = FakeRecordStore()

    result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_1",
            record_store=store,
            query=ImprovementRuntimeRecordQuery(run_id="missing-run"),
            expected_session_id="session-1",
        )
    )

    assert result.status == "not_found"
    assert result.records == ()
    assert result.summary is None
    assert result.session_id == "session-1"


@pytest.mark.asyncio
async def test_recover_runtime_chain_last_completed_handles_blocked_only_and_summary() -> None:
    blocked_store = NoSummaryRecordStore(
        records=[
            ImprovementRuntimeRecord(
                record_id="irtr_2",
                record_kind="runtime_not_enabled",
                session_id="session-1",
                run_id="ir_1",
                sequence=2,
                payload={"stage": "disabled"},
                created_at=20.0,
            ),
            ImprovementRuntimeRecord(
                record_id="irtr_1",
                record_kind="runtime_blocked",
                session_id="session-1",
                run_id="ir_1",
                sequence=2,
                payload={"stage": "blocked"},
                created_at=10.0,
            ),
        ]
    )
    blocked_result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_blocked",
            record_store=blocked_store,
            query=ImprovementRuntimeRecordQuery(run_id="ir_1"),
            expected_session_id="session-1",
        )
    )

    assert blocked_result.status == "recovered"
    assert blocked_result.last_completed_record_id is None
    assert blocked_result.last_completed_record_kind is None
    assert any("duplicate" in warning for warning in blocked_result.warnings)

    summary_store = SummaryOnlyRecordStore(
        summary=ImprovementRuntimeChainSummary(
            session_id="session-1",
            run_id="ir_2",
            record_count=2,
            status="completed",
            last_record_id="irtr_summary",
            last_sequence=2,
            last_record_kind="gate_evaluated",
        )
    )
    summary_result = await recover_improvement_runtime_chain(
        ImprovementRuntimeRecoveryRequest(
            request_id="irr_summary",
            record_store=summary_store,
            query=ImprovementRuntimeRecordQuery(run_id="ir_2"),
            expected_session_id="session-1",
        )
    )

    assert summary_result.status == "recovered"
    assert summary_result.records == ()
    assert summary_result.last_record_id == "irtr_summary"
    assert summary_result.last_completed_record_id == "irtr_summary"
    assert summary_result.last_completed_record_kind == "gate_evaluated"


def test_permission_requirement_and_policy_report_advisory_statuses() -> None:
    requirement = ImprovementRuntimePermissionRequirement(
        stage_id="verification",
        record_kind="verification_recorded",
        required_permissions=("filesystem_mutation", "shell"),
        reason="verification runner would need local shell access",
        metadata={"source": "verification-plan"},
    )
    policy = ImprovementRuntimePermissionPolicy()

    required = policy.evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="verification",
        requirements=(requirement,),
    )
    approved = policy.evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="verification",
        requirements=(requirement,),
        approved_permissions=("filesystem_mutation", "shell"),
    )
    blocked = policy.evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="verification",
        requirements=(requirement,),
        approved_permissions=("filesystem_mutation", "shell", "commit"),
    )
    not_required = policy.evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="observe",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="observe",
                required_permissions=("none",),
            ),
        ),
    )

    assert required.status == "required"
    assert required.missing_permissions == ("filesystem_mutation", "shell")
    assert approved.status == "approved"
    assert blocked.status == "blocked"
    assert blocked.extra_permissions == ("commit",)
    assert not_required.status == "not_required"

    snapshot = improvement_runtime_permission_report_to_dict(approved)
    json.dumps(snapshot)

    assert improvement_runtime_permission_report_from_dict(snapshot) == approved


def test_permission_requirement_rejects_duplicates_and_none_mixing() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="duplicates"):
        ImprovementRuntimePermissionRequirement(
            stage_id="verification",
            required_permissions=("shell", "shell"),
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="cannot mix none"):
        ImprovementRuntimePermissionRequirement(
            stage_id="verification",
            required_permissions=("none", "shell"),
        )


def test_permission_report_summary_omits_approval_adjacent_facts() -> None:
    report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="archive",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="archive",
                record_kind="archive_recorded",
                required_permissions=("filesystem_mutation",),
                reason="archive persistence needs a caller-owned store",
                metadata={"caller_note": "return-only report metadata"},
            ),
        ),
        approved_permissions=("filesystem_mutation", "commit"),
        metadata={"approver_identity": "return-only, not durable metadata"},
    )

    summary = report.to_summary()
    snapshot = improvement_runtime_permission_summary_to_dict(summary)
    serialized = json.dumps(snapshot)

    assert summary.status == "blocked"
    assert summary.required_permission_count == 1
    assert summary.missing_permission_count == 0
    assert summary.extra_permission_count == 1
    assert summary.requirement_labels == ("archive:archive_recorded",)
    assert "filesystem_mutation" not in serialized
    assert "commit" not in serialized
    assert "approver_identity" not in serialized
    assert "caller_note" not in serialized


def test_runtime_record_rejects_direct_full_permission_report_metadata() -> None:
    report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        stage_id="verification",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="verification",
                required_permissions=("filesystem_mutation", "shell"),
            ),
        ),
        approved_permissions=("filesystem_mutation", "shell"),
        metadata={"approver_identity": "operator-a"},
    )

    with pytest.raises(ImprovementRuntimeValidationError, match="permission_report"):
        ImprovementRuntimeRecord(
            record_id="irtr_1",
            record_kind="runtime_blocked",
            session_id="session-1",
            sequence=1,
            payload={"request_id": "irt_1", "status": "blocked"},
            metadata=cast(
                Mapping[str, FrozenJson],
                {
                    "permission_report": improvement_runtime_permission_report_to_dict(
                        report
                    )
                },
            ),
        )

    with pytest.raises(ImprovementRuntimeValidationError, match="permission_summary"):
        ImprovementRuntimeRecord(
            record_id="irtr_2",
            record_kind="runtime_blocked",
            session_id="session-1",
            sequence=1,
            payload={"request_id": "irt_1", "status": "blocked"},
            metadata=cast(
                Mapping[str, FrozenJson],
                {
                    "permission_summary": {
                        "stage_id": "verification",
                        "status": "approved",
                        "supplied_approved_permissions": (
                            "filesystem_mutation",
                            "shell",
                        ),
                    }
                },
            ),
        )

    record = ImprovementRuntimeRecord(
        record_id="irtr_3",
        record_kind="runtime_blocked",
        session_id="session-1",
        sequence=1,
        payload={"request_id": "irt_1", "status": "blocked"},
        metadata=_permission_summary_metadata(report),
    )

    assert record.metadata["permission_summary"] == {
        "stage_id": "verification",
        "status": "approved",
        "required_permission_count": 2,
        "missing_permission_count": 0,
        "extra_permission_count": 0,
        "requirement_labels": ("verification",),
    }


@pytest.mark.asyncio
async def test_permission_report_is_advisory_not_stage_authorization() -> None:
    plan = _runtime_candidate_plan()
    manager = FakeWorktreeManager()
    report_with_extra = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        stage_id="worktree",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="worktree",
                record_kind="worktree_allocated",
                required_permissions=("filesystem_mutation", "worktree"),
            ),
        ),
        approved_permissions=("filesystem_mutation", "worktree", "commit"),
    )

    assert report_with_extra.status == "blocked"

    allocation = await ImprovementPatchCandidateWorktreeAllocator(
        allocation_id_factory=lambda: "ipcw_1",
    ).allocate(
        plan,
        manager=cast(WorktreeManager, manager),
        cwd="/repo",
        approval=ImprovementPatchCandidateWorktreeApproval(
            approved_permissions=("filesystem_mutation", "worktree", "commit"),
            reason="stage approval permits valid extras for this stage",
        ),
        metadata=_permission_summary_metadata(report_with_extra),
    )

    assert allocation.status == "allocated"
    assert manager.create_calls == [(allocation.worktree_slug, "/repo")]

    approved_report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_2",
        session_id="session-1",
        stage_id="worktree",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="worktree",
                record_kind="worktree_allocated",
                required_permissions=("filesystem_mutation", "worktree"),
            ),
        ),
        approved_permissions=("filesystem_mutation", "worktree"),
    )

    assert approved_report.status == "approved"
    with pytest.raises(
        ImprovementPatchCandidateWorktreeValidationError,
        match="call-time approval",
    ):
        await ImprovementPatchCandidateWorktreeAllocator().allocate(
            plan,
            manager=cast(WorktreeManager, FakeWorktreeManager()),
            cwd="/repo",
            approval=None,
            metadata=_permission_summary_metadata(approved_report),
        )


@pytest.mark.asyncio
async def test_bridge_default_disabled_returns_not_enabled_without_adapters() -> None:
    result = await ImprovementRuntimeBridge(
        record_id_factory=lambda: "irtr_disabled"
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
        )
    )

    assert result.status == "not_enabled"
    assert result.blocked_reason == "improvement runtime bridge is not enabled"
    assert result.records[0].record_kind == "runtime_not_enabled"
    assert result.records[0].stop_reason == result.blocked_reason


@pytest.mark.asyncio
async def test_bridge_blocks_when_enabled_without_evidence_sources() -> None:
    result = await ImprovementRuntimeBridge(
        ImprovementRuntimeBridgeConfig(enabled=True),
        record_id_factory=lambda: "irtr_blocked",
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
        )
    )

    assert result.status == "blocked"
    assert result.blocked_reason == "no improvement evidence source adapters were supplied"
    assert result.records[0].record_kind == "runtime_blocked"


@pytest.mark.asyncio
async def test_bridge_collects_via_injected_adapter_and_store_only() -> None:
    adapter = FakeEvidenceSource()
    store = FakeRecordStore()
    record_ids = iter(("irtr_evidence",))

    result = await ImprovementRuntimeBridge(
        ImprovementRuntimeBridgeConfig(enabled=True, record_store=store),
        clock=lambda: 300.0,
        record_id_factory=lambda: next(record_ids),
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
            evidence_sources=(adapter,),
            metadata={"caller": "unit-test"},
        )
    )

    assert result.status == "completed"
    assert result.evidence == adapter.evidence
    assert result.warnings == ("source truncated",)
    assert result.records[0].record_kind == "evidence_collected"
    assert result.records[0].payload["evidence_count"] == 1
    assert result.records[0].payload["truncated"] is True
    assert result.records[0].created_at == 300.0
    assert result.summary is not None
    assert result.summary.status == "completed"
    assert result.summary.last_record_kind == "evidence_collected"
    assert adapter.requests == [_collection_request()]
    assert store.records == [result.records[0]]


@pytest.mark.asyncio
async def test_bridge_returns_permission_report_but_stores_only_sanitized_summary() -> None:
    adapter = FakeEvidenceSource()
    store = FakeRecordStore()
    permission_report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="verification",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="verification",
                record_kind="verification_recorded",
                required_permissions=("filesystem_mutation", "shell"),
                reason="verification runner requires explicit local approval",
                metadata={"return_only": "requirement metadata"},
            ),
        ),
        approved_permissions=("filesystem_mutation", "shell", "commit"),
        metadata={"approver_identity": "operator-a"},
    )

    result = await ImprovementRuntimeBridge(
        ImprovementRuntimeBridgeConfig(enabled=True, record_store=store),
        clock=lambda: 300.0,
        record_id_factory=lambda: "irtr_evidence",
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
            evidence_sources=(adapter,),
            permission_report=permission_report,
            metadata={"caller": "unit-test"},
        )
    )

    assert result.permission_report == permission_report
    report = result.permission_report
    assert report is not None
    assert report.supplied_approved_permissions == (
        "filesystem_mutation",
        "shell",
        "commit",
    )
    metadata = result.records[0].metadata
    assert metadata["caller"] == "unit-test"
    assert metadata["permission_summary"] == {
        "stage_id": "verification",
        "status": "blocked",
        "required_permission_count": 2,
        "missing_permission_count": 0,
        "extra_permission_count": 1,
        "requirement_labels": ("verification:verification_recorded",),
    }
    serialized_metadata = json.dumps(
        improvement_runtime_record_to_dict(result.records[0])["metadata"]
    )
    assert "supplied_approved_permissions" not in serialized_metadata
    assert "approver_identity" not in serialized_metadata
    assert "filesystem_mutation" not in serialized_metadata
    assert "commit" not in serialized_metadata


@pytest.mark.asyncio
async def test_bridge_rejects_permission_report_smuggling_in_record_metadata() -> None:
    with pytest.raises(ImprovementRuntimeValidationError, match="approved_permissions"):
        await ImprovementRuntimeBridge(
            ImprovementRuntimeBridgeConfig(enabled=True),
            record_id_factory=lambda: "irtr_evidence",
        ).collect_evidence(
            ImprovementRuntimeRequest(
                request_id="irt_1",
                session_id="session-1",
                runtime_session_id="runtime-1",
                collection_request=_collection_request(),
                evidence_sources=(FakeEvidenceSource(),),
                metadata={
                    "nested": {
                        "approved_permissions": (
                            "filesystem_mutation",
                            "shell",
                        )
                    }
                },
            )
        )


@pytest.mark.asyncio
async def test_bridge_emits_metadata_only_observability_event() -> None:
    observer = FakeImprovementRuntimeObserver()
    permission_report = ImprovementRuntimePermissionPolicy().evaluate(
        request_id="irt_1",
        session_id="session-1",
        runtime_session_id="runtime-1",
        stage_id="observe",
        requirements=(
            ImprovementRuntimePermissionRequirement(
                stage_id="observe",
                required_permissions=("none",),
            ),
        ),
    )

    result = await ImprovementRuntimeBridge(
        ImprovementRuntimeBridgeConfig(
            enabled=True,
            observability_sink=cast(ImprovementRuntimeObservabilitySink, observer),
        ),
        record_id_factory=lambda: "irtr_evidence",
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
            evidence_sources=(FakeEvidenceSource(),),
            permission_report=permission_report,
        )
    )

    assert result.status == "completed"
    assert observer.events == (
        [
            ImprovementRuntimeObservabilityEvent(
                request_id="irt_1",
                session_id="session-1",
                runtime_session_id="runtime-1",
                status="completed",
                record_ids=("irtr_evidence",),
                last_record_kind="evidence_collected",
                evidence_count=1,
                warning_count=1,
                truncated=True,
                stage_id="observe",
                permission_status="not_required",
                permission_required_count=1,
                permission_missing_count=0,
                permission_extra_count=0,
            )
        ]
    )
    snapshot = improvement_runtime_observability_event_to_dict(observer.events[0])
    assert improvement_runtime_observability_event_from_dict(snapshot) == observer.events[0]
    serialized = json.dumps(snapshot)
    assert "bounded transcript excerpt" not in serialized
    assert "source_uri" not in serialized
    assert "prompt" not in serialized
    assert "tool_result" not in serialized


@pytest.mark.asyncio
async def test_bridge_observability_failures_are_warnings_not_status_changes() -> None:
    observer = FakeImprovementRuntimeObserver(fail=True)

    result = await ImprovementRuntimeBridge(
        ImprovementRuntimeBridgeConfig(
            enabled=True,
            observability_sink=cast(ImprovementRuntimeObservabilitySink, observer),
        ),
        record_id_factory=lambda: "irtr_evidence",
    ).collect_evidence(
        ImprovementRuntimeRequest(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            collection_request=_collection_request(),
            evidence_sources=(FakeEvidenceSource(),),
        )
    )

    assert result.status == "completed"
    assert result.warnings[-1] == (
        "improvement runtime observability observer failed: RuntimeError"
    )


def test_kernel_event_observer_emits_metadata_only_bus_event() -> None:
    sink = CapturingKernelEventSink()
    bus = KernelEventBus((sink,), clock=lambda: 500.0, id_factory=lambda seq: f"ev_{seq}")
    observer = KernelEventImprovementRuntimeObserver(bus)

    observer.emit_transition(
        ImprovementRuntimeObservabilityEvent(
            request_id="irt_1",
            session_id="session-1",
            runtime_session_id="runtime-1",
            status="blocked",
            record_ids=("irtr_blocked",),
            last_record_kind="runtime_blocked",
            warning_count=1,
            stage_id="verification",
            permission_status="required",
            permission_required_count=2,
            permission_missing_count=2,
        )
    )

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.type == "improvement_runtime.transition"
    assert event.content_policy == "metadata_only"
    assert event.session_id == "session-1"
    assert event.runtime_session_id == "runtime-1"
    assert event.data["record_ids"] == ("irtr_blocked",)
    assert event.data["permission_status"] == "required"


@pytest.mark.asyncio
async def test_bridge_rejects_store_that_replaces_appended_record() -> None:
    adapter = FakeEvidenceSource()
    store = ReplacingRecordStore()

    with pytest.raises(ImprovementRuntimeValidationError, match="different runtime record"):
        await ImprovementRuntimeBridge(
            ImprovementRuntimeBridgeConfig(enabled=True, record_store=store),
            record_id_factory=lambda: "irtr_evidence",
        ).collect_evidence(
            ImprovementRuntimeRequest(
                request_id="irt_1",
                session_id="session-1",
                collection_request=_collection_request(),
                evidence_sources=(adapter,),
            )
        )


def test_explicit_transition_result_supports_required_statuses() -> None:
    completed = ImprovementRuntimeTransitionResult(
        request_id="irt_1",
        session_id="session-1",
        status="completed",
    )
    blocked = ImprovementRuntimeTransitionResult(
        request_id="irt_2",
        session_id="session-1",
        status="blocked",
        blocked_reason="missing evidence",
    )
    not_enabled = ImprovementRuntimeTransitionResult(
        request_id="irt_3",
        session_id="session-1",
        status="not_enabled",
    )

    assert (completed.status, blocked.status, not_enabled.status) == (
        "completed",
        "blocked",
        "not_enabled",
    )
    with pytest.raises(ImprovementRuntimeValidationError, match="blocked_reason"):
        ImprovementRuntimeTransitionResult(
            request_id="irt_4",
            session_id="session-1",
            status="blocked",
        )


def test_contract_module_does_not_import_runtime_or_product_layers() -> None:
    source = inspect.getsource(runtime_module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    called_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    forbidden_modules = {
        "raygent_harness.core.query_engine",
        "raygent_harness.sdk",
        "raygent_harness.goals",
        "raygent_harness.core.tool_execution",
        "raygent_harness.tools",
        "subprocess",
        "pathlib",
        "socket",
        "http",
        "urllib",
    }
    forbidden_names = {
        "QueryEngine",
        "RaygentSession",
        "create_raygent",
        "GoalRuntime",
        "run_tool_use",
        "Tool",
        "ToolUseContext",
        "Path",
        "subprocess",
    }

    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
    assert called_names.isdisjoint(forbidden_names)


def test_public_module_exports_match_all() -> None:
    exported = cast(tuple[str, ...], runtime_module.__all__)

    for name in exported:
        assert hasattr(runtime_module, name)
