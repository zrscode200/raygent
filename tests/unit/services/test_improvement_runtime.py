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
from raygent_harness.improvement import ImprovementEvidence, ImprovementEvidenceBounds
from raygent_harness.services.improvement_runtime import (
    DEFAULT_MAX_EVIDENCE_COLLECTION_EXCERPT_CHARS,
    DEFAULT_MAX_EVIDENCE_COLLECTION_ITEM_METADATA_CHARS,
    DEFAULT_MAX_EVIDENCE_COLLECTION_WARNINGS,
    DEFAULT_MAX_IMPROVEMENT_RUNTIME_METADATA_CHARS,
    DEFAULT_MAX_IMPROVEMENT_RUNTIME_SUMMARY_RECORDS,
    IMPROVEMENT_RUNTIME_RECORD_SCHEMA_VERSION,
    ImprovementEvidenceCollectionBounds,
    ImprovementEvidenceCollectionRequest,
    ImprovementEvidenceCollectionResult,
    ImprovementRuntimeBridge,
    ImprovementRuntimeBridgeConfig,
    ImprovementRuntimeChainSummary,
    ImprovementRuntimeRecord,
    ImprovementRuntimeRecordQuery,
    ImprovementRuntimeRequest,
    ImprovementRuntimeTransitionResult,
    ImprovementRuntimeValidationError,
    improvement_evidence_collection_result_from_dict,
    improvement_evidence_collection_result_to_dict,
    improvement_runtime_chain_summary_from_dict,
    improvement_runtime_chain_summary_to_dict,
    improvement_runtime_record_from_dict,
    improvement_runtime_record_to_dict,
    validate_improvement_evidence_collection,
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

    async def append_record(
        self,
        record: ImprovementRuntimeRecord,
    ) -> ImprovementRuntimeRecord:
        self.records.append(record)
        return record

    async def load_records(
        self,
        query: ImprovementRuntimeRecordQuery,
    ) -> tuple[ImprovementRuntimeRecord, ...]:
        return tuple(
            record for record in self.records if record.session_id == query.session_id
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


@dataclass
class ReplacingRecordStore(FakeRecordStore):
    async def append_record(
        self,
        record: ImprovementRuntimeRecord,
    ) -> ImprovementRuntimeRecord:
        self.records.append(record)
        return replace(record, session_id="other-session")


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
