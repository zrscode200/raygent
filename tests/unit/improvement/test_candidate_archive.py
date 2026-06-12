from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import cast

import pytest

import raygent_harness.improvement.candidate_archive as archive_module
from raygent_harness.improvement import (
    DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS,
    DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS,
    DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS,
    ImprovementPatchCandidateArchiveApproval,
    ImprovementPatchCandidateArchiveDecision,
    ImprovementPatchCandidateArchiveDecisionPolicy,
    ImprovementPatchCandidateArchiver,
    ImprovementPatchCandidateArchiveRecord,
    ImprovementPatchCandidateArchiveRequest,
    ImprovementPatchCandidateArchiveStoreResult,
    ImprovementPatchCandidateArchiveValidationError,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationResult,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidateOutcome,
    ImprovementPatchCandidateOutcomePolicy,
    ImprovementPatchOperation,
    improvement_patch_candidate_archive_record_from_dict,
    improvement_patch_candidate_archive_record_to_dict,
    improvement_patch_candidate_archive_request_from_dict,
    improvement_patch_candidate_archive_request_to_dict,
    improvement_patch_candidate_archive_store_result_from_dict,
    improvement_patch_candidate_archive_store_result_to_dict,
)


def _operation() -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id="op_1",
        kind="replace_text",
        relative_path="src/raygent_harness/improvement/candidate_archive.py",
        old_text="old",
        new_text="new",
    )


def _materialization() -> ImprovementPatchCandidateMaterialization:
    operation = _operation()
    return ImprovementPatchCandidateMaterialization(
        materialization_id="ipcm_1",
        allocation_id="ipcw_1",
        candidate_id="ipc_1",
        run_id="ir_1",
        proposal_id="ip_1",
        gate_evaluation_id="ige_1",
        base_revision="b1ce8fd",
        worktree_path="/tmp/raygent/ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        operations=(operation,),
        changed_files=(operation.relative_path,),
        patch_digest="sha256:" + "2" * 64,
        created_at=500.0,
        metadata={"phase": "rsi-003c"},
    )


def _evaluation_result(status: str = "fail") -> ImprovementPatchCandidateEvaluationResult:
    return ImprovementPatchCandidateEvaluationResult(
        result_id=f"res_{status}",
        kind="unit_tests",
        status=status,  # type: ignore[arg-type]
        summary=f"unit tests {status}",
        changed_files=("src/raygent_harness/improvement/candidate_archive.py",),
        output_reference="task-output:test-candidate-archive",
        required=True,
        created_at=600.0,
    )


def _evaluation(status: str = "fail") -> ImprovementPatchCandidateEvaluation:
    materialization = _materialization()
    return ImprovementPatchCandidateEvaluation(
        evaluation_id="ipce_1",
        materialization_id=materialization.materialization_id,
        allocation_id=materialization.allocation_id,
        candidate_id=materialization.candidate_id,
        run_id=materialization.run_id,
        proposal_id=materialization.proposal_id,
        gate_evaluation_id=materialization.gate_evaluation_id,
        results=(_evaluation_result(status),),
        created_at=700.0,
    )


def _outcome(decision: str = "reject") -> ImprovementPatchCandidateOutcome:
    status = "pass" if decision == "promotable" else "fail"
    return ImprovementPatchCandidateOutcomePolicy(
        clock=lambda: 800.0,
        outcome_id_factory=lambda: "ipco_1",
    ).decide(
        _materialization(),
        _evaluation(status),
        decision=decision,  # type: ignore[arg-type]
        summary=(
            "Evaluation passed."
            if decision == "promotable"
            else "Rejected candidate should be archived."
        ),
        promotion_blockers=None if decision == "promotable" else ("unit tests failed",),
        metadata={"phase": "outcome"},
    )


def _archive_decision(
    outcome: ImprovementPatchCandidateOutcome | None = None,
) -> ImprovementPatchCandidateArchiveDecision:
    return ImprovementPatchCandidateArchiveDecisionPolicy(
        clock=lambda: 900.0,
        archive_decision_id_factory=lambda: "ipcad_1",
    ).decide(
        outcome or _outcome("reject"),
        archive_reason="Rejected candidate should be retained for review.",
        summary="Rejected candidate should be archived.",
        failure_symptoms=("unit tests failed",),
        artifact_references=("task-output:test-candidate-archive",),
        metadata={"phase": "archive-decision"},
    )


def _approval() -> ImprovementPatchCandidateArchiveApproval:
    return ImprovementPatchCandidateArchiveApproval(
        approved_permissions=("filesystem_mutation",),
        reason="RSI-004B local archive approval",
        approved_by="tester",
        created_at=910.0,
        metadata={"approved": True},
    )


@dataclass
class FakeArchiveStore:
    storage_key: str = "archives/ipca_1.json"
    storage_kind: str = "local_archive"
    metadata: dict[str, bool] = field(default_factory=lambda: {"stored": True})
    requests: list[ImprovementPatchCandidateArchiveRequest] = field(
        default_factory=lambda: []
    )

    async def archive(
        self,
        request: ImprovementPatchCandidateArchiveRequest,
    ) -> ImprovementPatchCandidateArchiveStoreResult:
        self.requests.append(request)
        return ImprovementPatchCandidateArchiveStoreResult(
            storage_key=self.storage_key,
            storage_kind=self.storage_kind,
            metadata=self.metadata,
        )


async def _archive_record(
    *,
    outcome: ImprovementPatchCandidateOutcome | None = None,
    archive_decision: ImprovementPatchCandidateArchiveDecision | None = None,
    store: FakeArchiveStore | None = None,
) -> tuple[ImprovementPatchCandidateArchiveRecord, FakeArchiveStore]:
    selected_outcome = outcome or _outcome("reject")
    selected_decision = archive_decision or _archive_decision(selected_outcome)
    selected_store = store or FakeArchiveStore()
    record = await ImprovementPatchCandidateArchiver(
        clock=lambda: 1_000.0,
        archive_id_factory=lambda: "ipca_1",
    ).archive(
        selected_outcome,
        selected_decision,
        archive_store=selected_store,
        approval=_approval(),
        metadata={"phase": "archive-record"},
    )
    return record, selected_store


def test_archive_approval_requires_filesystem_mutation() -> None:
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="filesystem_mutation",
    ):
        ImprovementPatchCandidateArchiveApproval(
            approved_permissions=("human_review",),
            reason="review only",
        )

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="none",
    ):
        ImprovementPatchCandidateArchiveApproval(
            approved_permissions=("none", "filesystem_mutation"),
            reason="invalid approval",
        )

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="approved",
    ):
        ImprovementPatchCandidateArchiveApproval(
            approved_permissions=("filesystem_mutation",),
            reason="denied",
            approved=False,
        )


@pytest.mark.asyncio
async def test_archiver_requires_injected_store_and_call_time_approval() -> None:
    outcome = _outcome("reject")
    archive_decision = _archive_decision(outcome)
    service = ImprovementPatchCandidateArchiver()

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="ArchiveStore",
    ):
        await service.archive(
            outcome,
            archive_decision,
            archive_store=None,
            approval=_approval(),
        )

    store = FakeArchiveStore()
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="approval",
    ):
        await service.archive(
            outcome,
            archive_decision,
            archive_store=store,
            approval=None,
        )

    assert store.requests == []


@pytest.mark.asyncio
async def test_archiver_rejects_reserved_request_metadata_before_store_call() -> None:
    outcome = _outcome("reject")
    archive_decision = _archive_decision(outcome)
    store = FakeArchiveStore()
    service = ImprovementPatchCandidateArchiver()

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="reserved archive_store_metadata",
    ):
        await service.archive(
            outcome,
            archive_decision,
            archive_store=store,
            approval=_approval(),
            metadata={"archive_store_metadata": {"caller": True}},
        )
    assert store.requests == []

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="reserved archive_store_metadata",
    ):
        await service.archive(
            outcome,
            replace(
                archive_decision,
                metadata={"archive_store_metadata": {"decision": True}},
            ),
            archive_store=store,
            approval=_approval(),
        )
    assert store.requests == []


@pytest.mark.asyncio
async def test_archiver_invokes_injected_store_once_and_records_result() -> None:
    outcome = _outcome("reject")
    archive_decision = _archive_decision(outcome)
    record, store = await _archive_record(outcome=outcome, archive_decision=archive_decision)

    assert len(store.requests) == 1
    request = store.requests[0]
    assert request.archive_decision_id == archive_decision.archive_decision_id
    assert request.outcome_id == outcome.outcome_id
    assert request.materialization_id == outcome.materialization_id
    assert request.allocation_id == outcome.allocation_id
    assert request.candidate_id == outcome.candidate_id
    assert request.run_id == outcome.run_id
    assert request.proposal_id == outcome.proposal_id
    assert request.gate_evaluation_id == outcome.gate_evaluation_id
    assert request.base_revision == outcome.base_revision
    assert request.patch_digest == outcome.patch_digest
    assert request.evaluation_id == outcome.evaluation_id
    assert request.outcome_decision == outcome.decision
    assert request.archive_recommended is True
    assert request.archive_reason == archive_decision.archive_reason
    assert request.summary == archive_decision.summary
    assert request.failure_symptoms == archive_decision.failure_symptoms
    assert request.artifact_references == archive_decision.artifact_references
    assert request.metadata == {
        "phase": "archive-record",
    }

    assert record.archive_id == "ipca_1"
    assert record.archive_decision_id == archive_decision.archive_decision_id
    assert record.outcome_id == outcome.outcome_id
    assert record.materialization_id == outcome.materialization_id
    assert record.allocation_id == outcome.allocation_id
    assert record.candidate_id == outcome.candidate_id
    assert record.run_id == outcome.run_id
    assert record.proposal_id == outcome.proposal_id
    assert record.gate_evaluation_id == outcome.gate_evaluation_id
    assert record.base_revision == outcome.base_revision
    assert record.patch_digest == outcome.patch_digest
    assert record.evaluation_id == outcome.evaluation_id
    assert record.outcome_decision == "reject"
    assert record.archive_recommended is True
    assert record.archive_reason == archive_decision.archive_reason
    assert record.summary == archive_decision.summary
    assert record.failure_symptoms == ("unit tests failed",)
    assert record.artifact_references == ("task-output:test-candidate-archive",)
    assert record.storage_key == "archives/ipca_1.json"
    assert record.storage_kind == "local_archive"
    assert record.archive_digest.startswith("sha256:")
    assert record.status == "archived"
    assert record.created_at == 1_000.0
    assert record.metadata == {
        "phase": "archive-record",
        "archive_store_metadata": {"stored": True},
    }

    snapshot = improvement_patch_candidate_archive_record_to_dict(record)
    assert "approval" not in snapshot
    assert "approved_by" not in snapshot
    assert "approved_permissions" not in snapshot
    assert "reason" not in snapshot


@pytest.mark.asyncio
async def test_archive_records_and_requests_serialize_to_json_and_round_trip() -> None:
    record, store = await _archive_record()
    request = store.requests[0]
    result = ImprovementPatchCandidateArchiveStoreResult(
        storage_key=record.storage_key,
        storage_kind=record.storage_kind,
        metadata={"stored": True},
    )

    request_snapshot = improvement_patch_candidate_archive_request_to_dict(request)
    result_snapshot = improvement_patch_candidate_archive_store_result_to_dict(result)
    record_snapshot = improvement_patch_candidate_archive_record_to_dict(record)
    json.dumps(request_snapshot)
    json.dumps(result_snapshot)
    json.dumps(record_snapshot)

    assert improvement_patch_candidate_archive_request_from_dict(request_snapshot) == request
    assert improvement_patch_candidate_archive_store_result_from_dict(
        result_snapshot
    ) == result
    assert improvement_patch_candidate_archive_record_from_dict(record_snapshot) == record


@pytest.mark.asyncio
async def test_archive_digest_excludes_store_metadata_archive_id_and_created_at() -> None:
    first, _ = await _archive_record(store=FakeArchiveStore(metadata={"stored": True}))
    second, _ = await _archive_record(store=FakeArchiveStore(metadata={"stored": False}))

    assert first.archive_digest == second.archive_digest
    assert first.metadata["archive_store_metadata"] == {"stored": True}
    assert second.metadata["archive_store_metadata"] == {"stored": False}

    snapshot = improvement_patch_candidate_archive_record_to_dict(first)
    snapshot["archive_id"] = "ipca_other"
    snapshot["created_at"] = 2_000.0
    snapshot["metadata"] = {
        "phase": "archive-record",
        "archive_store_metadata": {"stored": "changed"},
    }
    restored = improvement_patch_candidate_archive_record_from_dict(snapshot)

    assert restored.archive_id == "ipca_other"
    assert restored.created_at == 2_000.0
    assert restored.archive_digest == first.archive_digest


@pytest.mark.asyncio
async def test_archive_digest_changes_when_request_or_storage_identity_changes() -> None:
    record, _ = await _archive_record()
    snapshot = improvement_patch_candidate_archive_record_to_dict(record)

    for field_name, changed_value in (
        ("summary", "changed summary"),
        ("archive_reason", "changed reason"),
        ("failure_symptoms", ["changed symptom"]),
        ("artifact_references", ["task-output:changed"]),
        ("storage_key", "archives/changed.json"),
        ("storage_kind", "changed_archive"),
        ("metadata", {"phase": "changed"}),
    ):
        changed = dict(snapshot)
        changed[field_name] = changed_value
        with pytest.raises(
            ImprovementPatchCandidateArchiveValidationError,
            match="archive_digest",
        ):
            improvement_patch_candidate_archive_record_from_dict(changed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome_field", "decision_field", "changed_value"),
    (
        ("outcome_id", "outcome_id", "ipco_other"),
        ("materialization_id", "materialization_id", "ipcm_other"),
        ("allocation_id", "allocation_id", "ipcw_other"),
        ("candidate_id", "candidate_id", "ipc_other"),
        ("run_id", "run_id", "ir_other"),
        ("proposal_id", "proposal_id", "ip_other"),
        ("gate_evaluation_id", "gate_evaluation_id", "ige_other"),
        ("base_revision", "base_revision", "other_revision"),
        ("patch_digest", "patch_digest", "sha256:" + "3" * 64),
        ("evaluation_id", "evaluation_id", "ipce_other"),
        ("decision", "outcome_decision", "needs_review"),
    ),
)
async def test_archiver_validates_outcome_archive_decision_linkage(
    outcome_field: str,
    decision_field: str,
    changed_value: str,
) -> None:
    del outcome_field
    outcome = _outcome("reject")
    archive_decision = replace(_archive_decision(outcome), **{decision_field: changed_value})
    store = FakeArchiveStore()

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match=decision_field,
    ):
        await ImprovementPatchCandidateArchiver().archive(
            outcome,
            archive_decision,
            archive_store=store,
            approval=_approval(),
        )

    assert store.requests == []


@pytest.mark.asyncio
async def test_archiver_rejects_promotable_and_non_recommended_inputs() -> None:
    promotable = _outcome("promotable")
    promotable_archive = ImprovementPatchCandidateArchiveDecisionPolicy().decide(
        promotable
    )
    store = FakeArchiveStore()

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="promotable",
    ):
        await ImprovementPatchCandidateArchiver().archive(
            promotable,
            promotable_archive,
            archive_store=store,
            approval=_approval(),
        )
    assert store.requests == []

    reject_outcome = _outcome("reject")
    non_recommended = cast(
        ImprovementPatchCandidateArchiveDecision,
        SimpleNamespace(
            archive_decision_id="ipcad_1",
            outcome_id=reject_outcome.outcome_id,
            materialization_id=reject_outcome.materialization_id,
            allocation_id=reject_outcome.allocation_id,
            candidate_id=reject_outcome.candidate_id,
            run_id=reject_outcome.run_id,
            proposal_id=reject_outcome.proposal_id,
            gate_evaluation_id=reject_outcome.gate_evaluation_id,
            base_revision=reject_outcome.base_revision,
            patch_digest=reject_outcome.patch_digest,
            evaluation_id=reject_outcome.evaluation_id,
            outcome_decision=reject_outcome.decision,
            archive_recommended=False,
            archive_reason="not recommended",
            summary=reject_outcome.summary,
            failure_symptoms=reject_outcome.promotion_blockers,
            artifact_references=(),
            metadata={},
        ),
    )
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="archive decision",
    ):
        await ImprovementPatchCandidateArchiver().archive(
            reject_outcome,
            non_recommended,
            archive_store=store,
            approval=_approval(),
        )
    assert store.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_name",
    (
        "archive_decision_id",
        "outcome_id",
        "materialization_id",
        "allocation_id",
        "candidate_id",
        "run_id",
        "proposal_id",
        "gate_evaluation_id",
        "base_revision",
        "patch_digest",
        "evaluation_id",
        "outcome_decision",
        "archive_reason",
        "summary",
        "failure_symptoms",
        "artifact_references",
    ),
)
async def test_archive_record_rejects_copied_field_drift(field_name: str) -> None:
    record, _ = await _archive_record()
    snapshot = improvement_patch_candidate_archive_record_to_dict(record)
    changed = dict(snapshot)
    changed[field_name] = (
        ["changed"]
        if field_name in {"failure_symptoms", "artifact_references"}
        else "needs_review"
        if field_name == "outcome_decision"
        else "changed"
    )

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="archive_digest",
    ):
        improvement_patch_candidate_archive_record_from_dict(changed)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    (
        ("storage_key", "", "non-empty"),
        (
            "storage_key",
            "x" * (DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS + 1),
            "STORAGE_KEY_CHARS",
        ),
        ("storage_key", "line one\nline two", "single-line"),
        ("storage_key", "diff --git a/file.py b/file.py", "raw output"),
        ("storage_key", "Traceback (most recent call last)", "raw output"),
        ("storage_key", "-----BEGIN PRIVATE KEY-----", "raw output"),
        ("storage_key", "@@ -1,2 +1,2 @@", "raw output"),
        ("storage_key", "def archive(): pass", "copied file"),
        ("storage_kind", "", "non-empty"),
        (
            "storage_kind",
            "x" * (DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS + 1),
            "STORAGE_KIND_CHARS",
        ),
        ("storage_kind", "line one\nline two", "single-line"),
        ("storage_kind", "diff --git a/file.py b/file.py", "raw output"),
        ("storage_kind", "Traceback (most recent call last)", "raw output"),
        ("storage_kind", "-----BEGIN PRIVATE KEY-----", "raw output"),
        ("storage_kind", "@@ -1,2 +1,2 @@", "raw output"),
        ("storage_kind", "class Archive: pass", "copied file"),
    ),
)
def test_store_result_rejects_bad_storage_references(
    field_name: str,
    bad_value: str,
    match: str,
) -> None:
    with pytest.raises(
        (ImprovementPatchCandidateArchiveValidationError, ValueError),
        match=match,
    ):
        if field_name == "storage_key":
            ImprovementPatchCandidateArchiveStoreResult(
                storage_key=bad_value,
                storage_kind="local_archive",
            )
        else:
            ImprovementPatchCandidateArchiveStoreResult(
                storage_key="archives/ipca_1.json",
                storage_kind=bad_value,
            )


def test_archive_request_and_record_reject_policy_inconsistent_snapshots() -> None:
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="archive_recommended",
    ):
        ImprovementPatchCandidateArchiveRequest(
            archive_decision_id="ipcad_1",
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="b1ce8fd",
            patch_digest="sha256:" + "2" * 64,
            evaluation_id="ipce_1",
            outcome_decision="reject",
            archive_recommended=False,
            archive_reason="not archived",
            summary="not archived",
        )

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="promotable",
    ):
        ImprovementPatchCandidateArchiveRequest(
            archive_decision_id="ipcad_1",
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="b1ce8fd",
            patch_digest="sha256:" + "2" * 64,
            evaluation_id="ipce_1",
            outcome_decision="promotable",
            archive_recommended=True,
            archive_reason="bad archive",
            summary="bad archive",
        )

    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="reserved archive_store_metadata",
    ):
        ImprovementPatchCandidateArchiveRequest(
            archive_decision_id="ipcad_1",
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="b1ce8fd",
            patch_digest="sha256:" + "2" * 64,
            evaluation_id="ipce_1",
            outcome_decision="reject",
            archive_recommended=True,
            archive_reason="archive",
            summary="archive",
            metadata={"archive_store_metadata": {"store": True}},
        )

    digest = "sha256:" + "0" * 64
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="archive_digest",
    ):
        ImprovementPatchCandidateArchiveRecord(
            archive_id="ipca_1",
            archive_decision_id="ipcad_1",
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="b1ce8fd",
            patch_digest="sha256:" + "2" * 64,
            evaluation_id="ipce_1",
            outcome_decision="reject",
            archive_recommended=True,
            archive_reason="archive",
            summary="archive",
            failure_symptoms=(),
            artifact_references=(),
            storage_key="archives/ipca_1.json",
            storage_kind="local_archive",
            archive_digest=digest,
        )


def test_archive_metadata_bounds_are_enforced() -> None:
    with pytest.raises(
        ImprovementPatchCandidateArchiveValidationError,
        match="METADATA_CHARS",
    ):
        ImprovementPatchCandidateArchiveApproval(
            approved_permissions=("filesystem_mutation",),
            reason="large metadata",
            metadata={"large": "x" * (DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS + 1)},
        )


def test_candidate_archive_module_uses_only_injected_archive_store_seam() -> None:
    source = inspect.getsource(archive_module)
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
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)

    forbidden_modules = {
        "os",
        "pathlib",
        "subprocess",
        "raygent_harness.core.permission_engine",
        "raygent_harness.core.streaming_tool_executor",
        "raygent_harness.core.tool_execution",
        "raygent_harness.core.tool_orchestration",
        "raygent_harness.sdk",
        "raygent_harness.services.worktree.manager",
        "raygent_harness.tools.bash_tool",
        "raygent_harness.tools.file_edit_tool",
        "raygent_harness.tools.file_text_utils",
    }
    forbidden_names = {
        "GitWorktreeManager",
        "PermissionHandler",
        "PermissionRequest",
        "Path",
        "create_raygent",
        "open",
        "write_text",
    }

    assert "hashlib" in imported_modules
    assert "json" in imported_modules
    assert imported_modules.isdisjoint(forbidden_modules)
    assert imported_names.isdisjoint(forbidden_names)
    assert "open" not in called_names


def test_candidate_archive_module_exports_only_public_rsi_004b_symbols() -> None:
    assert "json" not in archive_module.__all__
    assert "time" not in archive_module.__all__
    assert "Callable" not in archive_module.__all__
    assert "Mapping" not in archive_module.__all__
    assert "Sequence" not in archive_module.__all__
    assert set(archive_module.__all__) == {
        "DEFAULT_MAX_ARCHIVE_RECORD_METADATA_CHARS",
        "DEFAULT_MAX_ARCHIVE_STORAGE_KEY_CHARS",
        "DEFAULT_MAX_ARCHIVE_STORAGE_KIND_CHARS",
        "ImprovementPatchCandidateArchiveApproval",
        "ImprovementPatchCandidateArchiveError",
        "ImprovementPatchCandidateArchiveRecord",
        "ImprovementPatchCandidateArchiveRequest",
        "ImprovementPatchCandidateArchiveStatus",
        "ImprovementPatchCandidateArchiveStore",
        "ImprovementPatchCandidateArchiveStoreResult",
        "ImprovementPatchCandidateArchiveValidationError",
        "ImprovementPatchCandidateArchiver",
        "improvement_patch_candidate_archive_record_from_dict",
        "improvement_patch_candidate_archive_record_to_dict",
        "improvement_patch_candidate_archive_request_from_dict",
        "improvement_patch_candidate_archive_request_to_dict",
        "improvement_patch_candidate_archive_store_result_from_dict",
        "improvement_patch_candidate_archive_store_result_to_dict",
    }
