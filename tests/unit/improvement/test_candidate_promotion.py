from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

import raygent_harness.improvement.candidate_promotion as promotion_module
from raygent_harness.core.model_types import FrozenJson
from raygent_harness.improvement import (
    DEFAULT_MAX_PROMOTED_FILE_CHARS,
    DEFAULT_MAX_PROMOTED_FILES,
    DEFAULT_MAX_PROMOTION_KIND_CHARS,
    DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS,
    DEFAULT_MAX_PROMOTION_REF_CHARS,
    ImprovementPatchCandidateEvaluation,
    ImprovementPatchCandidateEvaluationResult,
    ImprovementPatchCandidateMaterialization,
    ImprovementPatchCandidateOutcome,
    ImprovementPatchCandidateOutcomePolicy,
    ImprovementPatchCandidatePromotionApproval,
    ImprovementPatchCandidatePromotionRecord,
    ImprovementPatchCandidatePromotionRequest,
    ImprovementPatchCandidatePromotionResult,
    ImprovementPatchCandidatePromotionService,
    ImprovementPatchCandidatePromotionValidationError,
    ImprovementPatchOperation,
    improvement_patch_candidate_promotion_record_from_dict,
    improvement_patch_candidate_promotion_record_to_dict,
    improvement_patch_candidate_promotion_request_from_dict,
    improvement_patch_candidate_promotion_request_to_dict,
    improvement_patch_candidate_promotion_result_from_dict,
    improvement_patch_candidate_promotion_result_to_dict,
)


def _operation() -> ImprovementPatchOperation:
    return ImprovementPatchOperation(
        operation_id="op_1",
        kind="replace_text",
        relative_path="src/raygent_harness/improvement/candidate_promotion.py",
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
        base_revision="dfb0b1a",
        worktree_path="/tmp/raygent/ipc_1",
        worktree_slug="ipc-ipc_1-a9b687a7",
        operations=(operation,),
        changed_files=(operation.relative_path,),
        patch_digest="sha256:" + "4" * 64,
        created_at=500.0,
        metadata={"phase": "rsi-003c"},
    )


def _evaluation_result(status: str = "pass") -> ImprovementPatchCandidateEvaluationResult:
    return ImprovementPatchCandidateEvaluationResult(
        result_id=f"res_{status}",
        kind="unit_tests",
        status=status,  # type: ignore[arg-type]
        summary=f"unit tests {status}",
        changed_files=("src/raygent_harness/improvement/candidate_promotion.py",),
        output_reference="task-output:test-candidate-promotion",
        required=True,
        created_at=600.0,
    )


def _evaluation(status: str = "pass") -> ImprovementPatchCandidateEvaluation:
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


def _outcome(
    decision: str = "promotable",
    *,
    required_permissions: tuple[str, ...] | None = None,
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementPatchCandidateOutcome:
    status = (
        "pass"
        if decision == "promotable"
        else "warn"
        if decision == "needs_review"
        else "fail"
    )
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
        promotion_blockers=(
            None
            if decision == "promotable"
            else ("evaluation produced warnings",)
            if decision == "needs_review"
            else ("unit tests failed",)
        ),
        required_permissions=required_permissions,
        metadata=metadata,
    )


def _promotion_result(
    *,
    promotion_ref: str = "promotions/ipcp_1",
    promotion_kind: str = "local_commit",
    source_worktree_ref: str = "worktrees/ipcw_1",
    target_ref: str = "refs/heads/ray-dev",
    target_revision: str = "abcd1234",
    promoted_files: tuple[str, ...] = (
        "src/raygent_harness/improvement/candidate_promotion.py",
    ),
    summary: str = "Promotion attempt recorded by caller-owned promoter.",
    metadata: Mapping[str, FrozenJson] | None = None,
) -> ImprovementPatchCandidatePromotionResult:
    return ImprovementPatchCandidatePromotionResult(
        promotion_ref=promotion_ref,
        promotion_kind=promotion_kind,
        source_worktree_ref=source_worktree_ref,
        target_ref=target_ref,
        target_revision=target_revision,
        promoted_files=promoted_files,
        summary=summary,
        metadata=metadata or {"promoted": True},
    )


def _approval(
    permissions: tuple[str, ...] = (
        "commit",
        "human_review",
        "filesystem_mutation",
    ),
) -> ImprovementPatchCandidatePromotionApproval:
    return ImprovementPatchCandidatePromotionApproval(
        approved_permissions=permissions,  # type: ignore[arg-type]
        reason="RSI-004C local promotion attempt approval",
        approved_by="tester",
        created_at=910.0,
        metadata={"approved": True},
    )


@dataclass
class FakePromoter:
    result: ImprovementPatchCandidatePromotionResult = field(
        default_factory=_promotion_result
    )
    requests: list[ImprovementPatchCandidatePromotionRequest] = field(
        default_factory=lambda: []
    )

    async def promote(
        self,
        request: ImprovementPatchCandidatePromotionRequest,
    ) -> ImprovementPatchCandidatePromotionResult:
        self.requests.append(request)
        return self.result


async def _promotion_record(
    *,
    outcome: ImprovementPatchCandidateOutcome | None = None,
    promoter: FakePromoter | None = None,
) -> tuple[ImprovementPatchCandidatePromotionRecord, FakePromoter]:
    selected_promoter = promoter or FakePromoter()
    record = await ImprovementPatchCandidatePromotionService(
        clock=lambda: 1_000.0,
        promotion_id_factory=lambda: "ipcp_1",
    ).promote(
        outcome or _outcome(metadata={"source": "outcome"}),
        promoter=selected_promoter,
        approval=_approval(),
        metadata={"phase": "promotion-record"},
    )
    return record, selected_promoter


@pytest.mark.parametrize(
    "permissions",
    (
        ("human_review", "filesystem_mutation"),
        ("human_review", "commit"),
        ("filesystem_mutation", "commit"),
    ),
)
def test_promotion_approval_requires_each_core_permission(
    permissions: tuple[str, ...],
) -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="missing required promotion permissions",
    ):
        _approval(permissions)


@pytest.mark.parametrize(
    "extra_permission",
    ("none", "model_provider", "shell", "worktree", "network", "external_service"),
)
def test_promotion_approval_rejects_permissions_outside_exact_core_set(
    extra_permission: str,
) -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match=extra_permission,
    ):
        _approval(
            (
                "human_review",
                "filesystem_mutation",
                "commit",
                extra_permission,
            )
        )


def test_promotion_approval_requires_denied_and_auditable_fields() -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="approved",
    ):
        ImprovementPatchCandidatePromotionApproval(
            approved_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            reason="denied",
            approved_by="tester",
            approved=False,
        )

    with pytest.raises(ValueError, match="reason"):
        ImprovementPatchCandidatePromotionApproval(
            approved_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            reason="",
            approved_by="tester",
        )

    with pytest.raises(ValueError, match="approved_by"):
        ImprovementPatchCandidatePromotionApproval(
            approved_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            reason="approved",
            approved_by="",
        )


@pytest.mark.asyncio
async def test_promotion_service_requires_injected_promoter_and_call_time_approval() -> None:
    outcome = _outcome()
    service = ImprovementPatchCandidatePromotionService()

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="Promoter",
    ):
        await service.promote(outcome, promoter=None, approval=_approval())

    promoter = FakePromoter()
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="approval",
    ):
        await service.promote(outcome, promoter=promoter, approval=None)

    denied_approval = cast(
        ImprovementPatchCandidatePromotionApproval,
        SimpleNamespace(
            approved_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            approved=False,
        ),
    )
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="approved",
    ):
        await service.promote(
            outcome,
            promoter=promoter,
            approval=denied_approval,
        )

    assert promoter.requests == []


@pytest.mark.asyncio
async def test_promotion_service_rejects_reserved_request_metadata_before_promoter_call() -> None:
    promoter = FakePromoter()
    service = ImprovementPatchCandidatePromotionService()

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="reserved promotion_result_metadata",
    ):
        await service.promote(
            _outcome(),
            promoter=promoter,
            approval=_approval(),
            metadata={"promotion_result_metadata": {"caller": True}},
        )
    assert promoter.requests == []

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="reserved promotion_result_metadata",
    ):
        await service.promote(
            _outcome(metadata={"promotion_result_metadata": {"outcome": True}}),
            promoter=promoter,
            approval=_approval(),
        )
    assert promoter.requests == []


@pytest.mark.asyncio
async def test_promotion_service_invokes_promoter_once_and_records_result() -> None:
    outcome = _outcome(metadata={"source": "outcome"})
    record, promoter = await _promotion_record(outcome=outcome)

    assert len(promoter.requests) == 1
    request = promoter.requests[0]
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
    assert request.evaluation_decision == "pass"
    assert request.outcome_decision == "promotable"
    assert request.summary == outcome.summary
    assert request.required_permissions == (
        "human_review",
        "filesystem_mutation",
        "commit",
    )
    assert request.metadata == {
        "source": "outcome",
        "phase": "promotion-record",
    }

    assert record.promotion_id == "ipcp_1"
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
    assert record.evaluation_decision == "pass"
    assert record.outcome_decision == "promotable"
    assert record.required_permissions == request.required_permissions
    assert record.promotion_ref == "promotions/ipcp_1"
    assert record.promotion_kind == "local_commit"
    assert record.source_worktree_ref == "worktrees/ipcw_1"
    assert record.target_ref == "refs/heads/ray-dev"
    assert record.target_revision == "abcd1234"
    assert record.promoted_files == (
        "src/raygent_harness/improvement/candidate_promotion.py",
    )
    assert record.promotion_digest.startswith("sha256:")
    assert record.status == "promotion_recorded"
    assert record.created_at == 1_000.0
    assert record.metadata == {
        "source": "outcome",
        "phase": "promotion-record",
        "promotion_result_metadata": {"promoted": True},
    }

    snapshot = improvement_patch_candidate_promotion_record_to_dict(record)
    assert "approval" not in snapshot
    assert "approved_by" not in snapshot
    assert "approved_permissions" not in snapshot
    assert "reason" not in snapshot


@pytest.mark.asyncio
async def test_promotion_records_requests_and_results_round_trip_through_json() -> None:
    record, promoter = await _promotion_record()
    request = promoter.requests[0]
    result = _promotion_result()

    request_snapshot = improvement_patch_candidate_promotion_request_to_dict(request)
    result_snapshot = improvement_patch_candidate_promotion_result_to_dict(result)
    record_snapshot = improvement_patch_candidate_promotion_record_to_dict(record)
    json.dumps(request_snapshot)
    json.dumps(result_snapshot)
    json.dumps(record_snapshot)

    assert improvement_patch_candidate_promotion_request_from_dict(
        request_snapshot
    ) == request
    assert improvement_patch_candidate_promotion_result_from_dict(
        result_snapshot
    ) == result
    assert improvement_patch_candidate_promotion_record_from_dict(record_snapshot) == record


@pytest.mark.asyncio
async def test_promotion_service_rejects_non_promotable_or_policy_blocked_outcomes() -> None:
    service = ImprovementPatchCandidatePromotionService()

    for outcome in (_outcome("reject"), _outcome("needs_review")):
        promoter = FakePromoter()
        with pytest.raises(
            ImprovementPatchCandidatePromotionValidationError,
            match="promotable",
        ):
            await service.promote(outcome, promoter=promoter, approval=_approval())
        assert promoter.requests == []

    valid = _outcome()
    invalid_cases = (
        ("evaluation_decision", "warn", "pass evaluation"),
        ("archive_recommended", True, "archive-recommended"),
        ("promotion_blockers", ("blocked",), "promotion_blockers"),
        (
            "required_permissions",
            ("human_review", "filesystem_mutation"),
            "missing required promotion permissions",
        ),
    )
    for field_name, changed_value, match in invalid_cases:
        promoter = FakePromoter()
        invalid = cast(
            ImprovementPatchCandidateOutcome,
            SimpleNamespace(
                outcome_id=valid.outcome_id,
                materialization_id=valid.materialization_id,
                allocation_id=valid.allocation_id,
                candidate_id=valid.candidate_id,
                run_id=valid.run_id,
                proposal_id=valid.proposal_id,
                gate_evaluation_id=valid.gate_evaluation_id,
                base_revision=valid.base_revision,
                patch_digest=valid.patch_digest,
                evaluation_id=valid.evaluation_id,
                evaluation_decision=(
                    changed_value
                    if field_name == "evaluation_decision"
                    else valid.evaluation_decision
                ),
                decision=valid.decision,
                summary=valid.summary,
                required_permissions=(
                    changed_value
                    if field_name == "required_permissions"
                    else valid.required_permissions
                ),
                archive_recommended=(
                    changed_value
                    if field_name == "archive_recommended"
                    else valid.archive_recommended
                ),
                promotion_blockers=(
                    changed_value
                    if field_name == "promotion_blockers"
                    else valid.promotion_blockers
                ),
                metadata=valid.metadata,
            ),
        )
        with pytest.raises(ImprovementPatchCandidatePromotionValidationError, match=match):
            await service.promote(invalid, promoter=promoter, approval=_approval())
        assert promoter.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_permission",
    ("none", "model_provider", "shell", "worktree", "network", "external_service"),
)
async def test_promotion_service_rejects_outcomes_with_extra_permissions(
    extra_permission: str,
) -> None:
    valid = _outcome()
    outcome = cast(
        ImprovementPatchCandidateOutcome,
        SimpleNamespace(
            outcome_id=valid.outcome_id,
            materialization_id=valid.materialization_id,
            allocation_id=valid.allocation_id,
            candidate_id=valid.candidate_id,
            run_id=valid.run_id,
            proposal_id=valid.proposal_id,
            gate_evaluation_id=valid.gate_evaluation_id,
            base_revision=valid.base_revision,
            patch_digest=valid.patch_digest,
            evaluation_id=valid.evaluation_id,
            evaluation_decision=valid.evaluation_decision,
            decision=valid.decision,
            summary=valid.summary,
            required_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
                extra_permission,
            ),
            archive_recommended=valid.archive_recommended,
            promotion_blockers=valid.promotion_blockers,
            metadata=valid.metadata,
        ),
    )
    promoter = FakePromoter()

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match=extra_permission,
    ):
        await ImprovementPatchCandidatePromotionService().promote(
            outcome,
            promoter=promoter,
            approval=_approval(),
        )

    assert promoter.requests == []


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    (
        ("promotion_ref", "", "non-empty"),
        (
            "promotion_ref",
            "x" * (DEFAULT_MAX_PROMOTION_REF_CHARS + 1),
            "PROMOTION_REF_CHARS",
        ),
        ("promotion_ref", "line one\nline two", "single-line"),
        ("promotion_ref", "diff --git a/file.py b/file.py", "raw output"),
        ("promotion_ref", "Traceback (most recent call last)", "raw output"),
        ("promotion_ref", "-----BEGIN PRIVATE KEY-----", "raw output"),
        ("promotion_ref", "@@ -1,2 +1,2 @@", "raw output"),
        ("promotion_ref", "def promote(): pass", "copied file"),
        ("promotion_kind", "", "non-empty"),
        (
            "promotion_kind",
            "x" * (DEFAULT_MAX_PROMOTION_KIND_CHARS + 1),
            "PROMOTION_KIND_CHARS",
        ),
        ("promotion_kind", "line one\nline two", "single-line"),
        ("promotion_kind", "class Promoter: pass", "copied file"),
        ("source_worktree_ref", "diff --git a/file.py b/file.py", "raw output"),
        ("target_ref", "Traceback (most recent call last)", "raw output"),
        ("target_revision", "-----BEGIN PRIVATE KEY-----", "raw output"),
    ),
)
def test_promotion_result_rejects_bad_references(
    field_name: str,
    bad_value: str,
    match: str,
) -> None:
    kwargs: dict[str, Any] = {field_name: bad_value}
    with pytest.raises(
        (ImprovementPatchCandidatePromotionValidationError, ValueError),
        match=match,
    ):
        _promotion_result(**kwargs)


@pytest.mark.parametrize(
    ("promoted_files", "match"),
    (
        ((), "must not be empty"),
        (("/absolute/path.py",), "relative"),
        (("../outside.py",), "parent traversal"),
        (("src/file.py\x00",), "NUL"),
        (("src\\file.py",), "POSIX-style"),
        (("src//file.py", "src/file.py"), "duplicate normalized paths"),
        (("x" * (DEFAULT_MAX_PROMOTED_FILE_CHARS + 1),), "PROMOTED_FILE_CHARS"),
        (
            tuple(f"src/file_{index}.py" for index in range(DEFAULT_MAX_PROMOTED_FILES + 1)),
            "PROMOTED_FILES",
        ),
    ),
)
def test_promotion_result_rejects_unsafe_promoted_files(
    promoted_files: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match=match,
    ):
        _promotion_result(promoted_files=promoted_files)


@pytest.mark.asyncio
async def test_promotion_digest_excludes_result_metadata_summary_id_and_created_at() -> None:
    first, _ = await _promotion_record(
        promoter=FakePromoter(
            result=_promotion_result(
                summary="first result summary",
                metadata={"promoted": True},
            )
        )
    )
    second, _ = await _promotion_record(
        promoter=FakePromoter(
            result=_promotion_result(
                summary="changed result summary",
                metadata={"promoted": False},
            )
        )
    )

    assert first.promotion_digest == second.promotion_digest
    assert first.metadata["promotion_result_metadata"] == {"promoted": True}
    assert second.metadata["promotion_result_metadata"] == {"promoted": False}

    snapshot = improvement_patch_candidate_promotion_record_to_dict(first)
    snapshot["promotion_id"] = "ipcp_other"
    snapshot["created_at"] = 2_000.0
    snapshot["metadata"] = {
        "source": "outcome",
        "phase": "promotion-record",
        "promotion_result_metadata": {"promoted": "changed"},
    }
    restored = improvement_patch_candidate_promotion_record_from_dict(snapshot)

    assert restored.promotion_id == "ipcp_other"
    assert restored.created_at == 2_000.0
    assert restored.promotion_digest == first.promotion_digest


@pytest.mark.asyncio
async def test_promotion_digest_changes_when_request_or_result_identity_changes() -> None:
    record, _ = await _promotion_record()
    snapshot = improvement_patch_candidate_promotion_record_to_dict(record)

    for field_name, changed_value in (
        ("summary", "changed outcome summary"),
        ("promotion_ref", "promotions/changed"),
        ("promotion_kind", "changed_kind"),
        ("source_worktree_ref", "worktrees/changed"),
        ("target_ref", "refs/heads/changed"),
        ("target_revision", "ffff0000"),
        ("promoted_files", ["src/changed.py"]),
        ("metadata", {"source": "changed", "phase": "promotion-record"}),
    ):
        changed = dict(snapshot)
        changed[field_name] = changed_value
        with pytest.raises(
            ImprovementPatchCandidatePromotionValidationError,
            match="promotion_digest",
        ):
            improvement_patch_candidate_promotion_record_from_dict(changed)


def test_promotion_request_and_record_reject_policy_inconsistent_snapshots() -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="pass",
    ):
        ImprovementPatchCandidatePromotionRequest(
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="dfb0b1a",
            patch_digest="sha256:" + "4" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="warn",
            outcome_decision="promotable",
            summary="bad promotion request",
            required_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="promotable",
    ):
        ImprovementPatchCandidatePromotionRequest(
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="dfb0b1a",
            patch_digest="sha256:" + "4" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="pass",
            outcome_decision="reject",
            summary="bad promotion request",
            required_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
        )

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="reserved promotion_result_metadata",
    ):
        ImprovementPatchCandidatePromotionRequest(
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="dfb0b1a",
            patch_digest="sha256:" + "4" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="pass",
            outcome_decision="promotable",
            summary="bad promotion request",
            required_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            metadata={"promotion_result_metadata": {"result": True}},
        )

    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="promotion_digest",
    ):
        ImprovementPatchCandidatePromotionRecord(
            promotion_id="ipcp_1",
            outcome_id="ipco_1",
            materialization_id="ipcm_1",
            allocation_id="ipcw_1",
            candidate_id="ipc_1",
            run_id="ir_1",
            proposal_id="ip_1",
            gate_evaluation_id="ige_1",
            base_revision="dfb0b1a",
            patch_digest="sha256:" + "4" * 64,
            evaluation_id="ipce_1",
            evaluation_decision="pass",
            outcome_decision="promotable",
            summary="bad digest",
            required_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            promotion_ref="promotions/ipcp_1",
            promotion_kind="local_commit",
            source_worktree_ref="worktrees/ipcw_1",
            target_ref="refs/heads/ray-dev",
            target_revision="abcd1234",
            promoted_files=("src/raygent_harness/improvement/candidate_promotion.py",),
            promotion_digest="sha256:" + "0" * 64,
        )


def test_promotion_metadata_bounds_are_enforced() -> None:
    with pytest.raises(
        ImprovementPatchCandidatePromotionValidationError,
        match="METADATA_CHARS",
    ):
        ImprovementPatchCandidatePromotionApproval(
            approved_permissions=(
                "human_review",
                "filesystem_mutation",
                "commit",
            ),
            reason="large metadata",
            approved_by="tester",
            metadata={
                "large": "x" * (DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS + 1)
            },
        )


def test_candidate_promotion_module_uses_only_injected_promotion_seam() -> None:
    source = inspect.getsource(promotion_module)
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
        "raygent_harness.improvement.candidate_archive",
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
    assert called_names.isdisjoint(forbidden_names)


def test_candidate_promotion_module_exports_public_rsi_004c_symbols() -> None:
    assert set(promotion_module.__all__) == {
        "DEFAULT_MAX_PROMOTED_FILES",
        "DEFAULT_MAX_PROMOTED_FILE_CHARS",
        "DEFAULT_MAX_PROMOTION_KIND_CHARS",
        "DEFAULT_MAX_PROMOTION_RECORD_METADATA_CHARS",
        "DEFAULT_MAX_PROMOTION_REF_CHARS",
        "DEFAULT_MAX_PROMOTION_SUMMARY_CHARS",
        "ImprovementPatchCandidatePromotionApproval",
        "ImprovementPatchCandidatePromotionError",
        "ImprovementPatchCandidatePromotionRecord",
        "ImprovementPatchCandidatePromotionRequest",
        "ImprovementPatchCandidatePromotionResult",
        "ImprovementPatchCandidatePromotionService",
        "ImprovementPatchCandidatePromotionStatus",
        "ImprovementPatchCandidatePromotionValidationError",
        "ImprovementPatchCandidatePromoter",
        "improvement_patch_candidate_promotion_record_from_dict",
        "improvement_patch_candidate_promotion_record_to_dict",
        "improvement_patch_candidate_promotion_request_from_dict",
        "improvement_patch_candidate_promotion_request_to_dict",
        "improvement_patch_candidate_promotion_result_from_dict",
        "improvement_patch_candidate_promotion_result_to_dict",
    }
