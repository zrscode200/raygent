"""Evidence bounding for improvement proposal runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from raygent_harness.improvement.models import (
    ImprovementEvidence,
    improvement_evidence_to_dict,
)

DEFAULT_MAX_EVIDENCE_ITEMS = 12
DEFAULT_MAX_EVIDENCE_ITEM_CHARS = 4_000
DEFAULT_MAX_TOTAL_EVIDENCE_CHARS = 12_000


class ImprovementEvidenceValidationError(ValueError):
    """Raised when improvement evidence is missing or exceeds configured bounds."""


@dataclass(frozen=True, slots=True)
class ImprovementEvidenceBounds:
    """Limits applied before an improvement proposal is generated."""

    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS
    max_item_text_chars: int = DEFAULT_MAX_EVIDENCE_ITEM_CHARS
    max_total_text_chars: int = DEFAULT_MAX_TOTAL_EVIDENCE_CHARS

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError("ImprovementEvidenceBounds.max_items must be >= 1")
        if self.max_item_text_chars < 1:
            raise ValueError("ImprovementEvidenceBounds.max_item_text_chars must be >= 1")
        if self.max_total_text_chars < self.max_item_text_chars:
            raise ValueError(
                "ImprovementEvidenceBounds.max_total_text_chars must be >= "
                "max_item_text_chars"
            )


@dataclass(frozen=True, slots=True)
class BoundedImprovementEvidence:
    """Validated evidence bundle plus accounting metadata."""

    evidence: tuple[ImprovementEvidence, ...]
    total_text_chars: int


def improvement_evidence_text_chars(evidence: ImprovementEvidence) -> int:
    """Return serialized character cost used for improvement evidence bounding."""

    return len(
        json.dumps(
            improvement_evidence_to_dict(evidence),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def validate_bounded_improvement_evidence(
    evidence: Sequence[ImprovementEvidence],
    *,
    bounds: ImprovementEvidenceBounds | None = None,
) -> BoundedImprovementEvidence:
    """Validate evidence count and text bounds before proposal generation."""

    resolved_bounds = bounds or ImprovementEvidenceBounds()
    items = tuple(evidence)
    if not items:
        raise ImprovementEvidenceValidationError("improvement evidence must not be empty")
    if len(items) > resolved_bounds.max_items:
        raise ImprovementEvidenceValidationError(
            "improvement evidence item count exceeds "
            f"{resolved_bounds.max_items}: {len(items)}"
        )

    total = 0
    for item in items:
        item_chars = improvement_evidence_text_chars(item)
        if item_chars > resolved_bounds.max_item_text_chars:
            raise ImprovementEvidenceValidationError(
                f"improvement evidence item {item.evidence_id!r} exceeds "
                f"{resolved_bounds.max_item_text_chars} characters"
            )
        total += item_chars
    if total > resolved_bounds.max_total_text_chars:
        raise ImprovementEvidenceValidationError(
            "improvement evidence total text exceeds "
            f"{resolved_bounds.max_total_text_chars} characters"
        )
    return BoundedImprovementEvidence(evidence=items, total_text_chars=total)


__all__ = (
    "DEFAULT_MAX_EVIDENCE_ITEMS",
    "DEFAULT_MAX_EVIDENCE_ITEM_CHARS",
    "DEFAULT_MAX_TOTAL_EVIDENCE_CHARS",
    "BoundedImprovementEvidence",
    "ImprovementEvidenceBounds",
    "ImprovementEvidenceValidationError",
    "improvement_evidence_text_chars",
    "validate_bounded_improvement_evidence",
)
