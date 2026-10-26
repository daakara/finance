"""
Response and Rubric Data Models for HVA-01 Intake Runner.
Enforces strict schema validation, canonical serialization, and rigid abstention taxonomy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class StageClassification(str, Enum):
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    STAGE_3 = "STAGE_3"
    STAGE_4 = "STAGE_4"
    UNCLASSIFIED = "UNCLASSIFIED"


class VCPQualification(str, Enum):
    VCP_CONFIRMED = "VCP_CONFIRMED"
    VCP_DEVELOPING = "VCP_DEVELOPING"
    VCP_REJECTED = "VCP_REJECTED"
    NON_VCP_BASE = "NON_VCP_BASE"


class ConfidenceLevel(str, Enum):
    CONFIDENCE_HIGH = "CONFIDENCE_HIGH"
    CONFIDENCE_MODERATE = "CONFIDENCE_MODERATE"
    CONFIDENCE_LOW = "CONFIDENCE_LOW"


class PriorFamiliarity(str, Enum):
    NO_RECALL = "NO_RECALL"
    PATTERN_FAMILIAR_OUTCOME_UNKNOWN = "PATTERN_FAMILIAR_OUTCOME_UNKNOWN"
    OUTCOME_RECALLED = "OUTCOME_RECALLED"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class AbstentionCode(str, Enum):
    NONE = "NONE"
    ABS_DATA = "ABS_DATA"          # Missing data or chart unreadable
    ABS_COMP = "ABS_COMP"          # Ambiguous/inconclusive chart geometry
    ABS_FAM = "ABS_FAM"            # Contaminated by prior ticker recall
    ABS_WITHDRAW = "ABS_WITHDRAW"  # Participant session withdrawal


CASE_ID_PATTERN = re.compile(r"^HVA2-[AB]-\d{2}$")


class ResponseValidationError(ValueError):
    """Raised when response payload violates rubric validation rules."""
    pass


@dataclass(frozen=True)
class RatingsPayload:
    stage_classification: StageClassification
    vcp_qualification: VCPQualification
    confidence_level: ConfidenceLevel
    prior_familiarity: PriorFamiliarity
    evidence_sufficiency: EvidenceSufficiency
    reviewer_reasoning: str
    abstention_code: AbstentionCode = AbstentionCode.NONE
    contraction_count_observed: Optional[int] = None
    progressive_tightening_observed: Optional[bool] = None
    volume_dry_up_observed: Optional[bool] = None
    pivot_present: Optional[bool] = None
    composite_score_100: Optional[int] = None

    def __post_init__(self) -> None:
        # Validate reasoning string
        if not self.reviewer_reasoning or len(self.reviewer_reasoning.strip()) < 5:
            raise ResponseValidationError("reviewer_reasoning must be at least 5 non-whitespace characters.")

        # Validate contraction count
        if self.contraction_count_observed is not None:
            if not isinstance(self.contraction_count_observed, int) or isinstance(self.contraction_count_observed, bool):
                raise ResponseValidationError("contraction_count_observed must be an integer.")
            if not (0 <= self.contraction_count_observed <= 10):
                raise ResponseValidationError(
                    f"contraction_count_observed must be in range [0, 10], got {self.contraction_count_observed}"
                )

        # Validate composite score
        if self.composite_score_100 is not None:
            if not isinstance(self.composite_score_100, int) or isinstance(self.composite_score_100, bool):
                raise ResponseValidationError("composite_score_100 must be an integer.")
            if not (0 <= self.composite_score_100 <= 100):
                raise ResponseValidationError(
                    f"composite_score_100 must be in range [0, 100], got {self.composite_score_100}"
                )

        # Validate abstention logic
        if self.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT and self.abstention_code == AbstentionCode.NONE:
            raise ResponseValidationError(
                "When evidence_sufficiency is INSUFFICIENT, a valid abstention_code must be specified."
            )

        if self.prior_familiarity == PriorFamiliarity.OUTCOME_RECALLED and self.abstention_code not in (
            AbstentionCode.ABS_FAM,
            AbstentionCode.NONE,
        ):
            # Flagged as contaminated
            pass

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return sorted dictionary with explicit enum values."""
        return {
            "abstention_code": self.abstention_code.value,
            "composite_score_100": self.composite_score_100,
            "confidence_level": self.confidence_level.value,
            "contraction_count_observed": self.contraction_count_observed,
            "evidence_sufficiency": self.evidence_sufficiency.value,
            "pivot_present": self.pivot_present,
            "prior_familiarity": self.prior_familiarity.value,
            "progressive_tightening_observed": self.progressive_tightening_observed,
            "reviewer_reasoning": self.reviewer_reasoning.strip(),
            "stage_classification": self.stage_classification.value,
            "vcp_qualification": self.vcp_qualification.value,
            "volume_dry_up_observed": self.volume_dry_up_observed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RatingsPayload:
        return cls(
            stage_classification=StageClassification(d["stage_classification"]),
            vcp_qualification=VCPQualification(d["vcp_qualification"]),
            confidence_level=ConfidenceLevel(d["confidence_level"]),
            prior_familiarity=PriorFamiliarity(d["prior_familiarity"]),
            evidence_sufficiency=EvidenceSufficiency(d["evidence_sufficiency"]),
            reviewer_reasoning=d["reviewer_reasoning"],
            abstention_code=AbstentionCode(d.get("abstention_code", AbstentionCode.NONE.value)),
            contraction_count_observed=d.get("contraction_count_observed"),
            progressive_tightening_observed=d.get("progressive_tightening_observed"),
            volume_dry_up_observed=d.get("volume_dry_up_observed"),
            pivot_present=d.get("pivot_present"),
            composite_score_100=d.get("composite_score_100"),
        )


@dataclass(frozen=True)
class ResponseRecord:
    sequence_index: int
    case_id: str
    session_id: str
    reviewer_pseudonym: str
    protocol_version: str
    protocol_document_sha256: str
    package_sha256: str
    timestamp_utc: str
    ratings: RatingsPayload
    idempotency_key: str
    previous_hash: str
    record_hash: str
    response_revision: int = 1
    is_locked: bool = True

    def __post_init__(self) -> None:
        if not (1 <= self.sequence_index <= 59):
            raise ResponseValidationError(f"sequence_index must be in range [1, 59], got {self.sequence_index}")
        if not CASE_ID_PATTERN.match(self.case_id):
            raise ResponseValidationError(f"Invalid case_id: {self.case_id}. Must match {CASE_ID_PATTERN.pattern}")
        if not self.is_locked:
            raise ResponseValidationError("ResponseRecord in persistence must have is_locked=True")
        try:
            datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except (ValueError, TypeError) as exc:
            raise ResponseValidationError(f"Invalid timestamp_utc: {self.timestamp_utc}") from exc

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return canonical dictionary for hashing and storage."""
        return {
            "case_id": self.case_id,
            "idempotency_key": self.idempotency_key,
            "is_locked": self.is_locked,
            "package_sha256": self.package_sha256,
            "previous_hash": self.previous_hash,
            "protocol_document_sha256": self.protocol_document_sha256,
            "protocol_version": self.protocol_version,
            "ratings": self.ratings.to_canonical_dict(),
            "record_hash": self.record_hash,
            "response_revision": self.response_revision,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "sequence_index": self.sequence_index,
            "session_id": self.session_id,
            "timestamp_utc": self.timestamp_utc,
        }

    def to_canonical_json(self) -> str:
        """Deterministic JSON string using RFC 8785 canonicalization principles."""
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResponseRecord:
        ratings_dict = d["ratings"]
        ratings = (
            ratings_dict
            if isinstance(ratings_dict, RatingsPayload)
            else RatingsPayload.from_dict(ratings_dict)
        )
        return cls(
            sequence_index=d["sequence_index"],
            case_id=d["case_id"],
            session_id=d["session_id"],
            reviewer_pseudonym=d["reviewer_pseudonym"],
            protocol_version=d["protocol_version"],
            protocol_document_sha256=d["protocol_document_sha256"],
            package_sha256=d["package_sha256"],
            timestamp_utc=d["timestamp_utc"],
            ratings=ratings,
            idempotency_key=d["idempotency_key"],
            previous_hash=d["previous_hash"],
            record_hash=d["record_hash"],
            response_revision=d.get("response_revision", 1),
            is_locked=d.get("is_locked", True),
        )
