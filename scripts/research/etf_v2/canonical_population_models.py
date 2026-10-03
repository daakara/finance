"""
scripts/research/etf_v2/canonical_population_models.py

Domain entities, data transfer objects, and status enums for the ETF V2
Canonical Population Authority.
Enforces strict immutability (frozen dataclasses) and explicit typing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple


class CanonicalStatus(str, Enum):
    """Authoritative legal lifecycle status of an admitted share class."""
    ADMITTED_CURRENT = "ADMITTED_CURRENT"
    ADMITTED_CURRENTNESS_NOT_ESTABLISHED = "ADMITTED_CURRENTNESS_NOT_ESTABLISHED"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"
    MERGED = "MERGED"
    TERMINATED = "TERMINATED"
    HISTORICAL = "HISTORICAL"


class AdmissionState(str, Enum):
    """Pure pipeline admission readiness state."""
    ADMISSION_READY = "ADMISSION_READY"
    ADMITTED = "ADMITTED"
    ADMISSION_BLOCKED = "ADMISSION_BLOCKED"


class AuthorityTier(str, Enum):
    """Evidence authority ranking."""
    TIER_1_OFFICIAL_STATUTORY = "TIER_1_OFFICIAL_STATUTORY"
    TIER_2_REGULATOR_OFFICIAL = "TIER_2_REGULATOR_OFFICIAL"
    TIER_3_VENDOR_MARKET = "TIER_3_VENDOR_MARKET"
    OPENFIGI = "OPENFIGI"

    def is_statutory_admissible(self) -> bool:
        return self in (AuthorityTier.TIER_1_OFFICIAL_STATUTORY, AuthorityTier.TIER_2_REGULATOR_OFFICIAL)


class HoldCategory(str, Enum):
    """Categorization for pre-admission quarantined candidates."""
    OFFICIAL_IDENTITY_EVIDENCE_NOT_ESTABLISHED_HOLD = "OFFICIAL_IDENTITY_EVIDENCE_NOT_ESTABLISHED_HOLD"
    IDENTITY_EXCEPTION_HOLD = "IDENTITY_EXCEPTION_HOLD"
    RELEVANT_SUPPLEMENT_ROUTE_HOLD = "RELEVANT_SUPPLEMENT_ROUTE_HOLD"
    PRELAUNCH_STATUS_HOLD = "PRELAUNCH_STATUS_HOLD"


class HoldState(str, Enum):
    """Status of an individual hold entry."""
    ACTIVE_HOLD = "ACTIVE_HOLD"
    RESOLVED = "RESOLVED"
    RELEASED = "RELEASED"


class CollisionState(str, Enum):
    """Exact classification of candidate vs canonical identity collision."""
    NEW_IDENTITY = "NEW_IDENTITY"
    EXACT_ALREADY_PRESENT = "EXACT_ALREADY_PRESENT"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    PROVENANCE_CONFLICT = "PROVENANCE_CONFLICT"
    HISTORICAL_CONTINUITY_REVIEW_REQUIRED = "HISTORICAL_CONTINUITY_REVIEW_REQUIRED"


@dataclass(frozen=True)
class CurrentnessDimensions:
    """Five-dimensional orthogonal currentness representation."""
    legal_identity_exists: bool = True
    current_regulatory_status: str = "AUTHORIZED"
    current_issuer_status: str = "ACTIVE"
    commercial_availability: str = "PUBLICLY_OFFERED"
    listing_status: str = "LISTED"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> CurrentnessDimensions:
        try:
            d = json.loads(json_str)
            return cls(**d)
        except Exception:
            return cls()


@dataclass(frozen=True)
class CanonicalParentEntity:
    """Represents a statutory fund umbrella (ICAV, SICAV, Trust)."""
    canonical_parent_id: str
    legal_umbrella_name: str
    domicile_iso2: str
    regulatory_jurisdiction: str
    regulator: str
    legal_entity_structure: str
    created_at: str
    updated_at: str
    national_regulator_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSubfund:
    """Represents a statutory sub-fund or compartment."""
    canonical_subfund_id: str
    canonical_parent_id: str
    legal_subfund_name: str
    subfund_currency: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalShareClass:
    """Represents an admitted statutory share class."""
    canonical_share_class_id: str
    isin: str
    canonical_subfund_id: str
    canonical_parent_id: str
    jurisdiction: str
    regulatory_framework: str
    legal_subfund_name: str
    legal_share_class_name: str
    canonical_status: str
    currentness_state: str
    authority_tier: str
    admission_state: str
    admitted_at: str
    admission_gate_id: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalProvenanceRecord:
    """Represents statutory evidence linkage for an admitted share class."""
    provenance_record_id: str
    canonical_share_class_id: str
    evidence_object_id: str
    source_candidate_id: str
    source_document_id: str
    source_url: str
    authority_tier: str
    evidence_type: str
    observed_at: str
    evidence_hash: str
    relationship_type: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalHoldRecord:
    """Represents a quarantined candidate held from admission."""
    hold_record_id: str
    candidate_identifier: str
    hold_category: str
    hold_state: str
    reason: str
    source_candidate_package: str
    entered_at: str
    hold_governance_gate: str
    reopen_condition: str = "EVENT_DRIVEN"
    status: str = "ACTIVE_HOLD"
    resolved_at: Optional[str] = None
    resolution_gate_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalAuditEvent:
    """Immutable audit ledger log entry."""
    audit_event_id: str
    admission_batch_id: str
    canonical_share_class_id: str
    isin: str
    change_type: str
    after_state_json: str
    authority_basis: str
    governance_gate_id: str
    recorded_at: str
    before_state_json: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PopulationScope:
    """Defines the boundary criteria for querying or evaluating canonical population."""
    jurisdiction: str
    regulator: str
    issuer: str
    regulatory_framework: str
    as_of_date: str
    product_type: str = "ETF"
    canonical_version: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSubmission:
    """Normalized input candidate for admission preflight and transactional write."""
    source_candidate_id: str
    isin: str
    legal_umbrella_name: str
    legal_subfund_name: str
    legal_share_class_name: str
    domicile_iso2: str
    regulatory_jurisdiction: str
    authority_tier: str
    evidence_references: Tuple[Dict[str, Any], ...]
    readiness_state: str
    source_artifact_identity: Dict[str, Any]
    subfund_currency: str = "EUR"
    regulator: str = "CBI"
    legal_entity_structure: str = "ICAV"
    national_regulator_code: Optional[str] = None


@dataclass(frozen=True)
class PreflightClassification:
    """Result of preflight validation for a candidate."""
    candidate: CandidateSubmission
    collision_state: CollisionState
    is_valid: bool
    error_message: Optional[str] = None
    existing_canonical_id: Optional[str] = None


@dataclass(frozen=True)
class BatchAdmissionResult:
    """Outcome of an atomic batch admission operation."""
    batch_id: str
    admitted_at: str
    governance_gate_id: str
    total_submitted: int
    admitted_count: int
    no_op_count: int
    rejected_count: int
    admitted_share_class_ids: Tuple[str, ...]
    no_op_share_class_ids: Tuple[str, ...]
    rejections: Tuple[Dict[str, Any], ...]
