"""
analyst_dashboard/security_master/source_governance_models.py

Immutable Evidence, Four-Layer Identity, Bitemporal Governance,
and Auditable Decision Ledger Models for ARX Security Master Sprint 2A.

Invariants Enforced:
- Raw evidence is immutable; raw provider records are never mutated in place.
- Four distinct identity layers: Issuer -> Security -> Listing, with ProviderInstrument mapping.
- SYMBOL_IS_IDENTITY = NO
- ALPACA_UUID_IS_UNIVERSAL_CROSS_PROVIDER_IDENTITY = NO
- FIRST_SEEN_AT != EFFECTIVE_FROM (no backdating of observation time).
- CURRENT_ALPACA_LIST MUST NEVER BE USED AS A HISTORICAL_POINT_IN_TIME_UNIVERSE.
- Every canonical fact is attributable to immutable evidence, a versioned policy,
  an explicit as_of, and a deterministic closed decision input hash.
"""

from __future__ import annotations

import json
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict


# =====================================================================
# Canonical Deterministic Serialization & Hashing
# =====================================================================

def canonical_json_dumps(obj: Any) -> str:
    """
    Produces deterministic UTF-8 JSON byte representations invariant to
    dict insertion order, key order, or whitespace.
    """
    def _normalize(val: Any) -> Any:
        if isinstance(val, dict):
            return {str(k): _normalize(v) for k, v in sorted(val.items())}
        elif isinstance(val, (list, tuple)):
            return [_normalize(x) for x in val]
        elif isinstance(val, set):
            return [_normalize(x) for x in sorted(list(val))]
        elif isinstance(val, Enum):
            return val.value
        elif hasattr(val, "model_dump"):
            return _normalize(val.model_dump())
        elif hasattr(val, "to_dict"):
            return _normalize(val.to_dict())
        return val

    normalized = _normalize(obj)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(obj: Any) -> str:
    """Computes deterministic SHA-256 digest of canonical serialized representation."""
    serialized = canonical_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


# =====================================================================
# Domain Enums
# =====================================================================

class SourceRole(str, Enum):
    PRIMARY_SECURITY_MASTER = "PRIMARY_SECURITY_MASTER"
    REFERENCE_AUTHORITY = "REFERENCE_AUTHORITY"
    CORROBORATING_PROVIDER = "CORROBORATING_PROVIDER"
    NON_AUTHORITATIVE = "NON_AUTHORITATIVE"


class SnapshotStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALID = "VALID"
    QUARANTINED = "QUARANTINED"
    SUPERSEDED = "SUPERSEDED"


class ConflictSeverity(str, Enum):
    S0_INFO = "S0_INFO"                       # Cosmetic / nonsemantic
    S1_WARNING = "S1_WARNING"                 # Genuine disagreement with no material downstream impact
    S2_DEGRADED = "S2_DEGRADED"               # Material disagreement resolved uniquely & deterministically by frozen policy
    S3_BLOCKING = "S3_BLOCKING"               # Material disagreement with no unique deterministic governed resolution
    S4_INTEGRITY_FAILURE = "S4_INTEGRITY_FAILURE" # Evidence/population construction itself cannot be trusted


class ConflictResolution(str, Enum):
    AGREED = "AGREED"
    RESOLVED_BY_PRECEDENCE = "RESOLVED_BY_PRECEDENCE"
    RESOLVED_BY_TEMPORAL_POLICY = "RESOLVED_BY_TEMPORAL_POLICY"
    MANUALLY_ADJUDICATED = "MANUALLY_ADJUDICATED"
    UNRESOLVED = "UNRESOLVED"


class MembershipState(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNRESOLVED = "UNRESOLVED"


class ListingState(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELISTED = "DELISTED"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class MembershipTransitionType(str, Enum):
    ADDED_NEW_LISTING = "ADDED_NEW_LISTING"
    ADDED_RELISTING = "ADDED_RELISTING"
    ADDED_PROVIDER_SCOPE_CHANGE = "ADDED_PROVIDER_SCOPE_CHANGE"
    REMOVED_DELISTING = "REMOVED_DELISTING"
    REMOVED_MERGER = "REMOVED_MERGER"
    REMOVED_SECURITY_CONVERSION = "REMOVED_SECURITY_CONVERSION"
    REMOVED_PROVIDER_SCOPE_CHANGE = "REMOVED_PROVIDER_SCOPE_CHANGE"
    SYMBOL_CHANGED = "SYMBOL_CHANGED"
    EXCHANGE_TRANSFERRED = "EXCHANGE_TRANSFERRED"
    LISTING_STATUS_CHANGED = "LISTING_STATUS_CHANGED"
    UNRESOLVED_ADDITION = "UNRESOLVED_ADDITION"
    UNRESOLVED_REMOVAL = "UNRESOLVED_REMOVAL"


class SurvivorshipStatus(str, Enum):
    SURVIVORSHIP_SAFE = "SURVIVORSHIP_SAFE"
    CURRENT_ONLY_UNIVERSE = "CURRENT_ONLY_UNIVERSE"
    PARTIAL_HISTORICAL_MEMBERSHIP = "PARTIAL_HISTORICAL_MEMBERSHIP"
    SURVIVORSHIP_RISK = "SURVIVORSHIP_RISK"
    UNKNOWN = "UNKNOWN"


class HistoricalMembershipAuthority(str, Enum):
    POINT_IN_TIME_VERIFIED = "POINT_IN_TIME_VERIFIED"
    CURRENT_ONLY = "CURRENT_ONLY"
    PARTIAL_HISTORY = "PARTIAL_HISTORY"
    UNRESOLVED = "UNRESOLVED"


class QuarantineScope(str, Enum):
    RECORD_QUARANTINE = "RECORD_QUARANTINE"
    LISTING_QUARANTINE = "LISTING_QUARANTINE"
    SECURITY_QUARANTINE = "SECURITY_QUARANTINE"
    SOURCE_SNAPSHOT_QUARANTINE = "SOURCE_SNAPSHOT_QUARANTINE"
    CANONICAL_RECONCILIATION_QUARANTINE = "CANONICAL_RECONCILIATION_QUARANTINE"


class PromotionStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    STALE_REJECTED = "STALE_REJECTED"


class DriftClassification(str, Enum):
    NO_DRIFT = "NO_DRIFT"
    COMPATIBLE_DRIFT = "COMPATIBLE_DRIFT"
    BLOCKING_DRIFT = "BLOCKING_DRIFT"


# =====================================================================
# Reason Code Taxonomy
# =====================================================================

class ReasonCode(str, Enum):
    IDENTITY_RESOLVED = "IDENTITY_RESOLVED"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    PRIMARY_AUTHORITY_MISSING = "PRIMARY_AUTHORITY_MISSING"
    REFERENCE_SOURCE_STALE = "REFERENCE_SOURCE_STALE"
    SECURITY_TYPE_CONFLICT = "SECURITY_TYPE_CONFLICT"
    LISTING_STATUS_CONFLICT = "LISTING_STATUS_CONFLICT"
    UNSUPPORTED_PROVIDER_ENUM = "UNSUPPORTED_PROVIDER_ENUM"
    TEMPORAL_EFFECTIVE_DATE_UNKNOWN = "TEMPORAL_EFFECTIVE_DATE_UNKNOWN"
    SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"
    UNKNOWN_SECURITY_TYPE = "UNKNOWN_SECURITY_TYPE"
    UNRESOLVED_REMOVAL = "UNRESOLVED_REMOVAL"
    UNRESOLVED_ADDITION = "UNRESOLVED_ADDITION"
    PRECEDENCE_RESOLVED = "PRECEDENCE_RESOLVED"
    STALE_EVIDENCE_REJECTED = "STALE_EVIDENCE_REJECTED"
    COLLISION_FAIL_CLOSED = "COLLISION_FAIL_CLOSED"
    MANUALLY_ADJUDICATED = "MANUALLY_ADJUDICATED"


# =====================================================================
# Layer 1: Raw Source Evidence Models
# =====================================================================

class RawSourceRecord(BaseModel):
    """Immutable observation record from a market data provider."""
    source_id: str
    source_snapshot_id: str
    provider_record_id: str
    provider_symbol: str
    raw_payload: Dict[str, Any]
    raw_record_hash: str = ""
    observed_at: str
    effective_as_of: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.raw_record_hash:
            # Deterministically hash payload + metadata
            computed = canonical_hash({
                "source_id": self.source_id,
                "provider_record_id": self.provider_record_id,
                "provider_symbol": self.provider_symbol,
                "raw_payload": self.raw_payload,
            })
            object.__setattr__(self, "raw_record_hash", computed)


class RawSourceSnapshot(BaseModel):
    """Immutable collection of raw source records representing a point-in-time observation."""
    source_snapshot_id: str
    source_id: str
    source_authority_version: str
    retrieved_at: str
    effective_as_of: str
    population_temporal_scope: str = "CURRENT_PROVIDER_POPULATION"
    records: List[RawSourceRecord] = Field(default_factory=list)
    raw_record_count: int = 0
    source_population_hash: str = ""
    snapshot_status: SnapshotStatus = SnapshotStatus.CANDIDATE
    implementation_sha: str = ""

    model_config = ConfigDict(frozen=True)

    def compute_population_hash(self) -> str:
        """Computes order-independent hash across all raw records in snapshot."""
        record_hashes = sorted([r.raw_record_hash for r in self.records])
        combined = "\n".join(record_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# =====================================================================
# Layer 2: Four-Layer Identity Models
# =====================================================================

class CanonicalIssuer(BaseModel):
    """Corporate or entity-level authority (1 Issuer -> N Securities)."""
    canonical_issuer_id: str
    issuer_name: str
    cik: Optional[str] = None
    country_of_incorporation: Optional[str] = None
    created_at: str
    last_verified_at: str

    model_config = ConfigDict(frozen=True)


class CanonicalSecurity(BaseModel):
    """Specific financial security of an issuer (1 Security -> N Market Listings)."""
    canonical_security_id: str
    canonical_issuer_id: Optional[str] = None
    security_type: str = "UNKNOWN"
    share_class: Optional[str] = None
    share_class_figi: Optional[str] = None
    is_voting: Optional[bool] = None
    security_hash: str = ""

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.security_hash:
            computed = canonical_hash({
                "security_id": self.canonical_security_id,
                "issuer_id": self.canonical_issuer_id,
                "security_type": self.security_type,
                "share_class": self.share_class,
                "share_class_figi": self.share_class_figi,
            })
            object.__setattr__(self, "security_hash", computed)


class CanonicalListing(BaseModel):
    """Specific venue market listing of a security (1 Listing -> 1 Security)."""
    canonical_listing_id: str
    canonical_security_id: str
    symbol: str
    canonical_mic: str
    listing_status: ListingState = ListingState.UNKNOWN
    composite_figi: Optional[str] = None
    effective_from: str
    effective_to: Optional[str] = None
    listing_hash: str = ""

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.listing_hash:
            computed = canonical_hash({
                "listing_id": self.canonical_listing_id,
                "security_id": self.canonical_security_id,
                "symbol": self.symbol,
                "canonical_mic": self.canonical_mic,
                "listing_status": self.listing_status.value,
                "composite_figi": self.composite_figi,
                "effective_from": self.effective_from,
                "effective_to": self.effective_to,
            })
            object.__setattr__(self, "listing_hash", computed)


class ProviderInstrumentRecord(BaseModel):
    """Mapping representation for an external provider's instrument record."""
    provider_name: str
    provider_record_id: str
    provider_symbol: str
    mapped_listing_id: Optional[str] = None
    mapped_security_id: Optional[str] = None
    effective_from: str
    effective_to: Optional[str] = None

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Layer 3: Decision & Conflict Ledgers
# =====================================================================

class CanonicalFieldDecision(BaseModel):
    """Immutable audit record for a single resolved canonical field value."""
    field_decision_id: str
    canonical_security_id: Optional[str] = None
    canonical_listing_id: Optional[str] = None
    canonical_field: str
    canonical_value: Any
    winning_source_id: str
    winning_snapshot_id: str
    winning_record_hash: str
    competing_source_ids: List[str] = Field(default_factory=list)
    competing_record_hashes: List[str] = Field(default_factory=list)
    field_policy_id: str
    field_policy_version: str
    field_policy_hash: str
    rule_id: str
    severity: ConflictSeverity = ConflictSeverity.S0_INFO
    as_of: str
    decision_input_hash: str = ""
    decision_hash: str = ""
    implementation_sha: str = ""

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.decision_hash:
            d_hash = canonical_hash({
                "field_decision_id": self.field_decision_id,
                "canonical_field": self.canonical_field,
                "canonical_value": str(self.canonical_value),
                "winning_source_id": self.winning_source_id,
                "winning_record_hash": self.winning_record_hash,
                "field_policy_id": self.field_policy_id,
                "field_policy_version": self.field_policy_version,
                "decision_input_hash": self.decision_input_hash,
                "as_of": self.as_of,
            })
            object.__setattr__(self, "decision_hash", d_hash)


class SourceConflictRecord(BaseModel):
    """First-class conflict record detailing multi-provider disagreement."""
    source_conflict_id: str
    canonical_field: str
    canonical_listing_id: Optional[str] = None
    canonical_security_id: Optional[str] = None
    source_a: str
    value_a: Any
    evidence_a_hash: str
    source_b: str
    value_b: Any
    evidence_b_hash: str
    resolution: ConflictResolution
    canonical_value: Any
    severity: ConflictSeverity
    identity_impact: str = "NONE"      # NONE, POSSIBLE, DEFINITE
    denominator_impact: str = "NONE"   # NONE, POSSIBLE, DEFINITE
    eligibility_impact: str = "NONE"   # NONE, POSSIBLE, DEFINITE
    readiness_impact: str = "NONE"     # NONE, POSSIBLE, DEFINITE
    policy_id: str
    policy_version: str
    decision_hash: str = ""

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Layer 4: Temporal Membership & Survivorship Ledgers
# =====================================================================

class MembershipEvent(BaseModel):
    """Immutable membership state change event for a market listing."""
    membership_event_id: str
    canonical_listing_id: str
    source_snapshot_id: str
    source_authority_id: str
    membership_state: MembershipState
    listing_state: ListingState
    effective_from: str
    effective_to: Optional[str] = None
    effective_time_authority: str = "OBSERVED_POINT_IN_TIME"
    observed_at: str
    transition_type: MembershipTransitionType
    reason_code: ReasonCode
    predecessor_event_id: Optional[str] = None
    source_record_hash: str
    decision_hash: str

    model_config = ConfigDict(frozen=True)


class BitemporalCorrectionRecord(BaseModel):
    """Append-only correction preserving both valid time and system belief time."""
    correction_id: str
    new_decision_id: str
    corrects_decision_id: str
    corrected_effective_from: str
    corrected_effective_to: Optional[str] = None
    new_evidence_hash: str
    correction_reason: str
    observed_at: str

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Layer 5: Enrichment Generation Accounting
# =====================================================================

class EnrichmentGenerationRecord(BaseModel):
    """Tracks asynchronous or rate-limited enrichment runs (e.g. OpenFIGI mapping)."""
    enrichment_generation_id: str
    source_snapshot_id: str
    enrichment_policy_hash: str
    started_at: str
    completed_at: Optional[str] = None
    requested_count: int = 0
    completed_count: int = 0
    cached_count: int = 0
    rate_limited_count: int = 0
    failed_count: int = 0
    unresolved_count: int = 0

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Layer 6: Manual Adjudication & Source Failover
# =====================================================================

class ManualAdjudicationRecord(BaseModel):
    """Scoped, effective-dated, auditable human adjudication."""
    adjudication_id: str
    conflict_id: str
    decision_value: Any
    scope: str
    effective_from: str
    expires_at: Optional[str] = None
    reason: str
    authorized_by_role: str
    approved_at: str

    model_config = ConfigDict(frozen=True)


class SourceFailoverRecord(BaseModel):
    """Formal audit record of an explicit, authorized source authority transition."""
    authority_change_id: str
    old_source: str
    new_source: str
    reason: str
    effective_at: str
    membership_differential: int
    field_differential: int
    authorized_by: str
    approved_at: str

    model_config = ConfigDict(frozen=True)


# =====================================================================
# Layer 7: Canonical Generation & Lifecycle
# =====================================================================

class CanonicalGeneration(BaseModel):
    """Complete, immutable candidate or active canonical reconciliation generation."""
    generation_id: str
    predecessor_generation_id: Optional[str] = None
    source_snapshot_id: str
    policy_generation_id: str
    as_of: str
    build_hash: str = ""
    reconciled_listings: Dict[str, CanonicalListing] = Field(default_factory=dict)
    reconciled_securities: Dict[str, CanonicalSecurity] = Field(default_factory=dict)
    field_decisions: List[CanonicalFieldDecision] = Field(default_factory=list)
    conflicts: List[SourceConflictRecord] = Field(default_factory=list)
    membership_events: List[MembershipEvent] = Field(default_factory=list)
    accounting_summary: Dict[str, int] = Field(default_factory=dict)
    validation_status: str = "PENDING"
    promotion_status: PromotionStatus = PromotionStatus.CANDIDATE
    promoted_at: Optional[str] = None
    canonical_schema_version: str = "2.0.0"
    minimum_reader_contract_version: str = "2.0.0"

    model_config = ConfigDict(frozen=True)

    def compute_build_hash(self) -> str:
        """Computes deterministic reconciliation build hash."""
        listing_hashes = sorted([l.listing_hash for l in self.reconciled_listings.values()])
        decision_hashes = sorted([d.decision_hash for d in self.field_decisions])
        payload = {
            "generation_id": self.generation_id,
            "predecessor": self.predecessor_generation_id,
            "source_snapshot": self.source_snapshot_id,
            "policy": self.policy_generation_id,
            "as_of": self.as_of,
            "listing_hashes": listing_hashes,
            "decision_hashes": decision_hashes,
            "accounting": self.accounting_summary,
        }
        return canonical_hash(payload)
