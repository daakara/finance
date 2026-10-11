"""
Certification Authority Data Models, Proof Envelopes, Receipt V2 Specifications,
Dispute State Machines, and State Composition Structures for Wave 2C Hardening.
Adheres strictly to ARX Terminal Smart Money Governance and SM_CERT_CANON_V1.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

from analyst_dashboard.smart_money.canonicalization import canonicalize_json, compute_canonical_hash


class EnumerationStatus(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class DisputeState(str, Enum):
    """Lifecycle state of an audit dispute."""
    DETECTED = "DETECTED"
    TRIAGE = "TRIAGE"
    ROOT_CAUSE_CLASSIFICATION = "ROOT_CAUSE_CLASSIFICATION"
    TECHNICAL_RECONCILIATION = "TECHNICAL_RECONCILIATION"
    REAUDIT = "REAUDIT"
    GOVERNANCE_ESCALATED = "GOVERNANCE_ESCALATED"
    GOVERNANCE_HOLD = "GOVERNANCE_HOLD"
    RESOLVED = "RESOLVED"


class SlaState(str, Enum):
    """Timer and SLA compliance status."""
    WITHIN_TRIAGE_SLA = "WITHIN_TRIAGE_SLA"
    TRIAGE_OVERDUE = "TRIAGE_OVERDUE"
    ROOT_CAUSE_OVERDUE = "ROOT_CAUSE_OVERDUE"
    RECONCILIATION_OVERDUE = "RECONCILIATION_OVERDUE"
    GOVERNANCE_ESCALATED = "GOVERNANCE_ESCALATED"
    MAX_SLA_EXCEEDED = "MAX_SLA_EXCEEDED"


class EscalationLevel(str, Enum):
    """Governance escalation hierarchy."""
    L0_ENGINEERING = "L0_ENGINEERING"
    L1_TECHNICAL_ADJUDICATION = "L1_TECHNICAL_ADJUDICATION"
    L2_GOVERNANCE_CLOSURE = "L2_GOVERNANCE_CLOSURE"
    L3_GOVERNANCE_HOLD = "L3_GOVERNANCE_HOLD"


class CertificationDisputeState(str, Enum):
    """Impact of dispute on active certified completeness."""
    LAST_UNDISPUTED_ACTIVE = "LAST_UNDISPUTED_ACTIVE"
    CANDIDATE_BLOCKED_BY_DISPUTE = "CANDIDATE_BLOCKED_BY_DISPUTE"
    PRIOR_CERTIFICATION_UNDER_INTEGRITY_REVIEW = "PRIOR_CERTIFICATION_UNDER_INTEGRITY_REVIEW"
    PRIOR_CERTIFICATION_INVALIDATED = "PRIOR_CERTIFICATION_INVALIDATED"
    POST_RESOLUTION_CERTIFICATION_PENDING = "POST_RESOLUTION_CERTIFICATION_PENDING"
    NEW_RECEIPT_CERTIFIED = "NEW_RECEIPT_CERTIFIED"


class ScopeKind(str, Enum):
    GOVERNED_SUBSET = "GOVERNED_SUBSET"
    MARKET_WIDE_AUTHORITATIVE = "MARKET_WIDE_AUTHORITATIVE"


class PipelineState(str, Enum):
    NOT_INITIALIZED = "NOT_INITIALIZED"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class CompletenessState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CERTIFIED = "CERTIFIED"
    BLOCKED_BY_GAPS = "BLOCKED_BY_GAPS"
    INTEGRITY_BREACH = "INTEGRITY_BREACH"


class FreshnessState(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class ScopedGapState(str, Enum):
    NONE = "NONE"
    OPEN_RETRYABLE = "OPEN_RETRYABLE"
    OPEN_DISCOVERY = "OPEN_DISCOVERY"
    OPEN_AMENDMENT = "OPEN_AMENDMENT"
    OPEN_SOURCE_INTEGRITY = "OPEN_SOURCE_INTEGRITY"


class ScanResultState(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    MATCHES = "MATCHES"
    NO_MATCHES = "NO_MATCHES"


class DecisionEligibilityState(str, Enum):
    ELIGIBLE_WITHIN_SCOPE = "ELIGIBLE_WITHIN_SCOPE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class ScopedCapabilityState(str, Enum):
    PIPELINE_PENDING = "PIPELINE_PENDING"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    STALE = "STALE"
    AVAILABLE_WITHIN_SCOPE = "AVAILABLE_WITHIN_SCOPE"
    UNAVAILABLE = "UNAVAILABLE"


class MarketWideCapabilityState(str, Enum):
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    AVAILABLE = "AVAILABLE"


@dataclass
class PartitionEnumerationReceipt:
    """
    Immutable audit receipt of an exhaustive discovery enumeration over a required partition.
    """
    partition_enumeration_receipt_id: str
    partition_id: str
    issuer_cik: str
    universe_build_id: str
    required_partition_set_hash: str
    window_start: str
    window_end: str
    source_contract_version: str
    discovery_policy_version: str
    pagination_policy_version: str
    source_snapshot_manifest_hash: str
    pages_expected_or_discovered: int
    pages_consumed: int
    pagination_exhausted: bool
    continuation_token_remaining: bool
    first_enumeration_accession_set_hash: str
    post_holdback_accession_set_hash: str
    accession_set_stable: bool
    discovery_gap_count: int
    enumeration_status: str
    enumerated_at: str
    receipt_hash: str = ""

    def compute_hash(self) -> str:
        d = asdict(self)
        d.pop("receipt_hash", None)
        return compute_canonical_hash(d)


@dataclass
class SourceSnapshot:
    """
    Content-addressed immutable store of raw SEC responses for offline reconstruction.
    """
    snapshot_id: str
    partition_id: str
    issuer_cik: str
    request_identity: str
    window_start: str
    window_end: str
    raw_payload: str
    payload_hash: str
    pagination_metadata: Dict[str, Any]
    fetched_at: str
    source_contract_version: str


@dataclass
class ExpectedAccession:
    """
    Canonical expected accession record binding source origin and availability.
    """
    accession_number: str
    issuer_cik: str
    public_availability_at: str
    source_snapshot_identity: str
    source_content_hash: str


@dataclass
class GapEvent:
    """
    Append-only gap lifecycle event for deterministic historical gap state reconstruction.
    """
    event_id: str
    event_sequence: int
    gap_id: str
    event_type: str  # GAP_OPENED, GAP_RETRY_ATTEMPTED, GAP_CLASSIFIED, GAP_RESOLVED, GAP_TERMINALLY_EXCLUDED
    partition_id: str
    issuer_cik: str
    gap_type: str
    window_start: str
    window_end: str
    event_payload: Dict[str, Any]
    occurred_at: str
    event_hash: str = ""

    def compute_hash(self) -> str:
        d = asdict(self)
        d.pop("event_hash", None)
        return compute_canonical_hash(d)


@dataclass
class ClosureProof:
    """
    Deterministic closure proof object binding all 16 certification inputs.
    Contains NO wall-clock issuance metadata.
    """
    previous_certified_frontier: str
    candidate_frontier: str
    universe_build_id: str
    universe_content_hash: str
    required_partition_set_hash: str
    partition_coverage_manifest_hash: str
    source_snapshot_manifest_hash: str
    enumeration_receipt_set_hash: str
    expected_accession_set_hash: str
    registered_accession_set_hash: str
    terminal_accession_set_hash: str
    gap_event_prefix_hash: str
    open_gap_set_hash: str
    amendment_state_hash: str
    source_ledger_hash: str
    policy_bundle_hash: str

    def compute_hash(self) -> str:
        return compute_canonical_hash(asdict(self))


@dataclass
class CertificationReceiptV2:
    """
    Cryptographically sealed, Merkle-linked certification receipt envelope V2.
    """
    schema_version: str
    receipt_id: str
    receipt_sequence: int
    previous_receipt_id: str
    previous_receipt_sha256: str
    previous_certified_frontier: str
    candidate_frontier: str
    certified_frontier: str
    closure_proof_sha256: str
    universe_build_id: str
    universe_content_hash: str
    required_partition_count: int
    required_partition_set_hash: str
    certified_at: str
    certifier_component_version: str
    certifier_functional_sha: str
    receipt_sha256: str = ""

    def compute_hash(self) -> str:
        d = asdict(self)
        d.pop("receipt_sha256", None)
        return compute_canonical_hash(d)


@dataclass
class AuditDispute:
    """
    Durable record of an audit dispute between certifier and auditor or between auditors.
    """
    dispute_id: str
    receipt_id: str
    candidate_frontier: str
    dispute_state: str  # DisputeState
    sla_state: str      # SlaState
    escalation_level: str  # EscalationLevel
    certification_state: str  # CertificationDisputeState
    first_divergent_step: str
    minimal_counterexample: Dict[str, Any]
    detected_at: str
    triage_deadline: str
    root_cause_deadline: str
    reconciliation_deadline: str
    governance_deadline: str
    sla_policy_version: str
    sla_policy_sha256: str
    resolved_at: Optional[str] = None
    resolution_artifact_hash: Optional[str] = None


@dataclass
class SmartMoneyCompositeState:
    """
    Scope-safe, formally structured Smart Money state composition.
    Prohibits ambiguous global booleans.
    """
    scope: Dict[str, Any]
    pipeline: str  # PipelineState
    completeness: str  # CompletenessState
    freshness: str  # FreshnessState
    gaps: str  # ScopedGapState
    scan_result: str  # ScanResultState
    decision_eligibility: str  # DecisionEligibilityState
    scoped_capability: str  # ScopedCapabilityState
    market_wide_capability: str  # MarketWideCapabilityState
    certified_through: Optional[str]
    certification_receipt_id: Optional[str]
    active_breach_count: int
    open_gap_count: int
    matched_count: Optional[int]
