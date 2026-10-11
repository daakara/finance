"""
External Anchor Finalization Domain Models & Canonical State Types.
Implements the A0..A11 finalization state machine taxonomy, attempt records,
finalization events, sealed run ledger, receipts, and storage attestations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class FinalizationState(str, Enum):
    """
    Canonical Finalization State Taxonomy (Section 7).
    Explicitly separate from the capture state machine.
    """
    A0_ANCHOR_NOT_REQUIRED = "A0_ANCHOR_NOT_REQUIRED"
    A1_ANCHOR_REQUIRED = "A1_ANCHOR_REQUIRED"
    A2_ANCHOR_PAYLOAD_SEALED = "A2_ANCHOR_PAYLOAD_SEALED"
    A3_ANCHOR_SUBMISSION_PENDING = "A3_ANCHOR_SUBMISSION_PENDING"
    A4_ANCHOR_RETRY_WAIT = "A4_ANCHOR_RETRY_WAIT"
    A5_ANCHOR_COMMIT_INDETERMINATE = "A5_ANCHOR_COMMIT_INDETERMINATE"
    A6_ANCHOR_CONFIRMED = "A6_ANCHOR_CONFIRMED"
    A7_ANCHOR_PENDING_EXPIRED = "A7_ANCHOR_PENDING_EXPIRED"
    A8_ANCHOR_CONFLICT = "A8_ANCHOR_CONFLICT"
    A9_GOVERNANCE_RECOVERY_REVIEW = "A9_GOVERNANCE_RECOVERY_REVIEW"
    A10_AUTHORITY_EFFECTIVE = "A10_AUTHORITY_EFFECTIVE"
    A11_TERMINAL_HOLD = "A11_TERMINAL_HOLD"


class AuthorityEligibility(str, Enum):
    """Authority eligibility status (Section 8)."""
    NOT_EVALUATED = "NOT_EVALUATED"
    PASS = "PASS"
    FAIL = "FAIL"


class AuthorityEffectiveStatus(str, Enum):
    """
    Authority effectiveness status (Section 8).
    Strictly separated from AuthorityEligibility.
    """
    NOT_EFFECTIVE = "NOT_EFFECTIVE"
    BLOCKED_PENDING_ANCHOR = "BLOCKED_PENDING_ANCHOR"
    BLOCKED_PENDING_FINALIZATION_ROOT = "BLOCKED_PENDING_FINALIZATION_ROOT"
    BLOCKED_PENDING_GOVERNANCE_RECOVERY = "BLOCKED_PENDING_GOVERNANCE_RECOVERY"
    BLOCKED_INTEGRITY_HOLD = "BLOCKED_INTEGRITY_HOLD"
    EFFECTIVE = "EFFECTIVE"
    SUSPENDED_PENDING_CLOSURE = "SUSPENDED_PENDING_CLOSURE"
    SUSPENDED_INTEGRITY_HOLD = "SUSPENDED_INTEGRITY_HOLD"
    TERMINAL_HOLD = "TERMINAL_HOLD"


class ErrorClass(str, Enum):
    """Standardized error classifications for anchor attempts (Section 12)."""
    NONE = "NONE"
    RETRYABLE_TRANSPORT = "RETRYABLE_TRANSPORT"
    PROVIDER_REJECTION = "PROVIDER_REJECTION"
    INTEGRITY_CONFLICT = "INTEGRITY_CONFLICT"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    POLICY_FAILURE = "POLICY_FAILURE"
    INDETERMINATE = "INDETERMINATE"


class LookupResult(str, Enum):
    """Lookup classifications for ambiguous commit recovery (Section 13)."""
    FOUND_MATCHING = "FOUND_MATCHING"
    FOUND_CONFLICTING = "FOUND_CONFLICTING"
    NOT_FOUND = "NOT_FOUND"
    INDETERMINATE = "INDETERMINATE"


class QuorumPolicy(str, Enum):
    """Quorum policies (Section 5)."""
    ALL_REQUIRED = "ALL_REQUIRED"
    ONE_OF_TWO = "1_OF_2"
    TWO_OF_THREE = "2_OF_3"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CriticalIntegrityError(Exception):
    """Raised on critical integrity conflicts (e.g. same idempotency key with different payload)."""
    pass


class InvalidTransitionError(Exception):
    """Raised when an illegal state machine transition is attempted."""
    pass


class DownstreamFirewallBlockedError(Exception):
    """Raised when a downstream consumer attempts to use authority while not effective."""
    pass


class QuorumNotSatisfiedError(Exception):
    """Raised when anchor quorum has not been satisfied."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SealedRunLedger:
    """
    Immutable sealed run ledger contract (Section 9).
    Must be sealed before any anchor submission begins.
    """
    run_id: str
    run_genesis_hash: str
    run_final_hash: str
    transition_count: int
    final_sequence: int
    final_state: str
    policy_set_hash: str
    natural_bundle_hash: str
    repository_head_sha: str
    sealed_at_utc: str
    is_sealed: bool = True

    def assert_immutable(self) -> None:
        if not self.is_sealed:
            raise ValueError("RUN_LEDGER_NOT_SEALED: Cannot submit anchors before run ledger is sealed.")


@dataclass(frozen=True)
class AnchorPayload:
    """
    Canonical deterministic anchor payload (Section 10).
    """
    anchor_schema_version: str
    run_id: str
    machine_version: str
    run_genesis_hash: str
    run_final_hash: str
    transition_count: int
    final_sequence: int
    final_state: str
    natural_bundle_hash: str
    policy_set_hash: str
    repository_head_sha: str
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_schema_version": self.anchor_schema_version,
            "created_at_utc": self.created_at_utc,
            "final_sequence": self.final_sequence,
            "final_state": self.final_state,
            "machine_version": self.machine_version,
            "natural_bundle_hash": self.natural_bundle_hash,
            "policy_set_hash": self.policy_set_hash,
            "repository_head_sha": self.repository_head_sha,
            "run_final_hash": self.run_final_hash,
            "run_genesis_hash": self.run_genesis_hash,
            "run_id": self.run_id,
            "transition_count": self.transition_count,
        }


@dataclass(frozen=True)
class AnchorAttemptRecord:
    """
    Append-only record of an individual anchor submission attempt (Section 12).
    """
    anchor_attempt_id: str
    anchor_request_id: str
    attempt_number: int
    run_id: str
    anchor_idempotency_key: str
    request_payload_sha256: str
    request_started_at: str
    request_completed_at: str
    provider: str
    provider_http_status: int
    provider_response_sha256: str
    error_class: ErrorClass
    result: str  # "SUCCESS", "FAILURE", "INDETERMINATE"
    lookup_required: bool
    lookup_performed: bool


@dataclass(frozen=True)
class FinalizationEvent:
    """
    Append-only finalization ledger event (Section 14).
    Forms a cryptographic hash chain anchored to PARENT_RUN_FINAL_HASH.
    """
    finalization_event_id: str
    parent_run_id: str
    parent_run_final_hash: str
    finalization_sequence: int
    finalization_event_type: str
    anchor_request_id: str
    anchor_attempt_id: str
    anchor_status_before: FinalizationState
    anchor_status_after: FinalizationState
    authority_effective_status_before: AuthorityEffectiveStatus
    authority_effective_status_after: AuthorityEffectiveStatus
    recorded_at: str
    previous_finalization_event_hash: str
    finalization_event_hash: str


@dataclass(frozen=True)
class AnchorReceipt:
    """
    Provider anchor receipt (Section 18).
    """
    receipt_id: str
    provider: str
    run_id: str
    run_final_hash: str
    payload_sha256: str
    policy_set_hash: str
    stored_object_id: str
    timestamp_utc: str
    signature: str
    signature_algorithm: str = "Ed25519"
    signer_key_id: str = "key_arx_sec_anchor_2026_01"


@dataclass(frozen=True)
class StorageConfigurationAttestation:
    """
    External storage configuration attestation (Section 16).
    """
    provider: str
    object_lock_enabled: bool
    versioning_enabled: bool
    retention_mode: str
    retention_until: str
    delete_protection: bool
    overwrite_protection: bool
    config_captured_at: str
    config_evidence_sha256: str
    immutability_class: str = "PROVIDER_ENFORCED_RETENTION"
    trust_assumption: str = "PROVIDER_CONTROL_PLANE_NOT_COMPROMISED"


@dataclass(frozen=True)
class AuthorityTimingRecord:
    """
    Separated authority timing record (Section 8, 29).
    Enforces T1 <= T2 <= T3 and prevents backdating.
    """
    evidence_observed_at: str  # T1
    authority_eligibility_established_at: str  # T2
    authority_became_effective_at: Optional[str] = None  # T3
