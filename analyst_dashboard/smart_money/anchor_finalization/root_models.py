"""
External Finalization Root Domain Models & Canonical State Types.
Implements data structures and enums for:
- Finalization Root Status Lifecycle
- Root Input & Canonical Root Payload
- Signed External Root Receipts
- Root Attempt Ledger Records
- External Storage Configuration Attestations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class FinalizationRootStatus(str, Enum):
    """
    Canonical Finalization Root Status Taxonomy (Section 21).
    Lifecycle states for externally rooting the finalization ledger.
    """
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "REQUIRED"
    PAYLOAD_SEALED = "PAYLOAD_SEALED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    RETRY_WAIT = "RETRY_WAIT"
    COMMIT_INDETERMINATE = "COMMIT_INDETERMINATE"
    CONFIRMED = "CONFIRMED"
    PENDING_EXPIRED = "PENDING_EXPIRED"
    CONFLICT = "CONFLICT"
    TERMINAL_HOLD = "TERMINAL_HOLD"


class RootPhase(str, Enum):
    """Two distinct root purposes (Sections 4, 5, 12)."""
    PRE_AUTHORITY_ADMISSION_ROOT = "PRE_AUTHORITY_ADMISSION_ROOT"
    POST_AUTHORITY_CLOSURE_ROOT = "POST_AUTHORITY_CLOSURE_ROOT"


class PostAuthorityClosureStatus(str, Enum):
    """Canonical lifecycle for post-authority closure root (Section 14)."""
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    EXPIRED = "EXPIRED"
    CONFLICT = "CONFLICT"


class R1ConfirmationStatus(str, Enum):
    """R1 external confirmation status taxonomy (Section 7)."""
    NOT_CONFIRMED = "NOT_CONFIRMED"
    CONFIRMED = "CONFIRMED"
    CONFLICT = "CONFLICT"


class RootLookupResult(str, Enum):
    """Lookup classifications for ambiguous root commit recovery (Section 20)."""
    FOUND_MATCHING = "FOUND_MATCHING"
    FOUND_CONFLICTING = "FOUND_CONFLICTING"
    NOT_FOUND = "NOT_FOUND"
    INDETERMINATE = "INDETERMINATE"


class RootErrorClass(str, Enum):
    """Standardized error classifications for root attempts (Section 19)."""
    NONE = "NONE"
    RETRYABLE_TRANSPORT = "RETRYABLE_TRANSPORT"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    PROVIDER_REJECTION = "PROVIDER_REJECTION"
    INTEGRITY_CONFLICT = "INTEGRITY_CONFLICT"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class FinalizationRootPayload:
    """
    Canonical finalization root payload committed to external storage (Sections 6, 8, 12).
    Binds finalization events, pre/post authority status, and two-phase linkage.
    """
    root_schema_version: str
    root_policy_id: str
    root_policy_version: str
    parent_run_id: str
    parent_run_final_hash: str
    finalization_event_count: int
    first_finalization_sequence: int
    last_finalization_sequence: int
    finalization_event_hashes: List[str]
    authority_eligibility: str
    authority_effective_status: str
    authority_became_effective_at: Optional[str]
    anchor_request_ids: List[str]
    anchor_receipt_hashes: List[str]
    policy_set_hash: str
    created_at_utc: str
    root_phase: Optional[RootPhase] = None
    authority_transition_intent: Optional[str] = None
    authority_effectiveness_precondition: Optional[str] = None
    r1_root_id: Optional[str] = None
    r1_root_hash: Optional[str] = None
    r1_receipt_hash: Optional[str] = None
    authority_transition_event_id: Optional[str] = None
    authority_transition_event_hash: Optional[str] = None
    authority_transition_sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Produces canonical sorted dictionary representation."""
        d: Dict[str, Any] = {
            "anchor_receipt_hashes": list(self.anchor_receipt_hashes),
            "anchor_request_ids": list(self.anchor_request_ids),
            "authority_became_effective_at": self.authority_became_effective_at,
            "authority_effective_status": self.authority_effective_status,
            "authority_eligibility": self.authority_eligibility,
            "created_at_utc": self.created_at_utc,
            "finalization_event_count": self.finalization_event_count,
            "finalization_event_hashes": list(self.finalization_event_hashes),
            "first_finalization_sequence": self.first_finalization_sequence,
            "last_finalization_sequence": self.last_finalization_sequence,
            "parent_run_final_hash": self.parent_run_final_hash,
            "parent_run_id": self.parent_run_id,
            "policy_set_hash": self.policy_set_hash,
            "root_policy_id": self.root_policy_id,
            "root_policy_version": self.root_policy_version,
            "root_schema_version": self.root_schema_version,
        }
        if self.root_phase is not None:
            d["root_phase"] = self.root_phase.value
        if self.authority_transition_intent is not None:
            d["authority_transition_intent"] = self.authority_transition_intent
        if self.authority_effectiveness_precondition is not None:
            d["authority_effectiveness_precondition"] = self.authority_effectiveness_precondition
        if self.r1_root_id is not None:
            d["r1_root_id"] = self.r1_root_id
        if self.r1_root_hash is not None:
            d["r1_root_hash"] = self.r1_root_hash
        if self.r1_receipt_hash is not None:
            d["r1_receipt_hash"] = self.r1_receipt_hash
        if self.authority_transition_event_id is not None:
            d["authority_transition_event_id"] = self.authority_transition_event_id
        if self.authority_transition_event_hash is not None:
            d["authority_transition_event_hash"] = self.authority_transition_event_hash
        if self.authority_transition_sequence is not None:
            d["authority_transition_sequence"] = self.authority_transition_sequence
        return d

    @property
    def root_id(self) -> str:
        if self.root_phase and self.root_phase == RootPhase.POST_AUTHORITY_CLOSURE_ROOT:
            return f"root_r2_{self.parent_run_id}_{self.parent_run_final_hash[:16]}"
        return f"root_{self.parent_run_id}_{self.parent_run_final_hash[:16]}"


@dataclass(frozen=True)
class AuthorityTransitionRecord:
    """
    Immutable atomic record of an authority transition triggered by verified R1 receipt (Sections 8, 9, 11).
    """
    authority_transition_event_id: str
    authority_transition_sequence: int
    parent_run_id: str
    r1_root_id: str
    r1_root_hash: str
    r1_receipt_id: str
    r1_receipt_hash: str
    authority_became_effective_at: str
    idempotency_key: str
    parent_run_final_hash: str = ""
    r1_confirmed_at: str = ""
    authority_scope: str = "ACCESSION_AUTHORITY"
    authorized_accession: str = ""
    previous_finalization_event_hash: str = ""
    authority_transition_event_hash: str = ""
    event_type: str = "AUTHORITY_BECAME_EFFECTIVE"
    authority_effective_status: Any = None
    t_eligible_utc: Optional[str] = None
    t_r1_confirmed_utc: Optional[str] = None
    recorded_at: Optional[str] = None

    def __post_init__(self):
        if self.authority_effective_status is None:
            from analyst_dashboard.smart_money.anchor_finalization.models import AuthorityEffectiveStatus
            object.__setattr__(self, "authority_effective_status", AuthorityEffectiveStatus.EFFECTIVE)
        if not self.r1_confirmed_at and self.t_r1_confirmed_utc:
            object.__setattr__(self, "r1_confirmed_at", self.t_r1_confirmed_utc)
        if self.t_r1_confirmed_utc is None:
            object.__setattr__(self, "t_r1_confirmed_utc", self.r1_confirmed_at)
        if self.t_eligible_utc is None:
            object.__setattr__(self, "t_eligible_utc", self.r1_confirmed_at or self.authority_became_effective_at)
        if self.recorded_at is None:
            object.__setattr__(self, "recorded_at", self.authority_became_effective_at)
        if not self.authority_transition_event_hash:
            from analyst_dashboard.smart_money.anchor_finalization.root_serializer import compute_authority_transition_event_hash
            h = compute_authority_transition_event_hash(
                parent_run_id=self.parent_run_id,
                r1_root_id=self.r1_root_id,
                r1_root_hash=self.r1_root_hash,
                r1_receipt_id=self.r1_receipt_id,
                r1_receipt_hash=self.r1_receipt_hash,
                t_eligible_utc=self.t_eligible_utc,
                t_r1_confirmed_utc=self.t_r1_confirmed_utc,
                authority_became_effective_at=self.authority_became_effective_at,
                authority_transition_sequence=self.authority_transition_sequence,
            )
            object.__setattr__(self, "authority_transition_event_hash", h)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_became_effective_at": self.authority_became_effective_at,
            "authority_scope": self.authority_scope,
            "authority_transition_event_hash": self.authority_transition_event_hash,
            "authority_transition_event_id": self.authority_transition_event_id,
            "authority_transition_sequence": self.authority_transition_sequence,
            "authorized_accession": self.authorized_accession,
            "idempotency_key": self.idempotency_key,
            "parent_run_final_hash": self.parent_run_final_hash,
            "parent_run_id": self.parent_run_id,
            "previous_finalization_event_hash": self.previous_finalization_event_hash,
            "r1_confirmed_at": self.r1_confirmed_at,
            "r1_receipt_hash": self.r1_receipt_hash,
            "r1_receipt_id": self.r1_receipt_id,
            "r1_root_hash": self.r1_root_hash,
            "r1_root_id": self.r1_root_id,
        }


@dataclass(frozen=True)
class PostAuthorityClosureRecord:
    """
    Record tracking post-authority closure lifecycle, deadline and status (Sections 14, 15, 16).
    """
    parent_run_id: str
    closure_status: PostAuthorityClosureStatus
    closure_id: str = ""
    r1_root_id: str = ""
    r2_root_id: Optional[str] = None
    closure_deadline_at: str = ""
    confirmed_at: Optional[str] = None
    created_at_utc: str = ""
    updated_at_utc: str = ""
    authority_transition_event_id: Optional[str] = None
    closure_deadline_utc: Optional[str] = None
    conflict_reason: Optional[str] = None

    def __post_init__(self):
        if not self.closure_deadline_at and self.closure_deadline_utc:
            object.__setattr__(self, "closure_deadline_at", self.closure_deadline_utc)
        elif not self.closure_deadline_utc and self.closure_deadline_at:
            object.__setattr__(self, "closure_deadline_utc", self.closure_deadline_at)


@dataclass(frozen=True)
class FinalizationRootReceipt:
    """
    Signed external receipt proving root commitment in an independent trust domain (Section 12).
    """
    root_receipt_id: str = "rcpt_root_001"
    parent_run_id: str = "run_001"
    parent_run_final_hash: str = "0" * 64
    finalization_ledger_root_hash: str = "0" * 64
    policy_set_hash: str = "0" * 64
    provider: str = "AWS_S3_OBJECT_LOCK_ROOT"
    receipt_signing_key_id: str = "DEFAULT_KEY"
    signature_algorithm: str = "HMAC-SHA256"
    receipt_payload_sha256: str = "0" * 64
    signature: str = ""
    created_at_utc: str = "2026-10-11T12:00:00Z"
    receipt_id: Optional[str] = None
    root_id: Optional[str] = None
    root_hash: Optional[str] = None
    external_provider: Optional[str] = None
    receipt_hash: Optional[str] = None
    confirmed_at_utc: Optional[str] = None
    object_version_id: Optional[str] = "ver_001"
    worm_retention_until: Optional[str] = "2036-10-11T12:00:00Z"
    root_phase: Optional[str] = None

    def __post_init__(self):
        if self.receipt_id and self.root_receipt_id == "rcpt_root_001":
            object.__setattr__(self, "root_receipt_id", self.receipt_id)
        if self.root_hash and self.finalization_ledger_root_hash == "0" * 64:
            object.__setattr__(self, "finalization_ledger_root_hash", self.root_hash)
        if self.receipt_hash and self.receipt_payload_sha256 == "0" * 64:
            object.__setattr__(self, "receipt_payload_sha256", self.receipt_hash)
        if self.external_provider and self.provider == "AWS_S3_OBJECT_LOCK_ROOT":
            object.__setattr__(self, "provider", self.external_provider)
        if self.confirmed_at_utc and self.created_at_utc == "2026-10-11T12:00:00Z":
            object.__setattr__(self, "created_at_utc", self.confirmed_at_utc)

        if self.root_id is None:
            rid = self.root_receipt_id.replace("rcpt_", "") if self.root_receipt_id.startswith("rcpt_") else self.root_receipt_id
            object.__setattr__(self, "root_id", rid)
        if self.root_hash is None:
            object.__setattr__(self, "root_hash", self.finalization_ledger_root_hash)
        if self.receipt_id is None:
            object.__setattr__(self, "receipt_id", self.root_receipt_id)
        if self.receipt_hash is None:
            object.__setattr__(self, "receipt_hash", self.receipt_payload_sha256)
        if self.confirmed_at_utc is None:
            object.__setattr__(self, "confirmed_at_utc", self.created_at_utc)
        if self.external_provider is None:
            object.__setattr__(self, "external_provider", self.provider)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "finalization_ledger_root_hash": self.finalization_ledger_root_hash,
            "parent_run_final_hash": self.parent_run_final_hash,
            "parent_run_id": self.parent_run_id,
            "policy_set_hash": self.policy_set_hash,
            "provider": self.provider,
            "receipt_payload_sha256": self.receipt_payload_sha256,
            "receipt_signing_key_id": self.receipt_signing_key_id,
            "root_receipt_id": self.root_receipt_id,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
        }


@dataclass(frozen=True)
class FinalizationRootAttemptRecord:
    """
    Append-only attempt record for external root submission (Section 19).
    """
    root_attempt_id: str
    root_id: str
    parent_run_id: str
    attempt_number: int
    idempotency_key: str
    request_payload_sha256: str
    request_started_at: str
    request_completed_at: str
    provider_response_hash: str
    result: str
    error_class: RootErrorClass
    lookup_required: bool
    lookup_result: Optional[RootLookupResult] = None


@dataclass(frozen=True)
class RootStorageConfigurationAttestation:
    """
    Storage configuration attestation for external root immutability (Section 11).
    """
    provider_id: str
    bucket_id: str
    object_versioning_enabled: bool
    object_lock_enabled: bool
    retention_mode: str
    retention_until: str
    delete_protection: bool
    overwrite_protection: bool
    immutability_class: str
    attestation_hash: str
    attested_at_utc: str
