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
    Canonical finalization root payload committed to external storage (Sections 6, 8).
    Binds all finalization events and authority status for a finalized run.
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

    def to_dict(self) -> Dict[str, Any]:
        """Produces canonical sorted dictionary representation."""
        return {
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


@dataclass(frozen=True)
class FinalizationRootReceipt:
    """
    Signed external receipt proving root commitment in an independent trust domain (Section 12).
    """
    root_receipt_id: str
    parent_run_id: str
    parent_run_final_hash: str
    finalization_ledger_root_hash: str
    policy_set_hash: str
    provider: str
    receipt_signing_key_id: str
    signature_algorithm: str
    receipt_payload_sha256: str
    signature: str
    created_at_utc: str

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
