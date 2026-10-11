"""
Sigstore / Rekor Interim Root Witness Domain Models — Epoch 001.
Implements data structures and enums for:
- Sigstore/Rekor Interim Trust Profile (Sections 2, 5)
- Sigstore Bundle Representation & Verification Results (Sections 5, 7, 16, 17)
- Synthetic R1/R2 Qualification Payloads & Authority Events (Sections 6, 10, 11)
- Fail-Closed Outage & Duplicate Signing Semantics (Sections 18, 19)
- Independent Reviewer Reconstruction Models (Section 20)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Final

# Pinned Trust Profile Constants (Sections 2, 5)
PINNED_SIGSTORE_IDENTITY: Final[str] = "53128824+daakara@users.noreply.github.com"
PINNED_SIGSTORE_ISSUER: Final[str] = "https://github.com/login/oauth"
IDENTITY_POLICY: Final[str] = "EXACT_PINNED"

EXTERNAL_ROOT_PROFILE: Final[str] = "SIGSTORE_REKOR_INTERIM_V1"
EXTERNAL_ROOT_FUNCTION: Final[str] = "PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS"
EXTERNAL_RAW_EVIDENCE_CUSTODY: Final[str] = "NOT_ESTABLISHED"
S3_OBJECT_LOCK_COMPLIANCE: Final[str] = "NOT_APPLICABLE"
AWS_EXTERNAL_CUSTODY: Final[str] = "DEFERRED"
SIGSTORE_TIMESTAMP_SEMANTICS: Final[str] = "SIGNED_EXTERNAL_TIMESTAMP_EVIDENCE"
SEC_PUBLIC_AVAILABILITY_TIME: Final[str] = "SEPARATE_AUTHORITY"
EXACT_FIRST_SEC_PUBLIC_TIME: Final[str] = "NOT_ESTABLISHED"
SMART_MONEY_MARKET_WIDE_AUTHORITY: Final[str] = "NOT_ESTABLISHED"
EMPIRICAL_SCANNER_QUALITY: Final[str] = "INSUFFICIENT_EVIDENCE"
MODEL_TUNING: Final[str] = "FROZEN"

# RFC / Fulcio / Rekor OIDs & Media Types
SIGSTORE_BUNDLE_MEDIA_TYPE_V03: Final[str] = "application/vnd.dev.sigstore.bundle.v0.3+json"
SIGSTORE_BUNDLE_MEDIA_TYPE_V02: Final[str] = "application/vnd.dev.sigstore.bundle+json;version=0.2"
OID_FULCIO_ISSUER_V1: Final[str] = "1.3.6.1.4.1.57264.1.1"
OID_FULCIO_ISSUER_V2: Final[str] = "1.3.6.1.4.1.57264.1.8"
OID_FULCIO_BUILD_SIGNER_URI: Final[str] = "1.3.6.1.4.1.57264.1.24"
OID_SCT_LIST: Final[str] = "1.3.6.1.4.1.11129.2.4.2"


class SigstoreVerificationStatus(str, Enum):
    """Canonical verification status taxonomy for Sigstore witnessing."""
    PASS = "PASS"
    FAIL = "FAIL"
    FAIL_AS_EXPECTED = "FAIL_AS_EXPECTED"
    TAMPER_DETECTED = "TAMPER_DETECTED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    ISSUER_MISMATCH = "ISSUER_MISMATCH"
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    OUTAGE_BLOCKED = "OUTAGE_BLOCKED"


@dataclass(frozen=True)
class SigstoreBundleModel:
    """
    Parsed in-memory model of a Sigstore keyless verification bundle.
    Encapsulates leaf X.509 certificate, cryptographic signature, Rekor transparency log proof,
    and RFC 3161 signed timestamp response.
    """
    media_type: str
    certificate_der_b64: str
    certificate_identity_san: str
    certificate_issuer_oidc: str
    signature_b64: str
    digest_b64: str
    digest_algorithm: str
    bundle_sha256: str
    tlog_log_index: Optional[int] = None
    tlog_integrated_time: Optional[int] = None
    tlog_tree_size: Optional[int] = None
    tlog_root_hash_b64: Optional[str] = None
    tlog_checkpoint: Optional[str] = None
    tlog_entry_count: int = 1
    rfc3161_timestamp_count: int = 1
    sct_present: bool = True
    raw_bundle_dict: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "certificate_identity_san": self.certificate_identity_san,
            "certificate_issuer_oidc": self.certificate_issuer_oidc,
            "signature_b64": self.signature_b64,
            "digest_b64": self.digest_b64,
            "digest_algorithm": self.digest_algorithm,
            "bundle_sha256": self.bundle_sha256,
            "tlog_log_index": self.tlog_log_index,
            "tlog_integrated_time": self.tlog_integrated_time,
            "tlog_tree_size": self.tlog_tree_size,
            "tlog_entry_count": self.tlog_entry_count,
            "rfc3161_timestamp_count": self.rfc3161_timestamp_count,
            "sct_present": self.sct_present,
        }


@dataclass(frozen=True)
class SigstoreVerificationResult:
    """Cryptographic and policy verification outcome for a Sigstore bundle and payload."""
    is_valid: bool
    signature_verification: str  # "PASS" or "FAIL"
    identity_verification: str   # "PASS" or "FAIL"
    issuer_verification: str     # "PASS" or "FAIL"
    digest_match: bool
    tamper_detected: bool
    tlog_present: bool
    rfc3161_present: bool
    sct_present: bool
    offline_verifiable: bool
    error_detail: Optional[str] = None


@dataclass(frozen=True)
class SyntheticR1Payload:
    """
    Canonical synthetic R1 (pre-authority admission root) payload.
    Binds run ID, root phase, payload digest, policy hash, qualification scope,
    and non-natural qualification boundaries.
    """
    run_id: str
    root_phase: str
    root_hash: str
    payload_sha256: str
    policy_set_hash: str
    qualification_scope: str
    natural_accession: bool
    authority_effective_status: str
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_effective_status": self.authority_effective_status,
            "created_at_utc": self.created_at_utc,
            "natural_accession": self.natural_accession,
            "payload_sha256": self.payload_sha256,
            "policy_set_hash": self.policy_set_hash,
            "qualification_scope": self.qualification_scope,
            "root_hash": self.root_hash,
            "root_phase": self.root_phase,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class SyntheticAuthorityEvent:
    """
    Test-only transition event record binding R1 root and bundle hashes.
    Guarantees no production or natural authority is granted.
    """
    transition_event_id: str
    transition_event_hash: str
    parent_run_id: str
    sequence: int
    r1_root_hash: str
    r1_bundle_hash: str
    policy_set_hash: str
    authority_event_scope: str
    production_authority_granted: bool
    natural_accession_authority_granted: bool
    recorded_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_event_scope": self.authority_event_scope,
            "natural_accession_authority_granted": self.natural_accession_authority_granted,
            "parent_run_id": self.parent_run_id,
            "policy_set_hash": self.policy_set_hash,
            "production_authority_granted": self.production_authority_granted,
            "r1_bundle_hash": self.r1_bundle_hash,
            "r1_root_hash": self.r1_root_hash,
            "recorded_at_utc": self.recorded_at_utc,
            "sequence": self.sequence,
            "transition_event_hash": self.transition_event_hash,
            "transition_event_id": self.transition_event_id,
        }


@dataclass(frozen=True)
class SyntheticR2Payload:
    """
    Canonical synthetic R2 (post-authority closure root) payload.
    Binds run ID, R1 root hash, R1 bundle hash, authority event hash,
    authority event sequence, policy hash, and qualification scope.
    """
    run_id: str
    root_phase: str
    root_hash: str
    payload_sha256: str
    r1_root_hash: str
    r1_bundle_hash: str
    authority_event_hash: str
    authority_event_sequence: int
    policy_set_hash: str
    qualification_scope: str
    authority_grant_function: str
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_event_hash": self.authority_event_hash,
            "authority_event_sequence": self.authority_event_sequence,
            "authority_grant_function": self.authority_grant_function,
            "created_at_utc": self.created_at_utc,
            "payload_sha256": self.payload_sha256,
            "policy_set_hash": self.policy_set_hash,
            "qualification_scope": self.qualification_scope,
            "r1_bundle_hash": self.r1_bundle_hash,
            "r1_root_hash": self.r1_root_hash,
            "root_hash": self.root_hash,
            "root_phase": self.root_phase,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class SigstoreInterimReviewerResult:
    """Outcome of independent reviewer reconstruction."""
    status: str
    reconstruction_mismatches: int
    r1_signature_valid: bool
    r1_identity_valid: bool
    r2_signature_valid: bool
    r2_identity_valid: bool
    r2_r1_linkage_valid: bool
    policy_set_hash_valid: bool
    privacy_review_passed: bool
    error_messages: List[str] = field(default_factory=list)
