"""
Sigstore / Rekor Operational Rooting Closure Models — Epoch 001.
Implements domain data structures and constants for:
- Signer Identity Lifecycle Governance (ASSUMP_005)
- Rekor Consistency Monitoring (ASSUMP_006)
- Sigstore/TUF Trust-Root Governance (ASSUMP_007)
- Local Raw-Evidence Durability & Disaster Recovery (ASSUMP_008)
- Same-Root Multiple-Bundle Substitution Enforcement (ASSUMP_009)
- Timestamp Semantics Separation (ASSUMP_010)
- Interim Natural Rooting Policy Ratification (ASSUMP_011)
- Unattended Workload Identity Contract (ASSUMP_004)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Final

# Policy Identifiers (Sections 13, 20)
SIGNER_IDENTITY_POLICY_ID: Final[str] = "SMART_MONEY_SIGSTORE_SIGNER_IDENTITY_POLICY_V1"
SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_ID: Final[str] = "SMART_MONEY_SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_V1"

# Unattended Workload Identity Contract Specification (Section 12)
UNATTENDED_SIGNER_OIDC_ISSUER_DEFAULT: Final[str] = "https://token.actions.githubusercontent.com"
UNATTENDED_WORKLOAD_SCOPE_DEFAULT: Final[str] = "repo:owner/finance:ref:refs/heads/main"
UNATTENDED_TOKEN_LIFETIME_CLASS: Final[str] = "SHORT_LIVED_EPHEMERAL_WORKLOAD_TOKEN"

# Rekor Consistency Monitoring Defaults (Section 14)
REKOR_CONSISTENCY_MONITORING_MODE: Final[str] = "SIGNED_TREE_HEAD_CONSISTENCY_WITH_FAIL_CLOSED_ALARM"
REKOR_CHECKPOINT_RETENTION_POLICY: Final[str] = "IMMUTABLE_LOCAL_LEDGER_APPEND_ONLY"
REKOR_CONSISTENCY_CHECK_FREQUENCY: Final[str] = "PER_ROOT_COMMIT_AND_SCHEDULED_HOURLY"
REKOR_SPLIT_VIEW_RESPONSE_POLICY: Final[str] = "TERMINAL_HOLD_AND_IMMEDIATE_ADMISSION_HALT"

# TUF Trust Root Constants (Section 15)
DEFAULT_SIGSTORE_TUF_ROOT_VERSION: Final[int] = 15
DEFAULT_SIGSTORE_TRUST_ROOT_SHA256: Final[str] = "6494e21ea73fa7ee769f85f57d5a3e6a08725eae1e38c755fc3517c9e6bc0b66"


class SignerCompromiseStatus(str, Enum):
    """Lifecycle status for a pinned signer identity."""
    ACTIVE = "ACTIVE"
    ROTATED = "ROTATED"
    COMPROMISED = "COMPROMISED"
    REVOKED = "REVOKED"


class UnattendedSigningStatus(str, Enum):
    """Status of unattended workload signing."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


@dataclass(frozen=True)
class SignerIdentityLifecyclePolicy:
    """
    Signer identity lifecycle governance contract (Section 13).
    Defines authorized identities, rotation protocols, and compromise handling.
    """
    policy_id: str
    authorized_interactive_identity: str
    authorized_interactive_issuer: str
    authorized_unattended_workflow: str
    authorized_unattended_issuer: str
    authorized_repository: str
    rotation_procedure: str
    compromise_procedure: str
    historical_valid_bundle_treatment: str
    emergency_suspension_procedure: str
    compromise_status: SignerCompromiseStatus = SignerCompromiseStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorized_interactive_identity": self.authorized_interactive_identity,
            "authorized_interactive_issuer": self.authorized_interactive_issuer,
            "authorized_repository": self.authorized_repository,
            "authorized_unattended_issuer": self.authorized_unattended_issuer,
            "authorized_unattended_workflow": self.authorized_unattended_workflow,
            "compromise_procedure": self.compromise_procedure,
            "compromise_status": self.compromise_status.value,
            "emergency_suspension_procedure": self.emergency_suspension_procedure,
            "historical_valid_bundle_treatment": self.historical_valid_bundle_treatment,
            "policy_id": self.policy_id,
            "rotation_procedure": self.rotation_procedure,
        }


@dataclass(frozen=True)
class RekorCheckpointRecord:
    """Signed tree head checkpoint record from Rekor transparency log (Section 14)."""
    origin: str
    tree_size: int
    root_hash_b64: str
    signature_b64: str
    key_id_b64: str
    raw_checkpoint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id_b64": self.key_id_b64,
            "origin": self.origin,
            "raw_checkpoint": self.raw_checkpoint,
            "root_hash_b64": self.root_hash_b64,
            "signature_b64": self.signature_b64,
            "tree_size": self.tree_size,
        }


@dataclass(frozen=True)
class RekorConsistencyAuditResult:
    """Outcome of Rekor consistency and inclusion audit (Section 14)."""
    status: str
    inclusion_verified: bool
    checkpoint_valid: bool
    split_view_detected: bool
    error_message: Optional[str] = None


@dataclass(frozen=True)
class TufTrustRootConfig:
    """Configuration and state for local Sigstore TUF trust-root governance (Section 15)."""
    tuf_root_version: int
    trust_root_path: str
    trust_root_sha256: str
    media_type: str
    ca_count: int
    tlog_count: int
    tsa_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ca_count": self.ca_count,
            "media_type": self.media_type,
            "tlog_count": self.tlog_count,
            "trust_root_path": self.trust_root_path,
            "trust_root_sha256": self.trust_root_sha256,
            "tsa_count": self.tsa_count,
            "tuf_root_version": self.tuf_root_version,
        }


@dataclass(frozen=True)
class RawEvidenceRecord:
    """Local raw evidence persistence record (Section 16)."""
    accession_or_id: str
    primary_path: str
    backup_path: str
    content_sha256: str
    size_bytes: int
    created_at_utc: str
    backup_verified_at_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accession_or_id": self.accession_or_id,
            "backup_path": self.backup_path,
            "backup_verified_at_utc": self.backup_verified_at_utc,
            "content_sha256": self.content_sha256,
            "created_at_utc": self.created_at_utc,
            "primary_path": self.primary_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RawEvidenceRecoveryResult:
    """Outcome of simulated raw evidence recovery from backup (Section 16)."""
    status: str
    accession_or_id: str
    original_sha256: str
    restored_sha256: str
    hash_match: bool
    root_hash_matches_witness: bool


@dataclass(frozen=True)
class DisaggregatedTimestampRecord:
    """
    Explicit separation of witness timestamps from SEC availability times (Section 18).
    Strictly prevents collapsing distinct time authorities into a single generic timestamp.
    """
    rfc3161_external_timestamp: str
    rfc3161_authority: str = "EXTERNAL_SIGNED_TIMESTAMP"
    rekor_integrated_time: str = ""
    rekor_integrated_time_unix: Optional[int] = None
    rekor_authority: str = "TRANSPARENCY_LOG_METADATA_TIME"
    local_capture_time_interval: str = ""
    local_capture_clock_authority: str = "INDEPENDENT_NTP_CLOCK_AUTHORITY"
    sec_public_availability_observed_at_interval: str = "NOT_APPLICABLE_NON_NATURAL"
    sec_public_availability_authority: str = "SEPARATE_OBSERVER_CONTRACT"
    exact_first_sec_public_time: str = "NOT_ESTABLISHED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_first_sec_public_time": self.exact_first_sec_public_time,
            "local_capture_clock_authority": self.local_capture_clock_authority,
            "local_capture_time_interval": self.local_capture_time_interval,
            "rekor_authority": self.rekor_authority,
            "rekor_integrated_time": self.rekor_integrated_time,
            "rekor_integrated_time_unix": self.rekor_integrated_time_unix,
            "rfc3161_authority": self.rfc3161_authority,
            "rfc3161_external_timestamp": self.rfc3161_external_timestamp,
            "sec_public_availability_authority": self.sec_public_availability_authority,
            "sec_public_availability_observed_at_interval": self.sec_public_availability_observed_at_interval,
        }


@dataclass(frozen=True)
class SigstoreOperationalReviewResult:
    """Outcome of independent Sigstore operational zero-trust review."""
    review_status: str
    mismatch_count: int
    unattended_signing_status: str
    rekor_consistency_verified: bool
    tuf_root_verified: bool
    evidence_durability_verified: bool
    same_root_substitution_rejected: bool
    timestamps_disaggregated: bool
    privacy_screening_passed: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "errors": self.errors,
            "evidence_durability_verified": self.evidence_durability_verified,
            "mismatch_count": self.mismatch_count,
            "privacy_screening_passed": self.privacy_screening_passed,
            "rekor_consistency_verified": self.rekor_consistency_verified,
            "review_status": self.review_status,
            "same_root_substitution_rejected": self.same_root_substitution_rejected,
            "timestamps_disaggregated": self.timestamps_disaggregated,
            "tuf_root_verified": self.tuf_root_verified,
            "unattended_signing_status": self.unattended_signing_status,
        }

