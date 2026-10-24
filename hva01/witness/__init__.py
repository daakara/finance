"""
HVA-01 Independent Witness Verification Package.
Provides canonical evidence contracts, synthetic timestamp anchoring,
RFC 3161 DER wire engine, dual-TSA failover, key trust and compromise evaluation,
and evidence batch verification.
"""

from .adapter import WitnessSessionAdapter
from .client import (
    DualTSAFallbackEngine,
    ProvenanceRecord,
    ProviderAttemptRecord,
    TSAClientError,
    TSAFailClosedError,
    TSAFailureClassification,
    TSAFailoverExhaustedError,
    TSAFallbackState,
    TSAProviderConfig,
    TSATimeoutError,
    TSATransportError,
)
from .models import (
    AttestationPayload,
    GenesisCommitment,
    WitnessAttestation,
    WitnessEnvelope,
)
from .policy import (
    KeyRevocationRecord,
    PolicyEvaluationResult,
    TrustStatus,
    WitnessTrustPolicy,
)
from .timestamp import (
    SyntheticTimestampEngine,
    SyntheticTimestampToken,
    compute_attestation_digest,
)
from .timestamp_wire import (
    ClockDriftExceededError,
    InvalidSignatureError,
    InvalidTimestampTokenError,
    MalformedAsn1Error,
    PKIStatusError,
    RequestResponseBindingError,
    TimestampWireError,
    TSAPolicyMismatchError,
    UntrustedTSACertificateError,
    VerifiedTimestamp,
    create_synthetic_timestamp_response,
    create_timestamp_request,
    parse_and_verify_timestamp_response,
    parse_timestamp_request,
    verify_tsa_certificate_trust,
)
from .verifier import EvidenceBatchVerifier

__all__ = [
    "GenesisCommitment",
    "AttestationPayload",
    "WitnessAttestation",
    "WitnessEnvelope",
    "SyntheticTimestampToken",
    "SyntheticTimestampEngine",
    "compute_attestation_digest",
    "KeyRevocationRecord",
    "TrustStatus",
    "PolicyEvaluationResult",
    "WitnessTrustPolicy",
    "EvidenceBatchVerifier",
    "WitnessSessionAdapter",
    # Wave 4C-1 Wire Engine & Fallback Client
    "VerifiedTimestamp",
    "create_timestamp_request",
    "parse_timestamp_request",
    "parse_and_verify_timestamp_response",
    "verify_tsa_certificate_trust",
    "create_synthetic_timestamp_response",
    "TimestampWireError",
    "MalformedAsn1Error",
    "PKIStatusError",
    "InvalidTimestampTokenError",
    "InvalidSignatureError",
    "UntrustedTSACertificateError",
    "RequestResponseBindingError",
    "ClockDriftExceededError",
    "TSAPolicyMismatchError",
    "DualTSAFallbackEngine",
    "TSAProviderConfig",
    "TSAFallbackState",
    "TSAFailureClassification",
    "TSAClientError",
    "TSATransportError",
    "TSATimeoutError",
    "TSAFailoverExhaustedError",
    "TSAFailClosedError",
    "ProvenanceRecord",
    "ProviderAttemptRecord",
]
