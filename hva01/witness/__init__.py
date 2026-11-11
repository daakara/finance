"""
HVA-01 Independent Witness Verification Package.
Provides canonical evidence contracts, synthetic timestamp anchoring,
key trust and compromise evaluation, and evidence batch verification.
"""

from .models import (
    AttestationPayload,
    GenesisCommitment,
    WitnessAttestation,
    WitnessEnvelope,
)
from .timestamp import (
    SyntheticTimestampEngine,
    SyntheticTimestampToken,
    compute_attestation_digest,
)
from .policy import (
    KeyRevocationRecord,
    PolicyEvaluationResult,
    TrustStatus,
    WitnessTrustPolicy,
)
from .verifier import EvidenceBatchVerifier
from .adapter import WitnessSessionAdapter

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
]
