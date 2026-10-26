"""
HVA-01 Key Trust, Compromise Horizon and Historical Revocation Policy Evaluator.
Implements the 4-level signature trust taxonomy, RFC 5280 compromise horizon checks,
and fail-closed indeterminate trust evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..crypto.signatures import verify_detached
from .models import WitnessEnvelope
from .timestamp import SyntheticTimestampEngine, compute_attestation_digest


class TrustStatus(str, Enum):
    VALID_HISTORICAL_RECORD = "VALID_HISTORICAL_RECORD"
    REJECTED_REVOKED_KEY = "REJECTED_REVOKED_KEY"
    REJECTED_SUSPECTED_COMPROMISE = "REJECTED_SUSPECTED_COMPROMISE"
    REJECTED_UNBOUND_SIGNATURE = "REJECTED_UNBOUND_SIGNATURE"
    REJECTED_UNTRUSTED_TSA = "REJECTED_UNTRUSTED_TSA"
    REJECTED_UNPINNED_KEY = "REJECTED_UNPINNED_KEY"
    REJECTED_INVALID_SIGNATURE = "REJECTED_INVALID_SIGNATURE"
    INDETERMINATE_TRUST = "INDETERMINATE_TRUST"


@dataclass(frozen=True)
class KeyRevocationRecord:
    """
    RFC 5280-compliant revocation record tracking both administrative revocation time
    and earliest suspected compromise horizon (invalidityDate).
    """
    key_fingerprint: str
    revoked_at_utc: str
    earliest_compromise_utc: Optional[str] = None  # None, ISO string, or "UNKNOWN"
    reason: str = "keyCompromise"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """
    Formal verdict issued by the trust policy evaluator.
    """
    status: TrustStatus
    is_accepted: bool
    rejection_reason: Optional[str] = None
    details: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.details is None:
            object.__setattr__(self, "details", {})


class WitnessTrustPolicy:
    """
    Enforces repository-pinned public keys and historical validity rules:
    Status = VALID iff:
      - Public key is in pinned manifest
      - Ed25519 signature is mathematically valid
      - RFC 3161 timestamp covers D_attest = SHA256(Root || Key || Sig)
      - T_TSA < T_revocation
      - T_TSA < T_earliest_compromise
      - Compromise timing is not indeterminate
    """

    def __init__(self) -> None:
        self._pinned_keys: Dict[str, ed25519.Ed25519PublicKey] = {}
        self._revocation_records: Dict[str, KeyRevocationRecord] = {}

    def pin_public_key(self, fingerprint: str, public_key: ed25519.Ed25519PublicKey) -> None:
        """Register a trusted witness public key in the pinned manifest."""
        self._pinned_keys[fingerprint] = public_key

    def record_revocation(self, revocation: KeyRevocationRecord) -> None:
        """Add an explicit revocation record."""
        self._revocation_records[revocation.key_fingerprint] = revocation

    def evaluate_attestation(
        self,
        envelope: WitnessEnvelope,
        timestamp_engine: SyntheticTimestampEngine,
        reference_time_utc: Optional[str] = None,
    ) -> PolicyEvaluationResult:
        """
        Evaluate complete attestation envelope against trust policy.
        Fails closed on any ambiguity or compromise horizon violation.
        """
        attestation = envelope.witness_attestation
        payload = envelope.attestation_payload
        key_fp = attestation.witness_pubkey_fingerprint

        # 1. Pinned Key Check
        if key_fp not in self._pinned_keys:
            return PolicyEvaluationResult(
                status=TrustStatus.REJECTED_UNPINNED_KEY,
                is_accepted=False,
                rejection_reason=f"Key fingerprint '{key_fp}' is not in pinned manifest.",
            )

        pub_key = self._pinned_keys[key_fp]

        # 2. Mathematical Signature Verification
        signed_bytes = payload.to_canonical_bytes()
        if not verify_detached(pub_key, signed_bytes, attestation.signature_b64):
            return PolicyEvaluationResult(
                status=TrustStatus.REJECTED_INVALID_SIGNATURE,
                is_accepted=False,
                rejection_reason="Ed25519 signature verification failed over canonical payload.",
            )

        # 3. Dual-Bound Attestation Digest Check
        calculated_digest = compute_attestation_digest(
            payload.merkle_root,
            key_fp,
            attestation.signature_b64,
        )
        if attestation.attestation_digest != calculated_digest:
            return PolicyEvaluationResult(
                status=TrustStatus.REJECTED_UNBOUND_SIGNATURE,
                is_accepted=False,
                rejection_reason=(
                    f"Attestation digest mismatch. Expected {calculated_digest}, "
                    f"got {attestation.attestation_digest}."
                ),
            )

        # 4. Timestamp Verification
        token_dict = attestation.timestamp_token
        if not token_dict:
            return PolicyEvaluationResult(
                status=TrustStatus.INDETERMINATE_TRUST,
                is_accepted=False,
                rejection_reason="No timestamp token attached. Undated signatures cannot prove historical existence.",
            )

        token_valid, token_err = timestamp_engine.verify_timestamp_token(
            token_dict,
            expected_attestation_digest=calculated_digest,
            reference_time_utc=reference_time_utc,
        )
        if not token_valid:
            return PolicyEvaluationResult(
                status=TrustStatus.REJECTED_UNTRUSTED_TSA,
                is_accepted=False,
                rejection_reason=f"Timestamp token verification failed: {token_err}",
            )

        # 5. Revocation & Compromise Horizon Evaluation
        t_tsa_str = token_dict.get("timestamp_utc", "")
        try:
            t_tsa = datetime.fromisoformat(t_tsa_str.replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            return PolicyEvaluationResult(
                status=TrustStatus.INDETERMINATE_TRUST,
                is_accepted=False,
                rejection_reason=f"Unparseable timestamp date '{t_tsa_str}': {e}",
            )

        if key_fp in self._revocation_records:
            rev_rec = self._revocation_records[key_fp]
            t_rev = datetime.fromisoformat(rev_rec.revoked_at_utc.replace("Z", "+00:00"))

            # Check administrative revocation date
            if t_tsa >= t_rev:
                return PolicyEvaluationResult(
                    status=TrustStatus.REJECTED_REVOKED_KEY,
                    is_accepted=False,
                    rejection_reason=(
                        f"Attestation timestamp ({t_tsa_str}) is at or after key revocation date "
                        f"({rev_rec.revoked_at_utc})."
                    ),
                    details={"revocation_record": rev_rec.to_dict()},
                )

            # Check earliest compromise horizon
            if rev_rec.earliest_compromise_utc == "UNKNOWN":
                return PolicyEvaluationResult(
                    status=TrustStatus.INDETERMINATE_TRUST,
                    is_accepted=False,
                    rejection_reason=(
                        "Key was revoked due to breach, but earliest compromise date is UNKNOWN. "
                        "Cannot prove signature predated compromise."
                    ),
                    details={"revocation_record": rev_rec.to_dict()},
                )
            elif rev_rec.earliest_compromise_utc:
                t_comp = datetime.fromisoformat(rev_rec.earliest_compromise_utc.replace("Z", "+00:00"))
                if t_tsa >= t_comp:
                    return PolicyEvaluationResult(
                        status=TrustStatus.REJECTED_SUSPECTED_COMPROMISE,
                        is_accepted=False,
                        rejection_reason=(
                            f"Attestation timestamp ({t_tsa_str}) is at or after earliest suspected "
                            f"compromise date ({rev_rec.earliest_compromise_utc}). Signature cannot be trusted."
                        ),
                        details={"revocation_record": rev_rec.to_dict()},
                    )

        # 6. All Checks Succeeded
        return PolicyEvaluationResult(
            status=TrustStatus.VALID_HISTORICAL_RECORD,
            is_accepted=True,
            rejection_reason=None,
            details={
                "witness_id": attestation.witness_id,
                "key_fingerprint": key_fp,
                "merkle_root": payload.merkle_root,
                "timestamp_utc": t_tsa_str,
            },
        )
