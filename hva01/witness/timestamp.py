"""
HVA-01 Synthetic RFC 3161 Timestamp Engine and Attestation Binding.
Implements offline synthetic timestamping over dual-bound attestation digests.
NONPRODUCTION TEST CONTRACT. No external network requests are made.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

from ..crypto.canonical import CanonicalJCS, canonicalize_json
from ..crypto.signatures import (
    generate_ed25519_keypair,
    get_key_fingerprint,
    sign_detached,
    verify_detached,
)


def compute_attestation_digest(
    merkle_root: str,
    key_fingerprint: str,
    signature_b64: str,
) -> str:
    """
    Compute dual-bound attestation digest D_attest:
    D_attest = "sha256:" || SHA-256(RFC8785({
        "merkle_root": merkle_root,
        "key_fingerprint": key_fingerprint,
        "signature_b64": signature_b64,
    }))
    
    This digest binds the data Merkle root, the witness identity, AND the witness signature,
    eliminating signature substitution and timestamp decoupling attacks.
    """
    payload = {
        "merkle_root": merkle_root,
        "key_fingerprint": key_fingerprint,
        "signature_b64": signature_b64,
    }
    canonical_bytes = canonicalize_json(payload)
    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"


@dataclass(frozen=True)
class SyntheticTimestampToken:
    """
    Structured representation of an RFC 3161 TimeStampToken.
    """
    tsa_identity: str
    tsa_policy_oid: str
    message_imprint_algorithm: str
    message_imprint_digest: str
    timestamp_utc: str
    serial_number: int
    nonce: int
    tsa_certificate_fingerprint: str
    tsa_signature_b64: str
    is_synthetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_canonical_bytes(self) -> bytes:
        data = {
            "tsa_identity": self.tsa_identity,
            "tsa_policy_oid": self.tsa_policy_oid,
            "message_imprint_algorithm": self.message_imprint_algorithm,
            "message_imprint_digest": self.message_imprint_digest,
            "timestamp_utc": self.timestamp_utc,
            "serial_number": self.serial_number,
            "nonce": self.nonce,
            "tsa_certificate_fingerprint": self.tsa_certificate_fingerprint,
        }
        return canonicalize_json(data)


class SyntheticTimestampEngine:
    """
    Local synthetic timestamp engine simulating an RFC 3161 Timestamp Authority.
    Operates 100% offline using an isolated synthetic Ed25519 signing keypair.
    """

    DEFAULT_POLICY_OID = "2.16.840.1.114412.7.1"

    def __init__(
        self,
        tsa_identity: str = "Synthetic-TSA-Desk-01",
        policy_oid: str = DEFAULT_POLICY_OID,
    ) -> None:
        self.tsa_identity = tsa_identity
        self.policy_oid = policy_oid
        self._priv, self._pub = generate_ed25519_keypair()
        self.tsa_cert_fingerprint = get_key_fingerprint(self._pub)
        self._serial_counter = 1000

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._pub

    def issue_timestamp(
        self,
        attestation_digest: str,
        policy_oid: Optional[str] = None,
        nonce: Optional[int] = None,
        timestamp_utc: Optional[str] = None,
    ) -> SyntheticTimestampToken:
        """
        Issue a signed synthetic timestamp token over the provided message imprint.
        """
        self._serial_counter += 1
        now_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()
        token_policy = policy_oid or self.policy_oid
        token_nonce = nonce if nonce is not None else self._serial_counter * 31337

        temp_token = SyntheticTimestampToken(
            tsa_identity=self.tsa_identity,
            tsa_policy_oid=token_policy,
            message_imprint_algorithm="SHA-256",
            message_imprint_digest=attestation_digest,
            timestamp_utc=now_utc,
            serial_number=self._serial_counter,
            nonce=token_nonce,
            tsa_certificate_fingerprint=self.tsa_cert_fingerprint,
            tsa_signature_b64="",
            is_synthetic=True,
        )

        signed_bytes = temp_token.to_canonical_bytes()
        sig_b64 = sign_detached(self._priv, signed_bytes)

        return SyntheticTimestampToken(
            tsa_identity=self.tsa_identity,
            tsa_policy_oid=token_policy,
            message_imprint_algorithm="SHA-256",
            message_imprint_digest=attestation_digest,
            timestamp_utc=now_utc,
            serial_number=self._serial_counter,
            nonce=token_nonce,
            tsa_certificate_fingerprint=self.tsa_cert_fingerprint,
            tsa_signature_b64=sig_b64,
            is_synthetic=True,
        )

    def verify_timestamp_token(
        self,
        token_dict: Dict[str, Any],
        expected_attestation_digest: str,
        pinned_policy_oid: Optional[str] = None,
        reference_time_utc: Optional[str] = None,
        max_drift_seconds: float = 5.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a synthetic RFC 3161 TimeStampToken against expected parameters.
        Returns (True, None) on success or (False, error_reason) on failure.
        """
        try:
            token = SyntheticTimestampToken(
                tsa_identity=token_dict["tsa_identity"],
                tsa_policy_oid=token_dict["tsa_policy_oid"],
                message_imprint_algorithm=token_dict["message_imprint_algorithm"],
                message_imprint_digest=token_dict["message_imprint_digest"],
                timestamp_utc=token_dict["timestamp_utc"],
                serial_number=token_dict["serial_number"],
                nonce=token_dict["nonce"],
                tsa_certificate_fingerprint=token_dict["tsa_certificate_fingerprint"],
                tsa_signature_b64=token_dict["tsa_signature_b64"],
                is_synthetic=token_dict.get("is_synthetic", True),
            )
        except (KeyError, TypeError) as e:
            return False, f"MALFORMED_TOKEN_STRUCTURE: {e}"

        # 1. Message Imprint Match
        if token.message_imprint_digest != expected_attestation_digest:
            return False, (
                f"IMPRINT_MISMATCH: Expected {expected_attestation_digest}, "
                f"got {token.message_imprint_digest}"
            )

        # 2. Policy OID Match
        expected_policy = pinned_policy_oid or self.DEFAULT_POLICY_OID
        if token.tsa_policy_oid != expected_policy:
            return False, (
                f"UNAUTHORIZED_POLICY_OID: Expected {expected_policy}, "
                f"got {token.tsa_policy_oid}"
            )

        # 3. Signature Verification
        signed_bytes = token.to_canonical_bytes()
        if not verify_detached(self._pub, signed_bytes, token.tsa_signature_b64):
            return False, "INVALID_TSA_SIGNATURE: Signature verification failed against TSA public key."

        # 4. Clock Drift Check (if reference time supplied)
        if reference_time_utc:
            try:
                t_token = datetime.fromisoformat(token.timestamp_utc.replace("Z", "+00:00"))
                t_ref = datetime.fromisoformat(reference_time_utc.replace("Z", "+00:00"))
                diff_seconds = abs((t_token - t_ref).total_seconds())
                if diff_seconds > max_drift_seconds:
                    return False, f"CLOCK_DRIFT_EXCEEDED: Drift of {diff_seconds:.2f}s exceeds {max_drift_seconds}s limit."
            except ValueError as e:
                return False, f"INVALID_TIMESTAMP_FORMAT: {e}"

        return True, None
