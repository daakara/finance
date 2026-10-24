"""
HVA-01 Key Rollover Attestation & Historical Evidence Quarantine Logic.
Implements Gate V3-01 and Gate V3-02 Three-State Historical Validity Framework
(VALID / INVALID / INDETERMINATE) per CTRL-4C2-14.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..crypto.canonical import canonicalize_json


class RolloverValidationError(Exception):
    """Raised when key rollover attestation verification fails."""
    pass


class EvidenceAdmissibilityState:
    VALID = "VALID"
    INVALID = "INVALID"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class RolloverVerificationResult:
    """Outcome of validating a key rollover attestation document."""
    is_valid: bool
    attestation_id: str
    sequence_number: int
    revoked_key_id: str
    activated_key_id: str
    compromise_suspected_timestamp_utc: Optional[str]
    effective_timestamp_utc: str
    errors: Optional[List[str]] = None


class KeyRolloverManager:
    """
    Manages generation, verification, and quarantine evaluation for
    KEY_ROLLOVER_ATTESTATION.json documents under dual-control authorization.
    """

    SCHEMA_VERSION = "1.0.0"

    @classmethod
    def create_rollover_attestation(
        cls,
        attestation_id: str,
        sequence_number: int,
        revoked_key_id: str,
        activated_key_id: str,
        effective_timestamp_utc: str,
        reason: str = "DEVICE_FAILURE",
        compromise_suspected_timestamp_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct raw dictionary for a rollover attestation document."""
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "attestation_id": attestation_id,
            "sequence_number": sequence_number,
            "revoked_key_id": revoked_key_id,
            "activated_key_id": activated_key_id,
            "reason": reason,
            "compromise_suspected_timestamp_utc": compromise_suspected_timestamp_utc,
            "effective_timestamp_utc": effective_timestamp_utc,
            "dual_control_signatures": [],
        }

    @classmethod
    def compute_canonical_attestation_hash(cls, attestation: Dict[str, Any]) -> Tuple[bytes, str]:
        """Compute RFC 8785 canonical hash of the attestation body (excluding signatures)."""
        body_to_sign = {
            "schema_version": attestation.get("schema_version", cls.SCHEMA_VERSION),
            "attestation_id": attestation.get("attestation_id", ""),
            "sequence_number": attestation.get("sequence_number", 1),
            "revoked_key_id": attestation.get("revoked_key_id", ""),
            "activated_key_id": attestation.get("activated_key_id", ""),
            "reason": attestation.get("reason", ""),
            "compromise_suspected_timestamp_utc": attestation.get("compromise_suspected_timestamp_utc"),
            "effective_timestamp_utc": attestation.get("effective_timestamp_utc", ""),
        }

        canonical_bytes = canonicalize_json(body_to_sign)
        digest = hashlib.sha256(canonical_bytes).digest()
        return canonical_bytes, digest.hex()

    @classmethod
    def add_signature(
        cls,
        attestation: Dict[str, Any],
        role: str,
        signer_public_key_bytes: bytes,
        signature_bytes: bytes,
        signed_timestamp_utc: str = "2026-10-24T12:00:00Z",
    ) -> None:
        """Add a dual-control signature to the rollover document."""
        fingerprint = hashlib.sha256(signer_public_key_bytes).hexdigest()
        sig_entry = {
            "role": role,
            "key_fingerprint_sha256": fingerprint,
            "signature_hex": signature_bytes.hex(),
            "signed_timestamp_utc": signed_timestamp_utc,
        }
        attestation["dual_control_signatures"].append(sig_entry)

    @classmethod
    def verify_rollover_attestation(
        cls,
        attestation: Dict[str, Any],
        trusted_owner_pubkey: bytes,
        trusted_custodian_pubkey: bytes,
        previous_sequence_number: int = 0,
    ) -> RolloverVerificationResult:
        """
        Verify rollover attestation integrity, monotonic sequence, and dual signatures.
        Requires both RESEARCH_OWNER and INSTITUTIONAL_CUSTODIAN co-signatures.
        """
        errors = []

        if attestation.get("schema_version") != cls.SCHEMA_VERSION:
            errors.append(f"Invalid schema_version: {attestation.get('schema_version')}")

        seq = attestation.get("sequence_number", 0)
        if seq <= previous_sequence_number:
            errors.append(
                f"Replay attack detected: sequence_number {seq} is not strictly "
                f"greater than previous sequence {previous_sequence_number}"
            )

        revoked_id = attestation.get("revoked_key_id", "")
        activated_id = attestation.get("activated_key_id", "")
        if not revoked_id or not activated_id:
            errors.append("Rollover attestation must specify revoked_key_id and activated_key_id")
        if revoked_id == activated_id:
            errors.append("revoked_key_id cannot equal activated_key_id")

        try:
            _, att_hash = cls.compute_canonical_attestation_hash(attestation)
            att_digest_bytes = bytes.fromhex(att_hash)
        except Exception as e:
            errors.append(f"Canonicalization failed: {e}")
            return RolloverVerificationResult(
                is_valid=False,
                attestation_id=attestation.get("attestation_id", ""),
                sequence_number=seq,
                revoked_key_id=revoked_id,
                activated_key_id=activated_id,
                compromise_suspected_timestamp_utc=attestation.get("compromise_suspected_timestamp_utc"),
                effective_timestamp_utc=attestation.get("effective_timestamp_utc", ""),
                errors=errors,
            )

        signatures = attestation.get("dual_control_signatures", [])
        owner_sig_found = False
        custodian_sig_found = False

        owner_fp = hashlib.sha256(trusted_owner_pubkey).hexdigest()
        custodian_fp = hashlib.sha256(trusted_custodian_pubkey).hexdigest()

        owner_verifier = ed25519.Ed25519PublicKey.from_public_bytes(trusted_owner_pubkey)
        custodian_verifier = ed25519.Ed25519PublicKey.from_public_bytes(trusted_custodian_pubkey)

        for sig in signatures:
            role = sig.get("role")
            sig_hex = sig.get("signature_hex", "")
            try:
                sig_bytes = bytes.fromhex(sig_hex)
            except ValueError:
                errors.append(f"Invalid signature hex for role {role}")
                continue

            if role == "RESEARCH_OWNER":
                if sig.get("key_fingerprint_sha256") != owner_fp:
                    errors.append("Owner signature fingerprint does not match trusted key")
                    continue
                try:
                    owner_verifier.verify(sig_bytes, att_digest_bytes)
                    owner_sig_found = True
                except InvalidSignature:
                    errors.append("Owner signature verification failed on rollover attestation")

            elif role == "INSTITUTIONAL_CUSTODIAN":
                if sig.get("key_fingerprint_sha256") != custodian_fp:
                    errors.append("Custodian signature fingerprint does not match trusted key")
                    continue
                try:
                    custodian_verifier.verify(sig_bytes, att_digest_bytes)
                    custodian_sig_found = True
                except InvalidSignature:
                    errors.append("Institutional Custodian signature verification failed on rollover attestation")

        if not owner_sig_found:
            errors.append("Missing valid RESEARCH_OWNER signature on rollover attestation")
        if not custodian_sig_found:
            errors.append("Missing valid INSTITUTIONAL_CUSTODIAN signature on rollover attestation")

        is_valid = len(errors) == 0 and owner_sig_found and custodian_sig_found

        return RolloverVerificationResult(
            is_valid=is_valid,
            attestation_id=attestation.get("attestation_id", ""),
            sequence_number=seq,
            revoked_key_id=revoked_id,
            activated_key_id=activated_id,
            compromise_suspected_timestamp_utc=attestation.get("compromise_suspected_timestamp_utc"),
            effective_timestamp_utc=attestation.get("effective_timestamp_utc", ""),
            errors=errors if errors else None,
        )

    @classmethod
    def evaluate_three_state_admissibility(
        cls,
        token_timestamp_utc: str,
        effective_revocation_utc: str,
        compromise_suspected_utc: Optional[str] = None,
    ) -> str:
        """
        Evaluate Gate V3-02 Three-State Historical Validity:
        - INVALID if token_timestamp >= effective_revocation
        - INDETERMINATE (Quarantined) if compromise_suspected <= token_timestamp < effective_revocation
        - VALID if token_timestamp < compromise_suspected (or effective_revocation if no compromise suspected)
        """
        if token_timestamp_utc >= effective_revocation_utc:
            return EvidenceAdmissibilityState.INVALID

        if compromise_suspected_utc and token_timestamp_utc >= compromise_suspected_utc:
            # Inside the evidence-derived Compromise Uncertainty Window:
            # Signature may be mathematically correct, but is quarantined as INDETERMINATE
            return EvidenceAdmissibilityState.INDETERMINATE

        return EvidenceAdmissibilityState.VALID
