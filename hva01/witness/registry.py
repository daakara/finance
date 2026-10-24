"""
HVA-01 Canonical RFC 8785 Witness Key Registry Management.
Implements the canonical JSON Schema, deterministic hashing, and dual-control
signature verification for WITNESS_KEY_REGISTRY.json per CTRL-4C2-10 and Gate V3-01.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..crypto.canonical import CanonicalJCS, canonicalize_json


class RegistryValidationError(Exception):
    """Raised when registry schema, canonicalization, or signature verification fails."""
    pass


@dataclass(frozen=True)
class RegistryVerificationResult:
    """Detailed verification outcome for a canonical witness key registry."""
    is_valid: bool
    registry_hash: str
    keys_count: int
    signatures_verified: int
    primary_key_id: Optional[str] = None
    backup_key_id: Optional[str] = None
    errors: Optional[List[str]] = None


class WitnessKeyRegistryManager:
    """
    Constructs, canonicalizes, dual-signs, and validates WITNESS_KEY_REGISTRY.json
    using RFC 8785 (JSON Canonicalization Scheme).
    """

    SCHEMA_VERSION = "1.0.0"

    @classmethod
    def create_empty_registry(
        cls,
        ceremony_id: str,
        epoch_utc: str = "2026-10-24T12:00:00Z",
    ) -> Dict[str, Any]:
        """Create initial registry dictionary structure."""
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "registry_epoch_utc": epoch_utc,
            "ceremony_identifier": ceremony_id,
            "keys": [],
            "dual_control_signatures": [],
        }

    @classmethod
    def add_key_entry(
        cls,
        registry: Dict[str, Any],
        key_id: str,
        public_key_bytes: bytes,
        token_serial: str,
        status: str = "ACTIVE_PRIMARY",
        effective_date_utc: str = "2026-10-24T12:00:00Z",
        attestation_status: str = "ATTESTATION_VERIFIED",
    ) -> None:
        """Add a validated hardware public key entry to the registry."""
        pub_hex = public_key_bytes.hex()
        fingerprint = hashlib.sha256(public_key_bytes).hexdigest()

        entry = {
            "key_id": key_id,
            "algorithm": "Ed25519",
            "public_key_hex": pub_hex,
            "fingerprint_sha256": fingerprint,
            "token_serial": token_serial,
            "attestation_status": attestation_status,
            "status": status,
            "effective_timestamp_utc": effective_date_utc,
        }
        registry["keys"].append(entry)

    @classmethod
    def compute_canonical_registry_hash(cls, registry: Dict[str, Any]) -> Tuple[bytes, str]:
        """
        Compute deterministic SHA-256 digest of the canonicalized registry content.
        Excludes the signatures block to prevent circular dependency.
        """
        content_to_sign = {
            "schema_version": registry.get("schema_version", cls.SCHEMA_VERSION),
            "registry_epoch_utc": registry.get("registry_epoch_utc", ""),
            "ceremony_identifier": registry.get("ceremony_identifier", ""),
            "keys": registry.get("keys", []),
        }

        # Canonicalize per RFC 8785
        canonical_bytes = canonicalize_json(content_to_sign)
        digest = hashlib.sha256(canonical_bytes).digest()
        return canonical_bytes, digest.hex()

    @classmethod
    def add_signature(
        cls,
        registry: Dict[str, Any],
        role: str,
        signer_public_key_bytes: bytes,
        signature_bytes: bytes,
        signed_timestamp_utc: str = "2026-10-24T12:00:00Z",
    ) -> None:
        """Add a detached signature to the registry's dual-control block."""
        fingerprint = hashlib.sha256(signer_public_key_bytes).hexdigest()
        sig_entry = {
            "role": role,
            "key_fingerprint_sha256": fingerprint,
            "signature_hex": signature_bytes.hex(),
            "signed_timestamp_utc": signed_timestamp_utc,
        }
        registry["dual_control_signatures"].append(sig_entry)

    @classmethod
    def verify_registry(
        cls,
        registry: Dict[str, Any],
        trusted_owner_pubkey: bytes,
        trusted_witness_pubkey: bytes,
    ) -> RegistryVerificationResult:
        """
        Verify the integrity, canonical hash, and dual signatures of the registry.
        Fail-closed: requires both valid RESEARCH_OWNER and INDEPENDENT_WITNESS signatures.
        """
        errors = []

        # 1. Structural schema assertions
        if registry.get("schema_version") != cls.SCHEMA_VERSION:
            errors.append(f"Invalid schema_version: {registry.get('schema_version')}")

        keys = registry.get("keys", [])
        if len(keys) < 2:
            errors.append(f"Registry requires at least 2 keys (Primary + Escrow), found {len(keys)}")

        primary_key_id = None
        backup_key_id = None
        for k in keys:
            if k.get("status") == "ACTIVE_PRIMARY":
                primary_key_id = k.get("key_id")
            elif k.get("status") == "COLD_BACKUP_ESCROW":
                backup_key_id = k.get("key_id")

        if not primary_key_id:
            errors.append("Registry missing ACTIVE_PRIMARY key entry")
        if not backup_key_id:
            errors.append("Registry missing COLD_BACKUP_ESCROW key entry")

        # 2. Canonical digest computation
        try:
            _, reg_hash = cls.compute_canonical_registry_hash(registry)
            reg_digest_bytes = bytes.fromhex(reg_hash)
        except Exception as e:
            errors.append(f"Canonicalization failed: {e}")
            return RegistryVerificationResult(
                is_valid=False,
                registry_hash="",
                keys_count=len(keys),
                signatures_verified=0,
                errors=errors,
            )

        # 3. Dual-control signature verification
        signatures = registry.get("dual_control_signatures", [])
        owner_sig_found = False
        witness_sig_found = False
        verified_count = 0

        owner_fingerprint = hashlib.sha256(trusted_owner_pubkey).hexdigest()
        witness_fingerprint = hashlib.sha256(trusted_witness_pubkey).hexdigest()

        owner_verifier = ed25519.Ed25519PublicKey.from_public_bytes(trusted_owner_pubkey)
        witness_verifier = ed25519.Ed25519PublicKey.from_public_bytes(trusted_witness_pubkey)

        for sig in signatures:
            role = sig.get("role")
            sig_hex = sig.get("signature_hex", "")
            try:
                sig_bytes = bytes.fromhex(sig_hex)
            except ValueError:
                errors.append(f"Invalid signature hex for role {role}")
                continue

            if role == "RESEARCH_OWNER":
                if sig.get("key_fingerprint_sha256") != owner_fingerprint:
                    errors.append("Owner signature fingerprint does not match trusted key")
                    continue
                try:
                    owner_verifier.verify(sig_bytes, reg_digest_bytes)
                    owner_sig_found = True
                    verified_count += 1
                except InvalidSignature:
                    errors.append("Owner signature verification failed")

            elif role == "INDEPENDENT_WITNESS":
                if sig.get("key_fingerprint_sha256") != witness_fingerprint:
                    errors.append("Witness signature fingerprint does not match trusted key")
                    continue
                try:
                    witness_verifier.verify(sig_bytes, reg_digest_bytes)
                    witness_sig_found = True
                    verified_count += 1
                except InvalidSignature:
                    errors.append("Independent Witness signature verification failed")

        if not owner_sig_found:
            errors.append("Missing valid RESEARCH_OWNER signature on registry")
        if not witness_sig_found:
            errors.append("Missing valid INDEPENDENT_WITNESS signature on registry")

        is_valid = len(errors) == 0 and owner_sig_found and witness_sig_found

        return RegistryVerificationResult(
            is_valid=is_valid,
            registry_hash=reg_hash,
            keys_count=len(keys),
            signatures_verified=verified_count,
            primary_key_id=primary_key_id,
            backup_key_id=backup_key_id,
            errors=errors if errors else None,
        )
