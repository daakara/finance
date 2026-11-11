"""
HVA-01 Synthetic Witness Asymmetric Signatures.
Implements detached Ed25519 (RFC 8032) signing and verification with pinned key fingerprints.
All keys generated in this module are explicitly marked as NONPRODUCTION SYNTHETIC FIXTURES.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def get_key_fingerprint(public_key: ed25519.Ed25519PublicKey) -> str:
    """
    Compute deterministic SHA-256 fingerprint of the 32-byte raw Ed25519 public key:
    fingerprint = "sha256:" || hex(SHA-256(raw_pubkey_bytes))
    """
    raw_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"


def generate_ed25519_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate a fresh Ed25519 keypair for local synthetic testing."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return priv, pub


def sign_detached(private_key: ed25519.Ed25519PrivateKey, data: bytes) -> str:
    """
    Compute a detached Ed25519 signature over raw data bytes.
    Returns standard Base64-encoded 64-byte signature string.
    """
    raw_sig = private_key.sign(data)
    return base64.b64encode(raw_sig).decode("ascii")


def verify_detached(
    public_key: ed25519.Ed25519PublicKey,
    data: bytes,
    signature_b64: str,
) -> bool:
    """
    Verify a detached Ed25519 Base64 signature over raw data bytes.
    Fails closed and returns False if the signature is invalid or malformed.
    """
    try:
        raw_sig = base64.b64decode(signature_b64.encode("ascii"))
        if len(raw_sig) != 64:
            return False
        public_key.verify(raw_sig, data)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class SyntheticWitnessKey:
    """
    Container representing a synthetic independent witness identity.
    NONPRODUCTION TEST ONLY. Zero operational signing authority.
    """
    witness_id: str
    public_key_fingerprint: str
    public_key_hex: str
    public_key_b64: str
    is_synthetic: bool = True
    environment: str = "SYNTHETIC_TEST_FIXTURE"

    @classmethod
    def from_public_key(cls, witness_id: str, pub: ed25519.Ed25519PublicKey) -> SyntheticWitnessKey:
        raw_bytes = pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        fingerprint = get_key_fingerprint(pub)
        return cls(
            witness_id=witness_id,
            public_key_fingerprint=fingerprint,
            public_key_hex=raw_bytes.hex(),
            public_key_b64=base64.b64encode(raw_bytes).decode("ascii"),
        )
