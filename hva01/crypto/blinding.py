"""
HVA-01 Blinding and Side-Channel Defenses.
Implements high-entropy salted nonces for low-entropy commitments and deterministic byte padding.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Optional, Tuple

from .canonical import canonicalize_json


def create_salted_commitment(
    value: Any,
    nonce: Optional[bytes] = None,
) -> Tuple[str, str]:
    """
    Create a confidentiality-preserving commitment for low-entropy values (e.g. classification 0 or 1).
    Uses a 256-bit (32-byte) cryptographically secure random nonce:
      Commitment = "sha256:" || SHA-256(RFC8785({"nonce": nonce_hex, "value": value}))
    
    Returns (commitment_digest, nonce_hex).
    """
    r_bytes = nonce if nonce is not None else secrets.token_bytes(32)
    if len(r_bytes) < 32:
        raise ValueError("Salted nonce must be at least 32 bytes (256 bits).")

    nonce_hex = r_bytes.hex()
    commitment_payload = {
        "nonce": nonce_hex,
        "value": value,
    }
    canonical_bytes = canonicalize_json(commitment_payload)
    commitment_digest = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
    return commitment_digest, nonce_hex


def verify_salted_commitment(
    value: Any,
    nonce_hex: str,
    expected_commitment: str,
) -> bool:
    """
    Verify an opened commitment against an expected commitment digest.
    Uses constant-time comparison to prevent timing side-channels.
    """
    try:
        r_bytes = bytes.fromhex(nonce_hex)
        if len(r_bytes) < 32:
            return False
        computed_commitment, _ = create_salted_commitment(value, nonce=r_bytes)
        return hmac.compare_digest(computed_commitment, expected_commitment)
    except (ValueError, TypeError):
        return False


def pad_payload(payload_bytes: bytes, target_size: int = 4096) -> bytes:
    """
    Pad payload to a uniform block size using 4-byte big-endian length prefix
    followed by payload and deterministic zero padding.
    Prevents response length from leaking classification or reasoning complexity.
    """
    payload_len = len(payload_bytes)
    if payload_len + 4 > target_size:
        raise ValueError(
            f"Payload length ({payload_len} bytes + 4 byte prefix) exceeds target size ({target_size} bytes)."
        )
    
    prefix = payload_len.to_bytes(4, byteorder="big")
    padding_needed = target_size - (4 + payload_len)
    return prefix + payload_bytes + (b"\x00" * padding_needed)


def unpad_payload(padded_bytes: bytes) -> bytes:
    """
    Extract the original payload from a padded block.
    Fails closed if the padded block is malformed or truncated.
    """
    if len(padded_bytes) < 4:
        raise ValueError("Padded buffer too short to contain length prefix.")
    
    payload_len = int.from_bytes(padded_bytes[:4], byteorder="big")
    if 4 + payload_len > len(padded_bytes):
        raise ValueError("Padded buffer corrupted: declared length exceeds buffer size.")
    
    return padded_bytes[4 : 4 + payload_len]
