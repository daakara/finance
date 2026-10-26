"""
ARX Terminal — Evidence Custody Cryptographic Implementation (Gate D)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Cryptographic Contract:
- Cipher: AES-256-GCM (NIST SP 800-38D authenticated encryption with associated data)
- Key size: 256 bits (32 bytes)
- Nonce: 96 bits (12 bytes) cryptographically random per message via os.urandom
- Authentication Tag: 128 bits (16 bytes) appended to ciphertext
- Pre-parse hashing: SHA-256 computed on raw wire bytes before JSON decoding
- Zero Key / Plaintext Logging
"""

import os
import hashlib
import binascii
import base64
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from analyst_dashboard.research.custody.exceptions import (
    CryptoTamperError,
    CustodyKeyError,
)

NONCE_LENGTH_BYTES = 12
TAG_LENGTH_BYTES = 16
KEY_LENGTH_BYTES = 32
MIN_ENCRYPTED_PAYLOAD_LENGTH = NONCE_LENGTH_BYTES + TAG_LENGTH_BYTES  # 28 bytes


def compute_sha256(data: bytes) -> str:
    """Compute deterministic SHA-256 hex digest of raw bytes."""
    if not isinstance(data, bytes):
        raise TypeError("compute_sha256 requires raw bytes input.")
    return hashlib.sha256(data).hexdigest()


def generate_key() -> bytes:
    """Generate a fresh cryptographically secure 256-bit key."""
    return AESGCM.generate_key(bit_length=256)


def generate_key_hex() -> str:
    """Generate a fresh 256-bit key as a 64-character hexadecimal string."""
    return binascii.hexlify(generate_key()).decode("ascii")


class CustodyCrypto:
    """
    Authenticated encryption module for evidence custody.
    Enforces AES-256-GCM, unique nonces, pre-parse digests, and tamper detection.
    """

    def __init__(self, key: Optional[bytes | str] = None):
        """
        Initialize crypto engine with 256-bit key.
        If key is None, attempts to load from environment variable ARX_CUSTODY_ENCRYPTION_KEY.
        Key can be provided as raw 32 bytes, 64-character hex string, or base64-encoded string.
        """
        raw_key = key or os.environ.get("ARX_CUSTODY_ENCRYPTION_KEY")
        if not raw_key:
            raise CustodyKeyError(
                "Custody encryption key is not provided and ARX_CUSTODY_ENCRYPTION_KEY is unset."
            )

        self._key = self._parse_key(raw_key)
        self._aesgcm = AESGCM(self._key)

    @staticmethod
    def _parse_key(key_input: bytes | str) -> bytes:
        """Parse and validate key into exact 32 bytes."""
        if isinstance(key_input, bytes):
            if len(key_input) == KEY_LENGTH_BYTES:
                return key_input
            raise CustodyKeyError(
                f"Invalid raw key length: expected {KEY_LENGTH_BYTES} bytes, got {len(key_input)}."
            )

        if isinstance(key_input, str):
            clean_str = key_input.strip()
            # Try hex decoding
            if len(clean_str) == 64:
                try:
                    k = binascii.unhexlify(clean_str)
                    if len(k) == KEY_LENGTH_BYTES:
                        return k
                except (binascii.Error, ValueError):
                    pass

            # Try base64 decoding
            try:
                k = base64.b64decode(clean_str)
                if len(k) == KEY_LENGTH_BYTES:
                    return k
            except Exception:
                pass

            # Try raw bytes if exactly 32 chars
            if len(clean_str.encode("utf-8")) == KEY_LENGTH_BYTES:
                return clean_str.encode("utf-8")

        raise CustodyKeyError(
            f"Encryption key must be a 32-byte binary or 64-character hex string (256-bit)."
        )

    def encrypt(
        self, plaintext_bytes: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Encrypt plaintext using AES-256-GCM.
        Returns: nonce (12 bytes) + ciphertext + authentication tag (16 bytes).
        """
        if not isinstance(plaintext_bytes, bytes):
            raise TypeError("Plaintext must be raw bytes.")

        nonce = os.urandom(NONCE_LENGTH_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext_bytes, associated_data)
        return nonce + ciphertext

    def decrypt(
        self, encrypted_payload: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt and authenticate AES-256-GCM payload.
        Expects: nonce (12 bytes) + ciphertext + authentication tag (16 bytes).
        Raises CryptoTamperError if tag authentication fails or data is corrupted.
        """
        if not isinstance(encrypted_payload, bytes):
            raise TypeError("Encrypted payload must be raw bytes.")

        if len(encrypted_payload) < MIN_ENCRYPTED_PAYLOAD_LENGTH:
            raise CryptoTamperError(
                f"Payload too short ({len(encrypted_payload)} bytes). Minimum is {MIN_ENCRYPTED_PAYLOAD_LENGTH} bytes."
            )

        nonce = encrypted_payload[:NONCE_LENGTH_BYTES]
        ciphertext_and_tag = encrypted_payload[NONCE_LENGTH_BYTES:]

        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext_and_tag, associated_data)
            return plaintext
        except InvalidTag as e:
            raise CryptoTamperError(
                "Ciphertext authentication failed: payload has been tampered with or key is incorrect."
            ) from e

    def __repr__(self) -> str:
        # Never leak key in string representations
        return "<CustodyCrypto: AES-256-GCM [REDACTED_KEY]>"
