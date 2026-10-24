"""
HVA-01 Hardware Token Interface & Synthetic Mock Isolation Subsystem.
Implements the hardware-native multi-key abstraction boundary defined in
HVA01_WAVE4C2_SCOPE_LOCK_AND_KEY_CEREMONY_SPECIFICATION_V3 (Gate V3-01 / CTRL-4C2-08).

Private keys are strictly born inside silicon (or mock memory isolation)
and are physically incapable of export across the abstraction boundary.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


class SecurityBoundaryViolation(Exception):
    """Raised when an operation attempts to breach cryptographic or hardware boundaries."""
    pass


class TokenLockedError(SecurityBoundaryViolation):
    """Raised when a hardware token is locked due to excessive PIN failures."""
    pass


class InvalidPinError(SecurityBoundaryViolation):
    """Raised when an incorrect PIN is supplied to the token interface."""
    pass


@dataclass(frozen=True)
class HardwareKeyMetadata:
    """Metadata describing an onboard-generated keypair."""
    key_id: str
    slot_id: int
    algorithm: str
    public_key_bytes: bytes
    public_key_hex: str
    fingerprint_sha256: str
    is_extractable: bool = False
    is_never_extractable: bool = True
    token_serial: str = "SYNTHETIC_TOKEN"


@dataclass(frozen=True)
class TokenDeviceInfo:
    """Device-level hardware certification information."""
    manufacturer: str
    model: str
    serial_number: str
    firmware_version: str
    is_mock: bool
    certification_level: str  # e.g., "CC EAL5+ / FIPS 140-3 Level 3"


class HardwareTokenInterface(ABC):
    """Abstract interface defining required hardware token operations."""

    @abstractmethod
    def initialize_token(self, so_pin: str, user_pin: str, label: str) -> None:
        """Initialize token slot, SO-PIN, User PIN, and security flags."""
        ...

    @abstractmethod
    def generate_keypair(self, slot_id: int, key_id: str, user_pin: str) -> HardwareKeyMetadata:
        """Generate Ed25519 keypair onboard silicon. Private key remains non-exportable."""
        ...

    @abstractmethod
    def get_public_key(self, slot_id: int, key_id: str) -> bytes:
        """Export 32-byte Ed25519 public key."""
        ...

    @abstractmethod
    def sign(self, slot_id: int, key_id: str, data: bytes, user_pin: str) -> bytes:
        """Execute detached Ed25519 signing inside token using onboard private key."""
        ...

    @abstractmethod
    def get_device_info(self) -> TokenDeviceInfo:
        """Return device hardware certification and serial metadata."""
        ...

    @abstractmethod
    def get_device_attestation(self, slot_id: int, key_id: str) -> Optional[bytes]:
        """Extract vendor-signed hardware attestation certificate (if supported)."""
        ...


class MockHardwareToken(HardwareTokenInterface):
    """
    Synthetic mock token simulating CC EAL5+ / FIPS 140-3 Level 3 hardware token.
    Enforces private key non-exportability and PIN retry lockout in Python memory.
    """

    def __init__(self, serial_number: str = "MOCK-YK5-001") -> None:
        self._serial_number = serial_number
        self._is_initialized = False
        self._so_pin = "12345678"
        self._user_pin = "123456"
        self._user_pin_retries_remaining = 3
        self._is_locked = False
        self._token_label = ""
        # Silicon memory simulation: private keys are stored in an isolated dictionary
        # and are never returned or exported.
        self._isolated_silicon_slots: Dict[int, Tuple[str, ed25519.Ed25519PrivateKey, bytes]] = {}

    def initialize_token(self, so_pin: str, user_pin: str, label: str) -> None:
        """Initialize or reset the mock token."""
        self._so_pin = so_pin
        self._user_pin = user_pin
        self._token_label = label
        self._user_pin_retries_remaining = 3
        self._is_locked = False
        self._isolated_silicon_slots.clear()
        self._is_initialized = True

    def _verify_user_pin(self, user_pin: str) -> None:
        """Verify user PIN enforcing 3-attempt lockout policy."""
        if self._is_locked:
            raise TokenLockedError(f"Token {self._serial_number} is LOCKED. SO-PIN reset required.")
        if user_pin != self._user_pin:
            self._user_pin_retries_remaining -= 1
            if self._user_pin_retries_remaining <= 0:
                self._is_locked = True
                raise TokenLockedError(f"Token {self._serial_number} LOCKED due to 3 consecutive invalid PIN attempts.")
            raise InvalidPinError(
                f"Invalid User PIN. Retries remaining: {self._user_pin_retries_remaining}"
            )
        # Successful PIN verification resets retry counter
        self._user_pin_retries_remaining = 3

    def unblock_with_so_pin(self, so_pin: str, new_user_pin: str) -> None:
        """Unblock a locked token slot using the Security Officer PIN."""
        if so_pin != self._so_pin:
            raise InvalidPinError("Invalid SO-PIN. Token remains locked.")
        self._user_pin = new_user_pin
        self._is_locked = False
        self._user_pin_retries_remaining = 3

    def generate_keypair(self, slot_id: int, key_id: str, user_pin: str) -> HardwareKeyMetadata:
        """
        Generate an Ed25519 keypair inside the simulated silicon slot.
        The private key is physically non-exportable.
        """
        self._verify_user_pin(user_pin)
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        import hashlib
        fingerprint = hashlib.sha256(pub_bytes).hexdigest()

        # Isolate private key in internal silicon storage
        self._isolated_silicon_slots[slot_id] = (key_id, private_key, pub_bytes)

        return HardwareKeyMetadata(
            key_id=key_id,
            slot_id=slot_id,
            algorithm="Ed25519",
            public_key_bytes=pub_bytes,
            public_key_hex=pub_bytes.hex(),
            fingerprint_sha256=fingerprint,
            is_extractable=False,
            is_never_extractable=True,
            token_serial=self._serial_number
        )

    def get_public_key(self, slot_id: int, key_id: str) -> bytes:
        """Export the public key for a slot."""
        if slot_id not in self._isolated_silicon_slots:
            raise KeyError(f"No keypair in slot {slot_id}")
        stored_id, _, pub_bytes = self._isolated_silicon_slots[slot_id]
        if stored_id != key_id:
            raise KeyError(f"Slot {slot_id} contains key '{stored_id}', not '{key_id}'")
        return pub_bytes

    def sign(self, slot_id: int, key_id: str, data: bytes, user_pin: str) -> bytes:
        """
        Execute detached Ed25519 signing inside the token.
        Private key never leaves the internal dictionary.
        """
        self._verify_user_pin(user_pin)
        if slot_id not in self._isolated_silicon_slots:
            raise KeyError(f"No keypair in slot {slot_id}")
        stored_id, private_key, _ = self._isolated_silicon_slots[slot_id]
        if stored_id != key_id:
            raise KeyError(f"Slot {slot_id} contains key '{stored_id}', not '{key_id}'")

        return private_key.sign(data)

    def export_private_key_disallowed(self, slot_id: int) -> None:
        """Proof of non-exportability: raises SecurityBoundaryViolation."""
        raise SecurityBoundaryViolation(
            f"CKA_EXTRACTABLE=FALSE / CKA_NEVER_EXTRACTABLE=TRUE: "
            f"Private keys in token {self._serial_number} cannot be exported."
        )

    def get_device_info(self) -> TokenDeviceInfo:
        return TokenDeviceInfo(
            manufacturer="Synthetic Security Labs",
            model="ARX-Simulated-HSM-Token",
            serial_number=self._serial_number,
            firmware_version="5.7.1-mock",
            is_mock=True,
            certification_level="CC EAL5+ / FIPS 140-3 Level 3 (Synthetic Simulation)"
        )

    def get_device_attestation(self, slot_id: int, key_id: str) -> Optional[bytes]:
        """Generate mock vendor attestation certificate."""
        if slot_id not in self._isolated_silicon_slots:
            return None
        pub_bytes = self.get_public_key(slot_id, key_id)
        # Synthetic attestation payload
        payload = (
            f"MOCK_DEVICE_ATTESTATION|SERIAL:{self._serial_number}|"
            f"SLOT:{slot_id}|KEY_ID:{key_id}|PUBKEY:{pub_bytes.hex()}|"
            f"EXTRACTABLE:FALSE|ONBOARD_TRNG:PTG.2|CERT:EAL5+"
        ).encode("utf-8")
        return payload
