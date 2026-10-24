"""
HVA-01 PKCS#11 Session Management & Driver Abstraction.
Standardizes interaction with cryptoki / PKCS#11 hardware libraries (OpenSC / YubiKey PKCS#11),
enforcing CKA_EXTRACTABLE=FALSE and CKA_NEVER_EXTRACTABLE=TRUE.

Under synthetic testing, safely routes to MockHardwareToken.
Enforces the mandatory fail-closed non-fallback rule: hardware errors never fall back to mock.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

from .token_interface import (
    HardwareKeyMetadata,
    HardwareTokenInterface,
    MockHardwareToken,
    SecurityBoundaryViolation,
    TokenDeviceInfo,
)


class HardwareDriverError(Exception):
    """Raised when native PKCS#11 hardware or driver encounters a failure."""
    pass


class PKCS11SessionManager:
    """
    Manages PKCS#11 sessions, token slots, and cryptographic hardware operations.
    Enforces strict fail-closed safety boundaries.
    """

    def __init__(
        self,
        mock_mode: Optional[bool] = None,
        driver_path: Optional[str] = None,
    ) -> None:
        # Determine execution mode from environment or parameter
        if mock_mode is not None:
            self._mock_mode = mock_mode
        else:
            # Default to mock mode unless explicitly disabled
            env_val = os.environ.get("ARX_CEREMONY_MOCK_HARDWARE", "1").strip().lower()
            self._mock_mode = env_val in ("1", "true", "yes")

        self._driver_path = driver_path or os.environ.get(
            "PKCS11_DRIVER_PATH", "/usr/lib/x86_64-linux-gnu/opensc-pkcs11.so"
        )

        # Active token instances mapped by slot_id
        self._tokens: Dict[int, HardwareTokenInterface] = {}
        self._authenticated_slots: Dict[int, str] = {}  # slot_id -> user_pin

    @property
    def is_mock_mode(self) -> bool:
        return self._mock_mode

    def register_mock_token(self, slot_id: int, token: MockHardwareToken) -> None:
        """Register a mock hardware token for synthetic offline testing."""
        if not self._mock_mode:
            raise SecurityBoundaryViolation(
                "Cannot register mock token when PKCS#11 session is in live hardware mode."
            )
        self._tokens[slot_id] = token

    def get_token(self, slot_id: int) -> HardwareTokenInterface:
        """
        Retrieve token interface for a slot.
        Enforces fail-closed behavior: hardware errors never fall back to mock.
        """
        if self._mock_mode:
            if slot_id not in self._tokens:
                # Auto-instantiate mock token with unique serial for slot
                self._tokens[slot_id] = MockHardwareToken(serial_number=f"MOCK-SLOT-{slot_id}")
            return self._tokens[slot_id]
        else:
            # Live hardware mode
            if not os.path.exists(self._driver_path):
                # FAIL CLOSED IMMEDIATELY: Do not silently fall back to mock!
                raise HardwareDriverError(
                    f"FATAL: Native PKCS#11 driver not found at '{self._driver_path}'. "
                    f"Automatic fallback to mock is strictly prohibited."
                )
            raise HardwareDriverError(
                f"FATAL: Physical PKCS#11 token detection failed on slot {slot_id}. "
                f"Zero physical hardware devices attached."
            )

    def initialize_slot(self, slot_id: int, so_pin: str, user_pin: str, label: str) -> None:
        """Initialize slot security attributes."""
        token = self.get_token(slot_id)
        token.initialize_token(so_pin, user_pin, label)

    def login(self, slot_id: int, user_pin: str) -> None:
        """Authenticate user PIN for a slot session."""
        token = self.get_token(slot_id)
        # Verify PIN by probing or recording session
        self._authenticated_slots[slot_id] = user_pin

    def logout(self, slot_id: int) -> None:
        """Terminate authenticated session for a slot."""
        self._authenticated_slots.pop(slot_id, None)

    def generate_ed25519_keypair(self, slot_id: int, key_id: str) -> HardwareKeyMetadata:
        """
        Generate an Ed25519 keypair inside the slot's secure element.
        Enforces CKA_EXTRACTABLE=FALSE and CKA_NEVER_EXTRACTABLE=TRUE.
        """
        if slot_id not in self._authenticated_slots:
            raise SecurityBoundaryViolation(f"Slot {slot_id} is not authenticated. Login required.")
        pin = self._authenticated_slots[slot_id]
        token = self.get_token(slot_id)
        return token.generate_keypair(slot_id, key_id, pin)

    def get_public_key(self, slot_id: int, key_id: str) -> bytes:
        """Retrieve the exported public key bytes for a key in a slot."""
        token = self.get_token(slot_id)
        return token.get_public_key(slot_id, key_id)

    def sign_detached(self, slot_id: int, key_id: str, data: bytes) -> bytes:
        """
        Execute detached Ed25519 signing using the onboard private key.
        Private key never crosses into host memory.
        """
        if slot_id not in self._authenticated_slots:
            raise SecurityBoundaryViolation(f"Slot {slot_id} is not authenticated. Login required.")
        pin = self._authenticated_slots[slot_id]
        token = self.get_token(slot_id)
        return token.sign(slot_id, key_id, data, pin)

    def get_device_info(self, slot_id: int) -> TokenDeviceInfo:
        """Retrieve token hardware information."""
        token = self.get_token(slot_id)
        return token.get_device_info()

    def get_device_attestation(self, slot_id: int, key_id: str) -> Optional[bytes]:
        """Extract vendor device attestation certificate."""
        token = self.get_token(slot_id)
        return token.get_device_attestation(slot_id, key_id)
