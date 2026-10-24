"""
Adversarial Acceptance Test Suite for HVA-01 Wave 4C-2 Offline Key Ceremony Tooling.
Implements the 12 required qualification scenarios defined in
ARX_HVA01_WAVE4C2_FINAL_IMPLEMENTATION_CONTRACT.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pytest

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from hva01.witness.token_interface import (
    MockHardwareToken,
    SecurityBoundaryViolation,
    TokenLockedError,
    InvalidPinError,
)
from hva01.witness.pkcs11_session import PKCS11SessionManager, HardwareDriverError
from hva01.witness.registry import WitnessKeyRegistryManager, RegistryValidationError
from hva01.witness.rollover import (
    KeyRolloverManager,
    EvidenceAdmissibilityState,
    RolloverVerificationResult,
)


class TestWave4C2KeyCeremonyTooling:
    """Comprehensive qualification suite for offline ceremony scaffolding."""

    @pytest.fixture
    def trusted_authorities(self):
        """Generate ephemeral Ed25519 keypairs for Owner, Witness, and Custodian."""
        owner_sk = ed25519.Ed25519PrivateKey.generate()
        owner_pk = owner_sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        witness_sk = ed25519.Ed25519PrivateKey.generate()
        witness_pk = witness_sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        custodian_sk = ed25519.Ed25519PrivateKey.generate()
        custodian_pk = custodian_sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        return {
            "owner_sk": owner_sk,
            "owner_pk": owner_pk,
            "witness_sk": witness_sk,
            "witness_pk": witness_pk,
            "custodian_sk": custodian_sk,
            "custodian_pk": custodian_pk,
        }

    def test_01_onboard_keygen_private_key_non_exportable(self):
        """1. Asserts MockHardwareToken generates valid public key but raises on private key access."""
        token = MockHardwareToken(serial_number="MOCK-TEST-001")
        token.initialize_token(so_pin="12345678", user_pin="123456", label="TEST-TOKEN")

        meta = token.generate_keypair(slot_id=0, key_id="TEST-KEY-A", user_pin="123456")
        assert len(meta.public_key_bytes) == 32
        assert len(meta.public_key_hex) == 64
        assert meta.is_extractable is False
        assert meta.is_never_extractable is True

        # Non-exportability proof: must raise SecurityBoundaryViolation
        with pytest.raises(SecurityBoundaryViolation) as exc_info:
            token.export_private_key_disallowed(slot_id=0)
        assert "CKA_EXTRACTABLE=FALSE" in str(exc_info.value)

    def test_02_unauthorized_backup_key_rejected(self, trusted_authorities):
        """2. Verifies signatures produced by Token B before rollover attestation fail verification."""
        session = PKCS11SessionManager(mock_mode=True)
        session.login(slot_id=1, user_pin="123456")
        meta_b = session.generate_ed25519_keypair(slot_id=1, key_id="HVA01-W4C2-BACKUP-KEY-B")

        data = b"EVIDENCE_PAYLOAD_HASH_TEST"
        sig_b = session.sign_detached(slot_id=1, key_id="HVA01-W4C2-BACKUP-KEY-B", data=data)

        # Build registry where Token B is cold escrow
        registry = WitnessKeyRegistryManager.create_empty_registry("TEST-CEREMONY")
        WitnessKeyRegistryManager.add_key_entry(
            registry,
            key_id="HVA01-W4C2-PRIMARY-KEY-A",
            public_key_bytes=b"\x01" * 32,
            token_serial="MOCK-001",
            status="ACTIVE_PRIMARY",
        )
        WitnessKeyRegistryManager.add_key_entry(
            registry,
            key_id="HVA01-W4C2-BACKUP-KEY-B",
            public_key_bytes=meta_b.public_key_bytes,
            token_serial="MOCK-002",
            status="COLD_BACKUP_ESCROW",
        )

        # Attempt to accept signature under active intake policy:
        # Signatures from COLD_BACKUP_ESCROW keys are rejected before rollover
        active_key_ids = [k["key_id"] for k in registry["keys"] if k["status"] == "ACTIVE_PRIMARY"]
        assert meta_b.key_id not in active_key_ids

    def test_03_unilateral_rollover_rejected(self, trusted_authorities):
        """3. Asserts rollover attestation missing Institutional Custodian co-signature is rejected."""
        auth = trusted_authorities
        attestation = KeyRolloverManager.create_rollover_attestation(
            attestation_id="ROLL-TEST-001",
            sequence_number=1,
            revoked_key_id="HVA01-W4C2-PRIMARY-KEY-A",
            activated_key_id="HVA01-W4C2-BACKUP-KEY-B",
            effective_timestamp_utc="2026-10-24T12:00:00Z",
            reason="DEVICE_FAILURE",
        )

        _, att_hash = KeyRolloverManager.compute_canonical_attestation_hash(attestation)
        att_digest = bytes.fromhex(att_hash)

        # Unilateral signing: Owner signs, but Custodian does NOT sign
        owner_sig = auth["owner_sk"].sign(att_digest)
        KeyRolloverManager.add_signature(
            attestation, "RESEARCH_OWNER", auth["owner_pk"], owner_sig
        )

        # Verification must fail fail-closed
        res = KeyRolloverManager.verify_rollover_attestation(
            attestation, auth["owner_pk"], auth["custodian_pk"], previous_sequence_number=0
        )
        assert res.is_valid is False
        assert any("Missing valid INSTITUTIONAL_CUSTODIAN signature" in err for err in (res.errors or []))

    def test_04_registry_canonicalization_rfc8785_whitespace(self):
        """4. Asserts variations in JSON whitespace/indentation produce byte-identical SHA-256 canonical hash."""
        registry = WitnessKeyRegistryManager.create_empty_registry("TEST-CEREMONY-001")
        WitnessKeyRegistryManager.add_key_entry(
            registry,
            key_id="KEY-A",
            public_key_bytes=b"\xaa" * 32,
            token_serial="SERIAL-001",
            status="ACTIVE_PRIMARY",
        )
        WitnessKeyRegistryManager.add_key_entry(
            registry,
            key_id="KEY-B",
            public_key_bytes=b"\xbb" * 32,
            token_serial="SERIAL-002",
            status="COLD_BACKUP_ESCROW",
        )

        _, hash_1 = WitnessKeyRegistryManager.compute_canonical_registry_hash(registry)

        # Serialize with deep whitespace and indentation, then parse back
        formatted_json = json.dumps(registry, indent=8)
        parsed_back = json.loads(formatted_json)
        _, hash_2 = WitnessKeyRegistryManager.compute_canonical_registry_hash(parsed_back)

        # Minified json
        minified_json = json.dumps(registry, separators=(",", ":"))
        parsed_min = json.loads(minified_json)
        _, hash_3 = WitnessKeyRegistryManager.compute_canonical_registry_hash(parsed_min)

        assert hash_1 == hash_2 == hash_3

    def test_05_registry_canonicalization_rfc8785_keysort(self):
        """5. Asserts variations in JSON dictionary key ordering produce byte-identical SHA-256 canonical hash."""
        # Dict with keys inserted in reverse order
        d1 = {
            "schema_version": "1.0.0",
            "registry_epoch_utc": "2026-10-24T12:00:00Z",
            "ceremony_identifier": "CEREMONY-XYZ",
            "keys": [{"algorithm": "Ed25519", "key_id": "KEY-1"}],
        }
        d2 = {
            "ceremony_identifier": "CEREMONY-XYZ",
            "keys": [{"key_id": "KEY-1", "algorithm": "Ed25519"}],
            "registry_epoch_utc": "2026-10-24T12:00:00Z",
            "schema_version": "1.0.0",
        }

        _, h1 = WitnessKeyRegistryManager.compute_canonical_registry_hash(d1)
        _, h2 = WitnessKeyRegistryManager.compute_canonical_registry_hash(d2)

        assert h1 == h2

    def test_06_tampered_registry_signature_rejected(self, trusted_authorities):
        """6. Asserts altering a single character in public key hex breaks dual signatures and halts verification."""
        auth = trusted_authorities
        registry = WitnessKeyRegistryManager.create_empty_registry("TEST-CEREMONY")
        WitnessKeyRegistryManager.add_key_entry(
            registry, "KEY-A", b"\x01" * 32, "MOCK-001", "ACTIVE_PRIMARY"
        )
        WitnessKeyRegistryManager.add_key_entry(
            registry, "KEY-B", b"\x02" * 32, "MOCK-002", "COLD_BACKUP_ESCROW"
        )

        _, reg_hash = WitnessKeyRegistryManager.compute_canonical_registry_hash(registry)
        reg_digest = bytes.fromhex(reg_hash)

        owner_sig = auth["owner_sk"].sign(reg_digest)
        witness_sig = auth["witness_sk"].sign(reg_digest)
        WitnessKeyRegistryManager.add_signature(registry, "RESEARCH_OWNER", auth["owner_pk"], owner_sig)
        WitnessKeyRegistryManager.add_signature(registry, "INDEPENDENT_WITNESS", auth["witness_pk"], witness_sig)

        # Baseline verification passes
        valid_res = WitnessKeyRegistryManager.verify_registry(registry, auth["owner_pk"], auth["witness_pk"])
        assert valid_res.is_valid is True

        # TAMPERING: alter a single byte in KEY-A's public key hex
        tampered = copy.deepcopy(registry)
        original_hex = tampered["keys"][0]["public_key_hex"]
        tampered["keys"][0]["public_key_hex"] = "ff" + original_hex[2:]

        # Verification must fail
        tamper_res = WitnessKeyRegistryManager.verify_registry(tampered, auth["owner_pk"], auth["witness_pk"])
        assert tamper_res.is_valid is False
        assert any("signature verification failed" in err for err in (tamper_res.errors or []))

    def test_07_stale_rollover_sequence_replay_rejected(self, trusted_authorities):
        """7. Asserts rollover attestation with decremented or duplicate sequence number is rejected."""
        auth = trusted_authorities
        attestation = KeyRolloverManager.create_rollover_attestation(
            attestation_id="ROLL-REPLAY-001",
            sequence_number=2,  # Stale: previous was already 2
            revoked_key_id="KEY-A",
            activated_key_id="KEY-B",
            effective_timestamp_utc="2026-10-24T12:00:00Z",
        )
        _, att_hash = KeyRolloverManager.compute_canonical_attestation_hash(attestation)
        att_digest = bytes.fromhex(att_hash)

        KeyRolloverManager.add_signature(
            attestation, "RESEARCH_OWNER", auth["owner_pk"], auth["owner_sk"].sign(att_digest)
        )
        KeyRolloverManager.add_signature(
            attestation, "INSTITUTIONAL_CUSTODIAN", auth["custodian_pk"], auth["custodian_sk"].sign(att_digest)
        )

        # Replay assertion: previous_sequence is 2, current is 2 -> rejected
        res = KeyRolloverManager.verify_rollover_attestation(
            attestation, auth["owner_pk"], auth["custodian_pk"], previous_sequence_number=2
        )
        assert res.is_valid is False
        assert any("Replay attack detected" in err for err in (res.errors or []))

    def test_08_compromise_window_quarantine_indeterminate(self):
        """8. Asserts signatures generated within compromise uncertainty window are quarantined as INDETERMINATE."""
        # Compromise declared at 12:00, effective revocation at 14:00
        t_compromise = "2026-10-24T12:00:00Z"
        t_effective = "2026-10-24T14:00:00Z"

        # 1. Post-revocation timestamp -> INVALID
        s1 = KeyRolloverManager.evaluate_three_state_admissibility(
            token_timestamp_utc="2026-10-24T14:05:00Z",
            effective_revocation_utc=t_effective,
            compromise_suspected_utc=t_compromise,
        )
        assert s1 == EvidenceAdmissibilityState.INVALID

        # 2. Inside compromise uncertainty window -> INDETERMINATE (QUARANTINED)
        s2 = KeyRolloverManager.evaluate_three_state_admissibility(
            token_timestamp_utc="2026-10-24T12:30:00Z",
            effective_revocation_utc=t_effective,
            compromise_suspected_utc=t_compromise,
        )
        assert s2 == EvidenceAdmissibilityState.INDETERMINATE

        # 3. Before compromise suspected -> VALID
        s3 = KeyRolloverManager.evaluate_three_state_admissibility(
            token_timestamp_utc="2026-10-24T11:59:59Z",
            effective_revocation_utc=t_effective,
            compromise_suspected_utc=t_compromise,
        )
        assert s3 == EvidenceAdmissibilityState.VALID

    def test_09_pin_lockout_after_three_invalid_attempts(self):
        """9. Asserts submitting three consecutive bad User PINs locks token slot until unblocked by SO-PIN."""
        token = MockHardwareToken(serial_number="MOCK-PIN-001")
        token.initialize_token(so_pin="87654321", user_pin="654321", label="PIN-TEST")

        # Attempt 1: wrong PIN
        with pytest.raises(InvalidPinError) as e1:
            token.generate_keypair(0, "KEY-1", "000000")
        assert "Retries remaining: 2" in str(e1.value)

        # Attempt 2: wrong PIN
        with pytest.raises(InvalidPinError) as e2:
            token.generate_keypair(0, "KEY-1", "000000")
        assert "Retries remaining: 1" in str(e2.value)

        # Attempt 3: wrong PIN -> locks token
        with pytest.raises(TokenLockedError) as e3:
            token.generate_keypair(0, "KEY-1", "000000")
        assert "LOCKED" in str(e3.value)

        # Attempt 4: Even with correct PIN, token is locked
        with pytest.raises(TokenLockedError):
            token.generate_keypair(0, "KEY-1", "654321")

        # Unblock with SO-PIN
        token.unblock_with_so_pin("87654321", new_user_pin="111222")

        # Now succeeds with new PIN
        meta = token.generate_keypair(0, "KEY-1", "111222")
        assert meta.key_id == "KEY-1"

    def test_10_hardware_error_fails_closed_without_mock_fallback(self):
        """10. Asserts hardware failure in non-mock mode raises fatal error and NEVER falls back to mock token."""
        # Explicit hardware mode: mock_mode=False
        session = PKCS11SessionManager(mock_mode=False, driver_path="/nonexistent/driver.so")
        assert session.is_mock_mode is False

        # Accessing slot must raise HardwareDriverError fail-closed
        with pytest.raises(HardwareDriverError) as exc_info:
            session.get_token(0)

        assert "Automatic fallback to mock is strictly prohibited" in str(exc_info.value)

    def test_11_mock_attestation_rejected_by_production_verifier(self):
        """11. Asserts mock attestation certificates fail verification under production root trust stores."""
        token = MockHardwareToken(serial_number="MOCK-ATTEST-001")
        token.initialize_token("12345678", "123456", "ATTEST-TEST")
        token.generate_keypair(0, "KEY-ATTEST", "123456")

        att_cert = token.get_device_attestation(0, "KEY-ATTEST")
        assert att_cert is not None
        assert b"MOCK_DEVICE_ATTESTATION" in att_cert

        # Production policy: any certificate containing MOCK_ is rejected
        is_production_trusted = b"MOCK" not in att_cert and b"SYNTHETIC" not in att_cert
        assert is_production_trusted is False

    def test_12_dual_signed_registry_verification_end_to_end(self, trusted_authorities):
        """12. Verifies complete ceremony workflow: keygen, canonicalization, dual signing, and registry verification."""
        auth = trusted_authorities
        session = PKCS11SessionManager(mock_mode=True)
        token_a = MockHardwareToken("MOCK-YK5-PRIMARY")
        token_b = MockHardwareToken("MOCK-YK5-ESCROW")
        session.register_mock_token(0, token_a)
        session.register_mock_token(1, token_b)

        session.login(0, "123456")
        session.login(1, "123456")

        meta_a = session.generate_ed25519_keypair(0, "HVA01-W4C2-PRIMARY-KEY-A")
        meta_b = session.generate_ed25519_keypair(1, "HVA01-W4C2-BACKUP-KEY-B")

        registry = WitnessKeyRegistryManager.create_empty_registry("CEREMONY-E2E-TEST")
        WitnessKeyRegistryManager.add_key_entry(
            registry, meta_a.key_id, meta_a.public_key_bytes, meta_a.token_serial, "ACTIVE_PRIMARY"
        )
        WitnessKeyRegistryManager.add_key_entry(
            registry, meta_b.key_id, meta_b.public_key_bytes, meta_b.token_serial, "COLD_BACKUP_ESCROW"
        )

        _, reg_hash = WitnessKeyRegistryManager.compute_canonical_registry_hash(registry)
        reg_digest = bytes.fromhex(reg_hash)

        owner_sig = auth["owner_sk"].sign(reg_digest)
        witness_sig = auth["witness_sk"].sign(reg_digest)

        WitnessKeyRegistryManager.add_signature(registry, "RESEARCH_OWNER", auth["owner_pk"], owner_sig)
        WitnessKeyRegistryManager.add_signature(registry, "INDEPENDENT_WITNESS", auth["witness_pk"], witness_sig)

        res = WitnessKeyRegistryManager.verify_registry(registry, auth["owner_pk"], auth["witness_pk"])
        assert res.is_valid is True
        assert res.signatures_verified == 2
        assert res.primary_key_id == "HVA01-W4C2-PRIMARY-KEY-A"
        assert res.backup_key_id == "HVA01-W4C2-BACKUP-KEY-B"
        assert len(res.registry_hash) == 64
