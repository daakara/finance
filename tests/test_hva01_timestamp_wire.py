"""
HVA-01 Wave 4C-1: RFC 3161 Wire Engine & Dual-TSA Fallback Acceptance Test Suite.
Verifies all 6 Acceptance Criteria (AC-01 to AC-06) and 4 Mandatory Negative-Test Families (NT-01 to NT-04).

GOVERNANCE CLASSIFICATION: TIER 3 — Cryptographic and Evidence-Integrity Controls.
FAIL-CLOSED / SYNTHETIC-ONLY / ZERO NETWORK EGRESS.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from hva01.witness.client import (
    DualTSAFallbackEngine,
    TSAFailClosedError,
    TSAFailoverExhaustedError,
    TSAFallbackState,
    TSAProviderConfig,
    TSATimeoutError,
    TSATransportError,
)
from hva01.witness.timestamp import compute_attestation_digest
from hva01.witness.timestamp_wire import (
    DEFAULT_TSA_POLICY_OID,
    ClockDriftExceededError,
    InvalidSignatureError,
    InvalidTimestampTokenError,
    MalformedAsn1Error,
    PKIStatusError,
    RequestResponseBindingError,
    TSAPolicyMismatchError,
    UntrustedTSACertificateError,
    UnsupportedAlgorithmError,
    create_synthetic_timestamp_response,
    create_timestamp_request,
    parse_and_verify_timestamp_response,
    parse_timestamp_request,
    verify_tsa_certificate_trust,
)


# ============================================================================
# Network Denial Guard (Enforces Zero External Network Egress)
# ============================================================================

@pytest.fixture(autouse=True)
def network_denial_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed guard ensuring no test attempts external socket connections."""
    def guarded_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("FAIL-CLOSED: External network socket access is strictly prohibited under Wave 4C-1!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ============================================================================
# Synthetic PKI Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def pki_setup() -> Dict[str, Any]:
    """
    Generate synthetic test Root CA, Primary TSA, and Secondary TSA keys and certificates.
    All certificates have valid validity dates and proper ExtendedKeyUsage for Time Stamping.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Root CA
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic ARX Root CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=10))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 2. Primary TSA (RSA 2048)
    p_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    p_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Primary Synthetic TSA Desk 01"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    p_cert = (
        x509.CertificateBuilder()
        .subject_name(p_subject)
        .issuer_name(ca_subject)
        .public_key(p_key.public_key())
        .serial_number(1001)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 3. Secondary TSA (ECDSA SECP256R1)
    s_key = ec.generate_private_key(ec.SECP256R1())
    s_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Secondary Synthetic TSA Desk 02"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    s_cert = (
        x509.CertificateBuilder()
        .subject_name(s_subject)
        .issuer_name(ca_subject)
        .public_key(s_key.public_key())
        .serial_number(1002)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 4. Untrusted Attacker TSA (Self-signed)
    bad_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Malicious Rogue TSA"),
    ])
    bad_cert = (
        x509.CertificateBuilder()
        .subject_name(bad_subject)
        .issuer_name(bad_subject)
        .public_key(bad_key.public_key())
        .serial_number(666)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(bad_key, hashes.SHA256())
    )

    # 5. Expired TSA
    exp_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    exp_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Expired TSA")])
    exp_cert = (
        x509.CertificateBuilder()
        .subject_name(exp_subject)
        .issuer_name(ca_subject)
        .public_key(exp_key.public_key())
        .serial_number(1003)
        .not_valid_before(now - datetime.timedelta(days=30))
        .not_valid_after(now - datetime.timedelta(days=1))  # Expired yesterday
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 6. Cert without Time Stamping EKU (e.g. Server Auth only)
    no_eku_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    no_eku_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "No EKU TSA")])
    no_eku_cert = (
        x509.CertificateBuilder()
        .subject_name(no_eku_subject)
        .issuer_name(ca_subject)
        .public_key(no_eku_key.public_key())
        .serial_number(1004)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # 7. Valid Intermediate CA (chained to ca_cert)
    inter_ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    inter_ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Authorized Synthetic Intermediate CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    inter_ca_cert = (
        x509.CertificateBuilder()
        .subject_name(inter_ca_subject)
        .issuer_name(ca_subject)
        .public_key(inter_ca_key.public_key())
        .serial_number(1005)
        .not_valid_before(now - datetime.timedelta(days=5))
        .not_valid_after(now + datetime.timedelta(days=730))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 8. Chained Leaf TSA (chained to inter_ca_cert)
    chained_tsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    chained_tsa_subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Chained Synthetic TSA Leaf"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ARX Terminal Research"),
    ])
    chained_tsa_cert = (
        x509.CertificateBuilder()
        .subject_name(chained_tsa_subject)
        .issuer_name(inter_ca_subject)
        .public_key(chained_tsa_key.public_key())
        .serial_number(1006)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=True, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(inter_ca_key, hashes.SHA256())
    )

    # 9. Defective Intermediate CA (BasicConstraints ca=False)
    bad_inter_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad_inter_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Defective Intermediate CA (ca=False)")])
    bad_inter_cert = (
        x509.CertificateBuilder()
        .subject_name(bad_inter_subject)
        .issuer_name(ca_subject)
        .public_key(bad_inter_key.public_key())
        .serial_number(1007)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 10. Chained Leaf under Defective Intermediate CA
    bad_inter_leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad_inter_leaf_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf under ca=False Intermediate")])
    bad_inter_leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(bad_inter_leaf_subject)
        .issuer_name(bad_inter_subject)
        .public_key(bad_inter_leaf_key.public_key())
        .serial_number(1008)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(bad_inter_key, hashes.SHA256())
    )

    # 11. Rogue Untrusted Root and Leaf
    rogue_root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rogue_root_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Untrusted Rogue Root CA")])
    rogue_root_cert = (
        x509.CertificateBuilder()
        .subject_name(rogue_root_subject)
        .issuer_name(rogue_root_subject)
        .public_key(rogue_root_key.public_key())
        .serial_number(2001)
        .not_valid_before(now - datetime.timedelta(days=10))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(rogue_root_key, hashes.SHA256())
    )
    rogue_leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rogue_leaf_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf Chained to Rogue Root")])
    rogue_leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(rogue_leaf_subject)
        .issuer_name(rogue_root_subject)
        .public_key(rogue_leaf_key.public_key())
        .serial_number(2002)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .sign(rogue_root_key, hashes.SHA256())
    )

    return {
        "ca_key": ca_key,
        "ca_cert": ca_cert,
        "primary_key": p_key,
        "primary_cert": p_cert,
        "secondary_key": s_key,
        "secondary_cert": s_cert,
        "bad_key": bad_key,
        "bad_cert": bad_cert,
        "expired_key": exp_key,
        "expired_cert": exp_cert,
        "no_eku_key": no_eku_key,
        "no_eku_cert": no_eku_cert,
        "inter_ca_key": inter_ca_key,
        "inter_ca_cert": inter_ca_cert,
        "chained_tsa_key": chained_tsa_key,
        "chained_tsa_cert": chained_tsa_cert,
        "bad_inter_key": bad_inter_key,
        "bad_inter_cert": bad_inter_cert,
        "bad_inter_leaf_key": bad_inter_leaf_key,
        "bad_inter_leaf_cert": bad_inter_leaf_cert,
        "rogue_root_key": rogue_root_key,
        "rogue_root_cert": rogue_root_cert,
        "rogue_leaf_key": rogue_leaf_key,
        "rogue_leaf_cert": rogue_leaf_cert,
        "trusted_certs": [p_cert, s_cert, ca_cert],
    }


# ============================================================================
# Category 1: Acceptance Criteria (AC-01 through AC-06)
# ============================================================================

class TestCategory1AcceptanceCriteria:
    """Verifies all six core acceptance criteria for Wave 4C-1."""

    def test_ac01_rfc3161_wire_correctness(self, pki_setup: Dict[str, Any]) -> None:
        """AC-01: Valid request serialization and response parsing under strict ASN.1 DER."""
        digest = "sha256:" + hashlib.sha256(b"evidence-payload-test").hexdigest()
        nonce = 9876543210

        req_der = create_timestamp_request(digest, nonce=nonce, cert_req=True)
        assert len(req_der) > 0

        parsed = parse_timestamp_request(req_der)
        assert parsed["version"] == 1
        assert parsed["hash_algorithm"] == "SHA-256"
        assert parsed["nonce"] == nonce
        assert parsed["cert_req"] is True
        assert parsed["hashed_message_hex"] == digest.split(":")[1]

    def test_ac02_cryptographic_timestamp_authenticity(self, pki_setup: Dict[str, Any]) -> None:
        """AC-02: Accepted tokens have verified CMS signature, trusted signer, and EKU."""
        digest = "sha256:" + hashlib.sha256(b"ac02-digest").hexdigest()
        nonce = 11223344

        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            nonce=nonce,
            serial_number=7001,
        )

        verified = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            expected_nonce=nonce,
            trusted_certs=pki_setup["trusted_certs"],
        )

        assert verified.serial_number == 7001
        assert verified.nonce == nonce
        assert verified.signer_common_name == "Primary Synthetic TSA Desk 01"
        assert verified.tsa_certificate_fingerprint.startswith("sha256:")
        assert verified.is_synthetic is True

    def test_ac03_request_response_integrity(self, pki_setup: Dict[str, Any]) -> None:
        """AC-03: Accepted tokens match message imprint digest and nonce exactly."""
        digest = "sha256:" + hashlib.sha256(b"ac03-integrity").hexdigest()
        nonce = 55667788

        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["secondary_key"],
            signing_cert=pki_setup["secondary_cert"],
            nonce=nonce,
            serial_number=7002,
        )

        verified = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            expected_nonce=nonce,
            trusted_certs=pki_setup["trusted_certs"],
        )
        assert verified.message_imprint_digest == digest
        assert verified.nonce == nonce

    def test_ac04_deterministic_dual_provider_failover(self, pki_setup: Dict[str, Any]) -> None:
        """AC-04: Failover follows policy with bounded attempts, no trust downgrade."""
        digest = "sha256:" + hashlib.sha256(b"ac04-failover").hexdigest()

        def primary_fail(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATransportError("Primary simulated network outage (HTTP 503)")

        def secondary_succeed(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                imprint_digest=digest,
                signing_key=pki_setup["secondary_key"],
                signing_cert=pki_setup["secondary_cert"],
                serial_number=8001,
            )

        p_cfg = TSAProviderConfig(
            provider_id="primary-tsa",
            endpoint_url="http://primary.tsa.local",
            trusted_certificates=[pki_setup["primary_cert"]],
            transport_handler=primary_fail,
            max_retries=1,
            backoff_base_seconds=0.01,
        )
        s_cfg = TSAProviderConfig(
            provider_id="secondary-tsa",
            endpoint_url="http://secondary.tsa.local",
            trusted_certificates=[pki_setup["secondary_cert"]],
            transport_handler=secondary_succeed,
            max_retries=1,
            backoff_base_seconds=0.01,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)

        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED
        assert prov.selected_provider_role == "SECONDARY"
        assert prov.selected_provider_id == "secondary-tsa"
        assert v_ts.serial_number == 8001
        assert len(prov.attempts) == 3  # Primary attempt 1 + retry attempt 2 + Secondary attempt 1

    def test_ac05_evidence_and_response_locking_integrity(self, pki_setup: Dict[str, Any]) -> None:
        """AC-05: Existing HVA-01 attestation digest D_attest and response locking preserved."""
        # Compute D_attest using existing authoritative formula
        merkle_root = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        key_fp = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
        sig_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
        d_attest = compute_attestation_digest(merkle_root, key_fp, sig_b64)

        # Wire client covers d_attest without mutation
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=d_attest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9001,
        )

        v_ts = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=d_attest,
            trusted_certs=pki_setup["trusted_certs"],
        )

        assert v_ts.message_imprint_digest == d_attest
        # Check compatible token dictionary format
        tok_dict = v_ts.to_token_dict()
        assert tok_dict["message_imprint_digest"] == d_attest
        assert tok_dict["verification_status"] == "VERIFIED_CRYPTOGRAPHIC_CMS"

    def test_ac06_reproducible_synthetic_verification(self, pki_setup: Dict[str, Any]) -> None:
        """AC-06: Verification executes 100% offline, deterministically, with zero external network traffic."""
        digest = "sha256:" + hashlib.sha256(b"ac06-synthetic").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9999,
        )
        verified = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            trusted_certs=pki_setup["trusted_certs"],
        )
        assert verified.is_synthetic is True


# ============================================================================
# Category 2: Negative Test Family NT-01 (ASN.1 & Wire Rejection)
# ============================================================================

class TestNegativeFamilyNT01WireRejection:
    """Verifies fail-closed rejection of malformed, oversized, or non-DER payloads."""

    def test_nt01_01_malformed_asn1_request(self) -> None:
        """Malformed ASN.1 bytes in request must be rejected."""
        with pytest.raises(MalformedAsn1Error):
            parse_timestamp_request(b"\x30\x05\xFF\xFF\xFF\xFF\xFF")

    def test_nt01_02_truncated_der_request(self) -> None:
        """Truncated DER request must be rejected."""
        req = create_timestamp_request("sha256:" + "a" * 64)
        with pytest.raises(MalformedAsn1Error):
            parse_timestamp_request(req[: len(req) // 2])

    def test_nt01_03_oversized_request(self) -> None:
        """Request exceeding max size limit must fail closed."""
        req = create_timestamp_request("sha256:" + "a" * 64)
        with pytest.raises(MalformedAsn1Error):
            parse_timestamp_request(req, max_size=len(req) - 5)

    def test_nt01_04_trailing_unparsed_bytes_request(self) -> None:
        """Non-strict DER with trailing garbage bytes must fail closed."""
        req = create_timestamp_request("sha256:" + "a" * 64)
        with pytest.raises(MalformedAsn1Error, match="Trailing unparsed bytes"):
            parse_timestamp_request(req + b"\x00\x00\xDE\xAD\xBE\xEF")

    def test_nt01_05_unsupported_hash_algorithm_request(self) -> None:
        """Unsupported algorithm (e.g. MD5) in request creation must fail closed."""
        with pytest.raises(UnsupportedAlgorithmError):
            create_timestamp_request("a" * 64, hash_algorithm="MD5")

    def test_nt01_06_malformed_response_der(self, pki_setup: Dict[str, Any]) -> None:
        """Garbage bytes passed as response must fail closed."""
        with pytest.raises(MalformedAsn1Error):
            parse_and_verify_timestamp_response(
                resp_der=b"NOT_A_VALID_DER_RESPONSE",
                expected_imprint_digest="sha256:" + "a" * 64,
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt01_07_trailing_bytes_in_response(self, pki_setup: Dict[str, Any]) -> None:
        """Response with trailing bytes must fail closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest, pki_setup["primary_key"], pki_setup["primary_cert"]
        )
        with pytest.raises(MalformedAsn1Error, match="Trailing unparsed bytes"):
            parse_and_verify_timestamp_response(
                resp_der=resp + b"\xDE\xAD",
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt01_08_missing_timestamp_token_in_granted_response(self, pki_setup: Dict[str, Any]) -> None:
        """Response claiming granted (status 0) but omitting token must fail closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            omit_token=True,
        )
        with pytest.raises(InvalidTimestampTokenError, match="timeStampToken is missing"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt01_09_pki_status_rejection(self, pki_setup: Dict[str, Any]) -> None:
        """Response with PKIStatus != granted must fail closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            pki_status=2,  # rejection
            status_string="TSA busy, please retry later",
            fail_info=0x40000000,
        )
        with pytest.raises(PKIStatusError, match="TSA rejected timestamp request"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
            )


# ============================================================================
# Category 3: Negative Test Family NT-02 (Cryptographic Trust Rejection)
# ============================================================================

class TestNegativeFamilyNT02TrustRejection:
    """Verifies fail-closed rejection of invalid signatures, rogue certificates, and expired certs."""

    def test_nt02_01_corrupted_signature_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Mathematically corrupted signature bytes must be rejected fail-closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            corrupt_signature=True,
        )
        with pytest.raises(InvalidSignatureError):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt02_02_untrusted_signing_certificate_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Certificate not signed by trusted CA or in trust store must be rejected."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["bad_key"],
            pki_setup["bad_cert"],
        )
        # Trust store contains only primary, secondary, and CA, but NOT bad_cert
        with pytest.raises(UntrustedTSACertificateError, match="is not in trusted store"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],  # Strict trust store
            )

    def test_nt02_03_expired_signing_certificate_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Expired TSA certificate must be rejected fail-closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["expired_key"],
            pki_setup["expired_cert"],
        )
        with pytest.raises(UntrustedTSACertificateError, match="not valid at timestamp generation time"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["expired_cert"], pki_setup["ca_cert"]],
            )

    def test_nt02_04_missing_timestamping_eku_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Certificate lacking ExtendedKeyUsage id-kp-timeStamping must be rejected."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["no_eku_key"],
            pki_setup["no_eku_cert"],
        )
        with pytest.raises(UntrustedTSACertificateError, match="missing required ExtendedKeyUsage"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["no_eku_cert"], pki_setup["ca_cert"]],
            )

    def test_nt02_05_unauthorized_policy_oid_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Timestamp returned with unexpected policy OID must fail closed."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            policy_oid="1.2.3.4.5.6.7.8.9",
        )
        with pytest.raises(TSAPolicyMismatchError, match="does not match expected policy"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
                expected_policy_oid=DEFAULT_TSA_POLICY_OID,
            )

    def test_nt02_06_clock_drift_exceeded_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Timestamp generation time exceeding maximum drift allowance must fail closed."""
        digest = "sha256:" + "a" * 64
        # Timestamp issued 5 minutes in the past
        now = datetime.datetime.now(datetime.timezone.utc)
        past_time = (now - datetime.timedelta(seconds=300)).isoformat()

        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            gen_time_utc=past_time,
        )

        with pytest.raises(ClockDriftExceededError, match="drift .* exceeds limit"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
                reference_time_utc=now.isoformat(),
                max_drift_seconds=30.0,
            )


# ============================================================================
# Category 4: Negative Test Family NT-03 (Request-Binding Rejection)
# ============================================================================

class TestNegativeFamilyNT03RequestBindingRejection:
    """Verifies fail-closed rejection of digest mismatches, nonce substitutions, and algorithm shifts."""

    def test_nt03_01_mismatched_imprint_digest_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Token returned with different message imprint digest must be rejected."""
        expected_digest = "sha256:" + "a" * 64
        different_digest = "sha256:" + "b" * 64

        resp = create_synthetic_timestamp_response(
            different_digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
        )

        with pytest.raises(RequestResponseBindingError, match="Message imprint digest mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=expected_digest,
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt03_02_mismatched_nonce_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Token returned with mismatched nonce must be rejected."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            nonce=123456,
        )

        with pytest.raises(RequestResponseBindingError, match="Nonce mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                expected_nonce=654321,  # Requested different nonce
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt03_03_missing_nonce_when_requested_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Token omitting nonce when nonce was requested must be rejected."""
        digest = "sha256:" + "a" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            nonce=None,  # Response omits nonce
        )

        with pytest.raises(RequestResponseBindingError, match="Nonce was requested but response contains no nonce"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                expected_nonce=123456,  # Requested nonce
                trusted_certs=pki_setup["trusted_certs"],
            )

    def test_nt03_04_corrupted_imprint_fails(self, pki_setup: Dict[str, Any]) -> None:
        """Token with corrupted digest bytes in message imprint must fail closed."""
        digest = "sha256:" + "c" * 64
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            corrupt_imprint=True,
        )

        with pytest.raises(RequestResponseBindingError):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=pki_setup["trusted_certs"],
            )


# ============================================================================
# Category 5: Negative Test Family NT-04 (Fallback & Recovery)
# ============================================================================

class TestNegativeFamilyNT04FallbackAndRecovery:
    """Verifies fail-closed dual-TSA failover behavior, retry bounds, and fault injection."""

    def test_nt04_01_primary_timeout_fallback_success(self, pki_setup: Dict[str, Any]) -> None:
        """Primary timeout triggers secondary fallback which succeeds."""
        digest = "sha256:" + hashlib.sha256(b"nt04-01").hexdigest()

        def timeout_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATimeoutError("Primary socket timeout")

        def secondary_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(digest, pki_setup["secondary_key"], pki_setup["secondary_cert"])

        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=timeout_handler,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)
        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED
        assert prov.selected_provider_role == "SECONDARY"

    def test_nt04_02_primary_http_500_fallback_success(self, pki_setup: Dict[str, Any]) -> None:
        """Primary HTTP 500 error triggers secondary fallback which succeeds."""
        digest = "sha256:" + hashlib.sha256(b"nt04-02").hexdigest()

        def err_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATransportError("HTTP 500 Internal Server Error")

        def secondary_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                digest, pki_setup["secondary_key"], pki_setup["secondary_cert"]
            )

        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=err_handler,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)
        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED
        assert prov.selected_provider_role == "SECONDARY"

    def test_nt04_03_primary_corrupt_response_fallback_success(self, pki_setup: Dict[str, Any]) -> None:
        """Primary returning corrupted DER triggers secondary fallback."""
        digest = "sha256:" + hashlib.sha256(b"nt04-03").hexdigest()

        def corrupt_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return b"\x30\x05GARBAGE_BYTES"

        def secondary_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                digest, pki_setup["secondary_key"], pki_setup["secondary_cert"]
            )

        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=corrupt_handler,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)
        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED

    def test_nt04_04_primary_untrusted_cert_fallback_success(self, pki_setup: Dict[str, Any]) -> None:
        """Primary returning rogue untrusted certificate triggers secondary fallback."""
        digest = "sha256:" + hashlib.sha256(b"nt04-04").hexdigest()

        def rogue_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            # Signs with bad key/cert not in trust store
            return create_synthetic_timestamp_response(digest, pki_setup["bad_key"], pki_setup["bad_cert"])

        def secondary_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                digest, pki_setup["secondary_key"], pki_setup["secondary_cert"]
            )

        # Primary config trusts only primary_cert; rogue response will be rejected
        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=rogue_handler,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)
        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED

    def test_nt04_05_both_providers_fail_fails_closed(self, pki_setup: Dict[str, Any]) -> None:
        """When both providers fail, engine transitions to FAILED_CLOSED and raises exception."""
        digest = "sha256:" + hashlib.sha256(b"nt04-05").hexdigest()

        def timeout_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATimeoutError("Simulated timeout")

        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=timeout_handler,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=timeout_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        with pytest.raises(TSAFailoverExhaustedError, match="Dual-TSA failover exhausted"):
            engine.request_timestamp(digest)

        assert engine.state == TSAFallbackState.FAILED_CLOSED

    def test_nt04_06_invalid_client_input_immediate_fail_closed(self, pki_setup: Dict[str, Any]) -> None:
        """Invalid client input (e.g. malformed digest hex) fails immediately without calling secondary."""
        called_secondary = False

        def secondary_handler(req: bytes, cfg: TSAProviderConfig) -> bytes:
            nonlocal called_secondary
            called_secondary = True
            return b""

        p_cfg = TSAProviderConfig(
            "p-tsa", "http://p.tsa", [pki_setup["primary_cert"]], max_retries=0
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_handler,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        with pytest.raises(TSAFailClosedError):
            engine.request_timestamp("INVALID_NOT_HEX_DIGEST")

        assert engine.state == TSAFallbackState.FAILED_CLOSED
        assert called_secondary is False, "Secondary provider must never be called on client invalid input"

    def test_nt04_07_audit_log_complete_traceability(self, pki_setup: Dict[str, Any]) -> None:
        """Verify audit trail records all transitions, states, and error reasons."""
        digest = "sha256:" + hashlib.sha256(b"nt04-07").hexdigest()

        def primary_fail(req: bytes, cfg: TSAProviderConfig) -> bytes:
            raise TSATransportError("Connection reset by peer")

        def secondary_succeed(req: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                digest, pki_setup["secondary_key"], pki_setup["secondary_cert"]
            )

        p_cfg = TSAProviderConfig(
            "p-tsa",
            "http://p.tsa",
            [pki_setup["primary_cert"]],
            transport_handler=primary_fail,
            max_retries=0,
        )
        s_cfg = TSAProviderConfig(
            "s-tsa",
            "http://s.tsa",
            [pki_setup["secondary_cert"]],
            transport_handler=secondary_succeed,
            max_retries=0,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)

        transitions = [entry["to_state"] for entry in engine.audit_log]
        assert transitions == [
            "PRIMARY_REQUESTED",
            "PRIMARY_REJECTED",
            "SECONDARY_REQUESTED",
            "SECONDARY_ACCEPTED",
        ]
        prov_dict = prov.to_dict()
        assert prov_dict["total_attempts"] == 2
        assert prov_dict["selected_provider_role"] == "SECONDARY"
        assert prov_dict["final_state"] == "SECONDARY_ACCEPTED"


# ============================================================================
# Category 6: SEC-4C1-01 Remediated Trust Enforcement (Mandatory Adversarial)
# ============================================================================

class TestCategory6RemediatedTrustEnforcementSEC01:
    """
    Verifies full fail-closed resolution of SEC-4C1-01:
    Mandatory trust anchor enforcement, multi-tier chain traversal, BasicConstraints CA validation,
    EKU timeStamping checks, and rejection of arbitrary self-signed signers.
    """

    def test_sec01_01_no_configured_trust_anchors_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 1: Passing empty or None trusted_certs must fail closed immediately."""
        digest = "sha256:" + hashlib.sha256(b"sec01-no-trust").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest, pki_setup["primary_key"], pki_setup["primary_cert"]
        )

        with pytest.raises(UntrustedTSACertificateError, match="No trusted certificates configured"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[],  # Empty list
            )

        with pytest.raises(UntrustedTSACertificateError, match="No trusted certificates configured"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=None,  # type: ignore[arg-type]
            )

    def test_sec01_02_untrusted_self_signed_with_valid_eku_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 9: Rogue self-signed TSA certificate with valid EKU must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec01-rogue-self-signed").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest, pki_setup["bad_key"], pki_setup["bad_cert"]
        )

        with pytest.raises(UntrustedTSACertificateError, match="is self-signed"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],  # Trust store only has root CA
            )

    def test_sec01_03_trusted_root_invalid_signer_chain_signature_mismatch(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Requirement 3: Signer claiming trusted CA issuer but signed with rogue key must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec01-forged-chain").hexdigest()
        now = datetime.datetime.now(datetime.timezone.utc)

        # Build cert claiming issuer is ca_cert, but signed with rogue key
        forged_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rogue_signer_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Forged TSA")]))
            .issuer_name(pki_setup["ca_cert"].subject)  # Claims CA is issuer
            .public_key(forged_key.public_key())
            .serial_number(7777)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
            .sign(rogue_signer_key, hashes.SHA256())  # NOT signed by ca_key!
        )

        resp = create_synthetic_timestamp_response(
            digest, forged_key, forged_cert
        )

        with pytest.raises(UntrustedTSACertificateError, match="signature verification failed"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec01_04_unauthorized_intermediate_ca_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 5: Chain rooted at unauthorized CA must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec01-unauthorized-ca").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["rogue_leaf_key"],
            pki_setup["rogue_leaf_cert"],
            additional_certs=[pki_setup["rogue_root_cert"]],
        )

        with pytest.raises(UntrustedTSACertificateError, match="not present in authorized trust anchors"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],  # Only authorized CA trusted
            )

    def test_sec01_05_intermediate_ca_missing_basic_constraints_ca_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """SEC-4C1-03: Intermediate certificate lacking basicConstraints ca=True must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec01-ca-false").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["bad_inter_leaf_key"],
            pki_setup["bad_inter_leaf_cert"],
            additional_certs=[pki_setup["bad_inter_cert"]],
        )

        with pytest.raises(UntrustedTSACertificateError, match="basicConstraints with ca=False"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec01_06_intermediate_ca_lacking_key_cert_sign_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 6: Intermediate certificate lacking keyCertSign in KeyUsage must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec01-no-key-cert-sign").hexdigest()
        now = datetime.datetime.now(datetime.timezone.utc)

        # Build intermediate CA with ca=True but keyCertSign=False
        no_kcs_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        no_kcs_sub = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "No KeyCertSign CA")])
        no_kcs_cert = (
            x509.CertificateBuilder()
            .subject_name(no_kcs_sub)
            .issuer_name(pki_setup["ca_cert"].subject)
            .public_key(no_kcs_key.public_key())
            .serial_number(8881)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=False,  # Lacks key_cert_sign!
                    crl_sign=False, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .sign(pki_setup["ca_key"], hashes.SHA256())
        )

        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        leaf_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Leaf under NoKCS CA")]))
            .issuer_name(no_kcs_sub)
            .public_key(leaf_key.public_key())
            .serial_number(8882)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
            .sign(no_kcs_key, hashes.SHA256())
        )

        resp = create_synthetic_timestamp_response(
            digest, leaf_key, leaf_cert, additional_certs=[no_kcs_cert]
        )

        with pytest.raises(UntrustedTSACertificateError, match="lacks keyCertSign in keyUsage"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec01_07_missing_or_incorrect_eku_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 6: Signer certificate lacking id-kp-timeStamping EKU must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec01-wrong-eku").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest, pki_setup["no_eku_key"], pki_setup["no_eku_cert"]
        )

        with pytest.raises(UntrustedTSACertificateError, match="missing required ExtendedKeyUsage"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec01_08_expired_or_not_yet_valid_signer_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 7: Signer certificate expired at generation time must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec01-expired").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest, pki_setup["expired_key"], pki_setup["expired_cert"]
        )

        with pytest.raises(UntrustedTSACertificateError, match="not valid at timestamp generation time"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec01_09_incorrect_tsa_policy_rejected(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 8: Unexpected policy OID must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec01-wrong-policy").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            policy_oid="1.3.6.1.4.1.99999.1.1",
        )

        with pytest.raises(TSAPolicyMismatchError, match="does not match expected policy"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
                expected_policy_oid=DEFAULT_TSA_POLICY_OID,
            )

    def test_sec01_10_authorized_signer_valid_3tier_chain_accepted(self, pki_setup: Dict[str, Any]) -> None:
        """Requirement 10: Authorized signer under valid 3-tier CA chain must be accepted."""
        digest = "sha256:" + hashlib.sha256(b"sec01-valid-3tier-chain").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["chained_tsa_key"],
            pki_setup["chained_tsa_cert"],
            additional_certs=[pki_setup["inter_ca_cert"]],
            serial_number=10001,
        )

        verified = parse_and_verify_timestamp_response(
            resp_der=resp,
            expected_imprint_digest=digest,
            trusted_certs=[pki_setup["ca_cert"]],  # Trust store contains only Root CA
        )

        assert verified.serial_number == 10001
        assert verified.signer_common_name == "Chained Synthetic TSA Leaf"
        assert verified.message_imprint_digest == digest

    def test_sec01_11_explicit_verify_tsa_certificate_trust_call(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Requirement 2: Explicit separation and direct invocation of verify_tsa_certificate_trust."""
        now = datetime.datetime.now(datetime.timezone.utc)
        chain = verify_tsa_certificate_trust(
            signer_cert=pki_setup["chained_tsa_cert"],
            trusted_certs=[pki_setup["ca_cert"]],
            candidate_intermediates=[pki_setup["inter_ca_cert"]],
            validation_time=now,
        )
        assert len(chain) == 3
        assert chain[0] == pki_setup["chained_tsa_cert"]
        assert chain[1] == pki_setup["inter_ca_cert"]
        assert chain[2] == pki_setup["ca_cert"]


# ============================================================================
# Category 7: SEC-4C1-02 Remediated SignedAttrs CMS Verification
# ============================================================================

class TestCategory7RemediatedSignedAttrsCMSSEC02:
    """
    Verifies full fail-closed resolution of SEC-4C1-02:
    Strict ASN.1 parsing of signedAttrs using attrType and attrValues, unwrapping inner OCTET STRING,
    validating contentType, enforcing duplicate attribute rejection, and correct DER SET OF encoding.
    """

    def test_sec02_01_synthetic_signed_attrs_correct_message_digest_accepted(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Valid response with signedAttrs and matching messageDigest must be accepted."""
        digest = "sha256:" + hashlib.sha256(b"sec02-signed-attrs-valid").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            serial_number=12001,
        )

        verified = parse_and_verify_timestamp_response(
            resp_der=resp,
            expected_imprint_digest=digest,
            trusted_certs=[pki_setup["primary_cert"]],
        )

        assert verified.serial_number == 12001
        assert verified.message_imprint_digest == digest

    def test_sec02_02_synthetic_signed_attrs_incorrect_message_digest_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Corrupted messageDigest attribute in signedAttrs must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec02-corrupted-md").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            corrupt_signed_attrs_digest=True,
        )

        with pytest.raises(InvalidSignatureError, match="messageDigest attribute mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_03_synthetic_signed_attrs_incorrect_content_type_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Incorrect contentType attribute in signedAttrs must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec02-wrong-content-type").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            corrupt_signed_attrs_content_type=True,
        )

        with pytest.raises(InvalidSignatureError, match="contentType attribute mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_04_synthetic_signed_attrs_missing_content_type_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """signedAttrs omitting mandatory contentType attribute must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec02-missing-ct").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            missing_signed_attrs_content_type=True,
        )

        with pytest.raises(InvalidSignatureError, match="missing required id-contentType attribute"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_05_synthetic_signed_attrs_missing_message_digest_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """signedAttrs omitting mandatory messageDigest attribute must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec02-missing-md").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            missing_signed_attrs_digest=True,
        )

        with pytest.raises(InvalidSignatureError, match="missing required id-messageDigest attribute"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_06_synthetic_signed_attrs_duplicate_attributes_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """signedAttrs containing duplicate attribute types must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec02-dup-attrs").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            duplicate_signed_attrs=True,
        )

        with pytest.raises(InvalidSignatureError, match="Duplicate signed attribute"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_07_tampered_signature_with_signed_attrs_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """Mathematically invalid signature over signedAttrs must be rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec02-tampered-sig").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            corrupt_signature=True,
        )

        with pytest.raises(InvalidSignatureError):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_08_signed_attrs_wrong_message_imprint_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """signedAttrs token with mismatched expected imprint must fail closed."""
        correct_digest = "sha256:" + hashlib.sha256(b"sec02-correct-imprint").hexdigest()
        wrong_digest = "sha256:" + hashlib.sha256(b"sec02-wrong-imprint").hexdigest()

        resp = create_synthetic_timestamp_response(
            correct_digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
        )

        with pytest.raises(RequestResponseBindingError, match="Message imprint digest mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=wrong_digest,
                trusted_certs=[pki_setup["primary_cert"]],
            )

    def test_sec02_09_signed_attrs_wrong_nonce_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """signedAttrs token with mismatched nonce must fail closed."""
        digest = "sha256:" + hashlib.sha256(b"sec02-nonce").hexdigest()
        resp = create_synthetic_timestamp_response(
            digest,
            pki_setup["primary_key"],
            pki_setup["primary_cert"],
            use_signed_attrs=True,
            nonce=12345678,
        )

        with pytest.raises(RequestResponseBindingError, match="Nonce mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=resp,
                expected_imprint_digest=digest,
                expected_nonce=87654321,
                trusted_certs=[pki_setup["primary_cert"]],
            )


# ============================================================================
# Category 8: OpenSSL 3.5.7 Independent Standards Interoperability
# ============================================================================

OPENSSL_EXE = r"C:\Program Files\Git\usr\bin\openssl.exe"


class TestCategory8OpenSSLInteroperability:
    """
    Verifies full interoperability with standards-conformant RFC 3161 timestamp responses
    generated offline by OpenSSL 3.5.7.
    """

    @pytest.fixture(scope="class")
    def openssl_tsa(self) -> Dict[str, Any]:
        """Generate offline synthetic OpenSSL TSA credentials and config."""
        bin_path = OPENSSL_EXE if os.path.exists(OPENSSL_EXE) else shutil.which("openssl")
        if not bin_path:
            pytest.skip("OpenSSL 3.5.7 executable not available in environment")

        workdir = Path(tempfile.mkdtemp())
        try:
            cnf_content = """
[ req ]
distinguished_name = req_dn
prompt = no

[ req_dn ]
CN = OpenSSL Interop TSA

[ ts_cert ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, nonRepudiation, digitalSignature
extendedKeyUsage = critical, timeStamping

[ tsa ]
default_tsa = tsa_config

[ tsa_config ]
dir = .
serial = ./serial
signer_cert = ./tsa.crt
certs = ./tsa.crt
signer_key = ./tsa.key
signer_digest = sha256
default_policy = 2.16.840.1.114412.7.1
digests = sha256, sha384, sha512
"""
            (workdir / "openssl.cnf").write_text(cnf_content, encoding="utf-8")
            (workdir / "serial").write_text("01\n", encoding="utf-8")

            # Generate RSA key and self-signed TSA cert
            subprocess.run(
                [
                    bin_path, "req", "-new", "-x509", "-nodes", "-days", "365",
                    "-keyout", "tsa.key", "-out", "tsa.crt", "-config", "openssl.cnf",
                    "-extensions", "ts_cert",
                ],
                check=True,
                capture_output=True,
                cwd=str(workdir),
            )

            cert_bytes = (workdir / "tsa.crt").read_bytes()
            cert = x509.load_pem_x509_certificate(cert_bytes)

            yield {"bin_path": bin_path, "workdir": workdir, "cert": cert}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_sec02_openssl_correctly_signed_response_accepted(
        self, openssl_tsa: Dict[str, Any]
    ) -> None:
        """OpenSSL 3.5.7 ts -reply token with signedAttrs is verified and accepted."""
        bin_path = openssl_tsa["bin_path"]
        workdir = openssl_tsa["workdir"]
        cert = openssl_tsa["cert"]

        data_bytes = b"openssl-interop-data-payload-12345"
        (workdir / "data.bin").write_bytes(data_bytes)

        subprocess.run(
            [bin_path, "ts", "-query", "-data", "data.bin", "-sha256", "-cert", "-out", "query.tsq"],
            check=True,
            capture_output=True,
            cwd=str(workdir),
        )

        subprocess.run(
            [bin_path, "ts", "-reply", "-config", "openssl.cnf", "-queryfile", "query.tsq", "-out", "reply.tsr"],
            check=True,
            capture_output=True,
            cwd=str(workdir),
        )

        reply_der = (workdir / "reply.tsr").read_bytes()
        digest = "sha256:" + hashlib.sha256(data_bytes).hexdigest()

        verified = parse_and_verify_timestamp_response(
            resp_der=reply_der,
            expected_imprint_digest=digest,
            trusted_certs=[cert],
        )

        assert verified.signer_common_name == "OpenSSL Interop TSA"
        assert verified.message_imprint_digest == digest
        assert verified.serial_number >= 1

    def test_sec02_openssl_untrusted_tsa_rejected(
        self, openssl_tsa: Dict[str, Any], pki_setup: Dict[str, Any]
    ) -> None:
        """OpenSSL token from TSA not in trust store must be rejected fail-closed."""
        workdir = openssl_tsa["workdir"]
        reply_der = (workdir / "reply.tsr").read_bytes()
        data_bytes = (workdir / "data.bin").read_bytes()
        digest = "sha256:" + hashlib.sha256(data_bytes).hexdigest()

        # Pass only pki_setup["ca_cert"], not openssl_tsa["cert"]
        with pytest.raises(UntrustedTSACertificateError, match="is not in trusted store"):
            parse_and_verify_timestamp_response(
                resp_der=reply_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec02_openssl_wrong_imprint_digest_rejected(
        self, openssl_tsa: Dict[str, Any]
    ) -> None:
        """OpenSSL token verified with mismatched message imprint must fail closed."""
        workdir = openssl_tsa["workdir"]
        reply_der = (workdir / "reply.tsr").read_bytes()
        cert = openssl_tsa["cert"]
        wrong_digest = "sha256:" + "0" * 64

        with pytest.raises(RequestResponseBindingError, match="Message imprint digest mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=reply_der,
                expected_imprint_digest=wrong_digest,
                trusted_certs=[cert],
            )

    def test_sec02_openssl_wrong_nonce_rejected(
        self, openssl_tsa: Dict[str, Any]
    ) -> None:
        """OpenSSL token verified with mismatched nonce expectation must fail closed."""
        workdir = openssl_tsa["workdir"]
        reply_der = (workdir / "reply.tsr").read_bytes()
        cert = openssl_tsa["cert"]
        data_bytes = (workdir / "data.bin").read_bytes()
        digest = "sha256:" + hashlib.sha256(data_bytes).hexdigest()

        with pytest.raises(RequestResponseBindingError, match="Nonce mismatch"):
            parse_and_verify_timestamp_response(
                resp_der=reply_der,
                expected_imprint_digest=digest,
                expected_nonce=999999999999,
                trusted_certs=[cert],
            )

    def test_sec02_openssl_tampered_signature_rejected(
        self, openssl_tsa: Dict[str, Any]
    ) -> None:
        """OpenSSL token with corrupted signature bytes must be rejected fail closed."""
        workdir = openssl_tsa["workdir"]
        reply_der = (workdir / "reply.tsr").read_bytes()
        cert = openssl_tsa["cert"]
        data_bytes = (workdir / "data.bin").read_bytes()
        digest = "sha256:" + hashlib.sha256(data_bytes).hexdigest()

        # Flip a byte near the end (in the signature field)
        tampered_reply = bytearray(reply_der)
        tampered_reply[-5] ^= 0x5A

        with pytest.raises((InvalidSignatureError, MalformedAsn1Error)):
            parse_and_verify_timestamp_response(
                resp_der=bytes(tampered_reply),
                expected_imprint_digest=digest,
                trusted_certs=[cert],
            )


class TestCategory9CriticalExtensionsSEC04:
    """
    SEC-4C1-04: Fail-closed RFC 3161 TSTInfo Critical Extension Enforcement.
    Verifies that unknown or unsupported critical extensions in TSTInfo cause deterministic
    token rejection, while non-critical unknown extensions are permitted per RFC 3161 § 2.4.2.
    """

    def test_sec04_01_no_extensions_accepted(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-03: TSTInfo with absent extensions is accepted normally."""
        digest = "sha256:" + hashlib.sha256(b"sec04-no-ext").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9001,
            tst_extensions=None,
        )
        res = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            trusted_certs=[pki_setup["ca_cert"]],
        )
        assert res.serial_number == 9001

    def test_sec04_02_unknown_noncritical_extension_accepted(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-02: Unknown non-critical extension in TSTInfo is ignored and token accepted."""
        digest = "sha256:" + hashlib.sha256(b"sec04-noncrit").hexdigest()
        # OID 1.3.6.1.4.1.99999.1 (custom non-critical extension)
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9002,
            tst_extensions=[("1.3.6.1.4.1.99999.1", False, b"noncritical-data")],
        )
        res = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            trusted_certs=[pki_setup["ca_cert"]],
        )
        assert res.serial_number == 9002

    def test_sec04_03_unknown_critical_extension_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-01: Unknown critical extension in TSTInfo causes deterministic rejection."""
        digest = "sha256:" + hashlib.sha256(b"sec04-crit").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9003,
            tst_extensions=[("1.3.6.1.4.1.99999.2", True, b"critical-policy-payload")],
        )
        with pytest.raises(
            InvalidTimestampTokenError,
            match="Unrecognized critical extension in TSTInfo: '1.3.6.1.4.1.99999.2'",
        ):
            parse_and_verify_timestamp_response(
                resp_der=resp_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_04_multiple_noncritical_extensions_accepted(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-02: Multiple distinct unknown non-critical extensions are accepted."""
        digest = "sha256:" + hashlib.sha256(b"sec04-multi-noncrit").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9004,
            tst_extensions=[
                ("1.3.6.1.4.1.99999.10", False, b"metadata-1"),
                ("1.3.6.1.4.1.99999.11", False, b"metadata-2"),
                ("1.3.6.1.4.1.99999.12", False, b"metadata-3"),
            ],
        )
        res = parse_and_verify_timestamp_response(
            resp_der=resp_der,
            expected_imprint_digest=digest,
            trusted_certs=[pki_setup["ca_cert"]],
        )
        assert res.serial_number == 9004

    def test_sec04_05_mixed_noncritical_and_critical_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-01: Token with both non-critical and critical extensions fails on critical."""
        digest = "sha256:" + hashlib.sha256(b"sec04-mixed").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9005,
            tst_extensions=[
                ("1.3.6.1.4.1.99999.20", False, b"noncritical-info"),
                ("1.3.6.1.4.1.99999.21", True, b"critical-restriction"),
            ],
        )
        with pytest.raises(
            InvalidTimestampTokenError,
            match="Unrecognized critical extension in TSTInfo: '1.3.6.1.4.1.99999.21'",
        ):
            parse_and_verify_timestamp_response(
                resp_der=resp_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_06_malformed_empty_extensions_sequence_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-04: TSTInfo with empty extensions sequence (SIZE 1..MAX violation) fails closed."""
        digest = "sha256:" + hashlib.sha256(b"sec04-empty-ext").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9006,
            tst_extensions=[],
        )
        with pytest.raises(
            InvalidTimestampTokenError,
            match="TSTInfo extensions field present but contains zero extensions",
        ):
            parse_and_verify_timestamp_response(
                resp_der=resp_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_07_malformed_extension_structure_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-04: Malformed extension ASN.1 structure inside eContent fails closed."""
        digest = "sha256:" + hashlib.sha256(b"sec04-malformed").hexdigest()
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9007,
            tst_extensions=[("1.3.6.1.4.1.99999.30", False, b"valid-ext")],
        )
        # Corrupt eContent bytes inside CMS SignedData
        corrupted = resp_der.replace(b"valid-ext", b"")
        with pytest.raises(
            (InvalidTimestampTokenError, InvalidSignatureError, MalformedAsn1Error)
        ):
            parse_and_verify_timestamp_response(
                resp_der=corrupted,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_08_duplicate_extension_identifiers_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """AC-SEC04-04: Duplicate extension identifiers in TSTInfo violate RFC 5280 and are rejected."""
        digest = "sha256:" + hashlib.sha256(b"sec04-duplicate").hexdigest()
        # Same OID repeated twice
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["primary_key"],
            signing_cert=pki_setup["primary_cert"],
            serial_number=9008,
            tst_extensions=[
                ("1.3.6.1.4.1.99999.40", False, b"first-instance"),
                ("1.3.6.1.4.1.99999.40", False, b"second-instance"),
            ],
        )
        with pytest.raises(
            InvalidTimestampTokenError,
            match="Duplicate extension identifier in TSTInfo: '1.3.6.1.4.1.99999.40'",
        ):
            parse_and_verify_timestamp_response(
                resp_der=resp_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_09_correctly_signed_token_with_critical_extension_rejected(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """
        AC-SEC04-01 & AC-SEC04-05: Cryptographically authentic, trusted token with valid CMS
        signature and valid certificate path is STILL rejected if TSTInfo has an unknown critical ext.
        """
        digest = "sha256:" + hashlib.sha256(b"sec04-authentic-crit").hexdigest()
        # Full multi-tier chain: chained_tsa -> inter_ca -> ca_cert
        resp_der = create_synthetic_timestamp_response(
            imprint_digest=digest,
            signing_key=pki_setup["chained_tsa_key"],
            signing_cert=pki_setup["chained_tsa_cert"],
            additional_certs=[pki_setup["inter_ca_cert"]],
            serial_number=9009,
            use_signed_attrs=True,
            tst_extensions=[
                ("1.3.6.1.4.1.99999.50", True, b"sovereign-audit-requirement")
            ],
        )
        with pytest.raises(
            InvalidTimestampTokenError,
            match="Unrecognized critical extension in TSTInfo: '1.3.6.1.4.1.99999.50'",
        ):
            parse_and_verify_timestamp_response(
                resp_der=resp_der,
                expected_imprint_digest=digest,
                trusted_certs=[pki_setup["ca_cert"]],
            )

    def test_sec04_10_primary_critical_ext_triggers_secondary_fallback(
        self, pki_setup: Dict[str, Any]
    ) -> None:
        """
        AC-SEC04-05: When primary TSA returns an unhandled critical extension, DualTSAClient
        catches the InvalidTimestampTokenError and falls over to secondary provider seamlessly.
        """
        digest = "sha256:" + hashlib.sha256(b"sec04-fallback").hexdigest()

        def primary_critical_ext_handler(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                imprint_digest=digest,
                signing_key=pki_setup["primary_key"],
                signing_cert=pki_setup["primary_cert"],
                serial_number=9101,
                tst_extensions=[
                    ("1.3.6.1.4.1.99999.60", True, b"unsupported-critical-ext")
                ],
            )

        def secondary_clean_handler(req_der: bytes, cfg: TSAProviderConfig) -> bytes:
            return create_synthetic_timestamp_response(
                imprint_digest=digest,
                signing_key=pki_setup["secondary_key"],
                signing_cert=pki_setup["secondary_cert"],
                serial_number=9102,
                tst_extensions=[
                    ("1.3.6.1.4.1.99999.61", False, b"harmless-noncritical-ext")
                ],
            )

        p_cfg = TSAProviderConfig(
            provider_id="primary-tsa",
            endpoint_url="http://primary.tsa.local",
            trusted_certificates=[pki_setup["primary_cert"]],
            transport_handler=primary_critical_ext_handler,
            max_retries=1,
            backoff_base_seconds=0.01,
        )
        s_cfg = TSAProviderConfig(
            provider_id="secondary-tsa",
            endpoint_url="http://secondary.tsa.local",
            trusted_certificates=[pki_setup["secondary_cert"]],
            transport_handler=secondary_clean_handler,
            max_retries=1,
            backoff_base_seconds=0.01,
        )

        engine = DualTSAFallbackEngine(p_cfg, s_cfg)
        v_ts, prov = engine.request_timestamp(digest)

        assert engine.state == TSAFallbackState.SECONDARY_ACCEPTED
        assert prov.selected_provider_role == "SECONDARY"
        assert prov.selected_provider_id == "secondary-tsa"
        assert v_ts.serial_number == 9102
        # Verify primary attempts recorded failure
        primary_attempts = [a for a in prov.attempts if a.provider_id == "primary-tsa"]
        assert len(primary_attempts) >= 1
        assert any("Unrecognized critical extension" in str(a.error_message) for a in primary_attempts)
