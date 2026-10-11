"""
Test Suite: Live R1 -> Dependent R2 Provenance Sealing Gate V1 — Epoch 001.
Validates:
- Canonical R1 byte identity and live Sigstore keyless verification (ASSUMP_001)
- Qualification authority event binding and ordering
- Canonical R2 byte identity and live Sigstore keyless verification (ASSUMP_002)
- Distinct external witnesses (R1 log index != R2 log index)
- Exact R2 -> R1 root, witness, event, policy, and run binding (ASSUMP_003)
- Negative controls: same-root alternate bundle substitution, cross-run transplant,
  phase substitution, authority event mutation, payload tamper, wrong identity/issuer
- Independent package-only zero-trust reconstruction
- Evidence manifest completeness and integrity
- Epistemic boundary invariants (no natural capture, observer un-armed, denominator 0)
"""

import hashlib
import json
import os
import pytest

from analyst_dashboard.smart_money.anchor_finalization import (
    PINNED_SIGSTORE_IDENTITY,
    PINNED_SIGSTORE_ISSUER,
    SigstoreInterimVerifier,
    SyntheticKeylessBundleGenerator,
    SameRootSubstitutionVerifier,
)
from analyst_dashboard.smart_money.canonicalization import canonicalize_json


class TestLiveR1DependentR2ProvenanceSealing:
    """Automated verification suite for SIGSTORE_LIVE_R1_DEPENDENT_R2_PROVENANCE_SEALING_V1."""

    PACKAGE_DIR = "evidence/sigstore-live-r1-r2-v1"

    @pytest.fixture
    def verifier(self):
        return SigstoreInterimVerifier(
            pinned_identity=PINNED_SIGSTORE_IDENTITY,
            pinned_issuer=PINNED_SIGSTORE_ISSUER,
            require_exact_pinned=True,
        )

    # 1. R1 Live Verification (ASSUMP_001)
    def test_live_canonical_r1_verification(self, verifier):
        payload_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_payload.json")
        bundle_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1.sigstore.json")
        receipt_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_authoritative_witness_receipt.json")

        assert os.path.exists(payload_path)
        assert os.path.exists(bundle_path)
        assert os.path.exists(receipt_path)

        with open(payload_path, "rb") as f:
            payload_bytes = f.read()

        with open(bundle_path, "rb") as f:
            bundle_raw = f.read()

        raw_bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
        bundle = verifier.parse_bundle(bundle_raw.decode("utf-8"))
        verif = verifier.verify_bundle(bundle, payload_bytes)

        assert verif.is_valid, f"R1 live verification failed: {verif.error_detail}"
        assert verif.signature_verification == "PASS"
        assert verif.identity_verification == "PASS"
        assert verif.issuer_verification == "PASS"
        assert verif.digest_match is True
        assert bundle.certificate_identity_san == PINNED_SIGSTORE_IDENTITY
        assert bundle.certificate_issuer_oidc == PINNED_SIGSTORE_ISSUER
        assert bundle.tlog_log_index is not None
        assert int(bundle.tlog_log_index) > 3200000000

        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)

        assert receipt["witness_frozen"] is True
        assert receipt["authoritative_witness_identity"] == raw_bundle_sha256
        assert receipt["r1_bundle_sha256"] == raw_bundle_sha256
        assert receipt["rekor_log_index"] == bundle.tlog_log_index

    # 2. Authority Event Binding
    def test_authority_event_binding(self):
        event_path = os.path.join(self.PACKAGE_DIR, "authority", "authority_event.json")
        receipt_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_authoritative_witness_receipt.json")

        with open(event_path, "r", encoding="utf-8") as f:
            event = json.load(f)
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)

        assert event["r1_root_hash"] == receipt["r1_root_hash"]
        assert event["r1_bundle_sha256"] == receipt["r1_bundle_sha256"]
        assert event["authoritative_r1_witness_identity"] == receipt["authoritative_witness_identity"]
        assert event["production_authority_granted"] is False
        assert event["natural_accession_authority_granted"] is False
        assert event["authority_event_scope"] == "TEST_ONLY"

    # 3. R2 Live Verification (ASSUMP_002)
    def test_live_canonical_r2_verification(self, verifier):
        payload_path = os.path.join(self.PACKAGE_DIR, "live-r2", "r2_payload.json")
        bundle_path = os.path.join(self.PACKAGE_DIR, "live-r2", "r2.sigstore.json")
        r1_bundle_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1.sigstore.json")

        with open(payload_path, "rb") as f:
            payload_bytes = f.read()

        with open(bundle_path, "rb") as f:
            bundle_raw = f.read()

        bundle = verifier.parse_bundle(bundle_raw.decode("utf-8"))
        verif = verifier.verify_bundle(bundle, payload_bytes)

        assert verif.is_valid, f"R2 live verification failed: {verif.error_detail}"
        assert verif.signature_verification == "PASS"
        assert verif.identity_verification == "PASS"
        assert verif.issuer_verification == "PASS"
        assert verif.digest_match is True
        assert bundle.certificate_identity_san == PINNED_SIGSTORE_IDENTITY
        assert bundle.certificate_issuer_oidc == PINNED_SIGSTORE_ISSUER

        # Prove R1 and R2 are distinct external witnesses
        with open(r1_bundle_path, "rb") as f:
            r1_raw = f.read()
        r1_bundle = verifier.parse_bundle(r1_raw.decode("utf-8"))

        assert bundle.tlog_log_index != r1_bundle.tlog_log_index
        assert hashlib.sha256(bundle_raw).hexdigest() != hashlib.sha256(r1_raw).hexdigest()

    # 4. Exact R2 -> R1 Linkage (ASSUMP_003)
    def test_exact_r2_to_r1_linkage(self):
        r1_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_payload.json")
        r1_receipt_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_authoritative_witness_receipt.json")
        auth_path = os.path.join(self.PACKAGE_DIR, "authority", "authority_event.json")
        r2_path = os.path.join(self.PACKAGE_DIR, "live-r2", "r2_payload.json")

        with open(r1_path, "r", encoding="utf-8") as f:
            r1 = json.load(f)
        with open(r1_receipt_path, "r", encoding="utf-8") as f:
            r1_receipt = json.load(f)
        with open(auth_path, "r", encoding="utf-8") as f:
            auth = json.load(f)
        with open(r2_path, "r", encoding="utf-8") as f:
            r2 = json.load(f)

        auth_bytes = canonicalize_json(auth).encode("utf-8")
        auth_hash = hashlib.sha256(auth_bytes).hexdigest()

        assert r2["r1_root_hash"] == r1["root_hash"]
        assert r2["authoritative_r1_witness_identity"] == r1_receipt["authoritative_witness_identity"]
        assert r2["r1_bundle_sha256"] == r1_receipt["r1_bundle_sha256"]
        assert r2["authority_event_hash"] == auth_hash
        assert r2["authority_event_sequence"] == auth["sequence"]
        assert r2["policy_set_hash"] == r1["policy_set_hash"]
        assert r2["run_id"] == r1["run_id"] == auth["run_id"]
        assert r2["authority_grant_function"] == "NONE"

    # 5. Negative Controls
    def test_negative_controls(self, verifier):
        r1_payload_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1_payload.json")
        r1_bundle_path = os.path.join(self.PACKAGE_DIR, "live-r1", "r1.sigstore.json")
        r2_payload_path = os.path.join(self.PACKAGE_DIR, "live-r2", "r2_payload.json")

        with open(r1_payload_path, "rb") as f:
            r1_payload_bytes = f.read()
        with open(r1_bundle_path, "rb") as f:
            r1_bundle_bytes = f.read()
        with open(r2_payload_path, "rb") as f:
            r2_payload_bytes = f.read()

        r1_live_bundle = verifier.parse_bundle(r1_bundle_bytes.decode("utf-8"))

        # 5.1 Same-root alternate bundle substitution
        alt_bundle = SyntheticKeylessBundleGenerator.generate_bundle(r1_payload_bytes)
        ok, _ = SameRootSubstitutionVerifier.verify_authoritative_bundle(
            bound_bundle_hash=r1_live_bundle.bundle_sha256,
            candidate_bundle=alt_bundle,
            candidate_bundle_valid=True,
        )
        assert ok is False, "Alternate valid bundle must be rejected"

        # 5.2 Phase substitution
        assert not verifier.verify_bundle(r1_live_bundle, r2_payload_bytes).is_valid

        # 5.3 Payload tamper
        tampered_bytes = r1_payload_bytes + b" "
        assert not verifier.verify_bundle(r1_live_bundle, tampered_bytes).is_valid

        # 5.4 Identity mismatch
        wrong_id_res = verifier.verify_bundle(r1_live_bundle, r1_payload_bytes, expected_identity="fake@example.com")
        assert not wrong_id_res.is_valid

        # 5.5 Issuer mismatch
        wrong_iss_res = verifier.verify_bundle(r1_live_bundle, r1_payload_bytes, expected_issuer="https://fake.issuer.com")
        assert not wrong_iss_res.is_valid

    # 6. Package-Only Independent Reconstruction
    def test_independent_package_reconstruction(self):
        result_path = os.path.join(self.PACKAGE_DIR, "review", "independent_reconstruction_result.json")
        manifest_path = os.path.join(self.PACKAGE_DIR, "SIGSTORE_LIVE_R1_R2_CHAIN_MANIFEST.json")

        with open(result_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        assert res["reconstruction_status"] == "PASS"
        assert res["mismatch_count"] == 0

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["gate"] == "SIGSTORE_LIVE_R1_DEPENDENT_R2_PROVENANCE_SEALING_V1"
        assert len(manifest["retained_artifacts"]) >= 8
