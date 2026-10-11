"""
Test Suite: Sigstore / Rekor Interim Root Witness Qualification — Epoch 001.
Covers Section 24 Qualification Gate Requirements:
- Exact Certificate Identity Pinning
- Exact OIDC Issuer Pinning
- R1 Construction, Witnessing, and Tamper Detection
- Wrong-Identity Negative Control
- Test-Only Authority Transition Event Binding
- R2 Construction, Witnessing, and Linkage
- R2 -> R1 Linkage Verification
- Cross-Run Substitution Rejection
- Cross-Phase Substitution Rejection
- Offline Bundle Verification
- Transparency Evidence Inspection (Tlog, SCT, RFC3161)
- Sigstore Outage Fail-Closed Invariants
- Duplicate Signing Semantics & Primary Acceptance Rule
- Independent Reviewer Zero-Trust Reconstruction
- Privacy Boundary Audit
- Live Witness Bundle Verification (if present)
- Epistemic Custody Boundary Invariants
"""

import copy
import hashlib
import json
import os
import pytest

from analyst_dashboard.smart_money.anchor_finalization import (
    PINNED_SIGSTORE_IDENTITY,
    PINNED_SIGSTORE_ISSUER,
    IDENTITY_POLICY,
    EXTERNAL_ROOT_PROFILE,
    EXTERNAL_ROOT_FUNCTION,
    SIGSTORE_TIMESTAMP_SEMANTICS,
    SigstoreBundleModel,
    SigstoreVerificationResult,
    SigstoreInterimVerifier,
    SyntheticKeylessBundleGenerator,
    SigstoreInterimHarness,
    SigstoreInterimReviewer,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_models import (
    EXTERNAL_RAW_EVIDENCE_CUSTODY,
    S3_OBJECT_LOCK_COMPLIANCE,
    AWS_EXTERNAL_CUSTODY,
    SEC_PUBLIC_AVAILABILITY_TIME,
    EXACT_FIRST_SEC_PUBLIC_TIME,
    SMART_MONEY_MARKET_WIDE_AUTHORITY,
    EMPIRICAL_SCANNER_QUALITY,
    MODEL_TUNING,
)


class TestSmartMoneySigstoreQualification:
    """Targeted qualification test suite for SIGSTORE_REKOR_INTERIM_R1_R2_QUALIFICATION_V1."""

    @pytest.fixture
    def harness(self):
        return SigstoreInterimHarness()

    @pytest.fixture
    def verifier(self):
        return SigstoreInterimVerifier()

    @pytest.fixture
    def reviewer(self):
        return SigstoreInterimReviewer()

    def test_epistemic_boundary_invariants(self):
        """Verifies hard epistemic boundaries and trust profile definitions (Section 2, 22)."""
        assert EXTERNAL_ROOT_PROFILE == "SIGSTORE_REKOR_INTERIM_V1"
        assert EXTERNAL_ROOT_FUNCTION == "PUBLIC_APPEND_ONLY_TRANSPARENCY_WITNESS"
        assert EXTERNAL_RAW_EVIDENCE_CUSTODY == "NOT_ESTABLISHED"
        assert S3_OBJECT_LOCK_COMPLIANCE == "NOT_APPLICABLE"
        assert AWS_EXTERNAL_CUSTODY == "DEFERRED"
        assert SIGSTORE_TIMESTAMP_SEMANTICS == "SIGNED_EXTERNAL_TIMESTAMP_EVIDENCE"
        assert SEC_PUBLIC_AVAILABILITY_TIME == "SEPARATE_AUTHORITY"
        assert EXACT_FIRST_SEC_PUBLIC_TIME == "NOT_ESTABLISHED"
        assert SMART_MONEY_MARKET_WIDE_AUTHORITY == "NOT_ESTABLISHED"
        assert EMPIRICAL_SCANNER_QUALITY == "INSUFFICIENT_EVIDENCE"
        assert MODEL_TUNING == "FROZEN"
        assert IDENTITY_POLICY == "EXACT_PINNED"

    def test_exact_certificate_identity_pinning(self, verifier):
        """Section 5: Verifies exact certificate identity matching policy without wildcards."""
        payload = b'{"qualification":"identity_check"}'
        bundle = SyntheticKeylessBundleGenerator.generate_bundle(
            payload,
            identity_san=PINNED_SIGSTORE_IDENTITY,
            issuer_oidc=PINNED_SIGSTORE_ISSUER,
        )

        # Exact pinned identity passes
        res_exact = verifier.verify_bundle(bundle, payload, expected_identity=PINNED_SIGSTORE_IDENTITY)
        assert res_exact.is_valid is True
        assert res_exact.identity_verification == "PASS"

        # Wildcard / prefix / suffix attempts fail
        res_prefix = verifier.verify_bundle(bundle, payload, expected_identity="daakara@users.noreply.github.com")
        assert res_prefix.is_valid is False
        assert res_prefix.identity_verification == "FAIL"

        res_wildcard = verifier.verify_bundle(bundle, payload, expected_identity="*@users.noreply.github.com")
        assert res_wildcard.is_valid is False
        assert res_wildcard.identity_verification == "FAIL"

    def test_exact_oidc_issuer_pinning(self, verifier):
        """Section 5: Verifies exact OIDC issuer matching policy without wildcards."""
        payload = b'{"qualification":"issuer_check"}'
        bundle = SyntheticKeylessBundleGenerator.generate_bundle(
            payload,
            identity_san=PINNED_SIGSTORE_IDENTITY,
            issuer_oidc=PINNED_SIGSTORE_ISSUER,
        )

        res_exact = verifier.verify_bundle(bundle, payload, expected_issuer=PINNED_SIGSTORE_ISSUER)
        assert res_exact.is_valid is True
        assert res_exact.issuer_verification == "PASS"

        # Unauthorized issuers fail
        res_other = verifier.verify_bundle(bundle, payload, expected_issuer="https://accounts.google.com")
        assert res_other.is_valid is False
        assert res_other.issuer_verification == "FAIL"

        res_truncated = verifier.verify_bundle(bundle, payload, expected_issuer="https://github.com")
        assert res_truncated.is_valid is False
        assert res_truncated.issuer_verification == "FAIL"

    def test_r1_canonical_payload_construction_and_witnessing(self, harness):
        """Sections 6, 7: Synthetic R1 construction, canonicalization, and Sigstore witnessing."""
        r1, r1_bytes, r1_root_hash = harness.build_synthetic_r1()

        assert r1.root_phase == "PRE_AUTHORITY_ADMISSION_ROOT"
        assert r1.qualification_scope == "TEST_ONLY"
        assert r1.natural_accession is False
        assert r1.authority_effective_status == "BLOCKED_PENDING_EXTERNAL_WITNESS"
        assert r1.root_hash == r1_root_hash

        bundle = harness.witness_r1(r1_bytes)
        assert bundle.certificate_identity_san == PINNED_SIGSTORE_IDENTITY
        assert bundle.certificate_issuer_oidc == PINNED_SIGSTORE_ISSUER

        verif = harness.verifier.verify_bundle(bundle, r1_bytes)
        assert verif.is_valid is True
        assert verif.signature_verification == "PASS"
        assert verif.identity_verification == "PASS"
        assert verif.issuer_verification == "PASS"
        assert verif.digest_match is True
        assert verif.tamper_detected is False

    def test_r1_tamper_detection_negative_control(self, harness):
        """Section 8: Tamper negative control fails verification as expected."""
        r1, r1_bytes, _ = harness.build_synthetic_r1()
        bundle = harness.witness_r1(r1_bytes)

        tamper_pass, result = harness.run_r1_tamper_control(r1_bytes, bundle)
        assert tamper_pass is True
        assert result.is_valid is False
        assert result.tamper_detected is True
        assert result.signature_verification == "FAIL" or result.digest_match is False

    def test_r1_wrong_identity_negative_control(self, harness):
        """Section 9: Wrong identity negative control fails verification as expected."""
        r1, r1_bytes, _ = harness.build_synthetic_r1()
        bundle = harness.witness_r1(r1_bytes)

        wrong_id_pass, result = harness.run_r1_wrong_identity_control(r1_bytes, bundle)
        assert wrong_id_pass is True
        assert result.is_valid is False
        assert result.identity_verification == "FAIL"

    def test_authority_transition_event_binding(self, harness):
        """Section 10: Test-only transition event binds R1 root and bundle hashes."""
        r1, r1_bytes, r1_root = harness.build_synthetic_r1()
        bundle = harness.witness_r1(r1_bytes)

        event = harness.build_test_authority_event(r1_root, bundle.bundle_sha256, sequence=1)
        assert event.authority_event_scope == "TEST_ONLY"
        assert event.production_authority_granted is False
        assert event.natural_accession_authority_granted is False
        assert event.r1_root_hash == r1_root
        assert event.r1_bundle_hash == bundle.bundle_sha256

        # Check deterministic hash
        expected_content = f"{event.parent_run_id}:{event.sequence}:{r1_root}:{bundle.bundle_sha256}:{event.policy_set_hash}"
        assert event.transition_event_hash == hashlib.sha256(expected_content.encode("utf-8")).hexdigest()

    def test_r2_canonical_payload_construction_and_witnessing(self, harness):
        """Sections 11, 12: Synthetic R2 construction, canonicalization, and Sigstore witnessing."""
        r1, r1_bytes, r1_root = harness.build_synthetic_r1()
        r1_bundle = harness.witness_r1(r1_bytes)
        event = harness.build_test_authority_event(r1_root, r1_bundle.bundle_sha256)

        r2, r2_bytes, r2_root = harness.build_synthetic_r2("run_synth_001", r1_root, r1_bundle.bundle_sha256, event)
        assert r2.root_phase == "POST_AUTHORITY_CLOSURE_ROOT"
        assert r2.qualification_scope == "TEST_ONLY"
        assert r2.authority_grant_function == "NONE"
        assert r2.r1_root_hash == r1_root
        assert r2.r1_bundle_hash == r1_bundle.bundle_sha256
        assert r2.authority_event_hash == event.transition_event_hash

        r2_bundle = harness.witness_r2(r2_bytes)
        verif = harness.verifier.verify_bundle(r2_bundle, r2_bytes)
        assert verif.is_valid is True
        assert verif.signature_verification == "PASS"
        assert verif.identity_verification == "PASS"

    def test_r2_to_r1_linkage_verification(self, harness):
        """Section 13: R2 -> R1 linkage verification."""
        r1, r1_bytes, r1_root = harness.build_synthetic_r1()
        r1_bundle = harness.witness_r1(r1_bytes)
        event = harness.build_test_authority_event(r1_root, r1_bundle.bundle_sha256)
        r2, r2_bytes, _ = harness.build_synthetic_r2("run_synth_001", r1_root, r1_bundle.bundle_sha256, event)

        link_pass, mismatches = harness.verify_r2_r1_linkage(r2, r1, r1_bundle, event)
        assert link_pass is True
        assert len(mismatches) == 0

        # Tampered linkage negative test
        bad_r2 = copy.copy(r2)
        object.__setattr__(bad_r2, "r1_root_hash", "0" * 64)
        bad_pass, bad_mismatches = harness.verify_r2_r1_linkage(bad_r2, r1, r1_bundle, event)
        assert bad_pass is False
        assert len(bad_mismatches) > 0

    def test_cross_run_substitution_rejection(self, harness):
        """Section 14: Cross-run substitution rejection."""
        r1, r1_bytes, r1_root = harness.build_synthetic_r1(run_id="run_A")
        r1_bundle = harness.witness_r1(r1_bytes)
        event = harness.build_test_authority_event(r1_root, r1_bundle.bundle_sha256, parent_run_id="run_A")
        r2, r2_bytes, _ = harness.build_synthetic_r2("run_A", r1_root, r1_bundle.bundle_sha256, event)
        r2_bundle = harness.witness_r2(r2_bytes)

        sig_rej, link_rej = harness.run_cross_run_substitution_test(
            run_a_r1_bytes=r1_bytes,
            run_a_r1_bundle=r1_bundle,
            run_a_r2_bytes=r2_bytes,
            run_a_r2_bundle=r2_bundle,
            run_a_r2=r2,
        )
        assert sig_rej is True
        assert link_rej is True

    def test_cross_phase_substitution_rejection(self, harness):
        """Section 15: Cross-phase substitution rejection."""
        r1, r1_bytes, r1_root = harness.build_synthetic_r1()
        r1_bundle = harness.witness_r1(r1_bytes)
        event = harness.build_test_authority_event(r1_root, r1_bundle.bundle_sha256)
        r2, r2_bytes, _ = harness.build_synthetic_r2("run_synth_001", r1_root, r1_bundle.bundle_sha256, event)
        r2_bundle = harness.witness_r2(r2_bytes)

        phase_sig_rej, phase_root_rej = harness.run_cross_phase_substitution_test(
            r1_bytes=r1_bytes,
            r1_bundle=r1_bundle,
            r2_bytes=r2_bytes,
            r2_bundle=r2_bundle,
            r1_root_hash=r1_root,
        )
        assert phase_sig_rej is True
        assert phase_root_rej is True

    def test_offline_bundle_verification(self, harness):
        """Section 16: Offline verification evaluation."""
        r1, r1_bytes, _ = harness.build_synthetic_r1()
        bundle = harness.witness_r1(r1_bytes)

        offline_pass, note = harness.verifier.verify_offline(bundle, r1_bytes)
        assert offline_pass is True
        assert "PASS" in note
        assert "Conditional" in note

    def test_transparency_evidence_inspection(self, harness):
        """Section 17: Transparency log and RFC 3161 timestamp inspection."""
        payload = b'{"test":"transparency_evidence"}'
        bundle = SyntheticKeylessBundleGenerator.generate_bundle(payload)

        assert bundle.tlog_entry_count >= 1
        assert bundle.rfc3161_timestamp_count >= 1
        assert bundle.sct_present is True
        assert bundle.media_type == "application/vnd.dev.sigstore.bundle.v0.3+json"
        assert bundle.tlog_log_index is not None
        assert bundle.tlog_integrated_time is not None

    def test_sigstore_outage_fail_closed(self, harness):
        """Section 18: Sigstore outage / fail-closed handling."""
        outage = harness.simulate_outage_fail_closed()
        assert outage["SIGSTORE_SERVICE_AVAILABLE"] is False
        assert outage["NEW_ROOT_SIGNING_WHEN_EXTERNAL_WITNESS_UNAVAILABLE"] == "BLOCKED"
        assert outage["LOCAL_UNSIGNED_ROOT_AUTHORITY"] == "NOT_GRANTED"
        assert outage["NATURAL_AUTHORITY_FALLBACK"] == "PROHIBITED"
        assert outage["SIGSTORE_OUTAGE_FAIL_CLOSED"] == "PASS"

    def test_duplicate_signing_behavior_and_primary_acceptance(self, harness):
        """Section 19: Duplicate signing behavior and deterministic primary acceptance."""
        r1, r1_bytes, _ = harness.build_synthetic_r1()
        dup = harness.run_duplicate_signing_test(r1_bytes)

        assert dup["IDENTICAL_PAYLOAD_ROOT_HASH_STABLE"] is True
        assert dup["BUNDLES_DIFFER"] is True
        assert dup["EPHEMERAL_KEYS_DIFFER"] is True
        assert dup["MULTIPLE_SIGSTORE_BUNDLES_FOR_SAME_ROOT"] == "ALLOWED"
        assert dup["PRIMARY_ACCEPTANCE_RULE"] == "DETERMINISTIC_FIRST_COMMITTED_IN_AUTHORITY_TRANSITION_RECORD"

    def test_independent_reviewer_reconstruction_pass(self, harness, reviewer):
        """Section 20: Independent reviewer zero-trust reconstruction."""
        full_res = harness.execute_full_qualification()

        review = reviewer.review_qualification(
            r1_payload_dict=full_res["r1_payload"].to_dict(),
            r1_bundle_dict=full_res["r1_bundle"].raw_bundle_dict,
            authority_event_dict=full_res["authority_event"].to_dict(),
            r2_payload_dict=full_res["r2_payload"].to_dict(),
            r2_bundle_dict=full_res["r2_bundle"].raw_bundle_dict,
            frozen_policy_set_hash="b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        )

        assert review.status == "PASS"
        assert review.reconstruction_mismatches == 0
        assert review.r1_signature_valid is True
        assert review.r1_identity_valid is True
        assert review.r2_signature_valid is True
        assert review.r2_identity_valid is True
        assert review.r2_r1_linkage_valid is True
        assert review.policy_set_hash_valid is True
        assert review.privacy_review_passed is True
        assert len(review.error_messages) == 0

    def test_independent_reviewer_catches_tampered_authority_event(self, harness, reviewer):
        """Section 20: Independent reviewer catches tampered event hash or scope."""
        full_res = harness.execute_full_qualification()
        tampered_event = copy.deepcopy(full_res["authority_event"].to_dict())
        tampered_event["production_authority_granted"] = True  # Forbidden violation

        review = reviewer.review_qualification(
            r1_payload_dict=full_res["r1_payload"].to_dict(),
            r1_bundle_dict=full_res["r1_bundle"].raw_bundle_dict,
            authority_event_dict=tampered_event,
            r2_payload_dict=full_res["r2_payload"].to_dict(),
            r2_bundle_dict=full_res["r2_bundle"].raw_bundle_dict,
            frozen_policy_set_hash="b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        )

        assert review.status == "FAIL"
        assert review.reconstruction_mismatches >= 1

    def test_privacy_boundary_review(self, verifier):
        """Section 21: Privacy boundary review filters secrets and confidential data."""
        clean_dict = {
            "root_phase": "PRE_AUTHORITY_ADMISSION_ROOT",
            "root_hash": "a" * 64,
            "qualification_scope": "TEST_ONLY",
        }
        passed, violations = verifier.check_privacy_boundary(clean_dict)
        assert passed is True
        assert len(violations) == 0

        # Inject fake secret
        dirty_dict = copy.deepcopy(clean_dict)
        dirty_dict["aws_key"] = "AKIA" + "1234567890ABCDEF"
        dirty_pass, dirty_viol = verifier.check_privacy_boundary(dirty_dict)
        assert dirty_pass is False
        assert len(dirty_viol) == 1

    def test_live_bundle_verification_if_present(self, verifier):
        """Verifies the live signed test bundle if present in the scratch directory."""
        live_bundle_path = "scratch/smart-money-rekor-qualification/arx-rekor-test.sigstore.json"
        live_payload_path = "scratch/smart-money-rekor-qualification/arx-rekor-test.json"
        tampered_path = "scratch/smart-money-rekor-qualification/arx-rekor-test-tampered.json"

        if os.path.exists(live_bundle_path) and os.path.exists(live_payload_path):
            with open(live_bundle_path, "r", encoding="utf-8") as f:
                bundle = verifier.parse_bundle(f.read())
            with open(live_payload_path, "rb") as f:
                payload = f.read()

            res = verifier.verify_bundle(bundle, payload)
            assert res.is_valid is True
            assert res.signature_verification == "PASS"
            assert res.identity_verification == "PASS"
            assert res.issuer_verification == "PASS"
            assert res.digest_match is True
            assert res.tlog_present is True
            assert res.rfc3161_present is True
            assert res.sct_present is True

            if os.path.exists(tampered_path):
                with open(tampered_path, "rb") as f:
                    tampered = f.read()
                tamper_res = verifier.verify_bundle(bundle, tampered)
                assert tamper_res.is_valid is False
                assert tamper_res.tamper_detected is True
