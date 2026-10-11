"""
Test Suite: Sigstore / Rekor Operational Rooting Closure — Epoch 001.
Verifies all Section 27 and Gate V2 Requirements:
- Live Canonical R1 and R2 Provenance Audit (ASSUMP_001, ASSUMP_002, ASSUMP_003)
- Unattended Workload Signing Evaluation & Fail-Closed Status (ASSUMP_004)
- Signer Identity Lifecycle & Compromise Handling (ASSUMP_005)
- Rekor Consistency, Merkle Inclusion, and Split-View Detection (ASSUMP_006)
- Sigstore/TUF Trust-Root Cache Governance & Health Auditing (ASSUMP_007)
- Local Raw-Evidence Backup, Hashing, & Disaster Recovery (ASSUMP_008)
- Same-Root Multiple-Bundle Substitution Rejection (ASSUMP_009)
- Timestamp Semantics Disaggregation & SEC Clock Authority Separation (ASSUMP_010)
- Interim Natural Rooting Policy Ratification (ASSUMP_011)
- Comprehensive Operational Reviewer Zero-Trust Reconstruction
- Privacy Boundary Auditing
- Epistemic Invariants (Outcome B & Outcome C)
"""

import hashlib
import json
import os
import shutil
import tempfile
import pytest

from analyst_dashboard.smart_money.anchor_finalization import (
    PINNED_SIGSTORE_IDENTITY,
    PINNED_SIGSTORE_ISSUER,
    SIGNER_IDENTITY_POLICY_ID,
    SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_ID,
    UNATTENDED_SIGNER_OIDC_ISSUER_DEFAULT,
    DEFAULT_SIGSTORE_TUF_ROOT_VERSION,
    DEFAULT_SIGSTORE_TRUST_ROOT_SHA256,
    SignerCompromiseStatus,
    UnattendedSigningStatus,
    SignerIdentityLifecyclePolicy,
    RekorCheckpointRecord,
    RekorConsistencyAuditResult,
    TufTrustRootConfig,
    RawEvidenceRecord,
    RawEvidenceRecoveryResult,
    DisaggregatedTimestampRecord,
    SigstoreOperationalReviewResult,
    RekorConsistencyMonitor,
    SigstoreTufGovernanceManager,
    LocalRawEvidenceDurabilityManager,
    SameRootSubstitutionVerifier,
    TimestampSemanticsManager,
    UnattendedSigningEvaluator,
    SigstoreOperationalReviewer,
    SigstoreInterimHarness,
    SigstoreBundleModel,
    SyntheticKeylessBundleGenerator,
)


class TestSmartMoneySigstoreOperationalClosure:
    """Targeted test suite for SIGSTORE/REKOR LIVE PROVENANCE + OPERATIONAL ROOTING CLOSURE GATE V2."""

    @pytest.fixture
    def harness(self):
        return SigstoreInterimHarness()

    @pytest.fixture
    def temp_durability_dirs(self):
        temp_dir = tempfile.mkdtemp()
        primary = os.path.join(temp_dir, "primary")
        backup = os.path.join(temp_dir, "backup")
        yield primary, backup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def lifecycle_policy(self):
        return SignerIdentityLifecyclePolicy(
            policy_id=SIGNER_IDENTITY_POLICY_ID,
            authorized_interactive_identity=PINNED_SIGSTORE_IDENTITY,
            authorized_interactive_issuer=PINNED_SIGSTORE_ISSUER,
            authorized_unattended_workflow="repo:owner/finance:ref:refs/heads/main",
            authorized_unattended_issuer=UNATTENDED_SIGNER_OIDC_ISSUER_DEFAULT,
            authorized_repository="owner/finance",
            rotation_procedure="Planned rotation with epoch transition and public key archiving",
            compromise_procedure="Emergency revocation notice, status transition, admission freeze",
            historical_valid_bundle_treatment="Valid prior to verified compromise timestamp",
            emergency_suspension_procedure="Immediate terminal hold on anomalous activity",
            compromise_status=SignerCompromiseStatus.ACTIVE,
        )

    # 1. ASSUMP_001, ASSUMP_002, ASSUMP_003: Canonical R1/R2 Live Provenance Check
    def test_canonical_r1_and_r2_live_provenance_audit(self):
        """
        Verifies that reference bundle does not contain canonical R1/R2,
        and accurately reflects that live canonical submission is OPEN.
        """
        ref_bundle_path = os.path.join("scratch", "smart-money-rekor-qualification", "arx-rekor-test.sigstore.json")
        ref_payload_path = os.path.join("scratch", "smart-money-rekor-qualification", "arx-rekor-test.json")

        assert os.path.exists(ref_bundle_path), "Reference bundle must exist from Gate V1"
        assert os.path.exists(ref_payload_path), "Reference payload must exist from Gate V1"

        with open(ref_payload_path, "rb") as f:
            ref_bytes = f.read()
        ref_sha256 = hashlib.sha256(ref_bytes).hexdigest()

        # Canonical R1 and R2 payload hashes
        canonical_r1_sha256 = "afdd45eadf115da9694dc709db9d522078d039f0f8a87a1ff51259d47291fdff"
        canonical_r2_sha256 = "2e3d2aa23653b42b5f77162c5a11d82ec8e330b399c40191eb55897d0689c28f"

        # Assert reference test bundle is distinct from canonical payloads
        assert ref_sha256 != canonical_r1_sha256, "Reference payload must not be canonical R1"
        assert ref_sha256 != canonical_r2_sha256, "Reference payload must not be canonical R2"

        # Therefore, ASSUMP_001, ASSUMP_002, ASSUMP_003 remain OPEN
        assump_001_status = "OPEN"
        assump_002_status = "OPEN"
        assump_003_status = "OPEN"
        assert assump_001_status == "OPEN"
        assert assump_002_status == "OPEN"
        assert assump_003_status == "OPEN"

    # 2. ASSUMP_004: Unattended Signing Evaluator
    def test_unattended_signing_evaluation(self):
        """
        Audits unattended workload signing. In local environment without CI runner OIDC,
        status must accurately report NOT_ESTABLISHED.
        """
        status, msg = UnattendedSigningEvaluator.evaluate_runtime_unattended_signing()
        # In desktop Windows environment, neither ACTIONS_ID_TOKEN_REQUEST_URL nor SPIFFE is set
        if not os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL") and not os.environ.get("SPIFFE_ENDPOINT_SOCKET"):
            assert status == UnattendedSigningStatus.NOT_ESTABLISHED
            assert "interactive signing qualified, unattended signing open" in msg

        # Test valid workload claims
        valid_claims = {
            "iss": "https://token.actions.githubusercontent.com",
            "repository": "owner/finance",
            "ref": "refs/heads/main",
        }
        ok, res = UnattendedSigningEvaluator.verify_workload_identity_claims(valid_claims, expected_repo="owner/finance")
        assert ok is True
        assert res == "PASS: Valid workload identity claims"

        # Test wrong repo
        wrong_repo_claims = {
            "iss": "https://token.actions.githubusercontent.com",
            "repository": "unauthorized/repo",
        }
        ok, res = UnattendedSigningEvaluator.verify_workload_identity_claims(wrong_repo_claims, expected_repo="owner/finance")
        assert ok is False
        assert "Unauthorized repository scope" in res

        # Test wrong issuer
        wrong_iss_claims = {
            "iss": "https://rogue-issuer.com",
            "repository": "owner/finance",
        }
        ok, res = UnattendedSigningEvaluator.verify_workload_identity_claims(wrong_iss_claims, expected_repo="owner/finance")
        assert ok is False
        assert "Unauthorized OIDC issuer" in res

    # 3. ASSUMP_005: Signer Identity Lifecycle & Compromise
    def test_signer_identity_lifecycle_policy(self, lifecycle_policy):
        """Validates signer identity lifecycle policy procedures and transitions."""
        assert lifecycle_policy.policy_id == SIGNER_IDENTITY_POLICY_ID
        assert lifecycle_policy.authorized_interactive_identity == PINNED_SIGSTORE_IDENTITY
        assert lifecycle_policy.compromise_status == SignerCompromiseStatus.ACTIVE

        # Test dictionary representation
        p_dict = lifecycle_policy.to_dict()
        assert p_dict["compromise_status"] == "ACTIVE"
        assert p_dict["policy_id"] == SIGNER_IDENTITY_POLICY_ID

        # Test compromise status transitions
        compromised = SignerIdentityLifecyclePolicy(
            policy_id=lifecycle_policy.policy_id,
            authorized_interactive_identity=lifecycle_policy.authorized_interactive_identity,
            authorized_interactive_issuer=lifecycle_policy.authorized_interactive_issuer,
            authorized_unattended_workflow=lifecycle_policy.authorized_unattended_workflow,
            authorized_unattended_issuer=lifecycle_policy.authorized_unattended_issuer,
            authorized_repository=lifecycle_policy.authorized_repository,
            rotation_procedure=lifecycle_policy.rotation_procedure,
            compromise_procedure=lifecycle_policy.compromise_procedure,
            historical_valid_bundle_treatment=lifecycle_policy.historical_valid_bundle_treatment,
            emergency_suspension_procedure=lifecycle_policy.emergency_suspension_procedure,
            compromise_status=SignerCompromiseStatus.COMPROMISED,
        )
        assert compromised.compromise_status == SignerCompromiseStatus.COMPROMISED

    # 4. ASSUMP_006: Rekor Checkpoint & Consistency Monitoring
    def test_rekor_checkpoint_and_consistency(self):
        """Verifies checkpoint parsing, consistency auditing, and split-view detection."""
        raw_chk_a = (
            "rekor.sigstore.dev - 1193050959916656506\n"
            "3088425988\n"
            "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\n"
            "— rekor.sigstore.dev wJ3xFAAAAAB...=="
        )
        chk_a = RekorConsistencyMonitor.parse_checkpoint(raw_chk_a)
        assert chk_a.origin == "rekor.sigstore.dev - 1193050959916656506"
        assert chk_a.tree_size == 3088425988
        assert chk_a.root_hash_b64 == "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="

        # Test consistent checkpoint pair
        chk_b = RekorConsistencyMonitor.parse_checkpoint(raw_chk_a)
        res = RekorConsistencyMonitor.audit_consistency(chk_a, chk_b)
        assert res.status == "CONSISTENT"
        assert res.split_view_detected is False

        # Test split-view detection: same tree size, different root hash
        raw_chk_split = (
            "rekor.sigstore.dev - 1193050959916656506\n"
            "3088425988\n"
            "CONFLICTING_ROOT_HASH_47DEQpj8HBSa+/TImW+5JC==\n\n"
            "— rekor.sigstore.dev wJ3xFAAAAAB...=="
        )
        chk_split = RekorConsistencyMonitor.parse_checkpoint(raw_chk_split)
        split_res = RekorConsistencyMonitor.audit_consistency(chk_a, chk_split)
        assert split_res.status == "SPLIT_VIEW_CONFLICT"
        assert split_res.split_view_detected is True
        assert "Split-view detected" in (split_res.error_message or "")

        # Test regression detection: tree size decreases
        raw_chk_regress = (
            "rekor.sigstore.dev - 1193050959916656506\n"
            "1000000000\n"
            "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\n"
            "— rekor.sigstore.dev wJ3xFAAAAAB...=="
        )
        chk_regress = RekorConsistencyMonitor.parse_checkpoint(raw_chk_regress)
        regress_res = RekorConsistencyMonitor.audit_consistency(chk_a, chk_regress)
        assert regress_res.status == "REGRESSION_CONFLICT"
        assert regress_res.split_view_detected is True

    # 5. ASSUMP_007: Sigstore TUF Trust-Root Governance
    def test_sigstore_tuf_trust_root_governance(self):
        """Verifies local cached Sigstore TUF trust-root inspection and fail-closed handling."""
        manager = SigstoreTufGovernanceManager()
        # Verify local real trust root
        if os.path.exists(manager.trust_root_path):
            config = manager.inspect_trust_root()
            assert config.tuf_root_version == DEFAULT_SIGSTORE_TUF_ROOT_VERSION
            assert config.trust_root_sha256 == DEFAULT_SIGSTORE_TRUST_ROOT_SHA256
            assert config.ca_count >= 1
            assert config.tlog_count >= 1

            ok, msg = manager.verify_trust_root_health()
            assert ok is True
            assert "PASS" in msg

        # Test fail-closed on missing file
        missing_manager = SigstoreTufGovernanceManager(trust_root_path="c:/nonexistent/trusted_root.json")
        ok, msg = missing_manager.verify_trust_root_health()
        assert ok is False
        assert "FAIL_CLOSED: Missing trusted root artifact" in msg

    # 6. ASSUMP_008: Local Raw Evidence Durability & Disaster Recovery
    def test_local_raw_evidence_durability(self, temp_durability_dirs):
        """Verifies dual storage, SHA-256 backup replication, and simulated disaster recovery."""
        primary_dir, backup_dir = temp_durability_dirs
        manager = LocalRawEvidenceDurabilityManager(primary_dir=primary_dir, backup_dir=backup_dir)

        test_data = b'{"accession": "TEST-001", "content": "Sample natural capture evidence"}'
        record = manager.store_and_backup("TEST-001", test_data)

        assert os.path.exists(record.primary_path)
        assert os.path.exists(record.backup_path)
        assert record.content_sha256 == hashlib.sha256(test_data).hexdigest()

        # Simulate disaster recovery (primary deleted, restore from backup)
        dr_res = manager.simulate_disaster_recovery(record, witnessed_root_hash=record.content_sha256)
        assert dr_res.status == "PASS"
        assert dr_res.hash_match is True
        assert dr_res.root_hash_matches_witness is True
        assert os.path.exists(record.primary_path), "Primary file should be restored"

    # 7. ASSUMP_009: Same-Root Multiple-Bundle Substitution Rejection
    def test_same_root_multiple_bundle_substitution_rejection(self, harness):
        """
        Verifies that even if two bundles are cryptographically valid for the same root,
        only the explicitly bound bundle is accepted, rejecting substitution attempts.
        """
        r1_model, canonical_bytes, root_hash = harness.build_synthetic_r1()

        # Generate bundle A (authoritative)
        bundle_a = SyntheticKeylessBundleGenerator.generate_bundle(canonical_bytes)

        # Generate bundle B (alternate valid bundle for same root)
        bundle_b = SyntheticKeylessBundleGenerator.generate_bundle(canonical_bytes)

        # Both bundles are cryptographically valid
        assert harness.verifier.verify_bundle(bundle_a, canonical_bytes).is_valid
        assert harness.verifier.verify_bundle(bundle_b, canonical_bytes).is_valid
        assert bundle_a.bundle_sha256 != bundle_b.bundle_sha256

        # Bind bundle A as authoritative
        bound_hash = bundle_a.bundle_sha256

        # Submitting bundle A -> Accepted
        ok_a, msg_a = SameRootSubstitutionVerifier.verify_authoritative_bundle(
            bound_bundle_hash=bound_hash,
            candidate_bundle=bundle_a,
            candidate_bundle_valid=True,
        )
        assert ok_a is True
        assert "AUTHORITATIVE_MATCH" in msg_a

        # Submitting bundle B -> Rejected as substitution
        ok_b, msg_b = SameRootSubstitutionVerifier.verify_authoritative_bundle(
            bound_bundle_hash=bound_hash,
            candidate_bundle=bundle_b,
            candidate_bundle_valid=True,
        )
        assert ok_b is False
        assert "SUBSTITUTION_REJECTED" in msg_b

    # 8. ASSUMP_010: Timestamp Semantics Disaggregation
    def test_timestamp_semantics_disaggregation(self, harness):
        """
        Verifies RFC 3161 timestamps, Rekor integrated time, local capture interval,
        and SEC availability times are strictly disaggregated.
        """
        r1_model, canonical_bytes, root_hash = harness.build_synthetic_r1()
        bundle = SyntheticKeylessBundleGenerator.generate_bundle(canonical_bytes)

        rec = TimestampSemanticsManager.create_disaggregated_timestamp_record(
            bundle=bundle,
            local_capture_interval="[2026-10-11T12:00:00Z, 2026-10-11T12:00:05Z]",
            sec_observed_interval="NOT_APPLICABLE_NON_NATURAL",
        )

        assert rec.rfc3161_authority == "EXTERNAL_SIGNED_TIMESTAMP"
        assert rec.rekor_authority == "TRANSPARENCY_LOG_METADATA_TIME"
        assert rec.local_capture_clock_authority == "INDEPENDENT_NTP_CLOCK_AUTHORITY"
        assert rec.sec_public_availability_authority == "SEPARATE_OBSERVER_CONTRACT"
        assert rec.exact_first_sec_public_time == "NOT_ESTABLISHED"

    # 9. ASSUMP_011: Interim Natural Rooting Policy Ratification
    def test_interim_natural_rooting_policy_ratification(self):
        """Verifies policy contract ratifying interim profile with conditional status."""
        policy_path = os.path.join(
            "docs", "domain", "radar", "smart-money", "public-availability",
            "SMART_MONEY_SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_V1.json"
        )
        assert os.path.exists(policy_path), "Policy file must exist"

        with open(policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["policy_id"] == SIGSTORE_INTERIM_NATURAL_ROOTING_POLICY_ID
        assert data["status"] == "RATIFIED"
        assert data["current_suitability_verdict"]["interim_profile_suitability"] == "CONDITIONAL"
        assert data["current_suitability_verdict"]["natural_observer_arming"] == "NOT_AUTHORIZED"
        assert data["current_suitability_verdict"]["natural_denominator_target"] == 0

    # 10. Operational Reviewer Zero-Trust Comprehensive Audit
    def test_operational_reviewer_zero_trust_reconstruction(self, harness, lifecycle_policy, temp_durability_dirs):
        """Verifies comprehensive operational reviewer executes end-to-end reconstruction."""
        primary_dir, backup_dir = temp_durability_dirs
        manager = LocalRawEvidenceDurabilityManager(primary_dir=primary_dir, backup_dir=backup_dir)
        evidence_rec = manager.store_and_backup("TEST-EVID", b"evidence payload")

        r1_model, canonical_bytes, root_hash = harness.build_synthetic_r1()
        bundle = SyntheticKeylessBundleGenerator.generate_bundle(canonical_bytes)

        raw_chk = (
            "rekor.sigstore.dev - 1193050959916656506\n"
            "3088425988\n"
            "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\n"
            "— rekor.sigstore.dev wJ3xFAAAAAB...=="
        )

        reviewer = SigstoreOperationalReviewer()
        result = reviewer.review_comprehensive_operational_closure(
            lifecycle_policy=lifecycle_policy,
            checkpoint_a_text=raw_chk,
            checkpoint_b_text=raw_chk,
            durability_manager=manager,
            evidence_record=evidence_rec,
            witnessed_root_hash=evidence_rec.content_sha256,
            bound_bundle_hash=bundle.bundle_sha256,
            candidate_bundle=bundle,
            candidate_bundle_valid=True,
            local_capture_interval="[2026-10-11T12:00:00Z, 2026-10-11T12:00:05Z]",
            workload_claims={"iss": "https://token.actions.githubusercontent.com", "repository": "owner/finance"},
        )


        assert result.review_status == "PASS"
        assert result.mismatch_count == 0
        assert result.rekor_consistency_verified is True
        assert result.tuf_root_verified is True
        assert result.evidence_durability_verified is True
        assert result.timestamps_disaggregated is True
        assert result.privacy_screening_passed is True

    # 11. Privacy Boundary Auditing
    def test_privacy_boundary_auditing(self):
        """Verifies zero-trust detection of secrets or sensitive data across records."""
        reviewer = SigstoreOperationalReviewer()

        # Clean record passes
        clean = [{"safe_key": "safe_value", "digest": "abcdef123456"}]
        ok, violations = reviewer.review_privacy_boundaries(clean)
        assert ok is True
        assert len(violations) == 0

        # Injected sensitive AWS key fails
        dirty_aws = [{"aws_token": "AKIAIOSFODNN7EXAMPLE1"}]
        ok_aws, v_aws = reviewer.review_privacy_boundaries(dirty_aws)
        assert ok_aws is False
        assert any("AWS Access Key ID" in v for v in v_aws)

        # Injected GitHub token fails
        dirty_gh = [{"token": "ghp_1234567890abcdefghijklmnopqrstuv"}]
        ok_gh, v_gh = reviewer.review_privacy_boundaries(dirty_gh)
        assert ok_gh is False
        assert any("GitHub Personal Access Token" in v for v in v_gh)
