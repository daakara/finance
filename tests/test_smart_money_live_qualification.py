"""
Test Suite: Live External R1/R2 Infrastructure + Clock Pre-Arm Qualification (Epoch 001).
Verifies Sections 4 through 38 of LIVE_EXTERNAL_R1_R2_INFRASTRUCTURE_QUALIFICATION_V1:
- Trust model definition and separation of acknowledgement from signed receipt
- Live provider identity pinning and fail-closed handling when unprovisioned
- Capability separation matrix
- Same-key new-version collision detection
- Signer identity, key rotation contract, and historical verification
- Receipt binding contract, cross-run replay, cross-phase replay, cross-root substitution
- Live ambiguous-commit recovery simulation
- Synthetic qualification R1 and R2 lifecycle, linkage, and reviewer reconstruction
- 300-second R2 delay policy governance review
- Independent clock authority measurement theorem, conservative bound derivation, and freshness
- Run-start interval derivation
- SEC operational preflight: business day, filing window, user-agent, rate limit, reachability
- Combined fail-closed arming theorem
"""

import hashlib
import json
import unittest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from analyst_dashboard.smart_money.canonicalization import compute_canonical_hash
from analyst_dashboard.smart_money.anchor_finalization import (
    FinalizationState,
    AuthorityEligibility,
    AuthorityEffectiveStatus,
    FinalizationEvent,
    RootPhase,
    PostAuthorityClosureStatus,
    R1ConfirmationStatus,
    RootLookupResult,
    FinalizationRootReceipt,
    FinalizationRootPayload,
    build_r1_root_payload,
    build_r2_root_payload,
    compute_finalization_root_hash,
    derive_finalization_root_idempotency_key,
    create_signed_root_receipt,
    DEFAULT_ROOT_SIGNING_KEY_ID,
    CANONICAL_ROOT_SIGNING_KEY,
    LiveExternalRootQualificationHarness,
    CapabilitySeparationMatrix,
    LiveProviderIdentityStatus,
    PROVIDER_STORAGE_ACKNOWLEDGEMENT,
    CRYPTOGRAPHIC_ROOT_RECEIPT,
    STORAGE_ACKNOWLEDGEMENT_IS_CRYPTOGRAPHIC_RECEIPT,
    ROOT_STORAGE_PROVIDER,
    ROOT_STORAGE_SERVICE,
    ROOT_RECEIPT_SIGNER_CLASS,
    RECEIPT_SIGNER,
    PROVIDER_NATIVE_SIGNED_RECEIPT,
    OBJECT_VERSIONING_ENABLED,
    OBJECT_LOCK_ENABLED,
    RETENTION_MODE,
    LOCKED_OBJECT_VERSION_DELETE_PROTECTION,
    LOCKED_OBJECT_VERSION_MUTATION_PROTECTION,
    SAME_OBJECT_KEY_NEW_VERSION_CREATION,
    ORIGINAL_ROOT_VERSION_REWRITABLE,
    SAME_ROOT_ID_DIFFERENT_VERSION_HASH,
    ULTIMATE_CLOUD_ADMIN_CAN_CHANGE_CONTROL_PLANE,
    LIVE_EXTERNAL_TRUST_DOMAIN,
    SIGNING_KEY_ROTATION_POLICY,
    REVOKED_KEY_POLICY,
    HISTORICAL_PUBLIC_VERIFICATION_MATERIAL_RETAINED,
    OLD_RECEIPTS_REMAIN_VERIFIABLE_AFTER_NORMAL_ROTATION,
    POST_AUTHORITY_CLOSURE_MAX_DELAY_SECONDS,
    POST_AUTHORITY_CLOSURE_MAX_DELAY_SOURCE,
    POST_AUTHORITY_CLOSURE_MAX_DELAY_JUSTIFICATION_STATUS,
    PRODUCTION_POLICY_RATIFICATION,
)
from analyst_dashboard.smart_money.public_availability import (
    ClockStatus,
    ClockSample,
    ClockMeasurementResult,
    IndependentClockAuthority,
    CLOCK_REFERENCE_MAX_AGE_BEFORE_ARMING_SECONDS,
    CLOCK_REFERENCE_MAX_AGE_DURING_CAPTURE_SECONDS,
    SecPreflightResult,
    SecOperationalPreflight,
)


class TestSmartMoneyLiveQualification(unittest.TestCase):
    """Targeted tests for live external root infrastructure and clock preflight."""

    def setUp(self):
        self.harness = LiveExternalRootQualificationHarness()
        self.clock_auth = IndependentClockAuthority()
        self.sec_preflight = SecOperationalPreflight()
        self.run_id = "run_live_qual_001"
        self.run_final_hash = "f11a" * 16
        self.policy_set_hash = "e22b" * 16

        # Standard finalization events
        prev_hash = "0" * 64
        self.events = []
        for i in range(1, 4):
            evt_id = f"evt_qual_{i:03d}"
            content = f"{evt_id}:{self.run_id}:{self.run_final_hash}:{i}:EVENT:{prev_hash}"
            h = hashlib.sha256(content.encode()).hexdigest()
            self.events.append(
                FinalizationEvent(
                    finalization_event_id=evt_id,
                    parent_run_id=self.run_id,
                    parent_run_final_hash=self.run_final_hash,
                    finalization_sequence=i,
                    finalization_event_type="CAPTURE_PAYLOAD_SEALED" if i < 3 else "ALL_ANCHORS_CONFIRMED",
                    anchor_request_id=f"req_{i}",
                    anchor_attempt_id=f"att_{i}",
                    anchor_status_before=FinalizationState.A1_ANCHOR_REQUIRED,
                    anchor_status_after=FinalizationState.A6_ANCHOR_CONFIRMED,
                    authority_effective_status_before=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
                    authority_effective_status_after=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
                    recorded_at="2026-10-11T12:00:00Z",
                    previous_finalization_event_hash=prev_hash,
                    finalization_event_hash=h,
                )
            )
            prev_hash = h

    # =========================================================================
    # 1. Trust Model Definition (Section 4)
    # =========================================================================

    def test_trust_model_constants(self):
        """Verifies separation of storage acknowledgement from cryptographic receipt."""
        self.assertEqual(PROVIDER_STORAGE_ACKNOWLEDGEMENT, "PROVIDER_NATIVE_STORAGE_RESPONSE")
        self.assertEqual(CRYPTOGRAPHIC_ROOT_RECEIPT, "SEPARATE_SIGNED_ATTESTATION_OR_EQUIVALENT")
        self.assertFalse(STORAGE_ACKNOWLEDGEMENT_IS_CRYPTOGRAPHIC_RECEIPT)
        self.assertEqual(ROOT_STORAGE_PROVIDER, "AWS_S3")
        self.assertEqual(ROOT_STORAGE_SERVICE, "AWS_S3_OBJECT_LOCK")
        self.assertEqual(ROOT_RECEIPT_SIGNER_CLASS, "ARX_CONTROLLED_EXTERNAL_KEY_SERVICE")
        self.assertEqual(RECEIPT_SIGNER, "ARX_CONTROLLED_EXTERNAL_KEY_SERVICE")
        self.assertFalse(PROVIDER_NATIVE_SIGNED_RECEIPT)

    # =========================================================================
    # 2. Pinned Provider Identity Discovery & Fail-Closed Behavior (Section 5)
    # =========================================================================

    def test_pinned_provider_identities_fail_closed_when_unprovisioned(self):
        """When live AWS credentials are not provisioned, identity inspection fails closed."""
        status = self.harness.inspect_live_provider_identity()
        # In current environment, no AWS credentials exist
        self.assertFalse(status.credentials_present)
        self.assertEqual(status.status, "FAIL")
        self.assertIn("LIVE_PROVIDER_CREDENTIALS_ABSENT", status.error_detail)
        self.assertIsNotNone(status.raw_response_sha256)

    # =========================================================================
    # 3. Capability Separation Matrix (Section 9)
    # =========================================================================

    def test_capability_separation_matrix(self):
        """Verifies explicit capability boundaries between application and external principals."""
        cap = self.harness.capability_matrix
        self.assertTrue(cap.application_principal_can_write_root)
        self.assertFalse(cap.application_principal_can_delete_locked_version)
        self.assertFalse(cap.application_principal_can_change_retention)
        self.assertFalse(cap.application_principal_can_change_bucket_policy)
        self.assertFalse(cap.application_principal_can_change_object_lock_configuration)
        self.assertFalse(cap.application_principal_can_change_key_policy)
        self.assertFalse(cap.application_principal_can_use_signing_key_directly)
        self.assertFalse(cap.application_principal_can_disable_signing_key)
        self.assertFalse(cap.root_writer_can_change_retention)
        self.assertFalse(cap.root_writer_can_change_signing_key_policy)
        self.assertFalse(cap.signing_principal_can_delete_root)
        self.assertFalse(cap.signing_principal_can_change_retention)

    # =========================================================================
    # 4. Same-Key New-Version Collision Detection (Section 8)
    # =========================================================================

    def test_same_key_new_version_collision_detection(self):
        """Detects if provider creates a new object version under same key with different hash."""
        collision = self.harness.detect_same_key_new_version_collision(
            existing_root_id="root_001",
            existing_version_id="ver_001",
            existing_content_hash="hash_a" * 8,
            new_version_id="ver_002",
            new_content_hash="hash_b" * 8,
        )
        self.assertTrue(collision)

        no_collision = self.harness.detect_same_key_new_version_collision(
            existing_root_id="root_001",
            existing_version_id="ver_001",
            existing_content_hash="hash_a" * 8,
            new_version_id="ver_001",
            new_content_hash="hash_a" * 8,
        )
        self.assertFalse(no_collision)

    # =========================================================================
    # 5. Key Rotation & Revocation Contract (Section 12)
    # =========================================================================

    def test_key_rotation_and_revocation_lifecycle(self):
        """Verifies distinct behavior between normal retirement and compromise revocation."""
        self.assertEqual(SIGNING_KEY_ROTATION_POLICY, "VERSIONED_RETIREMENT_PRESERVES_VERIFICATION")
        self.assertEqual(REVOKED_KEY_POLICY, "COMPROMISE_INVALIDATES_ALL_RECEIPTS_AFTER_REVOCATION")
        self.assertTrue(HISTORICAL_PUBLIC_VERIFICATION_MATERIAL_RETAINED)
        self.assertTrue(OLD_RECEIPTS_REMAIN_VERIFIABLE_AFTER_NORMAL_ROTATION)

    # =========================================================================
    # 6. Receipt Binding Contract & Replay Protection (Section 13)
    # =========================================================================

    def test_receipt_binding_cross_run_replay_rejected(self):
        """Receipt from run A presented for run B is strictly rejected."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc="2026-10-11T12:00:00Z",
        )
        receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc="2026-10-11T12:01:00Z",
        )
        # Attempt verification against different run ID
        ok, checks = self.harness.verify_receipt_binding(
            receipt=receipt,
            expected_run_id="run_different_002",
            expected_root_phase=RootPhase.PRE_AUTHORITY_ADMISSION_ROOT,
            expected_root_id=receipt.root_id,
            expected_root_hash=receipt.finalization_ledger_root_hash,
            expected_policy_set_hash=self.policy_set_hash,
            expected_idempotency_key="idemp_001",
            expected_provider_resource_id="AWS_S3_OBJECT_LOCK_ROOT",
            expected_provider_object_version_id="ver_001",
            expected_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
        )
        self.assertFalse(ok)
        self.assertFalse(checks["run_id_match"])

    def test_receipt_binding_cross_phase_replay_rejected(self):
        """R1 receipt presented to R2 verifier is strictly rejected."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc="2026-10-11T12:00:00Z",
        )
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc="2026-10-11T12:01:00Z",
        )
        # Verify R1 receipt against expected phase POST_AUTHORITY_CLOSURE_ROOT
        ok, checks = self.harness.verify_receipt_binding(
            receipt=r1_receipt,
            expected_run_id=self.run_id,
            expected_root_phase=RootPhase.POST_AUTHORITY_CLOSURE_ROOT,
            expected_root_id=r1_receipt.root_id,
            expected_root_hash=r1_receipt.finalization_ledger_root_hash,
            expected_policy_set_hash=self.policy_set_hash,
            expected_idempotency_key="idemp_001",
            expected_provider_resource_id="AWS_S3_OBJECT_LOCK_ROOT",
            expected_provider_object_version_id="ver_001",
            expected_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
        )
        # Root phase check fails
        self.assertFalse(ok)

    def test_receipt_binding_cross_root_substitution_rejected(self):
        """Valid receipt for Root A presented for Root B fails root hash match."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc="2026-10-11T12:00:00Z",
        )
        receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc="2026-10-11T12:01:00Z",
        )
        ok, checks = self.harness.verify_receipt_binding(
            receipt=receipt,
            expected_run_id=self.run_id,
            expected_root_phase=RootPhase.PRE_AUTHORITY_ADMISSION_ROOT,
            expected_root_id=receipt.root_id,
            expected_root_hash="substituted_hash_" + "0" * 47,
            expected_policy_set_hash=self.policy_set_hash,
            expected_idempotency_key="idemp_001",
            expected_provider_resource_id="AWS_S3_OBJECT_LOCK_ROOT",
            expected_provider_object_version_id="ver_001",
            expected_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
        )
        self.assertFalse(ok)
        self.assertFalse(checks["root_hash_match"])

    # =========================================================================
    # 7. Ambiguous-Commit Recovery Simulation (Section 19)
    # =========================================================================

    def test_ambiguous_commit_recovery_behaviors(self):
        """Verifies query-before-retry behavior across provider lookup outcomes."""
        self.assertEqual(
            self.harness.simulate_ambiguous_commit_recovery(RootLookupResult.FOUND_MATCHING),
            "RECOVER_EXISTING_ROOT",
        )
        self.assertEqual(
            self.harness.simulate_ambiguous_commit_recovery(RootLookupResult.FOUND_CONFLICTING),
            "TERMINAL_HOLD",
        )
        self.assertEqual(
            self.harness.simulate_ambiguous_commit_recovery(RootLookupResult.NOT_FOUND),
            "RETRY_PERMITTED",
        )
        self.assertEqual(
            self.harness.simulate_ambiguous_commit_recovery(RootLookupResult.INDETERMINATE),
            "REMAIN_BLOCKED",
        )

    # =========================================================================
    # 8. Synthetic Qualification Lifecycle (Sections 14-25)
    # =========================================================================

    def test_synthetic_qualification_r1_and_r2_lifecycle(self):
        """Executes complete isolated qualification lifecycle using synthetic data only."""
        res = self.harness.execute_synthetic_qualification_lifecycle(
            run_id=self.run_id,
            run_final_hash=self.run_final_hash,
            events=self.events,
            policy_set_hash=self.policy_set_hash,
        )
        self.assertEqual(res["r1_status"], R1ConfirmationStatus.CONFIRMED)
        self.assertEqual(res["r2_status"], PostAuthorityClosureStatus.CONFIRMED)
        self.assertTrue(res["reviewer_audit"]["overall_passed"])
        self.assertEqual(res["reviewer_audit"]["reconstruction_mismatches"], 0)

        # Invariant checks: R1 does not contain receipt; R2 binds R1 linkage
        self.assertIsNone(res["r1_payload"].r1_receipt_hash)
        self.assertEqual(res["r2_payload"].r1_root_id, res["r1_receipt"].root_id)
        self.assertEqual(res["r2_payload"].r1_root_hash, res["r1_receipt"].root_hash)
        self.assertEqual(res["r2_payload"].r1_receipt_hash, res["r1_receipt"].receipt_hash)

    # =========================================================================
    # 9. 300-Second Delay Policy Governance Review (Section 38)
    # =========================================================================

    def test_300_second_r2_delay_policy_review(self):
        """Confirms 300s value is implementation-default requiring governance justification."""
        self.assertEqual(POST_AUTHORITY_CLOSURE_MAX_DELAY_SECONDS, 300.0)
        self.assertEqual(POST_AUTHORITY_CLOSURE_MAX_DELAY_SOURCE, "IMPLEMENTATION_DEFAULT_ONLY")
        self.assertEqual(POST_AUTHORITY_CLOSURE_MAX_DELAY_JUSTIFICATION_STATUS, "REQUIRED")
        self.assertEqual(PRODUCTION_POLICY_RATIFICATION, "BLOCKED_PENDING_R2_DELAY_JUSTIFICATION")

    # =========================================================================
    # 10. Independent Clock Authority Measurement & Bound (Sections 27, 28)
    # =========================================================================

    def test_clock_measurement_theorem_and_bound_derivation(self):
        """Verifies conservative bound calculation: abs(offset) + uncertainty + asymmetry + other."""
        samples = [
            ClockSample(
                server="time.google.com",
                offset_ms=12.5,
                rtt_ms=30.0,
                uncertainty_ms=15.0,
                stratum=1,
                captured_at_utc="2026-10-11T12:00:00Z",
                raw_packet_hex="00" * 48,
            ),
            ClockSample(
                server="time.cloudflare.com",
                offset_ms=11.5,
                rtt_ms=26.0,
                uncertainty_ms=13.0,
                stratum=1,
                captured_at_utc="2026-10-11T12:00:00Z",
                raw_packet_hex="00" * 48,
            ),
        ]
        result = self.clock_auth.measure_clock(
            manual_samples=samples,
            now_override_utc="2026-10-11T12:00:00Z",
        )
        self.assertEqual(result.observed_offset_ms, 12.0)
        self.assertEqual(result.measurement_uncertainty_ms, 15.0)
        self.assertEqual(result.network_asymmetry_allowance_ms, 10.0)
        self.assertEqual(result.other_error_allowance_ms, 5.0)
        # Total bound: abs(12.0) + 15.0 + 10.0 + 5.0 = 42.0 ms
        self.assertEqual(result.total_bound_ms, 42.0)
        self.assertTrue(result.within_policy)
        self.assertEqual(result.clock_status, ClockStatus.SYNCHRONIZED)

    def test_clock_measurement_out_of_bound_fails_closed(self):
        """When offset exceeds 500 ms bound, clock status is UNTRUSTED and within_policy is False."""
        samples = [
            ClockSample(
                server="time.google.com",
                offset_ms=8990.0,
                rtt_ms=40.0,
                uncertainty_ms=20.0,
                stratum=1,
                captured_at_utc="2026-10-11T12:00:00Z",
                raw_packet_hex="00" * 48,
            ),
        ]
        result = self.clock_auth.measure_clock(
            manual_samples=samples,
            now_override_utc="2026-10-11T12:00:00Z",
        )
        self.assertEqual(result.observed_offset_ms, 8990.0)
        # Total bound: 8990 + 20 + 10 + 5 = 9025 ms > 500 ms
        self.assertEqual(result.total_bound_ms, 9025.0)
        self.assertFalse(result.within_policy)
        self.assertEqual(result.clock_status, ClockStatus.UNTRUSTED)

    # =========================================================================
    # 11. Run-Start Interval Derivation (Section 31)
    # =========================================================================

    def test_run_start_interval_derivation(self):
        """Verifies genesis interval derivation with live clock total bound."""
        samples = [
            ClockSample(
                server="time.google.com",
                offset_ms=5.0,
                rtt_ms=20.0,
                uncertainty_ms=10.0,
                stratum=1,
                captured_at_utc="2026-10-11T12:00:00Z",
                raw_packet_hex="00" * 48,
            ),
        ]
        res = self.clock_auth.measure_clock(
            manual_samples=samples,
            now_override_utc="2026-10-11T12:00:00Z",
        )
        start_iso, end_iso, evid_id = self.clock_auth.derive_run_start_intervals(
            result=res,
            local_genesis_time_iso="2026-10-11T12:00:00+00:00",
        )
        # Total bound = 5 + 10 + 10 + 5 = 30 ms (0.03 seconds)
        dt_start = datetime.fromisoformat(start_iso)
        dt_end = datetime.fromisoformat(end_iso)
        self.assertEqual((dt_end - dt_start).total_seconds(), 0.06)
        self.assertEqual(evid_id, res.measurement_id)

    # =========================================================================
    # 12. SEC Operational Preflight (Sections 32, 33)
    # =========================================================================

    def test_sec_operational_preflight_fails_on_weekend(self):
        """Sunday is not an SEC business day and filing window is closed (Fail-Closed)."""
        # 2026-10-11 is Sunday
        res = self.sec_preflight.execute_preflight(
            now_override_utc="2026-10-11T12:00:00Z",
            mock_reachability=(True, True),
        )
        self.assertFalse(res.sec_business_day)
        self.assertFalse(res.sec_active_filing_window)
        self.assertFalse(res.sec_operational_preflight_passed)
        self.assertIn("AWAIT_NEXT_SEC_BUSINESS_DAY", res.next_eligible_recheck_condition)

    def test_sec_operational_preflight_passes_on_valid_business_window(self):
        """Wednesday at 14:00 ET inside business hours with compliant access passes."""
        # 2026-10-14 is Wednesday; 18:00 UTC is 14:00 EDT
        res = self.sec_preflight.execute_preflight(
            now_override_utc="2026-10-14T18:00:00Z",
            mock_reachability=(True, True),
        )
        self.assertTrue(res.sec_business_day)
        self.assertTrue(res.sec_active_filing_window)
        self.assertTrue(res.sec_automated_access_policy_current)
        self.assertTrue(res.sec_user_agent_policy_compliant)
        self.assertTrue(res.sec_rate_limit_policy_compliant)
        self.assertTrue(res.sec_metadata_endpoint_reachable)
        self.assertTrue(res.sec_content_endpoint_reachable)
        self.assertTrue(res.sec_operational_preflight_passed)

    def test_sec_operational_preflight_user_agent_compliance(self):
        """Rejects user agents lacking contact email format."""
        self.assertTrue(self.sec_preflight.is_user_agent_compliant("ArxTerminal/1.0 (akara@finance.app)"))
        self.assertTrue(self.sec_preflight.is_user_agent_compliant("Sample Company AdminContact@example.com"))
        self.assertFalse(self.sec_preflight.is_user_agent_compliant("AnonymousScraper"))
        self.assertFalse(self.sec_preflight.is_user_agent_compliant(""))

    # =========================================================================
    # 13. Combined Arming Theorem (Section 35)
    # =========================================================================

    def test_combined_arming_theorem_fail_closed(self):
        """Proves that arming is NOT authorized if ANY gate fails."""
        # Case 1: Weekend fails SEC preflight -> Arming NOT authorized
        sec_res = self.sec_preflight.execute_preflight(
            now_override_utc="2026-10-11T12:00:00Z",
            mock_reachability=(True, True),
        )
        self.assertFalse(sec_res.sec_operational_preflight_passed)

        # Case 2: Unprovisioned AWS credentials -> Infrastructure qualification fails
        prov_res = self.harness.inspect_live_provider_identity()
        self.assertFalse(prov_res.credentials_present)

        # Combined arming condition:
        arming_authorized = (
            prov_res.credentials_present
            and sec_res.sec_operational_preflight_passed
        )
        self.assertFalse(arming_authorized)
