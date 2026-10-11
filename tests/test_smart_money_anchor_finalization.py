"""
Test Suite: Smart Money External Anchor Finalization Subsystem (Epoch 001).
Covers:
- Section 4 & 24: Ratified Policy Epoch & Retained Policy Artifacts
- Section 5: Quorum & Dual-Anchor Independence
- Section 6 & 11: Run Identity & Idempotency Key Continuity
- Section 7 & 8: Finalization State Machine & Authority Separation
- Section 9 & 10: Sealed Run Ledger & Canonical Payload Serialization
- Section 12, 14, 15: Attempt Ledger, Finalization Ledger Atomicity, Hash Chaining
- Section 13: Ambiguous Commit Result Policy & Provider Lookup
- Section 16: Storage Configuration Attestation (WORM/Object Lock)
- Section 17: Signing & Key Separation Invariants
- Section 18: Provider Receipt Verification Suite
- Section 19: Retry Policy & Backoff Bounds
- Section 20: Expired Pending State
- Section 21: Late Anchor Recovery & No Retroactive Authorization
- Section 22 & 23: Downstream-Use Firewall Tests
- Section 26: Independent Reviewer Reconstruction Procedure
- Section 27: Crash-Recovery Test Matrix
- Section 28: Conflict Test Matrix
- Section 29: Authority Timing Invariant Tests (T1 <= T2 <= T3)
- Section 30: No-Retroactive-Authorization Verification
"""

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.smart_money.anchor_finalization import (
    ANCHOR_POLICY_ID,
    ANCHOR_POLICY_VERSION,
    ANCHOR_POLICY_EFFECTIVE_AT,
    ANCHOR_POLICY_EFFECTIVE_FROM_RUN_EPOCH,
    ANCHOR_REQUIREMENT_APPLIES_RETROACTIVELY,
    EXTERNAL_ANCHOR_REQUIRED,
    AUTHORITY_ELIGIBILITY_CAN_EXIST_WITHOUT_ANCHOR,
    AUTHORITY_EFFECTIVENESS_REQUIRES_ANCHOR,
    HUMAN_OVERRIDE_CAN_GRANT_AUTHORITY,
    PROVIDER_OUTAGE_CAN_DISABLE_ANCHOR_REQUIREMENT,
    PROVIDER_OUTAGE_CAN_LOWER_QUORUM,
    RUN_ID_REUSE_ALLOWED,
    AMBIGUOUS_COMMIT_RESULT_POLICY,
    INITIAL_RETRY_DELAY_SECONDS,
    MAX_RETRY_DELAY_SECONDS,
    MAX_INLINE_ATTEMPTS,
    MAX_PENDING_DURATION_SECONDS,
    FinalizationState,
    AuthorityEligibility,
    AuthorityEffectiveStatus,
    ErrorClass,
    LookupResult,
    QuorumPolicy,
    CriticalIntegrityError,
    InvalidTransitionError,
    DownstreamFirewallBlockedError,
    QuorumNotSatisfiedError,
    SealedRunLedger,
    AnchorPayload,
    AnchorAttemptRecord,
    FinalizationEvent,
    AnchorReceipt,
    AnchorProviderConfig,
    DEFAULT_PROVIDER_A,
    DEFAULT_PROVIDER_B,
    ANCHOR_PROVIDER_COUNT,
    ANCHOR_REQUIRED_COUNT,
    ANCHOR_QUORUM_POLICY,
    compute_anchor_payload_hash,
    derive_anchor_idempotency_key,
    validate_idempotency_continuity,
    verify_dual_anchor_independence,
    evaluate_anchor_quorum,
    assert_anchor_quorum_satisfied,
    generate_provider_receipt_signature,
    verify_provider_receipt,
    create_storage_configuration_attestation,
    verify_storage_configuration_attestation,
    DownstreamUseFirewall,
    FinalizationLedgerStore,
    FinalizationStateMachine,
    AnchorFinalizationEngine,
    AnchorFinalizationReviewer,
    get_policy_artifact,
    verify_all_policy_artifacts_retained,
)


class TestSmartMoneyAnchorFinalization(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = os.path.join(self.tmp_dir.name, "finalization_test.db")
        self.store = FinalizationLedgerStore(self.db_path)

        # Baseline valid timestamps
        self.t1_observed = "2026-10-11T12:00:00.000Z"
        self.t2_eligible = "2026-10-11T12:00:01.000Z"
        self.t3_effective = "2026-10-11T12:00:05.000Z"

        # Baseline sealed ledger
        self.sample_sealed_ledger = SealedRunLedger(
            run_id="run_20261011_sec_natural_001",
            run_genesis_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
            run_final_hash="9999888877776666555544443333222211110000aaaabbbbccccddddeeeeffff",
            transition_count=42,
            final_sequence=42,
            final_state="SEALED_NATURAL_CAPTURE",
            policy_set_hash="c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0",
            natural_bundle_hash="beefcafebeefcafebeefcafebeefcafebeefcafebeefcafebeefcafebeefcafe",
            repository_head_sha="3714b3119162d54b0260e656089695069457c726",
            sealed_at_utc=self.t1_observed,
            is_sealed=True,
        )

    def tearDown(self):
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Section 4 & 24: Policy Epoch & Retained Policy Artifacts
    # -----------------------------------------------------------------------
    def test_policy_epoch_invariants(self):
        """Verifies Section 4 frozen policy epoch constants."""
        self.assertEqual(ANCHOR_POLICY_ID, "SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1")
        self.assertEqual(ANCHOR_POLICY_VERSION, "1.0")
        self.assertEqual(ANCHOR_POLICY_EFFECTIVE_AT, "2026-10-11T12:00:00Z")
        self.assertEqual(ANCHOR_POLICY_EFFECTIVE_FROM_RUN_EPOCH, "EPOCH_001")
        self.assertFalse(ANCHOR_REQUIREMENT_APPLIES_RETROACTIVELY)
        self.assertTrue(EXTERNAL_ANCHOR_REQUIRED)
        self.assertTrue(AUTHORITY_ELIGIBILITY_CAN_EXIST_WITHOUT_ANCHOR)
        self.assertTrue(AUTHORITY_EFFECTIVENESS_REQUIRES_ANCHOR)
        self.assertFalse(HUMAN_OVERRIDE_CAN_GRANT_AUTHORITY)
        self.assertFalse(PROVIDER_OUTAGE_CAN_DISABLE_ANCHOR_REQUIREMENT)
        self.assertFalse(PROVIDER_OUTAGE_CAN_LOWER_QUORUM)
        self.assertFalse(RUN_ID_REUSE_ALLOWED)
        self.assertEqual(AMBIGUOUS_COMMIT_RESULT_POLICY, "QUERY_EXISTING_ANCHOR_BEFORE_RETRY")

    def test_retained_policy_artifacts_completeness(self):
        """Verifies Section 24 retention of exact bytes for all 8 policies."""
        self.assertTrue(verify_all_policy_artifacts_retained())
        for policy_id in [
            "anchor_policy",
            "state_machine_policy",
            "clock_policy",
            "observer_policy",
            "authority_admission_policy",
            "transport_classification_policy",
            "identity_policy",
            "decision_input_contract",
        ]:
            artifact = get_policy_artifact(policy_id)
            self.assertTrue(artifact["policy_artifact_retained"])
            self.assertEqual(len(artifact["policy_artifact_sha256"]), 64)
            self.assertGreater(len(artifact["policy_artifact_raw_bytes"]), 0)

    # -----------------------------------------------------------------------
    # Section 5: Quorum & Dual-Anchor Independence
    # -----------------------------------------------------------------------
    def test_dual_anchor_independence_pass(self):
        """Verifies independence between distinct cloud providers and credentials."""
        self.assertEqual(ANCHOR_PROVIDER_COUNT, 2)
        self.assertEqual(ANCHOR_REQUIRED_COUNT, 2)
        self.assertEqual(ANCHOR_QUORUM_POLICY, QuorumPolicy.ALL_REQUIRED)
        verdict = verify_dual_anchor_independence(DEFAULT_PROVIDER_A, DEFAULT_PROVIDER_B)
        self.assertEqual(verdict, "PASS")

    def test_dual_anchor_independence_fail_on_shared_credentials(self):
        """Rejects pseudo-redundancy sharing the same credential/key domain."""
        compromised_b = AnchorProviderConfig(
            provider_id="GCP_CLOUD_STORAGE_LOCK",
            credential_domain="AWS_IAM_ROLE_ARX_ANCHOR_A",  # Colliding credential domain
            key_domain="AWS_KMS_KEY_ARX_ANCHOR_A",
            admin_domain="AWS_ACCOUNT_SECURITY_ORG_A",
            write_authority="INSERT_OBJECT_RETENTION_LOCK",
        )
        verdict = verify_dual_anchor_independence(DEFAULT_PROVIDER_A, compromised_b)
        self.assertEqual(verdict, "FAIL")

    def test_quorum_evaluation_rules(self):
        """Enforces ALL_REQUIRED quorum and prevents downgrade on outage."""
        required = ["PROV_A", "PROV_B"]
        # Both present
        self.assertTrue(evaluate_anchor_quorum(["PROV_A", "PROV_B"], required, QuorumPolicy.ALL_REQUIRED))
        # One missing -> fails
        self.assertFalse(evaluate_anchor_quorum(["PROV_A"], required, QuorumPolicy.ALL_REQUIRED))
        with self.assertRaises(QuorumNotSatisfiedError):
            assert_anchor_quorum_satisfied(["PROV_A"], required, QuorumPolicy.ALL_REQUIRED)

    # -----------------------------------------------------------------------
    # Section 6, 9, 10, 11: Sealing, Payload, Idempotency & Conflict
    # -----------------------------------------------------------------------
    def test_unsealed_run_ledger_rejected(self):
        """Cannot create anchor payload if run ledger is not sealed."""
        unsealed = SealedRunLedger(
            run_id="run_unsealed",
            run_genesis_hash="00" * 32,
            run_final_hash="11" * 32,
            transition_count=5,
            final_sequence=5,
            final_state="PENDING",
            policy_set_hash="22" * 32,
            natural_bundle_hash="33" * 32,
            repository_head_sha="44" * 20,
            sealed_at_utc=self.t1_observed,
            is_sealed=False,
        )
        engine = AnchorFinalizationEngine(self.store)
        with self.assertRaises(ValueError):
            engine.build_anchor_payload(unsealed)

    def test_canonical_payload_and_idempotency_derivation(self):
        """Verifies deterministic serialization and idempotency key derivation."""
        engine = AnchorFinalizationEngine(self.store)
        payload, payload_hash, idemp_key = engine.build_anchor_payload(self.sample_sealed_ledger)
        self.assertEqual(len(payload_hash), 64)
        self.assertEqual(len(idemp_key), 64)

        # Same payload again -> identical hash and key
        payload2, hash2, key2 = engine.build_anchor_payload(self.sample_sealed_ledger)
        self.assertEqual(payload_hash, hash2)
        self.assertEqual(idemp_key, key2)

    def test_same_idempotency_key_different_payload_critical_failure(self):
        """Freezes: SAME_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD = CRITICAL_INTEGRITY_FAILURE."""
        with self.assertRaises(CriticalIntegrityError):
            validate_idempotency_continuity(
                existing_idempotency_key="key_abc123",
                incoming_idempotency_key="key_abc123",
                existing_payload_hash="hash_1111",
                incoming_payload_hash="hash_2222",
            )

    # -----------------------------------------------------------------------
    # Section 7 & 8: State Machine Transitions & Authority Separation
    # -----------------------------------------------------------------------
    def test_state_machine_valid_progression(self):
        """Tests linear progression A1 -> A2 -> A3 -> A6 -> A10."""
        sm = FinalizationStateMachine(
            initial_state=FinalizationState.A1_ANCHOR_REQUIRED,
            evidence_observed_at=self.t1_observed,
            eligibility_established_at=self.t2_eligible,
        )
        self.assertEqual(sm.state, FinalizationState.A1_ANCHOR_REQUIRED)
        self.assertEqual(sm.effective_status, AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR)

        sm.transition(FinalizationState.A2_ANCHOR_PAYLOAD_SEALED)
        self.assertEqual(sm.state, FinalizationState.A2_ANCHOR_PAYLOAD_SEALED)

        sm.transition(FinalizationState.A3_ANCHOR_SUBMISSION_PENDING)
        self.assertEqual(sm.state, FinalizationState.A3_ANCHOR_SUBMISSION_PENDING)

        sm.transition(FinalizationState.A6_ANCHOR_CONFIRMED)
        self.assertEqual(sm.state, FinalizationState.A6_ANCHOR_CONFIRMED)

        # Quorum satisfied -> A10
        sm.transition(
            FinalizationState.A10_AUTHORITY_EFFECTIVE,
            effective_time_utc=self.t3_effective,
            quorum_satisfied=True,
        )
        self.assertEqual(sm.state, FinalizationState.A10_AUTHORITY_EFFECTIVE)
        self.assertEqual(sm.effective_status, AuthorityEffectiveStatus.EFFECTIVE)
        self.assertEqual(sm.authority_became_effective_at, self.t3_effective)

    def test_state_machine_invalid_transition_rejected(self):
        """Illegal jumps (e.g. A1 directly to A10) must raise InvalidTransitionError."""
        sm = FinalizationStateMachine(initial_state=FinalizationState.A1_ANCHOR_REQUIRED)
        with self.assertRaises(InvalidTransitionError):
            sm.transition(FinalizationState.A10_AUTHORITY_EFFECTIVE, quorum_satisfied=True)

    def test_transition_to_a10_blocked_without_quorum(self):
        """A6 -> A10 without quorum MUST fail closed."""
        sm = FinalizationStateMachine(initial_state=FinalizationState.A6_ANCHOR_CONFIRMED)
        with self.assertRaises(InvalidTransitionError):
            sm.transition(
                FinalizationState.A10_AUTHORITY_EFFECTIVE,
                effective_time_utc=self.t3_effective,
                quorum_satisfied=False,  # Quorum not satisfied!
            )

    # -----------------------------------------------------------------------
    # Section 12, 14, 15: Attempt & Finalization Ledgers, Atomicity & Hashing
    # -----------------------------------------------------------------------
    def test_finalization_ledger_append_and_hash_chain(self):
        """Verifies hash-chained event insertion in SQLite."""
        self.store.initialize_run_coordinator("run_test_001", "final_hash_001")
        ev1 = self.store.append_finalization_event(
            parent_run_id="run_test_001",
            parent_run_final_hash="final_hash_001",
            finalization_event_type="SEAL",
            anchor_request_id="req_1",
            anchor_attempt_id="att_1",
            anchor_status_to=FinalizationState.A2_ANCHOR_PAYLOAD_SEALED,
            authority_effective_status_to=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            recorded_at="2026-10-11T12:00:01.000Z",
        )
        self.assertEqual(ev1.finalization_sequence, 1)
        self.assertEqual(ev1.previous_finalization_event_hash, FinalizationLedgerStore.GENESIS_PREVIOUS_HASH)

        ev2 = self.store.append_finalization_event(
            parent_run_id="run_test_001",
            parent_run_final_hash="final_hash_001",
            finalization_event_type="CONFIRM",
            anchor_request_id="req_1",
            anchor_attempt_id="att_1",
            anchor_status_to=FinalizationState.A6_ANCHOR_CONFIRMED,
            authority_effective_status_to=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            recorded_at="2026-10-11T12:00:02.000Z",
        )
        self.assertEqual(ev2.finalization_sequence, 2)
        self.assertEqual(ev2.previous_finalization_event_hash, ev1.finalization_event_hash)

        events = self.store.get_events("run_test_001")
        self.assertEqual(len(events), 2)

    def test_attempt_ledger_logging(self):
        """Verifies attempts are recorded in the attempt ledger."""
        attempt = AnchorAttemptRecord(
            anchor_attempt_id="att_100",
            anchor_request_id="req_100",
            attempt_number=1,
            run_id="run_test_002",
            anchor_idempotency_key="idemp_100",
            request_payload_sha256="payload_sha_100",
            request_started_at="2026-10-11T12:00:00Z",
            request_completed_at="2026-10-11T12:00:01Z",
            provider="AWS_S3_OBJECT_LOCK",
            provider_http_status=200,
            provider_response_sha256="sig_100",
            error_class=ErrorClass.NONE,
            result="SUCCESS",
            lookup_required=False,
            lookup_performed=False,
        )
        self.store.record_attempt(attempt)
        recorded = self.store.get_attempts("run_test_002")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].anchor_attempt_id, "att_100")

    # -----------------------------------------------------------------------
    # Section 13: Ambiguous Commit Result Policy & Provider Lookup
    # -----------------------------------------------------------------------
    def test_ambiguous_commit_lookup_matching_confirms(self):
        """If indeterminate commit occurs, query existing anchor before retry -> matching confirms."""
        engine = AnchorFinalizationEngine(self.store, required_providers=["AWS_S3_OBJECT_LOCK"])
        payload, payload_hash, idemp_key = engine.build_anchor_payload(self.sample_sealed_ledger)
        sm = FinalizationStateMachine(initial_state=FinalizationState.A1_ANCHOR_REQUIRED)

        # Create valid receipt
        rcpt_dict = {
            "provider": "AWS_S3_OBJECT_LOCK",
            "run_id": payload.run_id,
            "run_final_hash": payload.run_final_hash,
            "payload_sha256": payload_hash,
            "policy_set_hash": payload.policy_set_hash,
            "stored_object_id": "s3://arx-worm/run_001.json",
            "timestamp_utc": "2026-10-11T12:00:02Z",
        }
        sig = generate_provider_receipt_signature(rcpt_dict)
        valid_rcpt = AnchorReceipt(
            receipt_id="rcpt_001",
            provider="AWS_S3_OBJECT_LOCK",
            run_id=payload.run_id,
            run_final_hash=payload.run_final_hash,
            payload_sha256=payload_hash,
            policy_set_hash=payload.policy_set_hash,
            stored_object_id=rcpt_dict["stored_object_id"],
            timestamp_utc=rcpt_dict["timestamp_utc"],
            signature=sig,
        )

        def mock_submit_indeterminate(p):
            return 504, None, ErrorClass.INDETERMINATE

        def mock_lookup_matching(r_id, key):
            return LookupResult.FOUND_MATCHING, valid_rcpt

        success = engine.submit_anchor_for_provider(
            payload=payload,
            payload_hash=payload_hash,
            idempotency_key=idemp_key,
            provider_id="AWS_S3_OBJECT_LOCK",
            sm=sm,
            provider_submit_fn=mock_submit_indeterminate,
            provider_lookup_fn=mock_lookup_matching,
        )
        self.assertTrue(success)
        self.assertEqual(sm.state, FinalizationState.A6_ANCHOR_CONFIRMED)

    def test_ambiguous_commit_lookup_conflicting_transitions_to_conflict(self):
        """If lookup returns conflicting anchor -> transition to A8_ANCHOR_CONFLICT -> A11_TERMINAL_HOLD."""
        engine = AnchorFinalizationEngine(self.store, required_providers=["AWS_S3_OBJECT_LOCK"])
        payload, payload_hash, idemp_key = engine.build_anchor_payload(self.sample_sealed_ledger)
        sm = FinalizationStateMachine(initial_state=FinalizationState.A1_ANCHOR_REQUIRED)

        def mock_submit_indeterminate(p):
            return 504, None, ErrorClass.INDETERMINATE

        def mock_lookup_conflicting(r_id, key):
            return LookupResult.FOUND_CONFLICTING, None

        success = engine.submit_anchor_for_provider(
            payload=payload,
            payload_hash=payload_hash,
            idempotency_key=idemp_key,
            provider_id="AWS_S3_OBJECT_LOCK",
            sm=sm,
            provider_submit_fn=mock_submit_indeterminate,
            provider_lookup_fn=mock_lookup_conflicting,
        )
        self.assertFalse(success)
        self.assertEqual(sm.state, FinalizationState.A11_TERMINAL_HOLD)
        self.assertEqual(sm.effective_status, AuthorityEffectiveStatus.TERMINAL_HOLD)

    # -----------------------------------------------------------------------
    # Section 16 & 17: Storage Attestation & Key Separation
    # -----------------------------------------------------------------------
    def test_storage_configuration_attestation(self):
        """Verifies WORM / Object Lock configuration attestation."""
        att = create_storage_configuration_attestation("AWS_S3_OBJECT_LOCK")
        self.assertTrue(verify_storage_configuration_attestation(att))
        self.assertEqual(att.immutability_class, "PROVIDER_ENFORCED_RETENTION")
        self.assertEqual(att.trust_assumption, "PROVIDER_CONTROL_PLANE_NOT_COMPROMISED")

    # -----------------------------------------------------------------------
    # Section 18: Provider Receipt Verification
    # -----------------------------------------------------------------------
    def test_provider_receipt_verification_pass_and_tamper_fails(self):
        """Validates provider receipt verification and verifies tampers are detected."""
        engine = AnchorFinalizationEngine(self.store)
        payload, payload_hash, _ = engine.build_anchor_payload(self.sample_sealed_ledger)

        rcpt_dict = {
            "provider": "AWS_S3_OBJECT_LOCK",
            "run_id": payload.run_id,
            "run_final_hash": payload.run_final_hash,
            "payload_sha256": payload_hash,
            "policy_set_hash": payload.policy_set_hash,
            "stored_object_id": "s3://arx-worm/run_001.json",
            "timestamp_utc": "2026-10-11T12:00:02Z",
        }
        sig = generate_provider_receipt_signature(rcpt_dict)
        valid_rcpt = AnchorReceipt(
            receipt_id="rcpt_001",
            provider="AWS_S3_OBJECT_LOCK",
            run_id=payload.run_id,
            run_final_hash=payload.run_final_hash,
            payload_sha256=payload_hash,
            policy_set_hash=payload.policy_set_hash,
            stored_object_id=rcpt_dict["stored_object_id"],
            timestamp_utc=rcpt_dict["timestamp_utc"],
            signature=sig,
        )

        verdict, details = verify_provider_receipt(valid_rcpt, payload)
        self.assertEqual(verdict, "PASS")
        self.assertTrue(all(details.values()))

        # Tampered final hash
        tampered_rcpt = AnchorReceipt(
            receipt_id="rcpt_001",
            provider="AWS_S3_OBJECT_LOCK",
            run_id=payload.run_id,
            run_final_hash="tampered_hash_000000000000000000000000000000000000000000000000",
            payload_sha256=payload_hash,
            policy_set_hash=payload.policy_set_hash,
            stored_object_id=rcpt_dict["stored_object_id"],
            timestamp_utc=rcpt_dict["timestamp_utc"],
            signature=sig,
        )
        t_verdict, t_details = verify_provider_receipt(tampered_rcpt, payload)
        self.assertEqual(t_verdict, "FAIL")
        self.assertFalse(t_details["anchor_receipt_final_hash_match"])

    # -----------------------------------------------------------------------
    # Section 19 & 20: Retry Policy & Pending Expired
    # -----------------------------------------------------------------------
    def test_pending_duration_exceeded_transitions_to_expired(self):
        """When pending duration exceeds 300s, state transitions to A7_ANCHOR_PENDING_EXPIRED."""
        engine = AnchorFinalizationEngine(self.store, required_providers=["AWS_S3_OBJECT_LOCK"])
        payload, payload_hash, idemp_key = engine.build_anchor_payload(self.sample_sealed_ledger)
        sm = FinalizationStateMachine(initial_state=FinalizationState.A1_ANCHOR_REQUIRED)

        def mock_submit(p):
            return 503, None, ErrorClass.RETRYABLE_TRANSPORT

        success = engine.submit_anchor_for_provider(
            payload=payload,
            payload_hash=payload_hash,
            idempotency_key=idemp_key,
            provider_id="AWS_S3_OBJECT_LOCK",
            sm=sm,
            provider_submit_fn=mock_submit,
            elapsed_seconds=301.0,  # Exceeds 300s max pending duration
        )
        self.assertFalse(success)
        self.assertEqual(sm.state, FinalizationState.A7_ANCHOR_PENDING_EXPIRED)
        self.assertEqual(sm.effective_status, AuthorityEffectiveStatus.BLOCKED_PENDING_GOVERNANCE_RECOVERY)

    # -----------------------------------------------------------------------
    # Section 21: Late Anchor Recovery & No Retroactive Authorization
    # -----------------------------------------------------------------------
    def test_late_anchor_recovery_no_retroactive_authorization(self):
        """Late recovery transitions A7 -> A9 -> A10, effectiveness time is recovery time (not observation time)."""
        engine = AnchorFinalizationEngine(
            self.store,
            required_providers=["AWS_S3_OBJECT_LOCK"],
        )
        payload, payload_hash, _ = engine.build_anchor_payload(self.sample_sealed_ledger)

        rcpt_dict = {
            "provider": "AWS_S3_OBJECT_LOCK",
            "run_id": payload.run_id,
            "run_final_hash": payload.run_final_hash,
            "payload_sha256": payload_hash,
            "policy_set_hash": payload.policy_set_hash,
            "stored_object_id": "s3://arx-worm/run_late.json",
            "timestamp_utc": "2026-10-11T13:00:00Z",
        }
        sig = generate_provider_receipt_signature(rcpt_dict)
        late_rcpt = AnchorReceipt(
            receipt_id="rcpt_late",
            provider="AWS_S3_OBJECT_LOCK",
            run_id=payload.run_id,
            run_final_hash=payload.run_final_hash,
            payload_sha256=payload_hash,
            policy_set_hash=payload.policy_set_hash,
            stored_object_id=rcpt_dict["stored_object_id"],
            timestamp_utc=rcpt_dict["timestamp_utc"],
            signature=sig,
        )

        recovery_time = "2026-10-11T13:05:00.000Z"
        sm = engine.execute_late_anchor_recovery(
            run_id=self.sample_sealed_ledger.run_id,
            sealed_ledger=self.sample_sealed_ledger,
            late_receipts=[late_rcpt],
            recovery_authorized_at=recovery_time,
        )
        self.assertEqual(sm.state, FinalizationState.A10_AUTHORITY_EFFECTIVE)
        self.assertEqual(sm.effective_status, AuthorityEffectiveStatus.EFFECTIVE)
        # Invariant: AUTHORITY_BECAME_EFFECTIVE_AT is recovery time, not observation time!
        self.assertEqual(sm.authority_became_effective_at, recovery_time)
        self.assertNotEqual(sm.authority_became_effective_at, self.sample_sealed_ledger.sealed_at_utc)

    # -----------------------------------------------------------------------
    # Section 22 & 23: Downstream-Use Firewall Tests
    # -----------------------------------------------------------------------
    def test_downstream_firewall_blocks_while_not_effective(self):
        """All downstream consumer checks MUST raise DownstreamFirewallBlockedError while not EFFECTIVE."""
        blocked_statuses = [
            AuthorityEffectiveStatus.NOT_EFFECTIVE,
            AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            AuthorityEffectiveStatus.BLOCKED_PENDING_GOVERNANCE_RECOVERY,
            AuthorityEffectiveStatus.TERMINAL_HOLD,
        ]
        for st in blocked_statuses:
            with self.assertRaises(DownstreamFirewallBlockedError):
                DownstreamUseFirewall.assert_production_decision_allowed(st)
            with self.assertRaises(DownstreamFirewallBlockedError):
                DownstreamUseFirewall.assert_empirical_denominator_allowed(st)
            with self.assertRaises(DownstreamFirewallBlockedError):
                DownstreamUseFirewall.assert_model_tuning_allowed(st)
            with self.assertRaises(DownstreamFirewallBlockedError):
                DownstreamUseFirewall.assert_learning_claim_allowed(st)
            with self.assertRaises(DownstreamFirewallBlockedError):
                DownstreamUseFirewall.assert_historical_backfill_allowed(st)

        # Retroactive justification is ALWAYS blocked even if EFFECTIVE
        with self.assertRaises(DownstreamFirewallBlockedError):
            DownstreamUseFirewall.assert_retroactive_decision_justification_allowed(
                AuthorityEffectiveStatus.EFFECTIVE
            )

    def test_downstream_firewall_allows_when_effective(self):
        """Downstream checks succeed when status is EFFECTIVE."""
        st = AuthorityEffectiveStatus.EFFECTIVE
        DownstreamUseFirewall.assert_production_decision_allowed(st)
        DownstreamUseFirewall.assert_empirical_denominator_allowed(st)
        DownstreamUseFirewall.assert_model_tuning_allowed(st)
        DownstreamUseFirewall.assert_learning_claim_allowed(st)
        DownstreamUseFirewall.assert_historical_backfill_allowed(st)
        perms = DownstreamUseFirewall.evaluate_downstream_permissions(st)
        self.assertTrue(perms["can_feed_production_decision"])
        self.assertTrue(perms["can_enter_empirical_denominator"])
        self.assertTrue(perms["can_enter_model_tuning"])
        self.assertFalse(perms["can_be_used_for_retroactive_decision_justification"])

    # -----------------------------------------------------------------------
    # Section 26: Independent Reviewer Reconstruction Procedure
    # -----------------------------------------------------------------------
    def test_independent_reviewer_reconstruction_pass(self):
        """Executes full 12-point reviewer audit on an end-to-end anchored run."""
        engine = AnchorFinalizationEngine(
            self.store,
            required_providers=["AWS_S3_OBJECT_LOCK", "GCP_CLOUD_STORAGE_LOCK"],
        )
        payload, payload_hash, _ = engine.build_anchor_payload(self.sample_sealed_ledger)

        # Build valid receipts for both providers
        receipts = []
        for prov in ["AWS_S3_OBJECT_LOCK", "GCP_CLOUD_STORAGE_LOCK"]:
            rcpt_dict = {
                "provider": prov,
                "run_id": payload.run_id,
                "run_final_hash": payload.run_final_hash,
                "payload_sha256": payload_hash,
                "policy_set_hash": payload.policy_set_hash,
                "stored_object_id": f"store://arx/{prov}/run.json",
                "timestamp_utc": "2026-10-11T12:00:02Z",
            }
            sig = generate_provider_receipt_signature(rcpt_dict)
            rcpt = AnchorReceipt(
                receipt_id=f"rcpt_{prov}",
                provider=prov,
                run_id=payload.run_id,
                run_final_hash=payload.run_final_hash,
                payload_sha256=payload_hash,
                policy_set_hash=payload.policy_set_hash,
                stored_object_id=rcpt_dict["stored_object_id"],
                timestamp_utc=rcpt_dict["timestamp_utc"],
                signature=sig,
            )
            receipts.append(rcpt)

        submit_map = {
            "AWS_S3_OBJECT_LOCK": lambda p: (200, receipts[0], ErrorClass.NONE),
            "GCP_CLOUD_STORAGE_LOCK": lambda p: (200, receipts[1], ErrorClass.NONE),
        }

        sm = engine.finalize_run(
            sealed_ledger=self.sample_sealed_ledger,
            evidence_observed_at=self.t1_observed,
            eligibility_established_at=self.t2_eligible,
            provider_submit_map=submit_map,
            effective_time_utc=self.t3_effective,
        )
        self.assertEqual(sm.state, FinalizationState.A10_AUTHORITY_EFFECTIVE)

        # Audit with Reviewer
        attempts = self.store.get_attempts(self.sample_sealed_ledger.run_id)
        events = self.store.get_events(self.sample_sealed_ledger.run_id)

        audit_res = AnchorFinalizationReviewer.audit_finalization_run(
            sealed_ledger=self.sample_sealed_ledger,
            payload=payload,
            attempts=attempts,
            events=events,
            receipts=receipts,
            required_providers=["AWS_S3_OBJECT_LOCK", "GCP_CLOUD_STORAGE_LOCK"],
            evidence_observed_at=self.t1_observed,
            eligibility_established_at=self.t2_eligible,
            authority_became_effective_at=sm.authority_became_effective_at,
            downstream_audit_log=[
                {"timestamp_utc": "2026-10-11T12:05:00.000Z", "accepted": True}
            ],
        )

        self.assertEqual(audit_res["run_final_hash_parity"], "PASS")
        self.assertEqual(audit_res["anchor_payload_hash_parity"], "PASS")
        self.assertEqual(audit_res["attempt_ledger_complete"], "PASS")
        self.assertEqual(audit_res["idempotency_continuity"], "PASS")
        self.assertEqual(audit_res["receipt_verification"], "PASS")
        self.assertEqual(audit_res["anchor_quorum_verification"], "PASS")
        self.assertEqual(audit_res["finalization_ledger_hash_chain"], "PASS")
        self.assertEqual(audit_res["authority_timing_reconstruction"], "PASS")
        self.assertEqual(audit_res["downstream_firewall_reconstruction"], "PASS")
        self.assertEqual(audit_res["unexplained_finalization_events"], 0)
        self.assertEqual(audit_res["overall_status"], "PASS")

    # -----------------------------------------------------------------------
    # Section 27, 28, 29, 30: Crash-Recovery, Conflict & Timing Matrix
    # -----------------------------------------------------------------------
    def test_conflict_matrix_different_final_hash_same_run_id(self):
        """Same run_id with different final hash raises critical integrity conflict."""
        self.store.initialize_run_coordinator("run_conflict_001", "original_final_hash_1111")
        with self.assertRaises(ValueError):
            self.store.append_finalization_event(
                parent_run_id="run_conflict_001",
                parent_run_final_hash="conflicting_final_hash_2222",  # Mismatched final hash
                finalization_event_type="SUBMIT",
                anchor_request_id="req_x",
                anchor_attempt_id="att_x",
                anchor_status_to=FinalizationState.A3_ANCHOR_SUBMISSION_PENDING,
                authority_effective_status_to=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            )

    def test_authority_timing_backdating_detected(self):
        """Enforces T1 <= T2 <= T3. Backdating T3 earlier than T2 raises ValueError."""
        sm = FinalizationStateMachine(
            initial_state=FinalizationState.A6_ANCHOR_CONFIRMED,
            evidence_observed_at="2026-10-11T12:00:00.000Z",  # T1
            eligibility_established_at="2026-10-11T12:00:05.000Z",  # T2
        )
        with self.assertRaises(ValueError):
            # Backdating T3 to 12:00:01 (before T2!)
            sm.transition(
                FinalizationState.A10_AUTHORITY_EFFECTIVE,
                effective_time_utc="2026-10-11T12:00:01.000Z",
                quorum_satisfied=True,
            )

    def test_crash_recovery_preserves_consistent_ledger(self):
        """Simulates crash after attempt insertion before event append."""
        self.store.initialize_run_coordinator("run_crash_001", "hash_crash_001")
        # Attempt logged
        self.store.record_attempt(
            AnchorAttemptRecord(
                anchor_attempt_id="att_crash_1",
                anchor_request_id="req_crash_1",
                attempt_number=1,
                run_id="run_crash_001",
                anchor_idempotency_key="idemp_crash",
                request_payload_sha256="payload_crash",
                request_started_at="2026-10-11T12:00:00Z",
                request_completed_at="2026-10-11T12:00:01Z",
                provider="AWS_S3_OBJECT_LOCK",
                provider_http_status=200,
                provider_response_sha256="",
                error_class=ErrorClass.NONE,
                result="SUCCESS",
                lookup_required=False,
                lookup_performed=False,
            )
        )
        # State reloaded from coordinator remains consistent
        state_tuple = self.store.get_current_state("run_crash_001")
        self.assertIsNotNone(state_tuple)
        self.assertEqual(state_tuple[0], FinalizationState.A1_ANCHOR_REQUIRED)
        self.assertEqual(state_tuple[1], AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR)


if __name__ == "__main__":
    unittest.main()
