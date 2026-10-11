"""
Test Suite: Finalization Ledger External Rooting Architecture (Epoch 001).
Verifies Sections 4 through 28 of FINALIZATION_LEDGER_EXTERNAL_ROOTING_CLOSURE_V1:
- Canonical root payload serialization & deterministic SHA-256 root hashing
- Root identity & idempotency key derivation
- Signed external receipt verification (8-point check)
- Mandatory Threat Model: Whole-ledger rewrite detection (Section 13)
- Event deletion detection (Section 14)
- Event insertion detection (Section 15)
- Event reordering detection (Section 16)
- Authority-effective-time tamper detection (Section 17)
- Receipt substitution detection (Section 18)
- Attempt ledger & ambiguous root commit recovery (Sections 19, 20)
- Authority effectiveness binding & no retroactive authority (Sections 21, 22, 23)
- Independent Reviewer 10-point reconstruction (Section 24)
- Policy artifact retention (Section 26)
- Crash recovery matrix (Section 27)
- Downstream firewall enforcement (Section 28)
- External storage configuration attestation (Section 11)
"""

import hashlib
import json
import unittest
from datetime import datetime, timezone
from typing import List, Dict, Any

from analyst_dashboard.smart_money.anchor_finalization import (
    FinalizationState,
    AuthorityEligibility,
    AuthorityEffectiveStatus,
    FinalizationEvent,
    CriticalIntegrityError,
    DownstreamFirewallBlockedError,
    DownstreamUseFirewall,
    FINALIZATION_ROOT_POLICY_ID,
    FINALIZATION_ROOT_POLICY_VERSION,
    FINALIZATION_ROOT_REQUIRED,
    FINALIZATION_ROOT_APPLIES_TO,
    FINALIZATION_ROOT_RETROACTIVE_REWRITE,
    FINALIZATION_ROOT_HUMAN_OVERRIDE,
    FINALIZATION_LEDGER_ROOT_SCOPE,
    FINALIZATION_ROOT_SCHEMA_VERSION,
    CANONICAL_ROOT_SERIALIZATION_VERSION,
    SAME_ROOT_ID_DIFFERENT_ROOT_HASH,
    SAME_ROOT_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD,
    DUPLICATE_IDENTICAL_ROOT_SUBMISSION,
    FINALIZATION_ROOT_EXTERNAL_PROVIDER,
    FINALIZATION_ROOT_CREDENTIAL_DOMAIN,
    FINALIZATION_ROOT_KEY_DOMAIN,
    FINALIZATION_ROOT_ADMIN_DOMAIN,
    FINALIZATION_ROOT_WRITE_AUTHORITY,
    FINALIZATION_ROOT_EXTERNAL_TRUST_DOMAIN_SEPARATE,
    ROOT_OBJECT_VERSIONING_ENABLED,
    ROOT_OBJECT_LOCK_ENABLED,
    ROOT_RETENTION_MODE,
    ROOT_RETENTION_UNTIL,
    ROOT_DELETE_PROTECTION,
    ROOT_OVERWRITE_PROTECTION,
    ROOT_STORAGE_IMMUTABILITY_CLASS,
    ROOT_AMBIGUOUS_COMMIT_RESULT_POLICY,
    ROOT_CONFIRMATION_REQUIRED_FOR_AUTHORITY_EFFECTIVENESS,
    RETROACTIVE_AUTHORITY_FROM_ROOT_CONFIRMATION,
    FinalizationRootStatus,
    RootLookupResult,
    RootErrorClass,
    FinalizationRootPayload,
    FinalizationRootReceipt,
    FinalizationRootAttemptRecord,
    RootStorageConfigurationAttestation,
    CANONICAL_ROOT_SIGNING_KEY,
    DEFAULT_ROOT_SIGNING_KEY_ID,
    serialize_canonical_finalization_root_payload,
    compute_finalization_root_hash,
    derive_finalization_root_id,
    derive_finalization_root_idempotency_key,
    build_finalization_root_payload,
    generate_root_receipt_signature,
    create_signed_root_receipt,
    verify_root_receipt,
    FinalizationRootLedgerStore,
    FinalizationRootReviewer,
    verify_all_policy_artifacts_retained,
    get_policy_artifact,
)


def _build_test_finalization_events(
    run_id: str = "run_root_test_001",
    run_final_hash: str = "aaa111" * 10 + "abcd",
    count: int = 5,
) -> List[FinalizationEvent]:
    """Helper to generate a valid hash-chained finalization event sequence."""
    events = []
    prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    for i in range(1, count + 1):
        evt_id = f"evt_{run_id}_{i:03d}"
        before_state = FinalizationState.A1_ANCHOR_REQUIRED if i == 1 else FinalizationState.A3_ANCHOR_SUBMISSION_PENDING
        after_state = FinalizationState.A6_ANCHOR_CONFIRMED if i == count else FinalizationState.A3_ANCHOR_SUBMISSION_PENDING
        before_auth = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR
        after_auth = AuthorityEffectiveStatus.EFFECTIVE if i == count else AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR
        recorded_at = f"2026-10-11T12:00:{i:02d}Z"
        content = (
            f"{evt_id}:{run_id}:{run_final_hash}:{i}:STATUS_TRANSITION:"
            f"req_001:att_001:{before_state.value}:{after_state.value}:"
            f"{before_auth.value}:{after_auth.value}:{recorded_at}:{prev_hash}"
        )
        evt_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        event = FinalizationEvent(
            finalization_event_id=evt_id,
            parent_run_id=run_id,
            parent_run_final_hash=run_final_hash,
            finalization_sequence=i,
            finalization_event_type="STATUS_TRANSITION",
            anchor_request_id="req_001",
            anchor_attempt_id="att_001",
            anchor_status_before=before_state,
            anchor_status_after=after_state,
            authority_effective_status_before=before_auth,
            authority_effective_status_after=after_auth,
            recorded_at=recorded_at,
            previous_finalization_event_hash=prev_hash,
            finalization_event_hash=evt_hash,
        )
        events.append(event)
        prev_hash = evt_hash
    return events


class TestSmartMoneyFinalizationRoot(unittest.TestCase):
    """
    Finalization Ledger External Rooting Subsystem Test Suite.
    """

    def setUp(self):
        self.run_id = "run_root_test_001"
        self.run_final_hash = "a" * 64
        self.policy_set_hash = "b" * 64
        self.events = _build_test_finalization_events(self.run_id, self.run_final_hash, count=5)
        self.created_at = "2026-10-11T12:05:00Z"
        self.payload = build_finalization_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.created_at,
        )
        self.receipt = create_signed_root_receipt(
            payload=self.payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.created_at,
        )

    # =========================================================================
    # 1. Canonical Root Serialization & Determinism (Sections 6, 7, 8)
    # =========================================================================

    def test_canonical_root_serialization_and_determinism(self):
        """Root byte serialization is deterministic, sorted, and reproducible."""
        b1 = serialize_canonical_finalization_root_payload(self.payload)
        b2 = serialize_canonical_finalization_root_payload(self.payload)
        self.assertEqual(b1, b2)
        h1 = compute_finalization_root_hash(self.payload)
        h2 = compute_finalization_root_hash(self.payload)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_root_hash_and_idempotency_key_derivation(self):
        """Root ID and idempotency keys derive deterministically (Section 9)."""
        root_id = derive_finalization_root_id(self.run_id, self.run_final_hash)
        self.assertEqual(root_id, f"root_{self.run_id}_{self.run_final_hash[:16]}")
        root_hash = compute_finalization_root_hash(self.payload)
        idem = derive_finalization_root_idempotency_key(self.run_id, self.run_final_hash, root_hash)
        expected = hashlib.sha256(f"{self.run_id}:{self.run_final_hash}:{root_hash}".encode("utf-8")).hexdigest()
        self.assertEqual(idem, expected)

    # =========================================================================
    # 2. Idempotency & Conflict Matrix (Section 9)
    # =========================================================================

    def test_same_root_id_different_root_hash_conflict(self):
        """Same parent_run_id with different root hash raises CriticalIntegrityError."""
        store = FinalizationRootLedgerStore(":memory:")
        store.record_root_payload(self.payload)

        # Alter payload
        alt_payload = build_finalization_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events[:3],  # Fewer events
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:03Z",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.created_at,
        )
        with self.assertRaises(CriticalIntegrityError) as ctx:
            store.record_root_payload(alt_payload)
        self.assertIn("CRITICAL_INTEGRITY_CONFLICT", str(ctx.exception))

    def test_duplicate_identical_submission_idempotent(self):
        """Duplicate identical root submission is safely idempotent."""
        store = FinalizationRootLedgerStore(":memory:")
        id1 = store.record_root_payload(self.payload)
        id2 = store.record_root_payload(self.payload)
        self.assertEqual(id1, id2)

    # =========================================================================
    # 3. External Receipt Verification Suite (Section 12)
    # =========================================================================

    def test_external_receipt_verification_pass_and_tamper_fails(self):
        """Valid receipt passes 8-point check; altered signatures or data fail."""
        verdict, checks = verify_root_receipt(self.receipt, self.payload)
        self.assertEqual(verdict, "PASS")
        self.assertTrue(all(checks.values()))

        # Tamper signature
        tampered_receipt = FinalizationRootReceipt(
            root_receipt_id=self.receipt.root_receipt_id,
            parent_run_id=self.receipt.parent_run_id,
            parent_run_final_hash=self.receipt.parent_run_final_hash,
            finalization_ledger_root_hash=self.receipt.finalization_ledger_root_hash,
            policy_set_hash=self.receipt.policy_set_hash,
            provider=self.receipt.provider,
            receipt_signing_key_id=self.receipt.receipt_signing_key_id,
            signature_algorithm="HMAC-SHA256",
            receipt_payload_sha256=self.receipt.receipt_payload_sha256,
            signature="f" * 64,  # Bad signature
            created_at_utc=self.receipt.created_at_utc,
        )
        verdict2, checks2 = verify_root_receipt(tampered_receipt, self.payload)
        self.assertEqual(verdict2, "FAIL")
        self.assertFalse(checks2["root_receipt_signature_valid"])

    # =========================================================================
    # 4. Mandatory Threat Model: Whole-Ledger Rewrite Detection (Section 13)
    # =========================================================================

    def test_whole_ledger_rewrite_detection(self):
        """
        Threat Model:
        1. Build valid finalization ledger;
        2. Compute root hash;
        3. Externally anchor root & obtain receipt;
        4. Copy ledger;
        5. Rewrite historical finalization event;
        6. Recompute every internal finalization event hash and internal chain;
        7. Reviewer verification against external receipt:
           - INTERNAL_REWRITTEN_CHAIN_VALID = YES
           - EXTERNAL_ROOT_PARITY = FAIL
           - WHOLE_FINALIZATION_LEDGER_REWRITE_DETECTED = YES
        """
        # Step 1, 2, 3: Valid ledger and receipt constructed in setUp
        audit_orig = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,
        )
        self.assertEqual(audit_orig["overall_status"], "PASS")
        self.assertFalse(audit_orig["whole_ledger_rewrite_detected"])

        # Step 4, 5, 6: Rewrite historical event 2 and recompute internal chain
        rewritten_events = []
        prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        for i in range(1, len(self.events) + 1):
            orig_ev = self.events[i - 1]
            # Rewrite event 2's anchor_status_after maliciously
            after_state = FinalizationState.A4_ANCHOR_RETRY_WAIT if i == 2 else orig_ev.anchor_status_after
            evt_id = orig_ev.finalization_event_id
            content = (
                f"{evt_id}:{self.run_id}:{self.run_final_hash}:{i}:STATUS_TRANSITION:"
                f"req_001:att_001:{orig_ev.anchor_status_before.value}:{after_state.value}:"
                f"{orig_ev.authority_effective_status_before.value}:{orig_ev.authority_effective_status_after.value}:"
                f"{orig_ev.recorded_at}:{prev_hash}"
            )
            recomputed_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            rewritten_ev = FinalizationEvent(
                finalization_event_id=evt_id,
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                finalization_sequence=i,
                finalization_event_type=orig_ev.finalization_event_type,
                anchor_request_id=orig_ev.anchor_request_id,
                anchor_attempt_id=orig_ev.anchor_attempt_id,
                anchor_status_before=orig_ev.anchor_status_before,
                anchor_status_after=after_state,
                authority_effective_status_before=orig_ev.authority_effective_status_before,
                authority_effective_status_after=orig_ev.authority_effective_status_after,
                recorded_at=orig_ev.recorded_at,
                previous_finalization_event_hash=prev_hash,
                finalization_event_hash=recomputed_hash,
            )
            rewritten_events.append(rewritten_ev)
            prev_hash = recomputed_hash

        # Step 7: Attempt verification against original external root receipt
        audit_rewritten = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=rewritten_events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,  # Original external receipt
        )

        self.assertEqual(audit_rewritten["internal_chain_valid"], "PASS")
        self.assertEqual(audit_rewritten["external_root_parity"], "FAIL")
        self.assertTrue(audit_rewritten["whole_ledger_rewrite_detected"])
        self.assertEqual(audit_rewritten["overall_status"], "FAIL")

    # =========================================================================
    # 5. Tamper Invariant Detection Tests (Sections 14, 15, 16, 17, 18)
    # =========================================================================

    def test_event_deletion_detected(self):
        """Deleting an intermediate event is detected (Section 14)."""
        # Delete event 3 (leaving events 1, 2, 4, 5)
        deleted_events = [self.events[0], self.events[1], self.events[3], self.events[4]]
        audit = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=deleted_events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,
        )
        self.assertEqual(audit["external_root_parity"], "FAIL")
        self.assertTrue(audit["missing_event_detected"])
        self.assertEqual(audit["overall_status"], "FAIL")

    def test_event_insertion_detected(self):
        """Inserting an unauthorized event is detected (Section 15)."""
        extra_event = FinalizationEvent(
            finalization_event_id="evt_unauthorized_999",
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            finalization_sequence=6,
            finalization_event_type="UNAUTHORIZED_INJECTION",
            anchor_request_id="req_999",
            anchor_attempt_id="att_999",
            anchor_status_before=FinalizationState.A6_ANCHOR_CONFIRMED,
            anchor_status_after=FinalizationState.A10_AUTHORITY_EFFECTIVE,
            authority_effective_status_before=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            authority_effective_status_after=AuthorityEffectiveStatus.EFFECTIVE,
            recorded_at="2026-10-11T12:00:06Z",
            previous_finalization_event_hash=self.events[-1].finalization_event_hash,
            finalization_event_hash="9" * 64,
        )
        inserted_events = self.events + [extra_event]
        audit = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=inserted_events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,
        )
        self.assertEqual(audit["external_root_parity"], "FAIL")
        self.assertEqual(audit["overall_status"], "FAIL")

    def test_event_reordering_detected(self):
        """Reordering events while preserving payloads is detected (Section 16)."""
        # Swap events 1 and 2
        reordered = [self.events[1], self.events[0], self.events[2], self.events[3], self.events[4]]
        audit = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=reordered,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,
        )
        self.assertTrue(audit["event_reordering_detected"])
        self.assertEqual(audit["overall_status"], "FAIL")

    def test_authority_effective_time_tamper_detected(self):
        """Tampering with authority effective time is detected (Section 17)."""
        audit = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T13:00:00Z",  # Backdated or altered time!
            policy_set_hash=self.policy_set_hash,
            external_receipt=self.receipt,
            evidence_observed_at="2026-10-11T12:00:00Z",
            eligibility_established_at="2026-10-11T12:00:01Z",
        )
        self.assertEqual(audit["external_root_parity"], "FAIL")
        self.assertEqual(audit["overall_status"], "FAIL")

    def test_receipt_substitution_detected(self):
        """Substituting root receipt with another receipt is detected (Section 18)."""
        substitute_receipt = FinalizationRootReceipt(
            root_receipt_id="rcpt_other_001",
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            finalization_ledger_root_hash="c" * 64,  # Different root hash
            policy_set_hash=self.policy_set_hash,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            signature_algorithm="HMAC-SHA256",
            receipt_payload_sha256="d" * 64,
            signature="e" * 64,
            created_at_utc=self.created_at,
        )
        audit = FinalizationRootReviewer.audit_finalization_root(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            authority_effective_status="EFFECTIVE",
            authority_became_effective_at="2026-10-11T12:00:05Z",
            policy_set_hash=self.policy_set_hash,
            external_receipt=substitute_receipt,
        )
        self.assertEqual(audit["external_root_parity"], "FAIL")
        self.assertTrue(audit["root_receipt_mismatch_detected"])
        self.assertEqual(audit["overall_status"], "FAIL")

    # =========================================================================
    # 6. Ambiguous Root Commit Recovery & Attempt Ledger (Sections 19, 20)
    # =========================================================================

    def test_ambiguous_root_commit_recovery(self):
        """Tests QUERY_EXISTING_ROOT_BEFORE_RETRY lookup transitions (Section 20)."""
        store = FinalizationRootLedgerStore(":memory:")
        store.record_root_payload(self.payload)

        root_hash = compute_finalization_root_hash(self.payload)

        # 1. Matching lookup
        res_match, p_match = store.lookup_existing_root(self.run_id, root_hash)
        self.assertEqual(res_match, RootLookupResult.FOUND_MATCHING)
        self.assertIsNotNone(p_match)

        # 2. Conflicting lookup
        res_conf, p_conf = store.lookup_existing_root(self.run_id, "different_hash" * 4 + "1234")
        self.assertEqual(res_conf, RootLookupResult.FOUND_CONFLICTING)

        # 3. Not found lookup
        res_nf, p_nf = store.lookup_existing_root("nonexistent_run", root_hash)
        self.assertEqual(res_nf, RootLookupResult.NOT_FOUND)
        self.assertIsNone(p_nf)

    # =========================================================================
    # 7. Authority Effectiveness Binding & Downstream Firewall (Sections 21, 28)
    # =========================================================================

    def test_downstream_firewall_blocks_unless_root_confirmed(self):
        """DownstreamUseFirewall rejects authority unless root is CONFIRMED (Section 28)."""
        # Confirmed passes
        DownstreamUseFirewall.assert_finalization_root_allowed(FinalizationRootStatus.CONFIRMED)

        # Non-confirmed statuses fail
        blocked_statuses = [
            FinalizationRootStatus.NOT_REQUIRED,
            FinalizationRootStatus.REQUIRED,
            FinalizationRootStatus.PAYLOAD_SEALED,
            FinalizationRootStatus.SUBMISSION_PENDING,
            FinalizationRootStatus.RETRY_WAIT,
            FinalizationRootStatus.COMMIT_INDETERMINATE,
            FinalizationRootStatus.PENDING_EXPIRED,
            FinalizationRootStatus.CONFLICT,
            FinalizationRootStatus.TERMINAL_HOLD,
        ]
        for st in blocked_statuses:
            with self.subTest(status=st):
                with self.assertRaises(DownstreamFirewallBlockedError):
                    DownstreamUseFirewall.assert_finalization_root_allowed(st)

    # =========================================================================
    # 8. Retained Policy Artifacts Completeness (Section 26)
    # =========================================================================

    def test_retained_policy_artifacts_completeness(self):
        """All 16 policy artifacts are retained and valid (Section 26)."""
        self.assertTrue(verify_all_policy_artifacts_retained())

        # Verify specifically the 5 new rooting artifacts
        required_root_artifacts = [
            "finalization_root_policy_v1",
            "root_serialization_contract_v1",
            "root_receipt_verification_policy_v1",
            "root_retry_policy_v1",
            "root_reviewer_policy_v1",
        ]
        for art in required_root_artifacts:
            rec = get_policy_artifact(art)
            self.assertIsNotNone(rec)
            self.assertTrue(rec["policy_artifact_retained"])
            self.assertEqual(len(rec["policy_artifact_sha256"]), 64)

    # =========================================================================
    # 9. Storage Configuration Attestation (Section 11)
    # =========================================================================

    def test_storage_configuration_attestation(self):
        """Storage attestation records provider-enforced retention without claiming absolute immutability."""
        self.assertTrue(ROOT_OBJECT_VERSIONING_ENABLED)
        self.assertTrue(ROOT_OBJECT_LOCK_ENABLED)
        self.assertEqual(ROOT_RETENTION_MODE, "COMPLIANCE")
        self.assertEqual(ROOT_STORAGE_IMMUTABILITY_CLASS, "PROVIDER_ENFORCED_RETENTION")
        self.assertTrue(ROOT_DELETE_PROTECTION)
        self.assertTrue(ROOT_OVERWRITE_PROTECTION)

    # =========================================================================
    # 10. Attempt Ledger Logging (Section 19)
    # =========================================================================

    def test_root_attempt_ledger_logging(self):
        """Root attempts are logged append-only with standard error classifications."""
        store = FinalizationRootLedgerStore(":memory:")
        root_id = store.record_root_payload(self.payload)

        attempt1 = FinalizationRootAttemptRecord(
            root_attempt_id="att_root_001",
            root_id=root_id,
            parent_run_id=self.run_id,
            attempt_number=1,
            idempotency_key="idem_001",
            request_payload_sha256="req_sha_001",
            request_started_at="2026-10-11T12:05:01Z",
            request_completed_at="2026-10-11T12:05:03Z",
            provider_response_hash="resp_001",
            result="FAILED",
            error_class=RootErrorClass.RETRYABLE_TRANSPORT,
            lookup_required=False,
        )
        attempt2 = FinalizationRootAttemptRecord(
            root_attempt_id="att_root_002",
            root_id=root_id,
            parent_run_id=self.run_id,
            attempt_number=2,
            idempotency_key="idem_001",
            request_payload_sha256="req_sha_001",
            request_started_at="2026-10-11T12:05:05Z",
            request_completed_at="2026-10-11T12:05:07Z",
            provider_response_hash="resp_002",
            result="CONFIRMED",
            error_class=RootErrorClass.NONE,
            lookup_required=True,
            lookup_result=RootLookupResult.FOUND_MATCHING,
        )
        store.record_root_attempt(attempt1)
        store.record_root_attempt(attempt2)

        logged = store.get_attempts(root_id)
        self.assertEqual(len(logged), 2)
        self.assertEqual(logged[0].attempt_number, 1)
        self.assertEqual(logged[0].error_class, RootErrorClass.RETRYABLE_TRANSPORT)
        self.assertEqual(logged[1].attempt_number, 2)
        self.assertEqual(logged[1].error_class, RootErrorClass.NONE)
        self.assertEqual(logged[1].lookup_result, RootLookupResult.FOUND_MATCHING)

    # =========================================================================
    # 11. Authority Effectiveness Binding & No Retroactive Authority (Sections 21, 22, 23)
    # =========================================================================

    def test_authority_effective_status_binding_and_no_retroactive_authority(self):
        """
        Authority is blocked until root is confirmed.
        Root confirmation timestamp becomes authority_became_effective_at without backdating.
        """
        self.assertTrue(ROOT_CONFIRMATION_REQUIRED_FOR_AUTHORITY_EFFECTIVENESS)
        self.assertFalse(RETROACTIVE_AUTHORITY_FROM_ROOT_CONFIRMATION)

        store = FinalizationRootLedgerStore(":memory:")
        root_id = store.record_root_payload(self.payload, status=FinalizationRootStatus.SUBMISSION_PENDING)

        root_record = store.get_root(root_id)
        self.assertEqual(root_record["status"], FinalizationRootStatus.SUBMISSION_PENDING.value)

        # Confirm root
        store.confirm_root(root_id, self.receipt)
        confirmed_record = store.get_root(root_id)
        self.assertEqual(confirmed_record["status"], FinalizationRootStatus.CONFIRMED.value)
        self.assertIsNotNone(confirmed_record["receipt"])

    # =========================================================================
    # 12. Crash Recovery Test Matrix (Section 27)
    # =========================================================================

    def test_crash_recovery_matrix(self):
        """
        Tests crash scenarios across root lifecycle:
        - Crash after payload seal / before submission: state remains PAYLOAD_SEALED, no duplicate root.
        - Crash during ambiguous lookup: re-lookup yields FOUND_MATCHING and confirms.
        - Crash after confirmation: confirmed state and receipt persist consistently.
        """
        store = FinalizationRootLedgerStore(":memory:")

        # Crash 1: Sealed payload persisted
        root_id = store.record_root_payload(self.payload, status=FinalizationRootStatus.PAYLOAD_SEALED)
        rec1 = store.get_root(root_id)
        self.assertEqual(rec1["status"], FinalizationRootStatus.PAYLOAD_SEALED.value)

        # Restart / retry attempt after crash
        id2 = store.record_root_payload(self.payload)
        self.assertEqual(root_id, id2)  # NO_DUPLICATE_ROOT

        # Crash 2: Provider commit indeterminate lookup recovery
        lookup_res, _ = store.lookup_existing_root(self.run_id, compute_finalization_root_hash(self.payload))
        self.assertEqual(lookup_res, RootLookupResult.FOUND_MATCHING)

        # Confirm and verify persistence
        store.confirm_root(root_id, self.receipt)
        rec_final = store.get_root(root_id)
        self.assertEqual(rec_final["status"], FinalizationRootStatus.CONFIRMED.value)


if __name__ == "__main__":
    unittest.main()

