"""
Test Suite: Two-Phase Finalization Root Semantics (Epoch 001).
Verifies Sections 4 through 28 of TWO_PHASE_FINALIZATION_ROOT_SEMANTICS_CLOSURE_V1:
- Phase R1 (PRE_AUTHORITY_ADMISSION_ROOT) pre-authority semantics & non-circularity
- Prohibition of root self-reference and current receipt circularity
- External receipt-driven authority transition trigger & immutable event emission
- Authority transition idempotency, atomicity, and clock-boundary monotonicity
- Phase R2 (POST_AUTHORITY_CLOSURE_ROOT) commitments, R1 linkage, and audit finality
- Closure deadline monitoring (300.0s), expiry suspension, and non-retroactivity
- Conflict detection (R1 conflict -> BLOCKED_INTEGRITY_HOLD, R2 conflict -> SUSPENDED_INTEGRITY_HOLD)
- Threat Model: Authority transition rewrite detection (Section 19)
- Threat Model: R1 receipt substitution detection (Section 20)
- Threat Model: Duplicate R1 notification handling (Section 21)
- Threat Model: R1 conflict handling (Section 22)
- Crash recovery matrix across multi-phase process lifecycle (Section 23)
- Independent Reviewer 10-point two-phase reconstruction (Section 25)
- Downstream firewall enforcement across pre-authority, post-authority, and suspension states
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
    RootLookupResult,
    RootPhase,
    PostAuthorityClosureStatus,
    R1ConfirmationStatus,
    FinalizationRootPayload,
    FinalizationRootReceipt,
    AuthorityTransitionRecord,
    PostAuthorityClosureRecord,
    build_r1_root_payload,
    build_r2_root_payload,
    derive_finalization_root_id,
    derive_finalization_root_idempotency_key,
    compute_finalization_root_hash,
    derive_authority_transition_idempotency_key,
    compute_authority_transition_event_hash,
    create_signed_root_receipt,
    verify_r1_root_receipt,
    verify_r2_root_receipt,
    check_authority_transition_authorized,
    FinalizationRootLedgerStore,
    FinalizationRootReviewer,
    DEFAULT_ROOT_SIGNING_KEY_ID,
    POST_AUTHORITY_CLOSURE_MAX_DELAY,
    ROOT_SELF_REFERENCE_PROHIBITED,
    R2_EXPIRY_RETROACTIVELY_REVOKES_PRIOR_VALID_DECISIONS,
    R2_PURPOSE,
    R2_AUTHORITY_GRANT_FUNCTION,
    DUPLICATE_IDENTICAL_AUTHORITY_TRIGGER,
    DUPLICATE_DIFFERENT_AUTHORITY_TRIGGER,
)


def _build_test_events(
    run_id: str = "run_twophase_001",
    run_final_hash: str = "111aaa" * 10 + "1234",
    count: int = 4,
) -> List[FinalizationEvent]:
    """Helper to generate pre-authority finalization events."""
    events = []
    prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    for i in range(1, count + 1):
        evt_id = f"evt_{run_id}_{i:03d}"
        before_state = FinalizationState.A1_ANCHOR_REQUIRED if i == 1 else FinalizationState.A3_ANCHOR_SUBMISSION_PENDING
        after_state = FinalizationState.A6_ANCHOR_CONFIRMED if i == count else FinalizationState.A3_ANCHOR_SUBMISSION_PENDING
        before_auth = AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR
        # In pre-authority R1, authority is NOT yet effective
        after_auth = AuthorityEffectiveStatus.BLOCKED_PENDING_FINALIZATION_ROOT if i == count else AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR
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


class TestSmartMoneyTwoPhaseRoot(unittest.TestCase):
    """
    Two-Phase Finalization Root Semantics Verification Test Suite.
    """

    def setUp(self):
        self.run_id = "run_twophase_001"
        self.run_final_hash = "c" * 64
        self.policy_set_hash = "d" * 64
        self.events = _build_test_events(self.run_id, self.run_final_hash, count=4)
        self.t_eligible = "2026-10-11T12:00:04Z"
        self.t_r1_created = "2026-10-11T12:00:05Z"
        self.t_r1_confirmed = "2026-10-11T12:00:06Z"
        self.t_effective = "2026-10-11T12:00:07Z"
        self.t_r2_created = "2026-10-11T12:00:08Z"
        self.t_r2_confirmed = "2026-10-11T12:00:09Z"

    # =========================================================================
    # 1. Phase R1 Pre-Authority Payload Semantics (Section 4, 5, 28.1)
    # =========================================================================

    def test_r1_pre_authority_payload_semantics(self):
        """R1 contains pre-authority status, null effective time, and intent."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )

        self.assertEqual(r1_payload.root_phase, RootPhase.PRE_AUTHORITY_ADMISSION_ROOT)
        self.assertEqual(
            r1_payload.authority_effective_status,
            AuthorityEffectiveStatus.BLOCKED_PENDING_FINALIZATION_ROOT,
        )
        self.assertIsNone(r1_payload.authority_became_effective_at)
        self.assertEqual(r1_payload.authority_transition_intent, "GRANT_IF_ROOT_CONFIRMED")
        self.assertEqual(
            r1_payload.authority_effectiveness_precondition,
            "R1_ROOT_RECEIPT_CONFIRMED",
        )
        # R2-specific linkage fields must be None in R1
        self.assertIsNone(r1_payload.r1_root_id)
        self.assertIsNone(r1_payload.r1_root_hash)
        self.assertIsNone(r1_payload.r1_receipt_hash)
        self.assertIsNone(r1_payload.authority_transition_event_id)
        self.assertIsNone(r1_payload.authority_transition_event_hash)
        self.assertIsNone(r1_payload.authority_transition_sequence)

    # =========================================================================
    # 2. Self-Reference & Circularity Elimination (Section 6, 7, 28.2)
    # =========================================================================

    def test_root_self_reference_prohibition(self):
        """Current receipt or circular effective state cannot be passed into root payload."""
        self.assertTrue(ROOT_SELF_REFERENCE_PROHIBITED)

        # Passing receipt to R1 builder raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_r1_root_payload(
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                events=self.events,
                authority_eligibility="PASS",
                policy_set_hash=self.policy_set_hash,
                created_at_utc=self.t_r1_created,
                receipt={"dummy": "receipt"},
            )
        self.assertIn("CIRCULAR_ROOT_PAYLOAD_PROHIBITED", str(ctx.exception))

        # Passing authority_effective_status = EFFECTIVE to R1 raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_r1_root_payload(
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                events=self.events,
                authority_eligibility="PASS",
                policy_set_hash=self.policy_set_hash,
                created_at_utc=self.t_r1_created,
                authority_effective_status=AuthorityEffectiveStatus.EFFECTIVE,
            )
        self.assertIn("CIRCULAR_ROOT_PAYLOAD_PROHIBITED", str(ctx.exception))

        # Passing authority_became_effective_at to R1 raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_r1_root_payload(
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                events=self.events,
                authority_eligibility="PASS",
                policy_set_hash=self.policy_set_hash,
                created_at_utc=self.t_r1_created,
                authority_became_effective_at="2026-10-11T12:00:00Z",
            )
        self.assertIn("CIRCULAR_ROOT_PAYLOAD_PROHIBITED", str(ctx.exception))

        # Passing receipt to R2 builder raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_r2_root_payload(
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                events=self.events,
                authority_eligibility="PASS",
                policy_set_hash=self.policy_set_hash,
                r1_root_id="root_001",
                r1_root_hash="hash_001",
                r1_receipt_hash="rcpt_001",
                authority_transition_event_id="evt_001",
                authority_transition_event_hash="hash_evt",
                authority_transition_sequence=5,
                authority_became_effective_at="2026-10-11T12:00:07Z",
                created_at_utc=self.t_r2_created,
                receipt={"future": "receipt"},
            )
        self.assertIn("CIRCULAR_ROOT_PAYLOAD_PROHIBITED", str(ctx.exception))

        # R2 builder missing R1 linkage raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_r2_root_payload(
                parent_run_id=self.run_id,
                parent_run_final_hash=self.run_final_hash,
                events=self.events,
                authority_eligibility="PASS",
                policy_set_hash=self.policy_set_hash,
                r1_root_id="",
                r1_root_hash="hash_001",
                r1_receipt_hash="rcpt_001",
                authority_transition_event_id="evt_001",
                authority_transition_event_hash="hash_evt",
                authority_transition_sequence=5,
                authority_became_effective_at="2026-10-11T12:00:07Z",
                created_at_utc=self.t_r2_created,
            )
        self.assertIn("R2 root payload requires r1_root_id", str(ctx.exception))

    # =========================================================================
    # 3. R1 Receipt-Triggered Authority Transition (Section 8, 9, 28.3)
    # =========================================================================

    def test_r1_receipt_triggered_transition(self):
        """Receipt confirmation strictly triggers immutable authority transition event."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )

        # 1. Verify R1 receipt
        r1_status, r1_err = verify_r1_root_receipt(r1_receipt, r1_payload)
        self.assertEqual(r1_status, R1ConfirmationStatus.CONFIRMED)
        self.assertTrue(all(r1_err.values()))

        # 2. Check transition authorization
        authorized = check_authority_transition_authorized(
            r1_receipt=r1_receipt,
            r1_payload=r1_payload,
            authority_eligibility=AuthorityEligibility.PASS,
            t_eligible_utc=self.t_eligible,
        )
        self.assertTrue(authorized)

        # 3. Record transition in store
        store = FinalizationRootLedgerStore()
        store.record_root_payload(r1_payload)

        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        self.assertEqual(transition.event_type, "AUTHORITY_BECAME_EFFECTIVE")
        self.assertEqual(transition.authority_effective_status, AuthorityEffectiveStatus.EFFECTIVE)
        self.assertEqual(transition.authority_became_effective_at, self.t_effective)
        self.assertEqual(transition.r1_root_id, r1_receipt.root_id)
        self.assertEqual(transition.r1_receipt_hash, r1_receipt.receipt_hash)

        # Verify closure initialized to PENDING
        closure = store.get_closure_record(self.run_id)
        self.assertIsNotNone(closure)
        self.assertEqual(closure.closure_status, PostAuthorityClosureStatus.PENDING)

    # =========================================================================
    # 4. Authority Transition Idempotency (Section 9, 28.4)
    # =========================================================================

    def test_authority_transition_idempotency(self):
        """Duplicate identical trigger returns existing transition record; different raises error."""
        store = FinalizationRootLedgerStore()
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )

        # First trigger
        t1 = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        # Duplicate identical trigger
        t2 = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )
        self.assertEqual(t1.authority_transition_event_id, t2.authority_transition_event_id)
        self.assertEqual(t1.authority_transition_event_hash, t2.authority_transition_event_hash)

        # Conflicting trigger with different r1_root_hash
        with self.assertRaises(CriticalIntegrityError) as ctx:
            store.record_authority_transition(
                parent_run_id=self.run_id,
                r1_root_id=r1_receipt.root_id,
                r1_root_hash="mismatched_root_hash_" + "9" * 44,
                r1_receipt_id=r1_receipt.receipt_id,
                r1_receipt_hash=r1_receipt.receipt_hash,
                t_eligible_utc=self.t_eligible,
                t_r1_confirmed_utc=self.t_r1_confirmed,
                authority_became_effective_at=self.t_effective,
            )
        self.assertIn("CONCURRENT_CONFLICT", str(ctx.exception))

    # =========================================================================
    # 5. Authority Transition Atomicity (Section 9, 28.5)
    # =========================================================================

    def test_authority_transition_atomicity(self):
        """Transition event recording and closure arming occur in single transaction."""
        store = FinalizationRootLedgerStore()
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )

        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        fetched_transition = store.get_authority_transition(self.run_id)
        fetched_closure = store.get_closure_record(self.run_id)
        self.assertIsNotNone(fetched_transition)
        self.assertIsNotNone(fetched_closure)
        self.assertEqual(fetched_closure.authority_transition_event_id, transition.authority_transition_event_id)
        self.assertEqual(fetched_closure.closure_status, PostAuthorityClosureStatus.PENDING)

    # =========================================================================
    # 6. Phase R2 Closure Payload & Commitments (Section 11, 28.6)
    # =========================================================================

    def test_r2_closure_payload(self):
        """R2 payload commits to R1 linkage, authority event, and audit finality."""
        self.assertEqual(R2_PURPOSE, "AUDIT_FINALITY")
        self.assertEqual(R2_AUTHORITY_GRANT_FUNCTION, "NONE")

        r2_payload = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id="root_r1_001",
            r1_root_hash="hash_r1_001" + "0" * 53,
            r1_receipt_hash="hash_rcpt_001" + "1" * 51,
            authority_transition_event_id="evt_auth_001",
            authority_transition_event_hash="hash_auth_001" + "2" * 51,
            authority_transition_sequence=5,
            authority_became_effective_at=self.t_effective,
            created_at_utc=self.t_r2_created,
        )

        self.assertEqual(r2_payload.root_phase, RootPhase.POST_AUTHORITY_CLOSURE_ROOT)
        self.assertEqual(r2_payload.authority_effective_status, AuthorityEffectiveStatus.EFFECTIVE)
        self.assertEqual(r2_payload.authority_became_effective_at, self.t_effective)
        self.assertEqual(r2_payload.r1_root_id, "root_r1_001")
        self.assertEqual(r2_payload.r1_root_hash, "hash_r1_001" + "0" * 53)
        self.assertEqual(r2_payload.r1_receipt_hash, "hash_rcpt_001" + "1" * 51)
        self.assertEqual(r2_payload.authority_transition_event_id, "evt_auth_001")
        self.assertEqual(r2_payload.authority_transition_event_hash, "hash_auth_001" + "2" * 51)
        self.assertEqual(r2_payload.authority_transition_sequence, 5)

    # =========================================================================
    # 7. R2 Linkage to Phase R1 Root (Section 12, 28.7)
    # =========================================================================

    def test_r2_linkage_to_r1(self):
        """R2 payload explicitly commits to R1 root ID, hash, and receipt hash."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        r1_hash = compute_finalization_root_hash(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )

        r2_payload_a = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_payload.root_id,
            r1_root_hash=r1_hash,
            r1_receipt_hash=r1_receipt.receipt_hash,
            authority_transition_event_id="evt_auth_001",
            authority_transition_event_hash="hash_auth_001" + "2" * 51,
            authority_transition_sequence=5,
            authority_became_effective_at=self.t_effective,
            created_at_utc=self.t_r2_created,
        )

        # Mutate R1 root hash in R2 linkage
        r2_payload_b = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_payload.root_id,
            r1_root_hash="tampered_r1_hash_" + "0" * 47,
            r1_receipt_hash=r1_receipt.receipt_hash,
            authority_transition_event_id="evt_auth_001",
            authority_transition_event_hash="hash_auth_001" + "2" * 51,
            authority_transition_sequence=5,
            authority_became_effective_at=self.t_effective,
            created_at_utc=self.t_r2_created,
        )

        h_a = compute_finalization_root_hash(r2_payload_a)
        h_b = compute_finalization_root_hash(r2_payload_b)
        self.assertNotEqual(h_a, h_b)

    # =========================================================================
    # 8. R2 Linkage to Authority Transition Event (Section 12, 28.8)
    # =========================================================================

    def test_r2_linkage_to_authority_transition(self):
        """R2 payload commits to authority transition event ID, hash, and sequence."""
        r2_payload_1 = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id="root_r1_001",
            r1_root_hash="hash_r1_001" + "0" * 53,
            r1_receipt_hash="hash_rcpt_001" + "1" * 51,
            authority_transition_event_id="evt_auth_001",
            authority_transition_event_hash="hash_auth_001" + "2" * 51,
            authority_transition_sequence=5,
            authority_became_effective_at=self.t_effective,
            created_at_utc=self.t_r2_created,
        )
        r2_payload_2 = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id="root_r1_001",
            r1_root_hash="hash_r1_001" + "0" * 53,
            r1_receipt_hash="hash_rcpt_001" + "1" * 51,
            authority_transition_event_id="evt_auth_001",
            authority_transition_event_hash="tampered_auth_hash" + "3" * 46,
            authority_transition_sequence=5,
            authority_became_effective_at=self.t_effective,
            created_at_utc=self.t_r2_created,
        )

        h1 = compute_finalization_root_hash(r2_payload_1)
        h2 = compute_finalization_root_hash(r2_payload_2)
        self.assertNotEqual(h1, h2)

    # =========================================================================
    # 9. Closure Deadline & Expiration (Section 13, 14, 28.9)
    # =========================================================================

    def test_r2_deadline_expiration(self):
        """Closure deadline of 300.0s transitions pending closure to EXPIRED."""
        self.assertEqual(POST_AUTHORITY_CLOSURE_MAX_DELAY, 300.0)

        store = FinalizationRootLedgerStore()
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        # 1. Within deadline: 200 seconds later -> still PENDING
        t_within = "2026-10-11T12:03:27Z"
        expired, status = store.check_and_apply_closure_deadline(self.run_id, t_within)
        self.assertFalse(expired)
        self.assertEqual(status, PostAuthorityClosureStatus.PENDING)

        # 2. Exceeded deadline: 301 seconds later -> transitions to EXPIRED
        t_expired = "2026-10-11T12:05:08Z"
        expired, status = store.check_and_apply_closure_deadline(self.run_id, t_expired)
        self.assertTrue(expired)
        self.assertEqual(status, PostAuthorityClosureStatus.EXPIRED)

        closure = store.get_closure_record(self.run_id)
        self.assertEqual(closure.closure_status, PostAuthorityClosureStatus.EXPIRED)

    # =========================================================================
    # 10. Authority Suspension After Expiry (Section 14, 28.10)
    # =========================================================================

    def test_authority_suspension_after_closure_expiry(self):
        """Expired closure suspends authority and blocks downstream actions without retroactive revoking."""
        self.assertFalse(R2_EXPIRY_RETROACTIVELY_REVOKES_PRIOR_VALID_DECISIONS)

        # Downstream firewall must block if authority is SUSPENDED_PENDING_CLOSURE
        with self.assertRaises(DownstreamFirewallBlockedError) as ctx:
            DownstreamUseFirewall.assert_authority_effective(
                authority_effective_status=AuthorityEffectiveStatus.SUSPENDED_PENDING_CLOSURE,
            )
        self.assertIn("AUTHORITY_SUSPENDED_PENDING_CLOSURE", str(ctx.exception))

        # Downstream firewall must block if closure record status is EXPIRED
        closure = PostAuthorityClosureRecord(
            parent_run_id=self.run_id,
            authority_transition_event_id="evt_auth_001",
            closure_status=PostAuthorityClosureStatus.EXPIRED,
            closure_deadline_utc="2026-10-11T12:05:07Z",
        )
        with self.assertRaises(DownstreamFirewallBlockedError) as ctx:
            DownstreamUseFirewall.assert_closure_status_allowed(closure)
        self.assertIn("CLOSURE_EXPIRED", str(ctx.exception))

    # =========================================================================
    # 11. Phase R2 Conflict Handling (Section 15, 28.11)
    # =========================================================================

    def test_r2_conflict(self):
        """Conflicting R2 receipt transitions closure to CONFLICT and suspends integrity."""
        store = FinalizationRootLedgerStore()
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        store.record_r2_conflict(self.run_id, conflict_reason="RECEIPT_SIGNATURE_MISMATCH")
        closure = store.get_closure_record(self.run_id)
        self.assertEqual(closure.closure_status, PostAuthorityClosureStatus.CONFLICT)
        self.assertEqual(closure.conflict_reason, "RECEIPT_SIGNATURE_MISMATCH")

        with self.assertRaises(DownstreamFirewallBlockedError) as ctx:
            DownstreamUseFirewall.assert_closure_status_allowed(closure)
        self.assertIn("CLOSURE_CONFLICT", str(ctx.exception))

        with self.assertRaises(DownstreamFirewallBlockedError) as ctx:
            DownstreamUseFirewall.assert_authority_effective(
                authority_effective_status=AuthorityEffectiveStatus.SUSPENDED_INTEGRITY_HOLD,
            )
        self.assertIn("AUTHORITY_SUSPENDED_INTEGRITY_HOLD", str(ctx.exception))

    # =========================================================================
    # 12. Threat Model: Authority Transition Rewrite (Section 19, 28.12)
    # =========================================================================

    def test_authority_transition_rewrite(self):
        """Auditor detects tampering or rewriting of internal authority transition record."""
        # 1. Honest execution
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        store = FinalizationRootLedgerStore()
        store.record_root_payload(r1_payload)
        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )
        r2_payload = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_hash=r1_receipt.receipt_hash,
            authority_transition_event_id=transition.authority_transition_event_id,
            authority_transition_event_hash=transition.authority_transition_event_hash,
            authority_transition_sequence=transition.authority_transition_sequence,
            authority_became_effective_at=transition.authority_became_effective_at,
            created_at_utc=self.t_r2_created,
        )
        r2_receipt = create_signed_root_receipt(
            payload=r2_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r2_confirmed,
                    )

        # 2. Tampered transition: forge event hash in transition record
        tampered_transition = AuthorityTransitionRecord(
            authority_transition_event_id=transition.authority_transition_event_id,
            parent_run_id=transition.parent_run_id,
            authority_transition_sequence=transition.authority_transition_sequence,
            event_type=transition.event_type,
            r1_root_id=transition.r1_root_id,
            r1_root_hash=transition.r1_root_hash,
            r1_receipt_id=transition.r1_receipt_id,
            r1_receipt_hash=transition.r1_receipt_hash,
            authority_effective_status=transition.authority_effective_status,
            t_eligible_utc=transition.t_eligible_utc,
            t_r1_confirmed_utc=transition.t_r1_confirmed_utc,
            authority_became_effective_at=transition.authority_became_effective_at,
            recorded_at=transition.recorded_at,
            idempotency_key=transition.idempotency_key,
            authority_transition_event_hash="forged_transition_hash_" + "0" * 41,
        )

        reviewer = FinalizationRootReviewer()
        result = reviewer.audit_two_phase_root(
            r1_payload=r1_payload,
            r1_receipt=r1_receipt,
            authority_transition=tampered_transition,
            r2_payload=r2_payload,
            r2_receipt=r2_receipt,
            authority_eligibility=AuthorityEligibility.PASS,
            t_eligible_utc=self.t_eligible,
        )

        self.assertFalse(result["overall_passed"])
        self.assertTrue(result["authority_transition_rewrite_detected"])

    # =========================================================================
    # 13. Threat Model: R1 Receipt Substitution (Section 20, 28.13)
    # =========================================================================

    def test_r1_receipt_substitution(self):
        """Auditor detects substitution of R1 receipt (external R2 parity failure)."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        r1_receipt_real = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        store = FinalizationRootLedgerStore()
        store.record_root_payload(r1_payload)
        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt_real.root_id,
            r1_root_hash=r1_receipt_real.root_hash,
            r1_receipt_id=r1_receipt_real.receipt_id,
            r1_receipt_hash=r1_receipt_real.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )
        r2_payload = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_receipt_real.root_id,
            r1_root_hash=r1_receipt_real.root_hash,
            r1_receipt_hash=r1_receipt_real.receipt_hash,
            authority_transition_event_id=transition.authority_transition_event_id,
            authority_transition_event_hash=transition.authority_transition_event_hash,
            authority_transition_sequence=transition.authority_transition_sequence,
            authority_became_effective_at=transition.authority_became_effective_at,
            created_at_utc=self.t_r2_created,
        )
        r2_receipt = create_signed_root_receipt(
            payload=r2_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r2_confirmed,
                    )

        # Attacker substitutes R1 receipt with a different receipt
        r1_receipt_substituted = FinalizationRootReceipt(
            receipt_id="rcpt_substituted_999",
            root_id=r1_receipt_real.root_id,
            root_hash=r1_receipt_real.root_hash,
            external_provider="AWS_S3_OBJECT_LOCK_ROOT",
            object_version_id="ver_sub_999",
            worm_retention_until="2036-10-11T12:00:00Z",
            confirmed_at_utc=self.t_r1_confirmed,
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            signature=r1_receipt_real.signature,
            receipt_hash="substituted_receipt_hash_" + "0" * 39,
        )

        reviewer = FinalizationRootReviewer()
        result = reviewer.audit_two_phase_root(
            r1_payload=r1_payload,
            r1_receipt=r1_receipt_substituted,
            authority_transition=transition,
            r2_payload=r2_payload,
            r2_receipt=r2_receipt,
            authority_eligibility=AuthorityEligibility.PASS,
            t_eligible_utc=self.t_eligible,
        )

        self.assertFalse(result["overall_passed"])
        self.assertTrue(result["r1_receipt_substitution_detected"])

    # =========================================================================
    # 14. Threat Model: Duplicate R1 Notification Handling (Section 21, 28.14)
    # =========================================================================

    def test_duplicate_r1_notification(self):
        """Duplicate R1 external notification emits 0 duplicate events."""
        self.assertEqual(DUPLICATE_IDENTICAL_AUTHORITY_TRIGGER, "RETURN_EXISTING_TRANSITION")

        store = FinalizationRootLedgerStore()
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )

        # First trigger
        t1 = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        # Duplicate delivery of notification
        t2 = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )

        self.assertEqual(t1.authority_transition_event_id, t2.authority_transition_event_id)
        # Check that only 1 record exists in database
        records = store.conn.execute(
            "SELECT count(*) FROM authority_transitions WHERE parent_run_id = ?",
            (self.run_id,),
        ).fetchone()[0]
        self.assertEqual(records, 1)

    # =========================================================================
    # 15. Threat Model: R1 Conflict Handling (Section 22, 28.15)
    # =========================================================================

    def test_r1_conflict(self):
        """Conflicting R1 receipt fails authorization and sets BLOCKED_INTEGRITY_HOLD."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        # Receipt with corrupted signature
        r1_receipt_invalid = FinalizationRootReceipt(
            receipt_id="rcpt_corrupted_001",
            root_id=r1_payload.root_id,
            root_hash="mismatched_hash_" + "0" * 48,
            external_provider="AWS_S3_OBJECT_LOCK_ROOT",
            object_version_id="ver_001",
            worm_retention_until="2036-10-11T12:00:00Z",
            confirmed_at_utc=self.t_r1_confirmed,
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            signature="corrupted_sig",
            receipt_hash="hash_001" * 8,
        )

        status, err = verify_r1_root_receipt(r1_receipt_invalid, r1_payload)
        self.assertEqual(status, R1ConfirmationStatus.CONFLICT)
        self.assertIsNotNone(err)

        authorized = check_authority_transition_authorized(
            r1_receipt=r1_receipt_invalid,
            r1_payload=r1_payload,
            authority_eligibility=AuthorityEligibility.PASS,
            t_eligible_utc=self.t_eligible,
        )
        self.assertFalse(authorized)

        # Downstream firewall must block BLOCKED_INTEGRITY_HOLD
        with self.assertRaises(DownstreamFirewallBlockedError) as ctx:
            DownstreamUseFirewall.assert_authority_effective(
                authority_effective_status=AuthorityEffectiveStatus.BLOCKED_INTEGRITY_HOLD,
            )
        self.assertIn("AUTHORITY_BLOCKED_INTEGRITY_HOLD", str(ctx.exception))

    # =========================================================================
    # 16. Crash Recovery Matrix (Section 23, 28.16)
    # =========================================================================

    def test_crash_recovery_matrix(self):
        """Verifies process recovery at each lifecycle interruption point."""
        store = FinalizationRootLedgerStore()

        # Step 1: Pre-authority root recorded, crash before receipt
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        store.record_root_payload(r1_payload)
        # Recovered state: R1 exists, transition does not exist, authority BLOCKED_PENDING_FINALIZATION_ROOT
        lookup_res, payload = store.lookup_existing_root(
            self.run_id,
            compute_finalization_root_hash(r1_payload),
            root_phase=RootPhase.PRE_AUTHORITY_ADMISSION_ROOT,
        )
        self.assertEqual(lookup_res, RootLookupResult.FOUND_MATCHING)
        self.assertIsNotNone(payload)
        self.assertIsNone(store.get_authority_transition(self.run_id))

        # Step 2: Receipt confirmed, transition recorded, crash before R2
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )
        # Recovered state: transition exists, closure is PENDING
        closure = store.get_closure_record(self.run_id)
        self.assertEqual(closure.closure_status, PostAuthorityClosureStatus.PENDING)

        # Step 3: R2 generated & submitted, receipt recorded
        r2_payload = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_hash=r1_receipt.receipt_hash,
            authority_transition_event_id=transition.authority_transition_event_id,
            authority_transition_event_hash=transition.authority_transition_event_hash,
            authority_transition_sequence=transition.authority_transition_sequence,
            authority_became_effective_at=transition.authority_became_effective_at,
            created_at_utc=self.t_r2_created,
        )
        store.record_root_payload(r2_payload)
        r2_receipt = create_signed_root_receipt(
            payload=r2_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r2_confirmed,
                    )
        store.record_r2_closure(
            parent_run_id=self.run_id,
            r2_root_id=r2_receipt.root_id,
            r2_root_hash=r2_receipt.root_hash,
            r2_receipt_id=r2_receipt.receipt_id,
            r2_receipt_hash=r2_receipt.receipt_hash,
        )

        closure_final = store.get_closure_record(self.run_id)
        self.assertEqual(closure_final.closure_status, PostAuthorityClosureStatus.CONFIRMED)

    # =========================================================================
    # 17. Independent Reviewer Two-Phase Reconstruction (Section 25, 28.17)
    # =========================================================================

    def test_two_phase_reviewer_reconstruction(self):
        """Independent Reviewer performs 10-point forensic audit of two-phase lifecycle."""
        r1_payload = build_r1_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            created_at_utc=self.t_r1_created,
        )
        r1_receipt = create_signed_root_receipt(
            payload=r1_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r1_confirmed,
                    )
        store = FinalizationRootLedgerStore()
        store.record_root_payload(r1_payload)
        transition = store.record_authority_transition(
            parent_run_id=self.run_id,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_id=r1_receipt.receipt_id,
            r1_receipt_hash=r1_receipt.receipt_hash,
            t_eligible_utc=self.t_eligible,
            t_r1_confirmed_utc=self.t_r1_confirmed,
            authority_became_effective_at=self.t_effective,
        )
        r2_payload = build_r2_root_payload(
            parent_run_id=self.run_id,
            parent_run_final_hash=self.run_final_hash,
            events=self.events,
            authority_eligibility="PASS",
            policy_set_hash=self.policy_set_hash,
            r1_root_id=r1_receipt.root_id,
            r1_root_hash=r1_receipt.root_hash,
            r1_receipt_hash=r1_receipt.receipt_hash,
            authority_transition_event_id=transition.authority_transition_event_id,
            authority_transition_event_hash=transition.authority_transition_event_hash,
            authority_transition_sequence=transition.authority_transition_sequence,
            authority_became_effective_at=transition.authority_became_effective_at,
            created_at_utc=self.t_r2_created,
        )
        store.record_root_payload(r2_payload)
        r2_receipt = create_signed_root_receipt(
            payload=r2_payload,
            provider="AWS_S3_OBJECT_LOCK_ROOT",
            receipt_signing_key_id=DEFAULT_ROOT_SIGNING_KEY_ID,
            created_at_utc=self.t_r2_confirmed,
                    )

        reviewer = FinalizationRootReviewer()
        result = reviewer.audit_two_phase_root(
            r1_payload=r1_payload,
            r1_receipt=r1_receipt,
            authority_transition=transition,
            r2_payload=r2_payload,
            r2_receipt=r2_receipt,
            authority_eligibility=AuthorityEligibility.PASS,
            t_eligible_utc=self.t_eligible,
        )

        self.assertTrue(result["overall_passed"])
        self.assertTrue(result["r1_payload_matches"])
        self.assertTrue(result["r1_receipt_verified"])
        self.assertTrue(result["transition_authorized"])
        self.assertTrue(result["transition_event_hash_matches"])
        self.assertTrue(result["transition_timing_valid"])
        self.assertTrue(result["r2_payload_matches"])
        self.assertTrue(result["r2_linkage_matches"])
        self.assertTrue(result["r2_receipt_verified"])
        self.assertFalse(result["authority_transition_rewrite_detected"])
        self.assertFalse(result["r1_receipt_substitution_detected"])
        self.assertFalse(result["r1_conflict_detected"])
        self.assertEqual(len(result["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
