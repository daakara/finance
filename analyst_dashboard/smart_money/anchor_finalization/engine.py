"""
External Anchor Finalization Engine & Coordination Controller.
Coordinates the end-to-end post-capture lifecycle:
sealing check, payload canonicalization, idempotent submission, ambiguous commit recovery,
receipt verification, quorum evaluation, finalization ledger commits, and late recovery.
"""

import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Tuple

from analyst_dashboard.smart_money.anchor_finalization.models import (
    SealedRunLedger,
    AnchorPayload,
    AnchorAttemptRecord,
    FinalizationEvent,
    AnchorReceipt,
    FinalizationState,
    AuthorityEligibility,
    AuthorityEffectiveStatus,
    ErrorClass,
    LookupResult,
    QuorumPolicy,
    CriticalIntegrityError,
    InvalidTransitionError,
    QuorumNotSatisfiedError,
)
from analyst_dashboard.smart_money.anchor_finalization.policy import (
    ANCHOR_POLICY_ID,
    INITIAL_RETRY_DELAY_SECONDS,
    MAX_RETRY_DELAY_SECONDS,
    MAX_INLINE_ATTEMPTS,
    MAX_PENDING_DURATION_SECONDS,
    AMBIGUOUS_COMMIT_RESULT_POLICY,
)
from analyst_dashboard.smart_money.anchor_finalization.payload_serializer import (
    ANCHOR_PAYLOAD_SCHEMA_VERSION,
    compute_anchor_payload_hash,
    derive_anchor_idempotency_key,
    validate_idempotency_continuity,
)
from analyst_dashboard.smart_money.anchor_finalization.quorum import (
    DEFAULT_PROVIDER_A,
    DEFAULT_PROVIDER_B,
    evaluate_anchor_quorum,
    assert_anchor_quorum_satisfied,
    verify_dual_anchor_independence,
)
from analyst_dashboard.smart_money.anchor_finalization.receipt_verifier import (
    verify_provider_receipt,
    generate_provider_receipt_signature,
)
from analyst_dashboard.smart_money.anchor_finalization.finalization_ledger import (
    FinalizationLedgerStore,
)
from analyst_dashboard.smart_money.anchor_finalization.state_machine import (
    FinalizationStateMachine,
)


class AnchorFinalizationEngine:
    """
    Control plane engine executing the External Anchor Finalization Subsystem.
    """

    def __init__(
        self,
        ledger_store: Optional[FinalizationLedgerStore] = None,
        required_providers: Optional[List[str]] = None,
        quorum_policy: QuorumPolicy = QuorumPolicy.ALL_REQUIRED,
    ):
        self.store = ledger_store or FinalizationLedgerStore()
        self.required_providers = required_providers or [
            DEFAULT_PROVIDER_A.provider_id,
            DEFAULT_PROVIDER_B.provider_id,
        ]
        self.quorum_policy = quorum_policy

        # Idempotency cache: key -> payload_hash
        self._idempotency_cache: Dict[str, str] = {}
        # Receipt storage: run_id -> provider -> AnchorReceipt
        self._confirmed_receipts: Dict[str, Dict[str, AnchorReceipt]] = {}
        # Active state machines: run_id -> FinalizationStateMachine
        self._state_machines: Dict[str, FinalizationStateMachine] = {}

    def build_anchor_payload(
        self,
        sealed_ledger: SealedRunLedger,
        created_at_utc: Optional[str] = None,
    ) -> Tuple[AnchorPayload, str, str]:
        """
        Validates sealed run ledger, constructs canonical AnchorPayload,
        computes payload hash and derives idempotency key.
        """
        sealed_ledger.assert_immutable()

        if created_at_utc is None:
            created_at_utc = sealed_ledger.sealed_at_utc

        payload = AnchorPayload(
            anchor_schema_version=ANCHOR_PAYLOAD_SCHEMA_VERSION,
            run_id=sealed_ledger.run_id,
            machine_version="1.0",
            run_genesis_hash=sealed_ledger.run_genesis_hash,
            run_final_hash=sealed_ledger.run_final_hash,
            transition_count=sealed_ledger.transition_count,
            final_sequence=sealed_ledger.final_sequence,
            final_state=sealed_ledger.final_state,
            natural_bundle_hash=sealed_ledger.natural_bundle_hash,
            policy_set_hash=sealed_ledger.policy_set_hash,
            repository_head_sha=sealed_ledger.repository_head_sha,
            created_at_utc=created_at_utc,
        )

        payload_hash = compute_anchor_payload_hash(payload)
        idempotency_key = derive_anchor_idempotency_key(
            run_id=sealed_ledger.run_id,
            run_final_hash=sealed_ledger.run_final_hash,
            policy_set_hash=sealed_ledger.policy_set_hash,
        )

        # Enforce idempotency continuity (Section 11)
        if idempotency_key in self._idempotency_cache:
            validate_idempotency_continuity(
                existing_idempotency_key=idempotency_key,
                incoming_idempotency_key=idempotency_key,
                existing_payload_hash=self._idempotency_cache[idempotency_key],
                incoming_payload_hash=payload_hash,
            )
        else:
            self._idempotency_cache[idempotency_key] = payload_hash

        return payload, payload_hash, idempotency_key

    def submit_anchor_for_provider(
        self,
        payload: AnchorPayload,
        payload_hash: str,
        idempotency_key: str,
        provider_id: str,
        sm: FinalizationStateMachine,
        provider_submit_fn: Callable[[AnchorPayload], Tuple[int, Optional[AnchorReceipt], ErrorClass]],
        provider_lookup_fn: Optional[Callable[[str, str], Tuple[LookupResult, Optional[AnchorReceipt]]]] = None,
        request_id: Optional[str] = None,
        elapsed_seconds: float = 0.0,
    ) -> bool:
        """
        Submits anchor to a single provider with retry backoff and ambiguous commit recovery.
        """
        run_id = payload.run_id
        if request_id is None:
            request_id = f"req_{run_id}_{provider_id}"

        # Initialize coordinator if not already present
        self.store.initialize_run_coordinator(
            parent_run_id=run_id,
            parent_run_final_hash=payload.run_final_hash,
        )

        # Transition A1 -> A2 (Sealed)
        if sm.state == FinalizationState.A1_ANCHOR_REQUIRED:
            sm.transition(FinalizationState.A2_ANCHOR_PAYLOAD_SEALED)
            self.store.append_finalization_event(
                parent_run_id=run_id,
                parent_run_final_hash=payload.run_final_hash,
                finalization_event_type="SEAL_PAYLOAD",
                anchor_request_id=request_id,
                anchor_attempt_id="none",
                anchor_status_to=FinalizationState.A2_ANCHOR_PAYLOAD_SEALED,
                authority_effective_status_to=sm.effective_status,
            )

        attempt_num = 1
        while attempt_num <= MAX_INLINE_ATTEMPTS:
            # Check expired pending duration (Section 20)
            if elapsed_seconds > MAX_PENDING_DURATION_SECONDS:
                sm.transition(FinalizationState.A7_ANCHOR_PENDING_EXPIRED)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="PENDING_EXPIRED",
                    anchor_request_id=request_id,
                    anchor_attempt_id=f"att_{run_id}_{provider_id}_{attempt_num}",
                    anchor_status_to=FinalizationState.A7_ANCHOR_PENDING_EXPIRED,
                    authority_effective_status_to=sm.effective_status,
                )
                return False

            attempt_id = f"att_{run_id}_{provider_id}_{attempt_num}"
            started_at = datetime.now(timezone.utc).isoformat()

            # Transition to A3 (Submission Pending)
            if sm.state != FinalizationState.A3_ANCHOR_SUBMISSION_PENDING:
                sm.transition(FinalizationState.A3_ANCHOR_SUBMISSION_PENDING)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="SUBMIT_PENDING",
                    anchor_request_id=request_id,
                    anchor_attempt_id=attempt_id,
                    anchor_status_to=FinalizationState.A3_ANCHOR_SUBMISSION_PENDING,
                    authority_effective_status_to=sm.effective_status,
                )

            # Invoke provider
            http_status, receipt, error_class = provider_submit_fn(payload)
            completed_at = datetime.now(timezone.utc).isoformat()

            # Record attempt in ledger
            attempt_rec = AnchorAttemptRecord(
                anchor_attempt_id=attempt_id,
                anchor_request_id=request_id,
                attempt_number=attempt_num,
                run_id=run_id,
                anchor_idempotency_key=idempotency_key,
                request_payload_sha256=payload_hash,
                request_started_at=started_at,
                request_completed_at=completed_at,
                provider=provider_id,
                provider_http_status=http_status,
                provider_response_sha256=receipt.signature if receipt else "",
                error_class=error_class,
                result="SUCCESS" if error_class == ErrorClass.NONE else "FAILURE",
                lookup_required=error_class == ErrorClass.INDETERMINATE,
                lookup_performed=False,
            )
            self.store.record_attempt(attempt_rec)

            if error_class == ErrorClass.NONE and receipt is not None:
                # Receipt verification (Section 18)
                verdict, details = verify_provider_receipt(receipt, payload)
                if verdict == "PASS":
                    if run_id not in self._confirmed_receipts:
                        self._confirmed_receipts[run_id] = {}
                    self._confirmed_receipts[run_id][provider_id] = receipt

                    sm.transition(FinalizationState.A6_ANCHOR_CONFIRMED)
                    self.store.append_finalization_event(
                        parent_run_id=run_id,
                        parent_run_final_hash=payload.run_final_hash,
                        finalization_event_type="RECEIPT_CONFIRMED",
                        anchor_request_id=request_id,
                        anchor_attempt_id=attempt_id,
                        anchor_status_to=FinalizationState.A6_ANCHOR_CONFIRMED,
                        authority_effective_status_to=sm.effective_status,
                    )
                    return True
                else:
                    # Receipt verification failure -> Conflict
                    sm.transition(FinalizationState.A8_ANCHOR_CONFLICT)
                    self.store.append_finalization_event(
                        parent_run_id=run_id,
                        parent_run_final_hash=payload.run_final_hash,
                        finalization_event_type="RECEIPT_VERIFICATION_FAILED",
                        anchor_request_id=request_id,
                        anchor_attempt_id=attempt_id,
                        anchor_status_to=FinalizationState.A8_ANCHOR_CONFLICT,
                        authority_effective_status_to=sm.effective_status,
                    )
                    sm.transition(FinalizationState.A11_TERMINAL_HOLD)
                    self.store.append_finalization_event(
                        parent_run_id=run_id,
                        parent_run_final_hash=payload.run_final_hash,
                        finalization_event_type="TERMINAL_HOLD_ENTERED",
                        anchor_request_id=request_id,
                        anchor_attempt_id=attempt_id,
                        anchor_status_to=FinalizationState.A11_TERMINAL_HOLD,
                        authority_effective_status_to=sm.effective_status,
                    )
                    return False

            elif error_class == ErrorClass.INDETERMINATE:
                # Ambiguous commit result policy: Query before retry (Section 13)
                sm.transition(FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="COMMIT_INDETERMINATE",
                    anchor_request_id=request_id,
                    anchor_attempt_id=attempt_id,
                    anchor_status_to=FinalizationState.A5_ANCHOR_COMMIT_INDETERMINATE,
                    authority_effective_status_to=sm.effective_status,
                )

                if provider_lookup_fn is not None:
                    lookup_res, lookup_receipt = provider_lookup_fn(run_id, idempotency_key)
                    if lookup_res == LookupResult.FOUND_MATCHING and lookup_receipt is not None:
                        verdict, _ = verify_provider_receipt(lookup_receipt, payload)
                        if verdict == "PASS":
                            if run_id not in self._confirmed_receipts:
                                self._confirmed_receipts[run_id] = {}
                            self._confirmed_receipts[run_id][provider_id] = lookup_receipt

                            sm.transition(FinalizationState.A6_ANCHOR_CONFIRMED)
                            self.store.append_finalization_event(
                                parent_run_id=run_id,
                                parent_run_final_hash=payload.run_final_hash,
                                finalization_event_type="LOOKUP_CONFIRMED",
                                anchor_request_id=request_id,
                                anchor_attempt_id=attempt_id,
                                anchor_status_to=FinalizationState.A6_ANCHOR_CONFIRMED,
                                authority_effective_status_to=sm.effective_status,
                            )
                            return True
                    elif lookup_res == LookupResult.FOUND_CONFLICTING:
                        sm.transition(FinalizationState.A8_ANCHOR_CONFLICT)
                        self.store.append_finalization_event(
                            parent_run_id=run_id,
                            parent_run_final_hash=payload.run_final_hash,
                            finalization_event_type="LOOKUP_CONFLICT",
                            anchor_request_id=request_id,
                            anchor_attempt_id=attempt_id,
                            anchor_status_to=FinalizationState.A8_ANCHOR_CONFLICT,
                            authority_effective_status_to=sm.effective_status,
                        )
                        sm.transition(FinalizationState.A11_TERMINAL_HOLD)
                        self.store.append_finalization_event(
                            parent_run_id=run_id,
                            parent_run_final_hash=payload.run_final_hash,
                            finalization_event_type="TERMINAL_HOLD_ENTERED",
                            anchor_request_id=request_id,
                            anchor_attempt_id=attempt_id,
                            anchor_status_to=FinalizationState.A11_TERMINAL_HOLD,
                            authority_effective_status_to=sm.effective_status,
                        )
                        return False
                    elif lookup_res == LookupResult.NOT_FOUND:
                        # Retry authorized
                        sm.transition(FinalizationState.A4_ANCHOR_RETRY_WAIT)
                        self.store.append_finalization_event(
                            parent_run_id=run_id,
                            parent_run_final_hash=payload.run_final_hash,
                            finalization_event_type="LOOKUP_NOT_FOUND_RETRY_WAIT",
                            anchor_request_id=request_id,
                            anchor_attempt_id=attempt_id,
                            anchor_status_to=FinalizationState.A4_ANCHOR_RETRY_WAIT,
                            authority_effective_status_to=sm.effective_status,
                        )
                else:
                    # No lookup available -> Remain blocked
                    return False

            elif error_class == ErrorClass.INTEGRITY_CONFLICT:
                sm.transition(FinalizationState.A8_ANCHOR_CONFLICT)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="INTEGRITY_CONFLICT",
                    anchor_request_id=request_id,
                    anchor_attempt_id=attempt_id,
                    anchor_status_to=FinalizationState.A8_ANCHOR_CONFLICT,
                    authority_effective_status_to=sm.effective_status,
                )
                sm.transition(FinalizationState.A11_TERMINAL_HOLD)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="TERMINAL_HOLD_ENTERED",
                    anchor_request_id=request_id,
                    anchor_attempt_id=attempt_id,
                    anchor_status_to=FinalizationState.A11_TERMINAL_HOLD,
                    authority_effective_status_to=sm.effective_status,
                )
                return False

            elif error_class == ErrorClass.RETRYABLE_TRANSPORT:
                sm.transition(FinalizationState.A4_ANCHOR_RETRY_WAIT)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=payload.run_final_hash,
                    finalization_event_type="RETRY_WAIT",
                    anchor_request_id=request_id,
                    anchor_attempt_id=attempt_id,
                    anchor_status_to=FinalizationState.A4_ANCHOR_RETRY_WAIT,
                    authority_effective_status_to=sm.effective_status,
                )
                backoff = min(
                    MAX_RETRY_DELAY_SECONDS,
                    INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt_num - 1)),
                )
                elapsed_seconds += backoff

            attempt_num += 1

        # Max inline attempts exhausted
        sm.transition(FinalizationState.A7_ANCHOR_PENDING_EXPIRED)
        self.store.append_finalization_event(
            parent_run_id=run_id,
            parent_run_final_hash=payload.run_final_hash,
            finalization_event_type="ATTEMPTS_EXHAUSTED_PENDING_EXPIRED",
            anchor_request_id=request_id,
            anchor_attempt_id=f"att_{run_id}_{provider_id}_max",
            anchor_status_to=FinalizationState.A7_ANCHOR_PENDING_EXPIRED,
            authority_effective_status_to=sm.effective_status,
        )
        return False

    def finalize_run(
        self,
        sealed_ledger: SealedRunLedger,
        evidence_observed_at: str,
        eligibility_established_at: str,
        provider_submit_map: Dict[str, Callable[[AnchorPayload], Tuple[int, Optional[AnchorReceipt], ErrorClass]]],
        provider_lookup_map: Optional[Dict[str, Callable[[str, str], Tuple[LookupResult, Optional[AnchorReceipt]]]]] = None,
        effective_time_utc: Optional[str] = None,
    ) -> FinalizationStateMachine:
        """
        Executes external anchor finalization for a run across all required providers.
        """
        run_id = sealed_ledger.run_id
        payload, payload_hash, idempotency_key = self.build_anchor_payload(sealed_ledger)

        sm = FinalizationStateMachine(
            initial_state=FinalizationState.A1_ANCHOR_REQUIRED,
            initial_eligibility=AuthorityEligibility.PASS,
            initial_effective_status=AuthorityEffectiveStatus.BLOCKED_PENDING_ANCHOR,
            evidence_observed_at=evidence_observed_at,
            eligibility_established_at=eligibility_established_at,
        )
        self._state_machines[run_id] = sm

        for provider_id in self.required_providers:
            submit_fn = provider_submit_map.get(provider_id)
            if not submit_fn:
                continue
            lookup_fn = provider_lookup_map.get(provider_id) if provider_lookup_map else None
            success = self.submit_anchor_for_provider(
                payload=payload,
                payload_hash=payload_hash,
                idempotency_key=idempotency_key,
                provider_id=provider_id,
                sm=sm,
                provider_submit_fn=submit_fn,
                provider_lookup_fn=lookup_fn,
            )
            if not success and sm.state in (FinalizationState.A8_ANCHOR_CONFLICT, FinalizationState.A11_TERMINAL_HOLD):
                return sm

        # Evaluate Quorum
        confirmed = list(self._confirmed_receipts.get(run_id, {}).keys())
        quorum_ok = evaluate_anchor_quorum(
            confirmed_providers=confirmed,
            required_providers=self.required_providers,
            quorum_policy=self.quorum_policy,
        )

        if quorum_ok and sm.state == FinalizationState.A6_ANCHOR_CONFIRMED:
            effective_now = effective_time_utc or datetime.now(timezone.utc).isoformat()
            sm.transition(
                FinalizationState.A10_AUTHORITY_EFFECTIVE,
                effective_time_utc=effective_now,
                quorum_satisfied=True,
            )
            self.store.append_finalization_event(
                parent_run_id=run_id,
                parent_run_final_hash=sealed_ledger.run_final_hash,
                finalization_event_type="AUTHORITY_EFFECTIVE_GRANTED",
                anchor_request_id="quorum_grant",
                anchor_attempt_id="none",
                anchor_status_to=FinalizationState.A10_AUTHORITY_EFFECTIVE,
                authority_effective_status_to=AuthorityEffectiveStatus.EFFECTIVE,
                recorded_at=effective_now,
            )

        return sm

    def execute_late_anchor_recovery(
        self,
        run_id: str,
        sealed_ledger: SealedRunLedger,
        late_receipts: List[AnchorReceipt],
        recovery_authorized_at: str,
    ) -> FinalizationStateMachine:
        """
        Executes Late Anchor Recovery (Section 21).
        Transitions PENDING_EXPIRED (A7) -> GOVERNANCE_RECOVERY_REVIEW (A9) -> AUTHORITY_EFFECTIVE (A10).
        Enforces NO RETROACTIVE AUTHORIZATION: authority_became_effective_at is recovery_authorized_at,
        NOT original observation time!
        """
        sm = self._state_machines.get(run_id)
        if sm is None:
            sm = FinalizationStateMachine(
                initial_state=FinalizationState.A7_ANCHOR_PENDING_EXPIRED,
                initial_eligibility=AuthorityEligibility.PASS,
                initial_effective_status=AuthorityEffectiveStatus.BLOCKED_PENDING_GOVERNANCE_RECOVERY,
                evidence_observed_at=sealed_ledger.sealed_at_utc,
                eligibility_established_at=sealed_ledger.sealed_at_utc,
            )
            self._state_machines[run_id] = sm

        if sm.state != FinalizationState.A7_ANCHOR_PENDING_EXPIRED:
            raise InvalidTransitionError(
                f"Late recovery requires state A7_ANCHOR_PENDING_EXPIRED, found {sm.state}"
            )

        # Transition A7 -> A9
        sm.transition(FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW)
        self.store.append_finalization_event(
            parent_run_id=run_id,
            parent_run_final_hash=sealed_ledger.run_final_hash,
            finalization_event_type="RECOVERY_REVIEW_INITIATED",
            anchor_request_id="recovery_review",
            anchor_attempt_id="none",
            anchor_status_to=FinalizationState.A9_GOVERNANCE_RECOVERY_REVIEW,
            authority_effective_status_to=sm.effective_status,
        )

        # Verify late receipts against payload
        payload, _, _ = self.build_anchor_payload(sealed_ledger)
        for rcpt in late_receipts:
            verdict, _ = verify_provider_receipt(rcpt, payload)
            if verdict != "PASS":
                sm.transition(FinalizationState.A11_TERMINAL_HOLD)
                self.store.append_finalization_event(
                    parent_run_id=run_id,
                    parent_run_final_hash=sealed_ledger.run_final_hash,
                    finalization_event_type="LATE_RECEIPT_INVALID_TERMINAL_HOLD",
                    anchor_request_id="recovery_review",
                    anchor_attempt_id="none",
                    anchor_status_to=FinalizationState.A11_TERMINAL_HOLD,
                    authority_effective_status_to=sm.effective_status,
                )
                return sm

            if run_id not in self._confirmed_receipts:
                self._confirmed_receipts[run_id] = {}
            self._confirmed_receipts[run_id][rcpt.provider] = rcpt

        # Verify quorum
        confirmed = list(self._confirmed_receipts.get(run_id, {}).keys())
        assert_anchor_quorum_satisfied(
            confirmed_providers=confirmed,
            required_providers=self.required_providers,
            quorum_policy=self.quorum_policy,
        )

        # Transition A9 -> A10 (Effective at recovery time, NOT backdated)
        sm.transition(
            FinalizationState.A10_AUTHORITY_EFFECTIVE,
            effective_time_utc=recovery_authorized_at,
            recovery_authorized=True,
        )
        self.store.append_finalization_event(
            parent_run_id=run_id,
            parent_run_final_hash=sealed_ledger.run_final_hash,
            finalization_event_type="LATE_AUTHORITY_EFFECTIVE_GRANTED",
            anchor_request_id="late_recovery_grant",
            anchor_attempt_id="none",
            anchor_status_to=FinalizationState.A10_AUTHORITY_EFFECTIVE,
            authority_effective_status_to=AuthorityEffectiveStatus.EFFECTIVE,
            recorded_at=recovery_authorized_at,
        )
        return sm
