"""
HVA-01 Intake Runner Coordinator.
Orchestrates FSM state transitions, cryptographic hashing, and transactional append-only storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .engine.fsm import AuditEvent, ResponseLockingFSM, StateTransitionError
from .engine.hasher import CanonicalHasher
from .models.response import (
    RatingsPayload,
    ResponseRecord,
    ResponseValidationError,
)
from .models.session import SessionModel, SessionState, ValidationError
from .storage.append_store import (
    AppendOnlyStore,
    ConflictingDuplicateSubmissionError,
    CorrectionRecord,
    StorageCorruptionError,
)


class IntakeRunner:
    """
    Core intake runner coordinating session lifecycle, response evaluation,
    deterministic locking, and crash recovery.
    """

    def __init__(
        self,
        session: SessionModel,
        store: AppendOnlyStore,
        fsm: Optional[ResponseLockingFSM] = None,
    ) -> None:
        self.session = session
        self.store = store
        self.fsm = fsm or ResponseLockingFSM(session)

    @classmethod
    def create_session(
        cls,
        session_id: str,
        reviewer_pseudonym: str,
        assigned_package: str,
        protocol_version: str,
        protocol_document_sha256: str,
        package_sha256: str,
        base_dir: Union[str, Path],
        total_cases: int = 59,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntakeRunner:
        """
        Create a new evaluation session with transactional storage and genesis hash computation.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        session = SessionModel(
            session_id=session_id,
            protocol_version=protocol_version,
            protocol_document_sha256=protocol_document_sha256,
            assigned_package=assigned_package,
            package_sha256=package_sha256,
            reviewer_pseudonym=reviewer_pseudonym,
            created_at_utc=now_utc,
            current_state=SessionState.SESSION_CREATED,
            current_sequence=1,
            total_cases=total_cases,
            metadata=metadata or {},
        )

        store = AppendOnlyStore(base_dir)
        store.init_session(session)
        fsm = ResponseLockingFSM(session)

        # Record session creation audit event
        runner = cls(session=session, store=store, fsm=fsm)
        runner._record_audit_event(
            event_code="EVT_SESSION_CREATED",
            details={
                "session_id": session_id,
                "reviewer_pseudonym": reviewer_pseudonym,
                "assigned_package": assigned_package,
                "total_cases": total_cases,
            },
        )
        return runner

    @classmethod
    def resume_session(cls, session_id: str, base_dir: Union[str, Path]) -> IntakeRunner:
        """
        Resume an existing session with crash recovery, hash-chain verification,
        and FSM state restoration from durable storage.
        """
        store = AppendOnlyStore(base_dir)
        store.cleanup_uncommitted_staging(session_id)

        session = store.load_session(session_id)
        records, latest_hash = store.load_master_records(session_id, verify=True)

        fsm = ResponseLockingFSM(session)
        locked_sequences = {rec.sequence_index for rec in records}

        committed_count = len(records)
        if committed_count >= session.total_cases:
            restored_state = SessionState.SESSION_CLOSED
            restored_sequence = session.total_cases
        elif committed_count > 0:
            restored_state = SessionState.RESPONSE_DRAFT
            restored_sequence = committed_count + 1
        else:
            restored_state = session.current_state
            restored_sequence = session.current_sequence

        fsm.restore_state(
            locked_cases=locked_sequences,
            current_sequence=restored_sequence,
            current_state=restored_state,
        )

        # Update session model on disk if recovered sequence or state changed
        if session.current_sequence != restored_sequence or session.current_state != restored_state:
            updated_session = SessionModel(
                session_id=session.session_id,
                protocol_version=session.protocol_version,
                protocol_document_sha256=session.protocol_document_sha256,
                assigned_package=session.assigned_package,
                package_sha256=session.package_sha256,
                reviewer_pseudonym=session.reviewer_pseudonym,
                created_at_utc=session.created_at_utc,
                current_state=restored_state,
                current_sequence=restored_sequence,
                total_cases=session.total_cases,
                schema_version=session.schema_version,
                expiry_utc=session.expiry_utc,
                metadata=session.metadata,
            )
            store.save_session(updated_session)
            session = updated_session

        runner = cls(session=session, store=store, fsm=fsm)
        runner._record_audit_event(
            event_code="EVT_SESSION_RESUMED",
            details={
                "committed_count": committed_count,
                "latest_hash": latest_hash,
                "restored_sequence": restored_sequence,
                "restored_state": restored_state.value,
            },
        )
        return runner

    def _sync_session_state(self) -> None:
        """Synchronize in-memory SessionModel with FSM state and persist to disk."""
        self.session = SessionModel(
            session_id=self.session.session_id,
            protocol_version=self.session.protocol_version,
            protocol_document_sha256=self.session.protocol_document_sha256,
            assigned_package=self.session.assigned_package,
            package_sha256=self.session.package_sha256,
            reviewer_pseudonym=self.session.reviewer_pseudonym,
            created_at_utc=self.session.created_at_utc,
            current_state=self.fsm.current_state,
            current_sequence=self.fsm.current_sequence,
            total_cases=self.session.total_cases,
            schema_version=self.session.schema_version,
            expiry_utc=self.session.expiry_utc,
            metadata=self.session.metadata,
        )
        self.store.save_session(self.session)

    def _record_audit_event(
        self,
        event_code: str,
        case_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        event = self.fsm._emit_event(
            event_code=event_code,
            actor=self.session.reviewer_pseudonym,
            from_state=self.fsm.current_state,
            to_state=self.fsm.current_state,
            case_id=case_id,
            details=details,
        )
        self.store.append_audit_event(self.session.session_id, event)
        return event

    def authorize(self) -> None:
        """
        Authorize the session to begin evaluation:
        Transitions: SESSION_CREATED -> SESSION_AUTHORIZED -> RESPONSE_DRAFT
        """
        event1 = self.fsm.transition_to(
            SessionState.SESSION_AUTHORIZED,
            actor=self.session.reviewer_pseudonym,
            event_code="EVT_SESSION_AUTHORIZED",
        )
        self.store.append_audit_event(self.session.session_id, event1)

        event2 = self.fsm.transition_to(
            SessionState.RESPONSE_DRAFT,
            actor=self.session.reviewer_pseudonym,
            event_code="EVT_CASE_DRAFT_OPENED",
            details={"sequence": self.fsm.current_sequence},
        )
        self.store.append_audit_event(self.session.session_id, event2)
        self._sync_session_state()

    def submit_and_lock_response(
        self,
        sequence_index: int,
        case_id: str,
        ratings: RatingsPayload,
    ) -> ResponseRecord:
        """
        Submit and lock a case evaluation:
        1. Validates FSM is in RESPONSE_DRAFT.
        2. Enters in-flight RESPONSE_SUBMITTED.
        3. Computes idempotency key, previous hash, and record hash.
        4. Writes to staging buffer, atomic append to master.jsonl, and fsync.
        5. Enters RESPONSE_LOCKED.
        6. Advances to next case (RESPONSE_DRAFT) or closes session (SESSION_CLOSED).
        """
        # Validate sequence matches FSM current sequence
        if sequence_index != self.fsm.current_sequence:
            raise ResponseValidationError(
                f"Sequence mismatch: current active sequence is {self.fsm.current_sequence}, "
                f"attempted submission for {sequence_index}"
            )

        # 1 & 2: Transition to RESPONSE_SUBMITTED
        event_sub = self.fsm.transition_to(
            SessionState.RESPONSE_SUBMITTED,
            actor=self.session.reviewer_pseudonym,
            case_id=case_id,
            event_code="EVT_RESPONSE_SUBMITTED",
            details={"sequence": sequence_index, "case_id": case_id},
        )
        self.store.append_audit_event(self.session.session_id, event_sub)

        try:
            # 3: Compute hashes
            records, latest_hash = self.store.load_master_records(self.session.session_id, verify=True)
            idempotency_key = CanonicalHasher.compute_idempotency_key(
                session_id=self.session.session_id,
                case_id=case_id,
                sequence_index=sequence_index,
            )

            now_utc = datetime.now(timezone.utc).isoformat()
            ratings_json = CanonicalHasher.canonical_json(ratings.to_canonical_dict())

            record_hash = CanonicalHasher.compute_record_hash(
                previous_hash=latest_hash,
                sequence_index=sequence_index,
                case_id=case_id,
                timestamp_utc=now_utc,
                ratings_canonical_json=ratings_json,
            )

            record = ResponseRecord(
                sequence_index=sequence_index,
                case_id=case_id,
                session_id=self.session.session_id,
                reviewer_pseudonym=self.session.reviewer_pseudonym,
                protocol_version=self.session.protocol_version,
                protocol_document_sha256=self.session.protocol_document_sha256,
                package_sha256=self.session.package_sha256,
                timestamp_utc=now_utc,
                ratings=ratings,
                idempotency_key=idempotency_key,
                previous_hash=latest_hash,
                record_hash=record_hash,
                response_revision=1,
                is_locked=True,
            )

            # 4: Durable persistence in storage
            persisted_record = self.store.append_response(self.session.session_id, record)

            # 5: Transition to RESPONSE_LOCKED
            event_lock = self.fsm.transition_to(
                SessionState.RESPONSE_LOCKED,
                actor=self.session.reviewer_pseudonym,
                case_id=case_id,
                event_code="EVT_RESPONSE_LOCK",
                details={
                    "sequence": sequence_index,
                    "case_id": case_id,
                    "record_hash": record_hash,
                },
            )
            self.store.append_audit_event(self.session.session_id, event_lock)

            # 6: Advance or Close
            if sequence_index < self.session.total_cases:
                event_adv = self.fsm.transition_to(
                    SessionState.RESPONSE_DRAFT,
                    actor=self.session.reviewer_pseudonym,
                    event_code="EVT_ADVANCE_NEXT_CASE",
                    details={"new_sequence": self.fsm.current_sequence + 1},
                )
                self.store.append_audit_event(self.session.session_id, event_adv)
            else:
                event_close = self.fsm.transition_to(
                    SessionState.SESSION_CLOSED,
                    actor=self.session.reviewer_pseudonym,
                    event_code="EVT_SESSION_SEALED",
                    details={"total_cases_locked": len(self.fsm.locked_cases)},
                )
                self.store.append_audit_event(self.session.session_id, event_close)

            self._sync_session_state()
            return persisted_record

        except Exception as exc:
            # Rollback to RESPONSE_DRAFT if submission validation failed before commit
            if self.fsm.current_state == SessionState.RESPONSE_SUBMITTED:
                event_rollback = self.fsm.transition_to(
                    SessionState.RESPONSE_DRAFT,
                    actor=self.session.reviewer_pseudonym,
                    case_id=case_id,
                    event_code="EVT_SUBMISSION_ROLLBACK",
                    details={"error": str(exc)},
                )
                self.store.append_audit_event(self.session.session_id, event_rollback)
                self._sync_session_state()
            raise

    def attempt_backtrack_to_case(self, target_sequence: int) -> None:
        """
        Attempt to navigate to or edit an earlier case.
        Raises StateTransitionError if the case has already been locked.
        """
        try:
            self.fsm.attempt_backtrack_to_case(
                target_sequence=target_sequence,
                actor=self.session.reviewer_pseudonym,
            )
        except StateTransitionError as exc:
            # Audit event is already emitted inside attempt_backtrack_to_case
            latest_audit = self.fsm.audit_log[-1]
            self.store.append_audit_event(self.session.session_id, latest_audit)
            raise

    def petition_correction(
        self,
        target_case_id: str,
        target_sequence_index: int,
        target_record_hash: str,
        revised_ratings: RatingsPayload,
        justification: str,
    ) -> CorrectionRecord:
        """
        Submit a formal correction petition:
        1. Transitions RESPONSE_LOCKED -> CORRECTION_REQUESTED.
        2. Verifies target record and appends to exceptions.jsonl.
        3. Transitions CORRECTION_REQUESTED -> CORRECTION_RECORDED -> RESPONSE_LOCKED.
        Master log is NEVER modified.
        """
        petition_id = f"PET-{self.session.session_id}-{target_sequence_index:02d}"
        now_utc = datetime.now(timezone.utc).isoformat()
        prior_state = self.fsm.current_state

        event_req = self.fsm.transition_to(
            SessionState.CORRECTION_REQUESTED,
            actor=self.session.reviewer_pseudonym,
            case_id=target_case_id,
            event_code="EVT_CORRECTION_REQUESTED",
            details={
                "petition_id": petition_id,
                "target_sequence": target_sequence_index,
                "target_case_id": target_case_id,
            },
        )
        self.store.append_audit_event(self.session.session_id, event_req)

        try:
            correction = CorrectionRecord.create(
                petition_id=petition_id,
                session_id=self.session.session_id,
                target_case_id=target_case_id,
                target_sequence_index=target_sequence_index,
                target_record_hash=target_record_hash,
                revised_ratings=revised_ratings,
                justification=justification,
                timestamp_utc=now_utc,
            )
            self.store.append_correction(self.session.session_id, correction)

            event_rec = self.fsm.transition_to(
                SessionState.CORRECTION_RECORDED,
                actor=self.session.reviewer_pseudonym,
                case_id=target_case_id,
                event_code="EVT_CORRECTION_RECORDED",
                details={
                    "petition_id": petition_id,
                    "petition_hash": correction.petition_hash,
                },
            )
            self.store.append_audit_event(self.session.session_id, event_rec)

            event_back = self.fsm.transition_to(
                prior_state,
                actor=self.session.reviewer_pseudonym,
                case_id=target_case_id,
                event_code="EVT_RETURN_FROM_CORRECTION",
            )
            self.store.append_audit_event(self.session.session_id, event_back)

            self._sync_session_state()
            return correction

        except Exception as exc:
            # Deny petition / return to prior state
            event_deny = self.fsm.transition_to(
                prior_state,
                actor=self.session.reviewer_pseudonym,
                case_id=target_case_id,
                event_code="EVT_CORRECTION_DENIED",
                details={"error": str(exc)},
            )
            self.store.append_audit_event(self.session.session_id, event_deny)
            self._sync_session_state()
            raise

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the session's complete cryptographic integrity.
        """
        genesis_hash = self.store.get_genesis_hash(self.session.session_id)
        records, latest_hash = self.store.load_master_records(self.session.session_id, verify=False)
        valid_chain, error_msg = CanonicalHasher.verify_chain(records, genesis_hash)

        corrections = self.store.load_corrections(self.session.session_id)
        audit_events = self.store.load_audit_events(self.session.session_id)

        return {
            "session_id": self.session.session_id,
            "reviewer_pseudonym": self.session.reviewer_pseudonym,
            "assigned_package": self.session.assigned_package,
            "current_state": self.fsm.current_state.value,
            "total_cases": self.session.total_cases,
            "committed_cases_count": len(records),
            "locked_cases": sorted(list(self.fsm.locked_cases)),
            "genesis_hash": genesis_hash,
            "latest_record_hash": latest_hash,
            "chain_valid": valid_chain,
            "chain_verification_error": error_msg,
            "corrections_count": len(corrections),
            "audit_events_count": len(audit_events),
            "is_sealed": self.fsm.current_state == SessionState.SESSION_CLOSED,
        }
