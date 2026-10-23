"""
HVA-01 Intake Runner Coordinator.
Orchestrates FSM state transitions, cryptographic hashing, and transactional append-only storage.
"""

from __future__ import annotations

import errno
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


class PackageAccessViolation(PermissionError):
    """Raised when an attempt is made to access assets outside the assigned package boundary (SC-10)."""
    pass


class RestrictedLabelAccessError(PermissionError):
    """Raised when an attempt is made to access restricted or unblinded evaluation labels (SC-11)."""
    pass


class StorageExhaustedError(OSError):
    """Raised when custody storage runs out of disk space during append (SC-19)."""
    pass


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
        expected_protocol_sha256: Optional[str] = None,
        expected_package_sha256: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntakeRunner:
        """
        Create a new evaluation session with transactional storage and genesis hash computation.
        Enforces SC-14 (protocol hash match) and SC-15 (package hash match).
        """
        if expected_protocol_sha256 is not None:
            if protocol_document_sha256.lower().strip() != expected_protocol_sha256.lower().strip():
                raise ValidationError(
                    f"PROTOCOL_VERSION_DIGEST_MISMATCH: protocol document SHA-256 "
                    f"'{protocol_document_sha256}' does not match approved '{expected_protocol_sha256}'."
                )

        if expected_package_sha256 is not None:
            if package_sha256.lower().strip() != expected_package_sha256.lower().strip():
                raise ValidationError(
                    f"PACKAGE_INTEGRITY_VERIFICATION_FAILED: package manifest SHA-256 "
                    f"'{package_sha256}' does not match sealed digest '{expected_package_sha256}'."
                )

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

        # Check if session was formally withdrawn
        if session.metadata.get("is_withdrawn", False):
            restored_state = SessionState.SESSION_CLOSED
            restored_sequence = len(records)
        else:
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
                "committed_count": len(records),
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

    def resolve_case_asset(self, case_id: str, relative_path: Optional[str] = None) -> Path:
        """
        Verify case and asset adherence to package assignment and blinding security rules.
        Enforces SC-10 (Package boundary confinement) and SC-11 (Restricted label isolation).
        """
        # Path traversal guard (checked first to prevent traversal attacks)
        if relative_path:
            norm = relative_path.replace("\\", "/")
            if (
                ".." in norm
                or norm.startswith("/")
                or norm.startswith("\\")
                or Path(relative_path).is_absolute()
                or bool(Path(relative_path).drive)
            ):
                self._record_audit_event(
                    event_code="EVT_ACCESS_VIOLATION",
                    case_id=case_id,
                    details={"reason": "Path traversal attempt", "path": relative_path},
                )
                raise PackageAccessViolation(f"Path traversal access violation: '{relative_path}'")

        # SC-11: Restricted label access check
        prohibited_terms = [
            "RESTRICTED",
            "UNBLIND",
            "MAPPING",
            "ANSWER_KEY",
            "GROUND_TRUTH",
            "ID_MAP",
        ]
        target_str = f"{case_id} {relative_path or ''}".upper()
        for term in prohibited_terms:
            if term in target_str:
                self._record_audit_event(
                    event_code="EVT_CUSTODY_BREACH",
                    case_id=case_id,
                    details={"requested": target_str, "reason": "Restricted label access attempt blocked."},
                )
                raise RestrictedLabelAccessError(
                    f"Restricted label access violation: '{term}' access prohibited during blinded evaluation."
                )

        # SC-10: Cross-package boundary check
        # PACKAGE_A reviewers can only evaluate HVA2-A-* cases from charts_a
        # PACKAGE_B reviewers can only evaluate HVA2-B-* cases from charts_b
        if self.session.assigned_package == "PACKAGE_A":
            if case_id.startswith("HVA2-B-") or (relative_path and "charts_b" in relative_path.lower()):
                self._record_audit_event(
                    event_code="EVT_ACCESS_VIOLATION",
                    case_id=case_id,
                    details={"assigned": "PACKAGE_A", "attempted": case_id, "path": relative_path},
                )
                raise PackageAccessViolation(
                    f"Package access violation: Reviewer assigned to PACKAGE_A cannot access '{case_id}' or PACKAGE_B assets."
                )
        elif self.session.assigned_package == "PACKAGE_B":
            if case_id.startswith("HVA2-A-") or (relative_path and "charts_a" in relative_path.lower()):
                self._record_audit_event(
                    event_code="EVT_ACCESS_VIOLATION",
                    case_id=case_id,
                    details={"assigned": "PACKAGE_B", "attempted": case_id, "path": relative_path},
                )
                raise PackageAccessViolation(
                    f"Package access violation: Reviewer assigned to PACKAGE_B cannot access '{case_id}' or PACKAGE_A assets."
                )

        return Path(relative_path or f"{case_id}.svg")

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
        1. Validates package boundary and blinding (SC-10, SC-11).
        2. Validates FSM is in RESPONSE_DRAFT.
        3. Enters in-flight RESPONSE_SUBMITTED.
        4. Computes idempotency key, previous hash, and record hash.
        5. Writes to staging buffer, atomic append to master.jsonl, and fsync.
        6. Enters RESPONSE_LOCKED.
        7. Advances to next case (RESPONSE_DRAFT) or closes session (SESSION_CLOSED).
        """
        # Validate package boundary and blinding
        self.resolve_case_asset(case_id)

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
                is_enospc = (
                    isinstance(exc, StorageExhaustedError)
                    or (isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC)
                )
                err_code = "EVT_STORAGE_EXHAUSTED" if is_enospc else "EVT_SUBMISSION_ROLLBACK"
                event_rollback = self.fsm.transition_to(
                    SessionState.RESPONSE_DRAFT,
                    actor=self.session.reviewer_pseudonym,
                    case_id=case_id,
                    event_code=err_code,
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
        except StateTransitionError:
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
        1. Transitions to CORRECTION_REQUESTED.
        2. Verifies target record and appends to exceptions.jsonl.
        3. Transitions CORRECTION_REQUESTED -> CORRECTION_RECORDED -> prior state.
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

    def withdraw_session(self, justification: str) -> Dict[str, Any]:
        """
        Formally withdraw the session, freezing all already committed cases and sealing the log.
        Enforces Scenario SC-20.
        """
        if self.fsm.current_state == SessionState.SESSION_CLOSED:
            raise StateTransitionError("Session is already closed.")

        now_utc = datetime.now(timezone.utc).isoformat()
        committed_count = len(self.fsm.locked_cases)
        coverage_str = f"{committed_count}/{self.session.total_cases}"

        details = {
            "committed_cases": committed_count,
            "coverage_ratio": coverage_str,
            "justification": justification,
            "withdrawal_timestamp_utc": now_utc,
        }

        event_seal = self.fsm.transition_to(
            SessionState.SESSION_CLOSED,
            actor=self.session.reviewer_pseudonym,
            event_code="EVT_WITHDRAWAL_SEAL",
            details=details,
            is_withdrawal=True,
        )
        self.store.append_audit_event(self.session.session_id, event_seal)

        # Update session metadata
        self.session.metadata["is_withdrawn"] = True
        self.session.metadata["withdrawal_justification"] = justification
        self.session.metadata["withdrawal_timestamp_utc"] = now_utc
        self.session.metadata["coverage_ratio"] = coverage_str
        self._sync_session_state()

        return {
            "session_id": self.session.session_id,
            "status": "WITHDRAWN",
            "committed_cases": committed_count,
            "coverage_ratio": coverage_str,
            "withdrawal_timestamp_utc": now_utc,
            "is_sealed": True,
        }

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
            "is_withdrawn": self.session.metadata.get("is_withdrawn", False),
        }
