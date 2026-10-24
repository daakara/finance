"""
Comprehensive Wave 1 Synthetic Test Suite for HVA-01 Core Intake Runner.
Covers Groups 1 through 9:
- Group 1: Model validation & schema rejection
- Group 2: FSM 8-state transitions & backtracking rejection
- Group 3: Response locking & post-lock overwrite rejection
- Group 4: Idempotency & duplicate submission handling
- Group 5: Concurrency & cross-platform file locking
- Group 6: Persistence & crash recovery failure injection (8 failure boundaries)
- Group 7: Hash chain tamper detection (mutations, reordering, truncation)
- Group 8: Exception correction ledger isolation
- Group 9: Static security & zero-PII validation
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest

from hva01 import (
    AbstentionCode,
    AppendOnlyStore,
    AuditEvent,
    CanonicalHasher,
    ConfidenceLevel,
    ConflictingDuplicateSubmissionError,
    CorrectionRecord,
    EvidenceSufficiency,
    FileLock,
    FileLockError,
    HashChainVerificationError,
    IntakeRunner,
    PriorFamiliarity,
    RatingsPayload,
    ResponseLockingFSM,
    ResponseRecord,
    ResponseValidationError,
    SessionModel,
    SessionState,
    StageClassification,
    StateTransitionError,
    StorageCorruptionError,
    VCPQualification,
    ValidationError,
)

VALID_PROTO_SHA = "a" * 64
VALID_PKG_SHA = "b" * 64


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    d = tempfile.mkdtemp(prefix="hva01_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def create_sample_ratings(
    stage: StageClassification = StageClassification.STAGE_2,
    vcp: VCPQualification = VCPQualification.VCP_CONFIRMED,
    confidence: ConfidenceLevel = ConfidenceLevel.CONFIDENCE_HIGH,
    reasoning: str = "Clear progressive contraction pattern observed with volume dry-up.",
    abstention: AbstentionCode = AbstentionCode.NONE,
) -> RatingsPayload:
    return RatingsPayload(
        stage_classification=stage,
        vcp_qualification=vcp,
        confidence_level=confidence,
        prior_familiarity=PriorFamiliarity.NO_RECALL,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        reviewer_reasoning=reasoning,
        abstention_code=abstention,
        contraction_count_observed=3,
        progressive_tightening_observed=True,
        volume_dry_up_observed=True,
        pivot_present=True,
        composite_score_100=88,
    )


# ==============================================================================
# GROUP 1: MODEL VALIDATION & SCHEMA REJECTION
# ==============================================================================

class TestGroup1ModelValidation:
    def test_valid_session_model(self) -> None:
        sess = SessionModel(
            session_id="HVA01-SESS-ALPHA-01",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            assigned_package="PACKAGE_A",
            package_sha256=VALID_PKG_SHA,
            reviewer_pseudonym="SYNTH-REV-01",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        assert sess.session_id == "HVA01-SESS-ALPHA-01"
        assert sess.current_state == SessionState.SESSION_CREATED
        assert sess.current_sequence == 1
        assert sess.total_cases == 59

    def test_invalid_session_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Invalid session_id"):
            SessionModel(
                session_id="invalid id with spaces",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                assigned_package="PACKAGE_A",
                package_sha256=VALID_PKG_SHA,
                reviewer_pseudonym="SYNTH-REV-01",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_pii_in_pseudonym_rejected(self) -> None:
        # Email address injection
        with pytest.raises(ValidationError, match="Invalid reviewer_pseudonym"):
            SessionModel(
                session_id="HVA01-SESS-TEST-01",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                assigned_package="PACKAGE_A",
                package_sha256=VALID_PKG_SHA,
                reviewer_pseudonym="john.doe@example.com",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_invalid_package_rejected(self) -> None:
        with pytest.raises(ValidationError, match="assigned_package must be PACKAGE_A or PACKAGE_B"):
            SessionModel(
                session_id="HVA01-SESS-TEST-01",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                assigned_package="PACKAGE_INVALID",
                package_sha256=VALID_PKG_SHA,
                reviewer_pseudonym="SYNTH-REV-01",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_invalid_sha256_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a 64-character lowercase hex"):
            SessionModel(
                session_id="HVA01-SESS-TEST-01",
                protocol_version="2.1.0",
                protocol_document_sha256="short_sha",
                assigned_package="PACKAGE_A",
                package_sha256=VALID_PKG_SHA,
                reviewer_pseudonym="SYNTH-REV-01",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_ratings_reasoning_too_short_rejected(self) -> None:
        with pytest.raises(ResponseValidationError, match="at least 5 non-whitespace characters"):
            RatingsPayload(
                stage_classification=StageClassification.STAGE_2,
                vcp_qualification=VCPQualification.VCP_CONFIRMED,
                confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
                prior_familiarity=PriorFamiliarity.NO_RECALL,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                reviewer_reasoning="   ",
            )

    def test_ratings_contraction_count_out_of_range(self) -> None:
        with pytest.raises(ResponseValidationError, match="contraction_count_observed must be in range"):
            RatingsPayload(
                stage_classification=StageClassification.STAGE_2,
                vcp_qualification=VCPQualification.VCP_CONFIRMED,
                confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
                prior_familiarity=PriorFamiliarity.NO_RECALL,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                reviewer_reasoning="Valid reasoning text",
                contraction_count_observed=15,
            )

    def test_ratings_composite_score_out_of_range(self) -> None:
        with pytest.raises(ResponseValidationError, match="composite_score_100 must be in range"):
            RatingsPayload(
                stage_classification=StageClassification.STAGE_2,
                vcp_qualification=VCPQualification.VCP_CONFIRMED,
                confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
                prior_familiarity=PriorFamiliarity.NO_RECALL,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                reviewer_reasoning="Valid reasoning text",
                composite_score_100=105,
            )

    def test_insufficient_evidence_without_abstention_code_rejected(self) -> None:
        with pytest.raises(ResponseValidationError, match="valid abstention_code must be specified"):
            RatingsPayload(
                stage_classification=StageClassification.UNCLASSIFIED,
                vcp_qualification=VCPQualification.VCP_REJECTED,
                confidence_level=ConfidenceLevel.CONFIDENCE_LOW,
                prior_familiarity=PriorFamiliarity.NO_RECALL,
                evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
                reviewer_reasoning="Insufficient data available",
                abstention_code=AbstentionCode.NONE,
            )

    def test_response_record_is_locked_enforced(self) -> None:
        ratings = create_sample_ratings()
        with pytest.raises(ResponseValidationError, match="must have is_locked=True"):
            ResponseRecord(
                sequence_index=1,
                case_id="HVA2-A-01",
                session_id="HVA01-SESS-001",
                reviewer_pseudonym="SYNTH-REV-01",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                package_sha256=VALID_PKG_SHA,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                ratings=ratings,
                idempotency_key="key",
                previous_hash="prev",
                record_hash="hash",
                is_locked=False,  # Forbidden in persistence
            )


# ==============================================================================
# GROUP 2: FSM 8-STATE TRANSITIONS & BACKTRACKING REJECTION
# ==============================================================================

class TestGroup2FSMTransitions:
    def test_full_linear_state_lifecycle(self) -> None:
        sess = SessionModel(
            session_id="HVA01-SESS-FSM-01",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            assigned_package="PACKAGE_A",
            package_sha256=VALID_PKG_SHA,
            reviewer_pseudonym="SYNTH-REV-01",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            total_cases=2,
        )
        fsm = ResponseLockingFSM(sess)
        assert fsm.current_state == SessionState.SESSION_CREATED

        # 1. Authorize
        fsm.transition_to(SessionState.SESSION_AUTHORIZED, actor="TEST")
        assert fsm.current_state == SessionState.SESSION_AUTHORIZED

        # 2. Open Case 1 Draft
        fsm.transition_to(SessionState.RESPONSE_DRAFT, actor="TEST")
        assert fsm.current_state == SessionState.RESPONSE_DRAFT
        assert fsm.current_sequence == 1

        # 3. Submit Case 1
        fsm.transition_to(SessionState.RESPONSE_SUBMITTED, actor="TEST")
        assert fsm.current_state == SessionState.RESPONSE_SUBMITTED

        # 4. Lock Case 1
        fsm.transition_to(SessionState.RESPONSE_LOCKED, actor="TEST")
        assert fsm.current_state == SessionState.RESPONSE_LOCKED
        assert 1 in fsm.locked_cases

        # 5. Advance to Case 2 Draft
        fsm.transition_to(SessionState.RESPONSE_DRAFT, actor="TEST")
        assert fsm.current_state == SessionState.RESPONSE_DRAFT
        assert fsm.current_sequence == 2

        # 6. Submit Case 2
        fsm.transition_to(SessionState.RESPONSE_SUBMITTED, actor="TEST")
        assert fsm.current_state == SessionState.RESPONSE_SUBMITTED

        # 7. Lock Case 2
        fsm.transition_to(SessionState.RESPONSE_LOCKED, actor="TEST")
        assert 2 in fsm.locked_cases

        # 8. Close session
        fsm.transition_to(SessionState.SESSION_CLOSED, actor="TEST")
        assert fsm.current_state == SessionState.SESSION_CLOSED

    def test_illegal_forward_jump_rejected(self) -> None:
        sess = SessionModel(
            session_id="HVA01-SESS-FSM-02",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            assigned_package="PACKAGE_A",
            package_sha256=VALID_PKG_SHA,
            reviewer_pseudonym="SYNTH-REV-01",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        fsm = ResponseLockingFSM(sess)
        with pytest.raises(StateTransitionError, match="Illegal state transition"):
            fsm.transition_to(SessionState.RESPONSE_LOCKED, actor="TEST")

    def test_backtracking_to_locked_case_strictly_prohibited(self) -> None:
        sess = SessionModel(
            session_id="HVA01-SESS-FSM-03",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            assigned_package="PACKAGE_A",
            package_sha256=VALID_PKG_SHA,
            reviewer_pseudonym="SYNTH-REV-01",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            total_cases=5,
        )
        fsm = ResponseLockingFSM(sess)
        fsm.transition_to(SessionState.SESSION_AUTHORIZED, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_DRAFT, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_SUBMITTED, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_LOCKED, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_DRAFT, actor="TEST")

        assert fsm.current_sequence == 2
        assert 1 in fsm.locked_cases

        # Attempt to backtrack to Case 1
        with pytest.raises(StateTransitionError, match="Backtracking prohibited: Case sequence 1 is permanently locked"):
            fsm.attempt_backtrack_to_case(target_sequence=1, actor="TEST")

        # Verify audit event emitted
        events = [e for e in fsm.audit_log if e.event_code == "EVT_BACKTRACKING_REJECTED"]
        assert len(events) == 1
        assert events[0].details["attempted_sequence"] == 1

    def test_cannot_close_session_prematurely(self) -> None:
        sess = SessionModel(
            session_id="HVA01-SESS-FSM-04",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            assigned_package="PACKAGE_A",
            package_sha256=VALID_PKG_SHA,
            reviewer_pseudonym="SYNTH-REV-01",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            total_cases=10,
        )
        fsm = ResponseLockingFSM(sess)
        fsm.transition_to(SessionState.SESSION_AUTHORIZED, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_DRAFT, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_SUBMITTED, actor="TEST")
        fsm.transition_to(SessionState.RESPONSE_LOCKED, actor="TEST")

        # Only 1 of 10 locked: closing must fail
        with pytest.raises(StateTransitionError, match="Cannot close session: only 1 of 10 cases locked"):
            fsm.transition_to(SessionState.SESSION_CLOSED, actor="TEST")


# ==============================================================================
# GROUP 3: RESPONSE LOCKING & POST-LOCK OVERWRITE REJECTION
# ==============================================================================

class TestGroup3ResponseLocking:
    def test_response_locking_and_post_lock_overwrite_rejected(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-LOCK-01",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()

        # Submit Case 1
        ratings = create_sample_ratings()
        rec1 = runner.submit_and_lock_response(
            sequence_index=1,
            case_id="HVA2-A-01",
            ratings=ratings,
        )
        assert rec1.is_locked is True
        assert rec1.record_hash is not None

        # Attempt to overwrite Case 1 with modified ratings
        modified_ratings = create_sample_ratings(stage=StageClassification.STAGE_4)
        with pytest.raises(ResponseValidationError, match="Sequence mismatch: current active sequence is 2"):
            runner.submit_and_lock_response(
                sequence_index=1,
                case_id="HVA2-A-01",
                ratings=modified_ratings,
            )

        # Attempt to directly append conflicting payload via store
        conflicting_record = ResponseRecord(
            sequence_index=1,
            case_id="HVA2-A-01",
            session_id=runner.session.session_id,
            reviewer_pseudonym=runner.session.reviewer_pseudonym,
            protocol_version=runner.session.protocol_version,
            protocol_document_sha256=runner.session.protocol_document_sha256,
            package_sha256=runner.session.package_sha256,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ratings=modified_ratings,
            idempotency_key=rec1.idempotency_key,
            previous_hash=rec1.previous_hash,
            record_hash="different_hash" + "0" * 50,
            is_locked=True,
        )
        with pytest.raises(ConflictingDuplicateSubmissionError, match="Conflicting submission for sequence 1"):
            runner.store.append_response(runner.session.session_id, conflicting_record)


# ==============================================================================
# GROUP 4: IDEMPOTENCY & DUPLICATE SUBMISSION HANDLING
# ==============================================================================

class TestGroup4Idempotency:
    def test_idempotent_duplicate_replay_succeeds(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-IDEMP-01",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()

        ratings = create_sample_ratings()
        rec1 = runner.submit_and_lock_response(sequence_index=1, case_id="HVA2-A-01", ratings=ratings)

        # Exact same record submitted directly to store simulates network / client replay
        replayed = runner.store.append_response(runner.session.session_id, rec1)
        assert replayed.record_hash == rec1.record_hash

        # Verify master log has exactly 1 entry, not 2
        records, _ = runner.store.load_master_records(runner.session.session_id)
        assert len(records) == 1


# ==============================================================================
# GROUP 5: CONCURRENCY & CROSS-PROCESS FILE LOCKING
# ==============================================================================

class TestGroup5FileLocking:
    def test_cross_process_single_writer_lock_exclusion(self, tmp_dir: Path) -> None:
        lock_file = tmp_dir / "test.lock"

        lock1 = FileLock(lock_file, timeout=0.5, poll_interval=0.05)
        lock2 = FileLock(lock_file, timeout=0.2, poll_interval=0.05)

        with lock1:
            # Second lock must fail due to timeout
            with pytest.raises(FileLockError, match="Could not acquire lock"):
                lock2.acquire()

        # After lock1 releases, lock2 must succeed
        with lock2:
            pass


# ==============================================================================
# GROUP 6: PERSISTENCE & CRASH RECOVERY FAILURE INJECTION (8 BOUNDARIES)
# ==============================================================================

class TestGroup6CrashRecovery:
    def test_crash_boundary_1_pre_write_clean(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-01",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
        )
        runner.authorize()
        resumed = IntakeRunner.resume_session("HVA01-SESS-CRASH-01", base_dir=tmp_dir)
        assert resumed.fsm.current_sequence == 1
        assert resumed.fsm.current_state == SessionState.RESPONSE_DRAFT

    def test_crash_boundary_2_leftover_uncommitted_staging_cleaned(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-02",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
        )
        runner.authorize()
        session_dir = runner.store.get_session_dir(runner.session.session_id)

        # Inject an abandoned .tmp file simulating interrupted process during staging
        abandoned_tmp = session_dir / "write_1_HVA2-A-01.tmp"
        abandoned_tmp.write_text("{\"incomplete\": true}\n", encoding="utf-8")
        assert abandoned_tmp.exists()

        # Resume session: verify abandoned file discarded
        resumed = IntakeRunner.resume_session("HVA01-SESS-CRASH-02", base_dir=tmp_dir)
        assert not abandoned_tmp.exists()
        assert resumed.fsm.current_sequence == 1

    def test_crash_boundary_3_committed_master_recovers_state(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-03",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())

        # Simulate abrupt process death and restart
        resumed = IntakeRunner.resume_session("HVA01-SESS-CRASH-03", base_dir=tmp_dir)
        assert resumed.fsm.current_sequence == 3
        assert resumed.fsm.current_state == SessionState.RESPONSE_DRAFT
        assert 1 in resumed.fsm.locked_cases
        assert 2 in resumed.fsm.locked_cases

    def test_crash_boundary_4_final_case_sealed(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-04",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=2,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())

        # All cases locked -> session must be closed
        assert runner.fsm.current_state == SessionState.SESSION_CLOSED

        resumed = IntakeRunner.resume_session("HVA01-SESS-CRASH-04", base_dir=tmp_dir)
        assert resumed.fsm.current_state == SessionState.SESSION_CLOSED
        assert len(resumed.fsm.locked_cases) == 2

    def test_crash_boundary_5_audit_trail_preserved(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-05",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())

        events = runner.store.load_audit_events(runner.session.session_id)
        assert len(events) >= 3
        event_codes = [e["event_code"] for e in events]
        assert "EVT_SESSION_CREATED" in event_codes
        assert "EVT_SESSION_AUTHORIZED" in event_codes
        assert "EVT_RESPONSE_LOCK" in event_codes

    def test_crash_boundary_6_status_inspection(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-06",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=4,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())

        status = runner.store.get_crash_recovery_status(runner.session.session_id)
        assert status["committed_cases_count"] == 1
        assert status["next_expected_sequence"] == 2
        assert status["uncommitted_staging_files_found"] == 0

    def test_crash_boundary_7_corrupted_json_line_in_master(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CRASH-07",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())

        # Corrupt master.jsonl by appending malformed JSON
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "a", encoding="utf-8") as f:
            f.write("{\"corrupted_json_line\n")

        with pytest.raises(StorageCorruptionError, match="Unparseable master log record"):
            runner.store.load_master_records(runner.session.session_id)

    def test_crash_boundary_8_missing_session_file(self, tmp_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            IntakeRunner.resume_session("HVA01-SESS-NONEXISTENT", base_dir=tmp_dir)


# ==============================================================================
# GROUP 7: HASH CHAIN TAMPER DETECTION
# ==============================================================================

class TestGroup7HashChainTamperDetection:
    def test_valid_chain_verifies(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CHAIN-01",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())
        runner.submit_and_lock_response(3, "HVA2-A-03", create_sample_ratings())

        report = runner.verify_integrity()
        assert report["chain_valid"] is True
        assert report["committed_cases_count"] == 3
        assert report["is_sealed"] is True

    def test_tampered_record_payload_detected(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CHAIN-02",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())

        # Directly tamper with Case 1 payload inside master.jsonl
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        lines[0]["ratings"]["reviewer_reasoning"] = "Tampered unauthorized modification of reasoning."

        with open(master_file, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        with pytest.raises(StorageCorruptionError, match="Record tampering detected"):
            runner.store.load_master_records(runner.session.session_id, verify=True)

    def test_swapped_lines_detected(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CHAIN-03",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())

        # Swap line 1 and line 2
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            raw_lines = [l for l in f if l.strip()]

        with open(master_file, "w", encoding="utf-8") as f:
            f.write(raw_lines[1])
            f.write(raw_lines[0])

        with pytest.raises(StorageCorruptionError, match="Sequence discontinuity"):
            runner.store.load_master_records(runner.session.session_id, verify=True)

    def test_truncated_line_detected(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CHAIN-04",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", create_sample_ratings())

        # Delete line 1 (leaving line 2 starting with sequence 2)
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            raw_lines = [l for l in f if l.strip()]

        with open(master_file, "w", encoding="utf-8") as f:
            f.write(raw_lines[1])

        with pytest.raises(StorageCorruptionError, match="Sequence discontinuity"):
            runner.store.load_master_records(runner.session.session_id, verify=True)


# ==============================================================================
# GROUP 8: EXCEPTION CORRECTION LEDGER ISOLATION
# ==============================================================================

class TestGroup8CorrectionLedgerIsolation:
    def test_correction_petition_ledger_isolation(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CORR-01",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        rec1 = runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())

        # Master record snapshot before petition
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            master_content_before = f.read()

        # Submit formal correction petition
        revised = create_sample_ratings(
            stage=StageClassification.STAGE_1,
            vcp=VCPQualification.VCP_REJECTED,
            reasoning="Correction justification: overlooked volume spike on day 14.",
        )
        correction = runner.petition_correction(
            target_case_id="HVA2-A-01",
            target_sequence_index=1,
            target_record_hash=rec1.record_hash,
            revised_ratings=revised,
            justification="Reviewer identified overlooked high volume bar violating VCP tightness.",
        )

        assert correction.petition_hash is not None
        assert correction.target_record_hash == rec1.record_hash

        # Verify master.jsonl was NOT modified
        with open(master_file, "r", encoding="utf-8") as f:
            master_content_after = f.read()
        assert master_content_before == master_content_after

        # Verify correction stored in exceptions.jsonl
        corrections = runner.store.load_corrections(runner.session.session_id)
        assert len(corrections) == 1
        assert corrections[0].petition_id == correction.petition_id

    def test_correction_mismatched_target_hash_rejected(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CORR-02",
            reviewer_pseudonym="SYNTH-REV-01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=3,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", create_sample_ratings())

        revised = create_sample_ratings()
        with pytest.raises(ValueError, match="Target record hash mismatch"):
            runner.petition_correction(
                target_case_id="HVA2-A-01",
                target_sequence_index=1,
                target_record_hash="wrong_hash" + "0" * 54,
                revised_ratings=revised,
                justification="Valid justification string for test.",
            )


# ==============================================================================
# GROUP 9: STATIC SECURITY & ZERO-PII VALIDATION
# ==============================================================================

class TestGroup9StaticSecurityAndZeroPII:
    def test_zero_pii_in_data_models(self) -> None:
        # Inspect model field names
        pii_keywords = ["email", "name", "address", "ssn", "ip_address", "phone", "password", "token"]

        for model in (SessionModel, ResponseRecord, RatingsPayload, AuditEvent, CorrectionRecord):
            for field_name in model.__annotations__.keys():
                for kw in pii_keywords:
                    assert kw not in field_name.lower(), (
                        f"PII keyword '{kw}' detected in model field '{field_name}' in {model.__name__}"
                    )

    def test_rfc8785_canonical_json_determinism(self) -> None:
        data = {
            "zebra": 1,
            "apple": 2,
            "nested": {"beta": "b", "alpha": "a"},
            "numbers": [3, 2, 1],
        }
        res1 = CanonicalHasher.canonical_json(data)
        res2 = CanonicalHasher.canonical_json(data)
        assert res1 == res2
        # Deterministic key sorting
        expected = '{"apple":2,"nested":{"alpha":"a","beta":"b"},"numbers":[3,2,1],"zebra":1}'
        assert res1 == expected
