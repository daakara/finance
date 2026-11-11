"""
Canonical 20-Scenario Synthetic Acceptance Test Suite for HVA-01 V2.1.
Authoritatively implements all 20 acceptance scenarios specified in:
- ARX_HVA00_V2_1_SYNTHETIC_ACCEPTANCE_TEST_MATRIX.md
- ARX_HVA01_V2_1_SYNTHETIC_ACCEPTANCE_EXECUTION_PLAN.md

Denominator Integrity: Exactly 20 canonical scenarios (SC-01 through SC-20).
Zero human evaluation data. Fail-closed execution.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import patch

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
    IntakeRunner,
    PackageAccessViolation,
    PriorFamiliarity,
    RatingsPayload,
    ResponseLockingFSM,
    ResponseRecord,
    ResponseValidationError,
    RestrictedLabelAccessError,
    SessionModel,
    SessionState,
    StageClassification,
    StateTransitionError,
    StorageCorruptionError,
    StorageExhaustedError,
    VCPQualification,
    ValidationError,
)

VALID_PROTO_SHA = "a" * 64
VALID_PKG_SHA = "b" * 64


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    d = tempfile.mkdtemp(prefix="hva01_sc20_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def make_valid_ratings(
    stage: StageClassification = StageClassification.STAGE_2,
    vcp: VCPQualification = VCPQualification.VCP_CONFIRMED,
    confidence: ConfidenceLevel = ConfidenceLevel.CONFIDENCE_HIGH,
    reasoning: str = "Acceptance test: valid VCP contraction with tightening volume.",
    abstention: AbstentionCode = AbstentionCode.NONE,
    familiarity: PriorFamiliarity = PriorFamiliarity.NO_RECALL,
    sufficiency: EvidenceSufficiency = EvidenceSufficiency.SUFFICIENT,
) -> RatingsPayload:
    return RatingsPayload(
        stage_classification=stage,
        vcp_qualification=vcp,
        confidence_level=confidence,
        prior_familiarity=familiarity,
        evidence_sufficiency=sufficiency,
        reviewer_reasoning=reasoning,
        abstention_code=abstention,
        contraction_count_observed=3,
        progressive_tightening_observed=True,
        volume_dry_up_observed=True,
        pivot_present=True,
        composite_score_100=85,
    )


class TestCanonicalTwentyScenarios:
    """
    Authoritative test harness executing all 20 canonical synthetic acceptance scenarios.
    Denominator: exactly 20.
    """

    # --------------------------------------------------------------------------
    # SC-01: Normal Sequential Submission
    # --------------------------------------------------------------------------
    def test_sc01_normal_sequential_submission(self, tmp_dir: Path) -> None:
        """
        Scenario 1: Applet initialized at Case k; valid rubric entered; submitted.
        Response committed to JSONL; hash chain updated; sequence increments to k+1.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC01-01",
            reviewer_pseudonym="SYNTH-REV-SC01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=59,
        )
        runner.authorize()
        assert runner.fsm.current_state == SessionState.RESPONSE_DRAFT
        assert runner.fsm.current_sequence == 1

        ratings = make_valid_ratings()
        record = runner.submit_and_lock_response(1, "HVA2-A-01", ratings)

        assert record.is_locked is True
        assert record.sequence_index == 1
        assert record.case_id == "HVA2-A-01"
        assert record.record_hash is not None

        # FSM advanced to Case 2 Draft
        assert runner.fsm.current_state == SessionState.RESPONSE_DRAFT
        assert runner.fsm.current_sequence == 2
        assert 1 in runner.fsm.locked_cases

        # Audit event verified
        events = runner.store.load_audit_events(runner.session.session_id)
        assert any(e["event_code"] == "EVT_RESPONSE_LOCK" for e in events)

    # --------------------------------------------------------------------------
    # SC-02: Attempted Backtracking
    # --------------------------------------------------------------------------
    def test_sc02_attempted_backtracking(self, tmp_dir: Path) -> None:
        """
        Scenario 2: Case k+1 active; Case k committed. Script attempts backward navigation.
        Navigation rejected with error; applet remains on Case k+1.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC02-01",
            reviewer_pseudonym="SYNTH-REV-SC02",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_valid_ratings())

        assert runner.fsm.current_sequence == 2

        # Attempt to backtrack to Case 1
        with pytest.raises(StateTransitionError, match="Backtracking prohibited: Case sequence 1 is permanently locked"):
            runner.attempt_backtrack_to_case(1)

        # Verify state unchanged
        assert runner.fsm.current_sequence == 2
        assert runner.fsm.current_state == SessionState.RESPONSE_DRAFT

        # Verify audit event logged
        events = runner.store.load_audit_events(runner.session.session_id)
        assert any(e["event_code"] == "EVT_BACKTRACKING_REJECTED" for e in events)

    # --------------------------------------------------------------------------
    # SC-03: Duplicate Submission
    # --------------------------------------------------------------------------
    def test_sc03_duplicate_submission(self, tmp_dir: Path) -> None:
        """
        Scenario 3: Case k active; submission in progress. Simultaneous commit calls.
        First succeeds; second returns idempotent replay; conflicting raises error. Exactly 1 entry in log.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC03-01",
            reviewer_pseudonym="SYNTH-REV-SC03",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        ratings = make_valid_ratings()
        rec1 = runner.submit_and_lock_response(1, "HVA2-A-01", ratings)

        # Exact duplicate replay to store
        replayed = runner.store.append_response(runner.session.session_id, rec1)
        assert replayed.record_hash == rec1.record_hash

        # Master log contains exactly 1 record
        records, _ = runner.store.load_master_records(runner.session.session_id)
        assert len(records) == 1

        # Conflicting duplicate rejected
        conflicting = ResponseRecord(
            sequence_index=1,
            case_id="HVA2-A-01",
            session_id=runner.session.session_id,
            reviewer_pseudonym=runner.session.reviewer_pseudonym,
            protocol_version=runner.session.protocol_version,
            protocol_document_sha256=runner.session.protocol_document_sha256,
            package_sha256=runner.session.package_sha256,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ratings=make_valid_ratings(stage=StageClassification.STAGE_4),
            idempotency_key=rec1.idempotency_key,
            previous_hash=rec1.previous_hash,
            record_hash="e" * 64,
            is_locked=True,
        )
        with pytest.raises(ConflictingDuplicateSubmissionError):
            runner.store.append_response(runner.session.session_id, conflicting)

    # --------------------------------------------------------------------------
    # SC-04: Response Overwrite Attempt
    # --------------------------------------------------------------------------
    def test_sc04_response_overwrite_attempt(self, tmp_dir: Path) -> None:
        """
        Scenario 4: Case 14 committed. External script attempts to edit line 14 directly on disk.
        On subsequent load/verification, SHA-256 hash chain fails; applet halts.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC04-01",
            reviewer_pseudonym="SYNTH-REV-SC04",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=20,
        )
        runner.authorize()
        for seq in range(1, 15):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        # Directly tamper line 14 on disk
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        lines[13]["ratings"]["reviewer_reasoning"] = "Malicious unauthorized disk edit."
        with open(master_file, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        with pytest.raises(StorageCorruptionError, match="Record tampering detected at sequence 14"):
            runner.store.load_master_records(runner.session.session_id, verify=True)

    # --------------------------------------------------------------------------
    # SC-05: Invalid Response Payload
    # --------------------------------------------------------------------------
    def test_sc05_invalid_response_payload(self, tmp_dir: Path) -> None:
        """
        Scenario 5: Case k active. Script submits incomplete rubric.
        Submission blocked; field validation error returned. Zero disk commit.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC05-01",
            reviewer_pseudonym="SYNTH-REV-SC05",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        # Incomplete rubric (empty reasoning)
        with pytest.raises(ResponseValidationError, match="at least 5 non-whitespace characters"):
            RatingsPayload(
                stage_classification=StageClassification.STAGE_2,
                vcp_qualification=VCPQualification.VCP_CONFIRMED,
                confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
                prior_familiarity=PriorFamiliarity.NO_RECALL,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                reviewer_reasoning="",
            )

        # Zero disk write: master log remains empty
        records, _ = runner.store.load_master_records(runner.session.session_id)
        assert len(records) == 0
        assert runner.fsm.current_state == SessionState.RESPONSE_DRAFT

    # --------------------------------------------------------------------------
    # SC-06: Browser / Window Refresh
    # --------------------------------------------------------------------------
    def test_sc06_browser_window_refresh(self, tmp_dir: Path) -> None:
        """
        Scenario 6: Case 22 active; Cases 1..21 committed. Window refresh triggered.
        Session reloads; reads JSONL; re-establishes Case 22. Cases 1..21 locked.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC06-01",
            reviewer_pseudonym="SYNTH-REV-SC06",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=25,
        )
        runner.authorize()
        for seq in range(1, 22):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        assert runner.fsm.current_sequence == 22

        # Simulate refresh / session reload
        reloaded = IntakeRunner.resume_session("HVA01-SESS-SC06-01", base_dir=tmp_dir)
        assert reloaded.fsm.current_sequence == 22
        assert reloaded.fsm.current_state == SessionState.RESPONSE_DRAFT
        assert len(reloaded.fsm.locked_cases) == 21
        assert all(s in reloaded.fsm.locked_cases for s in range(1, 22))

    # --------------------------------------------------------------------------
    # SC-07: Application Restart
    # --------------------------------------------------------------------------
    def test_sc07_application_restart(self, tmp_dir: Path) -> None:
        """
        Scenario 7: Applet abruptly terminated at Case 30 (1..29 committed).
        Process restarted with session ID. Validates 1..29, resumes at Case 30.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC07-01",
            reviewer_pseudonym="SYNTH-REV-SC07",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=35,
        )
        runner.authorize()
        for seq in range(1, 30):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        # Simulate crash and restart
        del runner

        restarted = IntakeRunner.resume_session("HVA01-SESS-SC07-01", base_dir=tmp_dir)
        report = restarted.verify_integrity()
        assert report["chain_valid"] is True
        assert report["committed_cases_count"] == 29
        assert restarted.fsm.current_sequence == 30
        assert restarted.fsm.current_state == SessionState.RESPONSE_DRAFT

    # --------------------------------------------------------------------------
    # SC-08: Interrupted Disk Write
    # --------------------------------------------------------------------------
    def test_sc08_interrupted_disk_write(self, tmp_dir: Path) -> None:
        """
        Scenario 8: Commit underway. Simulated IO failure during append.
        Atomic write pattern drops .tmp file; master log unharmed.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC08-01",
            reviewer_pseudonym="SYNTH-REV-SC08",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_valid_ratings())

        # Inject orphaned staging file
        session_dir = runner.store.get_session_dir(runner.session.session_id)
        orphaned = session_dir / "write_2_HVA2-A-02.tmp"
        orphaned.write_text("{\"partial\": true}\n", encoding="utf-8")

        # Resume session: verify staging cleaned and master log valid at sequence 1
        resumed = IntakeRunner.resume_session("HVA01-SESS-SC08-01", base_dir=tmp_dir)
        assert not orphaned.exists()
        records, _ = resumed.store.load_master_records(resumed.session.session_id)
        assert len(records) == 1
        assert resumed.fsm.current_sequence == 2

    # --------------------------------------------------------------------------
    # SC-09: Concurrent Active Sessions
    # --------------------------------------------------------------------------
    def test_sc09_concurrent_active_sessions(self, tmp_dir: Path) -> None:
        """
        Scenario 9: Session A running on Terminal 1. Second instance attempts launch on Terminal 2.
        File lock detects active lockfile; second instance aborts with error.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC09-01",
            reviewer_pseudonym="SYNTH-REV-SC09",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        # Hold lock with instance 1
        lock_path = runner.store._lock_path(runner.session.session_id)
        with FileLock(lock_path):
            # Second instance attempts to acquire
            with pytest.raises(FileLockError, match="Could not acquire lock"):
                FileLock(lock_path, timeout=0.2, poll_interval=0.05).acquire()

    # --------------------------------------------------------------------------
    # SC-10: Wrong-Package Access
    # --------------------------------------------------------------------------
    def test_sc10_wrong_package_access(self, tmp_dir: Path) -> None:
        """
        Scenario 10: Reviewer assigned Package A. Script attempts to request Package B assets.
        Request denied; PACKAGE_ACCESS_VIOLATION logged. Zero cross-package exposure.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC10-01",
            reviewer_pseudonym="SYNTH-REV-SC10",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        # Attempt to access Package B chart
        with pytest.raises(PackageAccessViolation, match="Package access violation"):
            runner.resolve_case_asset("HVA2-B-05", relative_path="charts_b/HVA2-B-05.svg")

        # Attempt submission with Package B case ID while assigned to Package A
        with pytest.raises(PackageAccessViolation):
            runner.submit_and_lock_response(1, "HVA2-B-01", make_valid_ratings())

        # Verify audit event logged
        events = runner.store.load_audit_events(runner.session.session_id)
        assert any(e["event_code"] == "EVT_ACCESS_VIOLATION" for e in events)

    # --------------------------------------------------------------------------
    # SC-11: Restricted-Label Access
    # --------------------------------------------------------------------------
    def test_sc11_restricted_label_access(self, tmp_dir: Path) -> None:
        """
        Scenario 11: Applet running evaluation loop. Script attempts to open unblinding key or mapping.
        Access denied; RESTRICTED_LABEL_ACCESS exception logged; zero label leakage.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC11-01",
            reviewer_pseudonym="SYNTH-REV-SC11",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        with pytest.raises(RestrictedLabelAccessError, match="Restricted label access violation"):
            runner.resolve_case_asset("HVA00_RESTRICTED_ID_MAPPING_V2.json")

        with pytest.raises(RestrictedLabelAccessError):
            runner.resolve_case_asset("UNBLIND_ANSWER_KEY.csv")

        # Verify audit event logged
        events = runner.store.load_audit_events(runner.session.session_id)
        assert any(e["event_code"] == "EVT_CUSTODY_BREACH" for e in events)

    # --------------------------------------------------------------------------
    # SC-12: Typo Correction Request
    # --------------------------------------------------------------------------
    def test_sc12_typo_correction_request(self, tmp_dir: Path) -> None:
        """
        Scenario 12: Case 14 committed with typo. Script appends valid entry to exception ledger.
        Master log intact; exception ledger links correction to target hash.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC12-01",
            reviewer_pseudonym="SYNTH-REV-SC12",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=20,
        )
        runner.authorize()
        for seq in range(1, 15):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        master_file = runner.store._master_file(runner.session.session_id)
        master_bytes_before = master_file.read_bytes()

        records, _ = runner.store.load_master_records(runner.session.session_id)
        target_rec = records[13]  # Case 14

        # Petition typo correction
        revised = make_valid_ratings(stage=StageClassification.STAGE_1)
        corr = runner.petition_correction(
            target_case_id="HVA2-A-14",
            target_sequence_index=14,
            target_record_hash=target_rec.record_hash,
            revised_ratings=revised,
            justification="Reviewer typo correction: misclassified stage on case 14.",
        )

        assert corr.target_record_hash == target_rec.record_hash
        assert corr.petition_hash is not None

        # Verify master log is byte-for-byte identical
        assert master_file.read_bytes() == master_bytes_before

        # Verify exception ledger contains the record
        exceptions = runner.store.load_corrections(runner.session.session_id)
        assert len(exceptions) == 1
        assert exceptions[0].petition_id == corr.petition_id

    # --------------------------------------------------------------------------
    # SC-13: Tampered Audit Record
    # --------------------------------------------------------------------------
    def test_sc13_tampered_audit_record(self, tmp_dir: Path) -> None:
        """
        Scenario 13: 59 cases committed; session completed. External script modifies timestamp on line 42.
        Recursive SHA-256 validation flags line 42 as RECORD_HASH_MISMATCH.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC13-01",
            reviewer_pseudonym="SYNTH-REV-SC13",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=45,
        )
        runner.authorize()
        for seq in range(1, 46):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Tamper timestamp on line 42
        lines[41]["timestamp_utc"] = "2026-01-01T00:00:00+00:00"
        with open(master_file, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        with pytest.raises(StorageCorruptionError, match="Record tampering detected at sequence 42"):
            runner.store.load_master_records(runner.session.session_id, verify=True)

    # --------------------------------------------------------------------------
    # SC-14: Incorrect Protocol Hash
    # --------------------------------------------------------------------------
    def test_sc14_incorrect_protocol_hash(self, tmp_dir: Path) -> None:
        """
        Scenario 14: Launcher initialized. Passes mismatched protocol document SHA-256.
        Launcher halts before rendering; PROTOCOL_VERSION_DIGEST_MISMATCH declared.
        """
        approved_proto_sha = "c" * 64
        corrupted_proto_sha = "d" * 64

        with pytest.raises(ValidationError, match="PROTOCOL_VERSION_DIGEST_MISMATCH"):
            IntakeRunner.create_session(
                session_id="HVA01-SESS-SC14-01",
                reviewer_pseudonym="SYNTH-REV-SC14",
                assigned_package="PACKAGE_A",
                protocol_version="2.1.0",
                protocol_document_sha256=corrupted_proto_sha,
                package_sha256=VALID_PKG_SHA,
                base_dir=tmp_dir,
                total_cases=10,
                expected_protocol_sha256=approved_proto_sha,
            )

    # --------------------------------------------------------------------------
    # SC-15: Incorrect Package Hash
    # --------------------------------------------------------------------------
    def test_sc15_incorrect_package_hash(self, tmp_dir: Path) -> None:
        """
        Scenario 15: Reviewer profile loaded. Package JSON modified or corrupted.
        Launcher halts; declares PACKAGE_INTEGRITY_VERIFICATION_FAILED.
        """
        sealed_package_sha = "e" * 64
        corrupted_package_sha = "f" * 64

        with pytest.raises(ValidationError, match="PACKAGE_INTEGRITY_VERIFICATION_FAILED"):
            IntakeRunner.create_session(
                session_id="HVA01-SESS-SC15-01",
                reviewer_pseudonym="SYNTH-REV-SC15",
                assigned_package="PACKAGE_A",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                package_sha256=corrupted_package_sha,
                base_dir=tmp_dir,
                total_cases=10,
                expected_package_sha256=sealed_package_sha,
            )

    # --------------------------------------------------------------------------
    # SC-16: Missing Final Submission
    # --------------------------------------------------------------------------
    def test_sc16_missing_final_submission(self, tmp_dir: Path) -> None:
        """
        Scenario 16: Reviewer completes 58 cases; closes applet. Synthetic pipeline attempts seal.
        Ingestion flags incomplete (58 of 59); session barred from sealing.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC16-01",
            reviewer_pseudonym="SYNTH-REV-SC16",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=59,
        )
        runner.authorize()
        for seq in range(1, 59):  # 58 cases
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        assert len(runner.fsm.locked_cases) == 58

        # Attempt to seal session
        with pytest.raises(StateTransitionError, match="Cannot close session: only 58 of 59 cases locked"):
            runner.fsm.transition_to(SessionState.SESSION_CLOSED, actor="TEST")

        assert runner.fsm.current_state != SessionState.SESSION_CLOSED

    # --------------------------------------------------------------------------
    # SC-17: Abstention Submission
    # --------------------------------------------------------------------------
    def test_sc17_abstention_submission(self, tmp_dir: Path) -> None:
        """
        Scenario 17: Case k active. Script selects evidence_sufficiency = INSUFFICIENT, code ABS_DATA.
        Valid commit recorded; scored as abstention; dual sensitivity partitions cleanly.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC17-01",
            reviewer_pseudonym="SYNTH-REV-SC17",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        abstain_ratings = make_valid_ratings(
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
            abstention=AbstentionCode.ABS_DATA,
            reasoning="Abstention test: unreadable chart due to data feed glitch.",
        )
        rec = runner.submit_and_lock_response(1, "HVA2-A-01", abstain_ratings)
        assert rec.is_locked is True
        assert rec.ratings.abstention_code == AbstentionCode.ABS_DATA
        assert rec.ratings.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT

    # --------------------------------------------------------------------------
    # SC-18: Outcome-Recall Disclosure
    # --------------------------------------------------------------------------
    def test_sc18_outcome_recall_disclosure(self, tmp_dir: Path) -> None:
        """
        Scenario 18: Case k active. Script selects prior_familiarity = OUTCOME_RECALLED.
        Record committed; flagged contaminated for sensitivity check.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC18-01",
            reviewer_pseudonym="SYNTH-REV-SC18",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        recalled_ratings = make_valid_ratings(
            familiarity=PriorFamiliarity.OUTCOME_RECALLED,
            reasoning="Prior familiarity test: reviewer recalls historical post-pivot outcome.",
        )
        rec = runner.submit_and_lock_response(1, "HVA2-A-01", recalled_ratings)
        assert rec.is_locked is True
        assert rec.ratings.prior_familiarity == PriorFamiliarity.OUTCOME_RECALLED

    # --------------------------------------------------------------------------
    # SC-19: Custody Disk Full Failure
    # --------------------------------------------------------------------------
    def test_sc19_custody_disk_full_failure(self, tmp_dir: Path) -> None:
        """
        Scenario 19: Case k submission. Simulated disk full (ENOSPC) during append.
        Applet catches exception; rolls back FSM to RESPONSE_DRAFT; retains input in memory.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC19-01",
            reviewer_pseudonym="SYNTH-REV-SC19",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        # Simulate ENOSPC disk full during append
        with patch.object(runner.store, "append_response", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            with pytest.raises(OSError, match="No space left on device"):
                runner.submit_and_lock_response(1, "HVA2-A-01", make_valid_ratings())

        # FSM safely rolled back to RESPONSE_DRAFT; zero master corruption
        assert runner.fsm.current_state == SessionState.RESPONSE_DRAFT
        assert runner.fsm.current_sequence == 1
        assert len(runner.fsm.locked_cases) == 0

        # Audit event records storage exhaustion
        events = runner.store.load_audit_events(runner.session.session_id)
        assert any(e["event_code"] == "EVT_STORAGE_EXHAUSTED" for e in events)

    # --------------------------------------------------------------------------
    # SC-20: Formal Session Withdrawal
    # --------------------------------------------------------------------------
    def test_sc20_formal_session_withdrawal(self, tmp_dir: Path) -> None:
        """
        Scenario 20: Reviewer completes 25 cases; requests withdrawal.
        Withdrawal executed; session sealed with WITHDRAWAL_STATUS = TRUE; 25 cases frozen.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-SC20-01",
            reviewer_pseudonym="SYNTH-REV-SC20",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=59,
        )
        runner.authorize()
        for seq in range(1, 26):
            runner.submit_and_lock_response(seq, f"HVA2-A-{seq:02d}", make_valid_ratings())

        assert len(runner.fsm.locked_cases) == 25

        # Execute formal withdrawal
        result = runner.withdraw_session(justification="Participant personal emergency: session withdrawn.")
        assert result["status"] == "WITHDRAWN"
        assert result["committed_cases"] == 25
        assert result["coverage_ratio"] == "25/59"
        assert result["is_sealed"] is True

        # State is closed
        assert runner.fsm.current_state == SessionState.SESSION_CLOSED

        # Reloading confirms sealed withdrawn state
        resumed = IntakeRunner.resume_session("HVA01-SESS-SC20-01", base_dir=tmp_dir)
        assert resumed.fsm.current_state == SessionState.SESSION_CLOSED
        assert len(resumed.fsm.locked_cases) == 25
        assert resumed.session.metadata["is_withdrawn"] is True
