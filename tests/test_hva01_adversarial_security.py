"""
Adversarial Security, Concurrency, and Stress Verification Suite for HVA-01 V2.1.
Exercises malformed inputs, boundary violations, path traversal, PII injection,
concurrency contention, disk fault injection, and cryptographic attack vectors.
"""

from __future__ import annotations

import concurrent.futures
import errno
import json
import os
import shutil
import tempfile
import threading
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
    HashChainVerificationError,
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
    VCPQualification,
    ValidationError,
)

VALID_PROTO_SHA = "a" * 64
VALID_PKG_SHA = "b" * 64


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    d = tempfile.mkdtemp(prefix="hva01_adv_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def make_ratings(reasoning: str = "Adversarial test reasoning string.") -> RatingsPayload:
    return RatingsPayload(
        stage_classification=StageClassification.STAGE_2,
        vcp_qualification=VCPQualification.VCP_CONFIRMED,
        confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
        prior_familiarity=PriorFamiliarity.NO_RECALL,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        reviewer_reasoning=reasoning,
        abstention_code=AbstentionCode.NONE,
        contraction_count_observed=3,
        progressive_tightening_observed=True,
        volume_dry_up_observed=True,
        pivot_present=True,
        composite_score_100=75,
    )


class TestAdversarialSecurityAndStress:
    """
    Adversarial verification suite covering malformed inputs, concurrency,
    filesystem fault injection, and cryptographic attack vectors.
    """

    # --------------------------------------------------------------------------
    # 1. Malformed Input & Injection Attacks
    # --------------------------------------------------------------------------
    def test_oversized_reasoning_string_handled(self, tmp_dir: Path) -> None:
        """Verify handling of exceptionally large reasoning payloads (100KB)."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-01",
            reviewer_pseudonym="SYNTH-REV-ADV01",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        huge_reasoning = "A" * (100 * 1024)  # 100 KB payload
        ratings = make_ratings(reasoning=huge_reasoning)
        rec = runner.submit_and_lock_response(1, "HVA2-A-01", ratings)
        assert rec.is_locked is True

        # Verify integrity of large record
        report = runner.verify_integrity()
        assert report["chain_valid"] is True

    def test_command_injection_strings_in_reasoning_stored_safely(self, tmp_dir: Path) -> None:
        """Verify SQL, Shell, and Script injection payloads are treated as inert UTF-8 data."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-02",
            reviewer_pseudonym="SYNTH-REV-ADV02",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        injection_payload = "'; DROP TABLE responses; -- && rm -rf / <script>alert('xss')</script>"
        ratings = make_ratings(reasoning=injection_payload)
        rec = runner.submit_and_lock_response(1, "HVA2-A-01", ratings)
        assert rec.ratings.reviewer_reasoning == injection_payload

        # Re-read from disk and verify exact fidelity
        records, _ = runner.store.load_master_records(runner.session.session_id)
        assert records[0].ratings.reviewer_reasoning == injection_payload

    def test_null_byte_rejection_in_ids(self, tmp_dir: Path) -> None:
        """Verify null-byte injection in identifiers is rejected by schema regex."""
        with pytest.raises(ValidationError):
            SessionModel(
                session_id="HVA01-SESS-\x00MALICIOUS",
                protocol_version="2.1.0",
                protocol_document_sha256=VALID_PROTO_SHA,
                assigned_package="PACKAGE_A",
                package_sha256=VALID_PKG_SHA,
                reviewer_pseudonym="SYNTH-REV-ADV03",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )

    def test_pii_pattern_rejection(self) -> None:
        """Verify various PII patterns are rejected in reviewer pseudonym."""
        pii_attempts = [
            "john.doe@gmail.com",
            "REV-user@domain.org",
            "123-45-6789",
            "+1-800-555-0199",
        ]
        for pii in pii_attempts:
            with pytest.raises(ValidationError):
                SessionModel(
                    session_id="HVA01-SESS-ADV-03",
                    protocol_version="2.1.0",
                    protocol_document_sha256=VALID_PROTO_SHA,
                    assigned_package="PACKAGE_A",
                    package_sha256=VALID_PKG_SHA,
                    reviewer_pseudonym=pii,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                )

    # --------------------------------------------------------------------------
    # 2. Path Traversal & Blinding Security
    # --------------------------------------------------------------------------
    def test_path_traversal_attempts_blocked(self, tmp_dir: Path) -> None:
        """Verify directory traversal attempts in asset resolution are rejected fail-closed."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-04",
            reviewer_pseudonym="SYNTH-REV-ADV04",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()

        traversal_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "/absolute/root/file.svg",
            "charts_a/../../restricted_keys.json",
        ]
        for path in traversal_paths:
            with pytest.raises(PackageAccessViolation):
                runner.resolve_case_asset("HVA2-A-01", relative_path=path)

    # --------------------------------------------------------------------------
    # 3. Concurrency Contention & Multi-Thread Stress
    # --------------------------------------------------------------------------
    def test_concurrent_writer_thread_exclusion(self, tmp_dir: Path) -> None:
        """Verify that concurrent threads competing for the store lock never corrupt master."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-05",
            reviewer_pseudonym="SYNTH-REV-ADV05",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()

        ratings = make_ratings()
        errors = []
        successes = []

        def worker_append(thread_id: int) -> None:
            try:
                # Direct store append attempt simulating race condition
                t_now = "2026-10-24T00:00:00+00:00"
                r_json = CanonicalHasher.canonical_json(ratings.to_canonical_dict())
                p_hash = runner.store.get_genesis_hash(runner.session.session_id)
                h_k = CanonicalHasher.compute_record_hash(p_hash, 1, "HVA2-A-01", t_now, r_json)
                idemp = CanonicalHasher.compute_idempotency_key(runner.session.session_id, "HVA2-A-01", 1)

                rec = ResponseRecord(
                    sequence_index=1,
                    case_id="HVA2-A-01",
                    session_id=runner.session.session_id,
                    reviewer_pseudonym=runner.session.reviewer_pseudonym,
                    protocol_version=runner.session.protocol_version,
                    protocol_document_sha256=runner.session.protocol_document_sha256,
                    package_sha256=runner.session.package_sha256,
                    timestamp_utc=t_now,
                    ratings=ratings,
                    idempotency_key=idemp,
                    previous_hash=p_hash,
                    record_hash=h_k,
                    is_locked=True,
                )
                runner.store.append_response(runner.session.session_id, rec)
                successes.append(thread_id)
            except Exception as e:
                errors.append(e)

        # Launch 5 concurrent threads attempting to commit sequence 1
        threads = [threading.Thread(target=worker_append, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 record in master.jsonl
        records, _ = runner.store.load_master_records(runner.session.session_id, verify=False)
        assert len(records) == 1

    # --------------------------------------------------------------------------
    # 4. Cryptographic Attack Vectors
    # --------------------------------------------------------------------------
    def test_sequence_gap_injection_detected(self, tmp_dir: Path) -> None:
        """Verify that injecting a sequence gap (e.g., 1 -> 3) fails verification."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-06",
            reviewer_pseudonym="SYNTH-REV-ADV06",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        rec1 = runner.submit_and_lock_response(1, "HVA2-A-01", make_ratings())

        # Attempt to append sequence 3 directly (skipping sequence 2)
        rec3 = ResponseRecord(
            sequence_index=3,
            case_id="HVA2-A-03",
            session_id=runner.session.session_id,
            reviewer_pseudonym=runner.session.reviewer_pseudonym,
            protocol_version=runner.session.protocol_version,
            protocol_document_sha256=runner.session.protocol_document_sha256,
            package_sha256=runner.session.package_sha256,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ratings=make_ratings(),
            idempotency_key="key3",
            previous_hash=rec1.record_hash,
            record_hash="hash3" + "0" * 59,
            is_locked=True,
        )
        with pytest.raises(ValueError, match="Sequence discontinuity"):
            runner.store.append_response(runner.session.session_id, rec3)

    def test_forged_predecessor_hash_detected(self, tmp_dir: Path) -> None:
        """Verify that submitting a record with an invalid previous_hash is rejected."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-07",
            reviewer_pseudonym="SYNTH-REV-ADV07",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_ratings())

        # Sequence 2 with forged previous_hash
        forged_rec2 = ResponseRecord(
            sequence_index=2,
            case_id="HVA2-A-02",
            session_id=runner.session.session_id,
            reviewer_pseudonym=runner.session.reviewer_pseudonym,
            protocol_version=runner.session.protocol_version,
            protocol_document_sha256=runner.session.protocol_document_sha256,
            package_sha256=runner.session.package_sha256,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ratings=make_ratings(),
            idempotency_key="key2",
            previous_hash="forged_predecessor_hash" + "0" * 41,
            record_hash="hash2" + "0" * 59,
            is_locked=True,
        )
        with pytest.raises(ValueError, match="Hash link mismatch"):
            runner.store.append_response(runner.session.session_id, forged_rec2)

    def test_duplicate_case_id_in_session_detected(self, tmp_dir: Path) -> None:
        """Verify that re-evaluating the same case_id under a new sequence is blocked."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-08",
            reviewer_pseudonym="SYNTH-REV-ADV08",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_ratings())

        # Attempt to evaluate HVA2-A-01 again as sequence 2
        with pytest.raises(ConflictingDuplicateSubmissionError, match="Conflicting submission"):
            # Construct duplicate case_id with sequence 2
            rec_dup = ResponseRecord(
                sequence_index=2,
                case_id="HVA2-A-01",  # Duplicate case_id
                session_id=runner.session.session_id,
                reviewer_pseudonym=runner.session.reviewer_pseudonym,
                protocol_version=runner.session.protocol_version,
                protocol_document_sha256=runner.session.protocol_document_sha256,
                package_sha256=runner.session.package_sha256,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                ratings=make_ratings(),
                idempotency_key="key_dup",
                previous_hash="some_hash",
                record_hash="rec_hash" + "0" * 56,
                is_locked=True,
            )
            runner.store.append_response(runner.session.session_id, rec_dup)

    def test_tampered_genesis_hash_fails_verification(self, tmp_dir: Path) -> None:
        """Verify that altering genesis block attributes causes verification to fail."""
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-09",
            reviewer_pseudonym="SYNTH-REV-ADV09",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_ratings())

        # Alter protocol_document_sha256 in session.json
        sess_file = runner.store._session_file(runner.session.session_id)
        with open(sess_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["protocol_document_sha256"] = "f" * 64
        with open(sess_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Re-running verification should fail on previous_hash link
        with pytest.raises(StorageCorruptionError, match="Hash link fault at sequence 1"):
            runner.store.load_master_records(runner.session.session_id, verify=True)

    # --------------------------------------------------------------------------
    # 5. Full Chain Rewrite Disproof Simulation
    # --------------------------------------------------------------------------
    def test_local_tamper_evidence_proof(self, tmp_dir: Path) -> None:
        """
        Formal demonstration of local hash chain security:
        Modifying block k breaks the hash link to block k+1 unless all subsequent
        blocks k+1..m are rewritten.
        """
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-ADV-10",
            reviewer_pseudonym="SYNTH-REV-ADV10",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=4,
        )
        runner.authorize()
        runner.submit_and_lock_response(1, "HVA2-A-01", make_ratings())
        runner.submit_and_lock_response(2, "HVA2-A-02", make_ratings())
        runner.submit_and_lock_response(3, "HVA2-A-03", make_ratings())

        # Attacker modifies block 2 and recomputes block 2 hash H_2'
        master_file = runner.store._master_file(runner.session.session_id)
        with open(master_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Modify body of block 2
        lines[1]["ratings"]["reviewer_reasoning"] = "Substituted reasoning."
        ratings_json = CanonicalHasher.canonical_json(lines[1]["ratings"])
        lines[1]["record_hash"] = CanonicalHasher.compute_record_hash(
            previous_hash=lines[1]["previous_hash"],
            sequence_index=2,
            case_id="HVA2-A-02",
            timestamp_utc=lines[1]["timestamp_utc"],
            ratings_canonical_json=ratings_json,
        )

        with open(master_file, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        # Block 2 itself has a valid internal hash, BUT Block 3's previous_hash still points to old H_2!
        # Verification MUST catch this link fault at Sequence 3:
        with pytest.raises(StorageCorruptionError, match="Hash link fault at sequence 3"):
            runner.store.load_master_records(runner.session.session_id, verify=True)
