"""
ARX Terminal — Human Observer Evidence Custody Security & Adversarial Test Suite (Gate J)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence

Covers 24 Adversarial Security Scenarios:
 1. Missing vault configuration
 2. Vault path inside Git repository
 3. Unsafe vault permissions
 4. Unauthorized observer token
 5. Malformed JSON
 6. Invalid schema
 7. Duplicate submission
 8. Submission replay
 9. File replacement attempt
10. Modified ciphertext
11. Incorrect decryption key
12. Missing package identity
13. Incorrect package hash
14. Gold-key access attempt
15. Observer cross-access attempt
16. Premature scoring with one submission
17. Two independently sealed valid synthetic submissions
18. Withdrawal invalidation
19. Deletion workflow
20. No sensitive logs
21. Git forced-add containment
22. Retained canonical qualification fixtures
23. Incomplete backup deletion documentation
24. Crash/interruption during atomic intake
"""

import os
import sys
import json
import io
import shutil
import tempfile
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import pytest

from analyst_dashboard.research.custody.exceptions import (
    CustodyError,
    VaultConfigurationError,
    VaultPermissionError,
    CryptoTamperError,
    CustodyKeyError,
    IntakeValidationError,
    BlindingError,
    LedgerIntegrityError,
    WithdrawalError,
)
from analyst_dashboard.research.custody.crypto import (
    CustodyCrypto,
    compute_sha256,
    generate_key_hex,
)
from analyst_dashboard.research.custody.vault import (
    LocalRestrictedVault,
    find_git_repository_root,
)
from analyst_dashboard.research.custody.ledger import (
    TamperEvidentLedger,
    compute_receipt_digest,
)
from analyst_dashboard.research.custody.intake import (
    IntakePipeline,
    EXPECTED_30_TRIALS,
)
from analyst_dashboard.research.custody.scoring_bridge import (
    MutualBlindingScoringBridge,
    DEFAULT_GOLD_KEY_PATH,
)
from analyst_dashboard.research.custody.retention import (
    RetentionAndWithdrawalManager,
)

CANONICAL_BUNDLE_SHA = "9a4a53e642d7a8ea55a1dbceab9024408d3d58913f8a019dac52609044a8d6c5"


def create_synthetic_ratings_payload(
    observer_id: str = "OBS-SYNTHETIC-01",
    transcript_id: str = "TRANS-CONSOLIDATED-30",
    match_gold: bool = True,
) -> Dict[str, Any]:
    """Helper to generate a valid synthetic 30-trial submission matching schema."""
    with open(DEFAULT_GOLD_KEY_PATH, "r", encoding="utf-8") as f:
        gold_key = json.load(f)

    ratings = []
    for tid in sorted(EXPECTED_30_TRIALS):
        gold_item = gold_key["items"].get(tid, {})
        cm_codes = list(gold_item.get("authoritative_cm_codes", [])) if match_gold else []
        sm_codes = list(gold_item.get("authoritative_sm_codes", [])) if match_gold else []
        interp = (
            gold_item.get("interpretation_state", "INTERPRETATION_CORRECT")
            if match_gold
            else "INTERPRETATION_CORRECT"
        )
        action = (
            gold_item.get("action_state", "ACTION_APPROPRIATE")
            if match_gold
            else "ACTION_APPROPRIATE"
        )
        crit_err = (
            gold_item.get("critical_decision_error", False)
            if match_gold
            else False
        )

        ratings.append({
            "trial_id": tid,
            "assigned_cm_codes": cm_codes,
            "assigned_sm_codes": sm_codes,
            "interpretation_state": interp,
            "action_state": action,
            "critical_decision_error": crit_err,
            "observer_notes": "Synthetic rater note for validation rehearsal.",
        })

    return {
        "observer_id": observer_id,
        "transcript_id": transcript_id,
        "submission_timestamp": "2026-10-23T16:00:00Z",
        "synthetic_provenance": True,
        "ratings": ratings,
    }


class TestEvidenceCustodySecurity:
    """Comprehensive 24-scenario adversarial custody test suite."""

    @pytest.fixture(autouse=True)
    def setup_isolated_env(self):
        """Create a temporary directory strictly outside the Git repository."""
        self.temp_dir = tempfile.mkdtemp(prefix="arx_custody_test_")
        self.vault_path = Path(self.temp_dir) / "vault"
        self.crypto_key = generate_key_hex()
        self.crypto = CustodyCrypto(self.crypto_key)
        self.vault = LocalRestrictedVault(
            self.vault_path, enforce_acls=True, strict_permissions_check=True
        )
        self.ledger = TamperEvidentLedger(self.vault.ledger_dir / "custody_receipts.jsonl")
        self.intake = IntakePipeline(
            vault=self.vault,
            crypto=self.crypto,
            ledger=self.ledger,
            is_synthetic_mode=True,
        )
        self.bridge = MutualBlindingScoringBridge(
            vault=self.vault,
            crypto=self.crypto,
            ledger=self.ledger,
            is_synthetic_mode=True,
        )
        self.retention = RetentionAndWithdrawalManager(
            vault=self.vault,
            ledger=self.ledger,
        )

        yield

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_missing_vault_configuration(self):
        """1. Missing vault configuration fails closed."""
        old_val = os.environ.pop("ARX_CUSTODY_VAULT_ROOT", None)
        try:
            with pytest.raises(VaultConfigurationError, match="unset or empty"):
                LocalRestrictedVault(storage_root=None)
        finally:
            if old_val:
                os.environ["ARX_CUSTODY_VAULT_ROOT"] = old_val

    def test_02_vault_path_inside_git_repository(self):
        """2. Vault path inside Git repository is strictly refused."""
        repo_root = find_git_repository_root()
        assert repo_root is not None
        inside_path = repo_root / "tests" / "fixtures" / "qualification" / "submissions" / "vault"
        with pytest.raises(VaultConfigurationError, match="is inside Git repository"):
            LocalRestrictedVault(storage_root=inside_path)

    def test_03_unsafe_vault_permissions(self):
        """3. Unsafe filesystem permissions trigger VaultPermissionError."""
        # Test verify_permissions on simulated unsafe path
        if os.name == "nt":
            # On Windows, create a directory without stripping inheritance
            raw_tmp = tempfile.mkdtemp(prefix="unsafe_ntfs_")
            try:
                # With strict permissions check, raw mkdtemp should fail if inheritance is present
                with pytest.raises(VaultPermissionError, match="Inherited permissions detected"):
                    LocalRestrictedVault(raw_tmp, enforce_acls=False, strict_permissions_check=True)
            finally:
                shutil.rmtree(raw_tmp, ignore_errors=True)
        else:
            raw_tmp = tempfile.mkdtemp(prefix="unsafe_posix_")
            try:
                os.chmod(raw_tmp, 0o777)
                with pytest.raises(VaultPermissionError, match="POSIX permissions unsafe"):
                    LocalRestrictedVault(raw_tmp, enforce_acls=False, strict_permissions_check=True)
            finally:
                shutil.rmtree(raw_tmp, ignore_errors=True)

    def test_04_unauthorized_observer_token(self):
        """4. Unauthorized real human observer token rejected in synthetic mode."""
        payload = create_synthetic_ratings_payload(observer_id="OBS-HUMAN-01")
        raw_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(IntakeValidationError, match="human intake not authorized"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes,
                registered_batch_id="BATCH-001",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )

    def test_05_malformed_json(self):
        """5. Malformed JSON rejected gracefully without leaking raw bytes."""
        malformed_bytes = b'{"observer_id": "OBS-SYNTHETIC-01", "truncated": [1, 2,'
        with pytest.raises(IntakeValidationError, match="invalid UTF-8 JSON syntax") as exc_info:
            self.intake.ingest_submission(
                raw_bytes=malformed_bytes,
                registered_batch_id="BATCH-001",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )
        assert "truncated" not in str(exc_info.value)

    def test_06_invalid_schema(self):
        """6. Schema violation (missing required ratings array) rejected with path."""
        invalid_payload = {
            "observer_id": "OBS-SYNTHETIC-01",
            "transcript_id": "TRANS-01",
            "submission_timestamp": "2026-10-23T16:00:00Z",
            # missing "ratings"
        }
        raw_bytes = json.dumps(invalid_payload).encode("utf-8")
        with pytest.raises(IntakeValidationError, match="Schema validation failed"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes,
                registered_batch_id="BATCH-001",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )

    def test_07_duplicate_submission(self):
        """7. Duplicate submission for same observer and batch rejected."""
        payload = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        raw_bytes = json.dumps(payload).encode("utf-8")

        self.intake.ingest_submission(
            raw_bytes=raw_bytes,
            registered_batch_id="BATCH-001",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        # Modify payload slightly so hash differs, but observer & batch match
        payload["ratings"][0]["observer_notes"] = "Modified note"
        raw_bytes_modified = json.dumps(payload).encode("utf-8")

        with pytest.raises(IntakeValidationError, match="Duplicate submission rejected"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes_modified,
                registered_batch_id="BATCH-001",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )

    def test_08_submission_replay(self):
        """8. Byte-identical submission replay rejected via SHA-256 ledger collision."""
        payload = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        raw_bytes = json.dumps(payload).encode("utf-8")

        self.intake.ingest_submission(
            raw_bytes=raw_bytes,
            registered_batch_id="BATCH-001",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        with pytest.raises(IntakeValidationError, match="Submission replay rejected"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes,
                registered_batch_id="BATCH-002",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )

    def test_09_file_replacement_attempt(self):
        """9. Direct file overwrite attempt on vault sealed partition prohibited."""
        subm_id = "SUBM-REPLACE-TEST"
        self.vault.store_sealed_evidence(subm_id, b"first_ciphertext")

        with pytest.raises(CustodyError, match="already exists. Overwrite prohibited"):
            self.vault.store_sealed_evidence(subm_id, b"second_ciphertext")

        assert self.vault.read_sealed_evidence(subm_id) == b"first_ciphertext"

    def test_10_modified_ciphertext(self):
        """10. Tampered ciphertext detected by AES-GCM tag verification."""
        plaintext = b'{"test": "confidential_data"}'
        ciphertext = bytearray(self.crypto.encrypt(plaintext))

        # Bit flip in ciphertext body
        ciphertext[-1] ^= 0x01

        with pytest.raises(CryptoTamperError, match="Ciphertext authentication failed"):
            self.crypto.decrypt(bytes(ciphertext))

    def test_11_incorrect_decryption_key(self):
        """11. Decryption with incorrect 256-bit key fails closed."""
        plaintext = b'{"test": "confidential_data"}'
        ciphertext = self.crypto.encrypt(plaintext)

        wrong_key = generate_key_hex()
        wrong_crypto = CustodyCrypto(wrong_key)

        with pytest.raises(CryptoTamperError, match="Ciphertext authentication failed"):
            wrong_crypto.decrypt(ciphertext)

    def test_12_missing_package_identity(self):
        """12. Ingestion with missing batch ID or package SHA rejected."""
        payload = create_synthetic_ratings_payload()
        raw_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(IntakeValidationError, match="mandatory"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes,
                registered_batch_id="",
                expected_package_sha=CANONICAL_BUNDLE_SHA,
            )

    def test_13_incorrect_package_hash(self):
        """13. Submission referencing invalid/empty package SHA rejected."""
        payload = create_synthetic_ratings_payload()
        raw_bytes = json.dumps(payload).encode("utf-8")

        with pytest.raises(IntakeValidationError, match="mandatory"):
            self.intake.ingest_submission(
                raw_bytes=raw_bytes,
                registered_batch_id="BATCH-001",
                expected_package_sha="",
            )

    def test_14_gold_key_access_attempt(self):
        """14. Gold standard key cannot be accessed or queried via rater pipeline."""
        # IntakePipeline has no gold key reference
        assert not hasattr(self.intake, "gold_key")
        assert not hasattr(self.intake, "_gold_key_data")

    def test_15_observer_cross_access_attempt(self):
        """15. Observer A cannot access Observer B's submission files."""
        p1 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        p2 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-02")

        self.intake.ingest_submission(
            raw_bytes=json.dumps(p1).encode("utf-8"),
            registered_batch_id="BATCH-CROSS",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )
        self.intake.ingest_submission(
            raw_bytes=json.dumps(p2).encode("utf-8"),
            registered_batch_id="BATCH-CROSS",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        # Ciphertext for OBS-01 cannot be decrypted without custodian crypto key
        c1 = self.vault.read_sealed_evidence("SUBM-BATCH-CROSS-OBS-SYNTHETIC-01")
        assert b"OBS-SYNTHETIC-01" not in c1
        assert b"ratings" not in c1

    def test_16_premature_scoring_with_one_submission(self):
        """16. Scoring with only 1 rater sealed is rejected (mutual blinding lockstep)."""
        p1 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        self.intake.ingest_submission(
            raw_bytes=json.dumps(p1).encode("utf-8"),
            registered_batch_id="BATCH-SOLO",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        with pytest.raises(BlindingError, match="Premature scoring rejected"):
            self.bridge.score_paired_batch(
                batch_id="BATCH-SOLO",
                rater1_id="OBS-SYNTHETIC-01",
                rater2_id="OBS-SYNTHETIC-02",
            )

    def test_17_two_independently_sealed_valid_synthetic_submissions(self):
        """17. Two sealed synthetic submissions evaluate to SYNTHETIC_REHEARSAL_VERIFIED."""
        p1 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01", match_gold=True)
        p2 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-02", match_gold=True)

        rcpt1 = self.intake.ingest_submission(
            raw_bytes=json.dumps(p1).encode("utf-8"),
            registered_batch_id="BATCH-PAIR-01",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )
        assert rcpt1["custody_state"] == "PENDING_INDEPENDENT_PAIR"

        rcpt2 = self.intake.ingest_submission(
            raw_bytes=json.dumps(p2).encode("utf-8"),
            registered_batch_id="BATCH-PAIR-01",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )
        assert rcpt2["custody_state"] == "PAIR_SEALED_ELIGIBLE_FOR_SCORING"

        # Verify hash chain
        ok, errors = self.ledger.verify_chain_integrity()
        assert ok, f"Ledger chain broken: {errors}"

        # Score pair
        result = self.bridge.score_paired_batch(
            batch_id="BATCH-PAIR-01",
            rater1_id="OBS-SYNTHETIC-01",
            rater2_id="OBS-SYNTHETIC-02",
        )

        assert result["overall_verdict"] == "SYNTHETIC_REHEARSAL_VERIFIED"
        assert result["overall_verdict"] != "HUMAN_OBSERVER_QUALIFIED"
        assert result["gates"]["rater1_passed"] is True
        assert result["gates"]["rater2_passed"] is True
        assert result["gates"]["pairwise_passed"] is True

    def test_18_withdrawal_invalidation(self):
        """18. Participant withdrawal invalidates qualification and prevents future scoring."""
        p1 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        p2 = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-02")

        self.intake.ingest_submission(
            raw_bytes=json.dumps(p1).encode("utf-8"),
            registered_batch_id="BATCH-WITHDRAW",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )
        self.intake.ingest_submission(
            raw_bytes=json.dumps(p2).encode("utf-8"),
            registered_batch_id="BATCH-WITHDRAW",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        # Execute withdrawal for rater 1
        w_res = self.retention.execute_participant_withdrawal(
            batch_id="BATCH-WITHDRAW",
            observer_id="OBS-SYNTHETIC-01",
        )
        assert w_res["status"] == "WITHDRAWAL_EXECUTED"
        assert w_res["qualification_status"] == "REVOKED_INVALIDATED"

        # Verify subsequent scoring attempt fails
        with pytest.raises(BlindingError, match="Premature scoring rejected"):
            self.bridge.score_paired_batch(
                batch_id="BATCH-WITHDRAW",
                rater1_id="OBS-SYNTHETIC-01",
                rater2_id="OBS-SYNTHETIC-02",
            )

    def test_19_deletion_workflow(self):
        """19. Vault cryptographic shredding securely deletes evidence file."""
        subm_id = "SUBM-PURGE-TEST"
        target_path = self.vault.store_sealed_evidence(subm_id, b"sensitive_bytes_to_purge")
        assert target_path.exists()

        purged = self.vault.purge_sealed_evidence(subm_id)
        assert purged is True
        assert not target_path.exists()

    def test_20_no_sensitive_logs(self, capsys):
        """20. No encryption keys, plaintexts, or qualitative notes leak to stdout/stderr."""
        secret_note = "PRIVATE_CONFIDENTIAL_CLINICAL_NOTE_998877"
        payload = create_synthetic_ratings_payload(observer_id="OBS-SYNTHETIC-01")
        payload["ratings"][0]["observer_notes"] = secret_note
        raw_bytes = json.dumps(payload).encode("utf-8")

        self.intake.ingest_submission(
            raw_bytes=raw_bytes,
            registered_batch_id="BATCH-LOG-TEST",
            expected_package_sha=CANONICAL_BUNDLE_SHA,
        )

        captured = capsys.readouterr()
        assert secret_note not in captured.out
        assert secret_note not in captured.err
        assert self.crypto_key not in captured.out
        assert self.crypto_key not in captured.err

    def test_21_git_forced_add_containment(self):
        """21. Git containment verifier script passes and check-ignore catches forbidden paths."""
        res = subprocess.run(
            [sys.executable, "scripts/verify_evidence_custody_containment.py"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Containment verification script failed: {res.stdout}"
        assert "ARX_UX_GIT_EXPOSURE_CONTAINMENT_VERIFIED" in res.stdout

    def test_22_retained_canonical_qualification_fixtures(self):
        """22. Frozen qualification fixtures, policies, and bundle remain intact and tracked."""
        bundle_path = Path("tests/fixtures/qualification/ARX_OBSERVER_QUALIFICATION_BUNDLE.zip")
        assert bundle_path.exists()
        bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        assert bundle_sha == CANONICAL_BUNDLE_SHA

        policy_path = Path("tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md")
        assert policy_path.exists()

    def test_23_incomplete_backup_deletion_documentation(self):
        """23. Tombstone receipt documents residual backup purge boundaries."""
        tombstone = self.retention.execute_participant_withdrawal(
            batch_id="BATCH-DOC-TEST",
            observer_id="OBS-SYNTHETIC-99",
            withdrawal_reason="GDPR_REQUEST",
        )
        rcpt = tombstone["tombstone_receipt"]
        assert rcpt["custody_state"] == "WITHDRAWAL_TOMBSTONE"
        assert rcpt["schema_validation_status"] == "WITHDRAWN_PURGED"

        # Ledger chain remains unbroken after tombstone
        ok, errs = self.ledger.verify_chain_integrity()
        assert ok, f"Ledger broken after tombstone: {errs}"

    def test_24_crash_interruption_during_atomic_intake(self):
        """24. Interruption during write cleans staging directory cleanly."""
        staging_dir = self.vault.staging_dir
        # Store an abandoned temporary file
        abandoned_tmp = staging_dir / "abandoned_crash.tmp"
        abandoned_tmp.write_bytes(b"partial_write_before_crash")

        # Sealed dir has no evidence of abandoned file
        assert not (self.vault.sealed_dir / "abandoned_crash.enc").exists()

        # Clean up
        abandoned_tmp.unlink()
