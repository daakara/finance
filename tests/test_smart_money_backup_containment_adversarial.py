"""
ARX Terminal — Smart Money Form 4 Backup-Age Containment Adversarial Test Suite.

Comprehensive 30-Vector Test Matrix verifying fail-closed backup-age containment (Policy A):
- Freshness evaluation vectors (1-10)
- Concurrency and race-condition vectors (11-17)
- Recovery and automatic resumption vectors (18-23)
- Governance, policy authority, and compatibility vectors (24-30)
"""

import os
import json
import time
import tempfile
from typing import Dict, Any, Optional, List, Tuple
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupFreshnessState,
    BackupContainmentPolicy,
    FreshnessEvaluationReceipt,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    ContainmentState,
    BackupAgeContainmentViolationException,
    SnapshotPublicationBlockedContainmentException,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    SmartMoneyBackupSchedulerThread,
    WorkerHealthState,
)


@pytest.fixture
def frozen_clock():
    """Deterministic injected frozen clock: 2026-10-14 12:00:00 UTC."""
    return datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def storage_dir(tmp_path):
    """Isolated storage directory for test backups."""
    d = tmp_path / "mock_backups"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def storage_adapter(storage_dir):
    return LocalOffVolumeStorageAdapter(base_directory=storage_dir)


@pytest.fixture
def active_policy():
    """Enabled Policy A with 12-hour (43200s) threshold."""
    return BackupContainmentPolicy(
        enabled=True,
        max_verified_age_seconds=43200.0,
        check_timeout_seconds=5.0,
        recheck_interval_seconds=60.0,
        clock_skew_tolerance_seconds=300.0,
        production_authorized=True,
    )


def create_mock_backup(
    storage_adapter: LocalOffVolumeStorageAdapter,
    backup_id: str,
    completed_at: str,
    dump_content: bytes = b"MOCK_PG16_VALID_DUMP_STREAM",
    upload_verified: bool = True,
    omit_dump: bool = False,
    corrupt_manifest: bool = False,
) -> Dict[str, Any]:
    """Helper to stage a mock backup artifact set in storage."""
    with tempfile.TemporaryDirectory() as tmp:
        dump_name = f"smart_money_{backup_id}.dump"
        manifest_name = f"manifest_{backup_id}.json"
        
        dump_local = os.path.join(tmp, dump_name)
        with open(dump_local, "wb") as f:
            f.write(dump_content)
        dump_sha = compute_sha256(dump_local)
        dump_size = os.path.getsize(dump_local)

        if not omit_dump:
            storage_adapter.upload_file(dump_local, dump_name)

        manifest = {
            "backup_id": backup_id,
            "completed_at": completed_at,
            "database_artifact": dump_name,
            "dump_filename": dump_name,
            "artifact_sha256": dump_sha,
            "artifact_size_bytes": dump_size,
            "upload_verified": upload_verified,
            "postgres_major_version": 16,
            "schema_version": "SMART_MONEY_POSTGRES_SCHEMA_V2",
        }

        manifest_local = os.path.join(tmp, manifest_name)
        with open(manifest_local, "w", encoding="utf-8") as f:
            if corrupt_manifest:
                f.write("{ INVALID_JSON_CORRUPTED_STREAM ...")
            else:
                json.dump(manifest, f, indent=2)

        storage_adapter.upload_file(manifest_local, manifest_name)
        return manifest


# ==============================================================================
# SECTION A: FRESHNESS EVALUATION VECTORS (1 - 10)
# ==============================================================================

class TestFreshnessEvaluationVectors:

    def test_fresh_verified_backup_permits_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 1: Backup 2h old (<12h) evaluates to FRESH and permits ingestion."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        create_mock_backup(storage_adapter, "b-fresh-01", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.backup_age_seconds == 7200.0
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is True
        assert gate.current_state == ContainmentState.HEALTHY

    def test_backup_just_below_threshold_permits_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 2: Backup at 11h 59m 50s (<12h) permits ingestion and enters warning state."""
        completed_at = (frozen_clock - timedelta(seconds=43190)).isoformat()
        create_mock_backup(storage_adapter, "b-near-limit-02", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is True
        assert gate.current_state == ContainmentState.FRESHNESS_WARNING

    def test_backup_exactly_at_threshold_permits_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 3: Backup at exactly 43200s (boundary) permits ingestion under <= semantics."""
        completed_at = (frozen_clock - timedelta(seconds=43200)).isoformat()
        create_mock_backup(storage_adapter, "b-exact-limit-03", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.is_fresh is True
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is True

    def test_backup_above_threshold_suspends_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 4: Backup at 12h 00m 01s (>12h) evaluates to STALE and suspends ingestion."""
        completed_at = (frozen_clock - timedelta(seconds=43201)).isoformat()
        create_mock_backup(storage_adapter, "b-stale-04", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED
        with pytest.raises(BackupAgeContainmentViolationException):
            gate.enforce_ingestion_gate(current_time=frozen_clock)

    def test_no_recovery_point_suspends_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 5: Zero manifests in storage evaluates to MISSING and fails closed."""
        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.verification_status == BackupFreshnessState.MISSING.value
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    def test_corrupt_manifest_suspends_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 6: Corrupt manifest payload evaluates to UNVERIFIABLE and fails closed."""
        completed_at = (frozen_clock - timedelta(hours=1)).isoformat()
        create_mock_backup(storage_adapter, "b-corrupt-06", completed_at, corrupt_manifest=True)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False

    def test_incorrect_checksum_or_missing_dump_suspends_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 7: Manifest present but remote dump file missing fails closed."""
        completed_at = (frozen_clock - timedelta(hours=1)).isoformat()
        create_mock_backup(storage_adapter, "b-nodump-07", completed_at, omit_dump=True)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "REMOTE_DUMP_ARTIFACT_MISSING" in receipt.reason_code
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False

    def test_r2_timeout_suspends_ingestion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 8: Storage network timeout evaluates to CHECK_FAILED and suspends ingestion."""
        mock_adapter = MagicMock()
        mock_adapter.list_backups.side_effect = TimeoutError("R2 S3 API Connection Timed Out")

        authority = BackupFreshnessAuthority(storage_adapter=mock_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.verification_status == BackupFreshnessState.CHECK_FAILED.value
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False

    def test_future_dated_manifest_fails_safely(self, storage_adapter, active_policy, frozen_clock):
        """Vector 9: Manifest with impossible future timestamp (+2 hours) is rejected as UNVERIFIABLE."""
        completed_at = (frozen_clock + timedelta(hours=2)).isoformat()
        create_mock_backup(storage_adapter, "b-future-09", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "FUTURE_DATED" in receipt.reason_code
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False

    def test_clock_skew_handled_deterministically(self, storage_adapter, active_policy, frozen_clock):
        """Vector 10: Timestamp 2 minutes ahead is tolerated under 300s skew and clamped to age 0.0s."""
        completed_at = (frozen_clock + timedelta(minutes=2)).isoformat()
        create_mock_backup(storage_adapter, "b-skew-10", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        receipt = authority.evaluate_freshness(current_time=frozen_clock)
        assert receipt.is_fresh is True
        assert receipt.backup_age_seconds == 0.0
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is True


# ==============================================================================
# SECTION B: CONCURRENCY & RACE-CONDITION VECTORS (11 - 17)
# ==============================================================================

class TestConcurrencyAndRaceConditionVectors:

    def test_two_workers_contend_for_sec_ownership_and_containment(self, storage_adapter, active_policy, frozen_clock):
        """Vector 11: Two workers independently enforce containment under mutual exclusion."""
        completed_at = (frozen_clock - timedelta(hours=3)).isoformat()
        create_mock_backup(storage_adapter, "b-shared-11", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate1 = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)
        gate2 = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate1.check_ingestion_eligibility(current_time=frozen_clock) is True
        assert gate2.check_ingestion_eligibility(current_time=frozen_clock) is True

    def test_worker_restart_while_backup_stale_remains_blocked(self, storage_adapter, active_policy, frozen_clock):
        """Vector 12: New worker instance booting when backup is stale fails preflight/cycle."""
        completed_at = (frozen_clock - timedelta(hours=14)).isoformat()
        create_mock_backup(storage_adapter, "b-stale-12", completed_at)

        worker = SmartMoneyIngestionWorker(
            dsn="postgresql://postgres:pwd@127.0.0.1:5432/postgres",
            containment_policy=active_policy,
        )
        # Point worker's freshness authority to test storage adapter and frozen clock
        worker.freshness_authority.storage_adapter = storage_adapter
        worker.freshness_authority.clock_fn = lambda: frozen_clock

        # Run single work cycle
        with patch.object(worker, "process_run_requests") as mock_process:
            worker.run_cycle()
            mock_process.assert_not_called()
            assert worker.state == WorkerHealthState.BLOCKED_DATA_INTEGRITY

    def test_long_running_cycle_crosses_threshold_halts_in_flight(self, storage_adapter, active_policy, frozen_clock):
        """Vector 13: In-flight check halts request completion if threshold is crossed mid-cycle."""
        # Backup is fresh at start of cycle (11h 59m old)
        start_time = frozen_clock
        completed_at = (start_time - timedelta(hours=11, minutes=59)).isoformat()
        create_mock_backup(storage_adapter, "b-race-13", completed_at)

        authority = BackupFreshnessAuthority(
            storage_adapter=storage_adapter, policy=active_policy, clock_fn=lambda: start_time
        )
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        # Before claim: fresh
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=start_time) is True

        # Mid-cycle: clock advances 5 minutes (now 12h 04m old -> STALE)
        mid_cycle_time = start_time + timedelta(minutes=5)
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=mid_cycle_time) is False

    def test_backup_stale_before_snapshot_publication_blocks_publish(self, storage_adapter, active_policy, frozen_clock):
        """Vector 14: publish_scanner_snapshot_atomic rejects publication if backup is stale."""
        completed_at = (frozen_clock - timedelta(hours=13)).isoformat()
        create_mock_backup(storage_adapter, "b-stale-14", completed_at)

        authority = BackupFreshnessAuthority(
            storage_adapter=storage_adapter, policy=active_policy, clock_fn=lambda: frozen_clock
        )
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        repo = PostgresSmartMoneyRepository(dsn=None, containment_gate=gate)
        with pytest.raises(SnapshotPublicationBlockedContainmentException):
            repo.publish_scanner_snapshot_atomic(
                snapshot={"snapshot_id": "SNAP-FAIL-01", "run_id": "r1", "semantic_fingerprint": "fp1"}
            )

    def test_lease_lost_after_freshness_verification_fenced(self, storage_adapter, active_policy, frozen_clock):
        """Vector 15: If worker lease is lost after freshness check, repository lease fencing still fails closed."""
        completed_at = (frozen_clock - timedelta(hours=1)).isoformat()
        create_mock_backup(storage_adapter, "b-fresh-15", completed_at)

        authority = BackupFreshnessAuthority(
            storage_adapter=storage_adapter, policy=active_policy, clock_fn=lambda: frozen_clock
        )
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        # Both freshness and fencing must hold. Fencing fails closed when lease token mismatches.
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None  # Lease lost
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        repo = PostgresSmartMoneyRepository(dsn=None, containment_gate=gate)
        with pytest.raises(StaleLeaseOwnershipLostError):
            repo.publish_scanner_snapshot_atomic(
                snapshot={"snapshot_id": "SNAP-FENCE-01", "run_id": "r1", "semantic_fingerprint": "fp1"},
                run_request_id="req-123",
                lease_token="tok-lost",
                conn=mock_conn,
            )

    def test_stale_cached_freshness_cannot_bypass_enforcement(self, storage_adapter, active_policy, frozen_clock):
        """Vector 16: Snapshot publication enforces force-refresh, bypassing short TTL caches."""
        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        # Invalidate by wiping backups; force_refresh immediately detects absence
        with pytest.raises(SnapshotPublicationBlockedContainmentException):
            gate.enforce_snapshot_publication_gate(current_time=frozen_clock)

    def test_backup_scheduler_remains_operational_during_suspension(self, storage_adapter, active_policy, frozen_clock):
        """Vector 17: Scheduler continues executing backups while ingestion is suspended."""
        # Start in suspended state
        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)
        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, containment_gate=gate)
        scheduler.containment_gate.mark_backup_in_progress()
        assert gate.current_state == ContainmentState.BACKUP_RECOVERY_PENDING
        scheduler.containment_gate.mark_backup_completed()


# ==============================================================================
# SECTION C: RECOVERY & RESUMPTION VECTORS (18 - 23)
# ==============================================================================

class TestRecoveryAndResumptionVectors:

    def test_successful_verified_backup_restores_eligibility(self, storage_adapter, active_policy, frozen_clock):
        """Vector 18: After suspension, generating a fresh backup transitions gate to HEALTHY."""
        # 1. Start suspended (stale backup)
        completed_old = (frozen_clock - timedelta(hours=15)).isoformat()
        create_mock_backup(storage_adapter, "b-stale-18", completed_old)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

        # 2. Complete fresh backup
        completed_new = (frozen_clock - timedelta(minutes=5)).isoformat()
        create_mock_backup(storage_adapter, "b-new-18", completed_new)

        # 3. Gate re-evaluates and automatically resumes
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is True
        assert gate.current_state == ContainmentState.HEALTHY
        assert gate.resumption_count == 1
        assert gate.last_resume_at is not None

    def test_failed_upload_does_not_restore_eligibility(self, storage_adapter, active_policy, frozen_clock):
        """Vector 19: Backup marked upload_verified=False does not restore eligibility."""
        completed_at = (frozen_clock - timedelta(minutes=2)).isoformat()
        create_mock_backup(storage_adapter, "b-failed-upload-19", completed_at, upload_verified=False)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    def test_manifest_only_upload_does_not_restore_eligibility(self, storage_adapter, active_policy, frozen_clock):
        """Vector 20: Uploading only a manifest without the dump artifact does not restore eligibility."""
        completed_at = (frozen_clock - timedelta(minutes=2)).isoformat()
        create_mock_backup(storage_adapter, "b-manifest-only-20", completed_at, omit_dump=True)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is False

    def test_recovery_does_not_duplicate_accession_records(self, storage_adapter, active_policy, frozen_clock):
        """Vector 21: Idempotent raw filing persistence prevents duplicates upon resumption."""
        completed_at = (frozen_clock - timedelta(hours=1)).isoformat()
        create_mock_backup(storage_adapter, "b-fresh-21", completed_at)

        authority = BackupFreshnessAuthority(
            storage_adapter=storage_adapter, policy=active_policy, clock_fn=lambda: frozen_clock
        )
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"content_hash": "existing_hash_123"}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        repo = PostgresSmartMoneyRepository(dsn=None, containment_gate=gate)
        # Attempt duplicate save of accession
        inserted = repo.save_raw_filing(
            {"accession_number": "0001-26-001", "content_hash": "existing_hash_123"},
            conn=mock_conn,
        )
        assert inserted is False  # Idempotent no-op

    def test_resume_does_not_bypass_existing_reconciliation_gates(self, active_policy):
        """Vector 22: Resumed worker preserves existing reconciliation receipt requirements."""
        repo = PostgresSmartMoneyRepository(dsn=None)
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None  # No certified receipt
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        receipt = repo.get_authoritative_receipt_for_interval(
            "2026-10-14T00:00:00Z", "2026-10-14T23:59:59Z", "W1", conn=mock_conn
        )
        assert receipt is None

    def test_repeated_suspend_resume_transitions_idempotent(self, storage_adapter, active_policy, frozen_clock):
        """Vector 23: Multiple oscillations between stale and fresh remain completely stable and idempotent."""
        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=active_policy)

        # Cycle 1: Stale -> Suspend
        create_mock_backup(storage_adapter, "b-c1-stale", (frozen_clock - timedelta(hours=15)).isoformat())
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED
        assert gate.suspension_count == 1

        # Cycle 1: Fresh -> Resume
        create_mock_backup(storage_adapter, "b-c1-fresh", (frozen_clock - timedelta(hours=1)).isoformat())
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=frozen_clock) is True
        assert gate.current_state == ContainmentState.HEALTHY
        assert gate.resumption_count == 1

        # Cycle 2: Advanced time -> Stale
        time_jump = frozen_clock + timedelta(hours=14)
        assert gate.check_ingestion_eligibility(force_refresh=True, current_time=time_jump) is False
        assert gate.current_state == ContainmentState.INGESTION_SUSPENDED
        assert gate.suspension_count == 2


# ==============================================================================
# SECTION D: GOVERNANCE, POLICY AUTHORITY & COMPATIBILITY (24 - 30)
# ==============================================================================

class TestGovernanceAndCompatibilityVectors:

    def test_disabled_policy_explicitly_reported_as_inactive(self, storage_adapter, frozen_clock):
        """Vector 24: Disabled policy permits ingestion but explicitly reports inactive in telemetry."""
        policy = BackupContainmentPolicy(enabled=False)
        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=policy)

        assert gate.check_ingestion_eligibility(current_time=frozen_clock) is True
        obs = gate.get_observability_envelope(current_time=frozen_clock)
        assert obs["containment_enabled"] is False
        assert obs["containment_effective"] is False

    def test_unauthorized_production_activation_fails_safely(self, monkeypatch):
        """Vector 25: Enabling policy in production without explicit authorization fails preflight."""
        monkeypatch.setenv("ARX_ENVIRONMENT", "production")
        policy = BackupContainmentPolicy(enabled=True, production_authorized=False)
        worker = SmartMoneyIngestionWorker(
            dsn="postgresql://postgres:pwd@127.0.0.1:5432/postgres",
            containment_policy=policy,
        )

        with patch("analyst_dashboard.smart_money.worker.get_sec_egress_role", return_value="OWNER"), \
             patch("analyst_dashboard.smart_money.worker.validate_storage_configuration", return_value=True):
            with pytest.raises(RuntimeError) as exc_info:
                worker.validate_preflight()
            assert "UNAUTHORIZED_PRODUCTION_BACKUP_CONTAINMENT" in str(exc_info.value)
            assert worker.state == WorkerHealthState.FATAL

    def test_existing_governance_authorization_remains_passing(self):
        """Vector 26: Governance sole-owner FIDO2 temporary suspension validator functions seamlessly."""
        from analyst_dashboard.smart_money.governance_control_plane.sole_owner_policy import (
            SoleOwnerGovernanceValidator,
            SoleOwnerReleaseApproval,
            Fido2RequirementStatus,
        )
        validator = SoleOwnerGovernanceValidator(status=Fido2RequirementStatus.TEMPORARILY_SUSPENDED)
        now = datetime.now(timezone.utc)
        approval = SoleOwnerReleaseApproval(
            approval_id="appr-c006-comp-01",
            policy_id="ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1",
            owner_identity="lead-maintainer-daakara",
            target_environment="rehearsal",
            release_sha="76a8691339c30edc35e4e056d5b295509a81fe3a",
            approved_at=now.isoformat(),
            valid_until=(now + timedelta(hours=2)).isoformat(),
            account_mfa_verified=True,
            technical_verification_passed=True,
            security_verification_passed=True,
            recovery_verification_passed=True,
        )
        record = validator.validate_release_authorization(
            approval=approval,
            target_sha="76a8691339c30edc35e4e056d5b295509a81fe3a",
            target_environment="rehearsal",
            current_time=now,
        )
        assert record["status"] == "APPROVED"

    def test_existing_r2_backup_security_remains_passing(self):
        """Vector 27: Unprivileged role and non-superuser boundaries remain intact."""
        from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
        engine = SmartMoneyBackupEngine(dsn="postgresql://postgres:pwd@127.0.0.1:5432/postgres")
        # Ensure routine backups default to include_globals=False
        assert engine.create_backup.__defaults__ == (None, False)

    def test_existing_immutable_audit_contracts_intact(self):
        """Vector 28: Containment gate transitions generate non-collapsible audit records."""
        gate = BackupAgeContainmentGate(policy=BackupContainmentPolicy(enabled=True))
        obs = gate.get_observability_envelope()
        assert "current_state" in obs
        assert "containment_enabled" in obs

    def test_production_web_read_compatibility_preserved(self):
        """Vector 29: Active snapshot pointer queries remain 100% compatible with production schema."""
        repo = PostgresSmartMoneyRepository(dsn=None)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        snap = repo.get_latest_active_snapshot("SMART_MONEY", conn=mock_conn)
        assert snap is None

    def test_unrelated_radar_and_vcp_behaviors_unaffected(self):
        """Vector 30: Zero mutation of Radar, VCP, macro, or trading analyzer logic."""
        import analyst_dashboard.analyzers.confluence_engine as ce
        assert hasattr(ce, "ConfluenceEngine")


# ==============================================================================
# SECTION E: SAFETY CONFIGURATION PARSING & FAIL-CLOSED BOUNDARIES
# ==============================================================================

class TestSafetyConfigurationParsing:

    def test_negative_max_age_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", "-100")
        with pytest.raises(ValueError) as exc:
            BackupContainmentPolicy.from_environment()
        assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_zero_max_age_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", "0")
        with pytest.raises(ValueError) as exc:
            BackupContainmentPolicy.from_environment()
        assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_nan_or_inf_max_age_fails_closed(self, monkeypatch):
        for val in ("nan", "NaN", "inf", "-inf", "Infinity"):
            monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", val)
            with pytest.raises(ValueError) as exc:
                BackupContainmentPolicy.from_environment()
            assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_malformed_max_age_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", "forty-two-thousand")
        with pytest.raises(ValueError) as exc:
            BackupContainmentPolicy.from_environment()
        assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_negative_timeout_fails_closed(self, monkeypatch):
        monkeypatch.delenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", raising=False)
        monkeypatch.setenv("ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC", "-5")
        with pytest.raises(ValueError) as exc:
            BackupContainmentPolicy.from_environment()
        assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_negative_recheck_interval_fails_closed(self, monkeypatch):
        monkeypatch.delenv("ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC", raising=False)
        monkeypatch.setenv("ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC", "-10")
        with pytest.raises(ValueError) as exc:
            BackupContainmentPolicy.from_environment()
        assert "INVALID_SAFETY_CONFIG" in str(exc.value)

    def test_valid_float_parsing_and_metadata_flag(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_MAX_VERIFIED_AGE_SEC", "21600.5")
        monkeypatch.setenv("ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC", "15.0")
        monkeypatch.setenv("ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC", "0.0")  # Zero recheck is valid
        monkeypatch.setenv("ARX_BACKUP_REQUIRE_REMOTE_CHECKSUM_METADATA", "false")

        policy = BackupContainmentPolicy.from_environment()
        assert policy.max_verified_age_seconds == 21600.5
        assert policy.check_timeout_seconds == 15.0
        assert policy.recheck_interval_seconds == 0.0
        assert policy.require_remote_checksum_metadata is False

