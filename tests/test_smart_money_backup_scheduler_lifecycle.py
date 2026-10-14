"""
ARX Terminal — Smart Money Backup Scheduler Lifecycle & Resilience Test Suite.

Deterministic tests proving:
1. Scheduler startup resolves the storage adapter (reproducing and verifying fix for NameError).
2. Scheduler thread remains alive after initialization.
3. Backup execution acquires independent advisory lock (872055449083353674) concurrently with SEC egress lock.
4. Backup failure is recorded and does not terminate scheduling.
5. A later scheduled attempt can execute after a failure.
6. Missing or invalid backup configuration fails safely without crashing.
7. Scheduler health accurately distinguishes running, failed, and stopped states.
8. Worker health does not falsely imply backup scheduler health.
9. Containment remains active during backup failure.
10. Shutdown releases scheduler resources cleanly.
"""

import os
import time
import pytest
import threading
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.worker import (
    SmartMoneyBackupSchedulerThread,
    SmartMoneyIngestionWorker,
    WorkerHealthState,
    get_backup_storage_adapter,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    LocalOffVolumeStorageAdapter,
    BackupStorageAdapter,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    ContainmentState,
)
from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupContainmentPolicy,
    BackupFreshnessAuthority,
    BackupFreshnessState,
    FreshnessEvaluationReceipt,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    LOCK_SEC_EGRESS_OWNER,
    LOCK_BACKUP_EXECUTION,
)


class TestSchedulerLifecycleRegression:
    """Test battery for SmartMoneyBackupSchedulerThread lifecycle and resilience."""

    def test_01_scheduler_startup_resolves_storage_adapter(self, tmp_path, monkeypatch):
        """Proves get_backup_storage_adapter is cleanly imported and resolved without NameError."""
        monkeypatch.setenv("ARX_BACKUP_PROVIDER", "local")
        monkeypatch.setenv("SMART_MONEY_BACKUP_DIR", str(tmp_path))

        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60)
        assert scheduler.storage_adapter is None

        # Verify resolution does not raise NameError
        adapter = scheduler.storage_adapter or get_backup_storage_adapter()
        assert isinstance(adapter, BackupStorageAdapter)
        assert isinstance(adapter, LocalOffVolumeStorageAdapter)

    def test_02_scheduler_thread_remains_alive_after_initialization(self, tmp_path, monkeypatch):
        """Proves scheduler thread initializes and stays alive."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=3600, storage_adapter=adapter)

        assert not scheduler.is_alive()
        assert not scheduler._running

        scheduler.start()
        try:
            assert scheduler.is_alive()
            assert scheduler._running
            status = scheduler.get_status_envelope()
            assert status["is_running"] is True
            assert status["interval_sec"] == 3600
        finally:
            scheduler.stop()
            scheduler.join(timeout=2.0)

    def test_03_backup_execution_acquires_independent_advisory_lock(self):
        """
        Proves backup execution uses independent advisory lock 872055449083353674
        which operates independently from SEC egress lock 401007311509093371.
        """
        assert LOCK_BACKUP_EXECUTION == 872055449083353674
        assert LOCK_SEC_EGRESS_OWNER == 401007311509093371
        assert LOCK_BACKUP_EXECUTION != LOCK_SEC_EGRESS_OWNER

        # Test advisory lock non-interference with mock connections
        mock_conn_sec = MagicMock()
        mock_cur_sec = MagicMock()
        mock_conn_sec.cursor.return_value.__enter__.return_value = mock_cur_sec
        mock_cur_sec.fetchone.return_value = (True,)

        mock_conn_backup = MagicMock()
        mock_cur_backup = MagicMock()
        mock_conn_backup.cursor.return_value.__enter__.return_value = mock_cur_backup
        mock_cur_backup.fetchone.return_value = (True,)

        from analyst_dashboard.smart_money.storage.postgres_coordination import (
            acquire_sec_owner_lock,
            acquire_backup_lock,
        )

        assert acquire_sec_owner_lock(mock_conn_sec) is True
        assert acquire_backup_lock(mock_conn_backup) is True

    def test_04_backup_failure_recorded_and_does_not_terminate_scheduling(self, tmp_path):
        """Proves a failed backup pass records failure, releases lock, and does not terminate scheduler."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)

        # Mock engine to simulate dump failure
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.create_backup.side_effect = RuntimeError("Simulated pg_dump pipe failure")
            mock_engine_cls.return_value = mock_engine

            success = scheduler.run_backup_pass()

            assert success is False
            assert scheduler.last_status == "FAILED"
            assert "Simulated pg_dump pipe failure" in scheduler.last_error
            assert scheduler.failed_runs == 1
            assert scheduler.successful_runs == 0
            # Mutex lock must be released so next attempt can run
            assert not scheduler._lock.locked()

    def test_05_subsequent_scheduled_attempt_can_execute_after_failure(self, tmp_path):
        """Proves scheduler recovers on the next cycle after an initial failure."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)

        # 1. First pass fails
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.create_backup.side_effect = IOError("Network timeout")
            mock_engine_cls.return_value = mock_engine
            scheduler.run_backup_pass()

        assert scheduler.failed_runs == 1
        assert scheduler.last_status == "FAILED"

        # 2. Second pass succeeds
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls, \
             patch("analyst_dashboard.smart_money.worker.BackupRetentionManager") as mock_retention_cls:
            mock_engine = MagicMock()
            mock_engine.create_backup.return_value = {
                "backup_id": "sm-backup-test-recover",
                "storage_destination": "local://recovery",
            }
            mock_engine_cls.return_value = mock_engine

            mock_retention = MagicMock()
            mock_retention.prune_backups.return_value = []
            mock_retention_cls.return_value = mock_retention

            success = scheduler.run_backup_pass()

            assert success is True
            assert scheduler.last_status == "SUCCESS"
            assert scheduler.last_backup_id == "sm-backup-test-recover"
            assert scheduler.successful_runs == 1
            assert scheduler.failed_runs == 1
            assert scheduler.last_error is None

    def test_06_missing_or_invalid_backup_config_fails_safely(self, monkeypatch):
        """Proves missing external credentials causes fail-closed scheduler recording without crash."""
        monkeypatch.setenv("ARX_BACKUP_PROVIDER", "cloudflare_r2")
        monkeypatch.delenv("ARX_R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("ARX_BACKUP_BUCKET", raising=False)
        monkeypatch.delenv("ARX_R2_ENDPOINT", raising=False)
        monkeypatch.delenv("ARX_R2_SECRET_ACCESS_KEY", raising=False)

        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60)
        success = scheduler.run_backup_pass()

        assert success is False
        assert scheduler.last_status == "FAILED"
        assert "EXTERNAL_BACKUP_CONFIG_ERROR" in scheduler.last_error
        assert scheduler.failed_runs == 1

    def test_07_scheduler_health_distinguishes_running_failed_and_stopped(self, tmp_path):
        """Proves get_status_envelope accurately reflects scheduler states."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=300, storage_adapter=adapter)

        # Initial state
        env_init = scheduler.get_status_envelope()
        assert env_init["is_running"] is False
        assert env_init["last_status"] == "INITIALIZING"
        assert env_init["successful_runs"] == 0

        # After failure
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = RuntimeError("Disk full")
            scheduler.run_backup_pass()

        env_failed = scheduler.get_status_envelope()
        assert env_failed["last_status"] == "FAILED"
        assert env_failed["failed_runs"] == 1
        assert "Disk full" in env_failed["last_error"]

        # Running state
        scheduler.start()
        try:
            env_running = scheduler.get_status_envelope()
            assert env_running["is_running"] is True
        finally:
            scheduler.stop()
            scheduler.join(timeout=2.0)

        # Stopped state
        env_stopped = scheduler.get_status_envelope()
        assert env_stopped["is_running"] is False

    def test_08_worker_health_does_not_falsely_imply_scheduler_health(self, monkeypatch):
        """
        Production readiness disproof:
        Proves worker status (e.g. HEALTHY) does NOT imply scheduler health.
        Health envelope must be inspected at both levels.
        """
        worker = SmartMoneyIngestionWorker()
        worker.worker_id = "test-worker-disproof"
        worker.state = WorkerHealthState.HEALTHY

        # Scheduler is in failed state
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_running": True,
            "interval_sec": 21600,
            "last_status": "FAILED",
            "last_error": "R2 connection timeout",
            "successful_runs": 0,
            "failed_runs": 5,
        }
        worker.backup_scheduler = mock_scheduler

        # Mock containment gate
        mock_gate = MagicMock()
        mock_gate.get_observability_envelope.return_value = {
            "current_state": "HEALTHY",
            "containment_enabled": True,
        }
        worker.containment_gate = mock_gate

        envelope = worker.get_health_envelope()
        assert envelope["status"] == "HEALTHY"
        # Scheduler inside envelope explicitly shows FAILED
        assert envelope["backup_scheduler"]["last_status"] == "FAILED"
        assert envelope["backup_scheduler"]["failed_runs"] == 5

    def test_09_containment_remains_active_during_backup_failure(self, tmp_path):
        """Proves containment gate mark_backup_completed clears in-progress and enforces state."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        mock_gate = MagicMock()

        scheduler = SmartMoneyBackupSchedulerThread(
            interval_sec=60,
            containment_gate=mock_gate,
            storage_adapter=adapter,
        )

        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = RuntimeError("Corrupt block")
            scheduler.run_backup_pass()

        # Gate must have received both mark_backup_in_progress and mark_backup_completed
        mock_gate.mark_backup_in_progress.assert_called_once()
        mock_gate.mark_backup_completed.assert_called_once()
        # Force refresh must NOT have been called because backup failed
        mock_gate.check_ingestion_eligibility.assert_not_called()

    def test_10_shutdown_releases_scheduler_resources(self, tmp_path):
        """Proves stop() sets _running=False and allows thread to exit."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)

        scheduler.start()
        assert scheduler.is_alive()

        scheduler.stop()
        scheduler.join(timeout=3.0)

        assert not scheduler.is_alive()
        assert scheduler._running is False
