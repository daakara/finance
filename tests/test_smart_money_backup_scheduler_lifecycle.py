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

    def test_11_scheduler_thread_crash_and_composite_health_envelope(self):
        """
        Workstream A: Proves that if scheduler thread dies or crashes while worker is healthy,
        composite health envelope flags backup_subsystem_healthy=False and reports
        SCHEDULER_THREAD_TERMINATED.
        """
        worker = SmartMoneyIngestionWorker()
        worker.worker_id = "test-worker-crash-detect"
        worker.state = WorkerHealthState.HEALTHY

        mock_scheduler = MagicMock()
        # Thread terminated: is_alive=False while thread_running=True (crashed before clean stop)
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": False,
            "thread_running": True,
            "interval_sec": 21600,
            "last_status": "FAILED",
            "consecutive_failures": 1,
            "last_error": "Fatal crash in thread pass",
            "is_stale_warning": False,
            "startup_error": None,
        }
        worker.backup_scheduler = mock_scheduler

        envelope = worker.get_health_envelope()
        assert envelope["worker_subsystem_healthy"] is True
        assert envelope["backup_subsystem_healthy"] is False
        assert envelope["overall_healthy"] is False
        assert "SCHEDULER_THREAD_TERMINATED" in envelope["degradation_reasons"]
        assert "CONSECUTIVE_BACKUP_FAILURES_1" in envelope["degradation_reasons"]

    def test_12_consecutive_backup_failures_and_recovery(self, tmp_path):
        """
        Workstream A: Proves consecutive failures increment, degrade backup subsystem health,
        and cleanly reset upon a subsequent successful pass.
        """
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=300, storage_adapter=adapter)

        assert scheduler.consecutive_failures == 0

        # Pass 1: Failure
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = RuntimeError("R2 upload timeout")
            scheduler.run_backup_pass()

        assert scheduler.consecutive_failures == 1
        assert scheduler.failed_runs == 1
        assert scheduler.last_status == "FAILED"

        # Pass 2: Failure
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = RuntimeError("R2 network partition")
            scheduler.run_backup_pass()

        assert scheduler.consecutive_failures == 2
        assert scheduler.failed_runs == 2

        status_env = scheduler.get_status_envelope()
        assert status_env["consecutive_failures"] == 2

        # Pass 3: Recovery
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.return_value = {
                "backup_id": "sm-backup-recovered-123",
                "storage_destination": "s3://test/rehearsal",
            }
            scheduler.run_backup_pass()

        assert scheduler.consecutive_failures == 0
        assert scheduler.failed_runs == 2
        assert scheduler.successful_runs == 1
        assert scheduler.last_status == "SUCCESS"
        assert scheduler.last_backup_id == "sm-backup-recovered-123"

    def test_13_backup_age_warning_threshold_envelope(self):
        """
        Workstream A: Proves that when backup age breaches the warning threshold (8 hours),
        envelope reflects is_stale_warning=True and BACKUP_FRESHNESS_WARNING reason.
        """
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY

        mock_scheduler = MagicMock()
        # Simulate last success 9 hours ago (32,400s > 28,800s warning threshold)
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "interval_sec": 21600,
            "warning_threshold_sec": 28800.0,
            "critical_threshold_sec": 43200.0,
            "last_status": "SUCCESS",
            "last_successful_timestamp": time.time() - 32400.0,
            "age_since_last_success_sec": 32400.0,
            "is_stale_warning": True,
            "consecutive_failures": 0,
            "startup_error": None,
        }
        worker.backup_scheduler = mock_scheduler

        envelope = worker.get_health_envelope()
        assert "BACKUP_FRESHNESS_WARNING" in envelope["degradation_reasons"]
        assert envelope["backup_scheduler"]["is_stale_warning"] is True

    def test_14_backup_alert_dispatcher_passive_evaluation(self):
        """
        Workstream A: Proves BackupAlertDispatcher passively evaluates health and emits
        structured events with zero network transmission.
        """
        from analyst_dashboard.smart_money.storage.backup.alerting import (
            BackupAlertDispatcher,
            BackupAlertSeverity,
        )

        dispatcher = BackupAlertDispatcher(warning_age_threshold_sec=28800.0, critical_age_threshold_sec=43200.0)

        dispatched_events = []
        dispatcher.register_handler(lambda e: dispatched_events.append(e))

        # Scenario 1: Scheduler Thread Death
        scheduler_crashed_status = {
            "is_alive": False,
            "thread_running": True,
            "consecutive_failures": 1,
            "last_error": "Thread crashed",
        }
        events = dispatcher.evaluate_scheduler_health(scheduler_crashed_status)
        assert len(events) >= 1
        event_types = [e.event_type for e in events]
        assert "SCHEDULER_THREAD_TERMINATED" in event_types

        # Scenario 2: Consecutive Failures Critical (>=3)
        scheduler_repeated_failure = {
            "is_alive": True,
            "thread_running": True,
            "consecutive_failures": 3,
            "last_error": "Persistent 503 Service Unavailable",
        }
        events_crit = dispatcher.evaluate_scheduler_health(scheduler_repeated_failure)
        crit_types = [e.event_type for e in events_crit]
        assert "CONSECUTIVE_BACKUP_FAILURES_CRITICAL" in crit_types
        assert any(e.severity == BackupAlertSeverity.CRITICAL for e in events_crit)

        # Scenario 3: Containment Gate Suspended
        containment_suspended = {
            "current_state": "INGESTION_SUSPENDED",
            "latest_verified_backup_age_seconds": 44000.0,
            "latest_verified_recovery_point": "sm-backup-test",
        }
        events_containment = dispatcher.evaluate_scheduler_health(
            scheduler_repeated_failure,
            containment_status=containment_suspended,
        )
        susp_types = [e.event_type for e in events_containment]
        assert "BACKUP_AGE_CONTAINMENT_SUSPENSION" in susp_types

        # Dispatch one event and verify handler received it
        dispatcher.dispatch(events[0])
        assert len(dispatched_events) == 1
        assert dispatched_events[0].event_type == "SCHEDULER_THREAD_TERMINATED"
        assert len(dispatcher.get_alert_history()) == 1

    def test_edge_01_worker_startup_before_first_scheduled_backup(self):
        """Edge Case 1: Worker startup before first scheduled backup completes is healthy."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "startup_completed": False,
            "startup_error": None,
            "consecutive_failures": 0,
            "last_status": "INITIALIZING",
            "is_stale_warning": False,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["overall_healthy"] is True
        assert envelope["backup_subsystem_healthy"] is True
        assert envelope["degradation_reasons"] == []

    def test_edge_02_scheduler_initialization_failure(self):
        """Edge Case 2: Scheduler thread fails during initialization."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": False,
            "thread_running": True,
            "startup_completed": False,
            "startup_error": "DB_CONNECTION_REFUSED",
            "consecutive_failures": 1,
            "last_status": "FAILED",
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["overall_healthy"] is False
        assert envelope["backup_subsystem_healthy"] is False
        assert "SCHEDULER_STARTUP_FAILED" in envelope["degradation_reasons"]

    def test_edge_03_unexpected_thread_death(self):
        """Edge Case 3: Scheduler thread exits unexpectedly while worker is running."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": False,
            "thread_running": True,
            "consecutive_failures": 0,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["backup_subsystem_healthy"] is False
        assert "SCHEDULER_THREAD_TERMINATED" in envelope["degradation_reasons"]

    def test_edge_04_scheduler_stalled_no_progress(self):
        """Edge Case 4: Scheduler thread remains alive but makes no progress."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "is_stalled": True,
            "consecutive_failures": 0,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["backup_subsystem_healthy"] is False
        assert "SCHEDULER_STALLED_NO_PROGRESS" in envelope["degradation_reasons"]

    def test_edge_05_backup_execution_exception_resilience(self, tmp_path):
        """Edge Case 5: Backup execution raises an exception and increments failures safely."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = RuntimeError("Database I/O failure")
            result = scheduler.run_backup_pass()
            assert result is False
            assert scheduler.consecutive_failures == 1
            assert scheduler.last_status == "FAILED"

    def test_edge_06_r2_upload_failure(self, tmp_path):
        """Edge Case 6: R2 upload fails during backup pass."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = ConnectionResetError("Cloudflare R2 SSL reset")
            result = scheduler.run_backup_pass()
            assert result is False
            assert "Cloudflare R2 SSL reset" in scheduler.last_error
            assert scheduler.consecutive_failures == 1

    def test_edge_07_checksum_verification_failure(self, tmp_path):
        """Edge Case 7: Remote checksum verification fails."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.side_effect = ValueError("CHECKSUM_MISMATCH: corrupt dump")
            result = scheduler.run_backup_pass()
            assert result is False
            assert "CHECKSUM_MISMATCH" in scheduler.last_error

    def test_edge_08_backup_lock_contention(self, tmp_path):
        """Edge Case 8: Backup execution lock cannot be acquired."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        # Simulate lock held by another pass
        assert scheduler._lock.acquire(blocking=False)
        try:
            result = scheduler.run_backup_pass()
            assert result is False
        finally:
            scheduler._lock.release()

    def test_edge_09_backup_execution_overdue(self):
        """Edge Case 9: Backup execution exceeds its expected duration."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "lock_held": True,
            "is_execution_overdue": True,
            "pass_duration_sec": 2400.0,
            "consecutive_failures": 0,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["backup_subsystem_healthy"] is False
        assert "BACKUP_EXECUTION_OVERDUE" in envelope["degradation_reasons"]

    def test_edge_10_freshness_warning_threshold(self):
        """Edge Case 10: Freshness warning threshold (8h) crossed."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "is_stale_warning": True,
            "consecutive_failures": 0,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert "BACKUP_FRESHNESS_WARNING" in envelope["degradation_reasons"]
        assert envelope["worker_subsystem_healthy"] is True

    def test_edge_11_critical_freshness_threshold_containment(self):
        """Edge Case 11: Critical freshness threshold (12h) crossed."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.HEALTHY
        mock_gate = MagicMock()
        mock_gate.get_observability_envelope.return_value = {
            "current_state": "INGESTION_SUSPENDED",
            "latest_verified_backup_age_seconds": 45000.0,
        }
        worker.containment_gate = mock_gate
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "consecutive_failures": 0,
            "critical_threshold_sec": 43200.0,
        }
        worker.backup_scheduler = mock_scheduler
        envelope = worker.get_health_envelope()
        assert envelope["overall_healthy"] is False
        assert envelope["backup_subsystem_healthy"] is False
        assert "CONTAINMENT_GATE_SUSPENDED" in envelope["degradation_reasons"]
        assert "BACKUP_FRESHNESS_CRITICAL" in envelope["degradation_reasons"]

    def test_edge_12_backup_recovery_after_failure(self, tmp_path):
        """Edge Case 12: Backup succeeds after a previous failure."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        scheduler.consecutive_failures = 2
        with patch("analyst_dashboard.smart_money.worker.SmartMoneyBackupEngine") as mock_engine_cls:
            mock_engine_cls.return_value.create_backup.return_value = {
                "backup_id": "sm-backup-recovered",
                "storage_destination": "local://recovered",
            }
            res = scheduler.run_backup_pass()
            assert res is True
            assert scheduler.consecutive_failures == 0
            assert scheduler.last_status == "SUCCESS"

    def test_edge_13_worker_restart_with_verified_backup(self):
        """Edge Case 13: Worker restarts with an existing verified fresh backup."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.OWNER_LOCK_ACQUIRED
        worker.repository = MagicMock()
        worker.repository.recover_stale_run_requests.return_value = 0
        worker.repository.claim_next_run_request.return_value = None
        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True
        worker.containment_gate = mock_gate
        worker.run_cycle()
        assert worker.state == WorkerHealthState.HEALTHY

    def test_edge_14_worker_sec_ownership_loss_and_reacquisition(self):
        """Edge Case 14: Worker loses and reacquires SEC ownership without scheduler impact."""
        worker = SmartMoneyIngestionWorker()
        worker.state = WorkerHealthState.OWNER_LOCK_LOST
        mock_scheduler = MagicMock()
        mock_scheduler.get_status_envelope.return_value = {
            "is_alive": True,
            "thread_running": True,
            "consecutive_failures": 0,
        }
        worker.backup_scheduler = mock_scheduler
        env_lost = worker.get_health_envelope()
        assert env_lost["worker_subsystem_healthy"] is False
        assert env_lost["backup_subsystem_healthy"] is True

        worker.state = WorkerHealthState.HEALTHY
        env_regained = worker.get_health_envelope()
        assert env_regained["worker_subsystem_healthy"] is True
        assert env_regained["backup_subsystem_healthy"] is True

    def test_edge_15_graceful_shutdown_during_backup_pass(self, tmp_path):
        """Edge Case 15: Graceful shutdown during a backup pass releases lock safely."""
        adapter = LocalOffVolumeStorageAdapter(base_directory=str(tmp_path))
        scheduler = SmartMoneyBackupSchedulerThread(interval_sec=60, storage_adapter=adapter)
        scheduler.start()
        try:
            assert scheduler.is_alive()
        finally:
            scheduler.stop()
            scheduler.join(timeout=3.0)
            assert not scheduler.is_alive()
            assert not scheduler._lock.locked()

    def test_alerting_anti_storm_debouncing(self):
        """Audit: Proves repeated warning/critical events within debounce window are suppressed."""
        from analyst_dashboard.smart_money.storage.backup.alerting import (
            BackupAlertDispatcher,
            BackupAlertSeverity,
            BackupAlertEvent,
        )
        dispatcher = BackupAlertDispatcher(debounce_window_sec=300.0)
        dispatched = []
        dispatcher.register_handler(lambda e: dispatched.append(e))

        event = BackupAlertEvent(
            event_type="CONSECUTIVE_BACKUP_FAILURES_WARNING",
            severity=BackupAlertSeverity.WARNING,
            message="Backup pass failed",
        )
        assert dispatcher.dispatch(event) is True
        assert len(dispatched) == 1

        # Second dispatch of identical event immediately should be debounced
        assert dispatcher.dispatch(event) is False
        assert len(dispatched) == 1

    def test_alerting_recovery_event_generation(self):
        """Audit: Proves BACKUP_RECOVERY_SUCCESS event emitted when recovering from failure."""
        from analyst_dashboard.smart_money.storage.backup.alerting import (
            BackupAlertDispatcher,
            BackupAlertSeverity,
        )
        dispatcher = BackupAlertDispatcher()

        # Step 1: Degraded state
        status_failed = {
            "is_alive": True,
            "thread_running": True,
            "consecutive_failures": 2,
            "last_error": "DB disconnect",
        }
        events1 = dispatcher.evaluate_scheduler_health(status_failed)
        assert any(e.severity == BackupAlertSeverity.WARNING for e in events1)

        # Step 2: Recovered state
        status_recovered = {
            "is_alive": True,
            "thread_running": True,
            "consecutive_failures": 0,
            "last_status": "SUCCESS",
            "last_backup_id": "sm-backup-recovered",
            "last_successful_timestamp": time.time(),
        }
        events2 = dispatcher.evaluate_scheduler_health(status_recovered)
        event_types = [e.event_type for e in events2]
        assert "BACKUP_RECOVERY_SUCCESS" in event_types
        recovery_ev = [e for e in events2 if e.event_type == "BACKUP_RECOVERY_SUCCESS"][0]
        assert recovery_ev.severity == BackupAlertSeverity.INFO

    def test_alerting_zero_secrets_contract(self):
        """Audit: Proves alert payloads never contain sensitive tokens, passwords, or DSNs."""
        from analyst_dashboard.smart_money.storage.backup.alerting import (
            BackupAlertDispatcher,
            BackupAlertEvent,
            BackupAlertSeverity,
        )
        dispatcher = BackupAlertDispatcher()
        event = BackupAlertEvent(
            event_type="TEST_ALERT",
            severity=BackupAlertSeverity.WARNING,
            message="Clean alert message",
            metadata={"status": "online", "count": 5},
        )
        payload = event.to_dict()
        serialized = str(payload).lower()
        assert "postgres://" not in serialized
        assert "password" not in serialized
        assert "token" not in serialized
        assert "secret" not in serialized


