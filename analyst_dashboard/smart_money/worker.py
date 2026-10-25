"""
ARX Terminal — Dedicated Smart Money SEC Egress & Ingestion Worker.

Production entrypoint for the sole authorized live SEC EDGAR egress worker.
Enforces:
1. ARX_SEC_EGRESS_ROLE=OWNER environment configuration;
2. Exclusive PostgreSQL session-level advisory lock (SEC_EGRESS_OWNER);
3. Observable machine-readable health states;
4. Discovery, ingestion, reconciliation, and snapshot publication cycles;
5. Background processing of operator run requests.
"""

import os
import sys
import time
import signal
import logging
from enum import Enum
from typing import Optional, Dict, Any, List

from analyst_dashboard.smart_money.storage.config import (
    get_database_url,
    get_database_role,
    ROLE_INGESTION,
    validate_storage_configuration,
)
from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    LOCK_SEC_EGRESS_OWNER,
    LOCK_BACKUP_EXECUTION,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.data.sec_edgar_fetcher import get_sec_egress_role
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    get_backup_storage_adapter,
    BackupStorageAdapter,
)
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
)
from analyst_dashboard.smart_money.continuous_pipeline import SmartMoneyContinuousPipeline
import threading

logger = logging.getLogger("smart_money.worker")


class SmartMoneyBackupSchedulerThread(threading.Thread):
    """
    Dedicated background thread executing periodic PostgreSQL logical backups.
    Enforces:
    1. Complete isolation from SEC ingestion cycles;
    2. Mutual exclusion via session-level LOCK_BACKUP_EXECUTION advisory lock;
    3. Configurable schedule interval (default: 21600s / 6 hours);
    4. Routine backups run with unprivileged role (include_globals=False);
    5. Automatic safe retention pruning protecting the latest verified recovery point;
    6. Missed-run detection, storage capacity monitoring, and failure alerting;
    7. Instant containment gate recheck upon successful backup completion.
    """

    def __init__(
        self,
        interval_sec: int = 21600,
        retention_count: int = 10,
        retention_days: int = 7,
        containment_gate: Optional[BackupAgeContainmentGate] = None,
        storage_adapter: Optional[BackupStorageAdapter] = None,
        warning_threshold_sec: Optional[float] = None,
        critical_threshold_sec: Optional[float] = None,
        max_execution_duration_sec: Optional[float] = None,
    ):
        super().__init__(daemon=True, name="SmartMoneyBackupScheduler")
        self.interval_sec = max(60, interval_sec)
        self.retention_count = retention_count
        self.retention_days = retention_days
        self.containment_gate = containment_gate
        self.storage_adapter = storage_adapter
        self.warning_threshold_sec = warning_threshold_sec or float(os.environ.get("ARX_BACKUP_WARNING_AGE_SEC", "28800.0"))
        self.critical_threshold_sec = critical_threshold_sec or float(os.environ.get("ARX_BACKUP_CRITICAL_AGE_SEC", "43200.0"))
        self.max_execution_duration_sec = max_execution_duration_sec or float(os.environ.get("ARX_BACKUP_MAX_DURATION_SEC", "1800.0"))
        self._running = False
        self._lock = threading.Lock()
        self.thread_started_at: Optional[float] = None
        self.last_attempt_timestamp: Optional[float] = None
        self.last_run_timestamp = 0.0
        self.last_successful_timestamp: Optional[float] = None
        self.last_status = "INITIALIZING"
        self.successful_runs = 0
        self.failed_runs = 0
        self.consecutive_failures = 0
        self.startup_completed = False
        self.startup_error: Optional[str] = None
        self.lock_held = False
        self.lock_key = LOCK_BACKUP_EXECUTION
        self.last_backup_id = None
        self.last_error = None

    def run_backup_pass(self) -> bool:
        if not self._lock.acquire(blocking=False):
            logger.warning("Backup scheduler pass skipped: previous pass still executing.")
            return False
        self.lock_held = True
        self.last_attempt_timestamp = time.time()
        if self.containment_gate:
            self.containment_gate.mark_backup_in_progress()
        try:
            logger.info("Executing scheduled automated backup pass (routine, unprivileged)...")
            adapter = self.storage_adapter or get_backup_storage_adapter()
            engine = SmartMoneyBackupEngine(storage_adapter=adapter)
            retention = BackupRetentionManager(
                storage_adapter=adapter,
                retention_count=self.retention_count,
                retention_days=self.retention_days,
            )
            manifest = engine.create_backup(include_globals=False)
            pruned = retention.prune_backups()
            now = time.time()
            self.last_run_timestamp = now
            self.last_successful_timestamp = now
            self.last_status = "SUCCESS"
            self.last_backup_id = manifest["backup_id"]
            self.last_error = None
            self.successful_runs += 1
            self.consecutive_failures = 0
            self.startup_completed = True
            logger.info(
                f"Scheduled backup {manifest['backup_id']} verified and stored in {manifest['storage_destination']}. "
                f"Pruned: {len(pruned)} expired sets."
            )
            # Instantly recheck freshness to restore ingestion eligibility
            if self.containment_gate:
                self.containment_gate.check_ingestion_eligibility(force_refresh=True)
            return True
        except Exception as e:
            self.last_status = "FAILED"
            self.last_error = str(e)
            self.failed_runs += 1
            self.consecutive_failures += 1
            if not self.startup_completed:
                self.startup_error = str(e)
            logger.error(f"Scheduled backup failed: {e}", exc_info=True)
            return False
        finally:
            self.lock_held = False
            if self.containment_gate:
                self.containment_gate.mark_backup_completed()
            self._lock.release()

    def run(self):
        self._running = True
        self.thread_started_at = time.time()
        logger.info(f"Backup scheduler thread started (interval={self.interval_sec}s / {self.interval_sec/3600:.1f}h).")
        # Run initial pass shortly after startup if configured (responsive to stop)
        elapsed_startup = 0.0
        startup_wait = min(5.0, float(self.interval_sec))
        while self._running and elapsed_startup < startup_wait:
            time.sleep(0.1)
            elapsed_startup += 0.1
        while self._running:
            self.run_backup_pass()
            elapsed = 0
            while self._running and elapsed < self.interval_sec:
                time.sleep(min(1.0, float(self.interval_sec - elapsed)))
                elapsed += 1
        logger.info("Backup scheduler thread stopped.")

    def stop(self):
        self._running = False

    def get_status_envelope(self) -> Dict[str, Any]:
        thread_alive = bool(self.is_alive())
        is_running = bool(self._running)
        is_alive_and_running = bool(thread_alive and is_running)
        now = time.time()
        age_since_last_success = (now - self.last_successful_timestamp) if self.last_successful_timestamp else None
        is_stale_warning = bool(age_since_last_success and age_since_last_success >= self.warning_threshold_sec)
        is_stale_critical = bool(age_since_last_success and age_since_last_success >= self.critical_threshold_sec)

        pass_duration_sec = (now - self.last_attempt_timestamp) if (self.lock_held and self.last_attempt_timestamp) else 0.0
        is_execution_overdue = bool(self.lock_held and pass_duration_sec > self.max_execution_duration_sec)

        time_since_last_attempt = (now - self.last_attempt_timestamp) if self.last_attempt_timestamp else None
        is_stalled = bool(
            self.startup_completed
            and thread_alive
            and is_running
            and not self.lock_held
            and time_since_last_attempt
            and (time_since_last_attempt > (self.interval_sec * 1.5))
        )

        return {
            "is_alive": is_alive_and_running,
            "thread_alive": thread_alive,
            "is_running": is_running,
            "thread_running": is_running,
            "thread_name": self.name,
            "thread_started_at": self.thread_started_at,
            "interval_sec": self.interval_sec,
            "warning_threshold_sec": self.warning_threshold_sec,
            "critical_threshold_sec": self.critical_threshold_sec,
            "max_execution_duration_sec": self.max_execution_duration_sec,
            "last_status": self.last_status,
            "last_attempt_timestamp": self.last_attempt_timestamp,
            "last_run_timestamp": self.last_run_timestamp,
            "last_successful_timestamp": self.last_successful_timestamp,
            "age_since_last_success_sec": age_since_last_success,
            "is_stale_warning": is_stale_warning,
            "is_stale_critical": is_stale_critical,
            "consecutive_failures": self.consecutive_failures,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "startup_completed": self.startup_completed,
            "startup_error": self.startup_error,
            "lock_held": self.lock_held,
            "lock_key": self.lock_key,
            "pass_duration_sec": pass_duration_sec,
            "is_execution_overdue": is_execution_overdue,
            "is_stalled": is_stalled,
            "last_backup_id": self.last_backup_id,
            "last_error": self.last_error,
        }


class WorkerHealthState(Enum):
    STARTING = "STARTING"
    OWNER_LOCK_ACQUIRED = "OWNER_LOCK_ACQUIRED"
    HEALTHY = "HEALTHY"
    SOURCE_DEGRADED = "SOURCE_DEGRADED"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    BLOCKED_DATA_INTEGRITY = "BLOCKED_DATA_INTEGRITY"
    OWNER_LOCK_LOST = "OWNER_LOCK_LOST"
    FATAL = "FATAL"


class SmartMoneyIngestionWorker:
    """Dedicated single-active SEC egress worker backed by PostgreSQL."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        containment_gate: Optional[BackupAgeContainmentGate] = None,
        containment_policy: Optional[BackupContainmentPolicy] = None,
    ):
        self.dsn = dsn or get_database_url()
        self.containment_policy = containment_policy or BackupContainmentPolicy.from_environment()
        self.freshness_authority = BackupFreshnessAuthority(
            dsn=self.dsn,
            policy=self.containment_policy,
        )
        self.containment_gate = containment_gate or BackupAgeContainmentGate(
            freshness_authority=self.freshness_authority,
            policy=self.containment_policy,
        )
        self.repository = PostgresSmartMoneyRepository(
            dsn=self.dsn,
            containment_gate=self.containment_gate,
        )
        self.state = WorkerHealthState.STARTING
        self.worker_id = f"worker-{os.getpid()}-{int(time.time())}"
        self._owner_conn = None
        self._running = False
        self.lease_duration_sec = int(os.environ.get("SMART_MONEY_LEASE_DURATION_SEC", "60"))
        self.heartbeat_interval_sec = int(os.environ.get("SMART_MONEY_HEARTBEAT_INTERVAL_SEC", "10"))
        self.stale_recovery_interval_sec = int(os.environ.get("SMART_MONEY_STALE_RECOVERY_INTERVAL_SEC", "30"))
        self._last_stale_recovery_at = 0.0
        self._is_executing_cycle = False
        self._shutdown_requested = False

    def validate_preflight(self) -> None:
        """
        Validates environment prerequisites before attempting lock or network operations:
        1. ARX_SEC_EGRESS_ROLE must equal 'OWNER'.
        2. Database configuration must be valid PostgreSQL.
        3. Containment policy configuration authorization in production.
        """
        role = get_sec_egress_role()
        if role != "OWNER":
            self.state = WorkerHealthState.FATAL
            raise RuntimeError(
                f"UNAUTHORIZED_WORKER_ROLE: Worker requires ARX_SEC_EGRESS_ROLE=OWNER. "
                f"Active role is '{role}'."
            )

        validate_storage_configuration()
        if not self.dsn:
            self.state = WorkerHealthState.FATAL
            raise RuntimeError("DATABASE_URL_REQUIRED: PostgreSQL connection string is required.")

        # Containment authorization check in production environment
        if self.containment_policy.enabled:
            is_prod = os.environ.get("ARX_ENVIRONMENT", "").strip().lower() == "production"
            if is_prod and not self.containment_policy.production_authorized:
                self.state = WorkerHealthState.FATAL
                raise RuntimeError(
                    "UNAUTHORIZED_PRODUCTION_BACKUP_CONTAINMENT: Policy A activation in production "
                    "requires explicit ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=True ratification."
                )

    def acquire_egress_authority(self, max_attempts: int = 5, retry_interval_sec: float = 2.0) -> bool:
        """
        Acquires dedicated PostgreSQL session-level advisory lock.
        Permits short retry window for deployment handover (SIGTERM / container swap).
        Fails closed if another worker holds the lock after retries.
        """
        import psycopg2
        self._owner_conn = psycopg2.connect(self.dsn)
        with self._owner_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'GMT';")
            cur.execute("SET application_name = %s;", (f"sm_worker_{self.worker_id}",))

        for attempt in range(1, max_attempts + 1):
            acquired = acquire_sec_owner_lock(self._owner_conn)
            if acquired:
                self.state = WorkerHealthState.OWNER_LOCK_ACQUIRED
                logger.info(f"Worker {self.worker_id} acquired exclusive SEC_EGRESS_OWNER advisory lock.")
                return True
            if attempt < max_attempts:
                logger.warning(
                    f"SEC_EGRESS_OWNER lock contention (attempt {attempt}/{max_attempts}). "
                    f"Waiting {retry_interval_sec}s for outgoing worker handover..."
                )
                time.sleep(retry_interval_sec)

        self.state = WorkerHealthState.OWNER_LOCK_LOST
        logger.error(
            "DUPLICATE_OWNER_REJECTED: Another worker currently holds SEC_EGRESS_OWNER lock. "
            "Failing closed to prevent aggregate rate violation."
        )
        self._owner_conn.close()
        self._owner_conn = None
        return False

    def release_egress_authority(self) -> None:
        """Releases session advisory lock on shutdown."""
        if self._owner_conn and not self._owner_conn.closed:
            try:
                release_sec_owner_lock(self._owner_conn)
            except Exception as e:
                logger.warning(f"Error releasing SEC owner lock: {e}")
            finally:
                self._owner_conn.close()
                self._owner_conn = None
        self.state = WorkerHealthState.OWNER_LOCK_LOST

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """
        Control B: Catches SIGTERM / SIGINT and drops session advisory lock (<= 1.0s).
        If a work cycle is currently in-flight, defers lock release until cycle completion
        to prevent dropping the session advisory lock underneath active state modification.
        """
        logger.info(
            f"Worker {self.worker_id} received shutdown signal {signum}."
        )
        self._running = False
        if getattr(self, "_is_executing_cycle", False):
            logger.warning(
                f"Worker {self.worker_id} is currently executing cycle; deferring lock release until cycle completion."
            )
            self._shutdown_requested = True
            return

        logger.info("Executing immediate Control B teardown (dropping advisory lock <= 1.0s)...")
        self.release_egress_authority()
        if getattr(self, "backup_scheduler", None):
            self.backup_scheduler.stop()
        sys.exit(0)

    def _setup_signals(self) -> None:
        """Registers signal handlers for graceful shutdown (Control B)."""
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        except (ValueError, AttributeError):
            pass

    def process_run_requests(self) -> int:
        """
        Checks for pending operator run requests, claims them with atomic lease,
        renews heartbeat, and marks them completed under lease fencing.
        Periodically executes stale claim recovery to self-heal abandoned work.
        Enforces backup freshness containment gate fail-closed.
        Returns the number of processed requests.
        """
        now = time.time()
        if now - self._last_stale_recovery_at >= self.stale_recovery_interval_sec:
            try:
                recovered = self.repository.recover_stale_run_requests()
                self._last_stale_recovery_at = now
                if recovered > 0:
                    logger.info(f"Self-healed {recovered} stale run request(s).")
            except Exception as e:
                logger.warning(f"Error during periodic stale request recovery: {e}")

        # Gate check before claiming request
        if not self.containment_gate.check_ingestion_eligibility():
            return 0

        claimed = self.repository.claim_next_run_request(
            self.worker_id, lease_duration_sec=self.lease_duration_sec
        )
        if not claimed:
            return 0

        req_id = claimed["request_id"]
        lease_token = claimed["lease_token"]
        logger.info(f"Claimed operator run request: {req_id} (token={lease_token})")

        # Heartbeat renewal during processing
        self.repository.heartbeat_run_request(req_id, lease_token, extension_sec=self.lease_duration_sec)

        # Gate check before completing request (prevent in-flight threshold breach)
        if not self.containment_gate.check_ingestion_eligibility():
            logger.warning(f"Aborting completion of request {req_id}: backup became stale during execution.")
            return 0

        self._is_executing_cycle = True
        try:
            pipeline = SmartMoneyContinuousPipeline(
                dsn=self.dsn,
                containment_gate=self.containment_gate,
                repository=self.repository,
                allow_test_mode=False,
            )
            res = pipeline.execute_cycle(
                run_request_id=req_id,
                lease_token=lease_token,
                trigger_type="OPERATOR",
            )
            logger.info(f"Completed operator run request: {req_id} -> snapshot {res.get('snapshot_id')}")
            return 1
        except StaleLeaseOwnershipLostError as e:
            logger.warning(f"Lease lost for request {req_id} (stale worker fenced): {e}")
            return 0
        except Exception as e:
            logger.error(f"Failed processing operator run request {req_id}: {e}")
            try:
                self.repository.complete_run_request(
                    req_id,
                    lease_token=lease_token,
                    status="FAILED",
                    failure_code=str(type(e).__name__),
                )
            except Exception as fail_err:
                logger.warning(f"Could not record failure status for request {req_id}: {fail_err}")
            return 0
        finally:
            self._is_executing_cycle = False
            if getattr(self, "_shutdown_requested", False):
                logger.info(
                    f"Worker {self.worker_id} completed in-flight cycle; executing deferred shutdown."
                )
                self.release_egress_authority()
                if getattr(self, "backup_scheduler", None):
                    self.backup_scheduler.stop()
                sys.exit(0)

    def run_cycle(self) -> None:
        """Executes a single operational work cycle."""
        # 1. Evaluate Backup Freshness Containment Gate
        if not self.containment_gate.check_ingestion_eligibility():
            self.state = WorkerHealthState.BLOCKED_DATA_INTEGRITY
            logger.warning(
                f"Work cycle suspended: Backup age containment gate closed "
                f"(status={self.containment_gate.current_state.value}, reason={self.containment_gate.last_error_code})."
            )
            return

        if self.state in (WorkerHealthState.OWNER_LOCK_ACQUIRED, WorkerHealthState.BLOCKED_DATA_INTEGRITY):
            self.state = WorkerHealthState.HEALTHY

        # Process any pending run requests
        try:
            self.process_run_requests()
        except Exception as e:
            logger.warning(f"Error processing run requests: {e}")

    def start(self, single_cycle: bool = False, poll_interval_sec: float = 5.0) -> None:
        """Starts worker event loop."""
        self.validate_preflight()
        self._setup_signals()
        if not self.acquire_egress_authority():
            sys.exit(1)

        # Startup recovery of abandoned/stale requests
        try:
            startup_recovered = self.repository.recover_stale_run_requests()
            if startup_recovered > 0:
                logger.info(f"Startup recovery: self-healed {startup_recovered} abandoned run request(s).")
        except Exception as e:
            logger.warning(f"Error during startup stale request recovery: {e}")

        self._running = True

        # Initialize automated scheduled backup thread if enabled
        self.backup_scheduler = None
        if os.environ.get("ARX_BACKUP_ENABLED", "").strip().lower() in ("true", "1", "yes"):
            interval_sec = int(os.environ.get("ARX_BACKUP_INTERVAL_SEC", "21600"))
            self.backup_scheduler = SmartMoneyBackupSchedulerThread(
                interval_sec=interval_sec,
                containment_gate=self.containment_gate,
            )
            self.backup_scheduler.start()

        def _sig_handler(signum, frame):
            logger.info(f"Received signal {signum}. Terminating worker gracefully...")
            self._running = False

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        logger.info(f"Worker {self.worker_id} is running with status {self.state.value}.")

        try:
            if single_cycle:
                self.run_cycle()
            else:
                while self._running:
                    self.run_cycle()
                    time.sleep(poll_interval_sec)
        finally:
            if self.backup_scheduler:
                self.backup_scheduler.stop()
            self.release_egress_authority()
            logger.info(f"Worker {self.worker_id} terminated cleanly.")

    def get_health_envelope(self) -> Dict[str, Any]:
        """
        Returns standardized machine-readable composite health envelope.
        Differentiates worker process health from backup subsystem health.
        Zero secrets or connection strings exposed.
        """
        scheduler_envelope = (
            self.backup_scheduler.get_status_envelope()
            if getattr(self, "backup_scheduler", None)
            else None
        )
        containment_envelope = (
            self.containment_gate.get_observability_envelope()
            if getattr(self, "containment_gate", None)
            else None
        )

        worker_subsystem_healthy = self.state in {
            WorkerHealthState.OWNER_LOCK_ACQUIRED,
            WorkerHealthState.HEALTHY,
        }

        degradation_reasons: List[str] = []
        backup_subsystem_healthy = True

        if getattr(self, "backup_scheduler", None) and scheduler_envelope:
            is_running = scheduler_envelope.get("is_running", scheduler_envelope.get("thread_running", False))
            is_alive = scheduler_envelope.get("is_alive", False)
            if not is_alive and is_running:
                backup_subsystem_healthy = False
                degradation_reasons.append("SCHEDULER_THREAD_TERMINATED")
            if scheduler_envelope.get("consecutive_failures", 0) > 0:
                backup_subsystem_healthy = False
                degradation_reasons.append(f"CONSECUTIVE_BACKUP_FAILURES_{scheduler_envelope.get('consecutive_failures')}")
            if scheduler_envelope.get("startup_error"):
                backup_subsystem_healthy = False
                degradation_reasons.append("SCHEDULER_STARTUP_FAILED")
            if scheduler_envelope.get("is_execution_overdue"):
                backup_subsystem_healthy = False
                degradation_reasons.append("BACKUP_EXECUTION_OVERDUE")
            if scheduler_envelope.get("is_stalled"):
                backup_subsystem_healthy = False
                degradation_reasons.append("SCHEDULER_STALLED_NO_PROGRESS")
            if scheduler_envelope.get("is_stale_warning"):
                degradation_reasons.append("BACKUP_FRESHNESS_WARNING")

        if containment_envelope:
            crit_thresh = (
                scheduler_envelope.get("critical_threshold_sec", 43200.0)
                if scheduler_envelope
                else 43200.0
            )
            age = containment_envelope.get("latest_verified_backup_age_seconds")
            if age is not None and age >= crit_thresh:
                backup_subsystem_healthy = False
                degradation_reasons.append("BACKUP_FRESHNESS_CRITICAL")
            if containment_envelope.get("current_state") == "INGESTION_SUSPENDED":
                backup_subsystem_healthy = False
                degradation_reasons.append("CONTAINMENT_GATE_SUSPENDED")

        if not worker_subsystem_healthy:
            degradation_reasons.append(f"WORKER_STATE_{self.state.value}")

        overall_healthy = worker_subsystem_healthy and backup_subsystem_healthy

        return {
            "worker_id": self.worker_id,
            "status": self.state.value,
            "overall_healthy": overall_healthy,
            "worker_subsystem_healthy": worker_subsystem_healthy,
            "backup_subsystem_healthy": backup_subsystem_healthy,
            "has_owner_lock": worker_subsystem_healthy,
            "degradation_reasons": degradation_reasons,
            "backup_scheduler": scheduler_envelope,
            "containment": containment_envelope,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main():
    """CLI Entrypoint: python -m analyst_dashboard.smart_money.worker"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    worker = SmartMoneyIngestionWorker()
    try:
        worker.start()
    except Exception as e:
        logger.critical(f"FATAL: Worker terminated with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
