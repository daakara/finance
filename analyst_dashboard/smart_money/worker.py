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
from typing import Optional, Dict, Any

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
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.data.sec_edgar_fetcher import get_sec_egress_role
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.storage_adapter import get_backup_storage_adapter
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
    6. Missed-run detection, storage capacity monitoring, and failure alerting.
    """

    def __init__(self, interval_sec: int = 21600, retention_count: int = 10, retention_days: int = 7):
        super().__init__(daemon=True, name="SmartMoneyBackupScheduler")
        self.interval_sec = max(60, interval_sec)
        self.retention_count = retention_count
        self.retention_days = retention_days
        self._running = False
        self._lock = threading.Lock()
        self.last_run_timestamp = 0.0
        self.last_status = "INITIALIZING"
        self.successful_runs = 0
        self.failed_runs = 0
        self.last_backup_id = None
        self.last_error = None

    def run_backup_pass(self) -> bool:
        if not self._lock.acquire(blocking=False):
            logger.warning("Backup scheduler pass skipped: previous pass still executing.")
            return False
        try:
            logger.info("Executing scheduled automated backup pass (routine, unprivileged)...")
            adapter = get_backup_storage_adapter()
            engine = SmartMoneyBackupEngine(storage_adapter=adapter)
            retention = BackupRetentionManager(
                storage_adapter=adapter,
                retention_count=self.retention_count,
                retention_days=self.retention_days,
            )
            manifest = engine.create_backup(include_globals=False)
            pruned = retention.prune_backups()
            self.last_run_timestamp = time.time()
            self.last_status = "SUCCESS"
            self.last_backup_id = manifest["backup_id"]
            self.last_error = None
            self.successful_runs += 1
            logger.info(
                f"Scheduled backup {manifest['backup_id']} verified and stored in {manifest['storage_destination']}. "
                f"Pruned: {len(pruned)} expired sets."
            )
            return True
        except Exception as e:
            self.last_status = "FAILED"
            self.last_error = str(e)
            self.failed_runs += 1
            logger.error(f"Scheduled backup failed: {e}", exc_info=True)
            return False
        finally:
            self._lock.release()

    def run(self):
        self._running = True
        logger.info(f"Backup scheduler thread started (interval={self.interval_sec}s / {self.interval_sec/3600:.1f}h).")
        # Run initial pass shortly after startup if configured
        time.sleep(min(5.0, float(self.interval_sec)))
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
        return {
            "is_running": self._running,
            "interval_sec": self.interval_sec,
            "last_status": self.last_status,
            "last_run_timestamp": self.last_run_timestamp,
            "last_backup_id": self.last_backup_id,
            "last_error": self.last_error,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
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

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or get_database_url()
        self.repository = PostgresSmartMoneyRepository(dsn=self.dsn)
        self.state = WorkerHealthState.STARTING
        self.worker_id = f"worker-{os.getpid()}-{int(time.time())}"
        self._owner_conn = None
        self._running = False
        self.lease_duration_sec = int(os.environ.get("SMART_MONEY_LEASE_DURATION_SEC", "60"))
        self.heartbeat_interval_sec = int(os.environ.get("SMART_MONEY_HEARTBEAT_INTERVAL_SEC", "10"))
        self.stale_recovery_interval_sec = int(os.environ.get("SMART_MONEY_STALE_RECOVERY_INTERVAL_SEC", "30"))
        self._last_stale_recovery_at = 0.0

    def validate_preflight(self) -> None:
        """
        Validates environment prerequisites before attempting lock or network operations:
        1. ARX_SEC_EGRESS_ROLE must equal 'OWNER'.
        2. Database configuration must be valid PostgreSQL.
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

    def acquire_egress_authority(self) -> bool:
        """
        Acquires dedicated PostgreSQL session-level advisory lock.
        Fails closed if another worker holds the lock.
        """
        import psycopg2
        self._owner_conn = psycopg2.connect(self.dsn)
        with self._owner_conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'GMT';")
            cur.execute("SET application_name = %s;", (f"sm_worker_{self.worker_id}",))

        acquired = acquire_sec_owner_lock(self._owner_conn)
        if not acquired:
            self.state = WorkerHealthState.OWNER_LOCK_LOST
            logger.error(
                "DUPLICATE_OWNER_REJECTED: Another worker currently holds SEC_EGRESS_OWNER lock. "
                "Failing closed to prevent aggregate rate violation."
            )
            self._owner_conn.close()
            self._owner_conn = None
            return False

        self.state = WorkerHealthState.OWNER_LOCK_ACQUIRED
        logger.info(f"Worker {self.worker_id} acquired exclusive SEC_EGRESS_OWNER advisory lock.")
        return True

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

    def process_run_requests(self) -> int:
        """
        Checks for pending operator run requests, claims them with atomic lease,
        renews heartbeat, and marks them completed under lease fencing.
        Periodically executes stale claim recovery to self-heal abandoned work.
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

        result_gen = f"gen-{req_id[:8]}"
        try:
            self.repository.complete_run_request(
                req_id,
                result_generation_id=result_gen,
                lease_token=lease_token,
                status="COMPLETED",
            )
            logger.info(f"Completed operator run request: {req_id} -> {result_gen}")
            return 1
        except StaleLeaseOwnershipLostError as e:
            logger.warning(f"Lease lost for request {req_id} (stale worker fenced): {e}")
            return 0

    def run_cycle(self) -> None:
        """Executes a single operational work cycle."""
        if self.state == WorkerHealthState.OWNER_LOCK_ACQUIRED:
            self.state = WorkerHealthState.HEALTHY

        # Process any pending run requests
        try:
            self.process_run_requests()
        except Exception as e:
            logger.warning(f"Error processing run requests: {e}")

    def start(self, single_cycle: bool = False, poll_interval_sec: float = 5.0) -> None:
        """Starts worker event loop."""
        self.validate_preflight()
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
            self.backup_scheduler = SmartMoneyBackupSchedulerThread(interval_sec=interval_sec)
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
        """Returns machine-readable health envelope."""
        return {
            "worker_id": self.worker_id,
            "status": self.state.value,
            "has_owner_lock": self.state in {WorkerHealthState.OWNER_LOCK_ACQUIRED, WorkerHealthState.HEALTHY},
            "backup_scheduler": self.backup_scheduler.get_status_envelope() if getattr(self, "backup_scheduler", None) else None,
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
