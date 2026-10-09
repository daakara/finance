"""
analyst_dashboard/coordination/store.py

Transactional Storage Engine for Distributed Run Coordination,
Lease Authority, Monotonic Fencing, and Append-Only Event Audits.
"""

import sqlite3
import os
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.data.market_db import retry_sqlite, DATA_DIR
from .contracts import (
    TriggerType,
    JobStatus,
    RunStatus,
    EventType,
    AcquisitionStatus,
    CurrentLease,
    LogicalJob,
    RunAttempt,
    LeaseAcquisitionResult,
    LeasePolicy,
    DEFAULT_COORDINATION_EPOCH,
    RESOURCE_KEY_VCP_PIPELINE,
)

logger = logging.getLogger(__name__)

COORDINATION_DB_PATH = os.path.join(DATA_DIR, ".finance_coordination_store.db")


class CoordinationStore:
    """Production-grade SQLite transactional store for distributed run coordination."""

    def __init__(self, db_path: str = COORDINATION_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    @retry_sqlite()
    def _init_schema(self):
        """Initialize all coordination tables, indexes, and immutability triggers."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_jobs (
                    job_id TEXT PRIMARY KEY,
                    logical_job_key TEXT NOT NULL UNIQUE,
                    resource_key TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    scheduled_for TEXT,
                    operator_request_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    job_status TEXT NOT NULL,
                    successful_publication_id TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_jobs_resource ON radar_jobs(resource_key, created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_jobs_status ON radar_jobs(job_status);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_runs (
                    run_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES radar_jobs(job_id),
                    resource_key TEXT NOT NULL,
                    coordination_epoch TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    owner_instance_id TEXT NOT NULL,
                    implementation_release_sha TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    failure_code TEXT,
                    universe_build_id TEXT,
                    scanner_snapshot_id TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_runs_job ON radar_runs(job_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_runs_resource ON radar_runs(resource_key, started_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_runs_status ON radar_runs(status);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_current_leases (
                    resource_key TEXT PRIMARY KEY,
                    coordination_epoch TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    owner_instance_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    acquired_at TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_coordination_epochs (
                    resource_key TEXT PRIMARY KEY,
                    current_epoch TEXT NOT NULL,
                    epoch_sequence INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_by TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_fencing_sequence (
                    resource_key TEXT NOT NULL,
                    coordination_epoch TEXT NOT NULL,
                    last_token INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (resource_key, coordination_epoch)
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS radar_lease_events (
                    event_id TEXT PRIMARY KEY,
                    resource_key TEXT NOT NULL,
                    coordination_epoch TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    metadata_json TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lease_events_resource ON radar_lease_events(resource_key, observed_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lease_events_run ON radar_lease_events(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lease_events_type ON radar_lease_events(event_type);")

            # Append-only triggers on radar_lease_events
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_radar_lease_events_no_update
                BEFORE UPDATE ON radar_lease_events
                BEGIN
                    SELECT RAISE(ABORT, 'CANONICAL_INVARIANT_VIOLATION: radar_lease_events is append-only');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_radar_lease_events_no_delete
                BEFORE DELETE ON radar_lease_events
                BEGIN
                    SELECT RAISE(ABORT, 'CANONICAL_INVARIANT_VIOLATION: radar_lease_events rows cannot be deleted');
                END;
            """)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def get_db_time_iso(self) -> str:
        """Fetch datastore-authoritative UTC ISO 8601 timestamp."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now') as now_iso;").fetchone()
            return row["now_iso"]
        finally:
            conn.close()

    @retry_sqlite()
    def get_current_epoch(self, resource_key: str) -> str:
        """Fetch the active coordination epoch for a resource key."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT current_epoch FROM radar_coordination_epochs WHERE resource_key = ?;",
                (resource_key,)
            ).fetchone()
            if row:
                return row["current_epoch"]
            return DEFAULT_COORDINATION_EPOCH
        finally:
            conn.close()

    @retry_sqlite()
    def acquire_or_join(
        self,
        resource_key: str,
        logical_job_key: str,
        trigger_type: TriggerType,
        owner_instance_id: str,
        implementation_release_sha: str,
        policy: LeasePolicy,
        scheduled_for: Optional[str] = None,
        operator_request_id: Optional[str] = None,
    ) -> LeaseAcquisitionResult:
        """
        Atomically acquire a lease or reject/deduplicate inside a single immediate transaction.
        Enforces:
        - Job idempotency (AT_MOST_ONCE successful publication per logical job)
        - Monotonic fencing token allocation
        - Authoritative DB expiration derivation
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # 1. Job Deduplication Check
            job_row = conn.execute(
                "SELECT * FROM radar_jobs WHERE logical_job_key = ?;",
                (logical_job_key,)
            ).fetchone()

            if job_row:
                if job_row["job_status"] == JobStatus.SUCCEEDED.value and job_row["successful_publication_id"]:
                    job = LogicalJob(
                        job_id=job_row["job_id"],
                        logical_job_key=job_row["logical_job_key"],
                        resource_key=job_row["resource_key"],
                        trigger_type=TriggerType(job_row["trigger_type"]),
                        scheduled_for=job_row["scheduled_for"],
                        operator_request_id=job_row["operator_request_id"],
                        created_at=job_row["created_at"],
                        job_status=JobStatus(job_row["job_status"]),
                        successful_publication_id=job_row["successful_publication_id"],
                    )
                    # Append suppressed duplicate event
                    event_id = f"evt-{uuid.uuid4().hex[:12]}"
                    conn.execute("""
                        INSERT INTO radar_lease_events (
                            event_id, resource_key, coordination_epoch, lease_id, run_id, job_id, fencing_token, event_type, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        event_id, resource_key, "N/A", "N/A", "N/A", job.job_id, 0,
                        EventType.DUPLICATE_JOB_SUPPRESSED.value,
                        json.dumps({"reason": "JOB_ALREADY_SUCCEEDED", "publication_id": job.successful_publication_id})
                    ))
                    conn.execute("COMMIT;")
                    return LeaseAcquisitionResult(
                        status=AcquisitionStatus.JOB_ALREADY_SUCCEEDED,
                        existing_job=job,
                        message=f"Logical job {logical_job_key} already completed successfully.",
                    )
                else:
                    job_id = job_row["job_id"]
                    conn.execute(
                        "UPDATE radar_jobs SET job_status = 'RUNNING' WHERE job_id = ?;",
                        (job_id,)
                    )
            else:
                job_id = f"job-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO radar_jobs (
                        job_id, logical_job_key, resource_key, trigger_type, scheduled_for, operator_request_id, job_status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING');
                """, (
                    job_id, logical_job_key, resource_key, trigger_type.value, scheduled_for, operator_request_id
                ))

            # 2. Check Active Lease on Resource Key (DB-authoritative time)
            lease_row = conn.execute("""
                SELECT * FROM radar_current_leases
                WHERE resource_key = ?;
            """, (resource_key,)).fetchone()

            is_currently_valid = False
            if lease_row:
                # Check unexpired according to DB time
                exp_check = conn.execute(
                    "SELECT (datetime(?) > datetime('now')) as is_valid;",
                    (lease_row["expires_at"],)
                ).fetchone()
                if exp_check and exp_check["is_valid"]:
                    is_currently_valid = True

            if is_currently_valid:
                # Active unexpired lease held by another worker/run
                active_lease = CurrentLease(
                    resource_key=lease_row["resource_key"],
                    coordination_epoch=lease_row["coordination_epoch"],
                    lease_id=lease_row["lease_id"],
                    run_id=lease_row["run_id"],
                    job_id=lease_row["job_id"],
                    owner_instance_id=lease_row["owner_instance_id"],
                    fencing_token=lease_row["fencing_token"],
                    acquired_at=lease_row["acquired_at"],
                    last_heartbeat_at=lease_row["last_heartbeat_at"],
                    expires_at=lease_row["expires_at"],
                )
                conn.execute("COMMIT;")
                return LeaseAcquisitionResult(
                    status=AcquisitionStatus.ALREADY_RUNNING,
                    lease=active_lease,
                    message=f"Resource {resource_key} is actively held by lease {active_lease.lease_id}.",
                )

            # 3. Acquire Lease (Fresh acquisition or takeover after expiry)
            epoch_row = conn.execute(
                "SELECT current_epoch FROM radar_coordination_epochs WHERE resource_key = ?;",
                (resource_key,)
            ).fetchone()
            current_epoch = epoch_row["current_epoch"] if epoch_row else DEFAULT_COORDINATION_EPOCH

            # Ensure epoch row exists
            if not epoch_row:
                conn.execute("""
                    INSERT OR IGNORE INTO radar_coordination_epochs (
                        resource_key, current_epoch, updated_by, reason
                    ) VALUES (?, ?, 'system_init', 'initial_epoch');
                """, (resource_key, current_epoch))

            # Monotonic fencing token allocation
            token_row = conn.execute("""
                INSERT INTO radar_fencing_sequence (resource_key, coordination_epoch, last_token)
                VALUES (?, ?, 1)
                ON CONFLICT(resource_key, coordination_epoch)
                DO UPDATE SET last_token = last_token + 1
                RETURNING last_token;
            """, (resource_key, current_epoch)).fetchone()
            fencing_token = int(token_row["last_token"])

            lease_id = f"lease-{uuid.uuid4().hex[:12]}"
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            was_takeover = (lease_row is not None)

            # Calculate DB time and expiry
            time_row = conn.execute("""
                SELECT 
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now') as now_iso,
                    strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '+' || ? || ' seconds') as exp_iso;
            """, (policy.lease_ttl_seconds,)).fetchone()
            now_iso = time_row["now_iso"]
            exp_iso = time_row["exp_iso"]

            # Upsert into radar_current_leases
            conn.execute("""
                INSERT INTO radar_current_leases (
                    resource_key, coordination_epoch, lease_id, run_id, job_id,
                    owner_instance_id, fencing_token, acquired_at, last_heartbeat_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_key) DO UPDATE SET
                    coordination_epoch = excluded.coordination_epoch,
                    lease_id = excluded.lease_id,
                    run_id = excluded.run_id,
                    job_id = excluded.job_id,
                    owner_instance_id = excluded.owner_instance_id,
                    fencing_token = excluded.fencing_token,
                    acquired_at = excluded.acquired_at,
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    expires_at = excluded.expires_at;
            """, (
                resource_key, current_epoch, lease_id, run_id, job_id,
                owner_instance_id, fencing_token, now_iso, now_iso, exp_iso
            ))

            # Record in radar_runs
            conn.execute("""
                INSERT INTO radar_runs (
                    run_id, job_id, resource_key, coordination_epoch, lease_id,
                    fencing_token, owner_instance_id, implementation_release_sha, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING');
            """, (
                run_id, job_id, resource_key, current_epoch, lease_id,
                fencing_token, owner_instance_id, implementation_release_sha
            ))

            # Record event in radar_lease_events
            event_type = EventType.TAKEN_OVER_AFTER_EXPIRY if was_takeover else EventType.ACQUIRED
            event_id = f"evt-{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO radar_lease_events (
                    event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                    fencing_token, event_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                event_id, resource_key, current_epoch, lease_id, run_id, job_id,
                fencing_token, event_type.value,
                json.dumps({
                    "owner_instance_id": owner_instance_id,
                    "lease_ttl_seconds": policy.lease_ttl_seconds,
                    "was_takeover": was_takeover,
                })
            ))

            conn.execute("COMMIT;")

            new_lease = CurrentLease(
                resource_key=resource_key,
                coordination_epoch=current_epoch,
                lease_id=lease_id,
                run_id=run_id,
                job_id=job_id,
                owner_instance_id=owner_instance_id,
                fencing_token=fencing_token,
                acquired_at=now_iso,
                last_heartbeat_at=now_iso,
                expires_at=exp_iso,
            )
            return LeaseAcquisitionResult(
                status=AcquisitionStatus.ACQUIRED,
                lease=new_lease,
                message=f"Lease {lease_id} acquired with fencing token {fencing_token}.",
            )
        except Exception as e:
            conn.execute("ROLLBACK;")
            logger.error(f"Error in acquire_or_join for {resource_key}: {e}")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def renew_lease(self, lease: CurrentLease, policy: LeasePolicy) -> bool:
        """
        Transactionally heartbeat / renew lease.
        Fails if expired, superseded, or epoch mismatched (NO LEASE RESURRECTION).
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            cur = conn.execute("""
                UPDATE radar_current_leases
                SET last_heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    expires_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '+' || ? || ' seconds')
                WHERE resource_key = ?
                  AND coordination_epoch = ?
                  AND lease_id = ?
                  AND fencing_token = ?
                  AND datetime(expires_at) > datetime('now');
            """, (
                policy.lease_ttl_seconds,
                lease.resource_key,
                lease.coordination_epoch,
                lease.lease_id,
                lease.fencing_token,
            ))

            if cur.rowcount == 1:
                event_id = f"evt-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO radar_lease_events (
                        event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                        fencing_token, event_type, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    event_id, lease.resource_key, lease.coordination_epoch, lease.lease_id,
                    lease.run_id, lease.job_id, lease.fencing_token,
                    EventType.RENEWED.value,
                    json.dumps({"extended_seconds": policy.lease_ttl_seconds})
                ))
                conn.execute("COMMIT;")
                return True
            else:
                # Renewal failed: lease expired or superseded
                event_id = f"evt-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO radar_lease_events (
                        event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                        fencing_token, event_type, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    event_id, lease.resource_key, lease.coordination_epoch, lease.lease_id,
                    lease.run_id, lease.job_id, lease.fencing_token,
                    EventType.LEASE_LOST.value,
                    json.dumps({"reason": "HEARTBEAT_REJECTED_EXPIRED_OR_SUPERSEDED"})
                ))
                conn.execute("""
                    UPDATE radar_runs
                    SET status = 'ABORTED_LEASE_EXPIRED', completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE run_id = ? AND status = 'RUNNING';
                """, (lease.run_id,))
                conn.execute("COMMIT;")
                return False
        except Exception as e:
            conn.execute("ROLLBACK;")
            logger.warning(f"Lease renewal error for {lease.lease_id}: {e}")
            return False
        finally:
            conn.close()

    @retry_sqlite()
    def release_lease(self, lease: CurrentLease) -> bool:
        """Explicitly release an active lease upon voluntary job conclusion or failure."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("""
                DELETE FROM radar_current_leases
                WHERE resource_key = ?
                  AND coordination_epoch = ?
                  AND lease_id = ?
                  AND fencing_token = ?;
            """, (
                lease.resource_key, lease.coordination_epoch, lease.lease_id, lease.fencing_token
            ))
            event_id = f"evt-{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO radar_lease_events (
                    event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                    fencing_token, event_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                event_id, lease.resource_key, lease.coordination_epoch, lease.lease_id,
                lease.run_id, lease.job_id, lease.fencing_token,
                EventType.RELEASED.value, json.dumps({"released": True})
            ))
            conn.execute("COMMIT;")
            return cur.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def reset_coordination_epoch(
        self,
        resource_key: str,
        new_epoch: str,
        reason: str,
        updated_by: str,
    ) -> None:
        """
        Explicit administrative epoch reset.
        Invalidates any existing lease and resets monotonic fencing sequence generation.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            # Fetch previous epoch
            prev_row = conn.execute(
                "SELECT current_epoch, epoch_sequence FROM radar_coordination_epochs WHERE resource_key = ?;",
                (resource_key,)
            ).fetchone()
            prev_epoch = prev_row["current_epoch"] if prev_row else DEFAULT_COORDINATION_EPOCH
            new_seq = (prev_row["epoch_sequence"] + 1) if prev_row else 2

            # Upsert new epoch
            conn.execute("""
                INSERT INTO radar_coordination_epochs (
                    resource_key, current_epoch, epoch_sequence, updated_at, updated_by, reason
                ) VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?)
                ON CONFLICT(resource_key) DO UPDATE SET
                    current_epoch = excluded.current_epoch,
                    epoch_sequence = excluded.epoch_sequence,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by,
                    reason = excluded.reason;
            """, (resource_key, new_epoch, new_seq, updated_by, reason))

            # Evict current lease immediately
            conn.execute("DELETE FROM radar_current_leases WHERE resource_key = ?;", (resource_key,))

            # Append EPOCH_RESET event
            event_id = f"evt-{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO radar_lease_events (
                    event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                    fencing_token, event_type, metadata_json
                ) VALUES (?, ?, ?, 'N/A', 'N/A', 'N/A', 0, ?, ?);
            """, (
                event_id, resource_key, new_epoch,
                EventType.EPOCH_RESET.value,
                json.dumps({
                    "previous_epoch": prev_epoch,
                    "new_epoch": new_epoch,
                    "reason": reason,
                    "updated_by": updated_by,
                })
            ))
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def reconcile_abandoned_runs(self, resource_key: str) -> int:
        """
        Find and reconcile abandoned runs (status=RUNNING without an active unexpired lease).
        Transitions them to ABORTED_LEASE_EXPIRED.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            rows = conn.execute("""
                SELECT run_id FROM radar_runs
                WHERE resource_key = ?
                  AND status = 'RUNNING'
                  AND run_id NOT IN (
                      SELECT run_id FROM radar_current_leases
                      WHERE resource_key = ? AND datetime(expires_at) > datetime('now')
                  );
            """, (resource_key, resource_key)).fetchall()

            count = len(rows)
            for r in rows:
                conn.execute("""
                    UPDATE radar_runs
                    SET status = 'ABORTED_LEASE_EXPIRED', completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE run_id = ?;
                """, (r["run_id"],))

            conn.execute("COMMIT;")
            return count
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def get_lease_events(self, resource_key: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve latest append-only lease events for telemetry and auditing."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM radar_lease_events
                WHERE resource_key = ?
                ORDER BY observed_at DESC
                LIMIT ?;
            """, (resource_key, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @retry_sqlite()
    def get_run(self, run_id: str) -> Optional[RunAttempt]:
        """Fetch details of a specific run attempt."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM radar_runs WHERE run_id = ?;", (run_id,)).fetchone()
            if not row:
                return None
            return RunAttempt(
                run_id=row["run_id"],
                job_id=row["job_id"],
                resource_key=row["resource_key"],
                coordination_epoch=row["coordination_epoch"],
                lease_id=row["lease_id"],
                fencing_token=row["fencing_token"],
                owner_instance_id=row["owner_instance_id"],
                implementation_release_sha=row["implementation_release_sha"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                status=RunStatus(row["status"]),
                failure_code=row["failure_code"],
                universe_build_id=row["universe_build_id"],
                scanner_snapshot_id=row["scanner_snapshot_id"],
            )
        finally:
            conn.close()
