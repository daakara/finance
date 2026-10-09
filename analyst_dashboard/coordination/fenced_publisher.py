"""
analyst_dashboard/coordination/fenced_publisher.py

Atomic Transactional Fenced Publisher for Radar Scanner Snapshots.
Enforces the mandatory invariant: NO_OVERLAPPING_AUTHORIZED_PUBLICATION.
"""

import sqlite3
import json
import logging
import uuid
from typing import Dict, Any, Optional

from analyst_dashboard.data.market_db import retry_sqlite
from .contracts import (
    CurrentLease,
    EventType,
    JobStatus,
    RunStatus,
    StaleLeasePublicationError,
    LeasePolicy,
    PRODUCTION_LEASE_POLICY,
)
from .store import CoordinationStore

logger = logging.getLogger(__name__)


class FencedPublisher:
    """
    Guarantees that a scanner snapshot is published ONLY if the executing worker
    holds the exact, unexpired lease authority inside the mutation transaction.
    """

    def __init__(self, coordination_store: CoordinationStore):
        self.store = coordination_store

    @retry_sqlite()
    def publish_snapshot_fenced(
        self,
        lease: CurrentLease,
        snapshot_data: Dict[str, Any],
        universe_build_id: Optional[str] = None,
        policy: LeasePolicy = PRODUCTION_LEASE_POLICY,
    ) -> str:
        """
        Atomically:
        1. Verifies exact lease authority:
           - resource_key == lease.resource_key
           - coordination_epoch == lease.coordination_epoch
           - lease_id == lease.lease_id
           - fencing_token == lease.fencing_token
           - expires_at > authoritative_db_time
        2. Inserts snapshot into scanner_snapshots.
        3. Updates radar_runs to SUCCEEDED.
        4. Updates radar_jobs to SUCCEEDED with successful_publication_id.
        5. Appends PUBLICATION_COMMITTED to radar_lease_events.
        6. Releases lease authority cleanly.
        7. Commits transaction.

        On lease check failure:
        - Aborts snapshot insertion.
        - Records STALE_WRITE_REJECTED in radar_lease_events.
        - Updates radar_runs to ABORTED_LEASE_LOST.
        - Raises StaleLeasePublicationError.
        """
        conn = self.store._get_connection()
        snapshot_id = snapshot_data["snapshot_id"]
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # 1. Authoritative Transactional Lease Verification
            lease_check = conn.execute("""
                SELECT 1 FROM radar_current_leases
                WHERE resource_key = ?
                  AND coordination_epoch = ?
                  AND lease_id = ?
                  AND fencing_token = ?
                  AND datetime(expires_at) > datetime('now');
            """, (
                lease.resource_key,
                lease.coordination_epoch,
                lease.lease_id,
                lease.fencing_token,
            )).fetchone()

            if not lease_check:
                # FENCE REJECTION: Stale or expired worker
                event_id = f"evt-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO radar_lease_events (
                        event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                        fencing_token, event_type, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    event_id,
                    lease.resource_key,
                    lease.coordination_epoch,
                    lease.lease_id,
                    lease.run_id,
                    lease.job_id,
                    lease.fencing_token,
                    EventType.STALE_WRITE_REJECTED.value,
                    json.dumps({
                        "snapshot_id": snapshot_id,
                        "reason": "LEASE_EXPIRED_OR_SUPERSEDED_DURING_PUBLICATION",
                    })
                ))

                conn.execute("""
                    UPDATE radar_runs
                    SET status = 'ABORTED_LEASE_LOST', completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE run_id = ?;
                """, (lease.run_id,))

                conn.execute("COMMIT;")

                msg = (
                    f"FENCED_REJECTED: Publication rejected for run {lease.run_id}. "
                    f"Lease {lease.lease_id} (token {lease.fencing_token}) is no longer authoritative or expired."
                )
                logger.error(msg)
                raise StaleLeasePublicationError(msg)

            # 2. Inject Lease Provenance into Snapshot
            provenance_dict = dict(snapshot_data.get("provenance", {}))
            provenance_dict["coordination"] = lease.to_provenance()

            # Ensure scanner_snapshots table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scanner_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    scanner_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    api_contract_version TEXT NOT NULL,
                    ruleset_version TEXT NOT NULL,
                    evidence_schema_version TEXT NOT NULL,
                    score_model_version TEXT NOT NULL,
                    data_provenance_version TEXT NOT NULL,
                    universe_version TEXT NOT NULL,
                    freshness_policy_version TEXT NOT NULL,
                    implementation_release_sha TEXT NOT NULL,
                    semantic_fingerprint TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    data_as_of TEXT NOT NULL,
                    status_at_publication TEXT NOT NULL,
                    universe_id TEXT NOT NULL,
                    universe_size INTEGER NOT NULL,
                    matched_count INTEGER NOT NULL,
                    results_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    freshness_json TEXT NOT NULL,
                    publication_decision TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Insert Immutable Snapshot Record
            conn.execute("""
                INSERT INTO scanner_snapshots (
                    snapshot_id, scanner_id, run_id, api_contract_version, ruleset_version,
                    evidence_schema_version, score_model_version, data_provenance_version,
                    universe_version, freshness_policy_version, implementation_release_sha,
                    semantic_fingerprint, generated_at, data_as_of, status_at_publication,
                    universe_id, universe_size, matched_count, results_json, provenance_json,
                    freshness_json, publication_decision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                snapshot_id,
                snapshot_data["scanner_id"],
                lease.run_id,
                snapshot_data["api_contract_version"],
                snapshot_data["ruleset_version"],
                snapshot_data["evidence_schema_version"],
                snapshot_data["score_model_version"],
                snapshot_data["data_provenance_version"],
                snapshot_data["universe_version"],
                snapshot_data["freshness_policy_version"],
                snapshot_data["implementation_release_sha"],
                snapshot_data["semantic_fingerprint"],
                snapshot_data["generated_at"],
                snapshot_data["data_as_of"],
                snapshot_data["status_at_publication"],
                snapshot_data["universe_id"],
                snapshot_data["universe_size"],
                snapshot_data["matched_count"],
                json.dumps(snapshot_data["results"]),
                json.dumps(provenance_dict),
                json.dumps(snapshot_data["freshness"]),
                snapshot_data["publication_decision"],
            ))

            # 4. Mark Run SUCCEEDED
            conn.execute("""
                UPDATE radar_runs
                SET status = 'SUCCEEDED',
                    scanner_snapshot_id = ?,
                    universe_build_id = ?,
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE run_id = ?;
            """, (snapshot_id, universe_build_id, lease.run_id))

            # 5. Bind Successful Publication to Logical Job
            conn.execute("""
                UPDATE radar_jobs
                SET job_status = 'SUCCEEDED',
                    successful_publication_id = ?
                WHERE job_id = ?;
            """, (snapshot_id, lease.job_id))

            # 6. Append PUBLICATION_COMMITTED Event
            event_id = f"evt-{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO radar_lease_events (
                    event_id, resource_key, coordination_epoch, lease_id, run_id, job_id,
                    fencing_token, event_type, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                event_id,
                lease.resource_key,
                lease.coordination_epoch,
                lease.lease_id,
                lease.run_id,
                lease.job_id,
                lease.fencing_token,
                EventType.PUBLICATION_COMMITTED.value,
                json.dumps({
                    "snapshot_id": snapshot_id,
                    "universe_build_id": universe_build_id,
                    "matched_count": snapshot_data["matched_count"],
                })
            ))

            # 7. Release Active Lease Authority
            conn.execute("""
                DELETE FROM radar_current_leases
                WHERE resource_key = ? AND lease_id = ?;
            """, (lease.resource_key, lease.lease_id))

            conn.execute("COMMIT;")
            logger.info(
                f"Fenced publication COMMITTED for run {lease.run_id}, "
                f"snapshot {snapshot_id}, fencing token {lease.fencing_token}."
            )
            return snapshot_id
        except StaleLeasePublicationError:
            raise
        except Exception as e:
            conn.execute("ROLLBACK;")
            logger.error(f"Fenced publication transaction error for {lease.run_id}: {e}")
            raise
        finally:
            conn.close()
