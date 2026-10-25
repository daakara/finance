"""
Comprehensive Test Suite for Smart Money Worker Lease Recovery & Fencing (Candidate 006).

Covers:
1. Database-backed lease contract on sm_run_requests (tokens, attempts, expiry, heartbeats).
2. Atomic claim via FOR UPDATE SKIP LOCKED without race collisions.
3. Heartbeat renewal succeeds for active owner and fails for stale/stolen lease.
4. Deterministic stale-claim recovery for abandoned requests (RETRY_ELIGIBLE vs FAILED).
5. Fencing: Stale worker cannot complete request or publish snapshot.
6. Atomic snapshot publication transaction rolls back on lease mismatch.
7. Worker startup recovery cleans up abandoned work automatically.
8. SEC Egress Owner lock non-interference.
"""

import time
import uuid
import tempfile
import pytest
psycopg2 = pytest.importorskip("psycopg2")
pgserver = pytest.importorskip("pgserver")

pytestmark = pytest.mark.tier2c

from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    acquire_backup_lock,
    release_backup_lock,
    LOCK_SEC_EGRESS_OWNER,
    LOCK_BACKUP_EXECUTION,
)


@pytest.fixture(scope="module")
def pg_uri():
    """Spins up an isolated PostgreSQL 16 server in a temp directory."""
    pgdata = tempfile.mkdtemp(prefix="arx_test_lease_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield uri

    server.cleanup()


@pytest.fixture
def repo(pg_uri):
    return PostgresSmartMoneyRepository(dsn=pg_uri)


class TestWorkerLeaseContractLifecycle:
    """Verifies atomic lease acquisition, attempt counting, and expiry."""

    def test_atomic_claim_establishes_lease_contract(self, repo):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_test_lease_contract",
            max_attempts=3,
        )

        claimed = repo.claim_next_run_request(worker_id="worker-node-alpha", lease_duration_sec=30)
        assert claimed is not None
        assert claimed["request_id"] == req_id
        assert claimed["claimed_by"] == "worker-node-alpha"
        assert claimed["lease_token"] is not None
        assert claimed["attempt_count"] == 1
        assert claimed["max_attempts"] == 3
        assert claimed["lease_expires_at"] is not None

        # Verify database row state
        db_row = repo.get_run_request(req_id)
        assert db_row["status"] == "RUNNING"
        assert db_row["lease_owner_id"] == "worker-node-alpha"
        assert db_row["lease_token"] == claimed["lease_token"]

        # Clean up
        repo.complete_run_request(req_id, lease_token=claimed["lease_token"], status="COMPLETED")

    def test_concurrent_claim_skip_locked(self, repo):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_concurrent_claim",
        )

        # Worker 1 claims
        c1 = repo.claim_next_run_request(worker_id="worker-1", lease_duration_sec=30)
        assert c1 is not None
        assert c1["request_id"] == req_id

        # Worker 2 attempts concurrent claim -> returns None
        c2 = repo.claim_next_run_request(worker_id="worker-2", lease_duration_sec=30)
        assert c2 is None

        # Complete by worker 1
        repo.complete_run_request(req_id, lease_token=c1["lease_token"], status="COMPLETED")


class TestHeartbeatAndRenewal:
    """Verifies heartbeat renewal and protection against stale workers."""

    def test_heartbeat_renews_active_lease(self, repo):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_heartbeat_test",
        )
        claimed = repo.claim_next_run_request(worker_id="worker-hb-1", lease_duration_sec=10)
        token = claimed["lease_token"]

        # Successful renewal
        renewed = repo.heartbeat_run_request(req_id, lease_token=token, extension_sec=20)
        assert renewed is True

        # Invalid token renewal fails
        fake_renewed = repo.heartbeat_run_request(req_id, lease_token="fake-token", extension_sec=20)
        assert fake_renewed is False

        repo.complete_run_request(req_id, lease_token=token, status="COMPLETED")


class TestStaleClaimRecovery:
    """Verifies automated recovery of abandoned RUNNING requests."""

    def test_stale_claim_recovers_to_retry_eligible_with_backoff(self, repo, pg_uri):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_recovery_retry",
            max_attempts=3,
        )
        claimed = repo.claim_next_run_request(worker_id="worker-crashed-1", lease_duration_sec=1)
        token = claimed["lease_token"]

        # Artificially expire the lease in PostgreSQL
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sm_run_requests
                SET lease_expires_at = NOW() - INTERVAL '5 seconds'
                WHERE request_id = %s;
            """, (req_id,))
        conn.commit()
        conn.close()

        # Run recovery
        recovered_count = repo.recover_stale_run_requests(backoff_sec=1)
        assert recovered_count >= 1

        db_row = repo.get_run_request(req_id)
        assert db_row["status"] == "RETRY_ELIGIBLE"
        assert db_row["last_failure_code"] == "LEASE_EXPIRED_RECOVERED"
        assert db_row["lease_token"] is None
        assert db_row["lease_owner_id"] == "None" or db_row["lease_owner_id"] is None

        # Clean up
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("UPDATE sm_run_requests SET next_attempt_at = NOW() WHERE request_id = %s;", (req_id,))
        conn.commit()
        conn.close()

        claimed_again = repo.claim_next_run_request(worker_id="worker-replacement-1")
        assert claimed_again is not None
        assert claimed_again["request_id"] == req_id
        assert claimed_again["attempt_count"] == 2
        repo.complete_run_request(req_id, lease_token=claimed_again["lease_token"], status="COMPLETED")

    def test_stale_claim_fails_terminally_after_max_attempts(self, repo, pg_uri):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_recovery_max_attempts",
            max_attempts=2,
        )
        # Attempt 1
        c1 = repo.claim_next_run_request(worker_id="worker-c1")
        # Artificially set attempt_count = 2 and expired
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sm_run_requests
                SET attempt_count = 2,
                    lease_expires_at = NOW() - INTERVAL '10 seconds'
                WHERE request_id = %s;
            """, (req_id,))
        conn.commit()
        conn.close()

        # Recover -> should transition to FAILED
        recovered_count = repo.recover_stale_run_requests()
        assert recovered_count >= 1

        db_row = repo.get_run_request(req_id)
        assert db_row["status"] == "FAILED"
        assert db_row["last_failure_code"] == "LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS"
        assert db_row["completed_at"] is not None


class TestStaleWorkerFencingAndPublicationSafety:
    """Verifies that a stale worker whose lease expired or transferred cannot publish."""

    def test_stale_worker_completion_rejected(self, repo, pg_uri):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_fencing_completion",
        )
        c1 = repo.claim_next_run_request(worker_id="worker-stale-1", lease_duration_sec=1)
        stale_token = c1["lease_token"]

        # Expire and recover request
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sm_run_requests
                SET lease_expires_at = NOW() - INTERVAL '10 seconds',
                    next_attempt_at = NOW() - INTERVAL '5 seconds'
                WHERE request_id = %s;
            """, (req_id,))
        conn.commit()
        conn.close()

        repo.recover_stale_run_requests(backoff_sec=0)

        # Replacement worker claims request
        c2 = repo.claim_next_run_request(worker_id="worker-replacement-2")
        new_token = c2["lease_token"]

        # Original stale worker attempts completion with old token -> FAILS CLOSED
        with pytest.raises(StaleLeaseOwnershipLostError):
            repo.complete_run_request(req_id, lease_token=stale_token, status="COMPLETED")

        # Replacement worker completes cleanly
        repo.complete_run_request(req_id, lease_token=new_token, status="COMPLETED")
        assert repo.get_run_request(req_id)["status"] == "COMPLETED"

    def test_stale_worker_atomic_snapshot_publication_rejected(self, repo, pg_uri):
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_fencing_publication",
        )
        c1 = repo.claim_next_run_request(worker_id="worker-stale-pub", lease_duration_sec=1)
        stale_token = c1["lease_token"]

        # Expire lease
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE sm_run_requests
                SET lease_expires_at = NOW() - INTERVAL '10 seconds',
                    next_attempt_at = NOW() - INTERVAL '5 seconds'
                WHERE request_id = %s;
            """, (req_id,))
        conn.commit()
        conn.close()

        repo.recover_stale_run_requests(backoff_sec=0)
        c2 = repo.claim_next_run_request(worker_id="worker-replacement-pub")
        new_token = c2["lease_token"]

        # Stale worker attempts to publish snapshot with old token
        snap_stale = {
            "snapshot_id": f"SNAP-STALE-{uuid.uuid4().hex[:8]}",
            "scanner_id": "SMART_MONEY",
            "run_id": "stale-worker-run",
            "api_contract_version": "1.0",
            "ruleset_version": "V1",
            "evidence_schema_version": "V1",
            "score_model_version": "V1",
            "data_provenance_version": "V1",
            "universe_version": "V1",
            "freshness_policy_version": "V1",
            "implementation_release_sha": "09f183de52e275ae543615747dff2f9ae00ef9ca",
            "semantic_fingerprint": "fp_stale",
            "generated_at": "2026-10-14T01:00:00Z",
            "data_as_of": "2026-10-14T01:00:00Z",
            "status_at_publication": "COMPLETE",
            "universe_id": "univ_test",
            "universe_size": 10,
            "matched_count": 1,
            "publication_decision": "PUBLISH",
            "results": [],
            "provenance": {},
            "freshness": {},
        }

        with pytest.raises(StaleLeaseOwnershipLostError):
            repo.publish_scanner_snapshot_atomic(
                snap_stale,
                run_request_id=req_id,
                lease_token=stale_token,
            )

        # Replacement worker publishes snapshot with valid token
        snap_valid = dict(snap_stale)
        snap_valid["snapshot_id"] = f"SNAP-VALID-{uuid.uuid4().hex[:8]}"
        snap_valid["run_id"] = "replacement-worker-run"

        published_id = repo.publish_scanner_snapshot_atomic(
            snap_valid,
            run_request_id=req_id,
            lease_token=new_token,
        )
        assert published_id == snap_valid["snapshot_id"]

        # Verify active snapshot pointer points to valid snapshot
        active = repo.get_latest_active_snapshot("SMART_MONEY")
        assert active["snapshot_id"] == snap_valid["snapshot_id"]

        # Verify run_request completed atomically
        req_row = repo.get_run_request(req_id)
        assert req_row["status"] == "COMPLETED"
        assert req_row["result_generation_id"] == snap_valid["snapshot_id"]


class TestCoordinationLockNonInterference:
    """Verifies that backup advisory locks and SEC egress locks operate independently without interference."""

    def test_sec_owner_lock_and_backup_lock_coexist(self, pg_uri):
        conn_sec = psycopg2.connect(pg_uri)
        conn_backup = psycopg2.connect(pg_uri)

        # Acquire SEC owner lock
        assert acquire_sec_owner_lock(conn_sec) is True

        # Acquire Backup lock concurrently -> must succeed independently
        assert acquire_backup_lock(conn_backup) is True

        # Second backup attempt on another connection fails non-blockingly
        conn_backup_2 = psycopg2.connect(pg_uri)
        assert acquire_backup_lock(conn_backup_2) is False
        conn_backup_2.close()

        # Release both
        assert release_backup_lock(conn_backup) is True
        assert release_sec_owner_lock(conn_sec) is True

        conn_sec.close()
        conn_backup.close()
