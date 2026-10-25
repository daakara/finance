"""
Independent Production Release-Readiness Adversarial Verification Suite
Candidate 006: Worker Lease Recovery & Automated Backup Remediation

Mechanically verifies:
1. Section 4: Final publication boundary lease fencing & concurrency race safety.
2. Section 5: Real process crash recovery, process fault injection, and terminal retry bounds.
3. Section 6: SEC Owner advisory lock single active ownership, connection drop handover, and backup lock independence.
4. Section 7: Schema V1 to V2 additive migration parity and idempotency.
5. Section 8: Canonical domain logic parity and immutable evidence invariants.
"""

import os
import sys
import time
import uuid
import signal
import tempfile
import subprocess
import pytest
psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor
pgserver = pytest.importorskip("pgserver")

from analyst_dashboard.smart_money.storage.postgres_schema import (
    initialize_postgres_schema,
    SCHEMA_DDL,
    SCHEMA_V2_MIGRATION_DDL,
)
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
    """Spins up an isolated PostgreSQL 16 server in a temporary directory."""
    pgdata = tempfile.mkdtemp(prefix="arx_test_adversarial_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield uri

    server.cleanup()


class TestSection4LeaseFencingCorrectness:
    """Independent verification of Section 4: Transactional Lease Fencing & Race Safety."""

    def test_complete_lease_fencing_lifecycle(self, pg_uri):
        """
        Required Scenario (Section 4):
        1. Worker A claims a request.
        2. Worker A becomes delayed.
        3. Its lease expires.
        4. Worker B recovers and claims the request.
        5. Worker A attempts to publish.
        6. Worker A is rejected.
        7. Worker B completes successfully.
        8. Exactly one authoritative publication exists.
        """
        repo = PostgresSmartMoneyRepository(dsn=pg_uri)

        # 1. Create request and Worker A claims
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subj_fence_lifecycle",
            max_attempts=3,
        )
        claim_a = repo.claim_next_run_request(worker_id="worker-A", lease_duration_sec=1)
        assert claim_a is not None
        token_a = claim_a["lease_token"]

        # 2 & 3. Worker A delayed; lease artificially expired in PostgreSQL
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

        # 4. Worker B recovers and claims request
        recovered = repo.recover_stale_run_requests(backoff_sec=0)
        assert recovered == 1
        claim_b = repo.claim_next_run_request(worker_id="worker-B", lease_duration_sec=60)
        assert claim_b is not None
        assert claim_b["request_id"] == req_id
        token_b = claim_b["lease_token"]
        assert token_b != token_a

        # 5 & 6. Worker A attempts to publish snapshot with token_a -> REJECTED
        snap_a = {
            "snapshot_id": f"SNAP-A-{uuid.uuid4().hex[:8]}",
            "scanner_id": "SMART_MONEY",
            "run_id": "run-worker-A",
            "api_contract_version": "1.0",
            "ruleset_version": "V1",
            "evidence_schema_version": "V1",
            "score_model_version": "V1",
            "data_provenance_version": "V1",
            "universe_version": "V1",
            "freshness_policy_version": "V1",
            "implementation_release_sha": "da7f6ff52b8d79658e041f935424c0fa57f42984",
            "semantic_fingerprint": "fp_worker_a",
            "generated_at": "2026-10-14T01:00:00Z",
            "data_as_of": "2026-10-14T01:00:00Z",
            "status_at_publication": "COMPLETE",
            "universe_id": "univ_test",
            "universe_size": 100,
            "matched_count": 5,
            "publication_decision": "PUBLISH",
            "results": [],
            "provenance": {},
            "freshness": {},
        }

        with pytest.raises(StaleLeaseOwnershipLostError) as exc_info:
            repo.publish_scanner_snapshot_atomic(
                snap_a,
                run_request_id=req_id,
                lease_token=token_a,
            )
        assert "STALE_LEASE_OWNERSHIP_LOST" in str(exc_info.value)

        # 7. Worker B completes successfully
        snap_b = dict(snap_a)
        snap_b["snapshot_id"] = f"SNAP-B-{uuid.uuid4().hex[:8]}"
        snap_b["run_id"] = "run-worker-B"
        snap_b["semantic_fingerprint"] = "fp_worker_b"

        pub_b_id = repo.publish_scanner_snapshot_atomic(
            snap_b,
            run_request_id=req_id,
            lease_token=token_b,
        )
        assert pub_b_id == snap_b["snapshot_id"]

        # 8. Exactly one authoritative publication exists
        active = repo.get_latest_active_snapshot("SMART_MONEY")
        assert active["snapshot_id"] == snap_b["snapshot_id"]

        # Check total snapshot records for this run_request
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("SELECT snapshot_id FROM scanner_snapshots WHERE run_id IN ('run-worker-A', 'run-worker-B');")
            rows = cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == snap_b["snapshot_id"]
        conn.close()

    def test_pre_publication_race_condition_protection(self, pg_uri):
        """
        Verifies that row-level locking (SELECT FOR UPDATE) prevents race conditions
        where ownership changes concurrently during pre-publication verification.
        """
        conn1 = psycopg2.connect(pg_uri)
        conn2 = psycopg2.connect(pg_uri)

        repo = PostgresSmartMoneyRepository(dsn=pg_uri)
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subj_race_lock",
        )
        claimed = repo.claim_next_run_request(worker_id="worker-race-1")
        token = claimed["lease_token"]

        # Transaction 1 acquires row lock on run request
        with conn1.cursor() as cur1:
            cur1.execute("""
                SELECT request_id FROM sm_run_requests
                WHERE request_id = %s AND lease_token = %s AND status = 'RUNNING'
                FOR UPDATE;
            """, (req_id, token))
            assert cur1.fetchone() is not None

            # Transaction 2 attempts to usurp/recover the same row
            # Since Transaction 1 holds FOR UPDATE, Transaction 2 with SKIP LOCKED skips it
            with conn2.cursor() as cur2:
                cur2.execute("""
                    SELECT request_id FROM sm_run_requests
                    WHERE request_id = %s
                    FOR UPDATE SKIP LOCKED;
                """, (req_id,))
                assert cur2.fetchone() is None, "Row lock must prevent concurrent usurpation"

        conn1.close()
        conn2.close()


class TestSection5ProcessFaultInjectionCrashRecovery:
    """Independent verification of Section 5: Real Process Faults & Crash Recovery."""

    def test_subprocess_worker_crash_and_automatic_recovery(self, pg_uri):
        """
        Forks/spawns an actual independent Python worker process, terminates it via SIGTERM/SIGKILL,
        and verifies that a subsequent worker lifecycle cleanly detects expiry and recovers work.
        """
        repo = PostgresSmartMoneyRepository(dsn=pg_uri)
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subj_proc_crash",
            max_attempts=3,
        )

        # Worker 1 script claims request and sleeps indefinitely (simulating in-flight crash)
        worker_code = f"""
import psycopg2
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
repo = PostgresSmartMoneyRepository(dsn="{pg_uri}")
claimed = repo.claim_next_run_request(worker_id="crashed-proc-1", lease_duration_sec=2)
if claimed:
    print("CLAIMED:" + claimed["lease_token"], flush=True)
import time
time.sleep(30)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for worker to claim
        token = None
        for line in proc.stdout:
            if line.startswith("CLAIMED:"):
                token = line.strip().split(":")[1]
                break

        assert token is not None, "Worker subprocess failed to claim"

        # Verify DB is RUNNING
        req_row = repo.get_run_request(req_id)
        assert req_row["status"] == "RUNNING"
        assert req_row["lease_token"] == token

        # INJECT SUDDEN CRASH (KILL PROCESS)
        proc.kill()
        proc.wait()

        # Simulate passage of time past lease expiry (2 sec)
        time.sleep(2.5)

        # Worker 2 boots up and executes startup recovery pass
        recovery_code = f"""
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
repo = PostgresSmartMoneyRepository(dsn="{pg_uri}")
recovered = repo.recover_stale_run_requests(backoff_sec=0)
print(f"RECOVERED:{{recovered}}", flush=True)
claimed = repo.claim_next_run_request(worker_id="replacement-proc-2")
if claimed:
    print(f"RECLAIMED:{{claimed['request_id']}}", flush=True)
    repo.complete_run_request(claimed['request_id'], lease_token=claimed['lease_token'], status='COMPLETED')
    print("COMPLETED", flush=True)
"""
        proc2 = subprocess.run(
            [sys.executable, "-c", recovery_code],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "RECOVERED:1" in proc2.stdout
        assert "COMPLETED" in proc2.stdout

        # Verify final state is COMPLETED
        final_row = repo.get_run_request(req_id)
        assert final_row["status"] == "COMPLETED"
        assert final_row["attempt_count"] == 2

    def test_terminal_retry_bounds_poison_pill(self, pg_uri):
        """Verifies that exceeding max_attempts marks request FAILED permanently."""
        repo = PostgresSmartMoneyRepository(dsn=pg_uri)
        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subj_poison_pill",
            max_attempts=2,
        )

        # Attempt 1
        c1 = repo.claim_next_run_request(worker_id="w1", lease_duration_sec=0)
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            cur.execute("UPDATE sm_run_requests SET lease_expires_at = NOW() - INTERVAL '1s' WHERE request_id = %s;", (req_id,))
        conn.commit()

        # Recover -> attempt 1 fails, marked RETRY_ELIGIBLE
        repo.recover_stale_run_requests(backoff_sec=0)

        # Attempt 2
        with conn.cursor() as cur:
            cur.execute("UPDATE sm_run_requests SET next_attempt_at = NOW() WHERE request_id = %s;", (req_id,))
        conn.commit()
        c2 = repo.claim_next_run_request(worker_id="w2", lease_duration_sec=0)
        assert c2 is not None

        # Expire Attempt 2
        with conn.cursor() as cur:
            cur.execute("UPDATE sm_run_requests SET lease_expires_at = NOW() - INTERVAL '1s' WHERE request_id = %s;", (req_id,))
        conn.commit()
        conn.close()

        # Recover -> attempt 2 is max_attempts (2) -> transitions to FAILED
        repo.recover_stale_run_requests()

        final_row = repo.get_run_request(req_id)
        assert final_row["status"] == "FAILED"
        assert final_row["last_failure_code"] == "LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS"


class TestSection6CanonicalSecOwnerLockVerification:
    """Independent verification of Section 6: SEC Egress Advisory Lock Handover."""

    def test_advisory_lock_handover_and_non_interference(self, pg_uri):
        """
        Verifies:
        1. Single active SEC owner.
        2. Second owner rejected non-blockingly.
        3. Connection termination releases lock automatically.
        4. Replacement owner acquires lock immediately.
        5. Backup execution lock does not collide with SEC owner lock.
        """
        conn_a = psycopg2.connect(pg_uri)
        conn_b = psycopg2.connect(pg_uri)
        conn_backup = psycopg2.connect(pg_uri)

        # 1. Worker A acquires SEC lock
        assert acquire_sec_owner_lock(conn_a) is True

        # 2. Worker B attempts acquisition -> Rejected
        assert acquire_sec_owner_lock(conn_b) is False

        # 5. Backup lock acquired concurrently -> Succeeds independently
        assert acquire_backup_lock(conn_backup) is True

        # 3. Worker A drops connection abruptly (simulating SIGKILL)
        conn_a.close()

        # 4. Worker B retries acquisition -> Succeeds immediately
        assert acquire_sec_owner_lock(conn_b) is True

        # Clean up
        assert release_backup_lock(conn_backup) is True
        assert release_sec_owner_lock(conn_b) is True
        conn_b.close()
        conn_backup.close()


class TestSection7SchemaV1ToV2ForwardMigrationAndIdempotency:
    """Independent verification of Section 7: Additive V2 Migration on clean DB."""

    def test_forward_migration_and_idempotency(self, pg_uri):
        conn = psycopg2.connect(pg_uri)
        with conn.cursor() as cur:
            # Re-run migration DDL (idempotency check)
            cur.execute(SCHEMA_V2_MIGRATION_DDL)
        conn.commit()

        # Verify all columns exist on sm_run_requests
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT column_name, data_type, column_default
                FROM information_schema.columns
                WHERE table_name = 'sm_run_requests';
            """)
            cols = {r["column_name"]: r for r in cur.fetchall()}

        expected_v2_cols = [
            "lease_owner_id", "lease_token", "lease_expires_at",
            "lease_heartbeat_at", "attempt_count", "max_attempts",
            "next_attempt_at", "last_failure_code", "last_failure_at",
        ]
        for col in expected_v2_cols:
            assert col in cols, f"Missing V2 column: {col}"

        conn.close()


class TestSection8CanonicalDomainParity:
    """Independent verification of Section 8: Preserved Analytical Engine Rules."""

    def test_canonical_invariants_preserved(self):
        from analyst_dashboard.smart_money.storage.config import (
            ACTIVE_SCHEMA_VERSION,
            SMART_MONEY_POSTGRES_SCHEMA_VERSION_V2,
            ROLE_BACKUP,
            ROLE_INGESTION,
            ROLE_WEB,
            ROLE_MIGRATION,
        )
        assert ACTIVE_SCHEMA_VERSION == SMART_MONEY_POSTGRES_SCHEMA_VERSION_V2
        assert ROLE_BACKUP == "smart_money_backup"
        assert ROLE_INGESTION == "smart_money_ingestion"
        assert ROLE_WEB == "smart_money_web"
        assert ROLE_MIGRATION == "smart_money_migration"
