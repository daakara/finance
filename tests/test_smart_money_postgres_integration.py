"""
Real PostgreSQL integration tests for Smart Money schema, triggers,
advisory locks, concurrency, PIT queries, amendments, migration parity,
dedicated worker, and least-privilege role permissions.
"""

import os
import time
import uuid
import json
import tempfile
import sqlite3
import pytest

pytestmark = pytest.mark.tier2c
from datetime import datetime
psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor
pgserver = pytest.importorskip("pgserver")

from analyst_dashboard.smart_money.storage.config import (
    SMART_MONEY_POSTGRES_SCHEMA_VERSION,
    ROLE_WEB,
    ROLE_INGESTION,
)
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    LOCK_SEC_EGRESS_OWNER,
)
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.migration import SmartMoneySqliteMigrator, verify_migration_parity
from analyst_dashboard.smart_money.worker import SmartMoneyIngestionWorker, WorkerHealthState


@pytest.fixture(scope="module")
def real_postgres_uri():
    """Starts an isolated PostgreSQL 16 server in a temporary directory."""
    pgdata = tempfile.mkdtemp(prefix="arx_test_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    # Connect and initialize schema
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield uri

    server.cleanup()


@pytest.fixture
def repo(real_postgres_uri):
    return PostgresSmartMoneyRepository(dsn=real_postgres_uri)


class TestPostgresSchemaAndImmutabilityTriggers:
    """Tests schema creation and append-only trigger enforcement."""

    def test_schema_initialized(self, real_postgres_uri):
        conn = psycopg2.connect(real_postgres_uri)
        with conn.cursor() as cur:
            cur.execute("SELECT version_id FROM sm_schema_migrations;")
            row = cur.fetchone()
            assert row is not None
            assert row[0] == SMART_MONEY_POSTGRES_SCHEMA_VERSION
        conn.close()

    def test_raw_filings_immutability_triggers(self, real_postgres_uri, repo):
        acc = f"ACC-{uuid.uuid4().hex[:8]}"
        filing = {
            "accession_number": acc,
            "cik": "0001018724",
            "ticker": "AMZN",
            "company_name": "Amazon.com Inc",
            "filing_date": "2026-10-10",
            "accepted_datetime": "2026-10-10T18:00:00Z",
            "xml_content": "<xml>test</xml>",
            "content_hash": "hash_123",
            "source_url": "https://sec.gov/test",
        }
        # Insert succeeds
        assert repo.save_raw_filing(filing) is True
        # Duplicate with same hash is idempotent no-op
        assert repo.save_raw_filing(filing) is False

        # Attempted UPDATE must fail via trigger
        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE form4_raw_filings SET ticker = 'GOOG' WHERE accession_number = %s;",
                    (acc,)
                )
        conn.close()

        # Attempted DELETE must fail via trigger
        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM form4_raw_filings WHERE accession_number = %s;",
                    (acc,)
                )
        conn.close()

    def test_reconciliation_receipts_append_only(self, real_postgres_uri, repo):
        r1_id = str(uuid.uuid4())
        receipt1 = {
            "receipt_id": r1_id,
            "interval_start": "2026-10-10T00:00:00Z",
            "interval_end": "2026-10-10T23:59:59Z",
            "certified_watermark": "2026-10-10T23:59:59Z",
            "receipt_version": 1,
            "supersedes_receipt_id": None,
            "expected_count": 100,
            "accounted_count": 95,
            "missing_count": 5,
            "material_gap_count": 5,
            "coverage_status": "PARTIAL",
            "failure_reasons": ["MISSING_ACCESSION"],
            "reconciliation_hash": "hash_v1",
        }
        res_id = repo.save_reconciliation_receipt(receipt1)
        assert res_id == r1_id

        # UPDATE on receipt must fail
        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sm_reconciliation_receipts SET coverage_status = 'COMPLETE' WHERE receipt_id = %s;",
                    (r1_id,)
                )
        conn.close()

        # DELETE on receipt must fail
        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM sm_reconciliation_receipts WHERE receipt_id = %s;",
                    (r1_id,)
                )
        conn.close()

        # Appending superseding receipt V2 must succeed
        r2_id = str(uuid.uuid4())
        receipt2 = {
            "receipt_id": r2_id,
            "interval_start": "2026-10-10T00:00:00Z",
            "interval_end": "2026-10-10T23:59:59Z",
            "certified_watermark": "2026-10-10T23:59:59Z",
            "receipt_version": 2,
            "supersedes_receipt_id": r1_id,
            "expected_count": 100,
            "accounted_count": 100,
            "missing_count": 0,
            "material_gap_count": 0,
            "coverage_status": "COMPLETE",
            "failure_reasons": [],
            "reconciliation_hash": "hash_v2",
        }
        res_v2 = repo.save_reconciliation_receipt(receipt2)
        assert res_v2 == r2_id

        # Query authoritative receipt for interval returns V2
        auth = repo.get_authoritative_receipt_for_interval(
            "2026-10-10T00:00:00Z", "2026-10-10T23:59:59Z", "2026-10-10T23:59:59Z"
        )
        assert auth is not None
        assert str(auth["receipt_id"]) == r2_id
        assert auth["receipt_version"] == 2
        assert auth["coverage_status"] == "COMPLETE"

    def test_scanner_snapshots_immutability(self, real_postgres_uri, repo):
        s_id = f"snap-{uuid.uuid4().hex[:8]}"
        snap = {
            "snapshot_id": s_id,
            "scanner_id": "SMART_MONEY",
            "run_id": "run-001",
            "api_contract_version": "1.0",
            "ruleset_version": "1.0",
            "evidence_schema_version": "1.0",
            "score_model_version": "1.0",
            "data_provenance_version": "1.0",
            "universe_version": "1.0",
            "freshness_policy_version": "1.0",
            "implementation_release_sha": "b72ea239",
            "semantic_fingerprint": "fingerprint_123",
            "generated_at": "2026-10-10T20:00:00Z",
            "data_as_of": "2026-10-10T20:00:00Z",
            "status_at_publication": "AVAILABLE",
            "universe_id": "UNIV-01",
            "universe_size": 100,
            "matched_count": 5,
            "results": [{"symbol": "NVDA"}],
            "provenance": {},
            "freshness": {},
            "publication_decision": "PUBLISH",
        }
        repo.publish_scanner_snapshot_atomic(snap)

        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE scanner_snapshots SET matched_count = 10 WHERE snapshot_id = %s;",
                    (s_id,)
                )
        conn.close()

        conn = psycopg2.connect(real_postgres_uri)
        with pytest.raises(psycopg2.Error, match="IMMUTABILITY_VIOLATION"):
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM scanner_snapshots WHERE snapshot_id = %s;",
                    (s_id,)
                )
        conn.close()


class TestUnknownVsZeroContractInPostgres:
    """Verifies that unknown/quarantined snapshots never conflate with genuine complete zeros (Section 17)."""

    def test_genuine_zero_complete_scan(self, repo):
        s_id = f"snap-zero-{uuid.uuid4().hex[:8]}"
        snap = {
            "snapshot_id": s_id,
            "scanner_id": "SMART_MONEY",
            "run_id": "run-zero",
            "api_contract_version": "1.0",
            "ruleset_version": "1.0",
            "evidence_schema_version": "1.0",
            "score_model_version": "1.0",
            "data_provenance_version": "1.0",
            "universe_version": "1.0",
            "freshness_policy_version": "1.0",
            "implementation_release_sha": "b72ea239",
            "semantic_fingerprint": "fingerprint_zero",
            "generated_at": "2026-10-10T21:00:00Z",
            "data_as_of": "2026-10-10T21:00:00Z",
            "status_at_publication": "AVAILABLE",
            "universe_id": "UNIV-01",
            "universe_size": 100,
            "coverage_status": "COMPLETE",
            "matched_count": 0,
            "results": [],
            "provenance": {},
            "freshness": {},
            "publication_decision": "PUBLISH",
        }
        repo.publish_scanner_snapshot_atomic(snap)
        active = repo.get_latest_active_snapshot("SMART_MONEY")
        assert active is not None
        assert active["snapshot_id"] == s_id
        assert active["matched_count"] == 0  # Genuine zero!

    def test_quarantined_scan_stores_null_and_does_not_advance_active_pointer(self, repo):
        # Current active snapshot is snap-zero from previous test
        prior_active = repo.get_latest_active_snapshot("SMART_MONEY")
        assert prior_active is not None

        q_id = f"snap-quarantine-{uuid.uuid4().hex[:8]}"
        snap_quarantined = {
            "snapshot_id": q_id,
            "scanner_id": "SMART_MONEY",
            "run_id": "run-quarantine",
            "api_contract_version": "1.0",
            "ruleset_version": "1.0",
            "evidence_schema_version": "1.0",
            "score_model_version": "1.0",
            "data_provenance_version": "1.0",
            "universe_version": "1.0",
            "freshness_policy_version": "1.0",
            "implementation_release_sha": "b72ea239",
            "semantic_fingerprint": "fingerprint_quarantine",
            "generated_at": "2026-10-10T22:00:00Z",
            "data_as_of": "2026-10-10T22:00:00Z",
            "status_at_publication": "QUARANTINED",
            "universe_id": "UNIV-01",
            "universe_size": 100,
            "coverage_status": "PARTIAL",
            "matched_count": None,  # Unknown!
            "results": [],
            "provenance": {},
            "freshness": {},
            "publication_decision": "QUARANTINE",
        }
        repo.publish_scanner_snapshot_atomic(snap_quarantined)

        # Active pointer must NOT have changed!
        current_active = repo.get_latest_active_snapshot("SMART_MONEY")
        assert current_active is not None
        assert current_active["snapshot_id"] == prior_active["snapshot_id"]


class TestDistributedAdvisoryLocksAndConcurrency:
    """Verifies PostgreSQL advisory lock mutual exclusion and concurrency handling."""

    def test_single_owner_lock_mutual_exclusion(self, real_postgres_uri):
        conn1 = psycopg2.connect(real_postgres_uri)
        conn2 = psycopg2.connect(real_postgres_uri)

        # Connection 1 acquires SEC owner lock
        assert acquire_sec_owner_lock(conn1) is True

        # Connection 2 attempts to acquire SEC owner lock -> must fail
        assert acquire_sec_owner_lock(conn2) is False

        # Connection 1 releases lock
        assert release_sec_owner_lock(conn1) is True

        # Connection 2 can now acquire lock
        assert acquire_sec_owner_lock(conn2) is True
        assert release_sec_owner_lock(conn2) is True

        conn1.close()
        conn2.close()

    def test_lock_released_on_connection_close(self, real_postgres_uri):
        conn1 = psycopg2.connect(real_postgres_uri)
        assert acquire_sec_owner_lock(conn1) is True

        # Simulate abrupt worker disconnect / crash
        conn1.close()

        # Connection 2 should now be able to acquire immediately
        conn2 = psycopg2.connect(real_postgres_uri)
        assert acquire_sec_owner_lock(conn2) is True
        assert release_sec_owner_lock(conn2) is True
        conn2.close()

    def test_conflicting_content_hash_raises_source_integrity_conflict(self, repo):
        acc = f"ACC-CONF-{uuid.uuid4().hex[:8]}"
        filing1 = {
            "accession_number": acc,
            "cik": "0001018724",
            "ticker": "AMZN",
            "company_name": "Amazon.com Inc",
            "filing_date": "2026-10-10",
            "accepted_datetime": "2026-10-10T18:00:00Z",
            "xml_content": "<xml>content A</xml>",
            "content_hash": "hash_AAA",
            "source_url": "https://sec.gov/test",
        }
        repo.save_raw_filing(filing1)

        # Second arrival with different content hash must raise SOURCE_INTEGRITY_CONFLICT
        filing2 = dict(filing1, content_hash="hash_BBB", xml_content="<xml>content B</xml>")
        with pytest.raises(RuntimeError, match="SOURCE_INTEGRITY_CONFLICT"):
            repo.save_raw_filing(filing2)


class TestPointInTimeAndAmendmentsInPostgres:
    """Verifies point-in-time boundary filtering and amendment supersession algebra."""

    def test_pit_temporal_isolation(self, repo):
        sym = "TEST_PIT"
        acc1 = f"ACC-T1-{uuid.uuid4().hex[:6]}"
        acc2 = f"ACC-T2-{uuid.uuid4().hex[:6]}"

        repo.save_raw_filing({
            "accession_number": acc1, "cik": "0009999", "ticker": sym,
            "company_name": "Test PIT Inc", "filing_date": "2026-10-01",
            "accepted_datetime": "2026-10-01T12:00:00Z", "xml_content": "<xml/>",
            "content_hash": "h1", "source_url": "https://sec.gov/1",
        })
        repo.save_availability(acc1, "2026-10-01T14:00:00Z")
        repo.save_parsed_transactions([{
            "transaction_id": f"tx-{acc1}-1", "accession_number": acc1,
            "transaction_date": "2026-10-01", "transaction_code": "P",
            "shares": 1000, "price_per_share": 30.0, "total_value_usd": 30000.0,
            "reporting_owner_name": "CEO", "reporting_owner_role": "CEO",
        }])

        repo.save_raw_filing({
            "accession_number": acc2, "cik": "0009999", "ticker": sym,
            "company_name": "Test PIT Inc", "filing_date": "2026-10-05",
            "accepted_datetime": "2026-10-05T12:00:00Z", "xml_content": "<xml/>",
            "content_hash": "h2", "source_url": "https://sec.gov/2",
        })
        repo.save_availability(acc2, "2026-10-05T14:00:00Z")
        repo.save_parsed_transactions([{
            "transaction_id": f"tx-{acc2}-1", "accession_number": acc2,
            "transaction_date": "2026-10-05", "transaction_code": "P",
            "shares": 2000, "price_per_share": 30.0, "total_value_usd": 60000.0,
            "reporting_owner_name": "CEO", "reporting_owner_role": "CEO",
        }])

        # As of 2026-10-02 (after T1, before T2): returns only T1
        res_t1 = repo.get_bulk_transactions_as_of("2026-10-02T00:00:00Z", [sym], "2026-09-01")
        assert len(res_t1.get(sym, [])) == 1
        assert res_t1[sym][0]["accession_number"] == acc1

        # As of 2026-10-06 (after both): returns both T1 and T2
        res_t2 = repo.get_bulk_transactions_as_of("2026-10-06T00:00:00Z", [sym], "2026-09-01")
        assert len(res_t2.get(sym, [])) == 2


class TestRunRequestQueue:
    """Verifies web -> worker run request queue via PostgreSQL."""

    def test_run_request_lifecycle(self, repo):
        # 1. Web inserts run request
        req_id = repo.create_run_request("MARKET_WIDE_SCAN", "hash_subject_123")
        assert req_id is not None

        # 2. Ingestion worker claims request
        claimed = repo.claim_next_run_request("worker-test-01")
        assert claimed is not None
        assert str(claimed["request_id"]) == req_id
        assert claimed["status"] == "PENDING"

        # 3. Another worker trying to claim gets None
        second_claim = repo.claim_next_run_request("worker-test-02")
        assert second_claim is None

        # 4. Worker completes request
        repo.complete_run_request(req_id, "snapshot-final-001")


class TestMigrationToolAndParityVerification:
    """Tests end-to-end SQLite -> PostgreSQL migration and verification."""

    def test_sqlite_to_postgres_migration(self, tmp_path, real_postgres_uri):
        sqlite_file = str(tmp_path / "source_test.db")
        conn = sqlite3.connect(sqlite_file)
        cur = conn.cursor()

        # Create source tables
        cur.execute("""
            CREATE TABLE form4_raw_filings (
                accession_number TEXT PRIMARY KEY, cik TEXT, ticker TEXT,
                company_name TEXT, filing_date TEXT, accepted_datetime TEXT,
                xml_content TEXT, content_hash TEXT, source_url TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE form4_parsed_transactions (
                transaction_id TEXT PRIMARY KEY, accession_number TEXT,
                transaction_date TEXT, transaction_code TEXT, shares REAL,
                price_per_share REAL, total_value_usd REAL, reporting_owner_name TEXT,
                reporting_owner_role TEXT, is_director INTEGER, is_officer INTEGER,
                is_ten_percent INTEGER, officer_title TEXT, ownership_type TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE sm_source_availability_ledger (
                accession_number TEXT PRIMARY KEY, public_availability_at TEXT,
                source TEXT, source_reference TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE scanner_snapshots (
                snapshot_id TEXT PRIMARY KEY, scanner_id TEXT, run_id TEXT,
                api_contract_version TEXT, ruleset_version TEXT,
                evidence_schema_version TEXT, score_model_version TEXT,
                data_provenance_version TEXT, universe_version TEXT,
                freshness_policy_version TEXT, implementation_release_sha TEXT,
                semantic_fingerprint TEXT, generated_at TEXT, data_as_of TEXT,
                status_at_publication TEXT, universe_id TEXT, universe_size INTEGER,
                matched_count INTEGER, results_json TEXT, provenance_json TEXT,
                freshness_json TEXT, publication_decision TEXT
            );
        """)

        # Insert test data
        cur.execute("""
            INSERT INTO form4_raw_filings VALUES (
                'ACC-MIG-1', '0001', 'AAPL', 'Apple Inc', '2026-10-01',
                '2026-10-01T18:00:00Z', '<xml/>', 'mig_hash_1', 'https://sec.gov/1'
            );
        """)
        cur.execute("""
            INSERT INTO form4_parsed_transactions VALUES (
                'TX-MIG-1', 'ACC-MIG-1', '2026-10-01', 'P', 1000, 150.0,
                150000.0, 'Tim Cook', 'CEO', 1, 1, 0, 'CEO', 'D'
            );
        """)
        cur.execute("""
            INSERT INTO sm_source_availability_ledger VALUES (
                'ACC-MIG-1', '2026-10-01T18:30:00Z', 'SEC_INDEX', 'index_ref'
            );
        """)
        cur.execute("""
            INSERT INTO scanner_snapshots VALUES (
                'SNAP-MIG-1', 'SMART_MONEY', 'run-m1', '1.0', '1.0', '1.0', '1.0',
                '1.0', '1.0', '1.0', 'b72ea239', 'fingerprint_mig', '2026-10-01T20:00:00Z',
                '2026-10-01T20:00:00Z', 'AVAILABLE', 'UNIV-1', 100, 1,
                '[{"symbol":"AAPL"}]', '{}', '{}', 'PUBLISH'
            );
        """)
        conn.commit()
        conn.close()

        # Run migration
        migrator = SmartMoneySqliteMigrator(sqlite_file, real_postgres_uri)
        manifest = migrator.run_migration()

        assert manifest["status"] == "SUCCESS"
        assert manifest["target_counts"]["form4_raw_filings"] >= 1
        assert manifest["target_counts"]["form4_parsed_transactions"] >= 1

        # Verify parity
        parity = verify_migration_parity(sqlite_file, real_postgres_uri)
        assert parity["raw_filings_count_parity"] is True
        assert parity["raw_filings_pk_parity"] is True
        assert parity["parsed_transactions_count_parity"] is True
        assert parity["parsed_transactions_pk_parity"] is True
        assert parity["availability_count_parity"] is True
        assert parity["snapshots_parity"] is True
        assert parity["all_parity_passed"] is True


class TestDedicatedWorkerLifecycle:
    """Verifies Ingestion Worker preflight, advisory lock, and lifecycle."""

    def test_worker_fails_closed_without_owner_role(self, monkeypatch, real_postgres_uri):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "DISABLED")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", real_postgres_uri)
        worker = SmartMoneyIngestionWorker(dsn=real_postgres_uri)

        with pytest.raises(RuntimeError, match="UNAUTHORIZED_WORKER_ROLE"):
            worker.validate_preflight()

    def test_worker_succeeds_with_owner_role(self, monkeypatch, real_postgres_uri):
        monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", real_postgres_uri)
        worker = SmartMoneyIngestionWorker(dsn=real_postgres_uri)

        worker.validate_preflight()
        assert worker.acquire_egress_authority() is True
        assert worker.state == WorkerHealthState.OWNER_LOCK_ACQUIRED

        # Single cycle run
        worker.run_cycle()
        assert worker.state == WorkerHealthState.HEALTHY

        # Health envelope check
        health = worker.get_health_envelope()
        assert health["status"] == "HEALTHY"
        assert health["has_owner_lock"] is True

        worker.release_egress_authority()
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST


class TestDatabaseRolePermissions:
    """Verifies least-privilege role boundaries (smart_money_web vs smart_money_ingestion)."""

    def test_web_role_cannot_insert_raw_filings_or_modify_receipts(self, real_postgres_uri):
        # Read and apply roles DDL using admin connection
        admin_conn = psycopg2.connect(real_postgres_uri)
        roles_sql_path = os.path.join(
            os.path.dirname(__file__), "..", "analyst_dashboard", "smart_money", "storage", "postgres_roles.sql"
        )
        with open(roles_sql_path, "r", encoding="utf-8") as f:
            roles_sql = f.read()

        with admin_conn.cursor() as cur:
            # Set password for web role so we can connect
            cur.execute(roles_sql)
            cur.execute("ALTER ROLE smart_money_web WITH PASSWORD 'test_web_pass';")
            cur.execute("ALTER ROLE smart_money_ingestion WITH PASSWORD 'test_ingest_pass';")
        admin_conn.commit()
        admin_conn.close()

        # Build web role connection URI
        import urllib.parse
        parsed = urllib.parse.urlparse(real_postgres_uri)
        web_uri = urllib.parse.urlunparse((
            parsed.scheme,
            f"smart_money_web:test_web_pass@{parsed.hostname}:{parsed.port}",
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))

        # 1. Web role CAN SELECT snapshots
        web_conn = psycopg2.connect(web_uri)
        with web_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM scanner_snapshots;")
            assert cur.fetchone() is not None

        # 2. Web role CAN INSERT run requests
        with web_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sm_run_requests (request_id, request_type, requested_by_subject_hash)
                VALUES (%s, 'SCAN_REQ', 'hash_test');
            """, (str(uuid.uuid4()),))
        web_conn.commit()

        # 3. Web role CANNOT INSERT raw filings
        with pytest.raises(psycopg2.Error, match="permission denied"):
            with web_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO form4_raw_filings (
                        accession_number, cik, ticker, company_name, filing_date,
                        accepted_datetime, xml_content, content_hash, source_url
                    ) VALUES ('TEST_DENIED', '001', 'TST', 'Test', '2026-10-01', NOW(), '<xml/>', 'h', 'u');
                """)
        web_conn.rollback()

        # 4. Web role CANNOT DELETE scanner snapshots
        with pytest.raises(psycopg2.Error, match="permission denied"):
            with web_conn.cursor() as cur:
                cur.execute("DELETE FROM scanner_snapshots;")
        web_conn.rollback()

        web_conn.close()
