"""
Comprehensive Integration Tests for Smart Money PostgreSQL Canonical Runtime Integration.

Covers:
1. Web snapshot read path uses PostgreSQL when configured (S1 in Postgres, SQLite empty).
2. Zero authoritative SQLite connections opened when ARX_STORAGE_BACKEND=postgres.
3. Operator scan enqueues into sm_run_requests with HTTP 202 in PostgreSQL profile.
4. Worker claims request via FOR UPDATE SKIP LOCKED without race collisions.
5. Migration conflict fails closed (hash divergence -> SOURCE_INTEGRITY_CONFLICT & rollback).
6. Pinned major-version target constants (PostgreSQL 16, postgres:16-alpine).
"""

import os
import uuid
import json
import tempfile
import sqlite3
import pytest

pytestmark = pytest.mark.tier2c
from datetime import datetime
psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor, Json
from fastapi import Response
pgserver = pytest.importorskip("pgserver")

from analyst_dashboard.smart_money.storage.config import (
    SMART_MONEY_POSTGRES_SCHEMA_VERSION,
    POSTGRES_MAJOR_VERSION_SUPPORTED,
    SMART_MONEY_POSTGRES_MAJOR,
    PRODUCTION_TARGET_POSTGRES_MAJOR,
    RAILWAY_POSTGRES_IMAGE_TARGET,
    get_storage_backend,
)
from analyst_dashboard.smart_money.storage.factory import (
    get_smart_money_storage_backend,
    get_smart_money_repository,
    get_smart_money_snapshot_store,
    get_smart_money_scanner_runner,
)
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.migration import (
    SmartMoneySqliteMigrator,
    SourceIntegrityConflictError,
)
from api.routes.screener import (
    get_operator_token,
    get_smart_money_snapshot,
    trigger_smart_money_scan,
    run_screener_get,
)


@pytest.fixture(scope="module")
def real_postgres_uri():
    """Starts an isolated PostgreSQL 16 server in a temporary directory."""
    pgdata = tempfile.mkdtemp(prefix="arx_test_runtime_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)
    conn.close()

    yield uri

    server.cleanup()


@pytest.fixture
def repo(real_postgres_uri):
    return PostgresSmartMoneyRepository(dsn=real_postgres_uri)


class TestPinnedMajorVersionConstants:
    """Verifies pinned PostgreSQL major version constants and container image targets."""

    def test_pinned_postgres_version_constants(self):
        assert SMART_MONEY_POSTGRES_MAJOR == 16
        assert PRODUCTION_TARGET_POSTGRES_MAJOR == 16
        assert POSTGRES_MAJOR_VERSION_SUPPORTED == 16
        assert RAILWAY_POSTGRES_IMAGE_TARGET == "postgres:16-alpine"
        assert SMART_MONEY_POSTGRES_SCHEMA_VERSION == "SMART_MONEY_POSTGRES_SCHEMA_V1"


class TestWebSnapshotReadPathPostgres:
    """Verifies that web API read path serves active snapshots directly from PostgreSQL."""

    def test_web_snapshot_reads_postgres_when_configured(self, real_postgres_uri, repo, monkeypatch):
        # Configure Postgres profile
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", real_postgres_uri)

        snap_id = f"SNAP-PG-TEST-{uuid.uuid4().hex[:8]}"
        run_id = f"RUN-{uuid.uuid4().hex[:8]}"
        snapshot_envelope = {
            "snapshot_id": snap_id,
            "scanner_id": "SMART_MONEY",
            "run_id": run_id,
            "api_contract_version": "1.0",
            "ruleset_version": "FORM4_RADAR_QUALIFICATION_V1",
            "evidence_schema_version": "1.0",
            "score_model_version": "1.0",
            "data_provenance_version": "1.0",
            "universe_version": "UNIVERSE_v1",
            "freshness_policy_version": "1.0",
            "implementation_release_sha": "b72ea23937cf6ed7b8c4b60096630767a7c4592b",
            "semantic_fingerprint": "fingerprint_123",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "data_as_of": "2026-10-10T23:59:59Z",
            "status_at_publication": "AVAILABLE",
            "universe_id": "GLOBAL_SEC_DISCOVERY_V1",
            "universe_size": 500,
            "matched_count": 1,
            "results_json": [
                {
                    "symbol": "NVDA",
                    "convictionScore": 95,
                    "signalStatus": "QUALIFIED_INSIDER_ACCUMULATION",
                    "insiderBuyerCount": 2,
                    "grossPurchaseValueUSD": 1500000.0,
                }
            ],
            "provenance_json": {"source": "SEC_EDGAR_DIRECT"},
            "freshness_json": {"latency_seconds": 12.5},
            "publication_decision": "PUBLISH",
        }

        # Publish snapshot into PostgreSQL
        repo.publish_scanner_snapshot_atomic(snapshot_envelope)

        # Call web snapshot endpoint
        resp = get_smart_money_snapshot()

        assert resp is not None
        assert resp["status"] == "AVAILABLE"
        assert resp["snapshot"]["snapshot_id"] == snap_id
        assert resp["snapshot"]["matched_count"] == 1
        assert len(resp["results"]) == 1
        assert resp["results"][0]["symbol"] == "NVDA"


class TestZeroAuthoritativeSqliteOpened:
    """Verifies that under PostgreSQL profile, zero SQLite connection files are initialized for Smart Money."""

    def test_zero_sqlite_connections_in_postgres_profile(self, real_postgres_uri, monkeypatch):
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", real_postgres_uri)

        # Intercept sqlite3.connect to assert no smart money databases are connected
        original_connect = sqlite3.connect

        def guarded_sqlite_connect(database, *args, **kwargs):
            db_str = str(database)
            if "smart_money" in db_str or "scanner_snapshots" in db_str:
                raise AssertionError(f"FORBIDDEN_SQLITE_CALL: Attempted to open SQLite database: {db_str}")
            return original_connect(database, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", guarded_sqlite_connect)

        runner = get_smart_money_scanner_runner()
        assert isinstance(runner.snapshot_store, PostgresSmartMoneyRepository)
        assert runner.coordinator is None

        # Fetch snapshot
        snapshot = runner.get_active_or_latest_snapshot()
        assert snapshot is not None


class TestOperatorScanAsynchronousEnqueue:
    """Verifies that triggering scan in PostgreSQL profile enqueues a run request with HTTP 202."""

    def test_operator_scan_enqueues_202_accepted(self, real_postgres_uri, monkeypatch):
        monkeypatch.setenv("ARX_STORAGE_BACKEND", "postgres")
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", real_postgres_uri)
        monkeypatch.setenv("ARX_OPERATOR_TOKEN", "valid-test-operator-secret-123")

        resp_obj = Response()
        result = trigger_smart_money_scan(
            response=resp_obj,
            x_operator_token="valid-test-operator-secret-123",
            x_operator_request_id="req-test-enqueue-001",
            as_of_watermark="2026-10-10T23:59:59Z",
            universe_build_id="build-test-01",
        )

        assert resp_obj.status_code == 202
        assert result["status"] == "ACCEPTED"
        assert result["mode"] == "ASYNCHRONOUS_QUEUE"
        req_id = result["request_id"]
        assert req_id is not None

        # Verify record in PostgreSQL sm_run_requests
        conn = psycopg2.connect(real_postgres_uri)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM sm_run_requests WHERE request_id = %s;", (req_id,))
            row = cur.fetchone()
            assert row is not None
            assert str(row["request_id"]) == str(req_id)
            assert row["request_type"] == "MARKET_WIDE_FORM4_SCAN"
            assert row["status"] == "PENDING"
        conn.close()


class TestWorkerClaimSkipLocked:
    """Verifies that worker claims pending requests via FOR UPDATE SKIP LOCKED without collision."""

    def test_worker_claims_request_via_skip_locked(self, real_postgres_uri, repo):
        # Drain any existing pending requests from previous tests to ensure clean queue state
        while True:
            drain = repo.claim_next_run_request(worker_id="drain-worker")
            if not drain:
                break
            repo.complete_run_request(drain["request_id"], status="DRAINED")

        req_id = repo.create_run_request(
            request_type="MARKET_WIDE_FORM4_SCAN",
            subject_hash="subject_skip_locked_test",
        )
        assert req_id is not None

        # First worker claims request
        claimed_1 = repo.claim_next_run_request(worker_id="worker-node-1")
        assert claimed_1 is not None
        assert claimed_1["request_id"] == req_id
        assert claimed_1["status"] == "PENDING"

        # Verify DB row transitioned to RUNNING
        db_row = repo.get_run_request(req_id)
        assert db_row is not None
        assert db_row["status"] == "RUNNING"
        assert db_row["claimed_by"] == "worker-node-1"

        # Second worker attempts to claim concurrently -> returns None (no pending request available)
        claimed_2 = repo.claim_next_run_request(worker_id="worker-node-2")
        assert claimed_2 is None

        # Worker completes request
        repo.complete_run_request(req_id, status="COMPLETED")
        status_row = repo.get_run_request(req_id)
        assert status_row is not None
        assert status_row["status"] == "COMPLETED"


class TestMigrationConflictFailClosed:
    """Verifies that SQLite to PostgreSQL migration fails closed on hash divergence."""

    def test_migration_conflict_divergent_hash_fails_closed(self, real_postgres_uri, tmp_path):
        # 1. Create SQLite DB with an accession and hash A
        sqlite_file = tmp_path / "legacy_source_conflict.db"
        sqlite_conn = sqlite3.connect(str(sqlite_file))
        sqlite_conn.execute("""
            CREATE TABLE form4_raw_filings (
                accession_number TEXT PRIMARY KEY,
                cik TEXT NOT NULL,
                ticker TEXT,
                company_name TEXT,
                filing_date TEXT NOT NULL,
                accepted_datetime TEXT NOT NULL,
                xml_content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_url TEXT
            );
        """)
        conflict_acc = "0001045810-26-999999"
        source_hash = "source_hash_aaa111"
        sqlite_conn.execute("""
            INSERT INTO form4_raw_filings VALUES (
                ?, '0001045810', 'NVDA', 'NVIDIA CORP', '2026-10-10', '2026-10-10T18:00:00Z', '<xml>source</xml>', ?, 'url'
            );
        """, (conflict_acc, source_hash))
        sqlite_conn.commit()
        sqlite_conn.close()

        # 2. In PostgreSQL, insert same accession with hash B
        pg_conn = psycopg2.connect(real_postgres_uri)
        existing_hash = "existing_pg_hash_bbb222"
        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date,
                    accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (
                conflict_acc, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-10",
                "2026-10-10T18:00:00Z", "<xml>target</xml>", existing_hash, "url"
            ))
        pg_conn.commit()
        pg_conn.close()

        # 3. Run migration -> must raise SourceIntegrityConflictError and record conflict manifest
        migrator = SmartMoneySqliteMigrator(
            sqlite_db_path=str(sqlite_file),
            pg_dsn=real_postgres_uri,
        )

        with pytest.raises(SourceIntegrityConflictError, match="SOURCE_INTEGRITY_CONFLICT"):
            migrator.run_migration()

        assert migrator.last_manifest is not None
        assert migrator.last_manifest["status"] == "SOURCE_INTEGRITY_CONFLICT"
        assert migrator.last_manifest["conflict_count"] == 1
        assert conflict_acc in migrator.last_manifest["conflicting_accessions"]

        # Verify PostgreSQL target record remains untouched
        pg_conn2 = psycopg2.connect(real_postgres_uri)
        with pg_conn2.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT content_hash FROM form4_raw_filings WHERE accession_number = %s;", (conflict_acc,))
            row = cur.fetchone()
            assert row is not None
            assert row["content_hash"] == existing_hash, "Target record must not be corrupted by failed migration"
        pg_conn2.close()

    def test_migration_identical_hash_succeeds_idempotently(self, real_postgres_uri, tmp_path):
        sqlite_file = tmp_path / "legacy_source_identical.db"
        sqlite_conn = sqlite3.connect(str(sqlite_file))
        sqlite_conn.execute("""
            CREATE TABLE form4_raw_filings (
                accession_number TEXT PRIMARY KEY,
                cik TEXT NOT NULL,
                ticker TEXT,
                company_name TEXT,
                filing_date TEXT NOT NULL,
                accepted_datetime TEXT NOT NULL,
                xml_content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_url TEXT
            );
        """)
        same_acc = "0001045810-26-888888"
        same_hash = "identical_hash_ccc333"
        sqlite_conn.execute("""
            INSERT INTO form4_raw_filings VALUES (
                ?, '0001045810', 'NVDA', 'NVIDIA CORP', '2026-10-10', '2026-10-10T18:00:00Z', '<xml>identical</xml>', ?, 'url'
            );
        """, (same_acc, same_hash))
        sqlite_conn.commit()
        sqlite_conn.close()

        # In PostgreSQL, insert same accession with same hash
        pg_conn = psycopg2.connect(real_postgres_uri)
        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date,
                    accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
            """, (
                same_acc, "0001045810", "NVDA", "NVIDIA CORP", "2026-10-10",
                "2026-10-10T18:00:00Z", "<xml>identical</xml>", same_hash, "url"
            ))
        pg_conn.commit()
        pg_conn.close()

        migrator = SmartMoneySqliteMigrator(
            sqlite_db_path=str(sqlite_file),
            pg_dsn=real_postgres_uri,
        )
        manifest = migrator.run_migration()
        assert manifest["status"] == "SUCCESS"
        assert manifest["conflict_count"] == 0
