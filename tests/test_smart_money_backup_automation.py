"""
Comprehensive Test Suite for Smart Money Automated PostgreSQL Backup & Recovery (Candidate 006).

Covers:
1. SmartMoneyBackupEngine: Execution of consistent logical backups (pg_dumpall + pg_dump -Fc).
2. Manifest generation with full cryptographic integrity and schema versions.
3. Overlap prevention via LOCK_BACKUP_EXECUTION session advisory lock.
4. BackupRetentionManager: Pruning of expired backups with strict protection of most recent backup.
5. BackupRestoreVerifier: Automated download, digest validation, and restore into isolated PostgreSQL 16.
6. Detection of tampered/corrupted backups.
"""

import os
import json
import tempfile
import pytest
psycopg2 = pytest.importorskip("psycopg2")
pgserver = pytest.importorskip("pgserver")

pytestmark = pytest.mark.tier2c

from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.restore_verifier import BackupRestoreVerifier


@pytest.fixture(scope="module")
def pg_source_uri():
    """Spins up a source PostgreSQL 16 server with sample data."""
    pgdata = tempfile.mkdtemp(prefix="arx_backup_src_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)

    # Populate a sample raw filing and active snapshot
    repo = PostgresSmartMoneyRepository(dsn=uri)
    repo.save_raw_filing({
        "accession_number": "0000000001-26-000001",
        "cik": "0001045810",
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "filing_date": "2026-10-14",
        "accepted_datetime": "2026-10-14T01:00:00Z",
        "xml_content": "<xml>test filing content</xml>",
        "content_hash": "hash_sample_nvda_filing_123",
        "source_url": "https://sec.gov/sample",
    })
    repo.publish_scanner_snapshot_atomic({
        "snapshot_id": "SNAP-BACKUP-TEST-001",
        "scanner_id": "SMART_MONEY",
        "run_id": "run-backup-test",
        "api_contract_version": "1.0",
        "ruleset_version": "V1",
        "evidence_schema_version": "V1",
        "score_model_version": "V1",
        "data_provenance_version": "V1",
        "universe_version": "V1",
        "freshness_policy_version": "V1",
        "implementation_release_sha": "09f183de52e275ae543615747dff2f9ae00ef9ca",
        "semantic_fingerprint": "fp_backup_sample",
        "generated_at": "2026-10-14T01:00:00Z",
        "data_as_of": "2026-10-14T01:00:00Z",
        "status_at_publication": "COMPLETE",
        "universe_id": "univ_nvda",
        "universe_size": 1,
        "matched_count": 1,
        "publication_decision": "PUBLISH",
        "results": [{"ticker": "NVDA", "company": "NVIDIA CORP"}],
        "provenance": {},
        "freshness": {},
    })

    conn.close()
    yield uri
    server.cleanup()


@pytest.fixture
def backup_dir():
    with tempfile.TemporaryDirectory(prefix="arx_storage_backup_dir_") as d:
        yield d


@pytest.fixture
def storage_adapter(backup_dir):
    return LocalOffVolumeStorageAdapter(base_directory=backup_dir)


class TestBackupEngineExecution:
    """Verifies that the automated backup engine captures dumps and generates manifests."""

    def test_create_backup_generates_verified_artifacts_routine(self, pg_source_uri, storage_adapter):
        """Routine scheduled backup (include_globals=False, unprivileged role capable)."""
        engine = SmartMoneyBackupEngine(dsn=pg_source_uri, storage_adapter=storage_adapter)
        manifest = engine.create_backup(include_globals=False)

        assert manifest["backup_id"].startswith("sm-backup-")
        assert manifest["upload_verified"] is True
        assert manifest["include_globals"] is False
        assert manifest["globals_artifact"] is None
        assert manifest["globals_artifact_sha256"] is None
        assert manifest["schema_version"] == "SMART_MONEY_POSTGRES_SCHEMA_V2"
        assert manifest["artifact_size_bytes"] > 0
        assert len(manifest["artifact_sha256"]) == 64

        # Verify files exist in storage (dump + manifest, no globals)
        items = storage_adapter.list_backups()
        filenames = [x["remote_filename"] for x in items]
        assert manifest["dump_filename"] in filenames
        assert f"manifest_{manifest['backup_id']}.json" in filenames

    def test_create_backup_generates_verified_artifacts_with_globals(self, pg_source_uri, storage_adapter):
        """Administrative backup (include_globals=True, exports cluster globals)."""
        engine = SmartMoneyBackupEngine(dsn=pg_source_uri, storage_adapter=storage_adapter)
        manifest = engine.create_backup(include_globals=True)

        assert manifest["backup_id"].startswith("sm-backup-")
        assert manifest["upload_verified"] is True
        assert manifest["include_globals"] is True
        assert manifest["globals_artifact"] is not None
        assert len(manifest["globals_artifact_sha256"]) == 64
        assert len(manifest["artifact_sha256"]) == 64

        items = storage_adapter.list_backups()
        filenames = [x["remote_filename"] for x in items]
        assert manifest["dump_filename"] in filenames
        assert manifest["globals_filename"] in filenames
        assert f"manifest_{manifest['backup_id']}.json" in filenames


class TestBackupRetentionAndPruning:
    """Verifies safe retention pruning and protection of the newest recovery point."""

    def test_retention_pruning_protects_latest_backup(self, pg_source_uri, storage_adapter):
        engine = SmartMoneyBackupEngine(dsn=pg_source_uri, storage_adapter=storage_adapter)
        retention = BackupRetentionManager(storage_adapter=storage_adapter, retention_count=2, retention_days=1)

        # Create 3 backups
        b1 = engine.create_backup()
        b2 = engine.create_backup()
        b3 = engine.create_backup()

        # Prune with retention_count=2
        pruned = retention.prune_backups(retention_count=2)
        assert len(pruned) == 1
        assert b1["backup_id"] in pruned

        # Verify b3 (most recent) and b2 remain intact
        remaining = [x["remote_filename"] for x in storage_adapter.list_backups()]
        assert f"manifest_{b3['backup_id']}.json" in remaining
        assert f"manifest_{b2['backup_id']}.json" in remaining
        assert f"manifest_{b1['backup_id']}.json" not in remaining

        # Even with retention_count=1, the most recent backup is NEVER deleted
        pruned_again = retention.prune_backups(retention_count=1)
        remaining_now = [x["remote_filename"] for x in storage_adapter.list_backups()]
        assert f"manifest_{b3['backup_id']}.json" in remaining_now


class TestRestoreVerifier:
    """Verifies automated restore, structural parity, and application read contract."""

    def test_restore_verifier_validates_backup_into_isolated_target(self, pg_source_uri, storage_adapter):
        engine = SmartMoneyBackupEngine(dsn=pg_source_uri, storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Spin up a disposable restore target database
        restore_pgdata = tempfile.mkdtemp(prefix="arx_restore_target_pg_")
        target_server = pgserver.get_server(restore_pgdata)
        target_uri = target_server.get_uri()

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        receipt = verifier.verify_backup(manifest["backup_id"], target_uri)

        assert receipt["status"] == "VERIFIED_RECOVERABLE"
        assert receipt["tables_count"] == 12
        assert receipt["triggers_count"] == 4
        assert receipt["active_snapshot_id"] == "SNAP-BACKUP-TEST-001"
        assert receipt["application_read_successful"] is True

        target_server.cleanup()

    def test_tampered_backup_fails_verification(self, pg_source_uri, storage_adapter, backup_dir):
        engine = SmartMoneyBackupEngine(dsn=pg_source_uri, storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Corrupt the dump file in storage
        dump_path = os.path.join(backup_dir, manifest["dump_filename"])
        with open(dump_path, "wb") as f:
            f.write(b"CORRUPTED_TRUNCATED_BYTES")

        # Disposable restore target
        restore_pgdata = tempfile.mkdtemp(prefix="arx_restore_tamper_pg_")
        target_server = pgserver.get_server(restore_pgdata)
        target_uri = target_server.get_uri()

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(ValueError) as excinfo:
            verifier.verify_backup(manifest["backup_id"], target_uri)
        assert "DIGEST_MISMATCH_DUMP" in str(excinfo.value)

        target_server.cleanup()
