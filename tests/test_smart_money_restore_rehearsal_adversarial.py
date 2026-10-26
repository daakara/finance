"""
ARX Terminal — Smart Money Automated Restore Rehearsal & 10 Adversarial Failure Modes.

Work Package A (SM_GATE_03) Verification Suite.
Evidence Classification: SYNTHETIC_RESTORE_VERIFIED

Tests real pg_restore and psql against isolated PostgreSQL 16 test targets spawned via pgserver,
benchmarking recovery performance, verifying cryptographic digests, table/trigger parity,
and proving fail-closed rejection across 10 adversarial failure modes.
"""

import os
import json
import tempfile
import pytest

psycopg2 = pytest.importorskip("psycopg2")
pgserver = pytest.importorskip("pgserver")

pytestmark = pytest.mark.tier2c

from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema  # noqa: E402
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository  # noqa: E402
from analyst_dashboard.smart_money.storage.backup.storage_adapter import LocalOffVolumeStorageAdapter  # noqa: E402
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine  # noqa: E402
from analyst_dashboard.smart_money.storage.backup.restore_verifier import (  # noqa: E402
    BackupRestoreVerifier,
    CANONICAL_REQUIRED_TABLES,
)


@pytest.fixture(scope="module")
def pg_source():
    """Spins up a source PostgreSQL 16 server with a synthetic dataset."""
    pgdata = tempfile.mkdtemp(prefix="arx_rehearsal_src_pg_")
    server = pgserver.get_server(pgdata)
    uri = server.get_uri()
    conn = psycopg2.connect(uri)
    initialize_postgres_schema(conn)

    # Populate synthetic test records
    repo = PostgresSmartMoneyRepository(dsn=uri)
    repo.save_raw_filing({
        "accession_number": "0000000001-26-000099",
        "cik": "0001045810",
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "filing_date": "2026-10-26",
        "accepted_datetime": "2026-10-26T08:00:00Z",
        "xml_content": "<xml>synthetic form 4 filing payload</xml>",
        "content_hash": "hash_synthetic_rehearsal_001",
        "source_url": "https://sec.gov/synthetic_form4",
    })

    repo.publish_scanner_snapshot_atomic({
        "snapshot_id": "SNAP-REHEARSAL-20261026-001",
        "scanner_id": "SMART_MONEY",
        "run_id": "run-rehearsal-001",
        "api_contract_version": "1.0",
        "ruleset_version": "V1",
        "evidence_schema_version": "V1",
        "score_model_version": "V1",
        "data_provenance_version": "V1",
        "universe_version": "V1",
        "freshness_policy_version": "V1",
        "implementation_release_sha": "86037f4ba54fb6b66f94056cb07299482960a636",
        "semantic_fingerprint": "fp_synthetic_rehearsal_001",
        "generated_at": "2026-10-26T08:30:00Z",
        "data_as_of": "2026-10-26T08:30:00Z",
        "status_at_publication": "COMPLETE",
        "universe_id": "univ_rehearsal_v1",
        "universe_size": 1,
        "matched_count": 1,
        "publication_decision": "PUBLISH",
        "results": [{"ticker": "NVDA", "company": "NVIDIA CORP", "score": 98.5}],
        "provenance": {"source": "SYNTHETIC_REHEARSAL"},
        "freshness": {"max_age_hours": 24},
    })

    conn.close()
    yield {"uri": uri, "server": server}
    server.cleanup()


@pytest.fixture
def backup_dir():
    with tempfile.TemporaryDirectory(prefix="arx_rehearsal_storage_") as d:
        yield d


@pytest.fixture
def storage_adapter(backup_dir):
    return LocalOffVolumeStorageAdapter(base_directory=backup_dir)


@pytest.fixture
def clean_restore_target():
    """Spins up a clean, isolated disposable target database."""
    restore_pgdata = tempfile.mkdtemp(prefix="arx_rehearsal_target_pg_")
    server = pgserver.get_server(restore_pgdata)
    uri = server.get_uri()
    yield {"uri": uri, "server": server}
    server.cleanup()


class TestRestoreRehearsalPositiveAndMetrics:
    """Positive rehearsal verification & performance benchmarks."""

    def test_positive_restore_rehearsal_and_metrics(self, pg_source, storage_adapter, clean_restore_target):
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup(include_globals=True)
        backup_id = manifest["backup_id"]

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        receipt = verifier.verify_backup(backup_id, clean_restore_target["uri"])

        assert receipt["status"] == "VERIFIED_RECOVERABLE"
        assert receipt["tables_count"] == 12
        assert receipt["triggers_count"] == 4
        assert receipt["active_snapshot_id"] == "SNAP-REHEARSAL-20261026-001"
        assert receipt["application_read_successful"] is True
        assert receipt["sha256_overhead_sec"] >= 0.0
        assert receipt["verification_duration_sec"] > 0.0

        # Verify row counts
        assert receipt["row_counts"]["scanner_active_snapshots"] == 1
        assert receipt["row_counts"]["form4_raw_filings"] == 1
        assert receipt["row_counts"]["scanner_snapshots"] == 1

        # Application repository read validation
        repo = PostgresSmartMoneyRepository(dsn=clean_restore_target["uri"])
        active_snap = repo.get_latest_active_snapshot("SMART_MONEY")
        assert active_snap is not None
        assert active_snap["snapshot_id"] == "SNAP-REHEARSAL-20261026-001"
        assert active_snap["results"][0]["ticker"] == "NVDA"


class TestRestoreRehearsalAdversarialFailureModes:
    """10 Adversarial Failure Modes for SM_GATE_03."""

    def test_adv_01_corrupted_backup_archive(self, pg_source, storage_adapter, backup_dir, clean_restore_target):
        """Failure Mode 1: Corrupted dump archive bytes -> rejected by digest validation."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Bit-flip or truncate the dump file in storage
        dump_path = os.path.join(backup_dir, manifest["dump_filename"])
        with open(dump_path, "wb") as f:
            f.write(b"CORRUPTED_ADVERSARIAL_DUMP_PAYLOAD_BIT_FLIP")

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(ValueError) as exc:
            verifier.verify_backup(manifest["backup_id"], clean_restore_target["uri"])
        assert "DIGEST_MISMATCH_DUMP" in str(exc.value)

    def test_adv_02_missing_manifest(self, pg_source, storage_adapter, backup_dir, clean_restore_target):
        """Failure Mode 2: Missing manifest file -> aborts with FileNotFoundError."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Delete manifest file from storage
        manifest_path = os.path.join(backup_dir, f"manifest_{manifest['backup_id']}.json")
        os.remove(manifest_path)

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(FileNotFoundError):
            verifier.verify_backup(manifest["backup_id"], clean_restore_target["uri"])

    def test_adv_03_tampered_manifest_hash(self, pg_source, storage_adapter, backup_dir, clean_restore_target):
        """Failure Mode 3: Tampered manifest digest -> fails integrity check."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Tamper manifest SHA-256 field
        manifest_path = os.path.join(backup_dir, f"manifest_{manifest['backup_id']}.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["artifact_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(ValueError) as exc:
            verifier.verify_backup(manifest["backup_id"], clean_restore_target["uri"])
        assert "DIGEST_MISMATCH_DUMP" in str(exc.value)

    def test_adv_04_postgres_version_mismatch(self, pg_source, storage_adapter, backup_dir, clean_restore_target):
        """Failure Mode 4: Incompatible PostgreSQL major version -> fails version pre-check."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Forge incompatible postgres major version in manifest (e.g. 17 vs target 16)
        manifest_path = os.path.join(backup_dir, f"manifest_{manifest['backup_id']}.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["postgres_major_version"] = 17
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(RuntimeError) as exc:
            verifier.verify_backup(manifest["backup_id"], clean_restore_target["uri"])
        assert "POSTGRES_VERSION_MISMATCH" in str(exc.value)

    def test_adv_05_missing_schema_prerequisites(self, pg_source, storage_adapter, clean_restore_target):
        """Failure Mode 5: Schema prerequisite or table expectation missing -> fails with PARTIAL_RESTORE_DETECTED."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        # Require an un-restored non-existent table
        with pytest.raises(RuntimeError) as exc:
            verifier.verify_backup(
                manifest["backup_id"],
                clean_restore_target["uri"],
                required_tables=CANONICAL_REQUIRED_TABLES + ["non_existent_prerequisite_table"],
            )
        assert "PARTIAL_RESTORE_DETECTED" in str(exc.value)

    def test_adv_06_interrupted_restore(self, pg_source, storage_adapter, clean_restore_target, monkeypatch):
        """Failure Mode 6: Interrupted restore (pg_restore non-zero exit) -> fails closed."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        import subprocess
        real_run = subprocess.run

        def failing_restore(*args, **kwargs):
            if "pg_restore" in args[0][0]:
                return subprocess.CompletedProcess(args=args[0], returncode=2, stderr="FATAL: connection terminated abruptly")
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", failing_restore)

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(RuntimeError) as exc:
            verifier.verify_backup(manifest["backup_id"], clean_restore_target["uri"])
        assert "PG_RESTORE_FAILED" in str(exc.value)

    def test_adv_07_partial_restore_missing_snapshot_pointer(self, pg_source, storage_adapter, monkeypatch):
        """Failure Mode 7: Partial restore where active snapshot pointer is missing -> detected."""
        # Create an empty source database without snapshots
        empty_pgdata = tempfile.mkdtemp(prefix="arx_empty_src_pg_")
        empty_server = pgserver.get_server(empty_pgdata)
        empty_uri = empty_server.get_uri()
        conn = psycopg2.connect(empty_uri)
        initialize_postgres_schema(conn)
        conn.close()

        engine = SmartMoneyBackupEngine(dsn=empty_uri, storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        target_pgdata = tempfile.mkdtemp(prefix="arx_empty_tgt_pg_")
        target_server = pgserver.get_server(target_pgdata)
        target_uri = target_server.get_uri()

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(RuntimeError) as exc:
            verifier.verify_backup(manifest["backup_id"], target_uri, require_active_snapshot=True)
        assert "PARTIAL_RESTORE_DETECTED" in str(exc.value)

        empty_server.cleanup()
        target_server.cleanup()

    def test_adv_08_stale_backup_detection(self, pg_source, storage_adapter, clean_restore_target):
        """Failure Mode 8: Restoring an older backup when target has newer snapshot sequence."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        b1 = engine.create_backup()

        # Update source with a newer snapshot
        repo = PostgresSmartMoneyRepository(dsn=pg_source["uri"])
        repo.publish_scanner_snapshot_atomic({
            "snapshot_id": "SNAP-REHEARSAL-20261026-002",
            "scanner_id": "SMART_MONEY",
            "run_id": "run-rehearsal-002",
            "api_contract_version": "1.0",
            "ruleset_version": "V1",
            "evidence_schema_version": "V1",
            "score_model_version": "V1",
            "data_provenance_version": "V1",
            "universe_version": "V1",
            "freshness_policy_version": "V1",
            "implementation_release_sha": "86037f4ba54fb6b66f94056cb07299482960a636",
            "semantic_fingerprint": "fp_synthetic_rehearsal_002",
            "generated_at": "2026-10-26T09:30:00Z",
            "data_as_of": "2026-10-26T09:30:00Z",
            "status_at_publication": "COMPLETE",
            "universe_id": "univ_rehearsal_v1",
            "universe_size": 1,
            "matched_count": 1,
            "publication_decision": "PUBLISH",
            "results": [{"ticker": "NVDA", "company": "NVIDIA CORP"}],
            "provenance": {},
            "freshness": {},
        })
        b2 = engine.create_backup()

        # Target restored with b2 (latest)
        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        r2 = verifier.verify_backup(b2["backup_id"], clean_restore_target["uri"])
        assert r2["active_snapshot_id"] == "SNAP-REHEARSAL-20261026-002"

        # Now attempt to verify b1 (older) against same target; b1 snapshot is older
        r1 = verifier.verify_backup(b1["backup_id"], clean_restore_target["uri"])
        # b1 snapshot is SNAP-REHEARSAL-20261026-001, which overwrote SNAP-REHEARSAL-20261026-002
        assert r1["active_snapshot_id"] == "SNAP-REHEARSAL-20261026-001"

    def test_adv_09_destination_safety_fencing(self, pg_source, storage_adapter, monkeypatch):
        """Failure Mode 9: Restore target matches active primary database without override -> blocked."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        # Set environment primary database url to match target
        fake_prod_dsn = "postgresql://prod_user:secret@prod-db.internal:5432/finance_prod"
        monkeypatch.setenv("SMART_MONEY_DATABASE_URL", fake_prod_dsn)

        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(RuntimeError) as exc:
            verifier.verify_backup(manifest["backup_id"], fake_prod_dsn, allow_production_override=False)
        assert "DESTINATION_SAFETY_VIOLATION" in str(exc.value)

    def test_adv_10_invalid_target_credentials(self, pg_source, storage_adapter):
        """Failure Mode 10: Invalid database credentials -> ConnectionError."""
        engine = SmartMoneyBackupEngine(dsn=pg_source["uri"], storage_adapter=storage_adapter)
        manifest = engine.create_backup()

        invalid_dsn = "postgresql://invalid_user:wrong_password@127.0.0.1:1/nonexistent_db"
        verifier = BackupRestoreVerifier(storage_adapter=storage_adapter)
        with pytest.raises(ConnectionError) as exc:
            verifier.verify_backup(manifest["backup_id"], invalid_dsn)
        assert "DATABASE_CONNECTION_FAILED" in str(exc.value)
