"""
ARX Terminal — Smart Money Form 4
Controlled Non-Production Fault-Injection Lifecycle Test (T0 - T10).

EVIDENCE CLASSIFICATION: LOCAL_SIMULATED_FAULT_INJECTION
This test runs deterministic in-memory simulations with LocalOffVolumeStorageAdapter,
MagicMock database connections, and simulated frozen clocks.
It proves contract conformance, state machine transitions, and repository gate wiring.
It does NOT constitute live PostgreSQL engine verification or real Cloudflare R2 evidence.
(Real PostgreSQL integration is independently executed in test_smart_money_backup_containment_postgres_integration.py).

Lifecycle Sequence:
T0: Initial state with fresh verified backup -> ingestion eligible, state HEALTHY
T1: Ingestion runs, repository mutations permitted under active gate, state remains HEALTHY
T2: Backup age approaches 12 hours -> warning state emitted (FRESHNESS_WARNING)
T3: Backup age exceeds 12 hours -> fail-closed containment trips (INGESTION_SUSPENDED)
T4: Repository mutations (save_raw_filing, save_parsed_transactions, publish_scanner_snapshot_atomic) strictly rejected
T5: Scheduled backup triggered, upload fails (simulated fault) -> containment remains tripped, state remains INGESTION_SUSPENDED
T6: Ingestion retry blocked -> verified fail-closed during active fault; zero unintended database mutations
T7: Subsequent backup succeeds, manifest written, dump verified -> recovery transitions to HEALTHY
T8: Containment gate re-evaluates -> idempotent repeated recovery confirmed
T9: SEC egress authority check -> worker confirms operational status
T10: Post-recovery ingestion cycle completes, deduplication logic skips ACC-001, persists ACC-002
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, call

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupFreshnessState,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    ContainmentState,
    BackupAgeContainmentViolationException,
    SnapshotPublicationBlockedContainmentException,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
)


def create_test_backup(
    adapter: LocalOffVolumeStorageAdapter,
    backup_id: str,
    completed_at_iso: str,
    started_at_iso: str = None,
    fail_upload: bool = False,
):
    with tempfile.TemporaryDirectory() as tmp:
        dump_name = f"smart_money_{backup_id}.dump"
        manifest_name = f"manifest_{backup_id}.json"

        if fail_upload:
            manifest = {
                "backup_id": backup_id,
                "started_at": started_at_iso or completed_at_iso,
                "completed_at": completed_at_iso,
                "upload_verified": False,
                "error": "Simulated R2 network timeout during chunk transfer",
            }
            manifest_local = os.path.join(tmp, manifest_name)
            with open(manifest_local, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            adapter.upload_file(manifest_local, manifest_name)
            return

        dump_local = os.path.join(tmp, dump_name)
        with open(dump_local, "wb") as f:
            f.write(f"PG16_DUMP_CONTENT_{backup_id}\n".encode("utf-8"))
        dump_sha = compute_sha256(dump_local)
        dump_size = os.path.getsize(dump_local)

        adapter.upload_file(dump_local, dump_name)

        manifest = {
            "backup_id": backup_id,
            "started_at": started_at_iso or (datetime.fromisoformat(completed_at_iso) - timedelta(minutes=2)).isoformat(),
            "completed_at": completed_at_iso,
            "database_artifact": dump_name,
            "dump_filename": dump_name,
            "artifact_sha256": dump_sha,
            "artifact_size_bytes": dump_size,
            "upload_verified": True,
            "postgres_major_version": 16,
            "schema_version": "SMART_MONEY_POSTGRES_SCHEMA_V2",
        }

        manifest_local = os.path.join(tmp, manifest_name)
        with open(manifest_local, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        adapter.upload_file(manifest_local, manifest_name)


def test_t0_to_t10_simulated_fault_injection_lifecycle(tmp_path):
    """
    Executes the full T0 - T10 lifecycle in an isolated test environment.
    Classification: LOCAL_SIMULATED_FAULT_INJECTION.
    """
    storage_dir = str(tmp_path / "fault_inject_backups")
    os.makedirs(storage_dir, exist_ok=True)
    adapter = LocalOffVolumeStorageAdapter(base_directory=storage_dir)

    policy = BackupContainmentPolicy(
        enabled=True,
        max_verified_age_seconds=43200.0,  # 12 Hours
        check_timeout_seconds=5.0,
        recheck_interval_seconds=60.0,
        clock_skew_tolerance_seconds=300.0,
        production_authorized=True,
        require_remote_checksum_metadata=True,
    )

    current_simulated_time = datetime(2026, 10, 14, 0, 0, 0, tzinfo=timezone.utc)

    def clock():
        return current_simulated_time

    authority = BackupFreshnessAuthority(
        storage_adapter=adapter,
        policy=policy,
        clock_fn=clock,
    )
    gate = BackupAgeContainmentGate(
        freshness_authority=authority,
        policy=policy,
        warning_threshold_ratio=0.8,
    )

    # Repository with mock database connection
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    repo = PostgresSmartMoneyRepository(dsn=None, containment_gate=gate)

    # --------------------------------------------------------------------------
    # T0: Initial state with fresh verified backup -> HEALTHY
    # --------------------------------------------------------------------------
    t0_backup_time = current_simulated_time - timedelta(hours=1)
    create_test_backup(
        adapter,
        "b-r2-20261013-230000",
        t0_backup_time.isoformat(),
        started_at_iso=(t0_backup_time - timedelta(minutes=3)).isoformat(),
    )
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    receipt_t0 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t0.is_fresh is True
    assert receipt_t0.backup_age_seconds == 3600.0
    assert receipt_t0.verification_status == BackupFreshnessState.FRESH.value

    # --------------------------------------------------------------------------
    # T1: Ingestion runs, repository mutations permitted under active gate
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(hours=2)
    filing_acc_001 = {
        "accession_number": "0001437749-26-000001",
        "cik": "0001018724",
        "ticker": "AMZN",
        "company_name": "Amazon Com Inc",
        "filing_date": "2026-10-14",
        "accepted_datetime": "2026-10-14 02:00:00",
        "xml_content": "<xml>content_001</xml>",
        "content_hash": "hash_acc_001",
        "source_url": "https://sec.gov/Archives/edgar/data/0001/0001.txt",
    }
    tx_acc_001 = [{
        "transaction_id": "tx-001",
        "accession_number": "0001437749-26-000001",
        "transaction_date": "2026-10-14",
        "transaction_code": "P",
        "shares": 1000,
        "price_per_share": 150.0,
        "total_value_usd": 150000.0,
        "reporting_owner_name": "Test Insider",
        "reporting_owner_role": "Director",
        "is_director": True,
        "is_officer": False,
        "is_ten_percent": False,
        "officer_title": None,
        "ownership_type": "D",
    }]

    mock_cur.reset_mock()
    inserted_filing = repo.save_raw_filing(filing_acc_001, conn=mock_conn)
    assert inserted_filing is True
    saved_tx = repo.save_parsed_transactions(tx_acc_001, conn=mock_conn)
    assert saved_tx == 1
    assert mock_cur.execute.called

    # --------------------------------------------------------------------------
    # T2: Backup age approaches 12 hours -> warning state emitted
    # --------------------------------------------------------------------------
    current_simulated_time = datetime(2026, 10, 14, 9, 30, 0, tzinfo=timezone.utc)  # Age: 10.5h
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.FRESHNESS_WARNING
    receipt_t2 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t2.backup_age_seconds == 37800.0

    # --------------------------------------------------------------------------
    # T3: Backup age exceeds 12 hours -> fail-closed containment trips
    # --------------------------------------------------------------------------
    current_simulated_time = datetime(2026, 10, 14, 11, 5, 0, tzinfo=timezone.utc)  # Age: 12h 05m
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is False
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED
    receipt_t3 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t3.is_fresh is False
    assert receipt_t3.reason_code == "RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD"

    # --------------------------------------------------------------------------
    # T4: Subsequent ingestion attempts blocked, database mutations rejected
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=5)
    mock_cur.reset_mock()

    # Ingestion gate check directly fails
    with pytest.raises(BackupAgeContainmentViolationException):
        gate.enforce_ingestion_gate(current_time=current_simulated_time)

    # Repository save_raw_filing fails closed
    filing_blocked = {"accession_number": "0001437749-26-000099", "content_hash": "hash_99"}
    with pytest.raises(BackupAgeContainmentViolationException):
        repo.save_raw_filing(filing_blocked, conn=mock_conn)

    # Repository save_parsed_transactions fails closed
    with pytest.raises(BackupAgeContainmentViolationException):
        repo.save_parsed_transactions([{"transaction_id": "tx-blocked"}], conn=mock_conn)

    # Snapshot publication fails closed
    with pytest.raises(SnapshotPublicationBlockedContainmentException):
        repo.publish_scanner_snapshot_atomic(
            snapshot={"snapshot_id": "snap-blocked", "run_id": "r1", "semantic_fingerprint": "fp"},
            conn=mock_conn,
        )

    # Verify zero database mutation attempts while suspended
    assert not mock_cur.execute.called
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    # --------------------------------------------------------------------------
    # T5: Scheduled backup triggered, upload fails (simulated fault) -> containment remains tripped
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=5)
    gate.mark_backup_in_progress()
    assert gate.current_state == ContainmentState.BACKUP_RECOVERY_PENDING

    # Staging corrupt/failed upload manifest
    create_test_backup(
        adapter,
        "b-r2-20261014-111500-failed",
        current_simulated_time.isoformat(),
        fail_upload=True,
    )
    gate.mark_backup_completed()

    # Re-evaluating: newest backup failed, older backup is 12h 15m old (>12h) -> MUST remain suspended
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is False
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    # --------------------------------------------------------------------------
    # T6: Ingestion retry blocked -> verified fail-closed during active fault
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=5)
    mock_cur.reset_mock()
    with pytest.raises(BackupAgeContainmentViolationException):
        repo.save_raw_filing(filing_blocked, conn=mock_conn)
    assert not mock_cur.execute.called

    # --------------------------------------------------------------------------
    # T7: Subsequent backup succeeds, manifest written, dump verified -> recovery transitions
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=10)
    gate.mark_backup_in_progress()
    new_backup_id = "b-r2-20261014-112500-success"
    create_test_backup(
        adapter,
        new_backup_id,
        current_simulated_time.isoformat(),
        started_at_iso=(current_simulated_time - timedelta(minutes=2)).isoformat(),
        fail_upload=False,
    )
    gate.mark_backup_completed()

    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    receipt_t7 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t7.is_fresh is True
    assert receipt_t7.recovery_point_id == new_backup_id

    # --------------------------------------------------------------------------
    # T8: Idempotent repeated recovery confirmed
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=1)
    resumption_count_before = gate.resumption_count
    assert gate.check_ingestion_eligibility(force_refresh=False, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    assert gate.resumption_count == resumption_count_before  # No duplicate resumption events

    # --------------------------------------------------------------------------
    # T9: SEC egress authority check -> worker confirms operational status
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=1)
    # Simulated worker authority re-check
    gate.enforce_ingestion_gate(current_time=current_simulated_time)
    assert gate.current_state == ContainmentState.HEALTHY

    # --------------------------------------------------------------------------
    # T10: Post-recovery ingestion cycle completes, deduplication skips ACC-001, saves ACC-002
    # --------------------------------------------------------------------------
    current_simulated_time += timedelta(minutes=3)
    mock_cur.reset_mock()

    # Replay of ACC-001: mock DB returns existing row -> idempotent skip (returns False)
    mock_cur.fetchone.return_value = {"content_hash": "hash_acc_001"}
    duplicate_saved = repo.save_raw_filing(filing_acc_001, conn=mock_conn)
    assert duplicate_saved is False

    # Insert new ACC-002: mock DB returns None -> saved (returns True)
    mock_cur.fetchone.return_value = None
    filing_acc_002 = {
        "accession_number": "0001437749-26-000002",
        "cik": "0000320193",
        "ticker": "AAPL",
        "company_name": "Apple Inc",
        "filing_date": "2026-10-14",
        "accepted_datetime": "2026-10-14 11:30:00",
        "xml_content": "<xml>content_002</xml>",
        "content_hash": "hash_acc_002",
        "source_url": "https://sec.gov/Archives/edgar/data/0002/0002.txt",
    }
    new_saved = repo.save_raw_filing(filing_acc_002, conn=mock_conn)
    assert new_saved is True
    assert mock_cur.execute.called
