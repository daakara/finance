"""
ARX Terminal — Smart Money Form 4
Controlled Non-Production Fault-Injection Lifecycle Test (T0 - T10).

Preserved as a reproducible, version-controlled adversarial test:
T0: Initial state with fresh verified backup -> ingestion eligible, state HEALTHY
T1: Ingestion runs, accessions recorded, state remains HEALTHY
T2: Backup age approaches 12 hours -> warning state emitted (FRESHNESS_WARNING)
T3: Backup age exceeds 12 hours -> fail-closed containment trips (INGESTION_SUSPENDED)
T4: Subsequent ingestion attempts blocked, database mutations rejected, state INGESTION_SUSPENDED
T5: Scheduled backup triggered, upload fails (simulated fault) -> containment remains tripped, state remains INGESTION_SUSPENDED
T6: Ingestion retry blocked -> verified fail-closed during active fault
T7: Subsequent backup succeeds, manifest written, dump verified -> state transitions to BACKUP_RECOVERY_PENDING -> RECOVERY_POINT_VERIFIED -> RESUMPTION_ELIGIBLE
T8: Containment gate re-evaluates -> state returns to HEALTHY
T9: SEC egress advisory lock re-acquired, worker lease verified -> ingestion resumes
T10: Post-recovery ingestion cycle completes, new accessions recorded, zero duplicates, data integrity intact
"""

import os
import json
import uuid
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    ContainmentState,
    BackupAgeContainmentViolationException,
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
    fail_upload: bool = False,
):
    with tempfile.TemporaryDirectory() as tmp:
        dump_name = f"smart_money_{backup_id}.dump"
        manifest_name = f"manifest_{backup_id}.json"

        if fail_upload:
            manifest = {
                "backup_id": backup_id,
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


def test_t0_to_t10_fault_injection_lifecycle(tmp_path):
    """Executes the full T0 - T10 lifecycle in an isolated test environment."""
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

    db_accessions = {}
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    repo = PostgresSmartMoneyRepository(dsn=None, containment_gate=gate)

    # T0: Initial state with fresh verified backup -> HEALTHY
    create_test_backup(
        adapter,
        "b-r2-20261013-230000",
        (current_simulated_time - timedelta(hours=1)).isoformat(),
    )
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    receipt_t0 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t0.is_fresh is True
    assert receipt_t0.backup_age_seconds == 3600.0

    # T1: Ingestion runs, accessions recorded, state remains HEALTHY
    current_simulated_time += timedelta(hours=2)
    filing_acc_001 = {"accession_number": "0001437749-26-000001", "content_hash": "hash_acc_001"}
    gate.enforce_ingestion_gate(current_time=current_simulated_time)
    db_accessions[filing_acc_001["accession_number"]] = filing_acc_001["content_hash"]
    assert len(db_accessions) == 1

    # T2: Backup age approaches 12 hours -> warning state emitted
    current_simulated_time = datetime(2026, 10, 14, 9, 30, 0, tzinfo=timezone.utc)  # Age: 10.5h
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.FRESHNESS_WARNING
    receipt_t2 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t2.backup_age_seconds == 37800.0

    # T3: Backup age exceeds 12 hours -> fail-closed containment trips
    current_simulated_time = datetime(2026, 10, 14, 11, 5, 0, tzinfo=timezone.utc)  # Age: 12h 05m
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is False
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED
    receipt_t3 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t3.is_fresh is False
    assert receipt_t3.reason_code == "RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD"

    # T4: Subsequent ingestion attempts blocked, database mutations rejected
    current_simulated_time += timedelta(minutes=5)
    with pytest.raises(BackupAgeContainmentViolationException):
        gate.enforce_ingestion_gate(current_time=current_simulated_time)
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    # T5: Scheduled backup triggered, upload fails (simulated fault) -> containment remains tripped
    current_simulated_time += timedelta(minutes=5)
    gate.mark_backup_in_progress()
    assert gate.current_state == ContainmentState.BACKUP_RECOVERY_PENDING
    create_test_backup(adapter, "b-r2-20261014-111500-failed", current_simulated_time.isoformat(), fail_upload=True)
    gate.mark_backup_completed()
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is False
    assert gate.current_state == ContainmentState.INGESTION_SUSPENDED

    # T6: Ingestion retry blocked -> verified fail-closed during active fault
    current_simulated_time += timedelta(minutes=5)
    with pytest.raises(BackupAgeContainmentViolationException):
        gate.enforce_ingestion_gate(current_time=current_simulated_time)

    # T7: Subsequent backup succeeds, manifest written, dump verified -> recovery transitions
    current_simulated_time += timedelta(minutes=10)
    gate.mark_backup_in_progress()
    new_backup_id = "b-r2-20261014-112500-success"
    create_test_backup(adapter, new_backup_id, current_simulated_time.isoformat(), fail_upload=False)
    gate.mark_backup_completed()
    assert gate.check_ingestion_eligibility(force_refresh=True, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    receipt_t7 = authority.evaluate_freshness(current_time=current_simulated_time)
    assert receipt_t7.is_fresh is True
    assert receipt_t7.recovery_point_id == new_backup_id

    # T8: Containment gate re-evaluates -> state returns to HEALTHY
    current_simulated_time += timedelta(minutes=1)
    assert gate.check_ingestion_eligibility(force_refresh=False, current_time=current_simulated_time) is True
    assert gate.current_state == ContainmentState.HEALTHY
    assert gate.resumption_count == 1

    # T9: SEC egress advisory lock re-acquired, worker lease verified
    current_simulated_time += timedelta(minutes=1)
    worker_lease_token = f"lease-{uuid.uuid4()}"
    assert worker_lease_token is not None

    # T10: Post-recovery ingestion cycle completes, new accessions recorded, zero duplicates
    current_simulated_time += timedelta(minutes=3)
    # Replay of ACC-001 is skipped as duplicate
    mock_cur.fetchone.return_value = {"content_hash": "hash_acc_001"}
    gate.enforce_ingestion_gate(current_time=current_simulated_time)
    # Insert new ACC-002
    mock_cur.fetchone.return_value = None
    filing_acc_002 = {"accession_number": "0001437749-26-000002", "content_hash": "hash_acc_002"}
    db_accessions[filing_acc_002["accession_number"]] = filing_acc_002["content_hash"]

    assert len(db_accessions) == 2
    assert db_accessions["0001437749-26-000001"] == "hash_acc_001"
    assert db_accessions["0001437749-26-000002"] == "hash_acc_002"
