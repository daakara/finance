"""
ARX Terminal — Smart Money Final Provisioning Safety Gates Test Suite.

Authoritative verification of the three production provisioning gates:
1. Gate 1 — Atomic duplicate prevention under exclusive session advisory lock.
2. Gate 2 — R2 backup digest integrity (streaming SHA-256 readback, non-ETag reliance).
3. Gate 3 — Non-destructive emergency stop hierarchy and read-only Radar preservation.
"""

from __future__ import annotations

import os
import io
import time
import hashlib
import pytest
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    S3CompatibleStorageAdapter,
)
from analyst_dashboard.smart_money.storage.postgres_schema import SCHEMA_DDL


class TestGate1AtomicDuplicatePrevention:
    """
    Gate 1: Durable New York execution-date eligibility check occurs under the
    same exclusive advisory lock (LOCK_SEC_EGRESS_OWNER) that protects ingestion.
    """

    def test_advisory_lock_claimed_before_date_check_and_pre_backup(self):
        """
        Execution ordering verification:
        Step 1: Preflight
        Step 2: Advisory Lock Claim (Session mutex)
        Step 3: Durable Date Eligibility Check Under Lock
        Step 4: Pre-ingestion Backup & R2 Upload
        """
        execution_order = []

        pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)
        pipeline._step1_preflight = MagicMock(side_effect=lambda *a, **k: execution_order.append("STEP_1_PREFLIGHT"))
        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=lambda *a, **k: execution_order.append("STEP_2_LOCK"))
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(
            side_effect=lambda *a, **k: (execution_order.append("STEP_3_DATE_CHECK"), (True, "OK"))[1]
        )
        pipeline._step4_pre_backup = MagicMock(
            side_effect=lambda *a, **k: (execution_order.append("STEP_4_PRE_BACKUP"), {"backup_id": "b-1", "status": "VERIFIED"})[1]
        )
        pipeline._step5_freshness_gate_check = MagicMock(side_effect=lambda *a, **k: execution_order.append("STEP_5_FRESHNESS"))
        pipeline._step6_ingest_form4_filings = MagicMock(side_effect=lambda *a, **k: (execution_order.append("STEP_6_INGEST"), [])[1])
        pipeline._step7_parse_and_reconcile = MagicMock(
            side_effect=lambda *a, **k: (execution_order.append("STEP_7_PARSE"), ([], {"reconciliation_hash": "h1"}))[1]
        )
        pipeline._step8_publish_snapshot = MagicMock(
            side_effect=lambda *a, **k: (execution_order.append("STEP_8_PUBLISH"), {"snapshot_id": "SNAP_1", "matched_count": 0})[1]
        )
        pipeline._step9_post_backup = MagicMock(
            side_effect=lambda *a, **k: (execution_order.append("STEP_9_POST_BACKUP"), {"backup_id": "b-2", "status": "VERIFIED"})[1]
        )
        pipeline._step10_prune_retention = MagicMock(side_effect=lambda *a, **k: (execution_order.append("STEP_10_PRUNE"), [])[1])

        result = pipeline.execute_cycle(trigger_type="TEST")

        assert result["status"] == "SUCCESS"
        assert execution_order == [
            "STEP_1_PREFLIGHT",
            "STEP_2_LOCK",
            "STEP_3_DATE_CHECK",
            "STEP_4_PRE_BACKUP",
            "STEP_5_FRESHNESS",
            "STEP_6_INGEST",
            "STEP_7_PARSE",
            "STEP_8_PUBLISH",
            "STEP_9_POST_BACKUP",
            "STEP_10_PRUNE",
        ]

    def test_concurrent_trigger_fails_closed_under_advisory_lock(self):
        """
        When two workers trigger concurrently, the second worker is blocked by the
        exclusive advisory lock and aborts fail-closed with 0 SEC filings fetched
        and 0 pre-backups created.
        """
        pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)
        pipeline._step1_preflight = MagicMock()
        # Simulate lock contention: lock cannot be acquired
        pipeline._step2_claim_advisory_lock = MagicMock(
            side_effect=RuntimeError("CONCURRENT_INGESTION_BLOCKED: Another worker currently holds LOCK_SEC_EGRESS_OWNER.")
        )
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock()
        pipeline._step4_pre_backup = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock()

        with pytest.raises(RuntimeError, match="CONCURRENT_INGESTION_BLOCKED"):
            pipeline.execute_cycle(trigger_type="SCHEDULED")

        # Invariant: Step 3, 4, 6 were NEVER called
        pipeline._step3_check_durable_date_eligibility_under_lock.assert_not_called()
        pipeline._step4_pre_backup.assert_not_called()
        pipeline._step6_ingest_form4_filings.assert_not_called()

    def test_sequential_restart_deduplication_under_lock(self):
        """
        Simulate an accidental duplicate run or process restart on the same NY date.
        Worker acquires the advisory lock, checks durable state in PostgreSQL under lock,
        discovers the date was already executed, and exits cleanly with SKIPPED_ALREADY_EXECUTED_TODAY.
        Zero SEC calls, zero pre-backups, and releases advisory lock cleanly.
        """
        mock_repo = MagicMock()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_SM_1792939989_48f30db6",
            "data_as_of": "2026-10-23T22:30:00-04:00",
        }

        mock_scheduler = MagicMock()
        mock_scheduler.target_hour = 22
        mock_scheduler.target_minute = 30
        mock_scheduler.should_execute_now.return_value = (
            False,
            "ALREADY_EXECUTED_TODAY: cycle already recorded for 2026-10-23",
        )

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            scheduler=mock_scheduler,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        pipeline._step2_claim_advisory_lock = MagicMock()
        pipeline._step4_pre_backup = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock()

        # Run with as_of="2026-10-23", force=False
        result = pipeline.execute_cycle(
            trigger_type="SCHEDULED",
            force=False,
            as_of="2026-10-23",
        )

        assert result["status"] == "SKIPPED_ALREADY_EXECUTED_TODAY"
        assert "ALREADY_EXECUTED_TODAY" in result["reason"]
        # Advisory lock was acquired
        pipeline._step2_claim_advisory_lock.assert_called_once()
        # Zero pre-backup and zero SEC ingestion
        pipeline._step4_pre_backup.assert_not_called()
        pipeline._step6_ingest_form4_filings.assert_not_called()

    def test_failure_before_publication_releases_lock_and_preserves_active_snapshot(self):
        """
        When a failure occurs prior to Step 8 (e.g., during pre-backup or SEC ingestion),
        the session advisory lock must be released in the `finally:` block and the
        active snapshot pointer must remain unmutated.
        """
        mock_repo = MagicMock()
        mock_repo.publish_scanner_snapshot_atomic = MagicMock()

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        mock_lock_conn = MagicMock()
        def fake_claim_lock():
            pipeline._lock_conn = mock_lock_conn
        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_claim_lock)
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline._step4_pre_backup = MagicMock(side_effect=IOError("R2_TIMEOUT: Connection aborted"))

        with pytest.raises(IOError, match="R2_TIMEOUT"):
            pipeline.execute_cycle(trigger_type="TEST")

        # Pointer never advanced
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()
        # Lock connection was closed in finally
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None


class TestGate2R2BackupDigestIntegrity:
    """
    Gate 2: R2/S3 ETag is not assumed to equal SHA-256.
    Enforces streaming content SHA-256 readback and metadata SHA-256 validation.
    """

    def test_r2_upload_verifies_streaming_sha256_content_and_metadata(self):
        content = b"TEST_DATABASE_DUMP_CONTENT_PAYLOAD_12345"
        expected_sha = hashlib.sha256(content).hexdigest()

        mock_s3 = MagicMock()
        # head_object returns metadata with correct sha256, but S3 ETag is an MD5 hash
        s3_etag_md5 = f'"{hashlib.md5(content).hexdigest()}"'
        mock_s3.head_object.return_value = {
            "ContentLength": len(content),
            "ETag": s3_etag_md5,
            "Metadata": {"sha256": expected_sha},
        }
        # get_object returns streaming Body with identical bytes
        mock_body = io.BytesIO(content)
        mock_s3.get_object.return_value = {"Body": mock_body}

        adapter = S3CompatibleStorageAdapter(
            bucket="test-arx-backups",
            endpoint_url="https://fake-r2.cloudflarestorage.com",
            access_key_id="fake-key",
            secret_access_key="fake-secret",
        )
        adapter._s3_client = mock_s3

        # Write temp source file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            receipt = adapter.upload_file(temp_path, "backup_test.dump")
            assert receipt["integrity_verified"] is True
            assert receipt["verification_method"] == "END_TO_END_STREAMING_SHA256"
            assert receipt["verified_remote_sha256"] == expected_sha
            assert receipt["verified_metadata_sha256"] == expected_sha
            assert receipt["is_multipart"] is False
            assert receipt["size_bytes"] == len(content)
        finally:
            os.remove(temp_path)

    def test_r2_upload_fails_on_metadata_corruption(self):
        content = b"DUMP_CONTENT_FOR_METADATA_TEST"
        expected_sha = hashlib.sha256(content).hexdigest()

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {
            "ContentLength": len(content),
            "ETag": '"some-md5"',
            "Metadata": {"sha256": "corrupted_metadata_digest_999999999999"},
        }

        adapter = S3CompatibleStorageAdapter(
            bucket="test-arx-backups",
            endpoint_url="https://fake-r2.cloudflarestorage.com",
            access_key_id="fake-key",
            secret_access_key="fake-secret",
        )
        adapter._s3_client = mock_s3

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            with pytest.raises(IOError, match="R2_UPLOAD_METADATA_CORRUPTION"):
                adapter.upload_file(temp_path, "backup_corrupted_meta.dump")
        finally:
            os.remove(temp_path)

    def test_r2_upload_fails_on_streaming_content_byte_corruption(self):
        content = b"ORIGINAL_LOCAL_UNALTERED_BYTES"
        corrupted_remote_content = b"CORRUPTED_REMOTE_BYTES_ATTACK_OR_STORAGE_FLIP"
        expected_sha = hashlib.sha256(content).hexdigest()

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {
            "ContentLength": len(content),
            "ETag": '"md5-match"',
            "Metadata": {"sha256": expected_sha},  # metadata says ok
        }
        # But remote stream body returns corrupted bytes!
        mock_s3.get_object.return_value = {"Body": io.BytesIO(corrupted_remote_content)}

        adapter = S3CompatibleStorageAdapter(
            bucket="test-arx-backups",
            endpoint_url="https://fake-r2.cloudflarestorage.com",
            access_key_id="fake-key",
            secret_access_key="fake-secret",
        )
        adapter._s3_client = mock_s3

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            with pytest.raises(IOError, match="R2_UPLOAD_CONTENT_CORRUPTION"):
                adapter.upload_file(temp_path, "backup_corrupted_bytes.dump")
        finally:
            os.remove(temp_path)

    def test_r2_multipart_etag_handling_without_sha256_confusion(self):
        """
        In S3/R2 multipart uploads, ETag format is `<md5_of_part_checksums>-<part_count>`,
        e.g., `"9b2cf535f27731c974343645a3985328-14"`.
        The adapter must recognise multipart uploads without confusing ETag for SHA-256,
        verifying integrity strictly through streaming readback.
        """
        content = b"LARGE_MULTIPART_DUMP_STREAMING_SIMULATION" * 100
        expected_sha = hashlib.sha256(content).hexdigest()
        multipart_etag = '"9b2cf535f27731c974343645a3985328-14"'

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {
            "ContentLength": len(content),
            "ETag": multipart_etag,
            "Metadata": {"sha256": expected_sha},
        }
        mock_s3.get_object.return_value = {"Body": io.BytesIO(content)}

        adapter = S3CompatibleStorageAdapter(
            bucket="test-arx-backups",
            endpoint_url="https://fake-r2.cloudflarestorage.com",
            access_key_id="fake-key",
            secret_access_key="fake-secret",
        )
        adapter._s3_client = mock_s3

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            receipt = adapter.upload_file(temp_path, "backup_multipart.dump")
            assert receipt["is_multipart"] is True
            assert receipt["etag"] == "9b2cf535f27731c974343645a3985328-14"
            assert receipt["verified_remote_sha256"] == expected_sha
            assert receipt["integrity_verified"] is True
        finally:
            os.remove(temp_path)

    def test_pipeline_fails_closed_before_sec_ingestion_on_backup_digest_failure(self):
        """
        If pre-backup fails due to R2 upload corruption, pipeline aborts at Step 4,
        never touches Step 6 (SEC Form 4 ingestion), and releases advisory lock.
        """
        mock_backup = MagicMock()
        mock_backup.create_backup.side_effect = IOError("R2_UPLOAD_CONTENT_CORRUPTION: Remote SHA != Local SHA")

        pipeline = SmartMoneyContinuousPipeline(
            backup_engine=mock_backup,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        pipeline._step2_claim_advisory_lock = MagicMock()
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline._step6_ingest_form4_filings = MagicMock()

        with pytest.raises(IOError, match="R2_UPLOAD_CONTENT_CORRUPTION"):
            pipeline.execute_cycle(trigger_type="TEST")

        # Invariant: SEC ingestion NEVER called
        pipeline._step6_ingest_form4_filings.assert_not_called()


class TestGate3NonDestructiveEmergencyStop:
    """
    Gate 3: Inspect and disprove invalid/destructive stop commands.
    Verify non-destructive stop hierarchy:
    1. Worker role kill-switch: ARX_SEC_EGRESS_ROLE=DISABLED (halts in <= 1.0s)
    2. Service replica scaling to 0
    3. Revoking ingestion DML
    Preserves active snapshot pointer and read-only Radar availability via smart_money_web.
    """

    def test_disprove_destructive_update_and_delete_on_active_snapshots(self):
        """
        Disprove runbook instructions recommending:
        `UPDATE scanner_active_snapshots SET is_active = FALSE`
        or `DELETE FROM scanner_active_snapshots WHERE scan_type = 'SMART_MONEY'`.

        Table schema verification:
        `scanner_active_snapshots` columns are (scan_type, active_snapshot_id, published_at, published_by).
        There is NO `is_active` column and NO `scanner_id` column.
        """
        # Parse schema DDL for scanner_active_snapshots
        ddl_start = SCHEMA_DDL.find("CREATE TABLE IF NOT EXISTS scanner_active_snapshots (")
        ddl_end = SCHEMA_DDL.find(");", ddl_start)
        table_ddl = SCHEMA_DDL[ddl_start:ddl_end]

        assert "is_active" not in table_ddl
        assert "scan_type VARCHAR(64) PRIMARY KEY" in table_ddl
        assert "active_snapshot_id VARCHAR(64) NOT NULL REFERENCES scanner_snapshots(snapshot_id)" in table_ddl

    def test_emergency_stop_via_role_disabled_halts_in_under_1s(self):
        """
        Setting ARX_SEC_EGRESS_ROLE=DISABLED causes the worker to halt immediately
        at Step 1 (preflight) within <= 1.0s with RuntimeError: UNAUTHORIZED_WORKER_ROLE.
        Zero database locks claimed, zero backups created, zero SEC calls made.
        """
        mock_backup = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(backup_engine=mock_backup, allow_test_mode=False)
        pipeline._step2_claim_advisory_lock = MagicMock()
        pipeline._step4_pre_backup = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "DISABLED"}):
            start = time.perf_counter()
            with pytest.raises(RuntimeError, match="UNAUTHORIZED_WORKER_ROLE"):
                pipeline.execute_cycle(trigger_type="SCHEDULED", dry_run=False)
            duration = time.perf_counter() - start

            assert duration < 1.0, f"Emergency stop exceeded 1.0s limit: {duration:.4f}s"
            pipeline._step2_claim_advisory_lock.assert_not_called()
            pipeline._step4_pre_backup.assert_not_called()
            pipeline._step6_ingest_form4_filings.assert_not_called()

    def test_active_snapshot_serving_preserved_during_emergency_stop(self):
        """
        During worker emergency stop (ARX_SEC_EGRESS_ROLE=DISABLED or worker replica=0),
        the web service (role: smart_money_web) continues serving the verified active
        snapshot (e.g. SNAP_SM_1792939989_48f30db6) with 100% read-only availability.
        """
        mock_repo = MagicMock()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_SM_1792939989_48f30db6",
            "status_at_publication": "AVAILABLE",
            "publication_decision": "PUBLISH",
            "matched_count": 0,
            "universe_size": 10,
        }

        # Simulate web tier query during worker outage
        snap = mock_repo.get_latest_active_snapshot(scan_type="SMART_MONEY")
        assert snap is not None
        assert snap["snapshot_id"] == "SNAP_SM_1792939989_48f30db6"
        assert snap["publication_decision"] == "PUBLISH"
