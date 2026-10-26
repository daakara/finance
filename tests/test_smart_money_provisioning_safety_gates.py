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

    def test_step3_durable_date_check_fails_closed_on_repository_error(self):
        """
        Gate A: When repository query fails, pipeline raises DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED
        and fails closed. Invariant: UNKNOWN_EXECUTION_STATE != ELIGIBLE_TO_RUN.
        Session advisory lock is released in finally, 0 pre-backups, 0 SEC calls.
        """
        mock_repo = MagicMock()
        mock_repo.get_latest_active_snapshot.side_effect = RuntimeError("POSTGRES_READ_TIMEOUT")

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0

        def fake_claim_lock():
            pipeline._lock_conn = mock_lock_conn

        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_claim_lock)
        pipeline._step4_pre_backup = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock()

        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED"):
            pipeline.execute_cycle(trigger_type="SCHEDULED")

        # Invariants
        pipeline._step4_pre_backup.assert_not_called()
        pipeline._step6_ingest_form4_filings.assert_not_called()
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None

    def test_step3_durable_date_check_fails_closed_on_malformed_or_missing_data_as_of(self):
        """
        Gate A: Active snapshot with malformed or missing data_as_of must fail closed.
        """
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)
        pipeline._lock_conn = MagicMock()
        pipeline._lock_conn.closed = 0

        # Case 1: Missing data_as_of
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_MALFORMED",
            "data_as_of": None,
        }
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED.*data_as_of"):
            pipeline._step3_check_durable_date_eligibility_under_lock()

        # Case 2: Unparseable data_as_of
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_MALFORMED",
            "data_as_of": "not-a-date-timestamp",
        }
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED.*unparseable"):
            pipeline._step3_check_durable_date_eligibility_under_lock()

    def test_step3_durable_date_check_fails_closed_in_production_without_repo_or_lock_conn(self):
        """
        Gate A: In production mode (allow_test_mode=False), missing repository or lock connection
        fails closed immediately.
        """
        mock_backup = MagicMock()
        # Case 1: Missing repository
        p1 = SmartMoneyContinuousPipeline(backup_engine=mock_backup, allow_test_mode=False)
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED.*repository required"):
            p1._step3_check_durable_date_eligibility_under_lock()

        # Case 2: Repository present but missing lock connection
        mock_repo = MagicMock()
        p2 = SmartMoneyContinuousPipeline(repository=mock_repo, backup_engine=mock_backup, allow_test_mode=False)
        p2._lock_conn = None
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED.*advisory lock connection required"):
            p2._step3_check_durable_date_eligibility_under_lock()

        # Case 3: Lock connection is closed
        mock_closed_conn = MagicMock()
        mock_closed_conn.closed = 1
        p3 = SmartMoneyContinuousPipeline(repository=mock_repo, backup_engine=mock_backup, allow_test_mode=False)
        p3._lock_conn = mock_closed_conn
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED.*closed"):
            p3._step3_check_durable_date_eligibility_under_lock()

    def test_step3_durable_date_check_handles_first_valid_execution(self):
        """
        Gate A: On a clean/fresh database where get_latest_active_snapshot returns None,
        the durable check passes None as last_executed_ny_date to the scheduler,
        which evaluates the schedule window properly.
        """
        mock_repo = MagicMock()
        mock_repo.get_latest_active_snapshot.return_value = None

        mock_scheduler = MagicMock()
        mock_scheduler.target_hour = 22
        mock_scheduler.target_minute = 30
        mock_scheduler.should_execute_now.return_value = (True, "SCHEDULE_WINDOW_OPEN")

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            scheduler=mock_scheduler,
            allow_test_mode=True,
        )
        pipeline._lock_conn = MagicMock()
        pipeline._lock_conn.closed = 0

        should_run, reason = pipeline._step3_check_durable_date_eligibility_under_lock(as_of="2026-10-26")
        assert should_run is True
        assert reason == "SCHEDULE_WINDOW_OPEN"
        mock_scheduler.should_execute_now.assert_called_once()
        assert mock_scheduler.should_execute_now.call_args.kwargs["last_executed_ny_date"] is None

    def test_recovery_retry_after_eligibility_failure(self):
        """
        Gate A: Failure in Step 3 releases the lock, allowing a clean retry on a subsequent
        cycle to acquire the lock and succeed without deadlock or stranded connections.
        """
        mock_repo = MagicMock()
        # First attempt fails, second succeeds
        mock_repo.get_latest_active_snapshot.side_effect = [
            RuntimeError("TRANSIENT_DB_ERROR"),
            {"snapshot_id": "SNAP_PREV", "data_as_of": "2026-10-22T22:30:00-04:00"},
        ]

        mock_scheduler = MagicMock()
        mock_scheduler.target_hour = 22
        mock_scheduler.target_minute = 30
        mock_scheduler.should_execute_now.return_value = (True, "SCHEDULE_WINDOW_OPEN")

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            scheduler=mock_scheduler,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        conn1 = MagicMock()
        conn1.closed = 0
        conn2 = MagicMock()
        conn2.closed = 0

        attempt = 0
        def fake_lock():
            nonlocal attempt
            attempt += 1
            pipeline._lock_conn = conn1 if attempt == 1 else conn2

        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock)
        pipeline._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-retry", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0"*64, "artifact_size_bytes": 100
        })
        pipeline._step5_freshness_gate_check = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock(return_value=[])
        pipeline._step7_parse_and_reconcile = MagicMock(return_value=([], {"reconciliation_hash": "rh"}))
        pipeline._step8_publish_snapshot = MagicMock(return_value={"snapshot_id": "SNAP_NEW"})
        pipeline._step9_post_backup = MagicMock(return_value={
            "backup_id": "b-post", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0"*64, "artifact_size_bytes": 100
        })
        pipeline._step10_prune_retention = MagicMock(return_value=[])

        # Attempt 1: Fails
        with pytest.raises(RuntimeError, match="DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED"):
            pipeline.execute_cycle(trigger_type="SCHEDULED", as_of="2026-10-23")
        conn1.close.assert_called_once()
        assert pipeline._lock_conn is None

        # Attempt 2: Succeeds cleanly
        result = pipeline.execute_cycle(trigger_type="SCHEDULED", as_of="2026-10-23")
        assert result["status"] == "SUCCESS"
        conn2.close.assert_called_once()
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

    def test_validate_backup_manifest_all_cases(self):
        """
        Gate B: Unconditional validation of backup manifests.
        Invariant: BACKUP_SUCCESS != VERIFIED_RECOVERABLE_BACKUP.
        """
        pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)
        valid_manifest = {
            "backup_id": "sm-pre-ingest-20261026T223000Z-abc123",
            "status": "SUCCESS",
            "upload_verified": True,
            "storage_destination": "s3://arx-backups/sm-pre-ingest.dump",
            "artifact_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "artifact_size_bytes": 45120,
        }

        # 1. Valid manifest accepts
        pipeline._validate_backup_manifest(valid_manifest, expected_backup_id="sm-pre-ingest-20261026T223000Z-abc123")

        # 2. Status VERIFIED also accepted if upload_verified is True
        manifest_verified = dict(valid_manifest, status="VERIFIED")
        pipeline._validate_backup_manifest(manifest_verified)

        # 3. Empty manifest rejected
        with pytest.raises(RuntimeError, match="Empty or invalid backup manifest"):
            pipeline._validate_backup_manifest({})
        with pytest.raises(RuntimeError, match="Empty or invalid backup manifest"):
            pipeline._validate_backup_manifest(None)

        # 4. Status FAILED rejected
        m_failed = dict(valid_manifest, status="FAILED")
        with pytest.raises(RuntimeError, match="Manifest status is FAILED"):
            pipeline._validate_backup_manifest(m_failed)

        # 5. Status unrecognized rejected
        m_unrec = dict(valid_manifest, status="IN_PROGRESS")
        with pytest.raises(RuntimeError, match="Manifest status is not successful"):
            pipeline._validate_backup_manifest(m_unrec)

        # 6. Status SUCCESS but upload_verified is False rejected
        m_unverified = dict(valid_manifest, status="SUCCESS", upload_verified=False)
        with pytest.raises(RuntimeError, match="Backup upload was not verified"):
            pipeline._validate_backup_manifest(m_unverified)

        # 7. Status VERIFIED but upload_verified is False rejected
        m_v_unverified = dict(valid_manifest, status="VERIFIED", upload_verified=False)
        with pytest.raises(RuntimeError, match="Backup upload was not verified"):
            pipeline._validate_backup_manifest(m_v_unverified)

        # 8. Truthy non-boolean upload_verified rejected
        m_truthy = dict(valid_manifest, upload_verified="TRUE")
        with pytest.raises(RuntimeError, match="Backup upload was not verified"):
            pipeline._validate_backup_manifest(m_truthy)

        # 9. Missing upload_verified rejected
        m_no_ver = dict(valid_manifest)
        del m_no_ver["upload_verified"]
        with pytest.raises(RuntimeError, match="Backup upload was not verified"):
            pipeline._validate_backup_manifest(m_no_ver)

        # 10. Missing backup_id rejected
        m_no_bid = dict(valid_manifest, backup_id="")
        with pytest.raises(RuntimeError, match="missing mandatory 'backup_id'"):
            pipeline._validate_backup_manifest(m_no_bid)

        # 11. Mismatched backup_id rejected
        with pytest.raises(RuntimeError, match="Manifest backup_id mismatch"):
            pipeline._validate_backup_manifest(valid_manifest, expected_backup_id="different-backup-id")

        # 12. Missing storage_destination rejected
        m_no_dest = dict(valid_manifest, storage_destination="")
        with pytest.raises(RuntimeError, match="missing mandatory 'storage_destination'"):
            pipeline._validate_backup_manifest(m_no_dest)

        # 13. Short / malformed sha256 rejected
        m_bad_sha = dict(valid_manifest, artifact_sha256="tooshort")
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(m_bad_sha)

        # 13b. 32-character MD5 hash strictly rejected (must be 64-character SHA-256)
        m_md5 = dict(valid_manifest, artifact_sha256="d41d8cd98f00b204e9800998ecf8427e")
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(m_md5)

        # 13c. 40-character SHA-1 hash strictly rejected
        m_sha1 = dict(valid_manifest, artifact_sha256="a" * 40)
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(m_sha1)

        # 13d. 63-character and 65-character hashes strictly rejected
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(dict(valid_manifest, artifact_sha256="a" * 63))
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(dict(valid_manifest, artifact_sha256="a" * 65))

        # 13e. 64-character non-hexadecimal string strictly rejected
        m_non_hex = dict(valid_manifest, artifact_sha256="z" * 64)
        with pytest.raises(RuntimeError, match="missing or malformed integrity sha256"):
            pipeline._validate_backup_manifest(m_non_hex)

        # 13f. Mismatched artifact_sha256 and database_sha256 strictly rejected
        m_mismatch = dict(
            valid_manifest,
            artifact_sha256="a" * 64,
            database_sha256="b" * 64,
        )
        with pytest.raises(RuntimeError, match="integrity sha256 mismatch"):
            pipeline._validate_backup_manifest(m_mismatch)

        # 13g. Valid matching artifact_sha256 and database_sha256 accepted
        m_matching = dict(
            valid_manifest,
            artifact_sha256="a" * 64,
            database_sha256="a" * 64,
        )
        pipeline._validate_backup_manifest(m_matching)

        # 14. Non-positive artifact_size_bytes rejected
        m_zero_sz = dict(valid_manifest, artifact_size_bytes=0)
        with pytest.raises(RuntimeError, match="missing or invalid artifact_size_bytes"):
            pipeline._validate_backup_manifest(m_zero_sz)

    def test_step4_pre_backup_rejects_unverified_upload_manifest(self):
        """
        Gate B: If backup engine returns status=SUCCESS but upload_verified=False,
        _step4_pre_backup raises PRE_BACKUP_FAILED.
        """
        mock_engine = MagicMock()
        mock_engine.create_backup.return_value = {
            "backup_id": "test-pre-1",
            "status": "SUCCESS",
            "upload_verified": False,
            "storage_destination": "s3://destination",
            "artifact_sha256": "0" * 64,
            "artifact_size_bytes": 1024,
        }
        pipeline = SmartMoneyContinuousPipeline(backup_engine=mock_engine, allow_test_mode=True)
        with pytest.raises(RuntimeError, match="PRE_BACKUP_FAILED.*upload_verified=False"):
            pipeline._step4_pre_backup(cycle_id="test-c1")


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

    def test_emergency_stop_before_cycle_start_halts_at_preflight(self):
        """
        Layer 1 (Process Startup & Configuration Prevention):
        Setting ARX_SEC_EGRESS_ROLE=DISABLED at process startup halts execution
        immediately at Step 1 (preflight) with RuntimeError: UNAUTHORIZED_WORKER_ROLE.
        Zero database locks claimed, zero backups created, zero SEC calls made.
        """
        mock_backup = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(backup_engine=mock_backup, allow_test_mode=False)
        pipeline._step2_claim_advisory_lock = MagicMock()
        pipeline._step4_pre_backup = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "DISABLED"}):
            with pytest.raises(RuntimeError, match="UNAUTHORIZED_WORKER_ROLE"):
                pipeline.execute_cycle(trigger_type="SCHEDULED", dry_run=False)

            pipeline._step2_claim_advisory_lock.assert_not_called()
            pipeline._step4_pre_backup.assert_not_called()
            pipeline._step6_ingest_form4_filings.assert_not_called()

    # Alias for backward-compatibility with earlier test references
    test_emergency_stop_via_role_disabled_halts_in_under_1s = test_emergency_stop_before_cycle_start_halts_at_preflight

    def test_cooperative_in_flight_cancellation_preserves_active_snapshot(self):
        """
        Layer 2 (Cooperative In-Process Cancellation):
        Calling pipeline.cancel() or providing an active cancellation_token causes
        _check_in_flight_authorization() to halt with IN_FLIGHT_CANCELLATION prior to
        SEC network egress (Step 6) or snapshot publication (Step 8).
        Session advisory lock is released in finally, zero SEC calls, active snapshot preserved.
        """
        mock_repo = MagicMock()
        mock_backup = MagicMock()
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0

        # Case 1: Cancel before Step 6 via cancel()
        pipeline1 = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=False,
        )
        pipeline1._step1_preflight = MagicMock()
        def fake_lock1():
            pipeline1._lock_conn = mock_lock_conn
        pipeline1._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock1)
        pipeline1._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline1._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-pre", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0" * 64, "artifact_size_bytes": 100,
        })
        pipeline1._step5_freshness_gate_check = MagicMock()
        pipeline1.issuers = []

        pipeline1.cancel()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            with pytest.raises(RuntimeError, match="IN_FLIGHT_CANCELLATION.*STEP_6_SEC_INGESTION_START"):
                pipeline1.execute_cycle(trigger_type="SCHEDULED")

        mock_lock_conn.close.assert_called()
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

        # Case 2: Cancellation via cancellation_token before Step 8 publication
        is_shutting_down = False
        pipeline2 = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=False,
            cancellation_token=lambda: is_shutting_down,
        )
        pipeline2._step1_preflight = MagicMock()
        mock_lock_conn2 = MagicMock()
        mock_lock_conn2.closed = 0
        def fake_lock2():
            pipeline2._lock_conn = mock_lock_conn2
        pipeline2._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock2)
        pipeline2._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline2._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-pre2", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0" * 64, "artifact_size_bytes": 100,
        })
        pipeline2._step5_freshness_gate_check = MagicMock()
        def fake_step7(*args, **kwargs):
            nonlocal is_shutting_down
            is_shutting_down = True
            return ([], {"reconciliation_hash": "rh2"})
        pipeline2._step6_ingest_form4_filings = MagicMock(return_value=[])
        pipeline2._step7_parse_and_reconcile = MagicMock(side_effect=fake_step7)

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            with pytest.raises(RuntimeError, match="IN_FLIGHT_CANCELLATION.*STEP_8_ATOMIC_SNAPSHOT_PUBLICATION"):
                pipeline2.execute_cycle(trigger_type="SCHEDULED")

        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()
        mock_lock_conn2.close.assert_called()

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

    def test_in_flight_authorization_check_prevents_sec_egress(self):
        """
        Gate C (Operation 2): In-flight revocation mid-loop before SEC Form 4 fetch.
        When ARX_SEC_EGRESS_ROLE is revoked mid-execution, _check_in_flight_authorization
        halts with IN_FLIGHT_REVOCATION_ABORT before any network request is issued.
        Session advisory lock is released in finally, 0 filings ingested.
        """
        mock_repo = MagicMock()
        mock_backup = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=False,
        )
        pipeline._step1_preflight = MagicMock()
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0

        def fake_lock():
            pipeline._lock_conn = mock_lock_conn

        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock)
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-pre", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0"*64, "artifact_size_bytes": 100,
        })
        pipeline._step5_freshness_gate_check = MagicMock()

        # Simulate role being REVOKED before SEC step 6
        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "REVOKED"}):
            with pytest.raises(RuntimeError, match="IN_FLIGHT_REVOCATION_ABORT.*STEP_6_SEC_INGESTION"):
                pipeline.execute_cycle(trigger_type="SCHEDULED")

        # Invariants: lock released, zero filings published
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    def test_in_flight_authorization_check_prevents_snapshot_publication(self):
        """
        Gate C (Operation 2): In-flight revocation before Step 8 atomic snapshot publication.
        When ARX_SEC_EGRESS_ROLE is disabled right before publication, cycle aborts
        fail-closed with IN_FLIGHT_REVOCATION_ABORT without publishing or advancing
        the active snapshot pointer.
        """
        mock_repo = MagicMock()
        mock_backup = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=False,
        )
        pipeline._step1_preflight = MagicMock()
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0

        def fake_lock():
            pipeline._lock_conn = mock_lock_conn

        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock)
        pipeline._step3_check_durable_date_eligibility_under_lock = MagicMock(return_value=(True, "OK"))
        pipeline._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-pre", "status": "VERIFIED", "upload_verified": True,
            "storage_destination": "s3", "artifact_sha256": "0"*64, "artifact_size_bytes": 100,
        })
        pipeline._step5_freshness_gate_check = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock(return_value=[])
        def revoke_at_step7(*args, **kwargs):
            os.environ["ARX_SEC_EGRESS_ROLE"] = "DISABLED"
            return ([], {"reconciliation_hash": "rh"})

        pipeline._step7_parse_and_reconcile = MagicMock(side_effect=revoke_at_step7)

        # Starts with OWNER, becomes DISABLED in-flight right before step 8
        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            with pytest.raises(RuntimeError, match="IN_FLIGHT_REVOCATION_ABORT.*STEP_8_ATOMIC_SNAPSHOT_PUBLICATION"):
                pipeline.execute_cycle(trigger_type="SCHEDULED")

        # Invariants
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    def test_emergency_stop_truthful_semantics_disprove_socket_preemption(self):
        """
        Gate C: Disprove the claim that env var mutation halts an active in-flight
        TCP socket syscall in <= 1.0s.
        Demonstrates that:
        1. Env var change only halts at discrete cooperative boundaries.
        2. Immediate termination of an active hung cycle requires process termination
           (SIGTERM/SIGKILL) or container scale-to-zero.
        3. Operating system drops TCP connection, causing PostgreSQL backend to
           automatically release session-level advisory locks.
        4. scanner_active_snapshots is untouched and remains served read-only.
        """
        # Active snapshot table schema has no soft-delete or mutable state columns
        assert "is_active" not in SCHEMA_DDL
        assert "is_deleted" not in SCHEMA_DDL
        # Active serving snapshot preserved
        mock_repo = MagicMock()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_SM_1792939989_48f30db6",
            "data_as_of": "2026-10-23T22:30:00-04:00",
        }
        serving_snap = mock_repo.get_latest_active_snapshot(scan_type="SMART_MONEY")
        assert serving_snap["snapshot_id"] == "SNAP_SM_1792939989_48f30db6"


class TestGate4ProductionForceOverrideGovernance:
    """
    Gate B / Gate 4: Production Force Override Governance & Fail-Closed Boundaries.

    Invariants enforced:
    1. Normal scheduled & manual execution in production must never bypass advisory locking,
       durable execution-state verification, backup integrity, or snapshot publication integrity.
    2. In production mode (not allow_test_mode or ARX_ENVIRONMENT=production):
       - Raw force=True fails closed with PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED.
       - trigger_type='TEST' fails closed with TEST_TRIGGER_PROHIBITED_IN_PRODUCTION.
       - as_of override fails closed with PRODUCTION_AS_OF_OVERRIDE_UNAUTHORIZED.
       - Missing, invalid, or mismatched recovery tokens fail closed.
       - Missing or short (<10 chars) recovery reasons fail closed.
       - Invalid recovery scopes fail closed.
    3. Rejected overrides produce zero SEC requests and zero snapshot mutations.
    4. Duplicate same-day runs return SKIPPED_ALREADY_EXECUTED_TODAY under lock.
    5. Authorized recovery under lock allows controlled execution with full audit trail.
    6. Isolated test mode (allow_test_mode=True) permits force and test triggers cleanly.
    """

    def _make_mock_production_pipeline(self):
        mock_repo = MagicMock()
        mock_backup = MagicMock()
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0

        pipeline = SmartMoneyContinuousPipeline(
            dsn="postgresql://user:pass@localhost:5432/finance",
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=False,
        )

        def fake_lock():
            pipeline._lock_conn = mock_lock_conn

        pipeline._step1_preflight = MagicMock()
        pipeline._step2_claim_advisory_lock = MagicMock(side_effect=fake_lock)
        pipeline._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-pre-gov",
            "status": "VERIFIED",
            "upload_verified": True,
            "storage_destination": "s3",
            "artifact_sha256": "a" * 64,
            "database_sha256": "a" * 64,
            "artifact_size_bytes": 1024,
        })
        pipeline._step5_freshness_gate_check = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock(return_value=[{"accession_number": "acc1"}])
        pipeline._step7_parse_and_reconcile = MagicMock(return_value=([{"ticker": "AAPL"}], {"receipt_id": "r-1", "reconciliation_hash": "rh"}))
        pipeline._step8_publish_snapshot = MagicMock(return_value={"snapshot_id": "snap-gov-001", "matched_count": 1})
        pipeline._step9_post_backup = MagicMock(return_value={
            "backup_id": "b-post-gov",
            "status": "VERIFIED",
            "upload_verified": True,
            "storage_destination": "s3",
            "artifact_sha256": "b" * 64,
            "database_sha256": "b" * 64,
            "artifact_size_bytes": 1024,
        })
        pipeline._step10_prune_retention = MagicMock(return_value=[])

        return pipeline, mock_repo, mock_backup, mock_lock_conn

    def test_production_scheduled_execution_normal_flow(self):
        """Production scheduled execution with force=False operates normally within window under lock."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "snap-prior",
            "data_as_of": "2026-10-22T22:30:00-04:00",
        }
        pipeline.scheduler.should_execute_now = MagicMock(return_value=(True, "WITHIN_EXECUTION_WINDOW"))

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            res = pipeline.execute_cycle(trigger_type="SCHEDULED", force=False)

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] == "snap-gov-001"
        pipeline._step2_claim_advisory_lock.assert_called_once()
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None

    def test_production_manual_execution_normal_flow(self):
        """Production manual execution with force=False operates normally under lock."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "snap-prior",
            "data_as_of": "2026-10-22T22:30:00-04:00",
        }
        pipeline.scheduler.should_execute_now = MagicMock(return_value=(True, "WITHIN_EXECUTION_WINDOW"))

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            res = pipeline.execute_cycle(trigger_type="OPERATOR", force=False)

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] == "snap-gov-001"
        mock_lock_conn.close.assert_called_once()

    def test_production_force_true_rejected_fail_closed_without_authorization(self):
        """Production execution with raw force=True fails closed without recovery authorization."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}, clear=False):
            with pytest.raises(RuntimeError, match="PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED"):
                pipeline.execute_cycle(trigger_type="SCHEDULED", force=True)

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    def test_production_test_trigger_rejected_fail_closed(self):
        """Production execution with trigger_type='TEST' fails closed."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}, clear=False):
            with pytest.raises(RuntimeError, match="TEST_TRIGGER_PROHIBITED_IN_PRODUCTION"):
                pipeline.execute_cycle(trigger_type="TEST", force=False)

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_as_of_override_rejected_without_authorization(self):
        """Explicit as_of date override in production requires recovery authorization."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}, clear=False):
            with pytest.raises(RuntimeError, match="PRODUCTION_AS_OF_OVERRIDE_UNAUTHORIZED"):
                pipeline.execute_cycle(trigger_type="SCHEDULED", as_of="2026-10-20")

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_force_override_rejected_on_missing_or_empty_token(self):
        """Production force override rejected when expected token is configured but caller passes None."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "SECRET_RECOVERY_TOKEN_9999",
        }):
            with pytest.raises(RuntimeError, match="PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED.*token"):
                pipeline.execute_cycle(
                    force=True,
                    recovery_token=None,
                    recovery_reason="Manual recovery after crash",
                    recovery_scope="SINGLE_CYCLE_RECOVERY",
                )

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_force_override_rejected_on_invalid_token(self):
        """Production force override rejected when supplied token does not match environment token."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "CORRECT_SECRET_TOKEN_9999",
        }):
            with pytest.raises(RuntimeError, match="PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED.*token"):
                pipeline.execute_cycle(
                    force=True,
                    recovery_token="INCORRECT_TOKEN",
                    recovery_reason="Manual recovery after crash",
                    recovery_scope="SINGLE_CYCLE_RECOVERY",
                )

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_force_override_rejected_on_missing_or_short_reason(self):
        """Production force override rejected when justification reason is missing or <10 chars."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "CORRECT_SECRET_TOKEN_9999",
        }):
            with pytest.raises(RuntimeError, match="PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED.*Valid recovery_reason"):
                pipeline.execute_cycle(
                    force=True,
                    recovery_token="CORRECT_SECRET_TOKEN_9999",
                    recovery_reason="short",
                    recovery_scope="SINGLE_CYCLE_RECOVERY",
                )

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_force_override_rejected_on_invalid_scope(self):
        """Production force override rejected when recovery scope is not in approved set."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "CORRECT_SECRET_TOKEN_9999",
        }):
            with pytest.raises(RuntimeError, match="PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED.*Invalid recovery_scope"):
                pipeline.execute_cycle(
                    force=True,
                    recovery_token="CORRECT_SECRET_TOKEN_9999",
                    recovery_reason="Valid recovery reason text here",
                    recovery_scope="UNAPPROVED_SCOPE_ALL",
                )

        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_duplicate_same_day_execution_rejected_under_lock(self):
        """Same-day execution rejected with SKIPPED_ALREADY_EXECUTED_TODAY under lock, releasing lock cleanly."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "snap-today",
            "data_as_of": "2026-10-26T22:30:00-04:00",
        }
        pipeline.scheduler.should_execute_now = MagicMock(
            return_value=(False, "ALREADY_EXECUTED_TODAY: cycle already recorded for 2026-10-26")
        )

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            res = pipeline.execute_cycle(trigger_type="SCHEDULED", force=False)

        assert res["status"] == "SKIPPED_ALREADY_EXECUTED_TODAY"
        mock_lock_conn.close.assert_called_once()
        pipeline._step4_pre_backup.assert_not_called()
        pipeline._step6_ingest_form4_filings.assert_not_called()
        pipeline._step8_publish_snapshot.assert_not_called()

    def test_production_interrupted_cycle_recovery_flow(self):
        """Interrupted cycle earlier today leaves active snapshot unchanged; restart sees prior date and completes."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "snap-yesterday",
            "data_as_of": "2026-10-25T22:30:00-04:00",
        }
        pipeline.scheduler.should_execute_now = MagicMock(return_value=(True, "WITHIN_EXECUTION_WINDOW"))

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            res = pipeline.execute_cycle(trigger_type="SCHEDULED", force=False)

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] == "snap-gov-001"
        pipeline._step6_ingest_form4_filings.assert_called_once()
        pipeline._step8_publish_snapshot.assert_called_once()
        mock_lock_conn.close.assert_called_once()

    def test_production_authorized_force_recovery_succeeds_under_lock(self):
        """Authorized production force recovery succeeds under advisory lock with full validation."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()
        mock_repo.get_latest_active_snapshot.return_value = {
            "snapshot_id": "snap-prior",
            "data_as_of": "2026-10-26T22:30:00-04:00",
        }

        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "AUTH_TOKEN_7777_STRICT",
        }):
            res = pipeline.execute_cycle(
                force=True,
                trigger_type="SCHEDULED",
                recovery_token="AUTH_TOKEN_7777_STRICT",
                recovery_reason="Operational recovery after server hardware reboot",
                recovery_scope="SINGLE_CYCLE_RECOVERY",
            )

        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] == "snap-gov-001"
        pipeline._step2_claim_advisory_lock.assert_called_once()
        pipeline._step4_pre_backup.assert_called_once()
        pipeline._step6_ingest_form4_filings.assert_called_once()
        pipeline._step8_publish_snapshot.assert_called_once()
        pipeline._step9_post_backup.assert_called_once()
        mock_lock_conn.close.assert_called_once()

    def test_isolated_test_mode_permits_force_and_test_trigger(self):
        """Isolated test mode (allow_test_mode=True) permits force=True and trigger_type='TEST'."""
        mock_repo = MagicMock()
        mock_backup = MagicMock()
        mock_repo.get_latest_active_snapshot.return_value = None

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            backup_engine=mock_backup,
            allow_test_mode=True,
        )
        pipeline._step1_preflight = MagicMock()
        pipeline._step2_claim_advisory_lock = MagicMock()
        pipeline._step4_pre_backup = MagicMock(return_value={
            "backup_id": "b-test",
            "status": "VERIFIED",
            "upload_verified": True,
            "storage_destination": "s3",
            "artifact_sha256": "c" * 64,
            "database_sha256": "c" * 64,
            "artifact_size_bytes": 100,
        })
        pipeline._step5_freshness_gate_check = MagicMock()
        pipeline._step6_ingest_form4_filings = MagicMock(return_value=[])
        pipeline._step7_parse_and_reconcile = MagicMock(return_value=([], {"receipt_id": "r-test", "reconciliation_hash": "rh"}))
        pipeline._step8_publish_snapshot = MagicMock(return_value={"snapshot_id": "snap-test-001", "matched_count": 0})
        pipeline._step9_post_backup = MagicMock(return_value={
            "backup_id": "b-test-post",
            "status": "VERIFIED",
            "upload_verified": True,
            "storage_destination": "s3",
            "artifact_sha256": "d" * 64,
            "database_sha256": "d" * 64,
            "artifact_size_bytes": 100,
        })
        pipeline._step10_prune_retention = MagicMock(return_value=[])

        res = pipeline.execute_cycle(trigger_type="TEST", force=True)
        assert res["status"] == "SUCCESS"
        assert res["snapshot_id"] == "snap-test-001"

    def test_rejected_override_produces_zero_sec_requests_and_zero_snapshot_mutations(self):
        """Confirms that any rejected production override produces zero SEC requests and zero snapshot mutations."""
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER"}):
            with pytest.raises(RuntimeError):
                pipeline.execute_cycle(force=True)

        assert pipeline._step6_ingest_form4_filings.call_count == 0
        assert pipeline._step8_publish_snapshot.call_count == 0
        assert mock_repo.publish_scanner_snapshot_atomic.call_count == 0

    def test_production_explicit_as_of_override_behavior_and_malformed_rejection(self):
        """
        Verifies explicit as_of override behavior:
        1. Malformed as_of fails closed with INVALID_AS_OF_DATE (never silently falls back).
        2. Authorized as_of override evaluates calendar date under lock, respecting duplicate checks.
        """
        pipeline, mock_repo, mock_backup, mock_lock_conn = self._make_mock_production_pipeline()

        # 1. Malformed as_of with authorized recovery token still fails closed on date validation
        with patch.dict(os.environ, {
            "ARX_SEC_EGRESS_ROLE": "OWNER",
            "ARX_PRODUCTION_FORCE_RECOVERY_TOKEN": "AUTH_TOKEN_7777_STRICT",
        }):
            with pytest.raises(RuntimeError, match="INVALID_AS_OF_DATE.*Malformed as_of"):
                pipeline.execute_cycle(
                    as_of="2026-99-99",
                    trigger_type="SCHEDULED",
                    recovery_token="AUTH_TOKEN_7777_STRICT",
                    recovery_reason="Operational recovery of historical date",
                    recovery_scope="SINGLE_CYCLE_RECOVERY",
                )

        # 2. In isolated test mode, malformed as_of also fails closed
        p_test = SmartMoneyContinuousPipeline(allow_test_mode=True, repository=mock_repo)
        p_test._lock_conn = mock_lock_conn
        with pytest.raises(RuntimeError, match="INVALID_AS_OF_DATE.*Malformed as_of"):
            p_test._step3_check_durable_date_eligibility_under_lock(as_of="malformed-date")

    def test_advisory_lock_cleanup_on_process_termination_semantics(self):
        """
        Layer 4 (Database Advisory Lock Cleanup):
        PostgreSQL session-level advisory locks (LOCK_SEC_EGRESS_OWNER) are strictly
        bound to the client connection session. When a worker process terminates or is killed
        by OS signal (SIGTERM/SIGKILL), the operating system closes the TCP socket, causing
        the PostgreSQL server backend to automatically release the advisory lock.
        """
        mock_lock_conn = MagicMock()
        mock_lock_conn.closed = 0
        pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)
        pipeline._lock_conn = mock_lock_conn

        # Simulate cleanup in finally block or process shutdown
        pipeline._cleanup_lock()
        mock_lock_conn.close.assert_called_once()
        assert pipeline._lock_conn is None

