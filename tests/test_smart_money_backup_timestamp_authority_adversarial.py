"""
ARX Terminal — Smart Money Form 4
Backup Freshness Authority — Timestamp Authority & Completion Integrity Adversarial Test Suite.

Tests 12 exhaustive adversarial vectors enforcing backup completion timestamp authority:
1. Valid completed backup.
2. Missing completed_at (key absent).
3. started_at present but completed_at absent.
4. Completion before start (completed_at precedes started_at).
5. Future-dated completion (exceeds clock skew tolerance).
6. Invalid timezone format.
7. Naive timestamp (normalized under explicit UTC contract).
8. Invalid timestamp format.
9. Upload failed after backup generation (upload_verified=False).
10. Manifest published before upload verification (upload_verified=None/False).
11. Backup completed but manifest incomplete (missing schema keys).
12. Recovery point exactly at the freshness threshold (age == max_verified_age_seconds).
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupFreshnessState,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)


@pytest.fixture
def frozen_clock():
    return datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def storage_adapter(tmp_path):
    storage_dir = tmp_path / "timestamp_test_backups"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return LocalOffVolumeStorageAdapter(base_directory=str(storage_dir))


@pytest.fixture
def active_policy():
    return BackupContainmentPolicy(
        enabled=True,
        max_verified_age_seconds=43200.0,
        check_timeout_seconds=5.0,
        recheck_interval_seconds=60.0,
        clock_skew_tolerance_seconds=300.0,
        production_authorized=True,
        require_remote_checksum_metadata=True,
    )


def stage_custom_manifest_backup(
    adapter: LocalOffVolumeStorageAdapter,
    backup_id: str,
    manifest_mutator=None,
    omit_dump: bool = False,
):
    with tempfile.TemporaryDirectory() as tmp:
        dump_name = f"smart_money_{backup_id}.dump"
        manifest_name = f"manifest_{backup_id}.json"

        dump_local = os.path.join(tmp, dump_name)
        with open(dump_local, "wb") as f:
            f.write(b"PG16_TIMESTAMP_TEST_CONTENT_123")

        actual_sha = compute_sha256(dump_local)
        actual_size = os.path.getsize(dump_local)

        if not omit_dump:
            adapter.upload_file(dump_local, dump_name)

        manifest = {
            "backup_id": backup_id,
            "started_at": "2026-10-14T09:55:00Z",
            "completed_at": "2026-10-14T10:00:00Z",
            "database_artifact": dump_name,
            "dump_filename": dump_name,
            "artifact_sha256": actual_sha,
            "artifact_size_bytes": actual_size,
            "upload_verified": True,
            "postgres_major_version": 16,
            "schema_version": "SMART_MONEY_POSTGRES_SCHEMA_V2",
        }

        if manifest_mutator:
            manifest = manifest_mutator(manifest)

        manifest_local = os.path.join(tmp, manifest_name)
        with open(manifest_local, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        adapter.upload_file(manifest_local, manifest_name)
        return manifest


class TestTimestampAuthorityAdversarial:

    def test_vector_01_valid_completed_backup(self, storage_adapter, active_policy, frozen_clock):
        """Vector 1: Valid completed backup evaluates to FRESH with exact completed_at."""
        stage_custom_manifest_backup(storage_adapter, "b-valid-01")

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.backup_completed_at == "2026-10-14T10:00:00Z"
        assert receipt.backup_age_seconds == 7200.0  # 12:00:00 - 10:00:00 = 2 hours

    def test_vector_02_missing_completed_at(self, storage_adapter, active_policy, frozen_clock):
        """Vector 2: Manifest lacking completed_at fails closed as UNVERIFIABLE."""
        def mutate(m):
            m.pop("completed_at", None)
            return m

        stage_custom_manifest_backup(storage_adapter, "b-missing-completed-02", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MISSING_COMPLETED_AT_TIMESTAMP"

    def test_vector_03_started_at_present_but_completed_at_absent(self, storage_adapter, active_policy, frozen_clock):
        """Vector 3: started_at present cannot substitute for completed_at; fails closed."""
        def mutate(m):
            m["started_at"] = "2026-10-14T10:00:00Z"
            m["completed_at"] = None
            return m

        stage_custom_manifest_backup(storage_adapter, "b-start-only-03", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MISSING_COMPLETED_AT_TIMESTAMP"

    def test_vector_04_completion_before_start(self, storage_adapter, active_policy, frozen_clock):
        """Vector 4: completed_at earlier than started_at violates causal order; fails closed."""
        def mutate(m):
            m["started_at"] = "2026-10-14T10:05:00Z"
            m["completed_at"] = "2026-10-14T10:00:00Z"  # Completed 5 minutes BEFORE start!
            return m

        stage_custom_manifest_backup(storage_adapter, "b-causal-inversion-04", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "COMPLETION_PRECEDES_START_TIMESTAMP"

    def test_vector_05_future_dated_completion(self, storage_adapter, active_policy, frozen_clock):
        """Vector 5: completed_at future-dated beyond clock skew tolerance fails closed."""
        def mutate(m):
            # Clock is 12:00:00; tolerance is 300s (5m); future date is 12:10:00 (+10m)
            m["completed_at"] = "2026-10-14T12:10:00Z"
            return m

        stage_custom_manifest_backup(storage_adapter, "b-future-05", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "FUTURE_DATED_RECOVERY_POINT_EXCEEDS_CLOCK_SKEW"

    def test_vector_06_invalid_timezone(self, storage_adapter, active_policy, frozen_clock):
        """Vector 6: Invalid timezone offset fails timestamp parsing and fails closed."""
        def mutate(m):
            m["completed_at"] = "2026-10-14T10:00:00+99:99"
            return m

        stage_custom_manifest_backup(storage_adapter, "b-bad-tz-06", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "INVALID_COMPLETED_AT_TIMESTAMP" in receipt.reason_code

    def test_vector_07_naive_timestamp(self, storage_adapter, active_policy, frozen_clock):
        """Vector 7: Naive timestamp without tzinfo is deterministically normalized to UTC."""
        def mutate(m):
            m["started_at"] = "2026-10-14T09:55:00"
            m["completed_at"] = "2026-10-14T10:00:00"  # Naive (no Z or offset)
            return m

        stage_custom_manifest_backup(storage_adapter, "b-naive-07", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.backup_age_seconds == 7200.0

    def test_vector_08_invalid_timestamp_format(self, storage_adapter, active_policy, frozen_clock):
        """Vector 8: Non-ISO timestamp string fails closed."""
        def mutate(m):
            m["completed_at"] = "Wednesday October 14th 2026 at 10am"
            return m

        stage_custom_manifest_backup(storage_adapter, "b-invalid-fmt-08", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "INVALID_COMPLETED_AT_TIMESTAMP" in receipt.reason_code

    def test_vector_09_upload_failed_after_generation(self, storage_adapter, active_policy, frozen_clock):
        """Vector 9: Backup generated but upload_verified=False fails closed."""
        def mutate(m):
            m["upload_verified"] = False
            return m

        stage_custom_manifest_backup(storage_adapter, "b-upload-failed-09", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MANIFEST_UPLOAD_NOT_VERIFIED"

    def test_vector_10_manifest_published_before_verification(self, storage_adapter, active_policy, frozen_clock):
        """Vector 10: Manifest published with upload_verified missing/None fails closed."""
        def mutate(m):
            m.pop("upload_verified", None)
            return m

        stage_custom_manifest_backup(storage_adapter, "b-no-verify-10", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MANIFEST_UPLOAD_NOT_VERIFIED"

    def test_vector_11_backup_completed_but_manifest_incomplete(self, storage_adapter, active_policy, frozen_clock):
        """Vector 11: Incomplete manifest missing essential schema keys fails closed."""
        def mutate(m):
            m.pop("artifact_size_bytes", None)
            return m

        stage_custom_manifest_backup(storage_adapter, "b-incomplete-11", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MALFORMED_MANIFEST_SCHEMA_KEYS"

    def test_vector_12_recovery_point_exactly_at_freshness_threshold(self, storage_adapter, active_policy, frozen_clock):
        """Vector 12: Recovery point exactly at boundary (43200.0s) permits ingestion under <= semantics."""
        # frozen_clock = 12:00:00; completed_at = 12:00:00 - 43200s = 00:00:00 (exactly 12h ago)
        boundary_time = (frozen_clock - timedelta(seconds=43200)).isoformat()
        start_time = (frozen_clock - timedelta(seconds=43200 + 300)).isoformat()
        def mutate(m):
            m["started_at"] = start_time
            m["completed_at"] = boundary_time
            return m

        stage_custom_manifest_backup(storage_adapter, "b-exact-threshold-12", manifest_mutator=mutate)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.backup_age_seconds == 43200.0
        assert receipt.reason_code == "VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD"
