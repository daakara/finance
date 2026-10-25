"""
ARX Terminal — Smart Money Form 4
Backup Freshness Authority — Remote Checksum Metadata Integrity Adversarial Test Suite.

Tests 12 exhaustive adversarial vectors defining the remote-metadata integrity contract:
1. Matching SHA-256 metadata.
2. Missing SHA-256 metadata.
3. Empty SHA-256 metadata.
4. Malformed SHA-256 metadata.
5. Uppercase/lowercase canonicalization.
6. Mismatched SHA-256 metadata.
7. Manifest SHA mismatch.
8. Remote object missing.
9. Remote size mismatch.
10. Metadata retrieval failure.
11. Historical object without metadata (policy require_remote_checksum_metadata toggle).
12. Object metadata modified after upload.
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupFreshnessState,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    ContainmentState,
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
    storage_dir = tmp_path / "checksum_test_backups"
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


def stage_backup(
    adapter: LocalOffVolumeStorageAdapter,
    backup_id: str,
    completed_at: str,
    dump_bytes: bytes = b"PG16_TEST_DUMP_BYTES_12345",
    manifest_sha_override: str = None,
    remote_meta_override: dict = None,
    omit_dump: bool = False,
    size_override: int = None,
    upload_verified: bool = True,
):
    with tempfile.TemporaryDirectory() as tmp:
        dump_name = f"smart_money_{backup_id}.dump"
        manifest_name = f"manifest_{backup_id}.json"

        dump_local = os.path.join(tmp, dump_name)
        with open(dump_local, "wb") as f:
            f.write(dump_bytes)

        actual_sha = compute_sha256(dump_local)
        actual_size = os.path.getsize(dump_local)

        if not omit_dump:
            adapter.upload_file(dump_local, dump_name)
            if remote_meta_override is not None:
                adapter.set_custom_metadata(dump_name, remote_meta_override)

        manifest = {
            "backup_id": backup_id,
            "started_at": (datetime.fromisoformat(completed_at) - timedelta(minutes=5)).isoformat(),
            "completed_at": completed_at,
            "database_artifact": dump_name,
            "dump_filename": dump_name,
            "artifact_sha256": manifest_sha_override or actual_sha,
            "artifact_size_bytes": size_override if size_override is not None else actual_size,
            "upload_verified": upload_verified,
            "postgres_major_version": 16,
            "schema_version": "SMART_MONEY_POSTGRES_SCHEMA_V2",
        }

        manifest_local = os.path.join(tmp, manifest_name)
        with open(manifest_local, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        adapter.upload_file(manifest_local, manifest_name)
        return manifest, actual_sha, actual_size


class TestChecksumMetadataIntegrityAdversarial:

    def test_vector_01_matching_sha256_metadata(self, storage_adapter, active_policy, frozen_clock):
        """Vector 1: Valid matching SHA-256 metadata returns FRESH."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(storage_adapter, "b-match-01", completed_at)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.reason_code == "VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD"
        assert receipt.verification_source == "STORAGE_MANIFEST_AND_REMOTE_METADATA"

    def test_vector_02_missing_sha256_metadata(self, storage_adapter, active_policy, frozen_clock):
        """Vector 2: Missing remote SHA-256 metadata fails closed as UNVERIFIABLE."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-missing-meta-02",
            completed_at,
            remote_meta_override={"sha256": None},
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "MISSING_REMOTE_CHECKSUM_METADATA"

    def test_vector_03_empty_sha256_metadata(self, storage_adapter, active_policy, frozen_clock):
        """Vector 3: Empty string remote SHA-256 metadata fails closed as UNVERIFIABLE."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-empty-meta-03",
            completed_at,
            remote_meta_override={"sha256": "   "},
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "EMPTY_REMOTE_CHECKSUM_METADATA"

    def test_vector_04_malformed_sha256_metadata(self, storage_adapter, active_policy, frozen_clock):
        """Vector 4: Malformed SHA-256 metadata (invalid length or non-hex) fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-malformed-meta-04",
            completed_at,
            remote_meta_override={"sha256": "not_a_valid_64_character_hex_string_too_short"},
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "MALFORMED_REMOTE_CHECKSUM_METADATA" in receipt.reason_code

    def test_vector_05_uppercase_lowercase_canonicalization(self, storage_adapter, active_policy, frozen_clock):
        """Vector 5: Uppercase remote metadata matches lowercase manifest under hex canonicalization."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        manifest, actual_sha, _ = stage_backup(storage_adapter, "b-canonical-05", completed_at)
        uppercase_sha = actual_sha.upper()
        storage_adapter.set_custom_metadata(f"smart_money_b-canonical-05.dump", {"sha256": uppercase_sha})

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is True
        assert receipt.verification_status == BackupFreshnessState.FRESH.value
        assert receipt.reason_code == "VERIFIED_RECOVERY_POINT_WITHIN_THRESHOLD"

    def test_vector_06_mismatched_sha256_metadata(self, storage_adapter, active_policy, frozen_clock):
        """Vector 6: Remote SHA-256 metadata different from manifest SHA fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        bogus_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        stage_backup(
            storage_adapter,
            "b-mismatch-06",
            completed_at,
            remote_meta_override={"sha256": bogus_sha},
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "DUMP_SHA256_MISMATCH" in receipt.reason_code

    def test_vector_07_manifest_sha_mismatch(self, storage_adapter, active_policy, frozen_clock):
        """Vector 7: Malformed SHA-256 recorded in manifest fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-bad-manifest-sha-07",
            completed_at,
            manifest_sha_override="invalid-manifest-sha",
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "MALFORMED_MANIFEST_SHA256" in receipt.reason_code

    def test_vector_08_remote_object_missing(self, storage_adapter, active_policy, frozen_clock):
        """Vector 8: Manifest references non-existent remote dump object fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-missing-dump-08",
            completed_at,
            omit_dump=True,
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert receipt.reason_code == "REMOTE_DUMP_ARTIFACT_MISSING"

    def test_vector_09_remote_size_mismatch(self, storage_adapter, active_policy, frozen_clock):
        """Vector 9: Manifest artifact_size_bytes does not match remote size fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-size-mismatch-09",
            completed_at,
            size_override=999999,
        )

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "DUMP_SIZE_MISMATCH" in receipt.reason_code

    def test_vector_10_metadata_retrieval_failure(self, storage_adapter, active_policy, frozen_clock, monkeypatch):
        """Vector 10: Storage adapter raising exception on get_backup_metadata fails closed."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(storage_adapter, "b-meta-fail-10", completed_at)

        def mock_get_metadata(filename):
            raise ConnectionResetError("Remote S3 endpoint connection reset by peer")

        monkeypatch.setattr(storage_adapter, "get_backup_metadata", mock_get_metadata)

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "REMOTE_STORAGE_METADATA_FETCH_FAILED" in receipt.reason_code

    def test_vector_11_historical_object_without_metadata(self, storage_adapter, frozen_clock):
        """Vector 11: Historical object lacking metadata is UNVERIFIABLE under standard policy."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        stage_backup(
            storage_adapter,
            "b-historical-11",
            completed_at,
            remote_meta_override={"sha256": None},
        )

        # Standard policy requires metadata -> UNVERIFIABLE
        strict_policy = BackupContainmentPolicy(enabled=True, require_remote_checksum_metadata=True)
        strict_auth = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=strict_policy)
        receipt_strict = strict_auth.evaluate_freshness(current_time=frozen_clock)
        assert receipt_strict.is_fresh is False
        assert receipt_strict.reason_code == "MISSING_REMOTE_CHECKSUM_METADATA"

        # Explicit historical audit policy (require_remote_checksum_metadata=False)
        hist_policy = BackupContainmentPolicy(enabled=True, require_remote_checksum_metadata=False)
        hist_auth = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=hist_policy)
        receipt_hist = hist_auth.evaluate_freshness(current_time=frozen_clock)
        assert receipt_hist.is_fresh is True  # Permitted under explicit historical audit mode

    def test_vector_12_object_metadata_modified_after_upload(self, storage_adapter, active_policy, frozen_clock):
        """Vector 12: Object metadata modified post-upload detected and rejected."""
        completed_at = (frozen_clock - timedelta(hours=2)).isoformat()
        manifest, actual_sha, _ = stage_backup(storage_adapter, "b-tamper-12", completed_at)

        # Simulate metadata corruption or tampering
        altered_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        storage_adapter.set_custom_metadata(f"smart_money_b-tamper-12.dump", {"sha256": altered_sha})

        authority = BackupFreshnessAuthority(storage_adapter=storage_adapter, policy=active_policy)
        receipt = authority.evaluate_freshness(current_time=frozen_clock)

        assert receipt.is_fresh is False
        assert receipt.verification_status == BackupFreshnessState.UNVERIFIABLE.value
        assert "DUMP_SHA256_MISMATCH" in receipt.reason_code
