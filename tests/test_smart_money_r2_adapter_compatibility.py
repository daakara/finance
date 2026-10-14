"""
Cloudflare R2 Adapter Compatibility & Failure Injection Test Suite
ARX Terminal — Smart Money Form 4 Backup Subsystem

Mechanically verifies:
1. Environment variable resolution for Cloudflare R2 standard (ARX_R2_*, ARX_BACKUP_*).
2. Endpoint derivation and region_name='auto' for R2.
3. S3CompatibleStorageAdapter operations against mock S3/R2 client.
4. Section 13 failure injection matrix:
   - Missing credentials fail-safe fallback.
   - Upload size/digest corruption detection.
   - Download tampering detection.
   - Retention pruning protection of latest backup.
"""

import os
import io
import pytest
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    get_backup_storage_adapter,
    S3CompatibleStorageAdapter,
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager


class TestCloudflareR2ConfigurationResolution:
    """Verifies environment variable resolution for Cloudflare R2."""

    def test_r2_env_vars_resolve_adapter(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_PROVIDER", "cloudflare_r2")
        monkeypatch.setenv("ARX_BACKUP_BUCKET", "arx-smart-money-backups")
        monkeypatch.setenv("ARX_BACKUP_PREFIX", "rehearsal")
        monkeypatch.setenv("ARX_R2_ACCOUNT_ID", "test-account-12345")
        monkeypatch.setenv("ARX_R2_ACCESS_KEY_ID", "test-access-key")
        monkeypatch.setenv("ARX_R2_SECRET_ACCESS_KEY", "test-secret-key")

        adapter = get_backup_storage_adapter()
        assert isinstance(adapter, S3CompatibleStorageAdapter)
        assert adapter.bucket == "arx-smart-money-backups"
        assert adapter.prefix == "rehearsal"
        assert adapter.endpoint_url == "https://test-account-12345.r2.cloudflarestorage.com"
        assert adapter.access_key_id == "test-access-key"
        assert adapter.secret_access_key == "test-secret-key"
        assert adapter.region_name == "auto"
        assert adapter.provider_name == "cloudflare_r2"
        assert adapter.is_external() is True

    def test_missing_credentials_fallback_to_local(self, monkeypatch):
        # Clear all S3/R2 variables and provider
        for k in list(os.environ.keys()):
            if "BACKUP" in k or "R2" in k or "S3" in k:
                monkeypatch.delenv(k, raising=False)

        adapter = get_backup_storage_adapter()
        assert isinstance(adapter, LocalOffVolumeStorageAdapter)
        assert adapter.is_external() is False
        assert adapter.provider_name == "local_off_volume"

    def test_r2_missing_credentials_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_PROVIDER", "cloudflare_r2")
        monkeypatch.delenv("ARX_BACKUP_BUCKET", raising=False)
        monkeypatch.delenv("ARX_R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("ARX_R2_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            get_backup_storage_adapter()
        assert "EXTERNAL_BACKUP_CONFIG_ERROR" in str(exc_info.value)
        assert "Fail-closed: silent fallback to local storage is strictly prohibited" in str(exc_info.value)

    def test_external_mandatory_without_credentials_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ARX_BACKUP_EXTERNAL_MANDATORY", "true")
        monkeypatch.delenv("ARX_BACKUP_BUCKET", raising=False)
        monkeypatch.delenv("ARX_R2_ACCESS_KEY_ID", raising=False)

        with pytest.raises(ValueError) as exc_info:
            get_backup_storage_adapter()
        assert "EXTERNAL_BACKUP_CONFIG_ERROR" in str(exc_info.value)


class TestR2StorageOperationsMocked:
    """Verifies S3/R2 API interactions using a mocked boto3 client."""

    @pytest.fixture
    def mock_s3_adapter(self, tmp_path):
        adapter = S3CompatibleStorageAdapter(
            bucket="arx-smart-money-backups",
            prefix="rehearsal",
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="auto",
        )
        adapter.provider_name = "cloudflare_r2"
        return adapter

    def test_upload_and_verify_head_object(self, mock_s3_adapter, tmp_path):
        test_file = tmp_path / "test_artifact.dump"
        test_file.write_bytes(b"test-database-content-bytes-12345")
        expected_sha = compute_sha256(str(test_file))
        file_size = len(b"test-database-content-bytes-12345")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": file_size}
        mock_s3_adapter._s3_client = mock_client

        receipt = mock_s3_adapter.upload_file(str(test_file), "test_artifact.dump")

        assert receipt["remote_filename"] == "test_artifact.dump"
        assert receipt["destination"] == "s3://arx-smart-money-backups/rehearsal/test_artifact.dump"
        assert receipt["sha256"] == expected_sha
        assert receipt["size_bytes"] == file_size
        assert receipt["is_external"] is True

        mock_client.put_object.assert_called_once()
        mock_client.head_object.assert_called_once_with(
            Bucket="arx-smart-money-backups",
            Key="rehearsal/test_artifact.dump",
        )

    def test_upload_corruption_size_mismatch_fails_closed(self, mock_s3_adapter, tmp_path):
        test_file = tmp_path / "corrupt_artifact.dump"
        test_file.write_bytes(b"1234567890")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 5}  # Mismatched size
        mock_s3_adapter._s3_client = mock_client

        with pytest.raises(IOError) as exc_info:
            mock_s3_adapter.upload_file(str(test_file), "corrupt_artifact.dump")
        assert "S3_UPLOAD_CORRUPTION" in str(exc_info.value)

    def test_list_and_delete_backups(self, mock_s3_adapter):
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "rehearsal/manifest_backup1.json",
                        "Size": 1024,
                        "LastModified": MagicMock(isoformat=lambda: "2026-10-14T05:00:00Z"),
                    }
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        mock_s3_adapter._s3_client = mock_client

        backups = mock_s3_adapter.list_backups()
        assert len(backups) == 1
        assert backups[0]["remote_filename"] == "manifest_backup1.json"

        assert mock_s3_adapter.delete_backup("manifest_backup1.json") is True
        mock_client.delete_object.assert_called_once_with(
            Bucket="arx-smart-money-backups",
            Key="rehearsal/manifest_backup1.json",
        )

    def test_download_tampering_raises_ioerror(self, mock_s3_adapter, tmp_path):
        dest_file = tmp_path / "downloaded.dump"

        mock_client = MagicMock()
        # Mock download_file writing corrupted bytes
        def fake_download(bucket, key, path):
            with open(path, "wb") as f:
                f.write(b"CORRUPTED_BYTES_FROM_TAMPERED_NETWORK")
        mock_client.download_file.side_effect = fake_download
        # Mock head_object returning an expected sha for the original valid content
        mock_client.head_object.return_value = {
            "Metadata": {"sha256": "original_valid_sha256_that_does_not_match"}
        }
        mock_s3_adapter._s3_client = mock_client

        with pytest.raises(IOError) as exc_info:
            mock_s3_adapter.download_file("test.dump", str(dest_file))
        assert "DOWNLOAD_CORRUPTION" in str(exc_info.value)
        assert not os.path.exists(str(dest_file))


class TestRetentionPruningProtection:
    """Verifies retention manager protects last recovery point."""

    def test_sole_remaining_backup_never_deleted(self):
        mock_adapter = MagicMock()
        # Single old backup
        mock_adapter.list_backups.return_value = [
            {
                "remote_filename": "manifest_backup_ancient.json",
                "size_bytes": 500,
                "modified_at": "2025-01-01T00:00:00Z",
            }
        ]
        manager = BackupRetentionManager(
            storage_adapter=mock_adapter,
            retention_count=1,
            retention_days=1,
        )
        pruned = manager.prune_backups()
        assert len(pruned) == 0
        mock_adapter.delete_backup.assert_not_called()
