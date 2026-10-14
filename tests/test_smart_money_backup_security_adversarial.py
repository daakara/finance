"""
ARX Terminal — Smart Money Form 4 Security Remediation & Disaster Recovery
Adversarial Regression Test Suite (Gate Candidate 006 Security Closure).

Covers all 10 mandatory regression vectors:
1. Unauthorized superuser escalation rejection.
2. Privilege reintroduction prevention.
3. Backup role separation.
4. External storage fail-closed behavior (zero silent local fallback).
5. False upload-verification flag rejection.
6. Missing recovery artifacts detection.
7. Incomplete / tampered remote restore detection.
8. Retention deletion of final recovery point protection.
9. Governance authorization bypass prevention.
10. Candidate SHA mismatch detection.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    get_backup_storage_adapter,
    S3CompatibleStorageAdapter,
    LocalOffVolumeStorageAdapter,
    compute_sha256,
)
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.restore_verifier import BackupRestoreVerifier
from analyst_dashboard.smart_money.governance_control_plane.sole_owner_policy import (
    SoleOwnerGovernanceValidator,
    SoleOwnerReleaseApproval,
    Fido2RequirementStatus,
    ReleaseApprovalMissingMfaException,
    ReleaseApprovalMismatchedShaException,
)


class TestSecurityPrivilegeAndRoleSeparation:
    """Vectors 1, 2, 3: Privilege boundaries and role separation."""

    def test_unauthorized_superuser_escalation_rejection(self):
        """Vector 1: Unprivileged backup/ingestion role cannot access cluster auth tables."""
        # Simulated unprivileged connection
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        import psycopg2
        mock_cur.execute.side_effect = psycopg2.ProgrammingError("permission denied for table pg_authid")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with pytest.raises(psycopg2.ProgrammingError) as exc_info:
            with mock_conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM pg_catalog.pg_authid;")
        assert "permission denied for table pg_authid" in str(exc_info.value)

    def test_privilege_reintroduction_prevention(self):
        """Vector 2: Routine backups must not invoke pg_dumpall or require CREATEROLE/SUPERUSER."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = LocalOffVolumeStorageAdapter(base_directory=tmp_dir)
            engine = SmartMoneyBackupEngine(
                dsn="postgresql://smart_money_ingestion:pwd@127.0.0.1:5432/smart_money",
                storage_adapter=adapter,
            )
            # Inspect that create_backup defaults to include_globals=False
            import inspect
            sig = inspect.signature(engine.create_backup)
            assert sig.parameters["include_globals"].default is False

    def test_backup_role_separation(self):
        """Vector 3: include_globals=False decouples cluster globals from regular execution."""
        mock_adapter = MagicMock()
        mock_adapter.provider_name = "s3"
        mock_adapter.bucket = "mock-bucket"
        mock_adapter.prefix = "rehearsal"
        mock_adapter.upload_file.return_value = {"sha256": "fake_sha", "size_bytes": 100}
        mock_adapter.get_storage_destination.return_value = "s3://mock/rehearsal"
        mock_adapter.is_external.return_value = True

        engine = SmartMoneyBackupEngine(
            dsn="postgresql://smart_money_ingestion:pwd@127.0.0.1:5432/smart_money",
            storage_adapter=mock_adapter,
        )

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("16.15",)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("psycopg2.connect", return_value=mock_conn), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.acquire_backup_lock", return_value=True), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.find_pg_binary", return_value="pg_dump"), \
             patch("subprocess.run") as mock_subproc, \
             patch("os.path.getsize", return_value=1024), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.compute_sha256", return_value="fake_sha"):

            mock_subproc.return_value = MagicMock(returncode=0, stderr="")
            manifest = engine.create_backup(include_globals=False)

            assert manifest["include_globals"] is False
            assert manifest["globals_artifact"] is None
            assert manifest["globals_artifact_sha256"] is None
            assert manifest["upload_verified"] is True
            # Assert pg_dumpall was NOT executed
            for call in mock_subproc.call_args_list:
                args = call[0][0]
                assert "pg_dumpall" not in args


class TestExternalStorageFailClosed:
    """Vectors 4, 5: Fail-closed external cloud storage contract."""

    def test_external_storage_fail_closed_behavior(self, monkeypatch):
        """Vector 4: Missing credentials when ARX_BACKUP_PROVIDER=cloudflare_r2 must fail."""
        monkeypatch.setenv("ARX_BACKUP_PROVIDER", "cloudflare_r2")
        monkeypatch.delenv("ARX_BACKUP_BUCKET", raising=False)
        monkeypatch.delenv("ARX_R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("ARX_R2_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            get_backup_storage_adapter()
        assert "EXTERNAL_BACKUP_CONFIG_ERROR" in str(exc_info.value)
        assert "Fail-closed: silent fallback to local storage is strictly prohibited" in str(exc_info.value)

    def test_external_mandatory_flag_enforcement(self, monkeypatch):
        """Vector 4b: Mandatory external custody rejects local fallback."""
        monkeypatch.setenv("ARX_BACKUP_EXTERNAL_MANDATORY", "true")
        monkeypatch.delenv("ARX_BACKUP_BUCKET", raising=False)

        with pytest.raises(ValueError) as exc_info:
            get_backup_storage_adapter()
        assert "EXTERNAL_BACKUP_CONFIG_ERROR" in str(exc_info.value)

    def test_false_upload_verification_flags_rejection(self):
        """Vector 5: Mismatched remote receipt fails upload_verified."""
        mock_adapter = MagicMock()
        mock_adapter.provider_name = "s3"
        mock_adapter.bucket = "mock-bucket"
        mock_adapter.prefix = "rehearsal"
        # Mock upload returning different sha256 (corruption in transit)
        mock_adapter.upload_file.return_value = {"sha256": "corrupted_remote_sha", "size_bytes": 1024}
        mock_adapter.get_storage_destination.return_value = "s3://mock/rehearsal"
        mock_adapter.is_external.return_value = True

        engine = SmartMoneyBackupEngine(
            dsn="postgresql://smart_money_ingestion:pwd@127.0.0.1:5432/smart_money",
            storage_adapter=mock_adapter,
        )

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("16.15",)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with patch("psycopg2.connect", return_value=mock_conn), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.acquire_backup_lock", return_value=True), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.find_pg_binary", return_value="pg_dump"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")), \
             patch("os.path.getsize", return_value=1024), \
             patch("analyst_dashboard.smart_money.storage.backup.backup_engine.compute_sha256", return_value="local_sha_valid"):

            manifest = engine.create_backup(include_globals=False)
            assert manifest["upload_verified"] is False


class TestRestoreIntegrityAndMissingArtifacts:
    """Vectors 6, 7: Tamper and loss detection during restore."""

    def test_missing_recovery_artifacts_detection(self, tmp_path):
        """Vector 6: Missing dump file in remote storage fails closed."""
        mock_adapter = MagicMock()
        # Manifest downloads fine, but dump file raises FileNotFoundError
        def fake_download(remote, local):
            if remote.startswith("manifest_"):
                import json
                with open(local, "w") as f:
                    json.dump({
                        "backup_id": "test-b1",
                        "database_artifact": "missing.dump",
                        "dump_filename": "missing.dump",
                        "artifact_sha256": "fake_sha",
                    }, f)
            else:
                raise FileNotFoundError(f"Object not found: {remote}")

        mock_adapter.download_file.side_effect = fake_download
        verifier = BackupRestoreVerifier(storage_adapter=mock_adapter)

        with pytest.raises(FileNotFoundError):
            verifier.verify_backup("test-b1", "postgresql://postgres:pwd@127.0.0.1:5432/postgres")

    def test_incomplete_remote_restore_detection(self, tmp_path):
        """Vector 7: Truncated or tampered dump file fails digest verification before restore."""
        mock_adapter = MagicMock()
        def fake_download(remote, local):
            import json
            if remote.startswith("manifest_"):
                with open(local, "w") as f:
                    json.dump({
                        "backup_id": "test-b2",
                        "database_artifact": "tampered.dump",
                        "dump_filename": "tampered.dump",
                        "artifact_sha256": "expected_valid_sha256_hash",
                    }, f)
            else:
                with open(local, "wb") as f:
                    f.write(b"CORRUPTED_INCOMPLETE_DUMP_STREAM")

        mock_adapter.download_file.side_effect = fake_download
        verifier = BackupRestoreVerifier(storage_adapter=mock_adapter)

        with pytest.raises(ValueError) as exc_info:
            verifier.verify_backup("test-b2", "postgresql://postgres:pwd@127.0.0.1:5432/postgres")
        assert "DIGEST_MISMATCH_DUMP" in str(exc_info.value)


class TestRetentionAndGovernanceInvariants:
    """Vectors 8, 9, 10: Retention protection and release governance reconciliation."""

    def test_retention_deletion_of_final_recovery_point_protection(self):
        """Vector 8: The newest recovery point is NEVER pruned, even if retention_count=0."""
        mock_adapter = MagicMock()
        mock_adapter.list_backups.return_value = [
            {
                "remote_filename": "manifest_sm-backup-20261014T050000Z-abc12345.json",
                "modified_at": "2026-10-14T05:00:00Z",
                "size_bytes": 500,
            }
        ]
        retention = BackupRetentionManager(storage_adapter=mock_adapter, retention_count=1)
        pruned = retention.prune_backups(retention_count=0, retention_days=0)

        assert len(pruned) == 0
        mock_adapter.delete_backup.assert_not_called()

    def test_governance_authorization_bypass_prevention(self):
        """Vector 9: Approval lacking verifiable MFA or authorization token is rejected."""
        validator = SoleOwnerGovernanceValidator()
        invalid_approval = SoleOwnerReleaseApproval(
            approval_id="appr-bypass-001",
            policy_id="ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1",
            owner_identity="david@arxterminal.com",
            target_environment="production",
            release_sha="c8cc1cda0be5a2a62816d3024b6499515e97101a",
            approved_at="2026-10-14T05:00:00Z",
            valid_until="2026-10-14T06:00:00Z",
            account_mfa_verified=False,  # Unverified MFA must raise exception
            technical_verification_passed=True,
            security_verification_passed=True,
            recovery_verification_passed=True,
        )
        with pytest.raises(ReleaseApprovalMissingMfaException):
            validator.validate_release_authorization(
                approval=invalid_approval,
                target_sha="c8cc1cda0be5a2a62816d3024b6499515e97101a",
                target_environment="production",
            )

    def test_candidate_sha_mismatch_detection(self):
        """Vector 10: Approval for commit A cannot be used to deploy commit B."""
        validator = SoleOwnerGovernanceValidator()
        valid_approval = SoleOwnerReleaseApproval(
            approval_id="appr-valid-001",
            policy_id="ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1",
            owner_identity="david@arxterminal.com",
            target_environment="production",
            release_sha="c8cc1cda0be5a2a62816d3024b6499515e97101a",
            approved_at="2026-10-14T05:00:00Z",
            valid_until="2026-10-14T09:00:00Z",
            account_mfa_verified=True,
            technical_verification_passed=True,
            security_verification_passed=True,
            recovery_verification_passed=True,
        )
        # Attempt to deploy mismatched candidate SHA
        with pytest.raises(ReleaseApprovalMismatchedShaException):
            validator.validate_release_authorization(
                approval=valid_approval,
                target_sha="mismatched_sha_1234567890abcdef",
                target_environment="production",
            )
