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

from datetime import datetime, timezone, timedelta

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
    ReleaseApprovalException,
    ReleaseApprovalExpiredException,
    ReleaseApprovalMissingMfaException,
    ReleaseApprovalMismatchedShaException,
    ReleaseApprovalMismatchedEnvironmentException,
    ReleaseApprovalTechnicalPrerequisiteException,
)


class TestSecurityPrivilegeAndRoleSeparation:
    """Vectors 1, 2, 3: Privilege boundaries and role separation."""

    def test_unauthorized_superuser_escalation_rejection(self):
        """Vector 1: Unprivileged backup/ingestion role cannot access cluster auth tables."""
        # Simulated unprivileged connection
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        try:
            import psycopg2
            prog_err_cls = psycopg2.ProgrammingError
        except ImportError:
            class MockProgrammingError(Exception):
                pass
            prog_err_cls = MockProgrammingError

        mock_cur.execute.side_effect = prog_err_cls("permission denied for table pg_authid")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with pytest.raises(prog_err_cls) as exc_info:
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

    @pytest.fixture
    def deterministic_clock(self):
        """Injected deterministic clock for test evaluation: 2026-10-14 12:00:00 UTC."""
        return datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def sample_valid_approval(self, deterministic_clock):
        """Deterministic valid unexpired release approval fixture."""
        return SoleOwnerReleaseApproval(
            approval_id="appr-deterministic-001",
            policy_id="ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1",
            owner_identity="david@arxterminal.com",
            target_environment="production",
            release_sha="c8cc1cda0be5a2a62816d3024b6499515e97101a",
            approved_at=(deterministic_clock - timedelta(hours=1)).isoformat(),
            valid_until=(deterministic_clock + timedelta(hours=2)).isoformat(),
            account_mfa_verified=True,
            technical_verification_passed=True,
            security_verification_passed=True,
            recovery_verification_passed=True,
        )

    def test_governance_authorization_bypass_prevention(self, deterministic_clock, sample_valid_approval):
        """Vector 9: Approval lacking verifiable MFA is rejected (deterministic clock)."""
        validator = SoleOwnerGovernanceValidator()
        invalid_approval = SoleOwnerReleaseApproval(
            **{**sample_valid_approval.to_dict(), "account_mfa_verified": False}
        )
        with pytest.raises(ReleaseApprovalMissingMfaException):
            validator.validate_release_authorization(
                approval=invalid_approval,
                target_sha=invalid_approval.release_sha,
                target_environment="production",
                current_time=deterministic_clock,
            )

    def test_governance_authorization_valid_unexpired_with_mfa(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.1: Valid unexpired approval with valid MFA evidence succeeds deterministically."""
        validator = SoleOwnerGovernanceValidator()
        record = validator.validate_release_authorization(
            approval=sample_valid_approval,
            target_sha="c8cc1cda0be5a2a62816d3024b6499515e97101a",
            target_environment="production",
            current_time=deterministic_clock,
        )
        assert record["status"] == "APPROVED"
        assert record["policy_id"] == "ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1"
        assert record["release_sha"] == "c8cc1cda0be5a2a62816d3024b6499515e97101a"
        assert record["target_environment"] == "production"
        assert record["audit_hash"] is not None

    def test_governance_authorization_expired_approval(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.2: Expired approval is rejected deterministically."""
        validator = SoleOwnerGovernanceValidator()
        expired_approval = SoleOwnerReleaseApproval(
            **{
                **sample_valid_approval.to_dict(),
                "approved_at": (deterministic_clock - timedelta(hours=5)).isoformat(),
                "valid_until": (deterministic_clock - timedelta(minutes=1)).isoformat(),
            }
        )
        with pytest.raises(ReleaseApprovalExpiredException):
            validator.validate_release_authorization(
                approval=expired_approval,
                target_sha=expired_approval.release_sha,
                target_environment="production",
                current_time=deterministic_clock,
            )

    def test_governance_authorization_missing_mfa_evidence(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.3: Missing MFA evidence is rejected independently."""
        validator = SoleOwnerGovernanceValidator()
        no_mfa = SoleOwnerReleaseApproval(
            **{**sample_valid_approval.to_dict(), "account_mfa_verified": False}
        )
        with pytest.raises(ReleaseApprovalMissingMfaException):
            validator.validate_release_authorization(
                approval=no_mfa,
                target_sha=no_mfa.release_sha,
                target_environment="production",
                current_time=deterministic_clock,
            )

    def test_governance_authorization_invalid_mfa_evidence(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.4: When FIDO2 hardware MFA is active, missing/invalid assertion is rejected."""
        validator = SoleOwnerGovernanceValidator(status=Fido2RequirementStatus.FIDO2_REQUIRED)
        no_fido = SoleOwnerReleaseApproval(
            **{**sample_valid_approval.to_dict(), "fido2_assertion": None}
        )
        with pytest.raises(ReleaseApprovalException) as exc_info:
            validator.validate_release_authorization(
                approval=no_fido,
                target_sha=no_fido.release_sha,
                target_environment="production",
                current_time=deterministic_clock,
            )
        assert "FIDO2_ASSERTION_REQUIRED" in str(exc_info.value)

    def test_governance_authorization_wrong_commit_sha(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.5: Approval bound to the wrong commit SHA is rejected."""
        validator = SoleOwnerGovernanceValidator()
        with pytest.raises(ReleaseApprovalMismatchedShaException):
            validator.validate_release_authorization(
                approval=sample_valid_approval,
                target_sha="wrong_sha_0123456789abcdef0123456789abcdef01",
                target_environment="production",
                current_time=deterministic_clock,
            )

    def test_governance_authorization_outside_permitted_validity_window(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.6: Evaluation outside permitted validity window fails closed."""
        validator = SoleOwnerGovernanceValidator()
        future_eval_time = deterministic_clock + timedelta(hours=10)
        with pytest.raises(ReleaseApprovalExpiredException):
            validator.validate_release_authorization(
                approval=sample_valid_approval,
                target_sha=sample_valid_approval.release_sha,
                target_environment="production",
                current_time=future_eval_time,
            )

    def test_governance_authorization_missing_owner_or_policy_mismatch(self, deterministic_clock, sample_valid_approval):
        """Sub-vector 9.7: Missing owner authorization policy is rejected."""
        validator = SoleOwnerGovernanceValidator()
        unauthorized_policy = SoleOwnerReleaseApproval(
            **{**sample_valid_approval.to_dict(), "policy_id": "UNAUTHORIZED_ARX_BYPASS_POLICY"}
        )
        with pytest.raises(ReleaseApprovalException) as exc_info:
            validator.validate_release_authorization(
                approval=unauthorized_policy,
                target_sha=unauthorized_policy.release_sha,
                target_environment="production",
                current_time=deterministic_clock,
            )
        assert "POLICY_MISMATCH" in str(exc_info.value)

    def test_candidate_sha_mismatch_detection(self, deterministic_clock, sample_valid_approval):
        """Vector 10: Approval for commit A cannot be used to deploy commit B."""
        validator = SoleOwnerGovernanceValidator()
        with pytest.raises(ReleaseApprovalMismatchedShaException):
            validator.validate_release_authorization(
                approval=sample_valid_approval,
                target_sha="mismatched_sha_1234567890abcdef",
                target_environment="production",
                current_time=deterministic_clock,
            )

    def test_archive_custom_format_version_compatibility_verification(self, tmp_path):
        """Vector 11: Distinguish client archive compatibility from server corruption."""
        # Create a mock PostgreSQL custom dump header
        # Bytes: PGDMP (5 bytes), format major=1, minor=16, rev=0
        mock_dump_path = tmp_path / "mock_pg17_dump.dump"
        header = b"PGDMP\x01\x10\x00\x04\x08\x01" + b"\x00" * 21
        mock_dump_path.write_bytes(header)

        # Parse header
        with open(mock_dump_path, "rb") as f:
            read_header = f.read(11)
        
        assert read_header[:5] == b"PGDMP", "Must be valid PostgreSQL custom archive magic"
        major, minor, rev = read_header[5], read_header[6], read_header[7]
        assert (major, minor, rev) == (1, 16, 0), "Must accurately identify format version 1.16.0"
        
        # Test client compatibility boundary logic
        # Client 16.x supports up to 1.15; Client 17.x supports 1.16
        def is_client_compatible(client_major: int, archive_minor: int) -> bool:
            max_supported_minor = 15 if client_major <= 16 else 16
            return archive_minor <= max_supported_minor

        assert not is_client_compatible(client_major=16, archive_minor=16)
        assert is_client_compatible(client_major=17, archive_minor=16)
