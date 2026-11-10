"""Focused positive, negative, and adversarial tests for the dedicated scheduled Smart Money backup runner."""
import os
import pytest

from analyst_dashboard.smart_money.storage.backup import backup_scheduled


SHA = "ba041b6d4e764f6450acb111f09ad056ed45cbb8"


@pytest.fixture
def safe_env():
    return {
        "ARX_BACKUP_SCHEDULED_AUTHORIZED": "true",
        "ARX_ENVIRONMENT": "production",
        "ARX_SEC_EGRESS_ROLE": "DISABLED",
        "ARX_RELEASE_SHA": SHA,
        "SMART_MONEY_DATABASE_ROLE": "smart_money_backup",
        "ARX_BACKUP_PROVIDER": "cloudflare_r2",
        "ARX_BACKUP_BUCKET": "arx-smart-money-backups",
        "ARX_BACKUP_PREFIX": "production",
    }


class FakeAdapter:
    provider_name = "cloudflare_r2"
    bucket = "arx-smart-money-backups"
    prefix = "production"

    def is_external(self):
        return True


# ---------------------------------------------------------------------------
# Preflight & Authorization Tests (Positive & Negative)
# ---------------------------------------------------------------------------

def test_preflight_success_with_matching_sha(safe_env):
    verified_sha = backup_scheduled.preflight(SHA, safe_env)
    assert verified_sha == SHA


def test_preflight_success_without_cli_arg(safe_env):
    verified_sha = backup_scheduled.preflight(None, safe_env)
    assert verified_sha == SHA


def test_preflight_success_with_railway_commit_sha(safe_env):
    del safe_env["ARX_RELEASE_SHA"]
    safe_env["RAILWAY_GIT_COMMIT_SHA"] = SHA
    verified_sha = backup_scheduled.preflight(None, safe_env)
    assert verified_sha == SHA


def test_preflight_accepts_trailing_slash_prefix(safe_env):
    safe_env["ARX_BACKUP_PREFIX"] = "production/"
    verified_sha = backup_scheduled.preflight(SHA, safe_env)
    assert verified_sha == SHA


def test_preflight_rejects_missing_authorization(safe_env):
    del safe_env["ARX_BACKUP_SCHEDULED_AUTHORIZED"]
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_AUTHORIZATION_MISSING"):
        backup_scheduled.preflight(SHA, safe_env)


def test_preflight_rejects_false_authorization(safe_env):
    safe_env["ARX_BACKUP_SCHEDULED_AUTHORIZED"] = "false"
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_AUTHORIZATION_MISSING"):
        backup_scheduled.preflight(SHA, safe_env)


def test_preflight_rejects_oneshot_authorization(safe_env):
    safe_env["ARX_BACKUP_ONESHOT_AUTHORIZED"] = "true"
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_REJECTS_ONESHOT_AUTHORIZATION"):
        backup_scheduled.preflight(SHA, safe_env)


@pytest.mark.parametrize("key,value,expected_error", [
    ("ARX_ENVIRONMENT", "rehearsal", "BACKUP_SCHEDULED_REQUIRES_PRODUCTION_ENVIRONMENT"),
    ("ARX_ENVIRONMENT", "staging", "BACKUP_SCHEDULED_REQUIRES_PRODUCTION_ENVIRONMENT"),
    ("ARX_SEC_EGRESS_ROLE", "OWNER", "BACKUP_SCHEDULED_REQUIRES_SEC_EGRESS_DISABLED"),
    ("ARX_SEC_EGRESS_ROLE", "", "BACKUP_SCHEDULED_REQUIRES_SEC_EGRESS_DISABLED"),
    ("SMART_MONEY_DATABASE_ROLE", "smart_money_ingestion", "BACKUP_SCHEDULED_REQUIRES_BACKUP_DATABASE_ROLE"),
    ("SMART_MONEY_DATABASE_ROLE", "postgres", "BACKUP_SCHEDULED_REQUIRES_BACKUP_DATABASE_ROLE"),
    ("ARX_BACKUP_PROVIDER", "local", "BACKUP_SCHEDULED_REQUIRES_CLOUDFLARE_R2"),
    ("ARX_BACKUP_PROVIDER", "s3", "BACKUP_SCHEDULED_REQUIRES_CLOUDFLARE_R2"),
    ("ARX_BACKUP_BUCKET", "wrong-bucket", "BACKUP_SCHEDULED_WRONG_BUCKET"),
    ("ARX_BACKUP_PREFIX", "rehearsal", "BACKUP_SCHEDULED_WRONG_PREFIX"),
    ("ARX_BACKUP_PREFIX", "staging", "BACKUP_SCHEDULED_WRONG_PREFIX"),
])
def test_preflight_fail_closed_on_invalid_env(safe_env, key, value, expected_error):
    safe_env[key] = value
    with pytest.raises(RuntimeError, match=expected_error):
        backup_scheduled.preflight(SHA, safe_env)


@pytest.mark.parametrize("bad_sha", [
    "",
    "a" * 39,
    "a" * 41,
    "not-a-sha",
    "G" * 40,
    "A" * 40,  # uppercase rejected
])
def test_preflight_rejects_invalid_release_sha(safe_env, bad_sha):
    safe_env["ARX_RELEASE_SHA"] = bad_sha
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_INVALID_RELEASE_IDENTITY"):
        backup_scheduled.preflight(None, safe_env)


def test_preflight_rejects_release_sha_mismatch(safe_env):
    other_sha = "1" * 40
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_RELEASE_IDENTITY_MISMATCH"):
        backup_scheduled.preflight(other_sha, safe_env)


def test_standby_role_permitted_in_preflight(safe_env):
    safe_env["ARX_SEC_EGRESS_ROLE"] = "STANDBY"
    assert backup_scheduled.preflight(SHA, safe_env) == SHA


# ---------------------------------------------------------------------------
# Database Identity Attestation Tests
# ---------------------------------------------------------------------------

def test_database_identity_verified_when_principal_matches(monkeypatch):
    monkeypatch.setattr(backup_scheduled, "get_database_url", lambda: "postgresql://fake@localhost/fake")
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            calls.append(sql)

        def fetchone(self):
            return ("smart_money_backup", "smart_money_backup")

    class Connection:
        def set_session(self, readonly):
            assert readonly is True
            calls.append("readonly")

        def cursor(self):
            return Cursor()

        def close(self):
            calls.append("close")

    monkeypatch.setattr(backup_scheduled.psycopg2, "connect", lambda *_a, **_k: Connection())
    backup_scheduled.verify_backup_database_identity()
    assert calls == ["readonly", "SELECT session_user, current_user", "close"]


@pytest.mark.parametrize("bad_principal", [
    ("postgres", "postgres"),
    ("smart_money_ingestion", "smart_money_ingestion"),
    ("smart_money_backup", "postgres"),
    ("postgres", "smart_money_backup"),
    None,
])
def test_database_identity_rejects_non_backup_principal(monkeypatch, bad_principal):
    monkeypatch.setattr(backup_scheduled, "get_database_url", lambda: "postgresql://fake@localhost/fake")

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql):
            pass

        def fetchone(self):
            return bad_principal

    class Connection:
        def set_session(self, readonly):
            pass

        def cursor(self):
            return Cursor()

        def close(self):
            pass

    monkeypatch.setattr(backup_scheduled.psycopg2, "connect", lambda *_a, **_k: Connection())
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_DATABASE_PRINCIPAL_MISMATCH"):
        backup_scheduled.verify_backup_database_identity()


def test_database_identity_connection_failure_redacts_credentials(monkeypatch):
    monkeypatch.setattr(backup_scheduled, "get_database_url", lambda: "postgresql://super_secret_pw@db.railway.internal/prod")

    def fails(*_args, **_kwargs):
        raise RuntimeError("Connection refused: postgresql://super_secret_pw@db.railway.internal/prod")

    monkeypatch.setattr(backup_scheduled.psycopg2, "connect", fails)
    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_DATABASE_PRINCIPAL_UNVERIFIABLE") as exc:
        backup_scheduled.verify_backup_database_identity()

    assert "super_secret_pw" not in str(exc.value)


# ---------------------------------------------------------------------------
# Storage Custody & Execution Tests
# ---------------------------------------------------------------------------

def test_rejects_local_or_internal_storage_custody(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)

    class FakeLocalAdapter:
        provider_name = "local_off_volume"
        bucket = "local"
        prefix = ""

        def is_external(self):
            return False

    monkeypatch.setattr(backup_scheduled, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_scheduled, "get_backup_storage_adapter", FakeLocalAdapter)
    monkeypatch.setattr(backup_scheduled, "SmartMoneyBackupEngine", lambda **_: pytest.fail("Engine must not be constructed"))

    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_EXTERNAL_CUSTODY_NOT_VERIFIED"):
        backup_scheduled.run_scheduled_backup(SHA)


def test_successful_scheduled_run_creates_backup_without_retention_or_globals(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)

    monkeypatch.setattr(backup_scheduled, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_scheduled, "get_backup_storage_adapter", FakeAdapter)

    calls = []

    class FakeEngine:
        def __init__(self, storage_adapter):
            assert isinstance(storage_adapter, FakeAdapter)

        def create_backup(self, include_globals):
            calls.append({"include_globals": include_globals})
            return {
                "backup_id": "sm-backup-20261110T230000Z-12345678",
                "status": "SUCCESS",
                "upload_verified": True,
                "is_external": True,
                "remote_bucket": "arx-smart-money-backups",
                "remote_object_prefix": "production",
                "source_release_sha": SHA,
                "artifact_sha256": "e" * 64,
                "database_sha256": "e" * 64,
                "artifact_size_bytes": 50400,
                "restore_verified": False,
                "restore_validation_status": "PENDING_VERIFICATION",
            }

    monkeypatch.setattr(backup_scheduled, "SmartMoneyBackupEngine", FakeEngine)

    result = backup_scheduled.run_scheduled_backup(SHA)

    # 1. Exactly one backup executed
    assert len(calls) == 1
    # 2. Never captures globals
    assert calls[0]["include_globals"] is False
    # 3. Verified return receipt
    assert result["upload_verified"] is True
    assert result["artifact_sha256"] == "e" * 64
    assert result["backup_id"] == "sm-backup-20261110T230000Z-12345678"
    assert result["restore_verified"] is False


@pytest.mark.parametrize("corrupt_field,corrupt_value", [
    ("status", "FAILED"),
    ("upload_verified", False),
    ("is_external", False),
    ("remote_bucket", "untrusted-bucket"),
    ("remote_object_prefix", "staging"),
    ("source_release_sha", "f" * 40),
    ("artifact_sha256", "bad_hash"),
    ("database_sha256", "different_hash"),
    ("backup_id", ""),
])
def test_manifest_integrity_violations_fail_closed(monkeypatch, safe_env, corrupt_field, corrupt_value):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)

    monkeypatch.setattr(backup_scheduled, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_scheduled, "get_backup_storage_adapter", FakeAdapter)

    class FakeEngine:
        def __init__(self, storage_adapter):
            pass

        def create_backup(self, include_globals):
            manifest = {
                "backup_id": "sm-backup-20261110T230000Z-12345678",
                "status": "SUCCESS",
                "upload_verified": True,
                "is_external": True,
                "remote_bucket": "arx-smart-money-backups",
                "remote_object_prefix": "production",
                "source_release_sha": SHA,
                "artifact_sha256": "e" * 64,
                "database_sha256": "e" * 64,
                "artifact_size_bytes": 50400,
            }
            manifest[corrupt_field] = corrupt_value
            return manifest

    monkeypatch.setattr(backup_scheduled, "SmartMoneyBackupEngine", FakeEngine)

    with pytest.raises(RuntimeError, match="BACKUP_SCHEDULED_UNVERIFIED_UPLOAD"):
        backup_scheduled.run_scheduled_backup(SHA)


def test_concurrent_backup_lock_contention_fails_closed(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)

    monkeypatch.setattr(backup_scheduled, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_scheduled, "get_backup_storage_adapter", FakeAdapter)

    class LockedEngine:
        def __init__(self, storage_adapter):
            pass

        def create_backup(self, include_globals):
            raise RuntimeError("CONCURRENT_BACKUP_BLOCKED: Another backup execution is active.")

    monkeypatch.setattr(backup_scheduled, "SmartMoneyBackupEngine", LockedEngine)

    with pytest.raises(RuntimeError, match="CONCURRENT_BACKUP_BLOCKED"):
        backup_scheduled.run_scheduled_backup(SHA)


# ---------------------------------------------------------------------------
# CLI Entry Point & Clean Exit Tests
# ---------------------------------------------------------------------------

def test_cli_success_exit_code_zero(monkeypatch, capsys):
    monkeypatch.setattr(
        backup_scheduled,
        "run_scheduled_backup",
        lambda expected_release_sha: {
            "backup_id": "sm-backup-test-1",
            "upload_verified": True,
            "artifact_sha256": "a" * 64,
            "restore_verified": False,
        },
    )

    exit_code = backup_scheduled.main(["--expected-release-sha", SHA])
    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "BACKUP_SCHEDULED_SUCCESS: sm-backup-test-1 sha256=" in stdout
    assert "BACKUP_SCHEDULED_RESTORE_VERIFICATION_PENDING" in stdout


def test_cli_failure_redacts_secrets_and_exits_one(monkeypatch, capsys):
    def fails_with_secret(expected_release_sha):
        raise RuntimeError("database error postgresql://secret_user:super_secret_pw@db.railway.internal:5432/db")

    monkeypatch.setattr(backup_scheduled, "run_scheduled_backup", fails_with_secret)

    exit_code = backup_scheduled.main([])
    assert exit_code == 1

    stderr = capsys.readouterr().err
    assert "super_secret_pw" not in stderr
    assert "postgresql://" not in stderr
    assert "BACKUP_SCHEDULED_FAILED: RuntimeError" in stderr
