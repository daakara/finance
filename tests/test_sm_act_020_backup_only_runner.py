"""Synthetic fail-closed tests for a backup-only, non-ingesting operator command."""
import pytest

from analyst_dashboard.smart_money.storage.backup import backup_once


SHA = "ba041b6d4e764f6450acb111f09ad056ed45cbb8"


@pytest.fixture
def safe_env():
    return {
        "ARX_BACKUP_ONESHOT_AUTHORIZED": "true",
        "ARX_ENVIRONMENT": "production",
        "ARX_SEC_EGRESS_ROLE": "DISABLED",
        "ARX_RELEASE_SHA": SHA,
        "SMART_MONEY_DATABASE_ROLE": "smart_money_backup",
        "ARX_BACKUP_PROVIDER": "cloudflare_r2",
        "ARX_BACKUP_BUCKET": "arx-smart-money-backups",
        "ARX_BACKUP_PREFIX": "production",
    }


@pytest.mark.parametrize("key,value", [
    ("ARX_BACKUP_ONESHOT_AUTHORIZED", "false"),
    ("ARX_ENVIRONMENT", "rehearsal"),
    ("ARX_SEC_EGRESS_ROLE", "OWNER"),
    ("ARX_SEC_EGRESS_ROLE", ""),
    ("ARX_RELEASE_SHA", "f" * 40),
    ("SMART_MONEY_DATABASE_ROLE", "smart_money_ingestion"),
    ("ARX_BACKUP_PROVIDER", "local"),
    ("ARX_BACKUP_BUCKET", "wrong-bucket"),
    ("ARX_BACKUP_PREFIX", "rehearsal"),
])
def test_fail_closed_when_any_explicit_gate_is_wrong(safe_env, key, value):
    safe_env[key] = value
    with pytest.raises(RuntimeError):
        backup_once.preflight(SHA, safe_env)


@pytest.mark.parametrize("expected", ["", "a" * 39, "not-a-commit", "A" * 40])
def test_requires_exact_lowercase_full_release_sha(safe_env, expected):
    with pytest.raises(RuntimeError):
        backup_once.preflight(expected, safe_env)


def test_fails_closed_if_authorization_omitted(safe_env):
    del safe_env["ARX_BACKUP_ONESHOT_AUTHORIZED"]
    with pytest.raises(RuntimeError):
        backup_once.preflight(SHA, safe_env)


def test_explicit_standby_role_permitted_but_never_owner(safe_env):
    safe_env["ARX_SEC_EGRESS_ROLE"] = "STANDBY"
    backup_once.preflight(SHA, safe_env)
    safe_env["ARX_SEC_EGRESS_ROLE"] = "OWNER"
    with pytest.raises(RuntimeError):
        backup_once.preflight(SHA, safe_env)


def test_missing_execute_does_not_touch_backup_engine(monkeypatch, capsys):
    monkeypatch.setattr(backup_once, "run_backup_once", lambda *_: pytest.fail("must not run"))
    assert backup_once.main(["--expected-release-sha", SHA]) == 2
    assert "NOT_EXECUTED" in capsys.readouterr().err


def test_preflight_failure_never_touches_storage_or_backup_engine(monkeypatch, safe_env):
    safe_env["ARX_BACKUP_ONESHOT_AUTHORIZED"] = "false"
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(backup_once, "get_backup_storage_adapter", lambda: pytest.fail("storage must not be accessed"))
    monkeypatch.setattr(backup_once, "SmartMoneyBackupEngine", lambda **_: pytest.fail("backup must not run"))
    with pytest.raises(RuntimeError, match="AUTHORIZATION"):
        backup_once.run_backup_once(SHA)


class FakeAdapter:
    provider_name = "cloudflare_r2"
    bucket = "arx-smart-money-backups"
    prefix = "production"

    def is_external(self):
        return True


def test_success_runs_exactly_once_without_globals_or_retention(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(backup_once, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_once, "get_backup_storage_adapter", FakeAdapter)
    calls = []

    class FakeEngine:
        def __init__(self, storage_adapter):
            assert isinstance(storage_adapter, FakeAdapter)

        def create_backup(self, include_globals):
            calls.append(include_globals)
            return {
                "backup_id": "synthetic-1",
                "status": "SUCCESS",
                "upload_verified": True,
                "is_external": True,
                "remote_bucket": "arx-smart-money-backups",
                "remote_object_prefix": "production",
                "source_release_sha": SHA,
                "restore_verified": False,
                "restore_validation_status": "PENDING_VERIFICATION",
            }

    monkeypatch.setattr(backup_once, "SmartMoneyBackupEngine", FakeEngine)
    result = backup_once.run_backup_once(SHA)
    assert calls == [False]
    assert result["upload_verified"] is True
    assert result["restore_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("status", "FAILED"),
    ("upload_verified", False),
    ("is_external", False),
    ("remote_bucket", "wrong"),
    ("remote_object_prefix", "rehearsal"),
    ("source_release_sha", "f" * 40),
])
def test_false_success_receipt_is_rejected(monkeypatch, safe_env, field, value):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(backup_once, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_once, "get_backup_storage_adapter", FakeAdapter)

    class FakeEngine:
        def __init__(self, storage_adapter):
            pass

        def create_backup(self, include_globals):
            receipt = {
                "backup_id": "synthetic-1",
                "status": "SUCCESS",
                "upload_verified": True,
                "is_external": True,
                "remote_bucket": "arx-smart-money-backups",
                "remote_object_prefix": "production",
                "source_release_sha": SHA,
            }
            receipt[field] = value
            return receipt

    monkeypatch.setattr(backup_once, "SmartMoneyBackupEngine", FakeEngine)
    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        backup_once.run_backup_once(SHA)


def test_local_adapter_never_executes_backup(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)

    class FakeLocal:
        provider_name = "local_off_volume"
        bucket = "local"
        prefix = ""
        def is_external(self):
            return False

    monkeypatch.setattr(backup_once, "verify_backup_database_identity", lambda: None)
    monkeypatch.setattr(backup_once, "get_backup_storage_adapter", FakeLocal)
    monkeypatch.setattr(backup_once, "SmartMoneyBackupEngine", lambda **_: pytest.fail("must not construct engine"))
    with pytest.raises(RuntimeError, match="EXTERNAL_CUSTODY"):
        backup_once.run_backup_once(SHA)


def test_cli_redacts_underlying_error(monkeypatch, capsys):
    def denied(_):
        raise RuntimeError("secret://should-not-appear")
    monkeypatch.setattr(backup_once, "run_backup_once", denied)
    assert backup_once.main(["--execute", "--expected-release-sha", SHA]) == 1
    stderr = capsys.readouterr().err
    assert "secret://" not in stderr
    assert "BACKUP_ONLY_FAILED" in stderr


def test_verified_db_identity_matches_real_session_user(monkeypatch):
    monkeypatch.setattr(backup_once, "get_database_url", lambda: "postgresql://fake@localhost/fake")
    events = []
    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql):
            assert sql == "SELECT session_user, current_user"
            events.append("query")
        def fetchone(self):
            return ("smart_money_backup", "smart_money_backup")
    class Connection:
        def set_session(self, readonly):
            assert readonly is True
            events.append("readonly")
        def cursor(self):
            return Cursor()
        def close(self):
            events.append("close")
    monkeypatch.setattr(backup_once.psycopg2, "connect", lambda *_args, **_kwargs: Connection())
    backup_once.verify_backup_database_identity()
    assert events == ["readonly", "query", "close"]


@pytest.mark.parametrize("identity", [
    ("smart_money_ingestion", "smart_money_ingestion"),
    ("postgres", "smart_money_backup"),
    ("smart_money_backup", "postgres"),
    None,
])
def test_actual_database_principal_is_fail_closed(monkeypatch, identity):
    monkeypatch.setattr(backup_once, "get_database_url", lambda: "postgresql://fake@localhost/fake")
    events = []
    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql):
            events.append("queried")
        def fetchone(self):
            return identity
    class Connection:
        def set_session(self, readonly):
            assert readonly
        def cursor(self):
            return Cursor()
        def close(self):
            events.append("closed")
    monkeypatch.setattr(backup_once.psycopg2, "connect", lambda *_a, **_k: Connection())
    with pytest.raises(RuntimeError, match="DATABASE_PRINCIPAL_MISMATCH"):
        backup_once.verify_backup_database_identity()
    assert events == ["queried", "closed"]


def test_database_connection_error_redacted_and_fail_closed(monkeypatch):
    monkeypatch.setattr(backup_once, "get_database_url", lambda: "postgresql://secret@localhost/fake")
    def fails(*_args, **_kwargs):
        raise RuntimeError("postgresql://secret@localhost/fake")
    monkeypatch.setattr(backup_once.psycopg2, "connect", fails)
    with pytest.raises(RuntimeError, match="BACKUP_ONLY_DATABASE_PRINCIPAL_UNVERIFIABLE") as exc:
        backup_once.verify_backup_database_identity()
    assert "secret@" not in str(exc.value)


def test_database_principal_denial_blocks_storage(monkeypatch, safe_env):
    for k, v in safe_env.items():
        monkeypatch.setenv(k, v)
    def deny():
        raise RuntimeError("BACKUP_ONLY_DATABASE_PRINCIPAL_MISMATCH")
    monkeypatch.setattr(backup_once, "verify_backup_database_identity", deny)
    monkeypatch.setattr(backup_once, "get_backup_storage_adapter", lambda: pytest.fail("R2 must not be initialized"))
    with pytest.raises(RuntimeError, match="PRINCIPAL_MISMATCH"):
        backup_once.run_backup_once(SHA)
