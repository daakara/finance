"""
Tests for Smart Money Lazy Dependency Loading & Fail-Closed Driver Isolation.
Verifies that SQLite development/testing runs cleanly without PostgreSQL drivers,
and that attempting to use PostgreSQL features without drivers fails closed.
"""

import pytest
from analyst_dashboard.smart_money.storage.factory import (
    get_smart_money_storage_backend,
    get_smart_money_scanner_runner,
    get_smart_money_snapshot_store,
    get_smart_money_repository,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    PSYCOPG2_AVAILABLE as REPO_PG_AVAIL,
)
from analyst_dashboard.smart_money.storage.postgres_connection import (
    get_connection_pool,
    PSYCOPG2_AVAILABLE as CONN_PG_AVAIL,
)
from analyst_dashboard.smart_money.storage.migration import (
    SmartMoneySqliteMigrator,
    PSYCOPG2_AVAILABLE as MIG_PG_AVAIL,
)
from analyst_dashboard.smart_money.storage.backup.backup_engine import (
    SmartMoneyBackupEngine,
    PSYCOPG2_AVAILABLE as BACKUP_PG_AVAIL,
)


def test_sqlite_mode_defaults_and_runner_creation():
    """Verify that default backend is sqlite and runner creates cleanly without postgres drivers."""
    backend = get_smart_money_storage_backend()
    assert backend == "sqlite"

    runner = get_smart_money_scanner_runner()
    assert runner.backend == "sqlite"
    assert hasattr(runner, "snapshot_store")

    env = runner.get_status_envelope()
    assert env["scanner_id"] == "SMART_MONEY"
    assert env["status"] == "PIPELINE_PENDING"
    assert env["snapshot"]["matched_count"] is None


def test_sqlite_snapshot_store_resolution():
    """Verify that snapshot store in sqlite mode resolves to ScannerSnapshotStore."""
    store = get_smart_money_snapshot_store(backend="sqlite")
    assert store.__class__.__name__ == "ScannerSnapshotStore"


def test_postgres_repository_unconfigured_dsn_fails_closed():
    """Verify that requesting postgres repository without configured DSN raises RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        get_smart_money_repository(dsn=None)
    assert "CANONICAL_POSTGRES_REPOSITORY_UNAVAILABLE" in str(exc_info.value)


def test_postgres_components_fail_closed_when_driver_missing():
    """
    Verify fail-closed semantics: if psycopg2 is absent, attempting to acquire connections
    or execute operations via PostgresSmartMoneyRepository, connection pool, migrator, or backup engine
    raises CANONICAL_POSTGRES_DRIVER_UNAVAILABLE.
    """
    if not REPO_PG_AVAIL:
        repo = PostgresSmartMoneyRepository(dsn="postgresql://user:pass@localhost:5432/db")
        with pytest.raises(RuntimeError) as exc_info:
            repo.get_latest_active_snapshot()
        assert "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE" in str(exc_info.value)

    if not CONN_PG_AVAIL:
        with pytest.raises(RuntimeError) as exc_info:
            get_connection_pool(dsn="postgresql://user:pass@localhost:5432/db")
        assert "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE" in str(exc_info.value)

    if not MIG_PG_AVAIL:
        with pytest.raises(RuntimeError) as exc_info:
            SmartMoneySqliteMigrator(sqlite_db_path="dummy.db", pg_dsn="postgresql://user:pass@localhost:5432/db")
        assert "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE" in str(exc_info.value)

    if not BACKUP_PG_AVAIL:
        engine = SmartMoneyBackupEngine(dsn="postgresql://user:pass@localhost:5432/db")
        with pytest.raises(RuntimeError) as exc_info:
            engine.create_backup()
        assert "CANONICAL_POSTGRES_DRIVER_UNAVAILABLE" in str(exc_info.value)
