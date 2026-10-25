"""
ARX Terminal — Smart Money SM-1 Lazy Import & Radar Conformance Test Suite.

Verifies:
1. SQLite-only environments load without psycopg2.
2. PostgreSQL mode fails with actionable POSTGRESQL_DRIVER_UNAVAILABLE error when psycopg2 is missing.
3. No silent fallback from production PostgreSQL to ephemeral SQLite.
4. Snapshot state distinguishes AVAILABLE, zero matches, and PIPELINE_PENDING.
5. Dual-capability integrity (VCP + SMART_MONEY) in get_radar_capabilities.
"""

import sys
import os
import pytest

pytestmark = pytest.mark.tier2c
from contextlib import contextmanager


@contextmanager
def simulate_missing_psycopg2():
    """Context manager that simulates an environment where psycopg2 is completely missing,
    properly evicting cached repository modules so re-import is forced."""
    orig_modules = {}
    keys_to_evict = [
        k for k in list(sys.modules.keys())
        if k == "psycopg2"
        or k.startswith("psycopg2.")
        or k.startswith("analyst_dashboard.smart_money.storage.")
        or k == "analyst_dashboard.smart_money.storage"
    ]
    for k in keys_to_evict:
        orig_modules[k] = sys.modules.pop(k, None)

    sys.modules["psycopg2"] = None
    sys.modules["psycopg2.extras"] = None
    sys.modules["psycopg2.extensions"] = None

    try:
        yield
    finally:
        sys.modules.pop("psycopg2", None)
        sys.modules.pop("psycopg2.extras", None)
        sys.modules.pop("psycopg2.extensions", None)
        for k, mod in orig_modules.items():
            if mod is not None:
                sys.modules[k] = mod


def test_sqlite_mode_loads_without_psycopg2():
    """Verify factory and storage modules load cleanly in SQLite mode when psycopg2 is unavailable."""
    with simulate_missing_psycopg2():
        import analyst_dashboard.smart_money.storage.factory as factory

        assert factory.get_smart_money_storage_backend() in ("sqlite", "postgres")

        # Load SQLite snapshot store and runner
        store = factory.get_smart_money_snapshot_store(backend="sqlite")
        assert store.__class__.__name__ == "ScannerSnapshotStore"

        runner = factory.get_smart_money_scanner_runner(backend="sqlite")
        assert runner.__class__.__name__ == "SmartMoneyScannerRunner"


def test_postgres_mode_fails_clearly_when_driver_unavailable():
    """Verify PostgreSQL mode raises actionable POSTGRESQL_DRIVER_UNAVAILABLE when psycopg2 is missing."""
    with simulate_missing_psycopg2():
        import analyst_dashboard.smart_money.storage.factory as factory

        # 1. Direct get_smart_money_repository with DSN
        with pytest.raises(RuntimeError) as exc_info:
            factory.get_smart_money_repository(dsn="postgresql://user:pass@localhost:5432/arx_db")
        assert "POSTGRESQL_DRIVER_UNAVAILABLE" in str(exc_info.value)
        assert "psycopg2-binary" in str(exc_info.value)

        # 2. get_smart_money_snapshot_store with postgres backend and configured URL
        orig_url = os.environ.get("SMART_MONEY_DATABASE_URL")
        try:
            os.environ["SMART_MONEY_DATABASE_URL"] = "postgresql://user:pass@localhost:5432/arx_db"
            with pytest.raises(RuntimeError) as exc_info2:
                factory.get_smart_money_snapshot_store(backend="postgres")
            assert "POSTGRESQL_DRIVER_UNAVAILABLE" in str(exc_info2.value)
        finally:
            if orig_url is not None:
                os.environ["SMART_MONEY_DATABASE_URL"] = orig_url
            else:
                os.environ.pop("SMART_MONEY_DATABASE_URL", None)


def test_no_silent_fallback_from_postgres_to_sqlite():
    """Verify that when postgres is selected, missing dependencies fail closed rather than falling back to SQLite."""
    with simulate_missing_psycopg2():
        orig_url = os.environ.get("SMART_MONEY_DATABASE_URL")
        try:
            os.environ["SMART_MONEY_DATABASE_URL"] = "postgresql://user:pass@localhost:5432/arx_db"
            import analyst_dashboard.smart_money.storage.factory as factory

            with pytest.raises(RuntimeError) as exc:
                store = factory.get_smart_money_snapshot_store(backend="postgres")
            # Must raise driver unavailable error, NOT return ScannerSnapshotStore
            assert "POSTGRESQL_DRIVER_UNAVAILABLE" in str(exc.value)
        finally:
            if orig_url is not None:
                os.environ["SMART_MONEY_DATABASE_URL"] = orig_url
            else:
                os.environ.pop("SMART_MONEY_DATABASE_URL", None)


def test_migrator_fails_clearly_when_driver_unavailable():
    """Verify SmartMoneySqliteMigrator fails with actionable error when driver is missing."""
    with simulate_missing_psycopg2():
        from analyst_dashboard.smart_money.storage.migration import SmartMoneySqliteMigrator

        with pytest.raises(RuntimeError) as exc:
            SmartMoneySqliteMigrator(sqlite_db_path="dummy.db", pg_dsn="postgresql://localhost/db")
        assert "POSTGRESQL_DRIVER_UNAVAILABLE" in str(exc.value)


def test_radar_capabilities_state_distinctions():
    """Verify get_radar_capabilities accurately distinguishes AVAILABLE (with matches & zero matches) vs PIPELINE_PENDING."""
    from api.routes.screener import get_radar_capabilities

    # Case 1: Pending snapshot
    pending_caps = get_radar_capabilities(sm_snapshot=None)
    assert pending_caps["SMART_MONEY"]["status"] == "PIPELINE_PENDING"
    assert pending_caps["SMART_MONEY"]["universeScreening"] == "PIPELINE_PENDING"
    assert pending_caps["SMART_MONEY"]["singleAssetAnalysis"] == "AVAILABLE"

    # Case 2: Available with matches
    active_snap = {
        "status": "AVAILABLE",
        "matched_count": 14,
        "freshness": {"data_as_of": "2026-10-24T20:00:00Z"},
    }
    avail_caps = get_radar_capabilities(sm_snapshot=active_snap)
    assert avail_caps["SMART_MONEY"]["status"] == "AVAILABLE"
    assert avail_caps["SMART_MONEY"]["universeScreening"] == "AVAILABLE"
    assert avail_caps["SMART_MONEY"]["matchedCount"] == 14
    assert avail_caps["SMART_MONEY"]["dataAsOf"] == "2026-10-24T20:00:00Z"

    # Case 3: Available with zero matches (distinct from pending!)
    zero_snap = {
        "status": "AVAILABLE",
        "matched_count": 0,
        "freshness": {"data_as_of": "2026-10-24T20:00:00Z"},
    }
    zero_caps = get_radar_capabilities(sm_snapshot=zero_snap)
    assert zero_caps["SMART_MONEY"]["status"] == "AVAILABLE"
    assert zero_caps["SMART_MONEY"]["universeScreening"] == "AVAILABLE"
    assert zero_caps["SMART_MONEY"]["matchedCount"] == 0
    assert zero_caps["SMART_MONEY"]["status"] != "PIPELINE_PENDING"


def test_radar_capabilities_dual_concurrency():
    """Verify get_radar_capabilities preserves both VCP and SMART_MONEY capabilities concurrently."""
    from api.routes.screener import get_radar_capabilities

    vcp_snap = {
        "status": "AVAILABLE",
        "matched_count": 8,
        "freshness": {"data_as_of": "2026-10-24T18:00:00Z", "status": "LIVE"},
        "universe": {"eligible_universe_count": 3500},
    }
    sm_snap = {
        "status": "AVAILABLE",
        "matched_count": 5,
        "freshness": {"data_as_of": "2026-10-24T19:00:00Z"},
    }

    caps = get_radar_capabilities(vcp_snapshot=vcp_snap, sm_snapshot=sm_snap)
    assert caps["VCP"]["status"] == "AVAILABLE"
    assert caps["VCP"]["matchedCount"] == 8
    assert caps["SMART_MONEY"]["status"] == "AVAILABLE"
    assert caps["SMART_MONEY"]["matchedCount"] == 5
