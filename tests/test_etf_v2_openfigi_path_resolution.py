"""
tests/test_etf_v2_openfigi_path_resolution.py

Regression and invariant test suite for OpenFIGI operational store path resolution,
cross-CWD coordination, limiter/persistence parity, and multi-process shared capacity.

Covered Invariants & Sections:
- Section 21: Cross-CWD Coordination (repo root, script dir, arbitrary cwd).
- Section 22: Shared Limiter / Persistence store parity under defaults and overrides.
- Section 23: Cross-Process rate limiter capacity ceiling (capacity=20, 21st blocked, recovery).
- Section 24: Multi-CWD rate limiter coordination sharing exactly 20 capacity (not 40).
- Section 25: Retry reservation accounting.
- Section 29: Live network fail-closed kill switch.
- Section 30: Operational DB initialization (WAL mode, busy timeout, idempotency).
- Section 31: Canonical firewall protection.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import List

import pytest

from scripts.research.etf_v2.openfigi_config import (
    CANONICAL_OPERATIONAL_DB_ENV_VAR,
    DEFAULT_OPERATIONAL_DB_PATH,
    LEGACY_RATE_LIMIT_ENV_VAR,
    REPO_ROOT,
    resolve_openfigi_operational_db_path,
)
from scripts.research.etf_v2.openfigi_rate_limiter import (
    CanonicalStoreContaminationError as LimiterCanonicalStoreContaminationError,
    GlobalSQLiteRateLimiter,
)
from scripts.research.etf_v2.openfigi_persistence import (
    CanonicalStoreContaminationError as PersistenceCanonicalStoreContaminationError,
    OpenFIGIPersistenceRepository,
)
from scripts.research.etf_v2.openfigi_client import (
    LiveNetworkProhibitedError,
    OpenFIGIClient,
)
from scripts.research.etf_v2.openfigi_models import OpenFIGIMappingJob


# ===========================================================================
# Section 21: Cross-CWD Coordination Test
# ===========================================================================
def test_cross_cwd_path_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Section 21: Path resolution invoked from different working directories returns
    the exact same operational DB path.
    """
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    original_cwd = os.getcwd()
    script_dir = REPO_ROOT / "scripts" / "research" / "etf_v2"
    arbitrary_dir = tmp_path / "arbitrary_cwd"
    arbitrary_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. From repository root
        os.chdir(str(REPO_ROOT))
        path_from_root = resolve_openfigi_operational_db_path()

        # 2. From script directory
        os.chdir(str(script_dir))
        path_from_script = resolve_openfigi_operational_db_path()

        # 3. From arbitrary temporary directory
        os.chdir(str(arbitrary_dir))
        path_from_arbitrary = resolve_openfigi_operational_db_path()

        assert path_from_root == DEFAULT_OPERATIONAL_DB_PATH
        assert path_from_script == DEFAULT_OPERATIONAL_DB_PATH
        assert path_from_arbitrary == DEFAULT_OPERATIONAL_DB_PATH

        assert path_from_root == path_from_script == path_from_arbitrary
        assert path_from_root.is_absolute()
    finally:
        os.chdir(original_cwd)


# ===========================================================================
# Section 22: Shared Limiter / Persistence Parity Test
# ===========================================================================
def test_limiter_persistence_store_parity_default(monkeypatch: pytest.MonkeyPatch):
    """Section 22: Under default configuration, limiter and persistence share the same DB path."""
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == DEFAULT_OPERATIONAL_DB_PATH


def test_limiter_persistence_store_parity_canonical_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Section 22: Under OPENFIGI_OPERATIONAL_DB, limiter and persistence resolve the exact same DB."""
    custom_db = tmp_path / "canonical_override.db"
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(custom_db))
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == custom_db.resolve()


def test_limiter_persistence_store_parity_legacy_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Section 22: Under OPENFIGI_RATE_LIMIT_DB, limiter and persistence resolve the exact same DB."""
    legacy_db = tmp_path / "legacy_override.db"
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.setenv(LEGACY_RATE_LIMIT_ENV_VAR, str(legacy_db))

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == legacy_db.resolve()


def test_limiter_persistence_store_parity_relative_env(monkeypatch: pytest.MonkeyPatch):
    """Section 22: Relative environment override is anchored to repository root across both."""
    rel_override = "data/operational/relative_test.db"
    expected = (REPO_ROOT / rel_override).resolve()

    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, rel_override)
    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == expected


def test_operational_db_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Section 13: Precedence order:
    explicit constructor > canonical env > legacy env > repository default
    """
    explicit_db = tmp_path / "explicit.db"
    canonical_env_db = tmp_path / "canonical_env.db"
    legacy_env_db = tmp_path / "legacy_env.db"

    # Both env vars set: canonical wins
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(canonical_env_db))
    monkeypatch.setenv(LEGACY_RATE_LIMIT_ENV_VAR, str(legacy_env_db))
    assert resolve_openfigi_operational_db_path() == canonical_env_db.resolve()

    # Explicit constructor wins over canonical env
    assert resolve_openfigi_operational_db_path(explicit_db) == explicit_db.resolve()

    # When canonical unset, legacy env wins over default
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    assert resolve_openfigi_operational_db_path() == legacy_env_db.resolve()

    # When legacy unset, default wins
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)
    assert resolve_openfigi_operational_db_path() == DEFAULT_OPERATIONAL_DB_PATH


# ===========================================================================
# Section 23: Cross-Process Rate-Limiter Test
# ===========================================================================
def test_cross_process_rate_limiter_capacity(tmp_path: Path):
    """
    Section 23: Multiple limiter instances/processes share the same reservation ceiling (20).
    Request 21 is blocked. After expiry, capacity recovers.
    """
    db_path = tmp_path / "shared_capacity.db"
    now = 10000.0

    # Instance A acquires 20 reservations
    limiter_a = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)
    for _ in range(20):
        granted, wait_sec = limiter_a.try_reserve()
        assert granted is True
        assert wait_sec == 0.0

    # Instance B attempts 21st reservation: blocked
    limiter_b = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)
    granted_21, wait_21 = limiter_b.try_reserve()
    assert granted_21 is False
    assert wait_21 > 0.0

    # Advance clock past 60-second window
    now_advanced = now + 60.1
    limiter_b_advanced = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now_advanced)
    granted_recovered, wait_recovered = limiter_b_advanced.try_reserve()
    assert granted_recovered is True
    assert wait_recovered == 0.0


# ===========================================================================
# Section 24: Multi-CWD Rate-Limiter Test
# ===========================================================================
def test_multi_cwd_shared_limiter_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Section 24: Two processes/instances operating from different working directories
    resolve the same operational DB and share the single 20-request window (not 40).
    """
    shared_db = tmp_path / "multi_cwd_test.db"
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(shared_db))

    original_cwd = os.getcwd()
    dir_1 = tmp_path / "proc_dir_1"
    dir_2 = tmp_path / "proc_dir_2"
    dir_1.mkdir(parents=True, exist_ok=True)
    dir_2.mkdir(parents=True, exist_ok=True)

    now = 50000.0
    try:
        # Process/context 1 in dir_1 reserves 12 slots
        os.chdir(str(dir_1))
        limiter_1 = GlobalSQLiteRateLimiter(clock=lambda: now)
        for _ in range(12):
            assert limiter_1.try_reserve()[0] is True

        # Process/context 2 in dir_2 reserves 8 slots
        os.chdir(str(dir_2))
        limiter_2 = GlobalSQLiteRateLimiter(clock=lambda: now)
        for _ in range(8):
            assert limiter_2.try_reserve()[0] is True

        # Exactly 20 reserved total: 21st attempt from dir_2 must fail
        assert limiter_2.try_reserve()[0] is False

        # Attempt from dir_1 must also fail
        os.chdir(str(dir_1))
        limiter_1_again = GlobalSQLiteRateLimiter(clock=lambda: now)
        assert limiter_1_again.try_reserve()[0] is False

        # Verify combined capacity was strictly 20, not 40
        assert limiter_1_again.get_active_count() == 20
    finally:
        os.chdir(original_cwd)


# ===========================================================================
# Section 25: Retry Reservation Preservation
# ===========================================================================
def test_retry_reservation_preservation(tmp_path: Path):
    """Section 25: Initial attempt and every retry consume a limiter reservation."""
    db_path = tmp_path / "retry_test.db"
    limiter = GlobalSQLiteRateLimiter(db_path=db_path)

    # Initial attempt
    acq1, _ = limiter.try_reserve()
    assert acq1 is True
    assert limiter.get_active_count() == 1

    # Retry 1
    acq2, _ = limiter.try_reserve()
    assert acq2 is True
    assert limiter.get_active_count() == 2

    # Retry 2
    acq3, _ = limiter.try_reserve()
    assert acq3 is True
    assert limiter.get_active_count() == 3


# ===========================================================================
# Section 29: Live Network Kill Switch
# ===========================================================================
def test_live_network_kill_switch_preserved(tmp_path: Path):
    """Section 29: Live network requests fail closed without authorized transport."""
    db_path = tmp_path / "kill_switch.db"
    limiter = GlobalSQLiteRateLimiter(db_path=db_path)
    client = OpenFIGIClient(api_key="TEST_KEY", transport=None, rate_limiter=limiter)

    with pytest.raises(LiveNetworkProhibitedError):
        client.post_mapping_jobs([OpenFIGIMappingJob(idType="ID_ISIN", idValue="IE00B3FL3272")])


# ===========================================================================
# Section 30: Operational DB Initialization
# ===========================================================================
def test_operational_db_initialization(tmp_path: Path):
    """
    Section 30: Verifies operational DB creation, parent directory creation,
    schema initialization, idempotence, WAL mode, and busy timeout.
    """
    deep_path = tmp_path / "deep" / "nested" / "operational.db"
    repo = OpenFIGIPersistenceRepository(db_path=deep_path, auto_init=True)

    assert deep_path.exists()

    with repo.connection() as conn:
        # Check WAL mode
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

        # Check busy timeout (30,000 ms)
        timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert timeout == 30000

        # Check operational tables exist
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "openfigi_observations" in tables
        assert "openfigi_active_mappings" in tables
        assert "schema_version" in tables

        # Verify canonical tables remain strictly absent
        assert "share_classes" not in tables
        assert "subfunds" not in tables
        assert "canonical_population" not in tables

    # Re-initialization is idempotent
    repo.initialize_schema()


# ===========================================================================
# Section 31: Canonical Firewall Test
# ===========================================================================
def test_canonical_firewall_rejection():
    """Section 31: Reject any path referencing canonical population store."""
    canonical_paths = [
        Path("data/canonical/etf_v2_canonical_population.db"),
        Path("etf_v2_canonical_population.db"),
    ]

    for p in canonical_paths:
        with pytest.raises(LimiterCanonicalStoreContaminationError):
            GlobalSQLiteRateLimiter(db_path=p)

        with pytest.raises(PersistenceCanonicalStoreContaminationError):
            OpenFIGIPersistenceRepository(db_path=p)
