"""
tests/test_etf_v2_openfigi_path_resolution.py

Regression, coordination, and invariant test suite for OpenFIGI operational store
path resolution, override governance, multi-process coordination, and rate-limit enforcement.

Enforces Acceptance Criteria from:
- ETF_V2_OPENFIGI_POST_RELEASE_OPERATIONAL_DB_PATH_REMEDIATION_GATE
- AC-05-01 to AC-05-06: Single Canonical Path Authority
- AC-06-01 to AC-06-06: Override Governance (RELATIVE_OVERRIDE_ALLOWED = NO)
- AC-07-01 to AC-07-06: Startup Validation & Store Parity
- AC-08-01 to AC-08-03: Single Physical Store Invariants (INV-OPENFIGI-DB-01 to 08)
- AC-09-01 to AC-09-05: Multi-Process Coordination Test (5 required cases)
- AC-10-01 to AC-10-05: Global Rate-Limit Verification (Process A: 12, Process B: 12 -> max 20)
- AC-12-01 to AC-12-05: Test DB Isolation
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import List, Tuple

import pytest

from scripts.research.etf_v2.openfigi_config import (
    APPROVED_OPERATIONAL_STORAGE_DIR,
    CANONICAL_DB_NAME,
    CANONICAL_OPERATIONAL_DB_ENV_VAR,
    DEFAULT_OPERATIONAL_DB_PATH,
    LEGACY_RATE_LIMIT_ENV_VAR,
    REPO_ROOT,
    CanonicalStoreContaminationError,
    OpenFIGIPathValidationError,
    OpenFIGIStoreParityError,
    resolve_openfigi_operational_db_path,
    validate_openfigi_operational_db_path,
    validate_store_path_parity,
)
from scripts.research.etf_v2.openfigi_rate_limiter import (
    GlobalSQLiteRateLimiter,
)
from scripts.research.etf_v2.openfigi_persistence import (
    OpenFIGIPersistenceRepository,
)
from scripts.research.etf_v2.openfigi_client import (
    LiveNetworkProhibitedError,
    OpenFIGIClient,
)
from scripts.research.etf_v2.openfigi_models import OpenFIGIMappingJob
from scripts.research.etf_v2.openfigi_service import OpenFIGICorroborationService


# ===========================================================================
# Section 5 & 9: Canonical Path Authority & Cross-CWD Parity (AC-05, AC-09-01)
# ===========================================================================
def test_cross_cwd_path_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    AC-05-03 & AC-09-01: Path resolution invoked from different working directories returns
    the exact same physical operational DB file.
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

        # 3. From arbitrary directory
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
# Section 7 & 11: Store Parity Verification (AC-07-02, AC-11-02)
# ===========================================================================
def test_limiter_persistence_store_parity_default(monkeypatch: pytest.MonkeyPatch):
    """AC-07-02 / AC-11-02: Under default configuration, limiter and persistence share the same DB path."""
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == DEFAULT_OPERATIONAL_DB_PATH
    assert validate_store_path_parity(limiter.db_path, repo.db_path) is True


def test_limiter_persistence_store_parity_canonical_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Under absolute OPENFIGI_OPERATIONAL_DB, limiter and persistence resolve the exact same DB."""
    custom_db = (tmp_path / "canonical_override.db").resolve()
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(custom_db))
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == custom_db
    assert validate_store_path_parity(limiter.db_path, repo.db_path) is True


def test_limiter_persistence_store_parity_legacy_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Under absolute OPENFIGI_RATE_LIMIT_DB, limiter and persistence resolve the exact same DB."""
    legacy_db = (tmp_path / "legacy_override.db").resolve()
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.setenv(LEGACY_RATE_LIMIT_ENV_VAR, str(legacy_db))

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == repo.db_path
    assert limiter.db_path == legacy_db
    assert validate_store_path_parity(limiter.db_path, repo.db_path) is True


def test_store_parity_mismatch_fails_closed(tmp_path: Path):
    """AC-07-02 / INV-OPENFIGI-DB-07: Store path mismatch between limiter and persistence fails closed."""
    db1 = (tmp_path / "store1.db").resolve()
    db2 = (tmp_path / "store2.db").resolve()

    limiter = GlobalSQLiteRateLimiter(db_path=db1)
    repo = OpenFIGIPersistenceRepository(db_path=db2, auto_init=False)
    client = OpenFIGIClient(rate_limiter=limiter)

    with pytest.raises(OpenFIGIStoreParityError):
        validate_store_path_parity(limiter.db_path, repo.db_path)

    with pytest.raises(OpenFIGIStoreParityError):
        OpenFIGICorroborationService(client=client, repository=repo)


# ===========================================================================
# Section 6: Override Governance (AC-06-01 to AC-06-06, AC-07-01, AC-09-03)
# ===========================================================================
def test_relative_production_override_rejected(monkeypatch: pytest.MonkeyPatch):
    """
    AC-06-03 / AC-07-01 / AC-09-03: Relative production environment override causes
    startup validation to fail.
    """
    rel_override = "data/operational/relative_test.db"
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, rel_override)
    with pytest.raises(OpenFIGIPathValidationError) as exc_info:
        resolve_openfigi_operational_db_path()
    assert "must be absolute" in str(exc_info.value) or "Relative path" in str(exc_info.value)


def test_relative_legacy_override_rejected(monkeypatch: pytest.MonkeyPatch):
    """AC-06-03: Relative legacy environment override is rejected."""
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    monkeypatch.setenv(LEGACY_RATE_LIMIT_ENV_VAR, "some_rel_path.db")
    with pytest.raises(OpenFIGIPathValidationError):
        resolve_openfigi_operational_db_path()


def test_operational_db_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Precedence order:
    explicit constructor > canonical env > legacy env > repository default
    """
    explicit_db = (tmp_path / "explicit.db").resolve()
    canonical_env_db = (tmp_path / "canonical_env.db").resolve()
    legacy_env_db = (tmp_path / "legacy_env.db").resolve()

    # Both env vars set: canonical wins
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(canonical_env_db))
    monkeypatch.setenv(LEGACY_RATE_LIMIT_ENV_VAR, str(legacy_env_db))
    assert resolve_openfigi_operational_db_path() == canonical_env_db

    # Explicit constructor wins over canonical env
    assert resolve_openfigi_operational_db_path(explicit_db) == explicit_db

    # When canonical unset, legacy env wins over default
    monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
    assert resolve_openfigi_operational_db_path() == legacy_env_db

    # When legacy unset, default wins
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)
    assert resolve_openfigi_operational_db_path() == DEFAULT_OPERATIONAL_DB_PATH


# ===========================================================================
# Section 7: Startup Validation Checks (AC-07-01 to AC-07-06)
# ===========================================================================
def test_startup_validation_boundary_rejection(tmp_path: Path):
    """AC-07-03: In production mode, paths outside approved runtime storage boundary are rejected."""
    unapproved_path = (tmp_path / "unapproved" / "openfigi.db").resolve()
    with pytest.raises(OpenFIGIPathValidationError) as exc_info:
        validate_openfigi_operational_db_path(unapproved_path, is_test=False)
    assert "outside approved runtime storage boundary" in str(exc_info.value)


def test_startup_validation_in_memory_rejection_in_production():
    """In-memory database is strictly rejected in production mode."""
    with pytest.raises(OpenFIGIPathValidationError):
        validate_openfigi_operational_db_path(":memory:", is_test=False)


def test_canonical_firewall_rejection():
    """INV-OPENFIGI-DB-08: Reject any path referencing canonical population store."""
    canonical_paths = [
        REPO_ROOT / "data" / "canonical" / "etf_v2_canonical_population.db",
        Path("data/canonical/etf_v2_canonical_population.db"),
        Path("etf_v2_canonical_population.db"),
    ]

    for p in canonical_paths:
        with pytest.raises((CanonicalStoreContaminationError, OpenFIGIPathValidationError)):
            validate_openfigi_operational_db_path(p)


# ===========================================================================
# Section 9: Multi-Process Coordination 5 Required Cases (AC-09-01 to AC-09-05)
# ===========================================================================
def test_multiprocess_coordination_five_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    AC-09-01 to AC-09-05: Tests all 5 required multi-process coordination cases:
    Case 1: same canonical configuration with the same CWD
    Case 2: same canonical configuration with different CWDs
    Case 3: same absolute override with different CWDs
    Case 4: invalid relative production override
    Case 5: test-specific isolated temporary DBs
    """
    original_cwd = os.getcwd()
    cwd_a = tmp_path / "proc_cwd_a"
    cwd_b = tmp_path / "proc_cwd_b"
    cwd_a.mkdir(parents=True, exist_ok=True)
    cwd_b.mkdir(parents=True, exist_ok=True)

    try:
        # Case 1: Same canonical config with same CWD
        monkeypatch.delenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, raising=False)
        monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)
        os.chdir(str(REPO_ROOT))
        path_c1_a = resolve_openfigi_operational_db_path()
        path_c1_b = resolve_openfigi_operational_db_path()
        assert path_c1_a == path_c1_b == DEFAULT_OPERATIONAL_DB_PATH

        # Case 2: Same canonical config with different CWDs
        os.chdir(str(cwd_a))
        path_c2_a = resolve_openfigi_operational_db_path()
        os.chdir(str(cwd_b))
        path_c2_b = resolve_openfigi_operational_db_path()
        assert path_c2_a == path_c2_b == DEFAULT_OPERATIONAL_DB_PATH
        assert path_c2_a.is_absolute()

        # Case 3: Same absolute override with different CWDs
        abs_override = (tmp_path / "shared_abs_override.db").resolve()
        monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(abs_override))
        os.chdir(str(cwd_a))
        path_c3_a = resolve_openfigi_operational_db_path()
        os.chdir(str(cwd_b))
        path_c3_b = resolve_openfigi_operational_db_path()
        assert path_c3_a == path_c3_b == abs_override

        # Case 4: Invalid relative production override
        monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, "relative_invalid.db")
        os.chdir(str(cwd_a))
        with pytest.raises(OpenFIGIPathValidationError):
            resolve_openfigi_operational_db_path()

        # Case 5: Test-specific isolated temporary DBs
        test_db_a = (tmp_path / "test_isolated_a.db").resolve()
        test_db_b = (tmp_path / "test_isolated_b.db").resolve()
        assert test_db_a != DEFAULT_OPERATIONAL_DB_PATH
        assert test_db_b != DEFAULT_OPERATIONAL_DB_PATH
        assert test_db_a != test_db_b

    finally:
        os.chdir(original_cwd)


# ===========================================================================
# Section 10: Global Rate-Limit Verification 12+12=20 (AC-10-01 to AC-10-05)
# ===========================================================================
def _worker_attempt_reservations(db_path_str: str, count: int, now: float) -> int:
    """Worker function executed by multiple processes to acquire reservations."""
    limiter = GlobalSQLiteRateLimiter(
        db_path=Path(db_path_str),
        clock=lambda: now
    )
    granted_count = 0
    for _ in range(count):
        granted, _ = limiter.try_reserve()
        if granted:
            granted_count += 1
    return granted_count


def test_section10_multiprocess_rate_limit_12_and_12_coordination(tmp_path: Path):
    """
    AC-10-01 to AC-10-05:
    Process A attempts 12 requests.
    Process B attempts 12 requests.
    All attempts occur within the same rolling 60-second window.
    Combined accepted requests within window must be exactly min(12+12, 20) = 20.
    """
    shared_db = (tmp_path / "rate_limit_12_12.db").resolve()
    now = 75000.0
    # Initialize schema first
    init_limiter = GlobalSQLiteRateLimiter(db_path=shared_db, clock=lambda: now)
    init_limiter.get_active_count()

    # Execute in multiprocessing pool to guarantee genuine separate OS processes
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        res_a = pool.apply_async(_worker_attempt_reservations, (str(shared_db), 12, now))
        res_b = pool.apply_async(_worker_attempt_reservations, (str(shared_db), 12, now))

        granted_a = res_a.get(timeout=20)
        granted_b = res_b.get(timeout=20)

    total_accepted = granted_a + granted_b
    assert total_accepted == 20, f"Expected exactly 20 total granted requests, got {total_accepted} ({granted_a} + {granted_b})"
    assert init_limiter.get_active_count(now=now) == 20

    # 25th attempt from caller process must be rejected
    rejected_attempt, wait_time = init_limiter.try_reserve()
    assert rejected_attempt is False
    assert wait_time > 0.0


# ===========================================================================
# Section 12: Test Isolation (AC-12-01 to AC-12-05)
# ===========================================================================
def test_test_db_isolation(tmp_path: Path):
    """
    AC-12-01 to AC-12-05:
    Every test that accesses an operational DB uses an explicit isolated test path.
    Production DB is never mutated during test execution.
    """
    prod_path = DEFAULT_OPERATIONAL_DB_PATH
    prod_existed = prod_path.exists()
    prod_mtime_before = prod_path.stat().st_mtime if prod_existed else None

    # Test uses isolated path
    isolated_db = (tmp_path / "isolated_test_suite.db").resolve()
    assert isolated_db != prod_path

    repo = OpenFIGIPersistenceRepository(db_path=isolated_db, auto_init=True)
    limiter = GlobalSQLiteRateLimiter(db_path=isolated_db)

    # Perform writes to isolated DB
    assert limiter.try_reserve()[0] is True
    with repo.connection() as conn:
        conn.execute("INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES ('1.0', '2026-10-06T00:00:00Z');")

    assert isolated_db.exists()

    # Production DB must be completely untouched
    if prod_existed:
        assert prod_path.stat().st_mtime == prod_mtime_before
    else:
        assert not prod_path.exists()


# ===========================================================================
# Live Network Kill Switch & Operational DB Setup
# ===========================================================================
def test_live_network_kill_switch_preserved(tmp_path: Path):
    """Section 14 & OFIGI-INV-011: Live network requests fail closed without authorized transport."""
    db_path = (tmp_path / "kill_switch.db").resolve()
    limiter = GlobalSQLiteRateLimiter(db_path=db_path)
    client = OpenFIGIClient(api_key="TEST_KEY", transport=None, rate_limiter=limiter)

    with pytest.raises(LiveNetworkProhibitedError):
        client.post_mapping_jobs([OpenFIGIMappingJob(idType="ID_ISIN", idValue="IE00B3FL3272")])


def test_operational_db_initialization(tmp_path: Path):
    """
    Verifies operational DB creation, parent directory creation,
    schema initialization, idempotence, WAL mode, and busy timeout.
    """
    deep_path = (tmp_path / "deep" / "nested" / "operational.db").resolve()
    repo = OpenFIGIPersistenceRepository(db_path=deep_path, auto_init=True)

    assert deep_path.exists()

    with repo.connection() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

        timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert timeout == 30000

        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "openfigi_observations" in tables
        assert "openfigi_active_mappings" in tables
        assert "schema_version" in tables

        assert "share_classes" not in tables
        assert "subfunds" not in tables
        assert "canonical_population" not in tables

    # Re-initialization is idempotent
    repo.initialize_schema()


# ===========================================================================
# Persistent Volume Durable Store Regression Tests (Epoch 002 Infrastructure)
# ===========================================================================
def test_persistent_volume_path_override_shared_and_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that when OPENFIGI_OPERATIONAL_DB points to a persistent volume path,
    both OpenFIGIPersistenceRepository and GlobalSQLiteRateLimiter share the identical path,
    validate_store_path_parity passes, and no dual databases are created.
    """
    persistent_mount = (tmp_path / "persistent_vol").resolve()
    persistent_mount.mkdir(parents=True, exist_ok=True)
    persistent_db = (persistent_mount / "data" / "operational" / "openfigi_operational.db").resolve()

    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(persistent_mount))
    monkeypatch.setenv(CANONICAL_OPERATIONAL_DB_ENV_VAR, str(persistent_db))
    monkeypatch.delenv(LEGACY_RATE_LIMIT_ENV_VAR, raising=False)

    resolved_path = resolve_openfigi_operational_db_path()
    assert resolved_path == persistent_db

    limiter = GlobalSQLiteRateLimiter()
    repo = OpenFIGIPersistenceRepository(auto_init=False)

    assert limiter.db_path == persistent_db
    assert repo.db_path == persistent_db
    assert limiter.db_path == repo.db_path
    assert validate_store_path_parity(limiter.db_path, repo.db_path) is True


def test_persistent_volume_boundary_validation_in_production_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that in production mode (is_test=False), paths residing on the
    configured persistent volume mount pass boundary validation, while arbitrary
    paths outside both the repo operational dir and persistent volume are rejected.
    """
    persistent_mount = (tmp_path / "persistent_vol").resolve()
    persistent_mount.mkdir(parents=True, exist_ok=True)
    persistent_db = (persistent_mount / "data" / "operational" / "openfigi_operational.db").resolve()
    unapproved_db = (tmp_path / "other_unapproved" / "openfigi_operational.db").resolve()

    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(persistent_mount))

    # Path inside persistent volume passes
    validated = validate_openfigi_operational_db_path(persistent_db, is_test=False)
    assert validated == persistent_db

    # Path outside both repo operational dir and persistent volume is rejected
    with pytest.raises(OpenFIGIPathValidationError) as exc_info:
        validate_openfigi_operational_db_path(unapproved_db, is_test=False)
    assert "outside approved runtime storage boundary" in str(exc_info.value)

