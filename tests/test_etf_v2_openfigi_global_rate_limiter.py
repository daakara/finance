"""
tests/test_etf_v2_openfigi_global_rate_limiter.py

Exhaustive offline deterministic verification of GlobalSQLiteRateLimiter (RLG-01 to RLG-18).
Enforces:
- Shared atomic rolling 60s window across multiple processes (RLG-01..04).
- Safe clock progression and regression handling (RLG-05, RLG-06, RLG-13, RLG-14).
- Retry consumption and pre-dispatch rejection isolation (RLG-07, RLG-08).
- Crash and restart recovery (RLG-09, RLG-10).
- Concurrent schema initialization and lock contention resilience (RLG-11, RLG-12).
- Zero canonical access and zero operational side effects (RLG-15..17).
- Preserved live network kill switch (RLG-18).
- Multi-process 8-worker stress test & rolling window audit (Section 20 & 21).
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

from scripts.research.etf_v2.openfigi_rate_limiter import (
    CanonicalStoreContaminationError,
    GlobalSQLiteRateLimiter,
    OpenFIGIRateLimitTimeoutError,
    DEFAULT_OPERATIONAL_DB_PATH
)
from scripts.research.etf_v2.openfigi_client import (
    OpenFIGIClient,
    LiveNetworkProhibitedError
)
from scripts.research.etf_v2.openfigi_models import (
    AuthorizedCanonicalInputRecord,
    OpenFIGIMappingJob
)
from scripts.research.etf_v2.openfigi_service import OpenFIGICorroborationService
from scripts.research.etf_v2.openfigi_persistence import OpenFIGIPersistenceRepository

pytestmark = pytest.mark.tier2c
CANONICAL_DB_PATH = Path("data/canonical/etf_v2_canonical_population.db")


# ---------------------------------------------------------------------------
# HELPERS & SUBPROCESS HARNESS
# ---------------------------------------------------------------------------
def _worker_attempt_reserve(db_path_str: str, fixed_now: float, count: int) -> Tuple[int, int]:
    """Worker process helper: attempts 'count' reservations at fixed_now."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    limiter = GlobalSQLiteRateLimiter(
        db_path=Path(db_path_str),
        clock=lambda: fixed_now
    )
    permitted = 0
    denied = 0
    for _ in range(count):
        acquired, _ = limiter.try_reserve()
        if acquired:
            permitted += 1
        else:
            denied += 1
    return permitted, denied


# ---------------------------------------------------------------------------
# TESTS RLG-01 TO RLG-18
# ---------------------------------------------------------------------------
def test_rlg01_single_process_ceiling(tmp_path: Path):
    """RLG-01: One process cannot reserve >20 dispatch attempts inside the same rolling 60s window."""
    db_path = tmp_path / "rlg01.db"
    now = 1000.0
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)

    permitted = 0
    for _ in range(25):
        acquired, wait_sec = limiter.try_reserve()
        if acquired:
            permitted += 1
        else:
            assert wait_sec > 0.0

    assert permitted == 20
    assert limiter.get_active_count() == 20


def test_rlg02_two_process_aggregate_ceiling(tmp_path: Path):
    """RLG-02: Two independent processes share authority; aggregate immediate capacity is 20, not 40."""
    db_path = tmp_path / "rlg02.db"
    fixed_now = 1000.0

    with multiprocessing.Pool(2) as pool:
        results = pool.starmap(
            _worker_attempt_reserve,
            [(str(db_path), fixed_now, 20), (str(db_path), fixed_now, 20)]
        )

    p1_perm, p1_den = results[0]
    p2_perm, p2_den = results[1]

    aggregate_permitted = p1_perm + p2_perm
    assert aggregate_permitted == 20
    assert (p1_den + p2_den) == 20

    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: fixed_now)
    assert limiter.get_active_count() == 20


def test_rlg03_four_process_aggregate_ceiling(tmp_path: Path):
    """RLG-03: Four independent processes sharing authority; aggregate immediate capacity is 20."""
    db_path = tmp_path / "rlg03.db"
    fixed_now = 2000.0

    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(
            _worker_attempt_reserve,
            [(str(db_path), fixed_now, 10) for _ in range(4)]
        )

    total_perm = sum(r[0] for r in results)
    total_den = sum(r[1] for r in results)

    assert total_perm == 20
    assert total_den == 20


def test_rlg04_contention_at_final_slot(tmp_path: Path):
    """RLG-04: Contention at 20th remaining slot; exactly one reservation succeeds immediately."""
    db_path = tmp_path / "rlg04.db"
    now = 1000.0
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)

    # Fill 19 slots
    for _ in range(19):
        acq, _ = limiter.try_reserve()
        assert acq

    # Now launch 4 contending processes for the remaining 1 slot
    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(
            _worker_attempt_reserve,
            [(str(db_path), now, 1) for _ in range(4)]
        )

    contention_wins = sum(r[0] for r in results)
    assert contention_wins == 1
    assert limiter.get_active_count() == 20


def test_rlg05_window_expiry(tmp_path: Path):
    """RLG-05: Advancing virtual time beyond 60s frees capacity correctly."""
    db_path = tmp_path / "rlg05.db"
    current_time = 1000.0

    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: current_time)

    # Fill 20 slots at t=1000
    for _ in range(20):
        acq, _ = limiter.try_reserve()
        assert acq

    assert limiter.try_reserve()[0] is False

    # Advance time by 60.1s -> all 20 slots expire
    current_time += 60.1
    acq, wait_sec = limiter.try_reserve()
    assert acq is True
    assert wait_sec == 0.0


def test_rlg06_rolling_window_boundary(tmp_path: Path):
    """RLG-06: Rolling-window boundary verification; no fixed-window reset burst allows >20."""
    db_path = tmp_path / "rlg06.db"
    current_time = 1000.0
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: current_time)

    # Reserve 10 at t=1000
    for _ in range(10):
        assert limiter.try_reserve()[0] is True

    # Advance to t=1030 (30s later), reserve 10
    current_time = 1030.0
    for _ in range(10):
        assert limiter.try_reserve()[0] is True

    # At t=1030, 20 active slots
    assert limiter.try_reserve()[0] is False

    # Advance to t=1060.1 (t=1000 batch expired, but t=1030 batch NOT expired)
    current_time = 1060.1
    # We should be able to reserve exactly 10, not 20!
    perm = 0
    for _ in range(15):
        if limiter.try_reserve()[0]:
            perm += 1
    assert perm == 10
    assert limiter.get_active_count(now=current_time) == 20


def test_rlg07_retry_consumes_capacity(tmp_path: Path):
    """RLG-07: Initial attempt plus each retry consumes a separate global reservation."""
    db_path = tmp_path / "rlg07.db"
    calls = 0

    def fake_transport(u, h, b):
        nonlocal calls
        calls += 1
        return 500, {}, b'{"error": "server error"}'

    limiter = GlobalSQLiteRateLimiter(db_path=db_path)
    client = OpenFIGIClient(
        api_key="TEST_KEY",
        transport=fake_transport,
        rate_limiter=limiter,
        sleep_func=lambda s: None
    )

    client.post_mapping_jobs([OpenFIGIMappingJob(idType="ID_ISIN", idValue="IE00B3FL3272")])

    # 1 initial attempt + 2 retries = 3 attempts total
    assert calls == 3
    # Exactly 3 reservations recorded in SQLite
    reservations = limiter.get_all_reservations()
    assert len(reservations) == 3


@pytest.fixture
def sample_canonical_record() -> AuthorizedCanonicalInputRecord:
    return AuthorizedCanonicalInputRecord(
        canonical_internal_id="etfs:v1:ISIN:IE00B3FL3272",
        isin="IE00B3FL3272",
        source_population_version="2.0.0",
        source_snapshot_sha256="938e0b00c4e7e5623b266ddddf6ad9eacf4614206b6525cabe1143db60cacaff",
        currency="USD"
    )


def test_rlg08_non_dispatched_local_rejection(tmp_path: Path, sample_canonical_record):
    """RLG-08: Locally rejected invalid request consumes no dispatch reservation."""
    db_path = tmp_path / "rlg08.db"
    repo = OpenFIGIPersistenceRepository(db_path=db_path)
    limiter = GlobalSQLiteRateLimiter(db_path=db_path)
    client = OpenFIGIClient(api_key="TEST_KEY", rate_limiter=limiter)
    service = OpenFIGICorroborationService(client=client, repository=repo)

    # Record violating OFIGI-INV-017 (both mic_code and exch_code)
    invalid_rec = AuthorizedCanonicalInputRecord(
        canonical_internal_id=sample_canonical_record.canonical_internal_id,
        isin=sample_canonical_record.isin,
        source_population_version="2.0.0",
        source_snapshot_sha256="938e0b00c4e7e5623b266ddddf6ad9eacf4614206b6525cabe1143db60cacaff",
        mic_code="XLON",
        exch_code="LN"
    )

    summary = service.run_corroboration([invalid_rec])
    assert summary.failures == 1

    # Zero reservations recorded
    assert len(limiter.get_all_reservations()) == 0


def test_rlg09_crash_after_reservation(tmp_path: Path):
    """RLG-09: Process terminates after reservation; reservation remains consumed until expiry."""
    db_path = tmp_path / "rlg09.db"
    now = 1000.0

    # Simulate sub-process reserving and then exiting
    cmd = f"""
import sys
from scripts.research.etf_v2.openfigi_rate_limiter import GlobalSQLiteRateLimiter
limiter = GlobalSQLiteRateLimiter(db_path=r'{db_path}', clock=lambda: {now})
limiter.try_reserve()
"""
    subprocess.run([sys.executable, "-c", cmd], check=True)

    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)
    assert limiter.get_active_count() == 1

    # Fill remaining 19
    for _ in range(19):
        assert limiter.try_reserve()[0] is True

    # 21st attempt blocked
    assert limiter.try_reserve()[0] is False


def test_rlg10_process_restart(tmp_path: Path):
    """RLG-10: Fresh process observes prior unexpired reservations."""
    db_path = tmp_path / "rlg10.db"
    now = 1000.0

    l1 = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)
    for _ in range(15):
        assert l1.try_reserve()[0] is True

    # New instance in new scope
    l2 = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: now)
    assert l2.get_active_count() == 15
    for _ in range(5):
        assert l2.try_reserve()[0] is True
    assert l2.try_reserve()[0] is False


def test_rlg11_concurrent_database_initialization(tmp_path: Path):
    """RLG-11: Multiple processes initialize authority concurrently without schema races."""
    db_path = tmp_path / "rlg11.db"
    now = 1000.0

    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(
            _worker_attempt_reserve,
            [(str(db_path), now, 5) for _ in range(4)]
        )

    assert sum(r[0] for r in results) == 20


def test_rlg12_lock_contention(tmp_path: Path):
    """RLG-12: Contention resolves deterministically without bypassing the limiter."""
    db_path = tmp_path / "rlg12.db"
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, timeout=0.1)

    # Intentionally lock DB externally
    conn = sqlite3.connect(str(db_path))
    conn.execute("BEGIN EXCLUSIVE")
    try:
        # try_reserve should fail fast with OperationalError or timeout, never bypass
        with pytest.raises(sqlite3.OperationalError):
            limiter.try_reserve()
    finally:
        conn.rollback()
        conn.close()


def test_rlg13_clock_advance(tmp_path: Path):
    """RLG-13: Injected deterministic time produces correct expiry behavior."""
    db_path = tmp_path / "rlg13.db"
    curr_time = 5000.0
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: curr_time)

    for _ in range(20):
        assert limiter.try_reserve()[0] is True

    # Advance 10 seconds: still full
    curr_time += 10.0
    assert limiter.try_reserve()[0] is False

    # Advance 50.1 seconds: capacity freed
    curr_time += 50.1
    assert limiter.try_reserve()[0] is True


def test_rlg14_clock_regression(tmp_path: Path):
    """RLG-14: Backward time movement cannot increase available capacity unsafely."""
    db_path = tmp_path / "rlg14.db"
    curr_time = 5000.0
    limiter = GlobalSQLiteRateLimiter(db_path=db_path, clock=lambda: curr_time)

    # Reserve 20 at t=5000
    for _ in range(20):
        assert limiter.try_reserve()[0] is True

    # Clock regresses backward to t=4950 (-50s)
    curr_time = 4950.0
    # Capacity must still be observed as full because future timestamps count!
    acq, wait_sec = limiter.try_reserve()
    assert acq is False
    assert wait_sec > 0.0


def test_rlg15_database_integrity(tmp_path: Path):
    """RLG-15: PRAGMA integrity_check returns ok after contention."""
    db_path = tmp_path / "rlg15.db"
    now = 1000.0

    with multiprocessing.Pool(4) as pool:
        pool.starmap(_worker_attempt_reserve, [(str(db_path), now, 10) for _ in range(4)])

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
    conn.close()


def test_rlg16_no_canonical_access(tmp_path: Path):
    """RLG-16: Contamination check rejects canonical database paths."""
    with pytest.raises(CanonicalStoreContaminationError):
        GlobalSQLiteRateLimiter(db_path=CANONICAL_DB_PATH)


def test_rlg17_no_production_operational_side_effect(tmp_path: Path):
    """RLG-17: Tests using isolated temporary databases do not mutate production operational DB."""
    prod_db = DEFAULT_OPERATIONAL_DB_PATH
    prod_stat_before = prod_db.stat() if prod_db.exists() else None

    isolated_db = tmp_path / "rlg17_isolated.db"
    limiter = GlobalSQLiteRateLimiter(db_path=isolated_db)
    assert limiter.try_reserve()[0] is True
    assert isolated_db.exists()

    if prod_stat_before is not None:
        prod_stat_after = prod_db.stat()
        assert prod_stat_after.st_mtime == prod_stat_before.st_mtime
        assert prod_stat_after.st_size == prod_stat_before.st_size
    else:
        assert not prod_db.exists()


def test_rlg18_live_network_kill_switch_preserved(tmp_path: Path):
    """RLG-18: Live network kill switch raises LiveNetworkProhibitedError."""
    db_path = tmp_path / "rlg18.db"
    limiter = GlobalSQLiteRateLimiter(db_path=db_path)
    client = OpenFIGIClient(api_key="TEST_KEY", transport=None, rate_limiter=limiter)
    with pytest.raises(LiveNetworkProhibitedError):
        client.post_mapping_jobs([OpenFIGIMappingJob(idType="ID_ISIN", idValue="IE00B3FL3272")])


# ---------------------------------------------------------------------------
# STRESS TEST & ROLLING-WINDOW AUDIT (SECTIONS 20 & 21)
# ---------------------------------------------------------------------------
def test_section20_and_21_multiprocess_stress_and_rolling_audit(tmp_path: Path):
    """
    Sections 20 & 21:
    - 8 processes, 25 attempts each = 200 total synthetic attempts at same logical time.
    - Exactly 20 immediately reserved, 180 immediately blocked/delayed.
    - Verified across STRESS_RUNS = 10.
    - Rolling window audit: max reservations in (T - 60s, T] <= 20 for all T.
    """
    db_path = tmp_path / "stress_test.db"
    logical_time = 10000.0

    for run_idx in range(10):
        # Reset DB for each run
        if db_path.exists():
            db_path.unlink()

        with multiprocessing.Pool(8) as pool:
            results = pool.starmap(
                _worker_attempt_reserve,
                [(str(db_path), logical_time + run_idx * 100.0, 25) for _ in range(8)]
            )

        total_perm = sum(r[0] for r in results)
        total_den = sum(r[1] for r in results)

        assert total_perm == 20, f"Run {run_idx}: Expected exactly 20 permitted, got {total_perm}"
        assert total_den == 180, f"Run {run_idx}: Expected exactly 180 denied, got {total_den}"

        # Audit rolling window
        limiter = GlobalSQLiteRateLimiter(db_path=db_path)
        reservations = limiter.get_all_reservations()
        assert len(reservations) == 20

        # Programmatically inspect rolling 60-second window invariant for all timestamps T
        timestamps = [r[3] for r in reservations]
        for t in timestamps:
            count_in_window = sum(1 for ts in timestamps if (t - 60.0) < ts <= t)
            assert count_in_window <= 20
