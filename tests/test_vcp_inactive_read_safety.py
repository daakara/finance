"""
tests/test_vcp_inactive_read_safety.py

Durable Regression Test Suite for VCP Inactive-Read Safety Contract.
Gate 4B-R: Proving public snapshot and read routes cannot initiate market scans,
acquire execution leases, or perform writes during cache misses, storage failures,
or release mismatches.

Tests Route-to-Runner execution path:
- GET /api/v1/screener/vcp/snapshot
- GET /api/v1/screener/run

Required Invariants Covered:
1. Existing usable snapshot: returns 200 OK, AVAILABLE, zero scans, zero leases, zero writes.
2. Empty storage/cache miss: returns 200 OK, PIPELINE_PENDING, zero scans, zero leases, zero writes.
3. Snapshot release mismatch: returns 200 OK, PIPELINE_PENDING, zero scans, zero leases, zero writes.
4. Stale snapshot: returns 200 OK, STALE status / freshness, zero scans, zero leases, zero writes.
5. Storage failure: returns 200 OK, ERROR status envelope, zero scans, zero leases, zero writes.
6. Repeated concurrent/sequential reads: zero scans, zero leases, zero writes across all iterations.
"""

import os
import sqlite3
import tempfile
import time
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.analyzers.scanner_contract import (
    ScannerStatus,
    FreshnessStatus,
    CANONICAL_VCP_UNIVERSE,
    RADAR_SCOPE_LABEL,
    VCP_API_CONTRACT_VERSION,
    VCP_RULESET_VERSION,
    VCP_EVIDENCE_SCHEMA_VERSION,
    VCP_SCORE_MODEL_VERSION,
    VCP_DATA_PROVENANCE_VERSION,
    VCP_UNIVERSE_VERSION,
    VCP_FRESHNESS_POLICY_VERSION,
)
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
import api.routes.screener as screener_module


client = TestClient(app)


@pytest.fixture
def isolated_vcp_environment(monkeypatch):
    """
    Provisions an isolated SQLite snapshot store and mounts it onto screener_module.vcp_scanner_runner.
    Instruments spies across execution, coordination, and publication paths to guarantee 0 side effects.
    """
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_vcp_scanner_store.db")
    store = ScannerSnapshotStore(db_path=db_path)

    # Initialize a clean runner with isolated store
    runner = VCPScannerRunner(snapshot_store=store)
    runner._cached_active_snapshot = None

    # Patch router-level runner
    monkeypatch.setattr(screener_module, "vcp_scanner_runner", runner)

    # Instrument spies on protected execution paths
    scan_spy = MagicMock(wraps=runner.execute_market_wide_scan)
    acquire_spy = MagicMock(wraps=runner.coordinator.acquire)
    release_spy = MagicMock(wraps=runner.coordinator.release)
    publish_spy = MagicMock(wraps=runner.fenced_publisher.publish_snapshot_fenced)
    save_spy = MagicMock(wraps=runner.snapshot_store.save_snapshot)

    monkeypatch.setattr(runner, "execute_market_wide_scan", scan_spy)
    monkeypatch.setattr(runner.coordinator, "acquire", acquire_spy)
    monkeypatch.setattr(runner.coordinator, "release", release_spy)
    monkeypatch.setattr(runner.fenced_publisher, "publish_snapshot_fenced", publish_spy)
    monkeypatch.setattr(runner.snapshot_store, "save_snapshot", save_spy)

    yield {
        "client": client,
        "runner": runner,
        "store": store,
        "db_path": db_path,
        "scan_spy": scan_spy,
        "acquire_spy": acquire_spy,
        "release_spy": release_spy,
        "publish_spy": publish_spy,
        "save_spy": save_spy,
    }

    # Cleanup temporary SQLite DB
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass


def _seed_snapshot(
    store: ScannerSnapshotStore,
    snapshot_id: str = "vcp-snap-test-usable-001",
    release_sha: str = CURRENT_IMPLEMENTATION_RELEASE_SHA,
    decision: str = "PUBLISH",
    status_at_publication: str = "AVAILABLE",
    freshness_status: str = "LIVE",
    matched_count: int = 12,
) -> Dict[str, Any]:
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = {
        "snapshot_id": snapshot_id,
        "scanner_id": "MINERVINI_VCP",
        "run_id": f"run-{snapshot_id}",
        "api_contract_version": VCP_API_CONTRACT_VERSION,
        "ruleset_version": VCP_RULESET_VERSION,
        "evidence_schema_version": VCP_EVIDENCE_SCHEMA_VERSION,
        "score_model_version": VCP_SCORE_MODEL_VERSION,
        "data_provenance_version": VCP_DATA_PROVENANCE_VERSION,
        "universe_version": VCP_UNIVERSE_VERSION,
        "freshness_policy_version": VCP_FRESHNESS_POLICY_VERSION,
        "implementation_release_sha": release_sha,
        "semantic_fingerprint": f"fp-{snapshot_id}",
        "generated_at": now_iso,
        "data_as_of": "2026-10-15",
        "status_at_publication": status_at_publication,
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": len(CANONICAL_VCP_UNIVERSE),
        "matched_count": matched_count,
        "results": [{"symbol": f"TST{i}", "current_price": 100.0 + i} for i in range(matched_count)],
        "provenance": {"role": "PRICE_VOLUME_HISTORY"},
        "freshness": {"status": freshness_status, "data_as_of": "2026-10-15", "generated_at": now_iso},
        "publication_decision": decision,
        "universe_metadata": {"scope_class": "US_EQUITIES", "display_name": RADAR_SCOPE_LABEL},
        "coverage_metadata": {"coverage_status": "COMPLETE"},
    }
    store.save_snapshot(rec)
    return rec


# ==============================================================================
# Case 1: Existing Usable Snapshot
# ==============================================================================

def test_01_existing_usable_snapshot(isolated_vcp_environment):
    """
    Valid usable snapshot: Returns permitted snapshot without scanning or lease acquisition.
    """
    env = isolated_vcp_environment
    client = env["client"]
    _seed_snapshot(env["store"], snapshot_id="vcp-snap-case-1", matched_count=15)
    env["save_spy"].reset_mock()

    # 1. GET /api/v1/screener/vcp/snapshot
    resp_snap = client.get("/api/v1/screener/vcp/snapshot")
    assert resp_snap.status_code == 200
    snap_data = resp_snap.json()
    assert snap_data["status"] == "AVAILABLE"
    assert snap_data["snapshot"]["snapshot_id"] == "vcp-snap-case-1"
    assert snap_data["snapshot"]["matched_count"] == 15
    assert len(snap_data["results"]) == 15
    assert snap_data["provenance"]["implementation_release_sha"] == CURRENT_IMPLEMENTATION_RELEASE_SHA
    assert snap_data["freshness"]["status"] == "LIVE"

    # 2. GET /api/v1/screener/run
    resp_run = client.get("/api/v1/screener/run")
    assert resp_run.status_code == 200
    run_data = resp_run.json()
    caps = run_data["capabilities"]
    assert caps["VCP"]["status"] == "AVAILABLE"
    assert caps["VCP"]["universeScreening"] == "AVAILABLE"
    assert caps["VCP"]["matchedCount"] == 15
    assert caps["VCP"]["snapshotId"] == "vcp-snap-case-1"

    # Spies: Zero scans, zero leases, zero publication calls, zero writes
    assert env["scan_spy"].call_count == 0, "Scan must not be called on existing usable snapshot"
    assert env["acquire_spy"].call_count == 0, "Execution lease must not be acquired on public read"
    assert env["release_spy"].call_count == 0, "Execution lease must not be released"
    assert env["publish_spy"].call_count == 0, "Fenced publication must not be invoked"
    assert env["save_spy"].call_count == 0, "Snapshot store writes must not occur on read"


# ==============================================================================
# Case 2: Empty Storage / Cache Miss
# ==============================================================================

def test_02_empty_storage_cache_miss(isolated_vcp_environment):
    """
    Empty storage / cache miss: Returns truthful PIPELINE_PENDING status envelope without scanning.
    Guarantees no HTTP 500 and no background execution on fresh deployment or empty store.
    """
    env = isolated_vcp_environment
    client = env["client"]

    # 1. GET /api/v1/screener/vcp/snapshot
    resp_snap = client.get("/api/v1/screener/vcp/snapshot")
    assert resp_snap.status_code == 200, f"Expected 200 OK on cache miss, got {resp_snap.status_code}"
    snap_data = resp_snap.json()

    assert snap_data["status"] == ScannerStatus.PIPELINE_PENDING.value
    assert snap_data["reason"] == "NO_ACTIVE_SNAPSHOT"
    assert "No published VCP scanner snapshot is active" in snap_data["detail"]
    assert snap_data["snapshot"]["snapshot_id"] is None
    assert snap_data["snapshot"]["matched_count"] is None  # Invariant: matched_count is None, NOT 0
    assert snap_data["results"] == []
    assert snap_data["freshness"]["status"] == FreshnessStatus.UNKNOWN.value

    # 2. GET /api/v1/screener/run
    resp_run = client.get("/api/v1/screener/run")
    assert resp_run.status_code == 200, f"Expected 200 OK on screener run, got {resp_run.status_code}"
    run_data = resp_run.json()
    caps = run_data["capabilities"]

    # Capabilities contract reflects base radar capabilities; no VCP matches qualified
    assert caps["VCP"]["status"] in ("AVAILABLE", "PIPELINE_PENDING")
    vcp_matches = [c for c in run_data.get("candidates", []) if "VCP" in c.get("categories", [])]
    assert len(vcp_matches) == 0, "No candidates can qualify as VCP without active snapshot"

    # Spies: Zero scans, zero leases, zero publication calls, zero writes
    assert env["scan_spy"].call_count == 0, "Scan must not be called on cache miss"
    assert env["acquire_spy"].call_count == 0, "Execution lease must not be acquired"
    assert env["release_spy"].call_count == 0, "Execution lease must not be released"
    assert env["publish_spy"].call_count == 0, "Publication must not occur"
    assert env["save_spy"].call_count == 0, "No writes on cache miss"


# ==============================================================================
# Case 3: Snapshot Release Mismatch
# ==============================================================================

def test_03_snapshot_release_mismatch(isolated_vcp_environment):
    """
    Release SHA mismatch: Persisted snapshot from prior release must NOT be returned as active
    and must NOT trigger a background scan or lease acquisition.
    """
    env = isolated_vcp_environment
    client = env["client"]
    _seed_snapshot(
        env["store"],
        snapshot_id="vcp-snap-old-release",
        release_sha="previous-release-sha-00000000000000000000",
        matched_count=10,
    )
    env["save_spy"].reset_mock()

    # 1. GET /api/v1/screener/vcp/snapshot
    resp_snap = client.get("/api/v1/screener/vcp/snapshot")
    assert resp_snap.status_code == 200
    snap_data = resp_snap.json()

    assert snap_data["status"] == ScannerStatus.PIPELINE_PENDING.value
    assert snap_data["reason"] == "RELEASE_MISMATCH"
    assert "previous-release-sha" in snap_data["detail"]
    assert snap_data["snapshot"]["snapshot_id"] is None
    assert snap_data["snapshot"]["matched_count"] is None
    assert snap_data["results"] == []

    # 2. GET /api/v1/screener/run
    resp_run = client.get("/api/v1/screener/run")
    assert resp_run.status_code == 200
    caps = resp_run.json()["capabilities"]
    assert caps["VCP"]["status"] in ("AVAILABLE", "PIPELINE_PENDING")
    vcp_matches = [c for c in resp_run.json().get("candidates", []) if "VCP" in c.get("categories", [])]
    assert len(vcp_matches) == 0, "No candidates can qualify as VCP on release mismatch"

    # Spies
    assert env["scan_spy"].call_count == 0, "Scan must not be called on release mismatch"
    assert env["acquire_spy"].call_count == 0, "Lease must not be acquired"
    assert env["release_spy"].call_count == 0, "Lease must not be released"
    assert env["publish_spy"].call_count == 0, "Publication must not occur"
    assert env["save_spy"].call_count == 0, "No writes on mismatch"


# ==============================================================================
# Case 4: Stale Snapshot According to Existing Policy
# ==============================================================================

def test_04_stale_snapshot_policy(isolated_vcp_environment):
    """
    Stale snapshot: Returns the snapshot with explicit STALE indication without triggering scan.
    """
    env = isolated_vcp_environment
    client = env["client"]
    _seed_snapshot(
        env["store"],
        snapshot_id="vcp-snap-stale-001",
        freshness_status="STALE",
        status_at_publication="STALE",
        matched_count=8,
    )
    env["save_spy"].reset_mock()

    # 1. GET /api/v1/screener/vcp/snapshot
    resp_snap = client.get("/api/v1/screener/vcp/snapshot")
    assert resp_snap.status_code == 200
    snap_data = resp_snap.json()

    assert snap_data["status"] == "STALE"
    assert snap_data["freshness"]["status"] == "STALE"
    assert snap_data["snapshot"]["snapshot_id"] == "vcp-snap-stale-001"
    assert snap_data["snapshot"]["matched_count"] == 8
    assert len(snap_data["results"]) == 8

    # 2. GET /api/v1/screener/run
    resp_run = client.get("/api/v1/screener/run")
    assert resp_run.status_code == 200
    caps = resp_run.json()["capabilities"]
    assert caps["VCP"]["status"] in ("AVAILABLE", "STALE")

    # Spies: zero execution side effects
    assert env["scan_spy"].call_count == 0, "Stale snapshot must not initiate market scan"
    assert env["acquire_spy"].call_count == 0, "Lease must not be acquired"
    assert env["release_spy"].call_count == 0, "Lease must not be released"
    assert env["publish_spy"].call_count == 0, "Publication must not occur"
    assert env["save_spy"].call_count == 0, "No writes on stale read"


# ==============================================================================
# Case 5: Storage Failure Handled Without Launching Execution
# ==============================================================================

def test_05_storage_failure(isolated_vcp_environment, monkeypatch):
    """
    Storage failure: Returns ERROR status envelope according to error contract without
    launching execution as recovery.
    """
    env = isolated_vcp_environment
    client = env["client"]

    # Simulate database disk/operational error
    def _faulty_get(*args, **kwargs):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr(env["runner"].snapshot_store, "get_latest_active_snapshot", _faulty_get)
    monkeypatch.setattr(env["runner"].snapshot_store, "get_latest_active_snapshot_id", _faulty_get)

    # 1. GET /api/v1/screener/vcp/snapshot
    resp_snap = client.get("/api/v1/screener/vcp/snapshot")
    assert resp_snap.status_code == 200
    snap_data = resp_snap.json()

    assert snap_data["status"] == ScannerStatus.ERROR.value
    assert snap_data["reason"] == "STORAGE_ERROR"
    assert "Failed to access scanner snapshot store" in snap_data["detail"]
    assert snap_data["snapshot"]["matched_count"] is None
    assert snap_data["results"] == []

    # 2. GET /api/v1/screener/run
    resp_run = client.get("/api/v1/screener/run")
    assert resp_run.status_code == 200
    caps = resp_run.json()["capabilities"]
    assert caps["VCP"]["status"] in ("AVAILABLE", "ERROR")
    vcp_matches = [c for c in resp_run.json().get("candidates", []) if "VCP" in c.get("categories", [])]
    assert len(vcp_matches) == 0, "No candidates qualify as VCP on storage failure"

    # Spies: Crucially, runner must NOT fall back to scan execution
    assert env["scan_spy"].call_count == 0, "Storage failure must never initiate scan as recovery"
    assert env["acquire_spy"].call_count == 0, "Lease must not be acquired on storage failure"
    assert env["release_spy"].call_count == 0, "Lease must not be released"
    assert env["publish_spy"].call_count == 0, "Publication must not occur"
    assert env["save_spy"].call_count == 0, "No writes on storage failure"


# ==============================================================================
# Case 6: Repeated Public Reads
# ==============================================================================

def test_06_repeated_reads(isolated_vcp_environment):
    """
    Repeated reads: 10 consecutive calls to /vcp/snapshot and 10 calls to /screener/run.
    Proves absolute idempotency with zero scans, zero leases, and zero writes.
    """
    from api.middleware.rate_limiter import in_memory_rate_store
    in_memory_rate_store.clear()

    env = isolated_vcp_environment
    client = env["client"]

    for i in range(10):
        headers = {"X-Forwarded-For": f"10.0.0.{i + 1}"}
        resp = client.get("/api/v1/screener/vcp/snapshot", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == ScannerStatus.PIPELINE_PENDING.value

    for i in range(10):
        headers = {"X-Forwarded-For": f"10.0.1.{i + 1}"}
        resp = client.get("/api/v1/screener/run", headers=headers)
        assert resp.status_code == 200
        caps = resp.json()["capabilities"]
        assert caps["VCP"]["status"] in ("AVAILABLE", "PIPELINE_PENDING")

    # Total across all 20 public reads: strictly 0
    assert env["scan_spy"].call_count == 0, "Repeated reads must never initiate scan"
    assert env["acquire_spy"].call_count == 0, "Repeated reads must never acquire lease"
    assert env["release_spy"].call_count == 0, "Repeated reads must never release lease"
    assert env["publish_spy"].call_count == 0, "Repeated reads must never publish"
    assert env["save_spy"].call_count == 0, "Repeated reads must never write to store"
