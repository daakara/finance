"""Regression checks for production VCP snapshot authority (Recovery 002)."""

import tempfile

from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.analyzers.scanner_runner import (
    VCPScannerRunner,
    CURRENT_IMPLEMENTATION_RELEASE_SHA,
)


def _snapshot(snapshot_id: str, status: str) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "scanner_id": "MINERVINI_VCP",
        "run_id": f"run-{snapshot_id}",
        "api_contract_version": "1.0.0",
        "ruleset_version": "1.0.0",
        "evidence_schema_version": "1.0.0",
        "score_model_version": "1.0.0",
        "data_provenance_version": "1.0.0",
        "universe_version": "1.0.0",
        "freshness_policy_version": "1.0.0",
        "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
        "semantic_fingerprint": "test-recovery-002",
        "generated_at": "2026-11-03T21:30:00Z",
        "data_as_of": "2026-11-03",
        "status_at_publication": status,
        "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
        "universe_size": 122,
        "matched_count": 0 if status == "AVAILABLE" else None,
        "results": [],
        "provenance": {},
        "freshness": {"status": "LIVE"},
        "publication_decision": "PUBLISH",
    }


def test_active_snapshot_and_id_exclude_nonavailable_publications(tmp_path):
    store = ScannerSnapshotStore(db_path=str(tmp_path / "scanner.db"))
    good = _snapshot("accepted-available", "AVAILABLE")
    store.save_snapshot(good)
    for status in ("ERROR", "PIPELINE_PENDING", "STALE"):
        store.save_snapshot(_snapshot(f"later-{status}", status))
    selected = store.get_latest_active_snapshot("MINERVINI_VCP")
    assert selected is not None
    assert selected["snapshot_id"] == "accepted-available"
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") == "accepted-available"


def test_no_eligible_snapshot_returns_none(tmp_path):
    store = ScannerSnapshotStore(db_path=str(tmp_path / "scanner.db"))
    store.save_snapshot(_snapshot("only-error", "ERROR"))
    assert store.get_latest_active_snapshot("MINERVINI_VCP") is None
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") is None


def test_production_boot_warmup_does_not_publish(tmp_path):
    store = ScannerSnapshotStore(db_path=str(tmp_path / "scanner.db"))
    runner = VCPScannerRunner(snapshot_store=store)
    result = runner.execute_market_wide_scan(
        universe_override=[],
        invocation_class="BOOT_WARMUP",
        startup_context=True,
    )
    assert result["status"] == "ERROR"
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") is None


def test_empty_universe_error_is_not_active(tmp_path):
    store = ScannerSnapshotStore(db_path=str(tmp_path / "scanner.db"))
    runner = VCPScannerRunner(snapshot_store=store)
    result = runner.execute_market_wide_scan(universe_override=[])
    assert result["status"] == "ERROR"
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") is None
