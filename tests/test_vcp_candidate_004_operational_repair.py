"""ARX Terminal — VCP Candidate 004 Minimal Operational Repair Test Suite.

Verifies:
1. Candidate 004 generation identity binding in natural trigger and warmup.
2. Fail-closed rejection of mismatched candidate generations.
3. Explicit scan, publication, and admission outcome statuses in natural trigger returns.
4. Multi-worker snapshot cache invalidation via indexed snapshot_id lookup.
5. POST /api/v1/screener/vcp/natural-scan scheduler authorization, disabled-by-default safety, and token auth.
6. Radar capability contract enhancement with VCP snapshot freshness disclosure.
7. Frozen semantic closure hash immutability verification.
"""

from __future__ import annotations

import os
import time
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from analyst_dashboard.vcp.sprint_3_shadow_governance import (
    CANDIDATE_004_GENERATION_ID,
    CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH,
)
from analyst_dashboard.vcp.natural_trigger import (
    NaturalVCPTriggerService,
    get_natural_vcp_trigger_service,
)
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner, CURRENT_IMPLEMENTATION_RELEASE_SHA
from analyst_dashboard.analyzers.scanner_contract import ScannerStatus
from analyst_dashboard.analyzers.scanner_publication_integrity import PublicationDecision
from api.main import app
from api.routes.screener import get_radar_capabilities, is_vcp_natural_scheduler_enabled, verify_scheduler_token


# ======================================================================
# 1. Candidate 004 Identity Binding & Fail-Closed Validation
# ======================================================================

def test_natural_trigger_candidate_004_binding():
    """Verify natural trigger service binds strictly to CANDIDATE_GENERATION_004."""
    mock_runner = MagicMock()
    mock_runner.execute_market_wide_scan.return_value = {
        "snapshot_id": "vcp-snap-test-001",
        "status": "AVAILABLE",
        "publication_decision": "PUBLISH",
        "matched_count": 25,
        "universe_size": 35,
        "results": [{"symbol": "NVDA"}],
        "freshness": {"data_as_of": "2026-10-14", "generated_at": "2026-10-14T16:00:00Z", "status": "LIVE"},
        "provenance": {"implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA},
    }

    svc = NaturalVCPTriggerService(scanner_runner=mock_runner)
    result = svc.trigger_natural_scan(scheduled_date="2026-10-14")

    # Verify runner was called with CANDIDATE_004_GENERATION_ID
    mock_runner.execute_market_wide_scan.assert_called_once()
    kwargs = mock_runner.execute_market_wide_scan.call_args.kwargs
    assert kwargs["candidate_generation_id"] == CANDIDATE_004_GENERATION_ID
    assert kwargs["invocation_class"] == "SCHEDULED_PRODUCTION"
    assert kwargs["originating_principal_type"] == "SCHEDULER"

    # Verify structured return statuses
    assert result["status"] == "SUCCESS"
    assert result["scan_execution_status"] == "SUCCESS"
    assert result["snapshot_publication_status"] == "PUBLISH"
    assert result["natural_observation_status"] == "RECORDED"
    assert result["prospective_admission_status"] == "ADMITTED"
    assert result["candidate_generation_id"] == CANDIDATE_004_GENERATION_ID
    assert result["snapshot_id"] == "vcp-snap-test-001"
    assert result["matched_count"] == 25
    assert result["universe_size"] == 35


def test_natural_trigger_candidate_generation_mismatch_fails_closed():
    """Verify natural trigger raises ValueError if an unauthorized candidate generation is requested."""
    mock_runner = MagicMock()
    svc = NaturalVCPTriggerService(scanner_runner=mock_runner)

    with pytest.raises(ValueError, match="CANDIDATE_GENERATION_MISMATCH"):
        svc.trigger_natural_scan(candidate_generation_id="CANDIDATE_GENERATION_003")

    # Verify runner was NOT called
    mock_runner.execute_market_wide_scan.assert_not_called()


def test_boot_warmup_candidate_004_binding():
    """Verify boot warmup scan binds strictly to CANDIDATE_GENERATION_004."""
    mock_runner = MagicMock()
    mock_runner.execute_market_wide_scan.return_value = {"snapshot_id": "vcp-snap-warmup"}

    svc = NaturalVCPTriggerService(scanner_runner=mock_runner)
    svc.trigger_boot_warmup()

    mock_runner.execute_market_wide_scan.assert_called_once()
    kwargs = mock_runner.execute_market_wide_scan.call_args.kwargs
    assert kwargs["candidate_generation_id"] == CANDIDATE_004_GENERATION_ID
    assert kwargs["invocation_class"] == "BOOT_WARMUP"
    assert kwargs["originating_principal_type"] == "BOOTSTRAP"
    assert kwargs["startup_context"] is True


# ======================================================================
# 2. Snapshot Cache Freshness & Invalidation
# ======================================================================

def test_cache_invalidation_on_new_snapshot_publication(tmp_path):
    """Verify VCPScannerRunner detects newly published snapshot and does not serve stale cached snapshot."""
    test_db = str(tmp_path / "test_scanner_store.db")
    store = ScannerSnapshotStore(db_path=test_db)

    def make_snapshot(snap_id: str, data_as_of: str, count: int) -> dict:
        return {
            "snapshot_id": snap_id,
            "scanner_id": "MINERVINI_VCP",
            "run_id": f"run-{snap_id}",
            "api_contract_version": "1.0.0",
            "ruleset_version": "2.0.0",
            "evidence_schema_version": "1.0.0",
            "score_model_version": "1.0.0",
            "data_provenance_version": "1.0.0",
            "universe_version": "1.0.0",
            "freshness_policy_version": "1.0.0",
            "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
            "semantic_fingerprint": "abc123fingerprint",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_as_of": data_as_of,
            "status_at_publication": "AVAILABLE",
            "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
            "universe_size": 35,
            "matched_count": count,
            "results": [{"symbol": f"SYM{i}", "current_price": 100.0} for i in range(count)],
            "provenance": {"role": "PRICE_VOLUME_HISTORY"},
            "freshness": {"status": "LIVE", "data_as_of": data_as_of, "generated_at": "2026-10-14T00:00:00Z"},
            "publication_decision": "PUBLISH",
            "universe_metadata": {"scope_class": "US_EQUITIES"},
            "coverage_metadata": {"coverage_status": "COMPLETE"},
        }

    # Step 1: Save snapshot A (e.g. from 2026-10-09)
    snap_a = make_snapshot("vcp-snap-A-old", "2026-10-09", 25)
    store.save_snapshot(snap_a)

    # Step 2: Initialize runner with store
    runner = VCPScannerRunner(snapshot_store=store)
    loaded_1 = runner.get_active_or_latest_snapshot()
    assert loaded_1["snapshot"]["snapshot_id"] == "vcp-snap-A-old"
    assert loaded_1["snapshot"]["matched_count"] == 25
    assert runner._cached_active_snapshot["snapshot_id"] == "vcp-snap-A-old"

    # Step 3: Fast-path cached return when no new snapshot exists
    loaded_2 = runner.get_active_or_latest_snapshot()
    assert loaded_2["snapshot"]["snapshot_id"] == "vcp-snap-A-old"

    # Step 4: Another worker or scheduled scan publishes snapshot B (e.g. from 2026-10-14)
    time.sleep(0.01)  # Ensure distinct created_at timestamp
    snap_b = make_snapshot("vcp-snap-B-new", "2026-10-14", 28)
    store.save_snapshot(snap_b)

    # Verify indexed ID query returns snap_b
    latest_id = store.get_latest_active_snapshot_id("MINERVINI_VCP")
    assert latest_id == "vcp-snap-B-new"

    # Step 5: Runner must detect the new snapshot and return snapshot B without serving stale cache
    loaded_3 = runner.get_active_or_latest_snapshot()
    assert loaded_3["snapshot"]["snapshot_id"] == "vcp-snap-B-new"
    assert loaded_3["snapshot"]["matched_count"] == 28
    assert runner._cached_active_snapshot["snapshot_id"] == "vcp-snap-B-new"


# ======================================================================
# 3. Comprehensive 12-Case Security & Execution Test Matrix (Phase B2)
# ======================================================================

def test_b2_case_01_scheduler_disabled_no_token(monkeypatch):
    """Case 1: Scheduler disabled, no token -> 503 Service Unavailable."""
    monkeypatch.delenv("VCP_NATURAL_SCHEDULER_ENABLED", raising=False)
    client = TestClient(app)
    res = client.post("/api/v1/screener/vcp/natural-scan")
    assert res.status_code == 503
    assert "NATURAL_SCHEDULER_DISABLED" in res.json()["detail"]


def test_b2_case_02_scheduler_disabled_valid_token(monkeypatch):
    """Case 2: Scheduler disabled, valid token -> 503 Service Unavailable (disabled state takes precedence)."""
    monkeypatch.delenv("VCP_NATURAL_SCHEDULER_ENABLED", raising=False)
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "valid-sched-token")
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "valid-sched-token"},
    )
    assert res.status_code == 503
    assert "NATURAL_SCHEDULER_DISABLED" in res.json()["detail"]


def test_b2_case_03_scheduler_enabled_no_configured_secret(monkeypatch):
    """Case 3: Scheduler enabled, no configured secret -> 403 Forbidden (fail closed)."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.delenv("VCP_SCHEDULER_SECRET", raising=False)
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "some-token"},
    )
    assert res.status_code == 403
    assert "SCHEDULER_AUTHENTICATION_FAILED" in res.json()["detail"]


def test_b2_case_04_scheduler_enabled_empty_configured_secret(monkeypatch):
    """Case 4: Scheduler enabled, empty configured secret -> 403 Forbidden (fail closed)."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "   ")
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "some-token"},
    )
    assert res.status_code == 403
    assert "SCHEDULER_AUTHENTICATION_FAILED" in res.json()["detail"]


def test_b2_case_05_scheduler_enabled_invalid_token(monkeypatch):
    """Case 5: Scheduler enabled, invalid token -> 403 Forbidden."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "legit-secret-key-999")
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "wrong-adversarial-token"},
    )
    assert res.status_code == 403
    assert "SCHEDULER_AUTHENTICATION_FAILED" in res.json()["detail"]


def test_b2_case_06_scheduler_enabled_telemetry_secret_fallback_rejected(monkeypatch):
    """Case 6: Cross-domain fallback to TELEMETRY_INTERNAL_SECRET is strictly rejected -> 403 Forbidden."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.delenv("VCP_SCHEDULER_SECRET", raising=False)
    monkeypatch.setenv("TELEMETRY_INTERNAL_SECRET", "telemetry-only-token")
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "telemetry-only-token"},
    )
    assert res.status_code == 403
    assert "SCHEDULER_AUTHENTICATION_FAILED" in res.json()["detail"]


def test_b2_case_07_scheduler_enabled_epoch_activation_secret_fallback_rejected(monkeypatch):
    """Case 7: Cross-domain fallback to ARX_EPOCH_ACTIVATION_KEY is strictly rejected -> 403 Forbidden."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.delenv("VCP_SCHEDULER_SECRET", raising=False)
    monkeypatch.setenv("ARX_EPOCH_ACTIVATION_KEY", "epoch-only-key")
    client = TestClient(app)
    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "epoch-only-key"},
    )
    assert res.status_code == 403
    assert "SCHEDULER_AUTHENTICATION_FAILED" in res.json()["detail"]


def test_b2_case_08_scheduler_enabled_valid_dedicated_token(monkeypatch):
    """Case 8: Scheduler enabled, valid dedicated VCP_SCHEDULER_SECRET -> 200 OK."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "dedicated-vcp-secret-1234")
    client = TestClient(app)

    mock_result = {
        "status": "SUCCESS",
        "scan_execution_status": "SUCCESS",
        "snapshot_publication_status": "PUBLISH",
        "natural_observation_status": "RECORDED",
        "prospective_admission_status": "ADMITTED",
        "candidate_generation_id": CANDIDATE_004_GENERATION_ID,
        "logical_scan_run_id": "run-sched-test",
        "snapshot_id": "vcp-snap-sched-001",
        "matched_count": 25,
        "universe_size": 35,
    }

    with patch("analyst_dashboard.vcp.natural_trigger.NaturalVCPTriggerService.trigger_natural_scan", return_value=mock_result) as mock_trigger:
        res = client.post(
            "/api/v1/screener/vcp/natural-scan",
            headers={"X-Scheduler-Token": "dedicated-vcp-secret-1234"},
            json={"scheduled_date": "2026-10-14"},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"
        mock_trigger.assert_called_once()


def test_b2_case_09_repeated_unauthorized_requests(monkeypatch):
    """Case 9: Repeated unauthorized attempts produce zero mutations and consistently fail closed."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "secret-token")
    client = TestClient(app)

    with patch("analyst_dashboard.vcp.natural_trigger.NaturalVCPTriggerService.trigger_natural_scan") as mock_trigger:
        for _ in range(5):
            res = client.post(
                "/api/v1/screener/vcp/natural-scan",
                headers={"X-Scheduler-Token": "attacker-token"},
            )
            assert res.status_code == 403
        mock_trigger.assert_not_called()


def test_b2_case_10_malformed_request_body(monkeypatch):
    """Case 10: Malformed request payload -> 422 Unprocessable Entity."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "secret-token")
    client = TestClient(app)

    res = client.post(
        "/api/v1/screener/vcp/natural-scan",
        headers={"X-Scheduler-Token": "secret-token", "Content-Type": "application/json"},
        content="not-valid-json{{{",
    )
    assert res.status_code == 422


def test_b2_case_11_unexpected_http_method():
    """Case 11: Unexpected HTTP method (GET) -> 405 Method Not Allowed."""
    client = TestClient(app)
    res = client.get("/api/v1/screener/vcp/natural-scan")
    assert res.status_code == 405


def test_b2_case_12_concurrent_execution_overlapping_scan_prohibited(monkeypatch):
    """Case 12: Overlapping concurrent scans map to HTTP 409 Conflict."""
    monkeypatch.setenv("VCP_NATURAL_SCHEDULER_ENABLED", "1")
    monkeypatch.setenv("VCP_SCHEDULER_SECRET", "secret-token")
    client = TestClient(app)

    with patch(
        "analyst_dashboard.vcp.natural_trigger.NaturalVCPTriggerService.trigger_natural_scan",
        side_effect=RuntimeError("OVERLAPPING_VCP_SCANS_PROHIBITED: A market-wide VCP scan is already in progress."),
    ):
        res = client.post(
            "/api/v1/screener/vcp/natural-scan",
            headers={"X-Scheduler-Token": "secret-token"},
            json={"scheduled_date": "2026-10-14"},
        )
        assert res.status_code == 409
        assert "OVERLAPPING_VCP_SCANS_PROHIBITED" in res.json()["detail"]


# ======================================================================
# 4. Cross-Process Cache Convergence & Invalidation Suite (Phase C2)
# ======================================================================

def test_c2_multi_reader_convergence_and_rejected_snapshot_isolation(tmp_path):
    """
    Phase C2 Scenario:
    1. Reader A & Reader B initialize and cache S1.
    2. Publisher commits S2 (PUBLISH). Both readers detect and serve S2.
    3. A rejected or staged snapshot S3 (publication_decision='REJECTED') is inserted.
    4. Both readers continue serving S2, ignoring S3.
    5. Valid S4 (PUBLISH) is committed. Both readers converge on S4.
    6. Reader A restarts: loads authoritative S4 directly.
    """
    test_db = str(tmp_path / "test_c2_store.db")
    store = ScannerSnapshotStore(db_path=test_db)

    def insert_snapshot(snap_id: str, decision: str = "PUBLISH", matches: int = 25) -> dict:
        data = {
            "snapshot_id": snap_id,
            "scanner_id": "MINERVINI_VCP",
            "run_id": f"run-{snap_id}",
            "api_contract_version": "1.0.0",
            "ruleset_version": "2.0.0",
            "evidence_schema_version": "1.0.0",
            "score_model_version": "1.0.0",
            "data_provenance_version": "1.0.0",
            "universe_version": "1.0.0",
            "freshness_policy_version": "1.0.0",
            "implementation_release_sha": CURRENT_IMPLEMENTATION_RELEASE_SHA,
            "semantic_fingerprint": f"fp-{snap_id}",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_as_of": "2026-10-14",
            "status_at_publication": "AVAILABLE" if decision == "PUBLISH" else "ERROR",
            "universe_id": "ARX_CANONICAL_LONG_TERM_V1",
            "universe_size": 35,
            "matched_count": matches,
            "results": [{"symbol": f"S{i}", "current_price": 50.0} for i in range(matches)],
            "provenance": {"role": "PRICE_VOLUME_HISTORY"},
            "freshness": {"status": "LIVE", "data_as_of": "2026-10-14", "generated_at": "2026-10-14T12:00:00Z"},
            "publication_decision": decision,
            "universe_metadata": {"scope_class": "US_EQUITIES"},
            "coverage_metadata": {"coverage_status": "COMPLETE"},
        }
        store.save_snapshot(data)
        return data

    # 1. Reader A and Reader B cache S1
    insert_snapshot("S1", decision="PUBLISH", matches=10)
    reader_a = VCPScannerRunner(snapshot_store=store)
    reader_b = VCPScannerRunner(snapshot_store=store)
    assert reader_a.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S1"
    assert reader_b.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S1"

    # 2. Publisher commits S2 (PUBLISH)
    insert_snapshot("S2", decision="PUBLISH", matches=20)
    assert reader_a.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S2"
    assert reader_b.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S2"

    # 3. A rejected or staged snapshot S3 is inserted
    insert_snapshot("S3", decision="REJECTED", matches=99)

    # 4. Both readers continue serving authoritative S2
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") == "S2"
    assert reader_a.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S2"
    assert reader_b.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S2"

    # 5. Publisher commits valid S4 (PUBLISH)
    insert_snapshot("S4", decision="PUBLISH", matches=25)
    assert store.get_latest_active_snapshot_id("MINERVINI_VCP") == "S4"
    assert reader_a.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S4"
    assert reader_b.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S4"

    # 6. Reader A restarts (new instance)
    reader_a_fresh = VCPScannerRunner(snapshot_store=store)
    assert reader_a_fresh.get_active_or_latest_snapshot()["snapshot"]["snapshot_id"] == "S4"


# ======================================================================
# 5. Radar Capabilities & Freshness Disclosure
# ======================================================================

def test_radar_capabilities_includes_vcp_freshness():
    """Verify get_radar_capabilities surfaces VCP snapshot freshness metadata."""
    vcp_snap = {
        "status": "AVAILABLE",
        "matched_count": 25,
        "snapshot_id": "vcp-snap-20261009",
        "data_as_of": "2026-10-09",
        "freshness": {
            "status": "LIVE",
            "data_as_of": "2026-10-09",
            "generated_at": "2026-10-09T21:16:07Z",
        },
    }

    caps = get_radar_capabilities(vcp_snapshot=vcp_snap)
    assert caps["VCP"]["status"] == "AVAILABLE"
    assert caps["VCP"]["matchedCount"] == 25
    assert caps["VCP"]["dataAsOf"] == "2026-10-09"
    assert caps["VCP"]["generatedAt"] == "2026-10-09T21:16:07Z"
    assert caps["VCP"]["snapshotId"] == "vcp-snap-20261009"
    assert caps["VCP"]["freshnessStatus"] == "LIVE"


# ======================================================================
# 6. Semantic Closure Hash Immutability
# ======================================================================

def test_semantic_closure_hash_immutability():
    """Verify frozen Candidate 004 semantic closure hash is completely intact and unchanged."""
    expected_hash = "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6"
    assert CANONICAL_CANDIDATE_SEMANTIC_CLOSURE_HASH == expected_hash
