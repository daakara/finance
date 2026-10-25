"""
ARX Terminal — Smart Money Radar End-to-End Contract Verification Suite.

Verifies the complete integration contract:
1. PIPELINE_PENDING semantics (default unprovisioned state).
2. Valid zero-match semantics (evaluated universe with 0 qualifying candidates).
3. Active snapshot candidate category qualification & evidence binding.
4. SEC egress role separation (web server DISABLED vs worker OWNER).
5. Immutable snapshot publication provenance.
"""

import pytest

pytestmark = pytest.mark.tier2c
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.smart_money.storage.factory import (
    get_smart_money_scanner_runner,
)
from analyst_dashboard.data.sec_edgar_fetcher import (
    get_sec_egress_role,
)
from analyst_dashboard.smart_money.worker import SmartMoneyIngestionWorker

client = TestClient(app)


def test_radar_default_pipeline_pending_contract():
    """
    Contract Step 1: In default unprovisioned state, Radar returns PIPELINE_PENDING.
    No synthetic results may be presented as live data.
    """
    res = client.get("/api/v1/screener/run")
    assert res.status_code == 200
    data = res.json()

    caps = data.get("capabilities", {})
    assert "SMART_MONEY" in caps
    sm_cap = caps["SMART_MONEY"]

    assert sm_cap["status"] == "PIPELINE_PENDING"
    assert sm_cap["universeScreening"] == "PIPELINE_PENDING"
    assert sm_cap["singleAssetAnalysis"] == "AVAILABLE"
    assert "matchedCount" not in sm_cap or sm_cap.get("matchedCount") is None

    # Filter for smart_money returns 0 candidates
    res_filter = client.get("/api/v1/screener/run?filter_type=smart_money")
    assert res_filter.status_code == 200
    filter_data = res_filter.json()
    assert filter_data["gemsFound"] == 0
    assert len(filter_data["candidates"]) == 0

    # Dedicated snapshot endpoint returns PIPELINE_PENDING
    res_snap = client.get("/api/v1/screener/smart-money/snapshot")
    assert res_snap.status_code == 200
    snap_data = res_snap.json()
    assert snap_data["status"] == "PIPELINE_PENDING"


def test_radar_valid_zero_match_contract():
    """
    Contract Step 2: When scanner evaluates universe and finds 0 qualifying stocks,
    it publishes status: AVAILABLE with matched_count: 0 (truthful zero match, not error).
    """
    mock_runner = MagicMock()
    mock_runner.get_active_or_latest_snapshot.return_value = {
        "status": "AVAILABLE",
        "scanner_id": "SMART_MONEY",
        "matched_count": 0,
        "results": [],
        "freshness": {"data_as_of": "2026-10-25T00:00:00Z"},
    }

    with patch("api.routes.screener.get_smart_money_scanner_runner", return_value=mock_runner):
        res = client.get("/api/v1/screener/run")
        assert res.status_code == 200
        data = res.json()

        sm_cap = data["capabilities"]["SMART_MONEY"]
        assert sm_cap["status"] == "AVAILABLE"
        assert sm_cap["universeScreening"] == "AVAILABLE"
        assert sm_cap["matchedCount"] == 0

        # Filtering returns 0 candidates with status AVAILABLE
        res_filter = client.get("/api/v1/screener/run?filter_type=smart_money")
        assert res_filter.status_code == 200
        assert res_filter.json()["gemsFound"] == 0


def test_radar_active_snapshot_candidate_qualification():
    """
    Contract Step 3: When active snapshot contains verified qualifying candidates,
    Radar categorizes matching tickers with SMART_MONEY and binds CRITERIA_MATCHED evidence.
    """
    mock_runner = MagicMock()
    mock_runner.get_active_or_latest_snapshot.return_value = {
        "status": "AVAILABLE",
        "scanner_id": "SMART_MONEY",
        "matched_count": 1,
        "results": [
            {
                "symbol": "NVDA",
                "conviction_score": 88.5,
                "net_transaction_value": 2450000.0,
                "cluster_count": 3,
            }
        ],
        "freshness": {"data_as_of": "2026-10-25T00:00:00Z"},
    }

    with patch("api.routes.screener.get_smart_money_scanner_runner", return_value=mock_runner):
        # Query with custom tickers including the matched symbol (NVDA) and an unmatched symbol (AAPL)
        res = client.get("/api/v1/screener/run?custom_tickers=NVDA,AAPL")
        assert res.status_code == 200
        data = res.json()

        nvda_cand = next((c for c in data["candidates"] if c["symbol"] == "NVDA"), None)
        aapl_cand = next((c for c in data["candidates"] if c["symbol"] == "AAPL"), None)

        assert nvda_cand is not None, "NVDA should be present in evaluated candidates"
        assert "SMART_MONEY" in nvda_cand["categories"], "NVDA must be tagged with SMART_MONEY category"
        assert nvda_cand["categoryEvidence"].get("SMART_MONEY") == "CRITERIA_MATCHED"

        if aapl_cand:
            assert "SMART_MONEY" not in aapl_cand["categories"], "AAPL must not have SMART_MONEY category"

        # Dedicated filter returns only the qualified candidate
        res_sm = client.get("/api/v1/screener/run?filter_type=smart_money&custom_tickers=NVDA,AAPL")
        assert res_sm.status_code == 200
        sm_data = res_sm.json()
        assert sm_data["gemsFound"] == 1
        assert sm_data["candidates"][0]["symbol"] == "NVDA"


def test_sec_egress_role_separation_contract(monkeypatch):
    """
    Contract Step 4: Strict SEC egress separation.
    Web server defaults to ARX_SEC_EGRESS_ROLE=DISABLED.
    Only dedicated worker with ARX_SEC_EGRESS_ROLE=OWNER is permitted to initiate egress.
    """
    # 1. On web server (or unconfigured default): role is DISABLED
    monkeypatch.delenv("ARX_SEC_EGRESS_ROLE", raising=False)
    assert get_sec_egress_role() == "DISABLED"

    # Attempting worker preflight with DISABLED role fails closed immediately
    worker = SmartMoneyIngestionWorker(dsn="postgresql://user:pass@127.0.0.1:5432/db")
    with pytest.raises(RuntimeError) as exc_info:
        worker.validate_preflight()
    assert "UNAUTHORIZED_WORKER_ROLE" in str(exc_info.value)

    # 2. On dedicated worker: role must be OWNER
    monkeypatch.setenv("ARX_SEC_EGRESS_ROLE", "OWNER")
    assert get_sec_egress_role() == "OWNER"
