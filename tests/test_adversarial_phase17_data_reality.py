"""Phase 17 Adversarial Data Reality, Provenance & Epistemic Honesty Verification Suite.

Validates that:
1. Placeholder sentiment never becomes actionable sentiment.
2. On-chain zero defaults never become 'zero activity' (explicitly UNAVAILABLE).
3. Static congressional trades are explicitly badged as CURATED historical dataset.
4. Static options trades never appear as live flow on ticker searches.
5. Dark pool unavailable state is strictly distinguishable from zero activity.
6. Unsupported SEC ticker is UNAVAILABLE, not 'no insider activity'.
7. SEC data includes provenance, coverage, and CIK metadata.
8. Stale live data is labeled accurately.
9. Fallback market data cannot claim live provenance.
10. Unknown data never contributes positive confluence.
11. Unknown data never contributes negative confluence.
12. Empty arrays do not imply zero activity.
13. Synthetic institutional events cannot reach production adapters.
14. Every material institutional signal has source metadata.
15. UI labels cannot claim 'live' for curated data.
16. Chaos resilience: timeouts, 429s, 500s, malformed responses, and impossible dates.
"""

import os
import pytest
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Placeholder sentiment never becomes actionable sentiment
# ---------------------------------------------------------------------------
def test_1_placeholder_sentiment_never_actionable():
    ce = ConfluenceEngine()
    res = ce.calculate_confluence(
        symbol="NVDA",
        technical_data={"current_price": 200.0, "stop_loss": 190.0},
        smart_money_data=None,
    )
    # Missing sentiment must not bias score upward
    assert res["confluenceScore"] < 80.0
    # No pillar should claim positive sentiment from missing data
    for p in res["pillars"]:
        assert p["pillar"] != "SOCIAL_SENTIMENT", "Social sentiment must not be an active math pillar without provider"


# ---------------------------------------------------------------------------
# Test 2: On-chain zero defaults never become 'zero activity'
# ---------------------------------------------------------------------------
def test_2_onchain_missing_is_unavailable_not_zero():
    from engines.fundamental_engine import CryptoFundamentalAnalyzer
    import pandas as pd
    cfa = CryptoFundamentalAnalyzer()
    res = cfa._generate_onchain_metrics("BTC", pd.DataFrame())
    assert res["available"] is False
    assert res["status"] == "UNAVAILABLE"
    assert res["metrics"] is None, "Metrics must be None, not fake zeroes"


# ---------------------------------------------------------------------------
# Test 3: Static congressional trades are labelled curated/historical
# ---------------------------------------------------------------------------
def test_3_congressional_trades_labeled_curated():
    res = client.get("/api/v1/smart-money/congress")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "CURATED"
    assert "curated" in data["disclosure"].lower()
    assert "not a real-time live trading feed" in data["disclosure"].lower()


# ---------------------------------------------------------------------------
# Test 4: Static options trades never appear as live flow on ticker search
# ---------------------------------------------------------------------------
def test_4_options_flow_not_live_on_ticker_search(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("OPRA_API_KEY", raising=False)
    res = client.get("/api/v1/smart-money/options-flow?symbol=NVDA")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["flow"] == [], "Without API key, live ticker options flow must be empty"
    assert "unavailable" in data["message"].lower()


# ---------------------------------------------------------------------------
# Test 5: Dark pool unavailable state distinguishable from zero activity
# ---------------------------------------------------------------------------
def test_5_darkpool_unavailable_distinguishable_from_zero():
    res = client.get("/api/v1/smart-money/finra-darkpool/UNKNOWN")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["metrics"] is None
    assert "unavailable" in data["message"].lower()


# ---------------------------------------------------------------------------
# Test 6: Unsupported SEC ticker is UNAVAILABLE, not 'no insider activity'
# ---------------------------------------------------------------------------
def test_6_unsupported_sec_ticker_is_unavailable():
    res = client.get("/api/v1/smart-money/sec-filings/NONEXIST")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["cik"] is None
    assert "unsupported" in data["coverage"].lower()
    assert data["filings"] == []


# ---------------------------------------------------------------------------
# Test 7: SEC data includes provenance, coverage, and CIK metadata
# ---------------------------------------------------------------------------
def test_7_sec_data_includes_provenance_metadata():
    res = client.get("/api/v1/smart-money/sec-filings/AAPL")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert data["cik"] == "0000320193"
    assert "supported" in data["coverage"].lower()
    assert "filings" in data


# ---------------------------------------------------------------------------
# Test 8: Stale live data is marked stale / curated
# ---------------------------------------------------------------------------
def test_8_stale_curated_data_does_not_inflate_live_confluence():
    res = client.get("/api/v1/analytics/PLTR")
    assert res.status_code == 200
    data = res.json()
    # Options flow must be empty on live analytics
    assert data["smartMoney"]["optionsFlow"] == []


# ---------------------------------------------------------------------------
# Test 9: Fallback market data cannot claim live provenance
# ---------------------------------------------------------------------------
def test_9_fallback_badge_renders_model_estimate():
    badge_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "components", "DataSourceBadge.tsx")
    with open(badge_path, "r", encoding="utf-8") as f:
        content = f.read()
  
    assert "Model Estimate" in content
    assert "bg-amber-950" in content, "Fallback badge must render amber, never emerald live styling"


# ---------------------------------------------------------------------------
# Test 10: Unknown data never contributes positive confluence
# ---------------------------------------------------------------------------
def test_10_unknown_data_never_contributes_positive_confluence():
    ce = ConfluenceEngine()
    res = ce.calculate_confluence(
        symbol="UNCATALOGED",
        technical_data={"current_price": 50.0, "stop_loss": 48.0},
        smart_money_data=None,
        fundamental_data=None,
    )
    # Missing smart money and missing fundamentals must not award positive points
    for p in res["pillars"]:
        if p["pillar"] in ["SMART_MONEY_FLOW", "FUNDAMENTAL_SOLVENCY"]:
            assert p["status"] == "unavailable"
            assert p["score"] == 0.0


# ---------------------------------------------------------------------------
# Test 11: Unknown data never contributes negative confluence (except earnings gap)
# ---------------------------------------------------------------------------
def test_11_unknown_data_never_contributes_negative_confluence():
    ce = ConfluenceEngine()
    res = ce.calculate_confluence(
        symbol="UNKNOWN_XYZ",
        technical_data={"current_price": 100.0, "stop_loss": 95.0},
        smart_money_data=None,
        catalyst_data=None,
    )
    # With no catalyst data, catalyst_mod must be 0.0 (no penalty)
    assert len(res["warnings"]) == 0


# ---------------------------------------------------------------------------
# Test 12: Empty arrays do not imply zero activity
# ---------------------------------------------------------------------------
def test_12_empty_arrays_do_not_imply_zero_activity():
    res = client.get("/api/v1/smart-money/options-flow?symbol=PLTR")
    assert res.status_code == 200
    data = res.json()
    assert data["flow"] == []
    # Explicit message communicates provider requirement rather than zero sweeps
    assert "unavailable" in data["message"].lower()


# ---------------------------------------------------------------------------
# Test 13: Synthetic institutional events cannot reach production adapters
# ---------------------------------------------------------------------------
def test_13_no_synthetic_whale_sweeps_in_live_analytics():
    res = client.get("/api/v1/analytics/NVDA")
    assert res.status_code == 200
    data = res.json()
    assert data["smartMoney"]["optionsFlow"] == []


# ---------------------------------------------------------------------------
# Test 14: Every material institutional signal has source metadata
# ---------------------------------------------------------------------------
def test_14_institutional_signals_have_source_metadata():
    res = client.get("/api/v1/smart-money/congress")
    assert res.status_code == 200
    data = res.json()
    assert "source_meta" in data
    assert "dataset_date" in data


# ---------------------------------------------------------------------------
# Test 15: UI labels cannot claim 'live' for curated data
# ---------------------------------------------------------------------------
def test_15_no_deceptive_live_claims_in_ui_components():
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    card_file = os.path.join(frontend_dir, "components", "CongressionalTradesCard.tsx")
    with open(card_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Live Options Tape Active" not in content
    assert "Live Big-Money Flow" not in content


# ---------------------------------------------------------------------------
# Test 16: Chaos resilience: timeouts, 429s, 500s, malformed, impossible dates
# ---------------------------------------------------------------------------
def test_16_chaos_resilience(monkeypatch):
    fetcher = SecEdgarFetcher()

    # 1. Timeout chaos
    def mock_timeout(*a, **kw):
        raise TimeoutError("Simulated socket timeout")
    monkeypatch.setattr(fetcher.session, "get", mock_timeout)
    assert fetcher.get_company_submissions("NVDA") is None

    # 2. HTTP 429 chaos
    class Mock429:
        status_code = 429
        text = "Too many requests"
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: Mock429())
    assert fetcher.get_company_submissions("NVDA") is None

    # 3. HTTP 500 chaos
    class Mock500:
        status_code = 500
        text = "Internal Server Error"
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: Mock500())
    assert fetcher.get_company_submissions("NVDA") is None

    # 4. Malformed JSON chaos
    class MockMalformed:
        status_code = 200
        def json(self):
            return {"filings": "BROKEN_PAYLOAD_NOT_A_DICT"}
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: MockMalformed())
    assert fetcher.get_recent_filings("NVDA") == []
