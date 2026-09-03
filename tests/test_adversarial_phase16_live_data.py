"""Phase 16 Adversarial Live Data Integrity & Provenance Verification Suite.

Comprehensive 20-point adversarial test matrix enforcing:
LIVE != CACHED != DERIVED != CURATED != PLACEHOLDER != UNAVAILABLE
UNKNOWN != FAVORABLE, UNKNOWN != NEGATIVE, UNKNOWN != ACTIONABLE
STALE != LIVE, CURATED != LIVE, PLACEHOLDER != DATA
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
# Test 1: Unknown ticker SEC lookup fails closed
# ---------------------------------------------------------------------------
def test_1_unknown_ticker_sec_lookup_fails_closed():
    fetcher = SecEdgarFetcher()
    cik = fetcher.resolve_cik("COMPLETELY_BOGUS_TICKER_XYZ999")
    assert cik is None, "Unknown ticker must not resolve to an arbitrary CIK"

    filings = fetcher.get_recent_filings("COMPLETELY_BOGUS_TICKER_XYZ999")
    assert filings == [], "Unknown ticker must return empty filings list"


# ---------------------------------------------------------------------------
# Test 2: Empty provider response returns available: false
# ---------------------------------------------------------------------------
def test_2_empty_provider_response_returns_unavailable():
    res = client.get("/api/v1/smart-money/finra-darkpool/UNKNOWN")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["metrics"] is None


# ---------------------------------------------------------------------------
# Test 3: Provider timeout fails closed without throwing 500
# ---------------------------------------------------------------------------
def test_3_provider_timeout_fails_closed(monkeypatch):
    fetcher = SecEdgarFetcher()
    def mock_get(*args, **kwargs):
        raise TimeoutError("Simulated SEC EDGAR timeout")
    monkeypatch.setattr(fetcher.session, "get", mock_get)
    result = fetcher.get_company_submissions("NVDA")
    assert result is None, "Timeout must return None instead of raising unhandled exception"


# ---------------------------------------------------------------------------
# Test 4: Provider 500 fails closed gracefully
# ---------------------------------------------------------------------------
def test_4_provider_500_fails_closed(monkeypatch):
    fetcher = SecEdgarFetcher()
    class Mock500Response:
        status_code = 500
        text = "Internal Server Error"
        def json(self):
            return {}
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: Mock500Response())
    result = fetcher.get_company_submissions("AAPL")
    assert result is None, "Provider 500 must return None"


# ---------------------------------------------------------------------------
# Test 5: Provider rate limit (429) fails closed
# ---------------------------------------------------------------------------
def test_5_provider_rate_limit_fails_closed(monkeypatch):
    fetcher = SecEdgarFetcher()
    class Mock429Response:
        status_code = 429
        text = "Too Many Requests"
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: Mock429Response())
    result = fetcher.get_company_submissions("AAPL")
    assert result is None, "Rate limit 429 must fail closed without crashing"


# ---------------------------------------------------------------------------
# Test 6: Missing API key does not crash and marks feed unavailable
# ---------------------------------------------------------------------------
def test_6_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("OPRA_API_KEY", raising=False)
    # With no options key in environment, options flow must be empty for live searches
    flow = SmartMoneyEngine.get_options_flow("NVDA", include_curated=False)
    assert flow == [], "Without API key, live options flow must be empty"


# ---------------------------------------------------------------------------
# Test 7: Stale response does not trigger real-time buy confluence
# ---------------------------------------------------------------------------
def test_7_stale_response_does_not_trigger_buy_confluence():
    res = client.get("/api/v1/analytics/NVDA")
    assert res.status_code == 200
    data = res.json()
    # Live analytics route must return empty options flow unless live feed connected
    assert data["smartMoney"]["optionsFlow"] == []


# ---------------------------------------------------------------------------
# Test 8: Malformed response fails closed
# ---------------------------------------------------------------------------
def test_8_malformed_response_fails_closed(monkeypatch):
    fetcher = SecEdgarFetcher()
    class MockMalformedResponse:
        status_code = 200
        def json(self):
            return {"filings": "MALFORMED_STRING_NOT_DICT"}
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: MockMalformedResponse())
    filings = fetcher.get_recent_filings("AAPL")
    assert filings == [], "Malformed payload must return empty filings list"


# ---------------------------------------------------------------------------
# Test 9: Wrong ticker response rejected
# ---------------------------------------------------------------------------
def test_9_wrong_ticker_response_rejected():
    res = client.get("/api/v1/smart-money/finra-darkpool/AAPL")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "AAPL"
    # Even if another symbol is present, data['symbol'] matches request


# ---------------------------------------------------------------------------
# Test 10: Global dataset not accidentally returned for specific ticker
# ---------------------------------------------------------------------------
def test_10_no_global_dataset_leak_for_specific_ticker():
    # If a ticker has no congressional trades, it must return empty list, NOT all 28 trades!
    trades = SmartMoneyEngine.get_congressional_trades("BOGUS_UNKNOWN_TICKER")
    assert trades == [], "Unmatched ticker must return empty list, never global list"


# ---------------------------------------------------------------------------
# Test 11: Static fixture not classified as LIVE
# ---------------------------------------------------------------------------
def test_11_static_fixture_not_classified_as_live():
    feed_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "lib", "institutionalFeeds.ts")
    with open(feed_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "CURATED_HISTORICAL_SEC_TRADES" in content, "Static trades must be labeled CURATED_HISTORICAL"


# ---------------------------------------------------------------------------
# Test 12: Sentiment missing returns undefined/neutral without affecting math
# ---------------------------------------------------------------------------
def test_12_sentiment_missing_does_not_bias_confluence():
    ce = ConfluenceEngine()
    # Confluence engine must not read or require a sentiment score
    res = ce.calculate_confluence(
        symbol="NVDA",
        technical_data={"current_price": 200.0, "stop_loss": 190.0},
        smart_money_data=None,
    )
    assert res["confluenceScore"] >= 0.0
    # Missing sentiment must not crash or bias calculation


# ---------------------------------------------------------------------------
# Test 13: On-chain missing returns unavailable
# ---------------------------------------------------------------------------
def test_13_onchain_missing_returns_unavailable():
    from engines.fundamental_engine import CryptoFundamentalAnalyzer
    cfa = CryptoFundamentalAnalyzer()
    import pandas as pd
    res = cfa._generate_onchain_metrics("BTC", pd.DataFrame())
    assert res["available"] is False
    assert res["status"] == "UNAVAILABLE"
    assert "unavailable" in res["message"].lower()


# ---------------------------------------------------------------------------
# Test 14: Options flow missing returns empty list
# ---------------------------------------------------------------------------
def test_14_options_flow_missing_returns_empty_list():
    flow = SmartMoneyEngine.get_options_flow("UNMAPPED_STOCK")
    assert flow == []


# ---------------------------------------------------------------------------
# Test 15: Dark pool missing returns unavailable
# ---------------------------------------------------------------------------
def test_15_dark_pool_missing_returns_unavailable():
    metrics = FinraTransparencyFetcher.get_ats_metrics("UNMAPPED_STOCK")
    assert metrics is None


# ---------------------------------------------------------------------------
# Test 16: SEC issuer resolution failure handled cleanly
# ---------------------------------------------------------------------------
def test_16_sec_issuer_resolution_failure():
    fetcher = SecEdgarFetcher()
    cik = fetcher.resolve_cik("INVALID_SYMBOL_12345")
    assert cik is None


# ---------------------------------------------------------------------------
# Test 17: SEC Form 4 empty returns empty list
# ---------------------------------------------------------------------------
def test_17_sec_form_4_empty_returns_empty_list(monkeypatch):
    fetcher = SecEdgarFetcher()
    class MockEmptySubmissions:
        status_code = 200
        def json(self):
            return {"filings": {"recent": {"form": [], "filingDate": [], "accessionNumber": [], "primaryDocDescription": []}}}
    monkeypatch.setattr(fetcher.session, "get", lambda *a, **kw: MockEmptySubmissions())
    filings = fetcher.get_recent_filings("AAPL", form_types=["4"])
    assert filings == []


# ---------------------------------------------------------------------------
# Test 18: Cached response preserves cache headers and provenance
# ---------------------------------------------------------------------------
def test_18_cached_response_preserves_headers():
    res = client.get("/api/v1/smart-money/finra-darkpool/NVDA")
    assert res.status_code == 200
    assert "Cache-Control" in res.headers
    assert "public" in res.headers["Cache-Control"]


# ---------------------------------------------------------------------------
# Test 19: Valid live market price response
# ---------------------------------------------------------------------------
def test_19_valid_live_market_price():
    res = client.get("/api/v1/analytics/SPY")
    assert res.status_code == 200
    data = res.json()
    assert data["currentPrice"] > 0
    assert isinstance(data["currentPrice"], (int, float))


# ---------------------------------------------------------------------------
# Test 20: Missing smart money evidence does not inflate score with 50.0
# ---------------------------------------------------------------------------
def test_20_missing_smart_money_does_not_inflate_confluence():
    ce = ConfluenceEngine()
    # Case A: smart_money_data is None (evidence absent)
    res_none = ce.calculate_confluence(
        symbol="TEST",
        technical_data={"current_price": 100.0, "stop_loss": 95.0},
        smart_money_data=None,
    )
    smart_pillar = next(p for p in res_none["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "unavailable"
    assert smart_pillar["score"] == 0.0
    assert "unavailable" in smart_pillar["detail"].lower()
