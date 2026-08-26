"""Comprehensive Monkey and Component Integrity Test Suite across all UI pathways."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SYMBOLS_TO_TEST = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "PLTR",
    "SPY", "QQQ", "SMH", "XLK", "IWM", "GLD", "TLT", "XLE",
    "BTC-USD", "ETH-USD", "SOL-USD"
]

def test_api_analytics_endpoint_every_asset():
    """Verify that every single asset in the watchlist returns valid 200 HTTP data with non-empty candles."""
    for symbol in SYMBOLS_TO_TEST:
        # Test daily mode
        res_daily = client.get(f"/api/v1/analytics/{symbol}?period=1y&interval=1d")
        assert res_daily.status_code == 200, f"Failed daily analytics for {symbol}"
        data_d = res_daily.json()
        assert data_d["symbol"] == symbol.upper()
        assert len(data_d["candles"]) > 0, f"Empty candles for {symbol} on 1d"
        assert "factorScores" in data_d
        assert "expectedReturn" in data_d

        # Test intraday mode
        res_intra = client.get(f"/api/v1/analytics/{symbol}?period=5d&interval=5m")
        assert res_intra.status_code == 200, f"Failed intraday analytics for {symbol}"
        data_i = res_intra.json()
        assert len(data_i["candles"]) > 0, f"Empty candles for {symbol} on 5m"
        assert "technicals" in data_i
        assert data_i["technicals"]["rsi_14"] is not None

def test_screener_run_and_archetype_filters():
    """Verify that the screener endpoint works for all 4 filter archetypes."""
    filters = ["all", "lynch", "greenblatt", "rule_breakers"]
    for f in filters:
        res = client.get(f"/api/v1/screener/run?filter_type={f}")
        assert res.status_code == 200, f"Screener failed for filter {f}"
        data = res.json()
        assert "candidates" in data
        assert len(data["candidates"]) > 0
        candidate = data["candidates"][0]
        assert "symbol" in candidate
        assert "companyName" in candidate
        assert "gemScore" in candidate
        assert "expertArchetype" in candidate

def test_macro_regimes_endpoint():
    """Verify macro regimes endpoint returns valid regime ratings."""
    res = client.get("/api/v1/regimes/current")
    assert res.status_code == 200
    data = res.json()
    assert "regime" in data

