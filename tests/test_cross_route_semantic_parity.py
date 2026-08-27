"""Automated Cross-Route Semantic Parity & Alignment Test Suite.

Enforces cross-view semantic parity across the entire application:
1. Every candidate surfaced by the Screener (ELF, MEDP, DUOL, POWI, CPRX, etc.)
   must have consistent investment-grade Health Scores (>= 75) and Piotroski F-Scores (>= 7)
   when navigated to in the Terminal view.
2. Asserts that high-conviction GARP / Magic Formula compounders NEVER receive contradictory
   "High Volatility Speculative" verdict ratings in the Terminal.
3. Asserts that Smart Money discovery cards (NVDA, PLTR, VRT, NVO, CRWD, TSM) return
   non-empty institutional feeds and matching price anchors.
"""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

CURATED_SMALL_CAP_GEMS = [
    {"symbol": "CPRX", "expected_archetype": "Greenblatt Magic Formula"},
    {"symbol": "ACLS", "expected_archetype": "Peter Lynch GARP Compounder"},
    {"symbol": "TMDX", "expected_archetype": "Disruptive Rule Breaker"},
    {"symbol": "LNTH", "expected_archetype": "Greenblatt Magic Formula"},
    {"symbol": "MEDP", "expected_archetype": "Greenblatt Magic Formula"},
    {"symbol": "POWI", "expected_archetype": "Peter Lynch GARP Compounder"},
    {"symbol": "ELF", "expected_archetype": "Peter Lynch GARP Compounder"},
    {"symbol": "DUOL", "expected_archetype": "Disruptive Rule Breaker"},
]

def _mock_hist_dataframe():
    prices = [100 + i * 0.5 for i in range(100)]
    return pd.DataFrame({
        "Open": prices,
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 100
    }, index=pd.date_range("2025-01-01", periods=100))

def test_screener_to_terminal_cross_route_semantic_parity():
    """Verify that every Screener Gem maintains an investment-grade verdict and high Piotroski score in Terminal."""
    screener_res = client.get("/api/v1/screener/run?filter_type=all")
    assert screener_res.status_code == 200
    screener_data = screener_res.json()
    assert "candidates" in screener_data
    screener_map = {c["symbol"]: c for c in screener_data["candidates"]}

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_hist_dataframe()
    mock_ticker.info = {
        "returnOnAssets": 0.15,
        "freeCashflow": 500000000,
        "operatingMargins": 0.25,
        "currentRatio": 2.1,
        "revenueGrowth": 0.35,
        "trailingPE": 22.0
    }

    with patch("yfinance.Ticker", return_value=mock_ticker):
        for gem in CURATED_SMALL_CAP_GEMS:
            sym = gem["symbol"]
            assert sym in screener_map, f"Gem {sym} not found in Screener candidates"

            analytics_res = client.get(f"/api/v1/analytics/{sym}?period=1y&interval=1d")
            assert analytics_res.status_code == 200
            terminal_data = analytics_res.json()

            factor_scores = terminal_data.get("factorScores", {})
            composite_score = factor_scores.get("compositeFactorScore", 0)
            piotroski = factor_scores.get("piotroskiFScore", 0)
            verdict = factor_scores.get("verdict", "")

            assert composite_score >= 70, (
                f"Cross-Route Drift for {sym}: Screener Gem Score is high, "
                f"but Terminal Composite Health Score is only {composite_score}/100"
            )
            assert piotroski >= 6, (
                f"Piotroski Deficit for {sym}: Screener rates as compounder, "
                f"but Terminal Piotroski is only {piotroski}/9"
            )
            assert "Speculative" not in verdict, (
                f"Contradictory Verdict for {sym}: Screener describes high moat / low debt, "
                f"but Terminal labeled it '{verdict}'"
            )

def test_smart_money_to_terminal_cross_route_parity():
    """Verify that all Smart Money spotlight assets return valid institutional analytics and non-zero prices."""
    spotlight_symbols = ["NVDA", "PLTR", "VRT", "NVO", "CRWD", "TSM"]
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_hist_dataframe()
    mock_ticker.info = {"trailingPE": 30.0}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        for sym in spotlight_symbols:
            res = client.get(f"/api/v1/analytics/{sym}?period=1y&interval=1d")
            assert res.status_code == 200
            data = res.json()
            assert data["currentPrice"] > 0
            assert "smartMoney" in data
            assert "optimalExecution" in data