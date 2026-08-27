"""Tests for Model Calibration: ETF scoring, ER drift damping, and Crypto Value moats."""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from api.routes.analytics import get_asset_analytics

def _mock_hist_dataframe():
    prices = [100 + i * 0.5 for i in range(100)]
    return pd.DataFrame({
        "Open": prices,
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 100
    }, index=pd.date_range("2025-01-01", periods=100))

def test_etf_quality_calibration():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_hist_dataframe()
    mock_ticker.info = {"trailingPE": 22.0}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_asset_analytics("SPY", "1y")
        assert res["factorScores"]["compositeFactorScore"] >= 68
        assert res["factorScores"]["qualityScore"] >= 80

def test_er_drift_damping():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_hist_dataframe()
    mock_ticker.info = {"trailingPE": 25.0}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_asset_analytics("INTC", "1y")
        assert res["expectedReturn"]["p50Expected"] < 60.0

def test_crypto_buffett_moat_calibration():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_hist_dataframe()
    mock_ticker.info = {}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_asset_analytics("ETH-USD", "1y")
        buffett = next(a for a in res["traderArchetypes"]["archetypes"] if "Warren Buffett" in a["name"])
        assert buffett["alignmentScore"] >= 70
        assert "Tier-1 Network Moat" in buffett["status"]