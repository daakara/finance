"""Tests for Day Trader Intraday Features & Position Sizing Suite."""

from unittest.mock import patch, MagicMock
import pytest
from api.routes.analytics import get_asset_analytics, compute_intraday_technicals
import pandas as pd


def test_intraday_interval_support():
    prices = [100 + i * 0.1 for i in range(50)]
    mock_df = pd.DataFrame({
        "Open": prices,
        "High": [p + 0.2 for p in prices],
        "Low": [p - 0.2 for p in prices],
        "Close": prices,
        "Volume": [10000] * 50
    }, index=pd.date_range("2026-01-01", periods=50, freq="5min"))

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df
    mock_ticker.info = {"trailingPE": 25.0}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = get_asset_analytics("AAPL", period="5d", interval="5m")
        assert "candles" in res
        assert len(res["candles"]) > 0
        assert "technicals" in res
        assert "vwap" in res["technicals"]
        assert "rsi_14" in res["technicals"]
        assert "atr_14" in res["technicals"]


def test_compute_intraday_technicals():
    df = pd.DataFrame({
        "High": [102, 104, 103, 105, 106, 107],
        "Low": [98, 100, 99, 102, 103, 104],
        "Close": [100, 103, 101, 104, 105, 106],
        "Volume": [1000, 2000, 1500, 3000, 2500, 4000],
    })
    technicals = compute_intraday_technicals(df)
    assert technicals["vwap"] is not None
    assert technicals["vwap"] > 100
    assert 0 <= technicals["rsi_14"] <= 100
    assert technicals["atr_14"] > 0