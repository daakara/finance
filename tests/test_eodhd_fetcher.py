"""Tests for EODHD Market Data Fetcher & Failover Pipeline."""

from unittest.mock import patch, MagicMock
from analyst_dashboard.data.eodhd_fetcher import EODHDMarketFetcher
import pandas as pd

def test_eodhd_fetcher_realtime_quote():
    fetcher = EODHDMarketFetcher()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"code": "AAPL.US", "close": 310.50, "volume": 50000000}

    with patch("requests.get", return_value=mock_resp):
        quote = fetcher.fetch_realtime_quote("AAPL")
        assert quote is not None
        assert "close" in quote
        assert quote["close"] == 310.50

def test_eodhd_fetcher_historical_candles():
    fetcher = EODHDMarketFetcher()
    mock_rows = [
        {"date": f"2026-01-{i+1:02d}", "open": 300+i, "high": 305+i, "low": 298+i, "close": 302+i, "volume": 1000000}
        for i in range(60)
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_rows

    with patch("requests.get", return_value=mock_resp):
        df = fetcher.fetch_historical_candles("AAPL")
        assert df is not None
        assert not df.empty
        assert "Close" in df.columns
        assert len(df) == 60