"""Tests for Alpaca Market Data Fetcher & IEX Feed Adapter."""

from unittest.mock import patch, MagicMock
from analyst_dashboard.data.alpaca_fetcher import AlpacaMarketFetcher
import pandas as pd
import pytest

pytestmark = pytest.mark.tier2c


def test_alpaca_fetcher_configuration():
    # Unconfigured
    unconf = AlpacaMarketFetcher(api_key_id="", api_secret_key="")
    assert not unconf.is_configured
    assert unconf.fetch_realtime_quote("AAPL") is None
    assert unconf.fetch_snapshots(["AAPL"]) == {}
    assert unconf.fetch_historical_candles("AAPL") is None

    # Configured
    conf = AlpacaMarketFetcher(api_key_id="PK_TEST", api_secret_key="SK_TEST", feed="iex")
    assert conf.is_configured
    assert conf._headers()["APCA-API-KEY-ID"] == "PK_TEST"
    assert conf._headers()["APCA-API-SECRET-KEY"] == "SK_TEST"


def test_alpaca_fetcher_realtime_quote():
    fetcher = AlpacaMarketFetcher(api_key_id="PK_TEST", api_secret_key="SK_TEST")

    mock_quote_resp = MagicMock()
    mock_quote_resp.status_code = 200
    mock_quote_resp.json.return_value = {
        "quote": {
            "ap": 230.50,
            "as": 100,
            "bp": 230.40,
            "bs": 50,
            "t": "2026-09-22T20:00:00Z"
        }
    }

    mock_trade_resp = MagicMock()
    mock_trade_resp.status_code = 200
    mock_trade_resp.json.return_value = {
        "trade": {
            "p": 230.45,
            "s": 25,
            "t": "2026-09-22T20:00:01Z"
        }
    }

    with patch.object(fetcher.session, "get", side_effect=[mock_quote_resp, mock_trade_resp]):
        quote = fetcher.fetch_realtime_quote("AAPL")
        assert quote is not None
        assert quote["symbol"] == "AAPL"
        assert quote["price"] == 230.45
        assert quote["bid"] == 230.40
        assert quote["ask"] == 230.50
        assert quote["feed"] == "iex"


def test_alpaca_fetcher_snapshots():
    fetcher = AlpacaMarketFetcher(api_key_id="PK_TEST", api_secret_key="SK_TEST")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "AAPL": {
            "latestTrade": {"p": 230.0, "t": "2026-09-22T20:00:00Z"},
            "dailyBar": {"c": 230.0, "h": 232.0, "l": 228.0, "o": 229.0, "v": 45000000},
            "prevDailyBar": {"c": 225.0}
        },
        "NVDA": {
            "latestTrade": {"p": 120.0, "t": "2026-09-22T20:00:00Z"},
            "dailyBar": {"c": 120.0, "h": 122.0, "l": 118.0, "o": 119.0, "v": 60000000},
            "prevDailyBar": {"c": 125.0}
        }
    }

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        snaps = fetcher.fetch_snapshots(["AAPL", "NVDA"])
        assert len(snaps) == 2
        assert snaps["AAPL"]["price"] == 230.0
        assert snaps["AAPL"]["change_pct"] == 2.22  # (230 - 225) / 225 * 100
        assert snaps["NVDA"]["price"] == 120.0
        assert snaps["NVDA"]["change_pct"] == -4.0   # (120 - 125) / 125 * 100


def test_alpaca_fetcher_historical_candles():
    fetcher = AlpacaMarketFetcher(api_key_id="PK_TEST", api_secret_key="SK_TEST")

    dates = pd.date_range("2026-01-01", periods=30, freq="D").strftime("%Y-%m-%dT00:00:00Z")
    mock_bars = [
        {
            "t": dates[i],
            "o": 200.0 + i,
            "h": 205.0 + i,
            "l": 198.0 + i,
            "c": 202.0 + i,
            "v": 1000000 * (i + 1),
            "n": 5000,
            "vw": 201.5 + i
        }
        for i in range(30)
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"bars": mock_bars, "symbol": "AAPL"}

    with patch.object(fetcher.session, "get", return_value=mock_resp):
        df = fetcher.fetch_historical_candles("AAPL", limit=30)
        assert df is not None
        assert not df.empty
        assert len(df) == 30
        assert "Close" in df.columns
        assert "VWAP" in df.columns
        assert df.index.name == "Date"
        assert df.iloc[-1]["Close"] == 231.0
