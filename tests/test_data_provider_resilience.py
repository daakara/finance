"""Data-Provider Resilience & Fail-Closed Test Suite (Phase 19A).
Tests cascading behavior across yfinance, EODHD, and SQLite stores,
ensuring the platform fails closed when data is missing, stale, or malformed,
and never fabricates synthetic candles, prices, or trade levels.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.data.market_db import MarketDatabaseEngine, DB_PATH
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine

client = TestClient(app)
db = MarketDatabaseEngine(db_path=DB_PATH)


def _seed_test_candles(symbol: str, count: int, base_price: float = 50.0, days_ago: int = 1):
    """Seed historical candles in SQLite with specific age and count."""
    start_date = datetime.utcnow() - timedelta(days=days_ago + count)
    records = []
    for i in range(count):
        d_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({
            "trade_date": d_str,
            "open": round(base_price + i * 0.1, 2),
            "high": round(base_price + i * 0.1 + 0.8, 2),
            "low": round(base_price + i * 0.1 - 0.8, 2),
            "close": round(base_price + i * 0.1 + 0.2, 2),
            "volume": 250_000,
        })
    db.save_daily_candles(symbol, records)


def test_provider_cascade_to_sqlite():
    """When Yahoo and EODHD are down, system cleanly falls back to SQLite store."""
    test_sym = "TST_CASCADE"
    _seed_test_candles(test_sym, count=35, base_price=42.0, days_ago=1)

    empty_df = pd.DataFrame()

    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {"longName": "Cascade Test Corp", "sector": "Technology"}
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200
        data = resp.json()

        assert data["symbol"] == test_sym
        assert len(data["candles"]) == 35
        assert data["freshness"]["providerSource"] == "sqlite_cache"
        assert data["freshness"]["candleCount"] == 35
        assert data["freshness"]["status"] in ["LIVE", "RECENT"]


def test_provider_complete_outage_fails_closed():
    """When all providers fail and SQLite has no data, returns HTTP 404 with zero fabricated candles."""
    test_sym = "TST_DEAD"
    empty_df = pd.DataFrame()

    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None), \
         patch("api.routes.analytics.market_db.get_daily_candles", return_value=[]), \
         patch("api.routes.analytics.market_db.get_candles_with_freshness", return_value={"candles": [], "freshness_status": "UNAVAILABLE"}):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data
        assert "No valid price action data found" in data["detail"]


def test_sqlite_stale_data_tagging():
    """When candles in SQLite are older than 5 trading days, tags them as STALE_HISTORICAL."""
    test_sym = "TST_STALE"
    # Seed 30 candles ending 45 days ago
    _seed_test_candles(test_sym, count=30, base_price=100.0, days_ago=45)

    fresh_info = db.get_candles_with_freshness(test_sym, limit=60)
    assert fresh_info["freshness_status"] == "STALE_HISTORICAL"
    assert fresh_info["staleness_days"] >= 40
    assert fresh_info["candle_count"] == 30

    empty_df = pd.DataFrame()
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {"longName": "Stale Asset Inc", "sector": "Energy"}
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["freshness"]["status"] == "STALE_HISTORICAL"
        assert data["freshness"]["stalenessDays"] >= 40
        assert data["freshness"]["providerSource"] == "sqlite_cache"


def test_malformed_payload_rejection():
    """Malformed price series (e.g. constant prices or zero-range bars) are rejected by is_valid_ohlcv."""
    flat_df = pd.DataFrame({
        "Open": [10.0] * 20,
        "High": [10.0] * 20,
        "Low": [10.0] * 20,
        "Close": [10.0] * 20,
        "Volume": [1000] * 20,
    }, index=pd.date_range("2026-01-01", periods=20))

    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None), \
         patch("api.routes.analytics.market_db.get_daily_candles", return_value=[]), \
         patch("api.routes.analytics.market_db.get_candles_with_freshness", return_value={"candles": [], "freshness_status": "UNAVAILABLE"}):
        mock_instance = MagicMock()
        mock_instance.history.return_value = flat_df
        mock_ticker.return_value = mock_instance

        resp = client.get("/api/v1/analytics/TST_FLAT?period=1y&interval=1d")
        # Since flat_df has high_max - low_min < 0.01, is_valid_ohlcv fails and cascades to 404
        assert resp.status_code == 404


def test_unseasoned_asset_fails_closed_in_execution_plan():
    """Unseasoned asset (< 50 bars) served via SQLite has null levels and INSUFFICIENT_HISTORY status."""
    test_sym = "TST_UNSEASONED"
    # Seed 25 candles (fewer than the required 50 sessions for stage analysis)
    _seed_test_candles(test_sym, count=25, base_price=22.0, days_ago=1)

    empty_df = pd.DataFrame()
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {"longName": "Unseasoned IPO", "sector": "Biotech"}
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200
        data = resp.json()

        exec_plan = data["optimalExecution"]
        assert exec_plan["execution_status"] == "INSUFFICIENT_HISTORY"
        assert exec_plan["optimal_entry_min"] is None
        assert exec_plan["optimal_entry_max"] is None
        assert exec_plan["stop_loss"] is None
        assert exec_plan["take_profit_1"] is None
        assert "< 50 Sessions" in exec_plan["setup_pattern"]
