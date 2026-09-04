"""Frontend / Backend API Contract Test Suite (Phase 19C).
Enforces non-negotiable data contracts between backend analytical engines
and frontend rendering components.

Guarantees:
  1. Null Preservation: Backend None serializes as JSON null and is NEVER coerced to 0, 0.0, or synthetic levels.
  2. Score Bounding: All scores remain strictly bounded in their canonical ranges.
  3. Freshness Governance: The freshness block is consistently emitted with valid enum status.
  4. Epistemic Invariants: UNKNOWN ≠ ACTIONABLE is maintained across serialization boundaries.
"""

import json
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.data.market_db import MarketDatabaseEngine, DB_PATH

client = TestClient(app)
db = MarketDatabaseEngine(db_path=DB_PATH)


def _seed_contract_candles(symbol: str, count: int, base_price: float = 30.0):
    """Seed candles in SQLite for contract validation."""
    start_date = datetime.utcnow() - timedelta(days=count + 1)
    records = []
    for i in range(count):
        d_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        records.append({
            "trade_date": d_str,
            "open": round(base_price + i * 0.1, 2),
            "high": round(base_price + i * 0.1 + 0.5, 2),
            "low": round(base_price + i * 0.1 - 0.5, 2),
            "close": round(base_price + i * 0.1 + 0.1, 2),
            "volume": 150_000,
        })
    db.save_daily_candles(symbol, records)


def test_contract_null_level_preservation():
    """Unseasoned asset (< 50 sessions) must serialize None as JSON null, never 0, 0.0, or synthetic levels."""
    test_sym = "CNT_NULL"
    _seed_contract_candles(test_sym, count=25, base_price=20.0)

    empty_df = pd.DataFrame()
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {"longName": "Contract Null Test", "sector": "Technology"}
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200

        # Verify raw JSON text serialization
        raw_text = resp.text
        parsed = json.loads(raw_text)

        opt_exec = parsed["optimalExecution"]
        assert opt_exec["execution_status"] == "INSUFFICIENT_HISTORY"
        assert opt_exec["optimal_entry_min"] is None
        assert opt_exec["optimal_entry_max"] is None
        assert opt_exec["stop_loss"] is None
        assert opt_exec["take_profit_1"] is None

        # Guarantee literal null in raw JSON payload for these fields
        assert '"optimal_entry_min": null' in raw_text or '"optimal_entry_min":null' in raw_text
        assert '"stop_loss": null' in raw_text or '"stop_loss":null' in raw_text
        assert '"take_profit_1": null' in raw_text or '"take_profit_1":null' in raw_text


def test_contract_score_bounding_and_schema():
    """All analytics scores must strictly respect canonical ranges and never emit NaN or uncompressed outliers."""
    test_sym = "CNT_SCORES"
    _seed_contract_candles(test_sym, count=60, base_price=55.0)

    empty_df = pd.DataFrame()
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {
            "longName": "Score Test Corp",
            "sector": "Healthcare",
            "operatingMargins": 0.22,
            "currentRatio": 1.8,
            "debtToEquity": 60,
            "grossMargins": 0.45,
            "returnOnEquity": 0.18,
            "revenueGrowth": 0.12,
        }
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200
        data = resp.json()

        # 1. Confluence Score Bounding
        confluence = data["confluence"]
        c_score = confluence["confluenceScore"]
        assert 0.0 <= c_score <= 100.0, f"confluenceScore {c_score} out of bounds"
        assert not pd.isna(c_score)

        # 2. Pillar Scores
        assert len(confluence["pillars"]) == 4
        for p_dict in confluence["pillars"]:
            assert "score" in p_dict
            p_score = p_dict["score"]
            assert 0.0 <= p_score <= 100.0, f"Pillar {p_dict.get('pillar')} score {p_score} out of bounds"

        # 3. Factor Scores
        factors = data["factorScores"]
        for key in ["growthScore", "qualityScore", "valuationScore", "momentumScore", "tailRiskScore"]:
            val = factors[key]
            if val is not None:
                assert 0 <= val <= 100, f"Factor {key} value {val} out of bounds"

        # 4. Piotroski F-Score (0-9 integer)
        if factors.get("piotroskiFScore") is not None:
            assert 0 <= factors["piotroskiFScore"] <= 9

        # 5. Confluence Metadata Bounding
        assert data["confluence"]["badgeColor"] in ["emerald", "cyan", "amber", "rose"]
        assert 0 <= data["confluence"]["positivesCount"] <= 4
        assert 0 <= data["confluence"]["warningsCount"] <= 4


def test_contract_freshness_metadata_schema():
    """The freshness block must always be present and contain valid schema keys."""
    test_sym = "CNT_FRESH"
    _seed_contract_candles(test_sym, count=30, base_price=70.0)

    empty_df = pd.DataFrame()
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("analyst_dashboard.data.eodhd_fetcher.EODHDMarketFetcher.fetch_historical_candles", return_value=None):
        mock_instance = MagicMock()
        mock_instance.history.return_value = empty_df
        mock_instance.info = {"longName": "Fresh Test", "sector": "Finance"}
        mock_ticker.return_value = mock_instance

        resp = client.get(f"/api/v1/analytics/{test_sym}?period=1y&interval=1d")
        assert resp.status_code == 200
        data = resp.json()

        assert "freshness" in data
        freshness = data["freshness"]
        assert freshness["status"] in ["LIVE", "RECENT", "STALE_HISTORICAL", "UNAVAILABLE"]
        assert freshness["providerSource"] in ["yfinance", "yfinance_crypto", "eodhd", "sqlite_cache"]
        assert isinstance(freshness["lastTradeDate"], str)
        assert isinstance(freshness["stalenessDays"], int)
        assert freshness["stalenessDays"] >= 0
        assert isinstance(freshness["candleCount"], int)
        assert freshness["candleCount"] == 30
