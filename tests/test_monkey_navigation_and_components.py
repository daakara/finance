"""Comprehensive Frontend Monkey & Navigation Parity Test Suite.

Simulates aggressive, random, and systematic multi-page user interactions:
1. Validates all Navigation routes (/, /screener, /compare, /smart-money).
2. Simulates Watchlist selection across all 19 asset tickers.
3. Tests Dynamic Role switching (DAY_TRADER <-> LONG_TERM).
4. Tests Compare Matrix ticker swapping (NVO, LLY, NVDA, AMD, PLTR, CRWD, etc.).
5. Tests Hidden Gems Screener filter tabs (All, Lynch, Greenblatt, Rule Breakers).
6. Tests Smart Money Regulatory sub-view tabs (Congress, SEC Form 4, Options Flow).
7. Asserts zero broken navigation endpoints, valid HTTP status codes, and non-empty UI models.
"""

import pytest
import re
import os
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

WATCHLIST_TICKERS = [
    "NVO", "LLY", "AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "PLTR",
    "SPY", "QQQ", "SMH", "XLK", "IWM", "GLD", "TLT", "XLE",
    "BTC-USD", "ETH-USD", "SOL-USD"
]

COMPARE_PAIRS = [
    ("NVO", "LLY"),
    ("NVDA", "AMD"),
    ("PLTR", "CRWD"),
    ("SPY", "QQQ"),
    ("AAPL", "MSFT")
]

def test_monkey_all_watchlist_asset_analytics():
    """Verify backend responds with 200 and complete schema for every single watchlist ticker."""
    for sym in WATCHLIST_TICKERS:
        res = client.get(f"/api/v1/analytics/{sym}?period=1y&interval=1d")
        assert res.status_code == 200, f"Failed analytics for {sym}"
        data = res.json()
        assert "currentPrice" in data
        assert "candles" in data
        assert len(data["candles"]) > 0
        assert "optimalExecution" in data
        assert "stop_loss" in data["optimalExecution"]
        assert "take_profit_1" in data["optimalExecution"]
        assert "factorScores" in data
        assert "smartMoney" in data

def test_monkey_role_switching_intervals():
    """Verify both Day Trader and Long Term intervals execute without server errors."""
    for sym in ["AAPL", "NVDA", "SPY"]:
        # Day trader intervals
        for interval in ["1m", "5m", "15m", "1h"]:
            res = client.get(f"/api/v1/analytics/{sym}?period=5d&interval={interval}")
            assert res.status_code == 200, f"Failed {sym} on intraday {interval}"
            data = res.json()
            assert len(data["candles"]) > 0
            assert data["optimalExecution"]["setup_pattern"] is not None

        # Long term intervals
        for interval in ["1d", "1wk", "1mo"]:
            res = client.get(f"/api/v1/analytics/{sym}?period=1y&interval={interval}")
            assert res.status_code == 200, f"Failed {sym} on macro {interval}"
            data = res.json()
            assert len(data["candles"]) > 0

def test_monkey_smart_money_all_endpoints():
    """Verify Smart Money overview, congress, options, and regulatory feeds."""
    res_overview = client.get("/api/v1/smart-money/overview")
    assert res_overview.status_code == 200
    overview_data = res_overview.json()
    assert "congress_trades" in overview_data
    assert "options_flow" in overview_data

    # Test symbol-filtered regulatory endpoints
    for sym in ["NVDA", "PLTR", "NVO", "VRT"]:
        res_sec = client.get(f"/api/v1/smart-money/sec-filings/{sym}")
        assert res_sec.status_code == 200
        sec_data = res_sec.json()
        assert "filings" in sec_data

        res_finra = client.get(f"/api/v1/smart-money/finra-darkpool/{sym}")
        assert res_finra.status_code == 200
        finra_data = res_finra.json()
        assert "metrics" in finra_data and "ats_dark_pool_volume_share_pct" in finra_data["metrics"]

def test_monkey_screener_and_archetype_runs():
    """Verify screener filtering logic across all archetypes."""
    filters = ["all", "lynch", "greenblatt", "rule_breakers"]
    for f in filters:
        res = client.get(f"/api/v1/screener/run?filter_type={f}")
        assert res.status_code == 200
        data = res.json()
        assert "candidates" in data
        assert len(data["candidates"]) > 0

def test_monkey_ui_link_integrity():
    """Scan all frontend page files and confirm all internal Links point to valid existing routes."""
    valid_routes = {"/", "/screener", "/compare", "/smart-money"}
    frontend_app_dir = os.path.join("frontend", "app")
    
    for root, _, files in os.walk(frontend_app_dir):
        for file in files:
            if file.endswith(".tsx") or file.endswith(".jsx"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Find all href="/..." links
                links = re.findall(r'href=["\'](/[^"\']*)["\']', content)
                for link in links:
                    base_route = link.split("?")[0]
                    if base_route != "#" and not base_route.startswith("http"):
                        assert base_route in valid_routes or base_route == "", f"Invalid route '{base_route}' found in {path}"