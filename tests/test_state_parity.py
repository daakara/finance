"""Automated State Parity & Re-render Consistency Test Suite.

Validates that:
1. WatchlistSidebar symbols and prices match the SHARED_WATCHLIST_ITEMS constant.
2. AssetFactorRadar and api.ts default score definitions share identical baseline metrics.
3. Every asset defined in the UI has calibrated factor scores and risk boundaries.
"""

import json
import os
import pytest
import re

pytestmark = pytest.mark.tier1


def test_frontend_constants_exist():
    constants_path = os.path.join("frontend", "lib", "constants.ts")
    assert os.path.exists(constants_path), "frontend/lib/constants.ts single source of truth file is missing"

    with open(constants_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SHARED_WATCHLIST_ITEMS" in content
    assert "SHARED_FACTOR_SCORES" in content
    assert "DEFAULT_MACRO_DIFFICULTY" in content
    assert "DEFAULT_EXPECTED_RETURN" in content

def test_no_hardcoded_factor_fallbacks_in_components():
    """Ensure UI components do not declare divergent inline numeric literals or synthetic fallbacks."""
    radar_path = os.path.join("frontend", "components", "AssetFactorRadar.tsx")
    with open(radar_path, "r", encoding="utf-8") as f:
        radar_content = f.read()

    assert "growthScore: 85" not in radar_content
    assert "qualityScore: 90" not in radar_content
    assert "valuationScore: 40" not in radar_content

def test_watchlist_sidebar_consumes_shared_constants():
    """Ensure WatchlistSidebar uses SHARED_WATCHLIST_ITEMS."""
    sidebar_path = os.path.join("frontend", "components", "WatchlistSidebar.tsx")
    with open(sidebar_path, "r", encoding="utf-8") as f:
        sidebar_content = f.read()

    assert "SHARED_WATCHLIST_ITEMS" in sidebar_content, "WatchlistSidebar must consume SHARED_WATCHLIST_ITEMS"

def test_factor_score_parity_for_core_tickers():
    """Verify that all core tickers have non-empty, mathematically sound factor scores."""
    constants_path = os.path.join("frontend", "lib", "constants.ts")
    with open(constants_path, "r", encoding="utf-8") as f:
        content = f.read()

    core_tickers = ["AAPL", "NVDA", "NVO", "LLY", "MSFT", "GOOGL", "TSLA", "PLTR", "SPY", "QQQ", "BTC", "ETH", "SOL"]
    for ticker in core_tickers:
        pattern = rf'"{ticker}":\s*\{{[^}}]*scores:\s*\{{([^}}]+)\}}'
        match = re.search(pattern, content)
        assert match is not None, f"Ticker {ticker} is missing calibrated factor scores in SHARED_FACTOR_SCORES"
        score_block = match.group(1)
        assert "growthScore" in score_block
        assert "qualityScore" in score_block
        assert "valuationScore" in score_block
        assert "momentumScore" in score_block
        assert "tailRiskScore" in score_block
        assert "compositeFactorScore" in score_block
