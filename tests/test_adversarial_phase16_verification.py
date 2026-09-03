"""Phase 16 Adversarial Live-System & Operational Safety Verification Suite.

Validates that:
1. calculate_piotroski_f_score strictly fails closed on empty or null info (UNKNOWN != FAVORABLE).
2. /api/v1/analytics route does not fabricate high-quality fundamental factor scores on uncataloged/missing info.
3. GET /api/v1/cache/clear is rejected with 405 Method Not Allowed (only POST allowed).
4. Content-Security-Policy in frontend/public/_headers includes query2.finance.yahoo.com.
5. Fallback optimalExecution and form 4 disclosures adhere to strict epistemic safety invariants.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.routes.analytics import calculate_piotroski_f_score

client = TestClient(app)


def test_adversarial_piotroski_empty_info_zero():
    """Verify that calculate_piotroski_f_score returns 0 on empty, None, or all-null info dicts."""
    # 1. Empty dict
    assert calculate_piotroski_f_score({}, {}) == 0

    # 2. All-None fields
    null_info = {
        "returnOnAssets": None,
        "freeCashflow": None,
        "operatingMargins": None,
        "currentRatio": None,
        "debtToEquity": None,
        "grossMargins": None,
        "returnOnEquity": None,
        "revenueGrowth": None,
    }
    assert calculate_piotroski_f_score(null_info, {}) == 0

    # 3. Valid positive signals award points
    positive_info = {
        "returnOnAssets": 0.12,
        "freeCashflow": 25000000,
        "operatingMargins": 0.22,
        "currentRatio": 2.5,
        "debtToEquity": 45,
        "grossMargins": 0.55,
        "returnOnEquity": 0.25,
        "revenueGrowth": 0.30,
    }
    score = calculate_piotroski_f_score(positive_info, {})
    assert score >= 7, f"Expected strong score on verified positive fundamentals, got {score}"


def test_adversarial_analytics_missing_info_fail_closed():
    """Verify that /api/v1/analytics does NOT fabricate 70+ factor scores or 'Strong Buy' on empty info."""
    mock_ticker = MagicMock()
    # Provide sufficient historical price candles
    mock_df = pd.DataFrame({
        "Open": [100.0] * 60,
        "High": [105.0] * 60,
        "Low": [98.0] * 60,
        "Close": [102.0] * 60,
        "Volume": [1000000] * 60,
    }, index=pd.date_range("2026-01-01", periods=60))
    mock_ticker.history.return_value = mock_df
    # Empty fundamental info (e.g. rate-limited by Yahoo Finance)
    mock_ticker.info = {}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        res = client.get("/api/v1/analytics/SYM_TEST_UNKNOWN?period=1y&interval=1d")
        assert res.status_code == 200
        data = res.json()

        factor_scores = data.get("factorScores", {})
        assert factor_scores.get("piotroskiFScore") == 0, "Piotroski must be 0 on missing info"
        assert factor_scores.get("growthScore") is None, "growthScore must be None on missing info"
        assert factor_scores.get("qualityScore") is None, "qualityScore must be None on missing info"
        assert factor_scores.get("valuationScore") is None, "valuationScore must be None on missing info"
        assert factor_scores.get("compositeFactorScore") is None, "compositeFactorScore must be None on missing info"
        assert "Awaiting" in factor_scores.get("verdict", ""), "Verdict must reflect unverified status"


def test_adversarial_cache_clear_get_rejected():
    """Verify that GET /api/v1/cache/clear returns 405 Method Not Allowed (destructive ops require POST)."""
    res = client.get("/api/v1/cache/clear")
    assert res.status_code == 405, f"Expected 405 Method Not Allowed for GET /cache/clear, got {res.status_code}"


def test_adversarial_csp_header_contains_query2():
    """Verify that frontend/public/_headers includes query2.finance.yahoo.com in CSP connect-src."""
    headers_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "_headers")
    assert os.path.exists(headers_path), "_headers file must exist in frontend/public"
    with open(headers_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "https://query2.finance.yahoo.com" in content, (
        "CSP connect-src must include https://query2.finance.yahoo.com for client failover"
    )


def test_adversarial_institutional_feeds_empty_on_unknown():
    """Verify that frontend/lib/institutionalFeeds.ts does not leak global Form 4 trades on unknown tickers."""
    feeds_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "lib", "institutionalFeeds.ts")
    assert os.path.exists(feeds_path), "institutionalFeeds.ts must exist"
    with open(feeds_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ensure the fallback to all trades on unmatched ticker has been removed
    assert "matched.length > 0 ? matched : LIVE_SEC_EDGAR_FORM4_TRADES" not in content, (
        "fetchSecForm4Insiders must not fall back to global trade list when ticker has 0 matches"
    )


def test_adversarial_fallback_analytics_non_actionable():
    """Verify that generateFallbackAnalytics sets risk_reward_ratio to 0 and setup_pattern to incomplete."""
    api_ts_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "lib", "api.ts")
    assert os.path.exists(api_ts_path), "api.ts must exist"
    with open(api_ts_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "risk_reward_ratio: 0," in content, "Fallback optimalExecution must set risk_reward_ratio: 0"
    assert "Trend Evidence Incomplete" in content, "Fallback optimalExecution must tag setup as incomplete"


def test_adversarial_error_boundaries_exist():
    """Verify that React Error Boundaries error.tsx and global-error.tsx exist in frontend/app."""
    error_tsx = os.path.join(os.path.dirname(__file__), "..", "frontend", "app", "error.tsx")
    global_error_tsx = os.path.join(os.path.dirname(__file__), "..", "frontend", "app", "global-error.tsx")
    assert os.path.exists(error_tsx), "frontend/app/error.tsx must exist"
    assert os.path.exists(global_error_tsx), "frontend/app/global-error.tsx must exist"
