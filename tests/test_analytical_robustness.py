"""ARX Analytical Robustness & Adversarial Data Suite (Phase 20B).

Adversarially tests edge-case data anomalies to prevent "garbage in -> plausible output":
  1. Zero volume / halted trading candles.
  2. Extreme artificial price spikes (> 500% intraday gap).
  3. Non-monotonic / duplicate timestamps.
  4. Missing or NaN fundamental ratios (P/E, margins).
  5. Stale fundamentals combined with fresh tape.
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
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState

client = TestClient(app)
db = MarketDatabaseEngine(db_path=DB_PATH)


def test_zero_volume_bars_technicals():
    """Candles with 0 volume (halted trading or illiquid periods) do not cause ZeroDivisionError in VWAP."""
    dates = pd.date_range("2026-06-01", periods=30, freq="D")
    zero_vol_df = pd.DataFrame({
        "Open": [50.0 + i * 0.1 for i in range(30)],
        "High": [50.0 + i * 0.1 + 0.5 for i in range(30)],
        "Low": [50.0 + i * 0.1 - 0.5 for i in range(30)],
        "Close": [50.0 + i * 0.1 + 0.1 for i in range(30)],
        "Volume": [0] * 30,  # Zero volume throughout
    }, index=dates)

    from api.routes.analytics import compute_intraday_technicals
    technicals = compute_intraday_technicals(zero_vol_df)

    # VWAP should be None when cumulative volume is 0, never throw ZeroDivisionError
    assert technicals["vwap"] is None
    assert technicals["rsi_14"] is not None
    assert technicals["ema_20"] is not None


def test_extreme_price_spike_robustness():
    """An extreme single-day 700% price spike does not crash execution engine or produce NaN levels."""
    dates = pd.date_range("2026-05-01", periods=60, freq="D")
    spike_df = pd.DataFrame({
        "Open": [10.0] * 59 + [80.0],  # 8x spike on final day
        "High": [10.5] * 59 + [85.0],
        "Low": [9.5] * 59 + [75.0],
        "Close": [10.0] * 59 + [80.0],
        "Volume": [100_000] * 60,
    }, index=dates)

    levels = OptimalExecutionEngine.calculate_trade_levels(
        price_df=spike_df,
        current_price=80.0,
        user_role="LONG_TERM",
    )

    # Must return valid structure without NaN
    assert levels is not None
    assert "execution_status" in levels
    if levels["stop_loss"] is not None:
        assert levels["stop_loss"] < 80.0
        assert not np.isnan(levels["stop_loss"])


def test_stale_fundamentals_with_fresh_tape_fails_closed():
    """When price tape is live but fundamental filings are absent, DecisionState must be EVIDENCE_INCOMPLETE."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TST_FRESH_PRICE_NO_FUND",
        current_price=150.0,
        candle_count=120,
        freshness_status="LIVE",
        has_fundamentals=False,  # Financial statements missing/uncataloged
        confluence_score=85.0,   # Even with strong technicals
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=3.5,
    )

    # Epistemic safeguard: Can never be ACTIONABLE_SETUP without verified fundamentals
    assert state["state"] == DecisionState.EVIDENCE_INCOMPLETE.value
    assert state["isActionable"] is False
    assert state["canSizeTrade"] is False
    assert "Fundamentals Missing" in state["label"]


def test_negative_margins_and_nan_ratios_safe_scoring():
    """Piotroski-F calculation handles negative margins, negative ROE, and missing ratios without crashing."""
    from api.routes.analytics import calculate_piotroski_f_score

    garbage_info = {
        "operatingMargins": -0.85,
        "profitMargins": -1.2,
        "currentRatio": 0.35,
        "debtToEquity": 850.0,
        "grossMargins": -0.15,
        "returnOnEquity": -0.45,
        "revenueGrowth": -0.30,
        "trailingPE": None,
    }

    score = calculate_piotroski_f_score(garbage_info, {})
    assert isinstance(score, int)
    # Distressed balance sheet must score 0
    assert score == 0


def test_duplicate_candles_sqlite_idempotency():
    """Saving duplicate daily candle records for the same date does not corrupt the store or cause duplicate rows."""
    test_sym = "TST_DEDUP"
    records = [
        {"trade_date": "2026-08-01", "open": 25.0, "high": 26.0, "low": 24.5, "close": 25.5, "volume": 10000},
        {"trade_date": "2026-08-01", "open": 25.0, "high": 26.0, "low": 24.5, "close": 25.5, "volume": 10000},
    ]

    db.save_daily_candles(test_sym, records)
    candles = db.get_daily_candles(test_sym, limit=10)

    # SQLite PRIMARY KEY (symbol, trade_date) must strictly deduplicate
    matching_date = [c for c in candles if c["time"] == "2026-08-01"]
    assert len(matching_date) == 1
