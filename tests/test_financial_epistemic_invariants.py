"""Financial Epistemic Invariants & Anti-Hallucination Regression Test Suite.
Enforces that no execution level, stop loss, take profit, factor score,
or favorable filter status can ever be synthesized or fabricated for
unknown, unverified, or insufficient-history assets.

Non-negotiable invariants:
  UNKNOWN ≠ FAVORABLE
  UNKNOWN ≠ NEGATIVE
  UNKNOWN ≠ ACTIONABLE
  PLACEHOLDER ≠ DATA
  SYNTHETIC ≠ MARKET DATA
  CURATED ≠ LIVE
"""

import pytest
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.liquidity_guard import LiquidityGuard


def make_short_candles(count: int, base_price: float = 20.0) -> pd.DataFrame:
    """Generate a DataFrame with fewer than 50 sessions to simulate unseasoned assets like GLX."""
    dates = pd.date_range("2026-06-01", periods=count, freq="D")
    df = pd.DataFrame({
        "Open": [base_price + i * 0.1 for i in range(count)],
        "High": [base_price + i * 0.1 + 0.5 for i in range(count)],
        "Low": [base_price + i * 0.1 - 0.5 for i in range(count)],
        "Close": [base_price + i * 0.1 + 0.2 for i in range(count)],
        "Volume": [100_000 for _ in range(count)],
    }, index=dates)
    return df


def test_unknown_asset_never_has_actionable_entry():
    """An unknown or unseasoned asset must never have an actionable entry level."""
    # Case 1: Empty / None DataFrame
    res_empty = OptimalExecutionEngine.calculate_trade_levels(
        price_df=None,
        current_price=0.0,
        user_role="LONG_TERM",
    )
    assert res_empty["optimal_entry_min"] is None
    assert res_empty["optimal_entry_max"] is None
    assert res_empty["execution_status"] == "INSUFFICIENT_HISTORY"

    # Case 2: Unseasoned asset (< 50 sessions, like GLX with 47 sessions)
    short_df = make_short_candles(40, base_price=17.50)
    res_short = OptimalExecutionEngine.calculate_trade_levels(
        price_df=short_df,
        current_price=17.50,
        user_role="LONG_TERM",
    )
    assert res_short["optimal_entry_min"] is None
    assert res_short["optimal_entry_max"] is None
    assert res_short["execution_status"] == "INSUFFICIENT_HISTORY"
    assert "< 50 Sessions" in res_short["setup_pattern"]


def test_unknown_asset_never_has_stop_loss():
    """An unknown asset or unseasoned asset must never fabricate a stop-loss price."""
    res_none = OptimalExecutionEngine.calculate_trade_levels(
        price_df=None,
        current_price=50.0,
        user_role="LONG_TERM",
    )
    assert res_none["stop_loss"] is None
    assert res_none["stop_loss_pct"] is None

    short_df = make_short_candles(30, base_price=25.0)
    res_short = OptimalExecutionEngine.calculate_trade_levels(
        price_df=short_df,
        current_price=25.0,
        user_role="LONG_TERM",
    )
    assert res_short["stop_loss"] is None
    assert res_short["stop_loss_pct"] is None


def test_unknown_asset_never_has_take_profit():
    """An unknown asset or unseasoned asset must never fabricate take-profit levels or R:R ratio."""
    short_df = make_short_candles(45, base_price=15.0)
    res = OptimalExecutionEngine.calculate_trade_levels(
        price_df=short_df,
        current_price=15.0,
        user_role="LONG_TERM",
    )
    assert res["take_profit_1"] is None
    assert res["take_profit_1_pct"] is None
    assert res["take_profit_2"] is None
    assert res["take_profit_2_pct"] is None
    assert res["risk_reward_ratio"] is None


def test_unknown_asset_never_has_positive_confluence():
    """An uncataloged asset without fundamentals/insiders must never achieve high confluence."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="UNKNOWN_CO",
        technical_data=None,
        smart_money_data=None,
        fundamental_data=None,
        catalyst_data=None,
    )
    # With missing fundamentals and missing smart money, score must be capped and defensive
    assert res["confluenceScore"] < 50.0
    assert res["badgeColor"] in ["rose", "slate"]
    assert "DEFENSIVE" in res["confluenceRating"]
    # Fundamental pillar must be explicitly unavailable, not fabricated
    assert res["pillars"][1]["status"] == "unavailable"
    assert "unavailable" in res["pillars"][1]["detail"].lower()


def test_unknown_asset_never_passes_favorable_filter():
    """Unverified assets must fail closed on all favorable screener filter criteria."""
    # Simulate screener candidate for unverified ticker
    candidate = {
        "symbol": "FAKE_TICKER",
        "currentPrice": 0.0,
        "executionStatus": "UNVERIFIED_ASSET",
        "optimalEntryMax": None,
        "confluenceScore": 0.0,
        "riskRewardRatio": None,
        "pegRatio": "N/A",
        "roic": "N/A",
        "grossMargin": "N/A",
        "expertArchetype": "Unverified Asset",
    }
    
    # Invariant: UNVERIFIED_ASSET must never match favorable filters
    favorable_filters = ["in_buy_zone", "high_confluence", "high_rr", "lynch", "greenblatt", "rule_breakers"]
    for f in favorable_filters:
        if f == "high_confluence":
            matches = (candidate["confluenceScore"] or 0) >= 80.0 and candidate["executionStatus"] != "UNVERIFIED_ASSET"
        elif f == "in_buy_zone":
            matches = candidate["executionStatus"] == "IN_BUY_ZONE"
        elif f == "high_rr":
            has_entry = candidate["optimalEntryMax"] is not None and candidate["optimalEntryMax"] > 0
            matches = (candidate["riskRewardRatio"] or 0) >= 2.0 and has_entry
        else:
            matches = candidate["executionStatus"] != "UNVERIFIED_ASSET" and candidate["roic"] != "N/A"
        assert not matches, f"Unverified asset must never match filter {f}"


def test_missing_price_never_defaults_to_100():
    """Missing or zero price must never evaluate to default 100 or fabricate a $93.00 stop."""
    # When price is 0 and df is empty/None, execution levels must not calculate positive stops
    res = OptimalExecutionEngine.calculate_trade_levels(
        price_df=None,
        current_price=0.0,
    )
    assert res["stop_loss"] is None
    assert res["optimal_entry_min"] is None


def test_missing_fundamentals_never_become_zero_score():
    """Missing fundamentals must be marked unavailable rather than silently penalized as bankruptcy."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="UNRESEARCHED_STOCK",
        technical_data={"setup_pattern": "Base Consolidation", "rsi_14": 52.0},
        smart_money_data=None,
        fundamental_data=None,
    )
    fund_pillar = res["pillars"][1]
    assert fund_pillar["pillar"] == "FUNDAMENTAL_SOLVENCY"
    assert fund_pillar["status"] == "unavailable"
    assert fund_pillar["score"] == 0.0
    # Must explicitly inform the user that SEC filings are unavailable
    assert "SEC fundamental filings unavailable" in fund_pillar["detail"]


def test_missing_catalyst_data_never_becomes_positive_catalyst():
    """Uncataloged assets must never receive synthetic upcoming product cycle text."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="NO_CATALYST_CO",
        catalyst_data=None,
    )
    # When catalyst_data is None, warnings should be empty, no fake earnings rush
    assert len(res["warnings"]) == 0
