"""Tests for ARX Cross-Engine Coherence and Lifecycle Harmony (Sprint 4).

Validates:
1. STOPPED_OUT semantic alignment: Prohibits 'awaiting pullback' guidance for stopped-out assets.
2. OHLCV column-name normalization: Case-insensitive entry boundary handling in OptimalExecutionEngine.
3. Factor score verdict language: Prevents legacy 'Strong Buy / Core Accumulation' advisory leakage.
4. Reward-to-Risk lifecycle stage separation: Verifies TP1 tactical boundary (1.85:1) vs Actionability hurdle (2.0:1).
"""

import math
import pytest
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine


@pytest.fixture
def synthetic_equity_df():
    """Deterministic 70-session equity DataFrame with standard Title Case columns."""
    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    base_price = 100.0
    prices = [base_price * (1.0 + 0.005 * i) for i in range(70)]
    
    return pd.DataFrame({
        "Open": [p * 0.99 for p in prices],
        "High": [p * 1.02 for p in prices],
        "Low": [p * 0.98 for p in prices],
        "Close": prices,
        "Volume": [1_000_000 for _ in range(70)],
    }, index=dates)


def test_reproduce_column_name_normalization_case_insensitivity(synthetic_equity_df):
    """Verify that OptimalExecutionEngine gracefully handles lowercase column names."""
    df_lower = synthetic_equity_df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    
    # Should not raise KeyError: 'Close'
    plan = OptimalExecutionEngine.calculate_trade_levels(
        price_df=df_lower,
        current_price=float(df_lower["close"].iloc[-1]),
        user_role="SWING",
    )
    
    assert plan is not None
    assert plan["stop_loss"] is not None
    assert plan["take_profit_1"] is not None
    assert plan["risk_reward_ratio"] is not None


def test_reproduce_stopped_out_semantic_disqualification():
    """Verify that a STOPPED_OUT asset does NOT instruct the user to 'await a pullback to buy zone'."""
    # When price is below stop floor, DecisionHierarchyEngine should explain stop-loss breach
    dec_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=115.00,
        candle_count=70,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=80.0,
        stage_phase=2,
        is_in_buy_zone=False,
        risk_reward_ratio=2.35,
        is_cataloged=True,
        is_confirmed=False,
        execution_status="STOPPED_OUT",
    )
    
    assert dec_state["state"] == DecisionState.VALID_SETUP.value
    assert dec_state["isActionable"] is False
    reason = dec_state["disqualificationReason"]
    assert "awaiting pullback to buy zone" not in reason.lower()
    assert "stop-loss" in reason.lower() or "invalidated" in reason.lower()
    assert "Stopped Out" in dec_state["label"] or "Invalidated" in dec_state["label"]


def test_rr_threshold_lifecycle_stages_demarcation():
    """Verify the functional demarcation between 1.85:1 (tactical TP1 geometry) and 2.0:1 (actionability hurdle)."""
    # Case A: Setup with RR = 1.90:1 (meets tactical 1.85 floor, but fails institutional 2.0 hurdle)
    dec_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="MSFT",
        current_price=125.00,
        candle_count=70,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=1.90,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
    )
    
    # Must NOT be actionable because RR < 2.0
    assert dec_state["isActionable"] is False
    assert "2.0:1" in dec_state["disqualificationReason"]
    assert "Risk/Reward" in dec_state["disqualificationReason"]
    
    # Case B: Setup with RR = 2.10:1 (satisfies both tactical and institutional hurdles)
    dec_state_actionable = DecisionHierarchyEngine.resolve_decision_state(
        symbol="MSFT",
        current_price=125.00,
        candle_count=70,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase=2,
        is_in_buy_zone=True,
        risk_reward_ratio=2.10,
        is_cataloged=True,
        is_confirmed=True,
        execution_status="IN_BUY_ZONE",
    )
    
    assert dec_state_actionable["isActionable"] is True
    assert dec_state_actionable["state"] == DecisionState.ACTIONABLE_SETUP.value
    assert dec_state_actionable["disqualificationReason"] is None


def test_factor_scores_verdict_eliminates_legacy_phrase():
    """Verify that factorScores verdict reflects fundamental profile rather than trading orders."""
    import pathlib
    analytics_path = pathlib.Path(__file__).parent.parent / "api" / "routes" / "analytics.py"
    content = analytics_path.read_text(encoding="utf-8")
    assert "Strong Buy / Core Accumulation" not in content
    assert "High Quality Profile" in content
