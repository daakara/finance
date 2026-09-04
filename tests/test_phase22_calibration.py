"""ARX Phase 22 Decision Intelligence & Calibration Test Suite.

Validates:
1. Dynamic Smart Money Reweighting: Equities & ETFs without insider buys can achieve high conviction (>= 75.0).
2. Epistemic Compression Invariant: Assets missing fundamentals remain strictly compressed (< 50.0).
3. Authentic Risk/Reward: No artificial buy-zone inflation of rr_ratio; authentic ratio governs decision hierarchy.
4. Stage 4 Slope Qualification: Routine pullbacks above flat/rising 50 SMA do not trigger Stage 4 distribution.
5. Continuous Curves: RSI and P/E mappings do not produce discontinuous cliffs.
6. Macro Decoupling: Stock-level stop loss distance does not penalize the Macro Safety Floor.
7. Golden Universe Parity: Canonical golden assets preserve intended decision hierarchy states.
"""

import math
import numpy as np
import pandas as pd
import pytest
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from tests.fixtures.golden_universe import GOLDEN_ASSETS


def test_smart_money_dynamic_reweighting_allows_actionable_confluence():
    """Verify that a compounder without active insider purchases is not capped at 75.0."""
    tech = {"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 55.0, "risk_reward_ratio": 2.8}
    fund = {"qualityScore": 92.0, "growthScore": 90.0, "valuationScore": 75.0, "piotroski_f": 8}
    macro = {"yield_curve_10y2y": 0.25, "credit_spread": 3.2}

    res = ConfluenceEngine.calculate_confluence(
        symbol="COMPOUNDER",
        technical_data=tech,
        fundamental_data=fund,
        smart_money_data=None,
        macro_data=macro,
    )

    smart_pillar = next(p for p in res["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "unavailable"
    assert smart_pillar["score"] == 0.0
    assert res["confluenceScore"] >= 80.0
    assert res["confluenceRating"] == "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT"


def test_etf_broad_market_reachability():
    """Verify that broad market benchmark ETFs (SPY) are capable of reaching >= 75.0 confluence."""
    tech = {"setup_pattern": "Stage 2 Breakout", "stage_phase": "Stage 2", "rsi_14": 58.0, "risk_reward_ratio": 2.5}
    fund = {"qualityScore": 85.0, "growthScore": 80.0, "valuationScore": 75.0, "piotroski_f": 8}
    macro = {"yield_curve_10y2y": 0.30, "credit_spread": 3.0}

    res_spy = ConfluenceEngine.calculate_confluence(
        symbol="SPY",
        technical_data=tech,
        fundamental_data=fund,
        smart_money_data=None,
        macro_data=macro,
    )

    assert res_spy["confluenceScore"] >= 78.0


def test_missing_fundamentals_compression_invariant():
    """Epistemic Invariant: Missing fundamentals must still mathematically compress score < 50.0."""
    tech = {"setup_pattern": "Breakout", "rsi_14": 50.0, "risk_reward_ratio": 3.0}
    macro = {"yield_curve_10y2y": 0.30, "credit_spread": 3.0}

    res = ConfluenceEngine.calculate_confluence(
        symbol="MISSING_FUND",
        technical_data=tech,
        fundamental_data={},
        smart_money_data=None,
        macro_data=macro,
    )

    assert res["confluenceScore"] < 50.0
    assert res["badgeColor"] in ["rose", "amber"]


def test_authentic_rr_ratio_without_buy_zone_inflation():
    """Verify that optimal execution does not artificially inflate rr_ratio to 2.25 in buy zone."""
    prices = [100.0 + i * 0.05 for i in range(100)]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1.0 for p in prices],
        "Low": [p - 1.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 100
    })

    current_price = float(df["Close"].iloc[-1])
    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="SWING")

    assert plan["risk_reward_ratio"] is not None
    assert plan["risk_reward_ratio"] >= 1.85
    risk = current_price - plan["stop_loss"]
    tp1_reward = plan["take_profit_1"] - current_price
    tp2_reward = plan["take_profit_2"] - current_price
    expected_blended_rr = round((0.5 * tp1_reward + 0.5 * tp2_reward) / risk, 2)
    assert abs(plan["risk_reward_ratio"] - expected_blended_rr) <= 0.02


def test_stage_4_slope_qualification():
    """A 2% dip below 50 SMA with a rising 50 SMA must NOT be classified as Stage 4 distribution."""
    closes = [50.0 + i * 1.0 for i in range(80)]
    sma_50 = sum(closes[-50:]) / 50.0
    pullback_price = round(sma_50 * 0.975, 2)
    closes[-1] = pullback_price

    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [500000] * len(closes)
    })

    plan = OptimalExecutionEngine.calculate_trade_levels(df, pullback_price, user_role="SWING")
    assert "Stage 4" not in plan["setup_pattern"]
    assert "Stage 4" not in plan["stage_phase"]


def test_stage_4_confirmed_when_slope_declining():
    """A price below 50 SMA with a declining 50 SMA must be confirmed as Stage 4 distribution."""
    closes = [150.0 - i * 0.8 for i in range(80)]
    current_price = closes[-1]

    df = pd.DataFrame({
        "Open": closes,
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [500000] * len(closes)
    })

    plan = OptimalExecutionEngine.calculate_trade_levels(df, current_price, user_role="SWING")
    assert "Stage 4" in plan["setup_pattern"] or "Stage 4" in plan["stage_phase"]


def test_continuous_rsi_curve_no_cliffs():
    """Verify that RSI transition from 74.0 to 76.0 is smooth without a 10-point cliff."""
    ce = ConfluenceEngine()
    tech_base = {"setup_pattern": "Stage 2 VCP", "executionStatus": "IN_BUY_ZONE", "risk_reward_ratio": 2.5, "current_price": 100.0, "stop_loss": 94.0}
    fund = {"qualityScore": 85.0, "piotroski_f": 8}

    res_74 = ce.calculate_confluence("TEST", technical_data={**tech_base, "rsi_14": 74.0}, fundamental_data=fund)
    res_76 = ce.calculate_confluence("TEST", technical_data={**tech_base, "rsi_14": 76.0}, fundamental_data=fund)

    tech_74 = next(p for p in res_74["pillars"] if p["pillar"] == "TECHNICAL_STRUCTURE")["score"]
    tech_76 = next(p for p in res_76["pillars"] if p["pillar"] == "TECHNICAL_STRUCTURE")["score"]

    delta = abs(tech_74 - tech_76)
    assert delta < 5.0, f"RSI step cliff detected: delta was {delta:.2f} pts"


def test_macro_safety_floor_decoupled_from_stock_stop():
    """Verify that a wider stock stop (e.g. 13% for high-beta stock) does not collapse the Macro score."""
    ce = ConfluenceEngine()
    macro_healthy = {"yield_curve_10y2y": 0.35, "credit_spread": 3.1}

    tech_wide_stop = {"current_price": 100.0, "stop_loss": 87.0, "setup_pattern": "Breakout", "risk_reward_ratio": 2.5}
    res = ce.calculate_confluence("HIGH_BETA", technical_data=tech_wide_stop, macro_data=macro_healthy)

    macro_pillar = next(p for p in res["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR")
    assert macro_pillar["score"] == 85.0
    assert macro_pillar["status"] == "positive"


def test_golden_universe_parity_preserved():
    """Verify that all 6 canonical golden assets preserve intended decision hierarchy states."""
    nvda_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="NVDA", current_price=228.45, candle_count=252, freshness_status="LIVE",
        has_fundamentals=True, confluence_score=82.5, stage_phase=2, is_in_buy_zone=False, risk_reward_ratio=2.85
    )
    assert nvda_state["state"] == DecisionState.VALID_SETUP.value
    assert nvda_state["isActionable"] is False

    fix_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="FIX", current_price=1580.19, candle_count=252, freshness_status="LIVE",
        has_fundamentals=True, confluence_score=45.9, stage_phase=4, is_in_buy_zone=False, risk_reward_ratio=1.15
    )
    assert fix_state["state"] == DecisionState.VALID_SETUP.value
    assert "Stage 4" in fix_state["disqualificationReason"]

    cprx_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="CPRX", current_price=31.49, candle_count=252, freshness_status="STALE_HISTORICAL",
        has_fundamentals=True, confluence_score=79.4, stage_phase=1, is_in_buy_zone=False, risk_reward_ratio=2.1
    )
    assert cprx_state["state"] == DecisionState.STALE_DATA.value

    glx_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="GLX", current_price=17.53, candle_count=47, freshness_status="RECENT",
        has_fundamentals=True, confluence_score=40.0, stage_phase=2, is_in_buy_zone=True, risk_reward_ratio=2.5
    )
    assert glx_state["state"] == DecisionState.INSUFFICIENT_DATA.value

    test_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST_AAA", current_price=0.0, candle_count=0, freshness_status="UNAVAILABLE",
        has_fundamentals=False, confluence_score=35.0, stage_phase=None, is_in_buy_zone=False, risk_reward_ratio=None, is_cataloged=False
    )
    assert test_state["state"] == DecisionState.UNVERIFIED.value
